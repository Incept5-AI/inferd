"""Pre-decode regression tests. No weights or network are required."""

import base64
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from PIL import Image

import vision


def image_with_color(value, compression=6):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (value, 12, 23)).save(buffer, "PNG", compress_level=compression)
    payload = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}}


def prompt(images):
    return [{"role": "user", "content": [{"type": "text", "text": "Look at these."}, *images]}]


class ImageProcessorFixture:
    def __init__(self):
        self.count = 0

    def __call__(self, images, **kwargs):
        self.count += len(images)
        pixels = np.array([image.resize((2, 2)).getpixel((0, 0)) for image in images])
        return {"pixel_values": np.repeat(pixels, 4, axis=0),
                "image_grid_thw": np.tile([1, 2, 2], (len(images), 1))}


class Qwen3VLProcessor:
    # A tiny stand-in for Qwen's layout and tokenization, exercising the
    # supported processor hook without requiring its tokenizer or a checkpoint.
    __module__ = "transformers.models.qwen3_vl.processing_qwen3_vl"

    def __init__(self):
        self.image_processor = ImageProcessorFixture()

    def apply_chat_template(self, messages, **kwargs):
        return "Look at these."

    def replace_image_token(self, image_inputs, image_idx, **kwargs):
        return [7] * (int(np.prod(image_inputs["image_grid_thw"][image_idx])) // 4)

    def _process_images(self, images, **kwargs):
        enc = self.image_processor(images, **kwargs)
        return enc, [self.replace_image_token(enc, i) for i in range(len(images))]

    def __call__(self, text, images, **kwargs):
        enc, replacement = self._process_images(images, **kwargs)
        tokens = [5, *[t for image in replacement for t in image], 6]
        return {**enc, "input_ids": [tokens]}


class SourceCacheTest(unittest.TestCase):
    def setUp(self):
        self.frontend = vision.Vision.__new__(vision.Vision)
        self.frontend.proc = Qwen3VLProcessor()
        self.frontend.feature_cache = vision.VisionFeatureCache(1 << 20)
        self.frontend.cfg = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
        self.frontend.rope = None
        self.frontend.image_token = 7
        self.frontend.vocab = 256
        self.frontend.prep_digest = b"test"
        self.frontend._encode_images = lambda pixels, grid: pixels[::4].astype("<u2")
        env = patch.dict(os.environ, {"INFERD_IMAGE_MAX_COUNT": "64", "INFERD_IMAGE_DECODE": "inline"})
        env.start()
        self.addCleanup(env.stop)
        ropes = patch.object(vision.qlang.LanguageModel, "get_rope_index",
                             side_effect=lambda _, tokens, *args: (mx.zeros((3, 1, tokens.shape[1])), mx.zeros(1)))
        ropes.start()
        self.addCleanup(ropes.stop)

    def test_all_hits_skip_decode_resize_and_encoding_at_19_and_47(self):
        for count in (19, 47):
            with self.subTest(count=count):
                self.frontend.feature_cache = vision.VisionFeatureCache(1 << 20)
                messages = prompt([image_with_color(i) for i in range(1, count + 1)])
                previous = self.frontend.prepare(messages)
                processor = self.frontend.proc.image_processor
                processor.count = 0
                with patch.object(vision.image_decode, "decode", side_effect=AssertionError("decode on a hot hit")):
                    with patch.object(self.frontend, "_encode_images", side_effect=AssertionError("tower on a hot hit")):
                        self.assertEqual(self.frontend.prepare(messages), previous)
                self.assertEqual(processor.count, 0)

    def test_mixed_request_prepares_the_new_image_only(self):
        older, newer = image_with_color(1), image_with_color(2)
        self.frontend.prepare(prompt([older]))
        processor = self.frontend.proc.image_processor
        processor.count = 0
        with patch.object(vision.image_decode, "decode", wraps=vision.image_decode.decode) as decode:
            mixed = self.frontend.prepare(prompt([older, newer, older]))
        self.assertEqual(processor.count, 1)
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(np.frombuffer(mixed[1], dtype="<u2").reshape(-1, 3)[:, 0].tolist(), [1, 2, 1])

    def test_warm_images_still_count_towards_all_request_limits(self):
        first, second = image_with_color(1), image_with_color(2)
        messages = prompt([first, second])
        self.frontend.prepare(messages)
        size = len(base64.b64decode(first["image_url"]["url"].split(",", 1)[1]))
        for limits, match in (
            ({"INFERD_IMAGE_MAX_COUNT": "1"}, "too many images"),
            ({"INFERD_IMAGE_REQ_PIXELS": "256"}, "pixels"),
            ({"INFERD_IMAGE_REQ_BYTES": str(size)}, "budget"),
        ):
            with self.subTest(match=match), patch.dict(os.environ, limits):
                with self.assertRaisesRegex(ValueError, match):
                    self.frontend.prepare(messages)

    def test_new_file_encoding_uses_normal_decode_and_pixel_identity(self):
        previous = self.frontend.prepare(prompt([image_with_color(1)]))
        with patch.object(vision.image_decode, "decode", wraps=vision.image_decode.decode) as decode:
            with patch.object(self.frontend, "_encode_images", side_effect=AssertionError("same pixels re-encoded")):
                again = self.frontend.prepare(prompt([image_with_color(1, compression=0)]))
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(again, previous)


if __name__ == "__main__":
    mx.set_default_device(mx.cpu)
    unittest.main()
