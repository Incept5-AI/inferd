# Vision for the front end: runs the checkpoint's vision tower (mlx_vlm's Qwen3.5 VisionModel, vision weights
# only) and builds what inferd needs for a multimodal prompt:
#   tokens  - the prompt with each image's pad tokens replaced by content-keyed ids >= vocab (so the prefix cache
#             can tell different images apart; inferd takes these rows' embeddings from `embeds`),
#   embeds  - bf16 [n_image_tokens, hidden] image features in prompt order,
#   mrope   - interleaved M-RoPE positions [L, 3] (t, h, w), and rope_delta for the generated tokens.
import base64, glob, hashlib, io, json, types, urllib.request

import mlx.core as mx
# Keep MLX's buffer cache small: the vision tower shares the GPU with inferd, whose KV pool is sized to the budget
# minus VISION_MARGIN_GIB, so freed image activations must go back to the system rather than sit in MLX's cache.
(getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit)(256 << 20)
import numpy as np
from PIL import Image
from transformers import AutoProcessor

from mlx_vlm.models.qwen3_5 import config as qcfg, language as qlang, vision as qvis


MAX_IMAGE_BYTES = 32 << 20
MAX_IMAGE_PIXELS = 64_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS  # decompression-bomb guard (PIL only raises above 2x this; we check below)


def _env_int(name, default):
    import os
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


class Budget:
    """Per-request image limits (Codex review 2026-11-02, S06): image count, total decoded bytes and total pixels.
    INFERD_IMAGE_MAX_COUNT (16), INFERD_IMAGE_REQ_BYTES (64 MiB), INFERD_IMAGE_REQ_PIXELS (128 M)."""

    def __init__(self):
        self.count, self.bytes, self.pixels = 0, 0, 0
        self.max_count = _env_int("INFERD_IMAGE_MAX_COUNT", 16)
        self.max_bytes = _env_int("INFERD_IMAGE_REQ_BYTES", 64 << 20)
        self.max_pixels = _env_int("INFERD_IMAGE_REQ_PIXELS", 128_000_000)

    def add(self, n_bytes, n_pixels=0):
        self.count += n_bytes > 0
        self.bytes += n_bytes
        self.pixels += n_pixels
        if self.count > self.max_count:
            raise ValueError(f"too many images in one request (limit {self.max_count})")
        if self.bytes > self.max_bytes:
            raise ValueError(f"images exceed the per-request budget of {self.max_bytes >> 20} MiB")
        if self.pixels > self.max_pixels:
            raise ValueError(f"images exceed the per-request budget of {self.max_pixels:,} pixels")


def _decode_data_url(url, limit):
    head, sep, b64 = url.partition(",")
    if not sep:
        raise ValueError("malformed data: URL (no comma)")
    if not head.endswith(";base64"):
        raise ValueError("data: image URLs must be base64-encoded")
    # Reject before allocating the decoded copy: base64 is 4 characters per 3 bytes (plus padding/whitespace slack).
    if len(b64) > (limit + 2) // 3 * 4 + 1024:
        raise ValueError(f"image larger than {limit >> 20} MiB")
    try:
        return base64.b64decode("".join(b64.split()), validate=True)
    except (ValueError, base64.binascii.Error) as e:
        raise ValueError(f"invalid base64 image data: {e}") from None


def load_image(url, budget=None):
    """Inline data: URLs always; http(s) only with INFERD_IMAGE_URLS=1; local paths only with INFERD_IMAGE_FILES=1.
    Both off by default: a caller-supplied path or URL let the server read local files or reach internal hosts
    (Codex review F12). `budget` (a Budget) enforces the per-request totals."""
    import os
    if not isinstance(url, str) or not url:
        raise ValueError("image must be a URL string")
    limit = MAX_IMAGE_BYTES
    if budget is not None:
        limit = min(limit, budget.max_bytes - budget.bytes)
        if limit <= 0:
            raise ValueError(f"images exceed the per-request budget of {budget.max_bytes >> 20} MiB")
    if url.startswith("data:"):
        data = _decode_data_url(url, limit)
    elif url.startswith(("http://", "https://")):
        if os.environ.get("INFERD_IMAGE_URLS") != "1":
            raise ValueError("image URLs are disabled (send the image inline as a data: URL, or set INFERD_IMAGE_URLS=1)")
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read(limit + 1)
    else:
        if os.environ.get("INFERD_IMAGE_FILES") != "1":
            raise ValueError("local image paths are disabled (send the image inline, or set INFERD_IMAGE_FILES=1)")
        with open(url.removeprefix("file://"), "rb") as f:
            data = f.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"image larger than {limit >> 20} MiB" if limit == MAX_IMAGE_BYTES else
                         f"images exceed the per-request budget of {budget.max_bytes >> 20} MiB")
    try:
        img = Image.open(io.BytesIO(data))  # lazy: only the header is parsed here
        w, h = img.size
    except Exception as e:
        raise ValueError(f"unreadable image: {e}") from None
    if w * h > MAX_IMAGE_PIXELS:
        raise ValueError(f"image has {w * h:,} pixels (limit {MAX_IMAGE_PIXELS:,})")
    if budget is not None:
        budget.add(len(data), w * h)
    return img.convert("RGB")


def image_keys(digest, n, vocab):
    """n prefix-cache key ids for one image, all in [vocab, vocab + 2^24): consecutive 24-bit pieces of a SHAKE-256
    stream over the image's full identity digest. The ids of an image of n >= 11 tokens carry 264 bits of the hash
    (two images share keys only if their 256-bit identities collide); a smaller image carries 24 * n bits (S06: the
    old scheme kept only a 24-bit residue for the whole image, whatever its size)."""
    stream = hashlib.shake_256(b"inferd-image-keys-v2" + digest).digest(3 * n)
    return [vocab + int.from_bytes(stream[3 * i:3 * i + 3], "little") for i in range(n)]


def image_identity(img, grid, prep_digest):
    """256-bit identity of one image as the model sees it: decoded RGB pixels, dimensions, the processor/vision
    configuration (resize, normalisation, patch/merge sizes) and the image's (t, h, w) grid."""
    h = hashlib.sha256(b"inferd-image-v2")
    h.update(prep_digest)
    h.update(json.dumps([img.mode, list(img.size), [int(x) for x in grid]]).encode())
    h.update(img.tobytes())
    return h.digest()


def has_images(messages):
    return any(isinstance(m.get("content"), list) and any(p.get("type") in ("image_url", "image") for p in m["content"])
               for m in messages)


class Vision:
    def __init__(self, model_dir):
        raw = json.load(open(f"{model_dir}/config.json"))
        self.cfg = qcfg.ModelConfig.from_dict(raw)
        vc = self.cfg.vision_config
        if isinstance(vc, dict):
            vc = qcfg.VisionConfig.from_dict(vc)
            self.cfg.vision_config = vc
        self.vocab = raw.get("text_config", raw)["vocab_size"]
        self.model = qvis.VisionModel(vc)
        w = {}
        for f in glob.glob(f"{model_dir}/*.safetensors"):
            for k, v in mx.load(f).items():
                if k.startswith("vision_tower."):
                    w[k.removeprefix("vision_tower.")] = v
        if hasattr(self.model, "sanitize"):
            w = self.model.sanitize(w)
        self.model.load_weights(list(w.items()))
        mx.eval(self.model.parameters())
        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.image_token = raw["image_token_id"]
        self.rope = types.SimpleNamespace(config=self.cfg)
        try:
            ip = self.proc.image_processor.to_dict()
        except Exception:
            ip = {k: v for k, v in vars(self.proc.image_processor).items() if not k.startswith("_")}
        self.prep_digest = hashlib.sha256(json.dumps({"processor": ip, "vision": raw.get("vision_config"),
                                                      "vocab": self.vocab, "image_token": self.image_token},
                                                     sort_keys=True, default=str).encode()).digest()

    def prepare(self, messages, tools=None, **template_kw):
        """OpenAI chat messages with image parts -> (tokens, embeds bytes, mrope flat list, rope_delta)."""
        images, msgs, budget = [], [], Budget()
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                parts = []
                for p in c:
                    if p.get("type") == "image_url":
                        u = p["image_url"]["url"] if isinstance(p["image_url"], dict) else p["image_url"]
                        images.append(load_image(u, budget))
                        parts.append({"type": "image"})
                    elif p.get("type") == "image":
                        images.append(load_image(p.get("image") or p.get("url"), budget))
                        parts.append({"type": "image"})
                    else:
                        parts.append(p)
                m = dict(m, content=parts)
            msgs.append(m)
        text = self.proc.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, **template_kw)
        enc = self.proc(text=[text], images=images, return_tensors="np")
        ids = np.array(enc["input_ids"])[0].astype(np.int64)
        grid_np = np.array(enc["image_grid_thw"])
        grid = mx.array(grid_np)
        feats, _ = self.model(mx.array(np.array(enc["pixel_values"])), grid)
        feats = feats.astype(mx.bfloat16)
        pos, delta = qlang.LanguageModel.get_rope_index(self.rope, mx.array(ids[None]), grid, None, None)
        mx.eval(feats, pos, delta)
        # Content-keyed ids for the pad rows: image k's tokens -> image_keys(full identity of image k), so the prefix
        # cache (which matches token ids) tells images apart by their full content, size and preprocessing.
        tokens = ids.copy()
        merge = self.cfg.vision_config.spatial_merge_size ** 2
        pads = np.flatnonzero(ids == self.image_token)
        if len(grid_np) != len(images):
            raise ValueError(f"processor returned {len(grid_np)} image grids for {len(images)} images")
        start = 0
        for k, g in enumerate(grid_np):
            n = int(np.prod(g)) // merge
            if start + n > len(pads):
                raise ValueError("image pad tokens do not match the image grids")
            tokens[pads[start:start + n]] = image_keys(image_identity(images[k], g, self.prep_digest), n, self.vocab)
            start += n
        emb = np.array(feats.view(mx.uint16)).astype("<u2").tobytes()
        mrope = np.array(pos)[:, 0, :].T.reshape(-1).astype(int).tolist()  # [L, 3] flattened
        return tokens.tolist(), emb, mrope, int(np.array(delta).reshape(-1)[0])
