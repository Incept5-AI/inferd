# Image decoding (image_decode.py; Codex review 2026-11-04, R01): the format allowlist, the review's XBM probe, the
# killable child process. Pure Pillow, no MLX or model: runs in CI.
#   python3 frontend/test_image_decode.py
import io, os, sys, time, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import image_decode as D  # noqa: E402
from PIL import Image  # noqa: E402

LIMIT = 64_000_000


def encode(fmt, size=(40, 24), color=(200, 10, 10)):
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, fmt)
    return b.getvalue()


def xbm_probe(k):
    """image_probe.py's input for GHSA-xgh8-jp6q-p5x8, verbatim (512 bytes; k=20 took 1.7 s in Pillow 12.3.0)."""
    data = b"#define" + b" " * k + b"_width 1\r" + b"#define" + b" " * k + b"_height 1\r"
    data += b"#define" + b" " * k + b"_x_hot 1\r" + b"#define" + b" " * k + b"_y_hot 1" + b"\r" * k + b"_bits"
    return data + b"A" * (512 - len(data))


class Decode(unittest.TestCase):
    def both(self, data, **kw):
        return D.decode_inline(data, LIMIT, **kw), D.decode_child(data, LIMIT, **kw)

    def test_allowed_formats(self):
        for fmt in D.FORMATS:
            a, b = self.both(encode(fmt))
            self.assertEqual((a.mode, a.size), ("RGB", (40, 24)), fmt)
            self.assertEqual(a.tobytes(), b.tobytes(), fmt)

    def test_other_formats_refused(self):
        for fmt in ("BMP", "TIFF", "PPM", "ICO", "TGA", "XBM", "PCX", "SGI"):
            b = io.BytesIO()
            Image.new("1" if fmt == "XBM" else "RGB", (16, 16)).save(b, fmt)
            for fn in (D.decode_inline, D.decode_child):
                with self.assertRaisesRegex(D.DecodeError, "unreadable", msg=fmt):
                    fn(b.getvalue(), LIMIT)

    def test_review_xbm_probe_refused_fast(self):
        for k in (10, 20, 60):  # 60: the advisory's minutes-long size
            for fn in (D.decode_inline, D.decode_child):
                t = time.monotonic()
                with self.assertRaisesRegex(D.DecodeError, "unreadable"):
                    fn(xbm_probe(k), LIMIT)
                self.assertLess(time.monotonic() - t, 1.0, (k, fn.__name__))

    def test_media_type_does_not_matter(self):
        # the data: URL's type is not trusted: a GIF labelled PNG decodes as GIF, an XBM labelled PNG is refused
        self.assertEqual(D.decode_inline(encode("GIF"), LIMIT).size, (40, 24))

    def test_pixel_limits(self):
        raw = encode("PNG", (100, 100))
        for fn in (D.decode_inline, D.decode_child):
            with self.assertRaises(D.DecodeError) as e:
                fn(raw, 9999)
            self.assertEqual(e.exception.kind, "pixels")
            with self.assertRaises(D.DecodeError) as e:
                fn(raw, LIMIT, budget_pixels=5000)
            self.assertEqual(e.exception.kind, "budget")

    def test_truncated_image(self):
        raw = encode("PNG", (64, 64))
        for fn in (D.decode_inline, D.decode_child):
            with self.assertRaises(D.DecodeError):
                fn(raw[:len(raw) // 2], LIMIT)

    def test_child_killed_at_deadline(self):
        t = time.monotonic()
        with self.assertRaises(D.DecodeError) as e:
            D.decode_child(b"x", 100, timeout=0.5, cmd=[sys.executable, "-c", "import time; time.sleep(30)"])
        self.assertEqual(e.exception.kind, "timeout")
        self.assertLess(time.monotonic() - t, 5)

    def test_crashing_child_fails_the_image_only(self):
        for code in ("import os; os.abort()", "import sys; sys.stdout.write('garbage')",
                     "import sys, json; sys.stdout.write(json.dumps({'ok': True, 'w': 4, 'h': 4}) + '\\n')"):
            with self.assertRaises(D.DecodeError) as e:
                D.decode_child(b"x", 100, cmd=[sys.executable, "-c", code])
            self.assertEqual(e.exception.kind, "crashed", code)

    def test_mode_switch(self):
        os.environ["INFERD_IMAGE_DECODE"] = "inline"
        try:
            self.assertEqual(D.decode(encode("JPEG"), LIMIT).size, (40, 24))
        finally:
            os.environ.pop("INFERD_IMAGE_DECODE")
        self.assertEqual(D.decode(encode("WEBP"), LIMIT).size, (40, 24))


if __name__ == "__main__":
    unittest.main(verbosity=1)
