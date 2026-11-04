# Image identity keys and input limits in vision.py (Codex review 2026-11-02, S06). No model needed, but vision.py
# imports mlx / mlx_vlm / transformers / PIL at module level, so run it in the front end's environment:
#   python3 frontend/test_vision.py
import base64, io, os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision
from PIL import Image

VOCAB = 248320


def png(w, h, color):
    b = io.BytesIO()
    Image.new("RGB", (w, h), color).save(b, "PNG")
    return b.getvalue()


def data_url(raw):
    return "data:image/png;base64," + base64.b64encode(raw).decode()


class Keys(unittest.TestCase):
    def test_range_and_determinism(self):
        k = vision.image_keys(b"\x01" * 32, 300, VOCAB)
        self.assertEqual(len(k), 300)
        self.assertTrue(all(VOCAB <= t < VOCAB + (1 << 24) for t in k))
        self.assertEqual(k, vision.image_keys(b"\x01" * 32, 300, VOCAB))

    def test_every_digest_bit_matters(self):
        base = vision.image_keys(bytes(32), 11, VOCAB)
        for bit in range(256):
            d = bytearray(32)
            d[bit // 8] ^= 1 << (bit % 8)
            self.assertNotEqual(base, vision.image_keys(bytes(d), 11, VOCAB))

    def test_old_residue_collision_now_distinct(self):
        # The old scheme: vocab + (h + i * c) mod 2^24 with h a 48-bit hash, so any two images whose h agreed mod 2^24
        # got identical keys. Two identities equal in their low 24 bits now give unrelated keys.
        a, b = (1).to_bytes(32, "little"), (1 + (1 << 24)).to_bytes(32, "little")
        old = lambda d, n: [VOCAB + (int.from_bytes(d[:6], "little") + i * 2654435761) % (1 << 24) for i in range(n)]
        self.assertEqual(old(a, 64), old(b, 64))
        self.assertNotEqual(vision.image_keys(a, 64, VOCAB), vision.image_keys(b, 64, VOCAB))

    def test_identity_covers_pixels_size_grid_and_prep(self):
        img = Image.new("RGB", (32, 32), (1, 2, 3))
        base = vision.image_identity(img, [1, 2, 2], b"p")
        self.assertNotEqual(base, vision.image_identity(Image.new("RGB", (32, 32), (1, 2, 4)), [1, 2, 2], b"p"))
        self.assertNotEqual(base, vision.image_identity(Image.new("RGB", (16, 64), (1, 2, 3)), [1, 2, 2], b"p"))
        self.assertNotEqual(base, vision.image_identity(img, [1, 4, 1], b"p"))
        self.assertNotEqual(base, vision.image_identity(img, [1, 2, 2], b"q"))
        self.assertEqual(len(base), 32)


class Limits(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(vision.load_image(data_url(png(8, 8, "red")), vision.Budget()).size, (8, 8))

    def test_oversize_base64_rejected_before_decode(self):
        calls = []
        orig = vision.base64.b64decode
        vision.base64.b64decode = lambda *a, **k: calls.append(1) or orig(*a, **k)
        try:
            with self.assertRaisesRegex(ValueError, "larger than"):
                vision.load_image("data:image/png;base64," + "A" * (vision.MAX_IMAGE_BYTES // 3 * 4 + 8192))
        finally:
            vision.base64.b64decode = orig
        self.assertEqual(calls, [])

    def test_bad_base64_and_non_base64(self):
        with self.assertRaisesRegex(ValueError, "invalid base64"):
            vision.load_image("data:image/png;base64,@@@@")
        with self.assertRaisesRegex(ValueError, "base64-encoded"):
            vision.load_image("data:image/png,rawbytes")
        with self.assertRaisesRegex(ValueError, "unreadable"):
            vision.load_image("data:image/png;base64," + base64.b64encode(b"not an image").decode())

    def test_request_budgets(self):
        b = vision.Budget()
        b.max_count = 2
        u = data_url(png(8, 8, "blue"))
        vision.load_image(u, b), vision.load_image(u, b)
        with self.assertRaisesRegex(ValueError, "too many images"):
            vision.load_image(u, b)
        b = vision.Budget()
        b.max_pixels = 100
        with self.assertRaisesRegex(ValueError, "pixels"):
            vision.load_image(data_url(png(20, 20, "blue")), b)
        raw = png(64, 64, "green")
        b = vision.Budget()
        b.max_bytes = len(raw) + 10
        vision.load_image(data_url(raw), b)
        with self.assertRaisesRegex(ValueError, "per-request budget"):
            vision.load_image(data_url(raw), b)

    def test_pixel_limit(self):
        orig = vision.MAX_IMAGE_PIXELS
        vision.MAX_IMAGE_PIXELS = 100
        try:
            with self.assertRaisesRegex(ValueError, "pixels"):
                vision.load_image(data_url(png(20, 20, "blue")))
        finally:
            vision.MAX_IMAGE_PIXELS = orig

    def test_formats_allowlist(self):  # Codex review 2026-11-04, R01 (decoder details: test_image_decode.py)
        for fmt in ("PNG", "JPEG", "WEBP", "GIF"):
            b = io.BytesIO()
            Image.new("RGB", (40, 24), (200, 10, 10)).save(b, fmt)
            img = vision.load_image(data_url(b.getvalue()), vision.Budget())
            self.assertEqual((img.mode, img.size), ("RGB", (40, 24)), fmt)
        b = io.BytesIO()
        Image.new("RGB", (16, 16), "red").save(b, "BMP")
        with self.assertRaisesRegex(ValueError, "unreadable"):
            vision.load_image(data_url(b.getvalue()), vision.Budget())

    def test_ocr_samples(self):  # the examples/ocr-agents documents (JPEG), as that demo sends them
        sd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../examples/ocr-agents/samples")
        files = sorted(f for f in os.listdir(sd) if f.endswith(".jpg")) if os.path.isdir(sd) else []
        for f in files:
            with open(os.path.join(sd, f), "rb") as fh:
                raw = fh.read()
            img = vision.load_image("data:image/jpeg;base64," + base64.b64encode(raw).decode(), vision.Budget())
            self.assertEqual(img.size, Image.open(io.BytesIO(raw)).size, f)

    def test_urls_and_files_off_by_default(self):
        os.environ.pop("INFERD_IMAGE_URLS", None), os.environ.pop("INFERD_IMAGE_FILES", None)
        with self.assertRaisesRegex(ValueError, "disabled"):
            vision.load_image("http://169.254.169.254/x")
        with self.assertRaisesRegex(ValueError, "disabled"):
            vision.load_image("/etc/passwd")


if __name__ == "__main__":
    unittest.main(verbosity=2)
