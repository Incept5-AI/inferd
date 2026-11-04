# Image decoding for vision.py, isolated from the server (Codex review 2026-11-04, R01). Only the formats the API
# documents are ever identified or decoded: PNG, JPEG, WebP and GIF (first frame). Pillow tries only those plugins, so
# a payload in any other format (XBM, JPEG 2000, TIFF, ...) is refused at its header, before any other decoder runs.
# By default the decode runs in a child process with a deadline (INFERD_IMAGE_DECODE_TIMEOUT, 10 s): a decoder that
# stalls is killed, and one that corrupts memory takes down only the child. A thread cannot be stopped that way, and
# the vision path holds the server's vision lock while it decodes. INFERD_IMAGE_DECODE=inline decodes in-process.
# Pure Pillow, no MLX: the child imports nothing else. Child protocol: image bytes on stdin; stdout is one JSON line,
# {"ok": true, "w": W, "h": H} followed by W*H*3 RGB bytes, or {"ok": false, "error": "...", "kind": "..."}.
#   python -I image_decode.py MAX_PIXELS BUDGET_PIXELS < image
import io, json, os, subprocess, sys

FORMATS = ("PNG", "JPEG", "WEBP", "GIF")


class DecodeError(ValueError):
    def __init__(self, message, kind="unreadable"):
        super().__init__(message)
        self.kind = kind


def decode_inline(data, max_pixels, budget_pixels=None):
    """bytes -> RGB PIL image. DecodeError kinds: unreadable, pixels (over max_pixels), budget (over budget_pixels)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = max_pixels  # decompression-bomb guard (PIL raises only above 2x this; checked below)
    try:
        img = Image.open(io.BytesIO(data), formats=FORMATS)  # lazy: only the header is parsed here
        w, h = img.size
    except Exception as e:
        raise DecodeError(f"unreadable image (supported formats: {', '.join(FORMATS)}): {e}") from None
    if w * h > max_pixels:
        raise DecodeError(f"image has {w * h:,} pixels (limit {max_pixels:,})", "pixels")
    if budget_pixels is not None and w * h > budget_pixels:
        raise DecodeError(f"{w * h:,} pixels", "budget")
    try:
        return img.convert("RGB")
    except Exception as e:
        raise DecodeError(f"unreadable image: {e}") from None


def _child_cmd(max_pixels, budget_pixels):
    return [sys.executable, "-I", os.path.abspath(__file__), str(max_pixels),
            str(-1 if budget_pixels is None else budget_pixels)]


def decode_child(data, max_pixels, budget_pixels=None, timeout=10.0, cmd=None):
    """decode_inline in a child process killed after `timeout` seconds."""
    from PIL import Image
    try:
        p = subprocess.run(cmd or _child_cmd(max_pixels, budget_pixels), input=data, capture_output=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:  # run() kills the child before raising
        raise DecodeError(f"image decoding took longer than {timeout:g} s (INFERD_IMAGE_DECODE_TIMEOUT)",
                          "timeout") from None
    head, _, raw = p.stdout.partition(b"\n")
    try:
        meta = json.loads(head)
    except ValueError:
        meta = None
    if p.returncode != 0 or not isinstance(meta, dict):
        raise DecodeError(f"image decoder failed (exit {p.returncode})", "crashed")
    if not meta.get("ok"):
        raise DecodeError(str(meta.get("error", "unreadable image"))[:500], meta.get("kind", "unreadable"))
    w, h = int(meta["w"]), int(meta["h"])
    if not (0 < w * h <= max_pixels) or len(raw) != w * h * 3:
        raise DecodeError("image decoder returned a malformed image", "crashed")
    return Image.frombytes("RGB", (w, h), raw)


def decode(data, max_pixels, budget_pixels=None):
    mode = os.environ.get("INFERD_IMAGE_DECODE", "subprocess")
    if mode == "inline":
        return decode_inline(data, max_pixels, budget_pixels)
    try:
        timeout = float(os.environ.get("INFERD_IMAGE_DECODE_TIMEOUT") or 10)
    except ValueError:
        timeout = 10.0
    return decode_child(data, max_pixels, budget_pixels, timeout=max(timeout, 0.1))


def _main():
    max_pixels, budget = int(sys.argv[1]), int(sys.argv[2])
    data = sys.stdin.buffer.read()
    out = sys.stdout.buffer
    try:
        img = decode_inline(data, max_pixels, None if budget < 0 else budget)
    except DecodeError as e:
        out.write(json.dumps({"ok": False, "error": str(e), "kind": e.kind}).encode() + b"\n")
        return
    out.write(json.dumps({"ok": True, "w": img.size[0], "h": img.size[1]}).encode() + b"\n")
    out.write(img.tobytes())


if __name__ == "__main__":
    _main()
