# rot_quality.py (engine-pf, 2026-11-02): quality of a group-free int8 prefill GEMM ("PF8"): activations rotated by a
# block-diagonal signed Hadamard (512) and quantized to int8 with ONE scale per row; weights dequantized, rotated the
# same way along K and requantized to int8 with one scale per output row. Fake-quant of every quantized linear except
# the lm_head, per-position logits vs the unmodified model on real text (as tools/a8_quality.py).
#   rot_quality.py <model> <text files...> [--modes g64,row,rot,rotw,rot256] [--max 2048]
import argparse, numpy as np, mlx.core as mx, mlx.nn as nn
from mlx_lm import load

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("files", nargs="+")
ap.add_argument("--modes", default="g64,row,rot,rotw"); ap.add_argument("--max", type=int, default=2048)
ap.add_argument("--block", type=int, default=512)
a = ap.parse_args()
model, tok = load(a.model)
orig = nn.QuantizedLinear.__call__
MODE = [""]
VOCAB = tok.vocab_size if hasattr(tok, "vocab_size") else 0

def hadamard(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)
rng = np.random.default_rng(7)
ROT = {}
def rot(K):
    b = a.block
    while K % b: b //= 2
    if b not in ROT:
        ROT[b] = mx.array((np.diag(rng.choice([-1.0, 1.0], b)) @ hadamard(b)).astype(np.float32))
    return ROT[b], b

def q_rows(x):  # symmetric int8, one scale per row (last axis)
    s = mx.maximum(mx.abs(x).max(axis=-1, keepdims=True), 1e-12) / 127.0
    return mx.round(x / s) * s

def q_groups(x, g):
    K = x.shape[-1]
    xr = x.reshape(*x.shape[:-1], K // g, g)
    s = mx.maximum(mx.abs(xr).max(axis=-1, keepdims=True), 1e-12) / 127.0
    return (mx.round(xr / s) * s).reshape(x.shape)

def rotate(x, R, b):
    sh = x.shape
    return (x.reshape(*sh[:-1], sh[-1] // b, b) @ R).reshape(sh)

WC = {}
def patched(self, x):
    m = MODE[0]
    W = mx.dequantize(self.weight, self.scales, self.biases, self.group_size, self.bits).astype(mx.float32)
    N, K = W.shape
    if VOCAB and N >= VOCAB - 1024:
        return orig(self, x)
    xf = x.astype(mx.float32)
    if m == "g64":
        return orig(self, q_groups(xf, 64).astype(x.dtype))
    if m == "row":
        y = q_rows(xf) @ q_rows(W).T
    else:
        R, b = rot(K)
        key = (id(self), m)
        if key not in WC:
            Wr = rotate(W, R, b)
            WC[key] = Wr if m == "rotw" else q_rows(Wr)
            mx.eval(WC[key])
        xr = rotate(xf, R, b)
        xq = q_groups(xr, 256) if m == "rot256" else (xr if m == "rotx" else q_rows(xr))
        y = xq @ WC[key].T
    if "bias" in self:
        y = y + self["bias"]
    return y.astype(x.dtype)

def logits(ids):
    return model(mx.array([ids]))[0].astype(mx.float32)

def stats(ref, out, ids):
    lr = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
    lo = out - mx.logsumexp(out, axis=-1, keepdims=True)
    kl = (mx.exp(lr) * (lr - lo)).sum(-1)
    top = (ref.argmax(-1) == out.argmax(-1)).astype(mx.float32)
    nxt = mx.array(ids[1:])
    nll_r = -mx.take_along_axis(lr[:-1], nxt[:, None], axis=-1).mean()
    nll_o = -mx.take_along_axis(lo[:-1], nxt[:, None], axis=-1).mean()
    mx.eval(kl, top, nll_r, nll_o)
    return float(kl.mean()), float(np.percentile(np.array(kl), 99)), float(top.mean()), float(mx.exp(nll_r)), float(mx.exp(nll_o))

texts = [open(f, errors="ignore").read() for f in a.files]
refs = []
for t in texts:
    ids = tok.encode(t)[: a.max]
    nn.QuantizedLinear.__call__ = orig
    r = logits(ids); mx.eval(r); refs.append((ids, r))
for m in a.modes.split(","):
    MODE[0] = m
    WC.clear()
    for f, (ids, ref) in zip(a.files, refs):
        nn.QuantizedLinear.__call__ = patched
        out = logits(ids)
        nn.QuantizedLinear.__call__ = orig
        s = stats(ref, out, ids)
        print(f"{m:6s} {f.split('/')[-1]:16s} T={len(ids)} KL mean {s[0]:.5f} p99 {s[1]:.4f} top1 {s[2]*100:.2f}%  "
              f"ppl {s[3]:.3f} -> {s[4]:.3f}", flush=True)
