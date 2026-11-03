# Quality of 8-bit activations for the int8 tensor path (W4A8): fake-quantizes the input of every quantized linear
# (symmetric int8, one scale per row per group of G values along K) and compares per-position logits with the
# unmodified model on real text.  a8_quality.py <model> <text files...> [--group 64] [--max 2048]
import argparse, numpy as np, mlx.core as mx, mlx.nn as nn
from mlx_lm import load

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("files", nargs="+")
ap.add_argument("--groups", default="64,128,0"); ap.add_argument("--max", type=int, default=2048)
a = ap.parse_args()
model, tok = load(a.model)
orig = nn.QuantizedLinear.__call__
G = [0]

def fq(x):
    g = G[0]
    K = x.shape[-1]
    xf = x.astype(mx.float32)
    gs = K if g == 0 else g
    xr = xf.reshape(*x.shape[:-1], K // gs, gs)
    s = mx.maximum(mx.abs(xr).max(axis=-1, keepdims=True), 1e-8) / 127.0
    return (mx.round(xr / s) * s).reshape(x.shape).astype(x.dtype)

def patched(self, x):
    return orig(self, fq(x))

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
for g in map(int, a.groups.split(",")):
    G[0] = g
    rows = []
    for f, t in zip(a.files, texts):
        ids = tok.encode(t)[: a.max]
        nn.QuantizedLinear.__call__ = orig
        ref = logits(ids)
        nn.QuantizedLinear.__call__ = patched
        out = logits(ids)
        nn.QuantizedLinear.__call__ = orig
        rows.append(stats(ref, out, ids))
        print(f"group {g or 'row'}: {f.split('/')[-1]:28s} T={len(ids)} KL mean {rows[-1][0]:.5f} p99 {rows[-1][1]:.4f} "
              f"top1 agree {rows[-1][2]*100:.2f}%  ppl {rows[-1][3]:.3f} -> {rows[-1][4]:.3f}", flush=True)
