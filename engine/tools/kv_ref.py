# Reference for kv_check (long-context accuracy): mlx_lm with its bf16 KV cache, at the positions kv_check keeps.
# Architectures only mlx-vlm has (Flash-Next's qwen4_exp) load its language model instead, with its own caches
# (QSAKVCache: the QSA sparse-attention indexer is applied past 2,048 tokens). QSA_REF_DENSE=1 disables the indexer's
# selection (every block kept: dense attention), to show that the reference's sparse selection changes the logits.
#   kv_ref.py make <model> <tokens.txt> <T> <text file>...   one sequence of T tokens per text file (repeated if short)
#   kv_ref.py ref <model> <tokens.txt> <ref.npy> <tail N> [rows R]
#   kv_ref.py compare <ref.npy> <engine.bin> [label]          KL(ref || engine), top-1 agreement, per sequence
#     (an R = 1 reference also serves R-row engine runs: every R-th row, the last of each step, is compared)
import sys
import numpy as np

cmd = sys.argv[1]
if cmd == "compare":
    ref = np.load(sys.argv[2])  # [B, S, V] float32
    eng = np.fromfile(sys.argv[3], dtype=np.float32)
    if eng.size != ref.size:  # an R-row engine run against an R = 1 reference: keep each step's last row
        R = ref.size // eng.size
        ref = ref[:, R - 1::R]
    eng = eng.reshape(ref.shape)
    label = sys.argv[4] if len(sys.argv) > 4 else sys.argv[3]

    def logp(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))

    lr, le = logp(ref.astype(np.float64)), logp(eng.astype(np.float64))
    kl = (np.exp(lr) * (lr - le)).sum(-1)  # [B, S]
    top = (ref.argmax(-1) == eng.argmax(-1))
    for b in range(ref.shape[0]):
        print(f"{label:28s} seq {b}: KL mean {kl[b].mean():.5f} median {np.median(kl[b]):.5f} "
              f"p95 {np.percentile(kl[b], 95):.5f} max {kl[b].max():.4f}  top-1 {top[b].mean() * 100:5.1f}%")
    print(f"{label:28s} all  : KL mean {kl.mean():.5f}  top-1 {top.mean() * 100:5.1f}%  ({kl.size} positions)")
    sys.exit(0)

import os
import mlx.core as mx

try:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(sys.argv[2])
except Exception:  # architectures only mlx-vlm has (e.g. Flash-Next's qwen4_exp): its language model
    from mlx_vlm import load as vlm_load

    _m, _proc = vlm_load(sys.argv[2])
    tok = getattr(_proc, "tokenizer", _proc)
    _lm = _m.language_model
    if os.environ.get("QSA_REF_DENSE") == "1":  # every block selected: the indexer's mask is plain causal
        n = 0
        for mod in _lm.modules():
            if hasattr(mod, "block_topk") and hasattr(mod, "select"):
                orig = mod.select

                def dense(*a, _orig=orig, **k):  # keep the indexer cache's updates, return "no selection"
                    _orig(*a, **k)
                    return None

                mod.select = dense
                n += 1
        print(f"QSA_REF_DENSE: indexer selection disabled in {n} layers", flush=True)

    def make_prompt_cache(_model):
        return _lm.make_cache()

    def model(ids, cache=None):
        out = _lm(ids, cache=cache)
        return getattr(out, "logits", out)
if cmd == "make":
    T = int(sys.argv[4])
    seqs = []
    for p in sys.argv[5:]:
        ids = tok.encode(open(p, errors="replace").read())
        while len(ids) < T:
            ids = ids + ids
        seqs.append(ids[:T])
    with open(sys.argv[3], "w") as f:
        f.write(f"{len(seqs)} {T}\n")
        for s in seqs:
            f.write(" ".join(map(str, s)) + "\n")
    sys.exit(0)

lines = open(sys.argv[3]).read().split("\n")
B, T = map(int, lines[0].split())
N, R = int(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 1
want = [T - N + s * R + R - 1 for s in range(N // R)]  # the last row of each kv_check step
out = []
for b in range(B):
    ids = list(map(int, lines[1 + b].split()))
    cache = make_prompt_cache(model)
    rows, t = [], 0
    while t < T:
        n = min(2048, T - t)
        logits = model(mx.array([ids[t:t + n]]), cache=cache)[0]
        keep = [p - t for p in want if t <= p < t + n]
        if keep:
            rows.append(np.array(logits[mx.array(keep)].astype(mx.float32)))
        mx.eval([c.state for c in cache])
        t += n
    out.append(np.concatenate(rows))
    print(f"seq {b}: {len(out[-1])} reference rows", flush=True)
np.save(sys.argv[4], np.stack(out))
