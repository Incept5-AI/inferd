# Ceiling probe: one batched decode step of a real model with MLX kernels, no scheduler.
# Prefill B identical-length random prompts of L tokens, then time decode steps at batch B.
import sys, time, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

path, L = sys.argv[1], int(sys.argv[2])
Bs = [int(b) for b in sys.argv[3].split(",")]
model, tok = load(path)
for B in Bs:
    cache = make_prompt_cache(model)
    x = mx.random.randint(0, 150000, (B, L))
    t = time.perf_counter()
    for i in range(0, L, 1024):  # chunked prefill
        mx.eval(model(x[:, i:i + 1024], cache=cache))
    tp = time.perf_counter() - t
    y = mx.random.randint(0, 150000, (B, 1))
    for _ in range(3): mx.eval(model(y, cache=cache))
    ts = []
    for _ in range(10):
        t = time.perf_counter(); mx.eval(model(y, cache=cache)); ts.append(time.perf_counter() - t)
    s = sorted(ts)[5]
    print(f"B={B:3d} L={L:6d} prefill {B*L/tp:7.0f} tok/s | decode step {s*1e3:7.2f} ms = {B/s:7.1f} tok/s total, {1/s:6.1f}/stream", flush=True)
    del cache; mx.clear_cache()
