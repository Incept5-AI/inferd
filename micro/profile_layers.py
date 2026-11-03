# Per-component decode cost at batch B: one GDN layer, one attention layer, one MLP, and the bare GDN recurrence.
import sys, time, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.gated_delta import gated_delta_update

path, L = sys.argv[1], int(sys.argv[2])
model, _ = load(path)
tc = getattr(model.args, "text_config", None); HID = tc["hidden_size"] if isinstance(tc, dict) else (tc.hidden_size if tc else model.args.hidden_size)
inner = model.language_model.model if hasattr(model, "language_model") else model.model
layers = inner.layers
gi = next(i for i, l in enumerate(layers) if l.is_linear)
ai = next(i for i, l in enumerate(layers) if not l.is_linear)
g = layers[gi].linear_attn
print(f"GDN: Hv={g.num_v_heads} Hk={g.num_k_heads} dk={g.head_k_dim} dv={g.head_v_dim}; "
      f"state/seq/layer = {g.num_v_heads*g.head_k_dim*g.head_v_dim*4/2**20:.2f} MiB fp32")

def bench(f, n=20):
    for _ in range(3): mx.eval(f())
    ts = []
    for _ in range(5):
        t = time.perf_counter(); mx.eval([f() for _ in range(n)]); ts.append((time.perf_counter() - t) / n)
    return sorted(ts)[2] * 1e3

for B in [int(b) for b in sys.argv[3].split(",")]:
    cache = make_prompt_cache(model)
    x = mx.random.randint(0, 150000, (B, L))
    for i in range(0, L, 1024):
        mx.eval(model(x[:, i:i + 1024], cache=cache))
    h = mx.random.normal((B, 1, HID)).astype(mx.bfloat16) * 0.1
    gc, ac = cache[gi], cache[ai]
    def gdn():
        s0, s1, off = gc[0], gc[1], gc.offset if hasattr(gc, "offset") else None
        r = layers[gi].linear_attn(layers[gi].input_layernorm(h), None, gc)
        gc[0], gc[1] = s0, s1  # keep the state fixed so repeated calls are identical
        return r
    def attn():
        r = layers[ai].self_attn(layers[ai].input_layernorm(h), None, ac)
        ac.offset -= 1  # undo the append
        return r
    mlp = lambda: layers[gi].mlp(h)
    st = gc[1]
    q = mx.random.normal((B, 1, g.num_k_heads, g.head_k_dim)).astype(mx.bfloat16)
    v = mx.random.normal((B, 1, g.num_v_heads, g.head_v_dim)).astype(mx.bfloat16)
    a = mx.random.normal((B, 1, g.num_v_heads)).astype(mx.bfloat16)
    bb = mx.random.normal((B, 1, g.num_v_heads)).astype(mx.bfloat16)
    rec = lambda: gated_delta_update(q, q, v, a, bb, g.A_log, g.dt_bias, st, None, use_kernel=True)
    tg, ta, tm, tr = bench(gdn), bench(attn), bench(mlp), bench(rec)
    nl = len(layers); ng = sum(l.is_linear for l in layers)
    est = ng * tg + (nl - ng) * ta + nl * tm
    print(f"B={B:3d} L={L}: GDN layer {tg:.3f} ms (recurrence {tr:.3f}) | attn layer {ta:.3f} | MLP {tm:.3f} | "
          f"sum over {nl} layers ≈ {est:.1f} ms  [GDN {ng*tg:.1f}, attn {(nl-ng)*ta:.1f}, MLP {nl*tm:.1f}]", flush=True)
    del cache; mx.clear_cache()
