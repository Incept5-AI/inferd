# For each spec/plain divergence in /tmp/spec_out.txt (plain rows first, then spec rows), report the reference
# (mlx_lm) logits of the two competing tokens at the first differing position, given the shared prefix.
import sys, numpy as np, mlx.core as mx
from mlx_lm import load
model, tok = load(sys.argv[1])
rows = [list(map(int, l.split())) for l in open("/tmp/spec_out.txt") if l.strip()]
n = len(rows) // 2
for b in range(n):
    p, s = rows[b], rows[n + b]
    k = next((i for i in range(min(len(p), len(s))) if p[i] != s[i]), None)
    if k is None:
        continue
    lg = np.array(model(mx.array([p[:k]]))[0, -1].astype(mx.float32))
    top = np.argsort(-lg)[:3]
    print(f"seq {b} @ {k}: plain {p[k]} ({lg[p[k]]:.3f})  spec {s[k]} ({lg[s[k]]:.3f})  gap {abs(lg[p[k]] - lg[s[k]]):.3f}  "
          f"ref top3 {[(int(t), round(float(lg[t]), 3)) for t in top]}")
