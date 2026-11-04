#!/usr/bin/env python3
# summ_mixed.py mixed.jsonl: per label, mean +- half-range over trials of total time, A_e2e, TTFT p50 / p95 / p99 / max,
# e2e p95; plus pooled TTFT p95/p99 over every user of every trial.
import json, sys, collections as C
def pct(v, p):
    v = sorted(v); k = (len(v) - 1) * p / 100; f = int(k)
    return v[f] + (v[min(f + 1, len(v) - 1)] - v[f]) * (k - f)
g = C.defaultdict(list)
for l in open(sys.argv[1]):
    r = json.loads(l)
    g[r["label"]].append(r)
f = lambda v: f"{sum(v)/len(v):6.1f} ± {(max(v)-min(v))/2:4.1f}" if v else "-"
print(f"{'label':28} {'k':>2} {'err':>3} | {'total s':>13} | {'turns/min':>13} | {'A_e2e':>13} | {'TTFT p50':>13} | {'p95':>13} | {'p99':>13} | {'pooled p95/p99':>15} | {'e2e p95':>13}")
for lab, rs in sorted(g.items()):
    ok = [r for r in rs if "total_s" in r]
    pool = [x["ttft"] for r in ok for x in r["rows"] if x.get("ttft") is not None]
    big = [x["ttft"] for r in ok for x in r["rows"] if x.get("ttft") is not None and x["plen"] >= 12000]  # aging evidence
    print(f"{lab:28} {len(ok):>2} {sum(r['errors'] for r in rs):>3} | {f([r['total_s'] for r in ok])} | {f([r.get('turns_per_min') or r['users'] * 60 / r['total_s'] for r in ok])} | {f([r['a_e2e'] for r in ok])} | "
          f"{f([r['ttft_p50'] for r in ok])} | {f([r['ttft_p95'] for r in ok])} | {f([r['ttft_p99'] for r in ok])} | "
          f"{pct(pool,95):6.1f}/{pct(pool,99):6.1f} | {f([r['e2e_p95'] for r in ok])} | "
          f"big p95 {pct(big,95) if big else 0:6.1f} max {max(big) if big else 0:6.1f}")
