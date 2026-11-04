#!/usr/bin/env python3
# summ.py stats.jsonl [label-substring...]: per label and N, mean +- half-range (and min..max) over trials of
# TTFT max (last user's first token), wall (last user's last token = total time), A_e2e, window decode.
import json, sys, collections as C
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
keep = sys.argv[2:]
g = C.defaultdict(list)
for r in rows:
    if keep and not any(k in r.get("label", "") for k in keep): continue
    g[(r["label"], r["n"])].append(r)
def f(v):
    if not v: return "-"
    m = sum(v) / len(v)
    return f"{m:7.1f} ± {(max(v)-min(v))/2:4.1f}"
print(f"{'label':28} {'N':>2} {'k':>2} {'err':>3} | {'TTFT max s':>14} | {'total s':>14} | {'A_e2e tok/s':>14} | {'decode all-gen':>14}")
for (lab, n), rs in sorted(g.items(), key=lambda x: (x[0][0], x[0][1])):
    ok = [r for r in rs if not r.get("errors")]
    print(f"{lab:28} {n:>2} {len(ok):>2} {sum(r.get('errors',0) for r in rs):>3} | {f([r['ttft_max_s'] for r in ok]):>14} | "
          f"{f([r['wall_s'] for r in ok]):>14} | {f([r['a_e2e_tok_s'] for r in ok]):>14} | "
          f"{f([r['window_tok_s'] for r in ok if r.get('window_tok_s')]):>14}")
