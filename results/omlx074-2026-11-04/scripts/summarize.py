#!/usr/bin/env python3
# summarize.py: tables from results/omlx074/raw/*.jsonl (conc, wall, speed, b2b, six). Prints markdown.
import json, os, sys, statistics as st, collections as C
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "raw")
def rows(f):
    p = os.path.join(R, f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def mr(v):  # mean ± half-range
    v = [x for x in v if x is not None]
    if not v: return "–"
    m = st.mean(v); h = (max(v) - min(v)) / 2
    return f"{m:.1f}" + (f" ± {h:.1f}" if len(v) > 1 else "")
print("## conc.py --mode followup, 16k\n\n| label | N | ok/err | total tok/s | agg decode | TTFT med / max s | cached share | wall s |\n|---|---:|---:|---:|---:|---:|---:|---:|")
for r in rows("conc.jsonl"):
    cs = r.get("cached_tokens") or 0; pt = r.get("prompt_tokens") or 1
    print(f"| {r['label']} | {r['n']} | {r['ok']}/{r['err']} | {r['agg_tps']} | {r['agg_decode_tps']} | {r['ttft_med']} / {r['ttft_max']} | {cs/pt:.2f} | {r['wall_s']} |")
print("\n## wall.py (14,336 in, 6,400 out), mean ± half-range over trials\n\n| label | N | trials | errors | total s | last 1st token s | TTFT med s | decode, all generating | A_e2e | gen tokens |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
g = C.OrderedDict()
for r in rows("wall.jsonl"): g.setdefault((r["label"], r["n"]), []).append(r)
for (l, n), rs in g.items():
    ok = [r for r in rs if not r.get("errors")]
    print(f"| {l} | {n} | {len(rs)} | {sum(r.get('errors',0) for r in rs)} | {mr([r.get('wall_s') for r in ok])} | {mr([r.get('ttft_max_s') for r in ok])} | {mr([r.get('ttft_med_s') for r in ok])} | {mr([r.get('window_tok_s') for r in ok])} | {mr([r.get('a_e2e_tok_s') for r in ok])} | {mr([r.get('gen_tokens') for r in ok])} |")
print("\n## speed.py single stream (median of reps): decode / prefill tok/s / TTFT s\n\n| label | size | prompt tok | decode | prefill | TTFT |\n|---|---|---:|---:|---:|---:|")
g = C.OrderedDict()
for r in rows("speed.jsonl"): g.setdefault((r["label"], r["size"]), []).append(r)
for (l, s), rs in g.items():
    md = lambda k: st.median([r[k] for r in rs if r.get(k) is not None]) if any(r.get(k) is not None for r in rs) else None
    print(f"| {l} | {s} | {md('prompt_tokens')} | {md('decode_tps'):.1f} | {md('prefill_tps'):.0f} | {md('ttft_s'):.2f} |")
print("\n## b2b (PR #4902 workload): time to the one token, s, per prompt in order\n")
g = C.OrderedDict()
for r in rows("b2b.jsonl"): g.setdefault(r["label"], []).append(r)
for l, rs in g.items():
    print(f"**{l}**\n\n| round | target | prompt tok | times s | slowest | mean |\n|---|---:|---:|---|---:|---:|")
    by = C.OrderedDict()
    for r in rs: by.setdefault((r["round"], r["target"]), []).append(r)
    for (rd, t), xs in by.items():
        ts = [x.get("ttft_s") for x in xs if x.get("ttft_s") is not None]; errs = sum(1 for x in xs if x.get("error"))
        print(f"| {rd} | {t} | {xs[0].get('prompt_tokens')} | {', '.join(f'{v:.2f}' for v in ts)}{' (+%d err)' % errs if errs else ''} | {max(ts) if ts else '–'} | {st.mean(ts) if ts else 0:.2f} |")
    print()
print("## six concurrent / one, 1,024 in, 1,024 out (decode window tok/s)\n\n| label | N | trials | errors | decode, all generating | per-user decode | A_e2e | TTFT max |\n|---|---:|---:|---:|---:|---:|---:|---:|")
g = C.OrderedDict()
for r in rows("six.jsonl"): g.setdefault((r["label"], r["n"]), []).append(r)
for (l, n), rs in g.items():
    ok = [r for r in rs if not r.get("errors")]
    print(f"| {l} | {n} | {len(rs)} | {sum(r.get('errors',0) for r in rs)} | {mr([r.get('window_tok_s') for r in ok])} | {mr([r.get('per_user_decode_tok_s') for r in ok])} | {mr([r.get('a_e2e_tok_s') for r in ok])} | {mr([r.get('ttft_max_s') for r in ok])} |")
