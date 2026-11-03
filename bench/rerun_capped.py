#!/usr/bin/env python3
"""rerun_capped.py -- re-run the tests that hit the token cap, at a bigger cap (2026-10-31).

Uses languages-benchmark's own runner unmodified (run_test, summarise): same prompts, grading, retry and loop guard.
Every test with an attempt that finished on "length" is re-run from scratch at --max-tokens; its record replaces the
capped one. The output is the 32k file with those records swapped and `capped_rerun_max_tokens` set, the field the
repo's report uses for its "→ full" rows.

  rerun_capped.py --bench ~/inference/languages-benchmark --in results/v1-x-r1.json --out results/v1-x-r1-rerun64k.json \
      --base-url http://127.0.0.1:8300/v1 --max-tokens 65536 --workers 32 [--label ...]
"""
import argparse, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--base-url", required=True)
ap.add_argument("--max-tokens", type=int, default=65536)
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--timeout", type=int, default=14400)
ap.add_argument("--label", default="")
a = ap.parse_args()
sys.path.insert(0, a.bench)
from langbench import runner, suites  # noqa: E402

st = json.loads(Path(a.inp).read_text())
capped = [r for r in st["records"] if any(x.get("finish_reason") == "length" for x in r["attempts"])]
print(f"{a.inp}: {len(capped)} of {len(st['records'])} tests hit the {st['max_tokens']} cap; re-running at {a.max_tokens}")
suite = suites.get(st["suite"])
names = sorted({r["challenge"] for r in capped})
suite.prepare(names) if hasattr(suite, "prepare") else suite.open()
cfg = {"model": st["model"], "mode": st["mode"], "effort": st["effort"], "sampling": st["sampling"],
       "max_tokens": a.max_tokens, "base_url": a.base_url, "api_key": runner.api_key(), "timeout": a.timeout,
       "extra_body": st.get("extra_body"), "save_replies": True,
       "loop_guard": st.get("loop_guard") if st.get("loop_guard") is not None else {}}
st["capped_rerun_max_tokens"] = a.max_tokens
st["capped_rerun_tests"] = len(capped)
st["score_at_cap"] = st["total_score"]
if a.label:
    st["label"] = a.label
st["tag"] = Path(a.out).stem
lock, t0 = threading.Lock(), time.time()
keyed = {(r["challenge"], r["language"]): i for i, r in enumerate(st["records"])}


def work(r):
    try:
        new = runner.run_test(suite, cfg, r["challenge"], r["language"])
    except Exception as e:  # noqa: BLE001
        new = {**r, "rerun_error": repr(e)[:300]}
    new["rerun_of_capped"] = True
    with lock:
        st["records"][keyed[(r["challenge"], r["language"])]] = new
        runner.summarise(st)
        st["rerun_elapsed_s"] = round(time.time() - t0, 1)
        Path(a.out).write_text(json.dumps(st, indent=1, ensure_ascii=False))
        last = new["attempts"][-1]
        print(f"  {r['challenge']:22s} {r['language']:10s} {r['score']:.1f} -> {new['score']:.1f}  "
              f"{last.get('finish_reason')} {last.get('completion_tokens')}", flush=True)


try:
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(work, capped))
finally:
    suite.close()
runner.summarise(st)
st["rerun_elapsed_s"] = round(time.time() - t0, 1)
Path(a.out).write_text(json.dumps(st, indent=1, ensure_ascii=False))
print(f"{st['tag']}: {st['score_at_cap']:.1f} at the cap -> {st['total_score']:.1f}/{st['max_score']} "
      f"(still capped: {st['hit_cap']}; {st['rerun_elapsed_s'] / 60:.0f} min)")
