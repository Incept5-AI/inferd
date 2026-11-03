#!/usr/bin/env python3
"""Cold single-stream speed test against an OpenAI-compatible server (stdlib only).

Same script, same prompts, on every machine, so Ultra-vs-Max rows are comparable:
- a fresh nonce at the very START of every prompt, so no prefix/session cache can hit;
- greedy decoding (temperature 0), so both machines generate the same tokens;
- token counts from the server's `usage`, never from local estimates;
- prefill = prompt_tokens / TTFT, decode = (completion_tokens - 1) / (t_last - t_first), on time.monotonic();
- every stream is validated by ../sse_client.py (normal finish_reason, data: [DONE], consistent usage); a failed or
  partial stream is an error row, never a measurement.

Usage: speed.py --url http://127.0.0.1:8091 --label studio-mtp3 --out results/x.jsonl
"""
import argparse, json, os, statistics, sys, time, uuid

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.dirname(_here)]  # sse_client.py lives in bench/ (or next to this file on the Studio)
import sse_client  # validated SSE: normal finish, [DONE], consistent usage (Codex review 2026-11-02, S04)

PARA = ("Section {i}. The ledger service reconciles settlement batches against the clearing house file. "
        "Each batch carries an identifier, a value date, a currency, a count of payments and a control sum. "
        "When the control sum disagrees with the recomputed total the batch is quarantined, an alert is raised, "
        "and an operator must either correct the source file or approve a manual adjustment with a reason code. ")
TASK = ("\n\nUsing the specification above as context, write a complete Python module that implements the "
        "batch reconciliation it describes, with dataclasses, type hints, error handling and a pytest test suite.")
# Paragraph count per size label. The legacy labels are kept for the 2026-10-13 Ultra-vs-Max rows. Otherwise
# "<n>k" targets n*1024 prompt tokens at ~81.7 tokens per paragraph (measured: 457 paragraphs -> 37,446 tokens);
# "256k" targets 250,000 so the prompt plus 512 output fits the 262,144 window. The real count always comes from usage.
LEGACY = {"0.5k": 6, "4k": 57, "16k": 228, "32k": 457}
TOK_PER_PARA, OVERHEAD = 81.7, 60


def n_para(size, legacy):
    if legacy and size in LEGACY:
        return LEGACY[size]
    target = 250_000 if size == "256k" else int(float(size.rstrip("k")) * 1024)
    return max(1, round((target - OVERHEAD) / TOK_PER_PARA))


def prompt(n_para):
    body = "".join(PARA.format(i=i) for i in range(n_para))
    return f"[run {uuid.uuid4().hex}]\n{body}{TASK}"


EXTRA = {}  # --extra: top-level request fields for engines that ignore chat_template_kwargs (e.g. Ollama reasoning_effort)


def one(url, model, n_para, max_tokens, timeout):
    req = {**EXTRA, "model": model, "stream": True, "stream_options": {"include_usage": True},
           "temperature": 0, "max_tokens": max_tokens,
           "messages": [{"role": "user", "content": prompt(n_para)}],
           "chat_template_kwargs": {"enable_thinking": False}}
    r = sse_client.stream_chat(url, req, timeout=timeout)
    if r["error"]:
        raise RuntimeError(("partial: " if r["partial"] else "") + r["error"])
    t0, t_first, t_last, usage, n_chunks = r["t_sent"], r["t_first"], r["t_last"], r["usage"], len(r["chunks"])
    text = [r["reasoning"], r["text"]]
    if t_first is None:
        raise RuntimeError(f"no content (usage={usage})")
    pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    ttft = t_first - t0
    dec = (ct - 1) / (t_last - t_first) if ct > 1 and t_last > t_first else float("nan")
    return dict(prompt_tokens=pt, completion_tokens=ct, ttft_s=round(ttft, 3),
                prefill_tps=round(pt / ttft, 1), decode_tps=round(dec, 2),
                total_s=round(t_last - t0, 3), chunks=n_chunks, head="".join(text)[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--model", default="default")
    ap.add_argument("--label", required=True)
    ap.add_argument("--sizes", default="0.5k,4k,16k,32k")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--legacy-sizes", action="store_true", help="paragraph counts used for the 2026-10-13 Ultra-vs-Max rows")
    ap.add_argument("--extra", default="{}", help="JSON merged into every request at top level")
    a = ap.parse_args()
    EXTRA.update(json.loads(a.extra))
    rows = []
    with open(a.out, "a") as f:
        for size in a.sizes.split(","):
            for rep in range(a.reps):
                row = dict(label=a.label, size=size, rep=rep, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
                try:
                    row.update(one(a.url, a.model, n_para(size, a.legacy_sizes), a.max_tokens, a.timeout))
                except Exception as e:  # recorded, never silently dropped from the denominator
                    row["error"] = repr(e)
                f.write(json.dumps(row) + "\n"); f.flush(); rows.append(row)
                print(json.dumps(row), flush=True)
    print(f"\n{a.label}: median over {a.reps} cold reps")
    print(f"{'size':>5} {'prompt':>7} {'TTFT s':>7} {'PP tok/s':>9} {'TG tok/s':>9} {'gen':>5} {'err':>4}")
    for size in a.sizes.split(","):
        ok = [r for r in rows if r["size"] == size and "error" not in r]
        err = sum(1 for r in rows if r["size"] == size and "error" in r)
        if not ok:
            print(f"{size:>5} {'-':>7} {'-':>7} {'-':>9} {'-':>9} {'-':>5} {err:>4}"); continue
        m = lambda k: statistics.median(r[k] for r in ok)
        print(f"{size:>5} {m('prompt_tokens'):>7.0f} {m('ttft_s'):>7.2f} {m('prefill_tps'):>9.0f} "
              f"{m('decode_tps'):>9.1f} {m('completion_tokens'):>5.0f} {err:>4}")


if __name__ == "__main__":
    main()
