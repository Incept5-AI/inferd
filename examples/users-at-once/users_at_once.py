#!/usr/bin/env python3
"""Many users arrive at once: N users released together, each with its own long prompt (--prompt tokens, distinct from
the first token) and a fixed output (--gen tokens). A server that prefills one prompt at a time makes the last user
wait for every prompt before it; one that interleaves prefill chunks with decoding and admits by memory budget gets
every user going early. A small version of bench/wall.py (same prompts, same metrics).

    python3 users_at_once.py                                       # inferd on http://127.0.0.1:8300/v1, N = 1 and 4
    python3 users_at_once.py --url http://127.0.0.1:8080/v1 --model <id> --n 1,2,4,8

Per level: time to first token (median, and the last user's = admission), the decode rate while every user is
generating, and end-to-end throughput (all completion tokens / (last token - release)). Greedy, thinking off,
ignore_eos (servers that ignore it may stop early: the 'gen' column shows what each user really got). Token counts
come from the server's usage. Python standard library only.
"""
import argparse, json, os, statistics as st, threading, time, urllib.error, urllib.request, uuid

PARA = ("Section {i}. The ledger service reconciles settlement batches against the clearing house file. "
        "Each batch carries an identifier, a value date, a currency, a count of payments and a control sum. "
        "When the control sum disagrees with the recomputed total the batch is quarantined, an alert is raised, "
        "and an operator must either correct the source file or approve a manual adjustment with a reason code. ")
TASK = ("\n\nUsing the specification above as context, write a complete Python module that implements the "
        "batch reconciliation it describes, with dataclasses, type hints, error handling and an exhaustive pytest "
        "test suite, then document every function in detail.")
TOK_PER_PARA, OVERHEAD = 81.7, 60


def user(a, model, release, rec):
    n_para = max(1, round((a.prompt - OVERHEAD) / TOK_PER_PARA))
    text = f"[user {uuid.uuid4().hex}]\n" + "".join(PARA.format(i=i) for i in range(n_para)) + TASK
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": a.gen, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    chunks, usage = [], {}
    try:
        while time.time() < release:
            time.sleep(0.001)
        req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                usage = ev.get("usage") or usage
                for c in ev.get("choices") or []:
                    d = c.get("delta") or {}
                    s = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                    if s:
                        chunks.append((time.time(), len(s)))
    except urllib.error.HTTPError as e:  # recorded, never silently dropped from a denominator
        rec["error"] = f"HTTP {e.code}: {e.read()[:160].decode(errors='replace')}"
    except Exception as e:
        rec["error"] = repr(e)[:200]
    rec.update(chunks=chunks, prompt=usage.get("prompt_tokens"), gen=usage.get("completion_tokens"))


def level(a, model, n):
    release = time.time() + 0.5
    recs = [{} for _ in range(n)]
    th = [threading.Thread(target=user, args=(a, model, release, recs[i])) for i in range(n)]
    [t.start() for t in th]; [t.join() for t in th]
    bad = [r for r in recs if r.get("error") or not r.get("chunks") or not r.get("gen")]
    out = {"label": a.label, "n": n, "errors": len(bad)}
    if bad:
        out["error_samples"] = [r.get("error") for r in bad][:3]
        return out
    firsts = [r["chunks"][0][0] - release for r in recs]
    lasts = [r["chunks"][-1][0] - release for r in recs]
    w0, w1, in_w = max(firsts), min(lasts), 0.0  # the window in which every user is generating
    for r in recs:  # tokens per chunk, apportioned by characters (a chunk may carry several speculated tokens)
        chars = sum(c for _, c in r["chunks"]) or 1
        in_w += sum(r["gen"] * c / chars for t, c in r["chunks"] if w0 <= t - release <= w1)
    out.update({"prompt": st.median(r["prompt"] for r in recs), "gen_min": min(r["gen"] for r in recs),
                "gen_med": st.median(r["gen"] for r in recs), "ttft_med_s": round(st.median(firsts), 2),
                "last_first_token_s": round(max(firsts), 2),
                "together_tok_s": round(in_w / (w1 - w0), 1) if w1 > w0 else None,
                "e2e_tok_s": round(sum(r["gen"] for r in recs) / max(lasts), 1), "wall_s": round(max(lasts), 1),
                "first_tokens_s": sorted(round(f, 1) for f in firsts)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=os.environ.get("INFERD_URL", "http://127.0.0.1:8300/v1"))
    ap.add_argument("--model", help="model id (default: the first one /v1/models lists)")
    ap.add_argument("--n", default="1,4", help="user counts to run, one level after another")
    ap.add_argument("--prompt", type=int, default=8192, help="tokens per user prompt (approximate)")
    ap.add_argument("--gen", type=int, default=1024, help="tokens generated per user")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--label", default="", help="engine name for the summary rows")
    ap.add_argument("--out", help="append JSON rows here")
    a = ap.parse_args()
    if not a.model:
        with urllib.request.urlopen(a.url + "/models", timeout=30) as r:
            a.model = json.load(r)["data"][0]["id"]
    print(f"users released together, {a.prompt // 1024}k-token prompts each, {a.gen} tokens generated each, "
          f"on {a.url} ({a.model})\n")
    print(f"{'users':>5} {'prompt':>7} {'gen':>9} {'TTFT med':>9} {'last user 1st tok':>17} {'all generating':>14} "
          f"{'end to end':>10} {'wall':>7}")
    for n in [int(x) for x in a.n.split(",")]:
        o = level(a, a.model, n)
        if o["errors"]:
            print(f"{n:>5}  {o['errors']} of {n} users failed: {o['error_samples']}")
        else:
            gen = f"{o['gen_med']:.0f}" if o["gen_min"] == o["gen_med"] else f"{o['gen_min']}-{o['gen_med']:.0f}"
            tog = f"{o['together_tok_s']} tok/s" if o["together_tok_s"] else "never"
            print(f"{n:>5} {o['prompt']:>7.0f} {gen:>9} {o['ttft_med_s']:>7.1f} s {o['last_first_token_s']:>15.1f} s "
                  f"{tog:>14} {o['e2e_tok_s']:>6.1f} t/s {o['wall_s']:>5.0f} s")
            print(f"{'':>5} first tokens at {o['first_tokens_s']} s")
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
