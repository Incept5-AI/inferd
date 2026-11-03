#!/usr/bin/env python3
"""shared_prefix.py -- agent fleet over one shared context (2026-10-30, stdlib only).

N agents work against the same large document (a repository, a contract set: --shared tokens of filler text, one
nonce per run so no earlier run's cache helps), each with its own task. Two phases, timed:
  fan-out  all N send their first turn at once: shared document + own task (~200 tokens), --max-tokens out.
           An engine that shares the document's prefill among the N pays for it once; one that does not, N times.
  turns    each agent then runs --turns more turns on its own clock: its last reply, a unique ~--tool-tokens "tool
           output", an instruction; --pause seconds of "tool time" before each.
Per phase: wall, TTFT median / max, generated tokens per second, cached prompt tokens (when the engine reports them).
Rows -> --out (JSON lines). Usage:
  shared_prefix.py --url http://127.0.0.1:8300 --model M --label L --shared 32k --n 8 --turns 2 --out rows.jsonl
"""
import argparse, json, random, statistics, threading, time, urllib.request, uuid

PARA = ("Module {i}: the settlement service reconciles batch {i} against the clearing house file, checks each "
        "payment's currency and amount, records adjustments with the operator who approved them, and raises an "
        "exception report when the control sum differs. Its tests cover partial files, duplicate references and "
        "late corrections.\n")
TOK_PER_PARA = 58.0  # measured roughly; the real count comes from usage
WORDS = "ledger batch payment currency amount control report adjust approve settle verify retry".split()


def call(url, model, messages, max_tokens, timeout):
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
            "max_tokens": max_tokens, "messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; text = []; usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    text.append(piece)
    gen = usage.get("completion_tokens", len(text))
    return {"text": "".join(text), "ttft": (first or time.perf_counter()) - t0,
            "decode_tps": (gen - 1) / (last - first) if first and last and last > first else None, "gen": gen,
            "prompt": usage.get("prompt_tokens"), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--shared", default="32k"); ap.add_argument("--n", default="8")
    ap.add_argument("--turns", type=int, default=2); ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tool-tokens", type=int, default=1000); ap.add_argument("--pause", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=7200); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    target = float(a.shared.rstrip("k")) * 1024
    for n in [int(x) for x in a.n.split(",")]:
        run = uuid.uuid4().hex[:12]
        doc = f"[project {run}]\n" + "".join(PARA.format(i=i) for i in range(int(target / TOK_PER_PARA)))
        system = "You are a coding agent working on this repository:\n\n" + doc
        convs = [[{"role": "system", "content": system},
                  {"role": "user", "content": f"Agent {k} ({uuid.uuid4().hex[:8]}): summarise how module {k * 7 % 97} "
                                              f"handles late corrections, in three sentences."}] for k in range(n)]
        rows = [None] * n

        def fan(k):
            try:
                rows[k] = call(a.url, a.model, convs[k], a.max_tokens, a.timeout)
            except Exception as e:  # errors are rows
                rows[k] = {"error": repr(e)}
        t0 = time.perf_counter()
        ts = [threading.Thread(target=fan, args=(k,)) for k in range(n)]
        [t.start() for t in ts]; [t.join() for t in ts]
        wall = time.perf_counter() - t0
        ok = [r for r in rows if r and "error" not in r]
        res = {"label": a.label, "phase": "fanout", "shared": a.shared, "n": n, "ok": len(ok), "err": n - len(ok),
               "wall_s": round(wall, 2), "ttft_med": round(statistics.median(r["ttft"] for r in ok), 2) if ok else None,
               "ttft_max": round(max(r["ttft"] for r in ok), 2) if ok else None,
               "gen_tps": round(sum(r["gen"] for r in ok) / wall, 1),
               "prompt_tokens": sum(r["prompt"] or 0 for r in ok), "cached_tokens": sum(r["cached"] or 0 for r in ok)}
        print(json.dumps(res)); open(a.out, "a").write(json.dumps(res) + "\n")
        for k, r in enumerate(rows):
            if r and "error" not in r:
                convs[k].append({"role": "assistant", "content": r["text"]})

        turn_rows = []
        lock = threading.Lock()

        def agent(k):
            rng = random.Random(k)
            for t in range(a.turns):
                if a.pause:
                    time.sleep(a.pause)
                tool = " ".join(rng.choice(WORDS) for _ in range(int(a.tool_tokens * 0.75)))
                convs[k].append({"role": "user", "content": f"Tool output ({t}):\n{tool}\n\nWhat do you change next, and why?"})
                try:
                    r = call(a.url, a.model, convs[k], a.max_tokens, a.timeout)
                except Exception as e:
                    r = {"error": repr(e)}
                with lock:
                    turn_rows.append(r)
                if "error" in r:
                    return
                convs[k].append({"role": "assistant", "content": r["text"]})
        t0 = time.perf_counter()
        ts = [threading.Thread(target=agent, args=(k,)) for k in range(n)]
        [t.start() for t in ts]; [t.join() for t in ts]
        wall = time.perf_counter() - t0
        ok = [r for r in turn_rows if "error" not in r]
        q = sorted(r["ttft"] for r in ok)
        res = {"label": a.label, "phase": "turns", "shared": a.shared, "n": n, "turns": a.turns, "ok": len(ok),
               "err": len(turn_rows) - len(ok), "wall_s": round(wall, 2), "turns_per_min": round(len(ok) / wall * 60, 2),
               "gen_tps": round(sum(r["gen"] for r in ok) / wall, 1),
               "ttft_med": round(statistics.median(q), 2) if q else None,
               "ttft_p95": round(q[int(0.95 * (len(q) - 1))], 2) if q else None,
               "decode_med": round(statistics.median(r["decode_tps"] for r in ok if r["decode_tps"]), 1) if ok else None,
               "prompt_tokens": sum(r["prompt"] or 0 for r in ok), "cached_tokens": sum(r["cached"] or 0 for r in ok)}
        print(json.dumps(res)); open(a.out, "a").write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
