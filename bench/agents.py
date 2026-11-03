#!/usr/bin/env python3
"""agents.py -- agent-fleet test for one server and one context size (2026-10-24, stdlib only).

Up to --agents agents, each with its own nonce'd context of --ctx tokens (speed.py filler; no shared prefix, the worst
case for memory). Agents are primed lazily, one at a time, the first time a cell needs them (prime rows record each
agent's prefill). A cell "N:K" then runs agents 0..N-1 for --rounds agentic turns each: the turn appends a ~1k-token
unique "tool output" plus an instruction and asks for 256 tokens (thinking off, greedy). Agents run independently
(no lockstep). Each cell starts every agent from its primed context. A client-side semaphore lets at most K requests be in flight, so K < N keeps the rest of the fleet
parked in the engine's cache (RAM or SSD) while K work.
Per turn: queue wait, TTFT, cached_tokens, decode, turn latency. Per cell: wall, generated tok/s, turns/min,
p50/p95 latency, cache-hit fraction, SSD-cache bytes (--cache-dir) and the largest "suspended=" the engine logged
(--server-log). A cell that passes --cell-budget seconds stops submitting new turns and is marked capped.
Errors are rows, never dropped. Rows -> --out (kind=prime|turn|cell).
"""
import argparse, json, os, re, statistics, threading, time, urllib.request, uuid
import speed

INSTR = ("Here is the latest tool output. Summarise what changed in two sentences, then say which file you will edit "
         "next and why.")

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--label", required=True)
ap.add_argument("--ctx", required=True); ap.add_argument("--agents", type=int, default=16)
ap.add_argument("--cells", default="4:4,8:8,8:4,16:16,16:4"); ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--max-tokens", type=int, default=256); ap.add_argument("--tool-paras", type=int, default=12)
ap.add_argument("--cell-budget", type=float, default=3600); ap.add_argument("--timeout", type=float, default=7200)
ap.add_argument("--cache-dir"); ap.add_argument("--server-log"); ap.add_argument("--out", required=True)
a = ap.parse_args()

lock = threading.Lock()


def emit(row):
    row = {"label": a.label, "ctx": a.ctx, **row}
    with lock:
        open(a.out, "a").write(json.dumps(row) + "\n")
    return row


def call(messages, max_tokens):
    req = {"model": a.model, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
           "max_tokens": max_tokens, "messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); tf = tl = None; usage = None; text = []; err = None; finish = None; done = False
    try:
        with urllib.request.urlopen(r, timeout=a.timeout) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if line == "data: [DONE]":
                    done = True
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                if ev.get("error"):
                    err = str(ev["error"])[:300]
                usage = ev.get("usage") or usage
                for ch in ev.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                    p = (ch.get("delta") or {}).get("content") or ""
                    if p:
                        now = time.perf_counter(); tf = tf or now; tl = now; text.append(p)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:300]
    # A turn counts only if the stream ended normally: an engine error, a missing [DONE], an unknown finish reason or
    # zero completion tokens is a failed turn (Codex review R04: a partial stream plus an error used to count).
    if not err and not done:
        err = "stream ended without [DONE]"
    if not err and finish not in ("stop", "length", "tool_calls"):
        err = f"finish_reason {finish!r}"
    if not err and usage and not usage.get("completion_tokens"):
        err = "zero completion tokens"
    if err or not usage or tf is None:
        return None, dict(error=err or f"no usage/content (usage={usage})", secs=round(time.perf_counter() - t0, 1))
    ct = usage["completion_tokens"]
    return "".join(text), dict(prompt_tokens=usage["prompt_tokens"], completion_tokens=ct,
                               cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                               ttft_s=round(tf - t0, 2), decode_tps=round((ct - 1) / (tl - tf), 2) if ct > 1 and tl > tf else None,
                               secs=round(time.perf_counter() - t0, 2))


def disk_bytes():
    if not a.cache_dir or not os.path.isdir(a.cache_dir):
        return None
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(a.cache_dir) for f in fs
               if os.path.exists(os.path.join(d, f)))


def log_offset():
    return os.path.getsize(a.server_log) if a.server_log and os.path.exists(a.server_log) else 0


def log_stats(off):
    if not a.server_log or not os.path.exists(a.server_log):
        return {}
    with open(a.server_log, errors="replace") as f:
        f.seek(off); t = f.read()
    sus = [int(x) for x in re.findall(r"suspended=(\d+)", t)]
    return dict(max_suspended=max(sus) if sus else 0, paused_events=t.count("growth paused"))


hist = []  # per agent: message list
np_ = speed.n_para(a.ctx, False)


def prime_to(n):
    while len(hist) < n:
        i = len(hist)
        body = "".join(speed.PARA.format(i=j) for j in range(np_))
        msgs = [{"role": "user", "content": f"[agent {i} · run {uuid.uuid4().hex}]\n{body}{speed.TASK}"}]
        txt, st = call(msgs, 64)
        emit(dict(kind="prime", agent=i, **st))
        print(f"prime agent {i}: {st}", flush=True)
        msgs.append({"role": "assistant", "content": txt or ""})
        hist.append(msgs)


def turn_msg(i, rnd):
    tool = "".join(speed.PARA.format(i=10_000 + 100 * rnd + j) for j in range(a.tool_paras))
    return {"role": "user", "content": f"[tool output · agent {i} · round {rnd} · {uuid.uuid4().hex}]\n{tool}\n{INSTR}"}


def cell(n, k):
    prime_to(n)
    for i in range(n):  # every cell starts each agent from its primed context, so turns don't pile up across cells
        del hist[i][2:]
    sem = threading.Semaphore(k); t0 = time.perf_counter(); off = log_offset(); d0 = disk_bytes()
    rows = []; capped = [False]

    def agent(i):
        for rnd in range(a.rounds):
            if time.perf_counter() - t0 > a.cell_budget:
                capped[0] = True; return
            hist[i].append(turn_msg(i, rnd))
            t_sub = time.perf_counter()
            with sem:
                t_go = time.perf_counter()
                txt, st = call(hist[i], a.max_tokens)
            done = time.perf_counter()
            r = emit(dict(kind="turn", n=n, k=k, agent=i, round=rnd, wait_s=round(t_go - t_sub, 2),
                          latency_s=round(done - t_sub, 2), t_done=round(done - t0, 2), **st))
            rows.append(r)
            if txt is None:
                hist[i].pop(); return
            hist[i].append({"role": "assistant", "content": txt})

    th = [threading.Thread(target=agent, args=(i,)) for i in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    ok = [r for r in rows if "error" not in r]
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    s = emit(dict(kind="cell", n=n, k=k, rounds=a.rounds, wall_s=round(wall, 1), turns=len(rows), ok=len(ok),
                  err=len(rows) - len(ok), errors=[r["error"][:120] for r in rows if "error" in r][:2], capped=capped[0],
                  gen_tps=round(sum(r["completion_tokens"] for r in ok) / wall, 1),
                  turns_per_min=round(len(ok) / wall * 60, 2),
                  lat_p50=q([r["latency_s"] for r in ok], .5), lat_p95=q([r["latency_s"] for r in ok], .95),
                  ttft_p50=q([r["ttft_s"] for r in ok], .5), ttft_p95=q([r["ttft_s"] for r in ok], .95),
                  wait_p50=q([r["wait_s"] for r in ok], .5),
                  decode_p50=q([r["decode_tps"] for r in ok if r.get("decode_tps")], .5),
                  new_tokens=sum(r["prompt_tokens"] - r["cached_tokens"] for r in ok),
                  cache_hit=round(sum(r["cached_tokens"] for r in ok) / max(1, sum(r["prompt_tokens"] for r in ok)), 3),
                  full_hits=sum(1 for r in ok if r["cached_tokens"] >= 0.9 * r["prompt_tokens"]),
                  disk_gib_before=round(d0 / 2**30, 2) if d0 is not None else None,
                  disk_gib_after=round((disk_bytes() or 0) / 2**30, 2) if d0 is not None else None, **log_stats(off)))
    print(json.dumps(s), flush=True)


for c in a.cells.split(","):
    n, k = map(int, c.split(":"))
    if n > a.agents:
        continue
    cell(n, k)
