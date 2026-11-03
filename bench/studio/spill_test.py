#!/usr/bin/env python3
"""spill_test.py -- SSD-tier check (2026-10-30): agents whose contexts no longer fit the KV pool must resume from the
SSD tier with identical output. Primes --agents agents with unique ~--ctx-token contexts one after another (each pushes
the previous ones out of a small pool), then sends each a follow-up. Prints per request: TTFT, cached tokens, and the
reply; --out writes the replies so two runs (reference pool vs small pool + SSD tier) can be compared.
Usage: spill_test.py --url http://127.0.0.1:8399 --out run.json [--agents 3 --ctx 6000]
Restart test (SPILL_PERSIST): --phase prime --state s.json, restart the server, then --phase follow --state s.json."""
import argparse, json, random, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--agents", type=int, default=3); ap.add_argument("--ctx", type=int, default=6000)
ap.add_argument("--max-tokens", type=int, default=64); ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("--phase", choices=["all", "prime", "follow"], default="all"); ap.add_argument("--state")
a = ap.parse_args()
WORDS = ("cache block state agent token kernel tensor buffer queue thread vector matrix layer prefix restore spill disk "
         "memory latency throughput window context schedule batch commit verify draft accept").split()


def context(i):  # ~1.3 tokens per word
    r = random.Random(1000 + i)
    paras = []
    for p in range(int(a.ctx / 1.3 / 60)):
        paras.append(f"Log {i}.{p}: " + " ".join(r.choice(WORDS) for _ in range(60)) + ".")
    return "\n".join(paras)


def chat(messages):
    body = json.dumps({"model": "q", "temperature": 0, "max_tokens": a.max_tokens, "stream": True,
                       "stream_options": {"include_usage": True}, "messages": messages,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(a.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0, ttft, text, usage = time.time(), None, "", {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                piece = c.get("delta", {}).get("content")
                if piece:
                    ttft = ttft or time.time() - t0
                    text += piece
    return text, ttft, usage


convs, out = [], {}
if a.phase == "follow":
    convs = json.load(open(a.state))
for i in range(a.agents if a.phase != "follow" else 0):
    msgs = [{"role": "user", "content": context(i) + "\n\nCount how often the word 'spill' appears in log %d.0." % i}]
    text, ttft, u = chat(msgs)
    msgs.append({"role": "assistant", "content": text})
    convs.append(msgs)
    print(f"prime  agent {i}: prompt {u.get('prompt_tokens')}, cached {u.get('prompt_tokens_details', {}).get('cached_tokens')}, TTFT {ttft:.2f} s")
if a.phase == "prime":
    json.dump(convs, open(a.state, "w"))
    raise SystemExit
for rnd in range(a.rounds):
  for i in range(a.agents):
    msgs = convs[i] + [{"role": "user", "content": "Now name the first three words of log %d.%d, then stop." % (i, rnd + 1)}]
    text, ttft, u = chat(msgs)
    msgs.append({"role": "assistant", "content": text})
    convs[i] = msgs
    out[f"{rnd}.{i}"] = text
    print(f"follow {rnd} agent {i}: prompt {u.get('prompt_tokens')}, cached {u.get('prompt_tokens_details', {}).get('cached_tokens')}, TTFT {ttft:.2f} s | {text[:70]!r}")
json.dump(out, open(a.out, "w"), indent=1)
