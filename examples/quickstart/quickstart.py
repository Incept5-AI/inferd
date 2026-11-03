#!/usr/bin/env python3
"""inferd quickstart: one chat request, one streamed request, and the prefix cache at work.

    python3 quickstart.py            # server on http://127.0.0.1:8300 (INFERD_URL to change)
Python standard library only.
"""
import json, os, time, urllib.request

URL = os.environ.get("INFERD_URL", "http://127.0.0.1:8300/v1")


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


model = json.load(urllib.request.urlopen(URL + "/models"))["data"][0]["id"]
print(f"server {URL}, model {model}\n")

# A long-ish system prompt, so the follow-up turn has a visible amount of cached prefix to reuse.
SYSTEM = "You are a concise assistant for a team of Mac developers. " + \
         "House rules: answer in at most three sentences; prefer concrete commands; never invent flags. " * 20

# 1. A plain chat request. Thinking off ("effort none"): the answer comes straight back.
t = time.time()
r = json.load(post("/chat/completions", {
    "model": model, "max_tokens": 200, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "How do I see which process listens on port 8300?"}]}))
print("== 1. chat (non-streaming)")
print(r["choices"][0]["message"]["content"])
print(f"finish_reason={r['choices'][0]['finish_reason']}  usage={json.dumps(r['usage'])}  {time.time() - t:.2f}s\n")

# 2. A follow-up turn, streamed: the whole conversation so far plus one new question. inferd resumes it from the
#    cached end of request 1 (KV blocks + the recurrent-state snapshot), so only the new tokens are prefilled.
#    Keep the effort the same: the template writes it into the system message, so a new effort is a new prefix.
print("== 2. follow-up turn, streamed")
t, first, usage = time.time(), None, None
resp = post("/chat/completions", {
    "model": model, "max_tokens": 400, "temperature": 0, "stream": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "How do I see which process listens on port 8300?"},
                 r["choices"][0]["message"],
                 {"role": "user", "content": "And how do I stop it without using pkill -f?"}]})
for line in resp:
    line = line.decode().strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    ev = json.loads(line[6:])
    if ev.get("usage"):
        usage = ev["usage"]
    for ch in ev.get("choices", []):
        d = ch["delta"]
        if first is None and (d.get("content") or d.get("reasoning_content")):
            first = time.time() - t
        if d.get("content"):
            print(d["content"], end="", flush=True)
dt = time.time() - t
print(f"\n\ntime to first token {first:.2f}s, total {dt:.2f}s, "
      f"{usage['completion_tokens'] / dt:.0f} tok/s")
print(f"usage={json.dumps(usage)}")
print(f"-> {usage['prompt_tokens_details']['cached_tokens']} of {usage['prompt_tokens']} prompt tokens came from "
      "the prefix cache (turn 1, in 32-token blocks)")
