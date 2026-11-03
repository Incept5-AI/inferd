#!/usr/bin/env python3
"""cascade_check.py -- same shared-prefix fan-out + one follow-up turn against a server, printing every reply, so two
servers (CASCADE=0 / 1) can be compared token for token. Fixed document and tasks (seeded), greedy, 96 tokens.
Usage: cascade_check.py URL N SHARED_TOKENS > replies.json"""
import json, sys, threading, urllib.request
url, n, shared = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
PARA = "Module {i}: the settlement service reconciles batch {i}, checks currency and amount, and logs adjustments.\n"
doc = "[project fixed-0001]\n" + "".join(PARA.format(i=i) for i in range(shared // 22))
def call(msgs):
    body = {"model": "q", "temperature": 0, "max_tokens": 96, "messages": msgs, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800)
    return json.load(r)["choices"][0]["message"]["content"]
convs = [[{"role": "system", "content": "Repository:\n" + doc}, {"role": "user", "content": f"Agent {k}: summarise module {k * 7}."}] for k in range(n)]
out = {}
def run(k, key):
    out[f"{key}.{k}"] = call(convs[k])
for phase in ("fan", "turn"):
    ts = [threading.Thread(target=run, args=(k, phase)) for k in range(n)]
    [t.start() for t in ts]; [t.join() for t in ts]
    for k in range(n):
        convs[k] += [{"role": "assistant", "content": out[f"{phase}.{k}"]}, {"role": "user", "content": f"Now module {k * 7 + 1}, briefly."}]
print(json.dumps(out, indent=1))
