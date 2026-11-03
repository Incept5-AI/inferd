#!/usr/bin/env python3
"""Per-agent budgets: each agent has its own context cap (max_context), reasoning effort and priority, all running
concurrently on one inferd server. Then two agents break their caps on purpose.

    python3 agent_budgets.py         # server on http://127.0.0.1:8300 (INFERD_URL to change)
Python standard library only.
"""
import json, os, threading, time, urllib.error, urllib.request

URL = os.environ.get("INFERD_URL", "http://127.0.0.1:8300/v1")
SERVER_MAX = int(os.environ.get("SERVER_MAX_CONTEXT", 32768))  # run.sh's last argument


def effort(level):
    """'none' = thinking off; otherwise the template's reasoning_effort (low | medium | xhigh, the default)."""
    return {"enable_thinking": False} if level == "none" else {"reasoning_effort": level}


def ask(agent, prompt, max_tokens):
    """One request with this agent's policy. -> (status, response JSON or error JSON, seconds)."""
    body = {"model": "qwen3.8-27b", "temperature": 0, "max_tokens": max_tokens,
            "max_context": agent["max_context"], "priority": agent["priority"],
            "chat_template_kwargs": effort(agent["effort"]),
            "messages": [{"role": "system", "content": agent["role"]}, {"role": "user", "content": prompt}]}
    t = time.time()
    req = urllib.request.Request(URL + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return 200, json.load(urllib.request.urlopen(req, timeout=900)), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, json.load(e), time.time() - t


ARTICLE = ("Apple Silicon Macs share one pool of memory between CPU and GPU, so a 27-billion-parameter model "
           "quantised to 4 bits (about 15 GB) fits next to a large key-value cache. Serving many agents at once is "
           "a scheduling problem: every agent's context must be kept, each step should decode for all of them "
           "together, and a follow-up turn should not re-read the context it already has. ") * 12

AGENTS = [  # name, cap on prompt + output, effort, priority (higher = preempted last), task, max_tokens
    dict(name="summariser", max_context=16384, effort="none", priority=0, max_tokens=300,
         role="You summarise documents in two sentences.", task="Summarise:\n\n" + ARTICLE),
    dict(name="coder", max_context=SERVER_MAX, effort="medium", priority=5, max_tokens=8000,
         role="You are a senior Python engineer. Reply with one code block and nothing else.",
         task="Write a function lru_cache_stats(maxsize) decorator that caches results and exposes "
              ".hits and .misses attributes on the wrapped function. Include a 3-line usage example."),
    dict(name="searcher-1", max_context=8192, effort="none", priority=-5, max_tokens=200,
         role="You answer factual questions in one sentence.", task="What is the capital of Australia?"),
    dict(name="searcher-2", max_context=8192, effort="low", priority=-5, max_tokens=1000,
         role="You answer factual questions in one sentence.", task="Which planet has the shortest day?"),
]

print(f"{len(AGENTS)} agents in parallel on {URL}\n")
results = {}
t0 = time.time()
threads = [threading.Thread(target=lambda a=a: results.__setitem__(a["name"], ask(a, a["task"], a["max_tokens"])))
           for a in AGENTS]
for th in threads:
    th.start()
for th in threads:
    th.join()
wall = time.time() - t0

print(f"{'agent':<11} {'max_context':>11} {'effort':>7} {'prio':>4} {'prompt':>6} {'output':>6} {'finish':>7} {'secs':>5}")
for a in AGENTS:
    status, r, dt = results[a["name"]]
    u, ch = r["usage"], r["choices"][0]
    print(f"{a['name']:<11} {a['max_context']:>11} {a['effort']:>7} {a['priority']:>4} {u['prompt_tokens']:>6} "
          f"{u['completion_tokens']:>6} {ch['finish_reason']:>7} {dt:>5.1f}")
print(f"wall time for all four: {wall:.1f}s\n")
for a in AGENTS:
    msg = results[a["name"]][1]["choices"][0]["message"]
    text = msg["content"].strip()
    think = len(msg.get("reasoning_content") or "")
    print(f"--- {a['name']}" + (f" (reasoned {think} chars first)" if think else ""))
    print(text if len(text) < 900 else text[:900] + "\n[...]")
    print()

# Breaking the caps on purpose.
print("=== over budget")
searcher = AGENTS[2]
status, r, dt = ask(searcher, "Summarise:\n\n" + ARTICLE * 10, 200)  # ~10k tokens into an 8k agent
print(f"searcher-1 (max_context 8192) sent a ~10k-token document -> HTTP {status}: "
      f"{r['error'].get('code')}: {r['error']['message']}")

tiny = dict(searcher, name="tiny", max_context=120)
status, r, dt = ask(tiny, "Write a 300-word story about a lighthouse.", 2000)
u = r["usage"]
print(f"tiny (max_context 120) asked for 300 words -> HTTP {status}, finish_reason "
      f"'{r['choices'][0]['finish_reason']}' after {u['completion_tokens']} tokens "
      f"(prompt {u['prompt_tokens']} + output {u['completion_tokens']} = {u['total_tokens']} <= 120)")
print("   ..." + r["choices"][0]["message"]["content"][-120:].replace("\n", " "))
