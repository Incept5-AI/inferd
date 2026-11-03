#!/usr/bin/env python3
"""A small research agent: the model calls a web_search tool (Wikipedia, no API key) until it can answer.
Tight budget: max_context 8192, reasoning effort low.

    python3 web_search_agent.py ["your question"]     # server on http://127.0.0.1:8300 (INFERD_URL to change)
Python standard library only. Needs inferd's front end with tool-call parsing (engine/frontend/serve.py, Nov 2026).
"""
import json, os, sys, time, urllib.error, urllib.parse, urllib.request

URL = os.environ.get("INFERD_URL", "http://127.0.0.1:8300/v1")
QUESTION = sys.argv[1] if len(sys.argv) > 1 else \
    "How tall is the Burj Khalifa, and which building held the title of world's tallest before it?"
MAX_CONTEXT, EFFORT, MAX_STEPS = 8192, "low", 5
UA = {"User-Agent": "inferd-examples/0.1 (local demo)"}

TOOLS = [{"type": "function", "function": {
    "name": "web_search",
    "description": "Search Wikipedia. Returns the top results with a short summary of each article.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "search terms"}},
                   "required": ["query"]}}}]


def web_search(query, n=3):
    """Wikipedia search + page summaries. Network failures come back as an error the model can read."""
    try:
        q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query, "format": "json",
                                    "srlimit": n})
        hits = json.load(urllib.request.urlopen(urllib.request.Request(
            "https://en.wikipedia.org/w/api.php?" + q, headers=UA), timeout=10))["query"]["search"]
        out = []
        for h in hits:
            title = urllib.parse.quote(h["title"].replace(" ", "_"))
            s = json.load(urllib.request.urlopen(urllib.request.Request(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + title, headers=UA), timeout=10))
            out.append({"title": h["title"], "url": s.get("content_urls", {}).get("desktop", {}).get("page"),
                        "summary": s.get("extract", "")[:1200]})
        return {"results": out} if out else {"results": [], "note": "no matches"}
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
        return {"error": f"search unavailable: {e}"}


def chat(messages):
    body = {"model": "qwen3.8-27b", "messages": messages, "tools": TOOLS, "temperature": 0, "max_tokens": 2048,
            "max_context": MAX_CONTEXT, "chat_template_kwargs": {"reasoning_effort": EFFORT}}
    req = urllib.request.Request(URL + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {json.load(e)['error']['message']}")


messages = [{"role": "system", "content": "You are a research assistant. Use web_search to check facts before "
                                          "answering. Answer in two or three sentences and name your sources."},
            {"role": "user", "content": QUESTION}]
print(f"Q: {QUESTION}\n   (max_context {MAX_CONTEXT}, effort {EFFORT})\n")
t0 = time.time()
for step in range(1, MAX_STEPS + 1):
    r = chat(messages)
    msg, usage = r["choices"][0]["message"], r["usage"]
    ctx = f"[context {usage['total_tokens']}/{MAX_CONTEXT}, cached {usage['prompt_tokens_details']['cached_tokens']}]"
    if not msg.get("tool_calls"):
        print(f"step {step}: final answer {ctx}  finish_reason={r['choices'][0]['finish_reason']}\n")
        print(msg["content"])
        break
    print(f"step {step}: model calls {len(msg['tool_calls'])} tool(s) {ctx}  finish_reason=tool_calls")
    messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg["tool_calls"]})
    for call in msg["tool_calls"]:
        args = json.loads(call["function"]["arguments"])
        print(f"  -> {call['function']['name']}({json.dumps(args)})")
        result = web_search(**args) if call["function"]["name"] == "web_search" else {"error": "unknown tool"}
        shown = result.get("error") or ", ".join(x["title"] for x in result["results"]) or "no results"
        print(f"  <- {shown}")
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
else:
    print(f"stopped after {MAX_STEPS} steps without an answer")
print(f"\n{time.time() - t0:.1f}s total")
