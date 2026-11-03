#!/usr/bin/env python3
"""Agent contexts: save a long context once, fork agents from it, park and resume agents, run one-off agents that
leave nothing behind. Uses inferd's agent-context API (docs/CONTEXTS.md).

    python3 agent_contexts.py [--url http://127.0.0.1:8300/v1] [--doc 8192] [--agents 4]

Steps:
  1. POST /v1/contexts: a ~--doc token service catalogue, effort none, persist "ssd" (when the server has an SSD
     tier): computed once, pinned in the prefix cache.
  2. --agents agents launched from it at once ("inferd": {"context": id, "session": ...}), each with its own task.
  3. A follow-up turn for agent 0 (its session resumes at the end of its own last turn).
  4. POST /v1/contexts/{id}/offload: the context leaves GPU memory for the SSD tier; a new agent launched from it
     restores it instead of recomputing it.
  5. POST /v1/sessions/{agent}/offload: park agent 1; its next turn restores from SSD.
  6. 20 one-off agents with "cache": "drop": the cache's snapshot count does not grow.
  7. DELETE everything.
Greedy, thinking off. Python standard library only.
"""
import argparse, json, os, random, threading, time, urllib.error, urllib.request, uuid

TEAMS = "Payments Ledger Identity Search Billing Risk Messaging Storage Platform Pricing Fraud Onboarding".split()
FIRST = "Priya Tomas Aiko Mateo Ingrid Kwame Lucia Omar Freya Dmitri Amara Jonas Mei Rafael Sanna Theo".split()
LAST = "Nair Okafor Lindqvist Moreau Haddad Tanaka Kowalski Ferreira Brennan Osei Varga Rossi Iyer Novak".split()


def catalogue(tokens, seed=7):
    rng, out = random.Random(seed), []
    for i in range(max(8, tokens // 34)):  # ~34 tokens per line
        out.append(f"Service S-{i:03d} is owned by the {rng.choice(TEAMS)} team; its on-call lead is "
                   f"{rng.choice(FIRST)} {rng.choice(LAST)} and its runbook is RB-{rng.randint(1000, 9999)}.\n")
    return "".join(out)


def call(url, method, path, body=None):
    req = urllib.request.Request(url + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": json.loads(e.read() or b"{}").get("error", e.code)}


def chat(url, messages, inferd, max_tokens=60):
    """Streamed; -> (text, ttft, usage)."""
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "inferd": inferd}
    t0, first, text, usage = time.time(), None, [], {}
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if "error" in ev:
                raise RuntimeError(ev["error"])
            usage = ev.get("usage") or usage
            for c in ev.get("choices") or []:
                s = (c.get("delta") or {}).get("content") or ""
                if s and first is None:
                    first = time.time()
                text.append(s)
    return "".join(text).strip(), (first or time.time()) - t0, usage


def line(tag, text, ttft, usage):
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"  {tag:10s} TTFT {ttft:5.2f} s  prompt {usage.get('prompt_tokens'):6}  cached {cached:6}  | {text[:70]}")


def resident(url, cid):
    v = call(url, "GET", f"/contexts/{cid}")["variants"][0]
    return f"pinned={v['pinned']} gpu={v['resident']['gpu']} ssd={v['resident']['ssd']} ({v['kv_mib']} MiB of KV)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("INFERD_URL", "http://127.0.0.1:8300/v1"))
    ap.add_argument("--doc", type=int, default=8192)
    ap.add_argument("--agents", type=int, default=4)
    a = ap.parse_args()
    url, nonce = a.url.rstrip("/"), uuid.uuid4().hex[:8]
    ssd = "ssd" in call(url, "GET", "/cache")
    cid, agents = f"catalogue-{nonce}", [f"agent-{nonce}-{k}" for k in range(a.agents)]

    print(f"1. save a context ({a.doc // 1024}k-token catalogue, effort none, persist {'ssd' if ssd else 'memory'})")
    system = f"[{nonce}] You answer questions about this service catalogue, briefly.\n\n" + catalogue(a.doc)
    r = call(url, "POST", "/contexts", {"name": cid, "system": system, "effort": "none",
                                                "persist": "ssd" if ssd else "memory"})
    if "error" in r:
        raise SystemExit(f"  failed: {r['error']}")
    v = r["variants"][0]
    print(f"  {r['id']}: {v['tokens']} tokens, computed in {v['prefill_s']} s; {resident(url, cid)}")

    print(f"\n2. launch {a.agents} agents from it at once, each with its own task")
    out, history = [None] * a.agents, {}

    def agent(k):
        msgs = [{"role": "user", "content": f"Who is the on-call lead of S-{10 + 7 * k:03d}? One sentence."}]
        text, ttft, usage = chat(url, msgs, {"context": cid, "session": agents[k], "effort": "none"})
        out[k] = (text, ttft, usage)
        history[k] = msgs + [{"role": "assistant", "content": text}]
    t0 = time.time()
    th = [threading.Thread(target=agent, args=(k,)) for k in range(a.agents)]
    [t.start() for t in th]
    [t.join() for t in th]
    for k, (text, ttft, usage) in enumerate(out):
        line(f"agent {k}", text, ttft, usage)
    print(f"  all {a.agents} answered in {time.time() - t0:.2f} s")

    print("\n3. two more turns for agent 0: a long answer, then a question about it (resumes after its own answer)")
    msgs = history[0] + [{"role": "user", "content": "List twenty services of that lead's team with their runbooks, "
                                                        "one per line."}]
    text, ttft, usage = chat(url, msgs, {"context": cid, "session": agents[0], "effort": "none"}, 400)
    line("agent 0", text.replace("\n", " / "), ttft, usage)
    msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": "Which of those has the highest "
                                                                                    "number? One sentence."}]
    text, ttft, usage = chat(url, msgs, {"context": cid, "session": agents[0], "effort": "none"})
    line("agent 0", text, ttft, usage)

    if ssd:
        print("\n4. offload the context to SSD, then launch a new agent from it")
        call(url, "POST", f"/contexts/{cid}/offload")
        print(f"  after offload: {resident(url, cid)}")
        text, ttft, usage = chat(url, [{"role": "user", "content": "Which team owns S-005? One sentence."}],
                                 {"context": cid, "session": f"agent-{nonce}-new", "effort": "none"})
        line("new agent", text, ttft, usage)
        print(f"  after the launch: {resident(url, cid)}")

        print("\n5. park agent 1 (its latest turn goes to SSD), then resume it")
        print(" ", call(url, "POST", f"/sessions/{agents[1]}/offload"))
        msgs = history[1] + [{"role": "user", "content": "Which team owns that service? One sentence."}]
        text, ttft, usage = chat(url, msgs, {"context": cid, "session": agents[1], "effort": "none"})
        line("agent 1", text, ttft, usage)
    else:
        print("\n4-5. skipped: the server has no SSD tier (start it with SPILL_DIR=...)")

    print("\n6. 20 one-off agents with \"cache\": \"drop\"")
    before = call(url, "GET", "/cache")
    for i in range(20):
        chat(url, [{"role": "user", "content": f"[{nonce}-{i}] " + catalogue(400, seed=100 + i) +
                    "Reply with one word: the team of S-000."}], {"cache": "drop", "effort": "none"}, 4)
    after = call(url, "GET", "/cache")
    print(f"  snapshots {before['snapshots']} -> {after['snapshots']}, cached KV blocks {before['kv_cached']} -> "
          f"{after['kv_cached']}")

    print("\n7. clean up")
    for s in agents + [f"agent-{nonce}-new"]:
        call(url, "DELETE", f"/sessions/{s}")
    print(" ", call(url, "DELETE", f"/contexts/{cid}"))
    c = call(url, "GET", "/cache")
    print(f"  cache now: {c['snapshots']} snapshots, {c['kv_cached']} cached KV blocks, {len(c['contexts'])} contexts")


if __name__ == "__main__":
    main()
