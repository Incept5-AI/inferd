#!/usr/bin/env python3
"""Agent hand-off: a lead agent reads a long document and works out an analysis; its whole conversation (document +
its turns) is then saved as a context and handed to agents with other settings, which start from the lead's cached
state instead of re-reading. At the end everything is deleted and the cache is checked to be empty of it.

    python3 agent_handoff.py [--url http://127.0.0.1:8300/v1] [--doc 8192] [--no-variant]

Steps (each one checks itself and prints PASS/FAIL; the exit code is 1 if any check failed):
  1. The lead (effort medium, max_context 32k, priority 5, a session) reads a --doc token incident catalogue and
     finds the service in an active incident; a second turn drafts a plan.
  2. POST /v1/contexts with the lead's messages (system, document, both turns): the lead's turns are already in the
     cache, so saving computes almost nothing.
  3. Three agents launched from it at once, each with its own settings: a writer (effort none, max_context 16k,
     priority 0, a session), a critic (effort medium: it reasons, priority 3, a session), a fact-checker (effort none,
     "cache": "drop": a one-off). Each should start in a fraction of the lead's reading time, with the context cached.
  4. (unless --no-variant) a reviewer at effort low: the effort is written into the chat template's prefix, so the
     first low agent computes a low variant of the context once (if it renders differently); a second one reuses it.
  5. DELETE the context and every session; GET /v1/cache must show the pinned blocks and the sessions released, and
     the same prompt sent fresh must find (almost) nothing cached.
Greedy. Python standard library only.
"""
import argparse, json, os, random, re, sys, threading, time, urllib.error, urllib.request, uuid

TEAMS = "Payments Ledger Identity Search Billing Risk Messaging Storage Platform Pricing Fraud Onboarding".split()
FIRST = "Priya Tomas Aiko Mateo Ingrid Kwame Lucia Omar Freya Dmitri Amara Jonas Mei Rafael Sanna Theo".split()
LAST = "Nair Okafor Lindqvist Moreau Haddad Tanaka Kowalski Ferreira Brennan Osei Varga Rossi Iyer Novak".split()
STORES = ["Postgres cluster pg-{}", "Redis shard rd-{}", "Kafka topic kt-{}", "S3 bucket s3-{}"]
FAILS = []


def check(ok, what):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        FAILS.append(what)
    return ok


def document(tokens, seed):
    """-> (text, facts of the service in the incident): a service catalogue of ~tokens tokens; one service is in an
    active SEV-1 incident, and its entry is the only one that says so."""
    rng = random.Random(seed)
    n = max(12, tokens // 48)  # ~48 tokens per entry (Qwen3.8 tokenizer)
    hot = rng.randrange(n // 3, n - 3)
    out, facts = [], None
    for i in range(n):
        f = dict(id=f"S-{i:03d}", team=rng.choice(TEAMS), lead=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                 store=rng.choice(STORES).format(rng.randint(1, 99)), rb=f"RB-{rng.randint(1000, 9999)}",
                 err=round(rng.uniform(0.1, 0.9), 1))
        status = (f"STATUS: SEV-1 INCIDENT OPEN since 03:12, error rate {rng.randint(31, 48)}% and rising"
                  if i == hot else f"status: healthy, error rate {f['err']}%")
        if i == hot:
            facts = f
        out.append(f"{f['id']}: owned by {f['team']}, on-call {f['lead']}, state in {f['store']}, runbook {f['rb']}; "
                   f"{status}.\n")
    return "".join(out), facts


def call(url, method, path, body=None):
    req = urllib.request.Request(url + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": json.loads(e.read() or b"{}").get("error", e.code)}


def chat(url, messages, inferd, max_tokens):
    """Streamed. TTFT is the first token of any kind (reasoning or content).
    -> dict(text, think, ttft, wall, prompt, cached, finish) or dict(error)."""
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "inferd": inferd}
    t0, first, text, think, usage, finish = time.time(), None, [], 0, {}, None
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                if "error" in ev:
                    return {"error": str(ev["error"])[:200]}
                usage = ev.get("usage") or usage
                for c in ev.get("choices") or []:
                    d = c.get("delta") or {}
                    s, rs = d.get("content") or "", d.get("reasoning_content") or ""
                    if (s or rs) and first is None:
                        first = time.time()
                    text.append(s)
                    think += len(rs)
                    finish = c.get("finish_reason") or finish
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"}
    return {"text": "".join(text).strip(), "think": think, "ttft": (first or time.time()) - t0,
            "wall": time.time() - t0, "prompt": usage.get("prompt_tokens") or 0, "finish": finish,
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0}


def show(tag, r):
    if "error" in r:
        print(f"  {tag:13s} ERROR {r['error']}")
        return
    pct = 100 * r["cached"] / max(1, r["prompt"])
    think = f" (reasoned {r['think']} chars)" if r["think"] else ""
    print(f"  {tag:13s} TTFT {r['ttft']:6.2f} s  prompt {r['prompt']:6}  cached {r['cached']:6} ({pct:3.0f}%)  "
          f"wall {r['wall']:6.1f} s{think} | {r['text'][:60].replace(chr(10), ' / ')}")


def cache_line(c):
    return (f"pinned {c.get('kv_pinned')} blocks, cached {c.get('kv_cached')} blocks, snapshots {c.get('snapshots')}, "
            f"session snapshots {c.get('session_snapshots')}, sessions {c.get('sessions')}, "
            f"contexts {len(c.get('contexts') or [])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=os.environ.get("INFERD_URL", "http://127.0.0.1:8300/v1"))
    ap.add_argument("--doc", type=int, default=8192, help="document size in tokens (approximate; < 14k so the "
                                                          "16k writer fits)")
    ap.add_argument("--lead-max-tokens", type=int, default=3000, help="the lead's budget per turn (reasoning + answer)")
    ap.add_argument("--no-variant", action="store_true", help="skip step 4 (the effort-low variant)")
    a = ap.parse_args()
    url, nonce = a.url.rstrip("/"), uuid.uuid4().hex[:8]
    cid = f"handoff-{nonce}"
    lead_s = f"lead-{nonce}"
    sessions = [lead_s]
    doc, hot = document(a.doc, seed=int(nonce, 16) % 1000)
    base = call(url, "GET", "/cache")
    if "error" in base:
        sys.exit(f"GET /v1/cache failed ({base['error']}): this demo needs inferd's agent-context API")
    print(f"server cache at start: {cache_line(base)}")

    # 1. The lead reads the document and works.
    print(f"\n1. the lead (effort medium, max_context 32k, priority 5) reads a ~{a.doc // 1024}k-token catalogue")
    lead = {"effort": "medium", "max_context": 32768, "priority": 5, "session": lead_s}
    msgs = [{"role": "system", "content": f"[{nonce}] You are the lead incident analyst. Be concise and exact."},
            {"role": "user", "content": "Service catalogue:\n\n" + doc + "\nExactly one service in this catalogue has "
             "an open SEV-1 incident. Which one is it? Give its id, owning team, on-call lead, state store and "
             "runbook, in one or two sentences."}]
    r1 = chat(url, msgs, lead, a.lead_max_tokens)
    show("lead turn 1", r1)
    if "error" in r1:
        check(False, f"lead turn 1 answered ({r1['error']})")
        return finish()
    check(r1["finish"] == "stop" and r1["text"], f"lead turn 1 finished with an answer (finish {r1['finish']})")
    check(hot["id"] in r1["text"] and hot["lead"] in r1["text"],
          f"lead found the incident: {hot['id']}, on-call {hot['lead']}")
    msgs += [{"role": "assistant", "content": r1["text"]},
             {"role": "user", "content": "Draft the working plan for the next hour: three numbered steps, one line "
                                         "each, naming the runbook."}]
    r2 = chat(url, msgs, lead, a.lead_max_tokens)
    show("lead turn 2", r2)
    if "error" in r2:
        check(False, f"lead turn 2 answered ({r2['error']})")
        return finish()
    check(r2["finish"] == "stop" and r2["text"], f"lead turn 2 finished with a plan (finish {r2['finish']})")
    check(r2["cached"] >= 0.9 * r1["prompt"], f"lead turn 2 resumed from its own turn 1 "
          f"({r2['cached']} of {r2['prompt']} prompt tokens cached)")
    msgs.append({"role": "assistant", "content": r2["text"]})
    read_s = r1["ttft"]

    # 2. Save the lead's conversation as a context.
    print("\n2. save the lead's whole conversation (system, document, 2 turns) as a context: POST /v1/contexts")
    t0 = time.time()
    ctx = call(url, "POST", "/contexts", {"name": cid, "messages": msgs, "effort": "medium", "persist": "memory"})
    if "error" in ctx:
        check(False, f"context saved ({ctx['error']})")
        return finish()
    v = ctx["variants"][0]
    print(f"  {cid}: {v['tokens']} tokens, saved in {time.time() - t0:.2f} s (engine prefill {v['prefill_s']} s), "
          f"{v['cached_at_save']} tokens were already cached, pinned={v['pinned']}, {v['kv_mib']} MiB of KV")
    check(v["pinned"] is True, "the context is pinned")
    check((v["cached_at_save"] or 0) >= 0.8 * v["tokens"],
          f"saving reused the lead's cached turns ({v['cached_at_save']} of {v['tokens']} tokens already cached)")
    ctx_tokens = v["tokens"]
    after_save = call(url, "GET", "/cache")
    print(f"  cache: {cache_line(after_save)}")

    # 3. Hand it to three agents with different settings, all at once.
    print("\n3. hand it to three agents at once, each with its own settings")
    roles = [
        ("writer", {"context": cid, "effort": "none", "max_context": 16384, "priority": 0,
                    "session": f"writer-{nonce}"}, 200,
         "Turn the plan above into a three-sentence status update for the company channel."),
        ("critic", {"context": cid, "effort": "medium", "priority": 3, "session": f"critic-{nonce}"}, 3000,
         "Critique the plan above: name its single biggest risk in one sentence."),
        ("fact-checker", {"context": cid, "effort": "none", "cache": "drop"}, 40,
         "Which service id did the analysis above identify as the incident? Reply with the id only."),
    ]
    sessions += [p["session"] for _, p, _, _ in roles if "session" in p]
    out = {}
    th = [threading.Thread(target=lambda n=n, p=p, m=m, q=q: out.__setitem__(n, chat(url, [{"role": "user",
          "content": q}], p, m))) for n, p, m, q in roles]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    print(f"  all three answered in {time.time() - t0:.1f} s (the lead took {read_s:.1f} s just to read the document)")
    for n, p, _, _ in roles:
        show(n, out[n])
    for n, p, _, _ in roles:
        r = out[n]
        if "error" in r:
            check(False, f"{n} answered ({r['error']})")
            continue
        check(r["cached"] >= 0.95 * ctx_tokens and r["ttft"] < 0.25 * read_s,
              f"{n} started from the lead's state: {r['cached']} of {r['prompt']} tokens cached, TTFT "
              f"{r['ttft']:.2f} s vs {read_s:.1f} s to read")
    if "error" not in out["fact-checker"]:
        check(hot["id"] in out["fact-checker"]["text"], f"fact-checker saw the lead's analysis "
              f"(answered {out['fact-checker']['text'][:20]!r}, expected {hot['id']})")
    if "error" not in out["writer"]:
        check(out["writer"]["finish"] == "stop", "writer finished inside its 16k max_context")

    # 4. Effort is part of the prefix: a low variant.
    if not a.no_variant:
        print("\n4. a reviewer at effort low (the effort is in the template's prefix: a variant of the context)")
        rv = {"context": cid, "effort": "low", "session": f"reviewer-{nonce}"}
        sessions.append(rv["session"])
        q = [{"role": "user", "content": "Is the plan above complete? Answer yes or no, then one sentence."}]
        r_a = chat(url, q, rv, 2000)
        show("reviewer 1", r_a)
        info = call(url, "GET", f"/contexts/{cid}")
        lows = [x for x in info.get("variants", []) if x["effort"] == "low"]
        shared = bool(lows) and lows[0]["tokens"] == ctx_tokens and lows[0]["prefill_s"] == v["prefill_s"]
        print(f"  variants now: " + ", ".join(f"{x['effort']} ({x['tokens']} tokens, prefill {x['prefill_s']} s)"
                                             for x in info.get("variants", [])))
        print("  low renders " + ("the same prefix as medium: no new computation" if shared else
                                 "a different prefix: the first low agent paid for that variant once"))
        r_b = chat(url, [{"role": "user", "content": "Name one stakeholder to inform. One sentence."}],
                   {**rv, "session": f"reviewer2-{nonce}"}, 2000)
        sessions.append(f"reviewer2-{nonce}")
        show("reviewer 2", r_b)
        if "error" in r_a or "error" in r_b:
            check(False, "reviewers answered")
        else:
            check(r_b["cached"] >= 0.9 * r_b["prompt"] - 64 and r_b["ttft"] < 0.25 * read_s,
                  f"second low agent reused the variant ({r_b['cached']} of {r_b['prompt']} cached, "
                  f"TTFT {r_b['ttft']:.2f} s)")

    # 5. Clean up, and verify.
    print("\n5. DELETE the context and all sessions, then verify with GET /v1/cache")
    before = call(url, "GET", "/cache")
    print(f"  before: {cache_line(before)}")
    d = call(url, "DELETE", f"/contexts/{cid}")
    ds = [call(url, "DELETE", f"/sessions/{s}") for s in sessions]
    check(d.get("deleted") is True, f"DELETE /v1/contexts/{cid}")
    check(all(x.get("deleted") for x in ds), f"DELETE /v1/sessions/... for {len(sessions)} sessions")
    after = call(url, "GET", "/cache")
    print(f"  after:  {cache_line(after)}")
    check(after["kv_pinned"] == base["kv_pinned"], f"pinned blocks released ({before['kv_pinned']} -> "
          f"{after['kv_pinned']}, {base['kv_pinned']} at start)")
    check(after["sessions"] == base["sessions"] and after["session_snapshots"] == base["session_snapshots"],
          f"sessions released ({before['sessions']} -> {after['sessions']}, session snapshots "
          f"{before['session_snapshots']} -> {after['session_snapshots']})")
    check(cid not in [c.get("name", "").split("@")[0] for c in after.get("contexts") or []],
          "the engine lists no variant of the context")
    gone = call(url, "GET", f"/contexts/{cid}")
    check("error" in gone, "GET /v1/contexts/{id} answers 404")
    r = chat(url, msgs + [{"role": "user", "content": "Reply with the word ok."}], {"effort": "none",
             "cache": "drop"}, 4)
    show("fresh request", r)
    if "error" in r:
        check(False, f"fresh request answered ({r['error']})")
    else:
        check(r["cached"] <= 0.1 * r["prompt"], f"the same prefix sent fresh is no longer cached "
              f"({r['cached']} of {r['prompt']} tokens cached; TTFT {r['ttft']:.1f} s, the cost of re-reading)")
    return finish()


def finish():
    print(f"\n{'PASS' if not FAILS else 'FAIL'}: agent-handoff, {len(FAILS)} failed check(s)")
    for f in FAILS:
        print(f"  failed: {f}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
