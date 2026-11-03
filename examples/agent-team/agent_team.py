#!/usr/bin/env python3
"""Agent team under memory pressure: one planner, a few workers, many one-off classifiers and unrelated filler
traffic on one inferd server, run twice: once with the agent-context API's cache controls (sessions, priorities,
"cache": "drop" for the one-offs) and once without them (the same requests, plain). Prints a side-by-side table.

    python3 agent_team.py [--url http://127.0.0.1:8300/v1] [--workers 4] [--classifiers 24] [--rounds 2]

Start the server short of memory, or there is no pressure to show (docs/CONTEXTS.md): e.g. a 1 GiB KV pool and 16
snapshot slots, `run.sh 8300 8 1024 1 8 32768` (16 slots). The filler per round is sized from the server's pool
(GET /v1/cache) and capped by --filler-max.

Roles (effort and max_context are the same in both runs; only the cache controls differ):
  planner      effort medium (reasons), a ~--planner-doc token brief, priority 8 and a session (with the API)
  workers      effort none, max_context 8k, each its own ~--worker-doc token spec, priority 2 and a session
  classifiers  effort none, max_context 2k, short unique tickets, one word out; "cache": "drop", priority -5
  filler       unrelated long prompts, a few tokens out, the default cache behaviour, no session (both runs)
Each run: the team's first turn; a burst of classifiers alone (does the cache grow?); then --rounds rounds of
[filler + classifiers, then the planner and workers resume]. A sampler polls GET /v1/cache for the peak cached KV.
Greedy. Python standard library only. Self-checks print PASS/FAIL; the exit code is 1 if any failed.
"""
import argparse, json, os, random, statistics as st, sys, threading, time, urllib.error, urllib.request, uuid

WORDS = ("ledger invoice token session quote refund payout ticket profile ranking alert export cluster shard topic "
         "bucket replica region tenant quota schema index cache queue worker deploy rollback canary budget").split()
GOOD = "great fast helpful resolved thanks excellent smooth friendly".split()
BAD = "broken slow refund angry failed outage unusable rude".split()
FAILS = []


def check(ok, what):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        FAILS.append(what)


def text(tokens, rng):
    """~tokens tokens of plausible spec prose (~1.3 tokens per word)."""
    out = []
    for i in range(max(4, int(tokens / 1.3 / 14))):
        w = [rng.choice(WORDS) for _ in range(9)]
        out.append(f"Item {i}: the {w[0]} {w[1]} must sync with {w[2]} {w[3]} before the {w[4]} {w[5]} "
                   f"reaches {w[6]} {w[7]} {w[8]}.")
    return " ".join(out)


def call(url, method, path, body=None):
    req = urllib.request.Request(url + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": json.loads(e.read() or b"{}").get("error", e.code)}


def chat(url, messages, inferd, max_tokens):
    """Streamed; TTFT = first token of any kind. -> dict(text, ttft, prompt, cached, finish) or dict(error)."""
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "inferd": inferd}
    t0, first, out, usage, finish = time.time(), None, [], {}, None
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
                    s = d.get("content") or ""
                    if (s or d.get("reasoning_content")) and first is None:
                        first = time.time()
                    out.append(s)
                    finish = c.get("finish_reason") or finish
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"}
    except Exception as e:
        return {"error": repr(e)[:200]}
    return {"text": "".join(out).strip(), "ttft": (first or time.time()) - t0, "finish": finish,
            "prompt": usage.get("prompt_tokens") or 0,
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0}


def parallel(jobs, width=64):
    """jobs: list of zero-argument callables -> their results, at most `width` at a time."""
    res, sem = [None] * len(jobs), threading.Semaphore(width)

    def run(i):
        with sem:
            res[i] = jobs[i]()
    th = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
    [t.start() for t in th]
    [t.join() for t in th]
    return res


class Sampler(threading.Thread):
    """Polls GET /v1/cache; keeps the peak cached KV (blocks)."""
    def __init__(self, url):
        super().__init__(daemon=True)
        self.url, self.peak, self.stop = url, 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            c = call(self.url, "GET", "/cache")
            self.peak = max(self.peak, c.get("kv_cached") or 0)
            self.stop.wait(0.5)


def run_team(url, a, api, filler_tokens):
    """One run. -> dict of per-role results."""
    nonce = uuid.uuid4().hex[:8]
    rng = random.Random(nonce)
    label = "with the API" if api else "without"
    print(f"\n=== run {label} ({'sessions, priorities, cache drop for one-offs' if api else 'plain requests'})")
    pol = (lambda **k: k) if api else (lambda **k: {x: y for x, y in k.items() if x in ("effort", "max_context")})
    sessions = []

    def agent(name, effort, prio, system, max_context=None):
        s = f"{name}-{nonce}"
        sessions.append(s)
        p = pol(effort=effort, priority=prio, session=s, **({"max_context": max_context} if max_context else {}))
        return {"name": name, "pol": p, "msgs": [{"role": "system", "content": f"[{nonce} {name}] {system}"}]}

    planner = agent("planner", "medium", 8, "You are the planner of an engineering team. Project brief:\n\n" +
                    text(a.planner_doc, rng) + "\n\nAnswer in one short sentence.")
    workers = [agent(f"worker{k}", "none", 2, f"You are worker {k}. Your component spec:\n\n" +
                     text(a.worker_doc, rng) + "\n\nAnswer in one short sentence.", 8192) for k in range(a.workers)]
    team = [planner] + workers

    def turn(ag, q):
        ag["msgs"].append({"role": "user", "content": q})
        r = chat(url, ag["msgs"], ag["pol"], 1500 if ag is planner else 48)
        ag["msgs"].append({"role": "assistant", "content": r.get("text", "")})
        return r

    def classifier(i):
        mood = rng.choice(["positive", "negative"])
        words = GOOD if mood == "positive" else BAD
        ticket = " ".join(rng.choice(words + WORDS) for _ in range(220))
        msgs = [{"role": "user", "content": f"[{nonce} c{i}-{rng.random():.6f}] Classify this support ticket as "
                 f"positive or negative. Reply with one word.\n\n{ticket}"}]
        p = pol(effort="none", max_context=2048, priority=-5, cache="drop")
        return lambda: dict(chat(url, msgs, p, 4), mood=mood)

    def filler(i):
        msgs = [{"role": "user", "content": f"[{nonce} filler {i} {rng.random():.6f}] " +
                 text(filler_tokens // a.filler_n, random.Random(f"{nonce}{i}")) + "\n\nSummarise in five words."}]
        return lambda: chat(url, msgs, {"effort": "none"}, 8)

    res = {"first": [], "planner": [], "workers": [], "classifiers": [], "filler": [], "probe": None}
    ssd0 = call(url, "GET", "/cache").get("ssd") or {}
    sampler = Sampler(url)
    sampler.start()
    t0 = time.time()
    first = parallel([lambda ag=ag: turn(ag, "What is your first step?") for ag in team])
    res["first"] = first
    print(f"  team's first turn: {len(team)} agents, TTFT max {max(r.get('ttft', 0) for r in first):.1f} s")

    c0 = call(url, "GET", "/cache")
    probe = parallel([classifier(i) for i in range(a.classifiers)], 8)
    c1 = call(url, "GET", "/cache")
    res["classifiers"] += probe
    res["probe"] = (c0["kv_cached"], c1["kv_cached"], c0["snapshots"], c1["snapshots"])
    print(f"  {a.classifiers} classifiers alone: cached KV {c0['kv_cached']} -> {c1['kv_cached']} blocks, "
          f"snapshots {c0['snapshots']} -> {c1['snapshots']}")

    n = a.classifiers
    for rnd in range(a.rounds):
        t1 = time.time()
        jobs = [filler(rnd * a.filler_n + i) for i in range(a.filler_n)] + \
               [classifier(n + i) for i in range(a.classifiers)]
        out = parallel(jobs, a.filler_n + 8)
        n += a.classifiers
        res["filler"] += out[:a.filler_n]
        res["classifiers"] += out[a.filler_n:]
        mid = call(url, "GET", "/cache")
        print(f"  round {rnd + 1}: filler {a.filler_n} x ~{filler_tokens // a.filler_n} tokens + {a.classifiers} "
              f"classifiers in {time.time() - t1:.1f} s; cache now {mid['kv_cached']} blocks cached, "
              f"{mid['snapshots']} snapshots, {mid['kv_free']} free")
        out = parallel([lambda ag=ag: turn(ag, f"Round {rnd + 1}: what is your next step?") for ag in team])
        res["planner"].append(out[0])
        res["workers"] += out[1:]
        w = [r for r in out[1:] if "error" not in r]
        print(f"  round {rnd + 1}: team resumed; planner TTFT {out[0].get('ttft', -1):.2f} s cached "
              f"{out[0].get('cached')}/{out[0].get('prompt')}, workers TTFT median "
              f"{st.median([r['ttft'] for r in w]) if w else -1:.2f} s, cached "
              f"{sum(r['cached'] for r in w)}/{sum(r['prompt'] for r in w)}")
    sampler.stop.set()
    sampler.join()
    res["peak"] = sampler.peak
    ssd1 = call(url, "GET", "/cache").get("ssd") or {}
    res["ssd"] = (ssd1.get("spilled", 0) - ssd0.get("spilled", 0), ssd1.get("restored", 0) - ssd0.get("restored", 0),
                  round(ssd1.get("restore_s", 0) - ssd0.get("restore_s", 0), 1)) if ssd1 else None
    res["wall"] = time.time() - t0
    for s in sessions:  # leave the server as we found it
        if api:
            call(url, "DELETE", f"/sessions/{s}")
    return res


def stats(rs):
    ok = [r for r in rs if "error" not in r]
    if not ok:
        return {"n": 0, "err": len(rs)}
    t = [r["ttft"] for r in ok]
    return {"n": len(ok), "err": len(rs) - len(ok), "med": st.median(t), "max": max(t),
            "share": sum(r["cached"] for r in ok) / max(1, sum(r["prompt"] for r in ok)),
            "stayed": sum(r["cached"] >= 0.9 * r["prompt"] - 64 for r in ok)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=os.environ.get("INFERD_URL", "http://127.0.0.1:8300/v1"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--classifiers", type=int, default=16, help="one-off classifiers per burst")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--planner-doc", type=int, default=2048, help="the planner's brief, tokens")
    ap.add_argument("--worker-doc", type=int, default=1024, help="each worker's spec, tokens")
    ap.add_argument("--filler-n", type=int, default=2, help="filler requests per round")
    ap.add_argument("--filler-max", type=int, default=4096, help="cap on filler tokens per round")
    ap.add_argument("--mode", choices=["both", "api", "plain"], default="both")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    c = call(url, "GET", "/cache")
    if "error" in c:
        sys.exit(f"GET /v1/cache failed ({c['error']}): this demo needs inferd's agent-context API")
    pool = c["kv_blocks"] * c.get("block_tokens", 32)
    team_tokens = a.planner_doc + a.workers * a.worker_doc
    filler = max(4096, min(a.filler_max, pool - team_tokens // 2))
    print(f"server: KV pool {pool} tokens ({c['kv_blocks']} blocks), {c['slots']} snapshot slots, "
          f"{c['kv_cached']} blocks cached at start")
    note = ("enough to cycle the pool" if filler >= pool - team_tokens else
            "less than the pool: the pressure comes mostly from snapshot slots, see README")
    print(f"team: planner ~{a.planner_doc} + {a.workers} workers x ~{a.worker_doc} tokens; filler ~{filler} tokens "
          f"per round ({note})")
    runs = {}
    for api in ([False, True] if a.mode == "both" else [a.mode == "api"]):
        runs[api] = run_team(url, a, api, filler)

    print("\n=== side by side (TTFT in seconds; cached = share of prompt tokens served from the cache)")
    cols = [k for k in (True, False) if k in runs]
    head = "".join(f" | {('with the API' if k else 'without'):>34}" for k in cols)
    print(f"{'':28}{head}")
    def row(name, f):
        print(f"{name:28}" + "".join(f" | {f(runs[k]):>34}" for k in cols))
    for role in ("planner", "workers", "classifiers", "filler"):
        row(f"{role} TTFT med / max", lambda r, role=role: (lambda s: f"{s['med']:.2f} / {s['max']:.2f}"
            if s['n'] else "-")(stats(r[role])))
        if role in ("planner", "workers"):
            row(f"{role} resumes cached", lambda r, role=role: (lambda s: f"{100 * s['share']:.0f}%, stayed "
                f"cached {s['stayed']}/{s['n']}" if s['n'] else "-")(stats(r[role])))
    row("errors (all roles)", lambda r: str(sum(stats(r[x])["err"] for x in ("first", "planner", "workers",
                                                                             "classifiers", "filler"))))
    row("one-off burst: cached KV", lambda r: f"{r['probe'][0]} -> {r['probe'][1]} blocks")
    row("one-off burst: snapshots", lambda r: f"{r['probe'][2]} -> {r['probe'][3]}")
    row("peak cached KV", lambda r: f"{r['peak']} blocks ({r['peak'] * 32 // 1000}k tokens)")
    row("SSD tier: spilled / restored", lambda r: f"{r['ssd'][0]} / {r['ssd'][1]} ({r['ssd'][2]} s restoring)"
        if r["ssd"] else "no SSD tier")
    row("wall", lambda r: f"{r['wall']:.0f} s")

    print("\n=== checks")
    for k in cols:
        r, tag = runs[k], "with the API" if k else "without"
        allr = r["first"] + r["planner"] + r["workers"] + r["classifiers"] + r["filler"]
        check(all("error" not in x for x in allr), f"{tag}: every request answered")
        cl = [x for x in r["classifiers"] if "error" not in x]
        right = sum(x["mood"] in x["text"].lower() for x in cl)
        check(right >= 0.9 * len(cl), f"{tag}: classifiers labelled {right}/{len(cl)} tickets correctly")
    if True in runs:
        r = runs[True]
        p, w = stats(r["planner"]), stats(r["workers"])
        check(p["n"] and p["stayed"] == p["n"], f"with the API: the planner stayed cached ({p.get('stayed')}/"
              f"{p['n']} resumes)")
        check(w["n"] and w["stayed"] == w["n"], f"with the API: the workers stayed cached ({w.get('stayed')}/"
              f"{w['n']} resumes, {100 * w.get('share', 0):.0f}% of prompt tokens)")
        k0, k1, s0, s1 = r["probe"]
        check(k1 <= k0 + 2 and s1 <= s0, f"with the API: one-offs left the cache flat (cached KV {k0} -> {k1} "
              f"blocks, snapshots {s0} -> {s1})")
    if True in runs and False in runs:
        wa, wp = stats(runs[True]["workers"]), stats(runs[False]["workers"])
        if wp["n"] and wp["stayed"] == wp["n"]:
            print(f"  SKIP  no pressure reached the team: without the API the workers stayed cached too "
                  f"({100 * wp['share']:.0f}%), so there is nothing to compare (use a server with fewer snapshot "
                  f"slots or a smaller KV pool, see README)")
        elif wa["n"] and wp["n"]:
            check(wa["share"] > wp["share"] and wa["med"] < wp["med"],
                  f"workers resume better with the API: median TTFT {wa['med']:.2f} vs {wp['med']:.2f} s, cached "
                  f"{100 * wa['share']:.0f}% vs {100 * wp['share']:.0f}%")
        k0, k1 = runs[False]["probe"][:2]
        print(f"  info  without the API the same one-off burst grew the cache {k0} -> {k1} blocks")
    print(f"\n{'PASS' if not FAILS else 'FAIL'}: agent-team, {len(FAILS)} failed check(s)")
    for f in FAILS:
        print(f"  failed: {f}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
