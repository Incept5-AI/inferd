#!/usr/bin/env python3
"""Agent scenarios: typical agentic load patterns under concurrency, measured from the client side.

    python3 agents_scenarios.py --scenario a|b|c|d|e|all --mode api|implicit|plain [--preset m6|m5pro|studio]
        [--url http://127.0.0.1:8300/v1] [--model ID] [--label NAME] [--out rows.jsonl] [--pid SERVER_PID]

Scenarios (each prints a table and appends one JSON row per phase to --out):
  a  many small agents: 32-64 agents, a shared ~1k system prompt + their own ~1k task, effort none, short replies,
     3 turns each, all at once.
  b  forked agents: one long shared context (32k on big machines), then N agents with different tasks, three ways:
     cold (everyone sends the full prompt), saved (computed once first: POST /v1/contexts in api mode, a one-token
     priming request otherwise), and relaunched after unrelated load has gone through the cache.
  c  re-used agents: K agents with their own contexts take a turn, go idle, then resume: still resident, after
     unrelated load evicted them (to SSD when the server has an SSD tier, else recomputed), and parked explicitly
     (api mode: POST /v1/sessions/{id}/offload). Resumes run one at a time, so TTFT is the resume cost alone.
  d  ephemeral vs long-lived: a few long-lived agents keep taking turns while hundreds/thousands of one-off requests
     stream past; does the long-lived agents' cache survive, and does memory stay flat? (api: one-offs send
     "cache": "drop", long-lived agents a "session").
  e  mixed fleet: a few long-context researchers at effort low (high priority) plus a stream of short helpers at
     effort none (low priority), arriving over time.

Modes: api = inferd's agent-context API (docs/CONTEXTS.md: contexts, sessions, cache drop, effort/priority/max_context); implicit =
inferd's per-request fields only (priority, max_context: what the build before the API had), the cache works by
itself; plain = OpenAI fields only (any server). Greedy, streamed with usage; every run starts its documents with a
fresh nonce, so nothing cached by an earlier run can help. Memory: GET /v1/cache when the server has it (sampled once
a second: KV blocks in use, cached, pinned, snapshots, SSD), and the server's RSS when --pid is given.
Python standard library only.
"""
import argparse, json, os, random, statistics as st, subprocess, sys, threading, time, urllib.error, urllib.request, uuid

# ---------------------------------------------------------------- synthetic text
TEAMS = "Payments Ledger Identity Search Billing Risk Messaging Storage Platform Pricing Fraud Onboarding".split()
FIRST = "Priya Tomas Aiko Mateo Ingrid Kwame Lucia Omar Freya Dmitri Amara Jonas Mei Rafael Sanna Theo".split()
LAST = "Nair Okafor Lindqvist Moreau Haddad Tanaka Kowalski Ferreira Brennan Osei Varga Rossi Iyer Novak".split()
STORES = ["Postgres cluster pg-{}", "Redis shard rd-{}", "Kafka topic kt-{}", "S3 bucket s3-{}", "DynamoDB table dd-{}"]
NOUNS = "ledger invoice token session quote refund payout ticket profile ranking alert export".split()
VERBS = "reconcile issue verify route price settle notify index archive score sync audit".split()
TOK_PER_PARA = 109.0  # Qwen3.8 tokens per paragraph (examples/agent-swarm)


def document(seed, tokens, prefix="S"):
    """-> (text, facts): about `tokens` tokens of service-catalogue paragraphs."""
    rng, out, facts = random.Random(seed), [], []
    for i in range(max(2, int(tokens / TOK_PER_PARA))):
        f = dict(id=f"{prefix}-{i:03d}", team=rng.choice(TEAMS), lead=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                 store=rng.choice(STORES).format(rng.randint(1, 99)), rb=f"RB-{rng.randint(1000, 9999)}")
        facts.append(f)
        out.append(f"Service {f['id']} ({rng.choice(NOUNS)}-{rng.choice(VERBS)}) is owned by the {f['team']} team; its "
                   f"on-call lead is {f['lead']}. It stores state in {f['store']}, retries failed downstream calls "
                   f"{rng.randint(2, 9)} times with a {rng.choice([50, 100, 250, 500])} ms backoff, and pages on-call "
                   f"when its error rate stays above {rng.randint(1, 9)}% for {rng.randint(2, 30)} minutes. Its "
                   f"runbook is {f['rb']}.\n")
    return "".join(out), facts


TOOLS_TEXT = """You are a worker agent in an automation fleet. You act by replying with exactly ONE line:
ACTION <tool> <argument>
Tools: search <query> | open <service id> | page <team> | ticket <summary> | done <one-line result>.
Never explain. Never add text after the line. Rules of the fleet:
""" + "".join(f"- Rule {i}: {r}\n" for i, r in enumerate(
    ["prefer open over search when an id is known", "page a team only for sustained errors",
     "a ticket summary is at most twelve words", "finish with done as soon as the task is answered",
     "never page the Platform team on weekends", "quote service ids exactly as written",
     "one action per reply", "if unsure, search first"] * 6, 1))


# ---------------------------------------------------------------- client
class Client:
    def __init__(self, url, model, mode, timeout=3600):
        self.url, self.model, self.mode, self.timeout = url.rstrip("/"), model, mode, timeout

    def _req(self, method, path, body=None, timeout=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data, {"Content-Type": "application/json"}, method=method)
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.load(r)

    def api(self, method, path, body=None):
        try:
            return self._req(method, path, body)
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"}

    def chat(self, messages, max_tokens, effort="none", inferd=None, tools=None):
        """One streamed request -> dict(text, ttft, done, prompt, completion, cached) or dict(error)."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}
        if self.mode == "api" and inferd is not None:
            body["inferd"] = dict(inferd, effort=effort)
        else:
            body["chat_template_kwargs"] = ({"enable_thinking": False} if effort == "none" else
                                            {"enable_thinking": True, "reasoning_effort": effort})
            if self.mode == "implicit" and inferd:
                body["inferd"] = {k: v for k, v in inferd.items() if k in ("priority", "max_context")}
        if tools:
            body["tools"] = tools
        t0, first, text, usage = time.time(), None, [], {}
        try:
            req = urllib.request.Request(self.url + "/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    ev = json.loads(line[5:])
                    if ev.get("error"):
                        return {"error": str(ev["error"])[:300], "ttft": None, "done": time.time() - t0}
                    usage = ev.get("usage") or usage
                    for c in ev.get("choices") or []:
                        d = c.get("delta") or {}
                        s = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                        if s and first is None:
                            first = time.time()
                        text.append(d.get("content") or "")
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}", "ttft": None,
                    "done": time.time() - t0}
        except Exception as e:
            return {"error": repr(e)[:200], "ttft": None, "done": time.time() - t0}
        if first is None and not usage.get("completion_tokens"):
            return {"error": "stream ended without a reply", "ttft": None, "done": time.time() - t0}
        return {"text": "".join(text), "ttft": (first or time.time()) - t0, "done": time.time() - t0,
                "prompt": usage.get("prompt_tokens") or 0, "completion": usage.get("completion_tokens") or 0,
                "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


class Monitor:
    """Samples GET /v1/cache (inferd with the agent-context API) and the server's RSS once a second."""

    def __init__(self, cl, pid=None):
        self.cl, self.pid, self.rows, self.stop = cl, pid, [], threading.Event()
        try:
            self.has_cache = "kv_blocks" in cl._req("GET", "/cache", timeout=10)
        except Exception:
            self.has_cache = False
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()

    def sample(self):
        r = {"t": time.time()}
        if self.has_cache:
            try:
                c = self.cl._req("GET", "/cache", timeout=10)
                bt = c.get("block_tokens", 32)
                r.update(kv_used=(c["kv_blocks"] - c["kv_free"]) * bt, kv_cached=c["kv_cached"] * bt,
                         kv_pinned=c["kv_pinned"] * bt, snapshots=c["snapshots"], running=c["running"],
                         waiting=c["waiting"], preempted=c["preempted"], kv_total=c["kv_blocks"] * bt,
                         ssd_gib=(c.get("ssd") or {}).get("gib"), ssd_restored=(c.get("ssd") or {}).get("restored"),
                         ssd_spilled=(c.get("ssd") or {}).get("spilled"))
            except Exception:
                pass
        if self.pid:
            try:
                r["rss_gib"] = int(subprocess.run(["ps", "-o", "rss=", "-p", str(self.pid)], capture_output=True,
                                                  text=True).stdout.strip() or 0) / 1048576
            except Exception:
                pass
        return r

    def run(self):
        while not self.stop.wait(1.0):
            self.rows.append(self.sample())

    def window(self, t0, t1):
        rows = [r for r in self.rows if t0 <= r["t"] <= t1] or self.rows[-1:]
        out = {}
        for k in ("kv_used", "kv_cached", "kv_pinned", "snapshots", "rss_gib", "ssd_gib"):
            v = [r[k] for r in rows if r.get(k) is not None]
            if v:
                out["peak_" + k] = round(max(v), 2) if isinstance(v[0], float) else max(v)
        last = rows[-1] if rows else {}
        for k in ("kv_total", "preempted", "ssd_restored", "ssd_spilled"):
            if last.get(k) is not None:
                out[k] = last[k]
        return out


# ---------------------------------------------------------------- helpers
def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None


def summarize(rows, wall=None):
    ok = [r for r in rows if r and "error" not in r]
    err = [r for r in rows if r and "error" in r]
    tt = [r["ttft"] for r in ok]
    p = sum(r["prompt"] for r in ok)
    c = [r["cached"] for r in ok]
    s = {"n": len(ok), "errors": len(err), "ttft_med": round(st.median(tt), 3) if tt else None,
         "ttft_p90": round(pct(tt, 0.9), 3) if tt else None, "ttft_max": round(max(tt), 3) if tt else None,
         "prompt_tokens": p, "cached_tokens": sum(c) if c and None not in c else None,
         "cached_share": round(sum(c) / p, 3) if c and None not in c and p else None,
         "gen_tokens": sum(r["completion"] for r in ok)}
    if wall:
        s["wall_s"] = round(wall, 2)
        s["gen_tok_s"] = round(s["gen_tokens"] / wall, 1)
    if err:
        s["first_error"] = err[0]["error"]
    return s


def show(name, s):
    cs = "-" if s.get("cached_share") is None else f"{100 * s['cached_share']:.0f}%"
    extra = " ".join(f"{k}={s[k]}" for k in ("setup_s", "wall_s", "gen_tok_s", "peak_kv_used", "peak_kv_cached",
                                              "peak_kv_pinned", "peak_snapshots", "peak_rss_gib", "preempted",
                                              "ssd_restored") if s.get(k) is not None)
    print(f"  {name:34s} n={s['n']:<4d} err={s['errors']:<3d} TTFT med {s['ttft_med']} p90 {s['ttft_p90']} "
          f"max {s['ttft_max']} s, cached {cs} {extra}")
    if s.get("first_error"):
        print(f"    first error: {s['first_error']}")


def parallel(fns):
    out = [None] * len(fns)

    def run(i):
        out[i] = fns[i]()
    th = [threading.Thread(target=run, args=(i,)) for i in range(len(fns))]
    [t.start() for t in th]
    [t.join() for t in th]
    return out


def pool(fns, workers):
    out, lock, it = [None] * len(fns), threading.Lock(), iter(range(len(fns)))

    def run():
        while True:
            with lock:
                i = next(it, None)
            if i is None:
                return
            out[i] = fns[i]()
    th = [threading.Thread(target=run) for _ in range(workers)]
    [t.start() for t in th]
    [t.join() for t in th]
    return out


PRESETS = {  # sizes per machine (tokens are approximate)
    "m6": dict(a_agents=32, a_sys=1000, a_task=800, b_ctx=24000, b_agents=8, c_agents=6, c_ctx=6000,
               d_long=3, d_ctx=6000, d_oneoffs=240, d_conc=6, d_prompt=400, e_res=2, e_res_ctx=12000, e_helpers=24,
               filler=4, filler_ctx=14000),
    "m5pro": dict(a_agents=48, a_sys=1200, a_task=1000, b_ctx=30000, b_agents=16, c_agents=8, c_ctx=8000,
                  d_long=4, d_ctx=8000, d_oneoffs=400, d_conc=8, d_prompt=300, e_res=3, e_res_ctx=16000,
                  e_helpers=32, filler=6, filler_ctx=20000),  # with inferd at kv_gib 4 (~123k tokens), 16 snapshots
    "studio": dict(a_agents=64, a_sys=1500, a_task=1200, b_ctx=32000, b_agents=32, c_agents=16, c_ctx=16000,
                   d_long=6, d_ctx=16000, d_oneoffs=2000, d_conc=16, d_prompt=500, e_res=4, e_res_ctx=32000,
                   e_helpers=64, filler=12, filler_ctx=120000),
}


# ---------------------------------------------------------------- scenarios
def filler_load(cl, P, nonce, tag):
    """Unrelated long prompts, all at once: pushes earlier agents' prefixes out of a full cache."""
    def one(k):
        doc, _ = document(hash((nonce, tag, k)) & 0xffffff, P["filler_ctx"], prefix=f"F{k}")
        return cl.chat([{"role": "system", "content": f"[filler {nonce} {tag} {k}]\n{doc}"},
                        {"role": "user", "content": "Reply with the word OK."}], 4, inferd={"cache": "keep"})
    t0 = time.time()
    rows = parallel([lambda k=k: one(k) for k in range(P["filler"])])
    s = summarize(rows, time.time() - t0)
    print(f"  (filler: {P['filler']} x ~{P['filler_ctx'] // 1000}k prompts in {s.get('wall_s')} s, {s['errors']} errors)")
    return s


def scen_a(cl, P, nonce, mon):
    n, turns = P["a_agents"], 3
    sysmsg = f"[fleet {nonce}]\n{TOOLS_TEXT}" + document(11, P["a_sys"] - 450, prefix="K")[0]
    tasks = [document(1000 + k, P["a_task"], prefix=f"T{k}")[0] for k in range(n)]
    ctx = None
    t0 = time.time()
    if cl.mode == "api":
        r = cl.api("POST", "/contexts", {"name": f"a-{nonce}", "messages": [{"role": "system", "content": sysmsg}],
                                         "effort": "none"})
        if "error" in r:
            print("  context save failed:", r["error"])
        else:
            ctx = r["id"]
    setup = time.time() - t0
    res = [[None] * turns for _ in range(n)]
    start = time.time()

    def agent(k):
        msgs = [] if ctx else [{"role": "system", "content": sysmsg}]
        msgs.append({"role": "user", "content": f"Your queue item:\n{tasks[k]}\nTask: find who leads T{k}-001 and "
                                                f"open a ticket. Reply with one ACTION line."})
        inf = {"session": f"a-{nonce}-{k}"}
        if ctx:
            inf["context"] = ctx
        for t in range(turns):
            r = cl.chat(msgs, 48, "none", inf)
            res[k][t] = r
            if "error" in r:
                return
            msgs += [{"role": "assistant", "content": r["text"]},
                     {"role": "user", "content": f"TOOL RESULT {t}: ok, {random.Random(k * 7 + t).randint(1, 999)} "
                                                 f"records matched. Next ACTION line."}]
    parallel([lambda k=k: agent(k) for k in range(n)])
    wall = time.time() - start
    t1 = summarize([res[k][0] for k in range(n)])
    fu = summarize([res[k][t] for k in range(n) for t in range(1, turns)])
    allr = summarize([res[k][t] for k in range(n) for t in range(turns)], wall)
    allr.update(mon.window(start, time.time()), setup_s=round(setup, 2))
    show("turn 1 (all agents at once)", t1)
    show("follow-up turns", fu)
    show(f"all {n * turns} turns", allr)
    if cl.mode == "api":
        for k in range(n):
            cl.api("DELETE", f"/sessions/a-{nonce}-{k}")
        if ctx:
            cl.api("DELETE", f"/contexts/{ctx}")
    return [dict(phase="turn1", **t1), dict(phase="followups", **fu), dict(phase="all", **allr)]


def scen_b(cl, P, nonce, mon):
    n, out = P["b_agents"], []
    tasks = [f"Task {k}: which team owns S-{(k * 37) % 200:03d}, who is its on-call lead, and what is its runbook? "
             f"One sentence." for k in range(n)]

    def launch(sysmsg, ctx, tag):
        def one(k):
            if ctx:
                return cl.chat([{"role": "user", "content": tasks[k]}], 64, "none",
                               {"context": ctx, "session": f"b-{nonce}-{tag}-{k}", "cache": "keep"})
            return cl.chat([{"role": "system", "content": sysmsg}, {"role": "user", "content": tasks[k]}], 64, "none",
                           {"cache": "keep"})
        t0 = time.time()
        rows = parallel([lambda k=k: one(k) for k in range(n)])
        s = summarize(rows, time.time() - t0)
        s.update(mon.window(t0, time.time()))
        return s

    # cold: everyone sends the full prompt at once
    doc_a = f"[doc {nonce}-cold]\nAnswer from the catalogue only.\n\n" + document(21, P["b_ctx"])[0]
    s = launch(doc_a, None, "cold")
    show(f"cold: {n} agents, full prompts", s)
    out.append(dict(phase="cold", **s))
    # saved once, then N launches
    doc_b = f"[doc {nonce}-saved]\nAnswer from the catalogue only.\n\n" + document(22, P["b_ctx"])[0]
    t0, ctx = time.time(), None
    if cl.mode == "api":
        r = cl.api("POST", "/contexts", {"name": f"b-{nonce}", "system": doc_b, "effort": "none"})
        if "error" in r:
            print("  context save failed:", r["error"])
        else:
            ctx = r["id"]
    else:  # prime the cache with one request (the shared prefix is the system message)
        r = cl.chat([{"role": "system", "content": doc_b}, {"role": "user", "content": "Reply OK."}], 1, "none")
    setup = time.time() - t0
    s = launch(doc_b, ctx, "w1")
    s["setup_s"] = round(setup, 2)
    show(f"saved once ({setup:.1f} s), {n} agents", s)
    out.append(dict(phase="saved", **s))
    # unrelated load, then the same context again
    fs = filler_load(cl, P, nonce, "b")
    s = launch(doc_b, ctx, "w2")
    show(f"relaunch after unrelated load", s)
    out.append(dict(phase="relaunch", filler_wall_s=fs.get("wall_s"), **s))
    if ctx:
        print("  context:", json.dumps(cl.api("GET", f"/contexts/{ctx}").get("variants")))
        for k in range(n):
            for tag in ("w1", "w2"):
                cl.api("DELETE", f"/sessions/b-{nonce}-{tag}-{k}")
        cl.api("DELETE", f"/contexts/{ctx}")
    return out


def scen_c(cl, P, nonce, mon):
    k_, out = P["c_agents"], []
    evicts = ["resident", "filler"] + (["offload"] if cl.mode == "api" else [])
    for ev in evicts:
        docs = [f"[agent {nonce}-{ev}-{k}]\nYou maintain these services. Answer briefly.\n\n" +
                document(300 + k, P["c_ctx"], prefix=f"A{k}")[0] for k in range(k_)]
        msgs = [[{"role": "system", "content": d}, {"role": "user", "content": f"Who leads A{k}-002? One sentence."}]
                for k, d in enumerate(docs)]
        sess = lambda k: {"session": f"c-{nonce}-{ev}-{k}"}
        t0 = time.time()
        first = parallel([lambda k=k: cl.chat(msgs[k], 48, "none", sess(k)) for k in range(k_)])
        s1 = summarize(first, time.time() - t0)
        for k, r in enumerate(first):
            if r and "error" not in r:
                msgs[k] += [{"role": "assistant", "content": r["text"]},
                            {"role": "user", "content": f"And which runbook covers A{k}-004? One sentence."}]
        t_ev = time.time()
        if ev == "filler":
            filler_load(cl, P, nonce, f"c{ev}")
        elif ev == "offload":
            for k in range(k_):
                r = cl.api("POST", f"/sessions/c-{nonce}-{ev}-{k}/offload")
                if "error" in r:
                    print("  offload:", r["error"])
        ev_s = time.time() - t_ev
        t0 = time.time()
        resumed = []
        for k in range(k_):  # one at a time: the resume cost alone
            resumed.append(cl.chat(msgs[k], 48, "none", sess(k)) if len(msgs[k]) > 2 else None)
        s2 = summarize(resumed, time.time() - t0)
        s2.update(mon.window(t0, time.time()), evict_s=round(ev_s, 2), turn1_ttft_med=s1["ttft_med"])
        show(f"resume after: {ev}", s2)
        out.append(dict(phase=ev, **s2))
        if cl.mode == "api":
            for k in range(k_):
                cl.api("DELETE", f"/sessions/c-{nonce}-{ev}-{k}")
    return out


def scen_d(cl, P, nonce, mon, oneoff_cache="drop"):
    L, M = P["d_long"], P["d_oneoffs"]
    docs = [f"[long-lived {nonce}-{k}]\nYou are a long-running agent. Answer briefly.\n\n" +
            document(500 + k, P["d_ctx"], prefix=f"L{k}")[0] for k in range(L)]
    stop = threading.Event()
    long_rows = [[] for _ in range(L)]

    def long_agent(k):
        msgs = [{"role": "system", "content": docs[k]}]
        t = 0
        while not stop.is_set() or t < 2:
            msgs.append({"role": "user", "content": f"Turn {t}: who leads L{k}-{t % 40:03d}? One sentence."})
            r = cl.chat(msgs, 48, "none", {"session": f"d-{nonce}-{k}", "priority": 2})
            long_rows[k].append(r)
            if "error" in r:
                return
            msgs.append({"role": "assistant", "content": r["text"]})
            t += 1
            stop.wait(P.get("d_idle", 2))  # idle between turns (waiting on a tool, a human): what LRU punishes

    def oneoff(i):
        rng = random.Random(i)
        body = document(10_000 + i, P["d_prompt"], prefix=f"Q{i}")[0]
        return cl.chat([{"role": "system", "content": f"[one-off {nonce} {i}] Classify the record."},
                        {"role": "user", "content": body + "\nReply with one word: the team of the first service."}],
                       8, "none", {"cache": oneoff_cache, "priority": 0})
    start = time.time()
    lt = [threading.Thread(target=long_agent, args=(k,)) for k in range(L)]
    [t.start() for t in lt]
    time.sleep(1.0)
    t_o = time.time()
    ones = pool([lambda i=i: oneoff(i) for i in range(M)], P["d_conc"])
    o_wall = time.time() - t_o
    stop.set()
    [t.join() for t in lt]
    wall = time.time() - start
    first = summarize([rows[0] for rows in long_rows if rows])
    later = summarize([r for rows in long_rows for r in rows[1:]])
    later.update(mon.window(t_o, time.time()))
    so = summarize(ones, o_wall)
    so["req_s"] = round(so["n"] / o_wall, 2)
    show("long-lived agents, first turn", first)
    show("long-lived agents, later turns", later)
    show(f"one-offs ({M}, {P['d_conc']} at a time)", so)
    print(f"  one-offs: {so['req_s']} requests/s; long-lived turns: {later['n']}; wall {wall:.1f} s")
    if cl.mode == "api":
        for k in range(L):
            cl.api("DELETE", f"/sessions/d-{nonce}-{k}")
    return [dict(phase="long_first", **first), dict(phase="long_later", **later), dict(phase="oneoffs", **so)]


def scen_e(cl, P, nonce, mon):
    R, H = P["e_res"], P["e_helpers"]
    rdocs = [f"[research {nonce}-{k}]\nYou are a careful analyst.\n\n" + document(700 + k, P["e_res_ctx"],
                                                                                   prefix=f"R{k}")[0] for k in range(R)]
    hsys = f"[helpers {nonce}]\n{TOOLS_TEXT}"
    hctx = None
    if cl.mode == "api":
        r = cl.api("POST", "/contexts", {"name": f"e-{nonce}", "system": hsys, "effort": "none"})
        hctx = r.get("id")
    res_rows, help_rows = [[] for _ in range(R)], [None] * H
    start = time.time()

    def researcher(k):
        msgs = [{"role": "system", "content": rdocs[k]}]
        for t, q in enumerate(["Which three teams own the most services here, and what does that imply for on-call "
                               "load? Be concise.", "Name two services whose failure would page the same lead. Be "
                               "concise."]):
            msgs.append({"role": "user", "content": q})
            r = cl.chat(msgs, 384, "low", {"session": f"e-{nonce}-r{k}", "priority": 5,
                                           "max_context": P["e_res_ctx"] + 4096})
            res_rows[k].append(r)
            if "error" in r:
                return
            msgs.append({"role": "assistant", "content": r["text"]})

    def helper(i):
        time.sleep(i * 0.75)  # arriving over time
        task = f"Item {i}: " + document(900 + i, 600, prefix=f"H{i}")[0] + "\nOne ACTION line."
        inf = {"priority": -1, "max_context": 4096, "cache": "drop"}
        if hctx:
            r = cl.chat([{"role": "user", "content": task}], 48, "none", dict(inf, context=hctx))
        else:
            r = cl.chat([{"role": "system", "content": hsys}, {"role": "user", "content": task}], 48, "none", inf)
        help_rows[i] = r
    parallel([lambda k=k: researcher(k) for k in range(R)] + [lambda i=i: helper(i) for i in range(H)])
    wall = time.time() - start
    rs = summarize([r for rows in res_rows for r in rows], wall)
    rok = [r for rows in res_rows for r in rows if "error" not in r]
    rs["res_decode_tok_s"] = round(st.mean(r["completion"] / max(1e-3, r["done"] - r["ttft"]) for r in rok), 1) if rok else None
    rs["res_turn_s_med"] = round(st.median(r["done"] for r in rok), 2) if rok else None
    hs = summarize(help_rows, wall)
    hs["help_done_p90"] = round(pct([r["done"] for r in help_rows if r and "error" not in r], 0.9) or 0, 2)
    allr = summarize([r for rows in res_rows for r in rows] + help_rows, wall)
    allr.update(mon.window(start, time.time()))
    show("researchers (effort low, priority 5)", rs)
    print(f"    researcher turn median {rs['res_turn_s_med']} s, decode {rs['res_decode_tok_s']} tok/s per stream")
    show("helpers (effort none, priority -1)", hs)
    print(f"    helper request p90 {hs['help_done_p90']} s")
    show("whole fleet", allr)
    if cl.mode == "api":
        for k in range(R):
            cl.api("DELETE", f"/sessions/e-{nonce}-r{k}")
        if hctx:
            cl.api("DELETE", f"/contexts/{hctx}")
    return [dict(phase="researchers", **rs), dict(phase="helpers", **hs), dict(phase="all", **allr)]


SCEN = {"a": ("many small agents", scen_a), "b": ("forked agents", scen_b), "c": ("re-used agents", scen_c),
        "d": ("ephemeral vs long-lived", scen_d), "e": ("mixed fleet", scen_e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=os.environ.get("INFERD_URL", "http://127.0.0.1:8300/v1"))
    ap.add_argument("--model")
    ap.add_argument("--mode", choices=("api", "implicit", "plain"), default="api")
    ap.add_argument("--scenario", default="all", help="a, b, c, d, e, a comma list, or all")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="m6")
    ap.add_argument("--set", action="append", default=[], help="override a size, e.g. --set b_agents=16")
    ap.add_argument("--d-oneoff-cache", choices=("keep", "drop"), default="drop",
                    help="api mode: what one-off requests send in scenario d")
    ap.add_argument("--label", default="")
    ap.add_argument("--tag", default="", help="free text stored with every row (e.g. pressure)")
    ap.add_argument("--out")
    ap.add_argument("--pid", type=int, help="server process to sample RSS from")
    a = ap.parse_args()
    P = dict(PRESETS[a.preset])
    for kv in a.set:
        k, v = kv.split("=", 1)
        P[k] = int(v)
    if not a.model:
        with urllib.request.urlopen(a.url.rstrip("/") + "/models", timeout=30) as r:
            a.model = json.load(r)["data"][0]["id"]
    cl = Client(a.url, a.model, a.mode)
    mon = Monitor(cl, a.pid)
    names = list(SCEN) if a.scenario == "all" else a.scenario.split(",")
    print(f"agent scenarios {','.join(names)} on {a.url} ({a.model}), mode {a.mode}, preset {a.preset}"
          f"{'' if mon.has_cache else ' (no /v1/cache: memory from --pid only)'}\n")
    for s in names:
        title, fn = SCEN[s]
        nonce = uuid.uuid4().hex[:10]
        print(f"[{s}] {title}")
        t0 = time.time()
        rows = fn(cl, P, nonce, mon, a.d_oneoff_cache) if s == "d" and a.mode == "api" else fn(cl, P, nonce, mon)
        print(f"  ({time.time() - t0:.0f} s)\n")
        if a.out:
            with open(a.out, "a") as f:
                for r in rows:
                    f.write(json.dumps(dict(label=a.label, tag=a.tag, d_oneoff_cache=a.d_oneoff_cache, mode=a.mode, preset=a.preset, scenario=s, sizes=P,
                                            time=time.strftime("%Y-%m-%d %H:%M:%S"), **r)) + "\n")
    mon.stop.set()


if __name__ == "__main__":
    main()
