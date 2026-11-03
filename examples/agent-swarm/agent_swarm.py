#!/usr/bin/env python3
"""Agent swarm on one shared document: N agents get the same long system prompt (a service catalogue of --doc tokens),
all send their first question at once (the fan-out), then each runs follow-up turns on its own clock. Every question
has a checkable answer in the catalogue, so you can also see that each engine answered the same way.

    python3 agent_swarm.py                                     # inferd on http://127.0.0.1:8300/v1
    python3 agent_swarm.py --url http://127.0.0.1:8080/v1 --model <id>   # any OpenAI-compatible server
    python3 agent_swarm.py --agents 16 --doc 32768 --turns 4

Greedy (temperature 0), thinking off (chat_template_kwargs.enable_thinking=false), streamed with usage. A fresh nonce
leads the catalogue on every run, so no cache from an earlier run (or an SSD cache that survives restarts) can help.
Token counts come from the server's usage; cached_tokens is shown when the server reports it ('-' when it does not).
Python standard library only.
"""
import argparse, json, os, random, re, statistics as st, threading, time, urllib.error, urllib.request, uuid

TEAMS = "Payments Ledger Identity Search Billing Risk Messaging Storage Platform Pricing Fraud Onboarding".split()
FIRST = "Priya Tomas Aiko Mateo Ingrid Kwame Lucia Omar Freya Dmitri Amara Jonas Mei Rafael Sanna Theo".split()
LAST = "Nair Okafor Lindqvist Moreau Haddad Tanaka Kowalski Ferreira Brennan Osei Varga Rossi Iyer Novak".split()
STORES = ["Postgres cluster pg-{}", "Redis shard rd-{}", "Kafka topic kt-{}", "S3 bucket s3-{}", "DynamoDB table dd-{}"]
NOUNS = "ledger invoice token session quote refund payout ticket profile ranking alert export".split()
VERBS = "reconcile issue verify route price settle notify index archive score sync audit".split()
DAYS = "Monday Tuesday Wednesday Thursday".split()
TOK_PER_SVC = 109.0  # Qwen3.8 tokens per service paragraph (measured from usage; only sizes the document)


def catalogue(seed, tokens):
    """-> (text, facts): a synthetic service catalogue of about `tokens` tokens and the facts to check answers by."""
    rng, facts, out = random.Random(seed), [], []
    for i in range(max(8, int(tokens / TOK_PER_SVC))):
        f = dict(id=f"S-{i:03d}", name=f"{rng.choice(NOUNS)}-{rng.choice(VERBS)}", team=rng.choice(TEAMS),
                 lead=f"{rng.choice(FIRST)} {rng.choice(LAST)}", store=rng.choice(STORES).format(rng.randint(1, 99)),
                 retries=rng.randint(2, 9), backoff=rng.choice([50, 100, 250, 500, 1000]), err=rng.randint(1, 9),
                 win=rng.randint(2, 30), deps=[rng.randrange(0, 400), rng.randrange(0, 400)], day=rng.choice(DAYS),
                 rb=f"RB-{rng.randint(1000, 9999)}")
        facts.append(f)
        out.append(f"Service {f['id']} ({f['name']}) is owned by the {f['team']} team; its on-call lead is {f['lead']}. "
                   f"It stores state in {f['store']}, retries failed downstream calls {f['retries']} times with a "
                   f"{f['backoff']} ms backoff, and pages on-call when its error rate stays above {f['err']}% for "
                   f"{f['win']} minutes. It depends on S-{f['deps'][0]:03d} and S-{f['deps'][1]:03d}, deploys on "
                   f"{f['day']}s, and its runbook is {f['rb']}.\n")
    return "".join(out), facts


QUESTIONS = [  # (question about service f, the fact a correct answer must contain)
    (lambda f: f"Which team owns {f['id']}, and who is its on-call lead? One sentence.", lambda f: f["lead"]),
    (lambda f: f"How many times does {f['id']} retry a failed call, and where does it store state? One sentence.",
     lambda f: f["store"]),
    (lambda f: f"Which runbook covers {f['id']}, and on which day does it deploy? One sentence.", lambda f: f["rb"]),
    (lambda f: f"At what error rate, sustained for how long, does {f['id']} page on-call? One sentence.",
     lambda f: f"{f['err']}%"),
]


def has_fact(text, fact):
    """True when the answer states the fact as a whole token ("rd-3" must not match "rd-31")."""
    return re.search(r"(?<![\w.-])" + re.escape(fact.lower()) + r"(?![\w])", text.lower()) is not None


def chat(url, model, messages, max_tokens, timeout):
    """One streamed request. -> dict(text, ttft, done, prompt, completion, cached, think, error)."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0, first, text, think, usage = time.time(), None, [], 0, {}
    try:
        req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                usage = ev.get("usage") or usage
                for c in ev.get("choices") or []:
                    d = c.get("delta") or {}
                    s, rs = d.get("content") or "", d.get("reasoning_content") or d.get("reasoning") or ""
                    if (s or rs) and first is None:
                        first = time.time()
                    text.append(s); think += len(rs)
    except urllib.error.HTTPError as e:  # e.g. 503 from a server that refuses work: recorded, never dropped
        return {"error": f"HTTP {e.code}: {e.read()[:160].decode(errors='replace')}", "ttft": None, "done": time.time() - t0}
    except Exception as e:
        return {"error": repr(e)[:200], "ttft": None, "done": time.time() - t0}
    if first is None and not usage.get("completion_tokens"):  # the stream ended without a reply: an error
        return {"error": "stream ended without a reply", "ttft": None, "done": time.time() - t0}
    return {"text": "".join(text), "ttft": (first or time.time()) - t0, "done": time.time() - t0,
            "prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), "think": think}


def first_model(url):
    with urllib.request.urlopen(url + "/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default=os.environ.get("INFERD_URL", "http://127.0.0.1:8300/v1"))
    ap.add_argument("--model", help="model id (default: the first one /v1/models lists)")
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--doc", type=int, default=16384, help="shared document size in tokens (approximate)")
    ap.add_argument("--turns", type=int, default=3, help="turns per agent, the first one being the fan-out")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--words", type=int, default=0,
                    help="also ask for an explanation of about this many words (longer replies: a decode-heavy load)")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--label", default="", help="engine name for the summary row")
    ap.add_argument("--out", help="append a JSON summary row here")
    a = ap.parse_args()
    model = a.model or first_model(a.url)
    more = (f" Then explain in about {a.words} words how you would check this during an incident, and what could go "
            f"wrong.") if a.words else ""
    nonce = uuid.uuid4().hex[:12]
    doc, facts = catalogue(7, a.doc)
    system = (f"[catalogue {nonce}]\nYou are one agent of a team answering questions about the service catalogue "
              f"below. Answer from the catalogue only, briefly.\n\n{doc}")
    rng = random.Random(1)
    picks = [rng.sample(facts, a.turns) for _ in range(a.agents)]  # each agent asks about its own services
    turns = [[None] * a.turns for _ in range(a.agents)]
    start = time.time()
    fanout_done = threading.Barrier(a.agents)

    def agent(k):
        msgs = [{"role": "system", "content": system}]
        for t in range(a.turns):
            q, _ = QUESTIONS[t % len(QUESTIONS)]
            msgs.append({"role": "user", "content": q(picks[k][t]) + more})
            sent = time.time() - start
            r = chat(a.url, model, msgs, a.max_tokens, a.timeout)
            r["sent"] = sent
            turns[k][t] = r
            if t == 0:
                fanout_done.wait()  # follow-ups start once the fan-out is answered, as an orchestrator would
            if "error" in r:
                return
            msgs.append({"role": "assistant", "content": r["text"]})

    print(f"{a.agents} agents, one shared {a.doc // 1024}k-token catalogue, {a.turns} turns each, "
          f"max_tokens {a.max_tokens}{f', ~{a.words}-word replies' if a.words else ''}, on {a.url} ({model})\n")
    th = [threading.Thread(target=agent, args=(k,)) for k in range(a.agents)]
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.time() - start

    print(f"{'agent':>5} {'turn1 TTFT':>10} {'follow-up TTFT':>14} {'prompt':>7} {'cached':>13} {'done at':>8} "
          f"{'correct':>7}")
    ok_all, err_all, correct_all, think_all = [], [], 0, 0
    for k in range(a.agents):
        rs = [r for r in turns[k] if r]
        ok = [r for r in rs if "error" not in r]
        err_all += [r for r in rs if "error" in r]
        ok_all += [(t, r) for t, r in enumerate(turns[k]) if r and "error" not in r]
        good = sum(1 for t, r in enumerate(turns[k]) if r and "error" not in r and
                   has_fact(r["text"], QUESTIONS[t % len(QUESTIONS)][1](picks[k][t])))
        correct_all += good
        think_all += sum(r.get("think", 0) for r in ok)
        t1 = turns[k][0]
        fu = [r["ttft"] for r in ok[1:]]
        fu_p = sum(r["prompt"] or 0 for r in ok[1:]); fu_c = [r["cached"] for r in ok[1:]]
        cached = f"{sum(fu_c)}/{fu_p}" if fu_c and None not in fu_c else "-"
        t1s = f"{t1['ttft']:.1f} s" if "error" not in t1 else "error"
        print(f"{k:>5} {t1s:>10} {(f'{st.median(fu):.2f} s' if fu else '-'):>14} {str(ok[-1]['prompt'] if ok else '-'):>7} "
              f"{cached:>13} {max(r['sent'] + r['done'] for r in rs):>7.1f}s {good:>3}/{a.turns}")

    fan = [turns[k][0] for k in range(a.agents)]
    fan_ok = [r for r in fan if "error" not in r]
    fu_ok = [r for t, r in ok_all if t > 0]
    def cached_share(rows):
        if not rows or any(r["cached"] is None for r in rows):
            return None
        return sum(r["cached"] for r in rows) / max(1, sum(r["prompt"] or 0 for r in rows))
    fan_c, fu_c = cached_share(fan_ok), cached_share(fu_ok)
    s = {"label": a.label, "model": model, "words": a.words, "max_tokens": a.max_tokens, "agents": a.agents, "doc": a.doc, "turns": a.turns,
         "prompt_tokens_turn1": fan_ok[0]["prompt"] if fan_ok else None,
         "fanout_last_first_token_s": round(max(r["ttft"] for r in fan_ok), 2) if fan_ok else None,
         "fanout_wall_s": round(max(r["done"] for r in fan), 2),
         "followup_ttft_med_s": round(st.median(r["ttft"] for r in fu_ok), 2) if fu_ok else None,
         "followup_ttft_max_s": round(max(r["ttft"] for r in fu_ok), 2) if fu_ok else None,
         "wall_s": round(wall, 2), "turns_ok": len(ok_all), "errors": len(err_all),
         "gen_tokens": sum(r["completion"] or 0 for t, r in ok_all),
         "gen_tok_s": round(sum(r["completion"] or 0 for t, r in ok_all) / wall, 1),
         "correct": correct_all, "cached_fanout": fan_c, "cached_followups": fu_c, "thinking_chars": think_all}
    pct = lambda x: "-" if x is None else f"{100 * x:.0f}%"
    print(f"\nfan-out ({a.agents} first questions at once): last agent's first token {s['fanout_last_first_token_s']} s, "
          f"all answered {s['fanout_wall_s']} s, cached {pct(fan_c)} of prompt tokens")
    print(f"follow-ups: TTFT median {s['followup_ttft_med_s']} s, max {s['followup_ttft_max_s']} s, "
          f"cached {pct(fu_c)} of prompt tokens")
    print(f"all {a.agents * a.turns} turns: {wall:.1f} s wall, {len(ok_all)} ok, {len(err_all)} errors, "
          f"{correct_all}/{a.agents * a.turns} answers contain the catalogue fact; {s['gen_tokens']} tokens generated "
          f"({s['gen_tok_s']} tok/s over the whole run)")
    if err_all:
        print("errors, e.g.:", err_all[0]["error"])
    if think_all:
        print(f"WARNING: the server returned {think_all} chars of reasoning: thinking was not off, not comparable")
    print("\nagent 0's answers:")
    for t, r in enumerate(turns[0]):
        if r and "error" not in r:
            print(f"  Q{t + 1}: {QUESTIONS[t % len(QUESTIONS)][0](picks[0][t])}\n  A{t + 1}: {r['text'].strip()[:300]}")
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
