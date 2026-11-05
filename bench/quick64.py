#!/usr/bin/env python3
"""Quick, repeatable check of the headline: N agents (default 64) taking turns on their own 16k contexts, in about
5 minutes, with optional wall-power readings (tokens per joule = tok/s per watt). Stdlib only; uses studio/conc.py's
prompts, priming and validated streaming, so the numbers are in the same units as docs/RESULTS.md.

  1. prime:    every agent sends its own nonce'd ~16k-token context at once (prefill, untimed for the score)
  2. idle:     --idle-s seconds of power readings with nothing running (only with --power-cmd)
  3. measured: --rounds times, all N agents send their next turn at once (512 tokens out each; each round extends the
               conversation with the previous turn and its reply, as an agent does); per round:
               agg tok/s = sum(completion_tokens) / (last token - first request), first-token median / max, and the
               mean wall power over the round

Power: --power-cmd is any shell command that prints the current wall power in watts as its last number (a smart plug,
a meter's CLI); it is run every --power-interval seconds in the background. Without it, only tok/s is reported.

Typical (Mac Studio, Flash-Next, run.sh start --model flash-next; server on port 8300):
  python3 bench/quick64.py --url http://127.0.0.1:8300 --power-cmd './watts.sh'
Result: a summary on stdout and one JSON line appended to --out.
"""
import argparse, json, os, re, statistics, subprocess, sys, threading, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "studio"))
import conc   # noqa: E402  (bench/studio/conc.py: FOLLOWUP, prime, body)
import sse_client  # noqa: E402  (bench/sse_client.py: validated SSE, keeps the text)
import speed  # noqa: E402  (bench/studio/speed.py: the nonce'd paragraphs)


class Power(threading.Thread):
    """Samples --power-cmd every interval seconds: (monotonic time, watts); trace keeps (unix time, watts)."""

    def __init__(self, cmd, interval):
        super().__init__(daemon=True)
        self.cmd, self.interval, self.samples, self.errors, self.stop = cmd, interval, [], 0, threading.Event()
        self.trace = []

    def run(self):
        while not self.stop.is_set():
            t = time.monotonic()
            try:
                out = subprocess.run(self.cmd, shell=True, capture_output=True, text=True, timeout=10).stdout
                nums = re.findall(r"[-+]?\d+(?:\.\d+)?", out)
                if nums:
                    self.samples.append((t, float(nums[-1])))
                    self.trace.append((round(time.time(), 2), float(nums[-1])))
                else:
                    self.errors += 1
            except Exception:
                self.errors += 1
            self.stop.wait(max(0.0, self.interval - (time.monotonic() - t)))

    def window(self, t0, t1):
        return [w for t, w in self.samples if t0 <= t <= t1]


def task(i, k):
    """Agent i's own sum in round k: different numbers per agent and round, so every reply must come from that agent's
    own request (a sum of two 3-digit numbers: easy enough that a wrong answer means a broken stream, not the model)."""
    a, b = 100 + (i * 37 + k * 11) % 900, 100 + (i * 53 + k * 29 + 7) % 900
    return a, b, None, a + b


def ask(i, k):
    a, b, _, _ = task(i, k)
    return (f"{conc.FOLLOWUP} Before anything else, on the first line write exactly 'ANSWER: <n>' where n = {a} + {b}"
            f" (you are agent {i}). Then carry on with the task.")


def check(text, want):
    m = re.search(r"ANSWER:\s*\**\s*(-?[\d,]+)", text or "")
    return bool(m) and int(m.group(1).replace(",", "")) == want, (m.group(1) if m else None)


def stream_one(url, model, messages, max_tokens, timeout, out, i, barrier):
    """conc.stream, but keeping the reply's text for the per-agent check."""
    req = conc.body(model, messages, max_tokens)
    barrier.wait()
    r = sse_client.stream_chat(url, req, timeout=timeout)
    row = dict(i=i, t_start=r["t_sent"])
    if not r["error"] and r["t_first"] is not None:
        u = r["usage"]
        row.update(prompt_tokens=u["prompt_tokens"], completion_tokens=u["completion_tokens"], cached_tokens=r["cached_tokens"],
                   t_first=r["t_first"], t_last=r["t_last"], ttft_s=round(r["t_first"] - r["t_sent"], 3), text=r["text"])
    else:
        row["error"] = (("partial: " if r["partial"] else "") + r["error"]) if r["error"] else f"no content (usage={r['usage']})"
    out[i] = row


def peak_concurrency(rows):
    """Most streams decoding at the same moment (between each one's first and last token), and the fraction of the
    round in which all of them were."""
    ev = sorted([(r["t_first"], 1) for r in rows] + [(r["t_last"], -1) for r in rows])
    cur = peak = 0
    for _, d in ev:
        cur += d; peak = max(peak, cur)
    lo, hi = max(r["t_first"] for r in rows), min(r["t_last"] for r in rows)
    span = max(r["t_last"] for r in rows) - min(r["t_first"] for r in rows)
    return peak, round(max(0.0, hi - lo) / span, 2) if span > 0 else None


def mean_or_none(xs):
    return round(statistics.mean(xs), 1) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8300")
    ap.add_argument("--model", default="", help="served model id (default: the first one /v1/models lists)")
    ap.add_argument("--agents", type=int, default=64)
    ap.add_argument("--ctx", default="16k", help="each agent's context (speed.py sizes: 4k, 8k, 16k, ...)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--power-cmd", default="", help="shell command printing wall watts (optional)")
    ap.add_argument("--power-interval", type=float, default=1.0)
    ap.add_argument("--idle-s", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--label", default="quick64")
    ap.add_argument("--out", default="quick64.jsonl")
    a = ap.parse_args()
    n = a.agents
    wall = lambda m: round(time.time() - (time.monotonic() - m), 2)  # monotonic -> unix time, for traces
    phases = []
    if not a.model:
        import urllib.request
        with urllib.request.urlopen(a.url + "/v1/models", timeout=30) as r:
            a.model = json.load(r)["data"][0]["id"]
    pw = Power(a.power_cmd, a.power_interval) if a.power_cmd else None
    if pw:
        pw.start()
        time.sleep(a.idle_s)
    t_idle = time.monotonic()
    idle_w = mean_or_none(pw.window(0, t_idle)) if pw else None

    # 1. prime every agent's context at once (prefill batches across agents)
    np_ = speed.n_para(a.ctx, False)
    convs = [[{"role": "user", "content": speed.prompt(np_)}] for _ in range(n)]
    replies = [None] * n

    def do_prime(i):
        replies[i] = conc.prime(a.url, a.model, convs[i], a.timeout)

    t0 = time.monotonic()
    th = [threading.Thread(target=do_prime, args=(i,)) for i in range(n)]
    [t.start() for t in th]; [t.join() for t in th]
    prime_s = round(time.monotonic() - t0, 1)
    phases.append(dict(name="prime", t0=wall(t0), t1=wall(time.monotonic())))
    if any(r is None for r in replies):
        sys.exit(f"quick64: {sum(r is None for r in replies)} of {n} agents failed to prime")
    for c, r in zip(convs, replies):
        c.append({"role": "assistant", "content": r})
    print(f"primed {n} agents x {a.ctx} in {prime_s} s", flush=True)

    # 3. measured rounds: the same follow-up turn on all agents at once
    rounds = []
    for k in range(a.rounds):
        out = [None] * n
        barrier = threading.Barrier(n)
        msgs = [convs[i] + [{"role": "user", "content": ask(i, k)}] for i in range(n)]  # this round's new turn
        th = [threading.Thread(target=stream_one, args=(a.url, a.model, msgs[i], a.max_tokens, a.timeout, out, i, barrier))
              for i in range(n)]
        [t.start() for t in th]; [t.join() for t in th]
        ok = [r for r in out if "error" not in r]
        rr = dict(round=k + 1, ok=len(ok), err=n - len(ok), errors=[r["error"] for r in out if "error" in r][:3])
        verdicts = []
        for r in out:
            good_ans, got = check(r.get("text"), task(r["i"], k)[3]) if "error" not in r else (False, None)
            verdicts.append(dict(i=r["i"], correct=good_ans, got=got, want=task(r["i"], k)[3],
                                 tokens=r.get("completion_tokens"), ttft_s=r.get("ttft_s"),
                                 sent=wall(r["t_start"]) if r.get("t_start") else None,
                                 first_tok=wall(r["t_first"]) if r.get("t_first") else None,
                                 last_tok=wall(r["t_last"]) if r.get("t_last") else None,
                                 first=(r.get("text") or "")[:80], error=r.get("error")))
        rr.update(correct=sum(v["correct"] for v in verdicts), agents_detail=verdicts)
        # the next round extends each conversation with this turn and its reply (an agent's next turn), not a branch
        for r in out:
            convs[r["i"]] = msgs[r["i"]] + [{"role": "assistant", "content": r.get("text") or ""}]
        if ok:
            pk, ov = peak_concurrency(ok)
            rr.update(peak_concurrent=pk, all_decoding_frac=ov)
        if ok:
            ts = min(r["t_start"] for r in out); te = max(r["t_last"] for r in ok)
            tot = sum(r["completion_tokens"] for r in ok)
            phases.append(dict(name=f"round {k + 1}", t0=wall(ts), t1=wall(te)))
            rr.update(wall_s=round(te - ts, 2), total_out=tot, agg_tps=round(tot / (te - ts), 1),
                      ttft_med=round(statistics.median(r["ttft_s"] for r in ok), 2),
                      ttft_max=round(max(r["ttft_s"] for r in ok), 2),
                      prompt_tokens=round(statistics.median(r["prompt_tokens"] for r in ok)),
                      cached_tokens=round(statistics.median(r.get("cached_tokens") or 0 for r in ok)))
            if pw:
                time.sleep(a.power_interval * 1.5)  # let the last reading inside the round land
                ws = pw.window(ts, te)
                rr.update(power_w=mean_or_none(ws), power_samples=len(ws),
                          tok_per_j=round(tot / (statistics.mean(ws) * (te - ts)), 2) if ws else None)
        rounds.append(rr)
        print(f"round {k + 1}: {rr.get('ok')}/{n} ok, {rr['correct']}/{n} answers correct, {rr.get('peak_concurrent')} decoding at"
              f" once, {rr.get('agg_tps')} tok/s, wall {rr.get('wall_s')} s, "
              f"first token median {rr.get('ttft_med')} s / max {rr.get('ttft_max')} s"
              + (f", {rr.get('power_w')} W wall ({rr.get('power_samples')} readings), {rr.get('tok_per_j')} tok/J"
                 if pw else ""), flush=True)
    if pw:
        pw.stop.set()

    good = [r for r in rounds if r.get("agg_tps")]
    res = dict(label=a.label, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), url=a.url, model=a.model, agents=n, ctx=a.ctx,
               max_tokens=a.max_tokens, prime_s=prime_s, idle_w=idle_w, rounds=rounds,
               agg_tps=round(statistics.mean(r["agg_tps"] for r in good), 1) if good else None,
               power_w=mean_or_none([r["power_w"] for r in good if r.get("power_w")]),
               tok_per_j=round(statistics.mean(r["tok_per_j"] for r in good if r.get("tok_per_j")), 2)
               if pw and any(r.get("tok_per_j") for r in good) else None,
               power_errors=pw.errors if pw else None, phases=phases,
               power_trace=pw.trace if pw else None)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    errs = sum(r.get("err", 0) for r in rounds)
    right = sum(r.get("correct", 0) for r in rounds)
    allin = all(r.get("peak_concurrent") == n for r in rounds)
    print(("PASS" if good and not errs and right == n * len(rounds) and allin else "FAIL")
          + f": {n} agents x {a.ctx}, {len(good)}/{len(rounds)} rounds, {errs} errors, {right}/{n * len(rounds)} answers"
          f" correct, all {n} decoding at once: {'yes' if allin else 'no'}")
    print(f"\n{n} agents x {a.ctx}: {res['agg_tps']} tok/s total"
          + (f"; wall power {res['power_w']} W (idle {idle_w} W); {res['tok_per_j']} tokens per joule "
             f"(= tok/s per watt)" if pw else "") + f"; prime {prime_s} s", flush=True)


if __name__ == "__main__":
    main()
