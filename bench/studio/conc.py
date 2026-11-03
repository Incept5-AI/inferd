#!/usr/bin/env python3
"""Concurrent-streams load test: N simultaneous requests, each with its own nonce'd C-token prompt (stdlib only).

Models an agent fleet in which every agent has a distinct context (there is no shared prefix, so no cache can help).
Per request it records t_start / t_first / t_last (time.monotonic, shared clock), TTFT and per-stream decode;
per cell it reports:
  agg_tps        = sum(completion_tokens) / (last t_last - first t_start)   end to end, prefill included
  agg_decode_tps = sum(completion_tokens) / (last t_last - last t_first)    once every stream is decoding
                   (only meaningful when the streams' decode windows overlap; see `overlap`)
  overlap        = fraction of the decode wall in which all N streams were decoding at once
--mode followup primes each agent's context untimed, then times turn 2 (a short follow-up) on all N at once:
the agent-fleet case, where contexts are already cached and decode dominates.
Streams are validated by ../sse_client.py (normal finish_reason, [DONE], consistent usage);
errors and partial streams are rows, never dropped. Usage: conc.py --label X --ctx 16k,32k --n 1,2,4,8,16 --out results/x.jsonl
"""
import argparse, json, statistics, threading, time, urllib.request
import speed  # also puts bench/ on sys.path
import sse_client  # validated SSE: normal finish, [DONE], consistent usage (Codex review 2026-11-02, S04)


FOLLOWUP = "Thanks. Now add type-checked validation for the currency field and extend the test suite to cover it."


THINKING = {"on": False}


def body(model, messages, max_tokens):
    kw = {"enable_thinking": THINKING["on"]}
    if THINKING["on"]:
        kw["reasoning_effort"] = THINKING.get("effort") or "low"
    return {"model": model, "stream": True, "stream_options": {"include_usage": True},
            "temperature": 0, "max_tokens": max_tokens, "messages": messages,
            "chat_template_kwargs": kw}


def prime(url, model, messages, timeout):
    """Untimed turn 1 (one agent's long context): returns the assistant reply so turn 2 extends it exactly."""
    req = dict(body(model, messages, 64), stream=False); req.pop("stream_options")
    r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def stream(url, model, messages, max_tokens, timeout, out, i, barrier):
    req = body(model, messages, max_tokens)
    barrier.wait()
    row = dict(i=i)
    r = sse_client.stream_chat(url, req, timeout=timeout)
    row["t_start"], t_first, t_last, usage = r["t_sent"], r["t_first"], r["t_last"], r["usage"]
    if not r["error"] and t_first is not None:
        ct = usage["completion_tokens"]
        row.update(prompt_tokens=usage["prompt_tokens"], completion_tokens=ct,
                   cached_tokens=r["cached_tokens"], t_first=t_first, t_last=t_last,
                   ttft_s=round(t_first - row["t_start"], 3),
                   decode_tps=round((ct - 1) / (t_last - t_first), 2) if t_last > t_first else None)
    else:
        row["error"] = (("partial: " if r["partial"] else "") + r["error"]) if r["error"] else f"no content (usage={usage})"
    out[i] = row


def cell(a, ctx, n):
    out = [None] * n; barrier = threading.Barrier(n)
    np_ = speed.n_para(ctx, False)
    convs = [[{"role": "user", "content": speed.prompt(np_)}] for _ in range(n)]
    prime_s = None
    if a.mode == "followup":  # each agent's context is primed (and cached) first; only turn 2 is timed
        t = time.monotonic()
        for c in convs:
            c += [{"role": "assistant", "content": prime(a.url, a.model, c, a.timeout)},
                  {"role": "user", "content": FOLLOWUP}]
        prime_s = round(time.monotonic() - t, 1)
    th = [threading.Thread(target=stream, args=(a.url, a.model, convs[i], a.max_tokens, a.timeout, out, i, barrier))
          for i in range(n)]
    [t.start() for t in th]; [t.join() for t in th]
    ok = [r for r in out if "error" not in r]
    res = dict(label=a.label, mode=a.mode, thinking=a.thinking or None, max_tokens=a.max_tokens, prime_s=prime_s, ctx=ctx, n=n, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), ok=len(ok), err=n - len(ok),
               errors=[r["error"] for r in out if "error" in r][:3])
    if ok:
        t0 = min(r["t_start"] for r in out); t_end = max(r["t_last"] for r in ok)
        last_first = max(r["t_first"] for r in ok); first_last = min(r["t_last"] for r in ok)
        tot = sum(r["completion_tokens"] for r in ok)
        dec_wall = t_end - min(r["t_first"] for r in ok)
        res.update(
            prompt_tokens=round(statistics.median(r["prompt_tokens"] for r in ok)),
            cached_tokens=round(statistics.median(r.get("cached_tokens") or 0 for r in ok)),
            wall_s=round(t_end - t0, 2), total_out=tot,
            agg_tps=round(tot / (t_end - t0), 1),
            agg_decode_tps=round(tot / (t_end - last_first), 1) if t_end > last_first else None,
            overlap=round(max(0.0, first_last - last_first) / dec_wall, 2) if dec_wall > 0 else None,
            ttft_med=round(statistics.median(r["ttft_s"] for r in ok), 2),
            ttft_max=round(max(r["ttft_s"] for r in ok), 2),
            stream_tps_med=round(statistics.median(r["decode_tps"] for r in ok if r["decode_tps"]), 1),
            stream_tps_min=round(min(r["decode_tps"] for r in ok if r["decode_tps"]), 1))
    res["rows"] = [{k: (round(v - out[0]["t_start"], 3) if k.startswith("t_") else v) for k, v in r.items()}
                   for r in out]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--model", default="mtplx-flash-next-optimized-speed")
    ap.add_argument("--label", required=True)
    ap.add_argument("--ctx", default="16k,32k,64k")
    ap.add_argument("--n", default="1,2,4,8,16")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--thinking", default="", help="reasoning effort to enable thinking with (e.g. low); off when empty")
    ap.add_argument("--mode", choices=["cold", "followup"], default="cold",
                    help="cold: every stream brings a brand-new context; followup: each agent's context is cached first")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    THINKING["on"] = bool(a.thinking); THINKING["effort"] = a.thinking or None
    print(f"{'ctx':>4} {'N':>3} {'ok':>3} {'err':>3} {'wall s':>7} {'agg tok/s':>9} {'agg dec':>8} {'ovl':>4} "
          f"{'TTFT med':>8} {'TTFT max':>8} {'stream med':>10} {'min':>6} {'cached':>7}", flush=True)
    with open(a.out, "a") as f:
        for ctx in a.ctx.split(","):
            for n in map(int, a.n.split(",")):
                r = cell(a, ctx, n)
                f.write(json.dumps(r) + "\n"); f.flush()
                g = lambda k: r.get(k, "-")
                print(f"{ctx:>4} {n:>3} {r['ok']:>3} {r['err']:>3} {g('wall_s'):>7} {g('agg_tps'):>9} "
                      f"{str(g('agg_decode_tps')):>8} {str(g('overlap')):>4} {g('ttft_med'):>8} {g('ttft_max'):>8} "
                      f"{g('stream_tps_med'):>10} {g('stream_tps_min'):>6} {g('cached_tokens'):>7}  {r['errors'][:1]}", flush=True)


if __name__ == "__main__":
    main()
