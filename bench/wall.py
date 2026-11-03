#!/usr/bin/env python3
# wall.py (2026-11-02): N users released at once, each with its own long prompt and a fixed output length: the shape of
# the public "8 users on an M3 Ultra" comparison (14,336 prompt + 6,400 generated tokens, N = 1..8). Reports
# per level: TTFT (median, max), admission time (release -> the last user's first token), the decode rate inside the
# window where every user is decoding, and end-to-end throughput A_e2e = all completion tokens / (last token - release).
# Standard library only. Prompts are distinct from their first token (a nonce leads), so no prefix is shared.
#   python3 wall.py --url http://127.0.0.1:8300 --model qwen --n 1,2,4,8 --trials 1 --out wall.jsonl
# Needs a server that honours ignore_eos (inferd, vLLM) so every user generates exactly --gen tokens.
# --shared (2026-11-02): the agent variant. All N users share ONE long context (a system message of ~--prompt tokens,
# fresh per level and trial) and ask different questions about it:
#   cold     the identical prefix is simply sent by everyone at once (any server; inferd's fan-out computes it once)
#   primed   one request with the prefix and max_tokens 1 goes first (the "warm the prompt cache" trick; any server)
#   context  inferd's agent-context API (docs/CONTEXTS.md): POST /v1/contexts once, then requests carry "inferd": {"context": id}
# primed/context report the setup time separately (setup_s) and with it (end_incl_setup_s). Columns: the last user's
# first token (last_first_s = TTFT max) and the last user's last token (last_end_s = wall), both from the release;
# total = setup + last end: the whole task's time, so cold, primed and context arms compare on equal terms.
# Streams go through sse_client.py (2026-11-02, Codex S04): a user counts only with a normal finish, data: [DONE],
# consistent usage and exactly --gen completion tokens; anything else is an error (partial if output had arrived), the
# whole level is reported as failed, and --raw keeps the failing streams' SSE payloads. Times are time.monotonic().
import argparse, json, os, statistics as st, sys, threading, time, urllib.request, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sse_client  # validated streaming client shared by the harnesses (Codex review 2026-11-02, S04)

PARA = ("Section {i}. The ledger service reconciles settlement batches against the clearing house file. "
        "Each batch carries an identifier, a value date, a currency, a count of payments and a control sum. "
        "When the control sum disagrees with the recomputed total the batch is quarantined, an alert is raised, "
        "and an operator must either correct the source file or approve a manual adjustment with a reason code. ")
TASK = ("\n\nUsing the specification above as context, write a complete Python module that implements the "
        "batch reconciliation it describes, with dataclasses, type hints, error handling and an exhaustive pytest "
        "test suite, then document every function in detail.")
TOK_PER_PARA, OVERHEAD = 81.7, 60


def shared_doc(args):
    n_para = max(1, round((args.prompt - OVERHEAD) / TOK_PER_PARA))
    return f"[shared {uuid.uuid4().hex}]\n" + "".join(PARA.format(i=i) for i in range(n_para)), n_para


def question(i, n_para):
    a, b = (i * 13) % n_para, (i * 29 + 7) % n_para
    return (f"You are agent {i}. Focus on sections {a} and {b} of the specification above. Using it as context, write "
            f"a complete Python module that implements what those sections describe, with dataclasses, type hints, "
            f"error handling and an exhaustive pytest test suite, then document every function in detail.")


def user(args, release, rec, i=0, shared=None):
    nonce = uuid.uuid4().hex
    n_para = max(1, round((args.prompt - OVERHEAD) / TOK_PER_PARA))
    text = f"[user {nonce}]\n" + "".join(PARA.format(i=i) for i in range(n_para)) + TASK
    msgs = [{"role": "user", "content": text}]
    inferd = None
    if shared is not None:  # the agent variant: one shared context, a different question each
        doc, n_para, ctx = shared
        if ctx:
            msgs, inferd = [{"role": "user", "content": question(i, n_para)}], {"context": ctx, "effort": "none"}
        else:
            msgs = [{"role": "system", "content": doc}, {"role": "user", "content": question(i, n_para)}]
    body = {"model": args.model, "messages": msgs, "max_tokens": args.gen,
            "temperature": args.temperature, "stream": True, "stream_options": {"include_usage": True},
            "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    if inferd:
        body["inferd"] = inferd
    if args.extra:
        body.update(json.loads(args.extra))
    while time.monotonic() < release:
        time.sleep(0.001)
    r = sse_client.stream_chat(args.url, body, timeout=args.timeout, expect_completion=args.gen,
                               keep_events=bool(args.raw))
    rec["sent"] = r["t_sent"]
    rec["chunks"] = r["chunks"]  # (monotonic t, characters) per delta
    rec["prompt_tokens"], rec["completion_tokens"] = r["prompt_tokens"], r["completion_tokens"]
    rec["cached"] = r["cached_tokens"]
    rec["finish_reason"], rec["done"] = r["finish_reason"], r["done"]
    if r["error"]:  # recorded, never silently dropped from a denominator; partial = output arrived before it failed
        rec["error"], rec["partial"] = r["error"], r["partial"]
    if args.raw:
        rec["events"] = r["events"]


def failed(rec):
    """A user counts only if its stream passed sse_client's validation (normal finish, [DONE], consistent usage,
    exactly --gen completion tokens) and produced output."""
    return bool(rec.get("error") or not rec.get("chunks") or not rec.get("completion_tokens"))


def post(args, path, body, timeout=None):
    r = urllib.request.urlopen(urllib.request.Request(args.url + path, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=timeout or args.timeout)
    return json.load(r)


def level(args, n):
    shared, setup = None, 0.0
    if args.shared != "off":
        doc, n_para = shared_doc(args)
        t0, ctx = time.monotonic(), None
        if args.shared == "context":
            ctx = post(args, "/v1/contexts", {"system": doc, "effort": "none"})["id"]
        elif args.shared == "primed":
            post(args, "/v1/chat/completions", {"model": args.model, "max_tokens": 1, "temperature": 0,
                                                "messages": [{"role": "system", "content": doc},
                                                             {"role": "user", "content": "Reply OK."}],
                                                "chat_template_kwargs": {"enable_thinking": False}})
        setup = time.monotonic() - t0
        shared = (doc, n_para, ctx)
    release, release_epoch = time.monotonic() + 0.5, time.time() + 0.5  # timing is monotonic; epoch for the log
    recs = [{} for _ in range(n)]
    th = [threading.Thread(target=user, args=(args, release, recs[i], i, shared)) for i in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    errs = [r for r in recs if failed(r)]
    out = {"n": n, "errors": len(errs), "partial": sum(1 for r in errs if r.get("partial")), "release": release_epoch,
           "shared": args.shared, "setup_s": round(setup, 3), "requested_gen": args.gen}
    if shared and shared[2]:
        try:
            urllib.request.urlopen(urllib.request.Request(args.url + f"/v1/contexts/{shared[2]}", method="DELETE"),
                                   timeout=60)
        except Exception:
            pass
    if errs:
        out["error_samples"] = [r.get("error") or "no output" for r in errs][:3]
        out["completion_tokens_seen"] = [r.get("completion_tokens") for r in recs]
        if args.raw:
            out["raw"] = [{"error": r.get("error"), "events": r.get("events")} for r in errs][:3]
        return out, recs
    firsts = [r["chunks"][0][0] for r in recs]
    lasts = [r["chunks"][-1][0] for r in recs]
    ttft = [f - release for f in firsts]
    total = sum(r["completion_tokens"] for r in recs)
    # Tokens per chunk, apportioned by characters (a chunk may carry several speculated tokens).
    w0, w1, in_w = max(firsts), min(lasts), 0.0
    for r in recs:
        chars = sum(c for _, c in r["chunks"]) or 1
        for t, c in r["chunks"]:
            if w0 <= t <= w1:
                in_w += r["completion_tokens"] * c / chars
    out.update({
        "prompt_tokens": st.median([r["prompt_tokens"] for r in recs]),
        "gen_tokens": st.median([r["completion_tokens"] for r in recs]),
        "ttft_med_s": st.median(ttft), "ttft_max_s": max(ttft), "admission_s": max(ttft),
        "window_s": max(0.0, w1 - w0), "window_tok_s": in_w / (w1 - w0) if w1 > w0 else None,
        "a_e2e_tok_s": total / (max(lasts) - release), "wall_s": max(lasts) - release,
        "last_first_s": max(ttft), "last_end_s": max(lasts) - release, "end_incl_setup_s": max(lasts) - release + setup,
        "cached_tokens_med": st.median([r.get("cached") or 0 for r in recs]),
        "per_user_decode_tok_s": st.median([r["completion_tokens"] / (r["chunks"][-1][0] - r["chunks"][0][0])
                                            for r in recs]),
    })
    return out, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--label", default="")
    ap.add_argument("--n", default="1,2,3,4,6,8")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--prompt", type=int, default=14336)
    ap.add_argument("--gen", type=int, default=6400)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--extra", default="", help="JSON merged into every request")
    ap.add_argument("--shared", choices=("off", "cold", "primed", "context"), default="off",
                    help="agent variant: N users share one context (see the header)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", action="store_true", help="keep failed streams' raw SSE payloads in the output")
    args = ap.parse_args()
    print(f"shared: {args.shared}; columns: last 1st = the last user's first token, last end = the last user's last "
          f"token (both from the release), setup = context save / priming before it", flush=True)
    print(f"{'N':>3} {'trial':>5} {'prompt':>7} {'cached':>7} {'gen':>5} {'TTFT med':>9} {'last 1st':>9} "
          f"{'last end':>9} {'setup':>6} {'total':>7} {'A_e2e':>7} {'win tok/s':>9} {'err':>3}", flush=True)
    for n in [int(x) for x in args.n.split(",")]:
        for t in range(args.trials):
            out, _ = level(args, n)
            out.update({"label": args.label, "trial": t, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
            with open(args.out, "a") as f:
                f.write(json.dumps(out) + "\n")
            if out["errors"]:
                print(f"{n:>3} {t:>5}  errors {out['errors']}: {out['error_samples']}", flush=True)
                continue
            print(f"{n:>3} {t:>5} {out['prompt_tokens']:>7} {out['cached_tokens_med']:>7} {out['gen_tokens']:>5} "
                  f"{out['ttft_med_s']:>9.1f} {out['last_first_s']:>9.1f} {out['last_end_s']:>9.1f} "
                  f"{out['setup_s']:>6.1f} {out['end_incl_setup_s']:>7.1f} {out['a_e2e_tok_s']:>7.1f} {out['window_tok_s'] or 0:>9.1f} "
                  f"{out['errors']:>3}", flush=True)


if __name__ == "__main__":
    main()
