#!/usr/bin/env python3
"""tfprompt.py -- single-stream timing on the TensorFold challenge's prompt (2026-10-31, stdlib only).

Sends the challenge's benchmark prompt tokens (mlx.fast / cuda.fast "longcopy" case, ~1,065 tokens) straight to an
inferd socket, greedy, and times prefill (to the first token) and decode (first to last token). Each rep prepends one
random token so the prefix cache cannot serve the prompt. Comparable in spirit, not in harness, to the challenge's
score (its decode window is 128 steps after a seed window; this reports both 128 and the full --max-tokens run).
  tfprompt.py --socket /tmp/inferd-8400.sock --prompt challenge.json [--max-tokens 512] [--reps 3]
"""
import argparse, json, random, socket, statistics, time


def run(sock_path, tokens, max_tokens, rid):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    s.sendall((json.dumps({"op": "gen", "id": rid, "tokens": tokens, "max_tokens": max_tokens, "temperature": 0,
                           "top_p": 1.0, "top_k": 0, "stop": []}) + "\n").encode())
    t0 = time.perf_counter(); times = []; buf = b""; done = None
    while done is None:
        chunk = s.recv(1 << 16)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            ev = json.loads(line)
            if ev.get("id") != rid:
                continue
            if "t" in ev:
                times.append(time.perf_counter())
            if "done" in ev:
                done = ev
    s.close()
    ttft = times[0] - t0
    dec = lambda n: (min(n, len(times)) - 1) / (times[min(n, len(times)) - 1] - times[0])
    return {"prompt": len(tokens), "gen": len(times), "ttft_s": round(ttft, 3),
            "prefill_tps": round(len(tokens) / ttft, 1), "decode128_tps": round(dec(129), 1),
            "decode_tps": round(dec(len(times)), 1), "done": done.get("done") if done else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True); ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=512); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--label", default="inferd"); ap.add_argument("--out")
    a = ap.parse_args()
    d = json.load(open(a.prompt))
    toks = d["benchmark"]["prefill_prompt_tokens"] if "benchmark" in d else d["cases"][0]["prompt_tokens"]
    run(a.socket, [random.randrange(1000, 100000)] + toks, 16, 1)  # warm-up
    rows = [run(a.socket, [random.randrange(1000, 100000)] + toks, a.max_tokens, 10 + i) for i in range(a.reps)]
    for r in rows:
        print(json.dumps(r))
    med = {k: statistics.median(r[k] for r in rows) for k in ("ttft_s", "prefill_tps", "decode128_tps", "decode_tps")}
    print(json.dumps({"label": a.label, "median": med}))
    if a.out:
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(dict(r, label=a.label)) + "\n")


if __name__ == "__main__":
    main()
