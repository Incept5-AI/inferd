#!/usr/bin/env python3
# b2b.py (2026-11-04): oMLX PR #4902's workload. Back-to-back prompts, one at a time: 3 x ~10.4k, 3 x ~20.8k, 5 x ~34.5k
# tokens of random dictionary words behind a unique marker (no prefix cache can hit), each answered with one greedy
# token; the time is release -> that token (streaming). Calibrates words per token with one untimed probe.
# Stdlib only. Rows: label, round, idx, target, prompt_tokens, ttft_s, error.
import argparse, json, random, time, urllib.request, uuid

def req(url, model, text, timeout):
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 1, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); first = None; usage = None
    r = urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=timeout)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("usage"):
            usage = ev["usage"]
        for c in ev.get("choices", []):
            d = c.get("delta", {})
            if first is None and ((d.get("content") or "") + (d.get("reasoning_content") or "")):
                first = time.time()
    end = time.time()
    return (first or end) - t0, end - t0, (usage or {}).get("prompt_tokens")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--plan", default="10400x3,20800x3,34500x5")
    a = ap.parse_args()
    words = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha()]
    rng = random.Random(4902)
    def text(n_words):
        return f"[marker {uuid.uuid4().hex}]\n" + " ".join(rng.choice(words) for _ in range(n_words)) + \
               "\n\nReply with one word."
    _, _, pt = req(a.url, a.model, text(4000), a.timeout)  # untimed probe
    wpt = 4000 / pt
    plan = [(int(t), int(k)) for t, k in (p.split("x") for p in a.plan.split(","))]
    for rd in range(a.rounds):
        idx = 0
        for tgt, k in plan:
            for _ in range(k):
                row = dict(label=a.label, round=rd, idx=idx, target=tgt, ts=time.strftime("%H:%M:%S"))
                try:
                    ttft, tot, pt = req(a.url, a.model, text(int(tgt * wpt)), a.timeout)
                    row.update(prompt_tokens=pt, ttft_s=round(ttft, 3), total_s=round(tot, 3))
                except Exception as e:
                    row["error"] = repr(e)[:300]
                print(json.dumps(row), flush=True)
                open(a.out, "a").write(json.dumps(row) + "\n")
                idx += 1

if __name__ == "__main__":
    main()
