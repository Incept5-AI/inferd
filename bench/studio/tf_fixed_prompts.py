# fixed.py URL LABEL OUT: speed.py's prompt with fixed run ids (identical prompts across engines), greedy, thinking off;
# streams and timestamps every content chunk, saves full text. 0.5k and 4k prompts x ids 0..4, 512 output tokens.
import json, os, sys, time, urllib.request
sys.path.insert(0, os.path.expanduser("~/inference/bench"))
from speed import PARA, TASK, n_para
url, label, out = sys.argv[1:4]
with open(out, "a") as f:
    for size in ("0.5k", "4k"):
        for rid in range(5):
            body = "".join(PARA.format(i=i) for i in range(n_para(size, False)))
            p = f"[run fixed-{rid:02d}]\n{body}{TASK}"
            req = {"model": "q27", "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 512,
                   "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
            t0 = time.time(); first = None; text = []; usage = None
            r = urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=600)
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]": continue
                ev = json.loads(line[6:])
                if ev.get("usage"): usage = ev["usage"]
                for c in ev.get("choices", []):
                    s = c.get("delta", {}).get("content")
                    if s:
                        if first is None: first = time.time()
                        text.append(s)
            t1 = time.time(); n = usage["completion_tokens"]
            rec = {"label": label, "size": size, "rid": rid, "n": n, "ttft": first - t0, "decode_tps": (n - 1) / (t1 - first), "text": "".join(text)}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"{label} {size} id{rid} n={n} decode {rec['decode_tps']:.0f} tok/s", flush=True)
