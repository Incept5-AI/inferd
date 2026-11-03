# Prefix-cache check for agents sharing a system prompt (2026-11-01): four requests, one 2.5k-token system prompt,
# ~1.8k-token distinct user turns. With branch-point snapshots the 2nd request on reports cached_tokens 2560.
#   python3 branch_snap_test.py http://127.0.0.1:PORT
import json, sys, time, urllib.request
url = sys.argv[1]
sysmsg = "You are a careful assistant for a logistics company. " + " ".join(f"Rule {i}: always check shipment {i*7} against manifest {i*13} before dispatch." for i in range(120))
def ask(q):
    body = {"model": "m", "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": q}],
            "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})))
    u = r["usage"]
    print(f"{q[:28]:30s} prompt {u['prompt_tokens']:5d} cached {u.get('prompt_tokens_details',{}).get('cached_tokens', u.get('cached_tokens'))!s:>6} {time.time()-t:.2f}s")
for k, q in enumerate(["What is rule 5 about?", "Summarise rule 17.", "Which manifest goes with shipment 70?", "Is rule 3 strict?"]):
    ask(q + " Context: " + " ".join(f"item {k}-{j} weighs {j*3+k} kg" for j in range(150)))
