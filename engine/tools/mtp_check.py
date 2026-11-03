# Greedy outputs through the OpenAI front end (temperature 0, thinking off), for comparing a server with MTP on
# against one with it off: the texts must be identical.
#   mtp_check.py <url> <out.json>        writes {prompt name: completion text}
#   mtp_check.py compare <a.json> <b.json>
import json, sys, urllib.request

if sys.argv[1] == "compare":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    same = 0
    for k in a:
        if a[k] == b.get(k):
            same += 1
            continue
        i = next((i for i, (x, y) in enumerate(zip(a[k], b[k])) if x != y), min(len(a[k]), len(b[k])))
        print(f"{k}: DIFFERS at char {i}: {a[k][max(0, i - 40):i + 40]!r} vs {b[k][max(0, i - 40):i + 40]!r}")
    print(f"{same}/{len(a)} identical")
    sys.exit(0 if same == len(a) else 1)

PARA = ("Section {i}. The ledger service reconciles settlement batches against the clearing house file. Each batch "
        "carries an identifier, a value date, a currency, a count of payments and a control sum. When the control sum "
        "disagrees with the recomputed total the batch is quarantined and an operator must approve an adjustment. ")
STORY = ("The lighthouse keeper on Skerry Rock kept a log of every ship that passed, the weather each morning, and the "
         "birds that nested in the cliffs below the lamp. ")
PROMPTS = {
    "short-fact": ("What is the capital of Australia, and why was it chosen over Sydney and Melbourne?", 300),
    "code-prime": ("Write a Python function is_prime(n) with a docstring, then a sieve of Eratosthenes, and tests.", 400),
    "explain": ("Explain how a hash map handles collisions, comparing chaining and open addressing, with examples.", 300),
    "story-600": ("Continue this story for several paragraphs.\n\n" + STORY * 22, 300),
    "spec-1200": ("".join(PARA.format(i=i) for i in range(18)) + "\n\nSummarise the specification above as a numbered "
                  "list of requirements, then write a Python dataclass for a batch.", 300),
    "spec-1700": ("".join(PARA.format(i=i) for i in range(26)) + "\n\nWrite a Python module that implements the batch "
                  "reconciliation described above.", 280),
}
url, out = sys.argv[1], sys.argv[2]
res = {}
for name, (text, mt) in PROMPTS.items():
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": mt,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    res[name] = r["choices"][0]["message"].get("content") or ""
    u = r["usage"]
    print(f"{name}: prompt {u['prompt_tokens']}, completion {u['completion_tokens']}, {r['choices'][0]['finish_reason']}",
          flush=True)
json.dump(res, open(out, "w"), indent=1)
