# vLLM OpenAI-server compatibility probe (docs/API-COMPAT.md). Sends one real request per vLLM endpoint / request field
# and classifies the answer:
#   works     the field or endpoint does what vLLM's does (checked on the response, not only the status)
#   ignored   accepted (200) but without vLLM's effect
#   rejected  a clean OpenAI-style error (4xx/501 with {"error": {"message", ...}})
#   broken    anything else: a 5xx, a hang, a non-JSON error, a wrong shape, a crash
#
#   python3 frontend/test_vllm_compat.py --url http://127.0.0.1:8300 [--key K] [--markdown out.md] [--only substr]
#       live: against a running server (the real model); prints the table, exits 1 if any item is broken or
#       differs from its expected class (EXPECT below)
#   python3 frontend/test_vllm_compat.py
#       fixture mode (CI): the real serve.py over a fake engine (test_hardening's stub tokenizer); the checks that need
#       a real model's text are skipped; exits 1 on a broken item or one off its expected class
# Stdlib only in live mode (urllib), so it runs from any Python.
import argparse, json, os, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
THINK_OFF = {"enable_thinking": False}
Q = "Count from 1 to 30, separated by single spaces. Output only the numbers."

# What inferd is meant to do with each item (docs/API-COMPAT.md). "works" unless listed.
EXPECT = {
    "POST /v1/completions suffix": "rejected", "POST /v1/embeddings": "rejected", "POST /v1/score": "rejected",
    "POST /v1/rerank": "rejected", "POST /rerank": "rejected", "POST /v1/audio/transcriptions": "rejected",
    "POST /v1/responses": "rejected", "GET /v1/nonexistent": "rejected", "prompt_logprobs": "rejected",
    "guided_grammar": "rejected", "cache_salt": "rejected", "best_of > n": "rejected", "chat_template (request)": "rejected",
    "skip_special_tokens": "ignored", "spaces_between_special_tokens": "ignored", "user": "ignored",
    "metadata": "ignored", "content part audio": "rejected", "suffix": "rejected",
    # sampled + seed: the self-tuning draft block changes the random draws a request consumes (docs/API-COMPAT.md)
    "temperature / top_p / top_k / seed (reproducible)": ("works", "ignored"),
}


class Live:
    def __init__(self, url, key=None, timeout=300):
        self.url, self.key, self.timeout = url.rstrip("/"), key, timeout
        self.model = None

    def req(self, method, path, body=None, headers=None, stream=False):
        """-> (status, headers, parsed JSON | text | list of SSE data payloads)."""
        h = {"Content-Type": "application/json", **(headers or {})}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.url + path, data=data, method=method, headers=h)
        try:
            resp = urllib.request.urlopen(r, timeout=self.timeout)
            status = resp.status
        except urllib.error.HTTPError as e:
            resp, status = e, e.code
        raw = resp.read().decode("utf-8", "replace")
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        if stream and status == 200 and "text/event-stream" in hdrs.get("content-type", ""):
            return status, hdrs, [ln[6:] for ln in raw.splitlines() if ln.startswith("data: ")]
        try:
            return status, hdrs, json.loads(raw)
        except ValueError:
            return status, hdrs, raw


class Fixture:
    """serve.py over the fake engine of test_hardening (no model): shapes and validation, not model behaviour. The
    fake answers JSON when the prompt asks for it, honours max_tokens and sends logprobs when asked."""

    def __init__(self):
        sys.path.insert(0, HERE)
        import asyncio
        import test_hardening as th
        self.th, self.asyncio = th, asyncio
        self.model = "m"
        th.StubTok.__len__ = lambda self: 1 << 16
        th.StubTok.convert_ids_to_tokens = lambda self, ids: [chr(i) for i in ids]

        class Fake(th.FakeInferd):
            async def handle(self, m):
                if m["op"] != "gen" or "ctx" in m:
                    return await super().handle(m)
                self.gens.append(m)
                prompt = "".join(chr(t) for t in m.get("tokens", []))
                sch = m.get("json_schema")
                if sch is not None:
                    text = json.dumps(sch["enum"][0]) if "enum" in sch else '"555-123-4567"' if "pattern" in sch \
                        else '{"city": "Tokyo", "population": 1}'
                elif "JSON" in prompt:
                    text = '{"city": "Tokyo", "population": 1}'
                else:
                    text = "hello world"
                toks = [ord(c) for c in text]
                cap = m.get("max_tokens", 4096)
                for t in toks[:cap]:
                    ev = {"id": m["id"], "t": [t]}
                    if "logprobs" in m:
                        ev["lp"] = [-0.5, [x for k in range(m["logprobs"]) for x in (t + k, -0.5 - k)]]
                    self.send(ev)
                self.send({"id": m["id"], "done": "length" if len(toks) > cap else "stop",
                           "p": len(m.get("tokens", [])), "c": min(cap, len(toks)), "cached": 0, "spec": 1})
        self.fake_cls = Fake

    def req(self, method, path, body=None, headers=None, stream=False):
        th, asyncio = self.th, self.asyncio

        async def go():
            eng = th.S.Engine()
            fake = self.fake_cls(eng)
            eng.writer, eng.alive, th.S.engine = fake, True, eng
            eng.background(eng.pump(fake.reader))
            async with th.client() as c:
                r = await c.request(method, path, json=body, headers=headers or {})
            return r
        r = asyncio.run(go())
        hdrs = {k.lower(): v for k, v in r.headers.items()}
        if stream and r.status_code == 200 and "text/event-stream" in hdrs.get("content-type", ""):
            return r.status_code, hdrs, [ln[6:] for ln in r.text.splitlines() if ln.startswith("data: ")]
        try:
            return r.status_code, hdrs, r.json()
        except ValueError:
            return r.status_code, hdrs, r.text


def clean_error(status, j):
    return status in (400, 401, 404, 405, 413, 422, 501) and isinstance(j, dict) and \
        isinstance(j.get("error"), dict) and isinstance(j["error"].get("message"), str)


def std(status, j):
    if status == 200:
        return None
    if clean_error(status, j):
        return "rejected", f"{status}: {j['error']['message'][:90]}"
    return "broken", f"{status}: {str(j)[:90]}"


def chat(p, **kw):
    body = {"model": p.model, "messages": [{"role": "user", "content": Q}], "max_tokens": 24, "temperature": 0,
            "chat_template_kwargs": THINK_OFF}
    body.update(kw)
    return {k: v for k, v in body.items() if v is not None}


def content(j, i=0):
    try:
        return j["choices"][i]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return None


def sse(chunks):
    """-> (parsed chunks, ended with [DONE])."""
    out = []
    for c in chunks:
        if c == "[DONE]":
            return out, True
        out.append(json.loads(c))
    return out, False


# ---- items: (name, run(p, live) -> (class, note)). Each run sends its own request(s).
ITEMS = []


def item(name):
    def deco(f):
        ITEMS.append((name, f))
        return f
    return deco


def simple(name, method, path, body_fn=None, check=None, stream=False):
    @item(name)
    def _(p, live):
        st, h, j = p.req(method, path, body_fn(p) if body_fn else None, stream=stream)
        r = std(st, j)
        if r:
            return r
        return check(j, h, live) if check else ("works", "")


def shape_chat(j, h, live):
    need = ("id", "object", "created", "model", "choices", "usage")
    miss = [k for k in need if k not in j]
    if miss or j["object"] != "chat.completion":
        return "broken", f"missing {miss}"
    c = j["choices"][0]
    if not {"index", "message", "finish_reason"} <= set(c) or c["finish_reason"] not in ("stop", "length", "tool_calls"):
        return "broken", f"choice {c}"
    extra = [k for k in ("stop_reason", "logprobs") if k not in c] + (["system_fingerprint"] if "system_fingerprint" not in j else [])
    if "cached_tokens" not in (j["usage"].get("prompt_tokens_details") or {}):
        return "broken", "usage.prompt_tokens_details.cached_tokens missing"
    return ("works", "") if not extra else ("ignored", f"no {', '.join(extra)}")


simple("POST /v1/chat/completions", "POST", "/v1/chat/completions", lambda p: chat(p), shape_chat)


@item("chat stream (SSE shape, [DONE], role chunk)")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, stream=True), stream=True)
    if r := std(st, j):
        return r
    cs, done = sse(j)
    if not done or not cs:
        return "broken", "no [DONE]"
    if any(c.get("object") != "chat.completion.chunk" for c in cs):
        return "broken", "chunk object"
    fin = [c for c in cs if c["choices"] and c["choices"][0].get("finish_reason")]
    if not fin:
        return "broken", "no finish_reason chunk"
    role = cs[0]["choices"][0]["delta"].get("role") == "assistant"
    return ("works", "") if role else ("ignored", "no first delta with role")


@item("stream_options.include_usage")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, stream=True, stream_options={"include_usage": True}),
                     stream=True)
    if r := std(st, j):
        return r
    cs, _ = sse(j)
    u = [c for c in cs if c.get("usage") and not c["choices"]]
    return ("works", "") if u else ("ignored", "no usage chunk")


@item("stream_options.continuous_usage_stats")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, stream=True, stream_options={
        "include_usage": True, "continuous_usage_stats": True}), stream=True)
    if r := std(st, j):
        return r
    cs, _ = sse(j)
    per = [c for c in cs if c["choices"] and c.get("usage")]
    return ("works", "") if per and len(per) >= len([c for c in cs if c["choices"]]) - 1 else ("ignored", "usage only at the end")


def completion(p, **kw):
    body = {"model": p.model, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}
    body.update(kw)
    return body


def shape_cmpl(j, h, live, n=1):
    if j.get("object") != "text_completion" or len(j.get("choices", [])) != n:
        return "broken", f"object {j.get('object')} choices {len(j.get('choices', []))}"
    if not all(isinstance(c.get("text"), str) for c in j["choices"]):
        return "broken", "text"
    if live and not all(c["text"] for c in j["choices"]):
        return "broken", "empty text"
    return "works", ""


simple("POST /v1/completions (string)", "POST", "/v1/completions", lambda p: completion(p), shape_cmpl)
simple("POST /v1/completions (array of strings)", "POST", "/v1/completions",
       lambda p: completion(p, prompt=["Paris is in", "Berlin is in"]), lambda j, h, l: shape_cmpl(j, h, l, 2))
simple("POST /v1/completions (token ids)", "POST", "/v1/completions",
       lambda p: completion(p, prompt=[785, 6722, 315, 9625, 374] if isinstance(p, Live) else [72, 105]), shape_cmpl)


@item("POST /v1/completions echo")
def _(p, live):
    st, h, j = p.req("POST", "/v1/completions", completion(p, echo=True))
    if r := std(st, j):
        return r
    t = j["choices"][0].get("text", "")
    return ("works", "") if t.startswith("The capital of France is") else ("ignored", repr(t[:40]))


simple("POST /v1/completions suffix", "POST", "/v1/completions", lambda p: completion(p, suffix=" done."))


@item("POST /v1/completions stream")
def _(p, live):
    st, h, j = p.req("POST", "/v1/completions", completion(p, stream=True), stream=True)
    if r := std(st, j):
        return r
    cs, done = sse(j)
    ok = done and cs and all(c.get("object") == "text_completion" and "text" in c["choices"][0] for c in cs if c["choices"])
    return ("works", "") if ok else ("broken", "chunk shape")


def models_check(j, h, live):
    d = (j.get("data") or [{}])[0]
    miss = [k for k in ("id", "object", "created", "owned_by", "max_model_len", "root") if k not in d]
    return ("works", "") if not miss else ("ignored", f"no {', '.join(miss)}")


simple("GET /v1/models", "GET", "/v1/models", None, models_check)
simple("GET /health", "GET", "/health")
simple("GET /ping", "GET", "/ping")
simple("GET /version", "GET", "/version", None,
       lambda j, h, l: ("works", "") if isinstance(j, dict) and "version" in j else ("broken", str(j)[:60]))


@item("POST /tokenize + /detokenize")
def _(p, live):
    st, h, j = p.req("POST", "/tokenize", {"model": p.model, "prompt": "Hello world"})
    if r := std(st, j):
        return r
    if not isinstance(j.get("tokens"), list) or j.get("count") != len(j["tokens"]):
        return "broken", str(j)[:80]
    st, h, k = p.req("POST", "/detokenize", {"model": p.model, "tokens": j["tokens"]})
    if r := std(st, k):
        return r
    st2, _, m = p.req("POST", "/tokenize", {"model": p.model, "messages": [{"role": "user", "content": "hi"}]})
    if st2 != 200 or not m.get("count"):
        return "broken", "messages form"
    return ("works", "") if k.get("prompt") == "Hello world" else ("broken", repr(k)[:60])


@item("GET /metrics")
def _(p, live):
    st, h, j = p.req("GET", "/metrics")
    if st != 200 or not isinstance(j, str):
        return std(st, j) or ("broken", "not text")
    need = ["inferd:num_requests_running", "inferd:num_requests_waiting", "inferd:prompt_tokens_total",
            "inferd:generation_tokens_total", "inferd:prefix_cache_hits_total", "inferd:kv_cache_usage_perc",
            "inferd:time_to_first_token_seconds_bucket", "inferd:e2e_request_latency_seconds_bucket"]
    miss = [n for n in need if n not in j]
    return ("works", "") if not miss else ("broken", f"missing {miss[:3]}")


for ep, body in (("/v1/embeddings", {"input": "x"}), ("/v1/score", {"text_1": "a", "text_2": "b"}),
                 ("/v1/rerank", {"query": "a", "documents": ["b"]}), ("/rerank", {"query": "a", "documents": ["b"]}),
                 ("/v1/audio/transcriptions", {}), ("/v1/responses", {"input": "hi"}), ("/v1/nonexistent", None)):
    simple(("GET " if body is None else "POST ") + ep, "GET" if body is None else "POST", ep,
           (lambda b: (lambda p: dict(b, model=p.model)))(body) if body is not None else None,
           lambda j, h, l: ("broken", "200 for an unsupported endpoint"))


# ---- request fields
@item("model (any name)")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, model="some-alias"))
    if r := std(st, j):
        return ("works", "404 for an unknown model, as vLLM") if st == 404 else r
    return "works", "accepted (vLLM: 404 unless a served name; INFERD_STRICT_MODEL=1)"


def messages_item(name, msgs, ok=None):
    @item(name)
    def _(p, live):
        st, h, j = p.req("POST", "/v1/chat/completions", chat(p, messages=msgs, max_tokens=12))
        if r := std(st, j):
            return r
        c = content(j)
        if c is None:
            return "broken", "no content"
        if live and ok and not ok(c):
            return "ignored", repr(c[:50])
        return "works", ""


messages_item("messages: content parts (text)", [{"role": "user", "content": [
    {"type": "text", "text": "Reply with the single word"}, {"type": "text", "text": "BANANA."}]}],
    lambda c: "BANANA" in c.upper())
messages_item("messages: input_text parts", [{"role": "user", "content": [
    {"type": "input_text", "text": "Reply with the single word BANANA."}]}], lambda c: "BANANA" in c.upper())
messages_item("messages: name, system", [
    {"role": "system", "content": "You answer in one word."}, {"role": "user", "name": "ann", "content": "Say hello."}])
messages_item("messages: developer role", [
    {"role": "developer", "content": "Answer with the single word YES."}, {"role": "user", "content": "Ready?"}],
    lambda c: "YES" in c.upper())
messages_item("messages: assistant tool_calls + tool result", [
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
        "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 21, \"sky\": \"sunny\"}"}],
    lambda c: "21" in c or "sunny" in c.lower())


@item("content part audio")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, messages=[{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]}]))
    return std(st, j) or ("ignored", "audio part accepted")


def count_tokens(j):
    return (j.get("usage") or {}).get("completion_tokens")


@item("max_tokens")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, max_tokens=5))
    if r := std(st, j):
        return r
    return ("works", "") if count_tokens(j) <= 5 and j["choices"][0]["finish_reason"] == "length" else ("broken", str(j["usage"]))


@item("max_completion_tokens")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, max_tokens=None, max_completion_tokens=5))
    if r := std(st, j):
        return r
    return ("works", "") if count_tokens(j) <= 5 else ("ignored", str(j["usage"]))


def sampled(p, **kw):
    return chat(p, temperature=1.0, max_tokens=16, messages=[{"role": "user", "content": "Write a random sentence."}],
                **kw)


@item("temperature / top_p / top_k / seed (reproducible)")
def _(p, live):
    a = p.req("POST", "/v1/chat/completions", sampled(p, top_p=0.9, top_k=40, seed=7))
    b = p.req("POST", "/v1/chat/completions", sampled(p, top_p=0.9, top_k=40, seed=7))
    if r := std(a[0], a[2]) or std(b[0], b[2]):
        return r
    if not live:
        return "works", ""
    return ("works", "") if content(a[2]) == content(b[2]) else ("ignored", "same seed, different text")


@item("min_p")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", sampled(p, min_p=0.99, seed=1))
    if r := std(st, j):
        return r
    if not live:
        return "works", ""
    g = p.req("POST", "/v1/chat/completions", chat(p, max_tokens=16, messages=[
        {"role": "user", "content": "Write a random sentence."}]))[2]
    # min_p 0.99 keeps only tokens within 1% of the best: in practice greedy
    return ("works", "") if content(j) == content(g) else ("ignored", "min_p 0.99 did not act like greedy")


@item("n (> 1)")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", sampled(p, n=3, seed=3))
    if r := std(st, j):
        return r
    ch = j.get("choices", [])
    if len(ch) != 3:
        return "ignored", f"{len(ch)} choice(s)"
    if [c["index"] for c in ch] != [0, 1, 2]:
        return "broken", "indexes"
    return "works", ""


@item("n (> 1) streaming")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", sampled(p, n=2, stream=True), stream=True)
    if r := std(st, j):
        return r
    cs, done = sse(j)
    idx = {c["choices"][0]["index"] for c in cs if c["choices"]}
    return ("works", "") if idx == {0, 1} and done else ("ignored", f"indexes {sorted(idx)}")


@item("best_of = n")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", sampled(p, n=2, best_of=2))
    if r := std(st, j):
        return r
    return ("works", "") if len(j["choices"]) == 2 else ("ignored", f"{len(j['choices'])} choice(s)")


@item("best_of > n")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", sampled(p, n=1, best_of=3))
    return std(st, j) or ("ignored", "accepted")


@item("stop (string / array)")
def _(p, live):
    a = p.req("POST", "/v1/chat/completions", chat(p, stop=" 5"))
    b = p.req("POST", "/v1/chat/completions", chat(p, stop=[" 7", " 5"]))
    if r := std(a[0], a[2]) or std(b[0], b[2]):
        return r
    if not live:
        return "works", ""
    ca, cb = content(a[2]), content(b[2])
    ok = ca.rstrip().endswith("4") and cb.rstrip().endswith("4") and a[2]["choices"][0]["finish_reason"] == "stop"
    sr = a[2]["choices"][0].get("stop_reason")
    if ok and sr != " 5":
        return "works", f"stop_reason {sr!r} (vLLM: the stop string)"
    return ("works", "") if ok else ("broken", repr(ca[-20:]))


@item("include_stop_str_in_output")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, stop=" 5", include_stop_str_in_output=True))
    if r := std(st, j):
        return r
    if not live:
        return "works", ""
    return ("works", "") if content(j).endswith(" 5") else ("ignored", repr(content(j)[-10:]))


def tok_id(p, text):
    st, h, j = p.req("POST", "/tokenize", {"model": p.model, "prompt": text, "add_special_tokens": False})
    if st == 200 and len(j.get("tokens", [])) == 1:
        return j["tokens"][0]
    try:  # before /tokenize existed: the checkpoint's tokenizer, when this Python has transformers
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(os.environ["PROBE_TOKENIZER"])
        ids = t.encode(text, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None
    except Exception:
        return None


@item("stop_token_ids")
def _(p, live):
    tid = tok_id(p, "5") if live else 53
    if tid is None:
        return "broken", "no token id for '5' (set PROBE_TOKENIZER)"
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, stop_token_ids=[tid]))
    if r := std(st, j):
        return r
    if not live:
        return "works", ""
    c = content(j)
    return ("works", f"stop_reason {j['choices'][0].get('stop_reason')}") if c.rstrip().endswith("4") else ("ignored", repr(c[-12:]))


@item("ignore_eos")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, messages=[{"role": "user", "content": "Say OK."}],
                                                          max_tokens=40, ignore_eos=True))
    if r := std(st, j):
        return r
    if not live:
        return "works", ""
    return ("works", "") if count_tokens(j) == 40 else ("ignored", f"{count_tokens(j)} tokens")


@item("min_tokens")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, messages=[{"role": "user", "content": "Say OK."}],
                                                          max_tokens=40, min_tokens=20))
    if r := std(st, j):
        return r
    if not live:
        return "works", ""
    return ("works", "") if count_tokens(j) >= 20 else ("ignored", f"{count_tokens(j)} tokens")


REP = [{"role": "user", "content": "Write the word 'cat' 40 times, separated by spaces."}]


def penalty_item(name, **kw):
    @item(name)
    def _(p, live):
        st, h, j = p.req("POST", "/v1/chat/completions", chat(p, messages=REP, max_tokens=40, **kw))
        if r := std(st, j):
            return r
        if not live:
            return "works", ""
        base = p.req("POST", "/v1/chat/completions", chat(p, messages=REP, max_tokens=40))[2]
        return ("works", "") if content(j) != content(base) else ("ignored", "same output as without it")


penalty_item("presence_penalty", presence_penalty=2.0)
penalty_item("frequency_penalty", frequency_penalty=2.0)
penalty_item("repetition_penalty", repetition_penalty=1.8)


@item("logit_bias")
def _(p, live):
    tid = tok_id(p, " banana") if live else 98
    if tid is None:
        return "broken", "no token id (set PROBE_TOKENIZER)"
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, logit_bias={str(tid): 100}, max_tokens=4))
    if r := std(st, j):
        return r
    if not live:
        return "works", ""
    return ("works", "") if "banana" in content(j) else ("ignored", repr(content(j)[:30]))


@item("logprobs + top_logprobs (chat)")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, logprobs=True, top_logprobs=3, max_tokens=6))
    if r := std(st, j):
        return r
    lp = j["choices"][0].get("logprobs")
    if not lp:
        return "ignored", "no choices[].logprobs"
    c = lp.get("content") or []
    if not c or not all({"token", "logprob", "bytes", "top_logprobs"} <= set(e) and len(e["top_logprobs"]) == 3 for e in c):
        return "broken", str(c[:1])[:80]
    if live and (abs(c[0]["logprob"] - c[0]["top_logprobs"][0]["logprob"]) > 1e-4 or c[0]["logprob"] > 0):
        return "broken", "greedy token is not the top logprob"
    return "works", ""


@item("logprobs (chat, streaming)")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, logprobs=True, top_logprobs=2, max_tokens=6, stream=True),
                     stream=True)
    if r := std(st, j):
        return r
    cs, _ = sse(j)
    n = sum(len((c["choices"][0].get("logprobs") or {}).get("content") or []) for c in cs if c["choices"])
    return ("works", "") if n >= 1 else ("ignored", "no logprobs in chunks")


@item("logprobs (completions, int)")
def _(p, live):
    st, h, j = p.req("POST", "/v1/completions", completion(p, logprobs=2, max_tokens=5))
    if r := std(st, j):
        return r
    lp = j["choices"][0].get("logprobs")
    if not lp:
        return "ignored", "no logprobs"
    ok = all(k in lp for k in ("tokens", "token_logprobs", "top_logprobs", "text_offset")) and \
        len(lp["tokens"]) == len(lp["token_logprobs"]) >= 1
    return ("works", "") if ok else ("broken", str(lp)[:80])


@item("prompt_logprobs")
def _(p, live):
    st, h, j = p.req("POST", "/v1/completions", completion(p, prompt_logprobs=1))
    return std(st, j) or ("ignored", "accepted")


simple("skip_special_tokens", "POST", "/v1/chat/completions", lambda p: chat(p, skip_special_tokens=False),
       lambda j, h, l: ("ignored", "accepted; stop tokens never reach the text either way"))
simple("spaces_between_special_tokens", "POST", "/v1/chat/completions",
       lambda p: chat(p, spaces_between_special_tokens=False), lambda j, h, l: ("ignored", "accepted"))


SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
          "required": ["city", "population"], "additionalProperties": False}
JQ = [{"role": "user", "content": "Give the largest city in Japan and its population as JSON."}]


def json_ok(c, schema=True):
    try:
        v = json.loads(c)
    except (TypeError, ValueError):
        return False
    return isinstance(v, dict) and (not schema or (isinstance(v.get("city"), str) and isinstance(v.get("population"), int)))


def structured_item(name, check=json_ok, **kw):
    @item(name)
    def _(p, live):
        st, h, j = p.req("POST", "/v1/chat/completions", chat(p, messages=JQ, max_tokens=120, **kw))
        if r := std(st, j):
            return r
        if not live:
            return "works", ""
        return ("works", "") if check(content(j)) else ("ignored", repr(content(j)[:50]))


structured_item("response_format text", check=lambda c: bool(c), response_format={"type": "text"})
structured_item("response_format json_object", check=lambda c: json_ok(c, False), response_format={"type": "json_object"})
structured_item("response_format json_schema strict", response_format={"type": "json_schema", "json_schema": {
    "name": "city", "schema": SCHEMA, "strict": True}})
structured_item("guided_json", guided_json=SCHEMA)
structured_item("structured_outputs.json (vLLM >= 0.10)", structured_outputs={"json": SCHEMA})


@item("guided_choice")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, messages=[
        {"role": "user", "content": "Is the sentiment of 'I love this' positive or negative?"}],
        guided_choice=["positive", "negative"], max_tokens=20))
    if r := std(st, j):
        return r
    if not live:
        return "works", ""
    return ("works", "") if content(j) in ("positive", "negative") else ("ignored", repr(content(j)[:40]))


@item("guided_regex")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, messages=[
        {"role": "user", "content": "Give a US phone number."}], guided_regex=r"[0-9]{3}-[0-9]{3}-[0-9]{4}",
        max_tokens=30))
    if r := std(st, j):
        return r
    if not live:
        return "works", ""
    import re
    return ("works", "") if re.fullmatch(r"[0-9]{3}-[0-9]{3}-[0-9]{4}", content(j) or "") else ("ignored", repr(content(j)[:40]))


@item("guided_grammar")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, guided_grammar='root ::= "yes" | "no"'))
    return std(st, j) or ("ignored", "accepted")


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                           "parameters": {"type": "object", "properties": {
                                               "city": {"type": "string"}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "get_time", "description": "Local time in a city",
                                           "parameters": {"type": "object", "properties": {
                                               "city": {"type": "string"}}, "required": ["city"]}}}]
TQ = [{"role": "user", "content": "What's the weather in Paris right now?"}]


def tool_item(name, check, **kw):
    @item(name)
    def _(p, live):
        st, h, j = p.req("POST", "/v1/chat/completions", chat(p, messages=kw.pop("messages", TQ), tools=TOOLS,
                                                              max_tokens=200, **kw))
        if r := std(st, j):
            return r
        if not live:
            return "works", ""
        return check(j["choices"][0])


def calls(ch):
    return [c["function"]["name"] for c in ch["message"].get("tool_calls") or []]


tool_item("tools + tool_choice auto", lambda ch: ("works", "") if calls(ch) == ["get_weather"] and
          ch["finish_reason"] == "tool_calls" else ("broken", f"{calls(ch)} {ch['finish_reason']}"), tool_choice="auto")
tool_item("tool_choice none", lambda ch: ("works", "") if not calls(ch) else ("ignored", f"called {calls(ch)}"),
          tool_choice="none")
tool_item("tool_choice required", lambda ch: ("works", "") if calls(ch) and set(calls(ch)) <= {"get_weather", "get_time"}
          else ("ignored", f"calls {calls(ch)}"),
          tool_choice="required", messages=[{"role": "user", "content": "Hello there!"}])
tool_item("tool_choice named function", lambda ch: ("works", "") if calls(ch) == ["get_time"] else
          ("ignored", f"called {calls(ch)}"), tool_choice={"type": "function", "function": {"name": "get_time"}})
tool_item("parallel_tool_calls false", lambda ch: ("works", "") if len(calls(ch)) == 1 else ("ignored", f"{calls(ch)}"),
          parallel_tool_calls=False, messages=[{"role": "user", "content":
                                                "Get the weather in Paris and in Tokyo, and the time in both."}])


@item("chat_template (request)")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, chat_template="{{ messages[0].content }}"))
    return std(st, j) or ("ignored", "accepted")


@item("chat_template_kwargs enable_thinking / reasoning_content")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, chat_template_kwargs={"enable_thinking": True,
                                                                                   "reasoning_effort": "low"},
                                                          max_tokens=400 if live else 24,
                                                          messages=[{"role": "user", "content": "What is 2+2?"}]))
    if r := std(st, j):
        return r
    m = j["choices"][0]["message"]
    if live and not m.get("reasoning_content"):
        return "broken", "no reasoning_content"
    return ("works", "") if not live or "reasoning" in m else ("works", "reasoning_content only (no vLLM 'reasoning')")


@item("add_generation_prompt false + continue_final_message")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, messages=[
        {"role": "user", "content": "List three fruits."}, {"role": "assistant", "content": "Here are three fruits: kiwi,"}],
        add_generation_prompt=False, continue_final_message=True, max_tokens=10))
    if r := std(st, j):
        return r
    if not live:
        return "works", ""
    c = content(j)
    return ("works", "") if "Here are" not in c and "kiwi" not in c else ("ignored", repr(c[:30]))


@item("echo (chat)")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, messages=[
        {"role": "user", "content": "List three fruits."}, {"role": "assistant", "content": "Here are three fruits: kiwi,"}],
        add_generation_prompt=False, continue_final_message=True, echo=True, max_tokens=10))
    if r := std(st, j):
        return r
    if not live:
        return "works", ""
    c = content(j)
    ok = c.startswith("Here are three fruits: kiwi,") and "Here are" not in c[10:]
    return ("works", "") if ok else ("ignored", repr(c[:30]))


simple("documents", "POST", "/v1/chat/completions", lambda p: chat(p, documents=[{"title": "a", "text": "b"}]))
simple("user", "POST", "/v1/chat/completions", lambda p: chat(p, user="u1"), lambda j, h, l: ("ignored", "accepted"))
simple("metadata", "POST", "/v1/chat/completions", lambda p: chat(p, metadata={"k": "v"}),
       lambda j, h, l: ("ignored", "accepted"))


@item("request_id / X-Request-Id")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, max_tokens=2), headers={"X-Request-Id": "probe-123"})
    if r := std(st, j):
        return r
    ok_h = h.get("x-request-id") == "probe-123"
    st, h2, j2 = p.req("POST", "/v1/chat/completions", chat(p, max_tokens=2, request_id="probe-456"))
    ok_b = st == 200 and "probe-456" in j2.get("id", "")
    return ("works", "") if ok_h and ok_b else ("ignored", f"header echo {ok_h}, id from request_id {ok_b}")


@item("priority")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, priority=1))
    return std(st, j) or ("works", "inferd semantics: higher = kept on preemption")


@item("truncate_prompt_tokens")
def _(p, live):
    long = [{"role": "user", "content": "word " * 300 + "\nSay OK."}]
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, messages=long, truncate_prompt_tokens=50, max_tokens=4))
    if r := std(st, j):
        return r
    pt = j["usage"]["prompt_tokens"]
    return ("works", "") if pt <= 50 else ("ignored", f"prompt_tokens {pt}")


@item("cache_salt")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, cache_salt="abc"))
    return std(st, j) or ("ignored", "accepted, cache shared")


@item("error object shape (bad request)")
def _(p, live):
    st, h, j = p.req("POST", "/v1/chat/completions", chat(p, temperature="hot"))
    return ("works", "") if clean_error(st, j) and st == 400 else ("broken", f"{st} {str(j)[:60]}")


def run_all(p, live, only=None):
    rows = []
    for name, f in ITEMS:
        if only and only not in name:
            continue
        t0 = time.time()
        try:
            cls, note = f(p, live)
        except Exception as e:  # a hang (timeout) or a crash in the server's answer
            cls, note = "broken", f"{type(e).__name__}: {str(e)[:80]}"
        rows.append((name, cls, note, time.time() - t0))
        print(f"{cls:9} {name:55} {note}", flush=True)
    return rows


def markdown(rows):
    out = ["| Item | inferd | Notes |", "|---|---|---|"]
    for name, cls, note, _ in rows:
        out.append(f"| {name} | {cls} | {note.replace('|', '/')} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--key", default=os.environ.get("INFERD_API_KEY"))
    ap.add_argument("--markdown")
    ap.add_argument("--only")
    ap.add_argument("--no-expect", action="store_true", help="report only; exit 0 unless an item is broken")
    a = ap.parse_args()
    if a.url:
        p, live = Live(a.url, a.key), True
        p.model = p.req("GET", "/v1/models")[2]["data"][0]["id"]
    else:
        p, live = Fixture(), False
    rows = run_all(p, live, a.only)
    if a.markdown:
        open(a.markdown, "w").write(markdown(rows))
    def on_target(r):
        want = EXPECT.get(r[0], "works")
        return r[1] in (want if isinstance(want, tuple) else (want,))
    bad = [r for r in rows if r[1] == "broken" or (not a.no_expect and not on_target(r))]
    counts = {c: sum(r[1] == c for r in rows) for c in ("works", "ignored", "rejected", "broken")}
    print(f"\n{len(rows)} items: " + ", ".join(f"{v} {k}" for k, v in counts.items()) +
          (f"; {len(bad)} off target: " + ", ".join(r[0] for r in bad) if bad else "; all on target"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
