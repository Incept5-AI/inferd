#!/usr/bin/env python3
"""Structured output: OpenAI `response_format` against inferd (or any OpenAI-compatible server).

    python3 structured_output.py            # server on http://127.0.0.1:8300 (INFERD_URL to change)
    python3 structured_output.py --strict   # also ask for constrained decoding (strict: true)
Python standard library only.
"""
import json, os, sys, time, urllib.error, urllib.request

URL = os.environ.get("INFERD_URL", "http://127.0.0.1:8300/v1")
STRICT = "--strict" in sys.argv


def post(body):
    req = urllib.request.Request(URL + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return 200, urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        return e.code, e


model = json.load(urllib.request.urlopen(URL + "/models"))["data"][0]["id"]
print(f"server {URL}, model {model}{', strict' if STRICT else ''}\n")
NONE = {"enable_thinking": False}  # effort none: the JSON comes straight back

# An invoice extractor's schema, as a pydantic model would emit it ($defs + $ref, additionalProperties false).
INVOICE = {
    "$defs": {"Line": {"type": "object", "properties": {
        "item": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}, "unit_price": {"type": "number"}},
        "required": ["item", "qty", "unit_price"], "additionalProperties": False}},
    "type": "object", "properties": {
        "number": {"type": "string", "pattern": "^INV-\\d+$"},
        "vendor": {"type": "string"},
        "currency": {"enum": ["GBP", "EUR", "USD"]},
        "lines": {"type": "array", "items": {"$ref": "#/$defs/Line"}, "minItems": 1},
        "vat_rate": {"type": "number", "minimum": 0, "maximum": 1}},
    "required": ["number", "vendor", "currency", "lines", "vat_rate"], "additionalProperties": False}
EMAIL = ("Hi, please find attached invoice INV-2291 from Acme Ltd: 3 x widget at £4.50, 1 x gadget at £12.00, "
         "plus 2 hours of installation at £40/hour. VAT at 20% applies. Thanks, Sam")


def show(title, code, r, t):
    print(f"== {title}")
    if code != 200:
        err = json.load(r)["error"]
        print(f"HTTP {code} {err.get('code')}: {err['message'][:200]}")
        print(f"last_output: {err.get('last_output', '')[:200]!r}\n")
        return None
    d = json.load(r)
    c = d["choices"][0]["message"]["content"]
    print(c)
    print(f"finish_reason={d['choices'][0]['finish_reason']}  attempts={d.get('inferd', {}).get('json_attempts')}  "
          f"completion_tokens={d['usage']['completion_tokens']}  {time.time() - t:.2f}s\n")
    if "inferd" not in d or "json_attempts" not in d.get("inferd", {}):
        sys.exit("This server ignored response_format: it needs inferd's structured-output front end "
                 "(engine/frontend/serve.py from 2 Nov 2026 or later).")
    return json.loads(c)


# 1. json_schema: extract an invoice. The content is the JSON itself (no fences, no prose), already validated.
t = time.time()
inv = show("1. json_schema: invoice extraction", *post({
    "model": model, "temperature": 0, "max_tokens": 1024, "chat_template_kwargs": NONE,
    "messages": [{"role": "user", "content": "Extract the invoice from this email:\n\n" + EMAIL}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "invoice", "schema": INVOICE,
                                                               "strict": STRICT}}}), t)
if inv:
    net = sum(x["qty"] * x["unit_price"] for x in inv["lines"])
    print(f"   -> net {net:.2f} {inv['currency']}, gross {net * (1 + inv['vat_rate']):.2f} (computed by the client)\n")

# 2. json_object: any JSON object, with thinking on (effort low). The reasoning stays in reasoning_content.
t = time.time()
show("2. json_object, effort low", *post({
    "model": model, "temperature": 0, "max_tokens": 2048,
    "chat_template_kwargs": {"reasoning_effort": "low"},
    "messages": [{"role": "user", "content": "Map each Apollo mission that landed on the Moon to its landing year, "
                                             "as a JSON object."}],
    "response_format": {"type": "json_object"}}), t)

# 3. Streaming: the server buffers until the JSON validates, then sends it (here in one content delta).
print("== 3. json_schema, streamed")
t = time.time()
code, r = post({
    "model": model, "temperature": 0, "max_tokens": 512, "stream": True, "chat_template_kwargs": NONE,
    "messages": [{"role": "user", "content": "Classify: 'The battery died after two hours, but support replaced it "
                                             "the same day.'"}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "sentiment", "strict": STRICT, "schema": {
        "type": "object", "properties": {"label": {"enum": ["positive", "negative", "mixed"]},
                                         "aspects": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
        "required": ["label", "aspects"], "additionalProperties": False}}}})
deltas = 0
for line in r:
    line = line.decode().strip()
    if line.startswith("data: {"):
        ch = json.loads(line[6:])["choices"]
        if ch and ch[0]["delta"].get("content"):
            deltas += 1
            print(ch[0]["delta"]["content"])
        if ch and ch[0]["finish_reason"]:
            print(f"finish_reason={ch[0]['finish_reason']}  ({deltas} content delta(s), {time.time() - t:.2f}s)\n")

# 4. A schema the model keeps missing: each failure goes back to the model with the validation error ("json_retries",
#    default 2). After the last attempt the server answers HTTP 422 json_validation_failed with the last output.
#    With --strict the length limit is enforced while decoding, so the first attempt is valid by construction.
t = time.time()
show("4. a 500-501 character limit (json_retries 1)", *post({
    "model": model, "temperature": 0, "max_tokens": 256, "chat_template_kwargs": NONE, "json_retries": 1,
    "messages": [{"role": "user", "content": "Give me a one-paragraph history of the Roman Empire."}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "tiny", "strict": STRICT, "schema": {
        "type": "object", "properties": {"history": {"type": "string", "minLength": 500, "maxLength": 501}},
        "required": ["history"], "additionalProperties": False}}}}), t)
