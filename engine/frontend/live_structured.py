# Live structured-output test against a running server: 20 schemas/prompts, thinking off and on (effort low).
#   python3 frontend/live_structured.py [--url http://127.0.0.1:8300/v1] [--modes off,low] [--strict] [--out f.json]
# For each case: one request with response_format (validated again here with structured.validate) and one baseline
# request with the same prompt and no response_format (latency reference). Reports validity on the first attempt,
# after retries, and the latency cost. Standard library only (plus structured.py next to this file).
import argparse, json, os, sys, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import structured as S

O = lambda props, req=None, extra=False: {"type": "object", "properties": props,  # noqa: E731
                                          "required": list(props) if req is None else req,
                                          "additionalProperties": extra}
STR, INT, NUM, BOOL = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"}

CASES = [
    ("person", "Ada Lovelace: give her name, birth year and the fields she worked in.",
     O({"name": STR, "born": INT, "fields": {"type": "array", "items": STR, "minItems": 1}})),
    ("sentiment", "Classify the sentiment of: 'The battery died after two hours, but support replaced it fast.'",
     O({"label": {"enum": ["positive", "negative", "mixed", "neutral"]}, "confidence": {"type": "number",
                                                                                         "minimum": 0, "maximum": 1}})),
    ("invoice", "Extract the invoice: 'INV-2291 from Acme Ltd, 3 x widget at 4.50, 1 x gadget at 12.00, VAT 20%.'",
     {"$defs": {"Line": O({"item": STR, "qty": {"type": "integer", "minimum": 1}, "unit_price": NUM})},
      **O({"number": {"type": "string", "pattern": "^INV-\\d+$"}, "vendor": STR,
           "lines": {"type": "array", "items": {"$ref": "#/$defs/Line"}, "minItems": 1}, "vat_rate": NUM})}),
    ("cities", "List exactly three European capitals with their country and approximate population in millions.",
     O({"cities": {"type": "array", "minItems": 3, "maxItems": 3,
                   "items": O({"city": STR, "country": STR, "population_m": NUM})}})),
    ("bool", "Is 2^31 - 1 a prime number? Answer with a JSON boolean field and a one-sentence reason.",
     O({"is_prime": BOOL, "reason": {"type": "string", "maxLength": 200}})),
    ("todo", "Turn this into tasks: 'buy milk tomorrow, call the bank urgently, maybe fix the bike someday'.",
     O({"tasks": {"type": "array", "items": O({"title": STR, "priority": {"enum": ["low", "medium", "high"]},
                                                "due": {"type": ["string", "null"]}})}})),
    ("recipe", "A pancake recipe for 2 people.",
     O({"title": STR, "servings": {"const": 2}, "ingredients": {"type": "array", "items": O(
         {"name": STR, "amount": STR}), "minItems": 3}, "steps": {"type": "array", "items": STR, "minItems": 2}})),
    ("code_review", "Review this Python: `def avg(xs): return sum(xs)/len(xs)`. Report issues.",
     O({"issues": {"type": "array", "items": O({"severity": {"enum": ["info", "warning", "error"]},
                                                 "line": {"type": "integer", "minimum": 1}, "message": STR})},
        "approved": BOOL})),
    ("shape_oneof", "Describe a circle of radius 2 as a shape object.",
     O({"shape": {"oneOf": [O({"kind": {"const": "circle"}, "radius": NUM}),
                             O({"kind": {"const": "rect"}, "w": NUM, "h": NUM})]}})),
    ("sql", "Write a SQL query that counts orders per customer in table orders(customer_id, id); also say which "
     "dialect it is for.", O({"sql": {"type": "string", "minLength": 10}, "dialect": {"enum": [
         "standard", "postgres", "mysql", "sqlite"]}})),
    ("nested", "Describe a small company org chart: a CEO with two direct reports, each with one report.",
     {"$defs": {"Node": O({"name": STR, "title": STR, "reports": {"type": "array",
                                                                  "items": {"$ref": "#/$defs/Node"}}})},
      **O({"root": {"$ref": "#/$defs/Node"}})}),
    ("translate", "Translate 'Good morning, how are you?' into French, German and Spanish.",
     O({"fr": STR, "de": STR, "es": STR})),
    ("dates", "When did Apollo 11 launch and land? Use ISO dates YYYY-MM-DD.",
     O({"launch": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
        "landing": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"}})),
    ("math", "Solve 3x + 7 = 22 and show the steps.",
     O({"steps": {"type": "array", "items": STR, "minItems": 1}, "x": NUM})),
    ("anyof_id", "Give an identifier for the user 'jdoe' (a username string) and their numeric id 1042.",
     O({"ids": {"type": "array", "items": {"anyOf": [STR, INT]}, "minItems": 2, "maxItems": 2}})),
    ("tags_unique", "Give 5 unique lowercase tags for a blog post about sourdough baking.",
     O({"tags": {"type": "array", "items": {"type": "string", "pattern": "^[a-z0-9-]+$"}, "minItems": 5,
                 "maxItems": 5, "uniqueItems": True}})),
    ("weather_tool_args", "Produce arguments for get_weather for Paris in celsius for the next 3 days.",
     O({"city": STR, "unit": {"enum": ["celsius", "fahrenheit"]}, "days": {"type": "integer", "minimum": 1,
                                                                           "maximum": 7}})),
    ("short_summary", "Summarise the plot of Hamlet in at most 120 characters, plus its 3 main characters.",
     O({"summary": {"type": "string", "maxLength": 120}, "characters": {"type": "array", "items": STR,
                                                                         "minItems": 3, "maxItems": 3}})),
    ("json_object", "Return a JSON object mapping each of the planets Mercury, Venus and Earth to its number of moons.",
     None),
    ("matrix", "Return the 3x3 identity matrix.",
     O({"matrix": {"type": "array", "minItems": 3, "maxItems": 3, "items": {
         "type": "array", "minItems": 3, "maxItems": 3, "items": INT}}})),
]

# Harder cases (--hard): constraints a model tends to miss on the first try, to exercise the retry path.
HARD = [
    ("haiku", "Write a haiku about the sea, then explain its imagery in a paragraph.",
     O({"lines": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "string", "maxLength": 22}}})),
    ("product_code", "Make up a product code for a blue electric kettle.",
     O({"code": {"type": "string", "pattern": "^[A-Z]{2}-\\d{4}-[A-Z]$"}})),
    ("tiny_summary", "Summarise the history of the Roman Empire.",
     O({"summary": {"type": "string", "maxLength": 40}})),
    ("tcp_bait", "Explain in detail, with markdown headings and code examples, how TCP's handshake works.",
     O({"summary": {"type": "string", "maxLength": 80}, "steps": {"type": "array", "items": STR, "maxItems": 3}})),
    ("numbers", "Pick four different integers between 10 and 20, sorted descending, and their sum.",
     O({"numbers": {"type": "array", "items": {"type": "integer", "minimum": 10, "maximum": 20}, "minItems": 4,
                    "maxItems": 4, "uniqueItems": True}, "sum": {"type": "integer", "multipleOf": 7}})),
]


def post(url, body, timeout=600):
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("INFERD_URL", "http://127.0.0.1:8300/v1"))
    ap.add_argument("--modes", default="off,low")
    ap.add_argument("--strict", action="store_true", help="send strict: true (constrained decoding when supported)")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--cases", default="")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--hard", action="store_true", help="run the HARD cases instead of the 20")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    nonce = f"[run {time.time():.0f}] "  # a fresh prefix per run: no answer comes from the prefix cache
    rows = []
    pick = set(a.cases.split(",")) if a.cases else None
    for mode in a.modes.split(","):
        kw = {"enable_thinking": False} if mode == "off" else {"reasoning_effort": mode}
        for name, prompt, schema in (HARD if a.hard else CASES):
            if pick and name not in pick:
                continue
            rf = {"type": "json_object"} if schema is None else {"type": "json_schema", "json_schema": {
                "name": name, "schema": schema, "strict": a.strict}}
            msgs = [{"role": "system", "content": nonce + "You are a helpful assistant."},
                    {"role": "user", "content": prompt}]
            base = dict(messages=msgs, temperature=0, max_tokens=a.max_tokens, chat_template_kwargs=kw)
            code, d, dt = post(a.url, {**base, "response_format": rf})
            row = {"mode": mode, "case": name, "http": code, "seconds": round(dt, 2)}
            if code == 200:
                c = d["choices"][0]["message"]["content"]
                sp = S.parse_response_format(rf)
                try:
                    v = json.loads(c)
                    errs = S.validate(v, sp["schema"])
                except ValueError as e:
                    errs = [f"not JSON: {e}"]
                row.update(valid=not errs, errors=errs[:3], attempts=d.get("inferd", {}).get("json_attempts"),
                           constrained=d.get("inferd", {}).get("json_constrained"),
                           completion_tokens=d["usage"]["completion_tokens"], content=c)
            else:
                row.update(valid=False, attempts=(d.get("error") or {}).get("attempts"),
                           errors=[(d.get("error") or {}).get("message", "")[:300]])
            if not a.no_baseline:
                bc, bd, bdt = post(a.url, base)
                row.update(base_seconds=round(bdt, 2),
                           base_tokens=bd["usage"]["completion_tokens"] if bc == 200 else None)
            rows.append(row)
            print(f"{mode:4} {name:18} http {code} valid {row['valid']!s:5} attempts {row.get('attempts')} "
                  f"{dt:6.2f}s" + (f"  (baseline {row['base_seconds']:6.2f}s)" if "base_seconds" in row else "")
                  + ("" if row["valid"] else f"  {row['errors'][:1]}"), flush=True)
    print()
    for mode in a.modes.split(","):
        r = [x for x in rows if x["mode"] == mode]
        if not r:
            continue
        first = sum(1 for x in r if x["valid"] and x["attempts"] == 1)
        ok = sum(1 for x in r if x["valid"])
        t = sum(x["seconds"] for x in r)
        line = f"{mode:4}: valid on first attempt {first}/{len(r)}, after retries {ok}/{len(r)}, " \
               f"retries used {sum((x['attempts'] or 1) - 1 for x in r)}, total {t:.1f}s"
        if all("base_seconds" in x for x in r):
            b = sum(x["base_seconds"] for x in r)
            line += f" vs baseline {b:.1f}s" + (f" ({(t / b - 1) * 100:+.0f}%)" if b > 0 else "")
        toks = sum(x.get("completion_tokens") or 0 for x in r)
        line += f", {toks} completion tokens" + (f", {toks / t:.1f} tok/s" if t > 0 else "")
        print(line)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
