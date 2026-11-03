# Structured output (response_format) on canned model text: stdlib unittest, no engine, no model needed.
#   python3 frontend/test_structured.py
# structured.py is imported directly; serve.py's structured_chat is lifted out of its source with ast (serve.py parses
# argv and loads a tokenizer at import) and driven by a fake `complete` that replays canned replies.
import ast, asyncio, json, os, sys, time, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import structured as S

PERSON = {"type": "object", "properties": {
    "name": {"type": "string", "minLength": 1}, "age": {"type": "integer", "minimum": 0, "maximum": 150},
    "email": {"type": ["string", "null"]}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
    "required": ["name", "age"], "additionalProperties": False}


def spec(schema):
    return S.parse_response_format({"type": "json_schema", "json_schema": {"name": "x", "schema": schema}})


class ParseFormat(unittest.TestCase):
    def test_kinds(self):
        self.assertIsNone(S.parse_response_format(None))
        self.assertIsNone(S.parse_response_format({"type": "text"}))
        self.assertEqual(S.parse_response_format({"type": "json_object"})["schema"], {"type": "object"})
        sp = S.parse_response_format({"type": "json_schema", "json_schema": {"name": "p", "schema": PERSON,
                                                                             "strict": True}})
        self.assertEqual((sp["kind"], sp["name"], sp["strict"]), ("json_schema", "p", True))
        for bad in ("json", {"type": "xml"}, {"type": "json_schema"}, {"type": "json_schema", "json_schema": {
                "schema": 3}}):
            with self.assertRaises(ValueError):
                S.parse_response_format(bad)

    def test_instruction_on_last_user_turn(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
        out = S.add_instruction(msgs, "DO JSON")
        self.assertEqual(out[0], msgs[0])  # the system prompt (a cached prefix) is untouched
        self.assertEqual(out[1]["content"], "q\n\nDO JSON")
        self.assertEqual(msgs[1]["content"], "q")  # not mutated
        parts = S.add_instruction([{"role": "user", "content": [{"type": "text", "text": "img"}]}], "J")
        self.assertEqual(parts[0]["content"][-1], {"type": "text", "text": "J"})
        tool = S.add_instruction([{"role": "user", "content": "q"}, {"role": "tool", "content": "r"}], "J")
        self.assertEqual(tool[-1], {"role": "user", "content": "J"})
        self.assertIn('"required": ["name", "age"]', S.instruction(spec(PERSON)))


class Extract(unittest.TestCase):
    def ok(self, text, schema=None, want=None):
        out, err = S.check(text, spec(schema) if schema is not None else S.parse_response_format(
            {"type": "json_object"}))
        self.assertIsNone(err, err)
        if want is not None:
            self.assertEqual(json.loads(out), want)
        return out

    def test_bare(self):
        self.assertEqual(self.ok('{"a": 1}'), '{"a": 1}')

    def test_fenced(self):
        self.ok('```json\n{"a": [1, 2]}\n```', want={"a": [1, 2]})
        self.ok('Sure:\n```\n{"a": 1}\n```\nHope that helps.', want={"a": 1})

    def test_prose_around(self):
        self.ok('Here is the result: {"name": "Ann", "age": 31} -- done.', PERSON, {"name": "Ann", "age": 31})

    def test_first_valid_candidate_wins(self):  # an example object that fails the schema, then the real answer
        self.ok('e.g. {"name": ""} then {"name": "Bo", "age": 2}', PERSON, {"name": "Bo", "age": 2})

    def test_failures(self):
        js = S.parse_response_format({"type": "json_object"})
        self.assertIn("empty", S.check("", js)[1])
        self.assertIn("no parseable", S.check("I cannot do that.", js)[1])
        self.assertIn("no parseable", S.check('{"a": 1', js)[1])  # truncated
        self.assertIn("no parseable", S.check('{"a": NaN}', js)[1])  # not JSON
        self.assertIn("expected object", S.check("[1, 2]", js)[1])  # json_object wants an object


class Validate(unittest.TestCase):
    def errs(self, v, schema):
        return S.validate(v, schema)

    def test_person(self):
        self.assertEqual(self.errs({"name": "A", "age": 3, "email": None, "tags": ["x"]}, PERSON), [])
        e = self.errs({"name": "", "age": 3.5, "nick": "z"}, PERSON)
        self.assertTrue(any("minimum is 1" in x for x in e), e)
        self.assertTrue(any("$.age: expected integer" in x for x in e), e)
        self.assertTrue(any("'nick' is not allowed" in x for x in e), e)
        self.assertIn("missing required property 'age'", self.errs({"name": "A"}, PERSON)[0])
        self.assertTrue(self.errs({"name": "A", "age": 200}, PERSON))
        self.assertTrue(self.errs({"name": "A", "age": 1, "tags": ["a", "b", "c", "d"]}, PERSON))

    def test_types(self):
        self.assertEqual(self.errs(2.0, {"type": "integer"}), [])  # 2.0 is an integer in JSON Schema
        self.assertTrue(self.errs(True, {"type": "integer"}))  # a boolean is not
        self.assertTrue(self.errs(True, {"type": "number"}))
        self.assertEqual(self.errs(None, {"type": "string", "nullable": True}), [])

    def test_enum_const(self):
        self.assertEqual(self.errs("b", {"enum": ["a", "b"]}), [])
        self.assertTrue(self.errs("c", {"enum": ["a", "b"]}))
        self.assertTrue(self.errs(True, {"enum": [1]}))  # true is not 1
        self.assertEqual(self.errs(1.0, {"const": 1}), [])
        self.assertTrue(self.errs({"a": [1]}, {"const": {"a": [2]}}))

    def test_nested_and_ref(self):  # pydantic-style $defs
        schema = {"$defs": {"Item": {"type": "object", "properties": {"sku": {"type": "string", "pattern": "^[A-Z]{3}-\\d+$"},
                                                                      "qty": {"type": "integer", "exclusiveMinimum": 0}},
                                     "required": ["sku", "qty"]}},
                  "type": "object", "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"},
                                                             "minItems": 1}}, "required": ["items"]}
        self.assertEqual(self.errs({"items": [{"sku": "ABC-1", "qty": 2}]}, schema), [])
        e = self.errs({"items": [{"sku": "ABC-1", "qty": 2}, {"sku": "abc", "qty": 0}]}, schema)
        self.assertTrue(any(x.startswith("$.items[1].sku") for x in e), e)
        self.assertTrue(any(x.startswith("$.items[1].qty") for x in e), e)
        self.assertTrue(self.errs({"items": []}, schema))

    def test_combinators(self):
        s = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        self.assertEqual(self.errs(3, s), [])
        self.assertTrue(self.errs(3.5, s))
        one = {"oneOf": [{"type": "integer"}, {"type": "number"}]}
        self.assertTrue(self.errs(3, one))  # matches both
        self.assertEqual(self.errs(3.5, one), [])
        self.assertTrue(self.errs(1, {"not": {"type": "integer"}}))
        self.assertTrue(self.errs({"a": 1}, {"allOf": [{"required": ["a"]}, {"required": ["b"]}]}))

    def test_arrays(self):
        self.assertTrue(self.errs([1, 1], {"type": "array", "uniqueItems": True}))
        self.assertEqual(self.errs([1, "a"], {"prefixItems": [{"type": "integer"}, {"type": "string"}]}), [])
        self.assertTrue(self.errs([1, "a", 3], {"prefixItems": [{"type": "integer"}], "items": {"type": "integer"}}))
        self.assertEqual(self.errs({"x": 1, "y": 2}, {"additionalProperties": {"type": "integer"}}), [])
        self.assertTrue(self.errs({"x": "1"}, {"additionalProperties": {"type": "integer"}}))


class Bounded(unittest.TestCase):  # Codex sweep 2026-11-02, S02 and the schema refusals
    def test_review_case_is_linear(self):  # (a+)+$ on 28 a's + '!': > 1 s with Python re, instant here
        t = time.time()
        self.assertFalse(S.pattern("(a+)+$").search("a" * 28 + "!"))
        self.assertTrue(S.pattern("(a+)+$").search("a" * 5000))
        self.assertFalse(S.pattern("^(a|aa)+$").search("a" * 20000 + "b"))
        self.assertLess(time.time() - t, 1.0)

    def test_same_as_re_where_defined(self):  # differential: our subset agrees with Python re on ASCII text
        import random, re
        pats = ["^INV-\\d+$", "[a-z]{2}(-[0-9]{1,3})?", "^\\d{4}-\\d{2}-\\d{2}$", "^(?:ab|a)c*$", "x.y", "^[^,]+,[^,]+$",
                "colou?r", "^\\w+@\\w+\\.com$", "a{2,3}b{0,}", "^\\s*$", "[\\]\\-]+", "^[A-Z][a-z]*( [A-Z][a-z]*)*$"]
        rnd = random.Random(1)
        alphabet = "aAbcxyz0123456789-,.@ _]\n" + "INV-colour"
        for p in pats:
            for _ in range(300):
                t = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 14)))
                if rnd.random() < 0.3:
                    t = rnd.choice(["INV-2291", "ab-12", "2026-11-02", "abccc", "color", "a@b.com", "aab", "Ann Lee"])
                py = p[:-1] + "\\Z" if p.endswith("$") else p  # Python's $ also matches before a final newline
                want = re.search(py, t) is not None
                self.assertEqual(S.pattern(p).search(t), want, (p, t))

    def test_patterns_refused(self):
        for bad in ["(a)\\1", "(?=a)b", "(?i)a", "a\\bc", "\\p{L}", "é", "a(b", "a)b", "[z-a]", "a{65}", "^a|^b",
                    "(" * 50000 + "a" + ")" * 50000, "(" * 33 + "a" + ")" * 33, "a" * 600, "((a{64}){64}){64}"]:
            t = time.time()
            with self.assertRaises(ValueError, msg=bad[:40]):
                S.Pattern(bad)
            self.assertLess(time.time() - t, 1.0, bad[:40])

    def test_schemas_refused(self):
        def bad(schema, frag):
            with self.assertRaises(ValueError) as cm:
                S.parse_response_format({"type": "json_schema", "json_schema": {"name": "x", "schema": schema}})
            self.assertIn(frag, str(cm.exception))
        bad({"type": "strng"}, "unknown type")
        bad({"if": {"type": "string"}, "then": {}}, "'if' is not supported")
        bad({"type": "object", "propertyNames": {"maxLength": 3}}, "propertyNames")
        bad({"$ref": "https://example.com/s.json"}, "only local")
        bad({"$ref": "#/$defs/Nope"}, "does not resolve")
        bad({"type": "string", "pattern": "(a)\\1"}, "escape")
        bad({"type": "object", "properties": {"a": False}}, "false subschemas")
        bad({"type": "string", "minLength": -1}, "non-negative")
        bad({"type": "number", "multipleOf": 0}, "positive")
        bad({"type": "object", "required": "a"}, "list of strings")
        bad({"anyOf": []}, "non-empty")
        bad({"enum": [0] * 20001}, "enum/const")
        bad({"description": "x" * (70 << 10)}, "larger than")
        deep = {}
        for _ in range(60):
            deep = {"type": "array", "items": deep}
        bad(deep, "nested deeper")
        with self.assertRaises(ValueError):
            S.parse_response_format({"type": "json_schema", "json_schema": {"name": "x", "schema": True}})

    def test_schemas_accepted(self):  # annotations and unknown non-validation keywords are fine
        sp = S.parse_response_format({"type": "json_schema", "json_schema": {"name": "x", "schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "T", "description": "d",
            "type": "object", "properties": {"d": {"type": "string", "format": "date", "examples": ["2026-01-01"]},
                                             "n": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None}},
            "required": ["d"], "additionalProperties": False, "x-vendor": 1}}})
        self.assertEqual(sp["checked_only"], ["anyOf"])
        self.assertEqual(S.parse_response_format({"type": "json_schema", "json_schema": {
            "schema": PERSON}})["checked_only"], ["maximum", "minimum"])

    def test_validation_is_bounded(self):  # exponential anyOf over a recursive $ref: refused in bounded time
        schema = {"$defs": {"N": {"anyOf": [{"$ref": "#/$defs/N"}, {"$ref": "#/$defs/N"}]}}, "$ref": "#/$defs/N"}
        t = time.time()
        out, err = S.check("1", {"schema": schema})
        self.assertIsNone(out)
        self.assertLess(time.time() - t, 10.0)
        t = time.time()  # a pattern over a long reply: linear, and capped by the budget
        out, err = S.check(json.dumps({"s": "a" * 200000}), spec({"type": "object", "properties": {
            "s": {"type": "string", "pattern": "(a|aa)+$"}}}))
        self.assertLess(time.time() - t, 10.0)

    def test_deep_reply(self):  # absurd nesting in the model's text is no crash
        js = S.parse_response_format({"type": "json_object"})
        self.assertIsNone(S.check("[" * 100000 + "]" * 100000, js)[0])

    def test_unique_items_linear(self):
        self.assertEqual(S.validate(list(range(50000)), {"uniqueItems": True}), [])
        self.assertTrue(S.validate([1, 1.0], {"uniqueItems": True}))
        self.assertEqual(S.validate([1, True, "1", [1], {"a": 1}, {"a": True}], {"uniqueItems": True}), [])


# ---------------------------------------------------------------- structured_chat with canned replies

_SRC = os.path.join(HERE, "serve.py")
_node = next(n for n in ast.parse(open(_SRC).read()).body
             if isinstance(n, ast.AsyncFunctionDef) and n.name == "structured_chat")


class FakeJSON:
    def __init__(self, content, status_code=200):
        self.status_code, self.content = status_code, content
        self.body = json.dumps(content).encode()


class FakeStream:
    def __init__(self, gen, media_type=None):
        self.gen = gen


def harness(replies, status=200):
    """-> (structured_chat bound to a fake complete that answers with `replies` in turn, list of bodies sent)."""
    sent, it = [], iter(replies)

    async def complete(body):
        sent.append(body)
        text = next(it)
        msg = {"role": "assistant", "content": text} if isinstance(text, str) else text
        return FakeJSON({"id": "c1", "object": "chat.completion", "created": 1, "model": "m",
                         "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                                   "prompt_tokens_details": {"cached_tokens": 4}}}, status)
    ns = {"structured": S, "json": json, "time": time, "asyncio": asyncio, "JSONResponse": FakeJSON,
          "StreamingResponse": FakeStream, "complete": complete}
    exec(compile(ast.Module(body=[_node], type_ignores=[]), _SRC, "exec"), ns)
    return ns["structured_chat"], sent


def run(replies, **body):
    fn, sent = harness(replies)
    body = {"messages": [{"role": "user", "content": "Who?"}], **body}
    return asyncio.run(fn(body)), sent


RF = {"type": "json_schema", "json_schema": {"name": "person", "schema": PERSON, "strict": True}}


class StructuredChat(unittest.TestCase):
    def test_first_try(self):
        r, sent = run(['```json\n{"name": "Ann", "age": 31}\n```'], response_format=RF)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content["choices"][0]["message"]["content"], '{"name": "Ann", "age": 31}')
        self.assertEqual(r.content["inferd"]["json_attempts"], 1)
        self.assertEqual(len(sent), 1)
        self.assertNotIn("response_format", sent[0])
        self.assertIn("JSON Schema", sent[0]["messages"][-1]["content"])

    def test_retry_with_error_fed_back(self):
        r, sent = run(['{"name": "Ann"}', '{"name": "Ann", "age": 31}'], response_format=RF)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content["inferd"]["json_attempts"], 2)
        self.assertEqual(r.content["usage"]["prompt_tokens"], 20)  # usage sums every attempt
        self.assertEqual(r.content["usage"]["prompt_tokens_details"]["cached_tokens"], 8)
        second = sent[1]["messages"]
        self.assertEqual(second[-2], {"role": "assistant", "content": '{"name": "Ann"}'})
        self.assertIn("missing required property 'age'", second[-1]["content"])

    def test_gives_up_with_422(self):
        r, sent = run(["no", "still no", "nope"], response_format=RF)
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.content["error"]["code"], "json_validation_failed")
        self.assertEqual(r.content["error"]["attempts"], 3)
        self.assertEqual(r.content["error"]["last_output"], "nope")
        self.assertEqual(len(sent), 3)

    def test_retries_knob(self):
        r, sent = run(["no"], response_format=RF, json_retries=0)
        self.assertEqual((r.status_code, len(sent)), (422, 1))
        r, _ = run([], response_format=RF, inferd={"json_retries": 99})
        self.assertEqual(r.status_code, 400)

    def test_bad_format_400(self):
        r, sent = run([], response_format={"type": "yaml"})
        self.assertEqual((r.status_code, sent), (400, []))

    def test_text_passthrough(self):
        r, sent = run(["hello"], response_format={"type": "text"})
        self.assertEqual(r.content["choices"][0]["message"]["content"], "hello")
        self.assertNotIn("response_format", sent[0])
        self.assertEqual(sent[0]["messages"][-1]["content"], "Who?")

    def test_reasoning_kept_content_cleaned(self):
        r, _ = run([{"role": "assistant", "content": 'Answer: {"ok": true}', "reasoning_content": "think"}],
                   response_format={"type": "json_object"})
        m = r.content["choices"][0]["message"]
        self.assertEqual((m["content"], m["reasoning_content"]), ('{"ok": true}', "think"))

    def test_tool_call_passes_through(self):
        call = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                                                                      "function": {"name": "f", "arguments": "{}"}}]}
        r, sent = run([call], response_format=RF)
        self.assertEqual(r.content["choices"][0]["message"]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(len(sent), 1)

    def test_constrained_routing(self):  # strict -> the schema goes to the engine as a grammar ("_grammar")
        r, sent = run(['{"name": "A", "age": 1}'], response_format=RF)
        self.assertEqual(sent[0]["_grammar"], PERSON)
        self.assertTrue(r.content["inferd"]["json_constrained"])
        loose = {"type": "json_schema", "json_schema": {"name": "p", "schema": PERSON}}
        r, sent = run(['{"name": "A", "age": 1}'], response_format=loose)
        self.assertNotIn("_grammar", sent[0])
        self.assertFalse(r.content["inferd"]["json_constrained"])
        self.assertNotIn("json_checked_only", r.content["inferd"])
        oneof = {"type": "json_schema", "json_schema": {"name": "s", "strict": True, "schema": {
            "oneOf": [{"type": "integer", "minimum": 1}, {"type": "string"}]}}}
        r, _ = run(["3"], response_format=oneof)
        self.assertEqual(r.content["inferd"]["json_checked_only"], ["minimum", "oneOf"])
        r, sent = run(['{"ok": 1}'], response_format={"type": "json_object"}, inferd={"constrained": True})
        self.assertEqual(sent[0]["_grammar"], {"type": "object"})
        r, sent = run(['{"name": "A", "age": 1}'], response_format=RF, constrained=False)
        self.assertNotIn("_grammar", sent[0])
        r, sent = run(["hi"], response_format={"type": "text"}, _grammar={"type": "object"})
        self.assertNotIn("_grammar", sent[0])  # a client cannot smuggle a grammar into a plain request

    def test_stream_buffers_then_sends(self):
        r, sent = run(['oops', '{"name": "B", "age": 2}'], response_format=RF, stream=True,
                      stream_options={"include_usage": True})
        self.assertNotIn("stream", sent[0])  # attempts run unstreamed

        async def collect():
            return [x async for x in r.gen]
        lines = asyncio.run(collect())
        evs = [json.loads(x[6:]) for x in lines if x.startswith("data: {")]
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in evs if e["choices"])
        self.assertEqual(json.loads(text), {"name": "B", "age": 2})
        self.assertEqual([e["choices"][0]["finish_reason"] for e in evs if e["choices"]][-1], "stop")
        self.assertEqual(evs[-1]["usage"]["completion_tokens"], 10)
        self.assertEqual(lines[-1], "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
