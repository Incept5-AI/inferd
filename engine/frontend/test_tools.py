# Tool-call parsing and stop strings in serve.py, on canned model text (stdlib unittest; no engine, no model needed).
#   python3 frontend/test_tools.py
# serve.py parses argv and loads a tokenizer at import, so the tool helpers are lifted out of its source with ast.
import ast, json, os, re, unittest, uuid

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve.py")
_WANT = {"TOOL_OPEN", "_TOOL_RE", "_PARAM_RE", "_param_types", "parse_tool_calls", "_template_messages",
         "parse_stop", "_first_stop", "cut_at_stop", "StopMatcher"}
_tree = ast.parse(open(_SRC).read())
_nodes = [n for n in _tree.body if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _WANT)
          or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in _WANT for t in n.targets))]
ns = {"re": re, "json": json, "uuid": uuid}
exec(compile(ast.Module(body=_nodes, type_ignores=[]), _SRC, "exec"), ns)
parse, template_messages = ns["parse_tool_calls"], ns["_template_messages"]
parse_stop, cut_at_stop, StopMatcher = ns["parse_stop"], ns["cut_at_stop"], ns["StopMatcher"]

TOOLS = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object", "properties": {
    "query": {"type": "string"}, "limit": {"type": "integer"}}}}},
         {"type": "function", "function": {"name": "shell", "parameters": {"type": "object", "properties": {
             "command": {"type": "string"}}}}}]


class ToolCalls(unittest.TestCase):
    def test_plain_text_untouched(self):
        self.assertEqual(parse("Paris is the capital of France.", TOOLS), ("Paris is the capital of France.", []))

    def test_single_call(self):
        text = "<tool_call>\n<function=web_search>\n<parameter=query>\nqwen 3.8 release\n</parameter>\n" \
               "<parameter=limit>\n3\n</parameter>\n</function>\n</tool_call>"
        content, calls = parse(text, TOOLS)
        self.assertIsNone(content)
        self.assertEqual(len(calls), 1)
        c = calls[0]
        self.assertEqual(c["type"], "function")
        self.assertTrue(c["id"].startswith("call_"))
        self.assertEqual(c["function"]["name"], "web_search")
        self.assertIsInstance(c["function"]["arguments"], str)  # OpenAI: a JSON string
        self.assertEqual(json.loads(c["function"]["arguments"]), {"query": "qwen 3.8 release", "limit": 3})

    def test_string_param_stays_string(self):  # "42" is a string because the schema says so
        text = "<tool_call>\n<function=web_search>\n<parameter=query>\n42\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(json.loads(parse(text, TOOLS)[1][0]["function"]["arguments"]), {"query": "42"})

    def test_content_before_and_two_calls(self):
        text = "Let me look.\n\n<tool_call>\n<function=shell>\n<parameter=command>\nls -la\n</parameter>\n" \
               "</function>\n</tool_call>\n<tool_call>\n<function=shell>\n<parameter=command>\ndf -h\n" \
               "</parameter>\n</function>\n</tool_call>"
        content, calls = parse(text, TOOLS)
        self.assertEqual(content, "Let me look.")
        self.assertEqual([json.loads(c["function"]["arguments"])["command"] for c in calls], ["ls -la", "df -h"])
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])

    def test_multiline_value_and_unknown_tool(self):
        text = "<tool_call>\n<function=write>\n<parameter=body>\nline 1\nline 2\n</parameter>\n" \
               "<parameter=opts>\n{\"a\": [1, 2]}\n</parameter>\n</function>\n</tool_call>"
        args = json.loads(parse(text, TOOLS)[1][0]["function"]["arguments"])
        self.assertEqual(args, {"body": "line 1\nline 2", "opts": {"a": [1, 2]}})

    def test_no_closing_tool_call_tag(self):  # generation stopped right after </function>
        text = "<tool_call>\n<function=shell>\n<parameter=command>\npwd\n</parameter>\n</function>"
        self.assertEqual(parse(text, TOOLS)[1][0]["function"]["name"], "shell")

    def test_malformed_left_as_text(self):
        text = "I would call <tool_call> but"
        self.assertEqual(parse(text, TOOLS), (text, []))

    def test_history_arguments_decoded_for_template(self):
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "shell", "arguments": "{\"command\": \"pwd\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "/tmp"}]
        out = template_messages(msgs)
        self.assertEqual(out[1]["tool_calls"][0]["function"]["arguments"], {"command": "pwd"})
        self.assertEqual(out[1]["content"], "")
        self.assertEqual(msgs[1]["tool_calls"][0]["function"]["arguments"], "{\"command\": \"pwd\"}")  # not mutated
        self.assertIs(out[2], msgs[2])


def stream(pieces, stops, keep=False):
    """Feed pieces through a StopMatcher as the server does: -> (emitted text, stopped, pieces consumed)."""
    sm, out = StopMatcher(stops, keep), []
    for n, p in enumerate(pieces, 1):
        t, hit = sm.feed(p)
        out.append(t)
        if hit:
            return "".join(out), True, n
    out.append(sm.flush())
    return "".join(out), False, len(pieces)


class StopStrings(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_stop(None), [])
        self.assertEqual(parse_stop("END"), ["END"])
        self.assertEqual(parse_stop(["a", "", "bc"]), ["a", "bc"])  # empty strings are ignored
        for bad in (3, ["a", 1], {"s": "a"}):
            with self.assertRaises(ValueError):
                parse_stop(bad)

    def test_cut(self):
        self.assertEqual(cut_at_stop("one, two, three", [", t"]), ("one", True))
        self.assertEqual(cut_at_stop("one, two", ["two", "one"]), ("", True))  # earliest wins
        self.assertEqual(cut_at_stop("abc", ["x"]), ("abc", False))
        self.assertEqual(cut_at_stop("a\n\nb", ["\n", "\n\n"], keep=True), ("a\n\n", True))  # same start: longest

    def test_stop_inside_one_piece(self):
        self.assertEqual(stream(["1, 2, 3, 4, 5"], ["4"]), ("1, 2, 3, ", True, 1))

    def test_stop_straddles_pieces(self):  # "###" arrives as "#", "##": nothing of it may be emitted
        text, hit, n = stream(["Answer: 42", "\n#", "## Next section"], ["###"])
        self.assertEqual((text, hit, n), ("Answer: 42\n", True, 3))

    def test_held_prefix_released(self):  # a partial match that does not complete goes out unchanged
        self.assertEqual(stream(["x <", "/b", "> y"], ["</a>"]), ("x </b> y", False, 3))
        sm = StopMatcher(["</a>"])
        self.assertEqual(sm.feed("x </"), ("x ", False))  # "</" is held back
        self.assertEqual(sm.flush(), "</")  # end of generation: the held text is released

    def test_keep_and_multi(self):
        self.assertEqual(stream(["a b ", "STOP c"], ["STOP", "END"], keep=True), ("a b STOP", True, 2))
        self.assertEqual(stream(["x EN", "D y STOP"], ["STOP", "END"]), ("x ", True, 2))

    def test_after_stop_nothing(self):
        sm = StopMatcher(["."])
        self.assertEqual(sm.feed("Hi. More"), ("Hi", True))
        self.assertEqual(sm.feed("text"), ("", True))

    def test_unicode(self):
        self.assertEqual(stream(["日本", "語。次", "の文"], ["。"]), ("日本語", True, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
