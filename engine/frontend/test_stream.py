# Streaming text plumbing in serve.py (Codex review 2026-11-02, S08): the incremental detokenizer and the terminal
# flush of the reasoning split. stdlib unittest; the Detok tests need a Qwen tokenizer (TOKENIZER=<dir>, default: the
# first Qwen3.5/3.8 snapshot in the Hugging Face cache) and are skipped without one.
#   python3 frontend/test_stream.py
import ast, glob, os, random, time, unittest

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve.py")
_tree = ast.parse(open(_SRC).read())
_top = {n.name: n for n in _tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
_split = next(n for n in _top["complete"].body if isinstance(n, ast.FunctionDef) and n.name == "split")
ns = {"THINK_END": "</think>"}
exec(compile(ast.Module(body=[_split, _top["drain_split"], _top["Detok"]], type_ignores=[]), _SRC, "exec"), ns)
split, drain_split, Detok = ns["split"], ns["drain_split"], ns["Detok"]


def _tokenizer():
    d = os.environ.get("TOKENIZER") or next(iter(sorted(
        os.path.dirname(p) for p in glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--*Qwen3.[58]*/snapshots/*/tokenizer.json")))), None)
    if not d:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(d)
    except Exception:
        return None


TOK = _tokenizer()


class Split(unittest.TestCase):
    def test_review_case_flushed(self):  # 'abcdefghij' as the final reasoning chunk: all of it reaches the client
        st = {"reasoning": True, "buf": ""}
        r1, _ = split(st, "abcdefghij")
        r2, c2 = drain_split(split, st, "")
        self.assertEqual(r1 + (r2 or ""), "abcdefghij")
        self.assertEqual((st["buf"], c2), ("", None))

    def test_partial_think_end_at_terminal(self):
        st = {"reasoning": True, "buf": ""}
        r1, _ = split(st, "thought </thi")
        r2, _ = drain_split(split, st, "")
        self.assertEqual(r1 + r2, "thought </thi")

    def test_after_reasoning_nothing_held(self):
        st = {"reasoning": True, "buf": ""}
        r, c = split(st, "x</think>\n\nanswer")
        self.assertEqual((r, c), ("x", "answer"))
        self.assertEqual(drain_split(split, st, "tail"), (None, "tail"))

    def test_streamed_equals_whole(self):
        rnd = random.Random(1)
        for _ in range(200):
            text = "".join(rnd.choice("ab</think>\n") for _ in range(rnd.randint(0, 40)))
            st, rs, cs, i = {"reasoning": True, "buf": ""}, [], [], 0
            while i < len(text):
                j = i + rnd.randint(1, 5)
                r, c = split(st, text[i:j])
                rs.append(r or ""), cs.append(c or "")
                i = j
            r, c = drain_split(split, st, "")
            rs.append(r or ""), cs.append(c or "")
            if "</think>" in text:
                want_r, want_c = text.split("</think>", 1)
                want_c = want_c.lstrip("\n")
            else:
                want_r, want_c = text, ""
            self.assertEqual("".join(rs), want_r, text)
            # content after </think>: the stream strips leading newlines of the first content piece only
            self.assertEqual("".join(cs).lstrip("\n"), want_c, text)


@unittest.skipIf(TOK is None, "no Qwen tokenizer found (set TOKENIZER=<dir>)")
class Detokenizer(unittest.TestCase):
    def setUp(self):
        ns["tok"] = TOK

    def stream(self, ids):
        d = Detok()
        return "".join(d.add(t) for t in ids) + d.flush()

    def test_matches_full_decode(self):
        rnd = random.Random(7)
        texts = ["Hello, world! Ünïcödé — 日本語のテキスト 🚀🎉 emoji\n\tcode: `x = [1, 2]`",
                 "<think>\nreasoning…</think>\n\n¿Qué? Ελληνικά, עברית, हिन्दी, 한국어 " * 3]
        for t in texts:
            ids = TOK.encode(t, add_special_tokens=False)
            self.assertEqual(self.stream(ids), TOK.decode(ids, skip_special_tokens=False))
        vocab = len(TOK)
        for _ in range(100):  # random ids, including byte tokens that split UTF-8 sequences
            ids = [rnd.randrange(vocab) for _ in range(rnd.randint(1, 60))]
            self.assertEqual(self.stream(ids), TOK.decode(ids, skip_special_tokens=False))

    def test_holds_partial_utf8_until_complete(self):
        ids = TOK.encode("🚀", add_special_tokens=False)
        if len(ids) < 2:
            self.skipTest("emoji is one token in this vocabulary")
        d = Detok()
        self.assertEqual(d.add(ids[0]), "")
        self.assertEqual("".join(d.add(t) for t in ids[1:]), "🚀")

    def test_linear_time(self):
        ids = TOK.encode("The quick brown fox jumps over the lazy dog. " * 400, add_special_tokens=False)

        def cost(n):
            d, t0 = Detok(), time.perf_counter()
            for t in ids[:n]:
                d.add(t)
            return time.perf_counter() - t0

        small, big = cost(len(ids) // 4), cost(len(ids))
        self.assertLess(big / small, 8, f"{small:.3f}s for n/4, {big:.3f}s for n: not linear")  # quadratic: ~16


if __name__ == "__main__":
    unittest.main(verbosity=2)
