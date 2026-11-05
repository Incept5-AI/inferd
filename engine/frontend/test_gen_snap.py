# The decode-snapshot probe in serve.py (_answers_rerendered): decode snapshots are only resume points when the chat
# template renders an earlier answer exactly as generated. Qwen3.8 keeps the empty think block (probe true); Qwen3.6
# drops it from earlier turns (probe false: its follow-ups resume at the previous prompt's end). stdlib unittest;
# checkpoint cases need TOKENIZERS=<dir>=<expected 0/1>[:...] (e.g. a Qwen3.6 dir=0, a Qwen3.8 dir=1), else skipped.
#   TOKENIZERS=<qwen3.6 dir>=0:<qwen3.8 dir>=1 python3 frontend/test_gen_snap.py
import ast, os, sys, unittest

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve.py")
_fn = next(n for n in ast.parse(open(_SRC).read()).body if isinstance(n, ast.FunctionDef) and n.name == "_answers_rerendered")


def probe(tok, env=None):
    ns = {"os": os, "sys": sys, "tok": tok}
    exec(compile(ast.Module(body=[_fn], type_ignores=[]), _SRC, "exec"), ns)
    old = os.environ.get("INFERD_GEN_SNAP")
    try:
        if env is None:
            os.environ.pop("INFERD_GEN_SNAP", None)
        else:
            os.environ["INFERD_GEN_SNAP"] = env
        return ns["_answers_rerendered"]()
    finally:
        if old is None:
            os.environ.pop("INFERD_GEN_SNAP", None)
        else:
            os.environ["INFERD_GEN_SNAP"] = old


class Broken:
    def apply_chat_template(self, *a, **k):
        raise ValueError("no template")


class Probe(unittest.TestCase):
    def test_override_and_fallback(self):
        self.assertTrue(probe(Broken(), "1"))
        self.assertFalse(probe(Broken(), "0"))
        self.assertTrue(probe(Broken()))  # a template that fails keeps the old behaviour
        with self.assertRaises(SystemExit):
            probe(Broken(), "yes")

    def test_checkpoints(self):
        dirs = [d for d in os.environ.get("TOKENIZERS", "").split(":") if d]
        if not dirs:
            self.skipTest("TOKENIZERS not set")
        from transformers import AutoTokenizer
        for e in dirs:
            d, want = e.rsplit("=", 1)
            with self.subTest(d=d):
                self.assertEqual(probe(AutoTokenizer.from_pretrained(d)), want == "1")


if __name__ == "__main__":
    unittest.main()
