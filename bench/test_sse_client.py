#!/usr/bin/env python3
"""Offline tests for sse_client.py and wall.py's success predicate (no network): python3 bench/test_sse_client.py
Includes the Codex review 2026-11-02 S04 reproduction: a partial stream + usage + SSE error + EOF, 100 tokens asked."""
import io, json, os, sys, types, unittest, urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sse_client, wall


def fake(events, status=200):
    lines = [(("data: " + (e if isinstance(e, str) else json.dumps(e))) + "\n\n").encode() for e in events]

    class Resp:
        def __init__(self):
            self.status = status

        def __iter__(self):
            for ln in lines:
                for part in ln.split(b"\n")[:-1]:
                    yield part + b"\n"

    return lambda req, timeout=None: Resp()


def delta(s, finish=None):
    return {"choices": [{"index": 0, "delta": {"content": s} if s else {}, "finish_reason": finish}]}


def usage(pt, ct, **kw):
    return {"choices": [], "usage": dict(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct, **kw)}


GOOD = [delta("hel"), delta("lo"), delta("", "length"), usage(10, 5), "[DONE]"]


class SSE(unittest.TestCase):
    def run_(self, events, **kw):
        return sse_client.stream_chat("http://unused", {"model": "q"}, opener=fake(events), **kw)

    def test_good(self):
        r = self.run_(GOOD, expect_completion=5)
        self.assertIsNone(r["error"])
        self.assertEqual((r["text"], r["completion_tokens"], r["finish_reason"], r["done"]), ("hello", 5, "length", True))

    def test_review_case_partial_error_eof(self):  # the S04 reproduction
        r = self.run_([delta("partial"), {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                       {"error": {"message": "engine failed"}}], expect_completion=100)
        self.assertIn("engine failed", r["error"])
        self.assertTrue(r["partial"])

    def test_error_then_done(self):  # inferd's own error shape: error event, then [DONE]
        r = self.run_([delta("x"), {"error": {"message": "error: oom"}}, "[DONE]"])
        self.assertIn("oom", r["error"])

    def test_eof_without_done(self):
        self.assertIn("[DONE]", self.run_(GOOD[:-1])["error"])

    def test_no_finish(self):
        self.assertIn("finish_reason", self.run_([delta("hello"), usage(10, 5), "[DONE]"])["error"])

    def test_bad_finish(self):
        self.assertIn("finish_reason", self.run_([delta("a", "abort"), usage(10, 1), "[DONE]"])["error"])

    def test_no_usage(self):
        self.assertIn("usage missing", self.run_([delta("a", "stop"), "[DONE]"])["error"])

    def test_inconsistent_usage(self):
        bad = {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 99}}
        self.assertIn("total_tokens", self.run_([delta("a", "stop"), bad, "[DONE]"])["error"])
        cached = usage(10, 5, prompt_tokens_details={"cached_tokens": 11})
        self.assertIn("cached", self.run_([delta("a", "stop"), cached, "[DONE]"])["error"])
        self.assertIn("completion_tokens is 0", self.run_([delta("a", "stop"), usage(10, 0), "[DONE]"])["error"])

    def test_short_fixed_length(self):
        r = self.run_(GOOD, expect_completion=100)
        self.assertIn("!= requested 100", r["error"])
        r = self.run_([delta("a", "stop"), usage(10, 5), "[DONE]"], expect_completion=5)
        self.assertIn("expected 'length'", r["error"])

    def test_data_after_done(self):
        self.assertIn("after [DONE]", self.run_(GOOD + [delta("late")])["error"])

    def test_non_json(self):
        self.assertIn("non-JSON", self.run_(["{oops"] + GOOD)["error"])

    def test_http_error(self):
        def op(req, timeout=None):
            raise urllib.error.HTTPError("u", 503, "x", {}, io.BytesIO(b'{"error":"engine disconnected"}'))
        r = sse_client.stream_chat("http://unused", {}, opener=op)
        self.assertIn("HTTP 503", r["error"])
        self.assertFalse(r["partial"])

    def test_raw_events_kept(self):
        r = self.run_(GOOD, keep_events=True)
        self.assertEqual(r["events"][-1], "[DONE]")


class Wall(unittest.TestCase):
    """wall.user + wall.failed, the level's success predicate, on the review's fake stream."""

    def user(self, events, gen=100):
        args = types.SimpleNamespace(prompt=100, model="q", gen=gen, temperature=0, extra=None, url="http://unused",
                                     timeout=1, raw=True)
        orig = sse_client.urllib.request.urlopen
        sse_client.urllib.request.urlopen = fake(events)
        try:
            rec = {}
            wall.user(args, 0.0, rec)
        finally:
            sse_client.urllib.request.urlopen = orig
        return rec

    def test_review_case_rejected(self):
        rec = self.user([delta("partial"), {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                         {"error": {"message": "engine failed"}}])
        self.assertTrue(wall.failed(rec))
        self.assertTrue(rec["partial"])

    def test_good_accepted(self):
        self.assertFalse(wall.failed(self.user(GOOD, gen=5)))

    def test_short_rejected(self):
        self.assertTrue(wall.failed(self.user(GOOD, gen=100)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
