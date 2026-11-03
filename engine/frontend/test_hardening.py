# Front-end hardening (Codex review 2026-11-03: R02 budgets, R03 validation, R04 context lifecycle, R07 submit
# cancellation, R08 auth and socket checks, R12 host-side retention). Imports the real serve.py (a stub tokenizer and a
# temporary model directory stand in for the checkpoint) and drives it through its ASGI stack (Guard + FastAPI) with
# httpx, against a fake inferd that speaks the JSON-lines protocol into the real Engine.pump. No engine, no model.
#   python3 frontend/test_hardening.py      (needs fastapi, httpx, jinja2: the front end's environment)
import asyncio, json, os, random, socket, subprocess, sys, tempfile, types, unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class StubTok:
    """Characters are tokens (id = code point); a ChatML-like template. Enough for render / render_context."""

    def apply_chat_template(self, msgs, tools=None, add_generation_prompt=False, tokenize=False, **kw):
        out = "".join(f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n" for m in msgs)
        if kw.get("reasoning_effort"):
            out = f"<|im_start|>system\neffort {kw['reasoning_effort']}<|im_end|>\n" + out
        if add_generation_prompt:
            out += "<|im_start|>assistant\n" + ("" if kw.get("enable_thinking", True) else "<think>\n\n</think>\n\n")
        return out

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def convert_tokens_to_ids(self, t):
        return 7

    def get_vocab(self):
        return {"<|im_end|>": 2}


def _import_serve():
    model = tempfile.mkdtemp(prefix="inferd-test-model-")
    json.dump({"eos_token_id": [2]}, open(os.path.join(model, "config.json"), "w"))
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path: StubTok())
    sys.modules["transformers"] = tf
    sys.modules["vision"] = types.SimpleNamespace(has_images=lambda msgs: False)  # no MLX here
    argv, sys.argv = sys.argv, ["serve.py", "--model", model, "--socket", "/nonexistent.sock"]
    os.environ.pop("INFERD_API_KEY", None)
    sys.path.insert(0, HERE)
    try:
        import serve
    finally:
        sys.argv = argv
    return serve


S = _import_serve()
import httpx  # noqa: E402


class FakeInferd:
    """The engine end of the socket: parses what Engine writes, answers through a real StreamReader into Engine.pump."""

    def __init__(self, eng):
        self.eng, self.reader = eng, asyncio.StreamReader()
        self.saved, self.cancelled, self.gens, self.ops = set(), [], [], []
        self.save_gate = None   # asyncio.Event: context saves wait for it
        self.gen_gate = None    # asyncio.Event: generations wait for it before emitting
        self.drain_gate = None  # asyncio.Event: Writer.drain blocks on it
        self.tokens = [ord(c) for c in "hello"]
        self.closing = False

    # StreamWriter surface
    def write(self, data):
        for line in data.decode().splitlines():
            m = json.loads(line)
            self.ops.append(m)
            asyncio.get_running_loop().create_task(self.handle(m))

    async def drain(self):
        if self.drain_gate is not None:
            await self.drain_gate.wait()

    def is_closing(self):
        return self.closing

    def send(self, ev):
        self.reader.feed_data((json.dumps(ev) + "\n").encode())

    async def handle(self, m):
        op, rid = m["op"], m["id"]
        if op == "cancel":
            self.cancelled.append(rid)
        elif op == "gen" and "ctx" in m:
            if self.save_gate is not None:
                await self.save_gate.wait()
            if rid in self.cancelled:
                return self.send({"id": rid, "done": "cancelled", "p": 0, "c": 0, "cached": 0})
            self.saved.add(m["ctx"])
            self.send({"id": rid, "done": "stop", "p": len(m["tokens"]), "c": 0, "cached": 0,
                       "ctx": {"name": m["ctx"], "tokens": len(m["tokens"]), "cached": 0}})
        elif op == "gen":
            self.gens.append(m)
            if self.gen_gate is not None:
                await self.gen_gate.wait()
            for t in self.tokens[:m.get("max_tokens", 4096)]:
                self.send({"id": rid, "t": [t]})
            self.send({"id": rid, "done": "stop", "p": len(m.get("tokens", [])), "c": len(self.tokens), "cached": 0})
        elif op in ("ctx_drop", "ctx_offload"):
            ok = m["ctx"] in self.saved
            if op == "ctx_drop":
                self.saved.discard(m["ctx"])
            self.send({"id": rid, "done": "ok" if ok else "error: no such context", "stats": {}})
        elif op in ("session_drop", "session_offload", "stats"):
            self.send({"id": rid, "done": "ok", "stats": {"contexts": [{"name": n} for n in sorted(self.saved)]}})


def fresh():
    """A new Engine wired to a new FakeInferd; front-end state reset."""
    eng = S.Engine()
    fake = FakeInferd(eng)
    eng.writer = fake
    eng.alive = True
    S.engine = eng
    S.contexts.clear()
    S._ctx_locks.clear()
    S._sessions.clear()
    S._prep_sem = None
    eng.background(eng.pump(fake.reader))
    return eng, fake


def client(app=None):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app or S.Guard(S.app)), base_url="http://t")


async def settle(n=20):
    for _ in range(n):
        await asyncio.sleep(0)
    await asyncio.sleep(0.01)


def run(coro):
    return asyncio.run(coro)


CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


# ------------------------------------------------------------------------------------------------ R07 submit cancel
class SubmitCancellation(unittest.TestCase):
    def test_review_repro_cancel_during_drain(self):  # frontend_repros.py: request 7 stayed registered
        async def main():
            eng, fake = fresh()
            fake.drain_gate = asyncio.Event()
            t = asyncio.create_task(eng.submit(7, {}))
            await settle()
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t
            self.assertNotIn(7, eng.queues)
            fake.drain_gate.set()  # the best-effort cancel was waiting for the lock / drain
            await settle()
            self.assertIn(7, fake.cancelled)  # it may have been sent: cancelled natively
        run(main())

    def test_cancel_while_awaiting_lock(self):
        async def main():
            eng, fake = fresh()
            await eng.lock.acquire()
            t = asyncio.create_task(eng.submit(8, {}))
            await settle()
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t
            eng.lock.release()
            await settle()
            self.assertNotIn(8, eng.queues)
            self.assertEqual([m for m in fake.ops if m["id"] == 8], [])  # never written, nothing to cancel
        run(main())

    def test_cancel_during_control_op(self):
        async def main():
            eng, fake = fresh()
            fake.save_gate = asyncio.Event()  # a context save that does not answer
            t = asyncio.create_task(eng.call("gen", ctx="x@none#1", tokens=[1], max_tokens=0))
            s = asyncio.create_task(eng.call("stats"))
            await settle()
            self.assertEqual(len(eng.queues), 1)  # stats answered, save pending
            self.assertTrue((await s)["done"] == "ok")
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t
            await settle()
            self.assertEqual(eng.queues, {})
            save_id = next(m["id"] for m in fake.ops if m.get("ctx") == "x@none#1")
            self.assertIn(save_id, fake.cancelled)
        run(main())

    def test_engine_error_unregisters(self):
        async def main():
            eng, fake = fresh()

            def boom(_):
                raise ConnectionResetError()
            fake.write = boom
            with self.assertRaises(ConnectionResetError):
                await eng.submit(9, {})
            self.assertNotIn(9, eng.queues)
        run(main())


# ------------------------------------------------------------------------------------------------ R04 lifecycle
def handles_of_live():
    return {v["engine"] for r in S.contexts.values() for v in r["variants"].values()}


class ContextLifecycle(unittest.TestCase):
    def test_review_repro_save_then_delete(self):
        # frontend_repros.py adapted: a variant save starts, the context is deleted, then the save completes. Before:
        # the record was gone while "demo@low" stayed saved (pinned) in the engine.
        async def main():
            eng, fake = fresh()
            async with client() as c:
                r = await c.post("/v1/contexts", json={"name": "demo", "system": "you are a demo " * 8})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(fake.saved, handles_of_live())
                fake.save_gate = asyncio.Event()
                launch = asyncio.create_task(c.post("/v1/chat/completions", json={
                    **CHAT, "inferd": {"context": "demo", "effort": "low"}}))
                await settle()
                d = asyncio.create_task(c.delete("/v1/contexts/demo"))
                await settle()
                self.assertNotIn("demo", S.contexts)  # unpublished at once
                fake.save_gate.set()
                self.assertEqual((await d).status_code, 200)
                r = await launch
                self.assertEqual(r.status_code, 404, r.text)  # the launch lost to the delete, cleanly
                await settle()
            self.assertEqual(fake.saved, set())  # no engine handle left behind
            self.assertEqual(S._ctx_locks, {})
        run(main())

    def test_extracted_ensure_variant_then_delete(self):  # the repro's own call sequence, against the new functions
        async def main():
            eng, fake = fresh()
            fake.save_gate = asyncio.Event()
            rec = dict(id="demo", gen=1, dead=False, messages=[{"role": "system", "content": "x" * 80}], tools=None,
                       persist="memory", effort="none", ttl=0, created=0, last_used=0, launches=0,
                       variants={"none": dict(engine="demo@none#1", tokens=[1], saved={})})
            fake.saved.add("demo@none#1")
            S.contexts["demo"] = rec
            t = asyncio.create_task(S.ensure_variant(rec, "low"))
            await settle()
            d = asyncio.create_task(S.delete_context("demo"))
            await settle()
            fake.save_gate.set()
            await d
            with self.assertRaises(S.Stale):
                await t
            await settle()
            self.assertEqual((fake.saved, S.contexts, S._ctx_locks), (set(), {}, {}))
        run(main())

    def test_replace_uses_new_handles(self):
        async def main():
            eng, fake = fresh()
            async with client() as c:
                for text in ("first context " * 8, "second context " * 8):
                    r = await c.post("/v1/contexts", json={"name": "doc", "system": text})
                    self.assertEqual(r.status_code, 200, r.text)
                await settle()
            self.assertEqual(len(fake.saved), 1)
            (name,) = fake.saved
            self.assertTrue(name.startswith("doc@none#"))
            self.assertEqual(fake.saved, handles_of_live())
        run(main())

    def test_cancelled_create_leaves_nothing(self):
        async def main():
            eng, fake = fresh()
            fake.save_gate = asyncio.Event()
            req = types.SimpleNamespace(body=None)

            async def body():
                return json.dumps({"name": "gone", "system": "y" * 100}).encode()
            req.body = body
            t = asyncio.create_task(S.create_context(req))
            await settle()
            t.cancel()  # the client went away mid-save
            with self.assertRaises(asyncio.CancelledError):
                await t
            fake.save_gate.set()
            await settle()
            self.assertEqual((fake.saved, S.contexts, S._ctx_locks), (set(), {}, {}))
        run(main())

    def test_random_races(self):
        # create / replace / delete / offload / expiry / lazy effort launches on a few names, interleaved at random
        # with saves of random latency. Invariant after the dust settles: the engine holds exactly the handles of the
        # live records (no orphans, nothing missing), and no lifecycle lock is left behind.
        async def main(seed):
            rnd = random.Random(seed)
            eng, fake = fresh()
            orig = fake.handle

            async def slow(m):
                await asyncio.sleep(rnd.random() * 0.006)
                await orig(m)
            fake.handle = slow
            async with client() as c:
                async def op():
                    await asyncio.sleep(rnd.random() * 0.03)  # spread out, so operations overlap in flight
                    name = rnd.choice(["a", "b", "c"])
                    k = rnd.random()
                    if k < 0.3:
                        await c.post("/v1/contexts", json={"name": name, "system": f"ctx {name} {rnd.random()} " * 6,
                                                           "ttl": rnd.choice([0, 0.001])})
                    elif k < 0.5:
                        await c.delete(f"/v1/contexts/{name}")
                    elif k < 0.6:
                        await c.post(f"/v1/contexts/{name}/offload")
                    elif k < 0.7:
                        await S._expire_once(now=10 ** 12)
                    else:
                        await c.post("/v1/chat/completions", json={**CHAT, "inferd": {
                            "context": name, "effort": rnd.choice(S.EFFORTS)}})
                await asyncio.gather(*(op() for _ in range(60)))
                await settle(50)
            self.assertEqual(fake.saved, handles_of_live(), f"seed {seed}")
            self.assertEqual(S._ctx_locks, {}, f"seed {seed}")
        for seed in range(25):
            run(main(seed))


# ------------------------------------------------------------------------------------------------ R03 validation
class Validation(unittest.TestCase):
    def post(self, path, payload, raw=None):
        async def main():
            fresh()
            async with client() as c:
                if raw is not None:
                    return await c.post(path, content=raw, headers={"content-type": "application/json"})
                return await c.post(path, json=payload)
        return run(main())

    def bad(self, payload, path="/v1/chat/completions", raw=None):
        r = self.post(path, payload, raw)
        self.assertEqual(r.status_code, 400, (payload, raw, r.text))
        self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
        return r.json()["error"]

    def test_chat_shapes(self):
        self.bad(None, raw=b"[1, 2]")
        self.bad(None, raw=b"{not json")
        self.bad(None, raw=b'{"messages": [{"role": "user", "content": "x"}], "temperature": NaN}')
        self.bad(None, raw=b'{"messages": [{"role": "user", "content": "x"}], "temperature": 1e999}')
        for extra in ({"temperature": "hot"}, {"temperature": -1}, {"temperature": True}, {"top_p": 0},
                      {"top_p": 1.5}, {"top_k": "5"}, {"top_k": -2}, {"max_tokens": 0}, {"max_tokens": -5},
                      {"max_tokens": 2.5}, {"max_completion_tokens": 0}, {"seed": "x"}, {"seed": 1 << 70},
                      {"stream": "yes"}, {"stream_options": []}, {"stop": 5}, {"stop": ["x"] * 100},
                      {"tools": {"a": 1}}, {"chat_template_kwargs": "x"}, {"enable_thinking": "no"},
                      {"inferd": []}, {"inferd": {"session": ""}}, {"inferd": {"session": "s" * 129}},
                      {"inferd": {"session": 5}}, {"inferd": {"context": "bad name!"}}, {"inferd": {"effort": "max"}},
                      {"inferd": {"cache": "never"}}, {"inferd": {"max_context": 0}}, {"inferd": {"priority": "hi"}},
                      {"session": "x" * 200}, {"response_format": "json"}):
            self.bad({**CHAT, **extra})
        for msgs in (None, [], "hi", [1], [{"content": "x"}], [{"role": "user", "content": 5}],
                     [{"role": "user", "content": ["x"]}], [{"role": "assistant", "tool_calls": {}}],
                     [{"role": "assistant", "tool_calls": [{"function": {"name": 1}}]}]):
            self.bad({"messages": msgs})

    def test_overlong_session_rejected_not_truncated(self):
        e = self.bad({**CHAT, "inferd": {"session": "a" * 129}})
        self.assertEqual(e["code"], "invalid_session_id")

    def test_explicit_values_reach_the_engine(self):
        async def main(extra):
            eng, fake = fresh()
            async with client() as c:
                r = await c.post("/v1/chat/completions", json={**CHAT, **extra})
            self.assertEqual(r.status_code, 200, r.text)
            return fake.gens[-1]
        g = run(main({"max_tokens": None, "temperature": 0, "top_k": -1, "top_p": 0.5}))
        self.assertEqual((g["max_tokens"], g["temperature"], g["top_k"], g["top_p"]), (4096, 0.0, 0, 0.5))
        g = run(main({"max_completion_tokens": 3, "temperature": None}))
        self.assertEqual((g["max_tokens"], g["temperature"]), (3, 0.7))
        g = run(main({"inferd": {"session": "a" * 128, "priority": 99}}))
        self.assertEqual((g["session"], g["priority"]), ("a" * 128, 10))

    def test_context_bodies(self):
        p = "/v1/contexts"
        for b in ({"system": "x", "ttl": -1}, {"system": "x", "ttl": "1h"}, {"system": "x", "ttl": 1e10},
                  {"system": "x", "name": "a" * 65}, {"system": "x", "name": 5}, {"system": 5},
                  {"documents": [1]}, {"documents": {"a": 1}}, {"messages": "x"}, {"messages": []},
                  {"system": "x", "persist": "disk"}, {"system": "x", "inferd": {"effort": "max"}},
                  {"system": "x", "tools": "t"}):
            self.bad(b, p)
        self.bad(None, p, raw=b'{"system": "x", "ttl": Infinity}')
        self.bad(None, p, raw=b'"x"')

    def test_path_ids(self):
        async def main():
            fresh()
            async with client() as c:
                return [(await c.delete("/v1/sessions/" + "s" * 129)).status_code,
                        (await c.post("/v1/sessions/" + "s" * 129 + "/offload")).status_code,
                        (await c.delete("/v1/contexts/" + "c" * 65)).status_code,
                        (await c.get("/v1/contexts/bad%20name")).status_code,
                        (await c.delete("/v1/sessions/" + "s" * 128)).status_code]
        self.assertEqual(run(main()), [400, 400, 400, 400, 200])


# ------------------------------------------------------------------------------------------------ R02 budgets
class Budgets(unittest.TestCase):
    def setUp(self):
        self.saved = {k: getattr(S.LIM, k) for k in vars(S.LIM) if not k.startswith("_")}

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(S.LIM, k, v)

    def test_body_size(self):
        S.LIM.body = 1000

        async def main():
            fresh()
            async with client() as c:
                big = await c.post("/v1/chat/completions", json={**CHAT, "x": "y" * 2000})

                async def chunks():  # no Content-Length: counted while reading
                    for _ in range(10):
                        yield b" " * 200
                streamed = await c.post("/v1/chat/completions", content=chunks())
                ok = await c.post("/v1/chat/completions", json=CHAT)
            return big, streamed, ok
        big, streamed, ok = run(main())
        self.assertEqual((big.status_code, streamed.status_code, ok.status_code), (413, 413, 200))
        self.assertEqual(big.json()["error"]["code"], "request_too_large")

    def test_inflight_limit_retry_after(self):
        S.LIM.inflight = 2

        async def main():
            eng, fake = fresh()
            fake.gen_gate = asyncio.Event()
            async with client() as c:
                held = [asyncio.create_task(c.post("/v1/chat/completions", json=CHAT)) for _ in range(2)]
                await settle()
                busy = await c.post("/v1/chat/completions", json=CHAT)
                health = await c.get("/health")
                fake.gen_gate.set()
                done = [await h for h in held]
                after = await c.post("/v1/chat/completions", json=CHAT)
            return busy, health, done, after
        busy, health, done, after = run(main())
        self.assertEqual(busy.status_code, 503)
        self.assertEqual(busy.headers["retry-after"], str(S.LIM.retry_after))
        self.assertEqual(busy.json()["error"]["code"], "server_overloaded")
        self.assertEqual(health.status_code, 200)  # /health is exempt from the in-flight budget
        self.assertEqual([d.status_code for d in done] + [after.status_code], [200, 200, 200])

    def test_inflight_bytes(self):
        S.LIM.inflight_bytes = 300

        async def main():
            eng, fake = fresh()
            fake.gen_gate = asyncio.Event()
            async with client() as c:
                held = asyncio.create_task(c.post("/v1/chat/completions", json={**CHAT, "pad": "p" * 150}))
                await settle()
                busy = await c.post("/v1/chat/completions", json={**CHAT, "pad": "p" * 150})
                fake.gen_gate.set()
                return busy, await held
        busy, held = run(main())
        self.assertEqual((busy.status_code, held.status_code), (503, 200))

    def test_preprocess_wait(self):
        S.LIM.preprocess, S.LIM.preprocess_wait = 1, 0.05

        async def main():
            fresh()
            S._prep_sem = asyncio.Semaphore(0)  # every slot taken
            async with client() as c:
                return await c.post("/v1/chat/completions", json=CHAT)
        r = run(main())
        self.assertEqual(r.status_code, 503)
        self.assertIn("retry-after", r.headers)

    def test_message_and_text_budgets(self):
        S.LIM.messages, S.LIM.prompt_chars = 3, 50

        async def main(body):
            fresh()
            async with client() as c:
                return (await c.post("/v1/chat/completions", json=body)).status_code
        self.assertEqual(run(main({"messages": [{"role": "user", "content": "x"}] * 4})), 413)
        self.assertEqual(run(main({"messages": [{"role": "user", "content": "x" * 51}]})), 413)
        self.assertEqual(run(main({"messages": [{"role": "user", "content": "x" * 20}]})), 200)

    def test_prompt_token_budget(self):
        S.LIM.prompt_tokens = 10

        async def main():
            fresh()
            async with client() as c:
                return await c.post("/v1/chat/completions", json=CHAT)
        r = run(main())
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, "context_length_exceeded"))

    def test_slow_client_is_cancelled_not_buffered(self):
        S.LIM.event_queue = 16

        async def main():
            eng, fake = fresh()
            q = await eng.submit(42, {"tokens": [1]})
            for i in range(100):  # nobody reads q
                fake.send({"id": 42, "t": [i]})
            await settle()
            self.assertEqual(q.qsize(), 1)
            ev = q.get_nowait()
            self.assertTrue(ev["done"].startswith("error: client too slow"), ev)
            self.assertNotIn(42, eng.queues)
            self.assertIn(42, fake.cancelled)
        run(main())

    def test_stream_end_to_end_and_cleanup(self):
        async def main():
            eng, fake = fresh()
            async with client() as c:
                r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True, "enable_thinking": False,
                                                               "stream_options": {"include_usage": True}})
            await settle()
            self.assertEqual(r.status_code, 200)
            text = "".join(json.loads(l[6:])["choices"][0]["delta"].get("content", "")
                           for l in r.text.splitlines() if l.startswith("data: {") and '"choices": [{' in l)
            self.assertEqual(text, "hello")
            self.assertTrue(r.text.rstrip().endswith("data: [DONE]"))
            self.assertEqual(eng.queues, {})
        run(main())


# ------------------------------------------------------------------------------------------------ R08 trust boundary
class Auth(unittest.TestCase):
    def tearDown(self):
        S.API_KEY = None

    def test_key_required_everywhere(self):
        S.API_KEY = "k" * 32

        async def main():
            fresh()
            async with client() as c:
                out = [(await c.get(p)).status_code for p in ("/health", "/v1/models", "/v1/cache", "/v1/contexts")]
                out.append((await c.post("/v1/chat/completions", json=CHAT,
                                         headers={"authorization": "Bearer wrong"})).status_code)
                out.append((await c.delete("/v1/sessions/x")).status_code)
                h = {"authorization": "Bearer " + "k" * 32}
                out.append((await c.post("/v1/chat/completions", json=CHAT, headers=h)).status_code)
                out.append((await c.get("/v1/models", headers=h)).status_code)
            return out
        self.assertEqual(run(main()), [401] * 6 + [200, 200])

    def test_non_loopback_needs_key(self):
        env = {k: v for k, v in os.environ.items() if k != "INFERD_API_KEY"}
        p = subprocess.run([sys.executable, os.path.join(HERE, "serve.py"), "--model", "/nonexistent",
                            "--host", "0.0.0.0"], capture_output=True, text=True, env=env, timeout=120)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("INFERD_API_KEY", p.stderr)
        env["INFERD_API_KEY"] = "short"
        p = subprocess.run([sys.executable, os.path.join(HERE, "serve.py"), "--model", "/nonexistent",
                            "--host", "203.0.113.5"], capture_output=True, text=True, env=env, timeout=120)
        self.assertIn("16+ characters", p.stderr)

    def test_loopback_detection(self):
        self.assertTrue(all(map(S._loopback, ("127.0.0.1", "127.1.2.3", "::1", "localhost"))))
        self.assertFalse(any(map(S._loopback, ("0.0.0.0", "::", "10.0.0.1", "example.com"))))

    def test_socket_checks(self):
        d = tempfile.mkdtemp()
        os.chmod(d, 0o700)
        f = os.path.join(d, "plain")
        open(f, "w").close()
        with self.assertRaises(SystemExit):
            S.Engine.check_socket(f)
        s = socket.socket(socket.AF_UNIX)
        path = os.path.join(d, "s.sock")
        s.bind(path)
        S.Engine.check_socket(path)  # ours, in an owner-only directory: fine
        s.close()


# ------------------------------------------------------------------------------------------------ R12 retention
class Retention(unittest.TestCase):
    def test_tokenizer_cache_counts_real_bytes(self):
        S._seg_cache.clear()
        S._seg_bytes = 0
        old, S._SEG_MAX_BYTES = S._SEG_MAX_BYTES, 20000
        try:
            for i in range(200):
                S.encode_cached(f"<|im_start|>user\n{i} " + "z" * 300)
            total = sum(c for _, c in S._seg_cache.values())
            self.assertEqual(total, S._seg_bytes)
            self.assertLessEqual(S._seg_bytes, 20000)
            ids, cost = next(iter(S._seg_cache.values()))
            self.assertGreaterEqual(cost, sys.getsizeof("z" * 300) + 4 * len(ids))
            self.assertEqual(S.encode_cached("<|im_start|>user\n199 " + "z" * 300),
                             [ord(c) for c in "<|im_start|>user\n199 " + "z" * 300])  # hits return the same ids
        finally:
            S._SEG_MAX_BYTES = old

    def test_sessions_bounded_lru(self):
        old = S.LIM.sessions
        S.LIM.sessions = 3

        async def main():
            eng, fake = fresh()
            async with client() as c:
                for n in range(6):
                    r = await c.post("/v1/chat/completions", json={**CHAT, "inferd": {"session": f"s{n}"}})
                    self.assertEqual(r.status_code, 200)
            await settle()
            dropped = [m["session"] for m in fake.ops if m["op"] == "session_drop"]
            return dropped, list(S._sessions)
        try:
            dropped, kept = run(main())
        finally:
            S.LIM.sessions = old
        self.assertEqual(dropped, ["s0", "s1", "s2"])
        self.assertEqual(kept, ["s3", "s4", "s5"])

    def test_session_ttl(self):
        old = S.LIM.session_ttl
        S.LIM.session_ttl = 10.0

        async def main():
            eng, fake = fresh()
            S.session_begin("busy")
            S.session_begin("idle")
            S.session_end("idle")
            await S._expire_once(now=S.time.time() + 100)
            return [m["session"] for m in fake.ops if m["op"] == "session_drop"], list(S._sessions)
        try:
            dropped, kept = run(main())
        finally:
            S.LIM.session_ttl = old
        self.assertEqual((dropped, kept), (["idle"], ["busy"]))

    def test_locks_reclaimed_with_many_short_lived_contexts(self):
        async def main():
            eng, fake = fresh()
            async with client() as c:
                for n in range(40):
                    await c.post("/v1/contexts", json={"name": f"tmp{n}", "system": "short-lived " * 8})
                    await c.post("/v1/chat/completions", json={**CHAT, "inferd": {"context": f"tmp{n}",
                                                                                  "effort": "low"}})
                    await c.delete(f"/v1/contexts/tmp{n}")
            await settle()
            return fake.saved
        self.assertEqual(run(main()), set())
        self.assertEqual((S._ctx_locks, S.contexts), ({}, {}))


if __name__ == "__main__":
    unittest.main()
