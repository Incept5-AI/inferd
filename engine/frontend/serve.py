# OpenAI-compatible HTTP front end for inferd (the native engine).
# Handles chat templating and tokenization, streams tokens back as SSE; inferd does all model work.
#   python serve.py --model <dir> --socket <run dir>/inferd.sock --port 8300 [--name qwen3.8-27b] (run.sh starts both)
import argparse, array, asyncio, base64, collections, contextlib, glob, hmac, ipaddress, itertools, json, math, os, re
import contextvars, stat, sys, threading, time, uuid

from jinja2 import TemplateError

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

TRUST = """\
Trust model: inferd serves one trusted local user. Anyone who can reach the HTTP port (or the engine's Unix socket)
can run generations and create, offload or delete any context or session: names are not ownership, and there are no
per-user namespaces or quotas. By default the front end listens on 127.0.0.1 only, which still admits every local
process. INFERD_API_KEY=<secret> makes every endpoint (including /health) require "Authorization: Bearer <secret>";
it is required (the server refuses to start without one, 16+ characters) when --host is not a loopback address.
One key is one tenant: it authenticates, it does not isolate agents from each other.
Images: inline data: URLs only, PNG, JPEG, WebP or GIF (first frame); any other format is refused unread. Images
are decoded in a child process killed after INFERD_IMAGE_DECODE_TIMEOUT s (10; INFERD_IMAGE_DECODE=inline: in this
process). INFERD_IMAGE_URLS=1 lets callers make the server fetch any http(s) URL (internal
hosts and redirects included) and INFERD_IMAGE_FILES=1 lets them read any file this user can read: enable them only
when every caller is trusted with that access.
Limits (environment; overload answers HTTP 503 with Retry-After, oversized input 413, malformed input 400):
  INFERD_MAX_BODY_MB (100) request body; INFERD_MAX_INFLIGHT (256) requests in progress, streams included;
  INFERD_MAX_INFLIGHT_MB (1024) their bodies together; INFERD_MAX_PREPROCESS (16) concurrent tokenize/render/vision
  jobs, waiting at most INFERD_PREPROCESS_WAIT s (60); INFERD_MAX_MESSAGES (10000); INFERD_MAX_PROMPT_CHARS
  (8000000) message text; INFERD_MAX_PROMPT_TOKENS (1048576); INFERD_MAX_OUTPUT_TOKENS (262144) max_tokens;
  INFERD_EVENT_QUEUE (8192) unread engine events per request before a non-reading client's request is cancelled;
  INFERD_RETRY_AFTER (2) s; INFERD_TOKCACHE_MB (256) tokenizer cache; INFERD_MAX_CONTEXTS (256) and
  INFERD_MAX_CONTEXT_MB (1024) retained context records (messages, tools, token ids) together;
  INFERD_EVENT_TIMEOUT (1800 s; 0 = never) without any engine event before a request is abandoned and cancelled;
  INFERD_MAX_SESSIONS (4096) session ids kept (least recently used idle ones are dropped from the engine);
  INFERD_SESSION_TTL (0 = never) seconds before an idle session is dropped.
"""

ap = argparse.ArgumentParser(description="OpenAI-compatible HTTP front end for inferd.", epilog=TRUST,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True)
ap.add_argument("--socket", default="/tmp/inferd.sock", help="inferd's Unix socket (run.sh: an owner-only directory)")
ap.add_argument("--host", default="127.0.0.1", help="anything but loopback requires INFERD_API_KEY")
ap.add_argument("--port", type=int, default=8300)
ap.add_argument("--name", default="qwen3.8-27b")
args = ap.parse_args()


def _loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


API_KEY = os.environ.get("INFERD_API_KEY") or None
if not _loopback(args.host) and (API_KEY is None or len(API_KEY) < 16):
    sys.exit(f"serve: --host {args.host} is not loopback: set INFERD_API_KEY (16+ characters) first. " + TRUST)


def _env_num(name, default, lo, hi, cast=int):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        v = cast(raw)
    except ValueError:
        sys.exit(f"serve: {name}={raw!r} is not a number")
    if not (math.isfinite(v) and lo <= v <= hi):
        sys.exit(f"serve: {name}={raw} is outside {lo}..{hi}")
    return v


class LIM:  # service-wide budgets (Codex review R02, R12), all checked before the expensive work
    body = _env_num("INFERD_MAX_BODY_MB", 100, 1, 4096) << 20
    inflight = _env_num("INFERD_MAX_INFLIGHT", 256, 1, 1 << 20)
    inflight_bytes = _env_num("INFERD_MAX_INFLIGHT_MB", 1024, 1, 1 << 20) << 20
    preprocess = _env_num("INFERD_MAX_PREPROCESS", 16, 1, 1024)
    preprocess_wait = _env_num("INFERD_PREPROCESS_WAIT", 60.0, 0.0, 86400.0, float)
    messages = _env_num("INFERD_MAX_MESSAGES", 10000, 1, 1 << 24)
    prompt_chars = _env_num("INFERD_MAX_PROMPT_CHARS", 8_000_000, 1, 1 << 34)
    prompt_tokens = _env_num("INFERD_MAX_PROMPT_TOKENS", 1 << 20, 1, 1 << 30)
    output_tokens = _env_num("INFERD_MAX_OUTPUT_TOKENS", 262144, 1, (1 << 31) - 1)
    event_queue = _env_num("INFERD_EVENT_QUEUE", 8192, 16, 1 << 24)
    retry_after = _env_num("INFERD_RETRY_AFTER", 2, 1, 3600)
    tokcache = _env_num("INFERD_TOKCACHE_MB", 256, 0, 1 << 20) << 20
    contexts = _env_num("INFERD_MAX_CONTEXTS", 256, 1, 1 << 20)
    context_bytes = _env_num("INFERD_MAX_CONTEXT_MB", 1024, 1, 1 << 20) << 20  # retained context records together
    event_timeout = _env_num("INFERD_EVENT_TIMEOUT", 1800.0, 0.0, 1e9, float)  # backstop: s without an engine event
    sessions = _env_num("INFERD_MAX_SESSIONS", 4096, 1, 1 << 24)
    session_ttl = _env_num("INFERD_SESSION_TTL", 0.0, 0.0, 1e9, float)

tok = AutoTokenizer.from_pretrained(args.model)
# Stop ids: generation_config.json when the checkpoint has one, else config.json (e.g. mlx-community Qwen3.5-9B).
_cfg_path = next(p for p in (f"{args.model}/generation_config.json", f"{args.model}/config.json") if os.path.exists(p))
gen_cfg = json.load(open(_cfg_path))


def _gen_default(key, ok):
    """A sampling default from the checkpoint's generation_config.json (as vLLM does), or None."""
    if not _cfg_path.endswith("generation_config.json"):
        return None
    v = gen_cfg.get(key)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and ok(v) else None


# Defaults for fields a request omits (Codex review 2026-11-04, R03): top_k and top_p from generation_config.json
# (Qwen3.8: top_k 20, top_p 0.95; Gemma-4: 64, 0.95), else no limit and 1.0. Temperature stays 0.7 when omitted. With
# top_k <= 64 the default sampled request is drawn exactly on the speculative path.
DEFAULT_TOP_K = int(_gen_default("top_k", lambda v: v == int(v) and 0 <= v <= 1 << 20) or 0)
SPEC_CANDIDATES = 64  # the engine's speculative candidates per row (SPEC_TOPK; src/sampling.h kSpecTopK)
DEFAULT_TOP_P = float(_gen_default("top_p", lambda v: 0 < v <= 1) or 1.0)
eos = gen_cfg.get("eos_token_id", gen_cfg.get("text_config", {}).get("eos_token_id", []))
_end = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|eot|>", "<end_of_turn>", "<turn|>") if t in tok.get_vocab()]
STOP = sorted(set(i for i in (eos if isinstance(eos, list) else [eos]) + _end if i is not None))
THINK_END = "</think>"
app = FastAPI()
ids = itertools.count(1)

# Vision: only for checkpoints that carry the vision tower's weights (the mlx-community 27B 4-bit build has none).
_vision = None
_vision_lock = threading.Lock()  # one image encode at a time (MLX in worker threads)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # run with -I: our own vision.py
import structured  # response_format: JSON extraction, schema validation, retries (structured.py)


def vision():
    global _vision
    if _vision is None:
        has = any("vision_tower." in k for f in glob.glob(f"{args.model}/*.index.json")
                  for k in json.load(open(f))["weight_map"])
        if not has:
            raise ValueError("this checkpoint has no vision tower; images are not supported")
        if os.environ.get("VISION") != "1":  # inferd sized its KV pool without room for the tower (see run.sh)
            print("serve: loading the vision tower without VISION=1: inferd left no GPU room for it; on machines with "
                  "64 GiB or less, restart with VISION=1 before image work", file=sys.stderr, flush=True)
        from vision import Vision
        _vision = Vision(args.model)
    return _vision


class Engine:
    """One socket to inferd, multiplexed by request id."""

    def __init__(self):
        self.queues = {}
        self.writer = None
        self.lock = asyncio.Lock()
        self.tasks = set()  # fire-and-forget cleanup (cancels, drops): referenced until done

    @staticmethod
    def check_socket(path):
        """The engine socket must be a socket owned by this user (Codex review R08); warn about a shared directory."""
        st = os.lstat(path)
        if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
            raise SystemExit(f"serve: {path} is not a socket owned by this user; refusing to connect")
        d = os.stat(os.path.dirname(os.path.abspath(path)))
        if d.st_uid != os.getuid() or d.st_mode & 0o077:
            print(f"serve: warning: {os.path.dirname(path)} is not an owner-only directory; other local users may "
                  "replace the socket (run.sh puts it in an owner-only run directory)", file=sys.stderr, flush=True)

    async def connect(self):
        self.check_socket(args.socket)
        reader, self.writer = await asyncio.open_unix_connection(args.socket, limit=1 << 24)
        asyncio.create_task(self.pump(reader))
        self.writer.write(await asyncio.to_thread(structured.vocab_op, tok))  # for constrained JSON (src/grammar.h)
        await self.writer.drain()

    alive = False

    def background(self, coro):
        t = asyncio.get_running_loop().create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    async def pump(self, reader):
        self.alive = True
        try:
            while line := await reader.readline():
                ev = json.loads(line)
                rid = ev["id"]
                q = self.queues.get(rid)
                if q is None:
                    continue
                if "done" not in ev and q.qsize() >= LIM.event_queue:
                    # The consumer (a client that stopped reading its stream) is that far behind: the request ends
                    # here, with one error event, instead of buffering without bound (Codex review R02).
                    self.queues.pop(rid, None)
                    while not q.empty():
                        q.get_nowait()
                    q.put_nowait({"id": rid, "done": f"error: client too slow ({LIM.event_queue} events unread, "
                                  "INFERD_EVENT_QUEUE); request cancelled", "p": 0, "c": 0, "cached": 0})
                    self.background(self.cancel_quiet(rid))
                    continue
                q.put_nowait(ev)
        finally:  # inferd closed the socket or died: end every open request instead of leaving it waiting forever
            self.alive = False
            for rid, q in list(self.queues.items()):
                q.put_nowait({"id": rid, "done": "error: engine disconnected", "p": 0, "c": 0, "cached": 0})

    async def submit(self, rid, body, op="gen"):
        q = asyncio.Queue()
        self.queues[rid] = q
        written = False
        try:
            async with self.lock:
                self.writer.write((json.dumps(dict(body, op=op, id=rid)) + "\n").encode())
                written = True  # in the transport's buffer: the engine may act on it from here on
                await self.writer.drain()
        except BaseException:  # failure or cancellation (CancelledError too, Codex review R07): no queue may outlive
            self.queues.pop(rid, None)  # the request, and a request the engine may have seen is cancelled there
            if written:
                self.background(self.cancel_quiet(rid))
            raise
        return q

    async def cancel(self, rid):
        async with self.lock:
            self.writer.write((json.dumps({"op": "cancel", "id": rid}) + "\n").encode())
            await self.writer.drain()

    async def cancel_quiet(self, rid):
        """Best-effort cancel (cleanup paths): an engine that has gone away has nothing left to cancel."""
        try:
            if self.writer is not None and not self.writer.is_closing():
                await self.cancel(rid)
        except Exception:
            pass

    def release(self, rid):
        self.queues.pop(rid, None)

    async def call(self, op, **fields):
        """One control op (or a context save, op "gen" with "ctx") -> its done event."""
        rid = next(ids)
        q = await self.submit(rid, fields, op=op)
        try:
            while True:
                ev = await next_event(q, rid)
                if ev.get("_timeout"):
                    self.background(self.cancel_quiet(rid))
                if "done" in ev:
                    return ev
        except BaseException:
            if op == "gen":  # a context save still computing: stop it (control ops answer at once)
                self.background(self.cancel_quiet(rid))
            raise
        finally:
            self.release(rid)


engine = Engine()


async def _send_json(send, status, payload, headers=()):
    body = json.dumps(payload).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                            *headers]})
    await send({"type": "http.response.body", "body": body})


def _err(message, typ, code):
    return {"error": {"message": message, "type": typ, "code": code}}


class Guard:
    """ASGI wrapper around the app: the API key, then the in-flight and body budgets, before any handler runs (Codex
    review R02, R08). The body is read here, bounded, and replayed to the app; a request counts as in flight until its
    response (a stream included) has finished."""

    def __init__(self, app):
        self.app, self.n, self.bytes = app, 0, 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        if API_KEY is not None:
            got = headers.get(b"authorization", b"")
            want = b"Bearer " + API_KEY.encode()
            if not hmac.compare_digest(got, want):
                return await _send_json(send, 401, _err("missing or invalid API key (Authorization: Bearer ...)",
                                                        "invalid_request_error", "invalid_api_key"),
                                        [(b"www-authenticate", b"Bearer")])
        retry = [(b"retry-after", str(LIM.retry_after).encode())]
        if scope["path"] != "/health" and self.n >= LIM.inflight:
            return await _send_json(send, 503, _err(f"server busy: {self.n} requests in progress "
                                                    "(INFERD_MAX_INFLIGHT); retry later", "server_error",
                                                    "server_overloaded"), retry)
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await _send_json(send, 400, _err("bad Content-Length", "invalid_request_error", "invalid_body"))
        if declared > LIM.body:
            return await _send_json(send, 413, _err(f"request body is {declared} bytes; the limit is {LIM.body} "
                                                    "(INFERD_MAX_BODY_MB)", "invalid_request_error",
                                                    "request_too_large"))
        if declared and self.bytes + declared > LIM.inflight_bytes:
            return await _send_json(send, 503, _err("server busy: too many request bytes in progress "
                                                    "(INFERD_MAX_INFLIGHT_MB); retry later", "server_error",
                                                    "server_overloaded"), retry)
        self.n += 1
        held = 0
        try:
            chunks, more = [], True
            while more:
                msg = await receive()
                if msg["type"] == "http.disconnect":
                    return
                chunk = msg.get("body", b"")
                more = msg.get("more_body", False)
                held += len(chunk)
                self.bytes += len(chunk)
                if held > LIM.body:
                    return await _send_json(send, 413, _err(f"request body is over {LIM.body} bytes "
                                                            "(INFERD_MAX_BODY_MB)", "invalid_request_error",
                                                            "request_too_large"))
                if self.bytes > LIM.inflight_bytes:
                    return await _send_json(send, 503, _err("server busy: too many request bytes in progress "
                                                            "(INFERD_MAX_INFLIGHT_MB); retry later", "server_error",
                                                            "server_overloaded"), retry)
                chunks.append(chunk)
            body, sent = b"".join(chunks), False
            chunks = None

            async def replay():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()
            await self.app(scope, replay, send)
        finally:
            self.n -= 1
            self.bytes -= held


@app.on_event("startup")
async def startup():
    await engine.connect()
    asyncio.create_task(_expire_loop())


@app.exception_handler(Exception)
async def _unexpected(req, e):  # a bug, not the client's fault: still an OpenAI-style error body, never a traceback
    print(f"serve: {req.url.path}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    return JSONResponse(_err("internal error", "server_error", "internal_error"), status_code=500)


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": args.name, "object": "model", "owned_by": "inferd"}]}


@app.get("/health")
async def health():
    if not engine.alive:
        return JSONResponse({"ok": False, "error": "engine disconnected"}, status_code=503)
    return {"ok": True}


# Tool calls, Qwen3.x markup (the chat template's own format):
#   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>
TOOL_OPEN = "<tool_call>"
_TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*(?:</tool_call>|$)", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def _param_types(tools):
    """tool name -> {parameter: JSON-schema type} from the request's tools."""
    out = {}
    for t in tools or []:
        f = t.get("function", t) if isinstance(t, dict) else {}
        props = ((f.get("parameters") or {}).get("properties") or {})
        out[f.get("name")] = {k: (v or {}).get("type") for k, v in props.items()}
    return out


def parse_tool_calls(text, tools=None):
    """-> (content before the first call or None, [OpenAI tool_calls]). No markup: (text, [])."""
    i = text.find(TOOL_OPEN)
    if i < 0:
        return text, []
    types, calls = _param_types(tools), []
    for m in _TOOL_RE.finditer(text, i):
        name, schema, arguments = m.group(1), types.get(m.group(1), {}), {}
        for k, v in _PARAM_RE.findall(m.group(2)):
            if schema.get(k) != "string":  # numbers, booleans, arrays, objects arrive as JSON text
                try:
                    v = json.loads(v)
                except ValueError:
                    pass
            arguments[k] = v
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
    if not calls:  # malformed markup: leave the text alone
        return text, []
    return (text[:i].strip() or None), calls


# User stop strings (OpenAI `stop`: a string or a list of strings). They apply to the answer text (content, tool-call
# markup included; with thinking on, not to the reasoning): generation ends at the first occurrence, the stop string
# is trimmed (vLLM's include_stop_str_in_output keeps it), finish_reason is "stop".
def parse_stop(v):
    """-> list of non-empty stop strings; ValueError for anything but a string or a list of strings."""
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
        raise ValueError("stop must be a string or a list of strings")
    return [s for s in v if s]


def _first_stop(text, stops):
    """-> (index, stop string) of the earliest (then longest) stop in text, or None."""
    hits = [(i, -len(s), s) for s in stops if (i := text.find(s)) >= 0]
    return (min(hits)[0], min(hits)[2]) if hits else None


def cut_at_stop(text, stops, keep=False):
    """-> (text up to the first stop string, matched)."""
    h = _first_stop(text, stops)
    if h is None:
        return text, False
    return text[:h[0]] + (h[1] if keep else ""), True


class StopMatcher:
    """Streaming stop-string search: feed(piece) -> (text safe to emit, stopped). A suffix that could still begin a
    stop string is held back until the next piece decides it; flush() returns it when generation ends otherwise."""

    def __init__(self, stops, keep=False):
        self.stops, self.keep, self.buf, self.done = list(stops), keep, "", False

    def feed(self, piece):
        if self.done:
            return "", True
        s = self.buf + piece
        h = _first_stop(s, self.stops)
        if h is not None:
            self.done, self.buf = True, ""
            return s[:h[0]] + (h[1] if self.keep else ""), True
        k = next((n for n in range(min(max(map(len, self.stops)) - 1, len(s)), 0, -1)
                  if any(st.startswith(s[-n:]) for st in self.stops)), 0)
        self.buf = s[len(s) - k:] if k else ""
        return s[:len(s) - k], False

    def flush(self):
        out, self.buf = self.buf, ""
        return out


async def _drain_done(q, timeout=30.0):
    """After a cancel: the request's final event (usage), skipping tokens still in flight."""
    try:
        while True:
            ev = await asyncio.wait_for(q.get(), timeout)
            if "done" in ev:
                return ev
    except asyncio.TimeoutError:
        return {"done": "cancelled", "p": 0, "c": 0, "cached": 0}


def _abandoned(rid=None):
    """The local terminal event for a request whose engine went quiet past INFERD_EVENT_TIMEOUT (Codex review
    2026-11-04, R05 backstop). `_local`: the engine never ended it, so the caller cancels it there."""
    return {"id": rid, "done": f"error: no engine event for {LIM.event_timeout:g} s (INFERD_EVENT_TIMEOUT); request "
            "abandoned", "p": 0, "c": 0, "cached": 0, "_local": True, "_timeout": True}


async def next_event(q, rid=None):
    """The request's next engine event, or _abandoned() after INFERD_EVENT_TIMEOUT seconds without one."""
    if not LIM.event_timeout:
        return await q.get()
    try:
        return await asyncio.wait_for(q.get(), LIM.event_timeout)
    except asyncio.TimeoutError:
        return _abandoned(rid)


# ---- Client disconnects (Codex review 2026-11-04, R07). A non-streaming request waits for the engine without reading
# the ASGI receive channel, so a client that hung up went unnoticed until generation finished. chat() publishes the
# request's receive here; complete() watches it while it waits and ends the request (cancelling the engine work) on
# http.disconnect. Streams are watched by Starlette (spec 2.3: listen_for_disconnect cancels the stream; _Stream cleans
# up).
_client_receive = contextvars.ContextVar("inferd_client_receive", default=None)
_GONE = {"done": "error: client disconnected", "p": 0, "c": 0, "cached": 0, "_local": True, "_gone": True}


async def _watch_disconnect(receive, q):
    while True:
        try:
            msg = await receive()
        except asyncio.CancelledError:
            raise
        except Exception:  # the server's receive channel failed: the client is as good as gone
            msg = {"type": "http.disconnect"}
        if msg.get("type") == "http.disconnect":
            q.put_nowait(dict(_GONE))
            return
        await asyncio.sleep(0.05)  # not a disconnect (nothing else is expected once the body is read): never spin


def _template_messages(messages):
    """OpenAI clients send tool_calls[].function.arguments as a JSON string; the template iterates a mapping."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                f = dict(c.get("function", c))
                if isinstance(f.get("arguments"), str):
                    try:
                        f["arguments"] = json.loads(f["arguments"]) if f["arguments"].strip() else {}
                    except ValueError:
                        pass
                calls.append({**c, "function": f})
            m = {**m, "tool_calls": calls, "content": m.get("content") or ""}
        out.append(m)
    return out


def _has_images(messages):
    """vision.has_images without importing vision.py: that import loads MLX (and mlx_vlm, PIL, transformers'
    processors) and creates a Metal device in this process. On a text-only first request it cost ~0.6-0.8 s of TTFT and
    stalled inferd's first command buffers by ~0.5 s more (M5 Pro, 2026-11-03), so it waits for the first image."""
    return any(isinstance(m.get("content"), list) and any(p.get("type") in ("image_url", "image") for p in m["content"])
               for m in messages)


def render(body):
    """-> (inferd request fields for the prompt, thinking on)."""
    kw = dict(body.get("chat_template_kwargs") or {})
    for k in ("enable_thinking", "reasoning_effort"):
        if k in body and k not in kw:
            kw[k] = body[k]
    tools = body.get("tools")
    body = {**body, "messages": _template_messages(body["messages"])}
    thinking = kw.get("enable_thinking", True)
    if _has_images(body["messages"]):
        with _vision_lock:
            tokens, emb, mrope, delta = vision().prepare(body["messages"], tools=tools, **kw)
        return {"tokens": tokens, "mrope": mrope, "rope_delta": delta,
                "embeds": base64.b64encode(emb).decode()}, thinking
    text = tok.apply_chat_template(body["messages"], tools=tools, add_generation_prompt=True, tokenize=False, **kw)
    out = {"tokens": encode_cached(text)}
    if text.startswith(_SEG + "system"):  # agents sharing this system prompt resume after it (prefix cache)
        end = text.find(_SEG, 1)
        if end > 0:
            out["snap_at"] = [len(encode_cached(text[:end]))]
    return out, thinking


# Per-turn tokenization: an agent's prompt is its previous prompt plus a few messages, but encoding the whole text
# again costs ~84 ms at 128k tokens, on the event loop (16 agents' turns queue behind each other). The rendered text
# is split before each <|im_start|> (a special token, which the tokenizer splits out before BPE, so the pieces encode
# independently) and each piece's ids are cached: a turn encodes only its new messages.
_SEG = "<|im_start|>"
_seg_cache = {}  # piece -> (ids as array('i'), retained bytes), least recently used first
_seg_bytes = 0
_SEG_MAX_BYTES = LIM.tokcache  # INFERD_TOKCACHE_MB
_SEG_ENTRY = 232  # per entry beyond the string and the array: a dict slot and the (ids, cost) tuple (CPython 3.12)


_seg_lock = threading.Lock()  # render runs in worker threads


def _seg_cost(piece, ids):
    """Retained bytes of one cache entry, measured (Codex review R12): the string object (1, 2 or 4 bytes per
    character plus its header), the token ids (an array of 4-byte ints, not a list of int objects) and the entry."""
    return sys.getsizeof(piece) + sys.getsizeof(ids) + _SEG_ENTRY


def encode_cached(text):
    global _seg_bytes
    out = []
    i = 0
    while i < len(text):
        j = text.find(_SEG, i + 1)
        j = len(text) if j < 0 else j
        piece = text[i:j]
        with _seg_lock:
            got = _seg_cache.pop(piece, None)
            if got is not None:
                _seg_cache[piece] = got  # most recently used last
        if got is None:
            ids = tok.encode(piece, add_special_tokens=False)
            if len(piece) >= 64:  # short pieces (the generation prompt) are cheaper to encode than to keep
                arr = array.array("i", ids)
                cost = _seg_cost(piece, arr)
                with _seg_lock:
                    if piece not in _seg_cache and cost <= _SEG_MAX_BYTES:
                        _seg_cache[piece] = (arr, cost)
                        _seg_bytes += cost
                    while _seg_bytes > _SEG_MAX_BYTES:
                        old = next(iter(_seg_cache))
                        _seg_bytes -= _seg_cache.pop(old)[1]
            out.extend(ids)
        else:
            out.extend(got[0])
        i = j
    return out


class Detok:
    """Incremental detokenizer that holds back incomplete UTF-8 sequences. Each token decodes only the window since
    the last emitted character boundary (prefix .. read offsets, the vLLM / HF TextStreamer scheme), so a long output
    costs O(n), not O(n^2) (Codex review 2026-11-02, S08). flush() returns whatever is still held at the end."""

    def __init__(self):
        self.ids, self.prefix, self.read = [], 0, 0

    def add(self, t):
        self.ids.append(t)
        before = tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=False)
        text = tok.decode(self.ids[self.prefix:], skip_special_tokens=False)
        if len(text) <= len(before) or text.endswith("\ufffd"):
            return ""
        self.prefix, self.read = self.read, len(self.ids)
        return text[len(before):]

    def flush(self):
        if self.read >= len(self.ids):
            return ""
        before = tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=False)
        text = tok.decode(self.ids[self.prefix:], skip_special_tokens=False)
        self.prefix = self.read = len(self.ids)
        return text[len(before):]


def drain_split(split, state, tail):
    """At the terminal event: route the detokenizer's held tail, then release the reasoning text split() held back
    as a possible partial </think> (it ended inside the reasoning). Streamed and non-streamed text then agree."""
    r, c = split(state, tail)
    if state["reasoning"] and state["buf"]:
        r, state["buf"] = (r or "") + state["buf"], ""
    return r, c


# ---- Agent contexts and sessions (docs/CONTEXTS.md) ----
# A saved context is a list of messages (a system prompt, documents, few-shot turns) plus optional tools, rendered at one
# effort, computed once by the engine, pinned in its prefix cache and optionally written to the SSD tier. Chat requests
# launch from it with "inferd": {"context": id}: the front end renders the context's messages before the request's, so
# the context is an exact token prefix and the engine resumes from its snapshot. The effort is part of the rendered
# system prompt (Qwen3.x templates), so each effort is its own variant, computed (once) the first time it is used.
EFFORTS = ("none", "low", "medium", "xhigh")
_SENT = "\x00inferd-context-end\x00"
_TURN_MARKERS = (_SEG, "<start_of_turn>", "<|turn>")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
contexts = {}          # id -> record (see create_context)
_ctx_locks = {}        # id -> [asyncio.Lock, holders + waiters]: one lifecycle step per context at a time; reclaimed
_ctx_gen = itertools.count(1)  # record generations: engine handles are "<id>@<effort>#<gen>", never reused
_SESSION_MAX = 128     # session ids: longer ones are rejected (they used to be truncated, so distinct ids could alias)
# Admission (Codex review 2026-11-04, R06): a create checks the limits and reserves its slot (a new id) and its text's
# bytes in one synchronous step, before any await, so concurrent creates cannot all see room; the reservation is
# released when the create publishes, fails or is cancelled. Bytes are an estimate of what a record retains (its
# messages and tools as text, plus its token id lists), checked against INFERD_MAX_CONTEXT_MB again when the tokens
# are known (publish, and each new effort variant).
_ctx_pending = {}      # id -> reserved bytes of a create in progress
_ctx_new = set()       # ids of creates in progress that will add a record (not replace one)
_TOKEN_BYTES = 36      # a Python int in a list (pointer + int object), conservatively


def _ctx_text_bytes(messages, tools):
    return len(json.dumps([messages, tools], ensure_ascii=False, default=str)) + 512


def _ctx_bytes(rec):
    """Estimated retained bytes of one record: its text plus each distinct token list (variants may share one)."""
    lists = {id(v["tokens"]): len(v["tokens"]) for v in rec["variants"].values()}
    return rec.get("text_bytes", 0) + _TOKEN_BYTES * sum(lists.values())


def _ctx_used(exclude=None):
    return sum(_ctx_bytes(r) for k, r in contexts.items() if k != exclude) + \
        sum(b for k, b in _ctx_pending.items() if k != exclude)


def _ctx_over(what):
    return ApiError(413, f"{what}: contexts would hold over {LIM.context_bytes / 2 ** 20:g} MiB (INFERD_MAX_CONTEXT_MB); "
                    "delete one first", "context_memory_limit")


class ApiError(Exception):
    def __init__(self, status, message, code=None, param=None, retry_after=None):
        super().__init__(message)
        self.status, self.message, self.code, self.param, self.retry_after = status, message, code, param, retry_after

    def response(self):
        err = {"message": self.message, "type": "invalid_request_error" if self.status < 500 else "server_error"}
        if self.code:
            err["code"] = self.code
        if self.param:
            err["param"] = self.param
        headers = {"Retry-After": str(self.retry_after)} if self.retry_after else None
        return JSONResponse({"error": err}, status_code=self.status, headers=headers)


# ---- Request validation (Codex review R03): one typed boundary for every JSON body, before any work. Malformed
# input is HTTP 400 with an OpenAI-style error; null means "not given" (the default); zero is never a default.
_I32 = (1 << 31) - 1


def _bad(param, msg, code="invalid_value"):
    return ApiError(400, f"{param}: {msg}", code, param=param)


def _reject_constant(c):
    raise ValueError(f"{c} is not valid JSON")


_SURROGATE = re.compile("[\ud800-\udfff]")


def _lone_surrogate(v):
    """-> a JSON path holding a surrogate code point (U+D800..U+DFFF), or None. Iterative: the body's depth is bounded
    by json.loads (RecursionError), not by this walk."""
    stack = [(v, "")]
    while stack:
        x, path = stack.pop()
        if isinstance(x, str):
            if _SURROGATE.search(x):
                return path or "body"
        elif isinstance(x, dict):
            for k, y in x.items():
                if _SURROGATE.search(k):
                    return f"{path}.{k!a}" if path else repr(k)
                stack.append((y, f"{path}.{k}" if path else k))
        elif isinstance(x, list):
            stack.extend((y, f"{path}[{i}]") for i, y in enumerate(x))
    return None


async def read_json(req):
    """The request body as a JSON object (NaN / Infinity rejected; Guard has bounded the size). Strings must be
    Unicode scalar values (Codex review 2026-11-04, R05): Python's decoder accepts an escaped lone surrogate ("\\ud800",
    and raw surrogate bytes via surrogatepass) that the engine's strict parser rejects, so such a body is a 400 here,
    before anything is forwarded."""
    raw = await req.body()
    try:
        b = json.loads(raw, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as e:
        raise ApiError(400, f"request body is not valid JSON: {str(e)[:200]}", "invalid_json")
    if not isinstance(b, dict):
        raise ApiError(400, "request body must be a JSON object", "invalid_json")
    # Surrogates reach Python strings only through a \u escape or raw ED A0..BF bytes: skip the walk otherwise.
    if (b"\\u" in raw or b"\xed" in raw) and (where := _lone_surrogate(b)) is not None:
        raise ApiError(400, f"{where}: strings must be valid Unicode (a lone surrogate \\uD800-\\uDFFF is not)",
                       "invalid_unicode", param=where)
    return b


def v_int(d, key, lo, hi, where=""):
    """d[key] as an int in [lo, hi] (integral floats accepted), None when absent or null."""
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or \
            (isinstance(v, float) and not (math.isfinite(v) and v.is_integer())):
        raise _bad(where + key, "must be an integer")
    if not lo <= v <= hi:
        raise _bad(where + key, f"must be between {lo} and {hi}")
    return int(v)


def v_num(d, key, lo, hi, lo_open=False, where=""):
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise _bad(where + key, "must be a finite number")
    if not (lo < v if lo_open else lo <= v) or v > hi:
        raise _bad(where + key, f"must be {'>' if lo_open else '>='} {lo} and <= {hi}")
    return float(v)


def v_type(d, key, types, what, where=""):
    v = d.get(key)
    if v is not None and (not isinstance(v, types) or (isinstance(v, bool) and bool not in
                                                        (types if isinstance(types, tuple) else (types,)))):
        raise _bad(where + key, f"must be {what}")
    return v


def v_choice(d, key, choices, where=""):
    v = d.get(key)
    if v is not None and (not isinstance(v, str) or v not in choices):
        raise _bad(where + key, f"must be one of {', '.join(choices)}")
    return v


def v_name(v, param):
    if not isinstance(v, str) or not _NAME_RE.match(v):
        raise _bad(param, "1-64 characters of [A-Za-z0-9._:-], starting with a letter or digit", "invalid_name")
    return v


def v_session(v, param="session"):
    if not isinstance(v, str) or not v:
        raise _bad(param, "must be a non-empty string")
    if len(v) > _SESSION_MAX or any(ord(ch) < 32 for ch in v):
        raise _bad(param, f"at most {_SESSION_MAX} characters, no control characters", "invalid_session_id")
    return v


def v_messages(msgs, param="messages"):
    """A non-empty list of message objects within the count and text budgets."""
    if not isinstance(msgs, list) or not msgs:
        raise _bad(param, "must be a non-empty list of message objects")
    if len(msgs) > LIM.messages:
        raise ApiError(413, f"{len(msgs)} messages; the limit is {LIM.messages} (INFERD_MAX_MESSAGES)",
                       "request_too_large", param=param)
    chars = 0
    for n, m in enumerate(msgs):
        p = f"{param}[{n}]"
        if not isinstance(m, dict):
            raise _bad(p, "must be an object")
        if not isinstance(m.get("role"), str):
            raise _bad(p + ".role", "must be a string")
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for k, part in enumerate(c):
                if not isinstance(part, dict):
                    raise _bad(f"{p}.content[{k}]", "must be a content-part object")
                if isinstance(part.get("text"), str):
                    chars += len(part["text"])
        elif c is not None:
            raise _bad(p + ".content", "must be a string, a list of content parts or null")
        v_type(m, "name", str, "a string", p + ".")
        v_type(m, "reasoning_content", str, "a string", p + ".")
        calls = m.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                raise _bad(p + ".tool_calls", "must be a list")
            for k, tc in enumerate(calls):
                f = tc.get("function", tc) if isinstance(tc, dict) else None
                if not isinstance(f, dict) or not isinstance(f.get("name"), str) or \
                        not isinstance(f.get("arguments", ""), (str, dict)):
                    raise _bad(f"{p}.tool_calls[{k}]", "must be {function: {name: string, arguments: string|object}}")
                chars += len(f["arguments"]) if isinstance(f.get("arguments"), str) else 0
        if chars > LIM.prompt_chars:
            raise ApiError(413, f"messages hold over {LIM.prompt_chars} characters of text (INFERD_MAX_PROMPT_CHARS)",
                           "request_too_large", param=param)
    return msgs


def v_tools(d):
    tools = d.get("tools")
    if tools is not None and (not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools)):
        raise _bad("tools", "must be a list of tool objects")
    return tools


def v_effort_fields(d, where=""):
    v_type(d, "enable_thinking", bool, "true or false", where)
    v_type(d, "reasoning_effort", str, "a string", where)
    kw = v_type(d, "chat_template_kwargs", dict, "an object", where)
    if kw:
        v_type(kw, "enable_thinking", bool, "true or false", where + "chat_template_kwargs.")
        v_type(kw, "reasoning_effort", str, "a string", where + "chat_template_kwargs.")


def v_policy(b):
    """The per-agent fields, top level or under "inferd" (which wins); -> the merged policy."""
    ext = v_type(b, "inferd", dict, "an object")
    pol = {**b, **(ext or {})}
    if pol.get("context") is not None:
        v_name(pol["context"], "inferd.context")
    if pol.get("effort") is not None:
        v_choice(pol, "effort", EFFORTS, "inferd.")
    if pol.get("session") is not None:
        v_session(pol["session"], "inferd.session")
    v_choice(pol, "cache", ("keep", "drop"), "inferd.")
    v_int(pol, "max_context", 1, _I32, "inferd.")
    v_int(pol, "priority", -_I32, _I32, "inferd.")  # clamped to -10..10 below (the engine clamps too)
    v_int(pol, "json_retries", -_I32, _I32, "inferd.")  # range checked by structured_chat
    v_type(pol, "constrained", bool, "true or false", "inferd.")
    v_choice(pol, "spec_sampling", ("exact", "truncate"), "inferd.")
    return pol


def check_chat(b):
    """Validate a /v1/chat/completions body (in place; raises ApiError)."""
    b.pop("_grammar", None)  # internal (structured_chat -> complete): never from a client
    v_type(b, "model", str, "a string")
    v_messages(b.get("messages"))
    v_type(b, "stream", bool, "true or false")
    so = v_type(b, "stream_options", dict, "an object")
    if so:
        v_type(so, "include_usage", bool, "true or false", "stream_options.")
    for k in ("max_tokens", "max_completion_tokens"):
        v_int(b, k, 1, LIM.output_tokens)  # 0 is not "the default": OpenAI and vLLM reject it too
    v_num(b, "temperature", 0, 2)
    v_num(b, "top_p", 0, 1, lo_open=True)
    v_int(b, "top_k", -1, 1 << 20)  # -1 (vLLM) and 0: off
    v_int(b, "seed", -(1 << 63), (1 << 63) - 1)
    for k in ("ignore_eos", "include_stop_str_in_output"):
        v_type(b, k, bool, "true or false")
    v_effort_fields(b)
    v_tools(b)
    try:
        stops = parse_stop(b.get("stop"))
    except ValueError as e:
        raise _bad("stop", str(e))
    if len(stops) > 64 or any(len(s) > 4096 for s in stops):
        raise _bad("stop", "at most 64 strings of up to 4096 characters")
    v_type(b, "response_format", dict, "an object")
    v_policy(b)
    return b


def check_context_body(b):
    """Validate a POST /v1/contexts body (raises ApiError)."""
    for k in ("name", "id"):
        if b.get(k) is not None:
            v_name(b[k], k)
    if b.get("messages") is not None:
        v_messages(b["messages"])
    v_type(b, "system", str, "a string")
    docs = b.get("documents")
    if docs is not None and not isinstance(docs, str) and \
            not (isinstance(docs, list) and all(isinstance(d, str) for d in docs)):
        raise _bad("documents", "must be a string or a list of strings")
    if sum(map(len, [b.get("system") or ""] + ([docs] if isinstance(docs, str) else docs or []))) > LIM.prompt_chars:
        raise ApiError(413, f"over {LIM.prompt_chars} characters of text (INFERD_MAX_PROMPT_CHARS)", "request_too_large")
    v_tools(b)
    v_effort_fields(b)
    v_policy(b)
    v_choice(b, "persist", ("memory", "ssd"))
    v_num(b, "ttl", 0, 1e9)  # seconds; 0 or null: no expiry


# ---- Bounded preprocessing (Codex review R02): tokenize / render / vision run in worker threads, at most
# INFERD_MAX_PREPROCESS at a time; a request that waits longer than INFERD_PREPROCESS_WAIT gets a retryable 503.
_prep_sem = None


async def preprocess(fn, *a, **kw):
    global _prep_sem
    if _prep_sem is None:
        _prep_sem = asyncio.Semaphore(LIM.preprocess)
    sem = _prep_sem
    try:
        await asyncio.wait_for(sem.acquire(), LIM.preprocess_wait)
    except asyncio.TimeoutError:
        raise ApiError(503, f"server busy: preprocessing queue full (INFERD_MAX_PREPROCESS={LIM.preprocess}); retry",
                       "server_overloaded", retry_after=LIM.retry_after)
    work = asyncio.ensure_future(asyncio.to_thread(fn, *a, **kw))
    work.add_done_callback(lambda _: sem.release())  # the slot is held until the thread really finishes
    return await asyncio.shield(work)


def effort_kwargs(effort):
    """'none' | 'low' | 'medium' | 'xhigh' -> chat_template_kwargs."""
    if effort not in EFFORTS:
        raise ApiError(400, f"effort must be one of {', '.join(EFFORTS)}", "invalid_effort")
    return {"enable_thinking": False} if effort == "none" else {"enable_thinking": True, "reasoning_effort": effort}


def body_effort(body, default=None):
    """The effort a request asks for: "inferd": {"effort"}, else chat_template_kwargs / top-level fields, else default."""
    pol = body.get("inferd") or {}
    if pol.get("effort") is not None:
        effort_kwargs(pol["effort"])
        return pol["effort"]
    kw = {**{k: body[k] for k in ("enable_thinking", "reasoning_effort") if k in body},
          **(body.get("chat_template_kwargs") or {})}
    if kw.get("enable_thinking") is False:
        return "none"
    if "reasoning_effort" in kw:
        return kw["reasoning_effort"]
    if "enable_thinking" in kw:
        return "xhigh"  # the template's default
    return default


def render_context(messages, tools, effort):
    """-> token ids of the context alone: the messages rendered as they will appear before any later turn."""
    msgs = _template_messages(messages) + [{"role": "user", "content": _SENT}]
    text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=False, tokenize=False,
                                   **effort_kwargs(effort))
    i = text.find(_SENT)
    cut = max(text.rfind(m, 0, i) for m in _TURN_MARKERS) if i >= 0 else -1
    if cut <= 0:
        raise ApiError(400, "cannot find where the context ends in the rendered template", "context_render")
    return encode_cached(text[:cut])


async def _engine_save(engine_name, tokens, persist):
    t0 = time.time()
    ev = await engine.call("gen", ctx=engine_name, tokens=tokens, pin=1, ssd=int(persist == "ssd"), max_tokens=0,
                           temperature=0, stop=[])
    if ev["done"] != "stop":
        raise ApiError(503 if "disconnected" in ev["done"] else 400, ev["done"], "context_save_failed")
    return dict(ev.get("ctx") or {}, seconds=round(time.time() - t0, 3))


# ---- Context lifecycle (Codex review R04). Every step that creates, replaces, expires, offloads or adds an effort
# variant to a context holds that context's lifecycle lock; delete unpublishes the record at once (marking it dead),
# then takes the lock to drop its engine handles. A record's engine handles carry its generation, so a replacement
# never shares names with the record it replaces. A variant save commits only if its record is still live: a save that
# completes for a dead record drops its own handle, and a save that fails or is cancelled drops it too (the engine may
# have registered it). Locks are reclaimed when nobody holds or waits for them.
class Stale(Exception):
    """The record was deleted, replaced or expired while waiting: start again from contexts[id]."""


@contextlib.asynccontextmanager
async def ctx_lock(cid):
    ent = _ctx_locks.get(cid)
    if ent is None:
        ent = _ctx_locks[cid] = [asyncio.Lock(), 0]
    ent[1] += 1
    try:
        async with ent[0]:
            yield
    finally:
        ent[1] -= 1
        if ent[1] == 0 and _ctx_locks.get(cid) is ent:
            del _ctx_locks[cid]


def _handles(rec):
    return {v["engine"] for v in rec["variants"].values()}


async def _drop_quiet(name):
    try:
        await engine.call("ctx_drop", ctx=name)
    except Exception:
        pass


async def _drop_all(names):
    for name in names:
        await _drop_quiet(name)


async def _drop_shielded(names):
    """Drop engine handles; a cancelled caller does not stop the drops (nothing may be left unreachable)."""
    if names:
        await asyncio.shield(engine.background(_drop_all(sorted(names))))


async def _save_variant(rec, effort):
    """Render and save `effort` for rec. The caller holds ctx_lock(rec id)."""
    tokens = await preprocess(render_context, rec["messages"], rec["tools"], effort)
    if rec["dead"]:
        raise Stale()
    same = next((x for x in rec["variants"].values() if x["tokens"] == tokens), None)
    if same is not None:  # this effort renders the same prefix (Qwen3.8: none and medium add no instructions)
        rec["variants"][effort] = dict(same, effort=effort)
        return rec["variants"][effort]
    name = f"{rec['id']}@{effort}#{rec['gen']}"
    try:
        info = await _engine_save(name, tokens, rec["persist"])
    except BaseException:  # failed or cancelled: the engine may still have registered the name
        engine.background(_drop_quiet(name))
        raise
    if rec["dead"]:  # deleted while saving: the deletion wins, the handle goes
        await _drop_shielded({name})
        raise Stale()
    if contexts.get(rec["id"]) is rec and \
            _ctx_used(rec["id"]) + _ctx_bytes(rec) + _TOKEN_BYTES * len(tokens) > LIM.context_bytes:
        await _drop_shielded({name})
        raise _ctx_over(f"context {rec['id']!r} effort {effort}")
    v = {"effort": effort, "engine": name, "tokens": tokens, "saved": info}
    rec["variants"][effort] = v
    return v


async def ensure_variant(rec, effort):
    """The context rendered and pinned at `effort` (computed on first use, once). Raises Stale if rec is not live."""
    v = rec["variants"].get(effort)
    if v is not None:
        return v
    async with ctx_lock(rec["id"]):
        if rec["dead"]:
            raise Stale()
        return rec["variants"].get(effort) or await _save_variant(rec, effort)


def _context_view(rec, stats=None):
    by = {c["name"]: c for c in (stats or {}).get("contexts", [])}
    out = {"id": rec["id"], "object": "context", "created": rec["created"], "effort": rec["effort"],
           "persist": rec["persist"], "tools": len(rec["tools"] or []), "messages": len(rec["messages"]),
           "tokens": len(rec["variants"][rec["effort"]]["tokens"]) if rec["effort"] in rec["variants"] else None,
           "ttl": rec["ttl"], "expires_at": round(rec["last_used"] + rec["ttl"], 1) if rec["ttl"] else None,
           "launches": rec["launches"], "variants": []}
    for e, v in rec["variants"].items():
        eng = by.get(v["engine"], {})
        out["variants"].append({"effort": e, "tokens": len(v["tokens"]), "kv_tokens": eng.get("tokens"),
                                "prefill_s": v["saved"].get("seconds"), "cached_at_save": v["saved"].get("cached"),
                                "pinned": eng.get("pinned"), "resident": {"gpu": eng.get("gpu"), "ssd": eng.get("ssd")},
                                "kv_mib": eng.get("kv_mib")})
    return out


async def _stats():
    ev = await engine.call("stats")
    return ev.get("stats") or {}


@app.post("/v1/contexts")
async def create_context(req: Request):
    try:
        b = await read_json(req)
        check_context_body(b)
        cid = b.get("name") or b.get("id") or f"ctx-{uuid.uuid4().hex[:12]}"
        messages = b.get("messages")
        if messages is None:  # a system prompt plus documents
            docs = b.get("documents") or []
            if isinstance(docs, str):
                docs = [docs]
            system = "\n\n".join([b.get("system") or ""] + docs).strip()
            if not system:
                raise ApiError(400, "give messages, or a system prompt and/or documents")
            messages = [{"role": "system", "content": system}]
        effort = body_effort(b, "none")
        effort_kwargs(effort)
        persist = b.get("persist") or "memory"
        ttl = float(b.get("ttl") or 0)
        if not engine.alive:
            raise ApiError(503, "engine disconnected", retry_after=LIM.retry_after)
        text_bytes = _ctx_text_bytes(messages, b.get("tools"))
        async with ctx_lock(cid):
            # Check and reserve with no await in between (R06): creates of distinct ids run concurrently.
            old = contexts.get(cid)
            if old is None and len(contexts) + len(_ctx_new) >= LIM.contexts:
                raise ApiError(400, f"{len(contexts)} contexts exist and {len(_ctx_new)} are being created "
                               "(INFERD_MAX_CONTEXTS); delete one first", "context_limit")
            if _ctx_used(cid) + text_bytes > LIM.context_bytes:
                raise _ctx_over(f"context {cid!r}")
            _ctx_pending[cid] = text_bytes
            if old is None:
                _ctx_new.add(cid)
            try:
                rec = {"id": cid, "gen": next(_ctx_gen), "dead": False, "messages": messages, "tools": b.get("tools"),
                       "effort": effort, "persist": persist, "ttl": ttl, "created": int(time.time()),
                       "last_used": time.time(), "launches": 0, "variants": {}, "text_bytes": text_bytes}
                try:
                    await _save_variant(rec, effort)
                except (TemplateError, ValueError, TypeError) as e:
                    raise ApiError(400, str(e), "context_render")
                old = contexts.get(cid)  # re-read: a delete may have unpublished it meanwhile (it drops its own handles)
                # The record's real size, against everything else retained (the old record it replaces excluded).
                if _ctx_used(cid) + _ctx_bytes(rec) > LIM.context_bytes:
                    await _drop_shielded(_handles(rec))
                    raise _ctx_over(f"context {cid!r}")
                if old is None and len(contexts) + len(_ctx_new - {cid}) >= LIM.contexts:
                    await _drop_shielded(_handles(rec))  # the record was deleted meanwhile and others filled up
                    raise ApiError(400, f"{len(contexts)} contexts exist (INFERD_MAX_CONTEXTS); delete one first",
                                   "context_limit")
                contexts[cid] = rec
            finally:  # published, failed or cancelled: the reservation goes
                _ctx_pending.pop(cid, None)
                _ctx_new.discard(cid)
            if old is not None:  # replaced: the old record's handles go (none are shared: names carry the generation)
                old["dead"] = True
                await _drop_shielded(_handles(old))
        return _context_view(rec, await _stats())
    except ApiError as e:
        return e.response()


@app.get("/v1/contexts")
async def list_contexts():
    stats = await _stats()
    return {"object": "list", "data": [_context_view(r, stats) for r in list(contexts.values())],
            "cache": {k: v for k, v in stats.items() if k != "contexts"}}


@app.get("/v1/contexts/{cid}")
async def get_context(cid: str):
    try:
        v_name(cid, "id")
    except ApiError as e:
        return e.response()
    if cid not in contexts:
        return ApiError(404, f"no context {cid!r}", "context_not_found").response()
    return _context_view(contexts[cid], await _stats())


@app.delete("/v1/contexts/{cid}")
async def delete_context(cid: str):
    try:
        v_name(cid, "id")
    except ApiError as e:
        return e.response()
    rec = contexts.pop(cid, None)
    if rec is None:
        return ApiError(404, f"no context {cid!r}", "context_not_found").response()
    rec["dead"] = True  # launches stop now; a save still running for it drops its own handle when it completes
    async with ctx_lock(cid):  # after any save or offload in progress
        await _drop_shielded(_handles(rec))
    return {"id": cid, "object": "context", "deleted": True}


@app.post("/v1/contexts/{cid}/offload")
async def offload_context(cid: str):
    """Write the context to the SSD tier and free its GPU memory (unpinned); the next launch restores it."""
    try:
        v_name(cid, "id")
    except ApiError as e:
        return e.response()
    async with ctx_lock(cid):
        rec = contexts.get(cid)
        if rec is None:
            return ApiError(404, f"no context {cid!r}", "context_not_found").response()
        for name in sorted(_handles(rec)):
            ev = await engine.call("ctx_offload", ctx=name)
            if ev["done"] != "ok":
                return ApiError(400, ev["done"], "offload_failed").response()
    return _context_view(rec, await _stats())


# ---- Sessions (Codex review R12): the front end tracks the session ids it has forwarded, least recently used first,
# and keeps at most INFERD_MAX_SESSIONS of them; the oldest idle one beyond that (and any idle longer than
# INFERD_SESSION_TTL, when set) is dropped from the engine, so a stream of one-off agent names cannot grow the
# engine's session table without bound. A session with a request in flight is never dropped.
_sessions = collections.OrderedDict()  # id -> [last used, requests in flight]


def session_begin(sid):
    ent = _sessions.pop(sid, None) or [0.0, 0]
    ent[0], ent[1] = time.time(), ent[1] + 1
    _sessions[sid] = ent
    if len(_sessions) > LIM.sessions:
        for old, e in list(_sessions.items()):
            if len(_sessions) <= LIM.sessions:
                break
            if e[1] == 0:
                del _sessions[old]
                engine.background(_session_drop_quiet(old))


def session_end(sid):
    ent = _sessions.get(sid)
    if ent is not None:
        ent[0], ent[1] = time.time(), max(0, ent[1] - 1)


async def _session_drop_quiet(sid):
    try:
        await engine.call("session_drop", session=sid)
    except Exception:
        pass


@app.delete("/v1/sessions/{sid}")
async def delete_session(sid: str):
    """Forget an agent session: its cached turns leave memory and the SSD tier (shared prefixes stay)."""
    try:
        v_session(sid, "id")
    except ApiError as e:
        return e.response()
    ent = _sessions.get(sid)
    if ent is not None and ent[1] == 0:
        del _sessions[sid]
    ev = await engine.call("session_drop", session=sid)
    return {"id": sid, "object": "session", "deleted": ev["done"] == "ok"}


@app.post("/v1/sessions/{sid}/offload")
async def offload_session(sid: str):
    """Park an idle agent: its latest turn goes to the SSD tier and leaves GPU memory; its next turn restores it."""
    try:
        v_session(sid, "id")
    except ApiError as e:
        return e.response()
    ev = await engine.call("session_offload", session=sid)
    if ev["done"] != "ok":
        return ApiError(400, ev["done"], "offload_failed").response()
    return {"id": sid, "object": "session", "offloaded": True}


@app.get("/v1/cache")
async def cache_stats():
    """KV pool, snapshot slots, pinned contexts, sessions and the SSD tier, in engine units (blocks of 32 tokens)."""
    return await _stats()


async def apply_context(body):
    """Resolve "inferd": {"context", "effort"}: -> (body to render, the context's token ids or None)."""
    pol = body.get("inferd") or {}
    cid = pol.get("context")
    if cid is None:
        if pol.get("effort") is not None:  # effort alone, on any request
            kw = {**(body.get("chat_template_kwargs") or {}), **effort_kwargs(pol["effort"])}
            body = {**body, "chat_template_kwargs": kw}
        return body, None
    for _ in range(4):  # a context replaced while this launch waited for it: launch from the live record
        rec = contexts.get(cid)
        if rec is None:
            raise ApiError(404, f"no context {cid!r}", "context_not_found")
        effort = body_effort(body, rec["effort"])
        tools = body.get("tools")
        if tools and rec["tools"] and tools != rec["tools"]:
            raise ApiError(400, "the request's tools differ from the context's (tools are part of the cached prefix)")
        if tools and not rec["tools"]:
            raise ApiError(400, "this context was saved without tools; save the tools with the context")
        if any(m.get("role") == "system" for m in body.get("messages", [])) and \
                any(m.get("role") == "system" for m in rec["messages"]):
            raise ApiError(400, "the context has the system message; leave it out of the request")
        try:
            v = await ensure_variant(rec, effort)
            break
        except Stale:
            continue
    else:
        raise ApiError(409, f"context {cid!r} kept changing while launching; retry", "context_busy",
                       retry_after=LIM.retry_after)
    rec["launches"] += 1
    rec["last_used"] = time.time()
    kw = {**(body.get("chat_template_kwargs") or {}), **effort_kwargs(effort)}
    body = {**body, "messages": list(rec["messages"]) + list(body.get("messages") or []), "tools": rec["tools"],
            "chat_template_kwargs": kw}
    return body, v["tokens"]


async def _expire_once(now=None):
    now = time.time() if now is None else now
    for cid, rec in list(contexts.items()):
        if not (rec["ttl"] and now - rec["last_used"] > rec["ttl"]) or cid in _ctx_locks:
            continue  # live, or busy (a save, offload or delete in progress: not idle)
        async with ctx_lock(cid):
            if contexts.get(cid) is rec and now - rec["last_used"] > rec["ttl"]:
                contexts.pop(cid, None)
                rec["dead"] = True
                await _drop_shielded(_handles(rec))
    if LIM.session_ttl:
        for sid, (last, busy) in list(_sessions.items()):
            if not busy and now - last > LIM.session_ttl:
                del _sessions[sid]
                await _session_drop_quiet(sid)


async def _expire_loop():
    while True:
        await asyncio.sleep(5)
        try:
            await _expire_once()
        except Exception as e:  # never let the expiry task die
            print(f"serve: expiry: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


@app.post("/v1/chat/completions")
async def chat(req: Request):
    _client_receive.set(req.receive)  # complete() watches it for http.disconnect while it waits (R07)
    try:
        body = check_chat(await read_json(req))  # one validation boundary (Codex review R03)
    except ApiError as e:
        return e.response()
    if body.get("response_format") is not None:
        return await structured_chat(body)
    return await complete(body)


class _Stream(StreamingResponse):
    """An SSE response whose cleanup also runs when the stream never started or was cut off (the client went away
    before or during it): the request's queue, session and engine work never outlive the response."""

    def __init__(self, gen, cleanup):
        super().__init__(gen, media_type="text/event-stream")
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._cleanup(False)


def _bad_request(e):
    return JSONResponse({"error": {"message": str(e)[:1000], "type": "invalid_request_error"}}, status_code=400)


async def complete(body):
    """One chat completion for a validated request body (check_chat) -> JSONResponse or StreamingResponse."""
    try:
        body, ctx_tokens = await apply_context(body)  # "inferd": {"context", "effort"} (agent contexts)
        prompt, thinking = await preprocess(render, body)  # bounded: INFERD_MAX_PREPROCESS
    except ApiError as e:
        return e.response()
    except (ValueError, TemplateError, TypeError, KeyError) as e:  # the chat template rejects the messages / effort
        return _bad_request(e)
    n_prompt = len(prompt.get("tokens", []))
    if n_prompt > LIM.prompt_tokens:
        return ApiError(400, f"prompt is {n_prompt} tokens; the limit is {LIM.prompt_tokens} "
                        "(INFERD_MAX_PROMPT_TOKENS)", "context_length_exceeded").response()
    if ctx_tokens is not None and prompt.get("tokens", [])[:len(ctx_tokens)] != ctx_tokens:
        return ApiError(400, "the rendered request does not extend the context token for token (chat template)",
                        "context_prefix_mismatch").response()
    stops = parse_stop(body.get("stop"))  # validated by check_chat
    keep_stop = bool(body.get("include_stop_str_in_output"))
    if not engine.alive:
        return ApiError(503, "engine disconnected", retry_after=LIM.retry_after).response()
    rid = next(ids)

    def given(*keys, default):  # the first field present and not null; 0 is a value, never "absent"
        return next((body[k] for k in keys if body.get(k) is not None), default)
    top_k = int(given("top_k", default=DEFAULT_TOP_K))
    gen = {**prompt, "max_tokens": int(given("max_tokens", "max_completion_tokens", default=4096)),
           "temperature": float(given("temperature", default=0.7)), "top_p": float(given("top_p", default=DEFAULT_TOP_P)),
           "top_k": max(0, top_k), "stop": [] if body.get("ignore_eos") else STOP}  # ignore_eos: vLLM's field
    if body.get("seed") is not None:
        gen["seed"] = int(body["seed"])
    # Sampling contract (Codex review 2026-11-04, R03; src/sampling.h): top_k 0 / -1 is no limit, over the whole
    # vocabulary, on every path. The speculative sampler sees 64 candidates per row: exact for top_k 1..64 and, with
    # top_k off, for top_p < 1 (a row whose nucleus does not fit is sampled by the engine's host sampler: "spec_host").
    # top_k above 64, or off with top_p 1, decodes one token per step unless the request opts into truncation to the
    # 64 candidates ("inferd": {"spec_sampling": "truncate"}). The effective settings come back in
    # "inferd": {"sampling": ...}.
    if ({**body, **(body.get("inferd") or {})}).get("spec_sampling") == "truncate":
        gen["spec_truncate"] = 1

    def sampling_info(ev):
        t, k, p = gen["temperature"], gen["top_k"], gen["top_p"]
        info = {"temperature": t, "top_p": p, "top_k": k or None}  # None: no limit
        if "spec" in ev:
            info["speculative"] = bool(ev["spec"])
            if ev["spec"] and t > 0 and (k > SPEC_CANDIDATES or (k == 0 and p >= 1)):
                info["speculative_top_k"] = SPEC_CANDIDATES  # the opt-in truncation applied on speculative steps
            if ev.get("spec_host"):
                info["host_sampled_rows"] = ev["spec_host"]  # nucleus beyond the GPU's candidates: sampled exactly
        return info
    if body.get("_grammar") is not None:  # structured output, constrained: the engine samples under the schema
        gen["json_schema"] = body["_grammar"]
        if thinking:  # the reasoning is free; the grammar starts after </think>
            gen["think_end"] = tok.convert_tokens_to_ids(THINK_END)
    # Per-agent policy (extra request fields; also accepted under "inferd": {...}): max_context caps this agent's
    # prompt + output tokens, priority protects it from preemption when memory runs short (higher = kept longer).
    policy = {**body, **(body.get("inferd") or {})}
    # Cache lifetime: "session" names the agent (its latest turn is kept longest, and goes to SSD before it leaves
    # memory); "cache": "drop" keeps nothing of this request (one-off agents: memory stays flat).
    no_cache = policy.get("cache") == "drop"
    if no_cache:
        gen["no_cache"] = 1
    session = policy.get("session")  # validated: 1..128 characters, never truncated
    if session is not None:
        gen["session"] = session
    if not thinking and not no_cache:  # the next turn re-renders this answer token for token: snapshot while decoding
        gen["gen_snap"] = 1
    for k in ("max_context", "priority"):
        if policy.get(k) is not None:
            gen[k] = int(policy[k])
    if "priority" in gen:
        gen["priority"] = max(-10, min(10, gen["priority"]))  # a bounded scale; the engine clamps too
    if gen.get("max_context") and n_prompt >= gen["max_context"]:
        return JSONResponse({"error": {"message": f"prompt is {n_prompt} tokens, max_context is "
                                       f"{gen['max_context']}", "type": "invalid_request_error",
                                       "code": "context_length_exceeded"}}, status_code=400)
    if session is not None:
        session_begin(session)
    try:
        q = await engine.submit(rid, gen)
    except BaseException as e:
        if session is not None:
            session_end(session)
        if isinstance(e, (ConnectionError, OSError, RuntimeError, AttributeError)):
            return ApiError(503, "engine disconnected", retry_after=LIM.retry_after).response()
        raise
    over = False

    def end_request(ended):
        """Once per request, whichever path ends it: unregister, close the session, cancel unfinished engine work."""
        nonlocal over
        if over:
            return
        over = True
        engine.release(rid)
        if session is not None:
            session_end(session)
        if not ended:
            engine.background(engine.cancel_quiet(rid))
    cid, created, model = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time()), body.get("model", args.name)

    def split(state, piece):
        """Route text to reasoning_content until </think>, then to content."""
        if not state["reasoning"]:
            return None, piece
        state["buf"] += piece
        if THINK_END in state["buf"]:
            r, c = state["buf"].split(THINK_END, 1)
            state["reasoning"], state["buf"] = False, ""
            return r, c.lstrip("\n")
        keep = len(THINK_END) - 1  # a partial </think> may be straddling pieces
        r, state["buf"] = state["buf"][:-keep] if len(state["buf"]) > keep else "", state["buf"][-keep:]
        return r, None

    def usage(ev):
        return {"prompt_tokens": ev["p"], "completion_tokens": ev["c"], "total_tokens": ev["p"] + ev["c"],
                "prompt_tokens_details": {"cached_tokens": ev["cached"]}}

    finish_map = {"stop": "stop", "length": "length"}
    tools = body.get("tools")

    def tool_hold(state, c):
        """With tools: stream content up to a <tool_call>, hold everything from there for the final delta."""
        if state["tool"] is not None:
            state["tool"] += c
            return ""
        s = state["pend"] + c
        i = s.find(TOOL_OPEN)
        if i >= 0:
            state["tool"], state["pend"] = s[i:], ""
            return s[:i]
        k = next((n for n in range(min(len(TOOL_OPEN) - 1, len(s)), 0, -1) if TOOL_OPEN.startswith(s[-n:])), 0)
        state["pend"] = s[len(s) - k:] if k else ""  # a partial <tool_call> may be straddling pieces
        return s[:len(s) - k]

    if body.get("stream"):
        async def events():
            dt, state = Detok(), {"reasoning": thinking, "buf": "", "tool": None, "pend": ""}
            sm, used = (StopMatcher(stops, keep_stop) if stops else None), 0

            def chunk(delta, finish=None, **extra):
                return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                              "model": model, "choices": [{"index": 0, "delta": delta,
                                                                           "finish_reason": finish}], **extra}) + "\n\n"
            try:
                while True:
                    ev = await next_event(q, rid)
                    if "t" in ev:
                        used += len(ev["t"])
                        piece = "".join(dt.add(t) for t in ev["t"])
                        r, c = split(state, piece)
                        hit = False
                        if c and sm:
                            c, hit = sm.feed(c)
                        delta = {}
                        if r:
                            delta["reasoning_content"] = r
                        if c and tools:
                            c = tool_hold(state, c)
                        if c:
                            delta["content"] = c
                        if delta:
                            yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                                         "model": model, "choices": [{"index": 0, "delta": delta,
                                                                                      "finish_reason": None}]}) + "\n\n"
                        if not hit:
                            continue
                        await engine.cancel_quiet(rid)  # a stop string: end the engine's work, finish as "stop"
                        ev = dict(await _drain_done(q), done="stop", c=used)
                    elif str(ev["done"]).startswith("error") or ev["done"] == "cancelled":
                        # an engine failure is an error event, never a normal finish (Codex review R04)
                        yield "data: " + json.dumps({"error": {"message": ev["done"], "type": "server_error",
                                                               "code": "engine_error"}}) + "\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    finish = finish_map.get(ev["done"], ev["done"])
                    if not (sm and sm.done):  # flush held text: a partial UTF-8 tail and the </think> look-behind
                        r, c = drain_split(split, state, dt.flush())
                        if c and sm:
                            c, hit = sm.feed(c)
                            finish = "stop" if hit else finish
                        if c and tools:
                            c = tool_hold(state, c)
                        if r or c:
                            yield chunk(dict(({"reasoning_content": r} if r else {}), **({"content": c} if c else {})))
                    if sm and not sm.done and (tail := sm.flush()):  # text held back as a possible stop prefix
                        tail = tool_hold(state, tail) if tools else tail
                        if tail:
                            yield chunk({"content": tail})
                    if state["pend"]:
                        yield chunk({"content": state["pend"]})
                    if state["tool"] is not None:  # tool calls go out whole, as one final delta
                        _, calls = parse_tool_calls(state["tool"], tools)
                        if calls:
                            yield chunk({"tool_calls": [dict(c, index=n) for n, c in enumerate(calls)]})
                            finish = "tool_calls" if finish == "stop" else finish
                        else:
                            yield chunk({"content": state["tool"]})
                    yield chunk({}, finish, inferd={"sampling": sampling_info(ev)})
                    if (body.get("stream_options") or {}).get("include_usage"):
                        yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                                     "model": model, "choices": [], "usage": usage(ev)}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return
            finally:
                end_request("ev" in locals() and "t" not in ev and not ev.get("_local"))
        return _Stream(events(), end_request)

    out = []
    # Stop strings: watch the answer text as it arrives (the same detokenizer and reasoning split as the stream) and
    # cancel at the first match; n_used = the tokens up to the one that completed it.
    sm, sdt, sst, n_used = (StopMatcher(stops) if stops else None), Detok(), {"reasoning": thinking, "buf": ""}, None
    ended = False
    recv = _client_receive.get()
    watch = engine.background(_watch_disconnect(recv, q)) if recv is not None else None
    try:
        while True:
            ev = await next_event(q, rid)
            if "t" in ev:
                out += ev["t"]
                if sm and n_used is None:
                    _, c = split(sst, "".join(sdt.add(t) for t in ev["t"]))
                    if c and sm.feed(c)[1]:
                        n_used = len(out)
                        await engine.cancel_quiet(rid)
            else:
                ended = not ev.get("_local")  # a timeout or a disconnect: the engine has not ended it
                break
    finally:  # the client went away (CancelledError, http.disconnect) or anything else: the engine work stops too
        if watch is not None:
            watch.cancel()
        end_request(ended)
    if ev.get("_gone"):  # nobody is listening; 499 (nginx's "client closed request") for the logs
        return JSONResponse(_err("client disconnected", "invalid_request_error", "client_disconnected"), status_code=499)
    if n_used is not None:  # ended by a stop string (the engine's own end, if it raced the cancel, is moot)
        out, ev = out[:n_used], dict(ev, done="stop", c=n_used)
    if str(ev["done"]).startswith("error"):
        code = 504 if ev.get("_timeout") else 503 if ("disconnected" in ev["done"] or "too slow" in ev["done"]) else 400 if ("too long" in ev["done"] or "max_context" in ev["done"]
                                                                or "json_schema" in ev["done"]
                                                                or "bad request" in ev["done"]) else 500
        return JSONResponse({"error": {"message": ev["done"], "type": "server_error" if code != 400 else "invalid_request_error",
                                       "code": "engine_error"}}, status_code=code)
    text = tok.decode(out, skip_special_tokens=False)
    reasoning = None
    if thinking and THINK_END in text:
        reasoning, text = text.split(THINK_END, 1)
        text = text.lstrip("\n")
    elif thinking:  # cut off inside the reasoning (max_tokens): it is all reasoning, as the stream reports it
        reasoning, text = text, ""
    if stops:
        text = cut_at_stop(text, stops, keep_stop)[0]
    finish = finish_map.get(ev["done"], ev["done"])
    calls = []
    if tools:
        text, calls = parse_tool_calls(text, tools)
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = calls
        finish = "tool_calls" if finish == "stop" else finish
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": model,
                         "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                         "usage": usage(ev), "inferd": {"sampling": sampling_info(ev)}})


# Structured output (response_format): the instruction goes on the last user turn, the answer's JSON is extracted and
# validated (structured.py), and an invalid answer is retried with the validation error as a new user turn (default 2
# retries; "json_retries" in the request or under "inferd"). Streaming buffers the answer and sends it once it is valid.
# After the last failed attempt the reply is HTTP 422, code "json_validation_failed", with the last output and error.
async def structured_chat(body):
    try:
        spec = structured.parse_response_format(body["response_format"])
        policy = {**body, **(body.get("inferd") or {})}
        retries = int(policy.get("json_retries", structured.DEFAULT_RETRIES))
        if not 0 <= retries <= structured.MAX_RETRIES:
            raise ValueError(f"json_retries must be 0..{structured.MAX_RETRIES}")
    except (ValueError, TypeError) as e:
        return JSONResponse({"error": {"message": str(e), "type": "invalid_request_error",
                                       "param": "response_format"}}, status_code=400)
    plain = {k: v for k, v in body.items() if k not in ("response_format", "stream", "stream_options", "_grammar")}
    # strict: true -> constrained decoding in the engine (src/grammar.h), validated afterwards all the same (the grammar
    # loosens anyOf/oneOf, numeric bounds, uniqueItems...); "constrained" in the request overrides either way.
    constrained = spec is not None and bool(policy.get("constrained", spec["strict"]))
    if constrained:
        plain["_grammar"] = spec["schema"]
    if spec is None:  # {"type": "text"}
        return await complete({k: v for k, v in body.items() if k not in ("response_format", "_grammar")})
    msgs = structured.add_instruction(body["messages"], structured.instruction(spec))
    tot, t0, err, content = {"p": 0, "c": 0, "cached": 0}, time.time(), None, ""
    for attempt in range(1, retries + 2):
        resp = await complete({**plain, "messages": msgs})
        if resp.status_code != 200:
            return resp
        d = json.loads(resp.body)
        u, choice = d["usage"], d["choices"][0]
        tot["p"] += u["prompt_tokens"]
        tot["c"] += u["completion_tokens"]
        tot["cached"] += u["prompt_tokens_details"]["cached_tokens"]
        msg = choice["message"]
        if msg.get("tool_calls"):  # the model chose a tool: that is the answer
            break
        content = msg.get("content") or ""
        out, err = await asyncio.to_thread(structured.check, content, spec)  # bounded work, off the event loop
        if out is not None:
            msg["content"], choice["finish_reason"] = out, "stop"
            break
        msgs = msgs + [{"role": "assistant", "content": content}, {"role": "user",
                                                                   "content": structured.retry_message(err)}]
    else:
        return JSONResponse({"error": {"message": f"no schema-valid JSON after {attempt} attempt(s): {err}",
                                       "type": "invalid_response_error", "code": "json_validation_failed",
                                       "attempts": attempt, "validation_error": err, "last_output": content}},
                            status_code=422)
    d["usage"] = {"prompt_tokens": tot["p"], "completion_tokens": tot["c"], "total_tokens": tot["p"] + tot["c"],
                  "prompt_tokens_details": {"cached_tokens": tot["cached"]}}
    d["inferd"] = {**(d.get("inferd") or {}), "json_attempts": attempt, "json_seconds": round(time.time() - t0, 3), "json_constrained": constrained}
    if constrained and spec.get("checked_only"):  # keywords validated after decoding, not enforced while decoding
        d["inferd"]["json_checked_only"] = spec["checked_only"]
    if not body.get("stream"):
        return JSONResponse(d)
    msg, finish = d["choices"][0]["message"], d["choices"][0]["finish_reason"]

    def chunk(delta, fin=None):
        return "data: " + json.dumps({"id": d["id"], "object": "chat.completion.chunk", "created": d["created"],
                                      "model": d["model"], "choices": [{"index": 0, "delta": delta,
                                                                        "finish_reason": fin}]}) + "\n\n"

    async def events():
        yield chunk({"role": "assistant"})
        if msg.get("reasoning_content"):
            yield chunk({"reasoning_content": msg["reasoning_content"]})
        if msg.get("content"):
            yield chunk({"content": msg["content"]})
        if msg.get("tool_calls"):
            yield chunk({"tool_calls": [dict(c, index=n) for n, c in enumerate(msg["tool_calls"])]})
        yield chunk({}, finish)
        if (body.get("stream_options") or {}).get("include_usage"):
            yield "data: " + json.dumps({"id": d["id"], "object": "chat.completion.chunk", "created": d["created"],
                                         "model": d["model"], "choices": [], "usage": d["usage"]}) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(Guard(app), host=args.host, port=args.port, log_level="warning")
