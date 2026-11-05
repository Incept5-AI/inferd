# OpenAI-compatible HTTP front end for inferd (the native engine).
# Handles chat templating and tokenization, streams tokens back as SSE; inferd does all model work.
#   python serve.py --model <dir> --socket <run dir>/inferd.sock --port 8300 [--name qwen3.8-27b] (run.sh starts both)
import argparse, array, asyncio, base64, collections, contextlib, glob, hmac, ipaddress, itertools, json, math, os, re
import contextvars, stat, sys, threading, time, uuid

from jinja2 import TemplateError

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
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
ap.add_argument("--name", default="qwen3.8-27b",
                help="served model name(s), comma-separated (vLLM's --served-model-name): the first is the id")
ap.add_argument("--max-model-len", type=int, default=int(os.environ.get("INFERD_MAX_MODEL_LEN") or 0),
                help="the engine's context cap (run.sh: INFERD_MAX_MODEL_LEN); 0: unknown")
args = ap.parse_args()
SERVED = [n.strip() for n in args.name.split(",") if n.strip()] or ["qwen3.8-27b"]
args.name = SERVED[0]
VERSION = "0.3.0-dev"  # inferd; GET /version
STARTED = int(time.time())


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
@contextlib.asynccontextmanager
async def _lifespan(_app):  # FastAPI's lifespan (on_event("startup") is deprecated); engine and _expire_loop are below
    await engine.connect()
    asyncio.create_task(_expire_loop())
    yield


app = FastAPI(lifespan=_lifespan)
ids = itertools.count(1)

# Vision: only for checkpoints that carry the vision tower's weights (older mlx-community 27B 4-bit snapshots have none).
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


@app.exception_handler(Exception)
async def _unexpected(req, e):  # a bug, not the client's fault: still an OpenAI-style error body, never a traceback
    print(f"serve: {req.url.path}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    return JSONResponse(_err("internal error", "server_error", "internal_error"), status_code=500)


def _model_card(name):
    return {"id": name, "object": "model", "created": STARTED, "owned_by": "inferd",
            "root": os.path.basename(os.path.normpath(args.model)), "parent": None,
            "max_model_len": args.max_model_len or None, "permission": []}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [_model_card(n) for n in SERVED]}


@app.get("/v1/models/{name:path}")
async def model_card(name: str):
    if name not in SERVED:
        return ApiError(404, f"The model `{name}` does not exist.", "model_not_found", param="model").response()
    return _model_card(name)


@app.get("/health")
async def health():
    if not engine.alive:
        return JSONResponse({"ok": False, "error": "engine disconnected"}, status_code=503)
    return {"ok": True}


@app.api_route("/ping", methods=["GET", "POST"])
async def ping():  # vLLM / SageMaker liveness: 200 while the engine is connected
    if not engine.alive:
        return JSONResponse({"ok": False, "error": "engine disconnected"}, status_code=503)
    return {"ok": True}


@app.get("/version")
async def version():
    return {"version": VERSION, "engine": "inferd"}


# Endpoints vLLM serves that inferd does not (pooling, scoring, audio, LoRA, the Responses API): a clean 501 with an
# OpenAI-style error, never a 404 HTML page or a hang (docs/API-COMPAT.md).
_UNSUPPORTED = {
    "/v1/embeddings": "embeddings", "/pooling": "pooling", "/classify": "classification", "/v1/classify": "classification",
    "/score": "scoring", "/v1/score": "scoring", "/rerank": "reranking", "/v1/rerank": "reranking",
    "/v2/rerank": "reranking", "/v1/audio/transcriptions": "audio", "/v1/audio/translations": "audio",
    "/v1/audio/speech": "audio", "/v1/load_lora_adapter": "LoRA adapters", "/v1/unload_lora_adapter": "LoRA adapters",
    "/v1/responses": "the Responses API", "/v1/images/generations": "image generation"}


def _unsupported_route(path, what):
    async def handler():
        return JSONResponse(_err(f"{path}: {what} is not supported by inferd (a generative chat model server); see "
                                 "docs/API-COMPAT.md", "invalid_request_error", "unsupported_endpoint"),
                            status_code=501)
    return handler


for _p, _w in _UNSUPPORTED.items():
    app.add_api_route(_p, _unsupported_route(_p, _w), methods=["GET", "POST"], include_in_schema=False)


@app.exception_handler(StarletteHTTPException)
async def _http_error(req, e):  # unknown routes, wrong methods: OpenAI-style JSON, not FastAPI's {"detail": ...}
    code = {404: "not_found", 405: "method_not_allowed"}.get(e.status_code, "http_error")
    msg = f"{req.method} {req.url.path}: {e.detail}"
    return JSONResponse(_err(msg, "invalid_request_error" if e.status_code < 500 else "server_error", code),
                        status_code=e.status_code)


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
        self.stops, self.keep, self.buf, self.done, self.hit = list(stops), keep, "", False, None

    def feed(self, piece):
        if self.done:
            return "", True
        s = self.buf + piece
        h = _first_stop(s, self.stops)
        if h is not None:
            self.done, self.buf, self.hit = True, "", h[1]
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
    """q: a request's queue, or a list of them (n choices)."""
    while True:
        try:
            msg = await receive()
        except asyncio.CancelledError:
            raise
        except Exception:  # the server's receive channel failed: the client is as good as gone
            msg = {"type": "http.disconnect"}
        if msg.get("type") == "http.disconnect":
            for x in (q if isinstance(q, list) else [q]):
                x.put_nowait(dict(_GONE))
            return
        await asyncio.sleep(0.05)  # not a disconnect (nothing else is expected once the body is read): never spin


def _template_messages(messages):
    """OpenAI clients send tool_calls[].function.arguments as a JSON string; the template iterates a mapping. Content
    parts in the other shapes vLLM accepts become the template's own: input_text / output_text / refusal -> text,
    input_image (a URL string) and image_url given as a bare string -> image_url {url}."""
    out = []
    for m in messages:
        if m.get("role") == "developer":  # OpenAI's newer name for the system role (templates know "system")
            m = {**m, "role": "system"}
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                t = p.get("type")
                if t in ("input_text", "output_text", "refusal"):
                    p = {"type": "text", "text": p.get("text", p.get("refusal", ""))}
                elif t == "input_image" or (t == "image_url" and isinstance(p.get("image_url"), str)):
                    u = p.get("image_url")
                    p = {"type": "image_url", "image_url": {"url": u.get("url") if isinstance(u, dict) else u}}
                parts.append(p)
            m = {**m, "content": parts}
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
    force = body.get("_force")  # tool_choice required / named: the call's opening markup, thinking off
    if force:
        kw["enable_thinking"] = False
    thinking = kw.get("enable_thinking", True)
    # vLLM's template fields: continue_final_message (the reply continues the last assistant message),
    # add_generation_prompt (default true), documents, chat_template (refused unless trusted, check_chat).
    cont = bool(body.get("continue_final_message"))
    agp = not cont and body.get("add_generation_prompt") is not False
    extra = {k: body[k] for k in ("documents", "chat_template") if body.get(k) is not None}
    if cont:
        extra["continue_final_message"] = True
    if cont or not agp:
        thinking = False  # the output continues a message as written: no reasoning split
    if _has_images(body["messages"]):
        if extra or not agp or force:
            raise ValueError("continue_final_message, add_generation_prompt: false, documents, chat_template and a "
                             "forced tool_choice are not supported with images")
        with _vision_lock:
            tokens, emb, mrope, delta = vision().prepare(body["messages"], tools=tools, **kw)
        return {"tokens": tokens, "mrope": mrope, "rope_delta": delta,
                "embeds": base64.b64encode(emb).decode()}, thinking
    text = tok.apply_chat_template(body["messages"], tools=tools, add_generation_prompt=agp, tokenize=False,
                                   **extra, **kw)
    if force:
        text += force
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
                if part.get("type") not in _PART_TYPES:
                    raise _bad(f"{p}.content[{k}].type", f"content parts of type {str(part.get('type'))[:40]!r} are "
                               "not supported (text, image_url and their input_* forms are)", "unsupported_content")
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


_PART_TYPES = ("text", "input_text", "output_text", "refusal", "image_url", "input_image", "image")


def v_tools(d):
    tools = d.get("tools")
    if tools is not None and (not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools)):
        raise _bad("tools", "must be a list of tool objects")
    return tools


MAX_N = _env_num("INFERD_MAX_N", 16, 1, 256)  # choices per request (n)
MAX_LOGPROBS = 20                             # top logprobs per token (vLLM's default max_logprobs)
STRICT_MODEL = os.environ.get("INFERD_STRICT_MODEL") == "1"  # unknown model names: 404, as vLLM (default: accepted)
TRUST_TEMPLATE = os.environ.get("INFERD_TRUST_REQUEST_CHAT_TEMPLATE") == "1"  # vLLM's --trust-request-chat-template
# vLLM's structured-output fields (guided_* before 0.10, structured_outputs since): -> one response_format-style spec.
_GUIDED = ("guided_json", "guided_regex", "guided_choice", "guided_grammar", "guided_decoding_backend",
           "guided_whitespace_pattern", "structured_outputs")


def v_sampling_extras(b, chat):
    """vLLM's request fields shared by chat and completions (docs/API-COMPAT.md); raises ApiError."""
    if STRICT_MODEL and b.get("model") is not None and b["model"] not in SERVED:
        raise ApiError(404, f"The model `{b['model']}` does not exist.", "model_not_found", param="model")
    n = v_int(b, "n", 1, MAX_N)
    best = v_int(b, "best_of", 1, MAX_N)
    if best is not None and best != (n or 1):
        raise _bad("best_of", "best_of different from n is not supported (as in vLLM V1); leave it out", "unsupported")
    v_num(b, "min_p", 0, 1)
    for k in ("presence_penalty", "frequency_penalty"):
        v_num(b, k, -2, 2)
    v_num(b, "repetition_penalty", 0, 100, lo_open=True)
    v_int(b, "min_tokens", 0, LIM.output_tokens)
    lb = v_type(b, "logit_bias", dict, "an object of token id -> bias")
    if lb:
        if len(lb) > 1024:
            raise _bad("logit_bias", "at most 1024 entries")
        for k, v in lb.items():
            if not (isinstance(k, str) and k.lstrip("-").isdigit()) or isinstance(v, bool) or \
                    not isinstance(v, (int, float)) or not math.isfinite(v) or not -100 <= v <= 100:
                raise _bad("logit_bias", "keys are token ids, values numbers in [-100, 100]")
    sti = v_type(b, "stop_token_ids", list, "a list of token ids")
    if sti is not None and (len(sti) > 64 or not all(isinstance(t, int) and not isinstance(t, bool) and t >= 0
                                                       for t in sti)):
        raise _bad("stop_token_ids", "at most 64 non-negative integers")
    for k in ("ignore_eos", "include_stop_str_in_output", "skip_special_tokens", "spaces_between_special_tokens",
              "echo", "add_special_tokens", "return_tokens_as_token_ids"):
        v_type(b, k, bool, "true or false")
    v_type(b, "user", str, "a string")
    v_type(b, "metadata", dict, "an object")
    v_type(b, "request_id", str, "a string")
    if b.get("request_id") is not None and (len(b["request_id"]) > 128 or any(ord(c) < 32 for c in b["request_id"])):
        raise _bad("request_id", "at most 128 characters, no control characters")
    v_int(b, "truncate_prompt_tokens", -1, LIM.prompt_tokens)
    if b.get("truncate_prompt_tokens") == 0:
        raise _bad("truncate_prompt_tokens", "must be -1 (the model's limit) or a positive number of tokens")
    so = b.get("stream_options") or {}
    v_type(so, "continuous_usage_stats", bool, "true or false", "stream_options.")
    if b.get("prompt_logprobs") is not None:
        raise _bad("prompt_logprobs", "prompt logprobs are not supported (inferd keeps no logits for prompt positions)",
                   "unsupported")
    if b.get("cache_salt") is not None:
        raise _bad("cache_salt", "not supported: inferd's prefix cache is shared by every request (one trusted user; "
                   "README, Security and limits); use \"cache\": \"drop\" to keep a request out of it", "unsupported")
    if b.get("lora_request") is not None or b.get("lora") is not None:
        raise _bad("lora_request", "LoRA adapters are not supported", "unsupported")
    if chat:
        v_type(b, "logprobs", bool, "true or false")
        tl = v_int(b, "top_logprobs", 0, MAX_LOGPROBS)
        if tl is not None and not b.get("logprobs"):
            raise _bad("top_logprobs", "needs logprobs: true")
    else:
        v_int(b, "logprobs", 0, MAX_LOGPROBS)
    return b


def v_guided(b):
    """-> a response_format equivalent of vLLM's guided_* / structured_outputs fields, or None. guided_choice and
    guided_regex become a JSON string schema (enum / anchored pattern) decoded under the grammar and unquoted."""
    so = v_type(b, "structured_outputs", dict, "an object") or {}
    g = {k[len("guided_"):]: b[k] for k in ("guided_json", "guided_regex", "guided_choice", "guided_grammar")
         if b.get(k) is not None}
    g.update({k: v for k, v in so.items() if v is not None and k in ("json", "regex", "choice", "grammar",
                                                                       "json_object", "structural_tag")})
    bad = [k for k, v in so.items() if v is not None and k not in ("json", "regex", "choice", "grammar", "json_object",
                                                                   "structural_tag", "backend", "whitespace_pattern",
                                                                   "disable_fallback", "disable_any_whitespace",
                                                                   "disable_additional_properties")]
    if bad:
        raise _bad("structured_outputs", f"unknown field(s): {', '.join(map(str, bad))[:200]}")
    if not g:
        return None
    if len(g) > 1:
        raise _bad("structured_outputs", "give one of json, regex, choice, grammar")
    if b.get("response_format") is not None and (b["response_format"] or {}).get("type", "text") != "text":
        raise _bad("structured_outputs", "give either response_format or guided_* / structured_outputs, not both")
    kind, v = next(iter(g.items()))
    if kind in ("grammar", "structural_tag"):
        raise _bad("guided_grammar" if kind == "grammar" else "structured_outputs.structural_tag",
                   "context-free grammars are not supported (JSON Schema, regex and choice are)", "unsupported")
    if kind == "json_object":
        return {"type": "json_object"} if v else None
    if kind == "json":
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                raise _bad("guided_json", "must be a JSON Schema (an object or its JSON text)")
        if not isinstance(v, dict):
            raise _bad("guided_json", "must be a JSON Schema object")
        return {"type": "json_schema", "json_schema": {"name": "guided", "schema": v, "strict": True}}
    if kind == "choice":
        if not isinstance(v, list) or not v or not all(isinstance(c, str) for c in v):
            raise _bad("guided_choice", "must be a non-empty list of strings")
        return {"type": "json_schema", "json_schema": {"name": "choice", "schema": {"enum": v}, "strict": True},
                "_unquote": True}
    if not isinstance(v, str) or not v:
        raise _bad("guided_regex", "must be a non-empty string")
    return {"type": "json_schema", "json_schema": {"name": "regex", "schema": {"type": "string",
                                                                              "pattern": f"^(?:{v})$"},
                                                   "strict": True}, "_unquote": True}


def v_tool_choice(b):
    """-> None (auto), "none", "required" or a function name; raises ApiError."""
    tc = b.get("tool_choice")
    v_type(b, "parallel_tool_calls", bool, "true or false")
    if tc is None or tc == "auto":
        return None
    tools = b.get("tools") or []
    if tc == "none":
        return "none"
    if not tools:
        raise _bad("tool_choice", "needs tools")
    if tc == "required":
        return "required"
    name = (tc.get("function") or {}).get("name") if isinstance(tc, dict) else None
    if not isinstance(name, str):
        raise _bad("tool_choice", "must be none, auto, required or {type: function, function: {name}}")
    if name not in [(t.get("function") or t).get("name") for t in tools]:
        raise _bad("tool_choice", f"function {name[:64]!r} is not in tools")
    return name


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
    b.pop("_force", None)    # internal (tool_choice)
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
    v_sampling_extras(b, chat=True)
    rf = v_guided(b)
    if rf is not None:
        b["response_format"] = rf
    v_tool_choice(b)
    for k in ("add_generation_prompt", "continue_final_message"):
        v_type(b, k, bool, "true or false")
    if b.get("continue_final_message") and b.get("add_generation_prompt"):  # absent counts as false here
        raise _bad("continue_final_message", "cannot be combined with add_generation_prompt: true")
    if b.get("continue_final_message") and b["messages"][-1].get("role") != "assistant":
        raise _bad("continue_final_message", "the last message must be the assistant's")
    if b.get("chat_template") is not None:
        v_type(b, "chat_template", str, "a string")
        if not TRUST_TEMPLATE:
            raise _bad("chat_template", "request chat templates are refused (as vLLM without "
                       "--trust-request-chat-template); start the server with INFERD_TRUST_REQUEST_CHAT_TEMPLATE=1",
                       "unsupported")
    docs = b.get("documents")
    if docs is not None and not (isinstance(docs, list) and all(isinstance(d, dict) for d in docs)):
        raise _bad("documents", "must be a list of objects")
    if (b.get("n") or 1) > 1 and b.get("response_format") is not None and \
            (b["response_format"].get("type") or "text") != "text":
        raise _bad("n", "n > 1 with structured output is not supported", "unsupported")
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


def _request_id(req, body):
    """X-Request-Id (vLLM): the response id is built from it (a body "request_id" wins) and it is echoed back."""
    xr = req.headers.get("x-request-id")
    if xr and len(xr) <= 128 and all(32 <= ord(c) < 127 for c in xr):
        body["_xrid"] = xr
        return xr
    body.pop("_xrid", None)
    return None


def _echo_id(resp, xr):
    if xr is not None:
        resp.headers["X-Request-Id"] = xr
    return resp


@app.post("/v1/chat/completions")
async def chat(req: Request):
    _client_receive.set(req.receive)  # complete() watches it for http.disconnect while it waits (R07)
    try:
        body = check_chat(await read_json(req))  # one validation boundary (Codex review R03)
    except ApiError as e:
        return e.response()
    xr = _request_id(req, body)
    tc = v_tool_choice(body)  # required / a named function: the reply opens with the call's markup (thinking off)
    if tc == "required":
        body["_force"] = TOOL_OPEN + "\n"  # not "<function=": a prefix that ends inside a token invents names
    elif tc not in (None, "none"):
        body["_force"] = f"{TOOL_OPEN}\n<function={tc}>\n"
    if body.get("response_format") is not None:
        return _echo_id(await structured_chat(body), xr)
    return _echo_id(await complete(body), xr)


# ---- /v1/completions (legacy text completions): the prompt is tokenized as given (no chat template), then the same
# engine path as chat. prompt: a string, a list of strings, a list of token ids or a list of those lists.
def check_completion(b):
    b.pop("_grammar", None)
    b.pop("_force", None)
    v_type(b, "model", str, "a string")
    pr = b.get("prompt")
    if isinstance(pr, str) or (isinstance(pr, list) and pr and all(isinstance(x, int) and not isinstance(x, bool)
                                                                   for x in pr)):
        prompts = [pr]
    elif isinstance(pr, list) and pr and all(isinstance(x, str) or (isinstance(x, list) and x and all(
            isinstance(t, int) and not isinstance(t, bool) for t in x)) for x in pr):
        prompts = list(pr)
    else:
        raise _bad("prompt", "must be a string, a list of strings, a list of token ids or a list of such lists")
    if len(prompts) > MAX_N:
        raise _bad("prompt", f"at most {MAX_N} prompts per request")
    if any(isinstance(x, str) and len(x) > LIM.prompt_chars for x in prompts):
        raise ApiError(413, f"a prompt is over {LIM.prompt_chars} characters (INFERD_MAX_PROMPT_CHARS)",
                       "request_too_large", param="prompt")
    if b.get("suffix") not in (None, ""):
        raise _bad("suffix", "suffix (fill-in-the-middle) is not supported, as in vLLM", "unsupported")
    v_type(b, "stream", bool, "true or false")
    so = v_type(b, "stream_options", dict, "an object")
    if so:
        v_type(so, "include_usage", bool, "true or false", "stream_options.")
    for k in ("max_tokens", "max_completion_tokens"):
        v_int(b, k, 1, LIM.output_tokens)
    v_num(b, "temperature", 0, 2)
    v_num(b, "top_p", 0, 1, lo_open=True)
    v_int(b, "top_k", -1, 1 << 20)
    v_int(b, "seed", -(1 << 63), (1 << 63) - 1)
    try:
        stops = parse_stop(b.get("stop"))
    except ValueError as e:
        raise _bad("stop", str(e))
    if len(stops) > 64 or any(len(s) > 4096 for s in stops):
        raise _bad("stop", "at most 64 strings of up to 4096 characters")
    v_policy(b)
    if (b.get("inferd") or {}).get("context") is not None or b.get("context") is not None:
        raise _bad("inferd.context", "saved contexts are for /v1/chat/completions")
    v_sampling_extras(b, chat=False)
    rf = v_guided(b) or (b.get("response_format") if isinstance(b.get("response_format"), dict) else None)
    if b.get("response_format") is not None and not isinstance(b["response_format"], dict):
        raise _bad("response_format", "must be an object")
    if rf is not None and (rf.get("type") or "text") != "text":
        if rf.get("_unquote"):
            raise _bad("guided_choice", "guided_choice / guided_regex are supported on /v1/chat/completions only",
                       "unsupported")
        try:
            spec = structured.parse_response_format(rf)
        except (ValueError, TypeError) as e:
            raise _bad("response_format", str(e))
        b["_grammar"] = spec["schema"]  # constrained decoding of the raw continuation (no instruction, no retries)
    return b, prompts


@app.post("/v1/completions")
async def completions(req: Request):
    _client_receive.set(req.receive)
    try:
        body, raw = check_completion(await read_json(req))
    except ApiError as e:
        return e.response()
    xr = _request_id(req, body)

    def tokenize_all():
        out = []
        for x in raw:
            if isinstance(x, str):
                ids_ = tok.encode(x, add_special_tokens=body.get("add_special_tokens", True) is not False)
                out.append({"tokens": ids_, "text": x})
            else:
                out.append({"tokens": list(x), "text": tok.decode(x, skip_special_tokens=False)
                            if body.get("echo") else ""})
        return out
    try:
        prompts = await preprocess(tokenize_all)
    except ApiError as e:
        return e.response()
    except (ValueError, TypeError, OverflowError) as e:
        return _bad_request(e)
    vocab = len(tok)
    for p in prompts:
        if not p["tokens"]:
            return _bad("prompt", "a prompt is empty").response()
        if any(t < 0 or t >= vocab for t in p["tokens"]):
            return _bad("prompt", f"token ids must be in [0, {vocab})").response()
    return _echo_id(await complete(body, prompts), xr)


# ---- vLLM's tokenizer endpoints
@app.post("/tokenize")
async def tokenize(req: Request):
    try:
        b = await read_json(req)
        v_type(b, "model", str, "a string")
        for k in ("add_special_tokens", "add_generation_prompt", "continue_final_message", "return_token_strs"):
            v_type(b, k, bool, "true or false")
        if b.get("messages") is not None:
            v_messages(b["messages"])
            v_tools(b)
            v_effort_fields(b)
            if b.get("chat_template") is not None and not TRUST_TEMPLATE:
                raise _bad("chat_template", "request chat templates are refused (INFERD_TRUST_REQUEST_CHAT_TEMPLATE)")
            body = dict(b)
            if body.get("add_generation_prompt") is None and not body.get("continue_final_message"):
                body["add_generation_prompt"] = True
            if _has_images(body["messages"]):
                raise _bad("messages", "/tokenize does not count image prompts")
            tokens = (await preprocess(render, body))[0]["tokens"]
        elif isinstance(b.get("prompt"), str):
            if len(b["prompt"]) > LIM.prompt_chars:
                raise ApiError(413, "prompt too long (INFERD_MAX_PROMPT_CHARS)", "request_too_large")
            tokens = await preprocess(tok.encode, b["prompt"], add_special_tokens=b.get("add_special_tokens", True)
                                      is not False)
        else:
            raise _bad("prompt", "give prompt (a string) or messages")
    except ApiError as e:
        return e.response()
    except (ValueError, TemplateError, TypeError, KeyError) as e:
        return _bad_request(e)
    out = {"count": len(tokens), "max_model_len": args.max_model_len or None, "tokens": list(tokens),
           "token_strs": tok.convert_ids_to_tokens(list(tokens)) if b.get("return_token_strs") else None}
    return out


@app.post("/detokenize")
async def detokenize(req: Request):
    try:
        b = await read_json(req)
        t = b.get("tokens")
        if not isinstance(t, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in t):
            raise _bad("tokens", "must be a list of token ids")
        if len(t) > LIM.prompt_tokens:
            raise ApiError(413, "too many tokens (INFERD_MAX_PROMPT_TOKENS)", "request_too_large")
        if any(x < 0 or x >= len(tok) for x in t):
            raise _bad("tokens", f"token ids must be in [0, {len(tok)})")
    except ApiError as e:
        return e.response()
    return {"prompt": await preprocess(tok.decode, t, skip_special_tokens=False)}


# ---- GET /metrics: Prometheus text. vLLM's names under an "inferd:" prefix (docs/API-COMPAT.md): request and token
# counters and latency histograms from this front end, gauges (running / waiting requests, KV usage, preemptions)
# from the engine's stats at scrape time.
class _Metrics:
    TTFT = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0,
            80.0, 160.0, 640.0, 2560.0)
    E2E = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0,
           960.0, 1920.0, 7680.0)

    def __init__(self):
        self.prompt = self.gen = self.cached = 0
        self.finished = collections.Counter()
        self.ttft = [0] * (len(self.TTFT) + 1) + [0.0]  # bucket counts (last: +Inf), then the sum
        self.e2e = [0] * (len(self.E2E) + 1) + [0.0]

    @staticmethod
    def _obs(h, bounds, v):
        h[next((i for i, b in enumerate(bounds) if v <= b), len(bounds))] += 1
        h[-1] += v

    def done(self, ev, t_sub, first):
        """One finished engine request (a choice): its done event, submit time and first-token time."""
        now = time.monotonic()
        self.prompt += ev.get("p", 0)
        self.gen += ev.get("c", 0)
        self.cached += ev.get("cached", 0)
        self.finished[str(ev.get("done"))[:16]] += 1
        if first is not None:
            self._obs(self.ttft, self.TTFT, first - t_sub)
        self._obs(self.e2e, self.E2E, now - t_sub)

    def text(self, st):
        lab = f'model_name="{args.name}"'
        o = []

        def metric(name, typ, help_, rows):
            o.append(f"# HELP inferd:{name} {help_}")
            o.append(f"# TYPE inferd:{name} {typ}")
            o.extend(rows)

        def hist(name, help_, h, bounds):
            rows, acc = [], 0
            for i, b in enumerate(bounds):
                acc += h[i]
                rows.append(f'inferd:{name}_bucket{{{lab},le="{b}"}} {acc}')
            acc += h[len(bounds)]
            rows += [f'inferd:{name}_bucket{{{lab},le="+Inf"}} {acc}', f"inferd:{name}_sum{{{lab}}} {h[-1]:.6f}",
                     f"inferd:{name}_count{{{lab}}} {acc}"]
            metric(name, "histogram", help_, rows)
        blocks = st.get("kv_blocks") or 0
        metric("num_requests_running", "gauge", "Sequences the engine is running (prefill or decode).",
               [f"inferd:num_requests_running{{{lab}}} {st.get('running', 0)}"])
        metric("num_requests_waiting", "gauge", "Sequences waiting for admission.",
               [f"inferd:num_requests_waiting{{{lab}}} {st.get('waiting', 0)}"])
        metric("kv_cache_usage_perc", "gauge", "Fraction of KV blocks held by running sequences (1 = full).",
               [f"inferd:kv_cache_usage_perc{{{lab}}} {(st.get('kv_running', 0) / blocks) if blocks else 0:.6f}"])
        metric("kv_cache_cached_perc", "gauge", "Fraction of KV blocks holding cached prefixes only (reclaimable).",
               [f"inferd:kv_cache_cached_perc{{{lab}}} {(st.get('kv_cached_only', 0) / blocks) if blocks else 0:.6f}"])
        metric("num_preemptions", "counter", "Sequences preempted when the KV pool ran dry.",
               [f"inferd:num_preemptions_total{{{lab}}} {st.get('preempted', 0)}"])
        metric("prompt_tokens", "counter", "Prompt tokens of finished requests.",
               [f"inferd:prompt_tokens_total{{{lab}}} {self.prompt}"])
        metric("generation_tokens", "counter", "Generated tokens of finished requests.",
               [f"inferd:generation_tokens_total{{{lab}}} {self.gen}"])
        metric("prefix_cache_queries", "counter", "Prompt tokens looked up in the prefix cache.",
               [f"inferd:prefix_cache_queries_total{{{lab}}} {self.prompt}"])
        metric("prefix_cache_hits", "counter", "Prompt tokens served from the prefix cache.",
               [f"inferd:prefix_cache_hits_total{{{lab}}} {self.cached}"])
        metric("request_success", "counter", "Finished requests by finish reason (engine terms).",
               [f'inferd:request_success_total{{{lab},finished_reason="{k}"}} {v}'
                for k, v in sorted(self.finished.items())] or [f'inferd:request_success_total{{{lab},finished_reason="stop"}} 0'])
        hist("time_to_first_token_seconds", "Time from submit to the first generated token.", self.ttft, self.TTFT)
        hist("e2e_request_latency_seconds", "Time from submit to the last event.", self.e2e, self.E2E)
        return "\n".join(o) + "\n"


METRICS = _Metrics()


@app.get("/metrics")
async def metrics():
    try:
        st = await asyncio.wait_for(_stats(), 5) if engine.alive else {}
    except Exception:
        st = {}
    return PlainTextResponse(METRICS.text(st), media_type="text/plain; version=0.0.4")


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


def _tok_str(t):
    return tok.decode([t], skip_special_tokens=False)


def logprobs_chat(lps):
    """Engine logprobs [(token, logprob, [id, logprob, ...])] -> OpenAI chat choices[].logprobs."""
    def ent(t, lp):
        s = _tok_str(t)
        return {"token": s, "logprob": lp, "bytes": list(s.encode("utf-8", "replace"))}
    return {"content": [dict(ent(t, lp), top_logprobs=[ent(top[k], top[k + 1]) for k in range(0, len(top), 2)])
                        for t, lp, top in lps]}


def logprobs_text(lps, offset):
    """-> OpenAI completions choices[].logprobs (tokens, token_logprobs, top_logprobs, text_offset) from `offset`."""
    out = {"tokens": [], "token_logprobs": [], "top_logprobs": [], "text_offset": []}
    for t, lp, top in lps:
        s = _tok_str(t)
        out["tokens"].append(s)
        out["token_logprobs"].append(lp)
        d = {_tok_str(top[k]): top[k + 1] for k in range(0, len(top), 2)}
        d.setdefault(s, lp)  # the sampled token is always listed (vLLM)
        out["top_logprobs"].append(d)
        out["text_offset"].append(offset)
        offset += len(s)
    return out


def _last_content(messages):
    c = messages[-1].get("content") if messages else None
    if isinstance(c, list):
        return "".join(p.get("text") or "" for p in c if isinstance(p, dict) and p.get("type") == "text")
    return c or ""


async def complete(body, prompts=None):
    """A chat completion for a validated request body (check_chat; prompts None), or a text completion
    (/v1/completions: prompts = [{"tokens", "text"}], body checked by check_completion) -> JSONResponse or
    StreamingResponse. n choices per prompt run as separate engine requests on the same prompt (the prefix cache
    shares it); choice i of prompt p has index p * n + i."""
    chat = prompts is None
    try:
        if chat:
            body, ctx_tokens = await apply_context(body)  # "inferd": {"context", "effort"} (agent contexts)
            prompt, thinking = await preprocess(render, body)  # bounded: INFERD_MAX_PREPROCESS
            prompts = [prompt]
        else:
            ctx_tokens, thinking = None, False
    except ApiError as e:
        return e.response()
    except (ValueError, TemplateError, TypeError, KeyError) as e:  # the chat template rejects the messages / effort
        return _bad_request(e)
    trunc = body.get("truncate_prompt_tokens")  # vLLM: keep the last k prompt tokens (-1: the model's limit)
    for prompt in prompts:
        if trunc is not None:
            keep = (args.max_model_len or LIM.prompt_tokens) if trunc == -1 else trunc
            if len(prompt["tokens"]) > keep:
                if "embeds" in prompt or ctx_tokens is not None:
                    return ApiError(400, "truncate_prompt_tokens: a prompt with images or a saved context cannot be "
                                    "cut", "invalid_value", param="truncate_prompt_tokens").response()
                prompt["tokens"] = prompt["tokens"][-keep:]
                prompt.pop("snap_at", None)
                if "text" in prompt:
                    prompt["text"] = tok.decode(prompt["tokens"], skip_special_tokens=False)
        n_prompt = len(prompt.get("tokens", []))
        if n_prompt > LIM.prompt_tokens:
            return ApiError(400, f"prompt is {n_prompt} tokens; the limit is {LIM.prompt_tokens} "
                            "(INFERD_MAX_PROMPT_TOKENS)", "context_length_exceeded").response()
    if ctx_tokens is not None and prompts[0].get("tokens", [])[:len(ctx_tokens)] != ctx_tokens:
        return ApiError(400, "the rendered request does not extend the context token for token (chat template)",
                        "context_prefix_mismatch").response()
    stops = parse_stop(body.get("stop"))  # validated by check_chat
    keep_stop = bool(body.get("include_stop_str_in_output"))
    min_tokens = int(body.get("min_tokens") or 0)  # stop ids are masked and stop strings ignored until then
    user_stop_ids = [int(t) for t in body.get("stop_token_ids") or []]
    if not engine.alive:
        return ApiError(503, "engine disconnected", retry_after=LIM.retry_after).response()

    def given(*keys, default):  # the first field present and not null; 0 is a value, never "absent"
        return next((body[k] for k in keys if body.get(k) is not None), default)
    top_k = int(given("top_k", default=DEFAULT_TOP_K))
    stop_ids = sorted(set(([] if body.get("ignore_eos") else STOP) + user_stop_ids))  # ignore_eos: vLLM's field
    gen = {"max_tokens": int(given("max_tokens", "max_completion_tokens", default=4096 if chat else 16)),
           "temperature": float(given("temperature", default=0.7)), "top_p": float(given("top_p", default=DEFAULT_TOP_P)),
           "top_k": max(0, top_k), "stop": stop_ids}
    if body.get("seed") is not None:
        gen["seed"] = int(body["seed"])
    # vLLM's logits processing (src/scheduler.h Request: host sampler, never speculated; only fields that change
    # anything are sent, so a request at the defaults keeps the speculative path).
    for k, neutral in (("presence_penalty", 0), ("frequency_penalty", 0), ("repetition_penalty", 1), ("min_p", 0)):
        if body.get(k) is not None and float(body[k]) != neutral:
            gen[k] = float(body[k])
    if min_tokens and stop_ids:
        gen["min_tokens"] = min_tokens
    if body.get("logit_bias"):
        gen["logit_bias"] = [[int(k), float(v)] for k, v in body["logit_bias"].items()]
    n_lp = (int(body.get("top_logprobs") or 0) if body.get("logprobs") else None) if chat else body.get("logprobs")
    if n_lp is not None:
        gen["logprobs"] = int(n_lp)
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
    if not thinking and not no_cache:  # the next turn re-renders this answer token for token: snapshot while decoding
        gen["gen_snap"] = 1
    for k in ("max_context", "priority"):
        if policy.get(k) is not None:
            gen[k] = int(policy[k])
    if "priority" in gen:
        gen["priority"] = max(-10, min(10, gen["priority"]))  # a bounded scale; the engine clamps too
    for prompt in prompts:
        n_prompt = len(prompt.get("tokens", []))
        if gen.get("max_context") and n_prompt >= gen["max_context"]:
            return JSONResponse({"error": {"message": f"prompt is {n_prompt} tokens, max_context is "
                                           f"{gen['max_context']}", "type": "invalid_request_error",
                                           "code": "context_length_exceeded"}}, status_code=400)
    n = int(body.get("n") or 1)
    gens = []  # one engine request per choice
    for p in prompts:
        for k in range(n):
            g = {**{f: v for f, v in p.items() if f != "text"}, **gen}
            if k and "seed" in g:
                g["seed"] = g["seed"] + k  # distinct, reproducible samples per choice
            if session is not None and not gens:
                g["session"] = session  # the agent's turn: its first choice
            gens.append(g)
    if session is not None:
        session_begin(session)
    rids, qs, t_sub = [], [], time.monotonic()
    try:
        for g in gens:
            rid = next(ids)
            qs.append(await engine.submit(rid, g))
            rids.append(rid)
    except BaseException as e:
        for rid in rids:
            engine.release(rid)
            engine.background(engine.cancel_quiet(rid))
        if session is not None:
            session_end(session)
        if isinstance(e, (ConnectionError, OSError, RuntimeError, AttributeError)):
            return ApiError(503, "engine disconnected", retry_after=LIM.retry_after).response()
        raise
    left = [len(rids)]

    def ender(rid):
        over = False

        def end_request(ended):
            """Once per choice, whichever path ends it: unregister, cancel unfinished engine work; the session closes
            with the last choice."""
            nonlocal over
            if over:
                return
            over = True
            engine.release(rid)
            left[0] -= 1
            if session is not None and left[0] == 0:
                session_end(session)
            if not ended:
                engine.background(engine.cancel_quiet(rid))
        return end_request
    ends = [ender(r) for r in rids]

    def end_all(ended):
        for e in ends:
            e(ended)
    base_id = body.get("request_id") or body.get("_xrid") or uuid.uuid4().hex[:24]
    cid = f"{'chatcmpl' if chat else 'cmpl'}-{base_id}"
    created, model = int(time.time()), body.get("model") or args.name
    obj = "chat.completion" if chat else "text_completion"

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

    def usage(evs):
        """Usage over choices: each prompt counted once (its first choice), completions summed."""
        firsts = evs[::n]
        p, c = sum(e["p"] for e in firsts), sum(e["c"] for e in evs)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
                "prompt_tokens_details": {"cached_tokens": sum(e["cached"] for e in firsts)}}

    finish_map = {"stop": "stop", "length": "length"}
    tc = v_tool_choice(body) if chat else None
    tools = body.get("tools") if chat and tc != "none" else None  # tool_choice none: tool markup stays text
    force = body.get("_force") if chat else None
    one_call = body.get("parallel_tool_calls") is False

    def keep_calls(calls):  # tool_choice naming a function: that call, once (vLLM); parallel_tool_calls false: one
        if tc not in (None, "none", "required"):
            calls = [c for c in calls if c["function"]["name"] == tc][:1]
        return calls[:1] if one_call else calls
    echo = (_last_content(body["messages"]) if chat and body["messages"][-1].get("role") == "assistant" else "") \
        if body.get("echo") else ""

    def stop_reason(ev, matched):
        if matched is not None:
            return matched
        t = ev.get("stop_tok")
        return t if t is not None and t in user_stop_ids else None

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
        so = body.get("stream_options") or {}

        async def choice_stream(i):
            """One choice's stream items: ("chunk", choice object, extra members), then ("end", ev) or ("error", msg)."""
            q, rid, prompt = qs[i], rids[i], prompts[i // n]
            n_p = len(prompt.get("tokens", []))
            dt, state = Detok(), {"reasoning": thinking, "buf": "", "tool": force, "pend": ""}
            sm, used, lps, off, first = (StopMatcher(stops, keep_stop) if stops else None), 0, [], 0, None

            def piece(delta, finish=None, **extra):
                nonlocal off
                c = {"index": i}
                if chat:
                    if "reasoning_content" in delta:
                        delta = {**delta, "reasoning": delta["reasoning_content"]}  # vLLM's newer name
                    c["delta"] = delta
                else:
                    c["text"] = delta.get("content", "")
                if lps and (delta or finish):
                    c["logprobs"] = logprobs_chat(lps) if chat else logprobs_text(lps, off)
                    off += sum(len(_tok_str(t)) for t, _, _ in lps) if not chat else 0
                    lps.clear()
                else:
                    c["logprobs"] = None
                c["finish_reason"] = finish
                if finish is not None or not chat:
                    c["stop_reason"] = extra.pop("stop_reason", None)
                if so.get("continuous_usage_stats"):
                    extra["usage"] = {"prompt_tokens": n_p, "completion_tokens": used, "total_tokens": n_p + used}
                return ("chunk", c, extra)
            ev = None
            try:
                if chat:
                    yield piece({"role": "assistant", "content": ""})
                if echo or (not chat and body.get("echo")):
                    text0 = echo if chat else prompt.get("text") or ""
                    off = len(text0)
                    if text0:
                        yield piece({"content": text0})
                while True:
                    ev = await next_event(q, rid)
                    if "t" in ev:
                        used += len(ev["t"])
                        first = first or time.monotonic()
                        if "lp" in ev:
                            lps.append((ev["t"][0], ev["lp"][0], ev["lp"][1]))
                        r, c = split(state, "".join(dt.add(t) for t in ev["t"]))
                        hit = False
                        if c and sm and used >= min_tokens:
                            c, hit = sm.feed(c)
                        delta = {}
                        if r:
                            delta["reasoning_content"] = r
                        if c and tools:
                            c = tool_hold(state, c)
                        if c:
                            delta["content"] = c
                        if delta:
                            yield piece(delta)
                        if not hit:
                            continue
                        await engine.cancel_quiet(rid)  # a stop string: end the engine's work, finish as "stop"
                        ev = dict(await _drain_done(q), done="stop", c=used)
                    elif str(ev["done"]).startswith("error") or ev["done"] == "cancelled":
                        yield ("error", ev["done"])  # an engine failure is an error event, never a normal finish
                        return
                    finish = finish_map.get(ev["done"], ev["done"])
                    if not (sm and sm.done):  # flush held text: a partial UTF-8 tail and the </think> look-behind
                        r, c = drain_split(split, state, dt.flush())
                        if c and sm and used >= min_tokens:
                            c, hit = sm.feed(c)
                            finish = "stop" if hit else finish
                        if c and tools:
                            c = tool_hold(state, c)
                        if r or c:
                            yield piece(dict(({"reasoning_content": r} if r else {}), **({"content": c} if c else {})))
                    if sm and not sm.done and (tail := sm.flush()):  # text held back as a possible stop prefix
                        tail = tool_hold(state, tail) if tools else tail
                        if tail:
                            yield piece({"content": tail})
                    if state["pend"]:
                        yield piece({"content": state["pend"]})
                    if state["tool"] is not None:  # tool calls go out whole, as one final delta
                        _, calls = parse_tool_calls(state["tool"], tools)
                        if calls:
                            calls = keep_calls(calls)
                            yield piece({"tool_calls": [dict(c, index=k) for k, c in enumerate(calls)]})
                            finish = "tool_calls" if finish == "stop" else finish
                        else:
                            yield piece({"content": state["tool"]})
                    yield piece({}, finish, inferd={"sampling": sampling_info(ev)},
                                stop_reason=stop_reason(ev, sm.hit if sm and sm.done else None))
                    METRICS.done(ev, t_sub, first)
                    yield ("end", ev)
                    return
            finally:
                ends[i](ev is not None and "t" not in ev and not ev.get("_local"))

        async def events():
            def line(choices, **extra):
                return "data: " + json.dumps({"id": cid, "object": obj + ".chunk" if chat else obj, "created": created,
                                              "model": model, "choices": choices, **extra}) + "\n\n"
            done_evs = [None] * len(qs)
            if len(qs) == 1:
                items = choice_stream(0)
                tasks = []
            else:  # n > 1 or several prompts: the choices' items interleave as they arrive
                outq = asyncio.Queue()

                async def run(i):
                    try:
                        async for it in choice_stream(i):
                            outq.put_nowait((i, it))
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:  # never leave the merger waiting
                        outq.put_nowait((i, ("error", f"error: {type(e).__name__}")))
                    finally:
                        outq.put_nowait((i, None))
                tasks = [asyncio.ensure_future(run(i)) for i in range(len(qs))]

                async def merged():
                    running = len(tasks)
                    while running:
                        i, it = await outq.get()
                        if it is None:
                            running -= 1
                            continue
                        yield it if it[0] != "end" else ("end", it[1], i)
                items = merged()
            try:
                async for it in items:
                    if it[0] == "chunk":
                        yield line([it[1]], **it[2])
                    elif it[0] == "error":
                        yield "data: " + json.dumps({"error": {"message": it[1], "type": "server_error",
                                                               "code": "engine_error"}}) + "\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    else:
                        done_evs[it[2] if len(it) > 2 else 0] = it[1]
                if so.get("include_usage"):
                    yield line([], usage=usage([e or {"p": 0, "c": 0, "cached": 0} for e in done_evs]))
                yield "data: [DONE]\n\n"
            finally:
                for t in tasks:
                    t.cancel()
                end_all(False)
        resp = _Stream(events(), end_all)
        return resp

    # Stop strings: watch the answer text as it arrives (the same detokenizer and reasoning split as the stream) and
    # cancel at the first match; n_used = the tokens up to the one that completed it.
    async def choice_full(i):
        q, rid = qs[i], rids[i]
        out, lps, first = [], [], None
        sm, sdt, sst, n_used = (StopMatcher(stops) if stops else None), Detok(), {"reasoning": thinking, "buf": ""}, None
        ended = False
        try:
            while True:
                ev = await next_event(q, rid)
                if "t" in ev:
                    out += ev["t"]
                    first = first or time.monotonic()
                    if "lp" in ev:
                        lps.append((ev["t"][0], ev["lp"][0], ev["lp"][1]))
                    if sm and n_used is None:
                        _, c = split(sst, "".join(sdt.add(t) for t in ev["t"]))
                        if c and len(out) >= min_tokens and sm.feed(c)[1]:
                            n_used = len(out)
                            await engine.cancel_quiet(rid)
                else:
                    ended = not ev.get("_local")  # a timeout or a disconnect: the engine has not ended it
                    if ended:
                        METRICS.done(dict(ev, done="stop", c=n_used) if n_used is not None else ev, t_sub, first)
                    break
        finally:  # the client went away (CancelledError, http.disconnect) or anything else: the engine work stops too
            ends[i](ended)
        return out, lps, n_used, ev

    recv = _client_receive.get()
    watch = engine.background(_watch_disconnect(recv, qs)) if recv is not None else None
    try:
        results = await asyncio.gather(*(choice_full(i) for i in range(len(qs))))
    finally:
        if watch is not None:
            watch.cancel()
        end_all(False)
    choices, evs = [], []
    for i, (out, lps, n_used, ev) in enumerate(results):
        if ev.get("_gone"):  # nobody is listening; 499 (nginx's "client closed request") for the logs
            return JSONResponse(_err("client disconnected", "invalid_request_error", "client_disconnected"),
                                status_code=499)
        if n_used is not None:  # ended by a stop string (the engine's own end, if it raced the cancel, is moot)
            out, ev = out[:n_used], dict(ev, done="stop", c=n_used)
        if str(ev["done"]).startswith("error"):
            code = 504 if ev.get("_timeout") else 503 if ("disconnected" in ev["done"] or "too slow" in ev["done"]) \
                else 400 if ("too long" in ev["done"] or "max_context" in ev["done"] or "json_schema" in ev["done"]
                             or "bad request" in ev["done"]) else 500
            return JSONResponse({"error": {"message": ev["done"], "type": "server_error" if code != 400 else
                                           "invalid_request_error", "code": "engine_error"}}, status_code=code)
        evs.append(ev)
        text = tok.decode(out, skip_special_tokens=False)
        reasoning = None
        if thinking and THINK_END in text:
            reasoning, text = text.split(THINK_END, 1)
            text = text.lstrip("\n")
        elif thinking:  # cut off inside the reasoning (max_tokens): it is all reasoning, as the stream reports it
            reasoning, text = text, ""
        matched = None
        if stops:
            start = 0
            if min_tokens:  # stop strings count only after min_tokens tokens: search past the text they make
                pre = tok.decode(out[:min_tokens], skip_special_tokens=False)
                if thinking:
                    pre = pre.split(THINK_END, 1)[1].lstrip("\n") if THINK_END in pre else ""
                start = min(len(pre), len(text))
            h = _first_stop(text[start:], stops)
            if h is not None:
                matched = h[1]
                text = text[:start + h[0]] + (h[1] if keep_stop else "")
        finish = finish_map.get(ev["done"], ev["done"])
        lp_obj = None
        if n_lp is not None:
            lps = lps[:len(out)]
            lp_obj = logprobs_chat(lps) if chat else logprobs_text(lps, len(prompts[i // n].get("text") or "")
                                                                    if body.get("echo") else 0)
        if not chat:
            if body.get("echo"):
                text = (prompts[i // n].get("text") or "") + text
            choices.append({"index": i, "text": text, "logprobs": lp_obj, "finish_reason": finish,
                            "stop_reason": stop_reason(ev, matched)})
            continue
        calls = []
        if force:
            text = force + text
        if tools:
            text, calls = parse_tool_calls(text, tools)
            calls = keep_calls(calls)
        if echo:
            text = echo + (text or "")
        msg = {"role": "assistant", "content": text}
        if calls:
            msg["tool_calls"] = calls
            finish = "tool_calls" if finish == "stop" else finish
        if reasoning is not None:
            msg["reasoning_content"] = msg["reasoning"] = reasoning
        choices.append({"index": i, "message": msg, "logprobs": lp_obj, "finish_reason": finish,
                        "stop_reason": stop_reason(ev, matched)})
    return JSONResponse({"id": cid, "object": obj, "created": created, "model": model, "system_fingerprint": None,
                         "choices": choices, "usage": usage(evs), "inferd": {"sampling": sampling_info(evs[0])}})


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
        if out is not None and body["response_format"].get("_unquote"):  # guided_choice / guided_regex: the string
            out = json.loads(out)
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
