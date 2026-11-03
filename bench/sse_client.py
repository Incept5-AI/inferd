#!/usr/bin/env python3
"""Validated SSE client for OpenAI-style streaming chat completions (standard library only).

Shared by every harness that streams (wall.py, studio/conc.py, studio/speed.py) so that a stream counts as a success
only when the server finished it normally (Codex review 2026-11-02, S04). A stream is OK only if:
  - the HTTP status is 2xx and every `data:` payload is JSON (or the `[DONE]` sentinel);
  - no payload carries an `error` object (inferd sends an error event and then [DONE]: that is still a failure);
  - choice 0 ends with a normal finish_reason (stop / length / tool_calls);
  - `data: [DONE]` arrives, with nothing after it;
  - `usage` is present (include_usage is forced on), prompt/completion tokens are non-negative integers,
    total_tokens (if sent) equals their sum, cached_tokens (if sent) does not exceed prompt_tokens, and a stream
    that produced text reports completion_tokens > 0;
  - for fixed-length runs (`expect_completion=N`, used with ignore_eos), completion_tokens == N and the finish
    reason is "length".
Anything else is recorded in `error` (never raised), with `partial` set when some output had already arrived, so a
caller can keep the trial in its denominator and report it as a failure or as partial service.

All times come from time.monotonic() (one clock across threads; not wall time). `chunks` holds (t, characters) per
content/reasoning delta; `events` keeps the raw `data:` payloads when keep_events=True.
"""
import json, time, urllib.error, urllib.request

NORMAL_FINISH = ("stop", "length", "tool_calls")


def _usage_problem(u, text_seen):
    if not isinstance(u, dict):
        return "usage missing"
    pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
    for k, v in (("prompt_tokens", pt), ("completion_tokens", ct)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            return f"usage.{k} invalid: {v!r}"
    tt = u.get("total_tokens")
    if tt is not None and tt != pt + ct:
        return f"usage.total_tokens {tt} != prompt {pt} + completion {ct}"
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is not None and (not isinstance(cached, int) or cached < 0 or cached > pt):
        return f"usage cached_tokens {cached!r} not within prompt_tokens {pt}"
    if text_seen and ct == 0:
        return "output received but usage.completion_tokens is 0"
    return None


def stream_chat(url, body, timeout=600.0, expect_completion=None, keep_events=False, path="/v1/chat/completions",
                headers=None, opener=None, on_send=None):
    """POST a streaming chat completion and validate it. Returns a dict (see the module docstring); never raises
    for a request/stream failure. `on_send` is called just before the request is sent (e.g. to stamp a release)."""
    body = dict(body, stream=True)
    so = dict(body.get("stream_options") or {})
    so["include_usage"] = True
    body["stream_options"] = so
    opener = opener or urllib.request.urlopen
    rec = {"t_sent": None, "t_first": None, "t_last": None, "t_end": None, "chunks": [], "text": "", "reasoning": "",
           "tool_calls": [], "usage": None, "finish_reason": None, "done": False, "error": None, "partial": False,
           "prompt_tokens": None, "completion_tokens": None, "cached_tokens": None, "events": [] if keep_events else None}
    text, reasoning, errors = [], [], []
    req = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **(headers or {})})
    if on_send:
        on_send()
    rec["t_sent"] = time.monotonic()
    try:
        resp = opener(req, timeout=timeout)
        status = getattr(resp, "status", None) or getattr(resp, "code", 200)
        if status and not 200 <= int(status) < 300:
            errors.append(f"HTTP {status}")
        with resp if hasattr(resp, "__enter__") else _nullctx(resp):
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue  # comments (":"), event:/id: fields and blank separators
                payload = line[5:].lstrip(" ")
                if keep_events:
                    rec["events"].append(payload)
                if rec["done"]:
                    errors.append("data after [DONE]")
                    break
                if payload == "[DONE]":
                    rec["done"] = True
                    continue
                try:
                    ev = json.loads(payload)
                except ValueError:
                    errors.append(f"non-JSON SSE payload: {payload[:120]!r}")
                    continue
                if not isinstance(ev, dict):
                    errors.append(f"SSE payload is not an object: {payload[:120]!r}")
                    continue
                if ev.get("error") is not None:
                    e = ev["error"]
                    errors.append("SSE error: " + str(e.get("message") if isinstance(e, dict) else e)[:300])
                if ev.get("usage"):
                    rec["usage"] = ev["usage"]
                for ch in ev.get("choices") or []:
                    if ch.get("index", 0) != 0:
                        continue
                    d = ch.get("delta") or {}
                    c, r = d.get("content") or "", d.get("reasoning_content") or d.get("reasoning") or ""
                    if d.get("tool_calls"):
                        rec["tool_calls"].extend(d["tool_calls"])
                    if c or r:
                        now = time.monotonic()
                        rec["t_first"] = rec["t_first"] or now
                        rec["t_last"] = now
                        rec["chunks"].append((now, len(c) + len(r)))
                        text.append(c)
                        reasoning.append(r)
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        errors.append(f"HTTP {e.code}: {detail}")
    except Exception as e:  # timeouts, resets, decode failures: recorded, never dropped from a denominator
        errors.append(repr(e)[:300])
    rec["t_end"] = time.monotonic()
    rec["text"], rec["reasoning"] = "".join(text), "".join(reasoning)
    u = rec["usage"]
    if isinstance(u, dict):
        rec["prompt_tokens"], rec["completion_tokens"] = u.get("prompt_tokens"), u.get("completion_tokens")
        rec["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if not errors:  # protocol checks only matter when nothing already failed
        if rec["finish_reason"] not in NORMAL_FINISH:
            errors.append(f"abnormal finish_reason {rec['finish_reason']!r}")
        if not rec["done"]:
            errors.append("stream ended without data: [DONE]")
        p = _usage_problem(u, bool(rec["chunks"] or rec["tool_calls"]))
        if p:
            errors.append(p)
        elif expect_completion is not None:
            if rec["completion_tokens"] != expect_completion:
                errors.append(f"completion_tokens {rec['completion_tokens']} != requested {expect_completion}")
            elif rec["finish_reason"] != "length":
                errors.append(f"fixed-length run finished with {rec['finish_reason']!r}, expected 'length'")
    if errors:
        rec["error"] = "; ".join(errors)[:600]
        rec["partial"] = bool(rec["chunks"] or rec["tool_calls"])
    return rec


class _nullctx:
    def __init__(self, r):
        self.r = r

    def __enter__(self):
        return self.r

    def __exit__(self, *a):
        return False
