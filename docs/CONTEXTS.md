# Agent contexts and sessions

> **Status: implemented** (merged 2 Nov 2026, `engine/src/scheduler.h`, `tools/server.mm`, `frontend/serve.py`).

inferd's prefix cache works implicitly: any prompt that shares a prefix with an earlier one resumes from the deepest
cached snapshot (docs/ARCHITECTURE.md, section 11). That is enough for an agent's own follow-up turns, but it does
not let you **name** a context, **keep** it while other traffic goes through the cache, **fork** agents from it
on purpose, **park** an idle agent on SSD, or tell the engine that a request is a **one-off** whose turns are not
worth keeping. The agent-context API adds those, in a few endpoints and one request block. It is an inferd
extension; everything else stays OpenAI-compatible.

| You want | Send |
|---|---|
| a long context computed once and kept | `POST /v1/contexts` {name, system + documents or messages, tools, effort, persist, ttl} |
| N agents forked from it | chat requests with `"inferd": {"context": "<name>"}` and only the agent's own messages |
| an agent at a given effort, context cap and priority | `"inferd": {"effort": "none|low|medium|xhigh", "max_context": N, "priority": -10..10}` |
| an agent whose turns survive other traffic | `"inferd": {"session": "<agent id>"}` |
| a one-off agent that leaves nothing behind | `"inferd": {"cache": "drop"}` |
| to see what is cached where | `GET /v1/contexts`, `GET /v1/cache` |
| to park / forget | `POST /v1/contexts/{id}/offload`, `POST /v1/sessions/{id}/offload`, `DELETE /v1/contexts/{id}`, `DELETE /v1/sessions/{id}` |

The example [examples/agent-contexts](../examples/agent-contexts/) runs all of it against a live server.

## Saved contexts

```http
POST /v1/contexts
{"name": "repo-docs",                       // optional; [A-Za-z0-9._:-], up to 64 characters (else ctx-<hex>)
 "system": "You review pull requests ...",  // a system prompt, and/or
 "documents": ["<file 1>", "<file 2>"],     //   documents appended to it,
 "messages": [...],                         //   or any message list (system, few-shot user/assistant turns)
 "tools": [...],                            // optional: tools are part of the cached prefix
 "effort": "none",                          // none | low | medium | xhigh (default none)
 "persist": "ssd",                          // memory (default): pinned in the GPU prefix cache; ssd: also written
                                            //   to the SSD tier (needs SPILL_DIR), survives eviction and, with
                                            //   SPILL_PERSIST=1, restarts of the engine
 "ttl": 3600}                               // optional: dropped after this many seconds without a launch
```

The reply carries the context's id, its token count, how long it took to compute (`prefill_s`), how much of it was
already cached (`cached_at_save`) and where it lives (`pinned`, `resident.gpu`, `resident.ssd`, `kv_mib`):

```json
{"id": "repo-docs", "object": "context", "effort": "none", "persist": "ssd", "tokens": 8236, ...,
 "variants": [{"effort": "none", "tokens": 8236, "kv_tokens": 8224, "prefill_s": 11.9, "cached_at_save": 0,
               "pinned": true, "resident": {"gpu": true, "ssd": true}, "kv_mib": 270.2}]}
```

What the engine does: the front end renders the messages exactly as they will appear in front of any later turn
(the chat template's text up to where the next message would start), tokenizes them, and sends a `gen` with
`"ctx": "<name>"`. The engine prefills the whole KV blocks of it (the last < 32 tokens are recomputed by each launch,
a few milliseconds), snapshots the recurrent state at the end, enters the chain into the prefix trie and **pins** it:
its blocks are exempt from eviction and from the KV admission budget, its snapshot is never dropped. Saving a
context that is already cached (or on SSD) computes nothing.

Pins are bounded so they cannot starve running agents: at most `PIN_MAX_SLOTS` pinned contexts (default half of the
snapshot slots beyond `max_running`, at least 1) and `PIN_MAX_FRAC` of the KV pool (0.5). A save over the limit
answers HTTP 400 with the reason.

**Effort is part of the prefix.** Qwen3.x chat templates write the reasoning effort into the system prompt, so a
context is rendered at one effort. Launching from it at another effort computes that variant once (concurrent
launches wait for one computation) and pins it beside the first; variants that render identically share one copy
(on Qwen3.8 `none` and `medium` add no instructions, so they share). Save the context at the effort its agents use.

`GET /v1/contexts` lists every context with its variants and residency plus the engine's cache occupancy;
`GET /v1/contexts/{id}` shows one. `DELETE /v1/contexts/{id}` unpins it and frees its blocks and snapshot (only
the part no running agent or other context shares) and removes it from the SSD tier.
`POST /v1/contexts/{id}/offload` writes it to the SSD tier and frees its GPU memory: it stays listed with
`resident.gpu: false`, and the next launch (or save) restores it from SSD instead of recomputing it.

## Launching agents from a context

```http
POST /v1/chat/completions
{"messages": [{"role": "user", "content": "Review PR #4411 for locking bugs."}],
 "max_tokens": 512,
 "inferd": {"context": "repo-docs", "session": "reviewer-7", "effort": "none", "max_context": 32768, "priority": 2}}
```

The front end puts the context's messages (and tools) in front of the request's messages, renders, and checks that
the context's tokens are an exact prefix of the result (HTTP 400 `context_prefix_mismatch` otherwise: it never
silently runs uncached). The engine finds the pinned snapshot and computes only the request's own tokens. Forking N
agents from one context computes the context once, however many arrive at once. Rules: the request must not bring its
own system message when the context has one, and its tools must be the context's (or none: the context's are used).

`effort`, `max_context` and `priority` work on any request, with or without a context. `effort` maps to the chat
template's `enable_thinking` / `reasoning_effort`; `max_context` caps the agent's prompt + output (over-cap prompt:
HTTP 400 `context_length_exceeded`; generation stops at the cap with `finish_reason: "length"`); `priority` decides
who is preempted first when the KV pool runs dry (lowest first).

## Sessions and cache lifetime

- **`"session": "<agent id>"`** names the agent a request belongs to. The engine marks the deepest snapshot each of
  its turns takes (the end of the prompt, and with thinking off the end of the answer), and eviction takes every
  unmarked cached prefix before a marked one. With an SSD tier, a marked snapshot that must leave memory goes to SSD
  first and is dropped from SSD only after unmarked entries; the next turn restores it. An agent's older turns are
  redundant once a newer one is cached and go first.
- **`POST /v1/sessions/{id}/offload`** parks an idle agent now: its latest turn is written to SSD and its GPU memory
  freed (the blocks it shares with others stay). Its next turn restores it.
- **`DELETE /v1/sessions/{id}`** forgets an agent: its cached turns leave memory and the SSD tier (shared prefixes
  and pinned contexts stay).
- **`"cache": "drop"`** marks a one-off request: the engine takes no snapshot for it, so nothing of its prompt or
  answer stays in the cache after it finishes; it still *reads* the cache (a saved context, a shared system prompt).
  Thousands of these leave memory flat and do not push long-lived agents out. `"cache": "keep"` is the default.

`GET /v1/cache` reports the engine's occupancy in KV blocks of 32 tokens: `kv_blocks` (pool), `kv_free`,
`kv_running` (held by running sequences), `kv_cached` / `kv_cached_only` (in the prefix trie / only there),
`kv_pinned`, `snapshots`, `session_snapshots`, `sessions`, `running`, `waiting`, `preempted`, and the SSD tier
(`entries`, `gib`, `spilled`, `restored`, seconds spent).

## Where it pays

Measured with [bench/agents_scenarios.py](../bench/agents_scenarios.py) (raw results and build states:
[results/agent-scenarios/](../results/agent-scenarios/)), Qwen3.8-27B 4-bit + DFlash2, inferd deliberately short of
memory (Mac mini M5 Pro: 4 GiB KV, ~129k tokens; Mac mini M6: 2.4 GiB, ~78k). "before" is the build without this API
(implicit prefix cache, SSD tier on); TTFT in seconds.

| scenario (machine) | with the API | before (SSD tier) | before, no SSD tier |
|---|---:|---:|---:|
| forked agents: relaunch 8 agents from a 19k context after ~66k tokens of unrelated load (M6) | **1.16** (pinned) | 2.06 (restored from SSD) | |
| re-used agents: resume 6 idle agents after unrelated load (M6) | **0.43** (sessions stayed resident) | 0.67 | 18.3 (0% cached: recomputed) |
| re-used agents: resume 8 agents x 8k after unrelated load (M5 Pro) | 0.26 | 0.34 | 15.1 (12% cached) |
| re-used agents: resume after an explicit offload to SSD (M5 Pro) | 0.31 | | |
| one-offs beside long-lived agents, same build: `cache: drop` vs `keep` (M6) | one-off TTFT 5.0, peak cached KV 63k | keep: 7.2, 77.6k (the whole pool) | |
| 16 agents forked from a 24k context, launch TTFT after save / prime (M5 Pro) | 2.1 (save 58.5 s) | 2.3 (prime 58.6 s) | |

- **The API buys guarantees under memory pressure, not speed on an idle machine.** A pinned context, an agent's
  session and the absence of one-off churn hold when other traffic fills the cache. With no pressure (the first M5
  Pro runs, whose fillers fitted in the pool) before and after are identical within noise, and priming the cache by
  hand gives the same launch TTFT as a saved context.
- **The SSD tier, which was already there, removes most of the eviction cost** (15-18 s recomputes become 0.3-0.7 s
  restores); pins and sessions remove the rest and make it predictable.
- Against other engines on the same cells (M5 Pro; each at its default memory, not capped like inferd), Splash fanned
  out the 24k context faster (42.7 vs 60.7 s: faster prefill) but refused 16 of 48 simultaneous agents (HTTP 503);
  oMLX refused 11 and took 438 s for the fan-out; TensorFold 1.2.1 lost its cache under unrelated load (relaunch 193 s,
  resumes 15 s), and its fcfs prefill policy did not help shared-prefix agents. In the mixed fleet (3 long researchers
  + 32 short helpers) inferd's helpers waited ~100 s behind the researchers' prefill (Splash 63 s, TensorFold 37 s) on the build
  tested then; the 3 Nov prefill budget (`PREFILL_PRIO`) brings them to 35 s (researchers 54 s, same wall time;
  RESULTS.md (m)).

## Errors and limits

Malformed bodies get HTTP 400 with an OpenAI-style error (wrong types, non-finite numbers, `max_tokens: 0`, session or
context ids over 128 characters). Over-budget servers answer 503 with `Retry-After`; oversized bodies 413. At most
`INFERD_MAX_CONTEXTS` (256) saved contexts and `INFERD_MAX_SESSIONS` (4,096, least recently used first) sessions are
kept. A create reserves its slot before it computes, so concurrent creates cannot overshoot the limit (HTTP 400
`context_limit`), and all contexts together may retain at most `INFERD_MAX_CONTEXT_MB` (1,024) of messages, tools and
token ids in the front end (HTTP 413 `context_memory_limit`, also for a new effort variant that would cross it). In `GET /v1/cache` and the engine log, a context's engine handles are named `<id>@<effort>#<generation>`: a
context that is replaced gets new handles, so a save that finishes after a delete or replace cannot attach to the
wrong record.

## Engine protocol (for other front ends)

On inferd's Unix socket (JSON lines, `engine/tools/server.mm`):

- `{"op":"gen", ..., "ctx":"<name>", "pin":1, "ssd":1}`: compute the tokens as a context (whole blocks), register it
  under the name, pin it, write it to SSD. Done event: `{"id":N,"done":"stop","p":P,"c":0,"cached":K,"ctx":{"name",
  "tokens","cached","pinned","gpu","ssd","kv_mib"}}`.
- `{"op":"gen", ..., "session":"<agent>"}`, `"no_cache":1`: as above.
- `{"op":"ctx_drop"|"ctx_offload","id":N,"ctx":"<name>"}`, `{"op":"session_drop"|"session_offload","id":N,
  "session":"<agent>"}`, `{"op":"stats","id":N}`: answered with `{"id":N,"done":"ok"|"error: ...","stats":{...}}`.

## What was already there, and what changed

Already implicit before this API: shared prefixes hit the cache (the block trie with GDN state snapshots), N agents
on one new document wait for one prefill (fan-out), system-prompt and branch-point snapshots, decode snapshots at the
end of answers, an SSD tier that receives evicted snapshot chains and restores them on a prefix match (persistent
across restarts with `SPILL_PERSIST=1`), per-agent `max_context` and `priority`.

Engine changes (`src/scheduler.h`, `src/spill.h`, `tools/server.mm`):
- trie nodes carry `pin` (chain), `pin_snap` (snapshot) and `sess` marks; eviction (`evict`, `redundant_victim`)
  skips pinned nodes and orders by (session mark, LRU); KV admission (`fits`) does not count pinned blocks as free;
- `prefill_only` requests: compute whole blocks, snapshot at the end, finish without sampling, register and pin
  (`ctx_finish`); a save that is fully cached (or restorable from SSD) computes nothing;
- explicit spill/restore of a chain: `ctx_offload` / `session_drop(offload)` write the chain and free its unshared
  tail (`drop_chain`); restores go through the existing admission path;
- `no_cache` requests take no snapshot points (prompt end, branch, hints, decode, donation);
- SSD entries carry a priority (named contexts and session turns are dropped after anonymous prefixes);
- server ops `ctx_drop`, `ctx_offload`, `session_drop`, `session_offload`, `stats`; the done event can carry extra
  members.

Front end (`frontend/serve.py`): the endpoints above, context rendering (`render_context`), effort variants
(`ensure_variant`), the prefix check, `effort` / `session` / `cache` in the `inferd` block, TTL expiry.

## Limits

- Contexts live in the front end's memory: restarting the front end forgets their names (the SSD tier's chains
  survive with `SPILL_PERSIST=1`, and saving the same context again restores it from SSD instead of recomputing it).
- Session marks are not persisted across engine restarts.
- Text only: contexts with images are refused.
- Without an SSD tier (no `SPILL_DIR`, and always on Flash-Next and Gemma-4) `persist: "ssd"` keeps the context in
  memory only (`resident.ssd: false`), and the offload endpoints answer HTTP 400.
