# agent-handoff

> Uses the agent-context API ([docs/CONTEXTS.md](../../docs/CONTEXTS.md)). A self-test as well as a demo: every step
> checks itself and prints `PASS`/`FAIL`; the exit code is 1 if anything failed.

A lead agent reads a long document and works out an analysis. Its **whole conversation so far** (system prompt,
document, its own turns) is then saved as a named context and handed to other agents, each with its own settings.
They start from the lead's cached state in about a second instead of re-reading the document. At the end the context
and every session are deleted, and the script checks that the engine really let go of them.

```bash
python3 agent_handoff.py                     # http://127.0.0.1:8300/v1 (or INFERD_URL); --doc 8192 --no-variant
```

Any server line from [../README.md](../README.md#start-the-server) works; nothing here needs an SSD tier.

## The calls

```python
# 1. the lead works normally, with a session (its turns are kept in the prefix cache)
POST /v1/chat/completions {"messages": [system, user: document + question],
                           "inferd": {"effort": "medium", "max_context": 32768, "priority": 5, "session": "lead"}}
# 2. hand over: save the lead's messages as a context (its turns are cached, so this computes almost nothing)
POST /v1/contexts {"name": "handoff", "messages": [system, user, assistant, user, assistant], "effort": "medium"}
# 3. agents with their own settings, sending only their own message
POST /v1/chat/completions {"messages": [user: "Turn the plan into a status update"],
     "inferd": {"context": "handoff", "effort": "none", "max_context": 16384, "priority": 0, "session": "writer"}}
     ... {"context": "handoff", "effort": "medium", "priority": 3, "session": "critic"}
     ... {"context": "handoff", "effort": "none", "cache": "drop"}           # a one-off fact-checker
# 4. another effort is another prefix: the first effort-low agent computes that variant once
     ... {"context": "handoff", "effort": "low", "session": "reviewer"}
# 5. clean up and verify
DELETE /v1/contexts/handoff      DELETE /v1/sessions/{lead,writer,critic,reviewer}      GET /v1/cache
```

## What the checks mean

| Step | Check |
|---|---|
| 1 | the lead names the service with the open incident and its on-call lead (both are in the document); its second turn resumes from its first (≥ 90% cached) |
| 2 | the context is pinned, and ≥ 80% of it was already cached when it was saved (the lead's turns) |
| 3 | each receiving agent has ≥ 95% of the context cached and a TTFT under a quarter of the lead's reading time; the fact-checker answers with the service id the *lead* found (so it saw the lead's turns); the writer finishes inside its 16k cap |
| 4 | a second effort-low agent reuses the low variant the first one computed |
| 5 | after the DELETEs, `GET /v1/cache` shows the pinned blocks, sessions and session snapshots back where they started, the context is gone (404), and the lead's conversation sent again as a plain request finds ≤ 10% of its tokens cached |

## Output

Mac mini M5 Pro (64 GB), Qwen3.8-27B 4-bit + DFlash2, build 1570a8e, server `run.sh 8300 16 1024 16 32 32768` with
`SPILL_DIR` and `VISION=1`, 3 November 2026, as `run_all.sh` ran it ([raw](../results/m5pro-1570/agent-handoff.txt)):

```
server cache at start: pinned 0 blocks, cached 177 blocks, snapshots 20, session snapshots 0, sessions 0, contexts 0

1. the lead (effort medium, max_context 32k, priority 5) reads a ~8k-token catalogue
  lead turn 1   TTFT  19.53 s  prompt   8175  cached      0 (  0%)  wall   22.2 s (reasoned 379 chars) | **S-100** — owned by **Search**, on-call **Aiko Tanaka**, st
  PASS  lead turn 1 finished with an answer (finish stop)
  PASS  lead found the incident: S-100, on-call Aiko Tanaka
  lead turn 2   TTFT   0.56 s  prompt   8283  cached   8128 ( 98%)  wall    9.0 s (reasoned 971 chars) | 1. **Triage (0–15 min):** Page Aiko Tanaka, open the inciden
  PASS  lead turn 2 finished with a plan (finish stop)
  PASS  lead turn 2 resumed from its own turn 1 (8128 of 8283 prompt tokens cached)

2. save the lead's whole conversation (system, document, 2 turns) as a context: POST /v1/contexts
  handoff-a3353cdf: 8447 tokens, saved in 0.45 s (engine prefill 0.414 s), 8256 tokens were already cached, pinned=True, 267.1 MiB of KV
  PASS  the context is pinned
  PASS  saving reused the lead's cached turns (8256 of 8447 tokens already cached)
  cache: pinned 263 blocks, cached 440 blocks, snapshots 23, session snapshots 1, sessions 1, contexts 1

3. hand it to three agents at once, each with its own settings
  all three answered in 10.7 s (the lead took 19.5 s just to read the document)
  writer        TTFT   0.40 s  prompt   8475  cached   8416 ( 99%)  wall    3.9 s | S-100 (Search) is experiencing a SEV-1 incident with a 32% e
  critic        TTFT   0.51 s  prompt   8472  cached   8448 (100%)  wall   10.7 s (reasoned 1501 chars) | The plan's sequential structure spends the first 15 minutes 
  fact-checker  TTFT   0.40 s  prompt   8477  cached   8448 (100%)  wall    0.5 s | S-100
  PASS  writer started from the lead's state: 8416 of 8475 tokens cached, TTFT 0.40 s vs 19.5 s to read
  PASS  critic started from the lead's state: 8448 of 8472 tokens cached, TTFT 0.51 s vs 19.5 s to read
  PASS  fact-checker started from the lead's state: 8448 of 8477 tokens cached, TTFT 0.40 s vs 19.5 s to read
  PASS  fact-checker saw the lead's analysis (answered 'S-100', expected S-100)
  PASS  writer finished inside its 16k max_context

4. a reviewer at effort low (the effort is in the template's prefix: a variant of the context)
  reviewer 1    TTFT  20.30 s  prompt   8498  cached   8448 ( 99%)  wall   27.9 s (reasoned 982 chars) | No — it lacks a contingency branch for the case where RB-803
  variants now: none (8447 tokens, prefill 0.414 s), medium (8447 tokens, prefill 0.414 s), low (8473 tokens, prefill 19.958 s)
  low renders a different prefix: the first low agent paid for that variant once
  reviewer 2    TTFT   0.30 s  prompt   8493  cached   8448 ( 99%)  wall    2.3 s (reasoned 214 chars) | Notify the Search team engineering manager that S-100 is in 
  PASS  second low agent reused the variant (8448 of 8493 cached, TTFT 0.30 s)

5. DELETE the context and all sessions, then verify with GET /v1/cache
  before: pinned 527 blocks, cached 709 blocks, snapshots 28, session snapshots 5, sessions 5, contexts 2
  PASS  DELETE /v1/contexts/handoff-a3353cdf
  PASS  DELETE /v1/sessions/... for 5 sessions
  after:  pinned 0 blocks, cached 177 blocks, snapshots 20, session snapshots 0, sessions 0, contexts 0
  PASS  pinned blocks released (527 -> 0, 0 at start)
  PASS  sessions released (5 -> 0, session snapshots 5 -> 0)
  PASS  the engine lists no variant of the context
  PASS  GET /v1/contexts/{id} answers 404
  fresh request TTFT  19.98 s  prompt   8465  cached      0 (  0%)  wall   20.1 s | ok
  PASS  the same prefix sent fresh is no longer cached (0 of 8465 tokens cached; TTFT 20.0 s, the cost of re-reading)

PASS: agent-handoff, 0 failed check(s)
```

What it shows:
- **Handing over costs almost nothing.** Saving the lead's 8.4k-token conversation took 0.45 s: 8,256 of its tokens
  were already in the cache from the lead's own turns, and only the last answer was computed.
- **The receiving agents start from the lead's state, not from the document.** Three agents with three different
  settings, launched at once, each had 99-100% of its prompt cached and its first token in 0.4-0.5 s, against the
  19.5 s the lead spent reading. The one-off fact-checker (`cache: drop`) still *reads* the saved context; it only
  leaves nothing of its own behind.
- **Effort none and medium share one copy; low does not.** Qwen3.8's template writes nothing extra into the prompt
  for none or medium, so the writer, critic and fact-checker all used the medium variant. Low adds an instruction, so
  the first low agent waited 20.3 s while the engine computed that variant once; the second started in 0.3 s. Note
  that the first low agent's `cached_tokens` still reads 99%: the variant is computed as a separate save just before
  the request runs, so its cost shows in TTFT and in the variant's `prefill_s`, not in the request's usage.
- **Deleting really frees it.** After `DELETE /v1/contexts/...` and `DELETE /v1/sessions/...` the pinned blocks,
  sessions and session snapshots are back to where they were before the demo (the 177 cached blocks and 20 snapshots
  at the start are an earlier demo's anonymous prefixes, untouched), and the lead's conversation sent again costs a
  full 20 s re-read with 0 tokens cached.

## Other machines

Same build and script, 3 November 2026 (raw: [M6](../results/m6-1570/agent-handoff.txt),
[M4 Pro](../results/m4pro-1570/agent-handoff.txt)). Seconds; every check passed on every machine.

| Machine | Lead reads ~8k tokens (TTFT) | Save | Writer / critic / fact-checker TTFT | First / second low agent | Fresh after DELETE |
|---|---:|---:|---:|---:|---:|
| Mac mini M5 Pro (64 GB) | 19.5 | 0.45 | 0.40 / 0.51 / 0.40 | 20.3 / 0.30 | 20.0 (0 cached) |
| Mac mini M6 (32 GB) | 31.9 | 0.58 | 0.74 / 1.23 / 0.74 | 32.3 / 0.47 | 31.9 (0 cached) |
| Mac mini M4 Pro (64 GB) | 65.0 | 1.63 | 1.34 / 1.55 / 0.48 | 67.6 / 0.52 | 67.1 (0 cached) |
