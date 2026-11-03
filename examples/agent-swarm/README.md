# agent-swarm

N agents share **one long system prompt** (a 16k-token service catalogue, standing in for a codebase, a spec or a
contract set). All of them send their first question at once (the fan-out), then each runs follow-up turns on its own
clock. Every question has a checkable answer in the catalogue (an on-call lead, a datastore, a runbook number), so the
run also shows that each engine gave the same answers.

```bash
python3 agent_swarm.py                                        # inferd on http://127.0.0.1:8300/v1, 8 agents x 16k x 3 turns
python3 agent_swarm.py --url http://127.0.0.1:8100/v1 --model <id>     # any OpenAI-compatible server
python3 agent_swarm.py --agents 16 --words 250 --max-tokens 512        # longer replies: a decode-heavy swarm
```

Options: `--agents`, `--doc` (shared document, tokens), `--turns`, `--max-tokens`, `--words` (also ask for an
explanation of about that many words), `--label`/`--out` (append a JSON summary row). Python standard library only.

## What it shows

- **The document is computed once.** When the eight first questions arrive together, inferd prefills the shared
  system prompt once and hands it to the other seven (snapshots at the system prompt's end and at branch points):
  87% of the fan-out's prompt tokens come from the cache. An engine without that sharing prefills it eight times.
- **Follow-ups pay only for the new question.** Each agent's next turn resumes from its own last turn: 99% of the
  follow-ups' prompt tokens are cached, and the time to first token is about 2 s for a 16.5k-token prompt.
- **Same answers everywhere.** Greedy decoding with thinking off; every turn that completed, on every engine, answered
  with the catalogue's fact (agent 0's answers were word for word the same on inferd and Splash).

## Head to head (Mac mini M5 Pro, 64 GB, 2 November 2026)

Same machine, one server at a time, the same `mlx-community/Qwen3.8-27B-4bit` weights, greedy, thinking off, a fresh
nonce per run. Each engine's start command is in [../compare_engines.sh](../compare_engines.sh). One run per cell; the
console output and JSON rows are in [../results/h2h-m5pro/](../results/h2h-m5pro/).

**8 agents × 16k shared document × 3 turns, 128-token answers** (the default):

| Engine | Last agent's first token (fan-out) | Fan-out cached | Follow-up TTFT median / max | All 24 turns | Correct |
|---|---:|---:|---:|---:|---:|
| inferd (engine-m2) | 38.9 s | 87% | 2.2 / 2.2 s | 46.6 s | 24/24 |
| **Splash 1.5.0** | **26.4 s** | 87% | 2.1 / 3.4 s | **33.5 s** | 24/24 |
| TensorFold 1.2.1 | 43.3 s | 87% | 1.5 / 66.0 s | 117.5 s | 24/24 |
| oMLX 0.7.1 | 300.3 s | 0% | 4.8 / 7.8 s | 328.3 s | 24/24 |
| mlx_lm.server (defaults) | crashed: Metal out of memory | | | | 0/24 |
| mlx_lm.server, `--prompt-concurrency 2` | 274.8 s | 0% | 1.8 / 3.2 s | 288.6 s | 19/24 (2 connections reset) |

**16 agents × 16k shared document × 3 turns, ~250-word answers (`--agents 16 --words 250 --max-tokens 512`)**: the
decode-heavy shape of a real agent fleet. Raw output: [../results/h2h-m5pro-long/](../results/h2h-m5pro-long/).

| Engine | Last agent's first token (fan-out) | Follow-up TTFT median / max | Tokens generated | Overall tok/s | All 48 turns | Correct |
|---|---:|---:|---:|---:|---:|---:|
| **inferd, 64 snapshots** | **40.8 s** | **4.6 / 21.7 s** | 19,344 | **95.1** | **203.5 s** | 48/48 |
| inferd, 32 snapshots (the examples' standard server line) | 41.2 s | 7.0 / 21.2 s | 18,859 | 87.8 | 214.8 s | 48/48 |
| Splash 1.5.0 | 71.9 s | 40.0 / 80.9 s | 18,006 | 78.6 | 229.1 s | 48/48 |
| TensorFold 1.2.1 | 75.1 s | 77.4 / 301.1 s | 18,157 | 35.1 | 516.7 s | 48/48 |

With 16 agents writing real answers, the work shifts from prefill to decoding many streams at once. inferd decodes
all sixteen in one batched speculative step, so each agent's next turn starts within seconds: the median follow-up
waits 5-7 s for its first token against 40 s on Splash (its follow-ups waited behind other agents' turns) and 77 s on TensorFold
(which drafts for one stream and evicted some prefixes: 78% of follow-up tokens cached against inferd's 96-97%). oMLX
and mlx_lm.server were not run at this size: at 8 agents they were already 6-7× slower than inferd. The two inferd
rows differ only in `run.sh`'s snapshot count (see [agent-followups](../agent-followups/) for why it matters).

## Honest notes

- **Splash wins this default cell on the M5 Pro.** The work is almost all prefill (one 16.5k document, 128-token
  answers), and Splash prefills about 1.6× faster here (its Neural Engine FFN split plus its kernels: ~740 tok/s vs
  inferd's ~460 on this GPU). Both share the document and both cache the follow-ups.
- **TensorFold** shares the fan-out too, but its prompt cache is sized at 5.3 GiB on this machine, so two agents' contexts
  were evicted and their follow-ups re-prefilled 16k tokens (66 s to first token).
- **oMLX** prefilled the shared document eight times (0% cached in the fan-out): one agent got its first token at 37 s,
  the other seven at about 300 s. Its follow-ups were then cached.
- **mlx_lm.server with its defaults** prefilled six 16k prompts at once and the GPU ran out of memory
  (`kIOGPUCommandBufferCallbackErrorOutOfMemory`); the generation thread died and every later request got HTTP 404.
  (The raw output of that run ends in a client traceback: the first version of this script could not print a row with
  no successful turn. Fixed since; the server log has the Metal error.)
  Limited to two prompts at a time (`--prompt-concurrency 2`) it ran to the end, but without sharing the document
  (0% cached in the fan-out), and two of the eight client connections were reset while waiting (counted as errors;
  those agents stopped).
- Speculative decoding differs by engine: inferd runs the DFlash2 drafter for every agent, Splash also runs DFlash2,
  TensorFold drafts for one stream (its startup log says so), oMLX and mlx_lm.server run none. That is part of what each engine is; the target weights are identical.
