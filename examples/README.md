# inferd examples: self-tests and demos

**What these are.** Small programs that confirm inferd works correctly on *your* machine and show what its API does:
agents with their own budgets, tool calls, structured output, vision, saved contexts and sessions. Point them at your
server; each prints what it did, and the agent-context demos check themselves and print `PASS`/`FAIL` lines.
[run_all.sh](run_all.sh) runs all of them and ends with a PASS/FAIL table.

**What they are not.** They are not a benchmark and not a ranking. The times below are what three particular Macs
measured on one day with one build; yours will differ with the chip, the memory, the build and whatever else the
machine is doing. Comparisons against other engines, with their methodology, are in
[docs/RESULTS.md](../docs/RESULTS.md).

Sixteen runs in fourteen folders, one script each (Python standard library, apart from Pillow for OCR), each with a
README that holds the output of a real run.

| Example | What it shows | Self-check | Run time (M5 Pro) |
|---|---|---|---:|
| [quickstart](quickstart/) | one chat request, one streamed follow-up turn, `usage` with `cached_tokens` | exit code | ~4 s |
| [agent-budgets](agent-budgets/) | four concurrent agents with their own `max_context`, effort and `priority`; what an over-cap agent gets back (HTTP 400 / `finish_reason: "length"`) | exit code | ~19 s |
| [tools-web-search](tools-web-search/) | an OpenAI-style tool loop with a keyless Wikipedia `web_search` tool, 8k context, effort low | exit code | ~9 s |
| [tools-shell](tools-shell/) | a `shell` tool restricted to read-only allow-listed commands in a sandbox, 4k context, effort none | exit code | ~8 s |
| [structured-output](structured-output/) | OpenAI `response_format` (`json_object`, `json_schema`): validated JSON in `content`, retries with the validation error, HTTP 422 when the model cannot comply | exit code | ~11 s |
| [coding-smoke](coding-smoke/) | languages-benchmark v1, Python + TypeScript (16 tests, 8 concurrent), effort none and medium | score (15.5-16/16) | 1-3 min each |
| [ocr-agents](ocr-agents/) | four vision agents extracting JSON from synthetic scans; how to size `max_context` per image; effort none vs low vs medium | fields correct (21/21) | ~10 s; `--compare 3` ~2 min |
| [agent-contexts](agent-contexts/) | the agent-context API: save a context once, fork agents from it, offload it and a parked agent to SSD and resume, one-off agents with `cache: drop` | exit code | ~50 s |
| [agent-handoff](agent-handoff/) | a lead agent's whole conversation saved as a context and handed to agents with other settings (effort, `max_context`, priority, `cache: drop`); they start in under a second instead of re-reading; then DELETE and verify the cache let go | **PASS/FAIL** (19 checks) | ~1.5 min |
| [agent-team](agent-team/) | planner + workers + one-off classifiers + filler on a server short of memory, run with and without sessions / priorities / `cache: drop`, side by side | **PASS/FAIL/SKIP** | ~3.5 min on a small server; ~3 min in `run_all.sh` |
| [agent-swarm](agent-swarm/) | 8-16 agents on one shared 16k document: the fan-out computes it once, follow-ups hit the cache; every answer checked | answers checked (24/24) | ~50 s |
| [agents-at-once](agents-at-once/) | N agents (8 by default; 64-128 on a Mac Studio) on their own contexts send their next turn at the same moment; every answer checked, how many generated at once, tok/s and (with `POWER_CMD`) tokens per joule | answers + all at once (PASS/FAIL) | ~1 min (8 x 4k) |
| [agent-followups](agent-followups/) | 8-16 agents, each with its own 4-8k context, taking follow-up turns with tool output | answers checked (24/24) | ~3 min |
| [users-at-once](users-at-once/) | N users with long prompts released together (a small [bench/wall.py](../bench/wall.py)): when does the last one start? | exit code | ~2 min |

agent-swarm, agent-followups and users-at-once also take `--url`/`--model` and run against any OpenAI-compatible
server, which is how the engine comparisons in [docs/RESULTS.md](../docs/RESULTS.md) were made; see
[Against other servers](#against-other-servers).

## Start the server

```bash
cd engine
export MODEL=~/models/Qwen3.8-27B-4bit          # mlx-community 4-bit (revision 10c35caa+ has the vision tower)
export DRAFT=~/models/Qwen3.8-27B-DFlash2       # incoai/Qwen3.8-27B-DFlash2
VISION=1 PY=~/.venvs/inferd/bin/python DRAFT_BITS=4 ./run.sh 8300 16 1024 16 32 32768
#                                                  port agents step KV-GiB snapshots max-context
curl -s localhost:8300/v1/models
```

`VISION=1` is needed for the OCR demos: with an automatic KV pool (KV-GiB 0) it leaves ~1 GiB of GPU memory for the
front end's vision tower on machines with 64 GB or less. Text-only runs can drop it and keep that memory for KV.

`run.sh` needs `MODEL` and the front end's Python (`PY`, default `python3`), so both are set above. Check and stop it
by port:

```bash
cd engine && ./run.sh status 8300
cd engine && ./run.sh stop 8300
```

Builds before 3 November 2026 have no `status`/`stop` and put the engine socket at `/tmp/inferd-PORT.sock` (newer
builds use an owner-only directory); stop those by port, never with `pkill -f` from a shell you care about:
`for p in $(lsof -ti tcp:8300 -sTCP:LISTEN) $(lsof -t /tmp/inferd-8300.sock); do kill $p; done`.

**On a laptop, give the KV pool an explicit size (`16` above) instead of `0` (auto).** Auto sizes the pool to the GPU
budget: 67 GiB of KV and 94.5 GiB in total on this 128 GB machine. With Slack, browsers and other apps open, macOS
ran out of free pages and the GPU stalled: a 7-token reply took 31 s and GPU utilisation sat at 2%. With 16 GiB of
KV (0.5 M tokens, plenty for these demos; 43 GiB in total) the same request runs at about 160 tok/s.

All scripts default to `http://127.0.0.1:8300/v1`; set `INFERD_URL` to point elsewhere.

## What carries over to your own agents

- **Effort** is the chat template's reasoning control, set per request in `chat_template_kwargs`:
  `{"enable_thinking": false}` (none), or `{"reasoning_effort": "low" | "medium" | "xhigh"}`. **`xhigh` is the
  default when you send nothing.** Other values (e.g. `"high"`) make the template raise; the server answers HTTP 400 `invalid_request_error` with the template's message.
- **`max_context`** caps one agent's prompt + output. An over-cap prompt gets HTTP 400 `context_length_exceeded`;
  generation stops at the cap with `finish_reason: "length"`. The server-wide maximum still applies: prompt +
  `max_tokens` above `run.sh`'s last argument is refused.
- **Tool calls** come back OpenAI-style (`message.tool_calls`, `finish_reason: "tool_calls"`), and `role: "tool"`
  results go back in. This needs the November 2026 front end (`engine/frontend/serve.py` with tool-call parsing).
  With streaming, content streams normally and the tool calls arrive whole in one final delta.
- **The prefix cache** resumes an agent's next turn from where its last prompt ended. Keep each agent's effort fixed:
  the effort is written into the system prompt, so changing it changes the prefix.
- **Vision**: thinking off was as accurate as low/medium at full resolution, about twice as fast, and the only effort
  with no misreads on downscaled scans. See [ocr-agents](ocr-agents/).

## Run them all: `run_all.sh`

```bash
./run_all.sh my-mac                                  # server on http://127.0.0.1:8300/v1
./run_all.sh my-mac http://127.0.0.1:18300/v1        # e.g. through an ssh tunnel to another Mac
ONLY=agent-team,agent-handoff ./run_all.sh my-mac    # re-run some; the summary keeps the others' latest results
```

Each demo's console output goes to `results/<name>/<demo>.txt` and its wall time to `timings.tsv`. At the end it prints
(and saves as `summary.txt`) one line per demo: PASS when the demo exited 0 and printed no `FAIL` line, the time, the
exit code, and how many of the demo's own checks passed; failed and skipped checks are listed under their demo. The
script exits 1 if any demo failed. coding-smoke clones languages-benchmark on first use (or set `LANGBENCH=...`; see
[coding-smoke](coding-smoke/)). agent-team will usually print `SKIP` for its with/without comparison here, because
the standard server has memory to spare; run it on its own against a small server to see the difference
([agent-team](agent-team/#give-it-a-server-that-is-short-of-memory)).

## What three Macs measured (3 November 2026)

`run_all.sh` on build 1570a8e, Qwen3.8-27B 4-bit with the vision tower (lmstudio-community) and the DFlash2 drafter,
`VISION=1` and an SSD tier (`SPILL_DIR`), clients on another Mac over an ssh tunnel. Servers: M5 Pro
`run.sh 8300 16 1024 16 32 32768` (16 GiB KV), M6 `run.sh 8300 16 1024 0 8 32768` (auto KV: 1.4 GiB),
M4 Pro `run.sh 8300 16 1024 0 16 32768` (auto KV: 27 GiB). One run each; raw output in `results/<machine>-1570/`. **These are
checks that each machine works, not a comparison between machines**: the servers were sized differently and each
number is a single run.

| Demo | Mac mini M5 Pro (64 GB) | Mac mini M6 (32 GB) | Mac mini M4 Pro (64 GB) |
|---|---:|---:|---:|
| quickstart | 3.9 s | 6.4 s | 9.0 s |
| agent-budgets | 18.7 s | 42.3 s | 80.0 s |
| tools-web-search | 8.6 s | 14.3 s | 21.2 s |
| tools-shell | 8.2 s | 15.3 s | 21.2 s |
| structured-output | 11.0 s | 24.8 s | 33.3 s |
| agent-contexts | 50.1 s | 78.2 s | 154.9 s |
| agent-handoff | 93.0 s | 165.3 s | 279.4 s |
| agent-team (standard server: comparison SKIP) | 193.3 s | 298.2 s | 604.4 s |
| ocr-agents | 10.0 s | 21.3 s | 31.2 s |
| ocr-agents --compare 3 | 111.1 s | 184.8 s | 319.3 s |
| coding-smoke, thinking off | 62.1 s | 104.7 s | 197.5 s |
| coding-smoke, effort medium | 170.6 s | 294.9 s | 481.7 s |
| agent-swarm (8 × 16k, 3 turns) | 48.7 s | 78.2 s | 160.8 s |
| agent-followups (8 × 8k, 3 turns) | 184.8 s | 303.6 s | 621.0 s |
| users-at-once (1 and 4 × 8k) | 134.7 s | 230.2 s | 418.8 s |
| **demos passed** | **15/15** | **15/15** | **15/15** |

- **Correctness**, the part that matters for a self-test: OCR 21/21 fields in every round on every machine; agent-swarm
  and agent-followups 24/24 answers with the catalogue's fact; agent-handoff 19/19 checks; coding-smoke 15.5 or 16 of
  16 (one TypeScript or Python test passing only after feedback costs half a point, and which one moves between runs).
- **agent-team** printed `SKIP` for its comparison on all three standard servers (no pressure reached the team; what
  the plain run lost, 2-10 states per machine, the SSD tier restored). Its run against a deliberately small
  server (`run.sh 8300 8 1024 1 8 32768`, no SSD tier) is the one that shows the API: workers resumed in 1.2 s (M5 Pro),
  1.8 s (M6) and 2.2 s (M4 Pro) with it, against 18 s, 25 s and 62 s without, where every resume re-read its whole prompt
  ([agent-team](agent-team/), raw in `results/<machine>-1570/agent-team-pressure.txt`).
- The M5 Pro's first agent-team run in `results/m5pro-1570/` (517 s) used an earlier version of the script with
  three times the filler; the table shows its re-run with the current script.

Earlier runs, on earlier builds and other machines (MacBook Pro M5 Max, Mac Studio), are kept in `results/` with the
build in the directory name.

## Against other servers

agent-swarm, agent-followups and users-at-once run unchanged against any OpenAI-compatible server (`--url`,
`--model`), and [compare_engines.sh](compare_engines.sh) starts each of five Mac engines the way it was measured on
2 November 2026. Those runs are recorded in the three demos' READMEs and in `results/h2h-m5pro*/`; the engine
comparisons, with methodology and caveats, are in [docs/RESULTS.md](../docs/RESULTS.md). One finding from them is
configuration advice for your own server rather than a ranking:

**Size inferd's snapshots for the agent count.** inferd keeps at most `snapshots` cached prefixes (`run.sh`'s fifth
argument; each costs ~0.14 GiB of DeltaNet state for this model). The standard server line above uses 32, enough for
8 agents; at 16 agents with their own contexts, 32 slots evicted agents' prefixes and the follow-ups re-read whole
documents (31% cached), while 64 slots kept them. See [agent-followups](agent-followups/) and
[ROADMAP P0-7](../docs/ROADMAP.md#p0-7-prefix-retention-is-capped-by-state-snapshot-slots-diagnosed-2026-11-01).
