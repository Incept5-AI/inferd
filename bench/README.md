# bench/: measurement harnesses and lab orchestration

**What these are for.** The harnesses here measure how inferd behaves on a machine: how long the last of N agents
waits for its first token, how much of a follow-up turn comes from the cache, how fast a stream decodes. Run them to
check that inferd works on *your* Mac and to see what it does under the load you care about. A number one of them
prints is what that machine measured on that day with that build; it is not a ranking, and Apple Silicon run-to-run
noise is several percent (more on a warm or busy machine). Comparisons against other engines, with their methodology
and run discipline, are in [docs/RESULTS.md](../docs/RESULTS.md). For a first check of a new machine, use
[examples/run_all.sh](../examples/) (pass/fail self-tests and demos) and `engine/tools/release_gate.sh` instead.

**Harnesses** (safe to run against any OpenAI-compatible server; they only send requests):

| File | What it measures |
|---|---|
| `wall.py` | N agents released at once (distinct long prompts, fixed output); `--shared cold/primed/context` for the agent variant. Total time, last first token, end-to-end and in-window decode. |
| `studio/conc.py`, `studio/speed.py` | Follow-up fleets (`--mode followup`) and single-stream speed. |
| `agents.py`, `agents_scenarios.py` | Multi-round agent fleets; the agent-context scenarios a–e. |
| `quick64.py`, `quick64_plot.py` | N agents (default 64) on their own 16k contexts send their next turn at once, every answer checked; tok/s, first token, and with `--power-cmd` wall power and tokens per joule (~5 min on a Mac Studio). The plot lines up wall power with macOS counters (`mactop --headless`). See [examples/agents-at-once](../examples/agents-at-once/). |
| `sse_client.py` | The validated streaming client all of the above use: a stream counts only with a normal finish, `[DONE]` and consistent usage. |

**Lab scripts.** The shell scripts that drove these harnesses on our machines (model paths, ports, external SSDs)
are not published: they assumed one lab's layout. Every result in `docs/RESULTS.md` names the harness, its arguments and
the raw file in `results/`, so any number can be reproduced with the harnesses above.
