# agent-followups

N agents, each with **its own long context** (a different 8k-token service catalogue per agent: think one repository,
ticket history or customer file per agent). All of them ask a first question at once, then take follow-up turns that
append a **tool output** (an ~500-token incident log) and a new question, as an agent does after every tool call. A
server that keeps every agent's context pays only for the new tokens on each follow-up; one that keeps fewer contexts
than it has agents re-reads whole documents.

```bash
python3 agent_followups.py                                    # inferd on http://127.0.0.1:8300/v1, 8 agents x 8k x 3 turns
python3 agent_followups.py --url http://127.0.0.1:8100/v1 --model <id> # any OpenAI-compatible server
python3 agent_followups.py --agents 16 --ctx 4096 --words 250 --max-tokens 512   # longer replies
```

Options: `--agents`, `--ctx` (each agent's document, tokens), `--turns`, `--tool` (tool output per follow-up, tokens),
`--max-tokens`, `--words`, `--label`/`--out`. Every document starts with a fresh nonce. Python standard library only.

## What it shows

- **Every agent's context stays cached.** inferd admits by KV budget and keeps each agent's prefix in its block trie
  (and on SSD with `SPILL_DIR`), so a follow-up prefills only the new tool output and question: 93% of the follow-ups'
  prompt tokens come from the cache.
- **The first round is prefill-bound.** Eight 8k documents are 66k tokens of prefill; on a small GPU that dominates
  the first round on every engine.

## Head to head (Mac mini M5 Pro, 64 GB, 2 November 2026)

Same machine, one server at a time, the same `mlx-community/Qwen3.8-27B-4bit` weights, greedy, thinking off.
Start commands: [../compare_engines.sh](../compare_engines.sh); raw output: [../results/h2h-m5pro/](../results/h2h-m5pro/).

**8 agents × own 8k document × 3 turns (~600 tokens of tool output per follow-up), 128-token answers** (the default):

| Engine | Last agent's first token | Follow-up TTFT median / max | All 16 follow-ups | Follow-ups cached | All 24 turns | Correct |
|---|---:|---:|---:|---:|---:|---:|
| inferd (engine-m2) | 157.0 s | 7.7 / 13.5 s | 28.4 s | 93% | 185.7 s | 24/24 |
| **Splash 1.5.0** | **103.6 s** | 6.8 / 18.5 s | **25.0 s** | 93% | **128.9 s** | 24/24 |
| oMLX 0.7.1 | 149.1 s | 14.6 / 24.9 s | 45.4 s | 93% | 197.7 s | 24/24 |
| TensorFold 1.2.1 | 159.6 s | 22.0 / 144.1 s | 148.6 s | 60% | 308.6 s | 24/24 |
| mlx_lm.server (defaults) | server already dead (Metal OOM in agent-swarm) | | | | | 0/24 |
| mlx_lm.server, `--prompt-concurrency 2` | 155.3 s | 20.7 / 133.2 s | 151.7 s | 68% | 309.2 s | 22/24 (1 connection reset) |

**16 agents × own 4k document × 3 turns, ~250-word answers (`--agents 16 --ctx 4096 --words 250 --max-tokens 512`)**.
Raw output: [../results/h2h-m5pro-long/](../results/h2h-m5pro-long/).

| Engine | Last agent's first token | Follow-up TTFT median / max | All 32 follow-ups | Follow-ups cached | All 48 turns | Correct |
|---|---:|---:|---:|---:|---:|---:|
| **inferd, 64 snapshots** | **161.3 s** | **13.8 / 41.1 s** | 184.9 s | 74% | **378.6 s** | 48/48 |
| inferd, 32 snapshots (the examples' standard server line) | 161.3 s | 38.8 / 152.5 s | 363.0 s | 31% | 555.6 s | 48/48 |
| Splash 1.5.0 | 216.0 s | 62.2 / 100.2 s | **183.0 s** | **82%** | 407.3 s | 48/48 |
| TensorFold 1.2.1 | 189.4 s | 153.8 / 232.9 s | 524.1 s | 2% | 728.1 s | 48/48 |

**Size inferd's snapshots for your agent count.** Qwen3.8 is a hybrid model: resuming an agent from its cached prefix
needs a saved DeltaNet state as well as the KV blocks, and inferd keeps at most `snapshots` of them (`run.sh`'s fifth
argument; [ROADMAP P0-7](../../docs/ROADMAP.md#p0-7-prefix-retention-is-capped-by-state-snapshot-slots-diagnosed-2026-11-01)).
With 32 slots and 16 agents, each holding a prefix per turn plus branch points, agents' prefixes were evicted: only 31%
of the follow-ups' prompt tokens were cached and the follow-ups re-prefilled whole documents. With 64 slots (`run.sh 8300 16 1024 0 64 32768`, 11.5
GiB of state instead of 6.9, 0.5M tokens of KV instead of 0.7M) the same run took 379 s instead of 556 s, and inferd
finished first: every agent's first answer by 161 s (Splash 216 s), and a follow-up's median wait for its first token
13.8 s against Splash's 62.2 s. **Still not perfect:** 74% cached, because several follow-ups resumed from the end of
the agent's document (~4.1k tokens) rather than the end of its previous turn, so they re-read the previous answer and
tool output. The all-follow-ups time is a tie with Splash (185 s vs 183 s).

## Honest notes

- **Splash wins this default cell on the M5 Pro**, for the same reason as in [agent-swarm](../agent-swarm/): the first
  round is 66k tokens of prefill and Splash prefills ~1.6× faster on this GPU. Its follow-ups are as well cached as
  inferd's. inferd's first answers arrive one after another, about every 20 s (24 s to 157 s): prefill is
  compute-bound, and interleaving it does not make it faster.
- **TensorFold** evicts: its prompt cache holds 5.3 GiB here, so only some of the eight contexts survive to the next
  turn and the rest are re-prefilled (60% cached; one follow-up waited 144 s for its first token).
- **oMLX** caches the follow-ups (93%) but decodes without a drafter, so each follow-up takes about twice as long.
- **mlx_lm.server**: with its defaults the server had already died in agent-swarm, so every request got HTTP 404.
  With `--prompt-concurrency 2` it ran, but its prompt cache kept only some contexts (68% cached, one follow-up
  waited 133 s) and one client connection was reset by the server while waiting (counted as an error; that agent
  stopped).
