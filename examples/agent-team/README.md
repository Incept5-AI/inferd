# agent-team

> Uses the agent-context API ([docs/CONTEXTS.md](../../docs/CONTEXTS.md)). A self-test as well as a demo: it prints
> `PASS`/`FAIL` (and `SKIP` when there was nothing to compare); the exit code is 1 if anything failed.

A mixed team on one server that is short of memory: a planner that reasons, a few workers, many one-off classifiers,
and unrelated filler traffic. The script runs the same workload twice in one go: **with** the API's cache controls
(sessions for the planner and workers, priorities, `"cache": "drop"` for the one-offs) and **without** them (the
same requests, plain). It prints the two side by side.

```bash
python3 agent_team.py                                   # http://127.0.0.1:8300/v1 (or INFERD_URL)
python3 agent_team.py --workers 8 --classifiers 32 --rounds 3 --planner-doc 4096 --worker-doc 2048   # bigger
python3 agent_team.py --mode api                        # one run only (api | plain | both)
```

| Role | Settings (both runs) | With the API, also |
|---|---|---|
| planner | effort medium (reasons), a ~2k-token brief, one-sentence answers | `session`, `priority: 8` |
| 4 workers | effort none, `max_context: 8192`, each its own ~1k-token spec | `session`, `priority: 2` |
| 16 classifiers per burst | effort none, `max_context: 2048`, a short unique ticket, one word out | `"cache": "drop"`, `priority: -5` |
| filler | unrelated ~2k-token prompts, a few tokens out | (nothing: default caching in both runs) |

Each run: the team's first turn; a burst of classifiers on their own (does the cache grow?); then two rounds of
[filler + classifiers, then the planner and workers take their next turn]. A sampler polls `GET /v1/cache` for the
peak cached KV.

## Give it a server that is short of memory

With plenty of memory nothing is evicted, both runs look the same, and the comparison prints `SKIP` (as it does in
`run_all.sh`, whose server has a large pool). To see the API work, start inferd small, for example:

```bash
cd engine && ./run.sh 8300 8 1024 1 8 32768       # 8 sequences, 1 GiB of KV (~32k tokens), 8 cached snapshots: 16 slots
```

The pressure in this demo comes mostly from **snapshot slots**, not KV blocks. Qwen3.8 is a hybrid model: a cached
prefix can only be resumed with its saved DeltaNet state, and the server holds a fixed number of those. Every one-off
request that is kept takes one, so a stream of short requests pushes the team's states out long before the KV pool
is full. With an SSD tier (`SPILL_DIR`) evicted states go to disk and come back on the next turn, which hides most of
the difference; the run below has no SSD tier.

## Output

Mac mini M4 Pro (64 GB), Qwen3.8-27B 4-bit (lmstudio-community) + DFlash2, build 1570a8e, server
`run.sh 8300 8 1024 1 8 32768` (no SSD tier), 3 November 2026 ([raw](../results/m4pro-1570/agent-team-pressure.txt)):

```
server: KV pool 32256 tokens (1008 blocks), 16 snapshot slots, 0 blocks cached at start
team: planner ~2048 + 4 workers x ~1024 tokens; filler ~4096 tokens per round (less than the pool: the pressure comes mostly from snapshot slots, see README)

=== run without (plain requests)
  team's first turn: 5 agents, TTFT max 61.0 s
  16 classifiers alone: cached KV 237 -> 353 blocks, snapshots 7 -> 9
  round 1: filler 2 x ~2048 tokens + 16 classifiers in 78.3 s; cache now 621 blocks cached, 10 snapshots, 387 free
  round 1: team resumed; planner TTFT 61.89 s cached 0/2614, workers TTFT median 62.29 s, cached 0/5368
  round 2: filler 2 x ~2048 tokens + 16 classifiers in 76.3 s; cache now 897 blocks cached, 11 snapshots, 111 free
  round 2: team resumed; planner TTFT 64.08 s cached 0/2667, workers TTFT median 55.74 s, cached 0/5559

=== run with the API (sessions, priorities, cache drop for one-offs)
  team's first turn: 5 agents, TTFT max 60.5 s
  16 classifiers alone: cached KV 944 -> 940 blocks, snapshots 12 -> 8
  round 1: filler 2 x ~2048 tokens + 16 classifiers in 75.0 s; cache now 902 blocks cached, 8 snapshots, 106 free
  round 1: team resumed; planner TTFT 3.97 s cached 2496/2590, workers TTFT median 3.64 s, cached 5056/5374
  round 2: filler 2 x ~2048 tokens + 16 classifiers in 75.0 s; cache now 941 blocks cached, 8 snapshots, 67 free
  round 2: team resumed; planner TTFT 2.92 s cached 2560/2641, workers TTFT median 2.08 s, cached 5408/5574

=== side by side (TTFT in seconds; cached = share of prompt tokens served from the cache)
                             |                       with the API |                            without
planner TTFT med / max       |                        3.44 / 3.97 |                      62.99 / 64.08
planner resumes cached       |             97%, stayed cached 2/2 |              0%, stayed cached 0/2
workers TTFT med / max       |                        2.20 / 3.64 |                      62.29 / 64.55
workers resumes cached       |             96%, stayed cached 8/8 |              0%, stayed cached 0/8
classifiers TTFT med / max   |                      17.98 / 55.33 |                      18.44 / 65.59
filler TTFT med / max        |                      57.51 / 65.46 |                      57.02 / 70.00
errors (all roles)           |                                  0 |                                  0
one-off burst: cached KV     |                  944 -> 940 blocks |                  237 -> 353 blocks
one-off burst: snapshots     |                            12 -> 8 |                             7 -> 9
peak cached KV               |            966 blocks (30k tokens) |            897 blocks (28k tokens)
SSD tier: spilled / restored |                        no SSD tier |                        no SSD tier
wall                         |                              281 s |                              417 s

=== checks
  PASS  with the API: every request answered
  PASS  with the API: classifiers labelled 48/48 tickets correctly
  PASS  without: every request answered
  PASS  without: classifiers labelled 48/48 tickets correctly
  PASS  with the API: the planner stayed cached (2/2 resumes)
  PASS  with the API: the workers stayed cached (8/8 resumes, 96% of prompt tokens)
  PASS  with the API: one-offs left the cache flat (cached KV 944 -> 940 blocks, snapshots 12 -> 8)
  PASS  workers resume better with the API: median TTFT 2.20 vs 62.29 s, cached 96% vs 0%
  info  without the API the same one-off burst grew the cache 237 -> 353 blocks

PASS: agent-team, 0 failed check(s)
```

What it shows:
- **Without the API the team lost everything between turns.** Sixteen kept one-offs and two filler prompts per round
  filled the 16 state slots, least recently used first, so the planner's and every worker's next turn found nothing
  cached (0%) and re-read its whole prompt: about a minute per turn on this machine.
- **With the API the same team stayed cached** (96-97% of prompt tokens) and resumed in 2-4 s. The one-offs took no
  snapshots, and the sessions' latest turns were the last thing eviction would touch.
- **One-offs leave the cache flat with `"cache": "drop"`**: the classifier burst added nothing to the cache (cached KV
  and snapshots went slightly *down*: the pool was full of the first run's leftovers, and the running classifiers'
  own blocks were taken from those); without it the same burst added 116 blocks and kept two more snapshots.
- **The one-offs and the filler did not get faster.** Their TTFT is queueing behind each other's prefill and is the
  same in both runs; the API decides what stays in memory, not how fast a cold prompt is computed.
- Peak cached KV is *higher* with the API: the team's turns stay in the pool instead of being replaced. This is the
  intent, not a leak; the script deletes its sessions at the end of each run.

## Other machines

Same build and script, same small server (`run.sh 8300 8 1024 1 8 32768`, no SSD tier), 3 November 2026; raw in
`results/<machine>-1570/agent-team-pressure.txt`. Every check passed on every machine. TTFT in seconds, median over the
two rounds' resumes.

| Machine | Planner resume, with / without | Workers resume, with / without | Workers cached, with / without | Wall, with / without |
|---|---:|---:|---:|---:|
| Mac mini M5 Pro (64 GB) | 1.16 / 18.13 | 1.16 / 18.43 | 94% / 0% | 90 / 121 s |
| Mac mini M6 (32 GB) | 1.86 / 31.25 | 1.79 / 24.57 | 94% / 0% | 150 / 212 s |
| Mac mini M4 Pro (64 GB) | 3.44 / 62.99 | 2.20 / 62.29 | 96% / 0% | 281 / 417 s |

Inside `run_all.sh` (the standard, roomier servers) the comparison printed `SKIP` on all three machines: both runs
kept the team cached, the plain one partly by restoring evicted states from the SSD tier (`results/<machine>-1570/
agent-team.txt`). That is the honest result for a machine with memory to spare: the cache controls do nothing you
can see until memory runs short.
