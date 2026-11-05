# agents-at-once: N agents working at the same moment, checked, with tok/s and (optionally) tokens per joule

Every agent gets its own long context (16k tokens by default), then all of them send their next turn at the same
moment, 512 tokens out each, for two rounds; each round continues the conversation, as an agent does. Each turn asks the
agent to start with `ANSWER: n`, the sum of two numbers that differ for every agent and every round, so a reply that
belongs to another agent, is cut short or garbled is caught. It reports:

- total tok/s per round (sum of completion tokens / the round's wall time) and the first-token median / max;
- how many answers were right, and the most agents generating at the same moment (from each stream's first and last
  token); `PASS` needs every answer right and every agent generating at once in every round;
- with `--power-cmd`, mean wall power per round and tokens per joule (= tok/s per watt).

The harness is [bench/quick64.py](../../bench/quick64.py) (standard library; the prompts, priming and validated streaming
of `bench/studio/conc.py`); [bench/quick64_plot.py](../../bench/quick64_plot.py) draws the power trace over the run.

```bash
# any Mac: a small self-test (run_all.sh uses this size)
./agents_at_once.sh                                   # 8 agents x 4k, server on 127.0.0.1:8300
# the Mac Studio headline (start it with: ./run.sh start --model flash-next)
AGENTS=64 CTX=16k ./agents_at_once.sh
# more than 64 agents: start the server with as many in flight and twice as many snapshots, e.g.
#   MODEL=<flash-next dir> MTP=1 ./run.sh 8300 128 2048 64 256 65536
AGENTS=128 CTX=16k ./agents_at_once.sh
# wall power: any command that prints watts (a smart plug, a meter's CLI)
POWER_CMD='my-plug --watts' AGENTS=64 CTX=16k ./agents_at_once.sh
```

Each run appends one JSON line to `$TMPDIR/agents_at_once.jsonl` (the script prints the path), outside the source tree;
`OUT=<file>` writes elsewhere.

**Measured on a Mac Studio M5 Ultra (256 GB), Qwen3.8-Flash-Next oQ4e + MTP, inferd 0.4.0 engine, 5 November 2026,**
wall power from a smart plug at the socket (the whole machine; idle 13–15 W), 2 rounds each:

| agents | answers right | all generating at once | tok/s (round 1 / 2) | wall power | tokens per joule | slowest first token |
|---:|---:|:---:|---:|---:|---:|---:|
| 64 | 128 / 128 | yes | 1,233 / 1,419 | 298 W | 4.2 / 4.7 | 2.5 / 4.7 s |
| 100 | 200 / 200 | yes | 1,359 / 1,556 | 300 W | 4.5 / 5.2 | 4.1 / 8.2 s |
| 128 | 256 / 256 | yes | 1,378 / 1,597 | 293 W | 4.6 / 5.6 | 5.2 / 9.5 s |

Priming (every agent's 16k context sent at once) takes most of the run: 284 s for 64 agents, 443 s for 100, 574 s for 128.
Raw results (per-agent answers and timings, the plug readings, macOS's power counters): [results/agents-at-once-2026-11-05](../../results/agents-at-once-2026-11-05/). macOS's own
counters (IOReport, via `mactop`) put the GPU at ~128 W and the chip's total at ~278 W during the rounds.
