# users-at-once

N users arrive **at the same moment**, each with its own long prompt (8k tokens, distinct from the first token) and a
fixed output (1,024 tokens). This is the shape of the public "8 users on an M3 Ultra" comparison, scaled
down so it runs in a few minutes on a Mac mini: a small version of [bench/wall.py](../../bench/wall.py), with the same
prompts and metrics.

```bash
python3 users_at_once.py                                      # inferd on http://127.0.0.1:8300/v1, N = 1 and 4
python3 users_at_once.py --url http://127.0.0.1:8100/v1 --model <id> --n 1,2,4,8
python3 users_at_once.py --n 8 --prompt 2048 --gen 1024        # many users, short prompts, long answers
```

Options: `--n` (user counts, one level after another), `--prompt`, `--gen`, `--label`/`--out`. Requests set
`ignore_eos` so every user gets exactly `--gen` tokens; the `gen` column shows what each server really produced.
Python standard library only.

## What it measures

- **TTFT median** and **last user's first token**: how long the slowest user waits before anything streams back.
- **All generating**: the decode rate (all users together) inside the window where every user is generating.
  "never" means the server never had all users generating at the same time.
- **End to end**: all generated tokens ÷ (last token − release).

## Head to head (Mac mini M5 Pro, 64 GB, 2 November 2026)

Same machine, one server at a time, the same `mlx-community/Qwen3.8-27B-4bit` weights, greedy, thinking off, one trial
per level. Start commands: [../compare_engines.sh](../compare_engines.sh); raw output:
[../results/h2h-m5pro/](../results/h2h-m5pro/).

**8k-token prompts, 1,024 tokens each** (the default):

| Engine | Users | TTFT median | Last user's first token | All generating | End to end | Wall |
|---|---:|---:|---:|---:|---:|---:|
| inferd (engine-m2) | 1 | 19.6 s | 19.6 s | 72.5 tok/s | 30.4 tok/s | 34 s |
| | 4 | 52.7 s | 78.3 s | **171.3 tok/s** | **40.3 tok/s** | **102 s** |
| Splash 1.5.0 | 1 | **12.5 s** | **12.5 s** | 71.2 tok/s | **38.1 tok/s** | **27 s** |
| | 4 | 55.4 s | 97.5 s | never | 36.6 tok/s | 112 s |
| TensorFold 1.2.1 | 1 | 19.1 s | 19.1 s | **100.0 tok/s** | 34.9 tok/s | 29 s |
| | 4 | 57.9 s | 80.5 s | **171.3 tok/s** | 40.1 tok/s | **102 s** |
| oMLX 0.7.1 | 1 | 18.1 s | 18.1 s | 17.2 tok/s | 13.2 tok/s | 78 s |
| | 4 | 65.5 s | 102.6 s | 40.7 tok/s | 21.0 tok/s | 196 s |
| mlx_lm.server (defaults) | 1, 4 | server already dead (Metal OOM in agent-swarm): HTTP 404 | | | | |
| mlx_lm.server, `--prompt-concurrency 2` | 1 | 19.1 s | 19.1 s | 16.7 tok/s | 12.7 tok/s | 80 s |
| | 4 | 60.9 s | **75.2 s** | 47.0 tok/s | 25.2 tok/s | 162 s |

**8 users × 2k-token prompts, 1,024 tokens each (`--n 8 --prompt 2048 --gen 1024`)**: many users, short prompts, long
answers. Raw output: [../results/h2h-m5pro-long/](../results/h2h-m5pro-long/).

| Engine | Users | TTFT median | Last user's first token | All generating | End to end | Wall |
|---|---:|---:|---:|---:|---:|---:|
| **inferd (engine-m2)** | 8 | **23.4 s** | **38.0 s** | **207.2 tok/s** | **107.4 tok/s** | **76 s** |
| inferd, second run (64 snapshots) | 8 | 23.2 s | 37.8 s | 208.1 tok/s | 107.7 tok/s | 76 s |
| Splash 1.5.0 | 8 | 32.2 s | 62.5 s | never | 105.2 tok/s | 78 s |
| TensorFold 1.2.1 | 8 | 32.9 s | 40.7 s | 196.4 tok/s | 101.2 tok/s | 81 s |

Here every user's prompt is admitted within 38 s on inferd, and all eight then decode together at 207 tok/s. Splash
reaches about the same total throughput, but its users started one after another until 62.5 s and there was never a
moment when all eight were generating. TensorFold 1.2.1 (8 streams by default) is close behind inferd
on every column.

## Honest notes

- **One user is not inferd's case.** Splash prefills 8k in 12.5 s against inferd's 19.6 s on this GPU, and TensorFold's
  single stream decodes faster (100 tok/s with DFlash2 and context-copy drafts vs inferd's 72).
- **At 4 × 8k the M5 Pro is prefill-bound.** 33k tokens of prompts at ~460 tok/s take ~70 s whatever the scheduler,
  so inferd's users still get their first tokens one after another (21, 43, 62, 78 s). It and TensorFold 1.2.1 (which
  now serves 8 streams by default) finish level; Splash never had all four users generating at once and finished
  10% later; oMLX, with no drafter, took twice as long. mlx_lm.server, prefilling two prompts at a time, got its
  last user going 3 s sooner than inferd (75 s) but, decoding without a drafter, finished 60 s later.
- With short prompts and long answers (8 × 2k + 1,024) the work is decode, and inferd leads on every column, though
  by only 5-6% over TensorFold on throughput; its clear win is time to first token for the last users (38 s vs 62.5 s
  on Splash).
- On a bigger GPU the picture moves towards inferd: on the Mac Studio M5 Ultra, 8 users × 14k, inferd's last user waited
  33-42 s against TensorFold's 222-332 s ([docs/RESULTS.md](../../docs/RESULTS.md#e7-many-users-with-long-prompts-released-at-once-2-nov-2026)).
