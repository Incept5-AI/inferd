# MacBook Pro M3 Max (external tester), 2–3 November 2026

MacBook Pro, Apple M3 Max (30-core GPU, 96 GB, **no GPU neural accelerators**), macOS 27.0 (26A428), on a Svalt cooling
stand, lid closed, external display, fans on System. Qwen3.8-27B 4-bit (mlx-community @ 10c35ca) + DFlash2 (@ 015e795),
the same weights for every engine. Harness: `bench/conc.py` (sweep: one cold prompt per size, 2 reps; conc: 32k
contexts, follow-up turns with 512-token replies, 1 and 16 agents). One run per engine except Splash (two rounds, plus
a first run that started hot and is kept as `*-hotstart`). Run order: TensorFold, Splash (hot), inferd, Splash; round 2
mirrored. `summary.txt` per run: GPU temperature, clock, power and thermal pressure per phase (nominal throughout).

| 32k contexts | inferd 819c396 | Splash 1.5.0 (r1 / r2) | TensorFold 1.2.1 |
|---|---|---|---|
| Prefill, one cold prompt (tok/s) | 147 | 242 / 242 | 126 |
| Decode, one agent (tok/s) | 34.0 | 58.0 / 56.5 | 19.9 |
| 1 agent, follow-up turn: total tok/s | 35.3 | 51.2 / 47.8 | 20.3 |
| 16 agents, follow-ups: total tok/s | **61.1** | 58.1 / 58.6 | 1.2 |
| 16 agents: wall (s) | **134.1** | 141.0 / 139.8 | 6,762 |
| 16 agents: first token, median / max (s) | **26.3 / 28.4** | 55.9 / 115.2, 53.8 / 113.0 | 2,441 / 6,738 |

The 8k/16k sweep rows are in each run's `sweep.txt`. `m3max-cmp-inferd-svalt-sys-draft-r1-c69b9b2` is a later
single-agent run on build c69b9b2 (Chrome and Colima open): 32k prefill 146 tok/s, decode 32.1 tok/s.

Reading: on a GPU without neural accelerators, Splash is clearly faster for one agent (1.6× prefill, 1.7× decode).
With 16 agents on 32k contexts the totals are level (inferd +4–5%), but inferd starts every agent within 28 s where
Splash's last agent waits 113–120 s. TensorFold 1.2.1 in this configuration did not reuse the shared prefix
(`cached` 3 tokens) and re-read every 32k context. One run per engine: treat differences under ~5% as noise.

Scrubbed: the tester's home directory in process listings is shown as `~`.
