# Mac mini M6 (32 GB) head-to-head, 4 November 2026

Boxes: mini-m6 (macOS 27.0) and mini-m6-2 (macOS 27.2), measured level within noise (single 4k decode 39.7/38.0 vs
38.6/36.8 tok/s; follow-ups 4k x 8 91.7 vs 87.3, 16k x 4 65.6 vs 72.3). Engines: inferd 0.2.0 (prebuilt), Splash
1.5.0 (latest stable; mini-m6 only), TensorFold 1.2.2. Weights: Qwen3.8-27B 4-bit (mlx-community, identical on both
boxes) + incoai DFlash2; Qwen3.5-9B 4-bit and 8-bit (mlx-community). Harness: conc.py (cold single, follow-ups),
agents.py (8k fleet, turns/min, 2 rounds), agents_scenarios.py (b, c). 0 errors and full denominators everywhere.
Raw files: mini-m6/ and mini-m6-2/ (scrubbed: no user paths or addresses).

## TensorFold 1.2.2 and the 27B on 32 GB

Refused in all 25 configurations tried on both boxes (--parallel 8/4/2/1, --context 32768 down to 1024, with and
without the drafter, --profile stable). Smallest: `--no-drafts --parallel 1 --context 1024`: "needs 2.7 GiB for its
streams and 512-row prompt scratch at a 1024-token context; 2.7 GiB remains beside the weights and an 8 GiB margin".
No CLI flag or environment variable lowers the 8 GiB margin. 1.2.1 refused the same way on 3 Nov.

## Qwen3.8-27B: inferd 0.2.0 vs Splash 1.5.0 (mini-m6)

| one agent (2 reps) | inferd | Splash |
|---|---|---|
| 4k first token / decode | 15.0–15.1 s / 38.0–39.7 tok/s | **8.0–8.2 s / 44.0** |
| 16k first token / decode | 62.6–65.6 s / 31.9–37.7 | **35.8–36.7 s / 38.9** |

| follow-ups, total tok/s (slowest first token) | inferd | Splash |
|---|---|---|
| 4k x 1 / 4 | 27.4 / 74.4 | **36.4 / 93.2** |
| 4k x 8 | 94.9 (3.7 s) | 94.9 (24.3 s) |
| 4k x 16 | **111.4** (9.8 s) | 88.5 (76.4 s) |
| 16k x 1 / 4 | 25.2 / 66.6 (2.2 s) | **32.6 / 69.6** (10.6 s) |
| 16k x 8 | 65.8 (35.6 s) | **86.4** (26.9 s) |
| 16k x 16 | 64.4 (101.9 s) | **88.6** (71.5 s) |

inferd's KV pool on 32 GB is 2.4 GiB (about four 16k contexts); above 4 agents at 16k it re-reads or restores from
SSD. Splash's prefill is ~1.8x faster for one agent (inferd's int8 prefill copy does not fit beside the KV pool).

| 8k agent fleet, turns/min (first token p50 / p95 s) | inferd GPU | inferd + Neural Engine (opt-in) | Splash |
|---|---|---|---|
| 8 agents | 9.67 (12.1 / 38.7) | **11.58** (19.7 / 36.8) | 8.71 (39.0 / 63.7) |
| 16 agents | 9.72 (55.8 / 91.1) | **14.33** (59.4 / 67.5) | 8.78 (88.2 / 116.3) |
| 32 agents | 9.57 (154.1 / 177.2) | **12.12** (125.9 / 150.4) | 8.82 (167.1 / 292.5) |

Scenarios (inferd, API mode): 8 agents forked from a long context: cold 75.2 s, saved once (73.1 s) then 2.36 s,
relaunched after unrelated load 1.09 s (100% cached); 6 agents resume 0.43 / 0.42 / 0.60 s (in memory / after load /
after SSD offload). Splash b/c: queue job 120 (pending).

## Qwen3.5-9B: inferd 0.2.0 vs TensorFold 1.2.2 (both M6s, same checkpoint, no drafter)

| follow-ups, total tok/s | inferd 4-bit | TF 4-bit | inferd 8-bit | TF 8-bit |
|---|---|---|---|---|
| 4k x 1 | 29.8–30.0 | 29.9–30.2 | 17.2–17.3 | 16.6–16.7 |
| 4k x 4 | 75.3–75.7 | **98.1–98.9** | 44.3–44.6 | **57.2–57.6** |
| 4k x 8 | **131.1–131.9** | 99.4–99.6 | **81.6–81.8** | 57.7–58.0 |
| 4k x 16 | **208.8–209.2** | 92.0–98.1 | **137.3–139.9** | 25.7–31.0 |
| 16k x 1 | 28.3–28.4 | 26.6–27.2 | 16.7–16.8 | 15.8 |
| 16k x 4 | 65.1–67.2 | **70.9–72.7** | **41.1–41.2** | 13.9–22.0 |
| 16k x 8 | **109.6–109.7** | 27.0–27.4 | **72.0–72.4** | 9.1–10.3 |
| 16k x 16 | **157.8–159.7** | 22.6–22.8 | **114.1–115.1** | 8.7 |

TensorFold loads the 9B with 7 streams of 32k and a 6.3 GiB prompt cache; from 8 agents at 16k (16 at 4k) the cache
no longer holds the contexts and every turn is re-read (0 cached tokens; slowest first token 336–905 s vs inferd 3 s).

## Where others lead on the M6

Splash: one agent (prefill ~1.8x, decode ~10%), 1–4 agents at 4k, 27B 16k fleets of 8–16 agents (+31–38%).
TensorFold: the 9B with 4 agents (+31% at 4k).
