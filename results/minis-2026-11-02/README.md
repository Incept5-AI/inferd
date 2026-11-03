# Mac mini runs, 2 November 2026 (evening): Qwen3.8-27B 4-bit + DFlash2

Raw outputs of `decode_test bench`, `speed.py`, `conc.py --mode followup` and `bench/wall.py` (validated SSE client).
Each directory has the inferd startup memory line (`server.txt`).

| Directory | Machine | Build | Server (running / snapshots / max ctx) | What |
|---|---|---|---|---|
| m6perf | M6 32 GB | engine-m2 (before the drafter-memory cut) | 8 / 8 / 32k | raw set |
| m6merge | M6 32 GB | merged tree (476c54c) | 8 / 8 / 32k | same raw set |
| m6fleet | M6 32 GB | 4feb7f5 | 16 / 8 + SSD tier / 32k | follow-ups 4k and 8k × 1/8/16 |
| m4final | M4 Pro 64 GB | merged tree (476c54c) | 16 / 16 / 64k | raw set |
| m4merge | M4 Pro 64 GB | merged tree | 16 / 16 / 64k | 16k × 8/16 (the cell that collapsed to 5.0 tok/s before prefix retention) |
| m4nodraft | M4 Pro 64 GB | merged tree, no drafter | 16 / 16 / 64k | follow-ups 4k and 16k × 8/16 |
| m5pfleet | M5 Pro 64 GB | 4feb7f5 + vision margin | 32 / 48 / 40k | follow-ups 16k × 1/8/16/32, 32k × 1/8/16 |

The M4 Pro has no GPU neural accelerators (MLX simdgroup GEMMs, bf16 activations); the M5 Pro and M6 use the NAX paths.
| m5praw | M5 Pro 64 GB | a4ee879 | 16 / 16 / 64k, VISION=1 | raw set (as m4final), then the examples suite |
| m6fleet16 | M6 32 GB | a4ee879 | 16 / 8 + SSD tier / 32k | follow-ups 16k × 4/8/16 |
| m4c16 | M4 Pro 64 GB | 476c54c tree | 16 / 16 / 64k | follow-ups 16k × 1/4 |
| m4ab | M4 Pro 64 GB | engine-m2 vs merged, alternating | 16 / 16 / 64k | 16k × 1/4 A/B: no regression |
