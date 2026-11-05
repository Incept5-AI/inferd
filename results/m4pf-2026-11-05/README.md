# M4 Pro prefill, 5 November 2026 (RESULTS.md section (v))

Mac mini M4 Pro (20-core GPU, 64 GB, macOS 27.0.1), Qwen3.8-27B 4-bit (mlx-community) + DFlash2.

| Files | What |
|---|---|
| `prefill-base.txt`, `prof-base.txt` | `decode_test prefill` 8,192 tokens at chunks 1,024 / 2,048, and `PROF_KERNELS=1` at 2,048 (19805b9) |
| `prefill-qmmh.txt`, `prof-qmmh.txt` | the same with an fp16-fragment GEMM that was rejected (not in the repo) |
| `ane-sweep.txt`, `ane-sweep2.txt` | `ANE_UNITS` 0-34 on the same prefill |
| `single-*.txt`, `single.jsonl` | `conc.py --mode cold`, one agent at 2k / 8k / 32k: `base` (19805b9), `cand` (split on by default, on 0.3.0), `cand2` (the same rebased on main, 8k only) |
| `followup-*.txt`, `followup.jsonl` | `conc.py --mode followup` at 8k, 1 / 4 / 16 agents (`cand2`: 4) |
| `mlxlm-*.txt` | `mlx_lm.benchmark -p <tokens> -g 16 -n 2` (mlx-lm 0.32.2, mlx 0.32.3) |
| `gate-m4pro-ane-default.log` | `tools/release_gate.sh` with the drafter, split on by default |
| `progress.log` | the session's log (server memory lines, summaries) |

Server: `run.sh 8320 16 1024 0 8 40960` with `DRAFT`, SSD tier on.
