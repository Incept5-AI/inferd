# Same-day quality runs, Mac Studio M5 Ultra, 2 November 2026

languages-benchmark v1 (48 tests: 8 challenges x 6 languages), Qwen3.8-27B, effort medium, 32k max tokens, 16 workers,
3 runs per arm, same day and machine. `v1-<arm>-r<n>.json` are the benchmark's own result files (`total_score`, per test
`records`); `bench-<arm>.log` the run logs. Scores (total of 48):

```
base     MLX 4-bit, inferd tree3 (DFlash2, W4A8, KV8)   [42.5, 43.5, 45.0] mean 43.7  2026-11-02
splash   MLX 4-bit, Splash 1.5.0 (defaults: DFlash2, in [39.5, 41.5, 44.5] mean 41.8  2026-11-02
nodraft  MLX 4-bit, inferd tree3, no drafter (W4A8, KV8 [41.5, 42.0, 43.0] mean 42.2  2026-11-02
q8       MLX 8-bit, inferd (DFlash2)                    [44.0, 43.5, 44.5] mean 44.0  2026-11-02
bf16num  MLX 4-bit, inferd tree3, W4A8=0 KV8=0 ATTN_IQ= [42.5, 44.0, 39.5] mean 42.0  2026-11-02
oq4e     Jundot oQ4e (5-bit parts widened to 8), inferd [42.5, 41.5, 45.0] mean 43.0  2026-11-02
lm4      lmstudio-community 4-bit, inferd (DFlash2)     [41.5, 44.5, 41.0] mean 42.3  2026-11-02
lm5      lmstudio-community 5-bit (widened to 8), infer [43.0, 44.0, 42.5] mean 43.2  2026-11-02
lm6      lmstudio-community 6-bit (widened to 8), infer [42.0, 43.0, 43.5] mean 42.8  2026-11-02
```

The website's 43.7 vs 41.8 is `base` (inferd default) vs `splash` (Splash 1.5.0 defaults). The spread between runs is about ±3,
so the two are level within run-to-run noise; the other arms are the quality ablations (no drafter, bf16 numerics, other
checkpoints). Scrubbed: the user name and host name in the system_info task outputs.
