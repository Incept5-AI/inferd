# v0.2.0 prebuilt tarball on the Mac Studio M5 Ultra (256 GB), 4 November 2026

The release tarball (sha256 verified) unpacked over this tree's `engine/`; Qwen3.8-27B 4-bit (mlx-community) +
DFlash2 (incoai).

- `gate*.{txt,log}`: `tools/release_gate.sh` passed: host and front-end tests, logits at chunks 1 and 24 (top-1 ok),
  bench at L = 2048: 19.9 / 30.0 / 69.6 ms per step at B = 1 / 16 / 64, speculative = plain.
- `chunks-256-1536.txt`: logits at 256- and 1,536-row chunks on 1,536-token prompts: top-1 ok, cos 0.9992–0.9995.
- `conc-followup.*`: `bench/studio/conc.py --mode followup`, 16k contexts, 512-token replies, server
  `run.sh 8320 64 2048 140 96 262144` (explicit 140 GiB KV pool), total tok/s, 0 errors:

| 16k contexts | 1 agent | 8 | 32 |
|---|---:|---:|---:|
| total tok/s | 158.5 | 492.3 | 619.9 |
| last agent's first token (s) | 0.10 | 0.59 | 2.25 |
