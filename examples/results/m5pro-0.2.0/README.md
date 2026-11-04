# v0.2.0 prebuilt tarball on a Mac mini M5 Pro (64 GB), 4 November 2026

The release tarball (sha256 verified) unpacked over this tree's `engine/`; Qwen3.8-27B 4-bit (mlx-community) +
DFlash2 (incoai); server `run.sh 8320 16 1024 0 32 32768` with the SSD tier on.

- `gate.txt`: `tools/release_gate.sh` passed: host and front-end tests, logits at chunks 1 and 24 (top-1 ok), bench
  1/16/64 at 2048, speculative = plain.
- `chunks-256-1536.txt`: logits at 256- and 1,536-row chunks on 1,536-token prompts: top-1 ok, cos 0.9992–0.9994.
- The rest: `examples/run_all.sh` self-tests, 13 / 13 PASS (coding-smoke not run here; it passed on the M4 Pro).

Fleet numbers (`bench/studio/conc.py --mode followup`, follow-up turns with 512-token replies, server
`run.sh 8320 16 1024 0 32 32768`), total tok/s, 0 errors, `conc-followup.*`:

| contexts | 1 agent | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| 4k | 52.7 | 122.1 | 166.9 | 190.1 |
| 16k | 46.6 | 115.0 | 150.4 | 169.3 |
