# v0.2.0 prebuilt tarball on a Mac mini M4 Pro (64 GB): fleet numbers, 4 November 2026

The release tarball (sha256 verified), lmstudio-community Qwen3.8-27B 4-bit, **no drafter** (run.sh's advice for
M3/M4 fleets of 16+ agents), server `run.sh 8320 16 1024 0 16 32768`. `bench/studio/conc.py --mode followup`,
512-token replies, total tok/s, 0 errors (`conc-followup.*`):

| contexts | 1 agent | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| 4k | 15.6 | 39.0 | 41.0 | 71.3 |
| 16k | 14.3 | 31.6 | 33.9 | 54.8 |

One agent is faster with the drafter on this machine (25–32 tok/s; the self-tests in this directory used it).
