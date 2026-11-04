# Agent scenarios on the Mac Studio M5 Ultra, v0.2.0 prebuilt tarball (4 November 2026)

`bench/agents_scenarios.py --mode api --preset studio --scenario all`; Qwen3.8-27B 4-bit + DFlash2; server
`run.sh 8320 64 2048 140 96 262144` with the SSD tier (64 GiB). 0 errors in every phase (`scenarios.txt`, rows in
`rows.jsonl`).

| scenario | result |
|---|---|
| a: 64 small agents × 3 turns | 192 turns in 72.6 s (follow-ups 97% cached) |
| b: 32 agents forked from a 25k-token context | cold 15.5 s first token; saved once (14.5 s): 1.08 s; relaunched after 12 × 120k of unrelated prompts: 1.03 s, 100% cached |
| c: 16 agents resume | resident 0.09 s; after 12 × 120k of unrelated prompts 0.09 s; after an explicit offload to SSD 0.97 s |
| d: 6 long-lived agents beside 2,000 one-off requests (`cache: drop`) | long-lived turns 2.45 s median, 100% cached, while one-offs ran at 3.7 requests/s |
| e: 8 researchers (effort low, priority 5) + 64 helpers (effort none) | helpers 6.7 s median first token, researchers decode 91.7 tok/s per stream, wall 86.8 s |
