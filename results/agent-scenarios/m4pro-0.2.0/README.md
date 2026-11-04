# Agent scenarios on a Mac mini M4 Pro (64 GB), v0.2.0 prebuilt tarball (4 November 2026)

`bench/agents_scenarios.py --mode api --preset m5pro --scenario all`; lmstudio-community Qwen3.8-27B 4-bit, **no
drafter** (run.sh's advice for M3/M4 fleets); server `run.sh 8320 32 1024 0 32 65536` with the SSD tier (48 GiB).
0 errors in every phase. The M4 Pro has no GPU neural accelerators and reads prompts at ~120–140 tok/s, so anything
that needs a new prompt read is slow here (first turns of a/d/e: 3–7 minutes at this preset's sizes); what the cache
keeps is fast:

| scenario | result |
|---|---|
| a: 48 small agents × 3 turns | 144 turns in 410 s; follow-ups 97% cached, 16.9 s median first token |
| b: 16 agents forked from a long context | cold 206 s; saved once: 6.6 s; relaunched after unrelated load: 6.8 s, 100% cached |
| c: 8 agents resume | resident 0.51 s; after unrelated load 0.52 s; after an explicit offload to SSD 0.59 s |
| d: 4 long-lived agents beside 400 one-offs (`cache: drop`) | long-lived turns 11.2 s median, 99% cached |
| e: 6 researchers + 32 helpers | wall 507 s (prefill-bound on this machine) |
