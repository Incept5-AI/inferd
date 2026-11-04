# inferd roadmap

This is what could still be improved, as of 2026-11-01 (release 0.1.1). Each item gives the evidence, with the
study-log section title in quotes or a file in [`results/`](../results), the expected payoff, and a
rough difficulty.

- **P0:** a correctness or robustness problem, or a gap that blocks the main use case (agent fleets on Qwen3.8-27B).
  Do these first.
- **P1:** a large, measured performance or memory win, or the next model.
- **P2:** worthwhile, but speculative or small.

Difficulty: **S** is days, M is about a week, L is weeks or open-ended research.

## Where we stand

On the Mac minis, inferd beats Splash 1.5.0 on agent fleets using the same weights and drafter ("Overnight agent-fleet
results").

| Machine | Context | inferd (turns/min) | Splash 1.5.0 (turns/min) |
|---|---|---|---|
| M6 | 8k | 9.5–9.7 (11.0–11.3 with the ANE split) | 6.8–8.9, and refused 32 of 128 turns at 64 agents |
| M6 | 16k | 8.2–9.3 | 6.8–8.2 |
| M5 Pro | 16k | 13.5–14.3 | 11.9–12.0 |
| M5 Pro | 32k | 11.0–12.8 | 10.3–10.6 |

On the Mac Studio, inferd leads from 4 agents up at 16k–64k ("Milestone inferd10": 16k × 16 is 563 tok/s against
369). One agent is close but not consistently ahead: 16k × 1 is 147 vs 172, and 256k × 1 is 95–111 vs 132. 128k × 4 is
level on a fresh server and behind after a long uptime.

---

## P0

### P0-1. Long-uptime slowdown at 128k × 4 (unexplained)
- **Evidence:** "Overnight agent-fleet results". A fresh server gives 256 / 249 tok/s, against 225 at the end of a
  21-cell sweep (270 best on 31 Oct). Two earlier causes were found and fixed: pages handed out in descending order
  ("Diagnosed 2026-10-30: a full KV pool…") and scattered groups ("v11: grouped KV allocation"). Something else
  remains.
- **Approach:** reproduce with a scripted long run, then compare `decode_test bench` with `BENCH_SHUFFLE=1` against
  the live server's step times. Log the allocator's group-contiguity (`ALLOC_LOG`) and the trie/snapshot sizes over
  time. Check whether the tuner's per-regime table (`SPEC_AUTO`, `INFERD_DEBUG=1`) drifts after many regimes.
- **Payoff:** 10–15% at long context on any server that stays up. **Difficulty:** M.

### Done 1 Nov 2026 (from a code review by OpenAI Codex)

- **F04** the SSD tier only ever clears or removes `<SPILL_DIR>/inferd-spill`, never `SPILL_DIR` itself.
- **F01** an SSD entry being restored is pinned (quota eviction skips it) and re-found before loading.
- **F06** the prefix-cache hash index drops empty buckets (it grew with every evicted prefix).
- **F03** the speculative scratch is allocated before the KV pool is auto-sized, and auto sizing keeps 12% of the
  budget free above 64 GiB (1.5 GiB on small machines, as before).
- **F10** if `inferd` dies, open requests end with an error and new ones get HTTP 503; `/health` returns 503.
- Per-agent `max_context` and `priority` (engine/README.md).

Second pass (same evening; Studio GPU verification pending at push time): R01 pending snapshots can't be evicted or
spilled before their GPU copy; R02 exact context limit and bounded verify width; R03 free-stack compaction; R04 engine
errors become API errors / SSE error events, failed submits clean up, `bench/agents.py` rejects failed streams; R05
no double-counted output after preemption; R07 SSD owner marker, recounted quota, rollback; R08 hardened JSON parser,
`seed: 0`; R09 SSD refused for Flash-Next/Gemma, Flash-Next limited to 2k tokens; F12 image URLs/paths off by default;
F13 socket 0600. Details: the study log "Codex review, second pass".

Still open from the review (P1 unless noted): F05 one
sampling contract (top-k capped at 16 on the speculative path, CPU path differs; `seed=0` ignored); F07 SSD quota can
be exceeded and is not transactional; F08 SSD cache identity/integrity (weights replaced at the same path); F09 bound
and validate every input (queue sizes, token ids, sampling values); F11 admission looks only at the queue head
(head-of-line blocking; add lookahead with ageing); F12 vision fetches arbitrary URLs/paths (SSRF) and keys images
by a 24-bit hash; F13 socket permissions and checkpoint validation; F14 front-end cache bounds and quadratic
detokenization. Engineering suggestions worth measuring: asynchronous SSD spill/restore (E01), adaptive active
concurrency (E02), `ref_logits.py` and `persist_test.sh` should fail on mismatch, per-cell build provenance in reports.

### P0-2. Auto KV sizing can hang the Mac Studio (mitigated 1 Nov: 12% headroom above 64 GiB)
- **Evidence:** "Diagnosed 2026-11-01: a server that hung on its first request…". With a small model, auto KV
  (`kv_gib 0`) took 222.5 of the 224 GiB budget, and the first prompt sat in an uninterruptible GPU wait.
  `tools/server.mm` leaves only `KV_MARGIN_GIB` (1.5) plus the ANE margin.
- **Approach:** cap the auto pool. Scale the margin with the budget, or cap the pool at the KV that `max_running ×
  max_ctx` could ever use. Add a start-up check that refuses a pool leaving less than N GiB for scratch growth.
- **Payoff:** removes an uninterruptible hang on the flagship machine. **Difficulty:** S.

### P0-3. Tool calls and structured output in the OpenAI API
- **Status (1 Nov 2026): tool calls done.** `serve.py` parses the Qwen `<tool_call>` markup into
  `message.tool_calls` and streaming `delta.tool_calls` (one final delta, `finish_reason: "tool_calls"`), converts
  earlier calls' JSON-string arguments back for the template, and has a unit test (`engine/frontend/test_tools.py`).
  Used by [examples/tools-web-search](../examples/tools-web-search/) and [examples/tools-shell](../examples/tools-shell/).
- **Structured output, phase 1 (2 Nov 2026):** `response_format` `json_object` / `json_schema` with schema validation and up to 2 retries (`frontend/structured.py`, HTTP 422 `json_validation_failed` when every attempt fails); 20/20 valid on the first attempt in the live set. Phase 2 (2 Nov 2026): `strict: true` is constrained decoding for a documented subset of JSON Schema (`src/grammar.h`), followed by the same schema validation; unsupported or over-budget schemas get HTTP 400. Greedy stays speculative (block tokens checked on the host, a rejected one replaced by its row's masked argmax); sampled strict requests are not speculated once the grammar is active (host sampler, frequency-tested). Strict live set: 20/20 first attempt, n = 20 schemas x 2 modes. Keyword table: [examples/structured-output](../examples/structured-output/#the-supported-subset).
- **Still open:** user `stop` strings, and `temperature` defaults to 0.7 when absent.
- **Approach:** accept `stop` strings. For JSON mode, start with a validate-and-retry, then grammar-constrained decoding. That needs a token
  mask in the sampler, which interacts with speculation (draft tokens must be checked against the mask).
- **Payoff:** agent frameworks can use inferd unmodified. This is the stated use case. **Difficulty:** S (parser), L
  (constrained decoding with speculation).

### P0-4. Re-run the 120-test coding benchmark (v2)
- **Evidence:** `results/studio/code-v2.log` shows that the run started, the server came up, and nothing more. Only
  v1 (48 tests) has results (`results/code/`; 42.5/48 vs Splash 42.0, "0.1.1"). The decision log asks for quality to
  be "held on HLE and codebench2" ("Decisions (2026-10-29)").
- **Payoff:** the quality evidence for 8-bit KV, W4A8 and speculative sampling at scale. **Difficulty:** S (machine
  time).

### P0-5. Robustness of the two-process server
- **Evidence:** in `serve.py`, `Engine.pump` exits silently when the `inferd` socket closes, so open requests wait
  forever, and `/health` returns `{"ok": true}` without asking `inferd`. Preemption (`Scheduler::preempt`) has never
  fired under test: the run that validated admission logged 0 preemptions ("v10: KV-budget admission + preemption").
- **Approach:** fail in-flight requests with a 5xx when the socket drops, have `/health` round-trip to `inferd`, and
  add a supervisor (launchd) restart policy. Write a test that forces preemption (a tiny pool with `ADMIT_RESERVE=0`
  and long generations) and checks replies against a large-pool reference.
- **Payoff:** no hung clients, and confidence in the preemption path. **Difficulty:** S–M.

### P0-7. Prefix retention is capped by state snapshot slots (diagnosed 2026-11-01)

The prefix cache keeps at most `snapshots` cached prefixes, even for models with no recurrent state (Muse Glimmer,
Gemma-4), where the slots cost nothing. With 16 agents at 32k and 16 slots, agents' prefixes were evicted and
follow-ups re-prefilled serially: worst TTFT 112–503 s on the Mac minis ([RESULTS, other models](RESULTS.md#e2-mac-minis)).
48 slots fixed it (35B-A3B on the M5 Pro: 57 → 174 tok/s at 16 agents). Fix: retain prefixes by KV budget alone when a
model has no state, and size `snapshots` from the agent count by default for hybrid models. Easy, high payoff.

### P0-6. Keep vision working: automate the check
- **Evidence:** vision was verified once, word for word against `mlx_vlm.generate` ("Vision (2026-10-29 night)").
  Several kernels and the KV layout have changed since (head planes, 8-bit KV default, chunk tiles), with no re-check
  recorded.
- **Approach:** script the synthetic-image check from [TESTING §2.4](TESTING.md#24-vision), then add multi-image,
  multi-agent and SSD-tier cases.
- **Payoff:** catches a silent regression on a must-have feature. **Difficulty:** S.

---

### P0-7b. Re-measure speculative throughput after the drafter scratch fix

The W4A8 scratch overflow fixed on 1 Nov (commit `abdc023`) silently lowered drafter acceptance on every machine
(M5 Pro spec_test 5.67 -> 8.12 tokens per cycle). Re-run the Studio comparison and the mini fleets on one build in one
session; this should both raise and clean up the headline numbers (one build per table, not the per-cell mix
`reports/build_compare.py` produces).

### P0-8. Test the M4 path properly

inferd now runs on an M4 Pro (1 Nov 2026, [MACHINES.md](MACHINES.md#mac-mini-m4-pro-works-lightly-tested-1-november-2026)),
but only a smoke pass: test 32k+ contexts, 16+ agents with the drafter, the SSD tier, vision and long uptime there; find
why some no-drafter requests stopped early; try a non-NAX quantized GEMM for prefill (W4A8 only reaches ~106 tok/s).

### P0-9. Qwen3.8-Flash-Next: from stage 1 to usable

Runs with MTP drafting (`MTP=1`): 114-118 tok/s single, 637 at 16 agents x 1k on the Studio (exact up to 2k
tokens). Next: the QSA sparse-attention indexer for > 2k contexts (target and MTP layer), K per load regime (K5-K6
measured faster single-stream), fewer dispatches (a decode step is ~1,250 dispatches; hyper-connections ~27% and MoE
~32% of a B=1 step), SSD spill of the PLE conv and MTP state, then NVIDIA's QAD FP4 checkpoint (FP4/FP8 kernels).

## P1: performance

### P1-1. Single stream: close the remaining cycle gap to Splash
- **Evidence:** "Closing the gap, continued" and "Milestone inferd10". The cycle is about 29.1 ms against Splash's
  about 24.5 ms at the same acceptance (5.3 tokens/cycle at 8k). About 3 ms of the gap is the 8-row GEMMs, and the
  rest is GDN, the drafter and overheads (about 1,377 dispatches per cycle at about 3–4 µs each). 16k × 1 is −15%;
  256k × 1 is −16 to −28%.
- **Ideas not yet done:** per-shape split laws for `qmst` ("Splash's source") once its drafter path is re-measured
  (the bug was fixed but not re-timed, "QMST drafter bug: cause found"); fusing more of the per-layer small kernels;
  explaining why `LANE=1` lowers drafter acceptance (2.82 → 2.33, cause not found).
- **Payoff:** +15–20% single stream. **Difficulty:** L.

### P1-2. Prefill, especially at long context and on the minis
- **Evidence:**
  - Fleet turns at long context are prefill-bound: about 21k prompt tokens per round at 64k × 16, and decoders stall
    behind the chunks ("Agent fleets (2026-10-30 evening)").
  - About 2/3 of a mini fleet's wall time is prefill ("Neural Engine FFN split").
  - The 128k sweep TTFT regressed by about 12% and nobody has explained why ("2026-10-31 morning").
  - 256k prefill is 767 tok/s against Splash's 847.
  - A 2,048-row chunk already runs at about 90% of bf16 peak via MLX qmm, and an int8 large-M tile was 4× slower.
- **Ideas:** TensorFold's prompt kernel (two tiles per simdgroup), which beats our MLX-qmm prefill ("Lessons from the
  mlx.fast competition"); an int8 prefill tile design that actually beats the bf16 ceiling; better interleaving of
  prefill chunks with decode steps so decoders don't stall.
- **Payoff:** TTFT and turns/min at 64k–256k, and directly on the minis. **Difficulty:** L.

### P1-3. Neural Engine prefill split: finish it
- **Evidence:** "Neural Engine FFN split for prefill" and "Diagnosed 2026-11-01: the ANE split…". It's +12–23% on
  the M6 and only +3% on the M5 Pro. **`ANE_A8D` is broken:** int8 down gives inf in layer 0 and zeros after. The
  suspect is a dequantization factor dropped in the int8 × int8 down matmul. `tools/ane_rot_test.mm` checks the parts
  in isolation; the next thing to check is the `[H, C] × [C, R]` shapes.
- **Next steps (from the study log):** fix `ANE_A8D`; overlap the next layer's re-quantization with the ANE; shared-event sync
  instead of host waits.
- **Payoff:** M6 prefill +29% at the microbenchmark level (more if A8D works); possibly makes the split worth it on M5
  Pro/Max. **Difficulty:** M.

### P1-4. Decode at 8–64 agents: wide GEMMs
- **Evidence:** "Diagnosed 2026-10-31: the W4A8 epilogue is not what holds wide GEMMs back" and "Wide GEMMs: tile
  variants tried and rejected". At M = 64 the int8 path runs at 56–90 TOPS of 223, and the limit is one 32×128×64
  `matmul2d` per scale group. W8A8 is no better per FLOP.
- **Ideas left in the study log:** stage decoded weights in threadgroup memory and reuse them across row tiles. Group-free
  scaling would allow longer K runs, but it isn't available with g64 weights. Rejected so far: dual int8 weight
  residency, 64-row tiles, 8-simdgroup tiles, staged scales, two MMAs in flight.
- **Payoff:** up to about 2× on the GEMM share of a 16–64-agent step, if the int8 path reached even 50% of peak.
  **Difficulty:** L (research).

### P1-5. Long-context verify attention for many agents
- **Evidence:** at 128k, 16 agents' attention is compute-bound at about 47–50 TFLOPS ("Agent fleets…", "Chunk tiles
  (mode 6)"). Cascade attention over shared prefixes is correct but slower (`CASCADE=1`, off). KV8 doesn't help the
  multi-agent steps of the old kernel.
- **Ideas:** revisit cascade once attention is closer to bandwidth; key-split tiling tuned for 16–32 agents at 128k;
  `attn_mpp` beyond GQ = 6 so the other models get it (P1-8).
- **Payoff:** 10–20% at 128k × 8–16. **Difficulty:** M–L.

## P1: memory

### P1-6. bf16 GDN state
- **Evidence:** "Where a 27B decode step goes", item 4. The recurrent state is fp32 (`Qwen35::slot_bytes`): 144 MiB
  per sequence, read and written every step, about 18 ms of a B = 64 step. Storing it in bf16 halves both the memory
  and the time, but **needs a quality check**, and none has been done.
- **Approach:** a `STATE_BF16` knob in the GDN kernels, snapshots and spill files (the spill MANIFEST must record it);
  `a8_quality.py`-style KL checks over long sequences, because recurrent errors accumulate; then the coding suite.
- **Payoff:** about 2× more snapshot and running slots in the same memory (it matters most on the M6), and a faster
  GDN at high B. **Difficulty:** M.

### P1-7. Sliding-window KV (ring blocks) and KV formats per model
- **Evidence:** "Other models on inferd". Gemma-4-31B needs bf16 KV (8-bit gives KL 1.96, because attention scale 1
  on normed q/k amplifies int8 key error about 20×). That's about 900 KB per token, so 16 × 32k would need about
  460 GB, and 32k is capped at about 5 agents. 50 of its layers are sliding-window, so ring blocks would cut that
  sharply. The conflict: ring blocks don't fit the prefix cache's immutable shared blocks.
- **Ideas:** ring blocks only for sliding layers, with snapshot semantics for the ring (like GDN state); a KV format
  with higher-precision keys (bf16 K + int8 V, or per-channel key scales) for models with attention scale 1. That's
  our suggestion; it hasn't been tried.
- **Payoff:** Gemma-4 at 16 × 32k (the stated target); Glimmer (window 2,048) gets the same memory saving.
  **Difficulty:** L.

### P1-8. Elastic KV on small machines
- **Evidence:** the 2026-11-01 correction in "0.1.1: speculative sampling". Splash averages 25.0 GiB wired on the M6
  against our 22.0. Its elastic KV takes the whole budget, while ours is a hand-sized 3 GiB beside worst-case scratch.
- **Payoff:** about 10–30% more resident context on the M6 (an estimate), so fewer SSD swaps. **Difficulty:** M.

## P1: features

### P1-9. Qwen3.8-Flash-Next
- **Evidence:** it's the next target model ("inference — vLLM-class concurrency…", CLAUDE.md). The architecture is in
  "The models are both hybrid GDN": 48 layers, 12 attention, 36 GDN, 512 experts top-10, about 3B active. The MoE
  path already exists for Qwen3.6-35B-A3B (`src/moe.h`, `kernels/moe.metal`: routed GEMV under 128 rows, a counting
  sort plus MLX gather-qmm above). The hypothesis from day 1: the batched MoE kernel, not bandwidth, is the limit, with
  a ceiling of about 1,000 tok/s at B = 16 against 269–324 measured on other Studio engines.
- **Work:** config/loader for its layout; KV heads 2 × 256 (attention kernels templated on GQ, P1-10); MoE at
  512 experts with top-10; its MTP head (below).
- **Payoff:** the second flagship model. **Difficulty:** L.

### P1-10. Drafters / MTP for the other models, and tuned kernels
- **Evidence:** "Other models on inferd". The 9B, 35B-A3B, Glimmer and Gemma ports have no drafter, and the fast
  `attn_paged_nax`/`attn_mpp` kernels are GQ = 6 / Dh 256 only. The 27B mlx-community 4-bit build has no MTP head.
- **Work:** an MTP head as a drafter (Flash-Next first, then any checkpoint that ships one); DFlash-class drafters
  where they exist; template `attn_mpp` on (GQ, Dh) as `attn_gqa` already is.
- **Payoff:** 2–3× single-stream for those models (the 27B went from 46.5 → 87.5 tok/s per stream at 16k with
  DFlash2). **Difficulty:** M per model.

### P1-11. 8-bit and bf16 27B as first-class configurations
- **Evidence:** "Weight precisions". Logits, `decode_test` and `spec_test` work at all three precisions, but no server
  fleet or quality run uses 8-bit or bf16. bf16 decode at B = 32 is 357 tok/s against about 600 for 4-bit.
- **Work:** a milestone run at 8-bit (the vision-capable precision), and a check of the W8A8 path under load.
  **Difficulty:** S.

---

## P2

### P2-1. One weight layout for all GEMMs (lane4/tile-major)
- **Evidence:** "Lessons from the mlx.fast competition". lane4 needs a repacked **copy** of the weights, so it's
  restricted to ≥ 96 GB and is currently off. To drop the copy, `qmv4`, W4A8 (M > 16) and the prefill GEMM all have to
  read the tile-major layout. **Payoff:** memory on the minis, and simpler routing. **Difficulty:** L.

### P2-2. Drafter vocabulary cut done right
- **Evidence:** "Drafter vocabulary cut reverted to opt-in". It saves about 1.4 ms per cycle on the laptop and about
  0.5 ms on the Ultra, but costs acceptance (4.46 → 4.15 at 16k). A cut chosen from the target's actual token
  frequencies might keep both. **Difficulty:** S–M.

### P2-3. SSD tier: spill in the background and survive more restarts
- **Evidence:** "SSD tier made content-addressed" and "Persistent SSD tier". Spills happen between steps with the GPU
  idle; the study log lists "spill in the background" as a next step. Persistence has been tested once.
  **Difficulty:** M.

### P2-4. Speculative sampling: lift the top-k ≤ 16 cap
- **Evidence:** `engine/README.md` and "0.1.1". The sampled path uses the GPU top-16 per verify row. Since 2026-11-04
  (Codex review R03) requests it cannot draw exactly (top-k 0 or > 16) decode one token per step on the host sampler
  instead of being truncated silently, so the default request (temperature 0.7, top-k 0) is no longer speculated.
  Lifting the cap means a full-vocabulary softmax (log-sum-exp per verify row) and a full-vocabulary resample on the
  GPU. **Difficulty:** M.

### P2-5. Batch invariance (deterministic output across load)
- **Evidence:** "DFlash2" lossless check. The engine isn't bit-level batch-invariant, and W4A8 makes speculative and
  plain decode differ at near-ties. Useful for reproducible evals. **Difficulty:** L.

### P2-6. Packaging and release
- **Evidence:**
  - `engine/run.sh` hard-codes a venv path (`~/.venvs/mlxvlm`), a model path on the original machine's external volume, and `~/inference/logs`.
  - The engine socket lives at `/tmp/inferd-<port>.sock`.
  - `bench/` scripts carry machine-specific paths.
  - `build.sh` prints two `-Wformat` warnings (`scheduler.h`).
  - There's no CI, no versioned binary release, and no launchd plist.
  - Neither `requirements.txt` nor the README lists `mlx-lm`, which `tools/ref_logits.py` imports.
- **Work:**
  - Make the paths configurable (`INFERD_LOG_DIR`, required `MODEL`).
  - Ship a signed tarball or Homebrew formula with a prebuilt `core.metallib`.
  - Add a launchd service.
  - Add a CI job that builds, then runs `decode_test check` and `bench` on a self-hosted M5 runner.
  - Generalise the bench drivers.
- **Difficulty:** S–M per item.

### P2-7. Configuration hygiene
- **Evidence:** there are about 80 environment knobs (`grep getenv engine/src engine/tools/server.mm`), many of them
  timing-only or dead experiments (`ATTN_V1`, `DEFER_SPLIT`, `NAX_X`, `QMVT_MAX`…). The docs disagree with the code in
  places: CLAUDE.md says `SPEC_WIDE` = 2, while `scheduler.h` defaults it to 4.
- **Work:** one table of supported knobs; delete or hide the experiment-only ones.
- **Difficulty:** S.

---

## Tried and rejected (don't repeat without a new idea)

the study log records each of these with numbers. In brief:

- Adaptive block sizes (`SPEC_BLOCKS` by load) before the tuner existed.
- Draft block 8 for one stream before the `draft_attn` fix.
- `qmvt4`, `qmvh`, `qmvc` and `qmvs` small-M kernels.
- 16/48/64-row W4A8 tiles for wide M.
- Split-K on wide shapes.
- Multi-tile NAX attention for follow-up prefills.
- A large-M int8 prefill tile.
- Cascade attention (correct, slower).
- `DEFER_SPLIT`.
- `RESIDENCY=1` without the ANE.
- `lane8`.
- A dual int8 weight copy.
- Scales staged in threadgroup memory.
- Touch-prefetching weights.

Several of these failed for a specific, recorded reason. Read the entry before you reopen one.
