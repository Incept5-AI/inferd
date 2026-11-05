# Machines

The hardware inferd was built and measured on between 29 October and 1 November 2026, how the engine behaves on each,
and the settings we recommend. Benchmark numbers are in [RESULTS.md](RESULTS.md); the reasoning behind each setting is
in the study log (summarised in [EXPERIMENTS.md](EXPERIMENTS.md)).

## The machines

| Machine | GPU | Memory | GPU memory budget | Measured memory speed | Role in the study |
|---|---|---|---|---|---|
| Mac Studio M5 Ultra | 80-core | 256 GB | 224 GiB (Metal's recommended working set) | 1,135–1,142 GB/s streaming read | Main development and benchmark machine |
| Mac mini M6 | 12-core | 32 GB | ~25 GiB (stock limit; Splash's `--max-memory 25559M` matches it) | ~159 GB/s effective in plain decode | Small-machine fleets; the Neural Engine split |
| Mac mini M5 Pro | 20-core | 64 GB | 51.8 GiB (stock limit, from the inferd start-up log) | ~292 GB/s effective in plain decode | Small-machine fleets; kernel checks |
| MacBook Pro M5 Max | 40-core | 128 GB | stock limit (not logged) | ~520 GB/s achieved by the `lane4` GEMM | Development laptop: builds, kernel experiments |
| Mac Studio M3 Ultra (**lightly tested**) | 80-core, no neural accelerators | 512 GB | 464 GiB | not measured (decode step 27 ms) | Portability and non-NAX tuning (1 Nov 2026) |
| MacBook Pro M4 Max, 36 GB (external tester, macOS 26.6.2) | no neural accelerators | 36 GB | 28 GiB | – | v0.4.0 macOS 26 prebuilt: sha256 ok, `install.sh --model 27b` found the models, the full release gate passed (bench skipped below 32 GiB of GPU budget), `run.sh start --model 27b` (auto KV 6 GiB) answered, `examples/run_all.sh` 15/15 PASS. The Neural Engine split stays off on Max chips (default only for M4 / M4 Pro). (5 Nov 2026) |
| Mac mini M4 Pro (**lightly tested**) | 20-core, no neural accelerators | 64 GB | 51.8 GiB | ~225 GB/s effective in plain decode (15 GB at 15 tok/s) | Portability check (1 Nov 2026) |
| MacBook Pro M4 Max (**lightly tested**, external tester) | 40-core, no neural accelerators | 64 GB | not logged | not measured (decode step 34 ms) | First source build on an M4 Max with macOS 27 (2d2a108, 4 Nov 2026) |
| MacBook Pro M3 Max (**lightly tested**, external tester) | 30-core, no neural accelerators | 36 GB | not logged | not measured | v0.4.0 self-tests 15/15 on macOS 27.2, Xcode 27.2 (5 Nov 2026; RESULTS.md (ab)) |
| A 128 GB Mac on **macOS 26.6** (**lightly tested**, external tester; chip not reported) | not reported | 128 GB | not logged | not measured (decode step 34 ms) | First macOS 26 run: source build and the v0.4.0 macOS 26 prebuilt, full release gate (5 Nov 2026) |

Sources: Studio bandwidth and working set, [results/micro-2026-10-29-r2.txt](../results/micro-2026-10-29-r2.txt); mini
core counts, memory and effective decode bandwidth (one 16 GB weight pass per plain decode step: M6 100 ms, M5 Pro
55 ms), the study log "Mac minis (2026-10-30)"; the M6 budget, the study log "0.1.1" correction of 1 Nov. "Effective" bandwidth is
what a full decode step achieved, not the chip's peak.

All Macs run the same binaries (built on the laptop or the Studio with Xcode 27's Metal toolchain; the M5 Pro has no
toolchain, so `build/` is copied to it) and the same weights: `mlx-community/Qwen3.8-27B-4bit` plus the
`incoai/Qwen3.8-27B-DFlash2` drafter. Logits match mlx_lm on every machine.

## What changes from machine to machine

### Decode is bandwidth-bound; speculation multiplies it

A plain decode step reads all ~16 GB of the 27B's 4-bit weights. Single-stream decode therefore tracks memory speed:

| Machine | Plain decode step (1 token) | Weight-read floor | Speculative single stream (8k, DFlash2) |
|---|---:|---:|---:|
| Mac Studio M5 Ultra | 16.7–17.1 ms (all linear layers; [night2.log](../results/studio/night2.log)) | 14.0 ms | 191 tok/s ([inferd6](../results/cmp/inferd6-sweep.jsonl)) |
| Mac mini M5 Pro | 55 ms | – | 58.5 tok/s (the study log, "Minis, generated tok/s") |
| Mac mini M6 | 100 ms | – | 28–31 tok/s (same) |

Because the weights are read once per step whatever the batch, extra rows are nearly free until compute catches up:
on the Studio, 1 to 16 rows cost 16.7 to 20.2 ms for all linear layers; on the M6, 20 rows cost the same as 5
(141 ms). That is why speculation (5–8 rows per agent) and many agents per step pay on every machine.

### Prefill is compute-bound, and on the minis it dominates a fleet

| Machine | Prefill (GPU alone) | What limits it |
|---|---|---|
| Mac Studio M5 Ultra | ~1,900 tok/s without attention, 1,455–1,730 in `speed.py` at 8k–32k | MLX's quantized GEMM at ~115 TFLOPS (~90% of the bf16 peak) |
| Mac mini M5 Pro | 451–475 tok/s | the GPU's bf16 rate |
| Mac mini M6 | 259–280 tok/s (~15 TFLOPS, ~80% of its bf16 peak) | the GPU's bf16 rate |

An agent turn adds ~1.2k new tokens of tool output and generates ~100–140, so on the minis about two thirds of a
fleet's wall time is prefill (the study log, "Neural Engine FFN split"). On the Studio the multi-agent cells are
decode-bound instead. Sources: the study log "Agent fleets (2026-10-30 evening)", "Mac minis (2026-10-30)",
[ane-prefill logs](../results/night/m6/ane-prefill.log).

### The Neural Engine split pays on the M6 and the M4 Pro, not on the M5 generation

`ANE_UNITS` moves part of every FFN layer (in 512-channel units, of 34) to the Neural Engine during prefill steps. It
pays only where the Neural Engine is fast relative to the GPU:

| Machine | Neural Engine, one FFN unit (1,024 rows) | GPU, one unit | Prefill sweep, GPU alone → best split | In the server |
|---|---|---|---|---|
| Mac mini M6 | 0.71 ms (22.7 TFLOPS) | ~2.1 ms at 2,048 rows | 259 → 335 tok/s at 24 units (+29%) | 8k fleet prime 31.2 → 25.5 s; fleets +13–17% at 8k, +8–13% at 16k |
| Mac mini M4 Pro (no GPU neural accelerators) | 1.63 ms (9.9 TFLOPS); 3.1 ms at 2,048 rows (10.3) | ~5.2 ms at 2,048 rows (MLX's `mlx_qmm_sg` at ~6.2 TFLOPS) | 114 → 218 tok/s at 26 units (+91%, 2,048-row chunks; 122 → 212 at 1,024) | **default since 5 Nov** (26 units): cold first token 8k 64.2 → 42.1–43.5 s, 32k 286 → 188–191 s; follow-ups level |
| Mac mini M5 Pro | 1.44 ms (11.2 TFLOPS) | ~0.55 ms | 475 → 488 tok/s at 14 units (+3%) | not used (gain below noise) |
| MacBook Pro M5 Max | 1.40 ms (11.5 TFLOPS); 0.99 ms with int8 activations | stronger than the Neural Engine | no gain (730–875 tok/s either way, noisy) | not used |
| Mac Studio M5 Ultra | not measured | – | – | not used (Splash's own calibration gives the split 3.8% there) |

Sources: the study log "Neural Engine FFN split for prefill (2026-11-01 night)" and its "Diagnosed" section;
[M6 sweep](../results/night/m6/ane-prefill.log), [M5 Pro sweep](../results/night/m5p/ane-prefill.log), and the fleet
files in RESULTS.md (a). Splash 1.5.0's own split share is 0.74 on the M6 and 0.41 on the M5 Pro.

Two things are required for the split to pay inside the server, both now defaults when `ANE_UNITS` is set:

- **A residency set over every buffer and the Neural Engine surfaces.** Without it, macOS unwired GPU memory while the
  GPU waited on the Neural Engine and re-wired it every layer: the M6's fleet prime went from 31 to 42–44 s (slower
  than no split) with 73k pageouts. With it, wired memory holds at 24.7 GiB and the prime takes 25.4 s.
- **`ANE_MIN_ROWS` = 1,024.** Mixed agent-turn steps of a few hundred prompt rows paid three command buffers and two
  host waits per layer: an M6 8-agent fleet ran 6.42 turns/min with the split at ≥ 512 rows, 10.88 at ≥ 1,024, and
  9.67 on the GPU alone.

The split also reserves 2.5 GiB of extra headroom, because the Neural Engine service's own buffers are outside Metal's
memory accounting.

### Memory budgets, KV pool and state slots

Each Qwen3.8-27B sequence needs paged KV (16 attention layers: 64 KiB per token at bf16, half that at the default
8-bit KV) plus a fixed recurrent (GDN) state of ~144–190 MiB, held in a state slot. Snapshots of that state let a
follow-up turn resume from the cached prefix.

| Machine | KV pool | Tokens resident (8-bit KV) | Sequences running | Snapshots | SSD tier |
|---|---|---|---|---|---|
| Mac Studio M5 Ultra | 140 GiB (explicit) | ~4.5M (e.g. 32 agents × 128k) | up to 64 | 96 | optional |
| Mac mini M5 Pro | 24 GiB (README) or auto from the budget (overnight fleets) | ~770k | 16 (README), 32 (overnight fleets) | 30 (README), 16 (overnight) | 96 GiB |
| Mac mini M6 | 3 GiB (README) or auto, 3.1–4 GiB (overnight) | ~96k–130k | 4 (README), 8 (overnight fleets) | 8 (README), 16 (overnight) | 64 GiB (README), 96 GiB (overnight) |

Sources: [engine/README.md](../engine/README.md) "Sizing by machine"; overnight settings from
`agents64g.sh` (lab script, not published) and the study log "Overnight agent-fleet results". The README and
overnight values differ because the overnight fleets used the newer auto-sized pool (`kv_gib` 0); both are listed.

What each machine taught us:

- **Mac mini M6 (32 GB): the SSD tier and admission by KV budget are essential.** Metal faults buffers in on first use,
  so an idle footprint (16 GB) hides the real one: a 5.5 GiB KV pool plus 15 slots reached 27 GB, swapped, and decode
  fell to 10 tok/s; 2.5–3 GiB fits (22 GB). With ~96k tokens of KV, a 16-agent 16k fleet is 5.5× the pool, so most
  contexts live on the SSD tier and come back at ~0.5 s per 33k context. Before admission counted KV blocks, a 32k
  fleet with 4 running deadlocked (all four grew until the pool was empty; 0.02 tok/s). Now a turn is admitted only if
  its blocks (prompt + 512 generated + a draft block) fit, and the newest sequence is preempted back to its cached
  prefix if the pool still runs dry (the study log, "v10: KV-budget admission + preemption").
- **Mac mini M5 Pro (64 GB): snapshot slots, not KV, were the limit.** With 16 agents in flight its 16:16 fleet cell
  fell from ~13 to ~5 turns/min three times, each a different snapshot-eviction bug (inner-node snapshots dropped
  without a spill, admission evicting the snapshot it had just restored). Fixed by 31 Oct; the cliff is gone (the study log,
  "Diagnosed 2026-10-30/31").
- **Mac Studio M5 Ultra (256 GB): size the KV pool explicitly.** Auto-sizing with a small model took 222.5 of the
  224 GiB budget and the server hung in an uninterruptible GPU wait on its first request (the study log, "Diagnosed
  2026-11-01: a server that hung"). An explicit 96–140 GiB is fine.
- **Long uptime on the Studio:** after hours of runs a full pool used to hand out scattered pages, costing 10–14% per
  verify step at 128k–256k; blocks now come in aligned groups of 4. Some slowdown after a long sweep remains (128k × 4:
  225 tok/s at the end of a sweep against 249–256 on a fresh server).

### Kernel choices that depend on the machine

| Feature | M5 Ultra (Studio) | M5 Max (laptop) | M5 Pro / M6 (minis) | Default |
|---|---|---|---|---|
| Split-K for narrow GEMM outputs | follows the GPU core count (below 480 tiles on ≥ 60 cores, else 3 × cores) | same rule | M6: no split (162 → 138.5 ms for 5-row GEMMs); M5 Pro 71.6 → 66.4 ms | automatic |
| `lane4` (TensorFold-style repacked GEMM for 3–16 rows) | slower: W4A8 already runs those rows at ~815 GB/s (M=5 19.6 vs 20.9 ms); end to end 16k × 1 147 → 125 tok/s | 12–15% faster per step (8 rows 35.9 → 31.5 ms) | not tried (it needs a weight copy, which the minis cannot afford) | off (`LANE=0`) |
| Verify attention on whole-tile matmuls (`attn_mpp`) | on; needs bf16 KV at first, 8-bit KV variant since 31 Oct | on | the minis' early 8-bit KV runs used the older kernel | on |
| 8-bit KV | default (halves KV memory; 10–12% faster verify steps at 64k–128k) | default | default (memory) | on; bf16 for Gemma-4 |
| Residency set over every buffer | no measured gain (`RESIDENCY=1` opt-in) | – | required with the Neural Engine split | on only with `ANE_UNITS` |
| Neural Engine FFN split | not used | no gain | M6: on (24 units); M5 Pro: off | off (`ANE_UNITS=0`), except M4 / M4 Pro: 26 units by default |
| Drafter linears | 4-bit (`DRAFT_BITS=4`) | 4-bit | 4-bit (the fleet scripts once pinned 8-bit; 4-bit gives 0.8 GiB back for KV) | 4-bit |

Sources: the study log sections "Mac minis (2026-10-30)", "Lessons from the mlx.fast competition", "attn_mpp", "Splash at
128k runs ~4 lanes", "Persistent SSD tier".

### The MacBook Pro M5 Max: a development machine, not a benchmark machine

The laptop was used for builds, kernel experiments, the SSD-tier tests and the competition prompt. It was often on
battery and it throttled during longer A/Bs: plain decode drifted from 32 to 45 ms per step during one end-to-end
comparison, and run-to-run noise was about ±7 ms per cycle while in use, which hides 1–2% effects. Treat its
end-to-end numbers as indicative, and judge small optimisations on the Studio (the study log, "Single-stream gap,
continued" and "Lessons from the mlx.fast competition"). Its measured ALU rates (`micro/fma_rate.mm`): f32 14.8, f16
19.2, half2 26.1 TFLOPS.

## Recommended settings

`run.sh` arguments are: port, max running sequences, step tokens, KV GiB (0 = auto from the GPU budget), snapshots,
max context. `run.sh start --model 27b` picks its sizing from the Mac's memory, using engine/README.md's "Sizing by
machine": under 48 GB the M6 line (`16 1024 0 8 32768`, SSD tier 64 GiB); under 128 GB the M5 Pro line with the pool
auto-sized, as its measured fleets ran (`16 1024 0 30 147456`, SSD tier 96 GiB); under 256 GB the laptop ("as the M5
Pro row", `16 1024 24 30 147456`); 256 GB and up the Studio row below. `--dry-run` shows the choice.

| Machine | `run.sh` | Environment | Notes |
|---|---|---|---|
| Mac Studio M5 Ultra, 256 GB | `8300 64 2048 140 192 262144` | defaults (`KV8=1`, `SPEC_AUTO=1`, `LANE=0`, `ANE_UNITS=0`); `SPILL_DIR` optional | Give the KV size explicitly; never 0 with a small model. Restart the server between long benchmark sweeps |
| Mac mini M5 Pro, 64 GB | `8300 16 1024 24 30 147456` (README); the overnight fleets used `8300 32 1024 0 16 <window>` | `SPILL_DIR=… SPILL_GIB=96`, `DRAFT_BITS=4` | GPU only: the Neural Engine split gains ~3% here |
| Mac mini M6, 32 GB | `8300 4 1024 3 8 40960` (README); the overnight fleets used 8 running, auto KV (3.1–4 GiB), 16 snapshots | `SPILL_DIR=… SPILL_GIB=64` (README) or 96 (fleets), `ANE_UNITS=24` (`ANE_MIN_ROWS` defaults to 1,024), `DRAFT_BITS=4` | The Neural Engine split is worth +8–17% in fleets. Agents beyond the KV pool queue and swap through the SSD |
| Mac mini M4 Pro, 64 GB (no GPU neural accelerators) | `8300 16 1024 0 8 40960` (the 5 Nov measurements) | `DRAFT=…`; the Neural Engine split is on by default (26 units; `ANE_UNITS=0` off) | Prefill ~1.5x with the split (RESULTS.md (v)); decode unchanged. Qwen3.6-35B-A3B: leave the drafter off from 4 agents (`--draft none`): 4k x 4 / 16 agents 152 / 211 tok/s without vs 83 / 133 with it (5 Nov); with 1 agent the drafter wins (107 vs 86) |
| MacBook Pro M5 Max | as the M5 Pro row | as the M5 Pro row | Benchmark on AC power and expect thermal drift; confirm small kernel wins on the Studio |

The README's mini rows were written for 0.1.0 before the auto-sized pool; the overnight values are what produced the
fleet results in RESULTS.md (a). When the two disagree, the overnight values are the measured ones.

## Mac mini M4 Pro: works, lightly tested (1 November 2026)

**Status: not fully tested.** One afternoon of checks on one machine (64 GB, 20-core GPU, macOS 27.0.1). Use it to
try inferd on older Macs, not for benchmarks or claims. Raw rows: [results/models/m4-pro/](../results/models/m4-pro/).

The M4 has Metal 4 but no GPU neural accelerators (`supportsFamily(MTLGPUFamilyApple10)` is false). What that changed:

- **Two bugs found, both fixed** (`src/gpu.h` `Gpu::nax`, `src/qwen35.h`, `src/dflash.h`):
  - MLX's NAX kernels we vendor (quantized matmul at M >= 256 rows, flash attention for chunks > 256 rows) compute wrong
    results on the M4 at prefill sizes (a 601-token prompt in one chunk: cos 0.29, wrong top-1). Without neural
    accelerators the quantized GEMMs no longer touch them: 1-10 rows go to MLX's `qmv_wide`, 11 rows and more
    (`SG_MIN`) to MLX's simdgroup `mlx_qmm_sg`, both with bf16 activations, so W4A8 is unused by default (`SG_MIN=0`
    sends 11+ rows back to it); every multi-row chunk runs on the key-split `attn_shared` path. (The first fix kept
    every GEMM on W4A8; this routing replaced it.)
    `NAX_GPU=0/1` overrides the detection. Our own `attn_paged_nax` (used only with `NAX_MPP=0`) is also wrong there.
  - **On every machine:** the DFlash2 drafter's input layer (`fc`, K = 25,600) is wider than the W4A8 activation
    scratch was sized for; a prefill chunk of more than ~0.68 x max_tokens rows overflowed it and filled the draft
    context with NaN (on the M4 this drafted junk, 1.0 tokens per cycle, and the server emitted garbage).
    `reserve_a8` now sizes it. M5 results are unchanged (27B logits identical; decode B 1/16/64 and verify blocks at
    parity on the Studio).
- **Correctness:** the 27B matches mlx_lm (decode, 24-token chunks and verify shapes 2, 3, 5 and 8 rows); greedy
  speculative decoding is lossless after the fix.
- **Speed (Qwen3.8-27B 4-bit), both columns on the fixed build** (`*-m4pro-nodraft-fixed.jsonl`, `*-m4pro-draft.jsonl`;
  earlier no-drafter rows were made with the wrong MLX kernels and are invalid):

| | no drafter | DFlash2 drafter |
|---|---:|---:|
| decode, single stream 0.5k / 4k / 16k (tok/s) | 15.1 / 14.7 / 13.9 | **24.6 / 24.7 / 20.3** |
| prefill (tok/s; 1 Nov, before the Neural Engine split default) | 98-109 | 106-107 |
| conc.py 4k, total decode tok/s at 1 / 4 / 8 / 16 agents | 14.9 / 20.6 / 39.0 / 73.1 | 19.0 / 55.7 / 60.9 / 65.0 |

  Drafter acceptance 4.72 tokens per cycle on chat prompts (3.64 over the whole run, smaller blocks at higher
  concurrency). At 16 agents the drafter stops paying: an 8-row verify step costs ~3.2x a decode step on the M4
  (1.4x on an M5). mlx_lm on the same machine: prefill 123-127 tok/s, decode 15.2-15.4. Correct prefill is ~110
  tok/s; the faster 139-143 measured earlier came from the wrong kernels.
- **Debug tools used to find these** (`engine/tools/`, build like the other tools: `clang++ -std=c++20 -O2 -fobjc-arc
  -framework Metal -framework Foundation -framework IOKit -framework IOSurface tools/<name>.mm -o build/<name>`):
  `gemm_dbg <bits> <M> <N> <K> <TM> [S] [NSG]` checks the W4A8/W8A8 kernel against a CPU reference on random weights;
  `draft_dbg <target> <draft> <tokens.txt> <reps>` runs the drafter repeatedly on identical inputs and prints draft
  tokens and buffer checksums (non-determinism there exposed the scratch overflow).
- **Still wrong on the M4, off the default path:** `NAX_MPP=0` (`attn_paged_nax`). **Not rerouted, probably wrong
  there (untested):** bf16 checkpoints (`mlx_gemm`). The Neural Engine split's K-slice GEMM (MLX's `mlx_qmm_sgw` on
  these GPUs) is correct: the release gate passes with the split on (5 Nov, below).
- **Prefill, 5 Nov: the Neural Engine split is on by default on M4 / M4 Pro** (26 of the 27B's 34 FFN units;
  `ANE_UNITS` overrides, `0` turns it off; `Qwen35::ane_units_default`). Where the time went before (8k, 2,048-row
  chunks, `PROF_KERNELS=1`): MLX's quantized GEMM 93.0%, attention 3.8%, GDN 1.7%, the rest < 1% each. That GEMM already
  runs at ~6.2 TFLOPS, level with mlx_lm (engine prefill 125 tok/s at 8k vs mlx_lm's 130.6), so a faster GPU kernel had
  little to win (an fp16-fragment GEMM was within ±3% end to end and was dropped); the Neural Engine runs the FFN at
  ~10.3 TFLOPS beside it. Cold first token for one agent through the server (27B + DFlash2, `run.sh 8300 16 1024 0 8
  40960`): 2k 16.1 → 12.4–12.8 s, 8k 64.2 → 42.1–43.5 s, 32k 286 → 188–191 s (~115 → ~173 tok/s; mlx_lm 118.8 at 32k).
  8k follow-ups at 1 / 4 / 16 agents 25.4 / 49.0 / 61.1 → 25.7 / 47.8 / 63.5 tok/s total. The split reserves its
  margin from the KV pool (29.7 → 28.8 GiB auto). Release gate passed with it on (logits at chunks 1/24/256/1536, cos
  >= 0.9995 on seqs 1-3, KL <= 0.0024). Not measured on M1-M3 or the M4 Max (a 40-core GPU leaves the Neural Engine
  a smaller share; try `ANE_UNITS` 14-26 there). Tables: [RESULTS.md (v)](RESULTS.md).
- **Long prompts are still slow on the M3 family.** The Neural Engine split's default covers M4 / M4 Pro only; on M3 it
  is off and untested. An external M3 Max (30-core GPU, 36 GB, 0.4.0) took 50.4 s to the first word of one ~8k-token
  prompt (RESULTS.md (ab)). Try `ANE_UNITS` (e.g. 14-26) there and check the release gate with it on.
- **External M4 Pro MacBook Pro (48 GB), 98a2b19 (the 0.4.0 engine + docs), 5 Nov, 2 trials each:** prefill 1.5-1.9x
  faster at 4k / 16k with the Neural Engine split on by default (speed.py first token 22.7 vs 35.3 s at 4k, 92.7 vs 157 s
  at 16k; decode unchanged). 4k follow-ups, drafter on vs off, total decode tok/s at 1 / 4 / 8 / 16 agents: 23.9-22.8 /
  52.2-49.0 / 63.8-66.0 / 74.2-67.6 vs 15.5-14.1 / 38.8-36.0 / 40.2-38.1 / 63.6-68.1: the drafter wins 1.3-1.7x to 8
  agents and is level at 16 (correcting the single-trial "69 vs 63" below); off halves the first token at 16 (6.6-7.1 vs
  11.9-13.1 s). The 1-8 agent shortfall reported against 0ef3457 is 2-10% here, within this build's 9% trial spread.
  Step times (drafter on): 1 seq block 16 230 ms, 4 seqs block 4 235, 16 seqs block 4 673 (9dd55ad: 252 / 256 / 732);
  acceptance 3.72 tokens per cycle. Correctness: top-1 matched mlx_lm at chunks 1 / 24 / 256 / 1,536.
- **External M4 Pro MacBook Pro (48 GB), 9dd55ad, 4 Nov:** 16 agents x 4k follow-ups with the drafter 69.0 tok/s total
  (0ef3457: 35.3; no drafter 63.2), verify step at block 4 732 ms (was 1,331), median first token 14 s (was 64; no drafter
  6.8). Drafter memory 6.0 -> 2.9 GiB (8-bit rings). Open: drafter-on totals at 1 / 4 / 8 agents 19.5 / 46.9 / 57.3 vs
  25.2 / 54.0 / 67.3 on the older build, with single-stream speed, step times and acceptance (3.62 vs 3.76) unchanged:
  repeated on a Mac mini M4 Pro (64 GB), 3 alternating trials each, 27B + DFlash2, 4k follow-ups x 1/4/8/16: v0.2.0
  23.7–25.0 / 50.7–53.4 / 60.7–60.9 / 62.4–64.4 vs the 0.3.0 engine 23.0–23.2 / 50.5–52.2 / 60.1–61.1 / 62.9–64.4 tok/s:
  level, so not a regression between those releases (the tester's older build is not in this repo). Top-1 matched at every batch size to 1,536 rows; 2 of 64 prompts just
  under cos 0.999 (the 0.995 gate passes).
- **Not tested on the M4:** 32k and longer contexts under load, the SSD tier,
  vision, long uptime, other models, Flash-Next, 8-bit and bf16 checkpoints, the base M4. (An M4 Max has since run
  the full self-test set; see the next section.)
  Some no-drafter single-user requests at 4k/16k errored or stopped after two tokens before the fixes; not re-checked.

## MacBook Pro M4 Max: source build on macOS 27, lightly tested (4 November 2026)

**Status: lightly tested, by an external tester.** One MacBook Pro (Mac16,6: M4 Max, 40-core GPU, 64 GB, no GPU
neural accelerators, so the M1-M4 kernel path above) on macOS 27.0 (26A428), Xcode 26.6 with the Metal toolchain
installed by `xcodebuild -downloadComponent MetalToolchain`, built from source at **2d2a108** (engine code identical
to v0.2.0). Model `mlx-community/Qwen3.8-27B-4bit` revision 10c35caa, drafter `incoai/Qwen3.8-27B-DFlash2` revision
015e7956. The raw outputs stayed on the tester's Mac; the findings and their fixes are in
[CHANGES-m4max-report-2026-11-04.md](CHANGES-m4max-report-2026-11-04.md).

- **Everything passed:** the build (35 s, no warnings), every host and front-end test, `tools/release_gate.sh` with the
  drafter, logits vs mlx_lm at chunks 1 and 24 (seqs 1-3 cos 0.99969-0.99997) and at 256 and 1,536 rows on 4 x 1,536
  tokens (cos 0.99959-0.99987, top-1 8/8), `KV8=0` at chunks 1/24/256 (cos >= 0.99983), the server smoke test and
  `examples/run_all.sh` 15/15 (OCR 21/21 on the 4-bit build above, which has the vision tower).
- **Speed (Qwen3.8-27B 4-bit), about 1.6-1.9x the M4 Pro:**

| Measure | M4 Max (2d2a108) | M4 Pro (above) |
|---|---:|---:|
| `decode_test bench`, 2,048 context, B = 1 / 16 / 64 | 33.96 / 101.44 / 333.95 ms per step (29.4 / 157.7 / 191.6 rows/s) | 15.1 tok/s plain decode, one stream |
| `spec_test`, 4 sequences (the gate's short prompts) | plain 78.9 tok/s total; speculative 115.6 tok/s, 11.75 tokens per cycle at block 16 (repetitive prompts) | |
| Server, one stream, drafter on | ~40 tok/s, 2.8 tokens per cycle (`SPEC_AUTO` chose blocks 4-5) | 24.6 tok/s |
| Prefill (`users-at-once`, 8,174-token prompt) | first token at 39.9 s, ~205 tok/s | ~110 tok/s |
| `coding-smoke` (16 tests), effort none / medium | 15.0 / 16 (14 first try), 16 / 16 | 15.5 / 16 (none) |

  The `agent-followups` timings from that run are invalid (the Mac slept during it); its 24/24 answers were correct.

## A 128 GB Mac on macOS 26.6: release gate, source build and prebuilt (5 November 2026)

**Status: lightly tested, by an external tester.** The first run of inferd on macOS 26 (26.6). The Mac has 128 GB; the
chip was not reported, so it is not named here. Both the source build of v0.4.0 and the v0.4.0 macOS 26 prebuilt
(`inferd-0.4.0-macos26-arm64.tar.gz`, built on macOS 26.6 with Xcode 26.5) passed the full release gate
(`tools/release_gate.sh` with the drafter):

- host, front-end and sampler tests all ok;
- logits vs mlx_lm on 4 x 2,048 tokens at chunks 1, 24, 256 and 1,536: top-1 ok, cos >= 0.9972;
- speculative = plain: passed (one exact tie);
- `decode_test bench`, 2,048 context, B = 1 / 16 / 64: 34.0 / 101.1 / 323.8 ms per step (source build) and
  34.1 / 101.1 / 324.0 ms (prebuilt).

Setup as a new user would do it: `./build.sh` took 43 s (after selecting Xcode: with the Command Line Tools as the active
developer directory it stopped at the first `xcrun metal`, which `build.sh` now handles itself); `install.sh` into a
fresh venv took 28 s and found both models in `~/models`; `run.sh start --model 27b` sized itself for 128 GB (16 agents,
24 GiB KV, 147k max context, 96 GiB SSD tier) and answered a `curl` request. The setup problems this run found, and their
fixes, are in engine/README.md "Changes" (Unreleased).

(An earlier note in these docs said an M4 Max had run a source build on macOS 26. That M4 Max ran macOS 27.0, build
26A428, as its report and the section above say; this 128 GB Mac is the first macOS 26 machine.)
