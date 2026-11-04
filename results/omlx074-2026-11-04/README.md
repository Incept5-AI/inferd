# oMLX 0.7.4.dev1 vs inferd 0.2.0 on the Mac Studio M5 Ultra (2026-11-04)

Everything ran on the Studio, one model process at a time, every arm under `studio_lock.sh` (re-sourced per acquire,
released after). Raw rows: `raw/`; logs: `logs/` (driver log, every server's log, harness output); 2 s memory samples:
`mem/`. Scripts in `scripts/`: `run_omlx074.sh` (main driver), `run_omlx074b.sh` (same plus the extra arms `i27pf`,
`o4_27mtp`, `ifn8k`), `b2b.py` (the oMLX #4902 workload, written for this), `summarize.py` (regenerates every table
from `raw/`: `python3 scripts/summarize.py`). Studio copies: `studio:~/inference/results/omlx074/`, scripts in
`studio:~/inference/omlx074/`.

## Setup

| | |
|---|---|
| inferd | 0.2.0 prebuilt tarball, `studio:~/inference/rel020/engine` (gated 4 Nov 04:11). 27B in two configs: the release gate's `run.sh 64 2048 140 96 262144` (KV 140 GiB, so **PF8 auto-off**: `i27`) and `run.sh 32 2048 0 96 32768` (auto KV, **PF8 on**: `i27pf`). Flash-Next `MTP=1 run.sh 32 2048 64 32 65536` (`ifn`). 8-bit `run.sh 64 2048 96 48 32768` with / without DFlash2 (`i8` / `i8nd`) |
| oMLX 0.7.4.dev1 | new venv `studio:~/.venvs/omlx074`, cp313 wheel; sha256 matches GitHub's asset digest (`40ba8e23...`, no signature published). mlx 0.32.4, mlx-lm 0.32.3.dev8 |
| oMLX 0.7.3 | new venv `studio:~/.venvs/omlx073` from `~/dev/tensorfold/dist-omlx-0.7.3` (`SHA256SUMS` OK). **Same mlx / mlx-lm builds as 0.7.4**, so the deltas are oMLX's own code. `~/.venvs/omlx` (0.7.1.dev1) untouched |
| oMLX server | isolated home per arm `<models>/omlx074/<arm>` (deleted only if it carries `.omlx074-marker`), weights symlinked, `--max-concurrent-requests 32 --paged-ssd-cache-dir <home>/cache --paged-ssd-cache-max-size 96GB --hot-cache-max-size 48GB`, thinking off in `model_settings.json`; stopped by pid then port |
| Weights | `mlx-community/Qwen3.8-27B-4bit` 10c35caa + DFlash2 `z-lab/Qwen3.8-27B-DFlash2` 50307d4c (byte-identical blobs to incoai: **the incoai snapshot directory on the SanDisk is now empty, blobs only**, so the rel020 gate's `DRAFT=` path no longer resolves); `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` c4ddfb54; `mlx-community/Qwen3.8-27B-8bit` 815b83c0; for oMLX's own MTP arm `Jundot/Qwen3.8-27B-oQ4e-mtp` 49c1d603 |
| oMLX's best equivalent | 27B: DFlash2 with the same drafter, batched (log: `DFlash drafter attached ... kind=dflash2, block=8, layers=[5, 19, 33, 47, 61]`). Its native MTP on its own oQ4e-mtp weights is slower (single 123-132 vs 159-166 tok/s; 8 agents 333.8 vs 286.3 s), so DFlash2 is the row quoted. Flash-Next: Lightning MTP on |
| Harnesses | `~/inference/bench/{conc,wall,speed}.py` unmodified (RESULTS (b1)/(h)). Untimed `wall.py --n 1 --prompt 2048 --gen 64` warm-up per server. Greedy, thinking off, `ignore_eos` (oMLX honours it: every row has exactly 6,400 / 1,024 tokens). **0 errors in every cell**, so no short denominators |

## 1. Qwen3.8-27B 4-bit + DFlash2

**Agent follow-ups, `conc.py --mode followup --ctx 16k`: total tok/s (TTFT median / max, s)**

| N | inferd 0.2.0, gate config | inferd 0.2.0, PF8 on | oMLX 0.7.4.dev1 | oMLX 0.7.3 |
|---:|---:|---:|---:|---:|
| 1 | 148.8 (0.09) | **150.2** (0.09) | 127.2 (0.64) | 40.9 (8.76), 0 cached |
| 8 | 484.6 (0.65 / 0.65) | **489.8** (0.53 / 0.53) | 154.4 (2.93 / 4.42) | 42.4 (72.9 / 76.8), 0 cached |
| 32 | 620.9 (1.96 / 2.17) | **628.0** (1.55 / 1.75) | 179.4 (12.1 / 29.5) | 44.5 (282 / 307), 0 cached |

The gate's own conc run of this tarball the same morning: 158.5 / 492.3 / 619.9. Both inferd and oMLX 0.7.4 hit the
prefix cache for 99-100% of every prompt; **0.7.3 reused nothing** (`re-prefills 16465 of 16465 tokens (reused 0)`), so
0.7.4's hybrid-cache work (#4994 in 0.7.3 notwithstanding) is what makes follow-ups work at all. Even so inferd
delivers 3.2x oMLX 0.7.4 at 8 agents and 3.5x at 32: oMLX serves the cache restores slowly (TTFT up to 29.5 s at 32
agents with 16,384 tokens cached each) and its batched DFlash decode peaks at 265 tok/s with all 32 streams running
(inferd 673).

**N agents at once, `wall.py` (14,336-token prompts, 6,400 forced tokens), 3 trials, mean ± half-range**

| N | inferd gate cfg, total s | inferd PF8, total s | oMLX 0.7.4, total s | oMLX 0.7.3, total s (1 trial) | inferd PF8: last 1st token / decode, all generating | oMLX 0.7.4: last 1st token / decode |
|---:|---:|---:|---:|---:|---|---|
| 1 | **32.9 ± 1.5** | 34.2 ± 4.4 | 43.2 ± 2.1 | – | 6.0 s / 231 | **5.3 s** / 169 |
| 4 | 80.5 ± 3.0 | **71.7 ± 1.0** | 163.2 ± 5.8 | – | 25.2 s / 568 | 33.6 s / 199 |
| 8 | 143.7 ± 4.5 | **125.0 ± 0.9** | 286.3 ± 3.9 | 302.5 (last 1st token 90.0 s, decode 228) | 51.8 s / 713 | 72.1 s / 241 |

inferd PF8 at 8 agents: 125.0 s, as RESULTS (k)/(m) (125.6). oMLX prefills all N prompts before anyone decodes (TTFT
median = max) and then decodes at 200-240 tok/s, so it needs 2.3x inferd's time at 4 and 8 agents. It has the faster
first token for one user (5.3 s vs 6.0 s), from its INT8-activation prefill (idea 1).

**Single stream, `speed.py`, median of 2 cold reps: decode / prefill tok/s**

| Prompt | inferd gate cfg | inferd PF8 | oMLX 0.7.4 DFlash2 | oMLX 0.7.4 MTP (oQ4e-mtp weights) |
|---|---|---|---|---|
| 1k | **239** / 1,787 | 219 / **2,544** | 166 / 1,546 | 132 / 1,864 |
| 8k | 212 / 1,888 | **214** / 2,473 | 159 / **2,702** | 123 / 2,540 |
| 16k | 207 / 1,822 | **211** / 2,382 | 162 / **2,582** | 124 / 2,494 |

oMLX 0.7.1.dev1 (research/omlx, 1 Nov) did 194 / 172 tok/s DFlash2 decode at 1k / 8k on the same weights, so 0.7.4's
single-stream DFlash decode is 10-15% slower than 0.7.1's (not investigated; its adaptive verify depth is the likely
suspect).

**Verdict (27B):** inferd leads decode by 30-35% and every multi-agent cell by 2.3-3.5x. oMLX leads 8-16k prefill by
5-10% over inferd's PF8 and by ~40% over inferd without PF8, **with no extra weight memory** (PF8 needs +22.7 GiB).

## 2. Qwen3.8-Flash-Next oQ4e (inferd MTP + QSA; oMLX Lightning MTP)

**`wall.py`, 3 trials** (0.7.3 and the inferd 8,192-step arm: 1 trial at N = 8)

| N | inferd total s | oMLX 0.7.4 total s | oMLX 0.7.3 total s | inferd: last 1st token / decode | oMLX 0.7.4: last 1st token / decode |
|---:|---:|---:|---:|---|---|
| 1 | 38.9 ± 0.6 | **34.8 ± 3.2** | – | 3.6 s / 182 | **2.3 s** (4.6 on the first trial) / **202** |
| 4 | **92.1 ± 2.4** | 121.0 ± 8.6 | – | 14.4 s / 331 | 17.1 s / 241 |
| 8 | 142.7 ± 1.7 | 191.7 ± 8.1 | 201.7 | 29.5 s / 464 | 32.8 s / 314 |
| 8, inferd `step_tokens` 8192 | **134.5** | | | 29.5 s / 481 | |

oMLX leads one user (prefill ~6k tok/s vs ~4k; decode 202 vs 182, its plain step being faster, as research/omlx found)
and falls behind from 4 users because its batched decode flattens (241 / 314 tok/s vs 331 / 464). 0.7.4 vs 0.7.3 at 8
users: -5%.

**Back-to-back long prompts** (#4902's workload: 3 x 10.4k, 3 x 20.8k, 5 x 34.5k random dictionary words behind a
unique marker, one greedy token each, 2 rounds; time to the token, s)

| Arm | 10.4k (6 prompts) | 20.8k (6) | 34.5k (10): mean / slowest |
|---|---|---|---|
| inferd 0.2.0 | 2.55-2.60 | 5.27-5.32 | 9.08 / 9.12 (3,800 tok/s, flat) |
| inferd, `step_tokens` 8192 | 2.55-2.61 | 5.27-5.31 | 9.09 / 9.18 (identical: an idle prefill already uses `PREFILL_STEP` 8,192 on Flash-Next) |
| oMLX 0.7.3, default guard | 1.97-2.29 | 4.02-4.08 | 6.62 / 6.66 |
| oMLX 0.7.4.dev1, default guard | 1.77-1.90 | 3.38-3.58 | **5.97** / 6.38 (5,800 tok/s) |
| oMLX 0.7.3, `--memory-guard-gb 118` (a 128 GB Mac's ceiling) | 1.96-2.29 | 3.92-4.23 | 6.72 / 7.28 |
| oMLX 0.7.4.dev1, `--memory-guard-gb 118` | 1.75-3.40 | 3.34-3.38 | **5.70 / 5.75** |

**Verdict on their claim** (slowest 34.5k prompt 90.7 -> 13.1 s on a 128 GB M5 Max): not reproducible here. 0.7.3
never collapsed on the Studio, even with the guard at 118 GB (which did engage: "Prefill above max_bytes at 8192
tokens: 106.0GB > 105.4GB"); the collapse needs a machine whose weights alone sit past the soft watermark. What is
confirmed: 0.7.4 is 10-15% faster than 0.7.3 at every size, guarded or not. **The finding that matters for us: oMLX
prefills Flash-Next 1.5x faster than inferd** (5.7-6.0 s vs 9.1 s at 34.5k, and 1.4x at 10k), and chunk width is
not the reason (inferd's 8,192-row arm is identical). It is kernels (idea 2).

## 3. Qwen3.8-27B 8-bit with 6 concurrent requests (their +16%)

`wall.py --n 6 --prompt 1024 --gen 1024`, 3 trials (and `--n 1`, 2 trials): decode tok/s while all streams generate

| Arm | 6 requests | 1 request |
|---|---:|---:|
| oMLX 0.7.3, no speculation | 129.4 ± 0.5 | 33.0 |
| oMLX 0.7.4.dev1, no speculation (log: "Batched decode affine kernels engaged (6 rows, vk)") | **148.1 ± 1.0** | 33.0 |
| inferd 0.2.0, no drafter | 146.8 ± 0.5 | **34.3** |
| oMLX 0.7.4.dev1 + DFlash2 | 255.9 ± 4.2 | 125.0 ± 2.5 |
| inferd 0.2.0 + DFlash2 | **491.6 ± 14.0** | **163.5 ± 2.6** |

**Verdict: the claim reproduces, +14.5% on the Studio** (+16% on their M5 Max with oQ8e). It brings oMLX's plain
6-row 8-bit decode level with inferd (148.1 vs 146.8), which already reads each weight tile once for 3-16 rows (W8A8
16-row tiles; `qmv8` only at M <= 2). With the drafter, inferd gives 1.9x oMLX at 6 requests and 1.3x at one.

## Peak wired memory (GiB, 2 s samples, `mem/`)

inferd pre-allocates its KV pool: i27 194.7, i27pf 202.1 (incl. the 22.7 GiB PF8 copies), ifn 192.0, i8 175.9. oMLX
grows with use: o4_27 132.7 (incl. the 48 GB hot cache), o4_fn 126.1, o4_8 37.1, o4_8d 52.4; 0.7.3 Flash-Next 117.5.
## Their changes, one by one: does inferd need it?

| oMLX change | What it fixes in oMLX | inferd today | Verdict |
|---|---|---|---|
| Bounded buffer pool through the prefill layer loop (#5031, DeepSeek V4.1, +43-48% prefill) | MLX's pool was emptied (`mx.clear_cache()`) after every layer of every wide chunk, so each layer re-allocated its buffers from Metal (system CPU 87-90 s per 256k prompt) | No allocator in the hot path: every activation, scratch, tape and KV buffer is allocated once at load (`Gpu::alloc` is called only from model/scheduler setup; the only MLX in the process is the front end's vision tower, already capped with `set_cache_limit(256 MB)`) | Nothing to take. One check worth keeping: the vision tower's MLX pool under long image prompts (cheap to measure) |
| 6-row 8-bit batched decode through `vk` (b5c7793e, +16% at 6 requests, oQ8e, M5 Max) | Stock `qmv_wide` reads the weights once per 5-row tile, so 6 rows read 8-bit weights twice | `qmv4/qmv8` only at M <= 2; from M = 3 the W4A8/W8A8 kernel reads every weight tile once for a 16-row tile (`W4A8_TM` auto 16 at M <= 16) | Already covered; measured below (6 vs 1 stream) |
| FP16 checkpoints on the fused HC/GDN kernels (#5028) | fp16 Flash-Next oQ2.5e fell off the fused path (+23-62% decode once on it) | The loader requires BF16 scales/biases for quantized linears (`need_dtype(... "BF16")`, `weights.h` / `qwen35.h:307`), so an fp16 checkpoint is rejected outright, not slowed | Only if we want fp16 oQ checkpoints (oQ2.5e for 64 GB machines). Low priority |
| Wide prefill chunks under memory pressure (#4902) | On 128 GB machines the adaptive throttle cut Flash-Next chunks to 32-512 tokens; each chunk re-streams all expert weights | Fixed step budget (`step_tokens`, 2048 default) independent of memory pressure; KV is pre-allocated and admission is by KV budget, so there is no throttle to mis-fire | Nothing to take for the Studio. Lesson for an elastic expert store on the 64 GB mini (NOTES idea 6): never shrink an MoE prefill chunk to save memory; each chunk pays the full expert read |
| Cluster prompt cache across tool turns (#5062) | Tool-result turns were typed "assistant" and evicted first; hybrid caches cannot be trimmed, so every tool turn re-prefilled | Snapshot a little before every prompt's end (`snap_margin`, chat templates re-render the previous turn's tail), decode snapshots (`SNAP_GEN`), system-prompt / branch snapshots (`SNAP_BRANCH_MIN`), agent sessions evicted last, SSD tier | Already covered; the bug class (a resume point typed so it is evicted first) is worth a regression test in the agent examples: N-turn tool loop, assert cached_tokens on turn 5 >= turn-4 prompt - margin |
| Prefill Accelerator (#5065) | M3 Ultra (slow GPU, 512 GB) + M5 Max: 1.4-2.9x prefill on 64k prompts | – | See the estimate below: real but small for our minis, and beaten by plain two-box sharding for a dense 27B |

## Prefill Accelerator for the 32 GB Mac mini M6: estimate (not implemented)

What crosses the link (oMLX's design, `docs/experimental/prefill-accelerator.md`, `prefill_helper/adapters/qwen4_exp.py`):
for each chunk the helper runs layers [0, K) and returns the residual stream at the cut, the KV rows its attention layers
appended, and the whole conv + recurrent state of every linear (GDN) layer it ran; the host runs [K, 64), appends the
helper's rows and keeps decoding alone. Qwen3.8-27B (config: hidden 5120, 64 layers, attention every 4th, 4 KV heads x
256, GDN 48 value heads x 128 x 128):

| Item | Size |
|---|---|
| Residual at the cut, per token (bf16) | 10.2 KB |
| KV per token per attention layer (inferd 8-bit KV + scales) | 2.1 KB (bf16: 4.1 KB) |
| DFlash2 context features (target layers 5/19/33/47/61; ~2 below a K=32 cut), per token | ~20 KB |
| GDN recurrent state per layer per sequence (fp32) + conv window | 3.15 MB + 61 KB |
| Per 2,048-token chunk at K = 32 (8 attention + 24 GDN layers), oMLX-style (GDN states every chunk) | 174 MB |
| Same, if the helper keeps its GDN states and ships them only at snapshot points / the prompt's end | ~98 MB |

Rates used (RESULTS / MACHINES.md): M6 GPU prefill 259-280 tok/s (335 with the ANE split); M5 Pro 451-475 bf16,
712-837 with PF8 (int8; the M6 cannot afford PF8's +22.7 GiB int8 copies, a helper can for its own layers). Links: the
M5 Pro <-> M6 Thunderbolt bridge measured 6.3 Gbit/s (~790 MB/s, cable negotiates 20 Gb/s); 10GbE ~1.15 GB/s; 1GbE
~117 MB/s.

| Pair (host + helper) | Layers on helper (balanced) | Pair prefill | Link need (2k chunks) | Transfer per chunk: TB / 10GbE / 1GbE | One 16k prompt, TTFT |
|---|---:|---:|---:|---|---|
| M6 + M6 | 32 | ~540 tok/s (1.8-1.9x) | 46-66 MB/s | 0.22 / 0.15 / 1.5 s (chunk compute 3.8 s) | 61 -> 32-34 s |
| M6 + M6, ANE split on both | 32 | ~670 (vs 335) | 57-82 MB/s | same | 49 -> 26-28 s |
| M6 + M5 Pro (bf16 on helper) | 40 | ~730 (2.5x) | 72-106 MB/s | 0.26 / 0.18 / 1.7 s | 61 -> 24-25 s |
| M6 + M5 Pro (PF8 on helper) | 48 | ~1,070 (3.5-3.7x) | 120-180 MB/s | 0.29 / 0.20 / 2.0 s (1GbE saturates) | 61 -> 16-17 s |

The link is not the problem on Thunderbolt or 10GbE (transfer is 5-8% of chunk compute and overlaps it); 1GbE becomes
the limit only with the M5 Pro helper. Pipeline fill costs one helper half-chunk per prompt (why 1.8x, not 2x).

**But for fleets (our M6 workload) it loses to plain sharding.** About two thirds of an M6 fleet's wall time is
prefill. A helper that takes half of each prefill chunk saves at most 1/3 of the wall (turns/min x1.5 with an M6
helper, x2 with a PF8 M5 Pro helper), and the host still decodes alone with its own 3 GiB KV pool, which is what limits
the M6's 16k x 8 cell. Two M6s each running inferd behind a session-affine router (agent -> box, so the prefix cache
keeps hitting) give ~2x on prefill *and* decode *and* KV capacity with no new engine code. The accelerator wins only
where sharding cannot help: one long prompt's time to first token (1.8x M6+M6, 3.5x with a PF8 M5 Pro), and models that
fit on the host only (not the 27B on a 32 GB box: it fits on every mini). Effort: high (oMLX's PR is ~10k lines incl.
1.4k of tests: chunk grid, wire format, split tuner, helper lifecycle and fallback); in inferd it would also have to
cross the scheduler's mixed prefill+decode steps, the DFlash2 context ring (target layers on both sides) and GDN
snapshots.

## Ideas for inferd, ranked

| # | Idea | Evidence | Expected gain | Effort |
|---:|---|---|---|---|
| 1 | **Large-M W4A8 prefill GEMM on the packed 4-bit weights** (no int8 weight copy): decode the 4-bit codes straight into int8 NAX fragment registers and multiply against per-row / per-64-group int8 activations, GS64 affine correction every 4 k-steps, nothing staged in threadgroup memory. oMLX's `omlx/custom_kernels/quant_matmul/csrc/oq_a8_nax.metal` (Apache-2.0; its weights are repacked once at load into `PackedLinear` tiles of 128 columns) | oMLX 27B prefill 2,582-2,702 tok/s at 8-16k with no extra memory, vs inferd PF8 2,382-2,473 (+22.7 GiB of int8 copies) and bf16 qmm 1,822-1,888. Our earlier large-M int8 tile staged weights through threadgroup memory and was 4x slower (NOTES "W4A8 at 2,048 rows") | Studio: +5-10% prefill over PF8 and the 22.7 GiB back for KV (PF8 is auto-off whenever KV is sized large, e.g. the gate config: then +40%). **Minis: PF8-class prefill where PF8 cannot fit**: M5 Pro fleets ~450 -> ~750-800 tok/s; M6 270 -> est. 400-500 (int8 at ~2x bf16 on the tensor units, if its rate scales like the M5's). With prefill ~2/3 of a mini fleet's wall, +25-40% turns/min | Medium-high: one kernel family + in-place repack at load (like the W4A8 flip) + the 256/1536-row logits gate. Would also replace PF8 |
| 2 | **Flash-Next prefill kernels**: oMLX runs routed-expert gate/up on int8 activations (`m5_gather_qmm_a8`, 48 MoE layers) and the dense GDN/MLP projections on the same QxA8 kernels, plus a pipelined GDN prefill kernel (`qwen35_gdn_chunked`, `impl=pipelined`) | 34.5k prompt 5.7-6.0 s vs inferd 9.1 s (1.5x); 14k first token 2.3 vs 3.6 s; not chunk width (inferd at 8,192 rows identical) | TTFT -35% on long Flash-Next prompts; wall.py N=1-2, where oMLX and TensorFold lead us; FN mixed-arrival tails | Medium: a gathered variant of idea 1's kernel for the experts; the GDN part separately |
| 3 | **`step_tokens` 8192 for Flash-Next when prefills share steps with decoders** (idle prefill already uses `PREFILL_STEP` 8192) | wall.py 8 users: 142.7 ± 1.7 s -> 134.5 s (1 trial, -6%), decode window 464 -> 481 tok/s, b2b unchanged. #4902's lesson: every MoE chunk re-reads all experts | ~5% at 8 users on the Studio; must check conc 16k fleets (longer decoder stalls) and the mixed-arrivals TTFT tail before changing the default | Low: a default / per-model rule, 3-trial A/B + fleet check |
| 4 | **Two inferd servers behind a session-affine router for the M6 pair** (instead of a Prefill Accelerator) | Estimate above: the accelerator's fleet ceiling is x1.5 (M6 helper) / x2 (PF8 M5 Pro helper) since the host still decodes alone with its 3 GiB KV pool; two boxes double prefill, decode and KV | ~2x turns/min for a fleet on two M6s (minus routing imbalance) | Low-medium: front-end proxy keyed by session / prefix hash; no engine change |
| 5 | **Tool-loop cache regression test** (the #5062 bug class: a resume point typed so it is evicted first, every tool turn re-prefilled) | inferd snapshots before every prompt end, decodes snapshots, keeps sessions last, so it is not affected today; oMLX 0.7.3 reused 0 tokens on our follow-ups | Protects the 3x follow-up lead | Low: N-turn tool loop in `examples/agent-*`, assert `cached_tokens` on turn 5 >= turn-4 prompt - margin |
| 6 | **Prefill Accelerator for the M6** (helper runs layers [0, K) of each chunk) | Estimate above | One long prompt's TTFT 1.8x (M6 helper) to 3.5x (M5 Pro helper with PF8 on its layers); fleets <= x1.5-2, below idea 4 | High (oMLX's PR is ~10k lines): chunk grid, wire format, split tuner, helper lifecycle; plus our mixed prefill+decode steps, DFlash2 context ring and GDN snapshots |
| 7 | fp16-scale checkpoints (#5028) | inferd rejects non-BF16 scales/biases at load | Only matters for fp16 oQ checkpoints (e.g. Flash-Next oQ2.5e on 64 GB machines) | Low-medium |
| – | Bounded buffer pool (#5031), 6-row 8-bit `vk` (b5c7793e), wide chunks under memory pressure (#4902) | inferd allocates everything at load, reads weights once from 3 rows (measured level: 146.8 vs 148.1), and has no memory-pressure throttle | Nothing to take | – |
