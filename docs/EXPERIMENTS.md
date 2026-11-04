# inferd experiments

A catalogue of what was tried while building inferd (2026-10-29 to 2026-11-01): what each idea was, what it measured,
and what became of it. Every number here comes from the study log (the dated study log) or from comments in the code; the
**Source** column gives the study log section heading (shortened) or the source file. `docs/ARCHITECTURE.md`
describes the engine as it stands.

**Machines.** *Ultra* = Mac Studio M5 Ultra (80 GPU cores, 256 GB; the main target). *Max* = MacBook Pro M5 Max
(the development laptop). *M5 Pro* = Mac mini M5 Pro (20 GPU cores, 64 GB). *M6* = Mac mini M6 (12 GPU cores, 32 GB).
Model: Qwen3.8-27B (mlx-community 4-bit unless stated) with the z-lab/incoai DFlash2
drafter.

**Benchmarks.** `conc.py --mode followup`: N agents with multi-turn follow-ups on a 16k–256k context, reporting
aggregate tok/s, per-stream tok/s and TTFT (the scoreboard). `speed.py`: one stream, cold prompts. `agents.py`: an
agent fleet with ~1.2k new tokens per turn, reported as turns per minute (cells written *running:agents*, e.g. 16:4).
`shared_prefix.py`: N agents fanning out over one shared document. `decode_test` / `spec_test`: engine-level step
timings.

**Verdicts.** *default* (on by default) · *kept* (in the code path, not a switch) · *opt-in* (a knob, off) · *off*
(code kept for re-tests, off) · *removed* (code deleted) · *broken* (known wrong, off) · *superseded* (replaced by a
later design) · *measurement* (a fact we designed around).

Contents

1. [Starting point and ceilings](#1-starting-point-and-ceilings)
2. [GEMM and quantisation](#2-gemm-and-quantisation)
3. [Attention](#3-attention)
4. [KV cache and memory](#4-kv-cache-and-memory)
5. [Scheduling, prefix cache and SSD tier](#5-scheduling-prefix-cache-and-ssd-tier)
6. [Speculative decoding](#6-speculative-decoding)
7. [Neural Engine](#7-neural-engine)
8. [Multi-model](#8-multi-model)
9. [Scoreboard milestones](#9-scoreboard-milestones)
10. [Ideas recorded but not tried, and open questions](#10-ideas-recorded-but-not-tried-and-open-questions)
11. [What is ours](#11-what-is-ours)
12. [Diagnosed problems](#12-diagnosed-problems)

---

## 1. Starting point and ceilings

The engines already on the machines, before inferd *(Source: "Starting point")*:

| model | machine / engine | N = 1 tok/s | best multi-user aggregate | how it fails |
|---|---|---:|---:|---|
| 27B 4-bit | Ultra, Splash + DFlash2 | 200 | 404 at N = 4 and 16 | batches ~4, queues the rest; stops long generations early |
| 27B 4-bit | Ultra, mlx-serve | 77 | 412 at 16 (27 per stream) | DFlash turns itself off; serial when speculating |
| Flash-Next 4-bit | Ultra, oMLX, MTP off | 155–170 (MTP on) | 269 (16 × 16k), 324 (16 × 3k) | memory guard at 16 × 64k; MTPLX cannot batch |

Measured ceilings and a profile of where a step goes:

| measurement | result | machine, date | Source |
|---|---|---|---|
| streaming read bandwidth | 1,135–1,142 GB/s | Ultra, 10-29 | "Measured ceilings, M5 Ultra" |
| fp16 `matmul2d` 4096³ by tile | 124 TFLOPS (128×64), 121 (64×64), 64 (the header's 64×32 example) | Ultra, 10-29 | same |
| decode-shaped GEMM (5120 → 17408) | flat 0.23–0.24 ms from M = 1 to 64 (~760 GB/s, 67% of peak); M = 128 0.39 ms, 512 1.5 ms, 2048 3.25 ms (112 TFLOPS) | Ultra, 10-29 | same |
| `matmul2d` element types | int8 × int8 → int32 223 TOPS vs fp16/bf16 124; bf16 × int8 gains nothing; block scales only power-of-two `fp8_ue8m0` per 32 (cannot carry MLX's affine group scales) | Ultra, 10-30 | "W4A8: int8 activations…" |
| ALU rate | f32 14.8, f16 19.2, f32x2 13.4, half2 26.1 TFLOPS | Max, 10-30 | "Single-stream gap, continued" |
| MLX 0.32.3 raw batched step, 2k context | B = 1 19.7 ms (51 tok/s; 8.6 ms = 44% host overhead), B = 16 68.3 ms (234), B = 64 108.3 ms (591); at B = 64 GDN 32.9, attention 32.8, MLP 24.0 ms | Ultra, 10-29 | "Where a 27B decode step goes" |
| GDN recurrence at B = 64 | 0.38 ms per layer = 192 MB of fp32 state read and written (at bandwidth); 3 MiB per sequence per layer, 144 MiB per sequence | Ultra, 10-29 | same |
| attention at B = 64 × 2k | 2.05 ms per layer vs ~0.47 ms KV read: 4× off bandwidth | Ultra, 10-29 | same |
| dispatch cost | 1,377 dispatches per speculative cycle at ~3–4 µs each; a concurrent encoder with barriers is no cheaper | Max, 10-30 | "Single-stream gap, continued" |

## 2. GEMM and quantisation

| # | experiment | what it was | measured | verdict | Source |
|---|---|---|---|---|---|
| G1 | Own 4-bit GEMMs, day 1 | (a) dequantize to threadgroup memory then `matmul2d`; (b) raw `uint4b` per group with scale-and-add in registers | M = 1/16/32/64/128: (a) 0.15/0.15/0.16/0.19/0.27 ms, (b) 0.13/0.16/0.25/0.47/0.89 ms, MLX `quantized_matmul` 0.064/0.127/0.085/0.122/0.248 ms (Ultra, 10-29). (b) proved MLX's nibble order = `uint4b_format` | removed; MLX's NAX kernels vendored as the baseline library | "4-bit GEMM, day 1" |
| G2 | `qmv4` matvec | our matvec for M ≤ 2, weights reused across 2 rows | R ≥ 4 spills registers: 381 ms per step at R = 8 | default (M ≤ 2) | "Native engine, day 1" |
| G3 | Split-K for narrow outputs | N = 5120 gives 80 tiles; split K, float partials, reduce + residual (`qmm_splitk`, derived from MLX) | `mlp.down` 150 → 530 GB/s; linear layers per step 50 → 27 ms at M = 8–32 (Ultra) | default | "Native engine, day 1" |
| G4 | `qmvt4` | small-M kernel staged through threadgroup memory | ~300 GB/s (activations reloaded per threadgroup), slower than MLX's qmm at M = 5 | off (`QMVT_MAX` = 0) | "Agent serving, prefix cache…" |
| G5 | 8-bit and bf16 checkpoints | `qmv8`, `embed8`, `gemv16`, MLX split-K and fused NAX bf16 GEMMs | 2k context, 4-bit / 8-bit / bf16: decode B = 1 50.5 / 34.2 / 18.5 tok/s (bf16 at 87% of bandwidth); B = 32 ~600 / 515 / 357; prefill 8k 1,707 / 1,632 / 1,699; spec B = 1 ~85 / 66.7 / 46.9; acceptance 4.4 tokens/cycle at every precision | kept | "Weight precisions…" |
| G6 | Activation quantization study | fake-quantize every quantized linear's input (`a8_quality.py`), 2k tokens of two texts | per row per 64-group: KL 0.0030 / 0.0018, top-1 97.7 / 98.9%, ppl 7.182 → 7.204 / 2.590 → 2.589; per 128-group KL 0.0038 / 0.0026; **per row KL 0.049 / 0.042, top-1 89 / 94% (unusable)**; reference: 4-bit weights vs bf16 KL 0.044, 8-bit 0.0016 | 64-group chosen | "W4A8: int8 activations…" |
| G7 | **W4A8 / W8A8** | int8 activations × the checkpoint's 4-/8-bit weights on the integer tensor path, affine epilogue per group | linear layers per step M = 5/40/80/160/2048: 29.8/44.0/83.5/116.1/950 → **22.5/38.8/56.2/95.6**/1,169 ms; verify step (R = 5, 2k) B = 1 32.1 → 25.8, B = 8 50.9 → 46.1 ms; conc 16k N = 8 40.7 → 53.3 per stream (agg 282 → 374, TTFT 1.6 → 1.0 s), N = 16 22.3 → 32.9 (317 → 455) (Ultra, 10-30). Logits: top-1 ok, KL ≤ 0.0007 normal text | default for 3 ≤ M < 256; MLX qmm above | "W4A8: int8 activations…" |
| G8 | Signed vs unsigned int4 | flip nibbles in place (q ^ 8) to run signed i8 × i4 | 15–20% faster than u8 × u4; no weight copy | default | same |
| G9 | Final-store branch | split-K partial vs direct store as a runtime branch | whole kernel 1.7× slower; made a template parameter | lesson | same |
| G10 | SLC-resident micro-benchmark | cycling 8 weight copies | served from the system-level cache: misleading; `decode_test gemm` now walks every layer's real weights | lesson | same |
| G11 | W4A8 tile rows, first pass | 16 / 32 / 48 / 64-row tiles | 32 × 128 beat the others at every M; 64-row tiles spill (10× slower) | 32 × 128 kept | same |
| G12 | 16-row tiles for M ≤ 16 | `w4a8_16x128_*`, split cap 2560 | linear layers at M = 5 22.5 → 20.4 ms; one stream 8k 122 → 131 tok/s, 16k 114 → 122; worse above 16 rows (M = 20 28.6 vs 24.6) | default (M ≤ 16) | "Head-to-head and the morning of 2026-10-30" |
| G13 | Epilogue tweaks, 10-30 morning | scales staged in threadgroup memory; two groups' MMAs in flight; 40/48-row tiles; TN = 64/256 at M = 5 | scales: no change at M = 40/80, worse at M = 5; two in flight: M = 40 38.7 → 48.4 ms; 40/48-row tiles 65 / 42 ms at M = 40; TN: no gain. A tile pass runs at ~115 of 223 TOPS; the 64-wide K per MMA call is the limit | rejected | same |
| G14 | More small-M variants (Max) | 8-row W4A8 tiles; single-simdgroup tiles (16×32, 16×64); more split-K at small M; group-major scale copies; compile-time epilogue capacity; dflash-mlx's `verify_qmm` | group-major scales +3% at 40–80 rows, nothing at 8; compile-time capacity broke 32-row tiles; `verify_qmm` slower than MLX's qmm; the rest no gain | rejected | "Diagnosed 2026-10-30: block 8 was useless…" (closing paragraph) |
| G15 | `qmvh`: lanes along K, 8 rows | half2 FMAs; second version packed half2 + 0x6400 dequant | 104–135 ms and 118–281 ms vs W4A8 36 ms at 8 rows: each lane re-reads 8 rows × 16 activations per 512-wide block (~8× the weight bytes) | removed | "Single-stream gap, continued" |
| G16 | `qmvc`: lanes along output columns | 8 activation rows staged as fp16 in threadgroup memory, broadcast | 116 ms (1 column per thread) / 193 ms (2) vs W4A8 39 ms: neighbouring threads read addresses a row apart | removed | same |
| G17 | Column-dot over a repacked copy | per 256-column block and group, 32-byte chunks and (s, b) contiguous | 100.8 vs 36.3 ms: ~600 instructions per 512 MACs, not the access pattern | removed | same |
| G18 | W4A8 without its epilogue; staged weights | timing variants | raw streaming 34.2 vs 37.2 ms at 8 rows (epilogue ~8%); threadgroup-staged double-buffered weights 39–44 ms (slower) | measurement | same |
| G19 | `qmvs` | `simdgroup_matrix` 8×8 MMAs, nibbles staged as bf16 via the 0x4300 trick, bf16 activations | correct and more accurate (KL 0.0028–0.005 on the repetitive sequence vs 0.009), but 32.8 ms vs W4A8 19.7 at 8 rows (Ultra) | removed | "attn_mpp: verify attention…" (closing bullet) |
| G20 | Split-K by GPU core count | split below 480 tiles on ≥ 60 cores, else 3 × cores | 5-row GEMMs M6 162 → 138.5 ms (no split), M5 Pro 71.6 → 66.4 ms | default | "Mac minis (2026-10-30)" |
| G21 | Split cap, `rmsnorm_a8`, `CB_SPLIT` | split-K cap 6 × / 12 × cores for 16/32-row tiles (was 2560/1280); norm fused with int8 quantization; commit the step in pieces | together: 8-row verify step 25.4 → 23.9 ms; 8k single stream 177 → 181.7 tok/s (Ultra). Code comments: the cap alone 25.4 → 24.2, `rmsnorm_a8` alone 24.4 → 23.9 | default | "Closing the gap…" (opening bullet), `qwen35.h` |
| G22 | `silu_mul_a8` / `rmsnorm_a8` on the Max | fused activation quantization | `silu_mul_a8` bit-identical, on; `rmsnorm_a8` first measured slower on the Max (49.1 → 50.9 ms per cycle, within its ±7 ms noise), later on after the Ultra measurement | default | "Single-stream gap, continued" |
| G23 | `DEFER_SPLIT` | fold the split-K reduce into the next norm, −290 dispatches | slower (one threadgroup per row): 23.9 → 24.7 ms per 8-row step (code) | off | "Closing the gap…", `qwen35.h` |
| G24 | Per-shape tuning at 8 rows | W4A8 vs the 1-row matvec per shape; split K for gate_up; touch-prefetch of the weight tile; 8-row tiles | W4A8 wins narrow-deep shapes (down 4.37 vs 4.52 ms) and loses wide ones (gate_up 8.3 vs 6.5); splitting K: gate_up 8.3 → 9.1; prefetch: no gain, the extra kernel parameter alone cost ~10% on in_proj; 8-row tiles gate_up 9.4 | rejected | "Closing the gap, continued" |
| G25 | **QMST** (Splash's staged tile) | weights repacked at load so a 32-column tile's payloads per 32-input step are contiguous; dequantize with scale to fp16 in a double-buffered stage; Rows × 32 × 32 `matmul2d`, no epilogue; golden-ratio spread start | per shape at 8 rows (Ultra) vs W4A8: down 3.97 vs 4.36 ms, lm_head 0.71 vs 0.88, attention in_proj 0.78 vs 1.00, gate_up 7.9–8.5 vs 8.25, GDN in_proj ~equal; without the spread start gate_up 9.77. In decode_test 8-row GEMMs 19.7 → 18.3 ms; **in the server 8k decode 189 → 111 tok/s** (drafter broken, §12) | off (`QMST=1`); bug fixed, not re-measured | "Splash's source", "Diagnosed 2026-10-31: QMST…", "QMST drafter bug: cause found" |
| G26 | Is the W4A8 epilogue the limit? | epilogue removed (`-DNOEPI`, timing only) | M = 16/32/64/128: 20.1/25.3/41.5/77.1 vs 19.4/25.7/42.9/81.5 ms; per shape at M = 64 56–90 TOPS. The `matmul2d` runs in K = 64 steps (one per scale group) are the limit | "dual residency" (per-channel int8 weight copy) dropped | "Diagnosed 2026-10-31: the W4A8 epilogue…" |
| G27 | Wide-GEMM tile variants | 32×128 on 8 simdgroups; 64×128 on 8; the 8-bit checkpoint's W8A8 | M = 32/64/128: default 25.4/41.5/76.9 ms; 8 simdgroups 40.5/75.7/143.1; 64×128 61.1/72.5/130.7; W8A8 M = 64 50.3 ms (53–71 TOPS): neither the int4 operand nor the epilogue is the limit | rejected (instances kept: `W4A8_NSGW`, `W4A8_TM=64`) | "Wide GEMMs: tile variants…" |
| G28 | Large-M int8 tile for prefill | 128 × 128, weights staged once per group, 8 simdgroups of 32×64×64 int8 matmuls | correct but 4,076 ms per 2,048-row chunk vs W4A8 1,174 and MLX qmm ~1.0 s (~115 TFLOPS, ~90% of bf16 peak) | removed | "Agent fleets (2026-10-30 evening)" |
| G29 | **lane4** (TensorFold-style) | weights repacked so a 32-column tile's codes per 64-group are one 1 KB block; bf16 activations from device memory into 16×32×64 `matmul2d`; affine per group; split K across 4 or 8 simdgroups | Max, all linears per step: W4A8 M = 5/8/16 36.3/35.9/37.0 → 30.3/30.9/31.9 ms (~520 GB/s), lm_head 1.77 → 1.31; above 16 rows compute-bound (M = 32 53.5 vs 44.8). **Ultra: slower** (M = 5 20.9 vs W4A8 19.6 ms); end to end 16k × 1 147 → 125, 64k × 1 166 → 129 tok/s; drafter acceptance 2.82 → 2.33 (cause not found) | off (`LANE=1`; costs a ~15 GB copy) | "Lessons from the mlx.fast competition" |
| G30 | lane8 | lane4 layout with int8 activations for 17–64 rows | M = 24/32/48/64: 43.1/44.1/66.5/78.6 vs W4A8 44.3/45.1/68.5/69.9 ms; 2 row tiles per simdgroup slower | removed | same |
| G31 | Wider loads for 8-bit KV decode | the KV8 decode path uses 4-byte loads | "wider loads are the next thing to try" | not tried | "8-bit KV cache" |

## 3. Attention

| # | experiment | what it was | measured | verdict | Source |
|---|---|---|---|---|---|
| A1 | Attention v2 (`attn_gqa`) | the 6 query heads sharing a KV head processed together (K/V read once, v1 read it 6×); lanes own positions; flash-decoding splits | decode total tok/s 2k: B = 1 50.5, B = 16 426, B = 64 865; 32k: B = 1 43.5, B = 16 227, B = 32 286 (v1 sweep: 8k B = 64 353, 32k B = 64 118, 64k B = 32 62). An intermediate version (lanes over dims) was slower than v1 | default (`ATTN_V1=1` restores v1) | "Ragged batches, prefill, and attention v2" |
| A2 | Flash prefill | gather paged KV, run MLX's NAX flash attention for ≥ 64 new rows | prefill 8k 1,488 → 1,707 tok/s; 32k 1,643 | default | "Agent serving, prefix cache…" |
| A3 | One multi-row kernel for everything | `attn_gqa` mode 2 for decode too | decode 1.7× slower (idle simdgroups) | rejected: separate row kernel | same |
| A4 | First speculative server | 5-row verify blocks fell into the flash path | N = 16 5.6 tok/s per stream (re-gathers the whole KV per step) | fixed with multi-row `attn_gqa` | same |
| A5 | `attn_paged_nax` | 6 heads × ≤ 5 rows packed in one 32-row NAX tile; one key tile = one KV page | 64k, R = 5: B = 8 159 → 92 ms, B = 32 510 → 285 ms; conc 64k N = 8 23.2 → 38.7 per stream (agg 297), N = 32 7.2 → 12.1 (378). Probes: a float P operand takes the slow fp32 path (cast to bf16: −13 ms); hoisting V loads and splitting D two ways were slower | superseded by attn_mpp (fallback) | "NAX verify attention" |
| A6 | `NAX_ROWS` multi-tile for follow-up prefills | 32- or 64-row tiles over longer runs | correct, but 1.6× slower than gather + flash at 128 rows × 64k (compute-bound) | rejected for prefill | "Where a 16-agent cycle goes now" |
| A7 | `NAX_ROWS` = 8 for 8-row verify | 8-row blocks through the paged kernel | 64k 64.6 vs 74.5 ms via gather + flash | default | "Diagnosed 2026-10-30: block 8 was useless…" |
| A8 | Long-context verify tuning (old kernel) | 48-row tile; head dim split 2; 64-key steps; `NO_NAX_ATTN`; split sizes 64–256; `ATTN_SPLIT_TGS` | at 64k: 48-row tile −0.5 ms; D split no change; 64-key steps 39.8–41.1 ms (worse); `NO_NAX_ATTN` 43 ms; split sizes all ~35 ms; `ATTN_SPLIT_TGS` 1024: 36.9 → 35.1 ms | `ATTN_SPLIT_TGS` 1024 default; rest rejected | "Studio after the drafter fix" |
| A9 | **attn_mpp** | whole-tile `matmul2d`: S^T = K Q^T with K as the left operand, 128-key groups on contiguous pages, online softmax in 64-key halves, lazy O rescale | per-call QK cost 8.7/6.0/4.9 ms for 32/64/128-key tiles (V side 4.0); 8-row verify step 8k 25.7 → 25.3, 32k 30.5 → 27.7, 64k 35.1 → 30.7, 128k 44.2 → 36.8 ms, 256k 48.9; server single stream 8k 174, 32k 146 (was 135), 64k 132 (124), 128k 103 (78 overnight). Pipelining QK one step ahead did nothing; on 2–5-row blocks 3–12% slower than paged NAX at first | default (6–8 rows) | "attn_mpp: verify attention on whole-tile matmuls" |
| A10 | attn_mpp with 8-bit KV | int8 K as left operand, V's scale folded into P | correct; not faster than bf16 then (128k 37.9 vs 35.4 ms per 8-row step) | kept for memory; later the default path (A14) | "Closing the gap, continued" |
| A11 | 32-row attn_mpp (mode 5) | for 2–5-row blocks (several agents) | 5-row verify step 128k × 16 227 → 196 ms, 128k × 4 70.6 → 61.6, 16k × 16 84.9 → 80.2, 16k × 8 55.3 → 51.9; 4-row 212 → 186 at 128k × 16 | default | "Agent fleets (2026-10-30 evening)" |
| A12 | Cascade attention | attend once to a physically shared prefix (48-row tiles from several sequences), suffix per sequence | correct (replies identical character for character), engaged in 163 of 200 fleet steps, but 5-row verify at 64k × 16: private prefixes 147 ms → shared blocks without cascade 110 ms → with cascade 124 ms (attention compute-bound at ~47 TFLOPS) | off (`CASCADE=1`) | same |
| A13 | Chunk tiles (mode 6) | a turn's 9–256 new rows as 8-row × 6-head tiles split over keys, instead of gathering the whole KV for flash | ms per step: 128k B = 1 × 32 177 → 79; 128k B = 16 × 32 2,866 → 844; 32k B = 16 × 32 888 → 440; 16k B = 16 × 16 300 → 198; ties at 2k, 1–2% slower at 512 rows (hence the 256 cap). Logits cos 0.9980/0.9994 vs flash 0.9978/0.9993 | default | "Chunk tiles (mode 6)" |
| A14 | **Verify-attention rework** (2026-10-31) | diagnosis by ablation (§12), then: head-plane KV layout; softmax lanes (4 lanes per row, xor shuffles, m/l in registers); scores `store()`d by the matmul (key-major); group scales in threadgroup memory; int8 queries (`ATTN_IQ`) | 8-bit KV step ms, old → new: 1 × 8k R8 24.2 → 23.5; 1 × 32k R8 26.8 → 25.0; 1 × 128k R8 37.6 → 30.7; 4 × 64k R8 59.3 → 45.7; 4 × 128k R8 88.6 → 61.2; 8 × 16k R4 38.1 → 37.1; 8 × 64k R4 58.6 → 55.1; 8 × 128k R4 85.9 → 79.0; **8 × 128k R8 153 → 109**; 16 × 64k R4 103.4 → 97.1. Layout alone 164 → 153; int8 queries ~3–5% alone; group scales no measurable gain alone; bf16 KV got slightly slower (121 → 125 at 8 × 128k R8) | default | "2026-10-31: long-context verify attention reworked", "Changes…" |
| A15 | Two threadgroups per core for one sequence | `ATTN_SPLIT_TGS_ONE` = max(128, 2 × cores) | 256k × 1 R8 38.5 → 36.4 ms, 128k 30.8 → 29.9, 32k unchanged; with 2–4 sequences 160 was 20–25% slower than 128 at 128k, so single-sequence only. Release note: 256k × 1 95 → 111 tok/s | default | `qwen35.h`, `engine/README.md` |
| A16 | Multi-agent attention with KV8 (old kernel) | 8-bit KV at 4–16 agents | no help: the old kernel was compute-bound at ~25 TFLOPS | superseded by A14 | "Agent fleets (2026-10-30 evening)" |
| A17 | Templated `attn_gqa_row` | template over (GQ, Dh) for other models | regression on the 27B (§12); fixed with named registers for Dh 256 | kept | "Diagnosed 2026-11-01: templating attn_gqa_row…" |

## 4. KV cache and memory

| # | experiment | what it was | measured | verdict | Source |
|---|---|---|---|---|---|
| K1 | 8-bit KV, first version | int8 K/V, float scale per (position, KV head), absmax/127 | KV memory halved (2.3M → ~4.5M tokens in 140 GiB); 64k B = 32 plain step 182 → 152 ms, verify 288 → 276; logits within noise except the flash path flipping top-1 on a fragile repetitive sequence (KL 1.1) | opt-in at the time | "8-bit KV cache" |
| K2 | KV8 vs bf16 at agent load | `decode_test bench`, R = 4 | 64k B = 8 64.8 vs 58.8 ms, B = 16 115.1 vs 103.8; 128k B = 8 97.9 vs 86.3, B = 16 181.4 vs 159.1 (10–12%); 3k-token logits top-1/top-5 correct. Splash stores KV as INT8 by default | led to the default | "Splash at 128k runs ~4 lanes…" |
| K3 | KV8 8-row verify at 128k (before rework) | blocks 4/5/8 at B = 8 | bf16 98.5/110.1/144.5 ms vs KV8 86.0/98.1/**164.1** (the KV8 8-row path was slower) | fixed by A14 | "2026-10-31 morning…" |
| K4 | **KV8 by default** | after the attention rework | server, 32k × 8 370 → 449 tok/s, 32k × 16 472 → 519, 128k × 8 220 → 263 (no tuner) / 289 (tuner); 128k × 8 follow-ups earlier: bf16 220 vs KV8 239 tok/s | default since 2026-10-31 (not Gemma-4) | "Server A/B after the attention rework", "2026-10-31 morning…" |
| K5 | Head-plane layout | `[Hkv][slots][Dh]` per layer | bit-identical logits; 164 → 153 ms at 8 × 128k R8 alone | default | "Changes…" |
| K6 | Contiguous page groups | `alloc_block(prefer)`: page i right after page i − 1 within groups of 4, so attn_mpp reads 128 keys in one matmul | part of A9 | default | "attn_mpp…" |
| K7 | Batch eviction | `EVICT_BATCH` = 64 instead of one block per allocation | scattered tables (`BENCH_SHUFFLE`) at 64k: 4 × 8 rows 54.9 → 61.5 ms, 8 × 8 95.9 → 109.2, 4 × 5 45.2 → 53.8, 8 × 5 76.9 → 93.6. After the fix (40 GiB pool filled by 128k × 4, then 64k): 64k × 4 273 tok/s, × 8 310 (fragmented 241/262; fresh 259/289); 128k × 4 228 (208) | default | "Diagnosed 2026-10-30: a full KV pool handed out pages in descending order" |
| K8 | Grouped allocation | a stack of fully free aligned groups of 4; a sequence starting a group takes a whole one | shuffled pages: 256k × 1 R8 38.3 → 42.1 ms (+10%), 128k × 4 R8 61.1 → 69.4 (+14%); a fresh server gave 256k × 1 117 tok/s vs the milestone's 95. Release note: 128k × 4 241 → 270 | default | "v11: grouped KV allocation" |
| K9 | `MTLResidencySet` over every buffer | what MLX and Splash do | no gain once the tape sizing was fixed (16 × 6k) | opt-in (`RESIDENCY=1`); default with the ANE split (§7) | "Diagnosed 2026-10-30: the "16-sequence non-GPU time"…" |
| K10 | M6 memory | 32 GB, buffers faulted in on first use | idle footprint 16 GB hid the real one; KV 5.5 GiB + 15 slots reached 27 GB and swapped (decode fell to 10 tok/s); 2.5–3 GiB fits (22 GB) | sized by hand, later auto | "Mac minis (2026-10-30)" |
| K11 | Auto-sized KV pool | from Metal's working-set budget less model, drafter, state and a margin | fine on the minis; on the Ultra with a small model it filled 222.5 of 224 GiB and hung (§12) | default on minis; explicit size on the Ultra | "Diagnosed 2026-11-01: a server that hung…" |
| K12 | ANE memory margin | KV headroom with the ANE split | study log: "2.5 GiB more KV headroom with the split"; code: `ANE_MARGIN_GIB` default 0.5 ("2.5 starved the KV pool, 4× slower") | 0.5 in code | "Diagnosed 2026-11-01: the ANE split…", `server.mm` |

## 5. Scheduling, prefix cache and SSD tier

| # | experiment | what it was | measured | verdict | Source |
|---|---|---|---|---|---|
| S1 | Native step encoding | one command buffer per step, fused projections, paged KV, slot pool | host overhead ~0.5 ms per step vs MLX's 8.6; 2k context B = 1/8/16/32/64: 20.7/37.7/40.4/50.8/83.4 ms (48/212/396/630/768 tok/s) vs raw MLX 19.7/40.1/68.3/74.6/108.3 ms | kept | "Native engine, day 1" |
| S2 | Ragged steps, chunked prefill | T rows from S sequences, prefill and decode in one pass | checked against mlx_lm at chunk 1, 7 and 24 | kept | "Ragged batches…" |
| S3 | Prefix cache with GDN snapshots | full 32-token blocks in a refcounted trie; snapshot at prompt end − 16, block-aligned | follow-up turns restore 16,352 of ~16,470 prompt tokens | kept | "Agent serving, prefix cache…" |
| S4 | Reply KV reuse | the template re-renders the previous turn, so its reply KV cannot be reused | TTFT at N = 16 ~2.6 s is compute-bound re-prefill of ~2k follow-up tokens; Splash reuses it ("deduplicated 64") | not done | "Agent serving…", "Diagnosed 2026-10-30: Splash's single-stream lead…" |
| S5 | Fan-out prefix sharing | a request sharing ≥ 1,024 tokens with a still-prefilling sequence asks it for a snapshot there and waits | 32k × 8 fan-out 188.8 s (283k prompt tokens, 0 cached) → **24.6 s** (Splash 22.9), turns 37.8/min vs Splash 33.8, TTFT median 2.4 vs 6.2 s; 64k × 16 fan-out 59.3 vs 53.8 s, turns 26.8 vs 26.1/min, TTFT median 5.4 vs 27.5 s | default | "Agent fleets (2026-10-30 evening)" |
| S6 | SSD tier v1 | evicted leaf chains (tokens, KV, state, drafter ring) to one file each | Max, 3 agents × 5.3k, pool for one agent: follow-up TTFT 0.24–0.33 s vs 7.2 s re-prefill, ~0.04 s per restore, replies identical to a large-pool reference | superseded by v2 | "SSD tier for the prefix cache" |
| S7 | Minis with the tier | fleets | M6 16:16 0.79 → 6.8 turns/min; M5 Pro 16:16 only 5.3 (spill churn: 136 full ~0.85 GB chain rewrites, ~23 s blocking I/O) | led to v2 | "Agent fleets (2026-10-30 evening)" |
| S8 | **SSD tier v2, content-addressed** | one file per block named by its prefix hash, one state file per chain; spill writes only new blocks, restore reads only missing ones | M5 Pro correctness 9 of 9 replies identical; spill I/O no longer the cliff (137 spills 11.5 s, restores 3 s total) | default when `SPILL_DIR` is set | same, "Results of the 2026-10-30 evening runs" |
| S9 | Spill inner nodes too | every evicted snapshot is spilled, not just leaves | M5 Pro replay 16:16 5.34 → 13.61 turns/min, new tokens 192k → 62.7k, hit 0.786 → 0.930 — but did not hold in the full fleet (§12) | kept | "Diagnosed 2026-10-30: the M5 Pro 16:16 cliff…" |
| S10 | Spare the matched snapshot | admission and `take_snapshots` never evict the node they are about to use | full fleet 16:16 5.02 → 14.33 turns/min, 211k → 59k new tokens; no failed restores | kept | "Diagnosed 2026-10-31: the 16:16 cliff…" |
| S11 | Persistent tier | token files renamed into place, MANIFEST, rebuild at start-up, flush on SIGTERM | flush of 3 contexts (1.5 GiB) in 2.7 s; restored from disk in ~0.23 s (re-prefill 3 s); 6 of 6 replies identical (Ultra) | opt-in (`SPILL_PERSIST=1`) | "Persistent SSD tier" |
| S12 | **KV-budget admission + preemption** | admit by blocks (prompt + 512 generated + draft block) against free + evictable; preempt the newest sequence when the pool runs dry | M6 32k × 4 deadlock (0.02 tok/s) → 12/12 turns in 121 s, 12.2 tok/s, 96.5% cache hits, ~0.5 s per 33k context swap through SSD, 0 preemptions | default | "v10: KV-budget admission + preemption" |
| S13 | Tokenization cache | encode only new messages (split before `<|im_start|>`) | 84 ms per turn at 126k tokens (serialised under the GIL: ~1.3 s for 16 agents) → 0.09 ms cached; 480/480 random conversations tokenize identically | default | "Front end: a turn re-tokenized…" |
| S14 | Server CPU cost | `sample` profile | 99% of the main thread in `waitUntilCompleted` | measurement | "Where a 16-agent cycle goes now" |
| S15 | Chunk tiles + tokenization cache, server | conc follow-ups (Ultra) | 32k N = 1 155 tok/s, TTFT 0.17 s (was 136 / 0.24; Splash 148); 32k N = 16 381 → 472 tok/s, TTFT 3.66 → 1.82 s (Splash 354, 8.3 s median); 64k N = 16 265 → 358 (Splash 300); 128k N = 16 160 → 184 (Splash 266) | — | "Smoke after chunk tiles + cached tokenization" |

## 6. Speculative decoding

| # | experiment | what it was | measured | verdict | Source |
|---|---|---|---|---|---|
| P1 | **DFlash2 integration** | drafter context append, draft block, target verify with GDN tapes, accept, replay, all in one command buffer; one CPU sync per cycle | acceptance 3.8 tokens/cycle on agent follow-ups (half the cycles accept all 5); cycle 38–42 ms (verify ~32, drafter ~6); conc 16k per stream N = 1/4/8/16: 46.5/26.0/21.0/17.3 → **87.5/62.8/40.4/22.4** | default | "Agent serving, prefix cache…" |
| P2 | Losslessness | `spec_test`: greedy speculative vs plain | identical except near-ties from different kernel paths (traced with `tie_check.py`); not batch-invariant at the bit level, like vLLM | kept | same |
| P3 | Adaptive block (`SPEC_BLOCKS`) | L = 4 above 6 decoders, 3 above 12, 2 above 24 | 16k N = 8/16/32: 45.6/21.2/10.3 vs 40.7/22.3/14.0 with L = 5 always: extra verify rows are nearly free next to the weights and KV they share | off (knob kept as an override) | "Full scoreboard with DFlash2" |
| P4 | Draft block 8 vs 5 (before the drafter fix) | DFlash2's trained block size | 4.02 → 4.08 tokens/cycle; one stream 113 vs 122 tok/s | rejected then; the cause was a bug (§12) | "Where a 16-agent cycle goes now", "Head-to-head…" |
| P5 | Tile-aware block | block 4 instead of 5 when it saves a 32-row W4A8 tile (8 agents 40 → 32 rows, 16 agents 80 → 64) | 16k N = 8 375 → 406 total (52.9 → 58.0 per agent), N = 16 464 → 488 | default (fixed rule) | "Head-to-head…" |
| P6 | `DRAFT_BITS=4` | the drafter's layer linears in 4-bit | 4.06 vs 4.08 tokens/cycle at block 5, ~1 ms per cycle; at block 8 5.71 vs 5.78, ~1–2 ms less. Exposed a view-offset bug (§12) | default since 10-30 | "Head-to-head…", "Diagnosed 2026-10-30: block 8 was useless…" |
| P7 | **Block 8 after the drafter fix** | `draft_attn` for 8 rows | 2k code prompt (Max): 4.28 → **5.68** tokens/cycle, 44 of 90 cycles accept all 8. Max A/B (2k / 8k, prefill / decode tok/s): Splash 1,185/136 and 972/124; inferd before 732/94 and 701/87; after 916/107 and 713/88. Ultra single stream 8k 122 → 163–174, 32k 106 → 135, 64k 95 → 124 (Splash 219/190/171); 5.28 tokens/cycle at 8k, level with Splash | default (`DRAFT_BLOCK` = 8) | "Diagnosed 2026-10-30: block 8 was useless…", "Studio after the drafter fix" |
| P8 | 8-row drafter attention shape | one 8-row simdgroup vs 2 groups of 4 across threadgroups | single simdgroup spills: 10.5 vs 8.5 ms per drafter block (Max) | `draft_attn8g` default | "Diagnosed 2026-10-30: block 8…" |
| P9 | Drafter time split (`DRAFT_SKIP`) | timing only | block 8 on the Max: attention ~2.1 ms, MLP ~3.6 (8-bit), head ~1.4, rest ~1 | measurement | same |
| P10 | Drafter attention splits, sharded top-16 | 64-key splits (`DRAFT_CHUNK`); top-16 over 64 shards then merge | drafter block 4.5 → 3.7 ms; 3.7 → 3.4 ms (Ultra); lossless check passes | default | "Closing the gap, continued" |
| P11 | Block 8 up to 4 decoders (`SPEC_WIDE` = 4) | wide block for few agents | 16k N = 4 314 → 339; above 4 block 5 better on real agent text (N = 8 445 vs 405, N = 16 515 vs 450), though `spec_test`'s prompts favoured 8 | default | "Agent fleets (2026-10-30 evening)" |
| P12 | Block cost by regime | step ms at blocks 4/5/8, KV8 | 8 × 16k 37/50/56; 8 × 64k 55/67/78; 8 × 128k 79/91/109; 16 × 16k 61/75/101; 16 × 64k 97/112/147; 16 × 128k 146/161/209; 4 × 16k 28/31/34; 4 × 128k 48/52/61. With ~3.5/4.0/5.3 tokens per cycle: block 8 wins for 8 agents from 64k (+7–10%), ties at 16k; 16 agents want 4 at 16k, 8 at 128k | led to P13 | "Block size by regime…" |
| P13 | **BlockTuner** (`SPEC_AUTO`) | per regime, pick the block maximising n · E(L) / T(L) from measured step times and acceptance | server: 32k × 8 449 → 447, 32k × 16 519 → 512, **128k × 8 263 → 289** (last agent done 15.6 → 14.2 s); single stream 8k neutral (181.8/187.1 vs 188.9/179.3). Exploration then made gentler (one step in 8; start from the fixed rule) | default | "Server A/B after the attention rework", "Tuner, single stream" |
| P14 | Block policy at 128k × 8 (before the rework) | blocks 4/5/8 | within ~5% at ~3.4/4.0/5.3 tokens per cycle: "block policy is not the lever" there | measurement | "2026-10-31 morning…" |
| P15 | Drafter vocabulary cut (`DRAFT_VOCAB`) | score ids [0, 98304) plus the special-token tail via row views of the LM head; 0.06% of code and 0.18% of English tokens lie between | Max, challenge prompt: drafter block 6.6 → 5.2 ms, acceptance unchanged (2.48). **Ultra, agent text: acceptance 4.46 → 4.15 (16k), 5.14 → 4.62 (64k)**, 148–150 vs 143–145 tok/s; the saving there is ~0.5 ms of ~26. M5 Pro fleet A/B: noise | reverted to opt-in | "Lessons from the mlx.fast competition", "M6 16k fleet" |
| P16 | **Speculative sampling** (0.1.1) | exact accept/resample for a point-mass proposal from each verify row's top-16 on the GPU | coding benchmark (48 tests, t = 0.6, 32 in flight): before, 9–14 min per run vs Splash ~7, 20 tok/s per request; M5 Pro 1 stream 16.9 → 42.9 tok/s, 4 streams 28 → 110 total; 400-sample distribution check matches plain sampling within noise | default (`SPEC_SAMPLE`) | "0.1.1: speculative sampling" |
| P17 | Short-window decode (mlx.fast challenge prompt) | batch 1, ~1,065-token prompt, 128 decode steps | Max: 51.6 tok/s over the first 128 tokens (48.7–62.3), 85.9 over 512; prefill 955 tok/s. Challenge leaders that day: Apple 59.2 decode / 1,140 prefill; local TensorFold baseline 61.8 / 1,115. Not the challenge harness | measurement: short windows are our weak spot | "mlx.fast / cuda.fast challenge" |

## 7. Neural Engine

| # | experiment | what it was | measured | verdict | Source |
|---|---|---|---|---|---|
| N1 | Decision | use the ANE only on work that does not compete with a saturated GPU batch (an earlier study saw a 12–17% cost beside a saturated GPU) | — | — | "Decisions (2026-10-29)" |
| N2 | Prefill on the small GPUs | where mini fleets spend time | M6 prefill ~15 TFLOPS (1,024-row chunk 3.65 s, ~280 tok/s), ~80% of its bf16 peak; Splash's 480 tok/s on the M6 is above what the GPU can compute, so its ANE assist matters there (Splash calibrates it at 3.8% on the Ultra) | led to N3 | "Mac minis (2026-10-30)", "Diagnosed 2026-10-30: Splash's single-stream lead…" |
| N3 | `ane_probe` | our own client of the private AppleNeuralEngine service: MIL text, weights as IOSurface inputs, one FFN slice (hidden 5120, int8 weights + per-row fp16 scales) | per 512-channel unit: Max W8A16 R = 1024/2048 1.40/2.69 ms (11.5/12.0 TFLOPS); Max int8 activations R = 1024 0.99 ms (16.3); M5 Pro W8A16 1.44 ms (11.2, GPU busy); M6 W8A16 0.71/1.52 ms (22.7/21.2, GPU busy). M6 GPU ~2.1 ms a unit at 2048 rows; M5 Pro GPU ~0.55 ms at 1024 | led to N4 | "Neural Engine FFN split for prefill" |
| N4 | **ANE FFN split** | per layer, the GPU computes channels [0, Cg), the ANE the rest; weights re-quantized per layer from our 4-bit weights | `decode_test prefill`, units 0/12/16/20/24 on the M6: 259/295/309/320/**335** tok/s (+29%, still rising); M5 Pro units 0/6/10/14/18: 475/462/474/**488**/476 (+3%) | default off; M6 uses 24 | "Diagnosed 2026-11-01: the ANE split made server prefill slower…" |
| N5 | Rotated W8A8 for gate/up | random-sign 512-point Walsh–Hadamard rotation, per-token scale per 2,560-channel segment | 1,000-token prompt (Max): GPU alone KL 0.00017, W8A16 split 0.00091, **W8A8 rotated 0.00015**; the Max's GPU is stronger than its ANE, so no gain there (730–875 tok/s either way) | default (`ANE_A8`) | "Neural Engine FFN split for prefill" |
| N6 | Residency set with the split | keep every buffer and the ANE surfaces wired | server fleet prime: before, M6 31 → 42–44 s and M5 Pro 40 → 70–83 s (worse than GPU alone); after, M6 8k prime 25.4 s vs 31.2 GPU alone (**+23% in the server**), wired steady at 24.7 GiB | default with `ANE_UNITS` | same |
| N7 | `ANE_MIN_ROWS` | smallest step that splits | M6 8k fleet, 8 agents: GPU alone 9.67 turns/min; split at ≥ 512 rows 6.42 (mixed steps pay three command buffers and two host waits per layer); at ≥ 1,024 rows **10.88** (+12.5%), prime 31.2 → 25.5 s | default 1024 | same |
| N8 | `ANE_A8D` | int8 intermediate for down (rotation as a grouped conv with a BLOBFILE constant, per-token quantize) | inf in layer 0, all-zero output from layer 1; the rotation and quantize pieces check out in isolation (`ane_rot_test`); suspect the int8 × int8 down matmul | broken, off | same |
| N9 | M5 Pro split | same design | +3% in `decode_test` (its ANE ~2.6× slower per unit than its GPU) | not pursued; M6 only | same |
| N10 | Fleets with the split | `agents.py`, 2 rounds | M6 8k, 8/16/32/64 agents: GPU 9.67/9.51/9.58/9.66; **ANE24 10.99/10.94/11.16/11.27**; Splash 8.73/7.54/8.85/6.82 (32 of 128 turns refused at 64). M6 16k, 8/16/32: GPU 8.22/8.16/8.25; ANE24 9.32/8.78/9.04; Splash 8.23/7.01/6.82. Gain +13–17% at 8k, +8–13% at 16k. Splash's own ANE share: 0.74 on the M6, 0.41 on the M5 Pro | — | "Overnight agent-fleet results", "M6 16k fleet" |

## 8. Multi-model

All smoke ports on 2026-11-01, matching mlx_lm (`ref_logits.py compare`, top-1 ok), no drafters *(Source: "Other models
on inferd")*.

| # | experiment | measured | verdict | Source |
|---|---|---|---|---|
| X1 | Qwen3.5-9B (same hybrid, GQ 4) | cos ≥ 0.9997 decode and chunked | kept | "Other models on inferd" |
| X2 | Qwen3.6-35B-A3B MoE: sort by expert + MLX gathered qmm vs per-row GEMV | sorting tripled prefill (1,919 → 5,858 tok/s at 8k); the GEMV is faster up to ~8 tokens (B = 1 6.05 vs 8.43 ms) | both kept, switch at 128 routed rows | same |
| X3 | Muse Glimmer 30B incl. sliding window 2048 (our kernels + patched MLX flash) | cos ≥ 0.9997 at 3,000 tokens in 1,024 chunks, 160 chunks, and decode past the window | kept | same |
| X4 | Gemma-4-31B with 8-bit KV | KL 1.96 / top-1 flip on a repetition prompt (0.06 bf16): attention scale 1 on normed q/k amplifies int8 key error ~20× | bf16 KV default for Gemma-4 | same |
| X5 | Gemma-4 KV capacity | ~900 KB per token in bf16; 16 × 32k would need ~460 GB; 32k capped at ~5 agents (160 GiB pool, 0.2M tokens). Needs sliding-window KV | open | same, "Studio results" |
| X6 | Templated attention kernels | 27B regression (§12) | fixed | "Diagnosed 2026-11-01: templating…" |

Ultra results without drafters *(Source: "Studio results (inferd, no drafters…)")*:

| model | decode tok/s (0.5k / 32k) | prefill tok/s (4k / 32k) | 32k × 1 / × 8 / × 16 agg tok/s | × 16 per stream | × 16 TTFT |
|---|---|---|---|---|---|
| Qwen3.5-9B | 136 / 115 | 5,877 / 5,216 | 117 / 480 / 656 | 45 | 1.2 s |
| Qwen3.6-35B-A3B | 154 / 136 | 7,163 / 6,858 | 138 / 414 / 547 | 38 | 1.5 s |
| Muse Glimmer 30B | 44 / 40 | 1,852 / 1,874 | 39 / 155 / 229 | 16 | 4.2 s |
| Gemma-4-31B (bf16 KV) | 28 / 20 | 1,493 / 1,155 | 18 / 37 (× 4) / –; 8k × 16: 124 | 9 (8k) | 12 s (8k) |

## 9. Scoreboard milestones

`conc.py` follow-ups, Qwen3.8-27B 4-bit + DFlash2, Mac Studio M5 Ultra, aggregate tok/s.

**First full run with DFlash2** (2026-10-29 night; per stream (aggregate), TTFT median) *(Source: "Full scoreboard
with DFlash2")*:

| ctx | N = 1 | N = 4 | N = 8 | N = 16 | N = 32 |
|---|---:|---:|---:|---:|---:|
| 16k | 85.0 (85), 0.19 s | 61.4 (242), 0.85 s | 40.7 (318), 1.6 s | 22.3 (351), 2.6 s | 14.0 (448), 5.5 s |
| 32k | 83.6 (84), 0.35 s | 54.5 (202), 2.0 s | 34.2 (270), 4.1 s | 18.9 (299), 6.2 s | 10.9 (361), 8.5 s |
| 64k | 68.8 (69), 0.44 s | 38.6 (154), 2.4 s | 23.2 (181), 4.8 s | 12.7 (195), 7.5 s | 7.2 (229), 13.4 s |

**Head-to-head, morning of 2026-10-30** (total tok/s, worst TTFT) *(Source: "Head-to-head and the morning of
2026-10-30")*:

| ctx × N | inferd | Splash 1.5.0 | Kapitsa 0.2.24 | mlx-serve |
|---|---:|---:|---:|---:|
| 16k × 8 | 372 (1.0 s) | 369 (6.4 s) | 179 | 70 (43 s) |
| 16k × 16 | 467 (1.9 s) | 369 (17 s) | 73 (74 s) | — |
| 32k × 8 | 334 (1.8 s) | 390 (6.3 s) | — | — |
| 64k × 8 | 259 (2.3 s) | 311 (8.0 s) | 9 (396 s) | 11 |
| 64k × 32 | 324 (9.8 s) | 311 (49 s) | — | — |
| 1 stream, 8k decode | 118 | 219 | 214 | 224 |

**Milestone inferd10** (2026-10-31: 8-bit KV, reworked verify attention, self-tuning block; Splash 1.5.0 in brackets)
*(Source: "Milestone inferd10")*:

| ctx | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 16k | 147 (172) | 366 (354) | 485 (369) | 563 (369) | 599 |
| 32k | 150 (148) | 405 (321) | 465 (391) | 504 (354) | 549 |
| 64k | 139 (147) | 323 (292) | 363 (311) | 390 (300) | 430 (311) |
| 128k | 146 (149) | 241 (265) | 265 (268) | 292 (266) | — |
| 256k | 95.3 (132.1) | 189.5 (187.2) | 187.0 | — | — |

Splash serves ~4 agents at a time and queues the rest (at 128k it holds ~266 tok/s from N = 4 up with 70–84 tok/s per
stream) *(Source: "Splash at 128k runs ~4 lanes…")*. The release table in `engine/README.md` (0.1.0, after grouped
allocation) differs in some cells: 16k × 1 142, 16k × 32 606, 32k × 16 518, 64k × 1 146, 128k × 4 270, 128k × 8 291,
256k × 1 111.

**Mac minis, final fleet build** (16k turns/min, Splash in brackets): M5 Pro 4:4 14.32 (14.4), 8:8 13.92 (15.5), 8:4
13.35 (14.5), 16:16 14.23 (14.4), 16:4 12.99 (14.6); M6 4:4 8.7 (8.8), 8:8 7.7 (8.5), 8:4 7.59 (9.4), 16:16 7.97
(8.9), 16:4 7.73 (9.1) *(Source: "Mac minis, final build")*. Overnight 2026-11-01 (`agents.py` N:N): M5 Pro 16k 8–64
agents 13.51–14.26 vs Splash 11.90–12.01; M5 Pro 32k 8–32 agents 11.03–12.76 vs 10.33–10.58 *(Source: "Overnight
agent-fleet results")*. At the stock wired limit Splash does 7.3–8.1 turns/min at 32k and 8.5–9.4 at 16k on the M6
*(Source: "0.1.1: speculative sampling", correction)*.

## 10. Ideas recorded but not tried, and open questions

| item | status | Source |
|---|---|---|
| GDN state in bf16 (halves the ~18 ms of state traffic per B = 64 step; needs a quality check) | not tried | "Where a 27B decode step goes" |
| Sliding-window KV (ring blocks) for Gemma-4; conflicts with prefix-cache block sharing | not done | "Other models on inferd" |
| Reuse of the re-rendered previous reply's KV on follow-ups (Splash does) | not done | "Diagnosed 2026-10-30: Splash's single-stream lead…" |
| An int8 prefill tile that beats the bf16 ceiling | needs a different tile design | "Agent fleets (2026-10-30 evening)" |
| lane4 as the only weight layout (no copy): qmv4, W4A8 and prefill GEMM would all have to read it; TensorFold's two-tiles-per-simdgroup prompt kernel beats our MLX-qmm prefill | not done | "Lessons from the mlx.fast competition" |
| QMST: per-shape split law, fused gate/up; re-measure after the drafter-view fix | not done | "Splash's source", "QMST drafter bug: cause found" |
| ANE: overlap the next layer's re-quantization with the ANE, int8 intermediate for down, shared-event sync | open | "Neural Engine FFN split for prefill" |
| Qwen3.8-Flash-Next (the stated second target) | not started | header of the study log |
| ~12% prefill regression at long context (128k TTFT 115 s on 10-30 → 129 s) | unexplained | "2026-10-31 morning…" |
| 64k × 16 evening vs morning (decode 315 vs 352) | one sample each, not chased | "Results of the 2026-10-30 evening runs", "Diagnosed 2026-10-30: the "16-sequence…"" |
| Long-uptime slowdown: 128k × 4 256/249 tok/s on a fresh server vs 225 after a 21-cell sweep | remains | "Overnight agent-fleet results" |
| lane4 drops drafter acceptance (2.82 → 2.33) | cause not found | "Lessons from the mlx.fast competition" |
| `spec_test` shows ~1.01 tokens/cycle on short chat prompts with every build | tool limitation, unexplained | same |
| One agent on a long context (256k × 1 28% behind Splash; 16k × 1 15% behind) | remaining gap | "Milestone inferd10" |

## 11. What is ours

### Borrowed, with credit

| source | what we use | how |
|---|---|---|
| **MLX** (ml-explore/mlx v0.32.3, MIT) | NAX quantized matmul (`affine_qmm_t_nax`), steel bf16 GEMM and split-K GEMM, NAX flash attention (`attention_nax`, `attention_nax_dsplit`), gathered quantized matmul (`affine_gather_qmm_rhs_nax`), `nax.h` MMA fragments, quantized loaders | vendored in `third_party/mlx` with three local patches (flip constant, K-slice row lengths, sliding window); our `qmm_splitk` is derived from `qmm_t_nax_tgp_impl` |
| **mlx_lm / mlx_vlm** | logits references (`ref_logits.py`); the Qwen3.5 vision tower and `get_rope_index` in the front end | run in Python, outside the engine |
| **DFlash2** drafter (z-lab / incoai, `Qwen3.8-27B-DFlash2`) | the trained drafter model and its architecture (block drafting, candidate selector with codebooks) | our own Metal implementation (`dflash.h`, `spec.metal`) |
| **Splash** (incoai, github.com/incoai/splash, Apache-2.0) | ideas: the ANE FFN split design (`AneFfn`), SiLU via tanh on the ANE, the "staged tile" decode GEMM (our QMST, off), INT8 KV by default, sharing in-flight prefill, parking KV and GDN state on SSD, drafter acceptance at block 8 as the reference | our implementations; no Splash code is vendored |
| **TensorFold** mlx.fast challenge | the lane-register GEMM design (`tf_lane_reg`) behind the promoted entries; the drafter vocabulary cut | lane4 (off), `DRAFT_VOCAB` (opt-in) |
| **vLLM** | the serving model: paged KV, continuous batching, chunked prefill, prefix caching, ragged steps | reimplemented for a hybrid GDN model |
| **Hugging Face transformers** | tokenizer, chat template, image processor | front end |

### Our own work, and what it measured

| # | contribution | measured advantage | Source |
|---|---|---|---|
| 1 | **Native step encoding** without an ML framework: fused projections, one forward pass for decode/verify/prefill rows, steps committed in pieces | host overhead ~0.5 ms per step vs MLX's 8.6 ms; B = 64 at 2k 768 vs 591 tok/s (raw MLX) on day 1 | "Native engine, day 1" |
| 2 | **W4A8/W8A8 on the integer tensor path with in-place flipped weights** (one int8 scale per row per 64-group, affine epilogue, no weight copy; patched MLX qmm reads the same flipped weights) | linear layers M = 80 83.5 → 56.2 ms; 16k × 16 agents 317 → 455 tok/s | "W4A8: int8 activations…" |
| 3 | **Core-count-aware split-K** for narrow outputs | `mlp.down` 150 → 530 GB/s; 50 → 27 ms per step; M6 162 → 138.5 ms by not splitting | "Native engine, day 1", "Mac minis (2026-10-30)" |
| 4 | **attn_mpp**: verify attention on whole-tile `matmul2d` with K as the left operand, 128-key groups on physically contiguous pages, softmax lanes, key-major score stores, lazy rescale, int8 queries × int8 KV; 32-row variant for multi-agent blocks | 8 × 128k 8-row verify 153 → 109 ms; 1 × 128k 37.6 → 30.7 ms; 128k × 16 (5-row) 227 → 196 ms | "attn_mpp…", "2026-10-31: long-context verify attention reworked", "Agent fleets…" |
| 5 | **Chunk tiles**: follow-up turns attend over paged KV with key splits instead of gathering it | 16 agents × 32-token turns at 128k 2,866 → 844 ms per step; 32k × 16 server 381 → 472 tok/s | "Chunk tiles (mode 6)", "Smoke after chunk tiles…" |
| 6 | **GDN-aware prefix cache**: block trie with recurrent-state snapshots at a margin before the prompt end; verify tapes + replay of committed rows only, so speculation and caching coexist with a recurrent model | follow-ups restore 16,352 of ~16,470 tokens; worst TTFT 1.0 / 1.9 s at 16k × 8 / × 16 vs Splash 6.4 / 17 s | "Agent serving…", "Head-to-head…" |
| 7 | **Fan-out leader snapshots** (Splash has an equivalent) | 32k × 8 fan-out 188.8 → 24.6 s; 64k × 16 TTFT median 5.4 vs Splash's 27.5 s | "Agent fleets (2026-10-30 evening)" |
| 8 | **Content-addressed, persistent SSD tier** for KV blocks and GDN/drafter state (Splash also spills; this layout is ours) | restores in ~0.23–0.33 s vs 3–7.2 s re-prefill; M6 16:16 0.79 → 7.97 turns/min; survives restarts | "SSD tier…", "Persistent SSD tier", "Mac minis, final build" |
| 9 | **KV-budget admission with preemption to the cache** | M6 32k × 4 deadlock (0.02 tok/s) → 12/12 turns in 121 s | "v10: KV-budget admission + preemption" |
| 10 | **Grouped, contiguity-preserving KV allocation** | recovers the 10–22% that fragmentation cost; 128k × 4 241 → 270, 256k × 1 95 → 111 (release notes) | "v11…", "Diagnosed 2026-10-30: a full KV pool…" |
| 11 | **BlockTuner**: draft block chosen per regime from measured step time × acceptance; tile-aware block rule | 128k × 8 263 → 289 tok/s; tile rule 16k × 8 375 → 406 | "Server A/B…", "Head-to-head…" |
| 12 | **Speculative sampling on the GPU** for temperature > 0 (exact since 2026-11-04: top-k 1–64, or top-p with rows beyond 64 candidates sampled on the host) | M5 Pro 16.9 → 42.9 tok/s single stream; 28 → 110 at 4 streams | "0.1.1: speculative sampling" |
| 13 | **Own AppleNeuralEngine client** (MIL generator, weights as per-layer re-quantized IOSurface inputs, two weight sets, rotated W8A8, residency) — design after Splash | M6 prime +23% in the server; M6 8k fleet 9.67 → 10.99 turns/min vs Splash 8.73 | "Neural Engine FFN split…", "Overnight agent-fleet results" |
| 14 | **Content-keyed image token ids** so the prefix cache distinguishes images | same image hits 128 cached tokens; a changed image hits 0 | "Vision" |
| 15 | **Incremental tokenization cache** in the front end | 84 ms → 0.09 ms per turn at 128k | "Front end: a turn re-tokenized…" |
| 16 | **One model class for five families** (hybrid dense, MoE, sliding-window, Gemma-4 global/sliding) | Qwen3.6-35B-A3B 547 tok/s at 32k × 16; MoE expert sort 1,919 → 5,858 tok/s prefill | "Other models on inferd", "Studio results" |

## 12. Diagnosed problems

Each entry: symptom → evidence → cause → fix. Dated headings are the study log's own post-mortems.

**Split-K scratch too small** *(CLAUDE.md gotchas)*. Symptom: corrupted logits. Cause: the split-K scratch was sized
for a narrower projection than the split condition could catch. Fix: sized for the widest (`kSplitMaxN`).

**First speculative server slower than plain (2026-10-29)** *(Source: "Agent serving…")*. Symptom: N = 16 at 5.6 tok/s
per stream. Evidence: 5-row verify blocks took the flash path. Cause: flash re-gathers each sequence's whole KV every
step. Fix: multi-row `attn_gqa` for verify blocks.

**Drafter acceptance 1.1 tokens/cycle on 8-bit checkpoints (2026-10-29)** *(Source: "Weight precisions…")*. Symptom:
acceptance collapsed but output stayed lossless. Cause: the drafter shares the target's embedding and used the 4-bit
lookup on an 8-bit table. Fix: `embed4/8/16` everywhere.

**Drafter k|v view offset (2026-10-30)** *(Source: "Head-to-head…")*. Exposed by `DRAFT_BITS=4`: the offset view of the
fused qkv weight assumed 1 byte per weight. Fix: `K * bits / 8`.

**Diagnosed 2026-10-30: Splash's single-stream lead is drafter acceptance, not kernel speed.** Symptom: Splash 219 tok/s
vs our 118 on one stream with the same files. Evidence: Splash's log reports 5.0–5.8 tokens per cycle at block 8 and a
~40 ms cycle (longer than our 32 ms); ours gave 4.02 (block 5) / 4.08 (block 8). Cause: a gap in our DFlash2
implementation (found next). 

**Diagnosed 2026-10-30: block 8 was useless because the drafter attention computed only 5 rows.** Symptom: block 8 ≈
block 5. Evidence: a per-cycle commit histogram (`INFERD_DEBUG`) showed 73 of 120 cycles committing exactly 5 and none
7 or 8; draft positions 6–7 were always the same junk ids. Cause: `draft_attn` had `DMAXR = 5`. Fix: 5- and 8-row
instances (8 as 2 × 4 across threadgroups): 4.28 → 5.68 tokens/cycle.

**Diagnosed 2026-10-30: the "16-sequence non-GPU time" was spec_test's own prefill sizing.** Symptom: 16 × 6k ran at
276 tok/s with ~150 ms per cycle unaccounted for, looking like a server regression. Evidence: the host sat in
`waitUntilCompleted`; 81 ms per cycle between submission and GPU start (residency work). Cause: spec_test prefilled all
prompts in one step, sizing the GDN verify tapes for 96k tokens (97 buffers, 177 GiB), all bound every cycle. Fix:
prefill one prompt per step: 758 tok/s, 93 ms per cycle.

**Diagnosed 2026-10-30: the M5 Pro 16:16 cliff was snapshots dropped from inner nodes without a spill.** Symptom: 5.34
turns/min after other cells vs 13.67 fresh. Evidence: 192k new tokens vs 60k, cache hit 0.786. Cause: `evict()`
spilled only leaves; a primed context becomes an inner node once its next turn extends it. Fix: spill every evicted
snapshot. (Did not hold in the full fleet; see 2026-10-31.)

**Diagnosed 2026-10-30: a full KV pool handed out pages in descending order.** Symptom: 64k × 4/8 at 263/287 tok/s after
a 16 × 128k run vs 305/352 fresh. Evidence: `BENCH_SHUFFLE` costs 12–22% per verify step. Cause: with a full pool each
allocation evicted the tail of the oldest chain, so a new sequence got X, X−1, X−2… and attn_mpp never saw a
contiguous ascending group. Fix: evict 64 blocks at once.

**Front end re-tokenized the whole conversation (2026-10-30)** *(Source: "Front end: a turn re-tokenized…")*. Symptom:
TTFT growing with context. Evidence: 84 ms per encode at 126k tokens under the GIL, ~1.3 s for 16 agents. Fix:
per-message piece cache.

**Diagnosed 2026-10-31: the 16:16 cliff was admission evicting the snapshot it had just restored.** Symptom: full fleet
16:16 still 5.5 turns/min, 191k new tokens. Evidence (`SPILL_DEBUG`): "restored 511 blocks (510 already in memory)"
then "admit … cached 0". Cause: the restore took the last free slot, and admission's eviction to replace it chose the
restored snapshot. Fix: `evict` takes a node to spare; admission and `take_snapshots` spare theirs. 16:16 14.33
turns/min, 59k new tokens.

**Diagnosed 2026-10-31: 8-row verify attention at long context was softmax-bound, not bandwidth-bound.** Symptom: 8-bit
KV 8-row verify at 325 GB/s while 5-row bf16 ran at 1.06 TB/s. Evidence: ablations at 8 × 128k (156 ms): without the
softmax 69 ms, without P V 92, without QK 132; then the score scatter loop was the next 36 ms; Splash's page layout
alone gave only 164 → 153. Fix: the attention rework (softmax lanes, matmul-stored scores, int8 queries): 153 → 109
ms.

**Diagnosed 2026-10-31: the W4A8 epilogue is not what holds wide GEMMs back.** Evidence: removing it changed M =
16–128 by ≤ 6%; the `matmul2d` runs in K = 64 steps are the limit. Consequence: the per-channel weight copy idea was
dropped.

**Diagnosed 2026-10-31: QMST in the server breaks the drafter** and **QMST drafter bug: cause found.** Symptom: 8k
decode 189 → 111 tok/s with `QMST=1`; target tokens identical, drafts junk from the second cycle (special-token ids),
mean commit 5.2 → 2.7; `decode_test` showed QMST as a win and `spec_test` (1.01 tokens/cycle on its prompts) could not
see it. Cause: the drafter's k|v offset view moved `wq`, `s`, `b` but kept the QMST copy, so q rows were computed as
k/v. Fix: views drop the copy (`rows_of`); the same bug later broke lane4 acceptance until fixed (2.54 → 2.10).

**Server sized the KV pool with its own KV8 default (2026-10-31)** *(Source: "Diagnosed 2026-10-31: QMST…")*.
Symptom: with 8-bit KV the Ultra's pool held 2.3M of 4.5M possible tokens. Fix: one KV8 default for both.

**M6 32k fleet deadlock (2026-10-31)** *(Source: "v10: KV-budget admission + preemption")*. Symptom: 0.02 tok/s,
two-hour timeouts. Cause: admission counted sequences; four 33k agents grew until the pool was empty and none could
free a block. Fix: KV-budget admission plus preemption.

**Fragmentation over a long uptime (2026-10-31)** *(Source: "v11: grouped KV allocation")*. Symptom: the milestone lost
128k × 4 and 256k × 1 on a server up for hours. Evidence: shuffled pages +10% / +14%; a fresh server gave 117 vs 95
tok/s at 256k × 1. Fix: grouped allocation.

**Diagnosed 2026-11-01: the ANE split made server prefill slower until the weights stayed wired.** Symptom: +29% in
`decode_test`, but in the server the M6 prime went 31 → 42–44 s and the M5 Pro 40 → 70–83 s. Evidence: wired memory
swung 8–15 GiB (M6) / 17–25 GiB (M5 Pro); 73k pageouts on the M6. Cause: the split's three command buffers per layer
leave the GPU idle, and with the KV pool filling the budget macOS unwired GPU memory and re-wired it each layer.
Fix: a residency set over every buffer and ANE surface: M6 prime 25.4 s (+23%).

**`ANE_A8D` overflow (2026-11-01, open)** *(Source: "Diagnosed 2026-11-01: the ANE split…")*. Symptom: inf in layer 0,
zeros after. Evidence: the rotation and quantization pieces are right in isolation. Suspect: an int8 × int8 down matmul
that drops the 1/128 dequantization (×16384, fp16 overflow). Off.

**Diagnosed 2026-11-01: a server that hung on its first request was the auto KV pool filling the Studio.** Symptom: an
uninterruptible GPU wait on the first prompt with a small model. Evidence: auto KV gave 222.5 of 224 GiB; an explicit
96 GiB worked. Fix: use an explicit size on the Ultra.

**Diagnosed 2026-11-01: templating attn_gqa_row cost the 27B 15–23% at 16–64 decode rows.** Symptom: 8k R = 1 B = 16
35.4 → 40.6 ms, B = 64 91.3 → 111.8 ms (verify blocks and logits unchanged). Cause: the V accumulator as an indexed
`float4` array was not kept in registers. Fix: two named `float4`s for Dh 256 (35.4 / 91.2 ms). Lesson: bench B 16/64
after any shared-kernel change.

**Gemma-4 with 8-bit KV (2026-11-01)** *(Source: "Other models on inferd")*. Symptom: KL 1.96 and a top-1 flip. Cause:
attention scale 1 on normed q/k amplifies int8 key error ~20× relative to 1/√Dh models. Fix: bf16 KV default for
Gemma-4.

**A comparison corrected (2026-11-01)** *(Source: "0.1.1: speculative sampling")*. The "Splash ~8 turns/min" on the M6
32k fleet was Splash's raised-wired-limit arm; at the stock limit (the budget inferd runs under) it does 7.3–8.1 at 32k.
Both peak at ~26 GiB wired, but Splash averages 25.0 against inferd's 22.0: its elastic KV takes the whole budget,
inferd's was a hand-sized 3 GiB beside an 8-bit drafter.
