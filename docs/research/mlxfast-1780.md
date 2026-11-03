# mlx.fast: what changed from 1.669 to 1.7802 (and the queued 1.7957), and what inferd can use

2026-11-03. Sources: `tensorfold/challenge/mlx-subs-now3.json` (927 submissions; notes and official metrics), the
competition repo's `main` (cloned read-only into the session scratchpad, because the local checkout at
`tensorfold/challenge/mlx` stops at `cb8d1770` = 1.669; `cb8d1770..d56c850f`: 61 files, +5,294/-515, all under
`Vendor/TensorFold/zig`), and the queued entry `ad25a33a` (terrapinelf, 1.7957). Prototypes: `engine-mlxf/`,
patch `engine-mlxf.patch` against `engine-mlxf-base/`.

## The regime, and the numbers

Scored: batch 1, the 1,065-token longcopy prompt, DFlash depth 1 (**at most 2 tokens per round**), 128 decode
steps; composite = decode^0.75 x prefill^0.25 against a live TensorFold control. The seed (prefill + first token,
~0.85 s) sits inside the decode leg, so prefill carries ~48% of the log-score in practice (mrv777's accounting).

| entry | who | score | decode x | prefill x | window ms/token | prefill tok/s | seed s |
|---|---|---|---|---|---|---|---|
| 0ee904e8 | a-github-name | 1.6691 | 1.849 | 1.228 | 16.03 | 1225 | 0.886 |
| 6e5f5d34 | ercumentyildirim | 1.6823 | 1.866 | 1.233 | 15.82 | 1230 | 0.884 |
| 4311baf2 | e-man07 | 1.7001 | 1.891 | 1.235 | 15.54 | 1232 | 0.883 |
| faf6648e | ercumentyildirim | 1.7170 | 1.910 | 1.246 | 15.27 | 1246 | 0.868 |
| c1d09a0f | jungjipdo | 1.7317 | 1.927 | 1.257 | 15.23 | 1254 | 0.867 |
| ed4ee746 | mrv777 | 1.7468 | 1.941 | 1.273 | 15.37 | 1250 | 0.871 |
| 4d076487 | ercumentyildirim | 1.7637 | 1.965 | 1.274 | 14.94 | 1272 | 0.852 |
| **05a60bc5** | **kshitij-hash** | **1.7802** | 1.983 | 1.288 | 14.90 | 1274 | 0.851 |
| 22b390be (queued) | terrapinelf | 1.7957 | 2.012 | 1.277 | 14.56 | 1276 | 0.849 |

So 1.669 -> 1.780 is **-7% decode window per token and +4% prefill**, made of ~40 small exact pieces, each worth
0.1-1%. There is no new big kernel: `tf_lane_reg` (already studied, our `lane4`) is still the engine. A round is ~29 ms
for ~1.95 tokens, i.e. the leader's 2-row verify plus drafter now sits at the M5 Max's weight-read floor (16 GB at
the 540 GB/s this laptop measures = ~29.6 ms). Everything they gained since 1.669 is latency hidden under that floor.

## Findings: idea -> evidence -> effect -> inferd has it? -> applicability

1. **Counted completion instead of barriers ("counted chains").** Producer kernels (out projection, MLP down, GDN
   prepare) count their threadgroups after their stores (device-scope fence + atomic add); the consumer (the norm,
   the star scan, the gate) is encoded in the *same barrier interval* of a concurrent encoder and one thread per
   threadgroup spins on the count before reading. Evidence: `TF_COUNT` in `lane_projection.metal`, `q27_norm_xs`
   `wait_for` (6e5f5d34, "two barriers a layer gone"); fkiene's star scan in the prepare's interval (a670d7e3,
   in 292d7fda); the queued entry hoists every independent read *before* the wait. Effect: ~0.2-0.5% each.
   inferd: no (serial encoder; NOTES: a concurrent encoder with barriers is no cheaper). **Prototyped as SK_FUSE
   below (the same mechanism applied to our split-K reduce): exact, but slower on the laptop.**
2. **Launch shapes for latency-bound kernels.** Decoded GDN star 4 -> 8 -> 16 SIMD groups per threadgroup (fewer
   pollers on counted waits), KV append/pack/gather 4 -> 1 SIMD group per threadgroup (more threadgroups spread
   over cores). Evidence: 17281cd7 in 05a60bc5 (`gdn.zig forwardStar`, nine `attention.zig` io dispatches),
   fkiene 8de0c6b5 in the queued entry. Effect: +0.3% decode. inferd: our small kernels are mostly 256-thread
   `threads()` dispatches; the gains here are tied to their counted waits. Low priority.
3. **Multi-column GDN scans** (each lane keeps 2/4 value columns; key, query, decay, beta loaded once per token).
   Evidence: kieranscotney 631553ae (in c1d09a0f, prompt scan, prefill -0.6%), terrapinelf's star pair (465d24fb).
   inferd: **already written** as `gdn_step_r` for Flash-Next only. **Prototyped as GDN_R for the 27B:
   bit-identical, neutral on decode/verify, see prefill below.**
4. **Prompt row-block bands** (threadgroup grid remap so a weight tile serves 32 row blocks back to back; mrv777
   ed4ee746, prefill -2.1% on an Ultra). Taken back out two entries later (292d7fda: "grid-order prompt walk gave
   the best prefills on box 2-1"). inferd: no. **Prototyped as QMM_BAND on MLX's qmm: bit-identical, no gain**
   (our prefill GEMM is compute-bound: MLX qmm ~58 of the laptop's measured 64 bf16 TFLOPS; weights re-read per row
   block cost ~3 ms of ~9 ms per gate/up at 1k rows, and that read hides under the math).
5. **Staggered prompt segments on 2-3 command queues** (4d076487, 2f94db80 in 05a60bc5): a >=768-row prompt pass is
   split near half (thirds), the second segment's layer-l mixer waits on an MTLEvent the first signals after its own
   layer-l mixer; everything else of the two segments overlaps. Effect: prefill +0.2-1%. inferd: no (one queue).
   Applicable to the server's chunked prefill (a chunk and the next chunk of the same prompt are exactly this), but
   the gain is filling the GPU's idle edges, small when the GEMMs are at 90% of peak. Not prototyped.
6. **2-bit sieve for the greedy verify head** (kshitij-hash 63a5493e, kappa 10 -> 7 -> 6, an eight-row scan in the
   queued entry): stream a 2-bit copy of lm_head (~0.32 GB), list candidate tiles whose bound can reach the max,
   rescore only those with the real head. Effect: +1-1.5% decode in their regime. **Its exactness is statistical**
   (the note says so: "not a certified bound"); the queued entry adds a "proof gate" for the 8-row scan only.
   inferd: no; our verify rows also feed `row_top16` for speculative sampling, which needs real top-16 logits.
   lm_head is 1.7 ms of a 40 ms 16-row step on the laptop (decode_test gemm), so the ceiling is ~2-2.5%. Not
   recommended unless made certified.
7. **Weight warm-up into the SLC during latency-bound intervals** (jungjipdo 867c8431 at ercumentyildirim's 9 MiB,
   bridge warm ee250bee; queued entry): read-only line requests for the next out projection / the next verify's
   first layer, issued beside the attention prefix or the drafter bridge. Effect: ~0.2-0.4%. inferd: no; needs a
   concurrent encoder to overlap. Low value.
8. **Commit cadence and host work off the critical path**: verifies commit every 8 layers (was 16), the first after
   layer 0's mixer (42a68177), round host work moved after the next frame's encode (0fe40d21), polled/sleeping
   request waits. inferd: steps are already committed in pieces; spec_test shows 0.3-0.5 ms submit->GPU start.
9. **Drafter**: 2-bit drafter linears with q4 copies used while the sibling landing share is below 0.95
   (faf6648e), drafter frame fusions (sums-emitting norm/conv/swiglu, fused residual conv + norm, per-head norm fused
   with rope; ArthurOstapenko c618f399), prompt catch-up on q4 copies (c1d09a0f), selector edges 16 candidates at a
   time. inferd's drafter is 4-bit, block 16, ~6-7 ms per cycle on the laptop (`SPEC_PROF`): a 2-bit drafter with an
   acceptance-driven q4 fallback is the transferable idea here (drafter ~15% of our cycle), not prototyped.
10. **Dead-row pruning in the last prompt layer** (fae7fff, live_from), **seed pick carried into round 1**, **one
    keep map a round**: regime-specific bookkeeping; inferd's prefill already computes the head for the last row only.

## What inferd already has

From the 1.669 study: lane4 (`LANE`, off), the drafter vocabulary cut (`DRAFT_VOCAB`, opt-in). Analogues: last-row
head in prefill, steps committed in pieces, `gdn_step_r` (Flash-Next only).

## Where inferd's single-stream time actually goes (laptop M5 Max, AC, 4k context, this build)

- Plain decode step 31.3 ms; 8-row verify 37.5-38 ms; 16-row verify 40.3-41 ms (decode_test bench, interleaved runs).
- All linears per step (decode_test gemm): 16 rows 35.8 ms at 447 GB/s vs the measured 540 GB/s read peak (floor
  ~29.6 ms). The non-linear remainder at 16 rows is ~5 ms; the drafter ~6-7 ms per cycle.
- **lane4 at 16 rows is 4.3 ms (11%) faster than W4A8 here** (40.4 -> 36.1 ms per step, two interleaved pairs), the
  single largest lever on the laptop; it stays off because it loses on the Ultra and spec_test acceptance fell.
  Measured below with `LANE_DRAFT=0` (drafter linears kept on W4A8) to separate the two.
- The dispatch count is not the bottleneck it was taken to be: SK_FUSE removed 144 dispatches per 16-row step
  (931 -> 787) and the step got slower, not faster (see results).

## Prototype results (laptop M5 Max, AC; `engine-mlxf`, every knob default off)

Method: interleaved A/B runs with a GPU-idle guard before each (no other engine process, GPU utilisation < 10%);
the laptop drifted thermally through the session (the same 16-row step read 40 ms early and ~48-51 ms inside
spec_test late), so only same-period pairs are compared.

**Gates (all three knobs).** Knob off: logits byte-identical to `engine-mlxf-base` (decode_test check, chunks 1/4/24/64
on the 4 x 64 toks.txt and 1,065 on the challenge prompt). Knob on: byte-identical as well (the changes are exact), so
the cos >= 0.9999 gate is met trivially. spec_test on the challenge prompt: with each knob on, the same output as the
base (diverges from plain decode at token 33, plain 10839 vs spec 11330, exactly like the unmodified tree: a known
W4A8 near-tie) and the same 2.89 tokens/cycle.

| knob | what | before -> after | verdict |
|---|---|---|---|
| `SK_FUSE=1` | W4A8 split-K: the last of the S threadgroups to arrive (device fence + atomic count) reduces in split order, bit-identical to `splitk_reduce`; modes 4/5 in `w4a8.metal` | dispatches per 16-row step 931 -> 787; step (GPU ms, 3 pairs) R=16 40.22/40.28/41.19 -> 40.99/41.17/41.24, R=8 37.78/37.62/37.67 -> 38.14/38.31/37.91; fleet R=5 at 2k B=8/16/32 102.9/155.2/271.8 and 101.0/153.9/268.3 -> 107.1/154.9/273.9 and 103.9/151.4/263.0 | **off** (+0.5-0.9 ms at 16 rows: the serial tail reduce costs more than the dispatch it saves) |
| `QMM_BAND=<RB>` | MLX qmm grid remapped into row-block bands (function constant 41) | 1,065-token prefill 963 -> 961/959/962 tok/s (RB 4/8/16); 8k at 512-row chunks 858 -> 849/827 (drifting) | **off** (prefill GEMM is compute-bound, ~58 of 64 bf16 TFLOPS) |
| `GDN_R=2/4` | 27B GDN recurrence on `gdn_step_r` (R value columns per simdgroup) | R=16 step 40.0-41.3 vs 40.4-41.3 base, R=8 37.6-38.4 vs 37.4-38.5, R=1 31.2-31.7 vs 31.3-31.6; prefill 1,065 paired 962/972 vs 963 | **off** (neutral; gdn_step is ~4% of a verify step and latency-bound on the sequential scan) |
| `LANE_DRAFT=0` (study knob) | lane4 for the target only, drafter linears on W4A8 | see below | diagnostic only |

**lane4's "acceptance drop" is not a drafter bug.** spec_test, 256 tokens, 2 reps:

| prompt | LANE=0 | LANE=1 | LANE=1, LANE_DRAFT=0 |
|---|---|---|---|
| challenge (1,065) | 2.89 tok/cycle, target 56.1 / 51.6 ms | 2.51, 52.3 / 50.1 ms | 2.51, 51.3 / 49.3 ms |
| p2k chat (1,982) | 7.19, 48.6 / 47.4 ms, identical to plain | 7.19, 49.4 / 47.3 ms, identical | 7.19, 49.3 / 47.4 ms, identical |

Keeping the drafter off lane4 changes nothing, and where the target's text is the same (p2k: all three arms produce
plain decode's 256 tokens) acceptance is identical to the digit. On the challenge prompt LANE=1 changes the target's
own continuation (it matches plain decode to token 74 instead of 33; bf16 activations vs W4A8's int8), and the lower
tokens/cycle is that different text, not the drafter. The 2.82 -> 2.33 on the Studio was very likely the same
effect. What remains open is speed: decode_test bench gives lane4 -4.3 ms at 16 rows (40.4 -> 36.1, fresh machine),
but inside spec_test's cycle on the now-throttled laptop the target step is level (p2k 47.3 vs 47.4 ms); and the
Ultra measured lane4 slower. Re-measure lane4 end-to-end on a cool laptop before any default change.

## Recommendation

- None of the three prototypes earns a default: SK_FUSE is a measured loss (drop it), QMM_BAND and GDN_R are exact
  but neutral (keep only as notes; GDN_R could be enabled for multi-sequence prefill if a fleet test ever shows
  gdn_step as the limit).
- The leader's 1.669 -> 1.780 gain is latency engineering underneath a 2-row, bandwidth-floor round (counted waits
  in a concurrent encoder, launch shapes, SLC warm-up, staggered prompt queues). inferd's single-stream gap is
  elsewhere: a 16-row verify at 447 GB/s vs 540 peak (lane4 territory, now unblocked on the acceptance question),
  ~5 ms of non-linear work per verify, and a 4-bit drafter at ~15% of the cycle (their 2-bit drafter with an
  acceptance-driven q4 fallback is the most transferable remaining idea).
- The certified-or-not head sieve is the one sizeable decode piece (+1-1.5% for them, ceiling ~2.5% for us); worth
  it only with a certified bound and a fallback that still produces top-16 for sampled rows.

## Follow-up 1: 2-bit drafter with acceptance-gated 4-bit fallback (`DRAFT_Q2`, 2026-11-03)

Implementation (`engine-mlxf`, default off): `DRAFT_Q2=1` loads a second copy of the drafter's six layer linears
(kp_attn, kp_mlp, qkv, o, gate_up, down) quantized to 3 levels per 64-group (2-bit values) and keeps the 4-bit copy
resident; `DRAFT_Q2_GATE=<t>` drafts on the 2-bit copy while a rolling share (EMA, 1/8 per cycle, of accepted drafts
/ drafted rows) is >= t, else on the 4-bit copy (t=0: always 2-bit). Context appends always use 4-bit. The 2-bit
values sit in 4-bit containers (no 2-bit 16-row kernel exists), so this measures **acceptance**, not speed; the
speed bound comes from `DRAFT_BITS=8` vs 4: +0.8 GB of drafter weights cost +2.0-2.8 ms of drafter block per cycle,
so a real 2-bit kernel would save at most ~1.0-1.4 ms of a ~48-58 ms laptop cycle (2-3%).

spec_test, 256 tokens, laptop (2 reps for 4-bit / always-2-bit / 8-bit, 1 for the gates):

| prompt | DRAFT_BITS=4 | always 2-bit | gate 0.5 | gate 0.3 | DRAFT_BITS=8 |
|---|---|---|---|---|---|
| challenge: tokens/cycle | 2.89 | 2.51 | 2.89 (6 of 93 cycles 2-bit) | 2.83 (12 of 94) | 2.85 |
| challenge: tok/s | 58.8 / 47.5 | 46.1 / 40.7 | 47.7 | 48.0 | 49.2 / 45.0 |
| challenge: drafter block ms | 6.5 / 8.5 | 7.0 / 8.6 | 8.4 | 8.2 | 9.3 / 10.4 |
| p2k: tokens/cycle | 7.19 | 6.00 | 6.45 (18 of 40) | 6.00 (35 of 43) | 7.19 |
| p2k: tok/s | 130.8 / 121.0 | 103.2 / 100.0 | 112.3 | 102.8 | 118.9 / 119.0 |
| p2k: drafter block ms | 7.6 / 8.4 | 8.2 / 8.6 | 8.1 | 8.3 | 10.1 / 10.1 |
| p2k: vs plain decode | identical | diverges at 131 | diverges at 131 | diverges at 131 | identical |

Fleet (spec_test, 8 / 16 sequences of 1,024-token windows of p2k, 128 tokens): B=8 382.4 -> 328.8 tok/s (2-bit),
329.1 (gate 0.5, the share stays ~0.7 so it never falls back); B=16 409.2 -> 342.5 / 333.6. Losslessness count vs
plain unchanged (6/8, 13/16, the usual W4A8 near-ties).

Exactness: verification is untouched, so committed tokens are target argmaxes; but the different acceptance pattern
moves block boundaries, and on p2k that exposes a near-tie at token 131 that the 4-bit run never hit (verify vs plain
differ only by W4A8 near-ties, documented in the study log). That fails the strict "spec equals plain" gate on p2k.

**Verdict: no.** Always-2-bit loses 13-17% of accepted tokens (challenge 2.89 -> 2.51, p2k 7.19 -> 6.00, fleet
-14 to -16% tok/s) against at most 2-3% of cycle time a real 2-bit kernel could save; the gate only recovers what it
gives back to 4-bit. The leader's regime differs: a 2-row round only needs the top sibling right, and their gate
watches exactly that. For a 16-row DFlash2 block the drafter's precision deep in the block matters. Nothing staged
against `engine/`.

## Follow-up 2: lane4 end to end on a cool machine (Mac mini M5 Pro, 2026-11-03 02:13-02:57)

`engine-mlxf` built on the laptop and rsynced (incl. build/) to a Mac mini M5 Pro; models
`~/models/Qwen3.8-27B-MLX-4bit` + DFlash2. The script (`~/inference/mlxf/mini_lane.sh`) waited for the demos agent's
inferd to exit, then before every run for no inferd/decode_test/spec_test and GPU < 10% for 3 samples plus 20 s
cooldown (in practice >= 3 min idle between runs). spec_test 256 tokens with SPEC_PROF (one extra sync per cycle, both
arms), LANE=0 and 1 alternated, 3 rounds. Results were stable to 0.1-0.4 ms across rounds:

| prompt | arm | spec tok/s (r1/r2/r3) | tokens/cycle | target step ms | drafter block ms | vs plain |
|---|---|---|---|---|---|---|
| p2k | LANE=0 | 71.3 / 74.5 / 74.3 | 7.00 | 78.7 / 78.8 / 79.0 | 13.1 / 12.9 / 12.9 | identical over 256 |
| p2k | LANE=1 | 82.8 / 87.2 / 87.1 | 7.19 | 66.8 / 67.1 / 67.3 | 11.5 / 11.2 / 11.3 | identical over 256 |
| challenge | LANE=0 | 34.8 / 34.8 / 34.8 | 3.19 | 77.8 / 77.8 / 77.8 | 11.7 / 11.7 / 11.8 | diverges at 34 |
| challenge | LANE=1 | 31.8 / 31.7 / 31.8 | 2.50 | 65.9 / 66.1 / 65.9 | 10.1 / 10.1 / 10.1 | diverges at 74 |

Plain decode (1 row, never on lane4) is 17.1 tok/s in every arm.

- **Cycle time: lane4 is a clear win on the M5 Pro**: target step -15% (78.8 -> 67.0 ms), drafter block -13%, on
  both prompts, every round (consistent with the fresh-laptop decode_test bench, -11% at 16 rows; the Ultra is the
  exception, where W4A8 already runs 3-16 rows at ~815 GB/s).
- **Same text: lane4 is +17% tok/s** (p2k 73.4 -> 85.7 average, both arms identical to plain decode; acceptance even
  7.19 vs 7.00).
- **Challenge prompt: -9% tok/s**, entirely from acceptance 3.19 -> 2.50 on a *different continuation*: lane4's target
  matches plain decode to token 74, W4A8's only to 34, so the two arms are drafting different text. LANE_DRAFT=0 on
  the laptop showed the drafter's precision does not move acceptance; the token stream does. Per committed token
  lane4 is 30.8 ms vs W4A8's 28.3 ms only because its (more faithful) text is harder to draft.

**Recommendation for lane4:** turn it on for non-Ultra Apple10 GPUs with memory to spare (a device rule such as
cores < 60 and RAM >= 64 GB; it stays off on the Ultra), after two checks this study did not make: (a) the copy
doubles weight memory (~15 GB more), which comes straight out of the KV pool on a 64 GB mini and rules out 32 GB
machines, so the honest fix is to make the lane layout the only layout (study log: qmv4, W4A8 > 16 rows and the prefill
GEMM must read it); (b) a server fleet check (conc.py follow-ups at 4/8 agents on the mini), because small fleets
(<= 3 sequences x 5 rows) also route through lane4. The "acceptance drop" should be retired from the study log as a blocker.

## Scoping: lane4 as the only 4-bit weight layout (design note, no implementation)

Goal: every reader of a 4-bit `QLinear` reads lane4's layout, so the ~15 GB second copy (`repack_lane`) disappears.
Layout (unchanged from `kernels/lane.metal`): `LP[tile = n/32][g = k/64][col 0..31][32 bytes]` signed int4 (flipped,
q - 8), one 1 KB block per (32-column tile, 64-group); `SB[g][N]` (s, b) bf16 pairs group-major. 8-bit weights,
bf16 weights and MoE experts stay in MLX's layout (they are not where the copy hurts, and the MoE kernels index
experts by row).

### Readers and what changes (from `grep .wq` in `engine/src`, plus kernels)

| reader | today | change | effort |
|---|---|---|---|
| **lane4** (3-16 rows) | lane layout | none | - |
| **qmv4 / qmvt4** (M <= 2, decode) | row-major ushorts, lanes along K | replace by lane4 at M = 1-2 (decode_test gemm already measured lane4 29.8/30.3 ms vs qmv4 30.5/39.4 on the M5 Max), or a lane-layout GEMV: a simdgroup per 8 columns reads 8 x 32 B per group contiguously. Not byte-identical (summation order) | 1 day incl. Ultra/M6 timing |
| **W4A8** (17 <= M < 256 after lane4 takes <= 16; also M <= 16 on the Ultra) | `tensor<int4b>(Wq, DT2(K, N))` sliced at (g*64, n0), TN 128, `S/B[n*KG+g]` | right operand per (tile, g) = the 1 KB block as `DT2(64, 32)` (exactly lane4's `BT`), TN 128 = 4 ops of 32 columns per group, epilogue reads `SB[g][n]`. Same integer products and the same per-group FMA order per element, so **byte-identical** outputs are achievable. Flip constant / xr no longer needed (layout is always signed) | 1.5-2 days; the risk is the Ultra, where the 128-wide op runs 3-16 rows at ~815 GB/s and four 32-wide ops may not |
| **MLX qmm** (M >= 256 prefill, NAX) and **qmm_splitk** | `QuantizedBlockLoader` reads BN rows x BK from row-major, dequantizes into threadgroup memory; function constant 40 flips back | a lane loader: BK = 64 is one group, BN = 64 is two 1 KB blocks (contiguous per tile); dequantize with `SB[g][n]`, no flip. Values written to threadgroup memory are identical, so **byte-identical**. K-slice views (`kslice_of`, the ANE split) become a group offset inside each tile with the full KG as the tile stride (replaces patch `ldk`) | 1.5 days (vendored-header patch, both kernels) |
| **mlx_qmm_sg** (no neural accelerators: M3/M4) and **qmv_wide** | MLX layout | same loader change, or keep these machines on the MLX layout (they never had lane4: lane4 is matmul2d on NAX) | keep MLX layout: 0 |
| **rows_of** views (drafter k/v view, `DRAFT_VOCAB`, ANE row views) | byte offsets | tile-aligned offsets already supported (`lo % 32 == 0`); non-aligned views must be rejected at load (today they silently drop the copy) | 0.5 day |
| **drafter** linears (our 4-bit quantizer) | MLX layout, flipped | write the lane layout directly in `load_qlinear_bf16_q8` | 0.5 day |
| **ANE** requantizers (`ane_req_rot`, `ane_rowscale`, `ane_quant` on gate_up/down) | read `wq/s/b` row-major | index the lane layout (dequant one value at (n, k) = block (n/32, k/64), byte (n%32)*32 + (k%64)/2) | 0.5 day + ANE gate |
| **PF8** `pf8_wrot` (int8 rotated prefill copy, +22.7 GiB, from engine-pf) | dequantizes `wq` | same lane dequant at load; PF8 itself is unaffected (it is its own copy) | 0.5 day |
| **embed4** | row gather from the embedding table | untouched for untied models (embedding is not a GEMM operand; keep MLX layout). Tied embeddings would need a lane gather (32 B per (col, group)) | 0 / 0.5 day |
| **QMST**, `copydraft`, `mtp.h`, tools (`gemm_dbg`, `weights_test`) | MLX layout | QMST (off, experimental) can be dropped; the others reload or get a lane dequant helper | 0.5 day |

Load path: weights are `read_into` a `g.alloc` buffer today (not mmapped), so repacking per tensor through one
staging buffer (largest tensor: lm_head 0.66 GB) keeps the peak at model + one tensor, and the copy is gone.

### Estimate, risks, prefill

- **Effort: ~6-8 working days** (about 5 implementing + 2-3 for gates: logits per family at chunks 1/24/1536,
  spec vs plain, fleet conc.py on mini/Studio/laptop, ANE and PF8 gates). A staged path: (1) loader + W4A8 + qmm on
  lane layout behind `LANE_ONLY=1` with byte-identity gates (these are exact), (2) the M <= 2 path, which is the only
  numerics change, (3) delete `repack_lane`.
- **Risks:** the Ultra's W4A8 at 3-16 rows (if 32-wide ops lose bandwidth there, the Ultra pays for the minis'
  memory); qmv replacement at M = 1 (spec_test once showed plain decode slower with lane4 at 1-2 rows: `LANE_MIN`=2
  exists for that reason; must be re-timed on the Ultra and the M6); every tool/kernel that indexes `wq` directly is a
  silent-garbage risk (the same class as the flipped-weights bugs in CLAUDE.md), so the conversion needs a startup
  self-check (dequantize a few rows both ways at load in debug builds) and a hard error for unconverted readers.
- **Large-M prefill should not lose speed:** MLX qmm at M >= 256 runs at ~90% of the bf16 peak (compute-bound; the
  QMM_BAND study showed weight-read order does not matter), and the lane loader reads the same bytes in larger
  contiguous runs. PF8 (the faster prefill path on 27B) is unaffected after load. The one place to watch is K-slice
  qmm under the ANE split (more, shorter tile reads), which needs its own timing.

## Follow-up 3: server fleet check, LANE=0 vs 1 (Mac mini M5 Pro, 2026-11-03 02:59-05:05)

`engine-mlxf` (knobs default off), `run.sh 8371 16 1024 0 16 32768`, mlx-community Qwen3.8-27B-4bit + incoai
DFlash2, no SSD tier; `conc.py --mode followup --ctx 4k,16k --n 1,4,8,16` (512 tokens, thinking off); arms
alternated LANE=0, 1, 0, 1, each on a fresh server after an idle GPU and 60 s cooldown (logs and jsonl in
the M5 Pro's run directory).

**Memory:** LANE=0 auto-sizes the KV pool to 27.8 GiB (28,056 blocks, ~0.9M tokens); LANE=1 to 13.5 GiB (13,643
blocks, ~0.44M tokens): model + state + scratch 19.4 -> 32.8 GiB, drafter 3.1 -> 4.0 GiB (13.4 + 0.9 GiB of lane
copies). **16 x 16k still fits** (~262k tokens of prompts plus reserves): 16/16 ok in both rounds, no errors, no
preemption logged. Headroom for cached contexts beyond the live set halves, which matters for agent workloads
that revisit many contexts without the SSD tier.

Aggregate tok/s (round 1 / round 2, mean) and TTFT p50 / max s:

| ctx x N | LANE=0 tok/s | LANE=1 tok/s | delta | LANE=0 TTFT r1; r2 | LANE=1 TTFT r1; r2 |
|---|---|---|---|---|---|
| 4k x 1 | 52.1 / 55.6 (53.9) | 62.0 / 59.8 (60.9) | **+13%** | 0.40/0.40; 0.39/0.39 | 0.40/0.40; 0.40/0.40 |
| 4k x 4 | 131.1 / 123.3 (127.2) | 127.9 / 121.9 (124.9) | -2% | 0.82/1.15; 1.56/1.56 | 1.57/1.57; 1.22/1.42 |
| 4k x 8 | 166.5 / 161.4 (164.0) | 164.4 / 159.9 (162.2) | -1% | 2.26/2.26; 2.44/2.44 | 1.57/1.86; 2.68/2.68 |
| 4k x 16 | 187.3 / 182.9 (185.1) | 185.5 / 181.1 (183.3) | -1% | 4.04/4.51; 4.51/5.23 | 4.12/4.67; 4.71/5.25 |
| 16k x 1 | 50.3 / 52.6 (51.5) | 55.4 / 52.8 (54.1) | **+5%** | 0.40/0.40; 0.40/0.40 | 0.40/0.40; 0.30/0.30 |
| 16k x 4 | 109.3 / 116.8 (113.1) | 109.0 / 105.9 (107.5) | -5% | 1.25/1.25; 1.50/1.50 | 1.04/1.04; 1.30/1.30 |
| 16k x 8 | 148.5 / 150.8 (149.7) | 148.9 / 147.3 (148.1) | -1% | 2.54/2.54; 2.42/2.42 | 2.32/2.32; 2.60/2.60 |
| 16k x 16 | 169.8 / 167.9 (168.9) | 166.4 / 167.3 (166.9) | -1% | 4.45/4.87; 4.71/5.23 | 4.76/5.28; 4.29/4.29 |

Reading: one agent gains (+13% at 4k, +5% at 16k, where attention is a larger share of the step); fleets are level
within the round-to-round spread (the 4/8/16-agent cells mostly verify more than 16 rows per step, which stays on
W4A8 either way; 16k x 4's -5% is inside its own r1/r2 spread of 7%); TTFT is unchanged. **Recommendation:** LANE=1
is safe to enable on the M5 Pro for the server shape tested (single-user gain, no fleet cost, 16 x 16k fits), at the
price of half the KV pool; that cost is what "lane4 as the only layout" (above) removes, and it is the reason not
to enable it on 32 GB machines or where long-lived context caches matter more than single-stream speed.
