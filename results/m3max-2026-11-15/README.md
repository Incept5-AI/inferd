# MacBook Pro M3 Max, 15 November 2026: decode without neural accelerators, and the Neural Engine split

Raw rows behind [docs/RESULTS.md](../../docs/RESULTS.md) sections (af) and (ag). MacBook Pro M3 Max (30-core GPU, 96 GB, macOS
27), Qwen3.8-27B 4-bit + DFlash2, branch `perf/decode-throughput` on top of 0.4.2.

The server files are JSON lines, one row per round: `n` streams start together on 512-token prompts that are already
in the prefix cache (`cached`: 480 of 512 tokens) and generate 256 tokens each, greedy. `sat_tps` is the tokens emitted
while all `n` streams were decoding, over that window; `agg_tps` is all tokens over the wall time.

| File | What |
|---|---|
| `server-tiles.jsonl` | (af) part 1: `base` = 0.4.2, `tiles` = 8-row tiles (`SG_TILES`); 1–16 streams, 3 rounds |
| `server-policy-fill.jsonl` | (af) part 2 on the 8-row-tile build: `old` = `SG_POLICY=0 SPEC_FILL=0`, `pol` = `SPEC_FILL=0`, `fill` = defaults; `-p1` and `-p2` are the two alternating passes of 2 rounds. One stream, 3 rounds: `chain1` = `TREE=0`, `tree16` = `TREE=16`, `tree8` = defaults |
| `server-one-stream.jsonl` | one stream, block-8 chain (`TREE=0`) against the 8-node tree, two alternating passes of 5 rounds |
| `spec-grid-0.4.2.tsv`, `spec-grid-tiles.tsv` | `spec_test`, streams × block on 2,048-token prompts: tok/s, tokens and GPU ms per cycle, the commit histogram, and the comparison with plain decode |
| `gemm-0.4.2.txt`, `gemm-tiles.txt` | `decode_test gemm`: the time of each linear shape by row count |
| `0.4.2-smoke.txt` | 0.4.2 on this machine: logits checks, `decode_test bench`, `spec_test` |
| `prof.txt` | (af) part 3: `PROF_KERNELS=1`, kernel ms per cycle at 1 × 8, 8 × 4 and 16 × 4 |
| `skip.txt` | (af) part 3: whole cycles, default against `SKIP=replay` and `STATE_BF16=1`, two runs of each in turn |
| `ane-probe.txt` | (ag) part 1: `tools/ane_probe`, an FFN slice of 20 units and of 1 unit at 64–2,048 rows; `A8=0` is W8A16, `A8=1` int8 activations |
| `ane-prefill.txt`, `ane-prefill-fine.txt` | (ag) part 2: `decode_test prefill`, 4,096 tokens, by `ANE_UNITS` (0 and 16–32 in steps of 4; then 18, 19, 21, 22), two passes, with the per-layer timings of `ANE_PROF` |
| `ane-cold.jsonl` | (ag) parts 3 and 4: cold turns through the server, one row per run. `n` streams start together on 2,048-token prompts that are not in the cache; `first_s` is the last stream's first token, `wall_s` the end of the last stream. `base*` = GPU alone, `ane20` and `pre*` = 20 units before the fix, `ane20fix` and `fix*` = with it, `*t8` = `TREE_SEQS=8`; `def8` and `off8` = the build with the 20-unit default, with nothing set and with `ANE_UNITS=0`; `re8` and `re8b` = nothing set again, on the two rebuilds that followed (the second is the committed source) |
| `ane-selfcheck.txt`, `ane-tree-stress.txt` | (ag) part 3: every generated token of the 8-stream runs against mlx_lm, teacher-forced (per stream: tokens that are not the reference top-1, the largest gap, the tokens more than 0.5 logits below it). The second file is the `TREE_SEQS=8` runs, with their rows and server lines |
| `ane-fix-smoke.txt`, `ane-fix-bench.txt` | (ag) part 3: the build with the fix against the one before it: logits at chunks 24 / 256 / 1,536 (byte comparison and `ref_logits.py compare-bin`), and `decode_test bench` in two alternating passes |
| `ane-default-verify.txt`, `ane-default-warm.jsonl` | (ag) part 4: the build with the 20-unit default against `ANE_UNITS=0`: start time and memory, 8 cold streams, logits, `decode_test bench` and `prefill`, warm decode at 1 / 5 / 7 streams (rows as in the server files above), and the whole-stream check of both cold runs |
| `ane-default-ab.jsonl` | (ag) part 4: warm decode with the split loaded against `ANE_UNITS=0` on the same build, 1 / 2 / 4 streams, two alternating passes of 3 rounds (`off-p*`, `def-p*`; rows as in the server files above) |
| `ane-default-recheck.txt` | (ag) part 4: the rounds of that comparison, with `decode_test bench` straight after it and a minute later; logits at chunk 256 under the default; the two rebuilds checked again (logits, bench, 8 cold streams, the whole-stream check); the automatic KV pool with and without the split; `spec_test` against plain decode and the near-tie check of its differences |

Not included: the `decode_test bench` and `prefill` output behind the step-time and prefill percentages in part 1
(not kept as files), and the token dumps behind the lossless checks.
