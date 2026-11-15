# MacBook Pro M3 Max, 15 November 2026: decode without neural accelerators

Raw rows behind [docs/RESULTS.md](../../docs/RESULTS.md) section (af). MacBook Pro M3 Max (30-core GPU, 96 GB, macOS
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

Not included: the `decode_test bench` and `prefill` output behind the step-time and prefill percentages in part 1
(not kept as files), and the token dumps behind the lossless checks.
