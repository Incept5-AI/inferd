# v0.2.0 prebuilt tarball on a Mac mini M6 (32 GB), 3–4 November 2026

The release tarball (`inferd-0.2.0-macos27-arm64.tar.gz`, sha256 verified) unpacked over this tree's `engine/`, with
Qwen3.8-27B 4-bit (mlx-community) + DFlash2 (incoai).

- `gate-part1.*`: `tools/release_gate.sh`. Host and front-end tests pass; logits at chunks 1 and 24 match mlx_lm
  (top-1 ok on every sequence). The bench step's 64 sequences x 2048 tokens does not fit a 32 GB GPU (out of
  memory), so it was re-run at 1 and 16 sequences in part 2.
- `gate-part2.txt`: logits at 256- and 1,536-row chunks on 1,536-token prompts (top-1 ok, cos 0.9996–0.9999,
  KL ≤ 0.0023); bench 104.7 / 186.7 ms per step at B = 1 / 16, L = 2048; speculative vs plain decode: 163.5 vs
  26.7 tok/s over 4 sequences (11.95 tokens per cycle). Two of the four very repetitive test prompts diverge after
  69 and 79 identical tokens: a near-tie flipped by the different kernel path in verify (documented in
  docs/EXPERIMENTS.md; the engine is not batch-invariant at the bit level, like vLLM).
