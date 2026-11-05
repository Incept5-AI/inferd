# CLAUDE.md

A native inference engine for Apple Silicon aimed at many concurrent **agents** on one Mac. Started 2026-10-29.
Primary model: Qwen3.8-27B (hybrid: 48 Gated DeltaNet + 16 gated attention layers); Qwen3.8-Flash-Next and four
smaller ports also run. Read `docs/ARCHITECTURE.md` (how it works), `docs/EXPERIMENTS.md` (what was tried, with
numbers) and `docs/RESULTS.md` (every measurement) before changing anything.

## Layout

- `micro/`: microbenchmarks (bandwidth, MPP `matmul2d` fp16/int4 GEMM, MLX baselines, per-layer MLX profiles).
- `engine/`: the engine. C++/Objective-C++ host plus Metal Shading Language, no MLX runtime.
  - `kernels/core.metal`: our kernels (qmv4, embed, norms, GDN conv/prep/step/out-norm, attention prep,
    `attn_gqa` + `attn_combine`, activations).
  - `kernels/mlx_qmm.metal`, `kernels/qmm_splitk.metal`: MLX's NAX quantized matmul (vendored unmodified in
    `third_party/mlx`, MIT), plus our split-K derivative for narrow outputs. `kernels/mlx_gemm*.metal`: MLX's NAX bf16
    GEMM and split-K GEMM for bf16 checkpoints. `kernels/mlx_attn.metal`: MLX's flash attention (prefill chunks).
  - `kernels/attn_paged.metal`: our paged GQA-packed NAX attention for speculative verify blocks (≤ 5 rows; more rows
    via `NAX_ROWS` work but are slower than gather + flash).
  - `kernels/w4a8.metal`: **W4A8/W8A8**, int8 activations (per row per 64-group) × the checkpoint's 4-/8-bit weights
    on the integer tensor path (223 TOPS vs 124 bf16). Weights are flipped in place at load (q ^ 8 / q ^ 0x80).
  - `kernels/spec.metal`, `src/dflash.h`: the DFlash2 drafter and speculative verify/accept/replay.
  - `src/qwen35.h`: the model. Fused projections, paged KV (32-token blocks), GDN state slots, ragged steps
    (`set_batch` + `encode_step`).
  - `src/scheduler.h`: continuous batching, chunked prefill, and the prefix cache (block trie plus GDN state
    snapshots). `src/spill.h`: the SSD tier under it (`SPILL_DIR`): evicted snapshot chains go to disk and come back on
    a later prefix match.
  - `tools/server.mm` → `build/inferd`: engine on a Unix socket (JSON lines). `frontend/serve.py`: the OpenAI API
    (chat template, tokenizer, SSE, usage with `cached_tokens`). `frontend/vision.py`: images (vision tower in MLX,
    M-RoPE positions, content-keyed image token ids).
  - `tools/decode_test.mm`: `check` (logits vs `tools/ref_logits.py`/mlx_lm), `bench`, `prefill`, `gemm`.

## Build and run

```bash
cd engine && ./build.sh                       # needs Xcode's Metal toolchain (Metal 4); build on the machine's own macOS
MODEL=<27B dir> DRAFT=<DFlash2 dir> ./run.sh 8300      # inferd + OpenAI front end on :8300
./run.sh start --model 27b [--dry-run]                 # the same, resolved from ~/models and sized from RAM (install.sh)
./run.sh stop 8300
./test_run.sh && ./test_run_start.sh                   # run.sh's tests (fake processes; after any run.sh change)
tools/package.sh <version>                             # release tarball for this macOS (never lower the Metal target)
```

Smoke tests, to run after every kernel change (each takes seconds):
`decode_test check $M /tmp/toks.txt /tmp/eng.bin {1,24,256}` then `ref_logits.py compare` (expect top-1 ok, cos ≥
0.999 on seqs 1–3, typically 0.9995+), and `decode_test bench $M 1,16,64 2048`. Self-tests: `examples/run_all.sh`. Benchmarks:
`bench/` (see `bench/README.md`); methodology in `docs/RESULTS.md`.

## Gotchas

- `matmul2d` **overwrites** a cooperative destination unless the descriptor has `mode::multiply_accumulate`.
- `threads_per_threadgroup` must have the same vector width as `threadgroup_position_in_grid` in a kernel signature.
- MLX's 4-bit nibble order equals Metal's `uint4b_format`, so MLX checkpoints can feed tensor ops directly.
- Narrow GEMM outputs (N=5120 → 80 tiles) starve the GPU: split-K. The split-K scratch must be sized for the
  *widest* projection the split condition can catch (that bug corrupted logits once).
- Never pass more than ~2 rows to `qmv4` (R ≥ 4 spills registers: 381 ms per step at R=8).
- The mlx-community 27B 4-bit checkpoint has no MTP head. Its current revision (10c35caa) does carry the vision tower
  (333 `vision_tower.*` tensors; the OCR demos score 21/21 through the server on it), as do the mlx-community
  8-bit/bf16 and lmstudio-community 4-bit builds; older snapshots of the 4-bit repo were text-only, so check for
  `vision_tower.*` keys in `model.safetensors.index.json`. Weight bits (4/8/16) are detected per tensor.
- spec_test prefills one prompt per step: a single step over B prompts sizes the per-layer verify tapes for B x T
  tokens (177 GiB at 16 x 6k) and the residency cost of binding them looked like a 16-sequence regression.
- Drafter kernels must cover every row of the block: `draft_attn` was written for 5 rows and silently drafted junk at
  positions 6–8 (block 8 looked useless until 2026-10-30). Check a per-cycle commit histogram when changing the block.
- KV layout is head planes (`[Hkv][slots][Dh]` per layer, `Qwen35::kv_ns`): kernels index `(kh * ns + slot) * Dh + d`, scales `kh * ns + slot`. A matmul2d `store()` to threadgroup memory keeps the destination's row index contiguous regardless of the strides passed (scores come out key-major).
- Anything that embeds tokens must handle all three precisions (`embed4/8/16`): the drafter shares the target's
  embedding, and a 4-bit lookup on an 8-bit table silently dropped acceptance to 1.1 tokens/cycle.
- **Flipped weights (W4A8, default on):** every 4-/8-bit linear except the embedding has its values XOR-flipped at
  load. Anything new that reads `QLinear.wq` must flip back: `qmv4/qmv8` take `GemmArgs.xr`, MLX's qmm and our
  split-K take function constant 40 (patched `quantized_nax.h`). A kernel that forgets reads garbage weights.
- Metal: a **runtime branch in a kernel's final store** made W4A8 1.7× slower (make it a template parameter), and a
  GEMM micro-benchmark that cycles a few weight copies is served from the SLC; `decode_test gemm` walks real layers.
- Env knobs: `DRAFT` (DFlash2 dir), `MODEL`, `KV8` (8-bit KV, default on since 2026-10-31; `KV8=0` bf16), `ATTN_IQ` (int8 queries in verify attention with KV8, default on), `SPEC_AUTO` (self-tuning draft block per load regime, default on; `INFERD_DEBUG=1` logs its table), `W4A8=0` (bf16 activations via MLX qmm),
  `W4A8_MAX` (default 256: larger M uses MLX qmm), `DRAFT_BLOCK` (the widest draft block, default `kMaxBlock` = 16 since 2026-11-02, `src/qwen35.h` / `src/dflash.h`
  `env_block`; DFlash2's trained block is 8 and the extra rows are mask tokens; MoE targets default to 8. Blocks past 8
  run only while decoders x block <= `SPEC_WIDE_ROWS` (32); above `SPEC_WIDE`=4 decoders it drops to 5, and without
  neural accelerators the scheduler starts at 5), `DRAFT_SKIP` (timing only), `SPEC_BLOCKS` (overrides the default tile-aware rule: block 4 when it saves a 32-row W4A8 tile),
  `DRAFT_BITS` (drafter linears, default 4; 8 for 8-bit), `SPILL_DIR`/`SPILL_GIB` (SSD tier, off unless set; default quota 64 GiB), `SNAP_BRANCH_MIN` (256: snapshot at the system prompt's end, from the front end's `snap_at`, and where a prompt branches off a cached one; 0 off), `ADMIT_RESERVE` (generated tokens budgeted per admitted turn, 512: admission is by KV budget, preemption re-queues the newest sequence when the pool runs dry), `W4A8_TM`/`W4A8_TN`/`W4A8_SPLIT_CAP` (auto: 16-row tiles at M ≤ 16), `NO_NAX_ATTN`, `NAX_ROWS`/`NAX_TILE` (8 / 32), `FLASH_ROWS`, `CHUNK_TILE_MAX` (256: chunks of 9..256 rows go to
  `attn_shared` as key-split tiles, mode 6; `CHUNK_TILE_CTX` = minimum history, 0), `RESIDENCY=1` (an MTLResidencySet
  over every buffer; no gain measured), `INFERD_DEBUG`. `decode_test bench <M> <B list> <L>
  [rows]` times verify blocks; `decode_test followup <M> <L> <B list> <chunk list>` times agent turns on a long context; `spec_test ... SPEC_PROF=1` splits a cycle into drafter context / block / target.
- A 64 GB Mac mini M5 Pro works as a dev box for kernel checks (the same neural accelerators, ~1/3 of the Studio's
  speed). If a machine has no Metal toolchain, build on another Mac with the same macOS and copy `build/`. Its GPU runs
  out of memory with 8 × 6k speculative runs; use 2k prompts.
- No neural accelerators (M3/M4, `!g.nax`): GEMMs are MLX's `qmv_wide` (1-10 rows) and `mlx_qmm_sg` (11+, tile per M
  from `Qwen35::sg_tile`), all bf16 activations, no W4A8. MLX's `BlockLoader` silently skips columns when BM*BK/128
  values per thread don't tile the block (BM 24/40/48 at BK 64): check any new tile shape with `decode_test check`
  at a chunk that lands on it. Knobs: `WIDE_MIN`/`WIDE_MAX`/`WIDE_NV`/`WIDE_KL`, `SG_MIN` (11), `SG_BM`/`SG_BK`/`SG_BN`/`SG_W`.
- KV blocks are allocated in aligned free groups of 4 (`grp_free`/`free_groups`): scattered pages cost 10–14% per verify step at 128k–256k. Anything that takes blocks off the pool must go through `alloc_block`/`push_free`.
- **Never lower the Metal kernels' target** (`-mmacosx-version-min` or `MACOSX_DEPLOYMENT_TARGET` for `xcrun metal`):
  a macOS 26 target made MLX's NAX matmuls compute wrong logits on macOS 27 at >= 256 rows (2026-11-03, reverted in
  540f107). Any build-system change needs a `decode_test check` at chunks of 256+ rows (e.g. 256 and 1536) against
  `ref_logits.py`, not only 1 and 24: small chunks never reach the large-M kernels.
