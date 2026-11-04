# Changes from the M4 Max test report (4 November 2026)

An external tester ran inferd on a MacBook Pro M4 Max (64 GB, no GPU neural accelerators) with macOS 27.0, built
from source at 2d2a108. Everything passed (machine details and numbers: [MACHINES.md](MACHINES.md)). The report found
four checks that could not fail, some out-of-date docs and a few setup papercuts. This page maps each item to its fix.

## Findings to act on

| # | Finding | Fix | Commit |
|---|---|---|---|
| 1 | `release_gate.sh` printed ok while `spec_test` reported 2/4 sequences identical; `spec_test` exits 0 on divergence | `spec_test` prints `lossless: <same>/<B> identical, <n> diverged[: seqs ...]` and, with `SPEC_STRICT=1`, exits 3 on divergence. Without the flag the exit status is unchanged, so existing callers keep working. The gate runs it strictly. On exit 3 it runs `tools/tie_check.py`, which now takes the outputs file, `--max-gap` and `--ignore-seqs` and exits 1 above the gap. The gate fails only when a sequence among 1–3 diverges by more than `TIE_MAX` = 0.5 reference logits (TESTING §2.2) | a7ba3f2 |
| 1 | Sequence 0 is known-sensitive | It is **excluded from the lossless criterion**: reported but not judged (`TIE_IGNORE=0`; `TIE_IGNORE=` judges it). Its 93-token prefix sits on a choice the reference scores 1.5 logits apart. The engine flips that choice without the drafter, from the last step's shape alone (5, 16 or 93 rows flip it; 1 or 8 rows do not), and with `KV8=0` every shape flips it. The M6's 0.2.0 gate diverged on the same sequence. Replacing the prompt would have changed the gate's long-standing short prompts, which other results are quoted against. Reasons are in the gate header and TESTING §2.2 | a7ba3f2, 51e7d6c |
| 2 | The gate's logits step ran chunks 1 and 24 on 4 × 24 tokens, and `ref_logits.py make` asserted beyond ~80 tokens | **Already fixed after the tested commit**, in 9dd55ad (review fixes, 4 Nov). `make` builds 2,048-token prompts from repository texts (sequence 0 stays the repetitive one), and the gate runs chunks 1, 24, 256 and 1,536 on 4 × 2,048 tokens. This laptop's gate run passed all four (below) | 9dd55ad |
| 3 | `ref_logits.py compare` failed only on top-1; the gate header said `cos >= 0.9999` | **Already fixed in 9dd55ad.** `compare` exits 1 on non-finite logits, a top-1 difference (explicit bf16 near-ties pass), cos below or KL above the thresholds: 0.995 / 0.03 for long prompts and 0.999 / 0.01 for ≤ 48 tokens on sequences 1–3, and 0.99 / 0.05 on sequence 0. `tools/test_ref_logits.py` proves it fails on bad logits. The header states those numbers | 9dd55ad |
| 4 | `examples/ocr_agents.py` always exited 0 and never printed FAIL | It prints a PASS/FAIL line per document (and per `--compare` round) and exits 1 on any failed request (HTTP error, connection error, malformed reply) or wrong field. Checked against a stub server: all correct gives exit 0; one wrong field, an HTTP 500 or no server gives a FAIL line and exit 1. `run_all.sh` shows those as FAIL. A note beside the M4 Pro 0.2.0 results ([README-ocr.md](../examples/results/m4pro-0.2.0/README-ocr.md)) records that their PASS hid a failed invoice request (15/21 fields) and 4 failed receipt rounds in the compare run. The raw files are unchanged | d803c61 |

## Docs out of date

| Where | Fix (verified against the code or the checkpoint) | Commit |
|---|---|---|
| CLAUDE.md, TESTING §1.2 / §2.4, examples READMEs, CONTRIBUTING, ARCHITECTURE, engine README: "mlx-community 4-bit is text-only" | Revision 10c35caa's index has 333 `vision_tower.*` tensors and no MTP tensors (checked on the Hub). The docs now say so, and that older snapshots were text-only | 51e7d6c |
| MACHINES.md (M4 Pro section) and the comment in `engine/src/qwen35.h` (no-NAX setup): "every quantized GEMM stays on W4A8" | 1–10 rows go to `qmv_wide` (`wide_route`, checked first in `gemm()`), and 11+ rows (`SG_MIN`) go to `mlx_qmm_sg`, both with bf16 activations. W4A8 is unused by default (`w4a8_max = SG_MIN`). This was a comment-only change: `core.metallib`, `inferd` and `decode_test` are byte-identical to a build of 48eeb38 | 51e7d6c |
| CLAUDE.md env knobs: "`DRAFT_BLOCK` (the drafter's block_size, 8)" | The default is `kMaxBlock` = 16 (`src/qwen35.h`, `Drafter::env_block` in `src/dflash.h`), and 8 is DFlash2's trained block. MoE targets default to 8. Blocks past 8 are limited by `SPEC_WIDE_ROWS`. The engine README's knob list had the same error | 51e7d6c |
| TESTING §2.1 KL guidance (W4A8 only) | Added the M1–M4 bf16-path expectation: KL up to ~0.0025 on the flat-distribution sequence 2 (0.00115 in decode, 0.00235 / 0.00246 at 256 / 1,536 rows). Gate sequence 0 at chunk 24 measured cos 0.99479 / KL 0.00284 | 51e7d6c |
| `release_gate.sh` header `cos >= 0.9999` | See finding 3 (9dd55ad) | 9dd55ad |
| MACHINES.md / TESTING §1.1 | Added the M4 Max with the report's numbers, labelled with commit 2d2a108 and 4 Nov 2026 | 51e7d6c |

## Small issues

| Issue | Fix | Commit |
|---|---|---|
| The Quick Start venv lacks `mlx-lm` | `engine/requirements.txt` pins `mlx-lm==0.32.2`, which resolves with mlx 0.32.3 and mlx-vlm 0.7.8 (installed fresh for the gate run below). README, engine README and CONTRIBUTING say so. The CONTRIBUTING and ROADMAP to-do items are done | f0be5d9, 51e7d6c |
| `run_all.sh` hard-coded `LANGBENCH` to a development path | Removed, so `coding_smoke.sh`'s `~/.cache/inferd/languages-benchmark` default applies | d803c61 |
| `run.sh status` printed the current `INFERD_LOGS`, not the directory `start` used | `start` records its absolute log directory in the run directory (`logs-<port>`), and `status` prints it. `test_run.sh` checks this (26/26) | f0be5d9 |
| `engine/gate-*.log` not ignored | Added to `.gitignore` | a7ba3f2 |
| `spec_test` / `tie_check.py` used a fixed `/tmp/spec_out.txt` | `SPEC_OUT`, or by default a fresh `$TMPDIR/spec_out-XXXXXX.txt` that is printed as `outputs:`. `tie_check.py` takes the path as an argument | a7ba3f2 |
| `serve.py` used the deprecated `@app.on_event("startup")` | Replaced by a lifespan handler. The front-end suites, which import the real `serve.py`, pass, and a live server started with `run.sh` connected and answered | f0be5d9 |
| `run_all.sh` gave no hint when coding-smoke exited 2 | The summary prints `tsx missing: ...`, or `toolchain missing: ...` for other missing tools | d803c61 |

## Evidence (MacBook Pro M5 Max, macOS 27.0.1, 5 Nov 2026)

Model `Qwen/Qwen3.8-27B-MLX-4bit` (the official 4-bit MLX build), drafter `incoai/Qwen3.8-27B-DFlash2`.

- **Gate passes, with a real near-tie.** `release_gate.sh` ran with the new venv and passed every step. Logits at
  chunks 1/24/256/1,536 on 4 × 2,048 tokens gave seqs 1–3 cos 0.99821–0.99965 and KL ≤ 0.0132. In the speculative
  step, seq 1 diverged after 79 tokens, with the same token on every run. tie_check gave
  `seq 1 @ 103: plain 52066 (18.000) spec 11386 (18.000) gap 0.000 ... near-tie`, an exact tie, so the step
  passed. The old gate would have printed ok here too, but without checking anything.
- **spec_test:** exit 0 by default, with outputs in a fresh `$TMPDIR` file. With `SPEC_STRICT=1` it exits 3 and
  prints `lossless: 3/4 identical, 1 diverged: seqs 1`.
- **A real gap above the threshold fails.** The same outputs file was doctored to simulate a bug: seq 2's
  speculative token at generated position 40 was replaced. tie_check measured
  `seq 2 @ 64: plain 220 (20.500) spec 1000 (6.250) gap 14.250 ... FAIL: gap > 0.5` and exited 1. With
  `--ignore-seqs 0,2` it reported the sequence as not judged and exited 0.
- **The gate fails end to end.** With `TIE_MAX=-1` (a threshold forced below the exact tie), the gate passed every
  earlier step and then stopped at the speculative step:
  `seq 1 @ 103: ... gap 0.000 ... FAIL: gap > -1.0`, `FAILED (see gate-....log)`, exit 1.
- **Server:** `run.sh` with a non-default `INFERD_LOGS` started the lifespan front end, which answered a chat
  request. `run.sh status` printed the start's log directory, and `run.sh stop` stopped both processes.
- **Byte-identical engine:** `core.metallib`, `inferd` and `decode_test` built from this tree have the same sha-256
  as a build of 48eeb38. The engine changes are comments, plus `spec_test`'s post-processing.
