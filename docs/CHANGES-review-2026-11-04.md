# engine-review.patch: fixes for review R01-R08 (codex/review-2026-11-04)

These fixes answer the review of commit 2d2a108. The patch (`engine-review.patch`, a git diff) is against
**Incept5-AI/inferd cc133d1** and applies cleanly to `git archive cc133d1`. The work was rebased from 2d2a108 to
1138540 and then to cc133d1 without conflicts. My hunks in the files that upstream also changed (qwen35.h, server.mm,
decode_test.mm, build.sh, README, RESULTS) are identical across the rebases. The working tree with its history is in
`~/dev/inference/engine-review` (local git: branch `review2` on `upcc133d1`). Nothing has been pushed.

## Per finding

- **R01 image decoder.**
  - Pillow is pinned at 13.0.0.
  - The new `frontend/image_decode.py` decodes only PNG, JPEG, WebP and GIF (first frame), via `Image.open(formats=...)`.
  - Decoding runs in a child process that is killed after `INFERD_IMAGE_DECODE_TIMEOUT` (10 s). `INFERD_IMAGE_DECODE=inline` decodes in-process.
  - The pixel and byte budgets are unchanged. They are checked in the child before full decode.
  - The review's XBM probe (k = 10/20/60) is now refused in 0.04 s (child) or less than 0.01 s (inline). It is refused on Pillow 12.3.0 too. The OCR samples still load.
  - The subprocess adds about 37 ms per image.
- **R02 geometry.**
  - `validate()` now rejects:
    - more than 512 experts (`moe_route`/`_bf`/`moe_sort`)
    - `conv_k != 4` and `linear_key_head_dim != 128` (GDN kernels)
    - `head_dim > 512` (`attn_prep`)
  - At load:
    - the fallback to `attn_decode` (no `attn_gqa`) requires `head_dim` 256
    - the MoE router, gate/up and down stacks are checked against the config
  - PF8 is skipped when K > 32768, and the ANE split refuses `hidden` > 5120.
  - `weights_test` has new cases: 512 accepted, 513 (the review's `moe_513`) and 65536 rejected, conv 3/5, `ldk` 64/256, `head_dim` 512/1024.
  - Every real config on the SanDisk still validates. The existing gemma-4-26b-a4b refusal is unchanged.
- **R03 sampling (extended at the coordinator's request).**
  - `src/sampling.h` is the one contract for every path: top_k, then top_p after top_k, over the whole vocabulary, ties by token id.
    - The host path's 1,024 cap is gone, and the grammar sampler uses the same contract.
  - The GPU speculative sampler now keeps **64** exact candidates (`row_top64_shard_v`/`top64_merge_v`).
    - A step whose sampled sequences all have top_k 1..16 uses the 16-candidate kernels.
    - With top_k off and top_p < 1, `verify_zpart` computes each row's full softmax normaliser. A row whose nucleus falls outside the 64 candidates is sampled exactly on the host and counted (`spec_host`, `/v1/cache` `spec_sampling`).
  - The GPU used to break logit ties by thread, which changed the support at the top_k boundary on bf16 ties. It now breaks them by id. The new GPU test found this.
  - Not speculated (one token per step):
    - top_k > 64
    - top_k off with top_p = 1
    - unless the request opts into `"inferd": {"spec_sampling": "truncate"}`
  - Omitted `top_k`/`top_p` come from `generation_config.json`, as vLLM does (Qwen3.8: 20 / 0.95). The default request is therefore speculated.
  - Effective settings come back in `inferd.sampling`. The docs no longer call sampling exact without saying where it is exact.
- **R04 release gate.**
  - `ref_logits.py make` builds long prompts: sequence 0 is repetitive, 1-3 are real text from repo docs and source.
  - `compare` fails on non-finite logits, top-1 (bf16 near-ties are reported as `TIE`), cos and KL.
    - Floors for T > 48: 0.995 / 0.03, calibrated on the Studio.
    - Floors for short prompts: 0.999 / 0.01.
    - Floors for sequence 0: 0.99 / 0.05.
  - `release_gate.sh`:
    - runs 4 × 2048 at chunks 1, 24, 256 and 1536
    - logs machine, OS, Xcode, metallib/inferd sha-256 and model hashes
    - runs the bad-logits fixtures (`tools/test_ref_logits.py`) and `spec_sample_test`
    - skips bench below 40 GiB (`decode_test budget`)
- **R05 Unicode.**
  - Lone surrogates get HTTP 400 `invalid_unicode` at the boundary.
  - Native `handle_line` sends a terminal error for any request whose id it can recover (`src/protocol.h`). If it cannot recover an id, it drops the connection.
  - New backstop: `INFERD_EVENT_TIMEOUT` (1800 s; HTTP 504 or an SSE error, plus a native cancel).
- **R06 contexts.**
  - The slot is reserved before any await and released on failure or cancel.
  - New `INFERD_MAX_CONTEXT_MB` (1024) budget on retained bytes, covering effort variants too.
- **R07 disconnects.** Non-streaming requests watch for `http.disconnect` and cancel natively (status 499). Real-socket tests cover both modes.
- **R08 CI.**
  - New `engine/requirements-test.txt`.
  - CI and the gate now run `test_hardening`, `test_image_decode` and the logits fixtures.

## Test results

**Host tests (ASan/UBSan):**

| Test | Result |
|---|---|
| grammar | 105/105 |
| json | 96/96 |
| spill | 39/39 |
| weights | 64/64 loaders + 21 headers, on the Studio with a metallib |

**Front end:**

| Test | Result |
|---|---|
| hardening | 44 |
| structured | 31 |
| tools | 16 |
| stream | 7 |
| image_decode | 9 |
| vision | 12 |
| logits fixtures | 8 |
| bench SSE | 16 |

All front-end tests also pass in a clean venv built from `requirements-test.txt`.

**Review repros:** context_limit gives [200, 400, 400, 400]. unicode_protocol returns 400 and nothing is pending. nonstream_disconnect sends a cancel, releases the slot and finishes the task. moe_513 exits 1.

**Studio** (M5 Ultra, macOS 27.0.1, 27B 4-bit + DFlash2, all under the lock), on the 1138540-based build:

- `release_gate.sh` passed, with compares at chunks 1, 24, 256 and 1536 all passing.
  - Real-text cos: 0.99719-0.99963. KL ≤ 0.0148.
  - Sequence 0 cos: 0.9960-0.9970.
- `spec_sample_test`: 41/41.
- spec = plain: 3/4 identical. Sequence 0 diverges after 69 tokens, the same as unmodified 1138540.
- Live checks: 14/14.

**Measured cost (Studio, 16k followup, sampled t 0.7 / top_p 0.95):**

| Setting | Before (2d2a108 / K=16) | Now |
|---|---|---|
| top_k 20, single stream | 46.3 tok/s, not speculated | 86.3 tok/s |
| top_k 20, 16k × 8 | 173.7 agg tok/s | 477-483 |
| top_k 16, 16k × 8 (fixed K: 64 vs 16) | 492.2 | K=64 467-470; adaptive 484-490 |
| top_k off, top_p 0.8, 16k × 8 | not exact | 481-482 (2 of 32,800 rows host-sampled) |
| greedy 16k × 1/8/32 | baseline 1138540: 160/413/614 | 145/491/624 (noise) |

**Final gate on the cc133d1 rebase** (QA8 auto-on; `studio:~/inference/engine-review-cc`, 21:39–21:48):

- `release_gate.sh` **passed**: host tests, `spec_sample_test` 41/41, front end, fixtures, and logits at 1/24/256/1536.
  - Real-text cos: 0.9982-0.9997. KL ≤ 0.0162.
  - Sequence 0 cos: 0.9967-0.9986.
- Bench ran. Spec = plain is unchanged from upstream: 3/4 identical, with sequence 0's known near-tie at token 69.
- Live checks: 14/14.
- conc greedy 16k × 1/8/32: 145.7 / 507.5 / 715.9.
- conc sampled (0.7 / 0.95 / top_k 20) 16k × 1/8: 158.2 / 514.0. Two rows were host-sampled.

Results: `studio:~/inference/results/engine-review/` (`ktopk/`, `kadapt/`, `cc133d1/`, `calib/`); copy in
`results/engine-review-2026-11-04/`.

## Caveats

- These checks cover the 27B 4-bit + DFlash2 only. Flash-Next (MTP drafting), Gemma and the MoE models ran no GPU gate. Their configs and MoE tensor shapes were checked against the new validation from the headers only.
- The opt-in `spec_top16` is now `spec_truncate` (64 candidates). The `spec` / `spec_host` done-event fields are additions to the protocol.
- One slip on my side: around 19:45 I ran `ref_logits.py make` on the Studio outside the lock. It loads the 27B. The lock log shows nobody held the lock at the time.
