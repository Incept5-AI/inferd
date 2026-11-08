# Feature matrix

State on `main`. **Shipped**: on by default or a documented request field, covered by tests or gates. **Opt-in**: in
the engine, off unless an environment variable or field enables it. **Not merged**: exists in a separate working tree
or patch and is not in `main`. Measurements quoted elsewhere name the build they came from; check them against this
table before reusing them.

| Area | Feature | State | Notes |
|---|---|---|---|
| Models | Qwen3.8-27B (4/8-bit, bf16), DFlash2 drafter, tree verification for one stream | Shipped | primary target |
| Models | Qwen3.8-Flash-Next (oQ4e), MTP + copy drafts, QSA sparse attention | Shipped | needs the Mac Studio (~170 GiB) |
| Models | Qwen3.6-35B-A3B (MoE), Qwen3.5-9B, Muse Glimmer 30B, Gemma-4-31B | Shipped | no drafter for these |
| Models | Vision (images through the checkpoint's own tower, in the front end) | Shipped | start with `VISION=1` for image work on ≤ 64 GB Macs |
| Numerics | 8-bit KV (`KV8`), W4A8/W8A8 int8 activations | Shipped (default on) | `KV8=0`, `W4A8=0` restore bf16 |
| Numerics | int8 prefill GEMM (PF8) | Shipped (auto) | on for 27B-class models where the KV pool still fits after the int8 copies (Studio; 128 GB laptops); off on 32–64 GB minis; Flash-Next opt-in (`PF8=1`) |
| Scheduling | Continuous batching, chunked prefill, KV-budget admission, preemption | Shipped | |
| Scheduling | Prefill ordering (`PREFILL_SHORT`, `PREFILL_PRIORITY`) | Shipped | |
| Scheduling | Prefill budget beside decoders (`PREFILL_PRIO`, default `max_running` since 8 Nov), MTP verify rows by load (`MTP_WIDE`), a start-up warning when an explicit KV pool exceeds the GPU budget (`KV_FIT=1` fits it), admission lookahead with aging (`ADMIT_LOOKAHEAD=8`) | Shipped | docs/RESULTS.md (k), (ae) |
| Speculation | Flash-Next copy drafts at any load (`COPY_SOLO=1` for the old rule) | Shipped | |
| Cache | Prefix trie + recurrent-state snapshots, branch-point and decode snapshots | Shipped | `SNAP_GEN_FORCE` opt-in |
| Cache | SSD tier (`SPILL_DIR`), checksummed entries | Opt-in | off unless `SPILL_DIR` is set |
| Agents | `/v1/contexts`: saved/pinned contexts, launching agents from them, sessions, `cache: drop` | Shipped | docs/CONTEXTS.md |
| Agents | Per-agent `max_context`, `effort`, `priority` | Shipped | |
| API | OpenAI chat completions, streaming, usage with `cached_tokens` | Shipped | |
| API | Tool calls, `stop` strings | Shipped | |
| API | `response_format`: `json_object`, `json_schema` (validate and retry) | Shipped | |
| API | `json_schema` with `strict: true` (constrained decoding for a documented subset, then validation) | Shipped | examples/structured-output/README.md |
| Hardware | M5/M6 neural-accelerator paths | Shipped | Studio M5 Ultra, M5 Pro, M6 qualified |
| Hardware | M3/M4 paths (no neural accelerators) | Shipped, lightly tested | M4 Pro measured; M3 Ultra by a contributor |
| Hardware | Neural Engine prefill split (`ANE_UNITS`) | Opt-in, experimental | private Apple API |
| Deployment | Loopback HTTP; engine socket in an owner-only directory; `run.sh start/stop/status` | Shipped | trusted local user only: SECURITY.md |
| Deployment | Bearer API key (`INFERD_API_KEY`; required off loopback), request/queue/body limits with 429/503/413 | Shipped | engine/README.md, "Security and limits" |
| Deployment | Checkpoint validation (shapes, dtypes, capacities) and per-request native validation | Shipped | malformed checkpoints fail before any copy |
