# Contributing to inferd

inferd is a native inference engine for Apple Silicon: a C++/Objective-C++ host plus Metal kernels, with no MLX
runtime in the engine. It's built for many concurrent agents on one model: continuous batching, DFlash2 speculative
decoding, 8-bit paged KV, a prefix cache with recurrent-state snapshots, and an SSD tier. The primary model is
Qwen3.8-27B.

Start with these:

1. **[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)**, the summary of the study log. Every design decision, measured number, failed experiment and dated
   post-mortem ("Diagnosed YYYY-MM-DD: …") is there. It's long, so read it with this file open. When it disagrees with
   anything else, it's the source of truth.
2. **[`engine/README.md`](engine/README.md)**, the release README: install, run, API, sizing per machine.
3. **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**: how a step, the scheduler and the caches fit together.
4. **[`docs/TESTING.md`](docs/TESTING.md)**: what's tested, what isn't, and how to test and report.
5. **[`docs/ROADMAP.md`](docs/ROADMAP.md)**: what to improve next, with evidence and priorities.

## Repository layout

| Path | What it is |
|---|---|
| `engine/src/qwen35.h` | The model: fused projections, paged KV (32-token blocks, head-plane layout), GDN state slots, ragged steps (`set_batch` + `encode_step`), the kernel routing per M |
| `engine/src/scheduler.h` | Continuous batching, chunked prefill, the prefix cache (block trie + GDN snapshots), admission, preemption, the draft-block tuner |
| `engine/src/spill.h` | The SSD tier under the prefix cache (content-addressed blocks, per-chain state files, optional persistence) |
| `engine/src/dflash.h`, `kernels/spec.metal` | The DFlash2 drafter, speculative verify / accept / replay, speculative sampling |
| `engine/src/moe.h`, `kernels/moe.metal` | MoE (Qwen3.6-35B-A3B smoke port) |
| `engine/src/ane.h`, `kernels/ane.metal` | The Neural Engine FFN split for prefill (`ANE_UNITS`) |
| `engine/src/gpu.h`, `weights.h`, `json.h` | Metal device/pipelines, checkpoint loading (bits detected per tensor), a small JSON parser |
| `engine/kernels/core.metal` | Our kernels: `qmv4/8`, embeddings, norms, GDN, attention prep, `attn_gqa` + `attn_combine` |
| `engine/kernels/w4a8.metal` | W4A8 / W8A8 GEMMs: int8 activations × the checkpoint's 4-/8-bit weights on the integer tensor path |
| `engine/kernels/attn_paged.metal`, `attn_mpp.metal` | Paged NAX verify attention, and the `matmul2d` verify/cascade/chunk-tile attention |
| `engine/kernels/mlx_*.metal`, `qmm_splitk.metal` | MLX's NAX quantized matmul, bf16 GEMM and flash attention (vendored), plus our split-K derivative |
| `engine/kernels/lane.metal`, `qmst.metal` | Experimental small-M GEMMs (off by default) |
| `engine/third_party/mlx/` | MLX v0.32.3 kernel headers, MIT. Unmodified except for the patches listed in its `README` |
| `engine/tools/server.mm` → `build/inferd` | The engine as a server on a Unix socket (JSON lines) |
| `engine/frontend/serve.py`, `vision.py` | The OpenAI-compatible API (chat template, tokenizer, SSE, usage) and the vision tower (MLX) |
| `engine/tools/decode_test.mm`, `spec_test.mm` | Correctness and timing tools (see below) |
| `engine/tools/ref_logits.py` | The mlx_lm reference for `decode_test check` |
| `bench/` | Load harnesses (`studio/conc.py`, `studio/speed.py`, `agents.py`, `shared_prefix.py`, `spill_test.py`) and the driver scripts that produced `results/` |
| `micro/` | Microbenchmarks from the first days (bandwidth, `matmul2d` types, MLX baselines) |
| `results/`, `reports/` | Raw benchmark rows (`.jsonl`) and the HTML reports built from them |

## Prerequisites

- **Hardware:** an M5-family Mac or newer is the target (the kernels use the Metal 4 tensor units, `matmul2d` /
  NAX, and the M5's GPU neural accelerators). **M4-family Macs work but are only lightly tested**: the engine detects
  a GPU without neural accelerators and routes around the MLX kernels that are wrong there (see docs/MACHINES.md).
  Tested on Mac Studio M5 Ultra, MacBook Pro M5 Max, Mac mini M5 Pro (64 GB) and Mac
  mini M6 (32 GB). The 27B at 4-bit needs about 15 GB of weights plus KV, so 32 GB is the practical minimum.
- **macOS 27 or 26.** Each prebuilt file is for one macOS major. macOS 26 is checked on one Mac so far (an external
  tester's 128 GB Mac on macOS 26.6: the source build and the macOS 26 prebuilt both passed the full release gate;
  docs/MACHINES.md).
- **Xcode with the Metal toolchain**, to build: the Command Line Tools alone have no Metal compiler. Check with
  `xcrun metal --version`. If that says `unable to find utility "metal"`, the Command Line Tools are probably the
  active developer directory (`xcode-select -p`); `./build.sh` then uses `/Applications/Xcode.app` by itself, and
  `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer` makes it permanent (`xcodebuild` refuses to run
  until Xcode is selected, or prefix one command with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`). If
  Xcode's toolchain itself is missing, install it with `xcodebuild -downloadComponent MetalToolchain`. Some machines
  can't get it: Apple's catalog has no toolchain for the Mac mini M5 Pro's macOS build. Build on another Mac and copy
  `build/` over (see Workflow).
- **Python 3.12** for the front end and the reference tools:

  ```bash
  python3.12 -m venv ~/.venvs/inferd
  ~/.venvs/inferd/bin/pip install -r engine/requirements.txt   # fastapi uvicorn transformers mlx mlx-vlm mlx-lm pillow numpy
  ```

  (`engine/install.sh` does this, and also handles a venv made by uv, which has no pip.)

  The benchmark harnesses in `bench/` use only the standard library.

## Getting a model

| Use | Hugging Face repo |
|---|---|
| Target, 4-bit (the benchmark default; revision 10c35caa has the vision tower, no MTP head) | `mlx-community/Qwen3.8-27B-4bit` |
| Target, 4-bit with vision tower (alternative) | `lmstudio-community` 4-bit build of Qwen3.8-27B |
| Target, 8-bit / bf16 (with vision tower) | `mlx-community/Qwen3.8-27B-8bit`, `mlx-community/Qwen3.8-27B-bf16` |
| Drafter (DFlash2) | `incoai/Qwen3.8-27B-DFlash2` (the `z-lab` DFlash2 also works) |

```bash
hf download mlx-community/Qwen3.8-27B-4bit --local-dir /Volumes/External/models/Qwen3.8-27B-4bit
hf download incoai/Qwen3.8-27B-DFlash2      --local-dir /Volumes/External/models/Qwen3.8-27B-DFlash2
```

**Keep weights, SSD-tier directories and results on an external drive.** Keep the internal SSD nearly empty. A
Hugging Face cache layout (`models--org--name/snapshots/<hash>`) works too: point `MODEL` at the snapshot directory (`install.sh` and `run.sh start --model` find it there themselves: README, "Models you already have").

The other models (Qwen3.5-9B, Qwen3.6-35B-A3B, Muse Glimmer 30B, Gemma-4-31B) load from their MLX 4-bit builds. The
family is chosen from `config.json`'s `model_type`. They're untuned smoke ports with no drafter
([ROADMAP P1-10](docs/ROADMAP.md)).

## Build

```bash
cd engine && ./build.sh
```

This compiles every `kernels/*.metal` to `build/core.metallib` (`xcrun metal -std=metal4.0`) and builds the server
`build/inferd`; the GPU tools `decode_test`, `spec_test`, `kv_check`, `qsa_bench`, `decode_test_fngemm`,
`decode_test_qsastats` and `weights_test`; and the host-only tests `grammar_test`, `json_test`, `spill_test` and
`tree_sim`. It takes about 15 s. `build/` is git-ignored. If you add a new `.metal` file, add it to both
the compile list and the `xcrun metallib` line.

## Run the server

```bash
cd engine
export MODEL=/path/to/Qwen3.8-27B-4bit DRAFT=/path/to/Qwen3.8-27B-DFlash2 PY=~/.venvs/inferd/bin/python
./run.sh 8300 64 2048 140 96 262144     # port, max agents, step tokens, KV GiB, snapshot slots, max context
curl localhost:8300/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","temperature":0,"messages":[{"role":"user","content":"Hello"}]}'
```

- `run.sh` starts `build/inferd` on `/tmp/inferd-<port>.sock`, then `frontend/serve.py` on the port. Logs go to
  `~/inference/logs/inferd-<port>.log` and `serve-<port>.log`, and the start-up lines report memory, KV pool size and
  slots. Running it again on the same port replaces the old server.
- **Always set `MODEL` and `PY`.** The script's defaults point at the original development machine's paths.
- Sizing per machine is in `engine/README.md`. For example, a 64 GB Mac uses `8300 16 1024 24 30 147456` with
  `SPILL_DIR=… SPILL_GIB=96`, and a 32 GB Mac uses `8300 4 1024 3 8 40960` with an SSD tier. **On the Mac Studio,
  give an explicit KV size**: auto sizing (`kv_gib` 0) has hung it with a small model.
- Without `DRAFT`, speculation is off. Requests with `temperature` > 0 use speculative sampling where it is exact: `top_k` 1–64 (the GPU sampler sees each verify row's top 64 logits), or `top_k` off with `top_p` < 1 (the nucleus is placed against the full softmax; a row whose nucleus does not fit in the 64 is sampled by the host, `inferd.sampling.host_sampled_rows`). `top_k` above 64, or off with `top_p` 1, decodes one token per step on the host sampler (the full distribution); `"inferd": {"spec_sampling": "truncate"}` opts into truncation to the 64 candidates instead. Omitted `top_k` / `top_p` come from the checkpoint's `generation_config.json` (Qwen3.8: 20 / 0.95), so the default sampled request is speculated. The response's `inferd.sampling` reports the effective settings (src/sampling.h, Codex review 2026-11-04 R03). When
  `temperature` is absent, the front end defaults it to 0.7 (`top_k`/`top_p` from generation_config.json): pass `0` for greedy.
- API: `/v1/chat/completions` (streaming or not, `usage.prompt_tokens_details.cached_tokens`, `reasoning_content` when
  thinking is on; `chat_template_kwargs: {"enable_thinking": false}` turns thinking off), `/v1/models` and `/health`.
  Also implemented: tool calls (`tools` / `tool_calls`, streaming and not), `stop` strings, `response_format`
  (`json_object`; `json_schema` with `strict: true` is constrained decoding for a documented subset of JSON Schema,
  then schema validation: examples/structured-output/README.md), and the agent-contexts API (`/v1/contexts`,
  sessions, cache drop: docs/CONTEXTS.md).

## Smoke tests

Run these after every kernel or scheduler change. They take seconds to minutes. Full details and thresholds are in
[TESTING §2–3](docs/TESTING.md).

```bash
cd engine
export M=$MODEL D=$DRAFT
$PY tools/ref_logits.py make $M /tmp/toks.txt 4 2048     # seq 0 repetitive, 1-3 real text (docs, scheduler source)
for c in 1 24 256 1536; do ./build/decode_test check $M /tmp/toks.txt /tmp/eng.bin $c && $PY tools/ref_logits.py compare $M /tmp/toks.txt /tmp/eng.bin; done
#   compare fails (exit 1) on non-finite logits, a top-1 difference (bf16 near-ties are reported as TIE), and on seqs 1-3
#   cos < 0.995 or KL > 0.03 at 2,048 tokens (0.999 / 0.01 for prompts <= 48 tokens), on seq 0 (a deliberately fragile
#   repetitive prompt) cos < 0.99 or KL > 0.05. Measured on the Studio at 2,048 tokens: cos 0.9972-0.9996, KL <= 0.015
#   (the code sequence lowest), with 8-bit and bf16 KV alike.
#   For a quick check after a small change, `make ... 4 24` with chunks 1 5 24 still works (it never reaches the 256+
#   row kernels).
./build/decode_test bench $M 1,16,64 2048          # decode step at B = 1/16/64
./build/decode_test bench $M 1,8,16 2048 5         # 5-row verify blocks
./build/decode_test bench $M 1 8192 8              # 8-row verify block (single-stream speculation)
./build/spec_test $M $D /tmp/toks.txt 128          # speculative output must equal plain decode
```

Full benchmarks (`conc.py`, `agents.py` fleets, the sweeps) run **only at milestones**, on a freshly started server.

## Development workflow

The workflow so far:

1. **Edit on any Mac.**
2. **Build on a Mac with the Metal toolchain** (`./build.sh`).
3. **Run where the hardware is.** Either sync the source and build there:
   ```bash
   rsync -a engine/ <studio>:inferd/engine/
   ssh <studio> 'cd inferd/engine && ./build.sh'
   ```
   or, for a machine with no toolchain, build locally and copy the build. Use the same macOS major version on both.
   ```bash
   rsync -a engine/build/ <mini>:inferd/engine/build/
   ```
4. **Smoke-test** (above). For kernel work, a Mac mini M5 Pro is a fine dev box: same NAX units, about 1/3 the speed.
   Use 2k prompts there, because its GPU runs out of memory with 8 × 6k speculative runs.
5. **Measure on a Mac Studio or Mac mini**, not on a laptop. The MacBook throttles, and its ±7 ms run-to-run noise
   hides 1–2% effects.
6. **Write it down.** Add a dated entry to the study log covering what changed, the numbers before and after on the same
   machine, and what was rejected and why. A root-cause investigation goes in a `### Diagnosed YYYY-MM-DD: <what it
   actually was>` section that keeps the misleading symptom, the evidence and the fix. Rejected experiments get
   recorded too, with their numbers. That's how nobody tries them a third time.

## Coding conventions

- **Match the surrounding style.** It's dense C++20 / Objective-C++ with ARC, header-only engine sources, 2-space
  indent, and short names inside kernels. Don't reformat code you aren't changing.
- **Comments explain why, with evidence.** The existing comments carry the measurement that justified a choice, for
  example `// off: in the server it cut 8k decode 189 -> 111 tok/s`. Keep that density. A non-obvious default or
  routing threshold needs its number and date.
- **Environment knob pattern.** Read a knob where it's used, with its default inline, and comment why the default is
  what it is:
  ```cpp
  int spec_wide = getenv("SPEC_WIDE") ? atoi(getenv("SPEC_WIDE")) : 4;   // why 4: measured ...
  ```
  New experimental paths go in **off by default** behind a knob, and are made default only after a server A/B on the
  Studio *and* a logits check. Document user-facing knobs in `engine/README.md`.
- **Kernels.**
  - Make modes template parameters or function constants, not runtime branches. A runtime branch in a final store
    made W4A8 1.7× slower.
  - Instantiate for every row count the caller can send.
  - Handle all three weight precisions (4/8/16).
- **Vendored MLX code stays unmodified** unless there's no alternative. Any patch must be minimal, gated by a function
  constant, and listed in `engine/third_party/mlx/README`.
- **After touching any shared kernel**, always measure the 27B decode at **B = 1 / 16 / 64**
  (`decode_test bench $M 1,16,64 2048`) **and** verify blocks (rows 5 and 8), and run the logits check. A change that
  only touched the head-shape templating once cost 15–23% at B = 16–64 while B = 1 and the logits looked fine.
- **Don't break the other paths you aren't testing:** 8-bit KV and bf16 KV (`KV8=0`), speculative and plain, the
  drafter, vision (non-text token ids ≥ vocab with external embeddings), and the SSD tier's file format (bump what the
  `MANIFEST` records if the layout changes).

## Gotchas

Each of these has already cost someone hours. Most come with a study-log post-mortem.

**Metal and the tensor units**

- **`matmul2d` overwrites its destination** unless the descriptor uses `mode::multiply_accumulate`. Accumulating
  across K tiles without it silently keeps only the last tile.
- **In a kernel signature, `threads_per_threadgroup` must have the same vector width as
  `threadgroup_position_in_grid`.**
- **A `matmul2d` `store()` to threadgroup memory keeps the destination's row index contiguous** whatever strides you
  pass. A `{BK, 1}` destination came out transposed and gave wrong logits. Scores are stored key-major for this
  reason.
- **A micro-benchmark that cycles a few weight copies runs from the system-level cache and lies.**
  `decode_test gemm` walks every layer's real weights. Use it.
- **Register spills are cliffs, not slopes.** `qmv4` with more than about 2 rows spills (381 ms per step at 8 rows).
  64-row W4A8 tiles are 10× slower. A single 8-row drafter-attention simdgroup spills too. An indexed `float4 v[N]`
  array the compiler didn't keep in registers cost 15–23%.

**Weights**

- **The checkpoint's quantized weights are flipped in place at load** for the int8 path (4-bit `q ^ 8`, 8-bit
  `q ^ 0x80`), for every linear except the embedding. Any new code that reads `QLinear.wq` must flip back: `qmv4/qmv8`
  take `GemmArgs.xr`, and MLX's qmm and our split-K take function constant 40. A kernel that forgets reads garbage
  weights.
- **MLX's 4-bit nibble order equals Metal's `uint4b_format`**, so MLX checkpoints feed the tensor ops directly.
- **Weight bits (4/8/16) are detected per tensor.** Anything that embeds tokens must handle `embed4/8/16`. The drafter
  shares the target's embedding and lm_head, and a 4-bit lookup on an 8-bit table dropped acceptance to 1.1
  tokens/cycle *without* breaking losslessness.
- **Weight views must carry every derived copy.** The drafter's k|v rows are an offset view of a fused weight. Twice,
  a repacked copy (`qmst`, then lane4) wasn't offset with the view, and the drafter quietly produced junk. Use
  `rows_of` (`weights.h`).
- **The mlx-community 27B 4-bit build has no MTP head.** Its current revision (10c35caa) has the vision tower (333
  `vision_tower.*` tensors; OCR 21/21 through the server); older snapshots were text-only, so check the index for
  `vision_tower.*` keys. The mlx-community 8-bit/bf16 and lmstudio-community 4-bit builds have it too.

**GEMM routing**

- **Narrow outputs starve the GPU.** N = 5120 is only 80 tiles, so those shapes use split-K, sized from the GPU core
  count. **The split-K scratch must be sized for the widest projection the split condition can catch.** Getting that
  wrong corrupted logits once.
- Routing by M: `qmv4` for 1–2 rows, W4A8 for 3–255 rows (16-row tiles at M ≤ 16), and MLX qmm at M ≥ 256 for prefill
  (`W4A8_MAX`).

**Speculation**

- **Drafter kernels must cover every row of the block.** `draft_attn` was written for 5 rows and silently drafted junk
  at positions 6–8, which made block 8 look useless for a day. When you change the block, check a per-cycle commit
  histogram (`INFERD_DEBUG=1`).
- **`spec_test` can't see every drafter bug, and on short prompts it reports about 1.01 tokens/cycle because of the
  tool itself.** Confirm acceptance in the server's commit lines.
- **`spec_test` prefills one prompt per step on purpose.** One step over B prompts sizes the per-layer verify tapes for
  B × T tokens (177 GiB at 16 × 6k), and the cost of binding them once looked like a 16-sequence regression.

**KV, memory and the scheduler**

- **The KV layout is head planes:** `[Hkv][slots][Dh]` per layer (`Qwen35::kv_ns`). Kernels index
  `(kh * ns + slot) * Dh + d`, and scales `kh * ns + slot`.
- **KV blocks are allocated in aligned groups of 4.** Scattered pages cost 10–14% per verify step at 128k–256k.
  Anything that takes blocks off the pool must go through `alloc_block` / `push_free`.
- **The prefix cache's snapshots are precious.** An `evict` that drops a snapshot without spilling it, or that evicts
  the snapshot admission just restored, makes agents silently re-prefill their whole context. That bug happened twice
  ("Diagnosed 2026-10-30/31"). Watch `cached_tokens` and the fleet's "new tokens" total.
- **8-bit KV isn't safe for every model.** It breaks Gemma-4, because attention scale 1 on normed q/k amplifies int8
  key error. Check each new model with `KV8=0` and `KV8=1`.
- **Buffers are faulted in on first use**, so an idle footprint hides the real one. On small Macs, size the KV pool
  from the steady-state wired memory (`bench/memsample.sh`), not from the idle number.

**Tooling**

- The tools load `build/core.metallib` relative to the current directory, so run them from `engine/`. `inferd` uses
  `INFERD_HOME`, which `run.sh` sets.
- On macOS, zsh doesn't word-split `${x:+--flag $x}`. Write `--flag=$x`.

## Release gate

Every pull request runs the host-only tests in CI (`.github/workflows/host-tests.yml`): the native JSON parser,
protocol, grammar/sampler and SSD-tier tests under AddressSanitizer and UndefinedBehaviorSanitizer, and the front
end's hardening (auth, budgets, validation, Unicode, cancellation, disconnects over a real socket, context admission),
structured-output, tool-call, stop-string, streaming, image-decoder, logits-gate-fixture and benchmark-client tests.
They need no GPU and no weights; `engine/requirements-test.txt` lists their Python packages.

A release candidate also passes the hardware gate on each qualified Mac, from `engine/` after `./build.sh`, with no
other model process running:

```bash
PY=~/.venvs/inferd/bin/python tools/release_gate.sh $M $D    # host tests, logits vs mlx_lm, bench, spec = plain
```

The gate's log starts with the machine, macOS version and build, the metallib and binary sha-256 and the model and
drafter config hashes. Its logits step runs 4 prompts of 2,048 tokens at chunks 1, 24, 256 and 1,536 (the 256+ row
kernels are where a wrong build target broke logits before), with the thresholds above. The bench step is skipped,
with a note in the log, when Metal's GPU budget is under 40 GiB. The speculative step passes when plain and speculative
decoding match, or when every diverging sequence among 1-3 is a near-tie (reference gap <= 0.5, `tools/tie_check.py`);
sequence 0, the repetitive prompt, is reported but not judged ([TESTING §2.2](docs/TESTING.md)). Run it on every
supported macOS version and build.

Then the fleet gates with a server running: `conc.py --mode followup` at 16k × 8/16/32, `wall.py` N = 1–8, mixed
arrivals and SSD pressure ([TESTING](docs/TESTING.md)). Record the commit, binary and metallib hashes, the model and
drafter revisions and the machine with every result, and compare like with like.

## Proposing changes

- **Branch from `main`** with a descriptive name (`kernels/attn-mpp-gq4`, `frontend/tool-calls`, `fix/auto-kv-hang`),
  and open a pull request. Keep a PR to one idea. Experiments that didn't pan out can still be merged as an
  off-by-default knob, or as a study-log entry, if the result is useful.
- **A PR description should include:**
  - what changed and why, linking the ROADMAP or TESTING item;
  - **before/after numbers from the same machine and build base**, from the right tool (`decode_test bench`/`gemm`
    for kernels, `conc.py` or `agents.py` for scheduler and server changes);
  - the `ref_logits.py compare` lines for any change on the compute path, and the `spec_test` result for anything
    near speculation;
  - the machine, macOS, env knobs and thermal state ([TESTING §4](docs/TESTING.md#4-how-to-report-a-result));
  - the study log entry.
- Don't commit weights, `build/`, SSD-tier directories or logs. Raw result rows (`.jsonl`) under `results/` are
  welcome when the PR quotes them.
- Don't put usernames, home paths, hostnames or IP addresses in code, scripts or results. Refer to machines by
  hardware name (Mac Studio M5 Ultra, Mac mini M6, …).

## Where you can help

Pick something, comment on its issue (or open one), and ask questions early.

**Metal kernels**

- Make `attn_paged_nax` / `attn_mpp` generic over (GQ, Dh) so the other models get the fast verify attention
  ([ROADMAP P1-10](docs/ROADMAP.md)).
- Debug `ANE_A8D`. The suspect is the int8 × int8 down matmul shapes; start from `tools/ane_rot_test.mm`
  ([P1-3](docs/ROADMAP.md)).
- bf16 GDN state behind a knob, plus its quality check ([P1-6](docs/ROADMAP.md)).
- Wide-GEMM research: reuse staged weights across row tiles ([P1-4](docs/ROADMAP.md)). Read the rejected list first.
- Re-measure `QMST=1` now that the drafter view bug is fixed ([P1-1](docs/ROADMAP.md)).

**C++ scheduler / engine**

- Fix the auto KV sizing hang on the Mac Studio ([P0-2](docs/ROADMAP.md)). This is a good first issue.
- Investigate the long-uptime slowdown at 128k × 4 ([P0-1](docs/ROADMAP.md)).
- A forced-preemption test and any fixes it finds ([P0-5](docs/ROADMAP.md), [TESTING §1.4](docs/TESTING.md)).
- Sliding-window ring blocks for Gemma-4 and Glimmer ([P1-7](docs/ROADMAP.md)).
- Elastic KV for 32 GB Macs ([P1-8](docs/ROADMAP.md)).
- Flash-Next bring-up ([P1-9](docs/ROADMAP.md)). This is a large piece of work, best done with the maintainers.

**Python front end**

- Fail in-flight requests when `inferd` dies, and make `/health` check the engine ([P0-5](docs/ROADMAP.md)).
- Look into the Glimmer replies that begin `to=self<|message|>` (`results/models/studio/speed-glimmer.jsonl`), which
  may be a chat-template or stop-token issue ([TESTING §1.2](docs/TESTING.md)).

**Benchmarking / QA**

- The 120-test coding benchmark (v2) re-run ([P0-4](docs/ROADMAP.md)).
- A scripted vision regression test ([P0-6](docs/ROADMAP.md), [TESTING §2.4](docs/TESTING.md#24-vision)).
- Run the smoke tests and a `conc.py` cell on a Mac not in the matrix, such as another M5 configuration
  ([TESTING §1.1](docs/TESTING.md)).
- Fleet and quality runs for the 27B at 8-bit, the vision-capable precision ([P1-11](docs/ROADMAP.md)).
- Anything marked **Untested** in [TESTING §1.4](docs/TESTING.md): client cancel, vision with the SSD tier, crash
  recovery.

**Packaging / release**

- Make `run.sh` path-free (required `MODEL`, configurable log directory and socket) and generalise the `bench/` driver
  scripts ([P2-6](docs/ROADMAP.md)).
- A launchd service definition, a release tarball with a prebuilt `core.metallib`, and a self-hosted CI job that
  builds and runs the smoke tests.

**Docs**

- One table of supported environment knobs, generated from the `getenv` calls, marking timing-only and dead ones
  ([P2-7](docs/ROADMAP.md)). Reconcile it with `CLAUDE.md` (it says `SPEC_WIDE` = 2; the code defaults to 4).
- Turn the study log post-mortems into a short "design overview" for newcomers: the step, the scheduler and the cache.
