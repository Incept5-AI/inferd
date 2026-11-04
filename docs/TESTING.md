# Testing inferd

This page says what needs testing, how to test it, and how to report what you find. All numbers quoted come from
the study log (summarised in [EXPERIMENTS.md](EXPERIMENTS.md)) or from the raw rows in [`results/`](../results). If a claim here disagrees
with the study log, the study log is right. Please open an issue for the mismatch.

Coverage legend used throughout:

| Mark | Meaning |
|---|---|
| **Tested** | Exercised repeatedly by the benchmarks in `bench/`, with results in `results/` |
| **Light** | One or two runs, or only a correctness check, with no load or long-run testing |
| **Untested** | Never run, as far as the study log and `results/` show |
| **Broken** | Known to fail. Don't use it, but do help fix it |
| **Unsupported** | Not implemented |

---

## 1. Test matrix

### 1.1 Machines

| Machine | GPU / memory | Coverage | Notes |
|---|---|---|---|
| Mac Studio M5 Ultra | 80-core GPU, 256 GB | **Tested** | Main development and scoreboard machine. All headline numbers come from it. |
| Mac mini M5 Pro | 20-core GPU, 64 GB | **Tested** (fleets) | Runs agent fleets at 16k and 32k from 8 to 64 agents, plus single-stream sweeps from 8k to 128k. It has no Metal toolchain, so you build elsewhere and copy `build/` over. |
| Mac mini M6 | 12-core GPU, 32 GB | **Tested** (fleets) | Runs agent fleets at 8k, 16k and 32k. Memory is tight: the KV pool is about 3 GiB (~96k tokens) and the SSD tier is essential. The Neural Engine prefill split (`ANE_UNITS=24`) only helps here. |
| MacBook Pro M5 Max | — | **Light** | Used for kernel development, the SSD-tier tests and the Splash A/B. It throttles: plain decode drifted from 32 to 45 ms in one A/B ("Lessons from the mlx.fast competition"), so don't trust its end-to-end A/Bs. |
| Other M5-family Macs (base M5, other memory sizes) | — | **Untested** | They should work: the kernels need the Metal 4 tensor units (NAX) found on M5-class and newer GPUs. |
| Mac mini M4 Pro | 20-core GPU, 64 GB, no neural accelerators | **Light** | Works since 1 Nov 2026 (NAX kernels routed around; drafter scratch fix). Correct vs mlx_lm; 27B decode 24.7 tok/s with the drafter. **Not fully tested**: see the list in docs/MACHINES.md. |
| Mac Studio M3 Ultra | 80-core GPU, 512 GB, no neural accelerators | **Light** | Works since 1 Nov 2026 with bf16-activation GEMMs (MLX `qmv_wide` / fitted `qmm` tiles, no W4A8). §3.1 smoke set and conc.py 16k × 1/4/8 only (the study log "M3 Ultra round 2"). |
| Other M4 Macs, other M3 Macs and older | — | **Untested** | M4 and M3 should behave like the M4 Pro / M3 Ultra (same Apple9 family). M2 and older are untested. |

### 1.2 Models and precisions

| Model / checkpoint | Precision | Drafter | Coverage | Evidence |
|---|---|---|---|---|
| Qwen3.8-27B, `mlx-community/Qwen3.8-27B-4bit` | 4-bit | incoai DFlash2 | **Tested** | Every scoreboard run. The checkpoint is text-only, with no vision tower and no MTP head. |
| Qwen3.8-27B, `mlx-community` 8-bit | 8-bit | DFlash2 | **Light** | Logits check, `decode_test` and `spec_test` numbers, and the vision check ("Weight precisions", "Vision"). No fleet runs. |
| Qwen3.8-27B, `mlx-community` bf16 | bf16 | DFlash2 | **Light** | Logits check and `decode_test`/`spec_test` only. |
| Qwen3.8-27B, `lmstudio-community` 4-bit | 4-bit | DFlash2 | **Light** | Logits check only. This is the 4-bit build that has a vision tower. |
| Qwen3.8-27B, official Qwen MLX 4-bit | 4-bit | z-lab DFlash2 | **Light** | Laptop runs only (SSD tier, the block-8 fix, the Splash A/B). |
| Qwen3.5-9B | 4-bit | none | **Light** (smoke port) | Logits cos ≥ 0.9997. One speed sweep and one 32k agent run per machine (`results/models/`). |
| Qwen3.6-35B-A3B (MoE) | 4-bit | none | **Light** (smoke port) | Logits top-1 ok. Smoke runs only (`results/models/`). |
| Muse Glimmer 30B | 4-bit | none | **Light** (smoke port) | Logits cos ≥ 0.9997, including decode past the 2,048-token sliding window. ⚠ The replies in `results/models/studio/speed-glimmer.jsonl` begin with `to=self<|message|>…`, which suggests a chat-template or stop-token problem. **Not investigated.** |
| Gemma-4-31B | 4-bit weights, **bf16 KV only** | none | **Light** (smoke port) | 8-bit KV breaks it (KL 1.96, top-1 flips), so it defaults to bf16 KV. 32k is capped at about 5 agents ("Other models on inferd"). |
| Qwen3.8-Flash-Next | — | — | **Unsupported** | This is the next target model; see [ROADMAP](ROADMAP.md). |

Every non-27B model is an **untuned smoke port**. The fast verify-attention kernels (`attn_paged_nax`, `attn_mpp`)
only handle the 27B's head grouping (GQ = 6, Dh = 256) and are switched off for the other models. None of those models
has a drafter.

### 1.3 Context × concurrency (Qwen3.8-27B 4-bit + DFlash2)

`conc.py --mode followup` cells measured on the M5 Ultra. Each cell is the context per agent × the number of agents
in flight.

| Context | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| 16k | Tested | Tested | Tested | Tested | Tested | Untested on the Studio |
| 32k | Tested | Tested | Tested | Tested | Tested | Untested |
| 64k | Tested | Tested | Tested | Tested | Tested | Untested |
| 128k | Tested | Tested | Tested | Tested | Untested | Untested |
| 256k | Tested | Tested | Tested | Untested | Untested | Untested |

On the minis, `agents.py` runs N:N fleets: M6 at 8 to 64 agents (8k and 16k), M5 Pro at 8 to 64 agents (16k) and 8 to
32 agents (32k). `decode_test bench` covers B = 64 at 2k, but **no server run on the Studio has gone above 32 agents.**

### 1.4 Features

| Feature | Coverage | How it was checked / what is missing |
|---|---|---|
| Greedy speculative decoding (DFlash2) | **Tested** | `spec_test` matches plain decode except near-ties (§2.2), plus acceptance on agent text (4–5.5 tokens/cycle). |
| Speculative *sampling* (temperature > 0) | **Light** | One distribution check (400 samples) and the 48-test coding suite. Exact by construction since 2026-11-04 (top-k 1–64; top-p with rows beyond 64 candidates sampled on the host). `spec_sample_test` (GPU, no model) checks the top-K order and the accept/resample distribution against `src/sampling.h` by chi-square; `grammar_test` checks the host sampler (top-k 0, 16, 20, top-p cuts, unconstrained and under a grammar). |
| Self-tuning draft block (`SPEC_AUTO`) | **Tested** | Studio and mini A/Bs. |
| 8-bit KV (default) vs bf16 KV | **Tested** (27B) | ⚠ It breaks Gemma-4. Each new model needs its own check. |
| Prefix cache (block trie + GDN snapshots) | **Tested** | Fleets reach 93–99% cache hits. |
| Fan-out prefix sharing (`FANOUT_MIN`) | **Light** | `shared_prefix.py` at 32k × 8 and 64k × 16. |
| SSD tier (`SPILL_DIR`) | **Tested** on the minis | `spill_test.py`: 9/9 replies identical on the M5 Pro. |
| Persistent SSD tier (`SPILL_PERSIST=1`) + SIGTERM flush | **Light** | One run of `persist_test.sh`: 6/6 identical replies. |
| KV-budget admission | **Tested** | Fixed the M6 32k deadlock. |
| **Preemption** (`preempt()`) | **Untested under load** | The run that validated admission logged 0 preemptions ("v10"). Nothing has forced the preemption path yet. |
| Vision (`image_url`) | **Light** | One synthetic image, word-for-word identical to `mlx_vlm.generate` (8-bit + DFlash2), plus a cache-hit check. **Untested:** multiple images per request, vision under multi-agent load, vision combined with the SSD tier, and the 4-bit lmstudio build through the server. |
| Thinking on (`enable_thinking`, `reasoning_content`) | **Light** | Most benchmarks run with thinking off. The coding suite ran at effort `medium`. |
| Streaming SSE, `stream_options.include_usage`, `cached_tokens` | **Tested** | The harnesses read these fields. |
| Client disconnect / cancel | **Untested** | `serve.py` sends `cancel` when a stream ends early, but nothing tests it. |
| **Tool calling** through the OpenAI API | **Works** (1 Nov 2026) | `<tool_call>` output comes back as `message.tool_calls` / streaming `delta.tool_calls`, parallel calls included. Unit test `engine/frontend/test_tools.py`; live in `examples/tools-*`. |
| **JSON mode / `response_format` / structured output** | **Unsupported** | `serve.py` ignores it. |
| User `stop` sequences | **Unsupported** | `serve.py` stops only on the model's EOS/end-of-turn ids. |
| Neural Engine prefill split (`ANE_UNITS`) | **Light** (M6 only) | +12–23% on M6 fleets. +3% on the M5 Pro, so not used there. |
| `ANE_A8D` (int8 down projection on the ANE) | **Broken** | Produces inf in layer 0 and all-zero output from layer 1 on. Off by default. |
| `QMST=1` (staged-tile GEMM) | **Light** | A drafter bug was fixed but **not re-measured**. Off by default. |
| `LANE=1` (lane4 GEMM) | **Light** | Slower on the Studio. Drafter acceptance fell (2.82 → 2.33) and the cause wasn't found. Off by default. |
| `CASCADE=1` (cascade attention) | **Light** | Correct but slower. Off by default. |
| Long uptime (hours, many cells) | **Known problem** | 128k × 4 runs at 256/249 tok/s on a fresh server against 225 after a 21-cell sweep ("Overnight agent-fleet results"). Unexplained. |
| Crash / recovery (inferd dies, front end dies) | **Untested** | If `inferd` exits, `serve.py`'s reader loop ends and open requests never complete. `/health` returns `ok` without checking `inferd`. |
| Studio auto KV sizing (`kv_gib` 0) | **Known problem** | With a small model, the auto pool filled 222.5 of the 224 GiB budget and the first request hung. On the Studio, pass an explicit `kv_gib`. |

---

## 2. Correctness checks

### 2.1 Logits against mlx_lm (`decode_test check` + `ref_logits.py compare`)

`decode_test check` feeds B sequences through the engine in ragged steps and writes the last position's logits
(float32 `[B, vocab]`). `ref_logits.py compare` runs `mlx_lm` on the same tokens and prints one line per sequence:

```
seq 1: cos 0.99996  max|diff| 0.412  KL 0.00004  top1 ok (' the')  top5 overlap 5/5
```

Run everything from `engine/`, because the tools load `build/core.metallib` relative to the current directory:

```bash
cd engine
export M=/path/to/Qwen3.8-27B-4bit
PY=/path/to/venv/bin/python        # needs mlx and mlx_lm (see CONTRIBUTING.md)

$PY tools/ref_logits.py make $M /tmp/toks.txt 4 24            # 4 sequences x 24 tokens (quick; see below for 2,048)
./build/decode_test check $M /tmp/toks.txt /tmp/eng.bin 1     # chunk 1: pure decode, token by token
$PY tools/ref_logits.py compare $M /tmp/toks.txt /tmp/eng.bin
./build/decode_test check $M /tmp/toks.txt /tmp/eng.bin 24    # one chunk of 24: the prefill path
$PY tools/ref_logits.py compare $M /tmp/toks.txt /tmp/eng.bin
./build/decode_test check $M /tmp/toks.txt /tmp/eng.bin 5     # verify-block-sized chunks
$PY tools/ref_logits.py compare $M /tmp/toks.txt /tmp/eng.bin
```

The optional last argument is `chunk`, or `first,chunk`: for example, `24,1` prefills 24 and then decodes one at a
time. Chunks 1, 5, 8 and 24 exercise different attention modes: decode, multi-row verify, and prefill.
`ref_logits.py make` with T <= 48 builds the historical short prompts from four built-in texts. Larger T (the
release gate uses `4 2048` at chunks 1, 24, 256 and 1,536) gives sequence 0 the repetitive "quick brown fox" text and
sequences 1-3 real text from the repository (`CORPUS`: docs/ARCHITECTURE.md, docs/RESULTS.md, the scheduler source),
so 256- and 1,536-row chunks really run. `compare` exits 1 on non-finite logits, a top-1 difference (an explicit
near-tie within one bf16 step is reported as `TIE` and passes), cos below `--cos-min` or KL above `--kl-max` on sequences
1-3 (0.999 / 0.01 for T <= 48; 0.995 / 0.03 for longer prompts: at 2,048 tokens the Studio measured 0.9972-0.9996 and
KL <= 0.015, with 8-bit and bf16 KV alike, the code sequence lowest), and below 0.99 / above 0.05 on sequence 0; `tools/test_ref_logits.py` proves it fails on bad
logits. For other long-context checks, write your own token file. The format is `B T` on the first line, then B lines of T
token ids.

**What the thresholds mean.**

- **top-1 ok:** the engine's most likely next token is the reference's. This is the hard requirement. A `DIFF` on a
  normal sequence is a bug.
- **cos ≥ 0.999 (soft):** the full logit vectors point the same way. Sequences 1–3 typically measure 0.9995–0.99997
  at chunks 1 and 24 and 0.9992–0.9999 at 256- and 1,536-row chunks (the 0.2.0 release gates on four Macs, RESULTS.md
  (n)); an external M4 Max on macOS 26 measured 0.9996–0.9997. A value in that range with top-1 and top-5 correct is a
  pass, not a regression. Sequence 0 is "The quick
  brown fox…" repeated, a deliberately fragile, highly repetitive prompt. It has measured 0.996 / KL 0.003 since day 1
  because the reference uses MLX's prefill path and the engine a different one, and that gap grows on repetitive
  text.
- **KL:** with the default int8 activations (W4A8), expect KL ≤ 0.0007 on normal sequences and 0.005–0.009 on the
  repetitive one. With `W4A8=0`, the repetitive one drops to about 0.004 ("W4A8").
- **The engine isn't batch-invariant at the bit level** (vLLM isn't either). Different kernel paths can flip a
  near-tie. That isn't a bug unless top-1 moves on a normal sequence.

**Per-model and per-path expectations** (all from the study log):

| Configuration | Expectation |
|---|---|
| 27B 4-bit, chunks 1/5/8/24, 8-bit or bf16 KV | top-1 ok on every sequence; cos ≥ 0.999 on sequences 1–3 (typically 0.9995+) |
| 27B 8-bit / bf16 checkpoints | cos ≥ 0.999 on `toks.txt` (chunks 1/5; typically 0.9999) |
| 27B at ~3k tokens, 8-bit KV (chunks of 100 after 1,000) | cos ~0.998 / 0.9994 on two sequences (bf16 KV is about the same), top-5 5/5 |
| Qwen3.5-9B | cos ≥ 0.9997, decode and chunked |
| Muse Glimmer 30B | cos ≥ 0.9997 at 3,000 tokens in 1,024 and 160 chunks, and decode past the 2,048 window |
| Qwen3.6-35B-A3B | top-1 ok (no cos figure recorded) |
| Gemma-4-31B | top-1 ok with **bf16 KV**. With `KV8=1`, expect failure (KL 1.96) |
| Mac minis vs the Studio | identical logits between the minis. W4A8 KL ≤ 0.00005 on normal text |

### 2.2 Speculative decoding matches plain decoding (`spec_test`)

```bash
export D=/path/to/Qwen3.8-27B-DFlash2
./build/spec_test $M $D /tmp/toks.txt 128        # greedy: plain decode, then speculative; must match
./build/spec_test $M $D /tmp/toks.txt 128 1      # B = 1
SPEC_PROF=1 ./build/spec_test $M $D prompt.txt 256 1   # cycle split: drafter context / block / target
```

The two outputs should match token for token, **except at near-ties**. Speculative verification scores several rows
at once through the GEMM kernels, while plain decode runs one row through the matrix-vector kernels; the two sum in a
different order (and on M5-class GPUs verify also uses int8 activations), so when the top two tokens are within a
fraction of a logit the choice can flip. The kernels are deterministic, so a flip **recurs at the same token on every
run**, and turning off `ATTN_IQ` or 8-bit KV does not remove it (those are not the only source). It is not a bug unless
the gap is large. Check it with `tools/tie_check.py`, which prints the reference (mlx_lm) logits of the two competing
tokens at the first differing position: a near-tie is a gap of about 0.5 or less. Expect a few such sequences in any
`spec_test` run on repetitive prompts: the 0.2.0 gate on a Mac mini M6 had 2 of 4 diverge after 69 and 79 identical
tokens, and an external M4 Max on macOS 26 had one. Like vLLM, the engine is not batch-invariant at the bit level. Two traps:

- **`spec_test` on short chat prompts reports about 1.01 tokens/cycle.** That comes from the tool, not the engine.
  Measure acceptance on a long prompt (about 1k+ tokens of code or prose), or on the server.
- **`spec_test` can't catch a broken drafter that only shows up in the server.** The `QMST=1` bug kept the target
  lossless while the drafts turned to junk. Always confirm acceptance in the server: run with `INFERD_DEBUG=1` and read
  the `idN commit K:` lines in the inferd log. On agent text, 16k × 1 should average about 4.5 tokens/cycle and 64k × 1
  about 5.1. If you change the draft block, check the **histogram** of commits too: at block 8 some cycles must commit
  7 and 8. Before the `draft_attn` fix, none ever did.

### 2.3 Speculative sampling

`build/spec_sample_test` (no model, seconds) checks the speculative sampler against the contract by chi-square for
top-k 16 / 20 / 64, top-p with top-k off, tied logits and unlikely drafts, and that rows beyond the 64 candidates go to
the host; `grammar_test` checks the host sampler. Run 400+ samples of a short open prompt at temperature 1 and top-k 16 (the study log used "three random animals"), once with
`SPEC_SAMPLE=0` and once with the default. Compare the frequency tables, which should match within noise. **Light**: it
has been done once.

### 2.4 Vision

There's no script. To repeat the original check:

1. Use a checkpoint that has the vision tower (mlx-community 8-bit or bf16, or lmstudio-community 4-bit). The
   mlx-community 4-bit build returns HTTP 400 for images.
2. Draw a synthetic image, for example a red circle, a blue rectangle and the text "HELLO 42", then send it as an
   `image_url` (a PNG/JPEG/WebP/GIF data URI) with `temperature: 0`.
3. Compare the reply against `mlx_vlm.generate` on the same checkpoint, image and prompt, greedy. It should match word
   for word.
4. Repeat the request. `cached_tokens` should be > 0 and the reply identical. Change one colour: `cached_tokens` should
   drop to about 0 (the image ids are content-keyed) and the reply should name the new colour.

Wanted: a scripted version, plus multi-image, multi-agent and SSD-tier variants (all **Untested**).

### 2.5 SSD tier and restart

```bash
# Reference: a pool large enough to hold every agent
./run.sh 8399 8 2048 16 16 32768
python3 bench/spill_test.py --url http://127.0.0.1:8399 --out /tmp/ref.json --agents 3 --ctx 6000 --rounds 2
# Test: a pool that holds about one agent, plus the SSD tier
SPILL_DIR=/Volumes/External/spill SPILL_GIB=32 ./run.sh 8399 8 2048 <small kv_gib> 16 32768
python3 bench/spill_test.py --url http://127.0.0.1:8399 --out /tmp/run.json --agents 3 --ctx 6000 --rounds 2
```

Expect every follow-up reply to be identical to the reference, `cached_tokens` to cover the context, and the log to
show `spill: restored` lines. Read `KV blocks (… tokens)` in the inferd start-up log to pick a pool that holds about
one 6k context. For the restart variant (`SPILL_PERSIST=1`, SIGTERM, restart, follow up), use `persist_test.sh` (lab script, not published)
as the template: it compares replies against a reference and expects all of them to be identical.

### 2.6 Tool calls and structured output

Tool calls are parsed (`python3 engine/frontend/test_tools.py`, 8 tests, no engine needed). Live checks: run
`examples/tools-web-search` and `examples/tools-shell`, streaming and not, thinking on and off, with multiple calls
per turn and arguments that contain JSON strings. Structured output (`response_format`) is still unsupported.

---

## 3. Smoke tests vs milestone benchmarks

### 3.1 Smoke tests: after every kernel or scheduler change (seconds to minutes)

```bash
cd engine && ./build.sh
# 1. Correctness (§2.1): chunks 1, 5, 24 (and 256, 1536 on `make ... 4 2048` after any build or GEMM change);
#    compare must pass: top-1 everywhere, cos >= 0.999 on seqs 1-3 (typically 0.9995+)
# 2. Decode step times at B = 1 / 16 / 64. ALWAYS run this after touching a shared kernel.
./build/decode_test bench $M 1,16,64 2048
# 3. Verify blocks (rows per sequence = 4, 5, 8) at short and long context
./build/decode_test bench $M 1,8,16 2048 5
./build/decode_test bench $M 1 8192 8
./build/decode_test bench $M 4,8 131072 8                  # long context; add BENCH_SHUFFLE=1 to scatter pages
# 4. Linear layers per step (walks every layer's real weights, so the SLC can't flatter it)
./build/decode_test gemm $M 1,2,5,8,16,32,64
# 5. Prefill and follow-up turns on a long context
./build/decode_test prefill $M 8192 1024,2048
./build/decode_test followup $M 16384 1,16 32,136
# 6. Speculation matches plain decode except near-ties (§2.2)
./build/spec_test $M $D /tmp/toks.txt 128
```

Why step 2 matters: templating `attn_gqa_row` over head shapes left logits and the B = 1 and R = 5 numbers unchanged,
but it cost the 27B **15–23% at 16–64 decode rows** ("Diagnosed 2026-11-01: templating attn_gqa_row…"). Only a B = 16
or 64 bench showed it.

Reference step times on the M5 Ultra (27B 4-bit, 8-bit KV), for orientation: 8-row verify at 8k ≈ 23.5 ms; 4 × 128k
R8 ≈ 61 ms; 8 × 128k R8 ≈ 109 ms; B = 16 / 64 decode at 8k ≈ 35.4 / 91.2 ms. The M5 Pro runs kernels at roughly 1/3 of the
Studio's speed, and the M6 slower still. When the result is a speed number, compare it against **your own baseline build on the
same machine**, not against these.

### 3.2 Server smoke (minutes)

```bash
MODEL=$M DRAFT=$D ./run.sh 8300 64 2048 140 96 262144     # Studio sizing; see engine/README.md for the minis
python3 bench/studio/conc.py --url http://127.0.0.1:8300 --model qwen3.8-27b --label smoke-<commit> \
  --mode followup --ctx 32k --n 1,16 --max-tokens 512 --timeout 7200 --out results/smoke-<commit>-agents.jsonl
```

This is the cell `smoke4.sh` (lab script, not published) ran. At 32k, expect about 150 tok/s at N = 1 and 500+ total at N = 16 on
the M5 Ultra.

### 3.3 Milestone benchmarks (hours; only at milestones)

**Studio scoreboard.** From `milestone.sh` (lab script, not published), run on a **freshly started** server:

```bash
for spec in 16k:1,4,8,16,32 32k:1,4,8,16,32 64k:1,4,8,16,32 128k:1,4,8,16; do
  python3 bench/studio/conc.py --url http://127.0.0.1:8300 --model qwen3.8-27b --label <tag> --mode followup \
    --ctx ${spec%%:*} --n ${spec#*:} --max-tokens 512 --timeout 7200 --out results/<tag>-agents.jsonl
done
```

**Single-stream sweep** (`compare.sh` (lab script, not published)): 2 cold repetitions with a nonce, greedy, 512 tokens out, thinking off.

```bash
python3 bench/studio/speed.py --url http://127.0.0.1:8300 --model qwen3.8-27b --label <tag> \
  --sizes 8k,16k,32k,64k,128k,256k --reps 2 --max-tokens 512 --timeout 3600 --out results/<tag>-sweep.jsonl
```

256k needs `max_ctx` 262144 on `run.sh`. For a head-to-head against another engine, `compare.sh` (lab script, not published) starts each
engine in turn, samples memory every 2 s (`bench/memsample.sh <out.tsv> <pid>`) and writes `<tag>-{sweep,agents}.jsonl`
and `<tag>-mem.tsv`. Its paths are machine-specific, so adapt them before you use it.

**Agent fleets** (`agents64g.sh` (lab script, not published)). These are the cells that matter for small GPUs: N:N, 2 rounds, about
1.2k new tokens per turn.

```bash
PYTHONPATH=bench/studio python3 bench/agents.py --url http://127.0.0.1:8300 --model qwen3.8-27b \
  --label <tag> --ctx 16k --agents 64 --cells 8:8,16:16,32:32,64:64 --rounds 2 --cell-budget 3600 \
  --server-log=$HOME/inference/logs/inferd-8300.log --out results/<tag>-16k.jsonl
```

(`agents.py` imports `speed.py`, which is why `PYTHONPATH` is set.) For a cell `N:K`, K requests are in flight and the
rest of the fleet stays parked in the cache. The older five-cell set is `4:4,8:8,8:4,16:16,16:4` with 3 rounds
(`fleet-inferd2.sh` (lab script, not published)). Server sizing for the minis is in `engine/README.md`. They use `kv_gib 0` (auto), an SSD
tier, and `max_ctx` = context + 8k (24576 for 16k, 40960 for 32k).

**Shared-prefix fleet** (`bench/shared_prefix.py`): N agents fan out on one shared document, then take turns of about
1k tool tokens each.

```bash
python3 bench/shared_prefix.py --url http://127.0.0.1:8300 --model qwen3.8-27b --label <tag> \
  --shared 32k --n 8 --turns 2 --out results/<tag>-shared.jsonl       # also 64k x 16
```

**Other models** (`models_smoke.sh` (lab script, not published)): `speed.py` at 0.5k/4k/32k with 1 rep, then `conc.py` at 32k × 1/8/16.

**Quality.** The coding suite (the internal languages-benchmark) runs through the OpenAI endpoint with 32 workers, a
32k cap, and a 64k re-run of any capped tests (`code.sh` (lab script, not published), `bench/rerun_capped.py`). The 48-test suite (v1)
scored 42.5/48 against Splash's 42.0 ("0.1.1"; rows in `results/code/`). **The 120-test suite (v2) is outstanding.** Its
only run started and stopped after the server came up (`results/studio/code-v2.log`), so no v2 result exists. It needs
a full run.

### 3.4 Benchmarking rules that have already cost wrong numbers

- **Use a fresh server for every milestone.** Long uptime fragments the KV pool and has slowed 128k × 4 by about 12%.
  Run `decode_test bench` with `BENCH_SHUFFLE=1` to see the scattered-page cost.
- **Compare across builds by generated tok/s and latency, not turns/min.** 8-bit attention changes greedy text and its
  length, so the same fleet takes a different number of turns ("Minis on the head-plane build").
- **Single-agent cells vary by ±10% from run to run** (16k × 1: 147–157). Run two repetitions, and quote a median, not
  a best.
- **The prompts in `speed.py`, `conc.py` and `agents.py` carry a nonce.** Don't remove it: identical prompts measure
  the prefix cache, not prefill.
- **Report errors and capped cells.** `agents.py` writes errors as rows and marks cells capped at `--cell-budget`. A
  cell with errors or a cap isn't comparable to a clean one.
- **Laptops throttle.** Judge 1–2% kernel effects on a Mac Studio or Mac mini, not a MacBook.
- **`spec_test` prefills one prompt per step** for a reason. A single step over B prompts sizes the verify tapes for
  B × T tokens (177 GiB at 16 × 6k), and that once looked like a regression.

---

## 4. How to report a result

Open an issue or a PR comment with the following, and attach the raw `.jsonl`. A summary table without the raw rows
can't be checked.

| Field | How to get it |
|---|---|
| Machine | Model and chip, GPU cores (`ioreg -r -c AGXAccelerator -d 1 \| grep gpu-core-count`), memory |
| macOS | `sw_vers` |
| Xcode / Metal toolchain | `xcodebuild -version` |
| Build | `git rev-parse --short HEAD`, plus `git status --porcelain` (say whether the tree was dirty) |
| Model + drafter | Hugging Face repo id **and** snapshot hash for each, and the precision |
| Server arguments | The full `run.sh` line: port, max_running, step_tokens, kv_gib, snapshots, max_ctx |
| Env knobs | Every non-default variable (`env \| grep -E '^(KV8\|SPEC\|DRAFT\|W4A8\|ANE\|SPILL\|NAX\|LANE\|QMST\|ADMIT\|CASCADE)'`) |
| Start-up lines | The `inferd: memory:` and `KV blocks` lines from the inferd log (budget, pool size, slots) |
| Thermal state | `pmset -g therm`. For laptops, say whether you were on power and whether the lid was open. |
| Uptime | Fresh server, or how many cells/hours it had served before this run |
| Other load | Anything else using the GPU or memory (other servers, browsers, the Neural Engine) |
| Raw output | The `--out` `.jsonl` (and `memsample.sh` `.tsv` if memory matters), and `INFERD_DEBUG=1` commit lines when acceptance matters |
| Correctness | The `ref_logits.py compare` lines for any kernel change |

Put result files under `results/<area>/` with a label that names the machine and the build (for example
`results/cmp/<tag>-agents.jsonl`), and add a dated entry to the study log that says what changed, what you measured, and
what you concluded, in the style of the existing entries.
