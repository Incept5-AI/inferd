# inferd engine

A native inference engine for Apple Silicon, built for many concurrent agents on one model: continuous batching,
speculative decoding with a DFlash2 drafter, 8-bit paged KV, a prefix cache that hands one agent's context to the
next, and an SSD tier that parks contexts on disk when memory runs out. C++/Objective-C++ and Metal; no MLX runtime
in the engine (the front end uses MLX only for the vision tower).

Model: **Qwen3.8-27B** (mlx-community 4-bit, 8-bit or bf16) with the **incoai Qwen3.8-27B-DFlash2** drafter.

## Requirements

- An M5-class or newer GPU (the kernels use the Metal 4 tensor units). M4-family GPUs work but are only lightly
  tested (Mac mini M4 Pro, 2026-11-01). Tested on M5 Ultra (Mac Studio), M5 Max
  (MacBook Pro), M5 Pro (Mac mini, 64 GB) and M6 (Mac mini, 32 GB).
- The macOS the prebuilt `build/` was made for (`build/BUILD-INFO`; a macOS 27 build refuses to start on macOS 26 and
  the other way round). On macOS 26 or 27 you can build from source with Xcode (not just the Command Line Tools) and
  its Metal toolchain (`./build.sh`; if it reports "missing Metal Toolchain", run
  `xcodebuild -downloadComponent MetalToolchain` once, about 690 MB). With the Command Line Tools selected
  (`xcode-select -p`), `./build.sh` uses `/Applications/Xcode.app` by itself; `sudo xcode-select -s
  /Applications/Xcode.app/Contents/Developer` makes that permanent.
- Python 3.12 with `fastapi uvicorn transformers` for the OpenAI front end, plus `mlx mlx-vlm pillow numpy` for images
  and `mlx-lm` for the release gate's logits reference, `tools/ref_logits.py` (`requirements.txt`: Pillow 13.0.0,
  which fixes decoder advisories in 12.3.0). `requirements-test.txt` is the host-test set (CI).
- Weights: `mlx-community/Qwen3.8-27B-4bit` (revision 10c35caa includes the vision tower; no MTP head),
  `lmstudio-community` 4-bit or mlx-community 8-bit, and `incoai/Qwen3.8-27B-DFlash2`.

## Run

```bash
./install.sh --model 27b                 # venv ~/.venvs/inferd + requirements; the 27B + drafter into ~/models,
                                         # unless a complete copy is already in ~/models, the HF cache or LM Studio
./install.sh --list                      # what each model name finds, and where (also ./run.sh models)
./run.sh start --model 27b               # sized from the Mac's memory (docs/MACHINES.md); --dry-run prints it
# or the explicit form, unchanged:
python3 -m venv ~/.venvs/inferd && ~/.venvs/inferd/bin/pip install -r requirements.txt
export MODEL=~/models/Qwen3.8-27B-4bit DRAFT=~/models/Qwen3.8-27B-DFlash2 PY=~/.venvs/inferd/bin/python
./run.sh 8300 64 2048 140 192 262144     # start: port, max agents, step tokens, KV GiB, snapshots, max context
curl localhost:8300/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello"}]}'
./run.sh status 8300                     # pids, engine socket, log paths (exit 0 when both processes run, 3 if not)
./run.sh stop 8300                       # stop what run.sh started on 8300; exit 0 when nothing is running
./run.sh clean 8300                      # delete 8300's SSD tier (it persists across restarts); not while running
```

**Stopping:** always `./run.sh stop PORT`. The engine socket is no longer `/tmp/inferd-PORT.sock` (it lives in an
owner-only run directory, below), so stopping by `lsof` on that old path finds nothing and would leave inferd running,
holding its GPU memory. `stop` signals only the processes recorded in its pid files (or, from an older `run.sh`, our
own inferd / front end found on the port or the old `/tmp` socket): SIGTERM, a wait of up to `STOP_WAIT` (60 s;
inferd writes the SSD tier on SIGTERM), then SIGKILL. Anything else on the port is reported and left alone. Starting
on a port first does the same stop. `./test_run.sh` checks stop / status / argument handling with fake processes (no
model needed); `./test_run_start.sh` checks the named start (options, memory-based sizing, `--dry-run`, where a model
is found, install.sh's use of a found copy) and that the positional form, stop and status behave as before.

**Where a named model is found** (`tools/models.zsh`, shared by `install.sh` and `run.sh start`): the first complete
copy of the exact repo id in `$INFERD_MODELS/<dir>` (default `~/models`), then the Hugging Face cache
(`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`; `models--<org>--<name>/snapshots/<rev>`, the
one `refs/main` names, else the newest complete one; links into `blobs/` are followed), then LM Studio
(`$LMSTUDIO_MODELS/<org>/<name>`, default `~/.lmstudio/models`, then `~/.cache/lm-studio/models`). Complete:
`config.json` and every shard in `model.safetensors.index.json` present and non-empty (or `model.safetensors`).
Exact repo ids are preferred; the known equivalent builds per name (`MI_EQUIV`, `MI_DRAFT_EQUIV`) are accepted by default
(since 0.4.0, with a note; `--exact` / `INFERD_EQUIVALENT=0` refuses them); every
candidate's `config.json` must match `MI_FACTS` (architecture, layers, hidden size, quantization bits and group size),
so an 8-bit checkpoint is refused for `27b`. The dry run and the start print the chosen copy and why; skipped copies
are reported as notes. `install.sh --download` forces `hf download` into the models directory. JSON is read with
JavaScript for Automation (`osascript`), so the search needs no Python.

**API:** a drop-in for vLLM's OpenAI-compatible server ([docs/API-COMPAT.md](../docs/API-COMPAT.md): every vLLM endpoint
and request field, and what inferd does with each). `/v1/chat/completions` and `/v1/completions` (streaming and not,
`n` choices, `logprobs` / `top_logprobs`, penalties, `min_p`, `logit_bias`, `min_tokens`, `stop_token_ids`, tools with
every `tool_choice`, `response_format` and vLLM's `guided_json` / `guided_choice` / `guided_regex` /
`structured_outputs`, `usage.prompt_tokens_details.cached_tokens`), `/v1/models`, `/tokenize`, `/detokenize`,
`/metrics` (Prometheus, `inferd:` names), `/version`, `/ping`, `/health` (503 when the engine process has gone; engine
errors come back as HTTP errors, or as an SSE `error` event once a stream has started). Embeddings, scoring, audio and
LoRA endpoints answer 501. Penalties, `min_p`, `logit_bias` and `min_tokens` decode one token per step on the host
sampler (no speculation) for the requests that use them; `logprobs` stays speculative.
`engine/frontend/test_vllm_compat.py --url http://127.0.0.1:8300` checks a running server against the whole list.

**Per-agent policy** (optional request fields, top level or under `"inferd": {...}`):

- `max_context`: this agent's cap on prompt + output tokens. A longer prompt gets HTTP 400 `context_length_exceeded`;
  generation stops at the cap with `finish_reason: "length"`. KV memory is allocated as each agent's context grows,
  so an 8k agent never holds 128k of memory; the cap stops an agent from growing past its budget.
- `priority` (integer, -10 to 10, default 0): when the KV pool runs dry, the lowest-priority sequence not yet
  scheduled in that step is preempted first (newest among equals). Give a long-context coding agent a higher priority
  than short search agents. It does not reorder admission.

Both act only when a request is admitted or the pool runs out, never per step: decode step times at 1/16/64
sequences were identical before and after (1 Nov 2026).

**Stop strings:** `stop` (a string or a list of strings) ends generation at the first occurrence in the answer text
(content and tool-call markup; with thinking on, not the reasoning), trims it (`include_stop_str_in_output: true` keeps
it) and reports `finish_reason: "stop"`, streaming or not.

**Images:** send them inline (`data:` URLs). Fetching `http(s)` URLs and reading local paths are off by default
(`INFERD_IMAGE_URLS=1`, `INFERD_IMAGE_FILES=1`; see the trust model below before enabling either).
Logs go to `~/inference/logs/` (`INFERD_LOGS`). `run.sh` requires `MODEL`, checks its arguments and the build, uses
`PY` (default `python3`), and on a restart stops only the processes it started on that port (pid files; SIGTERM, up
to `STOP_WAIT`=60 s, then SIGKILL; `run.sh stop PORT` does that alone).

### Security and limits

**Trust model: one trusted local user.** Anyone who can reach the HTTP port, or the engine's Unix socket, can run
generations and create, offload or delete *any* context or session: context and session names are not ownership,
and there are no per-user namespaces, quotas or authorization. The front end binds `127.0.0.1` by default, which
still admits every local process. Do not present inferd as a multi-tenant service.

- `INFERD_API_KEY=<secret>`: every endpoint (`/health` included) requires `Authorization: Bearer <secret>`. It is
  **required** (the front end refuses to start without one of 16+ characters) when `--host` (`INFERD_HOST=` for `run.sh`)
  is not a loopback address. One key is one tenant: it keeps other hosts out, it does not isolate agents.
- The engine socket lives in an owner-only directory (`$TMPDIR/inferd-<uid>`, mode 700; `INFERD_RUN_DIR`), the
  socket itself is mode 600, and `run.sh` refuses a stale path that is not a socket owned by you. The front end
  refuses to connect to a socket that is not yours.
- `INFERD_IMAGE_URLS=1` lets any caller make the server fetch any `http(s)` URL, internal hosts and redirects
  included; `INFERD_IMAGE_FILES=1` lets any caller read any image file your user can read. Enable them only when
  every caller is trusted with that access.

**Budgets** (environment; defaults leave 64 agents with 64k contexts untouched). Overload is HTTP 503 with
`Retry-After` (code `server_overloaded`); oversized input is 413 (`request_too_large`); malformed input is 400 with
an OpenAI-style error (`max_tokens: 0`, non-finite numbers, wrong types, a session id over 128 characters).

| Variable | Default | Bounds |
|---|---|---|
| `INFERD_MAX_BODY_MB` | 100 | one request body (inline images included) |
| `INFERD_MAX_INFLIGHT` | 256 | requests in progress, open streams included (`/health` exempt) |
| `INFERD_MAX_INFLIGHT_MB` | 1024 | their bodies together |
| `INFERD_MAX_PREPROCESS` / `INFERD_PREPROCESS_WAIT` | 16 / 60 s | concurrent tokenize/render/vision jobs; longest wait for one |
| `INFERD_MAX_MESSAGES` / `INFERD_MAX_PROMPT_CHARS` | 10000 / 8000000 | messages and text characters per request |
| `INFERD_MAX_PROMPT_TOKENS` / `INFERD_MAX_OUTPUT_TOKENS` | 1048576 / 262144 | rendered prompt; `max_tokens` |
| `INFERD_EVENT_QUEUE` | 8192 | engine events buffered for a client that stops reading its stream; beyond it the request is cancelled with an SSE error |
| `INFERD_TOKCACHE_MB` | 256 | tokenizer cache, measured in retained bytes (strings, 4-byte token arrays, entries) |
| `INFERD_MAX_CONTEXTS` | 256 | saved contexts (a create reserves its slot before it computes: concurrent creates cannot overshoot) |
| `INFERD_MAX_CONTEXT_MB` | 1024 | estimated bytes all saved contexts retain in the front end (messages, tools, token ids) |
| `INFERD_EVENT_TIMEOUT` | 1800 s (0: off) | a request with no engine event for this long is abandoned (HTTP 504 / SSE error) and cancelled |
| `INFERD_IMAGE_DECODE` / `INFERD_IMAGE_DECODE_TIMEOUT` | `subprocess` / 10 s | images (PNG, JPEG, WebP, GIF only) decode in a child process killed at the deadline; `inline` decodes in the front end |
| `INFERD_MAX_SESSIONS` / `INFERD_SESSION_TTL` | 4096 / 0 (off) | session ids kept; the least recently used idle ones, or those idle past the TTL, are dropped from the engine |
| `INFERD_GEN_SNAP` | `auto` | decode snapshots (state saved while an answer is generated, thinking off): `auto` takes them only when the chat template renders an earlier answer exactly as generated (Qwen3.8 does; Qwen3.6 drops the empty think block, so its follow-ups resume at the previous prompt's end); `1` / `0` force |
| `INFERD_RETRY_AFTER` | 2 s | the `Retry-After` value | Greedy requests are always speculated (the drafts are verified exactly). Sampled ones (temperature > 0) use
speculative sampling where it is exact: `top_k` 1–64 (the GPU sampler sees each verify row's top 64 logits), or `top_k` off with `top_p` < 1 (the nucleus is placed against the full softmax; a row whose nucleus does not fit in the 64 is sampled by the host, `inferd.sampling.host_sampled_rows`). `top_k` above 64, or off with `top_p` 1, decodes one token per step on the host sampler (the full distribution); `"inferd": {"spec_sampling": "truncate"}` opts into truncation to the 64 candidates instead. Omitted `top_k` / `top_p` come from the checkpoint's `generation_config.json` (Qwen3.8: 20 / 0.95), so the default sampled request is speculated. The response's `inferd.sampling` reports the effective settings (src/sampling.h, Codex review 2026-11-04 R03). `SPEC_SAMPLE=0` turns speculative sampling off.

### Sizing by machine

| Machine | `run.sh` arguments | SSD tier | Notes |
|---|---|---|---|
| M5 Ultra, 256 GB | `8300 64 2048 140 192 262144` | optional | 4.5M tokens of KV: 32 agents × 128k stay resident. 192 snapshots (5 Nov): with 96, a 64-agent fleet re-read contexts (8k fleet, 64 in flight: 59.6 → 95.3 turns/min with 192; +16 GiB of state) |
| M5 Max / M5 Pro, 64 GB | `8300 16 1024 24 30 147456` | `SPILL_DIR=… SPILL_GIB=96` | ~770k tokens resident |
| M6, 32 GB | `8300 16 1024 0 8 32768` | `SPILL_DIR=… SPILL_GIB=64` | auto KV: 2.9 GiB (~92k tokens) beside the drafter; agents beyond that queue and swap via SSD. `KV4=1` doubles it (182k: eight 16k agents resident) at a small accuracy cost |

**The drafter on small machines:** keep it on. On the 32 GB M6 (27B 4-bit, 2 Nov 2026) it gives 39 vs 9.5 tok/s
for one stream and 92 vs 41 / 89 vs 64 total tok/s for 8 / 16 agents with 8k contexts, although it leaves 2.4 GiB of
KV instead of 5.2; without it, 16 agents' worst time to first token is lower (13 s vs 54 s) because more contexts stay
resident. It costs ~2.8 GiB there (weights 1.1, context rings 42 MiB per state slot, scratch and verify tapes ~0.8), so
on 32 GB keep `snapshots` small and `step tokens` at 1024. On GPUs without neural accelerators (M3/M4) a verify row
costs ~3× a decode row; up to 0.2.0 the drafter lost at 16 agents there (M4 Pro 4k: 65 vs 73 tok/s). Since 9dd55ad (8-bit
context rings, verify scratch sized for the real rows) it wins: an external M4 Pro 48 GB measured 69.0 vs 63.2 tok/s at 16
agents x 4k (verify step at block 4: 732 vs 1,331 ms before), with a slower median first token (14 s vs 6.8 s without it).
Leave it on unless first-token latency matters more than throughput.

Snapshots (`run.sh`'s fifth argument): about 2 × the agent count. With thinking off, a turn is snapshotted while it
decodes (when a step ends on a block boundary, at most every 64 tokens, and at its end when that falls on one), so an
agent's next turn resumes near the end of its previous answer; snapshots made redundant by a newer one on the same
chain are evicted first. Pure-attention models (Muse Glimmer, Gemma-4) need no snapshots for their cached prefixes.

KV is 8-bit by default (`KV8=0` for bf16, which halves the pool). `KV4=1` stores it in 4 bits (Qwen3.5-family models
with 256-dim heads: the 27B, 9B, 35B-A3B), twice the tokens again: K and V are Hadamard-rotated per head before
quantizing (the queries rotate alike; the output is rotated back), one scale per token and head. It is not lossless:
on the 27B at 8k-32k it adds ~0.006 nats of KL against mlx_lm (8-bit KV: ~0.0014) and moves top-1 agreement from
98-99% to 96-98%, so it stays off unless the pool is what limits you (a 32 GB Mac with 16k agents: on the M6, `KV4=1` takes 8 agents at 16k from 66.6 to 92.6 tok/s and their worst
first token from 35 s to 4 s, all resident; 4 November 2026). A request whose context does not fit waits; when
the pool runs dry mid-turn the newest sequence is preempted and resumes from its cached prefix (RAM or SSD).

## What is different here

- **Agent-first scheduling.** Every admitted agent decodes in the same step (no lane cap), so time to first token
  stays at 1–5 s with 32 agents. Admission is by KV budget, not count, and preemption parks a context in the prefix
  cache or on SSD instead of failing.
- **Context hand-off.** A block trie with recurrent-state snapshots: an agent's follow-up turn, a fork, or a second
  agent on the same document resumes from the cached prefix (98–99% cache hits in fleet tests). Evicted chains are
  written to the SSD tier content-addressed and restored at ~2 GB/s on a later match.
- **Self-tuning speculation.** The draft block (4, 5 or 8) is chosen per load regime from measured step times and
  acceptance, so 1 agent and 32 agents each get the block that maximises their throughput.
- **Integer verify path.** W4A8 linears and int8 query × int8 KV attention on the tensor units, head-plane KV pages.

## Numbers (31 October 2026, M5 Ultra, Qwen3.8-27B 4-bit, agents with 512-token turns)

Total tok/s (Splash 1.5.0 on the same Mac in brackets):

| Context | 1 agent | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 16k | 142 (172) | 366 (354) | 485 (369) | 563 (369) | 606 |
| 32k | 150 (148) | 405 (321) | 465 (391) | 518 (354) | 549 |
| 64k | 146 (147) | 323 (292) | 363 (311) | 390 (300) | 430 (311) |
| 128k | 146 (149) | 270 (265) | 291 (268) | 292 (266) | |
| 256k | 111 (132) | 190 (187) | 187 | | |

The full comparison is in [docs/RESULTS.md](../docs/RESULTS.md).

## Known limits

- One model family (Qwen3.8 / Qwen3.5-style hybrid attention + Gated DeltaNet). The 4-bit mlx-community build has no
  MTP head (its current revision does have the vision tower).
- One agent at 16k (142 vs 172) and at 256k (111 vs 132) trails Splash; prefill is ~10% behind it.
- The SSD tier is a cache: it is cleared when the model, drafter or KV geometry changes. Otherwise it **persists across
  `run.sh stop` and restarts** (that is the point: agents' contexts come back from it), so it keeps its disk space
  (up to `SPILL_GIB`, 64–96 GiB under `run.sh start`) in `~/Library/Caches/inferd/spill-<port>` or `SPILL_DIR`.
  `./run.sh clean [port]` deletes that port's tier (`<dir>/inferd-spill` and its lock, in the directory the last start
  used) and refuses while a server runs on the port.

## Tuning knobs (environment)

`KV8` (default 1), `KV4` (4-bit KV, opt-in, 0: twice the tokens of 8-bit KV at a measured accuracy cost, see above),
`RING8` (8-bit drafter context rings: drafts only, the output stays exact; 1, 0 = bf16), `ATTN_IQ` (int8 queries, 1; 0 under `KV4`), `SPEC_AUTO` (self-tuning block, 1), `DRAFT_BLOCK` (widest block, 16; the drafter's trained block is 8),
`SPEC_BLOCKS` (fixed block by load, e.g. `8:4,16:3`), `ADMIT_RESERVE` (generated tokens budgeted per admitted turn,
512), `SPILL_DIR`/`SPILL_GIB`, `W4A8` (1), `INFERD_DEBUG` (per-step logs and the tuner table), `SPILL_DEBUG`,
`SPEC_ROWS` (verify rows to size the drafter and GDN tapes for; default from the block policy, 0 = the old
max_running x 16 sizing), `SNAP_GEN` (decode snapshot interval, 128; 0 off; taken only for requests the front end marks, see `INFERD_GEN_SNAP`), `SNAP_GEN_FORCE` (N > 0: with N or more decoders, shorten blocks to land
every `SNAP_GEN`; 0, default: cost 4-6%), `SNAP_REDUNDANT` (evict snapshots superseded on their chain first, 1), `PREFIX_ANY` (pure-attention models resume at any cached block without a snapshot slot, 1),
`TREE` (token-tree verification for one greedy decoder: nodes per tree, 16; 0 = chains), `TREE_SEQS` (decoders that
get trees, 1), `TREE_GPU` (build the trees on the GPU, 0), `TREE_TEMP` (1.5), `TREE_KIDS` (4).
`QA8` (int8-activation prefill GEMMs on the checkpoint's packed 4-bit weights, no copy; adapted from oMLX's
`oq_a8_nax`, see NOTICE; -1 auto = on with neural accelerators except Flash-Next, and an auto `PF8` then stays off;
0/1 force), `QA8_MIN` (rows, 256), `QA8_TILE` (tile index; default 2 = 128x64 on 30+ GPU cores, 1 = 64x64 below), `QA8_RF` (grid order, 0 = column tiles fastest).
`MOE_QA8` (Qwen3.6-35B-A3B: routed experts on int8 activations in prefill and verify steps of 16+ rows, 1),
`MOE_ROUTER_GEMM` (rows from which the 35B's router is a GEMM, 64; 0 off), `DRAFT_ATTN_MMA` (drafter attention on
simdgroup matrices for blocks of up to 8 rows, 1).
`WARMUP` (inferd runs two throwaway requests before it listens, so the first real request does not pay for the
GPU's first use of the weights; 1, 0 off), `PSO_LOG` (log every pipeline creation and its time).
See `CLAUDE.md` in the project root for the rest.

## Changes

- **Unreleased (0.4.1)** — first-run fixes from an external tester's report: `build.sh` uses `/Applications/Xcode.app`
  by itself when the Command Line Tools are the active developer directory (they have no Metal compiler) and otherwise
  says exactly what to install; `install.sh` works on a venv made by uv (no pip: `uv pip`, else `ensurepip`);
  `run.sh clean [port]` deletes a port's SSD tier, which persists across restarts by design; `examples/run_all.sh`
  defaults its machine name to the host name; `agents-at-once` writes its results under `$TMPDIR`, not the checkout.
  macOS 26: the source build and the 0.4.0 macOS 26 prebuilt passed the full release gate on a 128 GB Mac on macOS 26.6.

- **0.4.0 (2026-11-05)** — vLLM-compatible API: `/v1/completions`, logprobs (from the verify rows, speculation kept),
  presence/frequency/repetition penalties, `min_p`, `logit_bias`, `min_tokens`, `n`, guided JSON/choice/regex and
  `structured_outputs`, every `tool_choice`, `/tokenize`, `/detokenize`, `/version`, Prometheus `/metrics`; unsupported
  endpoints answer a clean 501 ([docs/API-COMPAT.md](../docs/API-COMPAT.md)). Qwen3.6-35B-A3B fleets: int8 experts on the
  QA8 path, the router as a matmul from 64 rows, drafter attention on simdgroup matrices (M5 Pro fleet 60–71 → 81–93
  turns/min), and decode snapshots off when a chat template drops earlier reasoning (Studio 16k x 32: 269 → 1,079 tok/s).
  M4 / M4 Pro: the Neural Engine FFN split on by default (8k prompt 64.2 → 42.8 s). `install.sh` / `run.sh start` find
  models in `~/models`, the Hugging Face cache and LM Studio, and accept known equivalent builds (with a note; `--exact`
  for exact repo ids). `run.sh start` sizing: the Mac Studio gets 192 snapshots (27B fleet at 64 in flight 59.6 → 95.3
  turns/min), Flash-Next 64 running / 128 snapshots (64 agents at 16k: 209–287 → 1,289–1,323 tok/s). A macOS 26 build
  made on macOS 26 itself. See [docs/RESULTS.md](../docs/RESULTS.md) (t)–(y).

- **0.3.0 (2026-11-05)** — getting started in four steps: a self-contained release tarball, `install.sh` (venv,
  requirements, `--model` downloads) and `run.sh start --model 27b|27b-8bit|flash-next|<dir>` sized from the Mac's
  memory. **QA8**: int8 prefill read straight from the packed 4-bit weights on GPUs with neural accelerators (no weight
  copy; replaces PF8 there), which frees memory for KV: 32 agents x 16k on the Mac Studio 711–727 tok/s (0.2.0:
  613–621). 8-bit drafter context rings (RING8, default) and opt-in 4-bit KV (`KV4=1`). One sampling contract on every
  path: exact speculative sampling up to top_k 64, defaults from the model's generation_config (Qwen3.8: top_k 20,
  top_p 0.95), `"spec_sampling": "truncate"` opt-in. Qwen3.6-35B-A3B: group size read per checkpoint (0.2.0 misloaded
  Qwen's official group-128 build), the DFlash2 drafter on MoE targets and an 8-row MoE verify path. Release gate:
  chunks 1/24/256/1536 on 4 x 2,048 tokens with cos/KL enforced, and speculative divergences judged by their
  reference logit gap. Fixes from the 4 November code and security review (R01–R08) and two external test reports.
  See [docs/RESULTS.md](../docs/RESULTS.md) (r)–(t) and the `docs/CHANGES-*` files.

- **0.2.0 (2026-11-03)** — first public-ready release: the agent-contexts API (`/v1/contexts`, sessions, `cache: drop`,
  per-agent `effort` / `max_context` / `priority`, `GET /v1/cache`), int8 prefill (PF8, automatic), tree verification
  for one agent, Flash-Next with QSA sparse attention, MTP and copy drafts, prefill budget beside decoders, admission
  lookahead, startup warm-up, strict JSON (constrained decoding), hardened front end and loader, SSD tier with
  checksums. See [docs/FEATURES.md](../docs/FEATURES.md).

- **0.1.1 (2026-10-31)** — speculative sampling for temperature > 0 (2.5–3.9× faster sampled decode; the 48-test coding
  benchmark at 32 in flight runs in 4–5 min against 9–14).

- **0.1.0 (2026-10-31)** — first test release. KV-budget admission and SSD-backed preemption (fixes a deadlock
  when agents' contexts exceed the KV pool, found on the 32 GB M6), 8-bit KV by default, reworked verify attention
  (4-lane softmax, int8 queries, head-plane KV), self-tuning draft block, KV blocks allocated in aligned groups of 4
  (a long-running server no longer fragments: 128k × 4 241 → 270 tok/s), two attention threadgroups per GPU core for
  one long sequence (256k × 1 95 → 111).
