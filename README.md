# inferd

**A native inference engine for Apple Silicon, built for many AI agents at once.** C++, Objective-C++ and Metal,
written from scratch in October–November 2026, with no MLX runtime in the token path. One model, many concurrent
agents with long contexts: continuous batching, speculative decoding for every agent, an 8-bit paged KV cache, a
prefix cache that hands one agent's context to the next, an SSD tier for contexts that do not fit in memory, and an
API that lets the agent framework decide what the cache keeps.

inferd is young, experimental software. It leads where agents pile up on one Mac; other engines are still faster for a
single stream, and it was built on the shoulders of the projects listed under [Acknowledgements](#acknowledgements).
Primary model: **Qwen3.8-27B** (4-bit) with the **incoai Qwen3.8-27B-DFlash2** drafter.

**Website: [inferd.incept5.ai](https://inferd.incept5.ai)**: what it is in plain language, the measurements as charts,
and getting started.

<p align="center"><img src="docs/img/overview-fleet.svg" alt="Total tok/s with 1-128 agents on 16k contexts on one Mac Studio: Qwen3.8-Flash-Next and Qwen3.8-27B" width="720"></p>

## Getting started

Up and running in four steps (click a step to open it). You need a Mac with Apple Silicon on macOS 27 or 26, and Python 3.12; Qwen3.8-27B with
its drafter runs from 32 GB. The same steps, with a client example in 14 languages, are on
**[inferd.incept5.ai](https://inferd.incept5.ai/#start)**.

<details><summary><b>1. Download and check</b>: the file for your macOS (27 or 26) from the v0.4.2 release</summary>

Download [inferd-0.4.2-macos27-arm64.tar.gz](https://github.com/Incept5-AI/inferd/releases/download/v0.4.2/inferd-0.4.2-macos27-arm64.tar.gz)
and its [.sha256](https://github.com/Incept5-AI/inferd/releases/download/v0.4.2/inferd-0.4.2-macos27-arm64.tar.gz.sha256)
from the [v0.4.2 release](https://github.com/Incept5-AI/inferd/releases/tag/v0.4.2). No Xcode needed.

```bash
cd ~/Downloads && shasum -a 256 -c inferd-0.4.2-macos27-arm64.tar.gz.sha256   # prints OK
tar xzf inferd-0.4.2-macos27-arm64.tar.gz && cd inferd-0.4.2
```

On **macOS 26** there is no 0.4.2 prebuilt yet: build it yourself (below), or use
[inferd-0.4.1-macos26-arm64.tar.gz](https://github.com/Incept5-AI/inferd/releases/download/v0.4.1/inferd-0.4.1-macos26-arm64.tar.gz)
(its [.sha256](https://github.com/Incept5-AI/inferd/releases/download/v0.4.1/inferd-0.4.1-macos26-arm64.tar.gz.sha256)), and on
an **M4 or M4 Pro** start it with `ANE_UNITS=0` (see the v0.4.2 release notes). During the gate
macOS may show a Python crash report: `test_image_decode` aborts a helper process on purpose, and it is not a failure. To build it yourself, then carry on from step 2
in `engine/`. That needs Xcode, not just the Command Line Tools, with its Metal toolchain (once:
`xcodebuild -downloadComponent MetalToolchain`); `./build.sh` uses `/Applications/Xcode.app` by itself when the
Command Line Tools are the active developer directory (see [Building from source](#building-from-source)):

```bash
git clone https://github.com/Incept5-AI/inferd && cd inferd && git checkout v0.4.2 && cd engine
./build.sh                                   # under a minute
```

</details>

<details><summary><b>2. Install</b>: <code>./install.sh --model 27b</code></summary>

One command sets up the small Python front end and downloads Qwen3.8-27B with its draft model into
`~/models` (about 17 GB). Safe to run again. Already have the model in LM Studio or the Hugging Face cache?
`install.sh` finds it and skips the download.

```bash
./install.sh --model 27b
```

</details>

<details><summary><b>3. Start it</b>: <code>./run.sh start --model 27b</code></summary>

The engine and its OpenAI-compatible server start on port 8300, sized for your Mac's memory.

```bash
./run.sh start --model 27b
```

</details>

<details><summary><b>4. Talk to it</b>: any OpenAI client at <code>http://localhost:8300/v1</code></summary>

Talk to it from anything that speaks the OpenAI API (agent frameworks connect by changing the base URL to
`http://localhost:8300/v1`):

```bash
curl localhost:8300/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello"}]}'
```

```python
from openai import OpenAI                        # pip install openai
client = OpenAI(base_url="http://localhost:8300/v1", api_key="unused")
reply = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Hello"}])
print(reply.choices[0].message.content)
```

</details>

<details><summary><b>More on these steps</b>: other models, options, checks, macOS versions</summary>

- `./run.sh status` and `./run.sh stop` manage the server; `./run.sh --help` lists everything. `--dry-run` prints the
  command lines it resolves; `--agents`, `--context`, `--kv` and `--port` override the sizing
  ([docs/MACHINES.md](docs/MACHINES.md)). Below 256 GB it also turns on the SSD tier, in
  `~/Library/Caches/inferd/spill-<port>` (up to 64–96 GiB; `--spill off` to disable). The tier **stays on disk after
  `./run.sh stop`**, on purpose: agents' contexts come back from it after a restart. `./run.sh clean [port]` deletes it
  (only that port's tier, and only while no server runs there).
- Other models: `--model 27b-8bit` (64 GB and up), `--model flash-next` (Mac Studio, 256 GB), or
  `--model <checkpoint dir>` with `--draft <drafter dir>`. `install.sh --model 27b` downloads
  `mlx-community/Qwen3.8-27B-4bit` and `incoai/Qwen3.8-27B-DFlash2`.
- **Models you already have.** `install.sh --model` and `run.sh start --model` look for each repo, in order, in
  `~/models` (`INFERD_MODELS`), the Hugging Face cache (`HF_HUB_CACHE`, else `HF_HOME/hub`, else
  `~/.cache/huggingface/hub`: the snapshot `refs/main` points to, else the newest complete one) and LM Studio
  (`LMSTUDIO_MODELS`, else `~/.lmstudio/models`, then `~/.cache/lm-studio/models`), and use the first complete copy
  where it is: no copy, no download. Complete means `config.json` plus every shard the index lists, non-empty. Same
  weights only: the exact repo id is preferred, and the known equivalent builds in `tools/models.zsh` are used too, with
  a note (for `27b`: `lmstudio-community/` and `Qwen/Qwen3.8-27B-MLX-4bit`; for the
  drafter: `z-lab/Qwen3.8-27B-DFlash2`, byte-identical); `--exact` (on both scripts) takes only the exact repo ids, as the
  benchmarks do. Every candidate's `config.json` must match the
  architecture, size, bits and group size, so an 8-bit build never stands in for `27b`. Both scripts say which copy
  they chose and why, and what they skipped; `./install.sh --list` (or `./run.sh models`) shows what each name finds,
  and `./install.sh --download` downloads into `~/models` anyway.
- The vLLM-style API (completions, logprobs, guided output, `/metrics`, …): [docs/API-COMPAT.md](docs/API-COMPAT.md).
- Optional check of the build on your Mac, a few minutes with no other model running:
  `PY=~/.venvs/inferd/bin/python tools/release_gate.sh ~/models/Qwen3.8-27B-4bit ~/models/Qwen3.8-27B-DFlash2`.
- **macOS versions.** The macOS 27 file is tested on Mac Studio M5 Ultra, Mac mini M6, M5 Pro and M4 Pro, and MacBook Pro
  M5 Max, M4 Max and M3 Max; it refuses to start on macOS 26 (`install.sh` and `run.sh` check). The macOS 26 file was
  built on macOS 26 itself. On macOS 26 both it and a source build passed the full release gate on one Mac so far (an
  external tester's 128 GB Mac on macOS 26.6, 5 Nov 2026: host, front-end and sampler tests, logits vs mlx_lm at
  chunks 1/24/256/1,536, speculative = plain; [docs/MACHINES.md](docs/MACHINES.md)). Never run a build made for macOS 26
  on macOS 27: its kernels compute wrong logits there at prefill chunks of 256+ rows.
- Release 0.2.0's files hold only `build/`: unpack one into `engine/` of a `v0.2.0` checkout instead.

</details>

## Where it stands (5 November 2026, release 0.4.1; the engine is 0.4.0's)

**64 agents at once on one Mac Studio: 849 tok/s on Qwen3.8-27B, 1,331–1,346 tok/s on Qwen3.8-Flash-Next**, every agent
on its own 16k context; the slowest agent's first token comes within 2 s on Flash-Next and 3.7 s on the 27B. And on a Mac mini, a fleet of agents doing real
work keeps going at 13–19 completed turns a minute on a 27-billion-parameter model.

Every number below is measured: same weights and drafter for every engine, same machine, same harness. Methodology,
every table and the raw file behind each number: [docs/RESULTS.md](docs/RESULTS.md).

### On a Mac mini

| Mac mini | What it does | inferd |
|---|---|---:|
| M6, 32 GB | Qwen3.8-27B + drafter, an agent fleet on 8k contexts (64-agent pool), completed turns per minute | **13.3–14.0** |
| M6, 32 GB | Qwen3.5-9B, 16 agents taking turns on 16k contexts, total tok/s | **164** (8-bit: 117) |
| M6, 32 GB | 27B agents launched from a saved context / resumed after other work: first token | **2.3 s / 0.4 s** |
| M5 Pro, 64 GB | Qwen3.8-27B agent fleet, turns per minute, on 16k / 32k contexts | **18.2–19.0 / 14.4–15.7** |
| M5 Pro, 64 GB | Qwen3.6-35B-A3B (MoE) agent fleet, turns per minute | **81–93** |
| M5 Pro, 64 GB | 35B, 32 agents taking turns on 4k / 16k contexts, total tok/s | **610 / 467** |
| M5 Pro, 64 GB | 27B on 32k contexts, 16 agents: the slowest agent's first token | **5.8–7.5 s** |
| M5 Pro, 64 GB | agents resumed after unrelated load (35B / 27B): first token | **0.14 s / 0.25 s** |
| M4 Pro, 64 GB | 27B, one agent, 8k prompt: first token (with the opt-in Neural Engine split, `ANE_UNITS=26`) | **42.8 s** (0.3.0: 64.2 s) |

<details><summary><b>How the minis compare with other engines</b> (Splash 1.5, TensorFold 1.2.2, mlx_lm)</summary>

| Mac mini | Workload | inferd | Other engine |
|---|---|---:|---:|
| M6, 32 GB | 27B agent fleet on 8k, turns per minute (back to back) | **13.3–13.5** | Splash 1.5: 8.8–8.9; TensorFold 1.2.2 cannot load the 27B on 32 GB |
| M6, 32 GB | 9B, 16 agents on 16k, total tok/s | **164** (8-bit 117) | TensorFold 1.2.2: 23 (8-bit 8.7) |
| M6, 32 GB | 27B, 8 / 16 agents on 16k, total tok/s (4-bit KV, `KV4=1`; main, 5 Nov) | **95** / 85 | Splash 1.5: 86 / 89 |
| M5 Pro, 64 GB | 27B agent fleet, 16k / 32k, turns per minute (back to back) | **18.2–19.0 / 14.4–15.7** | Splash 1.5: 11.4–12.0 / 10.2–10.4 |
| M5 Pro, 64 GB | 35B agent fleet, turns per minute (back to back) | **81–93** | Splash 1.5: 78–79 |
| M5 Pro, 64 GB | 9B agent fleet, 32 agents, turns per minute (4-bit / 8-bit) | **57.4 / 46.9** | TensorFold 1.2.2: 7.7 / 3.0 |
| M5 Pro, 64 GB | 16 forked agents relaunched / 8 resumed after unrelated load: first token | **2.1 s / 0.25 s** | TensorFold 1.2.2 (disk cache): 195 s / 15.4 s |
| M4 Pro, 64 GB | 27B, one agent, 8k prompt: first token | **42.8 s** | mlx_lm: ~63 s |

Where others lead on the minis: Splash is faster for one agent and for 4–8 agents taking turns on the M5 Pro and M6
(e.g. the 35B at 16k with 4 agents, 315 vs 237 tok/s on the M5 Pro), and for the 35B on the 32 GB M6 at 16k. Splash reads
one cold prompt sooner on the M5 Pro and M6. On M1–M4 GPUs leave the drafter off for the 35B from 4 agents
(`--draft none`).

</details>

### On a Mac Studio (M5 Ultra, 256 GB)

| What it does | inferd |
|---|---:|
| Qwen3.8-27B, 64 agents taking turns on 16k contexts, total tok/s, slowest first token (32 agents: 722 tok/s) | **849**, 3.4–3.7 s |
| Qwen3.8-Flash-Next, 64 agents on 16k contexts, total tok/s, slowest first token (v0.4.0, 3 trials) | **1,331–1,346**, 1.8–1.9 s |
| Qwen3.6-35B-A3B (MoE), 32 agents on 16k contexts, total tok/s, slowest first token | **1,079**, 1.7 s |
| Qwen3.8-27B agent fleet, 64 in flight, completed turns per minute | **95.3** |
| 32 agents on 64k contexts: the last agent's first token | **3.3–3.4 s** |
| One agent saves a 33.5k-token context, 31 more start from it: median first token / all 31 done | **0.72 s / 15.3 s** |
| 32 agents forked from a 25k context, relaunched after 1.4M tokens of unrelated work: first token | **1.03 s** |

Check it on your own Mac: [examples/agents-at-once](examples/agents-at-once/) runs 64 (or 100, or 128) agents at once in
about 5 minutes, checks every agent's answer, and with any watt meter reports tokens per joule. On our Studio: 64 / 100 / 128
agents, every answer right, 1,326 / 1,457 / 1,487 tok/s at about 300 W at the wall, 4.4–5.1 tokens per joule.

<details><summary><b>How the Studio compares with other engines</b> (Splash 1.5, TensorFold 1.2.2)</summary>

| Workload | inferd | Other engine |
|---|---:|---:|
| 27B, 32 agents on 16k contexts, total tok/s | **722** | Splash 1.5: levels off at ~369 |
| 27B agent fleet on 8k, 8 / 32 / 64 in flight, turns per minute | **80 / 89 / 95** | Splash 1.5: 49 / 50 / 50 (32 of 128 turns refused at 64) |
| 35B, 32 agents on 16k, total tok/s | **1,079** | Splash 1.5: 598 |
| 32 agents on 64k: the last agent's first token | **3.3–3.4 s** | Splash 1.5: 48.9 s |
| 8 Qwen3.8-27B agents start at once (14k prompts, 6,400 tokens out): all done | **125.6 s** | TensorFold 1.2.2: 177.6 s |
| 16 agents arriving at random, 2k–16k prompts: all done | **78.7 s** | TensorFold 1.2.2: 116.8 s |

Where others lead on the Studio: one agent on its own (Splash, TensorFold and mlx-serve decode a single stream faster),
and Flash-Next with 1–4 agents starting together (TensorFold 1.2.2 finishes sooner; level at 5, 105.9 vs 104.6 s; with 16 arriving at random the two are
level on total time and TensorFold's p95 first token is lower, 9.2 vs 13.7 s). The MacBook Pro M3 Max (external tester):
16 agents at 32k, the last agent's first token 28.4 s vs Splash 1.5's 113–115 s.

</details>

Quality: logits match the mlx_lm reference (top-1, cosine ≥ 0.9997); [HLE Lite](https://github.com/Incept5-AI/hle-lite) (173 items, effort medium) 32.4% for the
27B 4-bit, 34.1% for 8-bit and 41.6% for Flash-Next, inside the range of published runs on other engines; 100/100
strict-JSON replies valid on the first attempt. Overview with charts: [OVERVIEW.md](OVERVIEW.md).

## Building from source

Works on macOS 26 and 27, and needs **Xcode** (not just the Command Line Tools) with its Metal toolchain: the kernels
are Metal 4 and must be built on the machine's own macOS. Recent Xcode releases (e.g. 26.6) ship **without the Metal
compiler**: install it once with `xcodebuild -downloadComponent MetalToolchain` (about 690 MB). If the Command Line
Tools are the active developer directory (`xcode-select -p` prints `/Library/Developer/CommandLineTools`), `xcrun`
finds no `metal` and `xcodebuild` refuses to run; `./build.sh` then uses `/Applications/Xcode.app` for the build by
itself and says so. To make that permanent, and for `xcodebuild -downloadComponent`, select Xcode once:
`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer` (or prefix a single command with
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`).

```bash
git clone https://github.com/Incept5-AI/inferd && cd inferd/engine
./build.sh                                              # kernels + inferd + tools
./install.sh --model 27b && ./run.sh start --model 27b  # then steps 3-4 above
```

`engine/tools/package.sh <version>` turns a build into the release tarball. M5-class and newer GPUs (with neural
accelerators) are the main target; the M3/M4 family runs on its own kernel path and is lightly tested (the
drafter pays there too since 9dd55ad; `--draft none` gives a faster first token at 16+ agents); M1/M2 are untested. The positional form still works
(`MODEL=… DRAFT=… ./run.sh 8300 32 1024 0 32 65536`: port, max agents, step tokens, KV GiB with 0 = auto, snapshots,
max context). From a checkout, `examples/run_all.sh my-mac` (with the server up) runs the self-tests into
`examples/results/my-mac/` and prints a PASS/FAIL table (the name defaults to the Mac's short host name) (they check that
inferd works on your machine; they are not benchmarks). Per-machine settings: [docs/MACHINES.md](docs/MACHINES.md).

## The agent API

An ordinary OpenAI chat request plus one optional field:

```json
{"model": "qwen3.8-27b", "messages": [...],
 "inferd": {"context": "brief-42", "session": "agent-7", "effort": "low", "max_context": 32768, "priority": 1}}
```

- `POST /v1/contexts` saves and pins a context (a system prompt, documents, a whole conversation); any number of agents
  start from it and pay only for their own new tokens.
- `"session"` keeps an agent's latest turn cached, or parked on SSD, so it resumes in 0.2–0.7 s instead of a 15–18 s
  re-read. `"cache": "drop"` runs one-off work without disturbing anyone's cache.
- `effort`, `max_context` and `priority` are per agent, so a cheap helper and an expensive planner share one server.
- `GET /v1/cache` shows what is cached, pinned, parked and free; every reply reports `cached_tokens`.

Details: [docs/CONTEXTS.md](docs/CONTEXTS.md). Shipped vs experimental: [docs/FEATURES.md](docs/FEATURES.md).

## Documentation

| Document | What is in it |
|---|---|
| [OVERVIEW.md](OVERVIEW.md) | The short story with charts: why agents need a different engine, what is new, where others are better |
| [docs/RESULTS.md](docs/RESULTS.md) | Every benchmark and comparison, methodology, charts, caveats, raw-data links |
| [docs/MACHINES.md](docs/MACHINES.md) | Mac Studio, Mac minis and MacBook Pro: what differs, what pays where, recommended settings |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the engine works: kernels, attention, KV, scheduler, prefix cache, SSD tier, speculative decoding, every env knob |
| [docs/CONTEXTS.md](docs/CONTEXTS.md) | The agent-contexts API |
| [docs/API-COMPAT.md](docs/API-COMPAT.md) | vLLM / OpenAI API compatibility: every endpoint and request field, and what inferd does with it |
| [docs/FEATURES.md](docs/FEATURES.md) | What is shipped, what is opt-in, what is experimental |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Everything we tried, kept or rejected, with numbers; what is ours and what is borrowed |
| [docs/TESTING.md](docs/TESTING.md) | Correctness checks, smoke vs milestone benchmarks, how to report a result |
| [docs/ROADMAP.md](docs/ROADMAP.md) | What could still be improved, prioritised, with evidence |
| [examples/](examples/) | Thirteen small demos and self-tests: quickstart, per-agent budgets, tools, strict JSON, OCR agents, agent contexts, agent hand-off, an agent team under memory pressure, fleets |
| [bench/](bench/) | The measurement harnesses behind docs/RESULTS.md |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, build, workflow, conventions, gotchas, where you can help |
| [SECURITY.md](SECURITY.md) | The trust model (local, trusted use) and how to report a vulnerability |

## Repository layout

```
engine/      the engine: src/ (C++ headers), kernels/ (Metal), tools/ (server, tests), frontend/ (OpenAI API, vision),
             third_party/mlx (vendored MLX kernels, MIT, patched: see its README)
bench/       client harnesses (studio/conc.py, studio/speed.py, wall.py, agents.py, agents_scenarios.py)
results/     raw results (JSON lines, logs) behind every number in docs/RESULTS.md
examples/    small demos and self-tests, one folder each (see examples/README.md)
docs/        the documentation above; docs/img/ charts, regenerated by docs/make_charts.py
micro/       early microbenchmarks (bandwidth, GEMM, MLX baselines)
```

## Acknowledgements

inferd stands on the shoulders of these projects. Much of what it does well, they did first, and several of them are
better than inferd at important things.

- **[MLX](https://github.com/ml-explore/mlx)** (Apple's ml-explore team). inferd vendors MLX's quantized matrix-multiply,
  GEMM and flash-attention Metal kernels (MIT; see [NOTICE](NOTICE) and `engine/third_party/mlx`). Without them there
  would be no engine to speak of.
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)**: the reference every logit in inferd is checked against, and still
  the quickest way to try a new model on a Mac.
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)** (Prince Canuma): inferd's vision path runs its vision towers; it
  makes images on Apple Silicon simple.
- **[Qwen](https://huggingface.co/Qwen)**: open weights good enough to build on, including the hybrid Gated DeltaNet
  design this engine targets.
- **[DFlash](https://github.com/z-lab/dflash)** (z-lab) and the **[incoai DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)**
  drafter: the block-diffusion drafting that makes speculative decoding on the 27B worth doing.
- **TensorFold**: the fastest single stream on a Mac, superb prefill, and arrival-order scheduling that wins when a
  handful of agents start together. The mlx.fast competition around it moved everyone's kernels forward, ours included.
- **Splash**: the engine that showed what a Mac can do for agents (a Neural Engine split, int8 KV, parked state on
  SSD, shared prefill). Faster than inferd for one agent on short contexts, with excellent prefill on smaller chips.
- **oMLX** and **mlx-serve**: polished, quick-to-start Mac servers that are hard to beat for one user.
- **MiaAI-Lab**: their public benchmark recipes shaped our "N agents start at once" test.
- **[vLLM](https://github.com/vllm-project/vllm)**: paged KV, continuous batching and prefix caching, the serving design
  inferd reimplements for hybrid models.
- **[Hugging Face](https://huggingface.co)**: the hub every model and drafter here is downloaded from, and the tokenizers.

## Licence

[Apache License 2.0](LICENSE); third-party attributions in [NOTICE](NOTICE). Measured and maintained by
[Incept5](https://github.com/Incept5-AI).
