# coding-smoke

A real coding workload in about two minutes: the **Python and TypeScript** tests of
[languages-benchmark](https://github.com/Incept5-AI/languages-benchmark) v1, which has 8 challenges per language
(JSON merge, CSV analysis, log parsing, HTTP client, Dijkstra, file-tree walk, system info, matrices), so 16 tests.
For each test the model writes a program, the harness builds it, runs it and checks what it prints, and a failure
gets one feedback round. A test scores 1.0 on the first try and 0.5 after feedback. Eight tests run concurrently,
which makes this a quick check that a server change has not broken generation quality under load.

```bash
./coding_smoke.sh medium 8      # effort (none|low|medium|xhigh), workers; clones languages-benchmark on first use
LANGBENCH=<your checkout> ./coding_smoke.sh medium 8   # or use an existing checkout
```

`coding_smoke.sh` runs the benchmark unmodified (`bench.py run --suite v1 --langs python,typescript ...`), writes
its result file to a temp directory (the benchmark repo stays clean), and prints the report.

## Requirements

- `python3` 3.10+ (the benchmark is standard library only).
- **Node + `tsx`** for TypeScript: the harness runs `npx tsx program.ts`
  (`brew install node && npm i -g tsx`). `python3 bench.py doctor` checks the toolchains. The script prints its
  `v1:` line first. Java/Swift/C/Rust are not needed for this subset.
- The server's max context must hold prompt + `--max-tokens`. The script sends `MAX_TOKENS=30000` for a 32768-token
  server (v1 prompts are about 1k tokens). Larger values are refused with "context too long".

## Results (MacBook Pro M5 Max, Qwen3.8-27B 4-bit + DFlash2, 8 workers, 1 November 2026)

| Effort | Score | First try | Capped | Errors | Wall time |
|---|---:|---:|---:|---:|---:|
| medium | **16/16** | 16 | 0 | 0 | 128 s |
| none (`--no-thinking`) | **16/16** | 16 | 0 | 0 | 41 s |

```
v1: 16 of 16 tests to run · 8 workers · qwen3.8-27b · thinking · effort medium · max_tokens 30000
  [  1/16] csv_analysis           python     1.0  pass
  ...
  [ 16/16] file_tree_analysis     typescript 1.0  pass

qwen3.8-27b-medium-29k: 16.0/16  first-try 16  any 16  hit-cap 0  looped 0  errors 0  2 min
  Python        8.0/8
  TypeScript    8.0/8
wall time: 128s
```

Python and TypeScript v1 are close to solved for this model, so this smoke test catches **breakage** (garbled output,
errors, a server that stalls under 8 concurrent requests), not small quality differences. For model-vs-model work, run
the full v1 + v2 suites with `--runs 3`. A single run is noisy.
