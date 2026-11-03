# Agent scenarios: raw results (2 November 2026)

Produced by [bench/agents_scenarios.py](../../bench/agents_scenarios.py) through
a lab script (not published): one server at a time, the same Qwen3.8-27B 4-bit weights
(mlx-community) and the DFlash2 drafter; greedy, thinking off except the scenario e researchers (effort low). Every
run starts its documents with a fresh nonce. `rows.jsonl` holds one row per scenario phase; `<label>.txt` is the
console output; `*-inferd.log` the inferd start-up lines.

| folder | machine | inferd server line | harness preset |
|---|---|---|---|
| `m5pro/` | Mac mini M5 Pro, 64 GB | `run.sh 8300 16 1024 4 16 32768` + SSD tier (4 GiB KV, ~129k tokens: deliberately memory-short) | `m5pro` |
| `m6/` | Mac mini M6, 32 GB | `run.sh 8300 16 1024 0 8 32768` + SSD tier (auto KV, 2.4 GiB, ~78k tokens) | `m6`; `*-pressure` runs add `--set filler=6 --set d_idle=20` |

Labels (inferd source states; `scheduler.h` md5 prefixes):

- `before`: the tree before the agent-context API (f2cc8d1d), `--mode implicit` (per-request `priority` /
  `max_context` only; the prefix cache works by itself). `before-nossd`: the same without the SSD tier.
- `after`: the agent-context API as merged (8862e1c1; equal in behaviour to the pushed 4feb7f5), `--mode api`.
  `after-pressure-keep`: the same build, scenario d with one-offs sending `cache: keep` instead of `drop`.
- `after-final`, `after-final-pressure` (M6): the merged tree at an intermediate state that reserved 1 GiB for a vision
  tower these text runs never load (KV 1.4 GiB instead of 2.4). `after-final-pressure-nomargin`: the same with that
  reserve off, which is the default of the pushed build (the reserve became opt-in, `VISION=1`): use it for
  comparisons with `after` / `before`.
- `after2`, `after2-rerun`: the API plus two prefill-scheduling experiments that are **not** merged (`PREFILL_SHORT`,
  `PREFILL_PRIORITY`; 429c209c). `after2-CONTAMINATED.txt` (rows flagged `contaminated`) overlapped another 27B load on
  the same machine; use `after2-rerun`.
- `splash`, `omlx`, `tf` (TensorFold 1.2.1 defaults), `tf-fcfs` (`TENSORFOLD_PREFILL_POLICY=fcfs`,
  `TENSORFOLD_PREFILL_TOKENS=8192`): `--mode plain`, each engine at its own default memory settings (not capped like
  inferd), so the eviction scenarios are not like for like.

`ctx-gate.log`: the merge gate for the API (27B logits byte-identical to the build before it on chunks 1 and 24, and
the [agent-contexts](../../examples/agent-contexts/) example run end to end on the merged tree).

The tables and their reading are in [docs/CONTEXTS.md](../../docs/CONTEXTS.md#where-it-pays).
