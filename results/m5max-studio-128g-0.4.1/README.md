# Mac Studio M5 Max 128 GB (external tester), inferd 0.4.1

External measurements on a desktop Mac Studio M5 Max (128 GB), first reported in
[issue #3](https://github.com/Incept5-AI/inferd/issues/3). The 9 November run used the 128 GB
`run.sh` default (16 agents, 24 GiB KV). The 13 November runs use the 32-agent sizing requested
on that issue, plus same-evening default-sizing cells at N=16. Dated study-log entry:
[RESULTS.md (af)](../../docs/RESULTS.md#af-external-mac-studio-m5-max-128-gb-041-default-128-gb-sizing-and-a-32-agent-run-9-and-13-november).

## Machine

| Field | Value |
|---|---|
| Model | Mac Studio (`Mac17,14`) |
| Chip | Apple M5 Max |
| GPU cores | 40 (`gpu-core-count`) |
| CPU | 18 (6P + 12E) |
| Memory | 128 GB |
| macOS | 27.2 (`26B118`) — `sw_vers` |
| Metal | Metal 4 |
| Xcode / Metal toolchain | Xcode.app not installed; `xcodebuild` n/a; `xcrun metal` unavailable (CLT only). Prebuilt route used. |

Desktop (not a laptop): AC power, no lid. `pmset -g therm`: no thermal / performance / CPU-power warning on either day.

## Build

| Field | Value |
|---|---|
| Package | `inferd-0.4.1-macos27-arm64` official prebuilt tarball |
| BUILD-INFO commit | `4ad756ad70a300d420f92a0cbd5b51d8f23a76f7` |
| Source tree | no git checkout (prebuilt only; N/A dirty) |
| `build/inferd` sha256 | `cd16f8d80ca9ae22ae71418d8a28e7fbd1d9d0f388f088677949bdf28dcb40d7` |
| `build/core.metallib` sha256 | `991b9c871920909f845c6d69267c2343f5c53a76d2c4c6b565b79883140fe805` |
| Python | 3.12.16 (install.sh venv), Pillow 13.0.0 |

Both days used this same prebuilt and the same weights.

## Model + drafter

| Role | Hugging Face id | Snapshot | Precision |
|---|---|---|---|
| Target | `mlx-community/Qwen3.8-27B-4bit` | `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` | 4-bit |
| Drafter | `incoai/Qwen3.8-27B-DFlash2` | `015e795645c74b1a0eeef3b570031fb62e769bc5` | DFlash2 (block 16) |

## Server arguments and env knobs

No non-default `KV8` / `SPEC*` / `W4A8` / `ANE*` / `NAX` / `LANE` / `QMST` / `ADMIT` / `CASCADE` overrides.
`DRAFT`, `SPILL_DIR` and `SPILL_GIB=96` came from `run.sh`. Listen: `127.0.0.1:8300`. `max_ctx` 147456, `step_tokens` 1024.

**Default 128 GB sizing** — `./run.sh start --model 27b` (9 November full sweep, and two 13 November cells at N=16):

```text
#   max_running=16  step_tokens=1024  kv_gib=24  snapshots=30  max_ctx=147456
#   SPILL_GIB=96
#   engine argv: ./build/inferd <model> <socket> 16 1024 24 30 147456
inferd: memory: GPU budget 107.5 GiB; model + state + scratch 21.9, drafter 2.8, PF8 0.0, KV 24.0 -> 48.7 allocated
inferd: 8-bit KV, 24197 KV blocks (24 GiB, 0.8M tokens), 46 slots (6.6 GiB state), max ctx 147456
inferd: PF8: off (auto: QA8 is on); QA8 on (neural accelerators); SSD tier 96 GiB
```

The 13 November default-sizing starts printed the same memory line (48.7 allocated, 24,197 KV blocks, 46 slots).

**32-agent sizing** — `./run.sh start --model 27b --agents 32 --kv 48 --snapshots 64` (13 November, three reps):

```text
#   engine argv: ./build/inferd <model> <socket> 32 1024 48 64 147456
inferd: memory: GPU budget 107.5 GiB; model + state + scratch 29.1, drafter 4.3, PF8 0.0, KV 48.0 -> 81.4 allocated
inferd: 8-bit KV, 48395 KV blocks (48 GiB, 1.5M tokens), 96 slots (13.8 GiB state), max ctx 147456
```

Every 32-agent start printed that same memory line. 81.4 GiB is inside the 107.5 GiB budget. The SSD spill directory was left in place across restarts, as `run.sh` does by default; during the first 13 November session the spill tier reached its 96 GiB cap.

## Method (13 November `conc.py` reps)

All scored cells are `bench/studio/conc.py --mode followup`, 16k context, `max_tokens` 512, greedy, thinking off.

- Rep 1 of the 32-agent sizing (`…-a32-conc-16k-agents.jsonl`) was the first sweep after a fresh start, with no cooldown. `agents_at_once` 32 × 16k ran on that same server afterwards, without a restart.
- Before each later rep, inferd was stopped, the machine sat 240 s with port 8300 closed, the server was started, the sweep ran immediately, and the server was stopped again. That protocol covers 32-agent reps 2 and 3 (16k × 1/4/8/16/32) and two default-sizing cells at N=16.
- `thermalState` 0 (nominal) at the start and end of each of those later runs. `pmset -g therm` recorded no warnings on either day.

## Uptime, thermal, other load

| | 9 November | 13 November |
|---|---|---|
| Server | Fresh start for `release_gate.sh`; restarted once after the gate before `conc.py` / `agents_at_once` (same default sizing). Scored cells ~13:38–13:52 JST. | 32-agent rep 1: fresh start 22:01 JST for `conc.py`; `agents_at_once` 32 × 16k on the same server (~22:18–22:26 JST). Later reps: stop / 240 s cooldown / restart, measured immediately, then stopped (rep 2 ~22:42–22:56 JST, rep 3 ~23:01–23:15 JST; default N=16 ~23:20–23:32 JST). |
| Host uptime | ~7 days | ~11 days |
| Thermal | No warnings (`pmset -g therm`) | No warnings. `thermalState` 0 (nominal) at the start and end of every cooldown-protocol run. |
| Other load | LM Studio app service present but not serving. Another local OpenAI-compatible server on port 8000 was stopped for the run and restored afterward. | Same pattern: the port-8000 server was stopped for the runs; LM Studio present but not serving. |

## Correctness

- 9 November: `release_gate.sh` **PASS** (`EXIT=0`). Host / front-end / logits chunks 1, 24, 256, 1536 / bench / speculative=plain all ok. Speculative step: `lossless: 3/4 identical, 1 diverged: seqs 1`; tie_check `seq 1 @ 103` exact near-tie (gap 0.000) → ok. GPU budget 107.5 GiB.
- `conc.py`: 0 errors in all 15 cells of the 32-agent 3-rep sweep, and 0 errors in the two default-sizing N=16 cells. `agents_at_once` (`quick64.py`) passed with every answer correct and every agent decoding at once.

## Files

Harnesses: `bench/studio/conc.py --mode followup` (16k, `--n` as in each file, `max_tokens` 512, greedy, thinking off) and `examples/agents-at-once` (`quick64.py`).

| File | What |
|---|---|
| `m5max-studio-128g-0.4.1-20261109-conc-16k-agents.jsonl` | Default sizing, 16k × 1/4/8/16 (9 November, single trial) |
| `m5max-studio-128g-0.4.1-20261109-agents-at-once-8x4k.jsonl` | Default sizing, 8 × 4k, PASS 16/16, 356.1 tok/s |
| `m5max-studio-128g-0.4.1-20261109-agents-at-once-16x16k.jsonl` | Default sizing, 16 × 16k, PASS 32/32, 371.4 tok/s |
| `m5max-studio-128g-0.4.1-20261113-a32-conc-16k-agents.jsonl` | 32-agent sizing, 16k × 1/4/8/16/32, rep 1 (no cooldown) |
| `m5max-studio-128g-0.4.1-20261113-a32-rep2-conc-16k-agents.jsonl` | 32-agent sizing, same sweep, rep 2 (240 s cooldown) |
| `m5max-studio-128g-0.4.1-20261113-a32-rep3-conc-16k-agents.jsonl` | 32-agent sizing, same sweep, rep 3 (240 s cooldown) |
| `m5max-studio-128g-0.4.1-20261113-a32-agents-at-once-32x16k.jsonl` | 32-agent sizing, 32 × 16k, PASS 64/64, 372.6 tok/s (after rep 1, no restart) |
| `m5max-studio-128g-0.4.1-20261113-default-n16-r1-conc-16k-agents.jsonl` | Default sizing, 16k × 16, 13 November r1 (240 s cooldown), 323.7 tok/s |
| `m5max-studio-128g-0.4.1-20261113-default-n16-r2-conc-16k-agents.jsonl` | Default sizing, 16k × 16, 13 November r2 (240 s cooldown), 324.6 tok/s |

Headline numbers are in RESULTS.md (af). Logs, home paths, usernames, hostnames and non-loopback IPs are not included.

All times in this folder are JST (UTC+9). The `ts` fields in the `.jsonl` files carry no zone and are also JST.
