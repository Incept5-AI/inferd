# agent-contexts

> Uses the agent-context API ([docs/CONTEXTS.md](../../docs/CONTEXTS.md)), in the released front end since 2 Nov 2026.

Save a long context once, fork agents from it, park agents on SSD and resume them, and run one-off agents that leave
nothing in the cache: inferd's agent-context API ([docs/CONTEXTS.md](../../docs/CONTEXTS.md)).

```bash
python3 agent_contexts.py                       # http://127.0.0.1:8300/v1; --doc 8192 --agents 4
```

Start the server with an SSD tier to see steps 4-5 (`SPILL_DIR=/path/to/ssd/dir ./run.sh ...`); without one they
are skipped.

## The calls

```python
# 1. save: computed once, pinned in the prefix cache, also written to SSD
POST /v1/contexts {"name": "catalogue", "system": "...8k tokens...", "effort": "none", "persist": "ssd"}

# 2. fork agents from it: only their own messages; the context is put in front by the server
POST /v1/chat/completions {"messages": [{"role": "user", "content": "Who leads S-010?"}],
                           "inferd": {"context": "catalogue", "session": "agent-0", "effort": "none"}}

# 4. move the context to SSD (frees GPU memory); the next launch restores it
POST /v1/contexts/catalogue/offload
# 5. park an idle agent; its next turn restores it
POST /v1/sessions/agent-1/offload
# 6. one-off agents: nothing of theirs stays cached
POST /v1/chat/completions {..., "inferd": {"cache": "drop"}}
# 7. clean up
DELETE /v1/sessions/agent-0     DELETE /v1/contexts/catalogue     GET /v1/cache
```

## Output

Mac mini M5 Pro (64 GB), Qwen3.8-27B 4-bit (mlx-community) + DFlash2, server `run.sh 8300 16 1024 0 32 32768` with
`SPILL_DIR` set, 2 November 2026:

```
1. save a context (8k-token catalogue, effort none, persist ssd)
  catalogue-6aefbfa7: 8372 tokens, computed in 18.464 s; pinned=True gpu=True ssd=True (265.1 MiB of KV)

2. launch 4 agents from it at once, each with its own task
  agent 0    TTFT  0.51 s  prompt   8400  cached   8352  | The on-call lead for service S-010 is Aiko Osei.
  agent 1    TTFT  0.51 s  prompt   8400  cached   8384  | The on-call lead for service S-017 is Kwame Iyer.
  agent 2    TTFT  0.51 s  prompt   8400  cached   8384  | The on-call lead for service S-024 is Mei Varga.
  agent 3    TTFT  0.51 s  prompt   8400  cached   8384  | The on-call lead for service S-031 is Freya Rossi.
  all 4 answered in 0.90 s

3. two more turns for agent 0: a long answer, then a question about it (resumes after its own answer)
  agent 0    TTFT  0.21 s  prompt   8448  cached   8384  | S-002 RB-2408 / S-010 RB-1976 / S-015 RB-8353 / S-030 RB-8053 / S-031
  agent 0    TTFT  0.72 s  prompt   8712  cached   8448  | The service with the highest number among those listed is S-136, which

4. offload the context to SSD, then launch a new agent from it
  after offload: pinned=False gpu=False ssd=True (265.1 MiB of KV)
  new agent  TTFT  0.18 s  prompt   8396  cached   8352  | The Search team owns service S-005.
  after the launch: pinned=False gpu=True ssd=True (265.1 MiB of KV)

5. park agent 1 (its latest turn goes to SSD), then resume it
  {'id': 'agent-6aefbfa7-1', 'object': 'session', 'offloaded': True}
  agent 1    TTFT  0.21 s  prompt   8440  cached   8384  | The Messaging team owns service S-017.

6. 20 one-off agents with "cache": "drop"
  snapshots 6 -> 6, cached KV blocks 272 -> 272

7. clean up
  {'id': 'catalogue-6aefbfa7', 'object': 'context', 'deleted': True}
  cache now: 0 snapshots, 0 cached KV blocks, 0 contexts
```

What it shows:
- The 8k context is computed once (18.5 s); four agents launched from it at once all start in 0.5 s, each computing
  only its own ~30 tokens.
- Offloaded to SSD (GPU memory freed), the context comes back for the next launch in 0.18 s to first token instead
  of an 18 s recompute. A parked agent resumes from SSD in 0.21 s.
- One-off agents with `"cache": "drop"` leave the snapshot count and cached KV exactly where they were.
- Step 3, second turn: the cache resumed at the end of the previous *prompt* (8,448), not of the 264-token answer.
  A lone stream snapshots inside its answer only when a step happens to end on a block boundary (decode snapshots,
  `SNAP_GEN`); with two or more agents decoding, the engine lands blocks on the boundaries and the answer is cached
  too.
