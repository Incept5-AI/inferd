# agent-budgets

Four agents share one server, each with its own **context cap**, **reasoning effort** and **priority**, and all run at
the same time. Afterwards two agents break their caps on purpose, so you can see what an over-budget agent gets back.

```bash
python3 agent_budgets.py           # SERVER_MAX_CONTEXT=32768 (run.sh's last argument) sizes the coder
```

Start the server first (see [../README.md](../README.md#start-the-server)). Python standard library only.

## The three knobs (request fields)

| Field | What it does |
|---|---|
| `max_context` | This agent's cap on prompt + output tokens. A longer prompt gets **HTTP 400 `context_length_exceeded`**; generation stops at the cap with **`finish_reason: "length"`**. KV memory is allocated as the context grows, so an 8k agent never holds more than 8k of KV. |
| `chat_template_kwargs` | The effort. `{"enable_thinking": false}` = none (no reasoning). `{"reasoning_effort": "low" \| "medium" \| "xhigh"}` = thinking at that level. **`xhigh` is the template's default**, so leaving the field out means maximum thinking. Any other value (e.g. `"high"`) makes the template raise: HTTP 400 with the template's message. |
| `priority` | -10..10. When the KV pool runs dry, the lowest-priority sequence is preempted first and resumes later from its cache. It does not reorder admission. |

Both inferd fields may also go under `"inferd": {...}` if your client validates top-level fields.

| Agent | `max_context` | Effort | Priority | Task |
|---|---:|---|---:|---|
| summariser | 16384 | none | 0 | two-sentence summary of a ~1k-token text |
| coder | 32768 (server max) | medium | 5 | a decorator with hit/miss counters |
| searcher-1 | 8192 | none | -5 | one-line factual answer |
| searcher-2 | 8192 | low | -5 | one-line factual answer |

## Sample output (MacBook Pro M5 Max, 1 November 2026)

```
4 agents in parallel on http://127.0.0.1:8300/v1

agent       max_context  effort prio prompt output  finish  secs
summariser        16384    none    0   1074     62    stop   1.4
coder             32768  medium    5     67   1362    stop  12.7
searcher-1         8192    none   -5     32      3    stop   0.6
searcher-2         8192     low   -5     56    128    stop   2.4
wall time for all four: 12.7s

--- summariser
Apple Silicon Macs utilize a unified memory pool that allows a 4-bit quantized 27-billion-parameter model to coexist with a large key-value cache. Efficiently serving multiple agents simultaneously requires careful scheduling to maintain individual contexts, batch decoding steps, and avoid redundant re-reading of existing data.

--- coder (reasoned 4223 chars first)
```python
from collections import OrderedDict
import functools


def lru_cache_stats(maxsize=128):
    ...
[...]

--- searcher-1
Canberra

--- searcher-2 (reasoned 559 chars first)
Jupiter has the shortest day of any planet in our solar system, with a rotational period of approximately 9 hours and 56 minutes.

=== over budget
searcher-1 (max_context 8192) sent a ~10k-token document -> HTTP 400: context_length_exceeded: prompt is 10470 tokens, max_context is 8192
tiny (max_context 120) asked for 300 words -> HTTP 200, finish_reason 'length' after 82 tokens (prompt 38 + output 82 = 120 <= 120)
   ...se. The wind outside was a living thing, screaming through the narrow slits of the tower, trying to claw its way in, but
```

The short agents finish in under 3 s while the coder is still thinking: every admitted agent decodes in the same
step, so short work does not queue behind long work. The whole set takes as long as the coder alone.

## Notes

- The server's own maximum (`run.sh`'s last argument) still applies: **prompt + `max_tokens` above it is refused**
  ("context too long"), even when `max_context` is larger. Give the coder `max_tokens` that fits.
- An over-cap prompt is rejected before any work is done. Trim the history, or summarise it with a cheap agent, and
  retry.
