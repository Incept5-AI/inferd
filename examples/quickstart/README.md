# quickstart

One chat request, then a streamed follow-up turn, with `usage` printed for both, including how many prompt tokens came
from the prefix cache (`usage.prompt_tokens_details.cached_tokens`).

```bash
python3 quickstart.py              # INFERD_URL=http://host:port/v1 to point elsewhere
```

Start the server first (see [../README.md](../README.md#start-the-server)). Python standard library only.

## What it shows

- **A plain request** with thinking off (`chat_template_kwargs: {"enable_thinking": false}`): the answer comes straight back
  in `message.content`.
- **A streamed follow-up turn** (`stream: true`, `stream_options: {"include_usage": true}`): the request carries the
  whole conversation plus one new question, as an agent's next turn does. inferd resumes from the end of the first
  turn, so it prefills only the new tokens: time to first token is 0.05 s instead of the cold prompt's prefill.

## Sample output (MacBook Pro M5 Max, 1 November 2026)

```
server http://127.0.0.1:8300/v1, model qwen3.8-27b

== 1. chat (non-streaming)
Run `lsof -i :8300` to list the process ID and name bound to that port. If you need the full command line, follow up with `ps -p <PID> -o command`.
finish_reason=stop  usage={"prompt_tokens": 404, "completion_tokens": 48, "total_tokens": 452, "prompt_tokens_details": {"cached_tokens": 384}}  1.16s

== 2. follow-up turn, streamed
Run `kill <PID>` using the process ID you found with `lsof -i :8300`. If the process ignores the signal, escalate with `kill -9 <PID>`.

time to first token 0.05s, total 0.59s, 73 tok/s
usage={"prompt_tokens": 478, "completion_tokens": 43, "total_tokens": 521, "prompt_tokens_details": {"cached_tokens": 448}}
-> 448 of 478 prompt tokens came from the prefix cache (turn 1, in 32-token blocks)
```

On a fresh server, request 1 shows `cached_tokens: 0`. The 384 above is from an earlier run of the same script. The
tok/s figure on such a short reply is mostly fixed request overhead; a 190-token reply runs at about 160 tok/s on
this machine.

## Things worth knowing

- **What gets cached.** inferd resumes a prompt from a cached prefix only where it holds a snapshot of the model's
  recurrent (DeltaNet) state. It takes one where each prompt ends (so a follow-up turn hits), where the system
  prompt ends (the front end sends that position), and where a new prompt branches off a cached one. Agents that
  share a system prompt but ask different questions therefore reuse it from the second agent on: 2,560 of ~4,400
  prompt tokens cached, turn time 2.65 s → 1.22 s on the Studio (before 1 Nov 2026 this was 0 cached).
- **Effort is part of the prompt.** With thinking on, the chat template writes "Reasoning effort is set to low…" into
  the system message. Changing an agent's effort between turns therefore changes its prefix and loses the cache. Keep
  one effort per agent.
