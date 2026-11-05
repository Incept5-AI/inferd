# vLLM / OpenAI API compatibility

inferd's HTTP front end (`engine/frontend/serve.py`) aims to be a drop-in for vLLM's OpenAI-compatible server:
OpenAI SDKs, LiteLLM, LangChain and code written against vLLM's extra request fields should work unchanged. Every
documented vLLM field and endpoint is in one of three states:

- **works**: it does what vLLM does (checked on the response, not only the status code);
- **ignored**: accepted with no effect, for a documented reason (listed below);
- **rejected**: a clean OpenAI-style error (`{"error": {"message", "type", "code", "param"}}`, HTTP 400 or 501).

None of them should crash, hang or silently do something else. `engine/frontend/test_vllm_compat.py` checks all of
this: with `--url` it runs against a live server and prints the table below; without it, CI runs it against the real
front end and a fake engine (`.github/workflows/host-tests.yml`).

```bash
python3 engine/frontend/test_vllm_compat.py --url http://127.0.0.1:8300 [--key $INFERD_API_KEY] [--markdown out.md]
```

## Endpoints

| Endpoint | inferd |
|---|---|
| `POST /v1/chat/completions` | works (streaming and not, n choices, logprobs, tools, structured output) |
| `POST /v1/completions` | works: `prompt` as a string, a list of strings, token ids or lists of token ids; `echo`; `logprobs` (int); streaming; `guided_json` / `response_format` json_schema (constrained, no instruction added). `suffix` is rejected (400), as in vLLM. guided_choice / guided_regex are chat-only (400) |
| `GET /v1/models`, `GET /v1/models/{id}` | works: `id`, `object`, `created`, `owned_by`, `root` (the checkpoint directory's name), `parent`, `max_model_len`, `permission`. `--name a,b` serves several names (vLLM's `--served-model-name`) |
| `GET /health`, `GET`/`POST /ping` | works (503 when the engine process is gone) |
| `GET /version` | works: `{"version", "engine": "inferd"}` |
| `POST /tokenize` | works: `prompt` (+ `add_special_tokens`) or `messages` (chat template, + `tools`, `add_generation_prompt`, `continue_final_message`, `chat_template_kwargs`); returns `count`, `max_model_len`, `tokens`, `token_strs` (with `return_token_strs`) |
| `POST /detokenize` | works: `{"tokens"}` -> `{"prompt"}` |
| `GET /metrics` | works: Prometheus text, names below |
| `/v1/embeddings`, `/pooling`, `/classify`, `/score`, `/v1/score`, `/rerank`, `/v1/rerank`, `/v2/rerank`, `/v1/audio/*`, `/v1/responses`, `/v1/images/generations`, `/v1/load_lora_adapter`, `/v1/unload_lora_adapter` | rejected: 501, code `unsupported_endpoint` |
| any other path or method | rejected: 404 / 405 with an OpenAI-style error body (not FastAPI's `{"detail"}`) |

inferd's own endpoints (`/v1/contexts`, `/v1/sessions`, `/v1/cache`) are in [CONTEXTS.md](CONTEXTS.md).

## Request fields (chat and completions)

| vLLM field | inferd |
|---|---|
| `model` | accepted with any value and echoed in the response (vLLM answers 404 for a name it does not serve; `INFERD_STRICT_MODEL=1` does the same) |
| `messages` | string or content-part content; parts `text`, `input_text`, `output_text`, `refusal` (as text), `image_url` (object or string), `input_image`, `image`; `name`; `system` and `developer` (mapped to system) roles; `tool` messages; assistant `tool_calls` (arguments as a JSON string or an object); `reasoning_content`. Audio, video and file parts: 400 |
| `max_tokens`, `max_completion_tokens` | works (default 4096 for chat, 16 for completions as in vLLM / OpenAI; vLLM's chat default is the rest of the context) |
| `temperature`, `top_p`, `top_k` (-1 or 0 = off), `seed` | works. Omitted `top_k` / `top_p` come from `generation_config.json`, as vLLM. `seed` makes a request reproducible only while the speculative block choice repeats: the self-tuning draft block (`SPEC_AUTO`) depends on load, and it changes how many random draws a sampled request consumes, so two seeded runs can diverge after some tokens (greedy output is deterministic) |
| `min_p` | works (host sampler, see the cost note) |
| `presence_penalty`, `frequency_penalty`, `repetition_penalty` | works, vLLM's definitions: frequency/presence over the output tokens, repetition over prompt + output (positive logits divided, negative multiplied) |
| `logit_bias` | works (token id -> bias in [-100, 100], added to the logits) |
| `min_tokens` | works: stop ids are masked and stop strings are not matched until `min_tokens` tokens |
| `n` | works, up to `INFERD_MAX_N` (16): n engine requests on one prompt (the prefix cache shares it); with `seed`, choice i uses seed + i. Streaming interleaves the choices by `index`. n > 1 with structured output: 400 |
| `best_of` | equal to `n` (or absent): works. Any other value: 400 (vLLM V1 dropped it too) |
| `stop` (string or list) | works: trimmed, `finish_reason: "stop"`, `stop_reason` = the string |
| `stop_token_ids` | works: `stop_reason` = the id |
| `include_stop_str_in_output` | works |
| `ignore_eos` | works (stop ids from `stop_token_ids` still apply, as in vLLM) |
| `logprobs` + `top_logprobs` (chat), `logprobs` (int, completions) | works, up to 20: the raw target model's log-softmax (vLLM's default `raw_logprobs`), for every generated token (reasoning tokens included). Speculative requests stay speculative: each token's logprob comes from its verify row (they agree with plain decoding's to ~0.01 nats). Streaming chunks carry the logprobs of the tokens they release |
| `prompt_logprobs` | rejected (400): the engine keeps no logits for prompt positions |
| `skip_special_tokens`, `spaces_between_special_tokens` | ignored: the text never contains the stop ids, and the Qwen3.x special tokens that could appear (`<tool_call>`, `<think>`) are parsed out by the front end, so the output is what vLLM gives with the defaults |
| `stream`, `stream_options.include_usage`, `stream_options.continuous_usage_stats` | works (usage in every chunk with the latter) |
| `response_format` `text` / `json_object` / `json_schema` (`strict`) | works (chat: an instruction plus validation and retries, constrained decoding when `strict`; [FEATURES.md](FEATURES.md)) |
| `guided_json`, `structured_outputs.json` (vLLM >= 0.10), `structured_outputs.json_object` | works: as `response_format` json_schema with `strict: true` (constrained) |
| `guided_choice`, `structured_outputs.choice` | works (chat): decoded under a JSON `enum` schema and unquoted, so `content` is exactly one of the choices |
| `guided_regex`, `structured_outputs.regex` | works (chat) for the grammar's regex subset: decoded as an anchored JSON string pattern and unquoted. A pattern outside the subset is 400 |
| `guided_grammar`, `structured_outputs.grammar`, `structural_tag` | rejected (400): no context-free grammars |
| `guided_decoding_backend`, `guided_whitespace_pattern`, `structured_outputs.backend` / `disable_*` / `whitespace_pattern` | ignored |
| `tools`, `tool_choice` `auto` / `none` / `required` / named | works. `none`: tool markup stays text, no `tool_calls`. `required` / named: the reply is forced to open with the call's markup (thinking off for that request); named returns exactly one call to that function |
| `parallel_tool_calls: false` | works: the first call only |
| `chat_template` | rejected (400) unless the server runs with `INFERD_TRUST_REQUEST_CHAT_TEMPLATE=1` (vLLM's `--trust-request-chat-template`) |
| `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`, others) | works (passed to the template) |
| `add_generation_prompt`, `continue_final_message` | works (text prompts; with images: 400) |
| `echo` | works: chat prepends the last message when it is the assistant's; completions prepends the prompt |
| `documents` | works (passed to the template; Qwen3.x templates do not use them) |
| `user`, `metadata` | ignored (accepted) |
| `request_id`, `X-Request-Id` header | works: the response id is `chatcmpl-<id>` / `cmpl-<id>` (the body field wins), and the header is echoed back |
| `priority` | accepted with **inferd's** meaning: -10..10, higher is kept longer when the KV pool runs dry (vLLM: lower is scheduled first, and only with `--scheduling-policy priority`) |
| `truncate_prompt_tokens` | works: keeps the last k prompt tokens (-1: the model's limit); 400 for prompts with images or a saved context |
| `cache_salt` | rejected (400): the prefix cache is shared by every request (one trusted user, README "Security and limits"); `"cache": "drop"` keeps a request out of it |
| `lora_request` | rejected (400) |
| unknown fields | ignored, as vLLM |

**Cost of logits processing.** `min_p`, the three penalties, `logit_bias` and `min_tokens` change each token's
distribution with state the GPU's speculative sampler cannot see inside a block, so a request that uses any of them
(with a value that changes something) decodes one token per step on the engine's host sampler, never speculated: for
one stream that is roughly the plain-decode speed (no drafter gain). Other requests are unaffected (`src/scheduler.h`,
`Request::processed`). `logprobs` keeps speculation; it switches off token-tree verification for that request (chains
only) and costs a host pass over the vocabulary per token.

## Response shape

- `id`, `object` (`chat.completion` / `chat.completion.chunk` / `text_completion`), `created`, `model`,
  `system_fingerprint` (null), `choices`, `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`,
  `prompt_tokens_details.cached_tokens`), plus inferd's `inferd.sampling` (the effective sampling settings).
- `choices[]`: `index`, `message` (`role`, `content`, `tool_calls`, `reasoning_content` and vLLM's newer `reasoning`
  alias when thinking is on), `logprobs` (null unless asked), `finish_reason` (`stop`, `length`, `tool_calls`),
  `stop_reason` (null, the stop string or the stop token id).
- Streams: a first chunk with `{"role": "assistant", "content": ""}`, content / reasoning / tool-call deltas, a
  chunk with `finish_reason` and `stop_reason`, the usage chunk when asked, then `data: [DONE]`. An engine failure after
  the stream has started is a `data: {"error": ...}` event followed by `[DONE]`.
- Errors: `{"error": {"message", "type", "code", "param"}}`; 400 malformed, 404 unknown path or model (strict), 413
  oversized, 422 structured output that never validated, 501 unsupported endpoint, 503 overloaded or engine gone
  (`Retry-After`), 504 engine silent past `INFERD_EVENT_TIMEOUT`.

## Metrics (`GET /metrics`)

vLLM's names under an `inferd:` prefix, label `model_name`. From the engine's stats at scrape time:
`inferd:num_requests_running`, `inferd:num_requests_waiting`, `inferd:kv_cache_usage_perc` (blocks held by running
sequences), `inferd:kv_cache_cached_perc` (blocks holding only cached prefixes), `inferd:num_preemptions_total`. From
the front end (per finished engine request, so n choices count n times): `inferd:prompt_tokens_total`,
`inferd:generation_tokens_total`, `inferd:prefix_cache_queries_total`, `inferd:prefix_cache_hits_total`,
`inferd:request_success_total{finished_reason}`, and the histograms `inferd:time_to_first_token_seconds` and
`inferd:e2e_request_latency_seconds` (vLLM's buckets). Counters reset when the front end restarts.

## Probe results

Live run, Qwen3.8-27B 4-bit + DFlash2 on an M5 Max, 5 November 2026 (`--url`, 78 items). Before (main at 19805b9): 21 works, 36 ignored, 2 rejected, 19 broken. After: 60 works, 4 ignored, 14 rejected, 0 broken. "Broken" before was mostly a missing endpoint answered with FastAPI's `{"detail": "Not Found"}` instead of an OpenAI error. Seeded sampling is reproducible over short outputs only (see `seed` above); the probe accepts either result for it.

| Item | Before (19805b9) | After | Notes (after) |
|---|---|---|---|
| POST /v1/chat/completions | ignored | works |  |
| chat stream (SSE shape, [DONE], role chunk) | ignored | works |  |
| stream_options.include_usage | works | works |  |
| stream_options.continuous_usage_stats | ignored | works |  |
| POST /v1/completions (string) | broken | works |  |
| POST /v1/completions (array of strings) | broken | works |  |
| POST /v1/completions (token ids) | broken | works |  |
| POST /v1/completions echo | broken | works |  |
| POST /v1/completions suffix | broken | rejected | 400: suffix: suffix (fill-in-the-middle) is not supported, as in vLLM |
| POST /v1/completions stream | broken | works |  |
| GET /v1/models | ignored | works |  |
| GET /health | works | works |  |
| GET /ping | broken | works |  |
| GET /version | broken | works |  |
| POST /tokenize + /detokenize | broken | works |  |
| GET /metrics | broken | works |  |
| POST /v1/embeddings | broken | rejected | 501: /v1/embeddings: embeddings is not supported by inferd (a generative chat model server); se |
| POST /v1/score | broken | rejected | 501: /v1/score: scoring is not supported by inferd (a generative chat model server); see docs/A |
| POST /v1/rerank | broken | rejected | 501: /v1/rerank: reranking is not supported by inferd (a generative chat model server); see doc |
| POST /rerank | broken | rejected | 501: /rerank: reranking is not supported by inferd (a generative chat model server); see docs/A |
| POST /v1/audio/transcriptions | broken | rejected | 501: /v1/audio/transcriptions: audio is not supported by inferd (a generative chat model server |
| POST /v1/responses | broken | rejected | 501: /v1/responses: the Responses API is not supported by inferd (a generative chat model serve |
| GET /v1/nonexistent | broken | rejected | 404: GET /v1/nonexistent: Not Found |
| model (any name) | works | works | accepted (vLLM: 404 unless a served name; INFERD_STRICT_MODEL=1) |
| messages: content parts (text) | works | works |  |
| messages: input_text parts | works | works |  |
| messages: name, system | works | works |  |
| messages: developer role | rejected | works |  |
| messages: assistant tool_calls + tool result | works | works |  |
| content part audio | rejected | rejected | 400: messages[0].content[0].type: content parts of type 'input_audio' are not supported (text, |
| max_tokens | works | works |  |
| max_completion_tokens | works | works |  |
| temperature / top_p / top_k / seed (reproducible) | works | works |  |
| min_p | ignored | works |  |
| n (> 1) | ignored | works |  |
| n (> 1) streaming | ignored | works |  |
| best_of = n | ignored | works |  |
| best_of > n | ignored | rejected | 400: best_of: best_of different from n is not supported (as in vLLM V1); leave it out |
| stop (string / array) | works | works |  |
| include_stop_str_in_output | works | works |  |
| stop_token_ids | ignored | works | stop_reason 20 |
| ignore_eos | works | works |  |
| min_tokens | ignored | works |  |
| presence_penalty | ignored | works |  |
| frequency_penalty | ignored | works |  |
| repetition_penalty | ignored | works |  |
| logit_bias | ignored | works |  |
| logprobs + top_logprobs (chat) | ignored | works |  |
| logprobs (chat, streaming) | ignored | works |  |
| logprobs (completions, int) | broken | works |  |
| prompt_logprobs | broken | rejected | 400: prompt_logprobs: prompt logprobs are not supported (inferd keeps no logits for prompt posi |
| skip_special_tokens | ignored | ignored | accepted; stop tokens never reach the text either way |
| spaces_between_special_tokens | ignored | ignored | accepted |
| response_format text | works | works |  |
| response_format json_object | works | works |  |
| response_format json_schema strict | works | works |  |
| guided_json | ignored | works |  |
| structured_outputs.json (vLLM >= 0.10) | ignored | works |  |
| guided_choice | ignored | works |  |
| guided_regex | ignored | works |  |
| guided_grammar | ignored | rejected | 400: guided_grammar: context-free grammars are not supported (JSON Schema, regex and choice are |
| tools + tool_choice auto | works | works |  |
| tool_choice none | ignored | works |  |
| tool_choice required | ignored | works |  |
| tool_choice named function | ignored | works |  |
| parallel_tool_calls false | ignored | works |  |
| chat_template (request) | ignored | rejected | 400: chat_template: request chat templates are refused (as vLLM without --trust-request-chat-te |
| chat_template_kwargs enable_thinking / reasoning_content | works | works |  |
| add_generation_prompt false + continue_final_message | ignored | works |  |
| echo (chat) | ignored | works |  |
| documents | works | works |  |
| user | ignored | ignored | accepted |
| metadata | ignored | ignored | accepted |
| request_id / X-Request-Id | ignored | works |  |
| priority | works | works | inferd semantics: higher = kept on preemption |
| truncate_prompt_tokens | ignored | works |  |
| cache_salt | ignored | rejected | 400: cache_salt: not supported: inferd's prefix cache is shared by every request (one trusted u |
| error object shape (bad request) | works | works |  |
