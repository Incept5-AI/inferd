# structured-output

OpenAI **`response_format`**: `{"type": "json_object"}` and `{"type": "json_schema", "json_schema": {"name",
"schema", "strict"}}`. The reply's `content` is the JSON itself (no code fences, no prose around it), already checked
against your schema. Four requests: a pydantic-style invoice schema (`$defs`/`$ref`, enum, pattern, nested array),
a `json_object` with thinking on, a streamed reply, and a schema the model keeps missing, which ends in HTTP 422.

```bash
python3 structured_output.py              # validate-and-retry (any OpenAI-compatible server works too)
python3 structured_output.py --strict     # strict: true: constrained decoding, then the same validation
```

Start the server first (see [../README.md](../README.md#start-the-server)). Python standard library only.
It needs the front end with **`response_format`** (`engine/frontend/serve.py` + `structured.py`, November 2026).

## How it works (front end, `engine/frontend/structured.py`)

1. **Tell the model.** An instruction with the schema is appended to the **last user turn** (not the system prompt,
   so the system prompt and the conversation so far stay a cached prefix). Thinking may stay on: the reasoning goes
   to `reasoning_content`, and only the answer after it is parsed.
2. **Extract.** The answer's JSON is taken from the whole reply, else from a code fence, else from the first `{` or
   `[` that starts a complete value (leading prose dropped). `NaN`/`Infinity` are not JSON and are refused.
3. **Validate** with a small standard-library JSON Schema validator: `type` (incl. lists and `nullable`), `enum`,
   `const`, `properties`, `required`, `additionalProperties`, `patternProperties`, `min/maxProperties`, `items`
   (schema or tuple), `prefixItems`, `min/maxItems`, `uniqueItems`, `contains`, `min/maxLength`, `pattern`,
   `minimum`/`maximum`/`exclusive*`, `multipleOf`, `anyOf`/`oneOf`/`allOf`/`not`, local `$ref` (`#/$defs/...`,
   `#/definitions/...`). `format` is ignored. Patterns use the bounded regex subset described under
   [the supported subset](#the-supported-subset), and validation runs off the event loop with a work cap. A schema
   outside the subset is refused up front with **HTTP 400** (same rules with or without `strict`).
4. **Retry.** An invalid answer goes back to the model as an assistant turn plus a user turn with the validation
   error (`"$.lines[1].qty: expected integer, got string"`), up to **`json_retries`** times (default 2, 0-5; a
   top-level request field or under `"inferd": {...}`). Retries resume from the prefix cache.
5. **Reply.** Valid: HTTP 200, `content` = the JSON text, `finish_reason: "stop"`, `usage` summed over every
   attempt, and `"inferd": {"json_attempts": N, "json_seconds": S}`. Still invalid after the last attempt:
   **HTTP 422**, `error.code: "json_validation_failed"`, with `attempts`, `validation_error` and `last_output` (the
   model's last reply, unmodified) so a client can salvage it if it wants to.

Streaming (`"stream": true`) buffers the answer until it validates, then sends it (reasoning, then content, then
the finish chunk, then usage if asked for): you get nothing until the JSON is known to be good. If the model calls
one of your `tools` instead, the tool call is returned as usual (`finish_reason: "tool_calls"`).

## Sample output (MacBook Pro M5 Max, 2 November 2026)

Qwen3.8-27B 4-bit (lmstudio-community) with the DFlash2 drafter, greedy.

```
server http://127.0.0.1:8300/v1, model qwen3.8-27b

== 1. json_schema: invoice extraction
{
  "number": "INV-2291",
  "vendor": "Acme Ltd",
  "currency": "GBP",
  "lines": [
    {
      "item": "widget",
      "qty": 3,
      "unit_price": 4.5
    },
    {
      "item": "gadget",
      "qty": 1,
      "unit_price": 12.0
    },
    {
      "item": "installation",
      "qty": 2,
      "unit_price": 40.0
    }
  ],
  "vat_rate": 0.2
}
finish_reason=stop  attempts=1  completion_tokens=153  1.22s

   -> net 105.50 GBP, gross 126.60 (computed by the client)

== 2. json_object, effort low
{"Apollo 11":1969,"Apollo 12":1969,"Apollo 14":1971,"Apollo 15":1971,"Apollo 16":1972,"Apollo 17":1972}
finish_reason=stop  attempts=1  completion_tokens=214  1.65s

== 3. json_schema, streamed
{"label":"mixed","aspects":["battery","support"]}
finish_reason=stop  (1 content delta(s), 0.37s)

== 4. retries, then 422 (json_retries 1)
HTTP 422 json_validation_failed: no schema-valid JSON after 2 attempt(s): the JSON does not match the schema: $.history: string is 1194 characters, maximum is 501
last_output: '{"history": "The Roman Empire, established in 27 BC when Augustus became the first emperor, marked the transition from the Roman Republic to a centralized autocratic state. It expanded to encompass mu'
```

Request 4 is deliberately unreasonable (a paragraph of exactly 500-501 characters): it shows what a client gets
when the model cannot meet the schema.

### With `--strict` (constrained decoding)

Requests 1-3 come back the same apart from formatting the grammar leaves to the model (`"unit_price": 4.50` instead
of `4.5`; `"customer support"` instead of `"support"`), in 1.27 s, 1.51 s and 0.34 s. Request 4 changes:

```
== 4. a 500-501 character limit (json_retries 1)
{"history": "The Roman Empire, established in 27 BC when Augustus became the first emperor, marked the transition from the Roman Republic to a centralized autocratic state. It expanded to encompass much of the Mediterranean basin, North Africa, and parts of Europe and the Middle East, reaching its greatest territorial extent under Emperor Trajan in 117 AD. The empire was characterized by its sophisticated legal system, extensive road networks, and military prowess, which facilitated trade, communication, and "}
finish_reason=stop  attempts=1  completion_tokens=102  2.32s
```

Valid on the first attempt: `maxLength` is one of the keywords the grammar enforces, so the engine would not let the
string grow past 501 characters and closed it there, mid-sentence. That is what constrained decoding can do and a
prompt cannot. It is also why a hard length limit is a blunt tool: the model did not plan for it.

## How often does the model get it right? (live test, same machine)

[`engine/frontend/live_structured.py`](../../engine/frontend/live_structured.py): 20 varied schemas (enums, patterns,
`$defs` recursion, `oneOf`, tuples of fixed length, nullable fields, `json_object`), greedy, thinking off and effort
low, each case checked again by the client.

| | valid on the first attempt | valid after retries | retries used |
|---|---:|---:|---:|
| thinking off | 20/20 | 20/20 | 0 |
| effort low | 20/20 | 20/20 | 0 |
| 5 harder cases (`--hard`), thinking off | 3/5 | 4/5 | 4 |
| 5 harder cases, effort low | 5/5 | 5/5 | 0 |

The harder cases are built to fail (a 22-character line limit on a haiku, a summary of the Roman Empire in 40
characters, four distinct integers whose sum is a multiple of 7). The validation itself costs well under a
millisecond; the latency cost is the retries, one extra turn each from the cached prefix (the haiku's two retries
took 1.6 s in all). JSON replies are also much shorter than prose: the 20 cases took 14.6 s with `response_format`
against 39.8 s for the same prompts without it (thinking off).

## `strict: true`: constrained decoding for a documented subset of JSON Schema, followed by schema validation

With `"strict": true` (or `"inferd": {"constrained": true}`, which also works for `json_object`), the front end sends
the schema to the engine, which samples every token of the answer under a grammar compiled from it: a token is allowed
only when all of its bytes keep the reply a valid prefix of a value the grammar accepts. The grammar enforces part of
the subset below; the front end then validates the result against the whole schema and retries (or answers 422)
exactly as without `strict`. A strict reply's `"inferd"` block lists the schema's keywords that were only validated,
not enforced, as `json_checked_only`.

### The supported subset

| Keyword | Enforced while decoding (grammar) | Validated after decoding |
|---|---|---|
| JSON syntax (no `NaN`, no raw control characters in strings, no duplicate keys) | yes | yes |
| `type` (a name or a list), OpenAPI `nullable`, integer vs number | yes | yes |
| `enum`, `const` | yes (as their compact JSON texts) | yes |
| `properties`, `required` (`}` is refused until every required key is present) | yes | yes |
| `additionalProperties` (`false`: unknown keys cannot be started; a schema: applied) | yes | yes |
| `items` (schema or tuple), `prefixItems`, `additionalItems`, `minItems`, `maxItems` | yes | yes |
| `minLength`, `maxLength` (characters) | yes | yes |
| `pattern` (the regex subset below) | yes | yes |
| `format` `date`, `date-time`, `time`, `uuid` | yes | no (annotation) |
| local `$ref` (`#/$defs/...`, `#/definitions/...`), recursion included | yes | yes |
| `minimum` / `maximum` / `exclusive*` | only `minimum >= 0` (no `-`) and integer ranges of up to 1,001 values | yes |
| `anyOf`, `oneOf`, `allOf` | as the union of the branches (looser) | yes |
| `not`, `contains`, `minProperties`, `maxProperties`, `multipleOf`, `uniqueItems`, `patternProperties` | no | yes |
| `title`, `description`, `default`, `examples`, `$schema`, `$id`, `$comment`, other `format`s, `x-...` and other unknown keywords | ignored (annotations) | ignored |

**Refused with HTTP 400** (the request never reaches the model): the JSON Schema validation keywords outside the
subset (`if`/`then`/`else`, `dependentRequired`, `dependentSchemas`, `dependencies`, `propertyNames`,
`unevaluatedProperties`, `unevaluatedItems`, `minContains`, `maxContains`, `$dynamicRef`, `$recursiveRef`,
`contentSchema`), remote or unresolvable `$ref`, malformed keyword values (an unknown type name, a negative
`minLength`, `required` that is not a list of strings...), `false` subschemas other than `additionalProperties` and
`items`, and schemas over a budget: 64 KiB of compact JSON, 48 levels of nesting, 20,000 `enum`/`const` values,
patterns over 512 characters, 32 nested groups, `{m,n}` bounds over 64 or 2,000 NFA states. The engine applies the
same budgets again (plus 4,096 compiled subschemas and 20,000 literals including enumerated integer ranges) and
refuses with a 400 too.

**Patterns** use one regex subset in both the grammar and the validator, matched by a Thompson NFA (linear time, no
backtracking): literals, `.` (not newline), classes `[a-z]` `[^...]`, `\d \w \s \D \W \S \n \t \r \f \v` and escaped
punctuation, groups `( )` `(?: )`, `|`, `* + ? {m} {m,} {m,n}`, `^` only at the start and `$` only at the end, ASCII
only; unanchored ends search, as JSON Schema's `pattern` does. Back-references, lookaround, `\b`, `\p{...}`, inline
flags and non-ASCII characters are refused. 20,000 random pattern/string pairs gave the same answer (or the same
refusal) in the engine's and the validator's implementation. Validation work is capped per reply (2,000,000 NFA
steps and validator calls together); a reply over the cap counts as invalid.

### Measured (laptop M5 Max, Qwen3.8-27B 4-bit + DFlash2, 2 November 2026)

`live_structured.py --strict`: 20 schemas × 2 modes (thinking off, effort low), greedy, each reply validated again by
the client.

| Set | n | valid on the first attempt | valid after retries | end-to-end tok/s |
|---|---:|---:|---:|---:|
| strict, thinking off | 20 | 20/20 | 20/20 | 108.7 |
| strict, effort low | 20 | 20/20 | 20/20 | 101.0 |
| validate-and-retry only, thinking off (same day, earlier build) | 20 | 20/20 | 20/20 | 76.6 |
| validate-and-retry only, effort low (same day, earlier build) | 20 | 20/20 | 20/20 | 82.6 |
| strict, 5 harder cases, thinking off | 5 | 4/5 | 4/5 | |
| strict, 5 harder cases, effort low | 5 | 5/5 | 5/5 | |
| strict at once with 4 plain requests: temperature 0.7 thinking off / 0.7 effort low / greedy | 20 + 20 + 20 | 60/60 | 60/60 | |

The tok/s of the two builds are not a like-for-like comparison (other engine changes landed in between); on the same
build earlier in the day strict ran at 68.7 / 71.4 against 76.6 / 82.6. The one hard-set failure is `multipleOf`, which
is validated but not enforced: three attempts, then 422. The haiku case (22-character lines) that needed three
attempts without the grammar passed on the first with it. Unconstrained requests are unaffected: `decode_test check`
logits byte-identical at chunks 1 and 24, single stream and `conc` 16k × 8 within run-to-run noise.

### How sampling works under the grammar

- **Thinking stays free:** the reasoning is unconstrained, and the grammar starts at `</think>`.
- **Greedy (temperature 0) stays speculative.** The drafter's block is verified as usual, then its committed tokens
  are checked against the grammar on the host. The first token the grammar rejects is replaced by the masked argmax
  of that row's logits, which is exactly what one-token-at-a-time masked greedy decoding would choose; if drafts after
  it had already been folded into the model state, the sequence re-prefills from the prefix cache.
- **Sampled (temperature > 0) strict requests are not speculated once the grammar is active.** The GPU's speculative
  sampler keeps 16 candidates per row, and correcting its draws afterwards does not give the masked distribution (an
  independent review showed P(a) = 0.5348 against the correct 0.5025 on a 17-token case). So a sampled request is
  speculated during its reasoning only; the block is cut right after `</think>`, and from there every token comes from
  the host sampler. Its contract: keep only the tokens the grammar allows, take the `top_k` of them by logit (1,024
  when `top_k` is 0, as the unconstrained host sampler does), apply the temperature, then the `top_p` nucleus, then
  draw.
- **Evidence for the sampled path** (`engine/tools/grammar_test.cc`, run by `build.sh`): 200,000 draws per case from
  `Matcher::sample`, compared with the exact masked distribution computed independently, Pearson chi-square, pass when
  chi2 < df + 6 sqrt(2 df) + 10:

  | Case | chi2 | df |
  |---|---:|---:|
  | the review's 17-token case (legal `"a"` in the top 16, legal `"b"` just outside, 15 illegal): P(a) = 0.5025 | 3.5 | 1 |
  | 8 digits, odd ones illegal, full distribution | 7.3 | 3 |
  | the same, `top_k` 2 | 0.9 | 1 |
  | `top_k` 3, temperature 0.7 | 2.0 | 2 |
  | `top_p` 0.6 | 0.9 | 1 |
  | `top_p` 0.9, `top_k` 3, temperature 1.3 | 5.1 | 2 |

  These test the host sampler that sampled strict requests now use for every constrained token. There is no live
  GPU frequency test of the speculative path, because that path no longer serves sampled strict tokens. Treat sampled
  strict mode as tested at the sampler level; greedy is the mode measured end to end above.
