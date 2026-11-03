# tools-web-search

A research agent in about 100 lines. The model gets one tool, `web_search` (Wikipedia's search and page-summary APIs,
no key), and loops: tool calls, tool results, then an answer with sources. The budget is tight: **`max_context` 8192,
reasoning effort `low`**.

```bash
python3 web_search_agent.py                                  # the default question
python3 web_search_agent.py "Who designed the Sydney Opera House, and when did it open?"
```

Start the server first (see [../README.md](../README.md#start-the-server)). Python standard library only.
It needs the front end with **tool-call parsing** (`engine/frontend/serve.py`, November 2026): older front ends return
the raw `<tool_call>` markup as text.

## What it shows

- **OpenAI-style tool calling.** Send `tools`. The reply carries `message.tool_calls` (each with `id`, `type`, and
  `function.name` and `function.arguments` as a JSON string) and `finish_reason: "tool_calls"`. Send results back as
  `{"role": "tool", "tool_call_id": ..., "content": ...}` messages. Any OpenAI client works the same way.
- **Parallel calls.** The model issues two searches in one turn.
- **The cache across turns.** Each loop iteration is a follow-up turn, so it resumes from the previous prompt
  (`cached` below).
- **Network failure.** A failed search returns `{"error": ...}` as the tool result, and the model says it could not
  verify instead of crashing the loop.

## Sample output (MacBook Pro M5 Max, 1 November 2026)

```
Q: How tall is the Burj Khalifa, and which building held the title of world's tallest before it?
   (max_context 8192, effort low)

step 1: model calls 2 tool(s) [context 460/8192, cached 0]  finish_reason=tool_calls
  -> web_search({"query": "Burj Khalifa height world's tallest building"})
  <- History of the world's tallest buildings, Burj Khalifa, List of tallest buildings
  -> web_search({"query": "building world's tallest before Burj Khalifa"})
  <- Burj Khalifa, History of the world's tallest buildings, List of tallest residential buildings
step 2: final answer [context 1312/8192, cached 320]  finish_reason=stop

The Burj Khalifa in Dubai stands 828 m (2,717 ft) to its roof, making it the world's tallest building since its 2011 inauguration. Before it, the title was held by **Taipei 101** in Taiwan, which had been the record-holder since 2004. (Source: Wikipedia, "Burj Khalifa" and "History of the world's tallest buildings.")

7.5s total
```

With the network cut off (`https_proxy=http://127.0.0.1:9`):

```
step 1: model calls 2 tool(s) [context 460/8192, cached 320]  finish_reason=tool_calls
  -> web_search({"query": "Burj Khalifa height world's tallest building"})
  <- search unavailable: <urlopen error [Errno 61] Connection refused>
  -> web_search({"query": "building world's tallest before Burj Khalifa"})
  <- search unavailable: <urlopen error [Errno 61] Connection refused>
step 2: final answer [context 715/8192, cached 320]  finish_reason=stop

The Burj Khalifa in Dubai stands 828 meters (2,717 feet) tall, and before it the title of world's tallest building was held by Taipei 101 (508 m / 1,668 ft), which had held the record since 2004 until the Burj Khalifa's completion in 2010. Note: I was unable to verify these facts via web search (the search service was unavailable), so I'm relying on established knowledge rather than a live source.

5.4s total
```

## Notes

- The whole run used 1.3k of its 8k budget. Each search returns at most 3 summaries of 1,200 characters, which keeps
  a few rounds well inside 8k. If you allow more steps, an over-cap prompt comes back as HTTP 400
  `context_length_exceeded`, and the script stops with that message.
- Streaming works with tools too: text streams normally, and the tool calls arrive whole in one final delta before
  `finish_reason: "tool_calls"`. This demo uses `stream: false` for simplicity.
