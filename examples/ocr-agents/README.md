# ocr-agents

Four OCR agents run in parallel on the vision model. Each gets one scanned-looking document, sent inline as a `data:`
JPEG, and returns its fields as JSON. Every field is checked against the known ground truth. The documents are drawn
with Pillow, so the answers are known exactly:

| [invoice](samples/invoice.jpg) | [receipt](samples/receipt.jpg) | [shipping label](samples/shipping_label.jpg) | [letter](samples/letter.jpg) |
|---|---|---|---|
| vendor, number, date, customer, total, currency | store, date (printed 28/10/2026), time, total, card | name, street, city, postcode, tracking, weight | signed by, recipient, date (in words), subject |

```bash
~/.venvs/mlxvlm/bin/python ocr_agents.py              # the demo (needs Pillow)
~/.venvs/mlxvlm/bin/python ocr_agents.py --compare 3  # the test behind its settings: effort none/low/medium x 3 rounds
```

It needs a checkpoint **with the vision tower**. The mlx-community 4-bit build is text-only; use
`lmstudio-community` 4-bit (`~/.lmstudio/models/Qwen/Qwen3.8-27B-MLX-4bit`) or mlx-community 8-bit. Images must be inline
`data:` URLs: inferd does not fetch URLs or read files unless `INFERD_IMAGE_URLS=1` / `INFERD_IMAGE_FILES=1`.

## Sample output (MacBook Pro M5 Max, 1 November 2026)

```
4 OCR agents in parallel, effort none; max_context = image tokens + 110 + 150

receipt         max_context   876  used 685+71  5/5 fields  6.0s  finish=stop
   {"store": "GREENLEAF MARKET", "date": "2026-10-28", "time": "18:42", "total": "23.47", "card_brand": "VISA"}
letter          max_context   960  used 767+63  4/4 fields  6.0s  finish=stop
   {"signed_by": "Dr. Helen Marsh", "recipient_name": "Mr. Tomas Ruiz", "date": "2026-11-03", "subject": "Renewal of laboratory lease"}
shipping_label  max_context   764  used 578+93  6/6 fields  6.2s  finish=stop
   {"recipient": "Maria Okafor", "street": "88 Kingsland Road", "city": "London", "postcode": "E2 8DP", "tracking_number": "1Z 999 AA1 01 2345 6784", "weight_kg": "2.4"}
invoice         max_context   932  used 744+89  6/6 fields  6.2s  finish=stop
   {"vendor": "NORTHWIND COMPONENTS LTD", "invoice_number": "INV-2026-04817", "date": "2026-10-14", "customer": "Incept5 Labs", "total": "1,284.60", "currency": "EUR"}

21/21 fields correct, 6.4s wall for all four
```

## Sizing each agent's context

The cost of an image is known before you send it. Qwen3.x's vision tower uses 16-pixel patches merged 2×2, so the cost
is **one token per 32×32 pixels**. Each side is rounded to a multiple of 32, with a 64-token minimum.

| Document | Pixels | Image tokens (estimate) | `prompt_tokens` measured | Rest of the prompt |
|---|---|---:|---:|---:|
| invoice | 900×760 | 672 | 744 | 72 |
| receipt | 900×720 | 616 | 685 | 69 |
| shipping label | 900×560 | 504 | 578 | 74 |
| letter | 900×800 | 700 | 767 | 67 |

The rule the demo uses:

```
max_context = round(W/32) × round(H/32)        # image tokens
            + 110                              # template + instruction: 67–74 measured, +28 at effort low
            + output budget                    # 150 at effort none (largest reply measured: 99 tokens)
```

At effort low or medium the output budget has to cover the thinking as well. The largest replies we measured were 306
(low) and 341 (medium) tokens, so the script budgets 600 and 700. An agent sized this way cannot run away. If its
budget is too small, the reply stops with `finish_reason: "length"` and you see it on that line.

## Effort none vs low vs medium (the test behind the demo)

`--compare 3`: all four documents run concurrently, three rounds per effort, with temperature 0, scored field by
field (21 fields per round).

**Full resolution (900 px wide, 578–767 prompt tokens):**

| Effort | Fields correct | Wall time, 4 docs (3 rounds) | Output tokens per doc |
|---|---:|---|---|
| **none** | **63/63** | **4.6, 4.3, 6.5 s** | 58–94 |
| low | 63/63 | 9.5, 9.1, 8.9 s | 165–306 |
| medium | 63/63 | 9.6, 9.3, 9.6 s | 190–332 |

**Half resolution (450 px wide, 200–268 prompt tokens), same documents:**

| Effort | Fields correct | Wall time (3 rounds) | Misreads |
|---|---:|---|---|
| **none** | **63/63** | **3.2, 2.8, 3.0 s** | none |
| low | 61/63 | 5.8, 6.0, 6.2 s | "88 Kingsland Road" → "8 Kingsland Road" (2 rounds) |
| medium | 60/63 | 6.0, 6.6, 6.2 s | the same misread, every round |

What we can say:

- **At full resolution, thinking buys no accuracy and costs about 2× the time.** All three efforts read every field.
  The extra 100–250 tokens per document are reasoning about text that was already legible.
- **On a harder image, thinking was slightly worse.** At half resolution only effort none read every field. Low and
  medium "corrected" 88 to 8 in 5 of 6 rounds. That is one field on a small sample, so treat it as a hint rather than a
  result. It is consistent with the hypothesis that vision extraction works best with thinking off.
- **Half resolution was still exact at effort none**, at a third of the prompt tokens and half the time. For clean
  typed documents, downscaling is the cheaper lever. Check it on your own documents before relying on it.
- **Ambiguous field names fail before OCR does.** In a first run the letter scored 1/4 at every effort: the model put
  the letterhead under "sender", name plus company under "recipient", and converted "3 November 2026" to ISO
  (which the prompt allowed). The reading was correct; the spec was not. Renaming the fields `signed_by` and
  `recipient_name` and asking for every date as YYYY-MM-DD fixed it.

So the demo runs at **effort none**, full resolution, with the sizing rule above.

Repeated runs at temperature 0 are not always identical. Between two effort-none rounds the invoice reply went from
94 to 89 tokens and the letter's from 58 to 63, with the same fields. Concurrent batching and speculative
verification can change low-order bits, so run several rounds before trusting a single difference.
