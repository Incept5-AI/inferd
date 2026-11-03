#!/usr/bin/env python3
"""OCR agents: four synthetic documents (invoice, receipt, shipping label, letter), one concurrent vision request
each, fields extracted as JSON and checked against the known ground truth.

    python3 ocr_agents.py                    # the demo: effort none, max_context sized per image
    python3 ocr_agents.py --compare 2        # the test behind it: none / low / medium, 2 rounds each
Needs Pillow (pip install pillow) and a checkpoint with the vision tower (e.g. lmstudio-community 4-bit).
"""
import base64, io, json, math, os, re, sys, threading, time, urllib.error, urllib.request
from PIL import Image, ImageDraw, ImageFilter, ImageFont

URL = os.environ.get("INFERD_URL", "http://127.0.0.1:8300/v1")
FONT = {k: f"/System/Library/Fonts/{v}.ttc" for k, v in (("sans", "Helvetica"), ("mono", "Courier"), ("serif", "Times"))}
DOCS = {
    "invoice": ("sans", {"vendor": "Northwind Components Ltd", "invoice_number": "INV-2026-04817",
                         "date": "2026-10-14", "customer": "Incept5 Labs", "total": "1,284.60", "currency": "EUR"},
                ["NORTHWIND COMPONENTS LTD", "12 Harbour Road, Bristol BS1 4XX", "", "INVOICE", "",
                 "Invoice no: INV-2026-04817", "Date: 2026-10-14", "Bill to: Incept5 Labs", "",
                 "3 x USB-C dock          EUR 389.70", "5 x 2m TB5 cable        EUR 495.00",
                 "1 x Rack shelf          EUR 185.80", "VAT 20%                 EUR 214.10", "",
                 "TOTAL DUE               EUR 1,284.60", "Payment within 30 days."]),
    "receipt": ("mono", {"store": "GREENLEAF MARKET", "date": "2026-10-28", "time": "18:42", "total": "23.47",
                         "card_brand": "VISA"},
                ["   GREENLEAF MARKET", "  41 Rue Paradis, Marseille", "", "Date 28/10/2026", "Time 18:42",
                 "", "Baguette            1.30", "Tomatoes 1kg        3.95", "Olive oil 75cl     11.90",
                 "Comte 250g          6.32", "", "TOTAL              23.47", "Paid VISA ****4417", "", "   MERCI !"]),
    "shipping_label": ("sans", {"recipient": "Maria Okafor", "street": "88 Kingsland Road", "city": "London",
                                "postcode": "E2 8DP", "tracking_number": "1Z 999 AA1 01 2345 6784",
                                "weight_kg": "2.4"},
                       ["FROM: Inferd Hardware, 3 Mill Lane, Leeds LS1 2AB", "", "SHIP TO:", "Maria Okafor",
                        "88 Kingsland Road", "London E2 8DP", "UNITED KINGDOM", "", "WEIGHT: 2.4 kg   SERVICE: NEXT DAY",
                        "", "TRACKING: 1Z 999 AA1 01 2345 6784"]),
    "letter": ("serif", {"signed_by": "Dr. Helen Marsh", "recipient_name": "Mr. Tomas Ruiz", "date": "2026-11-03",
                         "subject": "Renewal of laboratory lease"},
               ["Marsh & Partners, 7 College Green, Oxford OX1 3BB", "", "3 November 2026", "", "Mr. Tomas Ruiz",
                "Ruiz Property Group", "", "Subject: Renewal of laboratory lease", "", "Dear Mr. Ruiz,", "",
                "Thank you for your letter of 20 October. We would like to renew",
                "the lease on Unit 4 for a further three years from 1 January.", "", "Yours sincerely,", "",
                "Dr. Helen Marsh"]),
}
# Sizing (measured with --compare, see README): the largest JSON reply was 99 tokens at effort none, 306 at low and
# 341 at medium (thinking included); the budgets give ~1.5-2x headroom.
OUTPUT_BUDGET = {"none": 150, "low": 600, "medium": 700}


def render(kind):
    """Draw the document; -> (JPEG bytes, width, height)."""
    style, _, lines = DOCS[kind]
    font = ImageFont.truetype(FONT[style], 26)
    w, h = 900, 120 + 40 * len(lines)
    img = Image.new("RGB", (w, h), (250, 249, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, w - 20, h - 20], outline=(60, 60, 60), width=3)
    for i, line in enumerate(lines):
        d.text((60, 60 + 40 * i), line, fill=(20, 20, 30), font=font)
    # Make it look scanned: a slight skew, a little blur and sensor noise.
    img = img.rotate(0.8 if len(lines) % 2 else -0.6, resample=Image.BICUBIC, fillcolor=(235, 233, 228))
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    img = Image.blend(img, Image.merge("RGB", [Image.effect_noise((w, h), 40)] * 3), 0.08)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue(), w, h


def image_tokens(w, h):
    """Qwen3.x vision: 16-px patches merged 2x2 -> one token per 32x32 pixels (sides rounded to 32)."""
    return round(h / 32) * round(w / 32)


def norm(v):
    """Field comparison: numbers by value (1,284.60 == 1284.6), text without case, spaces or punctuation."""
    s = str(v).strip().lower()
    try:
        return float(s.replace(",", "").replace(" ", ""))
    except ValueError:
        return re.sub(r"[^a-z0-9]", "", s)


def extract(kind, effort, max_context):
    jpg, w, h = render(kind)
    fields = list(DOCS[kind][1])
    body = {"model": "qwen3.8-27b", "temperature": 0, "max_tokens": max_context, "max_context": max_context,
            "chat_template_kwargs": {"enable_thinking": False} if effort == "none" else {"reasoning_effort": effort},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}},
                {"type": "text", "text": f"Extract these fields from the document as a JSON object: {fields}. "
                                         "Copy names, codes and numbers exactly as printed; write every date as "
                                         "YYYY-MM-DD; amounts as plain numbers. Reply with JSON only."}]}]}
    t = time.time()
    req = urllib.request.Request(URL + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        return {"kind": kind, "error": json.load(e)["error"]["message"], "secs": time.time() - t}
    text = r["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    try:
        got = json.loads(m.group(0)) if m else {}
    except ValueError:
        got = {}
    truth = DOCS[kind][1]
    ok = [f for f in truth if norm(got.get(f, "")) == norm(truth[f])]
    return {"kind": kind, "secs": time.time() - t, "usage": r["usage"], "finish": r["choices"][0]["finish_reason"],
            "got": got, "ok": len(ok), "n": len(truth), "wrong": [f for f in truth if f not in ok]}


def run_all(effort, sizing):
    """All four documents concurrently. sizing(kind) -> max_context."""
    out = {}
    ths = [threading.Thread(target=lambda k=k: out.__setitem__(k, extract(k, effort, sizing(k)))) for k in DOCS]
    t = time.time()
    [th.start() for th in ths]
    [th.join() for th in ths]
    return out, time.time() - t


TEXT_TOKENS = 110  # template + instruction: prompt_tokens - image tokens measured 67-74, +28 for effort low's line


def sized(effort):
    def rule(kind):
        _, w, h = render(kind)
        return image_tokens(w, h) + TEXT_TOKENS + OUTPUT_BUDGET[effort]
    return rule


if len(sys.argv) > 2 and sys.argv[1] == "--compare":
    print(f"{'effort':<7} {'round':>5} {'fields ok':>9} {'wall s':>7}  output tokens {list(DOCS)}  wrong")
    for effort in ("none", "low", "medium"):
        for rnd in range(int(sys.argv[2])):
            res, wall = run_all(effort, lambda k: 16384)  # generous cap while measuring
            ok = sum(r.get("ok", 0) for r in res.values()); n = sum(len(DOCS[k][1]) for k in DOCS)
            outs = [res[k]["usage"]["completion_tokens"] if "usage" in res[k] else "err" for k in DOCS]
            wrong = {k: r.get("wrong") or r.get("error") for k, r in res.items() if r.get("wrong") or r.get("error")}
            print(f"{effort:<7} {rnd + 1:>5} {ok:>4}/{n:<4} {wall:>7.1f} {str(outs):>22}  {wrong or ''}")
            if rnd == 0 and effort != "none":
                print(f"        max output this round: {max(o for o in outs if o != 'err')} tokens")
            if rnd == 0 and effort == "none":
                for k, r in res.items():
                    _, w, h = render(k)
                    print(f"        {k}: {w}x{h} px -> {image_tokens(w, h)} image tokens (estimate), "
                          f"prompt_tokens {r['usage']['prompt_tokens']}")
    sys.exit()

EFFORT = "none"
rule = sized(EFFORT)
print(f"4 OCR agents in parallel, effort {EFFORT}; max_context = image tokens + {TEXT_TOKENS} + {OUTPUT_BUDGET[EFFORT]}\n")
res, wall = run_all(EFFORT, rule)
for k, r in res.items():
    if "error" in r:
        print(f"{k}: ERROR {r['error']}"); continue
    u = r["usage"]
    print(f"{k:<15} max_context {rule(k):>5}  used {u['prompt_tokens']}+{u['completion_tokens']}  "
          f"{r['ok']}/{r['n']} fields  {r['secs']:.1f}s  finish={r['finish']}")
    print("   " + json.dumps(r["got"], ensure_ascii=False))
total = sum(r.get("ok", 0) for r in res.values()); n = sum(len(DOCS[k][1]) for k in DOCS)
print(f"\n{total}/{n} fields correct, {wall:.1f}s wall for all four")
