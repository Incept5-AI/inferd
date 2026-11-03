#!/usr/bin/env python3
"""make_charts.py -- static SVG charts for docs/RESULTS.md, built from the raw result files (Python 3.9+, stdlib only).

Every chart reads the raw rows under results/ and writes one self-contained SVG to docs/img/. Each SVG has an explicit
white background (so it reads on light and dark pages), labelled axes with units, a legend, and a footer naming the
source files and the date of the data.

  python3 docs/make_charts.py            # writes docs/img/*.svg and checks each one parses as XML
"""
import glob
import json
import os
import statistics
import sys
from collections import defaultdict
from xml.dom import minidom
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "img")

# One colour per entity, fixed across every chart (categorical slots in a fixed order, never cycled).
COLOR = {
    "inferd": "#2a78d6",          # blue
    "splash": "#eb6834",          # orange
    "inferd_ane": "#1baf7a",      # aqua
    "kapitsa": "#eda100",         # yellow
    "mlxserve": "#e87ba4",        # magenta
}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#77766f", "#e4e3df"
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------- data helpers
def rows(path):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, path))):
        for line in open(p):
            line = line.strip()
            if line.startswith("{"):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def agents(path, label=None):
    """conc.py follow-up rows -> {ctx: {n: row}} (last row wins)."""
    t = defaultdict(dict)
    for r in rows(path):
        if "agg_tps" not in r or (label and r.get("label") != label):
            continue
        if r.get("mode") == "followup" or str(r.get("label", "")).startswith("pair:"):
            t[r["ctx"]][r["n"]] = r
    return t


def sweep(path, label=None):
    """speed.py rows -> {size: {prefill, decode, ttft, date}} (medians over reps, error rows excluded)."""
    t = defaultdict(list)
    for r in rows(path):
        if r.get("error") or "decode_tps" not in r or (label and r.get("label") != label):
            continue
        t[r["size"]].append(r)
    return {k: dict(prefill=statistics.median(x["prefill_tps"] for x in v),
                    decode=statistics.median(x["decode_tps"] for x in v),
                    ttft=statistics.median(x["ttft_s"] for x in v),
                    date=(v[0].get("ts") or "")[:10]) for k, v in t.items()}


def dates(rs):
    ds = sorted({(r.get("ts") or "")[:10] for r in rs if r.get("ts")})
    return ds[0] if len(ds) == 1 else (f"{ds[0]} to {ds[-1]}" if ds else "")


# ---------------------------------------------------------------- SVG primitives
def fmt(v):
    return f"{v:,.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}".rstrip("0").rstrip("."))


def nice_top(v):
    for step in (1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000):
        if v / step <= 6:
            return step * (int(v / step) + 1), step
    return v * 1.1, v / 5


def text(x, y, s, size=12, anchor="start", fill=INK, weight="normal", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
            f'font-weight="{weight}" {extra}>{escape(str(s))}</text>')


def text_w(s, size):
    return 0.56 * size * len(str(s))


def wrap(s, size, width):
    """Greedy word wrap by estimated text width."""
    out, cur = [], ""
    for word in str(s).split(" "):
        cand = (cur + " " + word).strip()
        if cur and text_w(cand, size) > width:
            out.append(cur)
            cur = word
        else:
            cur = cand
    return out + ([cur] if cur else [])


class Layout:
    """Header (title, wrapped subtitle, wrapped legend) and footer heights for a chart of width w."""

    def __init__(self, w, title, subtitle, footer, legend):
        self.w = w
        self.sub = wrap(subtitle, 12, w - 40)
        self.leg = []  # rows of (x, name, color)
        row, x = [], 20
        for name, color in legend:
            iw = 17 + text_w(name, 12) + 22
            if row and x + iw > w - 20:
                self.leg.append(row)
                row, x = [], 20
            row.append((x, name, color))
            x += iw
        if row:
            self.leg.append(row)
        self.foot = [l for f in footer for l in wrap(f, 10, w - 40)]
        self.top = 47 + 16 * (len(self.sub) - 1) + 12 + 20 * len(self.leg) + 22  # plot area starts here
        self.bottom = 14 * len(self.foot) + 12  # footer height

    def render(self, h, title, body):
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{h}" viewBox="0 0 {self.w} {h}" '
               f'font-family="{FONT}" role="img" aria-label="{escape(title, {chr(34): "&quot;"})}">',
               f'<title>{escape(title)}</title>',
               f'<rect x="0" y="0" width="{self.w}" height="{h}" fill="#ffffff"/>',
               text(20, 28, title, 16, weight="600")]
        for i, line in enumerate(self.sub):
            out.append(text(20, 47 + 16 * i, line, 12, fill=INK2))
        ly = 47 + 16 * (len(self.sub) - 1) + 14
        for r, row in enumerate(self.leg):
            for x, name, color in row:
                out.append(f'<rect x="{x:.1f}" y="{ly + 20 * r:.1f}" width="12" height="12" rx="2" fill="{color}"/>')
                out.append(text(x + 17, ly + 20 * r + 11, name, 12, fill=INK2))
        out += body
        for i, line in enumerate(self.foot):
            out.append(text(20, h - 12 - 14 * (len(self.foot) - 1 - i), line, 10, fill=MUTED))
        out.append("</svg>")
        return "\n".join(out)


def y_axis(x0, x1, y0, y1, top, step, label):
    out = []
    v = 0.0
    while v <= top + 1e-9:
        y = y1 - (y1 - y0) * v / top
        out.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(text(x0 - 6, y + 4, f"{v:,.0f}" if v == int(v) else f"{v:g}", 11, "end", MUTED))
        v += step
    out.append(text(16, (y0 + y1) / 2, label, 12, "middle", INK2,
                    extra=f'transform="rotate(-90 16 {(y0 + y1) / 2:.1f})"'))
    out.append(f'<line x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}" stroke="{INK2}" stroke-width="1"/>')
    return out


def bar_chart(fname, title, subtitle, groups, series, ylabel, xlabel, footer, notes=None, w=860, h=300, labels=True):
    """Grouped bars. series = [(name, color, [value or None per group])]; notes = {(series_i, group_i): 'text'}."""
    notes = notes or {}
    vals = [v for _, _, vs in series for v in vs if v]
    top, step = nice_top(max(vals) * 1.08)
    L = Layout(w, title, subtitle, footer, [(n, c) for n, c, vs in series if any(vs)])
    h = L.top + h + 60 + L.bottom  # h is the plot height
    x0, x1, y0, y1 = 70, w - 20, L.top, h - L.bottom - 60
    body = y_axis(x0, x1, y0, y1, top, step, ylabel)
    gw = (x1 - x0) / len(groups)
    ns = len(series)
    bw = min(40, (gw - 16) / ns)
    for gi, g in enumerate(groups):
        gx = x0 + gi * gw + (gw - bw * ns) / 2
        for si, (name, color, vs) in enumerate(series):
            v = vs[gi]
            bx = gx + si * bw
            if v is None:
                if (si, gi) in notes:
                    body.append(text(bx + (bw - 2) / 2 + 3, y1 - 6, notes[(si, gi)], 10, "start", MUTED,
                                     extra=f'transform="rotate(-90 {bx + (bw - 2) / 2 + 3:.1f} {y1 - 6})"'))
                continue
            bh = (y1 - y0) * v / top
            by = y1 - bh
            r = min(4, (bw - 2) / 2, bh)
            # rounded data end, square at the baseline
            body.append(f'<path d="M{bx:.1f},{y1} V{by + r:.1f} Q{bx:.1f},{by:.1f} {bx + r:.1f},{by:.1f} '
                        f'H{bx + bw - 2 - r:.1f} Q{bx + bw - 2:.1f},{by:.1f} {bx + bw - 2:.1f},{by + r:.1f} V{y1} Z" '
                        f'fill="{color}"><title>{escape(f"{name}, {g}: {fmt(v)}")}</title></path>')
            if labels and bw >= 15:
                body.append(text(bx + (bw - 2) / 2, by - 4, fmt(v), 9 if bw < 26 else 10, "middle", INK))
            if (si, gi) in notes:
                body.append(text(bx + (bw - 2) / 2, y1 + 26, notes[(si, gi)], 9, "middle", MUTED))
        body.append(text(x0 + gi * gw + gw / 2, y1 + 15, g, 12, "middle", INK2))
    body.append(text((x0 + x1) / 2, y1 + 44, xlabel, 12, "middle", INK2))
    write(fname, L.render(h, title, body))


def line_chart(fname, title, subtitle, xs, xlabels, series, ylabel, xlabel, footer, w=860, h=300):
    """Lines with markers. xs = numeric x positions (already transformed); series = [(name, color, [(x, y)])]."""
    vals = [y for _, _, pts in series for _, y in pts]
    top, step = nice_top(max(vals) * 1.08)
    L = Layout(w, title, subtitle, footer, [(n, c) for n, c, _ in series])
    h = L.top + h + 54 + L.bottom
    x0, x1, y0, y1 = 70, w - 40, L.top, h - L.bottom - 54
    body = y_axis(x0, x1, y0, y1, top, step, ylabel)
    lo, hi = min(xs), max(xs)
    px = lambda x: x0 + 20 + (x1 - x0 - 40) * (x - lo) / (hi - lo if hi > lo else 1)
    py = lambda y: y1 - (y1 - y0) * y / top
    for x, lab in zip(xs, xlabels):
        body.append(text(px(x), y1 + 16, lab, 11, "middle", INK2))
    for name, color, pts in series:
        pts = sorted(pts)
        if len(pts) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(pts))
            body.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
        for x, y in pts:
            body.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4.5" fill="{color}" stroke="#ffffff" '
                        f'stroke-width="2"><title>{escape(f"{name}: {fmt(y)}")}</title></circle>')
        if pts:  # direct label at the line end
            x, y = pts[-1]
            body.append(text(px(x) + 8, py(y) + 4, fmt(y), 10, "start", INK))
    body.append(text((x0 + x1) / 2, y1 + 38, xlabel, 12, "middle", INK2))
    write(fname, L.render(h, title, body))


WRITTEN = []


def write(fname, svg):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, fname)
    with open(path, "w") as f:
        f.write(svg)
    WRITTEN.append(path)


# ---------------------------------------------------------------- charts
def fleet_charts():
    """agents.py fleets on the Mac minis, 2026-11-01: turns per minute, inferd vs Splash 1.5.0."""
    boxes = {"m6": "Mac mini M6 (32 GB)", "m5p": "Mac mini M5 Pro (64 GB)"}
    for box, ctx in (("m6", "8k"), ("m6", "16k"), ("m5p", "16k"), ("m5p", "32k")):
        files = {"inferd": f"results/night/{box}/a64-inferd-{box}-{ctx}.jsonl",
                 "inferd_ane": f"results/night/{box}/a64-inferdane24-{box}-{ctx}.jsonl",
                 "splash": f"results/night/{box}/a64-splash-{box}-{ctx}.jsonl"}
        cells = {k: {r["n"]: r for r in rows(p) if r.get("kind") == "cell"} for k, p in files.items()}
        ns = sorted({n for c in cells.values() for n in c})
        names = {"inferd": "inferd (GPU)", "inferd_ane": "inferd + Neural Engine split (24 units)",
                 "splash": "Splash 1.5.0"}
        series, notes = [], {}
        for si, k in enumerate(k for k in ("inferd", "inferd_ane", "splash") if cells[k]):
            vs = []
            for gi, n in enumerate(ns):
                r = cells[k].get(n)
                vs.append(r["turns_per_min"] if r else None)
                if r and r.get("err"):
                    notes[(si, gi)] = f"{r['err']} of {r['turns']} refused"
                elif not r:
                    notes[(si, gi)] = "not run"
            series.append((names[k], COLOR[k], vs))
        used = [os.path.relpath(os.path.join(ROOT, p), ROOT) for k, p in files.items() if cells[k]]
        bar_chart(f"fleet-{box}-{ctx}.svg",
                  f"Agent fleet, {boxes[box]}, {ctx} contexts",
                  "Qwen3.8-27B 4-bit + DFlash2 on both engines. agents.py, 2 rounds of ~1.2k-token turns, all agents in "
                  "flight. Higher is better.",
                  [f"{n} agents" for n in ns], series, "completed turns per minute", "agents in the fleet",
                  ["Source: " + ", ".join(used), "Data: overnight 2026-10-31/11-01 (the study log, 'Overnight agent-fleet "
                   "results'). Splash's refused turns are HTTP 503 and are not counted as completed."], notes)


STUDIO_SERIES = [  # (key, display name, file glob, label filter)
    ("inferd", "inferd, 1 Nov build", "results/cmp/inferd14-agents.jsonl", None),
    ("splash", "Splash 1.5.0", "results/cmp/splash-agents.jsonl", None),
    ("kapitsa", "Kapitsa 0.2.24", "results/cmp/kapitsa-agents.jsonl", None),
    ("mlxserve", "mlx-serve 26.10.2", "results/cmp/mlxserve-agents.jsonl", None),
]


def studio_charts():
    data = {k: agents(p, lab) for k, _, p, lab in STUDIO_SERIES}
    for ctx in ("16k", "32k", "64k", "128k", "256k"):
        present = [(k, name) for k, name, _, _ in STUDIO_SERIES if data[k].get(ctx)]
        ns = [n for n in (1, 4, 8, 16, 32) if any(n in data[k][ctx] for k, _ in present)]
        series = [(name, COLOR[k], [data[k][ctx].get(n, {}).get("agg_tps") for n in ns]) for k, name in present]
        if not series:
            continue
        used = [p for k, _, p, _ in STUDIO_SERIES if data[k].get(ctx)]
        bar_chart(f"studio-agents-{ctx}.svg",
                  f"Qwen3.8-27B agents at {ctx}: total tok/s, Mac Studio M5 Ultra",
                  "conc.py --mode followup: each agent's context primed, then all N send a follow-up at once, 512 "
                  "tokens each. End-to-end tok/s (prefill included). Higher is better.",
                  [f"N={n}" for n in ns], series, "total tokens per second", "concurrent agents", [
                      "Source: " + ", ".join(sorted(set(used))),
                      "Data: Splash, Kapitsa and mlx-serve 2026-10-30; inferd 2026-11-01 (one sweep, single runs, see "
                      "RESULTS.md for the long-uptime caveat)" + ". A missing bar was not run."],
                  w=900, labels=len(series) <= 3)


def sweep_charts():
    """speed.py single stream on the Studio: decode and prefill against context."""
    sizes = ["8k", "16k", "32k", "64k", "128k", "256k"]
    src = [("inferd", "inferd, 31 Oct", "results/cmp/inferd6-sweep.jsonl"),
           ("splash", "Splash 1.5.0", "results/cmp/splash-sweep.jsonl"),
           ("kapitsa", "Kapitsa 0.2.24", "results/cmp/kapitsa-sweep.jsonl"),
           ("mlxserve", "mlx-serve 26.10.2", "results/cmp/mlxserve-sweep.jsonl")]
    data = {k: sweep(p) for k, _, p in src}
    for metric, ylabel, title in (("decode", "decode tokens per second", "Single-stream decode against context"),
                                  ("prefill", "prefill tokens per second", "Single-stream prefill against context")):
        series = [(name, COLOR[k], [(i, data[k][s][metric]) for i, s in enumerate(sizes) if s in data[k]])
                  for k, name, _ in src if data[k]]
        line_chart(f"studio-sweep-{metric}.svg", f"{title} (Qwen3.8-27B, cold prompts)",
                   "speed.py: a nonce at the start of every prompt, greedy, 512 tokens out, median of 2 cold runs. "
                   "Mac Studio M5 Ultra.",
                   list(range(len(sizes))), sizes, series, ylabel, "prompt length (tokens)",
                   ["Source: " + ", ".join(p for _, _, p in src),
                    "Data: 2026-10-30 (Splash, Kapitsa, mlx-serve), 2026-10-31 (inferd). "
                    "inferd at 256k on the later 0.1.0 build: 112 tok/s decode (results/cmp/inferd10-sweep.jsonl)."])


def ane_chart():
    series = []
    for box, name, key in (("m6", "Mac mini M6", "inferd_ane"), ("m5p", "Mac mini M5 Pro", "inferd")):
        pts = []
        for line in open(os.path.join(ROOT, f"results/night/{box}/ane-prefill.log")):
            f = line.split()
            if len(f) == 2 and f[0].isdigit() and f[1].isdigit():
                pts.append((int(f[0]), int(f[1])))
        series.append((name, COLOR[key], pts))
    xs = sorted({x for _, _, p in series for x, _ in p})
    line_chart("ane-prefill.svg", "Neural Engine FFN split: prefill throughput by units on the Neural Engine",
               "decode_test prefill, Qwen3.8-27B 4-bit, 8k prompt in 1,024-token chunks. 0 units = GPU alone; each "
               "unit is 512 of the FFN's 17,408 channels.",
               xs, [str(x) for x in xs], series, "prefill tokens per second", "units on the Neural Engine (of 34)",
               ["Source: results/night/m6/ane-prefill.log, results/night/m5p/ane-prefill.log",
                "Data: 2026-11-01 (one run per point). Microbenchmark, no server: the in-server gain differs "
                "(the study log, 'Neural Engine FFN split')."])


MODELS = [  # (display, speed file, speed label, conc file, conc label)
    ("Qwen3.5-9B", "results/models/studio/speed-q9.jsonl", "q9-inferd", "results/models/studio/conc-q9.jsonl", "q9-fix"),
    ("Qwen3.6-35B-A3B", "results/models/studio/speed-q35moe.jsonl", "q35moe-inferd2",
     "results/models/studio/conc-q35moe.jsonl", "q35moe-fix"),
    ("Muse Glimmer 30B", "results/models/studio/speed-glimmer.jsonl", "glimmer-inferd",
     "results/models/studio/conc-glimmer.jsonl", "glimmer-fix"),
    ("Gemma-4-31B", "results/models/studio/speed-gemma31.jsonl", "gemma31-inferd",
     "results/models/studio/conc-gemma31.jsonl", "gemma31-inferd"),
]


def model_charts():
    names = [m[0] for m in MODELS]
    sp = [sweep(s, sl) for _, s, sl, _, _ in MODELS]
    bar_chart("models-studio-decode.svg", "Other models on inferd, Mac Studio M5 Ultra: single-user decode",
              "speed.py, cold prompts, greedy, 512 tokens out, no drafter (speculation off). Median of 2 runs "
              "(Qwen3.6-35B-A3B: the second session).",
              names, [("0.5k-token prompt", COLOR["inferd"], [s["0.5k"]["decode"] for s in sp]),
                      ("32k-token prompt", COLOR["splash"], [s["32k"]["decode"] for s in sp])],
              "decode tokens per second", "model", ["Source: " + ", ".join(m[1] for m in MODELS),
                                                    "Data: 2026-11-01. Smoke ports, not tuned."], w=860)
    agg, notes = [], {}
    for gi, (_, _, _, c, cl) in enumerate(MODELS):
        a = agents(c, cl)
        r = a.get("32k", {}).get(16)
        if r is None and a.get("8k", {}).get(16):
            r = a["8k"][16]
            notes[(0, gi)] = "8k contexts"
        agg.append(r["agg_tps"] if r else None)
    bar_chart("models-studio-agents.svg", "Other models on inferd, Mac Studio M5 Ultra: 16 agents",
              "conc.py --mode followup, 16 agents at 32k each (Gemma-4-31B at 8k: its bf16 KV fits ~5 agents at 32k), "
              "512 tokens each. End-to-end tok/s.",
              names, [("16 agents, total tok/s", COLOR["inferd"], agg)], "total tokens per second", "model",
              ["Source: " + ", ".join(m[3] for m in MODELS), "Data: 2026-11-01. Single runs."], notes, w=860)
    # single-user decode at 4k on every machine that ran the model
    boxes = [("Mac Studio M5 Ultra", "studio", COLOR["inferd"]), ("Mac mini M5 Pro", "m5-pro", COLOR["splash"]),
             ("Mac mini M6", "m6", COLOR["inferd_ane"])]
    keys = {"Qwen3.5-9B": "q9", "Qwen3.6-35B-A3B": "q35moe", "Muse Glimmer 30B": "glimmer", "Gemma-4-31B": "gemma31"}
    series, notes = [], {}
    for si, (bname, bdir, color) in enumerate(boxes):
        vs = []
        for gi, (m, s, sl, _, _) in enumerate(MODELS):
            if bdir == "studio":
                d = sweep(s, sl)
            else:
                d = sweep(f"results/models/{bdir}/speed-{keys[m]}-{bdir}.jsonl", f"{keys[m]}-{bdir}")
            v = d.get("4k", {}).get("decode")
            vs.append(v)
            if v is None:
                notes[(si, gi)] = "pending"
        series.append((bname, color, vs))
    bar_chart("models-machines-decode.svg", "Other models on inferd: single-user decode at 4k, by machine",
              "speed.py, cold 4k-token prompt, greedy, 512 tokens out, no drafter. Minis: one run per cell.",
              names, series, "decode tokens per second", "model",
              ["Source: results/models/{studio,m5-pro,m6}/speed-*.jsonl",
               "Data: 2026-11-01. 'pending' = no result file yet."], notes, w=860)


def main():
    fleet_charts()
    studio_charts()
    sweep_charts()
    ane_chart()
    model_charts()
    bad = 0
    for p in WRITTEN:
        try:
            minidom.parse(p)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"INVALID {p}: {e}")
    for p in WRITTEN:
        print(f"wrote {os.path.relpath(p, ROOT)} ({os.path.getsize(p) // 1024} KB)")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
