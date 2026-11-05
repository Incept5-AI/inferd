#!/usr/bin/env python3
"""Plot a quick64.py run: wall power (--power-cmd readings) beside macOS's own power counters, over the run's phases.

  quick64_plot.py RESULT.jsonl [--line -1] [--mactop mactop.jsonl] [--out quick64.svg]

RESULT.jsonl: quick64.py's output (the last line by default). --mactop: `mactop --headless --format json -i 1000`
output captured on the machine under test (one JSON sample per line; cpu/gpu/ane/dram/total power from IOReport, no
sudo). Times are matched by wall clock, so the client and the server may be different Macs (keep their clocks in sync).
Writes a self-contained SVG (light and dark) with the prime and measured rounds shaded, each round's tok/s and tok/J.
"""
import argparse, datetime, html, json

W, H, L, R, T, B = 960, 420, 64, 200, 56, 64
SERIES = [  # (key in a mactop sample's soc_metrics, label, colour)
    ("total_power", "macOS: total (IOReport)", "#2a78d6"),
    ("gpu_power", "macOS: GPU", "#1aa37a"),
    ("cpu_power", "macOS: CPU", "#b06ad8"),
    ("dram_power", "macOS: DRAM", "#d9822b"),
]


def mactop_samples(path):
    out = []
    for ln in open(path, errors="replace"):
        ln = ln.strip().lstrip("[,").rstrip("],")  # one JSON array, a sample per line (and maybe no closing ])
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        for s in d if isinstance(d, list) else [d]:
            ts, m = s.get("timestamp"), s.get("soc_metrics") or {}
            if ts and m:
                t = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                out.append((t, m))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result")
    ap.add_argument("--line", type=int, default=-1)
    ap.add_argument("--mactop", default="")
    ap.add_argument("--out", default="quick64.svg")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    res = json.loads(open(a.result).read().strip().splitlines()[a.line])
    wall = res.get("power_trace") or []
    mt = mactop_samples(a.mactop) if a.mactop else []
    phases = res.get("phases") or []
    ts = [t for t, _ in wall] + [t for t, _ in mt] + [p["t0"] for p in phases] + [p["t1"] for p in phases]
    if not ts:
        raise SystemExit("quick64_plot: nothing to plot (no power trace, mactop samples or phases)")
    t0, t1 = min(ts), max(ts)
    vmax = max([w for _, w in wall] + [m.get(k, 0) or 0 for _, m in mt for k, *_ in SERIES] + [1])
    top = (int(vmax / 50) + 1) * 50
    pw, ph = W - L - R, H - T - B
    X = lambda t: L + pw * (t - t0) / max(1e-9, t1 - t0)
    Y = lambda v: T + ph * (1 - v / top)
    e = html.escape
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="-apple-system, Helvetica, Arial, sans-serif">',
         "<style>text{fill:#222}.g{stroke:#ddd}.ph{fill:#2a78d6;opacity:.07}.pr{fill:#888;opacity:.08}"
         "@media(prefers-color-scheme:dark){text{fill:#ddd}.g{stroke:#444}rect.bg{fill:#151515}}</style>",
         f'<rect class="bg" width="{W}" height="{H}" fill="#fff"/>']
    title = a.title or (f"{res['agents']} agents x {res['ctx']}: {res.get('agg_tps')} tok/s"
                        + (f", {res.get('power_w')} W at the wall, {res.get('tok_per_j')} tok/J" if res.get("power_w") else ""))
    o.append(f'<text x="{L}" y="26" font-size="16" font-weight="600">{e(title)}</text>')
    o.append(f'<text x="{L}" y="44" font-size="12">{e(res.get("model", ""))}, {e(res.get("ts", ""))}; '
             f'idle {res.get("idle_w")} W at the wall before the run</text>')
    for k in range(6):
        v = top * k / 5
        o.append(f'<line class="g" x1="{L}" x2="{L + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end">{v:.0f}</text>')
    o.append(f'<text x="16" y="{T + ph / 2}" font-size="12" transform="rotate(-90 16 {T + ph / 2})" text-anchor="middle">watts</text>')
    for p in phases:
        x0, x1 = X(p["t0"]), X(p["t1"])
        o.append(f'<rect class="{"pr" if p["name"] == "prime" else "ph"}" x="{x0:.1f}" y="{T}" width="{max(1, x1 - x0):.1f}" height="{ph}"/>'
                 f'<text x="{(x0 + x1) / 2:.1f}" y="{T + 14}" font-size="11" text-anchor="middle">{e(p["name"])}</text>')
    for r in res.get("rounds") or []:
        p = next((p for p in phases if p["name"] == f"round {r['round']}"), None)
        if p and r.get("agg_tps"):
            lab = f"{r['agg_tps']:.0f} tok/s" + (f", {r['tok_per_j']} tok/J" if r.get("tok_per_j") else "")
            o.append(f'<text x="{(X(p["t0"]) + X(p["t1"])) / 2:.1f}" y="{T + 28}" font-size="11" text-anchor="middle">{e(lab)}</text>')
    legend = []

    def line(pts, colour, width, label):
        pts = [(X(t), Y(v)) for t, v in pts if v is not None]
        if len(pts) > 1:
            o.append(f'<polyline fill="none" stroke="{colour}" stroke-width="{width}" points="'
                     + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
            legend.append((label, colour))

    line(wall, "#d62828", 2.4, "wall (plug)")
    for k, label, colour in SERIES:
        line([(t, m.get(k)) for t, m in mt], colour, 1.6, label)
    for i, (label, colour) in enumerate(legend):
        y = T + 10 + i * 20
        o.append(f'<rect x="{L + pw + 16}" y="{y - 9}" width="14" height="4" fill="{colour}"/>'
                 f'<text x="{L + pw + 36}" y="{y - 3}" font-size="12">{e(label)}</text>')
    dur = t1 - t0
    for k in range(0, int(dur) + 1, max(30, int(dur / 8) // 30 * 30 or 30)):
        x = X(t0 + k)
        o.append(f'<text x="{x:.1f}" y="{T + ph + 18}" font-size="11" text-anchor="middle">{k}s</text>')
    o.append(f'<text x="{L + pw / 2}" y="{H - 14}" font-size="12" text-anchor="middle">seconds from the start of the run'
             f' (shaded: priming the agents\' contexts, then the measured rounds)</text></svg>')
    open(a.out, "w").write("\n".join(o))
    print(a.out)


if __name__ == "__main__":
    main()
