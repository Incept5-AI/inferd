#!/usr/bin/env python3
# collect.py: pull every mini's benchmark queue results to ~/dev/inference/results/fleet-queue/<host>/ and build the
# progress report reports/m6-campaign-progress.html (queue state per box, the latest measurements, tables per job).
# Usage: python3 collect.py   (prints the report path; Telegram it from the session)
import html, json, pathlib, re, subprocess, time

HOSTS = ["mini-m6", "mini-m6-2", "mini-m5-pro", "mini-m5-pro-2", "mini-m4-pro"]
LOCAL = pathlib.Path.home() / "dev/inference/results/fleet-queue"
OUT = pathlib.Path.home() / "dev/inference/reports/m6-campaign-progress.html"
E = html.escape


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300).stdout


def pull(h):
    d = LOCAL / h
    d.mkdir(parents=True, exist_ok=True)
    sh(f"rsync -a --exclude 'mem-*.tsv' --exclude 'logs/' {h}:inference/results/queue/ {d}/results/ 2>/dev/null")
    sh(f"rsync -a {h}:inference/queue/queue.log {h}:inference/queue/summary.log {d}/ 2>/dev/null")
    st = {"pending": sh(f"ssh -o ConnectTimeout=6 {h} 'ls ~/inference/queue/pending 2>/dev/null'").split(),
          "running": sh(f"ssh -o ConnectTimeout=6 {h} 'ls ~/inference/queue/running 2>/dev/null'").split(),
          "failed": sh(f"ssh -o ConnectTimeout=6 {h} 'ls ~/inference/queue/failed 2>/dev/null'").split()}
    return d, st


def conc_table(path):
    """conc.py's text table -> rows (ctx, N, ok, err, wall, agg tok/s, TTFT med, TTFT max)."""
    rows = []
    for ln in path.read_text(errors="replace").splitlines():
        m = re.match(r"\s*(\d+k)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)", ln)
        if m:
            rows.append(m.groups())
    return rows


sections, state_rows = [], []
for h in HOSTS:
    d, st = pull(h)
    state_rows.append(f"<tr><td>{h}</td><td>{E(', '.join(x[:-3] for x in st['running']) or '–')}</td>"
                      f"<td>{len(st['pending'])}</td><td>{E(', '.join(x[:-3] for x in st['failed']) or '–')}</td></tr>")
    res = d / "results"
    for job in sorted(p for p in res.glob("*") if p.is_dir()):
        parts = []
        for f in sorted(job.glob("followup-*.txt")) + sorted(job.glob("single-*.txt")):
            rows = conc_table(f)
            if not rows:
                continue
            body = "".join("<tr>" + "".join(f"<td>{E(c)}</td>" for c in r) + "</tr>" for r in rows)
            parts.append(f"<h4>{E(f.stem)}</h4><table><tr><th>ctx</th><th>agents</th><th>ok</th><th>err</th><th>wall s</th>"
                         f"<th>total tok/s</th><th>TTFT med</th><th>TTFT max</th></tr>{body}</table>")
        for f in sorted(job.glob("scen-*.txt")) + sorted(job.glob("fleet-*.txt")):
            lines = [l for l in f.read_text(errors="replace").splitlines() if "n=" in l or l.startswith("[") or "turns" in l]
            if lines:
                parts.append(f"<h4>{E(f.stem)}</h4><pre>{E(chr(10).join(lines[:40]))}</pre>")
        prog = job / "progress.log"
        if prog.exists():
            parts.insert(0, f"<pre class='p'>{E(prog.read_text(errors='replace')[-3000:])}</pre>")
        if parts:
            sections.append(f"<section><h3>{E(h)} · {E(job.name)}</h3>{''.join(parts)}</section>")

summ = []
for h in HOSTS:
    f = LOCAL / h / "summary.log"
    if f.exists():
        summ += [(l[:19], h, l[20:]) for l in f.read_text(errors="replace").splitlines() if l.strip()]
summ.sort(reverse=True)
latest = "".join(f"<tr><td>{E(t)}</td><td>{E(h)}</td><td>{E(m[:400])}</td></tr>" for t, h, m in summ[:40])

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>M6 Campaign Progress</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mute:#6b6a65;--rule:#e2dfd7;--panel:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141413;--fg:#ecebe6;--mute:#a3a19a;--rule:#33322f;--panel:#1b1b1a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px 48px}} h1{{margin:0 0 4px}} .m{{color:var(--mute)}}
table{{border-collapse:collapse;font-size:.86rem;margin:6px 0 12px;display:block;overflow-x:auto}} th,td{{padding:5px 8px;border-bottom:1px solid var(--rule);text-align:left;white-space:nowrap}}
td:last-child{{white-space:normal}} th{{color:var(--mute)}} section{{border-top:1px solid var(--rule);padding-top:8px;margin-top:18px}}
pre{{font:12px ui-monospace,Menlo,monospace;background:var(--panel);border:1px solid var(--rule);padding:8px;overflow-x:auto;white-space:pre-wrap}}
</style></head><body><main>
<h1>M6 benchmark campaign: progress</h1>
<p class="m">Generated {time.strftime('%Y-%m-%d %H:%M')}. Every mini runs a job queue (one model at a time); Mac mini M6 is the target.
Engines: inferd 0.2.0 (prebuilt), TensorFold 1.2.2, Splash 1.5 where installed. Models: Qwen3.8-27B 4-bit + DFlash2, Qwen3.5-9B 4-bit and 8-bit.</p>
<h2>Queues</h2><table><tr><th>machine</th><th>running</th><th>pending</th><th>failed</th></tr>{''.join(state_rows)}</table>
<h2>Latest measurements</h2><table><tr><th>time</th><th>machine</th><th>result</th></tr>{latest}</table>
<h2>By job</h2>{''.join(sections) or '<p class="m">No finished measurements yet.</p>'}
</main></body></html>"""
OUT.write_text(page, encoding="utf-8")
print(OUT)
