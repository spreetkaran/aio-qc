# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Data Studio profile report (Qt-free): one self-contained HTML file — overview, data-quality score, one card per column
(type, missing, distinct, statistics, a small chart, the warnings), strong correlations and the recipe that produced
the table. Light and dark themes follow the reader's system.
"""
from __future__ import annotations

import html
import time

from . import steps as S
from .profile import describe_value

CSS = """
:root{--bg:#fff;--fg:#1d2329;--muted:#5f6b76;--line:#e3e7eb;--card:#f7f9fa;--acc:#00838f;--warn:#b26a00;--bad:#c62828}
@media (prefers-color-scheme: dark){:root{--bg:#15191d;--fg:#e6eaee;--muted:#9aa6b1;--line:#2b3238;--card:#1c2227;
--acc:#4dd0e1;--warn:#ffb74d;--bad:#ef5350}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,
Roboto,sans-serif}main{max-width:1180px;margin:0 auto;padding:28px 20px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:17px;margin:28px 0 10px}.muted{color:var(--muted)}.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:130px}
.kpi b{display:block;font-size:20px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card h3{margin:0;font-size:15px;word-break:break-all}.tag{display:inline-block;font-size:11px;padding:1px 7px;
border-radius:9px;border:1px solid var(--line);color:var(--muted);margin-left:6px}table{border-collapse:collapse;width:100%}
td{padding:2px 0;vertical-align:top}td+td{text-align:right}.w{color:var(--warn)}.bad{color:var(--bad)}
svg{display:block;margin:8px 0}ol{padding-left:22px}code{font-size:12.5px}
"""


def _svg_hist(counts, w=300, h=60, color="var(--acc)") -> str:
    if not counts:
        return ""
    m = max(counts) or 1
    bw = w / len(counts)
    bars = "".join(f"<rect x='{i * bw:.1f}' y='{h - c / m * h:.1f}' width='{max(1.0, bw - 1):.1f}' "
                   f"height='{c / m * h:.1f}' fill='{color}' opacity='.85'/>" for i, c in enumerate(counts))
    return f"<svg viewBox='0 0 {w} {h}' width='100%' height='{h}' preserveAspectRatio='none'>{bars}</svg>"


def _svg_top(top, w=300) -> str:
    if not top:
        return ""
    m = max(n for _v, n in top) or 1
    rows = []
    for i, (v, n) in enumerate(top[:6]):
        y = i * 18
        label = html.escape(str(v))[:28]
        rows.append(f"<rect x='0' y='{y + 2}' width='{n / m * (w - 120):.1f}' height='13' fill='var(--acc)' "
                    f"opacity='.75'/><text x='{n / m * (w - 120) + 4:.1f}' y='{y + 13}' font-size='11' "
                    f"fill='currentColor'>{label} · {n:,}</text>")
    return f"<svg viewBox='0 0 {w} {len(top[:6]) * 18 + 4}' width='100%'>{''.join(rows)}</svg>"


def build(name: str, prof: dict, steps: list, source: str = "", sampled: bool = False) -> str:
    e = html.escape
    cards = []
    for c in prof["columns"]:
        rows = [("type", c.kind), ("missing", f"{c.missing:,} ({c.missing_pct:.1%})"), ("distinct", f"{c.distinct:,}")]
        st = c.stats
        if c.kind in ("number", "integer") and st:
            rows += [("min · median · max", f"{describe_value(st['min'])} · {describe_value(st['median'])} · "
                                             f"{describe_value(st['max'])}"),
                     ("mean ± std", f"{describe_value(st['mean'])} ± {describe_value(st['std'])}"),
                     ("outliers (IQR)", f"{st['outliers']:,}")]
        elif c.kind == "date" and st:
            rows += [("from", st["min"]), ("to", st["max"])]
        elif st:
            rows += [("length", f"{st['min_len']}–{st['max_len']} (avg {st['mean_len']:.1f})")]
        chart = _svg_hist(c.hist[0]) if c.hist else _svg_top(c.top)
        warn = "".join(f"<div class='w'>⚠ {e(t)}</div>" for _k, t in c.warnings)
        tbl = "".join(f"<tr><td class='muted'>{e(a)}</td><td>{e(str(b))}</td></tr>" for a, b in rows)
        cards.append(f"<div class='card'><h3>{e(c.name)}<span class='tag'>{e(c.dtype)}</span></h3>{chart}"
                     f"<table>{tbl}</table>{warn}</div>")
    corr = "".join(f"<tr><td>{e(a)} ↔ {e(b)}</td><td>{r:+.2f}</td></tr>" for a, b, r in prof["correlations"])
    rec = "".join(f"<li>{'' if st.get('on', True) else '<s>'}{e(S.OPS[st['op']].describe(st['p']))}"
                  f"{'' if st.get('on', True) else '</s> (off)'}</li>" for st in steps) or "<li>none</li>"
    kpis = [("Rows", f"{prof['rows']:,}"), ("Columns", prof["cols"]), ("Missing cells", f"{prof['missing_cells']:,}"),
            ("Duplicate rows", f"{prof['duplicate_rows']:,}"), ("Quality score", f"{prof['score']} / 100")]
    k = "".join(f"<div class='kpi'><span class='muted'>{a}</span><b>{b}</b></div>" for a, b in kpis)
    note = " · statistics from a 200,000-row sample" if prof.get("sampled") else ""
    note += " · the app showed a sample of the file" if sampled else ""
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>{e(name)} — data profile</title><style>{CSS}</style></head><body><main>"
            f"<h1>{e(name)}</h1><div class='muted'>{e(source)} · {time.strftime('%Y-%m-%d %H:%M')}{note}</div>"
            f"<div class='kpis'>{k}</div><h2>Columns</h2><div class='grid'>{''.join(cards)}</div>"
            f"<h2>Strong correlations (|r| ≥ 0.5)</h2>"
            f"{'<table>' + corr + '</table>' if corr else '<p class=muted>none</p>'}"
            f"<h2>Recipe</h2><ol>{rec}</ol><p class='muted'>AIO QC · Data Studio</p></main></body></html>")
