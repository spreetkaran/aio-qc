# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
QC presets (Qt-free): a named set of checks, each with its settings and a limit, run on a dataset with one click —
the same checks, the same way, on every delivery — and a pass / fail report for the people who fix it.

    presets = load()                                     app-wide (<data home>/qc_presets.json)
    p = {"name": "LPR delivery", "note": "...", "checks": [
            {"tool": "validation", "title": "Validation", "params": {...}, "max_files": 0, "count": "any"}, ...]}
    res = evaluate(check, rows)          rows: [(file key, severity, text)] -> {"files", "findings", "passed", ...}
    paths = write_report(out_dir, preset, results, dataset)   HTML (for people) + CSV (one line per finding)

A check passes when at most `max_files` files have a finding — counting every finding ("any") or only the errors
("errors"). Presets can be exported to a .json file and imported on another computer (Share).
"""
from __future__ import annotations

import csv
import html
import json
import os
import time
from collections import Counter
from pathlib import Path

FILE = "qc_presets.json"
VERSION = 1


def _file() -> Path:
    from .appdirs import data_home
    return data_home() / FILE


def load() -> list[dict]:
    try:
        d = json.loads(_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [p for p in (d.get("presets") or []) if isinstance(p, dict) and p.get("name")]


def save(presets: list[dict]) -> None:
    p = _file()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": VERSION, "presets": presets}, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def unique_name(name: str, presets: list[dict]) -> str:
    names = {p["name"].lower() for p in presets}
    if name.lower() not in names:
        return name
    k = 2
    while f"{name} ({k})".lower() in names:
        k += 1
    return f"{name} ({k})"


def export_preset(preset: dict, path) -> None:
    Path(path).write_text(json.dumps({"format": "aio-qc-preset", "version": VERSION, "preset": preset}, indent=1,
                                     ensure_ascii=False), encoding="utf-8")


def import_preset(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    p = d.get("preset") if isinstance(d, dict) and d.get("format") == "aio-qc-preset" else d
    if not isinstance(p, dict) or not p.get("name") or not isinstance(p.get("checks"), list):
        raise ValueError("not an AIO QC preset file")
    for c in p["checks"]:
        if not isinstance(c, dict) or not c.get("tool"):
            raise ValueError("a check without a tool in the preset file")
        c.setdefault("params", {})
        c.setdefault("max_files", 0)
        c.setdefault("count", "any")
    return p


def new_check(tool: str, title: str, params: dict | None = None) -> dict:
    return {"tool": tool, "title": title, "params": dict(params or {}), "max_files": 0, "count": "any"}


def evaluate(check: dict, rows, error: str = "") -> dict:
    """rows: [(key, severity "error" | "warn" | "info", text)]."""
    count = check.get("count", "any")
    keep = [r for r in rows if count == "any" or r[1] == "error"]
    files = Counter(k for k, _s, _t in keep)
    limit = int(check.get("max_files", 0))
    passed = not error and len(files) <= limit
    return {"tool": check["tool"], "title": check.get("title") or check["tool"], "files": len(files),
            "findings": len(keep), "limit": limit, "count": count, "passed": passed, "error": error,
            "rows": keep, "worst": files.most_common(10)}


def verdict(results: list[dict]) -> str:
    if not results:
        return "no checks"
    if any(r["error"] for r in results):
        return "incomplete"
    return "PASS" if all(r["passed"] for r in results) else "FAIL"


def write_report(out_dir, preset: dict, results: list[dict], dataset: str, rel=None) -> tuple[Path, Path]:
    """-> (html path, csv path). rel(key) -> the file name shown (default: the key)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in preset["name"])[:60] or "preset"
    base = out / f"{safe}_{stamp}"
    rel = rel or (lambda k: k)
    csv_path = base.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["check", "result", "file", "severity", "finding"])
        for r in results:
            word = "error" if r["error"] else ("pass" if r["passed"] else "fail")
            if not r["rows"]:
                w.writerow([r["title"], word, "", "", r["error"]])
            for k, sev, text in r["rows"]:
                w.writerow([r["title"], word, rel(k), sev, text])
    v = verdict(results)
    color = {"PASS": "#2e7d32", "FAIL": "#c62828"}.get(v, "#ef6c00")
    e = html.escape
    rows = []
    for r in results:
        res = "⚠ could not run" if r["error"] else ("✓ pass" if r["passed"] else "✗ fail")
        tone = "#ef6c00" if r["error"] else ("#2e7d32" if r["passed"] else "#c62828")
        what = "errors" if r["count"] == "errors" else "findings"
        rows.append(f"<tr><td>{e(r['title'])}</td><td style='color:{tone};font-weight:600'>{res}</td>"
                    f"<td class=n>{r['files']:,}</td><td class=n>≤ {r['limit']:,}</td><td>{what}</td>"
                    f"<td>{e(r['error'])}</td></tr>")
    detail = []
    for r in results:
        if not r["rows"]:
            continue
        by = {}
        for k, sev, text in r["rows"]:
            by.setdefault(k, []).append((sev, text))
        items = "".join(f"<li><b>{e(rel(k))}</b><ul>" + "".join(f"<li class={s}>{e(t)}</li>" for s, t in v[:20])
                        + (f"<li>… {len(v) - 20} more</li>" if len(v) > 20 else "") + "</ul></li>"
                        for k, v in list(by.items())[:2000])
        more = f"<p>… {len(by) - 2000:,} more files in the CSV.</p>" if len(by) > 2000 else ""
        detail.append(f"<h3>{e(r['title'])} — {len(by):,} file(s)</h3><ul>{items}</ul>{more}")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(preset['name'])} — QC</title>
<style>body{{font:14px/1.5 system-ui,Segoe UI,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}}
.n{{text-align:right}}.error{{color:#c62828}}.warn{{color:#ef6c00}}.info{{color:#555}}
.v{{font-size:28px;font-weight:700;color:{color}}}ul ul{{margin:2px 0 8px}}</style></head><body>
<p class=v>{v}</p><h1>{e(preset['name'])}</h1>
<p>{e(dataset)}<br>{time.strftime('%Y-%m-%d %H:%M')} · {len(results)} check(s) · AIO QC</p>
{f"<p>{e(preset.get('note', ''))}</p>" if preset.get('note') else ''}
<table><tr><th>Check</th><th>Result</th><th class=n>Files</th><th class=n>Allowed</th><th>Counted</th><th></th></tr>
{''.join(rows)}</table><h2>What to fix</h2>{''.join(detail) or '<p>Nothing — every check passed.</p>'}
</body></html>"""
    html_path = base.with_suffix(".html")
    html_path.write_text(doc, encoding="utf-8")
    return html_path, csv_path
