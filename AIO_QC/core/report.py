# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
One-click QC report (Qt-free): a self-contained HTML page (and, via the dialog, a PDF) of where the dataset stands.

    snap = {...}                              taken on the GUI thread (cheap copies — see widgets/report_dialog)
    data = gather(snap, history_events, last)   (worker)  numbers for every section
    html, images = render(data, thumbs)         (worker)  images = {"thumb:N": jpeg bytes} referenced by <img src>
    inline(html, images) -> one file with data: URIs   ·  the PDF path adds them as document resources instead
    save_snapshot(root, data)                   so the next report shows "changes since the last report"

Sections: summary · class balance (per split) · review coverage · findings by tool · QC sample estimate ·
riskiest files (thumbnails with their boxes) · changes since the last report (edits, review marks, merges,
outside changes, and how the headline numbers moved).
Bars are table cells (render in browsers AND in Qt's PDF writer); no scripts, no external files.
"""
from __future__ import annotations

import base64
import html
import json
import math
import time
from collections import Counter
from pathlib import Path

ACCENT, OK, WARN, ERR, MUTED = "#00838f", "#2e7d32", "#b26a00", "#c62828", "#777777"


def esc(x) -> str:
    return html.escape(str(x))


def gather(snap: dict, events: list[dict], last: dict | None) -> dict:
    """snap: root, name, type, version, user, samples [(key, split, display)], index {key: Counter}, status {key: st},
    issues {"summary": {...}, "titles": {...}, "ranked": [(key, risk)], "by_file": {key: [(sev, tool, text)]}},
    samples_qc [(name, Estimate-like dict)], classes [names]."""
    samples = snap["samples"]
    index = snap["index"]
    status = snap["status"]
    splits = sorted({s for _k, s, _d in samples}, key=lambda s: ({"train": 0, "val": 1, "valid": 1, "test": 2}.get(
        s.lower(), 9), s))
    per_split = {sp: {"images": 0, "instances": Counter(), "images_with": Counter(), "empty": 0,
                      "status": Counter()} for sp in splits}
    total_inst = Counter()
    for k, sp, _d in samples:
        c = index.get(k) or Counter()
        d = per_split[sp]
        d["images"] += 1
        d["instances"].update(c)
        d["images_with"].update({lab: 1 for lab in c})
        if not c:
            d["empty"] += 1
        d["status"][status.get(k) or "none"] += 1
        total_inst.update(c)
    n = len(samples)
    st = Counter(status.get(k) or "none" for k, _s, _d in samples)
    reviewed = n - st.get("none", 0)
    iss = snap.get("issues") or {}
    headline = {"files": n, "instances": sum(total_inst.values()), "classes": len(total_inst),
                "reviewed": reviewed, "flag": st.get("flag", 0), "fixed": st.get("fixed", 0), "ok": st.get("ok", 0),
                "issue_files": (iss.get("summary") or {}).get("files", 0)}
    ch = Counter(e["action"] for e in events)
    files_touched = len({e["rel"] for e in events})
    by_tool = Counter(e["tool"] for e in events if e["action"] != "review")
    deltas = {}
    if last:
        for k, v in headline.items():
            if k in (last.get("headline") or {}):
                deltas[k] = v - last["headline"][k]
    return {"snap": {k: snap[k] for k in ("root", "name", "type", "version", "user") if k in snap},
            "when": time.time(), "splits": splits, "per_split": per_split, "total_inst": total_inst,
            "status": st, "headline": headline, "issues": iss, "sampling": snap.get("samples_qc") or [],
            "changes": {"since": (last or {}).get("when"), "actions": ch, "files": files_touched,
                        "tools": by_tool, "deltas": deltas, "events": len(events)}}


def save_snapshot(root, data: dict):
    p = Path(root) / ".aio_qc" / "reports" / "last.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"when": data["when"], "headline": data["headline"]}), encoding="utf-8")


def load_snapshot(root) -> dict | None:
    try:
        return json.loads((Path(root) / ".aio_qc" / "reports" / "last.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------- render
def _bar(frac: float, color: str = ACCENT, width: int = 160) -> str:
    w = max(0, min(width, int(round(frac * width))))
    if w <= 0:
        return f"<table class=bar width={width} cellspacing=0 cellpadding=0><tr><td bgcolor='#e6e6e6' " \
               f"height=8></td></tr></table>"
    rest = width - w
    return (f"<table class=bar width={width} cellspacing=0 cellpadding=0><tr><td bgcolor='{color}' width={w} "
            f"height=8></td>" + (f"<td bgcolor='#e6e6e6' width={rest}></td>" if rest else "") + "</tr></table>")


def _card(title, value, sub="", color=""):
    return (f"<td class=card><div class=ct>{esc(title)}</div><div class=cv style='color:{color}'>{value}</div>"
            f"<div class=cs>{sub}</div></td>")


def _delta(v):
    if v is None or v == 0:
        return ""
    return f" <span style='color:{MUTED}'>({'+' if v > 0 else ''}{v:,})</span>"


CSS = """
body{font-family:'Segoe UI',Helvetica,Arial,sans-serif;color:#222;margin:24px;font-size:10pt;}
h1{font-size:18pt;margin:0 0 2px 0;color:#0b4f55;} h2{font-size:13pt;margin:22px 0 6px 0;color:#0b4f55;
border-bottom:1px solid #cfd8dc;padding-bottom:3px;}
.muted{color:#777;} table{border-collapse:collapse;} td,th{padding:3px 8px;vertical-align:middle;}
th{text-align:left;background:#eef3f4;font-weight:600;} .n{text-align:right;}
.card{border:1px solid #d5dde0;padding:8px 12px;} .ct{color:#666;font-size:8.5pt;} .cv{font-size:16pt;font-weight:600;}
.cs{color:#888;font-size:8pt;} .grid td{border-bottom:1px solid #eceff1;} .bar td{padding:0;}
.thumb{border:1px solid #d5dde0;padding:6px;vertical-align:top;width:33%;}
"""


def render(data: dict, thumbs: list[dict], title: str = "QC Report") -> tuple[str, dict]:
    """thumbs: [{"key", "display", "risk", "status", "issues": [(sev, tool, text)], "jpeg": bytes|None}]."""
    sn, H = data["snap"], data["headline"]
    images = {}
    o = [f"<html><head><meta charset='utf-8'><title>{esc(title)} — {esc(sn.get('name', ''))}</title>"
         f"<style>{CSS}</style></head><body>"]
    o.append(f"<h1>{esc(title)}: {esc(sn.get('name', ''))}</h1><div class=muted>{esc(sn.get('root', ''))} · "
             f"{esc(sn.get('type', ''))} · {time.strftime('%Y-%m-%d %H:%M', time.localtime(data['when']))} · "
             f"{esc(sn.get('user', ''))} · AIO QC {esc(sn.get('version', ''))}</div>")
    D = data["changes"]["deltas"]
    n = max(1, H["files"])
    cov = H["reviewed"] / n
    o.append("<h2>Summary</h2><table cellspacing=6><tr>")
    o.append(_card("Files", f"{H['files']:,}{_delta(D.get('files'))}", f"{len(data['splits'])} split(s)"))
    o.append(_card("Objects", f"{H['instances']:,}{_delta(D.get('instances'))}", f"{H['classes']} classes"))
    o.append(_card("Reviewed", f"{cov:.1%}", f"{H['reviewed']:,} files{_delta(D.get('reviewed'))}",
                   OK if cov >= 0.95 else WARN if cov >= 0.3 else ""))
    o.append(_card("Flagged", f"{H['flag']:,}{_delta(D.get('flag'))}", f"{H['fixed']:,} fixed · {H['ok']:,} OK",
                   ERR if H["flag"] else OK))
    o.append(_card("Files with findings", f"{H['issue_files']:,}{_delta(D.get('issue_files'))}",
                   f"{H['issue_files'] / n:.1%} of files", WARN if H["issue_files"] else OK))
    est = data["sampling"][0] if data["sampling"] else None
    if est:
        e = est[1]
        vcol = {"pass": OK, "fail": ERR}.get(e.get("verdict"), WARN)
        o.append(_card("Estimated error rate", f"{e['rate']:.1%}",
                       f"{e['lo']:.1%}–{e['hi']:.1%} · {esc(e.get('verdict', ''))}", vcol))
    o.append("</tr></table>")

    # ---- class balance
    tot = data["total_inst"]
    if tot:
        sp = data["splits"]
        mx = max(tot.values())
        o.append("<h2>Class balance</h2><table class=grid><tr><th>Class</th><th class=n>Objects</th><th></th>" +
                 "".join(f"<th class=n>{esc(s or 'all')}</th>" for s in sp if len(sp) > 1) +
                 "<th class=n>Share</th></tr>")
        allsum = sum(tot.values())
        for lab, c in tot.most_common(80):
            share = c / allsum
            col = ERR if c < 10 else WARN if share < 0.01 else ACCENT
            cells = ""
            if len(sp) > 1:
                for s in sp:
                    v = data["per_split"][s]["instances"].get(lab, 0)
                    cells += f"<td class=n style='color:{ERR if v == 0 else ''}'>{v:,}</td>"
            o.append(f"<tr><td>{esc(lab)}</td><td class=n>{c:,}</td><td>{_bar(c / mx, col)}</td>{cells}"
                     f"<td class=n>{share:.1%}</td></tr>")
        if len(tot) > 80:
            o.append(f"<tr><td colspan=9 class=muted>… {len(tot) - 80} more classes</td></tr>")
        o.append("</table>")
        r = max(tot.values()) / max(1, min(tot.values()))
        o.append(f"<p class=muted>Most / least common class: {r:,.0f}×. Classes in red have fewer than 10 objects "
                 "(or none in a split).</p>")

    # ---- review coverage
    o.append("<h2>Review coverage</h2><table class=grid><tr><th>Split</th><th class=n>Files</th><th>Reviewed</th><th></th>"
             "<th class=n>OK</th><th class=n>Flagged</th><th class=n>Fixed</th><th class=n>Not reviewed</th></tr>")
    for s in data["splits"]:
        d = data["per_split"][s]
        st = d["status"]
        rv = d["images"] - st.get("none", 0)
        f = rv / max(1, d["images"])
        o.append(f"<tr><td>{esc(s or '—')}</td><td class=n>{d['images']:,}</td><td>{_bar(f, OK)}</td><td class=n>{f:.0%}</td>"
                 f"<td class=n>{st.get('ok', 0):,}</td><td class=n>{st.get('flag', 0):,}</td>"
                 f"<td class=n>{st.get('fixed', 0):,}</td><td class=n>{st.get('none', 0):,}</td></tr>")
    o.append("</table>")

    # ---- findings by tool
    iss = data["issues"] or {}
    summ = iss.get("summary") or {}
    tools = summ.get("tools") or {}
    if tools:
        mx = max(tools.values())
        titles = iss.get("titles") or {}
        o.append("<h2>Findings by tool</h2><table class=grid><tr><th>Tool</th><th class=n>Findings</th><th></th>"
                 "<th class=n>Files</th><th>Last run</th></tr>")
        files_per = iss.get("files_per_tool") or {}
        when = iss.get("when") or {}
        for t, c in sorted(tools.items(), key=lambda x: -x[1]):
            w = when.get(t) or ""
            o.append(f"<tr><td>{esc(titles.get(t, t))}</td><td class=n>{c:,}</td><td>{_bar(c / mx, WARN)}</td>"
                     f"<td class=n>{files_per.get(t, 0):,}</td><td class=muted>"
                     f"{esc(w)}</td></tr>")
        o.append("</table>")
        sev = summ.get("severity") or {}
        o.append(f"<p class=muted>{sev.get('error', 0):,} errors · {sev.get('warn', 0):,} warnings · "
                 f"{sev.get('info', 0):,} notes across {summ.get('files', 0):,} files.</p>")

    # ---- sampling
    if data["sampling"]:
        o.append("<h2>QC sample estimate</h2><table class=grid><tr><th>Sample</th><th class=n>Reviewed</th>"
                 "<th class=n>Errors found</th><th class=n>Error rate</th><th>95% range</th><th>Verdict</th></tr>")
        for name, e in data["sampling"][:5]:
            vcol = {"pass": OK, "fail": ERR}.get(e.get("verdict"), WARN)
            o.append(f"<tr><td>{esc(name)}</td><td class=n>{e['reviewed']:,} / {e['target']:,}</td>"
                     f"<td class=n>{e['flagged']:,}</td><td class=n>{e['rate']:.2%}</td>"
                     f"<td>{e['lo']:.2%} – {e['hi']:.2%}</td><td style='color:{vcol}'><b>{esc(e.get('verdict'))}"
                     f"</b></td></tr>")
        o.append("</table>")

    # ---- worst files
    if thumbs:
        o.append("<h2>Riskiest files</h2><table cellspacing=4>")
        for i in range(0, len(thumbs), 3):
            o.append("<tr>")
            for j, t in enumerate(thumbs[i:i + 3]):
                ref = f"thumb:{i + j}"
                img = ""
                if t.get("jpeg"):
                    images[ref] = t["jpeg"]
                    img = f"<img src='{ref}' width=220><br>"
                lines = "".join(f"<div style='color:{ERR if s == 'error' else WARN if s == 'warn' else MUTED}'>• "
                                f"{esc(tool)}: {esc(text[:90])}</div>" for s, tool, text in t["issues"][:4])
                o.append(f"<td class=thumb>{img}<b>{esc(t['display'])}</b><br><span class=muted>risk "
                         f"{t['risk']:.0f} · {esc(t.get('status') or 'not reviewed')}</span>{lines}</td>")
            o.append("</tr>")
        o.append("</table>")

    # ---- changes
    C = data["changes"]
    o.append("<h2>Changes since the last report</h2>")
    if C["since"] is None:
        o.append("<p class=muted>This is the first report for this dataset — the next one lists what changed "
                 "after now.</p>")
    elif not C["events"]:
        o.append(f"<p class=muted>Nothing recorded since {time.strftime('%Y-%m-%d %H:%M', time.localtime(C['since']))}"
                 ".</p>")
    else:
        from .history import ACTIONS
        o.append(f"<p>Since {time.strftime('%Y-%m-%d %H:%M', time.localtime(C['since']))}: {C['events']:,} recorded "
                 f"change(s) to {C['files']:,} file(s).</p><table class=grid><tr><th>What</th><th class=n>Count</th>"
                 "</tr>")
        for a, c in C["actions"].most_common():
            o.append(f"<tr><td>{esc(ACTIONS.get(a, a))}</td><td class=n>{c:,}</td></tr>")
        o.append("</table>")
        if C["tools"]:
            o.append("<p class=muted>Edits by: " + " · ".join(f"{esc(t)} {c:,}" for t, c in C["tools"].most_common(8))
                     + "</p>")
    o.append(f"<p class=muted style='margin-top:24px'>Generated by AIO QC · {esc(sn.get('root', ''))}</p>")
    o.append("</body></html>")
    return "".join(o), images


def inline(html_text: str, images: dict) -> str:
    for ref, data in images.items():
        html_text = html_text.replace(f"src='{ref}'", f"src='data:image/jpeg;base64,{base64.b64encode(data).decode()}'")
    return html_text


def estimate_dict(est) -> dict:
    return {"rate": est.rate, "lo": est.low, "hi": est.high, "reviewed": est.reviewed, "flagged": est.flagged,
            "target": est.target, "verdict": est.verdict}


def nice_risk(r: float) -> str:
    return f"{r:.0f}" if not math.isnan(r) else "—"
