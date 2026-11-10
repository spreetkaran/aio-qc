# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
HTML reports (one self-contained file: inline SVG charts, embedded thumbnails, no network) + CSV / JSON.

Report types
  summary     one page: verdict in plain words, leaderboard, PR curves, speed vs accuracy, per-class AP
  detailed    summary + per model: confusion matrix, F1 vs confidence, per-class table, error breakdown,
              recall by object size, worst images with the errors drawn on them
  comparison  two models head to head: per-class AP difference, where each one wins (with images),
              both confusion matrices
  video       label-free behaviour on videos: stability (flicker / blips), detections over time, model
              agreement, confidence distribution, flagged moments with frames, scores vs a reference model
"""
import base64
import csv
import datetime as _dt
import json
import os
from html import escape

import numpy as np

from AIO_QC.core.testing import charts as C
from AIO_QC.core.testing import metrics as M
from AIO_QC.core.testing.annotate import draw, thumb, crop_focus, focus_rect, model_bgr

TYPES = {"summary": "Summary", "detailed": "Detailed", "comparison": "Comparison", "video": "Video behaviour"}

CSS = """
*{box-sizing:border-box}
html{background:var(--c-page)}
body{margin:0;background:var(--c-page);color:var(--c-ink);font:14px/1.5 Inter,system-ui,-apple-system,'Segoe UI',sans-serif;font-feature-settings:'tnum' 1;
     -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 64px}
header.top{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;margin-bottom:24px}
.eyebrow{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--c-muted);margin-bottom:6px}
h1{font-size:28px;line-height:1.2;margin:0 0 6px;font-weight:650;letter-spacing:-.01em}
h2{font-size:18px;margin:40px 0 12px;font-weight:620}
h3{font-size:14px;margin:0;font-weight:600}
.sub{color:var(--c-ink2);margin:0}
.btn{font:inherit;font-size:12px;color:var(--c-ink2);background:var(--c-surface);border:1px solid var(--c-grid);
     border-radius:8px;padding:6px 10px;cursor:pointer;white-space:nowrap}
.btn:hover{color:var(--c-ink)}
.card{background:var(--c-surface);border:1px solid rgba(127,127,127,.18);border-radius:14px;padding:18px 18px 10px;margin:0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}
.grid.one{grid-template-columns:1fr}
figcaption{margin-bottom:10px}
figcaption p{margin:2px 0 0;color:var(--c-ink2);font-size:13px}
.chart{overflow-x:auto}
.chart svg{max-width:100%;height:auto;display:block}
details{margin:8px 0 6px}
summary{cursor:pointer;color:var(--c-muted);font-size:12px;padding:4px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--c-grid);white-space:nowrap}
th{font-weight:600;color:var(--c-ink2);font-size:12px;cursor:pointer;user-select:none;position:sticky;top:0;background:var(--c-surface)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:rgba(127,127,127,.06)}
.tablebox{max-height:520px;overflow:auto;border-radius:10px}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px;vertical-align:-1px}
.cellbar{display:flex;align-items:center;gap:8px;justify-content:flex-end}
.cellbar i{display:block;height:8px;border-radius:0 4px 4px 0;min-width:2px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:20px 0 16px}
.tile{background:var(--c-surface);border:1px solid rgba(127,127,127,.18);border-radius:14px;padding:14px 16px}
.tile .l{font-size:12px;color:var(--c-ink2)}
.tile .v{font-size:26px;font-weight:620;margin-top:2px}
.tile .s{font-size:12px;color:var(--c-muted);margin-top:2px}
ul.verdict{list-style:none;padding:0;margin:0;display:grid;gap:8px}
ul.verdict li{background:var(--c-surface);border:1px solid rgba(127,127,127,.18);border-radius:12px;padding:10px 14px;
              display:flex;gap:10px;align-items:baseline}
ul.verdict b{font-weight:600}
.k{font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--c-ink2);min-width:74px}
.k.warn::before{content:"⚠ ";color:var(--c-critical)}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.shot{background:var(--c-surface);border:1px solid rgba(127,127,127,.18);border-radius:12px;overflow:hidden}
.shot img{display:block;width:100%;height:auto;background:#000}
.shot div{padding:8px 10px;font-size:12px;color:var(--c-ink2)}
.shot b{color:var(--c-ink);font-weight:600}
.legendrow{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--c-ink2);margin:6px 0 12px}
.key{display:inline-block;width:18px;height:0;border-top:2px solid;vertical-align:middle;margin-right:6px}
.key.dash{border-top-style:dashed}
.note{color:var(--c-ink2);font-size:13px}
footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--c-grid);color:var(--c-muted);font-size:12px}
footer p{margin:4px 0}
#tip{position:fixed;pointer-events:none;z-index:9;background:var(--c-ink);color:var(--c-surface);font-size:12px;
     padding:6px 9px;border-radius:8px;max-width:340px;opacity:0;transition:opacity .08s;box-shadow:0 4px 18px rgba(0,0,0,.18)}
[data-tip]:hover{opacity:.82}
@media (max-width:640px){.wrap{padding:20px 16px 48px}.grid{grid-template-columns:1fr}h1{font-size:22px}
  header.top{flex-direction:column}}
@media print{.btn,#tip{display:none}.card,.tile,.shot{break-inside:avoid}}
"""

JS = """
(function(){
  var tip=document.getElementById('tip');
  document.addEventListener('pointermove',function(e){
    var t=e.target.closest&&e.target.closest('[data-tip]');
    if(!t){tip.style.opacity=0;return}
    tip.textContent=t.getAttribute('data-tip');
    var x=e.clientX+14,y=e.clientY+14,r=tip.getBoundingClientRect();
    if(x+r.width>innerWidth-8)x=e.clientX-r.width-14; if(y+r.height>innerHeight-8)y=e.clientY-r.height-14;
    tip.style.left=x+'px';tip.style.top=y+'px';tip.style.opacity=1;});
  document.querySelectorAll('table.sortable th').forEach(function(th,i){
    th.addEventListener('click',function(){
      var tb=th.closest('table').tBodies[0],rows=[].slice.call(tb.rows),asc=th.dataset.asc!=='1';
      th.closest('tr').querySelectorAll('th').forEach(function(o){delete o.dataset.asc});th.dataset.asc=asc?'1':'0';
      rows.sort(function(a,b){var x=a.cells[i].dataset.v||a.cells[i].textContent,y=b.cells[i].dataset.v||b.cells[i].textContent;
        var nx=parseFloat(x),ny=parseFloat(y);if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;return asc?x.localeCompare(y):y.localeCompare(x)});
      rows.forEach(function(r){tb.appendChild(r)});});});
  var b=document.getElementById('theme'),modes=['auto','light','dark'];
  b&&b.addEventListener('click',function(){var r=document.documentElement,c=r.getAttribute('data-theme')||'auto',
    n=modes[(modes.indexOf(c)+1)%3];if(n==='auto')r.removeAttribute('data-theme');else r.setAttribute('data-theme',n);
    b.textContent='Theme: '+n;});
})();
"""


_FONT_CSS = None


def font_css():
    """@font-face for Inter (Latin web subset bundled in AIO QC/fonts, ~50 KB): embedded so a report looks
    the same on every machine and stays one self-contained file."""
    global _FONT_CSS
    if _FONT_CSS is None:
        from AIO_QC.core.testing.textdraw import FONT_DIR as d
        rules = []
        for w, num in (("Regular", 400), ("SemiBold", 600)):
            p = os.path.join(d, f"inter-{w}-web.woff")
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    rules.append(f"@font-face{{font-family:'Inter';font-style:normal;font-weight:{num};"
                                 f"font-display:swap;src:url(data:font/woff;base64,{base64.b64encode(f.read()).decode()})"
                                 " format('woff')}")
        _FONT_CSS = "\n".join(rules)
    return _FONT_CSS


def _b64(jpg):
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode()


def _sw(slot):
    return f'<span class="swatch" style="background:var(--s{int(slot) % 8})"></span>'


def _num(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def _table(head, rows, numeric=None, sortable=True, data_v=None):
    numeric = numeric or set()
    h = "".join(f'<th class="{"n" if i in numeric else ""}">{escape(x)}</th>' for i, x in enumerate(head))
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            v = c if isinstance(c, tuple) else (c, None)
            html, sortv = v
            dv = f' data-v="{escape(str(sortv))}"' if sortv is not None else ""
            cells.append(f'<td class="{"n" if i in numeric else ""}"{dv}>{html}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="tablebox"><table class="{"sortable" if sortable else ""}"><thead><tr>{h}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _fig(title, desc, svg, table_html=None, wide=False):
    t = f"<details><summary>Show data</summary>{table_html}</details>" if table_html else ""
    span = ' style="grid-column:1/-1"' if wide else ""          # (no nested quotes: Python 3.10 / 3.11 f-strings)
    return (f'<figure class="card"{span}><figcaption><h3>{escape(title)}</h3>'
            f'<p>{escape(desc)}</p></figcaption><div class="chart">{svg}</div>{t}</figure>')


class ReportBuilder:
    def __init__(self, run=None, title="Model evaluation", slots=None, frame_getter=None, gallery_n=12,
                 video=None, version=""):
        """run: evaluation.LabRun (or None for a video-only report); slots: colour slot per model (fixed);
        frame_getter(item) -> BGR image; video: [(video path, videoeval result)]."""
        self.run = run
        self.title = title
        n = len(run.specs) if run else (len(video[0][1]["models"]) if video else 0)
        self.slots = slots or list(range(n))
        self.ink = C.Ink("html")
        self.get_frame = frame_getter
        self.gallery_n = gallery_n
        self.video = video or []
        self.version = version

    # ---------------------------------------------------------------- pieces
    def _header(self, kind):
        r = self.run
        bits = [_dt.datetime.now().strftime("%d %b %Y, %H:%M")]
        if r:
            bits.append(f"{len(r.items):,} sampled video frames" if r.source == "videos" else f"{len(r.items):,} images")
            if r.ground == "labels":
                bits.append(f"{len(r.eval_idx):,} with ground truth")
            elif r.ground == "reference":
                bits.append(f"scored against {r.specs[r.reference].label} (reference model)")
            bits.append(f"{len(r.names)} classes")
            bits.append(f"{len(r.specs)} model{'s' if len(r.specs) != 1 else ''}")
        for path, _ in self.video:
            bits.append(os.path.basename(path))
        return (f'<header class="top"><div><div class="eyebrow">AIO QC · Testing · {escape(TYPES[kind])} report'
                f'</div><h1>{escape(self.title)}</h1><p class="sub">{escape(" · ".join(bits))}</p></div>'
                f'<button class="btn" id="theme" type="button">Theme: auto</button></header>')

    def _verdict(self):
        items = self.run.verdict() if self.run else []
        if not items:
            return ""
        lab = {"best": "Best", "fast": "Speed", "tradeoff": "Trade-off", "weak": "Weakness", "warn": "Check"}
        li = "".join(f'<li><span class="k {k}">{lab[k]}</span><span>{escape(t)}</span></li>' for k, t in items)
        return f'<h2>In short</h2><ul class="verdict">{li}</ul>'

    def _tiles(self):
        tiles = self.run.kpis()
        if not tiles:
            return ""
        return '<div class="tiles">' + "".join(
            f'<div class="tile"><div class="l">{escape(a)}</div><div class="v">{escape(b)}</div>'
            f'<div class="s">{escape(c)}</div></div>' for a, b, c in tiles) + "</div>"

    def _leaderboard(self):
        r = self.run
        rows = r.summary_table()
        if not rows:
            return ""
        top = max(x["map5095"] for x in rows) or 1
        out = []
        for x in rows:
            slot = self.slots[x["idx"]]
            bar = (f'<span class="cellbar"><i style="width:{x["map5095"] / top * 90:.0f}px;background:var(--s{slot % 8})"></i>'
                   f'{x["map5095"]:.3f}</span>')
            out.append([(str(x["rank"]), x["rank"]), f'{_sw(slot)}{escape(x["model"])}', (bar, x["map5095"]),
                        (f'{x["map50"]:.3f}', x["map50"]), (f'{x["precision"]:.3f}', x["precision"]),
                        (f'{x["recall"]:.3f}', x["recall"]), (f'{x["f1"]:.3f}', x["f1"]),
                        (f'{x["best_conf"]:.2f}', x["best_conf"]),
                        (_num(x["speed_ms"], 1), x["speed_ms"] or 1e9), (_num(x["fps"], 0), x["fps"] or 0),
                        (f'{x["fp"]:,}', x["fp"]), (f'{x["fn"]:,}', x["fn"])])
        head = ["#", "Model", "mAP50-95", "mAP50", "Precision", "Recall", "F1", "Best conf", "ms / image", "Images/s",
                "False det.", "Missed"]
        note = ("<p class='note'>Precision / recall / F1 at each model's best-F1 confidence (YOLO-val method). False "
                "detections and missed objects counted at the operating threshold. Click a column to sort.</p>")
        return "<h2>Leaderboard</h2>" + _table(head, out, set(range(2, 12)) | {0}) + note

    def _pr_curves(self):
        r = self.run
        series = []
        for i, res in r.scored:
            if res.prec_at_rec is None or not len(res.prec_at_rec):
                continue
            y = res.prec_at_rec.mean(0)
            xs = np.linspace(0, 1, 101)
            series.append(dict(name=res.name, color=self.slots[i], x=xs.tolist(),
                               y=np.interp(xs, res.px, y).tolist(),
                               tip=f"{res.name}: mAP50 {res.map50:.3f}"))
        if not series:
            return ""
        svg = C.line_chart(series, "Recall", "Precision", (0, 1), (0, 1), 560, 330, self.ink,
                           "Precision vs recall (IoU 0.5, mean over classes)", end_labels=False)
        rows = [[f'{_sw(self.slots[i])}{escape(res.name)}'] + [f"{np.interp(q, res.px, res.prec_at_rec.mean(0)):.3f}"
                                                                for q in (0.25, 0.5, 0.75, 0.9)]
                for i, res in r.scored if res.prec_at_rec is not None and len(res.prec_at_rec)]
        return _fig("Precision vs recall", "Higher and further right is better. The area under each curve is "
                    "mAP50.", svg, _table(["Model", "P @ R 0.25", "P @ R 0.50", "P @ R 0.75", "P @ R 0.90"], rows, {1, 2, 3, 4}))

    def _f1_curves(self, only=None):
        r = self.run
        series, markers = [], []
        for i, res in r.scored:
            if only is not None and i != only:
                continue
            if res.f1_curve is None or not len(res.f1_curve):
                continue
            f1 = res.f1_curve.mean(0)
            xs = res.px[::10]
            series.append(dict(name=res.name, color=self.slots[i], x=xs.tolist(), y=f1[::10].tolist()))
            bi = int(np.argmin(np.abs(res.px - res.best_conf)))
            markers.append(dict(x=res.best_conf, y=float(f1[bi]), color=self.slots[i],
                                tip=f"{res.name}: best confidence {res.best_conf:.2f} → F1 {f1[bi]:.3f}",
                                label=f"{res.best_conf:.2f}" if only is not None or len(r.scored) <= 2 else None))
        if not series:
            return ""
        svg = C.line_chart(series, "Confidence threshold", "F1 (mean over classes)", (0, 1), (0, 1), 560, 330, self.ink,
                           "F1 vs confidence", markers=markers, end_labels=False)
        rows = [[f'{_sw(self.slots[i])}{escape(res.name)}', f"{res.best_conf:.2f}", f"{res.mf1:.3f}"]
                for i, res in r.scored if only is None or i == only]
        return _fig("F1 vs confidence threshold", "Where to set the confidence threshold: the dot marks the best "
                    "balance of precision and recall.", svg, _table(["Model", "Best confidence", "F1 there"], rows, {1, 2}))

    def _speed_scatter(self):
        pts = [dict(name=res.name, color=self.slots[i], x=res.speed_ms, y=res.map5095,
                    tip=f"{res.name}: {res.speed_ms:.1f} ms · mAP50-95 {res.map5095:.3f}")
               for i, res in self.run.scored if res.speed_ms]
        if len(pts) < 2:
            return ""
        svg = C.scatter(pts, "Milliseconds per image (lower is faster)", "mAP50-95", 560, 330, self.ink,
                        "Speed vs accuracy", x_fmt=lambda v: f"{v:g}", y_fmt=lambda v: f"{v:.2f}")
        return _fig("Speed vs accuracy", "Up and left is better. Speed measured on this machine; image loading "
                    "excluded, first batch ignored.", svg)

    def _per_class(self, metric="ap50", cap=24):
        r = self.run
        if not r.scored:
            return ""
        counts = r.class_counts()
        classes = sorted(counts, key=lambda c: -counts[c])
        shown = classes[:cap]
        series = []
        for i, res in r.scored:
            d = {row["cls"]: row for row in res.class_rows()}
            series.append(dict(name=res.name, color=self.slots[i], links=[f"class:{i}:{c}" for c in shown],
                               values=[d[c][metric] if c in d else None for c in shown],
                               tips=[f"{res.name} · {r.names.get(c, c)}: AP50 {d[c]['ap50']:.3f}, AP50-95 {d[c]['ap']:.3f}, "
                                     f"P {d[c]['p']:.2f}, R {d[c]['r']:.2f} ({d[c]['gt']} objects)" if c in d else None
                                     for c in shown]))
        cats = [f"{r.names.get(c, c)} ({counts[c]})" for c in shown]
        svg = C.bar_chart(cats, series, 560, self.ink, "AP50 per class", value_fmt=lambda v: f"{v:.2f}", max_value=1.0,
                          cat_links=[f"class:-1:{c}" for c in shown])
        head = ["Class", "Objects"] + [res.name for _, res in r.scored]
        rows = []
        for c in classes:
            row = [escape(str(r.names.get(c, c))), (f"{counts[c]:,}", counts[c])]
            for i, res in r.scored:
                d = {x["cls"]: x for x in res.class_rows()}
                row.append((f"{d[c]['ap50']:.3f}", d[c]["ap50"]) if c in d else "–")
            rows.append(row)
        more = f" Showing the {cap} most frequent of {len(classes)} classes; all in the table." if len(classes) > cap else ""
        return _fig("AP50 per class", "Bars per model; the number in brackets is how many objects of that class are "
                    "in the ground truth." + more, svg, _table(head, rows, set(range(1, len(head)))), wide=len(r.scored) > 2)

    def _errors(self, only=None):
        r = self.run
        types = ["cls", "loc", "dup", "bkg", "missed", "missed_low_conf"]
        series = [dict(name=res.name, color=self.slots[i], values=[res.errors.get(t, 0) for t in types],
                       links=[f"err:{i}:{t}" for t in types])
                  for i, res in r.scored if only is None or i == only]
        if not series:
            return ""
        svg = C.bar_chart([M.ERROR_LABELS[t] for t in types], series, 560, self.ink, "Errors by type",
                          value_fmt=lambda v: f"{int(v):,}")
        rows = [[f'{_sw(self.slots[i])}{escape(res.name)}'] + [f"{res.errors.get(t, 0):,}" for t in types]
                for i, res in r.scored if only is None or i == only]
        return _fig("What goes wrong", "False detections split by cause, then missed objects. 'Found below threshold' "
                    "means a lower confidence threshold would catch it.", svg,
                    _table(["Model"] + [M.ERROR_LABELS[t] for t in types], rows, set(range(1, 7))))

    def _sizes(self):
        r = self.run
        sizes = ["small", "medium", "large"]
        series = [dict(name=res.name, color=self.slots[i],
                       values=[(res.recall_by_size[s][0] / res.recall_by_size[s][1]) if res.recall_by_size[s][1] else None
                               for s in sizes],
                       tips=[f"{res.name} · {s}: {res.recall_by_size[s][0]} of {res.recall_by_size[s][1]} found"
                             for s in sizes]) for i, res in r.scored]
        if not series:
            return ""
        totals = r.results[r.scored[0][0]].recall_by_size
        cats = [f"{s} ({totals[s][1]:,})" for s in sizes]
        svg = C.bar_chart(cats, series, 560, self.ink, "Recall by object size", value_fmt=lambda v: f"{v * 100:.0f}%",
                          max_value=1.0)
        return _fig("Recall by object size", "Share of objects found at the operating threshold. Small < 32×32 px, "
                    "medium < 96×96 px, large above (COCO areas).", svg)

    def _class_distribution(self):
        r = self.run
        counts = r.class_counts()
        if not counts:
            return ""
        cls = sorted(counts, key=lambda c: -counts[c])[:30]
        svg = C.bar_chart([str(r.names.get(c, c)) for c in cls], [dict(name="objects", color="muted",
                          values=[counts[c] for c in cls])], 560, self.ink, "Objects per class", value_fmt=lambda v: f"{int(v):,}",
                          legend=False)
        return _fig("Ground truth: objects per class", "How the test set is balanced - classes with few objects have "
                    "less reliable scores.", svg)

    def _confusion(self, i, res, title=None):
        r = self.run
        cm = res.confusion
        present = [c for c in range(cm.shape[0] - 1) if cm[c].sum() or cm[:, c].sum()]
        if not present:
            return ""
        if len(present) > 30:
            present = sorted(present, key=lambda c: -cm[c].sum())[:30]
        idx = present + [cm.shape[0] - 1]
        sub = cm[np.ix_(idx, idx)]
        labels = [str(r.names.get(c, c)) for c in present] + ["background"]
        cls_of = present + [-1]                                   # -1 = background
        svg = C.heatmap(sub, labels, labels, self.ink, f"Confusion matrix · {res.name}", normalize="row",
                        y_label="True class" if r.ground == "labels" else "Reference model",
                        link=lambda a, b: f"cm:{i}:{cls_of[a]}:{cls_of[b]}" if a != b else f"class:{i}:{cls_of[a]}")
        rows = [[escape(labels[a])] + [f"{int(v):,}" for v in sub[a]] for a in range(len(labels))]
        desc = (f"At confidence {res.op_conf:.2f}, IoU > 0.45. Rows: what is really there; columns: what the model "
                "said. Colour = share of the row; 'background' row = false detections, column = missed objects.")
        return _fig(title or f"Confusion matrix · {res.name}", desc, svg,
                    _table(["True \\ predicted"] + labels, rows, set(range(1, len(labels) + 1)), sortable=False), wide=True)

    def _class_table(self, i, res):
        rows = [[escape(x["name"]), (f"{x['gt']:,}", x["gt"]), (f"{x['ap50']:.3f}", x["ap50"]), (f"{x['ap']:.3f}", x["ap"]),
                 (f"{x['p']:.3f}", x["p"]), (f"{x['r']:.3f}", x["r"]), (f"{x['f1']:.3f}", x["f1"]),
                 (f"{x['tp']:,}", x["tp"]), (f"{x['fp']:,}", x["fp"]), (f"{x['fn']:,}", x["fn"])]
                for x in sorted(res.class_rows(), key=lambda x: x["ap50"])]
        extra = ""
        if res.unscored_classes:
            extra = (f"<p class='note'>Predicted but absent from the ground truth (not scored): "
                     f"{escape(', '.join(str(self.run.names.get(c, c)) for c in res.unscored_classes))}.</p>")
        return (f'<figure class="card" style="grid-column:1/-1"><figcaption><h3>Per class · {escape(res.name)}</h3>'
                f'<p>Sorted weakest first. P / R / F1 at best confidence {res.best_conf:.2f}; TP / FP / FN at the '
                f'operating threshold {res.op_conf:.2f}.</p></figcaption>' +
                _table(["Class", "Objects", "AP50", "AP50-95", "Precision", "Recall", "F1", "Correct", "False det.", "Missed"],
                       rows, set(range(1, 10))) + extra + "</figure>")

    def _layers(self, item_idx, m):
        """(gt or None, preds, per-image error record or None) of model m on one test image."""
        r = self.run
        it = r.items[item_idx]
        res = r.results[m]
        pi = None
        if res is not None:
            pi = next((p for p in res.per_image if r.eval_idx[p["i"]] == item_idx), None)
        pr = r.preds[m][item_idx]
        pr = pr if pr is not None else np.zeros((0, 6), np.float32)
        if r.ground == "labels":
            gt = (it.boxes, it.cls) if it.boxes is not None else None
        elif item_idx in r.eval_idx and r.scored:
            gt = r.ground_truth()[1][r.eval_idx.index(item_idx)]
        else:
            gt = None
        return gt, pr, pi

    def model_conf(self, m):
        """Operating threshold of model m (its scored one, else the model's own setting)."""
        res = self.run.results[m]
        if res is not None:
            return res.op_conf
        if self.run.ground == "reference" and m == self.run.reference:
            return self.run._ref_conf()              # the threshold that makes it the answer key
        return 0.25                                   # unscored: the usual detection threshold (never the cache floor)

    def plot_image(self, item_idx, mi, show_gt=True, errors=True, conf=None, tw=None, img=None):
        """A test image "plotted": model mi's detections in its colour with "class 0.87" labels (Inter), plus the
        ground truth dashed (show_gt) and errors in red (errors, needs ground truth). tw = width the image will
        be shown at (labels are sized to stay readable there); None = full resolution (saving). None when the
        image cannot be read."""
        if img is None:
            if self.get_frame is None:
                return None
            img = self.get_frame(self.run.items[item_idx])
            if img is None:
                return None
        gt, pr, pi = self._layers(item_idx, mi)
        if self.run.ground == "reference" and mi == self.run.reference:
            show_gt = errors = False             # the reference IS the answer key: draw its detections only
        conf = self.model_conf(mi) if conf is None else conf
        scale = max(0.5, img.shape[1] / tw * 1.1) if tw else None
        names, col = self.run.names, model_bgr(self.slots[mi])
        if errors and pi and gt is not None and conf == self.model_conf(mi):
            return draw(img, gt if show_gt else None, pr, names, col, conf, pi["pred_type"], pi["keep"],
                        pi["gt_type"] if show_gt else None, gt_names=names, scale=scale, fit_labels=bool(tw))
        return draw(img, gt if show_gt else None, pr, names, col, conf, gt_names=names, scale=scale,
                    fit_labels=bool(tw))

    def shot_image(self, item_idx, mi, focus=True, compare=None, tw=420):
        """BGR image of one test image with the errors of model mi drawn (zoomed to the errors when focus), or
        two models side by side for compare=(a, b). None when the image cannot be read."""
        if self.get_frame is None:
            return None
        img = self.get_frame(self.run.items[item_idx])
        if img is None:
            return None

        def one(m):
            gt, pr, pi = self._layers(item_idx, m)
            bad = []
            if pi and gt is not None:
                bad = [pr[k, :4] for k, t in zip(pi["keep"], pi["pred_type"]) if t != "tp"]
                bad += [gt[0][k] for k, t in enumerate(pi["gt_type"]) if t != "tp"]
            crop_w = img.shape[1]
            zoom = focus and bad and min(img.shape[:2]) > 400
            if zoom:
                _, _, cw, _ = focus_rect(img.shape, np.array(bad))
                crop_w = cw
            shown = tw / 2 if compare else tw               # side by side: each half gets half the width
            im = self.plot_image(item_idx, m, True, True, None, shown * img.shape[1] / crop_w, img=img)
            if zoom:
                im, _ = crop_focus(im, np.array(bad))
            return im
        if compare:
            import cv2
            a, b = (one(m) for m in compare)
            h = min(a.shape[0], b.shape[0])
            a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
            b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
            return np.hstack([a, np.full((h, max(4, h // 120), 3), 255, np.uint8), b])
        return one(mi)

    def _shot(self, item_idx, mi, caption, focus=True, compare=None):
        """Thumbnail with the errors of model mi drawn (or two models side by side for compare=(a, b))."""
        im = self.shot_image(item_idx, mi, focus, compare, 360 if compare else 420)
        if im is None:
            return ""
        jpg = thumb(im, 720 if compare else 420, 78)
        it = self.run.items[item_idx]
        where = it.meta.get("video")
        loc = (f"{os.path.basename(where)} · frame {it.meta.get('frame')} · {it.meta.get('time', 0):.2f}s"
               if where else os.path.basename(it.path))
        return (f'<div class="shot"><img src="{_b64(jpg)}" alt="{escape(loc)}"><div><b>{escape(caption)}</b>'
                f'<br>{escape(loc)}</div></div>')

    def _gallery_legend(self):
        gt = ("reference model (answer key)" if self.run is not None and self.run.ground == "reference"
              else "ground truth")
        return ("<div class='legendrow'><span><span class='key dash' style='border-color:var(--c-ink2)'></span>" + gt +
                "</span><span><span class='key' style='border-color:var(--s0)'></span>correct detection (model colour)"
                "</span><span><span class='key' style='border-color:var(--c-critical)'></span>error, labelled with its type"
                "</span><span><span class='key dash' style='border-color:var(--c-critical)'></span>missed object</span></div>")

    def _worst(self, i, res):
        rows = self.run.worst_images(i, self.gallery_n)
        shots = "".join(self._shot(ii, i, f"{fp} false · {fn} missed · {tp} correct") for ii, tp, fp, fn, *_ in rows)
        if not shots:
            return ""
        return (f'<figure class="card" style="grid-column:1/-1"><figcaption><h3>Hardest images · {escape(res.name)}</h3>'
                f'<p>Most errors first; zoomed to the errors.</p></figcaption>{self._gallery_legend()}'
                f'<div class="gallery">{shots}</div></figure>')

    # ---------------------------------------------------------------- video
    def _video_section(self, path, v):
        if not v["models"]:
            return ""
        name = os.path.basename(path)
        out = [f"<h2>{escape(name)}</h2>"]
        rows = []
        for i, m in enumerate(v["models"]):
            slot = self.slots[i] if i < len(self.slots) else i
            rows.append([f"{_sw(slot)}{escape(m['label'])}", (f"{m['op_conf']:.2f}", m["op_conf"]),
                         (f"{m['detections']:,}", m["detections"]),
                         (f"{m['frames_with_dets']:,}", m["frames_with_dets"]), (f"{m['tracks']:,}", m["tracks"]),
                         (f"{m['flicker']:,}", m["flicker"]), (f"{m['blip']:,}", m["blip"]),
                         (f"{m['stability'] * 100:.1f}%", m["stability"])])
        out.append('<div class="grid one">' + (
            '<figure class="card"><figcaption><h3>Stability</h3><p>Flicker: an object disappears for a frame or two '
            'and comes back (missed detections). Blip: an object that exists for at most two samples (likely false). '
            'Stability: share of detections not involved in either.</p></figcaption>' +
            _table(["Model", "Threshold", "Detections", "Frames with detections", "Objects tracked", "Flicker", "Blips",
                    "Stability"], rows, set(range(1, 8))) + "</figure></div>"))
        figs = []
        times = v["times"]
        series = [dict(name=m["label"], color=self.slots[i] if i < len(self.slots) else i,
                       y=m["counts"].sum(1).tolist()) for i, m in enumerate(v["models"])]
        events = []
        for e in v["events"]:
            if e["type"] in ("flicker", "blip"):
                mi = next(i for i, m in enumerate(v["models"]) if m["label"] == e["model"])
                events.append(dict(t=e["time"], color=self.slots[mi] if mi < len(self.slots) else mi,
                                   tip=f"{e['model']}: {e['type']} ({e['cls']}) at {e['time']:.2f}s, frame {e['frame']}",
                                   link=f"vframe:{e['frame']}"))
        svg = C.timeline(times, series, events[:3000], 1080, 240 + 10 * len(series), self.ink, "Detections over time",
                         "Detections per frame")
        figs.append(_fig("Detections over time", "Objects detected per sampled frame; ticks under the axis mark each "
                         "model's flicker / blip moments (one row per model).", svg, wide=True))
        if v["pairs"]:
            # pairs are their own entities: slots after the models', so no pair borrows a model's colour
            base = len(v["models"])
            ps = [dict(name=f"{p['a']} vs {p['b']}", color=(base + k) % 8, y=p["agreement"].tolist())
                  for k, p in enumerate(v["pairs"][:max(1, 8 - base)])]
            svg = C.timeline(times, ps, None, 1080, 220, self.ink, "Model agreement", "Agreement (F1)", y_max=1.0,
                             y_fmt=lambda x: f"{x:.1f}")
            prow = [[escape(p["a"]), escape(p["b"]), (f"{p['mean']:.3f}", p["mean"])] for p in v["pairs"]]
            figs.append(_fig("Agreement between models", "1.0 = both models found the same objects; dips show where "
                             "they disagree (worth a look - one of them is wrong).", svg,
                             _table(["Model A", "Model B", "Mean agreement"], prow, {2}), wide=True))
        hs = [dict(name=m["label"], color=self.slots[i] if i < len(self.slots) else i, counts=m["conf_hist"])
              for i, m in enumerate(v["models"])]
        figs.append(_fig("Confidence distribution", "All stored detections (from the 0.001 floor). Mass near the "
                         "threshold means many borderline decisions.",
                         C.histogram_lines(v["models"][0]["conf_edges"], hs, 560, 300, self.ink, "Confidence distribution")))
        cls = sorted({c for m in v["models"] for c in m["per_class"]})
        if cls:
            bs = [dict(name=m["label"], color=self.slots[i] if i < len(self.slots) else i,
                       values=[m["per_class"].get(c, 0) for c in cls]) for i, m in enumerate(v["models"])]
            figs.append(_fig("Detections per class", "Total detections at each model's threshold over the sampled frames.",
                             C.bar_chart(cls, bs, 560, self.ink, "Detections per class", value_fmt=lambda x: f"{int(x):,}")))
        out.append('<div class="grid">' + "".join(figs) + "</div>")
        if v.get("reference"):
            ref = v["reference"]
            rrows = [[escape(res.name), (f"{res.map50:.3f}", res.map50), (f"{res.map5095:.3f}", res.map5095),
                      (f"{res.op_prf()[0]:.3f}", 0), (f"{res.op_prf()[1]:.3f}", 0)] for res in ref["results"]]
            out.append(f'<div class="grid one"><figure class="card"><figcaption><h3>Compared with the reference model '
                       f'({escape(ref["label"])})</h3><p>The reference model\'s detections are treated as ground truth: '
                       f'this measures how closely each model reproduces it, not true accuracy.</p></figcaption>' +
                       _table(["Model", "mAP50 vs ref.", "mAP50-95 vs ref.", "Precision vs ref.", "Recall vs ref."], rrows,
                              {1, 2, 3, 4}) + "</figure></div>")
        if self.get_frame is not None:
            shots = []
            seen = []
            span = (v["times"][-1] - v["times"][0]) if len(v["times"]) > 1 else 1.0
            min_gap = span / max(1, self.gallery_n * 2)          # spread over the video, not one burst
            for e in sorted(v["events"], key=lambda e: (e["type"] != "disagree", e.get("score", 1))):
                if len(shots) >= self.gallery_n:
                    break
                if any(abs(e["time"] - t) < min_gap for t in seen):
                    continue
                seen.append(e["time"])
                img = self.get_frame(dict(video=path, frame=e["frame"]))
                if img is None:
                    continue
                sample = e["sample"]
                drawn = img
                for mi, m in enumerate(v["models"]):
                    p = v.get("preds", [[]] * len(v["models"]))[mi]
                    if p and sample < len(p) and p[sample] is not None:
                        drawn = draw(drawn, None, p[sample], v.get("model_names", [{}] * len(v["models"]))[mi],
                                     model_bgr(self.slots[mi] if mi < len(self.slots) else mi), m["op_conf"])
                kind = {"disagree": "Models disagree", "flicker": "Flicker", "blip": "Blip"}[e["type"]]
                shots.append(f'<div class="shot"><img src="{_b64(thumb(drawn, 420, 78))}">'
                             f'<div><b>{escape(kind)}</b> · {escape(e["model"])}<br>frame {e["frame"]} · {e["time"]:.2f}s</div></div>')
            if shots:
                out.append('<div class="grid one"><figure class="card"><figcaption><h3>Moments worth a look</h3><p>'
                           'Biggest disagreements first, then flicker / blips. All models drawn in their colours.</p>'
                           f'</figcaption><div class="gallery">{"".join(shots)}</div></figure></div>')
        return "".join(out)

    # ---------------------------------------------------------------- comparison
    def _comparison_reference(self, ref, m):
        """A model next to the reference model (unlabelled / videos-only runs): how much of what the reference finds
        it finds, how much of what it finds the reference confirms, per class, and the frames where they differ."""
        r = self.run
        res = r.results[m]
        rn, mn = r.specs[ref].label, res.name
        p, rc, _ = res.op_prf()
        errs = r.image_errors(m)
        misses = sorted(((ii, fn) for ii, tp, fp, fn, *_ in errs if fn), key=lambda x: -x[1])
        extras = sorted(((ii, fp) for ii, tp, fp, fn, *_ in errs if fp), key=lambda x: -x[1])
        differ = len({ii for ii, _ in misses} | {ii for ii, _ in extras})
        sr = (r.infos.get(r.specs[ref].key) or {}).get("speed_ms")
        sm = res.speed_ms
        what = "frames" if r.source == "videos" else "images"
        tiles = [(f"Finds what {rn} finds", f"{rc * 100:.0f}%", f"recall vs the reference at conf {res.op_conf:.2f}"),
                 ("Confirmed by reference", f"{p * 100:.0f}%", "precision vs the reference"),
                 ("mAP50 vs reference", f"{res.map50:.3f}", f"mAP50-95 {res.map5095:.3f}"),
                 ("Speed", f"{sm:.1f} ms" if sm else "–",
                  f"{(sm - sr) / sr * 100:+.0f}% vs {rn}" if sm and sr else "not measured"),
                 (f"{what.capitalize()} that differ", f"{differ:,}", f"of {len(r.eval_idx):,}")]
        out = [f"<h2>{escape(mn)} compared with the reference {escape(rn)}</h2>",
               "<p class='note'>No labels: the reference model's detections are the answer key, so these numbers measure "
               "agreement with it, not true accuracy.</p>",
               '<div class="tiles">' + "".join(
                   f'<div class="tile"><div class="l">{escape(x)}</div><div class="v">{escape(y)}</div>'
                   f'<div class="s">{escape(z)}</div></div>' for x, y, z in tiles) + "</div>"]
        out.append('<div class="grid">' + self._confusion(m, res) + self._class_table(m, res) + "</div>")
        for title, lst, note in ((f"{mn} misses what {rn} finds", misses, "Most missed objects first."),
                                 (f"{mn} finds what {rn} does not", extras, "Most extra detections first.")):
            shots = "".join(self._shot(ii, m, f"left {rn} · right {mn}", focus=False, compare=(ref, m))
                            for ii, _ in lst[:self.gallery_n])
            if shots:
                out.append(f'<div class="grid one"><figure class="card"><figcaption><h3>{escape(title)} ({len(lst):,} '
                           f'{what})</h3><p>Left: {escape(rn)} (reference), right: {escape(mn)}. {note}</p></figcaption>'
                           f'{self._gallery_legend()}<div class="gallery" style="grid-template-columns:repeat(auto-fill,'
                           f'minmax(420px,1fr))">{shots}</div></figure></div>')
        return "".join(out)

    def _comparison(self, a, b):
        r = self.run
        if r.ground == "reference" and r.reference in (a, b) and a != b:
            ref = r.reference
            m = b if a == ref else a
            if r.results[m] is not None:
                return self._comparison_reference(ref, m)
        cmp_ = r.comparison(a, b)
        if not cmp_:
            return "<p class='note'>Both models need ground truth (or a reference) to be compared.</p>"
        ra, rb = r.results[a], r.results[b]
        out = []
        sa, sb = ra.speed_ms, rb.speed_ms
        tiles = [("mAP50-95", f"{rb.map5095:.3f}", f"{cmp_['map_delta']:+.3f} vs {ra.name}"),
                 ("mAP50", f"{rb.map50:.3f}", f"{cmp_['map50_delta']:+.3f} vs {ra.name}"),
                 ("Speed", f"{sb:.1f} ms" if sb else "–",
                  (f"{(sb - sa) / sa * 100:+.0f}% vs {ra.name}" if sa and sb else "not measured")),
                 ("Images better", f"{len(cmp_['b_better']):,}", f"{len(cmp_['a_better']):,} better with {ra.name}")]
        out.append(f"<h2>{escape(rb.name)} compared with {escape(ra.name)}</h2>")
        out.append('<div class="tiles">' + "".join(
            f'<div class="tile"><div class="l">{escape(x)}</div><div class="v">{escape(y)}</div><div class="s">{escape(z)}</div></div>'
            for x, y, z in tiles) + "</div>")
        pc = sorted(cmp_["per_class"], key=lambda x: x["delta"])
        svg = C.diverging_bars([x["name"] for x in pc], [x["delta"] for x in pc], 560, self.ink, "AP50 difference per class",
                               labels=(f"{ra.name} better", f"{rb.name} better"))
        rows = [[escape(x["name"]), _num(x["a"]), _num(x["b"]), (f"{x['delta']:+.3f}", x["delta"])] for x in pc]
        figs = [_fig("Per-class difference", f"AP50 of {rb.name} minus {ra.name}. Blue: {rb.name} better; red: worse.",
                     svg, _table(["Class", ra.name, rb.name, "Difference"], rows, {1, 2, 3}))]
        figs.append(self._pr_curves_subset([a, b]))
        out.append('<div class="grid">' + "".join(figs) + "</div>")
        for title, lst in ((f"Where {rb.name} does better", cmp_["b_better"]), (f"Where {ra.name} does better", cmp_["a_better"])):
            shots = "".join(self._shot(ii, a, f"left {ra.name} · right {rb.name}", focus=False, compare=(a, b))
                            for ii, _ in lst[:self.gallery_n])
            if shots:
                out.append(f'<div class="grid one"><figure class="card"><figcaption><h3>{escape(title)}</h3><p>Left: '
                           f'{escape(ra.name)}, right: {escape(rb.name)}. Largest differences first.</p></figcaption>'
                           f'{self._gallery_legend()}<div class="gallery" style="grid-template-columns:repeat(auto-fill,minmax(420px,1fr))">'
                           f'{shots}</div></figure></div>')
        out.append('<div class="grid">' + self._confusion(a, ra) + self._confusion(b, rb) + "</div>")
        return "".join(out)

    def _pr_curves_subset(self, idx):
        keep = self.run.results
        saved = self.run.results
        try:
            self.run.results = [r if i in idx else None for i, r in enumerate(keep)]
            return self._pr_curves()
        finally:
            self.run.results = saved

    # ---------------------------------------------------------------- assemble
    def _footer(self):
        r = self.run
        lines = ["Metrics follow the YOLO (Ultralytics) validation method: predictions at confidence ≥ 0.001, IoU "
                 "thresholds 0.50:0.05:0.95, 101-point interpolated AP; P / R / F1 at the confidence that maximises "
                 "mean F1. Confusion matrix at the operating threshold, IoU > 0.45. Error types at IoU 0.5."]
        if r:
            for s in r.specs:
                info = r.infos.get(s.key) or {}
                lines.append(f"{s.label}: {s.path} · fingerprint {s.key} · image size {s.imgsz or 'model default'} · "
                             f"NMS IoU {s.iou} · max {s.max_det} boxes · {info.get('backend') or 'cached'}")
            if r.source:
                lines.append(f"Test data: {r.source}")
        lines.append(f"Generated by AIO QC {self.version} on {_dt.datetime.now():%Y-%m-%d %H:%M}.")
        return "<footer>" + "".join(f"<p>{escape(x)}</p>" for x in lines) + "</footer>"

    def build(self, kind="summary", pair=None):
        body = [self._header(kind)]
        r = self.run
        if kind in ("summary", "detailed") and r:
            body += [self._verdict(), self._tiles(), self._leaderboard(), "<h2>Accuracy</h2>",
                     '<div class="grid">', self._pr_curves(), self._speed_scatter() or self._f1_curves(),
                     self._per_class(), self._errors() if kind == "summary" else "", "</div>"]
            if kind == "detailed":
                body += ['<div class="grid">', self._f1_curves(), self._errors(), self._sizes(),
                         self._class_distribution(), "</div>"]
                for i in r.ranking():
                    res = r.results[i]
                    body += [f"<h2>{_sw(self.slots[i])}{escape(res.name)}</h2>", '<div class="grid">',
                             self._confusion(i, res), self._class_table(i, res), self._worst(i, res), "</div>"]
        elif kind == "comparison" and r:
            rank = r.ranking()
            if pair:
                a, b = pair
            elif len(rank) > 1:
                a, b = rank[1], rank[0]
            elif r.ground == "reference" and r.reference is not None and rank:
                a, b = r.reference, rank[0]
            else:
                a, b = 0, 0
            body += [self._verdict(), self._comparison(a, b)]
        if kind == "video" or (self.video and kind != "comparison"):
            for path, v in self.video:
                body.append(self._video_section(path, v))
        body.append(self._footer())
        return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
                "content='width=device-width,initial-scale=1'>"
                f"<title>{escape(self.title)}</title><style>{font_css()}{C.css_tokens()}{CSS}</style></head><body>"
                f"<div class='wrap'>{''.join(x for x in body if x)}</div><div id='tip'></div><script>{JS}</script>"
                "</body></html>")


# ------------------------------------------------------------------ exports
def write_exports(run, folder):
    """metrics.json, per_class.csv, per_image.csv. Returns the written paths."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    p = os.path.join(folder, "metrics.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(run.to_json(), f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    paths.append(p)
    p = os.path.join(folder, "per_class.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "class", "objects", "ap50", "ap50_95", "precision", "recall", "f1", "tp", "fp", "fn"])
        for _, res in run.scored:
            for x in res.class_rows():
                w.writerow([res.name, x["name"], x["gt"], f"{x['ap50']:.4f}", f"{x['ap']:.4f}", f"{x['p']:.4f}",
                            f"{x['r']:.4f}", f"{x['f1']:.4f}", x["tp"], x["fp"], x["fn"]])
    paths.append(p)
    p = os.path.join(folder, "per_image.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "image", "video", "frame", "time", "correct", "false_detections", "missed"] +
                   [M.ERROR_LABELS[t] for t in M.ERROR_TYPES + M.MISS_TYPES])
        for mi, res in run.scored:
            for ii, tp, fp, fn, pt, keep, gtt in run.image_errors(mi):
                it = run.items[ii]
                cnt = {t: 0 for t in M.ERROR_TYPES + M.MISS_TYPES}
                for t in list(pt) + list(gtt):
                    if t != "tp":
                        cnt[t] += 1
                w.writerow([res.name, it.path, it.meta.get("video", ""), it.meta.get("frame", ""), it.meta.get("time", ""),
                            tp, fp, fn] + [cnt[t] for t in M.ERROR_TYPES + M.MISS_TYPES])
    paths.append(p)
    return paths


def write_report(path, html):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(tmp, path)
    return path


_FIG_RE = None


def extract_figures(html):
    """[(title, description, svg)] from report fragments - the app shows the same charts natively (QtSvg)."""
    global _FIG_RE
    import re
    from html import unescape
    if _FIG_RE is None:
        _FIG_RE = re.compile(r'<figure class="card"[^>]*><figcaption><h3>([^<]*)</h3><p>([^<]*)</p></figcaption>'
                             r'<div class="chart">(<svg.*?</svg>)</div>', re.S)
    return [(unescape(t), unescape(d), svg) for t, d, svg in _FIG_RE.findall(html or "")]
