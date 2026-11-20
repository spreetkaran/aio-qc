# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Reports for classifier runs (cls_eval.ClsLabRun): the same single-file HTML as detection reports, with the parts a
classifier is judged by — top-1 / top-5 accuracy, per-class precision / recall / F1, confusion matrices, accuracy vs
how many images it answers (confidence), speed vs accuracy, the surest mistakes as pictures, A-vs-B, and video
behaviour (class flips, agreement). The app's results view shows these same charts natively. Qt-free.
"""
from __future__ import annotations

import csv
import json
import os
from html import escape

import numpy as np

from AIO_QC.core.testing import charts as C
from AIO_QC.core.testing.annotate import draw_answers, draw_cls, model_bgr, thumb
from AIO_QC.core.testing.report import CSS, JS, TYPES, ReportBuilder, _b64, _fig, _sw, _table, font_css, _num


class ClsReportBuilder(ReportBuilder):
    """A ReportBuilder for ClsLabRun: every detection-only piece replaced."""

    def _leaderboard(self):
        rows = self.run.summary_table()
        if not rows:
            return ""
        top = max(x["top1"] for x in rows) or 1
        out = []
        for x in rows:
            slot = self.slots[x["idx"]]
            bar = (f'<span class="cellbar"><i style="width:{x["top1"] / top * 90:.0f}px;background:var(--s{slot % 8})">'
                   f'</i>{x["top1"]:.1%}</span>')
            out.append([(str(x["rank"]), x["rank"]), f'{_sw(slot)}{escape(x["model"])}', (bar, x["top1"]),
                        (f'{x["top5"]:.1%}', x["top5"]), (f'{x["precision"]:.3f}', x["precision"]),
                        (f'{x["recall"]:.3f}', x["recall"]), (f'{x["f1"]:.3f}', x["f1"]),
                        (f'{x["answered"]:.0%} · {x["accuracy_answered"]:.1%}', x["accuracy_answered"]),
                        (_num(x["speed_ms"], 1), x["speed_ms"] or 1e9), (_num(x["fps"], 0), x["fps"] or 0),
                        (f'{x["wrong"]:,}', x["wrong"])])
        head = ["#", "Model", "Top-1", "Top-5", "Precision", "Recall", "F1", "At confidence: answered · right",
                "ms / image", "Images/s", "Wrong"]
        thr = rows[0]["op_conf"]
        note = ("<p class='note'>Top-1: the model's best class is the image's class. Precision / recall / F1: macro "
                "averages over the classes. " + (f"At confidence {thr:.2f}: answers below it count as unsure — the "
                                                  "share it answers and how often those are right. " if thr else "")
                + "Click a column to sort.</p>")
        return "<h2>Leaderboard</h2>" + _table(head, out, set(range(2, 11)) | {0}) + note

    def _pr_curves(self):                    # accuracy vs answered: the classifier's operating curve
        series = []
        for i, res in self.run.scored:
            if not res.curve:
                continue
            series.append(dict(name=res.name, color=self.slots[i], x=[c[1] for c in res.curve],
                               y=[c[2] for c in res.curve]))
        if not series:
            return ""
        svg = C.line_chart(series, "Share of images answered", "Accuracy on those", (0, 1), (0, 1), 560, 330,
                           self.ink, "Accuracy vs coverage", x_fmt=lambda v: f"{v:.0%}", y_fmt=lambda v: f"{v:.0%}")
        return _fig("Accuracy vs coverage", "Raising the confidence a model needs before it answers: fewer images "
                    "answered (left), more of them right (up). A curve higher up is better.", svg)

    def _f1_curves(self, only=None):
        return ""

    def _speed_scatter(self):
        pts = [dict(name=res.name, color=self.slots[i], x=res.speed_ms, y=res.top1,
                    tip=f"{res.name}: {res.speed_ms:.1f} ms · top-1 {res.top1:.1%}")
               for i, res in self.run.scored if res.speed_ms]
        if len(pts) < 2:
            return ""
        svg = C.scatter(pts, "Milliseconds per image (lower is faster)", "Top-1 accuracy", 560, 330, self.ink,
                        "Speed vs accuracy", x_fmt=lambda v: f"{v:g}", y_fmt=lambda v: f"{v:.0%}")
        return _fig("Speed vs accuracy", "Up and left is better. Speed measured on this machine; image loading "
                    "excluded, first batch ignored.", svg)

    def _per_class(self, metric="f1", cap=24):
        r = self.run
        rank = r.ranking()
        if not rank:
            return ""
        best = r.results[rank[0]]
        cats = sorted(best.classes, key=lambda k: best.classes[k][metric])[:cap]
        series = [dict(name=r.results[i].name, color=self.slots[i],
                       values=[(r.results[i].classes.get(k) or {}).get(metric, 0.0) for k in cats],
                       tips=[f"{r.results[i].name} · {k}: F1 {(r.results[i].classes.get(k) or {}).get('f1', 0):.3f}, "
                             f"recall {(r.results[i].classes.get(k) or {}).get('recall', 0):.0%}" for k in cats])
                  for i in rank[:6]]
        svg = C.bar_chart(cats, series, 560, self.ink, "F1 per class", value_fmt=lambda v: f"{v:.2f}", max_value=1.0)
        return _fig("F1 per class", "Weakest classes first (by the best model).", svg)

    def _errors(self, only=None):
        return ""

    def _sizes(self):
        return ""

    def _class_distribution(self):
        cc = self.run.class_counts()
        if not cc:
            return ""
        cats = sorted(cc, key=lambda k: -cc[k])[:30]
        svg = C.bar_chart(cats, [dict(name="images", color=0, values=[cc[k] for k in cats])], 560, self.ink,
                          "Images per class", value_fmt=lambda v: f"{int(v):,}", legend=False)
        return _fig("Images per class", "What the scores rest on: classes with few images swing a lot.", svg)

    def _confusion(self, i, res, title=None):
        names = [k for k, _ in sorted(res.classes.items(), key=lambda kv: -kv[1]["support"])
                 if res.classes[k]["support"] or any((res.confusion.get(g) or {}).get(k) for g in res.confusion)]
        if not names:
            return ""
        names = names[:30]
        m = np.array([[(res.confusion.get(a) or {}).get(b, 0) for b in names] for a in names], float)
        svg = C.heatmap(m, names, names, self.ink, f"Confusion matrix · {res.name}", normalize="row",
                        y_label="True class" if self.run.ground == "labels" else "Reference model",
                        link=lambda a, b: f"cm:{i}:{names[a]}:{names[b]}" if a != b else f"class:{i}:{names[a]}")
        rows = [[escape(names[a])] + [f"{int(v):,}" for v in m[a]] for a in range(len(names))]
        return _fig(title or f"Confusion matrix · {res.name}", "Rows: the image's class; columns: the model's best "
                    "class. Colour = share of the row.", svg,
                    _table(["True \\ predicted"] + names, rows, set(range(1, len(names) + 1)), sortable=False),
                    wide=True)

    def _class_table(self, i, res):
        rows = [[escape(x["name"]), (f"{x['support']:,}", x["support"]), (f"{x['precision']:.3f}", x["precision"]),
                 (f"{x['recall']:.3f}", x["recall"]), (f"{x['f1']:.3f}", x["f1"]), (f"{x['fp']:,}", x["fp"]),
                 (f"{x['fn']:,}", x["fn"])] for x in res.class_rows()]
        return (f'<figure class="card" style="grid-column:1/-1"><figcaption><h3>Per class · {escape(res.name)}</h3>'
                '<p>Weakest first. Taken for it: images of other classes it was called; missed: its images called '
                'something else.</p></figcaption>' +
                _table(["Class", "Images", "Precision", "Recall", "F1", "Taken for it", "Missed"], rows,
                       set(range(1, 7))) + "</figure>")

    # ---- pictures: the image with the model's answer in a band under it (used by the app's viewers and galleries)
    def truth_of(self, item_idx):
        """The image's class names (labels, or the reference model's answer), or None."""
        r = self.run
        if r.ground == "labels":
            return list(r.items[item_idx].tags or []) or None
        if r.ground == "reference" and r.reference is not None:
            p = r.preds[r.reference][item_idx]
            return [p[0][0]] if p and p[0][1] >= r._ref_conf() else None
        return None

    def model_conf(self, m):
        res = self.run.results[m] if 0 <= m < len(self.run.results) else None
        if res is not None:
            return res.op_conf
        if self.run.ground == "reference" and m == self.run.reference:
            return self.run._ref_conf()
        return float(self.run.op_conf) if isinstance(self.run.op_conf, (int, float)) else 0.0

    def plot_image(self, item_idx, mi, show_gt=True, errors=True, conf=None, tw=None, img=None, compact=False):
        if img is None:
            if self.get_frame is None:
                return None
            img = self.get_frame(self.run.items[item_idx])
            if img is None:
                return None
        conf = self.model_conf(mi) if conf is None else conf
        truth = self.truth_of(item_idx) if show_gt and not (self.run.ground == "reference"
                                                             and mi == self.run.reference) else None
        top = self.run.preds[mi][item_idx] or []
        return draw_cls(img, top, truth, model_bgr(self.slots[mi]), conf, tw, compact=compact)

    def shot_image(self, item_idx, mi, focus=True, compare=None, tw=420):
        if self.get_frame is None:
            return None
        img = self.get_frame(self.run.items[item_idx])
        if img is None:
            return None
        if compare:
            import cv2
            a, b = (self.plot_image(item_idx, m, True, True, None, tw * 0.3, img=img, compact=True) for m in compare)
            h = max(a.shape[0], b.shape[0])
            a = cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(250, 250, 250))
            b = cv2.copyMakeBorder(b, 0, h - b.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(250, 250, 250))
            if a.shape[1] != b.shape[1]:
                b = cv2.resize(b, (a.shape[1], int(b.shape[0] * a.shape[1] / b.shape[1])))
                h = max(a.shape[0], b.shape[0])
                a = cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(250, 250, 250))
                b = cv2.copyMakeBorder(b, 0, h - b.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(250, 250, 250))
            return np.hstack([a, np.full((h, max(4, h // 120), 3), 255, np.uint8), b])
        return self.plot_image(item_idx, mi, True, True, None, tw * 0.55, img=img, compact=True)

    def cls_shot(self, item_idx, caption):
        """A thumbnail of the image with a caption (a classifier has nothing to draw on it)."""
        if self.get_frame is None:
            return ""
        it = self.run.items[item_idx]
        try:
            im = self.get_frame(it)
        except Exception:                                         # noqa: BLE001
            im = None
        if im is None:
            return ""
        jpg = thumb(im, 300, 78)
        where = it.meta.get("video")
        loc = (f"{os.path.basename(where)} · frame {it.meta.get('frame')} · {it.meta.get('time', 0):.2f}s"
               if where else os.path.basename(it.path))
        return (f'<div class="shot"><img src="{_b64(jpg)}" alt="{escape(loc)}"><div><b>{escape(caption)}</b>'
                f'<br>{escape(loc)}</div></div>')

    def _worst(self, i, res):
        rows = self.run.mistakes(i, self.gallery_n)
        shots = "".join(self.cls_shot(ii, f"{', '.join(g)} → {p[0][0]} {p[0][1]:.0%}") for ii, g, p in rows)
        if not shots:
            return ""
        return (f'<figure class="card" style="grid-column:1/-1"><figcaption><h3>Surest mistakes · '
                f'{escape(res.name)}</h3><p>True class → the model\'s answer, the most confident first: often a '
                'wrong label rather than a wrong model.</p></figcaption>'
                f'<div class="gallery">{shots}</div></figure>')

    def _comparison(self, a, b):
        r = self.run
        cmp = r.comparison(a, b)
        if cmp is None:
            return ""
        ra, rb = r.results[a], r.results[b]
        head = (f"<h2>{_sw(self.slots[a])}{escape(ra.name)} vs {_sw(self.slots[b])}{escape(rb.name)}</h2>"
                f"<p>Top-1 {ra.top1:.1%} → {rb.top1:.1%} ({cmp['top1_delta'] * 100:+.1f} points) · macro F1 "
                f"{ra.f1:.3f} → {rb.f1:.3f} · {len(cmp['b_better']):,} image(s) only {escape(rb.name)} gets right, "
                f"{len(cmp['a_better']):,} only {escape(ra.name)}.</p>")
        rows = [x for x in cmp["per_class"] if x["a"] is not None or x["b"] is not None]
        svg = C.diverging_bars([x["name"] for x in rows], [x["delta"] for x in rows], 560, self.ink,
                               f"Recall per class: {rb.name} minus {ra.name}", value_fmt=lambda v: f"{v:+.0%}",
                               labels=(f"{ra.name} better", f"{rb.name} better")) if rows else ""
        fig = _fig("Per class", "How much more (blue) or less (red) of each class the second model gets right.",
                   svg) if svg else ""
        def gal(lst, who):
            shots = "".join(self.cls_shot(ii, f"only {who} right") for ii, _c in lst[:self.gallery_n])
            return f'<div class="gallery">{shots}</div>' if shots else ""
        return (head + '<div class="grid">' + fig + "</div>" +
                f"<h3>Only {escape(rb.name)} right</h3>" + gal(cmp["b_better"], rb.name) +
                f"<h3>Only {escape(ra.name)} right</h3>" + gal(cmp["a_better"], ra.name))

    # ---- videos: a classifier's answer per sampled frame — class flips, drop-outs and agreement over time
    def _video_section(self, path, v):
        if v.get("task") != "classify":
            return super()._video_section(path, v)
        if not v["models"]:
            return ""
        out = [f"<h2>{escape(os.path.basename(path))}</h2>"]
        n = max(1, len(v["times"]))
        rows = []
        for i, m in enumerate(v["models"]):
            slot = self.slots[i] if i < len(self.slots) else i
            rows.append([f"{_sw(slot)}{escape(m['label'])}", (f"{m['op_conf']:.2f}", m["op_conf"]),
                         (f"{m['frames_with_dets'] / n:.0%}", m["frames_with_dets"]), (f"{m['tracks']:,}", m["tracks"]),
                         (f"{m['blip']:,}", m["blip"]), (f"{m['flicker']:,}", m["flicker"]),
                         (f"{m['stability'] * 100:.1f}%", m["stability"])])
        out.append('<div class="grid one"><figure class="card"><figcaption><h3>Stability</h3><p>A classifier should '
                   'keep its answer while the scene stays the same. Class runs: stretches with one answer. Brief '
                   'flips: an answer that lasts at most two samples. Drop-outs: frames where it is suddenly unsure. '
                   'Stability: share of frames in neither.</p></figcaption>' +
                   _table(["Model", "Confidence", "Frames answered", "Class runs", "Brief flips", "Drop-outs",
                           "Stability"], rows, set(range(1, 7))) + "</figure></div>")
        figs = []
        times = v["times"]
        cls = sorted({c for m in v["models"] for c in m["per_class"]})
        series = []
        for i, m in enumerate(v["models"]):
            p = (v.get("preds") or [[]] * len(v["models"]))[i] or []
            series.append(dict(name=m["label"], color=self.slots[i] if i < len(self.slots) else i,
                               y=[float(a[0][4]) if a is not None and len(a) else 0.0 for a in p]))
        events = []
        for e in v["events"]:
            if e["type"] in ("flicker", "blip"):
                mi = next(i for i, m in enumerate(v["models"]) if m["label"] == e["model"])
                what = "brief flip" if e["type"] == "blip" else "drop-out"
                events.append(dict(t=e["time"], color=self.slots[mi] if mi < len(self.slots) else mi,
                                   tip=f"{e['model']}: {what} ({e['cls']}) at {e['time']:.2f}s, frame {e['frame']}",
                                   link=f"vframe:{e['frame']}"))
        if any(len(x["y"]) == len(times) for x in series):
            svg = C.timeline(times, [x for x in series if len(x["y"]) == len(times)], events[:3000], 1080,
                             240 + 10 * len(series), self.ink, "Confidence over time", "Confidence", y_max=1.0,
                             y_fmt=lambda x: f"{x:.1f}")
            figs.append(_fig("Confidence over time", "How sure each model is of its answer per sampled frame; ticks "
                             "under the axis mark brief flips and drop-outs (click one to see the frame).", svg,
                             wide=True))
        if v["pairs"]:
            base = len(v["models"])
            ps = [dict(name=f"{p['a']} vs {p['b']}", color=(base + k) % 8, y=p["agreement"].tolist())
                  for k, p in enumerate(v["pairs"][:max(1, 8 - base)])]
            svg = C.timeline(times, ps, None, 1080, 220, self.ink, "Model agreement", "Same answer", y_max=1.0,
                             y_fmt=lambda x: f"{x:.1f}")
            prow = [[escape(p["a"]), escape(p["b"]), (f"{p['mean']:.1%}", p["mean"])] for p in v["pairs"]]
            figs.append(_fig("Agreement between models", "1 = both models give the same class; dips are where one of "
                             "them is wrong.", svg, _table(["Model A", "Model B", "Same answer"], prow, {2}), wide=True))
        if cls:
            bs = [dict(name=m["label"], color=self.slots[i] if i < len(self.slots) else i,
                       values=[m["per_class"].get(c, 0) for c in cls]) for i, m in enumerate(v["models"])]
            figs.append(_fig("Frames per class", "How many sampled frames each model gave each class.",
                             C.bar_chart(cls, bs, 560, self.ink, "Frames per class", value_fmt=lambda x: f"{int(x):,}")))
        out.append('<div class="grid">' + "".join(figs) + "</div>")
        if self.get_frame is not None:
            shots, seen = [], []
            span = (times[-1] - times[0]) if len(times) > 1 else 1.0
            min_gap = span / max(1, self.gallery_n * 2)
            for e in sorted(v["events"], key=lambda e: (e["type"] != "disagree", e.get("score", 1))):
                if len(shots) >= self.gallery_n:
                    break
                if any(abs(e["time"] - t) < min_gap for t in seen):
                    continue
                seen.append(e["time"])
                img = self.get_frame(dict(video=path, frame=e["frame"]))
                if img is None:
                    continue
                ans = []
                for mi, m in enumerate(v["models"]):
                    p = (v.get("preds") or [[]] * len(v["models"]))[mi]
                    names = (v.get("model_names") or [{}] * len(v["models"]))[mi] or {}
                    a = p[e["sample"]] if p and e["sample"] < len(p) else None
                    col = model_bgr(self.slots[mi] if mi < len(self.slots) else mi)
                    if a is not None and len(a):
                        ans.append((m["label"], names.get(int(a[0][5]), str(int(a[0][5]))), float(a[0][4]), col))
                    else:
                        ans.append((m["label"], None, 0.0, col))
                kind = {"disagree": "Models disagree", "flicker": "Drop-out", "blip": "Brief flip"}[e["type"]]
                shots.append(f'<div class="shot"><img src="{_b64(thumb(draw_answers(img, ans, 420), 420, 78))}">'
                             f'<div><b>{escape(kind)}</b> · {escape(e["model"])}<br>frame {e["frame"]} · '
                             f'{e["time"]:.2f}s</div></div>')
            if shots:
                out.append('<div class="grid one"><figure class="card"><figcaption><h3>Moments worth a look</h3><p>'
                           'Disagreements first, then brief flips / drop-outs, with every model\'s answer.</p>'
                           f'</figcaption><div class="gallery">{"".join(shots)}</div></figure></div>')
        return "".join(out)

    def _footer(self):
        r = self.run
        import datetime as _dt
        lines = ["Classification metrics: top-1 = the best class is the image's class (several classes: any of "
                 "them); top-5 = it is among the five best; precision / recall / F1 per class, macro averaged over the "
                 "classes that have images."]
        if r:
            for s in r.specs:
                info = r.infos.get(s.key) or {}
                lines.append(f"{s.label}: {s.path} · fingerprint {s.key} · image size {s.imgsz or 'model default'} · "
                             f"{info.get('backend') or 'cached'}")
            if r.source:
                lines.append(f"Test data: {r.source}")
        lines.append(f"Generated by AIO QC {self.version} on {_dt.datetime.now():%Y-%m-%d %H:%M}.")
        return "<footer>" + "".join(f"<p>{escape(x)}</p>" for x in lines) + "</footer>"

    def build(self, kind="summary", pair=None):
        body = [self._header(kind)]
        r = self.run
        if kind in ("summary", "detailed") and r:
            body += [self._verdict(), self._tiles(), self._leaderboard(), "<h2>Accuracy</h2>", '<div class="grid">',
                     self._pr_curves(), self._speed_scatter(), self._per_class(), self._class_distribution(),
                     "</div>"]
            ranks = r.ranking() if kind == "detailed" else r.ranking()[:1]
            for i in ranks:
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


def write_cls_exports(run, folder):
    """metrics.json, per_class.csv, per_image.csv for a classifier run."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    p = os.path.join(folder, "metrics.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(run.to_json(), f, indent=1, default=str)
    paths.append(p)
    p = os.path.join(folder, "per_class.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "class", "images", "precision", "recall", "f1", "tp", "fp", "fn"])
        for _, res in run.scored:
            for x in res.class_rows():
                w.writerow([res.name, x["name"], x["support"], f"{x['precision']:.4f}", f"{x['recall']:.4f}",
                            f"{x['f1']:.4f}", x["tp"], x["fp"], x["fn"]])
    paths.append(p)
    p = os.path.join(folder, "per_image.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "image", "video", "frame", "time", "true", "predicted", "confidence", "correct", "top5"])
        for mi, res in run.scored:
            for i, g, pr, right in res.per_image:
                it = run.items[run.eval_idx[i]]
                w.writerow([res.name, it.path, it.meta.get("video", ""), it.meta.get("frame", ""),
                            it.meta.get("time", ""), ";".join(g), pr[0][0], f"{pr[0][1]:.4f}", int(right),
                            ";".join(f"{k}:{c:.3f}" for k, c in pr)])
    paths.append(p)
    return paths


__all__ = ["ClsReportBuilder", "write_cls_exports", "TYPES"]
