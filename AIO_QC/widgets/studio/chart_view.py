# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Charts for Data Studio (QPainter + numpy; no plotting library): histogram, bar, line, scatter and box plots with
nice axes, a legend and hover read-outs, in the theme's colours. Data is aggregated / sampled before drawing, so
charts stay instant on millions of rows. Save as PNG.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolTip, QVBoxLayout, QWidget,
)

from AIO_QC.utils import theme_color

SERIES = ["#1E88E5", "#FB8C00", "#43A047", "#E53935", "#8E24AA", "#00ACC1", "#C0CA33", "#6D4C41"]
TYPES = ["Histogram", "Bar", "Line", "Scatter", "Box"]
AGGS = ["count", "sum", "mean", "median", "min", "max"]
MAX_POINTS = 20_000


def nice_ticks(lo, hi, n=6):
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return []
    if hi <= lo:
        return [lo]
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + step * 1e-9:
        out.append(round(v, 12))
        v += step
    return out


def tick_text(v):
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:g}B"
    if a >= 1e6:
        return f"{v / 1e6:g}M"
    if a >= 1e4:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


class ChartCanvas(QWidget):
    """spec = {"type", "title", "x_label", "y_label", ...type data} — see ChartPanel._spec()."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spec = None
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._hits = []                                     # [(QRectF, text)]

    def set_spec(self, spec):
        self.spec = spec
        self.update()

    # ---- geometry
    def _plot_rect(self):
        return QRectF(64, 36, max(10, self.width() - 64 - 24), max(10, self.height() - 36 - 54))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._hits = []
        text, muted, line = theme_color("text"), theme_color("muted"), theme_color("line")
        sp = self.spec
        if not sp or sp.get("error"):
            p.setPen(muted)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, (sp or {}).get("error") or
                       "Choose what to plot above")
            return
        f = QFont(self.font())
        f.setBold(True)
        p.setFont(f)
        p.setPen(text)
        p.drawText(QRectF(12, 6, self.width() - 24, 24), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   sp.get("title", ""))
        p.setFont(self.font())
        R = self._plot_rect()
        t = sp["type"]
        if t in ("Bar", "Box"):
            self._categorical(p, R, sp, text, muted, line)
        else:
            self._numeric(p, R, sp, text, muted, line)
        p.setPen(muted)
        p.drawText(QRectF(R.left(), R.bottom() + 30, R.width(), 20), Qt.AlignmentFlag.AlignCenter,
                   sp.get("x_label", ""))
        p.save()
        p.translate(14, R.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-R.height() / 2, -10, R.height(), 20), Qt.AlignmentFlag.AlignCenter, sp.get("y_label", ""))
        p.restore()
        self._legend(p, sp, text)

    def _yaxis(self, p, R, lo, hi, muted, line):
        ticks = nice_ticks(lo, hi)
        for v in ticks:
            y = R.bottom() - (v - lo) / ((hi - lo) or 1) * R.height()
            p.setPen(QPen(line, 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(R.left(), y), QPointF(R.right(), y))
            p.setPen(muted)
            p.drawText(QRectF(0, y - 9, R.left() - 6, 18), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       tick_text(v))
        return lambda v: R.bottom() - (v - lo) / ((hi - lo) or 1) * R.height()

    def _numeric(self, p, R, sp, text, muted, line):
        xs_all = [np.asarray(s["x"], float) for s in sp["series"]]
        ys_all = [np.asarray(s["y"], float) for s in sp["series"]]
        if sp["type"] == "Histogram":
            edges = np.asarray(sp["edges"], float)
            xlo, xhi = float(edges[0]), float(edges[-1])
        else:
            xx = np.concatenate([x[np.isfinite(x)] for x in xs_all]) if xs_all else np.array([0.0])
            xlo, xhi = (float(xx.min()), float(xx.max())) if len(xx) else (0.0, 1.0)
        yy = np.concatenate([y[np.isfinite(y)] for y in ys_all]) if ys_all else np.array([0.0])
        ylo = 0.0 if sp["type"] in ("Histogram", "Line") and (not len(yy) or yy.min() >= 0) else \
            (float(yy.min()) if len(yy) else 0.0)
        yhi = float(yy.max()) if len(yy) else 1.0
        if yhi <= ylo:
            yhi = ylo + 1
        pad = (yhi - ylo) * 0.05
        yhi += pad
        if sp["type"] == "Scatter":
            ylo -= pad
        Y = self._yaxis(p, R, ylo, yhi, muted, line)
        X = lambda v: R.left() + (v - xlo) / ((xhi - xlo) or 1) * R.width()
        for v in nice_ticks(xlo, xhi):
            x = X(v)
            p.setPen(muted)
            label = sp["x_fmt"](v) if sp.get("x_fmt") else tick_text(v)
            p.drawText(QRectF(x - 50, R.bottom() + 4, 100, 18), Qt.AlignmentFlag.AlignHCenter, label)
        p.setPen(QPen(line, 1))
        p.drawLine(R.bottomLeft(), R.bottomRight())
        for i, (s, x, y) in enumerate(zip(sp["series"], xs_all, ys_all)):
            col = QColor(s.get("color") or SERIES[i % len(SERIES)])
            if sp["type"] == "Histogram":
                edges = sp["edges"]
                col.setAlpha(200)
                for j, c in enumerate(y):
                    r = QRectF(X(edges[j]) + 0.5, Y(c), max(1.0, X(edges[j + 1]) - X(edges[j]) - 1), R.bottom() - Y(c))
                    p.fillRect(r, col)
                    self._hits.append((r.adjusted(0, -R.height(), 0, 0),
                                       f"{tick_text(edges[j])} – {tick_text(edges[j + 1])}: {int(c):,}"))
            elif sp["type"] == "Line":
                p.setPen(QPen(col, 2))
                pts = [QPointF(X(a), Y(b)) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
                for a, b in zip(pts, pts[1:]):
                    p.drawLine(a, b)
                if len(pts) <= 60:
                    p.setBrush(col)
                    for q in pts:
                        p.drawEllipse(q, 2.5, 2.5)
                for a, b, q in zip(x, y, pts):
                    self._hits.append((QRectF(q.x() - 5, q.y() - 5, 10, 10), f"{s.get('name', '')} "
                                       f"{sp['x_fmt'](a) if sp.get('x_fmt') else tick_text(a)}: {b:,.4g}"))
            else:                                                       # scatter
                col.setAlpha(150 if len(x) < 3000 else 70)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(col)
                rad = 3.0 if len(x) < 3000 else 1.6
                for a, b in zip(x, y):
                    if math.isfinite(a) and math.isfinite(b):
                        p.drawEllipse(QPointF(X(a), Y(b)), rad, rad)

    def _categorical(self, p, R, sp, text, muted, line):
        cats = sp["cats"]
        n = max(1, len(cats))
        slot = R.width() / n
        if sp["type"] == "Bar":
            vals = [v for s in sp["series"] for v in s["y"] if v is not None and math.isfinite(v)]
            lo = min(0.0, min(vals)) if vals else 0.0
            hi = max(vals) * 1.05 if vals and max(vals) > 0 else 1.0
        else:
            vals = [v for b in sp["boxes"] for v in (b or (None,) * 5)[:5] if v is not None]
            lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
            pad = (hi - lo) * 0.05 or 1
            lo, hi = lo - pad, hi + pad
        Y = self._yaxis(p, R, lo, hi, muted, line)
        p.setPen(QPen(line, 1))
        p.drawLine(R.bottomLeft(), R.bottomRight())
        fm = p.fontMetrics()
        for i, c in enumerate(cats):
            x0 = R.left() + i * slot
            label = fm.elidedText(str(c), Qt.TextElideMode.ElideRight, int(max(slot - 4, 20)))
            p.setPen(muted)
            if slot < 46:
                p.save()
                p.translate(x0 + slot / 2 + 4, R.bottom() + 6)
                p.rotate(45)
                p.drawText(0, 8, fm.elidedText(str(c), Qt.TextElideMode.ElideRight, 80))
                p.restore()
            else:
                p.drawText(QRectF(x0, R.bottom() + 4, slot, 18), Qt.AlignmentFlag.AlignHCenter, label)
            if sp["type"] == "Bar":
                k = len(sp["series"])
                bw = slot * 0.78 / k
                for j, s in enumerate(sp["series"]):
                    v = s["y"][i]
                    if v is None or not math.isfinite(v):
                        continue
                    col = QColor(s.get("color") or SERIES[j % len(SERIES)])
                    top, base = Y(max(v, 0)), Y(min(v, 0))
                    r = QRectF(x0 + slot * 0.11 + j * bw, top, max(1.0, bw - 1), base - top)
                    p.fillRect(r, col)
                    self._hits.append((QRectF(r.x(), R.top(), r.width(), R.height()),
                                       f"{c}{' · ' + s['name'] if k > 1 else ''}: {v:,.4g}"))
            else:
                b = sp["boxes"][i]
                if not b:
                    continue
                mn, q1, md, q3, mx = b[:5]
                col = QColor(SERIES[0])
                cx, w = x0 + slot / 2, min(slot * 0.5, 60)
                p.setPen(QPen(col, 1.5))
                p.drawLine(QPointF(cx, Y(mn)), QPointF(cx, Y(q1)))
                p.drawLine(QPointF(cx, Y(q3)), QPointF(cx, Y(mx)))
                fill = QColor(col)
                fill.setAlpha(60)
                r = QRectF(cx - w / 2, Y(q3), w, Y(q1) - Y(q3))
                p.fillRect(r, fill)
                p.drawRect(r)
                p.setPen(QPen(col, 2.5))
                p.drawLine(QPointF(cx - w / 2, Y(md)), QPointF(cx + w / 2, Y(md)))
                self._hits.append((QRectF(x0, R.top(), slot, R.height()),
                                   f"{c}: min {mn:,.4g} · q1 {q1:,.4g} · median {md:,.4g} · q3 {q3:,.4g} · "
                                   f"max {mx:,.4g} (whiskers 1.5 × IQR)"))

    def _legend(self, p, sp, text):
        series = [s for s in sp.get("series", []) if s.get("name")]
        if len(series) < 2:
            return
        x = self.width() - 16
        y = 10
        for i, s in reversed(list(enumerate(series))):
            name = str(s["name"])[:24]
            w = p.fontMetrics().horizontalAdvance(name)
            x -= w + 22
            p.fillRect(QRectF(x, y + 4, 10, 10), QColor(s.get("color") or SERIES[i % len(SERIES)]))
            p.setPen(text)
            p.drawText(QPointF(x + 14, y + 13), name)

    def mouseMoveEvent(self, e):
        for r, t in self._hits:
            if r.contains(e.position()):
                QToolTip.showText(e.globalPosition().toPoint(), t, self)
                return
        QToolTip.hideText()


class ChartPanel(QWidget):
    """Controls + canvas. set_frame(df) after the data changes."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.st = settings
        self.df = None
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.type = QComboBox()
        self.type.addItems(TYPES)
        self.x, self.y, self.by = QComboBox(), QComboBox(), QComboBox()
        self.agg = QComboBox()
        self.agg.addItems(AGGS)
        for label, w in (("Chart", self.type), ("x", self.x), ("y", self.y), ("as", self.agg), ("colour by", self.by)):
            bar.addWidget(QLabel(label))
            bar.addWidget(w, 1 if w in (self.x, self.y, self.by) else 0)
        save = QPushButton("Save PNG…")
        save.clicked.connect(self._save)
        bar.addWidget(save)
        lay.addLayout(bar)
        self.canvas = ChartCanvas()
        lay.addWidget(self.canvas, 1)
        self.note = QLabel("")
        lay.addWidget(self.note)
        for w in (self.type, self.x, self.y, self.agg, self.by):
            w.currentIndexChanged.connect(self.redraw)
        self.type.currentIndexChanged.connect(self._enable)

    def set_frame(self, df, kinds: dict):
        self.df, self.kinds = df, kinds
        cols = [str(c) for c in df.columns]
        num = [c for c in cols if kinds.get(c) in ("number", "integer")]
        for w, items, none in ((self.x, cols, False), (self.y, num, True), (self.by, cols, True)):
            cur = w.currentText()
            w.blockSignals(True)
            w.clear()
            if none:
                w.addItem("(none)")
            w.addItems(items)
            if cur in items or (none and cur == "(none)"):
                w.setCurrentText(cur)
            w.blockSignals(False)
        self._enable()
        self.redraw()

    def _enable(self):
        t = self.type.currentText()
        self.y.setEnabled(t != "Histogram")
        self.agg.setEnabled(t in ("Bar", "Line"))
        self.by.setEnabled(t in ("Line", "Scatter", "Bar", "Histogram"))
        self.redraw()

    def redraw(self):
        try:
            spec = self._spec()
        except Exception as exc:
            spec = {"error": f"Cannot draw this: {exc}"}
        self.canvas.set_spec(spec)
        self.note.setText((spec or {}).get("note", "") if spec else "")

    # ---- data -> spec
    def _spec(self):
        import pandas as pd
        df = self.df
        if df is None or not len(df.columns):
            return None
        t, x = self.type.currentText(), self.x.currentText()
        y = self.y.currentText() if self.y.currentText() != "(none)" else None
        by = self.by.currentText() if self.by.isEnabled() and self.by.currentText() not in ("(none)", x) else None
        agg = self.agg.currentText()
        if x not in df.columns:
            return None
        kx = self.kinds.get(x)
        note = ""
        if t == "Histogram":
            if kx not in ("number", "integer", "date"):
                return {"error": "A histogram needs a number or date column as x — try a Bar chart for categories"}
            xs = df[x]
            date = kx == "date"
            v = (pd.to_datetime(xs, errors="coerce").astype("int64") / 1e9) if date else pd.to_numeric(xs, errors="coerce")
            v = v[np.isfinite(v.to_numpy(dtype=float, na_value=np.nan))]
            if not len(v):
                return {"error": "no values to plot"}
            lo, hi = float(v.min()), float(v.max())
            edges = np.histogram_bin_edges(v.to_numpy(float), bins=40 if hi > lo else 1,
                                           range=(lo, hi) if hi > lo else (lo - 0.5, lo + 0.5))
            series = []
            groups = self._groups(df, by, 6) if by else [(None, df.index)]
            for name, idx in groups:
                cnt, _e = np.histogram(v.reindex(idx).dropna().to_numpy(float), bins=edges)
                series.append({"name": None if name is None else str(name), "x": edges[:-1], "y": cnt})
            return {"type": t, "title": f"Distribution of {x}", "x_label": x, "y_label": "rows", "edges": edges,
                    "series": series, "x_fmt": _date_fmt if date else None}
        if t == "Bar":
            top = df[x].value_counts(dropna=False).head(30).index.tolist()
            sub = df[df[x].isin(top)]
            series = []
            groups = self._groups(sub, by, 5) if by else [(None, sub.index)]
            for name, idx in groups:
                part = sub.loc[idx]
                g = part.groupby(x, dropna=False)
                vals = g.size() if agg == "count" or not y else getattr(pd.to_numeric(part[y], errors="coerce")
                                                                      .groupby(part[x], dropna=False), agg)()
                lookup = {_key(k): float(v) for k, v in vals.items()}
                series.append({"name": None if name is None else str(name), "y": [lookup.get(_key(c)) for c in top]})
            if df[x].nunique(dropna=False) > 30:
                note = "Top 30 categories by row count"
            ylab = "rows" if agg == "count" or not y else f"{agg} of {y}"
            return {"type": t, "title": f"{ylab} by {x}", "x_label": x, "y_label": ylab, "cats": [_c(c) for c in top],
                    "series": series, "note": note}
        if t == "Box":
            if not y:
                return {"error": "A box plot needs a number column as y"}
            if kx in ("number", "integer") and df[x].nunique() > 12:
                cats, sub = ["all rows"], [("all rows", df[y])]
            else:
                top = df[x].value_counts().head(12).index.tolist()
                cats = [_c(c) for c in top]
                sub = [(c, df.loc[df[x] == c, y]) for c in top]
            boxes = []
            for _c_, s in sub:
                a = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
                if not len(a):
                    boxes.append(None)
                    continue
                q1, md, q3 = np.percentile(a, [25, 50, 75])
                iqr = q3 - q1
                lo = float(a[a >= q1 - 1.5 * iqr].min())
                hi = float(a[a <= q3 + 1.5 * iqr].max())
                boxes.append((lo, float(q1), float(md), float(q3), hi))
            return {"type": t, "title": f"{y} by {x}", "x_label": x, "y_label": y, "cats": cats, "boxes": boxes,
                    "series": []}
        if not y:
            return {"error": f"A {t.lower()} chart needs a number column as y"}
        date = kx == "date"
        xv = (pd.to_datetime(df[x], errors="coerce").astype("int64") / 1e9) if date else \
            pd.to_numeric(df[x], errors="coerce")
        if xv.notna().sum() == 0:
            return {"error": f"{x} holds no numbers or dates — use a Bar chart for categories"}
        yv = pd.to_numeric(df[y], errors="coerce")
        series = []
        groups = self._groups(df, by, 8) if by else [(None, df.index)]
        for name, idx in groups:
            xs, ys = xv.loc[idx], yv.loc[idx]
            m = xs.notna() & ys.notna()
            xs, ys = xs[m], ys[m]
            if t == "Line":
                if xs.nunique() > 400:                                 # many x values: bin them
                    bins = np.linspace(xs.min(), xs.max(), 201)
                    key = pd.cut(xs, bins, include_lowest=True, labels=bins[:-1])
                    g = ys.groupby(key, observed=True)
                    s = g.size() if agg == "count" else getattr(g, agg)()
                    xs2, ys2 = np.asarray(s.index, float), s.to_numpy(float)
                    note = "x binned into 200 steps"
                else:
                    g = ys.groupby(xs)
                    s = g.size() if agg == "count" else getattr(g, agg)()
                    xs2, ys2 = s.index.to_numpy(float), s.to_numpy(float)
                series.append({"name": None if name is None else str(name), "x": xs2, "y": ys2})
            else:
                if len(xs) > MAX_POINTS:
                    pick = np.random.default_rng(1).choice(len(xs), MAX_POINTS, replace=False)
                    xs, ys = xs.iloc[pick], ys.iloc[pick]
                    note = f"{MAX_POINTS:,} random points shown"
                series.append({"name": None if name is None else str(name), "x": xs.to_numpy(float),
                               "y": ys.to_numpy(float)})
        ylab = y if t == "Scatter" else (f"{agg} of {y}" if agg != "count" else "rows")
        return {"type": t, "title": f"{ylab} vs {x}", "x_label": x, "y_label": ylab, "series": series, "note": note,
                "x_fmt": _date_fmt if date else None}

    @staticmethod
    def _groups(df, by, k):
        top = df[by].value_counts().head(k).index.tolist()
        return [(c, df.index[df[by] == c]) for c in top]

    def _save(self):
        from AIO_QC.utils.save_dirs import ask_save
        path, _ = ask_save(self, "Save chart", "chart.png", "PNG (*.png)", kind="images")
        if path:
            self.canvas.grab().save(path, "PNG")


def _key(v):
    return "\x00missing" if v is None or (isinstance(v, float) and math.isnan(v)) or str(v) in ("<NA>", "NaT") else v


def _c(v):
    return "(missing)" if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _date_fmt(v):
    import datetime as _dt
    try:
        return _dt.datetime.fromtimestamp(v, tz=_dt.timezone.utc).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return tick_text(v)
