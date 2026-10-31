# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Lightweight, theme-aware charts (QPainter + numpy; no matplotlib). Built for scale: data is binned / rasterised
once, painting only blits the result plus a few interactive marks.

    StripChart    per-class distribution strip (density + normal-range band + median + clickable outlier dots)
    ScatterBrush  log–log width × height density scatter with a draggable selection rectangle
    CoverageMap   every file as a cell coloured by review state, sectioned by split; click to open
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPalette, QPen
from PyQt6.QtWidgets import QSizePolicy, QToolTip, QWidget

from AIO_QC.utils import theme_color

LABEL_W = 150
ROW_H = 34
AXIS_H = 34


def _nice_ticks(lo: float, hi: float, log: bool):
    """Tick positions in data space (log10 values when log)."""
    if log:
        a, b = math.floor(lo), math.ceil(hi)
        return [float(v) for v in range(a, b + 1)]
    span = hi - lo or 1.0
    step = 10 ** math.floor(math.log10(span / 5))
    for m in (1, 2, 5, 10):
        if span / (step * m) <= 6:
            step *= m
            break
    start = math.ceil(lo / step) * step
    return list(np.arange(start, hi + step * 0.5, step))


# =============================================================================
class StripChart(QWidget):
    """
    groups: [{"label", "values": np.ndarray (already transformed, e.g. log), "lo", "hi", "med",
              "outliers": [(value, row, "small" | "large")]}]
    fmt(value_in_chart_space) -> str for axis labels and tooltips.
    """
    activated = pyqtSignal(object)            # the ResultRow of a clicked outlier

    def __init__(self, parent=None):
        super().__init__(parent)
        self.groups: list[dict] = []
        self.fmt = lambda v: f"{v:.3g}"
        self.log_axis = False
        self.lo = self.hi = 0.0
        self._dots: list[tuple[QPointF, object, str]] = []
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, groups, fmt, log_axis: bool):
        self.groups, self.fmt, self.log_axis = groups, fmt, log_axis
        allv = [g["values"] for g in groups if len(g["values"])]
        if allv:
            cat = np.concatenate(allv)
            self.lo, self.hi = float(np.nanpercentile(cat, 0.1)), float(np.nanpercentile(cat, 99.9))
            ol = [v for g in groups for v, *_ in g["outliers"]]
            if ol:
                self.lo, self.hi = min(self.lo, min(ol)), max(self.hi, max(ol))
            pad = (self.hi - self.lo) * 0.04 or 1.0
            self.lo, self.hi = self.lo - pad, self.hi + pad
        for g in groups:                                   # pre-bin once: painting is O(bins), not O(shapes)
            v = g["values"]
            h, _ = np.histogram(v, bins=160, range=(self.lo, self.hi)) if len(v) else (np.zeros(160), None)
            g["hist"] = h / h.max() if h.max() else h
        self.setFixedHeight(AXIS_H + ROW_H * max(1, len(groups)) + 8)
        self.update()

    def _x(self, v, w):
        return LABEL_W + (v - self.lo) / max(self.hi - self.lo, 1e-12) * (w - LABEL_W - 16)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        text, muted, line = pal.color(QPalette.ColorRole.Text), theme_color("muted"), theme_color("line")
        w = self.width()
        self._dots = []
        if not self.groups:
            p.setPen(muted)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Run a scan to see the distributions")
            return
        band = QColor(theme_color("ok"))
        band.setAlpha(38)
        for i, g in enumerate(self.groups):
            y = 6 + i * ROW_H
            mid = y + ROW_H / 2
            if i % 2:
                p.fillRect(QRect(0, y, w, ROW_H), pal.color(QPalette.ColorRole.AlternateBase))
            p.setPen(text)
            p.drawText(QRect(8, y, LABEL_W - 16, ROW_H), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       self.fontMetrics().elidedText(f"{g['label']}  ({len(g['values']):,})", Qt.TextElideMode.ElideRight,
                                                     LABEL_W - 16))
            # normal range band
            x1, x2 = self._x(g["lo"], w), self._x(g["hi"], w)
            p.fillRect(QRectF(x1, y + 5, max(1.0, x2 - x1), ROW_H - 10), band)
            # density strip
            nb = len(g["hist"])
            bw = (w - LABEL_W - 16) / nb
            for b, d in enumerate(g["hist"]):
                if d <= 0:
                    continue
                c = QColor(muted)
                c.setAlphaF(0.15 + 0.75 * float(d))
                p.fillRect(QRectF(LABEL_W + b * bw, mid - 3 - 7 * d, max(bw, 1.0), 6 + 14 * d), c)
            # median
            xm = self._x(g["med"], w)
            p.setPen(QPen(text, 2))
            p.drawLine(QPointF(xm, y + 6), QPointF(xm, y + ROW_H - 6))
            # outliers
            for v, row, kind in g["outliers"]:
                c = theme_color("error" if kind == "large" else "warn")
                pt = QPointF(self._x(v, w), mid)
                p.setPen(QPen(pal.color(QPalette.ColorRole.Base), 1.5))
                p.setBrush(c)
                p.drawEllipse(pt, 4.5, 4.5)
                self._dots.append((pt, row, f"{g['label']}: {self.fmt(v)} ({kind})"))
        # axis
        ya = 6 + len(self.groups) * ROW_H + 4
        p.setPen(QPen(line, 1))
        p.drawLine(QPointF(LABEL_W, ya), QPointF(w - 16, ya))
        p.setPen(muted)
        for t in _nice_ticks(self.lo, self.hi, self.log_axis):
            if self.lo <= t <= self.hi:
                xt = self._x(t, w)
                p.drawLine(QPointF(xt, ya), QPointF(xt, ya + 4))
                p.drawText(QRectF(xt - 40, ya + 5, 80, 16), Qt.AlignmentFlag.AlignHCenter, self.fmt(t))
        p.end()

    def _hit(self, pos: QPoint):
        best, bd = None, 8.0
        for pt, row, tip in self._dots:
            d = math.hypot(pt.x() - pos.x(), pt.y() - pos.y())
            if d < bd:
                best, bd = (row, tip), d
        return best

    def mouseMoveEvent(self, e):
        h = self._hit(e.position().toPoint())
        self.setCursor(Qt.CursorShape.PointingHandCursor if h else Qt.CursorShape.ArrowCursor)
        if h:
            QToolTip.showText(e.globalPosition().toPoint(), h[1] + "\nclick to open", self)

    def mouseReleaseEvent(self, e):
        h = self._hit(e.position().toPoint())
        if h and e.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(h[0])


# =============================================================================
class ScatterBrush(QWidget):
    """
    Log–log density scatter of width × height. Drag to select a rectangle; click without dragging to clear.
    selectionChanged(wmin, wmax, hmin, hmax) — full range when cleared (0, inf, 0, inf).
    """
    selectionChanged = pyqtSignal(float, float, float, float)
    MARGIN_L, MARGIN_B, MARGIN_T, MARGIN_R = 58, 40, 12, 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 300)
        self.w = self.h = np.zeros(0)
        self.unit = "px"
        self.dom = (0.0, 1.0, 0.0, 1.0)            # log10 domain: x0, x1, y0, y1
        self._img: QImage | None = None
        self._sel: tuple | None = None             # data-space log10 (x0, x1, y0, y1)
        self._drag = None
        self.count_text = ""

    def set_data(self, w: np.ndarray, h: np.ndarray, unit: str):
        ok = (w > 0) & (h > 0)
        self.w, self.h, self.unit = np.log10(w[ok]), np.log10(h[ok]), unit
        if len(self.w):
            x0, x1 = np.percentile(self.w, [0.05, 99.95])
            y0, y1 = np.percentile(self.h, [0.05, 99.95])
            px, py = (x1 - x0) * 0.05 or 0.5, (y1 - y0) * 0.05 or 0.5
            self.dom = (x0 - px, x1 + px, y0 - py, y1 + py)
        self._img = None
        self.update()

    def set_selection(self, wmin, wmax, hmin, hmax):
        if wmin <= 0 and hmin <= 0 and math.isinf(wmax) and math.isinf(hmax):
            self._sel = None
        else:
            lg = lambda v, d: math.log10(v) if 0 < v < float("inf") else d
            x0, x1, y0, y1 = self.dom
            self._sel = (lg(wmin, x0), lg(wmax, x1), lg(hmin, y0), lg(hmax, y1))
        self.update()

    # geometry ------------------------------------------------------------------
    def _plot(self) -> QRect:
        return QRect(self.MARGIN_L, self.MARGIN_T, self.width() - self.MARGIN_L - self.MARGIN_R,
                     self.height() - self.MARGIN_T - self.MARGIN_B)

    def _to_px(self, x, y):
        r = self._plot()
        x0, x1, y0, y1 = self.dom
        return (r.x() + (x - x0) / (x1 - x0) * r.width(), r.bottom() - (y - y0) / (y1 - y0) * r.height())

    def _to_data(self, px, py):
        r = self._plot()
        x0, x1, y0, y1 = self.dom
        return (x0 + (px - r.x()) / r.width() * (x1 - x0), y0 + (r.bottom() - py) / r.height() * (y1 - y0))

    def _raster(self):
        """2-D histogram -> image (computed once per data / size change)."""
        r = self._plot()
        bw, bh = max(1, r.width() // 2), max(1, r.height() // 2)
        x0, x1, y0, y1 = self.dom
        H, _, _ = np.histogram2d(self.h, self.w, bins=(bh, bw), range=((y0, y1), (x0, x1)))
        H = np.log1p(H[::-1])
        a = (H / H.max() * 235).astype(np.uint8) if H.max() else np.zeros_like(H, np.uint8)
        a[(H > 0) & (a < 40)] = 40
        c = theme_color("info")
        rgba = np.zeros((bh, bw, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = c.red(), c.green(), c.blue(), a
        rgba = np.ascontiguousarray(rgba)
        self._img = QImage(rgba.data, bw, bh, 4 * bw, QImage.Format.Format_RGBA8888).copy()
        self._img_size = r.size()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        muted, line = theme_color("muted"), theme_color("line")
        r = self._plot()
        p.fillRect(r, pal.color(QPalette.ColorRole.Base))
        p.setPen(QPen(line, 1))
        p.drawRect(r)
        if not len(self.w):
            p.setPen(muted)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Run a scan to see every shape's width × height")
            return
        if self._img is None or getattr(self, "_img_size", None) != r.size():
            self._raster()
        p.drawImage(r, self._img)
        # grid + ticks (decades)
        p.setPen(muted)
        x0, x1, y0, y1 = self.dom
        for t in _nice_ticks(x0, x1, True):
            if x0 <= t <= x1:
                px, _ = self._to_px(t, y0)
                p.setPen(QPen(line, 1, Qt.PenStyle.DotLine))
                p.drawLine(QPointF(px, r.top()), QPointF(px, r.bottom()))
                p.setPen(muted)
                p.drawText(QRectF(px - 40, r.bottom() + 4, 80, 16), Qt.AlignmentFlag.AlignHCenter, self._fmt(t))
        for t in _nice_ticks(y0, y1, True):
            if y0 <= t <= y1:
                _, py = self._to_px(x0, t)
                p.setPen(QPen(line, 1, Qt.PenStyle.DotLine))
                p.drawLine(QPointF(r.left(), py), QPointF(r.right(), py))
                p.setPen(muted)
                p.drawText(QRectF(0, py - 8, self.MARGIN_L - 6, 16), Qt.AlignmentFlag.AlignRight, self._fmt(t))
        p.drawText(QRectF(r.x(), self.height() - 18, r.width(), 16), Qt.AlignmentFlag.AlignHCenter, f"width ({self.unit})")
        p.save()
        p.translate(12, r.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-60, -8, 120, 16), Qt.AlignmentFlag.AlignHCenter, f"height ({self.unit})")
        p.restore()
        # selection
        sel = self._drag or self._sel
        if sel:
            ax, ay = self._to_px(sel[0], sel[3])
            bx, by = self._to_px(sel[1], sel[2])
            rect = QRectF(QPointF(min(ax, bx), min(ay, by)), QPointF(max(ax, bx), max(ay, by))).intersected(QRectF(r))
            c = QColor(theme_color("accent"))
            fill = QColor(c)
            fill.setAlpha(40)
            p.setPen(QPen(c, 2))
            p.setBrush(fill)
            p.drawRect(rect)
            if self.count_text:
                p.setPen(c)
                p.drawText(rect.adjusted(6, 4, -6, -4), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                           self.count_text)
        p.end()

    def _fmt(self, t):
        v = 10 ** t
        return f"{v:.3g}" + ("%" if self.unit.startswith("%") else "")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._plot().contains(e.position().toPoint()):
            x, y = self._to_data(e.position().x(), e.position().y())
            self._drag_origin = (x, y, e.position())
            self._drag = None

    def mouseMoveEvent(self, e):
        o = getattr(self, "_drag_origin", None)
        if o is None:
            return
        x, y = self._to_data(e.position().x(), e.position().y())
        self._drag = (min(o[0], x), max(o[0], x), min(o[1], y), max(o[1], y))
        self.update()

    def mouseReleaseEvent(self, e):
        o = getattr(self, "_drag_origin", None)
        self._drag_origin = None
        if o is None:
            return
        moved = (e.position() - o[2]).manhattanLength() > 4
        if moved and self._drag:
            self._sel, self._drag = self._drag, None
            s = self._sel
            self.selectionChanged.emit(10 ** s[0], 10 ** s[1], 10 ** s[2], 10 ** s[3])
        else:                                          # click: clear selection
            self._sel = self._drag = None
            self.selectionChanged.emit(0.0, float("inf"), 0.0, float("inf"))
        self.update()


# =============================================================================
class CoverageMap(QWidget):
    """
    Every file as a small cell coloured by review state (unreviewed / OK / flagged), sectioned by split.
    Rendered into one QImage (fast for 100k+ files); review changes repaint single cells.
    """
    activated = pyqtSignal(str)
    STATES = {None: 0, "ok": 1, "flag": 2, "fixed": 3}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.sections: list[tuple[str, list[str]]] = []
        self.status_of = lambda k: None
        self.name_of = lambda k: k
        self.current: str | None = None
        self._layout = []            # [(title, y_title, y0, keys, cols)]
        self._pos: dict[str, tuple[int, int]] = {}
        self._img: QImage | None = None
        self.cell = 8

    def set_data(self, sections, status_of, name_of):
        self.sections, self.status_of, self.name_of = sections, status_of, name_of
        self._relayout()

    def _relayout(self):
        w = max(200, self.width() - 16)
        # biggest cell (2..14 px) that keeps the map within ~2.5 screens of height
        for cell in (14, 12, 10, 8, 6, 5, 4, 3, 2):
            cols = max(1, w // cell)
            rows = sum(math.ceil(len(k) / cols) for _t, k in self.sections)
            if rows * cell + 26 * len(self.sections) <= max(600, self.parent().height() * 2.5 if self.parent() else 1500):
                break
        self.cell, cols = cell, max(1, w // cell)
        y, self._layout, self._pos = 4, [], {}
        for title, keys in self.sections:
            self._layout.append((title, y, y + 20, keys, cols))
            for i, k in enumerate(keys):
                self._pos[k] = (8 + (i % cols) * cell, y + 20 + (i // cols) * cell)
            y += 20 + math.ceil(len(keys) / cols) * cell + 10
        self.setMinimumHeight(y + 4)
        self._img = None
        self.update()

    def _colors(self):
        return [theme_color("placeholder"), theme_color("ok"), theme_color("warn"), theme_color("info")]

    def _render(self):
        img = QImage(max(1, self.width()), max(1, self.minimumHeight()), QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        cols = self._colors()
        c = self.cell
        gap = 1 if c >= 4 else 0
        for title, _yt, y0, keys, ncol in self._layout:
            for i, k in enumerate(keys):
                x, y = self._pos[k]
                p.fillRect(x, y, c - gap, c - gap, cols[self.STATES.get(self.status_of(k), 0)])
        p.end()
        self._img = img

    def update_keys(self, keys):
        """Review marks changed: recolour just these cells."""
        if self._img is None:
            return self.update()
        p = QPainter(self._img)
        cols = self._colors()
        c = self.cell
        gap = 1 if c >= 4 else 0
        for k in keys:
            if k in self._pos:
                x, y = self._pos[k]
                p.fillRect(x, y, c - gap, c - gap, cols[self.STATES.get(self.status_of(k), 0)])
        p.end()
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if e.oldSize().width() != e.size().width():
            self._relayout()

    def paintEvent(self, _):
        p = QPainter(self)
        if self._img is None:
            self._render()
        f = p.font()
        f.setBold(True)
        for title, yt, _y0, keys, _c in self._layout:
            p.setFont(f)
            p.setPen(self.palette().color(QPalette.ColorRole.Text))
            p.drawText(QRect(8, yt, self.width() - 16, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       title)
        p.drawImage(0, 0, self._img)
        if self.current in self._pos:
            x, y = self._pos[self.current]
            p.setPen(QPen(self.palette().color(QPalette.ColorRole.Text), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(x - 2, y - 2, self.cell + 2, self.cell + 2)
        p.end()

    def _key_at(self, pos: QPoint):
        c = self.cell
        for title, _yt, y0, keys, cols in self._layout:
            if y0 <= pos.y() < y0 + math.ceil(len(keys) / cols) * c:
                col, row = (pos.x() - 8) // c, (pos.y() - y0) // c
                i = row * cols + col
                if 0 <= col < cols and 0 <= i < len(keys):
                    return keys[i]
        return None

    def mouseMoveEvent(self, e):
        k = self._key_at(e.position().toPoint())
        self.setCursor(Qt.CursorShape.PointingHandCursor if k else Qt.CursorShape.ArrowCursor)
        if k:
            st = {None: "unreviewed", "ok": "✓ OK", "flag": "⚑ flagged", "fixed": "🔧 fixed"}.get(self.status_of(k), "")
            QToolTip.showText(e.globalPosition().toPoint(), f"{self.name_of(k)}\n{st}\nclick to open", self)

    def mouseReleaseEvent(self, e):
        k = self._key_at(e.position().toPoint())
        if k and e.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(k)

    def set_current(self, key):
        self.current = key
        self.update()
