# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Explorer charts, painted with QPainter (no plotting library): the confidence histogram with a draggable threshold,
the precision / recall curve and the embedding map (scatter with rubber-band selection).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QToolTip, QWidget

from AIO_QC.utils import theme_color

TP_COLOR, FP_COLOR = "#22c55e", "#ef4444"
M_L, M_R, M_T, M_B = 34, 10, 10, 34


class _Chart(QWidget):
    def __init__(self, parent=None, h=150):
        super().__init__(parent)
        self.setMinimumHeight(h)
        self.setMouseTracking(True)

    def plot_rect(self) -> QRectF:
        return QRectF(M_L, M_T, max(10, self.width() - M_L - M_R), max(10, self.height() - M_T - M_B))

    def _frame(self, p, xlabel, ylabel=""):
        r = self.plot_rect()
        p.setPen(QPen(theme_color("line"), 1))
        p.drawRect(r)
        p.setPen(theme_color("muted"))
        fm = QFontMetrics(p.font())
        for t in (0.0, 0.5, 1.0):
            x = r.left() + t * r.width()
            p.drawText(QRectF(x - 20, r.bottom() + 2, 40, fm.height()), Qt.AlignmentFlag.AlignCenter, f"{t:g}")
        p.drawText(QRectF(r.left(), r.bottom() + 2 + fm.height(), r.width(), fm.height()), Qt.AlignmentFlag.AlignCenter,
                   xlabel)
        if ylabel:
            p.save()
            p.translate(10, r.center().y())
            p.rotate(-90)
            p.drawText(QRectF(-r.height() / 2, -8, r.height(), 16), Qt.AlignmentFlag.AlignCenter, ylabel)
            p.restore()


class ConfHistogram(_Chart):
    """Stacked TP (green) / FP (red) counts over confidence; the threshold line is draggable."""
    thresholdChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent, 140)
        self.tp = self.fp = np.zeros(0)
        self.edges = np.linspace(0, 1, 51)
        self.thr = 0.25
        self.best = float("nan")
        self._drag = False
        self.setToolTip("Labels the model matched (green) and its other detections (red), by confidence.\n"
                        "Drag the line to change this model's threshold.")

    def set_data(self, c: dict, thr: float):
        self.tp, self.fp, self.edges = c["tp"], c["fp"], c["edges"]
        self.best, self.thr = c.get("best_conf", float("nan")), float(thr)
        self.update()

    def set_threshold(self, thr: float):
        self.thr = float(thr)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.plot_rect()
        self._frame(p, "confidence")
        tot = (self.tp + self.fp) if len(self.tp) else np.zeros(0)
        top = float(np.sqrt(tot.max())) if len(tot) and tot.max() > 0 else 1.0       # sqrt scale: both tails show
        n = len(tot)
        for i in range(n):
            x0 = r.left() + self.edges[i] * r.width()
            w = max(1.0, (self.edges[i + 1] - self.edges[i]) * r.width() - 1)
            hf = np.sqrt(self.fp[i]) / top * r.height() if self.fp[i] else 0
            ht = np.sqrt(tot[i]) / top * r.height() if tot[i] else 0
            p.fillRect(QRectF(x0, r.bottom() - ht, w, ht - hf), QColor(TP_COLOR))
            p.fillRect(QRectF(x0, r.bottom() - hf, w, hf), QColor(FP_COLOR))
        if self.best == self.best:
            x = r.left() + self.best * r.width()
            p.setPen(QPen(theme_color("muted"), 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        x = r.left() + self.thr * r.width()
        p.setPen(QPen(theme_color("accent"), 2))
        p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        p.setBrush(theme_color("accent"))
        p.drawEllipse(QPointF(x, r.top() + 4), 4, 4)
        p.setPen(theme_color("text"))
        p.drawText(QRectF(min(x + 4, r.right() - 40), r.top(), 44, 16), Qt.AlignmentFlag.AlignLeft, f"{self.thr:.2f}")

    def _value(self, x):
        r = self.plot_rect()
        return float(np.clip((x - r.left()) / r.width(), 0.0, 1.0))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = True
            self._move(e.position().x())

    def mouseMoveEvent(self, e):
        if self._drag:
            self._move(e.position().x())
        else:
            v = self._value(e.position().x())
            i = min(len(self.tp) - 1, int(v * len(self.tp))) if len(self.tp) else -1
            if i >= 0:
                QToolTip.showText(e.globalPosition().toPoint(), f"conf {self.edges[i]:.2f}–{self.edges[i + 1]:.2f}: "
                                  f"{int(self.tp[i]):,} matched · {int(self.fp[i]):,} other", self)

    def mouseReleaseEvent(self, e):
        if self._drag:
            self._drag = False
            self.thresholdChanged.emit(round(self.thr, 2))

    def _move(self, x):
        self.thr = round(self._value(x), 2)
        self.update()


class PRCurve(_Chart):
    def __init__(self, parent=None):
        super().__init__(parent, 150)
        self.prec = self.rec = self.conf = np.zeros(0)
        self.at = None
        self.best = None
        self.setToolTip("Precision / recall of this model as its threshold goes from high to low.\n"
                        "● current threshold · ○ best F1")

    def set_data(self, c: dict, thr: float):
        self.prec, self.rec, self.conf = c["precision"], c["recall"], c["conf"]
        self.at = None
        self.best = None
        if len(self.conf):
            n = int(np.searchsorted(-self.conf, -thr, side="right"))
            if n:
                self.at = (float(self.rec[n - 1]), float(self.prec[n - 1]))
            if c.get("best_conf") == c.get("best_conf"):
                k = int(np.searchsorted(-self.conf, -c["best_conf"], side="right"))
                if k:
                    self.best = (float(self.rec[k - 1]), float(self.prec[k - 1]))
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.plot_rect()
        self._frame(p, "recall", "precision")
        if len(self.prec) < 2:
            p.setPen(theme_color("muted"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "no predictions")
            return
        step = max(1, len(self.prec) // 2000)
        path = QPainterPath()
        pt = lambda rc, pr: QPointF(r.left() + rc * r.width(), r.bottom() - pr * r.height())
        path.moveTo(pt(self.rec[0], self.prec[0]))
        for i in range(step, len(self.prec), step):
            path.lineTo(pt(self.rec[i], self.prec[i]))
        path.lineTo(pt(self.rec[-1], self.prec[-1]))
        p.setPen(QPen(theme_color("accent"), 2))
        p.drawPath(path)
        if self.best is not None:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(theme_color("text"), 1.5))
            p.drawEllipse(pt(*self.best), 5, 5)
        if self.at is not None:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme_color("accent"))
            p.drawEllipse(pt(*self.at), 4.5, 4.5)


class ScatterMap(QWidget):
    """Points (0..1 coordinates) coloured per point; drag a rectangle to select; double-click opens a point."""
    selected = pyqtSignal(object)        # np.ndarray of point indices
    activated = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 240)
        self.setMouseTracking(True)
        self.xy = np.zeros((0, 2), np.float32)
        self.colors = []
        self.sel = np.zeros(0, bool)
        self.tips = None
        self._band = None
        self._img = None

    def set_points(self, xy, colors, tips=None):
        self.xy = np.asarray(xy, np.float32)
        self.colors = list(colors)
        self.sel = np.zeros(len(self.xy), bool)
        self.tips = tips
        self._img = None
        self.update()

    def recolor(self, colors):
        self.colors = list(colors)
        self._img = None
        self.update()

    def _rect(self) -> QRectF:
        return QRectF(12, 12, max(10, self.width() - 24), max(10, self.height() - 24))

    def _screen(self):
        r = self._rect()
        return np.stack([r.left() + self.xy[:, 0] * r.width(), r.top() + (1 - self.xy[:, 1]) * r.height()], 1) \
            if len(self.xy) else np.zeros((0, 2))

    def resizeEvent(self, e):
        self._img = None
        super().resizeEvent(e)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), theme_color("surface"))
        if not len(self.xy):
            p.setPen(theme_color("muted"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       "No map yet — choose the embeddings and press Compute.")
            return
        dpr = self.devicePixelRatioF()
        if self._img is None:                                  # points cached as an image: hover / band stay fast
            img = QImage(int(self.width() * dpr), int(self.height() * dpr), QImage.Format.Format_ARGB32_Premultiplied)
            img.setDevicePixelRatio(dpr)
            img.fill(Qt.GlobalColor.transparent)
            q = QPainter(img)
            q.setRenderHint(QPainter.RenderHint.Antialiasing)
            q.setPen(Qt.PenStyle.NoPen)
            s = self._screen()
            rad = 2.2 if len(s) > 20_000 else 3.0
            by_color: dict = {}
            for n, c in enumerate(self.colors):
                by_color.setdefault(c, []).append(n)
            for c, idx in by_color.items():
                col = QColor(c)
                col.setAlpha(190)
                q.setBrush(col)
                for n in idx:
                    q.drawEllipse(QPointF(*s[n]), rad, rad)
            q.end()
            self._img = img
        p.drawImage(0, 0, self._img)
        if self.sel.any():
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(QPen(theme_color("text"), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            s = self._screen()
            for n in np.nonzero(self.sel)[0][:5000]:
                p.drawEllipse(QPointF(*s[n]), 4.5, 4.5)
        if self._band is not None:
            p.setPen(QPen(theme_color("accent"), 1, Qt.PenStyle.DashLine))
            c = QColor(theme_color("accent"))
            c.setAlpha(40)
            p.setBrush(c)
            p.drawRect(QRectF(self._band[0], self._band[1]).normalized())

    def _nearest(self, pos, tol=8.0):
        if not len(self.xy):
            return None
        s = self._screen()
        dd = np.hypot(s[:, 0] - pos.x(), s[:, 1] - pos.y())
        n = int(np.argmin(dd))
        return n if dd[n] <= tol else None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._band = [e.position(), e.position()]

    def mouseMoveEvent(self, e):
        if self._band is not None:
            self._band[1] = e.position()
            self.update()
            return
        n = self._nearest(e.position())
        if n is not None and self.tips is not None:
            QToolTip.showText(e.globalPosition().toPoint(), self.tips(n), self)

    def mouseReleaseEvent(self, e):
        if self._band is None:
            return
        a, b = self._band
        self._band = None
        r = QRectF(a, b).normalized()
        if r.width() < 4 and r.height() < 4:
            n = self._nearest(e.position())
            self.sel = np.zeros(len(self.xy), bool)
            if n is not None:
                self.sel[n] = True
        else:
            s = self._screen()
            self.sel = (s[:, 0] >= r.left()) & (s[:, 0] <= r.right()) & (s[:, 1] >= r.top()) & (s[:, 1] <= r.bottom())
        self.update()
        self.selected.emit(np.nonzero(self.sel)[0])

    def mouseDoubleClickEvent(self, e):
        n = self._nearest(e.position())
        if n is not None:
            self.activated.emit(int(n))
