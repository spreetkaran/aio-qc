# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
RegionCanvas — draw / move / resize one rectangle on an image frame of a fixed pixel size (Dim / Area ▸ Region).

Shows a sample image of that size, a heat-map of where the annotations lie, and the matching boxes. Coordinates are
always image pixels. Drag on empty space draws, drag inside moves, drag an edge / corner resizes, right-click or Esc
clears. regionChanging(rect | None) fires while dragging (cheap counts), regionChanged(rect | None) on release.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from AIO_QC.core.region import clamp_rect
from AIO_QC.utils import theme_color

EDGE = 7            # px grab tolerance for edges / corners


class RegionCanvas(QWidget):
    regionChanging = pyqtSignal(object)
    regionChanged = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 180)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.W = self.H = 0
        self.bg: QImage | None = None
        self.heat_img: QImage | None = None
        self.boxes: list = []                  # [(x1, y1, x2, y2, QColor)] in image pixels
        self.region = None                     # (x1, y1, x2, y2) image pixels
        self.count_text = ""
        self.empty_text = "Scan first — then pick an image size"
        self._drag = None                      # (mode, start image point, start region)
        self.show_heat = True
        self.show_boxes = True

    # ------------------------------------------------------------------ data
    def set_frame(self, W: int, H: int):
        self.W, self.H = int(W), int(H)
        self.update()

    def set_background(self, img: QImage | None):
        self.bg = img if img is not None and not img.isNull() else None
        self.update()

    def set_heat(self, grid: np.ndarray | None):
        """Coverage grid -> translucent accent-coloured image (log-scaled so sparse places still show)."""
        if grid is None or not grid.size or grid.max() <= 0:
            self.heat_img = None
        else:
            g = np.log1p(grid)
            a = (g / g.max() * 170).astype(np.uint8)
            a[(grid > 0) & (a < 28)] = 28
            c = theme_color("accent")
            rgba = np.zeros(grid.shape + (4,), np.uint8)
            rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = c.red(), c.green(), c.blue(), a
            rgba = np.ascontiguousarray(rgba)
            h, w = grid.shape
            self.heat_img = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        self.update()

    def set_boxes(self, boxes):
        self.boxes = boxes
        self.update()

    def set_region(self, rect):
        self.region = rect
        self.update()

    # ------------------------------------------------------------------ geometry
    def _frame(self) -> QRectF:
        """Where the W × H image is drawn (aspect kept, centred)."""
        if not (self.W and self.H):
            return QRectF()
        m = 10
        aw, ah = self.width() - 2 * m, self.height() - 2 * m - 20
        k = min(aw / self.W, ah / self.H)
        w, h = self.W * k, self.H * k
        return QRectF(m + (aw - w) / 2, m + (ah - h) / 2, w, h)

    def _k(self) -> float:
        f = self._frame()
        return f.width() / self.W if self.W else 1.0

    def to_img(self, p: QPointF) -> tuple[float, float]:
        f, k = self._frame(), self._k()
        return (min(max((p.x() - f.x()) / k, 0.0), self.W), min(max((p.y() - f.y()) / k, 0.0), self.H))

    def to_widget(self, x, y) -> QPointF:
        f, k = self._frame(), self._k()
        return QPointF(f.x() + x * k, f.y() + y * k)

    def _rect_w(self, r) -> QRectF:
        return QRectF(self.to_widget(r[0], r[1]), self.to_widget(r[2], r[3]))

    def _hit(self, pos: QPointF) -> str:
        """'' outside · 'move' inside · edge code (l, r, t, b, lt, rb, …) near an edge."""
        if self.region is None:
            return ""
        r = self._rect_w(self.region)
        x, y = pos.x(), pos.y()
        near_l, near_r = abs(x - r.left()) <= EDGE, abs(x - r.right()) <= EDGE
        near_t, near_b = abs(y - r.top()) <= EDGE, abs(y - r.bottom()) <= EDGE
        inside_x = r.left() - EDGE <= x <= r.right() + EDGE
        inside_y = r.top() - EDGE <= y <= r.bottom() + EDGE
        code = ""
        if inside_y and (near_l or near_r):
            code += "l" if near_l else "r"
        if inside_x and (near_t or near_b):
            code += "t" if near_t else "b"
        if code:
            return code
        return "move" if r.contains(pos) else ""

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        f = self._frame()
        muted = theme_color("muted")
        if f.isEmpty():
            p.setPen(muted)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
            return
        if self.bg is not None:
            p.drawImage(f, self.bg)
        else:
            p.fillRect(f, theme_color("placeholder"))
        if self.show_heat and self.heat_img is not None:
            p.drawImage(f, self.heat_img)
        if self.show_boxes and self.boxes:
            k = self._k()
            pens = {}
            for x1, y1, x2, y2, col in self.boxes:
                pen = pens.get(col.rgba())
                if pen is None:
                    pen = pens[col.rgba()] = QPen(col, 1)
                p.setPen(pen)
                p.drawRect(QRectF(f.x() + x1 * k, f.y() + y1 * k, max((x2 - x1) * k, 1), max((y2 - y1) * k, 1)))
        p.setPen(QPen(theme_color("line"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(f)
        if self.region is not None:
            r = self._rect_w(self.region)
            shade = QPainterPath()
            shade.addRect(f)
            hole = QPainterPath()
            hole.addRect(r)
            p.fillPath(shade.subtracted(hole), QBrush(QColor(0, 0, 0, 110)))     # dim what is outside
            acc = theme_color("accent")
            p.setPen(QPen(acc, 2))
            p.drawRect(r)
            p.setBrush(acc)
            for c in (r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()):
                p.drawRect(QRectF(c.x() - 3.5, c.y() - 3.5, 7, 7))
            x1, y1, x2, y2 = self.region
            label = f"{x1:.0f}, {y1:.0f} · {x2 - x1:.0f} × {y2 - y1:.0f} px" + (
                f"  —  {self.count_text}" if self.count_text else "")
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label) + 12
            tx = min(max(r.left(), f.left()), f.right() - tw)
            ty = r.top() - 22 if r.top() - 22 >= f.top() else r.bottom() + 4
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 170))
            p.drawRoundedRect(QRectF(tx, ty, tw, 18), 4, 4)
            p.setPen(QColor("white"))
            p.drawText(QRectF(tx, ty, tw, 18), Qt.AlignmentFlag.AlignCenter, label)
        p.setPen(muted)
        hint = (f"{self.W} × {self.H} px · drag to draw · drag inside to move · drag an edge to resize · "
                "right-click to clear")
        p.drawText(QRectF(0, f.bottom() + 4, self.width(), 18), Qt.AlignmentFlag.AlignHCenter,
                   p.fontMetrics().elidedText(hint, Qt.TextElideMode.ElideRight, self.width() - 8))
        p.end()

    # ------------------------------------------------------------------ mouse
    _CURSORS = {"l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
                "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
                "lt": Qt.CursorShape.SizeFDiagCursor, "rb": Qt.CursorShape.SizeFDiagCursor,
                "rt": Qt.CursorShape.SizeBDiagCursor, "lb": Qt.CursorShape.SizeBDiagCursor,
                "move": Qt.CursorShape.SizeAllCursor}

    def mousePressEvent(self, e):
        if not (self.W and self.H):
            return
        if e.button() == Qt.MouseButton.RightButton:
            self.clear()
            return
        if e.button() != Qt.MouseButton.LeftButton or not self._frame().adjusted(-EDGE, -EDGE, EDGE, EDGE).contains(
                e.position()):
            return
        mode = self._hit(e.position()) or "new"
        self._drag = (mode, self.to_img(e.position()), self.region)

    def mouseMoveEvent(self, e):
        if self._drag is None:
            self.setCursor(QCursor(self._CURSORS.get(self._hit(e.position()), Qt.CursorShape.CrossCursor)))
            return
        mode, (sx, sy), start = self._drag
        x, y = self.to_img(e.position())
        if mode == "new":
            r = (sx, sy, x, y)
        elif mode == "move":
            x1, y1, x2, y2 = start
            dx = min(max(x - sx, -x1), self.W - x2)
            dy = min(max(y - sy, -y1), self.H - y2)
            r = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        else:
            x1, y1, x2, y2 = start
            if "l" in mode:
                x1 = x
            if "r" in mode:
                x2 = x
            if "t" in mode:
                y1 = y
            if "b" in mode:
                y2 = y
            r = (x1, y1, x2, y2)
        r = clamp_rect(r, self.W, self.H)
        if r is not None:
            self.region = tuple(round(v) for v in r)
            self.regionChanging.emit(self.region)
            self.update()

    def mouseReleaseEvent(self, e):
        if self._drag is None:
            return
        self._drag = None
        self.regionChanged.emit(self.region)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.region is not None:
            self.clear()
            return
        super().keyPressEvent(e)

    def clear(self):
        self._drag = None
        if self.region is not None:
            self.region = None
            self.update()
            self.regionChanged.emit(None)
