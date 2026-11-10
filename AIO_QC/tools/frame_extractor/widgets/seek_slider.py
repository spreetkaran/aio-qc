# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Media-player seekbar.

Performance: the platform style is never asked to draw it (a styled QSlider repaints the whole widget on every
value change). Instead:
  * the static layer (track, in/out band, bookmark ticks) is rendered once into a cached pixmap and reused
    until something on it changes (resize, theme, marks);
  * the playhead is a float frame position with sub-pixel x, so it glides at the display rate instead of
    jumping between whole frames / pixels;
  * a playhead move repaints only the thin strip between the old and new position (+ the handle), a few
    hundred pixels instead of the whole bar.
During playback the owner calls set_playhead() for every presented frame; the QSlider value (what dragging
and clicking use) is synchronised only when it matters, so value changes don't trigger full repaints.
"""
from PyQt6.QtCore import Qt, QRectF, QPoint, QPointF, QRect, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPolygonF, QPixmap, QPen
from PyQt6.QtWidgets import QSlider

IN_OUT_COLOR = QColor(122, 79, 214)
BOOKMARK_COLOR = QColor(217, 138, 0)
TRACK_H = 4          # px
HANDLE_R = 6.5       # px
PAD = 9              # px left / right: room for the handle at both ends


class SeekSlider(QSlider):
    hovered = pyqtSignal(int, QPoint)   # value under cursor, global point at the slider's top
    hover_left = pyqtSignal()

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(30)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.time_text = lambda v: str(v)   # set by the owner (frame -> "HH:MM:SS.mmm")
        self._in = self._out = None
        self._marks = []
        self._ph = 0.0                       # playhead (float frame)
        self._ph_x = None                    # last painted playhead x
        self._cache = None
        self._cache_key = None
        self._dragging = False
        self.valueChanged.connect(self._on_value)

    # ------------------------------------------------------------------ data
    def set_in_out(self, in_frame, out_frame):
        self._in, self._out = in_frame, out_frame
        self._cache = None
        self.update()

    def set_bookmarks(self, frames):
        self._marks = sorted(frames)
        self._cache = None
        self.update()

    def set_playhead(self, frame):
        """Cheap position update for playback: float frame, partial repaint, no valueChanged."""
        if self._dragging:
            return
        frame = float(frame)
        if frame == self._ph:
            return
        old_x = self._ph_x if self._ph_x is not None else self._x_f(self._ph)
        self._ph = frame
        new_x = self._x_f(frame)
        if abs(new_x - old_x) < 0.2:           # sub-pixel change too small to see
            return
        a, b = sorted((old_x, new_x))
        r = int(HANDLE_R) + 3
        self.update(QRect(int(a) - r, 0, int(b - a) + 2 * r + 2, self.height()))

    def sync_value(self, frame):
        """Bring the QSlider value in line with the playhead without a full repaint."""
        if self._dragging or self.value() == int(frame):
            return
        self._quiet = True
        self.blockSignals(True)
        try:
            self.setValue(int(frame))
        finally:
            self.blockSignals(False)
            self._quiet = False

    def sliderChange(self, change):
        # QAbstractSlider repaints the whole widget on every change; a quiet sync is drawn by set_playhead
        if getattr(self, "_quiet", False):
            return
        super().sliderChange(change)

    def _on_value(self, v):
        self._ph = float(v)

    # ------------------------------------------------------------------ geometry (used by the timeline strip too)
    def _span(self):
        return max(1.0, self.width() - 2 * PAD)

    def _x_f(self, value):
        lo, hi = self.minimum(), self.maximum()
        if hi <= lo:
            return float(PAD)
        return PAD + (min(max(value, lo), hi) - lo) / (hi - lo) * self._span()

    def x_for_value(self, value):
        return self._x_f(value)

    def value_at(self, x):
        return self._value_at(x)

    def _value_at(self, x):
        lo, hi = self.minimum(), self.maximum()
        if hi <= lo:
            return lo
        f = (x - PAD) / self._span()
        return int(round(lo + min(1.0, max(0.0, f)) * (hi - lo)))

    def _handle_rect(self):
        x, cy = self._x_f(self._ph), self.height() / 2
        return QRectF(x - HANDLE_R - 2, cy - HANDLE_R - 2, 2 * HANDLE_R + 4, 2 * HANDLE_R + 4)

    # ------------------------------------------------------------------ mouse (no style involvement)
    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton or self.maximum() <= self.minimum():
            return super().mousePressEvent(ev)
        self._dragging = True
        self.setSliderDown(True)              # emits sliderPressed
        self.setValue(self._value_at(ev.position().x()))   # emits valueChanged + sliderMoved (slider is down)
        self.update()
        ev.accept()

    def mouseMoveEvent(self, ev):
        x = ev.position().x()
        if self._dragging:
            v = self._value_at(x)
            if v != self.value():
                self.setValue(v)                   # emits sliderMoved while the slider is down
                self.update()
        if self.maximum() > self.minimum():
            self.hovered.emit(self._value_at(x), self.mapToGlobal(QPoint(int(x), 0)))
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if self._dragging and ev.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setSliderDown(False)          # emits sliderReleased
            self.update()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def enterEvent(self, ev):
        self.update(self._handle_rect().toAlignedRect().adjusted(-2, -2, 2, 2))
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self.hover_left.emit()
        self.update(self._handle_rect().toAlignedRect().adjusted(-2, -2, 2, 2))
        super().leaveEvent(ev)

    # ------------------------------------------------------------------ painting
    def resizeEvent(self, ev):
        self._cache = None
        super().resizeEvent(ev)

    def changeEvent(self, ev):
        from PyQt6.QtCore import QEvent
        if ev.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange, QEvent.Type.EnabledChange):
            self._cache = None
        super().changeEvent(ev)

    def _render_static(self):
        dpr = self.devicePixelRatioF()
        pm = QPixmap(max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr)))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        h, cy = self.height(), self.height() / 2
        track = pal.mid().color()
        track.setAlpha(150)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(PAD, cy - TRACK_H / 2, self._span(), TRACK_H), TRACK_H / 2, TRACK_H / 2)
        if self.maximum() > self.minimum():
            if self._in is not None or self._out is not None:
                a = self._x_f(self._in if self._in is not None else self.minimum())
                b = self._x_f(self._out if self._out is not None else self.maximum())
                p.fillRect(QRectF(a, h - 5, max(2.0, b - a), 4), IN_OUT_COLOR)
                p.setPen(QPen(IN_OUT_COLOR, 1))
                for x in ((a,) if self._in is not None else ()) + ((b,) if self._out is not None else ()):
                    p.drawLine(QPointF(x, h - 12), QPointF(x, h - 1))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(BOOKMARK_COLOR)
            for m in self._marks:
                x = self._x_f(m)
                p.drawPolygon(QPolygonF([QPointF(x - 4, 1), QPointF(x + 4, 1), QPointF(x, 7)]))
        p.end()
        return pm

    def paintEvent(self, ev):
        key = (self.width(), self.height(), self.minimum(), self.maximum(), self.isEnabled())
        if self._cache is None or self._cache_key != key:
            self._cache, self._cache_key = self._render_static(), key
        p = QPainter(self)
        clip = ev.rect()
        p.drawPixmap(clip, self._cache, QRect(int(clip.x() * self._cache.devicePixelRatio()),
                                              int(clip.y() * self._cache.devicePixelRatio()),
                                              int(clip.width() * self._cache.devicePixelRatio()),
                                              int(clip.height() * self._cache.devicePixelRatio())))
        if self.maximum() <= self.minimum():
            p.end()
            self._ph_x = None
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        cy = self.height() / 2
        x = self._x_f(self._ph)
        hl = pal.highlight().color() if self.isEnabled() else pal.mid().color()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(hl)
        p.drawRoundedRect(QRectF(PAD, cy - TRACK_H / 2, max(0.0, x - PAD), TRACK_H), TRACK_H / 2, TRACK_H / 2)
        r = HANDLE_R + (1.0 if (self._dragging or self.underMouse()) else 0.0)
        p.setBrush(pal.button().color())
        p.setPen(QPen(hl, 2))
        p.drawEllipse(QPointF(x, cy), r, r)
        p.end()
        self._ph_x = x
