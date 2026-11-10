# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Toolbar zoom control: a "Zoom" caption above the level to one decimal (e.g. "137.5 %") in a rounded field.

  · click the number to type a level (Enter applies, Esc cancels, ↑ / ↓ step ±10 %)
  · scroll over it to zoom in / out around the image centre
  · ▾ opens presets 10 … 800 % (Fit window / Actual size are toolbar buttons already)
"""
from __future__ import annotations

import re

from PyQt6.QtCore import QObject, QRegularExpression, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QRegularExpressionValidator
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QMenu, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from AIO_QC.utils import style as apply_style

PRESETS = [10, 25, 50, 75, 100, 150, 200, 400, 800]

# Windows Photos zoom steps (percent): 10-point steps up to 200 %, 25-point steps up to 600 %, 50-point steps above;
# below 10 %, 1-point steps. A click goes to the next level of this grid (from 137 %: 140 % in, 130 % out).
_BANDS = ((0.0, 10.0, 1.0), (10.0, 200.0, 10.0), (200.0, 600.0, 25.0), (600.0, 1_000_000.0, 50.0))


def photos_step(zoom: float, direction: int, lo: float = 0.01, hi: float = 10.0) -> float:
    """The next zoom level in (direction > 0) or out (< 0), as in Windows Photos (1.0 = 100 %)."""
    import math
    z = zoom * 100.0
    eps = 1e-6
    if direction > 0:
        for a, b, step in _BANDS:
            if z < b - eps:
                nxt = (math.floor((z + eps) / step) + 1) * step
                z = max(nxt, a + step if z < a else nxt)
                break
    else:
        for a, b, step in reversed(_BANDS):
            if z > a + eps:
                z = (math.ceil((z - eps) / step) - 1) * step
                break
    return max(lo, min(hi, z / 100.0))


class _Value(QLineEdit):
    stepped = pyqtSignal(int)            # +1 / -1 (arrow keys, wheel)
    cancelled = pyqtSignal()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Up:
            self.stepped.emit(1)
        elif e.key() == Qt.Key.Key_Down:
            self.stepped.emit(-1)
        elif e.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
        else:
            super().keyPressEvent(e)

    def wheelEvent(self, e):
        d = e.angleDelta().y()
        if d:
            self.stepped.emit(1 if d > 0 else -1)
        e.accept()

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        self.selectAll()                 # click -> type a new value straight away


class ZoomWidget(QWidget):
    zoomRequested = pyqtSignal(float)    # 1.0 = 100 %
    STEP = 1.1                           # ±10 % per key / wheel step

    def __init__(self, min_zoom: float, max_zoom: float, parent=None):
        super().__init__(parent)
        self.min_zoom, self.max_zoom = min_zoom, max_zoom
        self._zoom: float | None = None

        self.caption = QLabel("Zoom")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        apply_style(self.caption, "color:@muted;")

        self.value = _Value()
        self.value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value.setFrame(False)
        self.value.setTextMargins(0, 0, 0, 0)
        # normal weight, same font as the rest of the UI; just wide enough for "1000.0 %"
        self.value.setFixedWidth(self.value.fontMetrics().horizontalAdvance("1000.0%") + 6)
        self.value.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\s*\d{0,4}([.,]\d{0,1})?\s*%?\s*$")))
        self.value.setToolTip("Type a zoom level and press Enter · scroll or ↑ / ↓ to step · ▾ for presets")
        self.value.returnPressed.connect(self._apply_typed)
        self.value.editingFinished.connect(self._show)          # focus lost without Enter: restore
        self.value.stepped.connect(self._step)
        self.value.cancelled.connect(self._cancel)
        apply_style(self.value, "QLineEdit{background:transparent; color:@text; padding:0; border:none;}")

        self.more = QToolButton()
        self.more.setText("▾")
        self.more.setAutoRaise(True)
        self.more.setToolTip("Zoom presets")
        self.more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more.setFixedSize(QSize(14, 20))
        self.menu = QMenu(self)
        self.menu.aboutToShow.connect(self._fill_menu)
        self.more.setMenu(self.menu)

        self.field = QFrame()
        self.field.setObjectName("zoomField")
        row = QHBoxLayout(self.field)
        row.setContentsMargins(4, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.value)
        row.addWidget(self.more)
        apply_style(self.field, "QFrame#zoomField{border:1px solid @line; border-radius:7px; background:@surface;}"
                                "QFrame#zoomField:hover{border:1px solid @muted;}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(8)                      # clear gap between the "Zoom" caption and the field
        lay.addWidget(self.caption)
        lay.addWidget(self.field)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)   # toolbar never stretches it
        self.set_zoom(None)

    # ------------------------------------------------------------------ public
    def set_compact(self, on: bool):
        """Compact toolbar: caption beside the field (one row, same height as the icon buttons) instead of above."""
        lay = self.layout()
        lay.setDirection(lay.Direction.LeftToRight if on else lay.Direction.TopToBottom)
        lay.setSpacing(6 if on else 8)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight if on
                                  else Qt.AlignmentFlag.AlignHCenter)
        self.updateGeometry()
        self.adjustSize()

    # ------------------------------------------------------------------ public
    def set_zoom(self, zoom: float | None):
        self._zoom = zoom
        self.value.setEnabled(zoom is not None)
        self.more.setEnabled(zoom is not None)
        if not self.value.hasFocus():
            self._show()

    # ------------------------------------------------------------------ internals
    @staticmethod
    def fmt(zoom: float) -> str:
        """100 -> "100%", 137.5 -> "137.5%" (no ".0" on whole values)."""
        v = round(zoom * 100, 1)
        return f"{v:.0f}%" if abs(v - round(v)) < 1e-9 else f"{v:.1f}%"

    def _show(self):
        self.value.setText("–" if self._zoom is None else self.fmt(self._zoom))

    def _clamp(self, z: float) -> float:
        return max(self.min_zoom, min(self.max_zoom, z))

    def _apply_typed(self):
        m = re.search(r"\d+(?:[.,]\d+)?", self.value.text())
        if m and self._zoom is not None:
            z = self._clamp(float(m.group(0).replace(",", ".")) / 100)
            if z > 0:
                self.zoomRequested.emit(z)
        self.value.clearFocus()
        self._show()

    def _cancel(self):
        self.value.clearFocus()
        self._show()

    def _step(self, direction: int):
        if self._zoom is None:
            return
        self.zoomRequested.emit(self._clamp(photos_step(self._zoom, direction, self.min_zoom, self.max_zoom)))

    def wheelEvent(self, e):                       # caption / frame area too
        d = e.angleDelta().y()
        if d:
            self._step(1 if d > 0 else -1)
        e.accept()

    def _fill_menu(self):
        self.menu.clear()
        cur = round((self._zoom or 0) * 100, 1)
        for p in PRESETS:
            act = self.menu.addAction(f"{p}%", lambda p=p: self.zoomRequested.emit(self._clamp(p / 100)))
            act.setCheckable(True)
            act.setChecked(abs(cur - p) < 0.05)


class ZoomSlider(QWidget):
    """Paint-style zoom slider: a thin track, a round handle, logarithmic (every step is the same ratio, so 10 %
    → 20 % takes as much travel as 400 % → 800 %). Dragging zooms live around the image centre; it follows the
    canvas zoom however it changes."""
    zoomRequested = pyqtSignal(float)
    STEPS = 1000

    def __init__(self, min_zoom: float, max_zoom: float, parent=None):
        super().__init__(parent)
        import math
        from PyQt6.QtWidgets import QSlider
        self._log = math.log
        self.min_zoom, self.max_zoom = min_zoom, max_zoom
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.slider.setFixedWidth(150)
        self.slider.setPageStep(50)
        self.slider.setToolTip("Zoom (drag; the zoom box shows the level)")
        apply_style(self.slider,
                    "QSlider::groove:horizontal{height:4px; border-radius:2px; background:@line;}"
                    "QSlider::sub-page:horizontal{height:4px; border-radius:2px; background:@accent;}"
                    "QSlider::handle:horizontal{width:14px; height:14px; margin:-6px 0; border-radius:8px;"
                    " background:@accent; border:2px solid palette(base);}"
                    "QSlider::handle:horizontal:hover{width:16px; height:16px; margin:-7px 0; border-radius:9px;}"
                    "QSlider::handle:horizontal:disabled{background:@line;}")
        self.slider.valueChanged.connect(self._moved)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.addWidget(self.slider)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._quiet = False
        self.set_zoom(None)

    def _to_pos(self, z: float) -> int:
        a, b = self._log(self.min_zoom), self._log(self.max_zoom)
        return int(round((self._log(max(self.min_zoom, min(z, self.max_zoom))) - a) / (b - a) * self.STEPS))

    def _to_zoom(self, v: int) -> float:
        import math
        a, b = math.log(self.min_zoom), math.log(self.max_zoom)
        return math.exp(a + (b - a) * v / self.STEPS)

    def set_zoom(self, zoom: float | None):
        self.slider.setEnabled(zoom is not None)
        if zoom is None or self.slider.isSliderDown():
            return
        self._quiet = True
        self.slider.setValue(self._to_pos(zoom))
        self._quiet = False

    def _moved(self, v):
        if not self._quiet:
            self.zoomRequested.emit(self._to_zoom(v))


class HoldToZoom(QObject):
    """Zoom in / out buttons, Windows Photos style.

    Press: one ×1.25 step at once (a glide; quick repeated clicks are that many steps). Keep holding: after a short
    pause (0.4 s, like a key's repeat delay) the zoom runs on continuously — easing in to a steady 2.5× per
    second, computed from the elapsed time on every display frame (never stepped) — and eases out to a stop on
    release. The point at the view centre stays put; the zoom box and slider follow every frame."""
    RATE = 0.9163                                 # ln(2.5) per second at full speed
    RAMP = 0.25                                   # s to reach full speed
    STEP = 1.25
    DELAY = 400                                   # ms held before the continuous zoom starts (like key repeat)

    def __init__(self, canvas, button, direction: int):
        super().__init__(button)
        from PyQt6.QtCore import QElapsedTimer, QTimer
        self.c, self.button, self.dir = canvas, button, direction
        self.clock = QElapsedTimer()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(8)                 # faster than any display: each frame shows the exact time-based zoom
        self.timer.timeout.connect(self._frame)
        self.hold = QTimer(self)
        self.hold.setSingleShot(True)
        self.hold.setInterval(self.DELAY)         # pause between the first step and the continuous zoom
        self.hold.timeout.connect(self._start_continuous)
        self.pressed = False
        self.z0 = self.anchor = None
        button.installEventFilter(self)

    def _travel(self, t: float) -> float:
        """log-zoom travelled after t seconds (ease-in, then constant speed)."""
        r = self.RAMP
        return self.RATE * (t * t / (2 * r) if t < r else t - r / 2)

    def eventFilter(self, obj, e):
        from PyQt6.QtCore import QEvent
        t = e.type()
        if t in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick) \
                and e.button() == Qt.MouseButton.LeftButton and obj.isEnabled():
            self._press()
            obj.setDown(True)
            return True                           # a fast second click is another step, not a double-click
        if t == QEvent.Type.MouseButtonRelease and e.button() == Qt.MouseButton.LeftButton and self.pressed:
            obj.setDown(False)
            self._release()
            return True                           # handled here: the action does not add another step
        return False

    def _press(self):
        c = self.c
        if c.pixmap_item.pixmap().isNull():
            return
        self.pressed = True
        c.zoom_step(self.dir)                     # the click's step (Windows Photos levels), right away
        self.hold.start()

    def _start_continuous(self):
        c = self.c
        if not self.pressed or c.pixmap_item.pixmap().isNull():
            return
        target = c._target_zoom if c.is_animating() else c.current_zoom
        c.stop_zoom_animation()
        if abs(c.current_zoom - target) > 1e-9:   # the step's glide had not landed: finish it without a jump
            c._apply_view(target, c.clamp_center(target, c._view_center()), animating=True)
        self.anchor = c._view_center()
        self.z0 = c.current_zoom
        c._set_smooth(c.current_zoom, True)
        c._holding = True                         # the image uses its fast filter meanwhile
        self.clock.start()
        self.timer.start()

    def _zoom_at(self, t: float) -> float:
        import math
        z = self.z0 * math.exp(self.dir * self._travel(t))
        return max(self.c.MIN_ZOOM, min(self.c.MAX_ZOOM, z))

    def _frame(self):
        c = self.c
        if self.z0 is None or c.pixmap_item.pixmap().isNull():
            self.timer.stop()
            return
        z = self._zoom_at(self.clock.elapsed() / 1000.0)
        c._apply_view(z, c.clamp_center(z, self.anchor), animating=True)

    def _release(self):
        import math
        self.pressed = False
        self.hold.stop()
        c = self.c
        if self.z0 is None:                        # released before the pause ended: it was a click (step done)
            return
        self.timer.stop()
        c._holding = False
        anchor = self.anchor
        self.z0 = None
        if c.pixmap_item.pixmap().isNull():
            return
        t = self.clock.elapsed() / 1000.0
        speed = self.RATE * min(1.0, t / self.RAMP)
        z = c.current_zoom
        ms = 220                                   # ease out (OutCubic starts at 3× its mean speed: distance
        target = z * math.exp(self.dir * speed * ms / 1000 / 3)      # = speed · T / 3 → no speed bump on release)
        c.animate_to(target, anchor_scene=anchor, anchor_view=c.viewport().rect().center(), duration=ms)
