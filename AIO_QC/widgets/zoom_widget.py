# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Toolbar zoom control: a "Zoom" caption above the level to one decimal (e.g. "137.5 %") in a rounded field.

  · click the number to type a level (Enter applies, Esc cancels, ↑ / ↓ step ±10 %)
  · scroll over it to zoom in / out around the image centre
  · ▾ opens presets 10 … 800 % (Fit window / Actual size are toolbar buttons already)
"""
from __future__ import annotations

import re

from PyQt6.QtCore import QRegularExpression, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QRegularExpressionValidator
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QMenu, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from AIO_QC.utils import style as apply_style

PRESETS = [10, 25, 50, 75, 100, 150, 200, 400, 800]


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
        self.zoomRequested.emit(self._clamp(self._zoom * (self.STEP if direction > 0 else 1 / self.STEP)))

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
