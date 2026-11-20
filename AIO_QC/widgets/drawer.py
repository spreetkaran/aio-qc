# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
OptionsDrawer — the one "More options" control of the app. The extra options sit ABOVE the button and slide open like
a drawer (height eased per display refresh, content fading in), the button then reads "Less options"; closing slides
them shut again. Settings ▸ General ▸ Animations off: it opens / closes at once.

    d = OptionsDrawer(content_widget, more="More options", less="Less options")
    layout.addWidget(d)
    d.set_open(True, animate=False)       # restore a remembered state
    d.toggled.connect(...)                # bool: open
    d.is_open()
"""
from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, Qt, pyqtSignal
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from AIO_QC import utils
from . import anim

OPEN_MAX = 16777215                     # QWIDGETSIZE_MAX: when open the content may grow freely


class OptionsDrawer(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, content: QWidget, more: str = "More options", less: str = "Less options",
                 open_: bool = False, ms: int = 300, tip: str = "", parent=None):
        super().__init__(parent)
        self.content = content
        self.more_text, self.less_text, self.ms = more, less, ms
        self._open = False
        self.clip = QWidget()
        self.clip.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        cl = QVBoxLayout(self.clip)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(content)
        self.clip.setMaximumHeight(0)
        self.clip.setVisible(False)
        self.button = QToolButton()
        self.button.setCheckable(True)
        self.button.setAutoRaise(True)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.setObjectName("drawerButton")
        utils.style(self.button, "#drawerButton{color:@accent; font-weight:600; padding:4px 6px; border:0;}"
                                 "#drawerButton:hover{background:@pill; border-radius:6px;}")
        if tip:
            self.button.setToolTip(tip)
        self.button.toggled.connect(lambda on: self.set_open(on))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.clip)                                  # the options lie above …
        lay.addWidget(self.button, 0, Qt.AlignmentFlag.AlignLeft)  # … the button that opens them
        self._label()
        if open_:
            self.set_open(True, animate=False)

    # ------------------------------------------------------------------ api
    def is_open(self) -> bool:
        return self._open

    def set_open(self, on: bool, animate: bool = True):
        on = bool(on)
        if self.button.isChecked() != on:
            self.button.blockSignals(True)
            self.button.setChecked(on)
            self.button.blockSignals(False)
        if on == self._open and (self.clip.maximumHeight() in (0, OPEN_MAX)):
            return
        self._open = on
        self._label()
        self.toggled.emit(on)
        if not animate or not anim.ENABLED or not self.isVisible():
            self._finish(on)
            return
        self._slide(on)

    # ------------------------------------------------------------------ motion
    def _target(self) -> int:
        self.content.adjustSize()
        h = self.content.sizeHint().height()
        return max(h, self.content.minimumSizeHint().height(), 1)

    def _slide(self, on: bool):
        start = self.clip.height() if self.clip.isVisible() else 0
        end = self._target() if on else 0
        eff = self.content.graphicsEffect()
        if not isinstance(eff, QGraphicsOpacityEffect):
            eff = QGraphicsOpacityEffect(self.content)
            self.content.setGraphicsEffect(eff)
        o0 = eff.opacity() if self.clip.isVisible() else 0.0
        o1 = 1.0 if on else 0.0
        self.clip.setMaximumHeight(start)
        self.clip.setVisible(True)

        def step(t):
            self.clip.setMaximumHeight(int(round(start + (end - start) * t)))
            # content fades a little behind the height (opening) / ahead of it (closing)
            k = min(1.0, t * 1.35) if not on else max(0.0, (t - 0.15) / 0.85)
            eff.setOpacity(o0 + (o1 - o0) * k)
            if on:
                self._reveal()                         # a scroll area follows the opening drawer, frame by frame

        anim._run(self.clip, "drawer", self.ms if abs(end - start) > 4 else 60, step, lambda: self._finish(on),
                  QEasingCurve.Type.OutCubic if on else QEasingCurve.Type.InOutCubic)

    def _finish(self, on: bool):
        if self.content.graphicsEffect() is not None:
            self.content.setGraphicsEffect(None)            # crisp text, no offscreen pass once still
        self.clip.setMaximumHeight(OPEN_MAX if on else 0)
        self.clip.setVisible(on)
        if on:
            self._reveal()

    def _reveal(self):
        """Inside a scroll area: keep the opened options and the button in view."""
        p = self.parentWidget()
        while p is not None and not isinstance(p, QScrollArea):
            p = p.parentWidget()
        if isinstance(p, QScrollArea) and p.widget() is not None:
            p.ensureWidgetVisible(self.button, 0, 24)

    def _label(self):
        self.button.setText(self.less_text if self._open else self.more_text)
        self.button.setIcon(utils.newIcon("chevron-up" if self._open else "chevron-down"))
