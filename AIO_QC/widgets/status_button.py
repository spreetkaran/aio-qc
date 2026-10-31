# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
from PyQt6.QtCore import QPropertyAnimation
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsColorizeEffect, QHBoxLayout, QLabel, QWidget

from AIO_QC.utils import newButton

# state -> (icon, background, hover, pressed)
STATES = {
    "default": ("", None, None, None),
    "success": ("✅", "#1e7e34", "#259a41", "#155d27"),
    "warning": ("⚠️", "#8a6d00", "#a58300", "#6b5500"),
    "error":   ("❌", "#a4282a", "#bf3234", "#7f1e20"),
}


class StatusButton(QWidget):
    """Push button with a coloured status (default / success / warning / error) and an attention flash."""

    def __init__(self, text, icon: str | None = None, slot=None, parent=None):
        """icon = name of a file in AIO_QC/icons (e.g. "open"); slot is connected via newButton."""
        super().__init__(parent)
        self._layout = QHBoxLayout(self)          # NOT self.layout: that would shadow QWidget.layout()
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(4)
        self.status_icon = QLabel()
        self.status_icon.setFixedWidth(20)
        self.button = newButton(text, icon=icon, slot=slot)
        self._layout.addWidget(self.status_icon)
        self._layout.addWidget(self.button)
        self.state = "default"
        self._base_tip = ""
        self._anim: QPropertyAnimation | None = None
        self._effect: QGraphicsColorizeEffect | None = None

    def set_base_tooltip(self, tip: str):
        self._base_tip = tip
        self.button.setToolTip(tip)

    def set_status(self, state: str, detail: str = ""):
        """state = 'default' | 'success' | 'warning' | 'error'; detail is appended to the tooltip."""
        if state not in STATES:
            state = "default"
        self.state = state
        icon, bg, hover, pressed = STATES[state]
        self.status_icon.setText(icon)
        self.status_icon.setToolTip(detail)
        self.button.setToolTip(f"{self._base_tip}\n{detail}".strip())
        if bg is None:
            self.button.setStyleSheet("")
            return
        self.button.setStyleSheet(
            f"QPushButton{{background-color:{bg};color:white;border:1px solid {pressed};"
            f"border-radius:4px;padding:3px 10px;}}"
            f"QPushButton:hover{{background-color:{hover};}}"
            f"QPushButton:pressed{{background-color:{pressed};}}"
            f"QPushButton:disabled{{background-color:#9a9a9a;border-color:#8a8a8a;}}")

    def flash(self, state: str = "warning", detail: str = "", loops: int = 4):
        """Pulse the button to draw attention, then keep `state`."""
        self.set_status(state, detail)
        self._stop_flash()
        effect = QGraphicsColorizeEffect(self.button)
        effect.setColor(QColor("#ffd24d") if state != "error" else QColor("#ff6b6b"))
        effect.setStrength(0.0)
        self.button.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"strength", self)
        anim.setDuration(500)
        anim.setLoopCount(loops)
        anim.setKeyValueAt(0.0, 0.0)
        anim.setKeyValueAt(0.5, 1.0)
        anim.setKeyValueAt(1.0, 0.0)
        anim.finished.connect(self._stop_flash)
        self._anim, self._effect = anim, effect
        anim.start()

    def _stop_flash(self):
        anim, self._anim = self._anim, None
        if anim is not None:
            anim.stop()           # stop BEFORE the effect it animates is destroyed
            anim.deleteLater()
        if self._effect is not None:
            self._effect = None
            self.button.setGraphicsEffect(None)   # Qt deletes the old effect
