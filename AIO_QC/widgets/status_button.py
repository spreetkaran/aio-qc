# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
from PyQt6.QtCore import QPropertyAnimation
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsColorizeEffect, QHBoxLayout, QWidget

from AIO_QC.utils import IconLabel

# state -> (icon file, background, hover, pressed) — sharp SVG icons, not emoji
STATES = {
    "default": ("", None, None, None),
    "success": ("status-ok", "#1e7e34", "#259a41", "#155d27"),
    "warning": ("status-warning", "#8a6d00", "#a58300", "#6b5500"),
    "error":   ("status-error", "#a4282a", "#bf3234", "#7f1e20"),
}


class StatusButton(QWidget):
    """3.0: a toolbar button [icon] Name [key hint] [● status] — the status is a small coloured dot inside the button
    (green / amber / red), with the details in the tooltip; an attention flash for problems."""

    DOTS = {"success": "ok", "warning": "warn", "error": "error"}

    def __init__(self, text, icon: str | None = None, slot=None, parent=None, hint: str = "",
                 dot_on_success: bool = True):
        """icon = name of a file in AIO_QC/icons (e.g. "folder-line"); hint = a key shown in the button."""
        super().__init__(parent)
        from AIO_QC.widgets.chrome import RichButton
        self._layout = QHBoxLayout(self)          # NOT self.layout: that would shadow QWidget.layout()
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(4)
        self.button = RichButton(text, icon or "", hint)
        if slot is not None:
            self.button.clicked.connect(lambda _c=False: slot())
        self.status_icon = IconLabel("", 16)        # kept for callers; the dot inside the button shows the status
        self.status_icon.setVisible(False)
        self._layout.addWidget(self.button)
        self.dot_on_success = dot_on_success
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
        icon = STATES[state][0]
        self.status_icon.set_icon(icon)
        self.status_icon.setToolTip(detail)
        self.button.setToolTip(f"{self._base_tip}\n{detail}".strip())
        role = self.DOTS.get(state)
        if state == "success" and not self.dot_on_success:
            role = None
        from AIO_QC.widgets.design import tokens
        self.button.dot.set_color(tokens()[role] if role else None)

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
