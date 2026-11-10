# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""The two-way switch in the menu-bar corner of both windows: Image QC | Data Studio."""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import QButtonGroup, QHBoxLayout, QToolButton, QWidget

from AIO_QC.utils import newIcon, style as apply_style

MODES = (("image", "Image QC", "mode-image", "Annotated image datasets — LabelMe / YOLO quality control"),
         ("data", "Data Studio", "mode-data", "Tables — CSV, Excel, Parquet … analysis, cleaning, features, Spark"))


class ModeSwitch(QWidget):
    switched = pyqtSignal(str)

    def __init__(self, current: str, parent=None):
        super().__init__(parent)
        self.setObjectName("modeSwitch")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 6, 2)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        for i, (key, text, icon, tip) in enumerate(MODES):
            b = QToolButton()
            b.setText(text)
            b.setIcon(newIcon(icon))
            b.setProperty("aio_icon", icon)
            b.setIconSize(QSize(16, 16))
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            b.setCheckable(True)
            b.setChecked(key == current)
            b.setToolTip(f"{text} — {tip}  (Ctrl+Alt+M switches)")
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setObjectName("modeLeft" if i == 0 else "modeRight")
            b.clicked.connect(lambda _c=False, k=key: self._clicked(k))
            self.group.addButton(b)
            self.buttons[key] = b
            lay.addWidget(b)
        apply_style(self, "QToolButton{border:1px solid @line; padding:2px 10px; background:transparent;} "
                          "QToolButton#modeLeft{border-top-left-radius:11px; border-bottom-left-radius:11px; "
                          "border-right:none;} "
                          "QToolButton#modeRight{border-top-right-radius:11px; border-bottom-right-radius:11px;} "
                          "QToolButton:checked{background:@accent; color:white; border-color:@accent;} "
                          "QToolButton:hover:!checked{background:@surface;}")
        self.current = current

    def _clicked(self, key):
        if key != self.current:
            self.buttons[self.current].setChecked(True)          # the window that opens shows its own switch
            self.switched.emit(key)

    def set_current(self, key):
        self.current = key
        self.buttons[key].setChecked(True)
