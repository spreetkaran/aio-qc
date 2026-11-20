# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""The switch in the menu-bar corner of every mode window: Frame Extractor | Image QC | Data Studio."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QButtonGroup, QHBoxLayout, QToolButton, QWidget


MODES = (("video", "Frame Extractor", "mode-video", "Videos — play, mark and extract frames for annotation"),
         ("image", "Image QC", "app", "Annotated image datasets — LabelMe / YOLO quality control"),
         ("data", "Data Studio", "mode-data", "Tables — CSV, Excel, Parquet … analysis, cleaning, features, Spark"))


class ModeSwitch(QWidget):
    """3.0: a grey segmented track; the active mode is a white pill (names only, as in the approved mock-up)."""
    switched = pyqtSignal(str)

    def __init__(self, current: str, parent=None):
        super().__init__(parent)
        from AIO_QC.widgets.chrome import Segmented
        self.setObjectName("modeSwitch")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 3, 8, 3)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        btns = []
        for key, text, icon, tip in MODES:
            b = QToolButton()
            b.setText(text)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            b.setCheckable(True)
            b.setChecked(key == current)
            b.setToolTip(f"{text} — {tip}  (Ctrl+Alt+M switches)")
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setMinimumHeight(26)
            b.clicked.connect(lambda _c=False, k=key: self._clicked(k))
            self.group.addButton(b)
            self.buttons[key] = b
            btns.append(b)
        self.track = Segmented(btns)
        from PyQt6.QtWidgets import QSizePolicy
        self.track.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)   # never squeezed by the chip
        from AIO_QC.widgets.training_monitor import TrainingChip
        self.training_chip = TrainingChip(self)          # shows only while a model trains
        lay.addWidget(self.training_chip)
        lay.addSpacing(6)
        lay.addWidget(self.track)
        self.current = current
    def _clicked(self, key):
        if key != self.current:
            # the pill glides here first; the page that comes to the front continues the same glide
            self.switched.emit(key)
            if self.current != key:                              # not switched (e.g. a mode that can't run)
                self.buttons[self.current].setChecked(True)

    def set_current(self, key, from_key=None, animate=True):
        """The window that comes to the front shows its pill gliding over from the mode before."""
        self.current = key
        b = self.buttons[key]
        b.blockSignals(True)
        b.setChecked(True)
        b.blockSignals(False)
        if animate and from_key in self.buttons and from_key != key:
            self.track.glide_to(b, self.buttons[from_key])
        else:
            self.track.snap()
