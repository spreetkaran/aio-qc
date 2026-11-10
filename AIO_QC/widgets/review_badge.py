# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The open file's review mark in the status bar: a sharp icon + one short word (Pending · OK · Flagged · Fixed) in a
fixed width, so nothing to its left moves from one image to the next. A flag note is shown by a note icon; the note
itself is in the tooltip.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from AIO_QC import utils

WORDS = ("Pending", "OK", "Flagged", "Fixed")
# status -> (icon, word, colour role, bold)
LOOK = {
    None: ("review-pending", "Pending", "muted", False),
    "ok": ("review-ok", "OK", "ok", True),
    "flag": ("review-flag", "Flagged", "warn", True),
    "fixed": ("review-fixed", "Fixed", "info", True),
}


class ReviewBadge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        lay.setSpacing(4)
        self.icon = utils.IconLabel("review-pending", 14)
        self.word = QLabel("")
        self.note = utils.IconLabel("review-note", 14)
        self.note.setVisible(False)
        lay.addWidget(self.icon)
        lay.addWidget(self.word)
        lay.addWidget(self.note)
        lay.addStretch(1)
        self.status = "unset"
        self._fix_width()

    def _fix_width(self):
        f = self.word.font()
        f.setBold(True)
        from PyQt6.QtGui import QFontMetrics
        widest = max(QFontMetrics(f).horizontalAdvance(w) for w in WORDS)
        self.word.setFixedWidth(widest + 2)
        self.setFixedWidth(6 + 14 + 4 + widest + 2 + 4 + 14 + 6)

    def text(self) -> str:
        return self.word.text()

    def set_status(self, status, note: str = "", tip: str = ""):
        icon, word, role, bold = LOOK.get(status, LOOK[None])
        if status != self.status:
            self.status = status
            self.icon.set_icon(icon)
            self.word.setText(word)
            utils.style(self.word, f"color:@{role};" + (" font-weight:bold;" if bold else ""))
        self.note.setVisible(bool(note))
        self.setToolTip(tip)
        self.word.setToolTip(tip)
        # paint now: loading a large image right after this must not leave the previous mark on screen meanwhile
        self.repaint()

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (e.Type.FontChange, e.Type.StyleChange):
            self._fix_width()


class IconText(QWidget):
    """A sharp icon + a label (instead of an emoji in the text): the dataset name in the toolbar, Drag / Select in
    the status bar. setText / text / setIcon like a label."""

    def __init__(self, icon: str = "", text: str = "", size: int = 14, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.setSpacing(5)
        self.icon = utils.IconLabel(icon, size)
        self.icon.setVisible(bool(icon))
        self.label = QLabel(text)
        lay.addWidget(self.icon)
        lay.addWidget(self.label)

    def setIcon(self, icon: str):
        self.icon.set_icon(icon)
        self.icon.setVisible(bool(icon))

    def setText(self, text: str):
        self.label.setText(text)

    def text(self) -> str:
        return self.label.text()
