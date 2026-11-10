# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Get Started: a short animated tour shown on first launch (and from Help ▸ Get Started).

Six cards - what the app does, opening and playing, picking frames, saving at scale, search and Image QC,
and where to find help - each with an action button that does the real thing. Pages slide and cross-fade, the
page indicator glides (utils.anim, display-rate). Arrow keys / Enter navigate.
"""
from PyQt6.QtCore import Qt, QRectF, QSize, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QKeySequence, QShortcut, QPalette
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QCheckBox)

from AIO_QC.tools.frame_extractor import __appname__, __version__
from AIO_QC.widgets import motion as anim
from AIO_QC.tools.frame_extractor.utils import newIcon
from AIO_QC.utils.fx_settings import fx_settings

PAGES = [
    ("icon", f"Welcome to {__appname__} {__version__}",
     "Turn hours of video into clean, annotation-ready images — and find out how well your detection models really "
     "perform on that footage.<br><br>This tour takes a minute. You can reopen it any time from <b>Help ▸ Get Started</b>.",
     None),
    ("open", "Open and play",
     "Open a video with <b>Ctrl+O</b>, or drop it on the window. <b>Space</b> plays and pauses, <b>,</b> and <b>.</b> step "
     "one frame, the mouse wheel steps too. <b>Ctrl + wheel</b> zooms to the original pixels; <b>Ctrl+0</b> fits the "
     "window again.<br><br>Click the position under the video to type any time, timecode or frame.",
     ("Open a video…", "open")),
    ("marker_add", "Pick the right frames",
     "Press <b>M</b> to drop a marker — even while the video plays. <b>Snap</b> moves markers to the sharpest nearby frame, "
     "<b>Suggest</b> proposes scene starts or the sharpest frame per window, and <b>Ctrl+E</b> saves them all at full "
     "resolution.<br><br>Press <b>R</b> to draw a region of interest; saved frames are cropped to it.",
     None),
    ("batch", "Extract at scale",
     "<b>Ctrl+B</b> opens the batch queue: many videos in parallel, with blur, duplicate and scene filters, cross-video "
     "de-duplication, model-based mining and pre-annotations. Stop at any time — it resumes where it left off.<br><br>"
     "<b>Ctrl+G</b> reviews the results and rejects bad frames before they reach annotators.",
     ("Open the batch queue", "batch")),
    ("search", "Find moments, then QC the frames",
     "<b>Ctrl+F</b> searches your videos by description or by example. When frames are extracted, <b>File ▸ Open output "
     "in Image QC</b> makes them (with their pre-labels) the dataset in AIO QC's Image QC mode — to measure models on "
     "them, use <b>Testing</b> in Image QC's Models tab.",
     ("Open output in Image QC…", "qc")),
    ("help", "Help is always one key away",
     "<b>F1</b> opens the User's Guide, <b>Ctrl+/</b> the keyboard shortcuts, and <b>Ctrl+K</b> the command palette — "
     "type any command's name to run it.<br><br>Hover any control for a short explanation.",
     ("Open the User's Guide", "guide")),
]


class PageDots(QWidget):
    """Page indicator: the active dot is a pill that glides to its new place."""

    def __init__(self, n, parent=None):
        super().__init__(parent)
        self.n, self.pos = n, 0.0
        self.f = anim.Follower(self._set, 0.0, tau=0.06, eps=0.002, widget=self)
        self.setFixedSize(QSize(n * 18 + 16, 16))

    def _set(self, v):
        self.pos = v
        self.update()

    def set_index(self, i):
        self.f.set_target(float(i))

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        dim = QColor(pal.color(QPalette.ColorRole.WindowText))
        dim.setAlphaF(0.25)
        on = pal.color(QPalette.ColorRole.Highlight)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(dim)
        for i in range(self.n):
            p.drawEllipse(QRectF(4 + i * 18, 5, 6, 6))
        x = 4 + self.pos * 18
        p.setBrush(on)
        p.drawRoundedRect(QRectF(x - 3, 4, 14, 8), 4, 4)
        p.end()


class GetStarted(QDialog):
    action = pyqtSignal(str)          # "open" | "batch" | "qc" | "guide"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Get Started — {__appname__}")
        self.setMinimumSize(520, 380)
        from AIO_QC.tools.frame_extractor.utils.qt_utils import preferred_size
        preferred_size(self, 720, 480)
        self.st = fx_settings()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 18)
        self.stack = anim.AnimatedStack()
        for icon, title, text, act in PAGES:
            self.stack.addWidget(self._page(icon, title, text, act))
        lay.addWidget(self.stack, 1)
        bottom = QHBoxLayout()
        self.show_again = QCheckBox("Show at startup")
        self.show_again.setChecked(self.st.value("ui/get_started_startup", True, type=bool))
        self._touched = False
        self.show_again.toggled.connect(self._toggled)
        bottom.addWidget(self.show_again)
        bottom.addStretch(1)
        self.dots = PageDots(len(PAGES))
        bottom.addWidget(self.dots)
        bottom.addStretch(1)
        self.back = QPushButton("Back")
        self.back.clicked.connect(lambda: self.go(self.stack.currentIndex() - 1))
        self.next = QPushButton("Next")
        self.next.setDefault(True)
        self.next.clicked.connect(self._next)
        bottom.addWidget(self.back)
        bottom.addWidget(self.next)
        lay.addLayout(bottom)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=self._next)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self.go(self.stack.currentIndex() - 1))
        self.go(0)

    def _page(self, icon, title, text, act):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(14)
        from AIO_QC.utils import IconLabel
        ic = IconLabel(newIcon(icon), 72)              # sharp at any screen scaling
        v.addWidget(ic)
        t = QLabel(title)
        f = t.font()
        f.setPointSizeF(f.pointSizeF() * 1.7)
        f.setWeight(f.Weight.DemiBold)
        t.setFont(f)
        t.setWordWrap(True)
        v.addWidget(t)
        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        f2 = body.font()
        f2.setPointSizeF(f2.pointSizeF() * 1.12)
        body.setFont(f2)
        v.addWidget(body)
        if act:
            b = QPushButton(act[0])
            b.setMinimumHeight(32)
            b.clicked.connect(lambda _=False, k=act[1]: self._act(k))
            row = QHBoxLayout()
            row.addWidget(b)
            row.addStretch(1)
            v.addLayout(row)
        v.addStretch(1)
        return w

    def _toggled(self, on):
        self._touched = True
        self.st.setValue("ui/get_started_startup", on)

    def _complete(self):
        """Seen the tour: stop showing it at startup unless the user asked for that explicitly."""
        if not self._touched:
            self.st.setValue("ui/get_started_startup", False)

    def _act(self, key):
        self._complete()
        self.accept()
        self.action.emit(key)

    def _next(self):
        i = self.stack.currentIndex()
        if i >= self.stack.count() - 1:
            self._complete()
            self.accept()
        else:
            self.go(i + 1)

    def go(self, i):
        i = max(0, min(self.stack.count() - 1, i))
        self.stack.setCurrentIndex(i)
        self.dots.set_index(i)
        self.back.setEnabled(i > 0)
        self.next.setText("Start using AIO QC" if i == self.stack.count() - 1 else "Next")


def should_show_at_startup():
    """First launch, or the user ticked "Show at startup"."""
    return fx_settings().value("ui/get_started_startup", True, type=bool)
