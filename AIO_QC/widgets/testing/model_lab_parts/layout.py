# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Testing (Model Lab) — the page layout.

A mixin of AIO_QC.widgets.testing.model_lab.ModelLab: the methods use the window's own attributes, set up in its __init__.
"""
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QComboBox, QFrame,
)
from AIO_QC.widgets import motion as anim
from AIO_QC.widgets.testing.lab_common import (
    IssueBar,
)

from AIO_QC.widgets.testing.model_lab_common import (STEPS, _hint, _scroll)


class LayoutMixin:

    # ------------------------------------------------------------------ layout
    def _build(self):
        central = QWidget()
        h = QVBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        # the steps as a stepper across the top (fits the Models tab's tool windows): ⌂ Projects | 1 … 5
        top = QWidget()
        top.setObjectName("testingStepper")
        sv = QHBoxLayout(top)
        sv.setContentsMargins(8, 6, 8, 4)
        sv.setSpacing(6)
        home_b = QPushButton("⌂  Projects")
        home_b.setFlat(True)
        home_b.setToolTip("Start page: new / open / recent projects")
        home_b.clicked.connect(lambda: self._show_home())
        sv.addWidget(home_b)
        self.nav = QListWidget()
        self.nav.setFlow(QListWidget.Flow.LeftToRight)
        self.nav.setWrapping(False)
        self.nav.setIconSize(QSize(16, 16))
        self.nav.setSpacing(2)
        self.nav.setFixedHeight(40)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.nav.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for i, (t, sub) in enumerate(STEPS):
            it = QListWidgetItem(f"{i + 1}  {t}" + ("   ›" if i < len(STEPS) - 1 else ""))
            it.setToolTip(sub)
            it.setSizeHint(QSize(150, 32))
            self.nav.addItem(it)
        self.nav.currentRowChanged.connect(self._go)
        self.nav.setFrameShape(QFrame.Shape.NoFrame)
        sv.addWidget(self.nav, 1)
        h.addWidget(top)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        h.addWidget(line)
        self.pages = anim.AnimatedStack()
        h.addWidget(self.pages, 1)
        self.setCentralWidget(central)
        self.pages.addWidget(self._page_data())
        # every page scrolls when the window is small: the window never needs more than the screen
        self.pages.addWidget(_scroll(self._page_models()))
        self.pages.addWidget(_scroll(self._page_run()))
        self.pages.addWidget(self._page_results())
        self.pages.addWidget(_scroll(self._page_report()))
        self.pages.addWidget(self._page_home())              # HOME
        self.pages.addWidget(_scroll(self._page_review()))   # REVIEW
        self.status = self.statusBar()
        # long model / file names must not widen the window: combos elide, labels wrap
        for cb in self.findChildren(QComboBox):
            cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(10)
        self.run_info.setWordWrap(True)
        self.nav.setCurrentRow(0)

    def _go(self, row):
        if row < 0:
            return
        self.pages.setCurrentIndex(self.REVIEW if row == 5 else row)
        if row == 5:
            self._refresh_review_page()
        if row == 1:
            self._refresh_mapping_hint()
        if row == 2:
            self._refresh_run_page()
        if row == 4:
            self._refresh_report_page()

    def _page(self, title, subtitle, step=None):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)
        t = QLabel(f"<span style='font-size:15pt; font-weight:600'>{title}</span>")
        lay.addWidget(t)
        if subtitle:
            lay.addWidget(_hint(subtitle))
        if step is not None:
            self.issue_bars[step] = IssueBar()
            lay.addWidget(self.issue_bars[step])
        return w, lay
