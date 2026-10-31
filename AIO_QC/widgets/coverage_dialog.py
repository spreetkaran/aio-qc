# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
✅ Review coverage map: every file as a cell coloured by review state, grouped by split / folder.
Answers "which parts of the dataset have been QC'ed?" at a glance; click any cell to open that file.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QProgressBar, QScrollArea, QVBoxLayout

from AIO_QC.core.dataset import SPLIT_ORDER
from AIO_QC.core.state import REVIEW_FIXED, REVIEW_FLAG, REVIEW_OK
from AIO_QC.utils import newButton, style as apply_style
from .charts import CoverageMap


class CoverageDialog(QDialog):
    TITLE, ICON = "Coverage Map", "✅"
    SUPPORTED = None

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.setWindowTitle(self.TITLE)
        self.resize(int(app.width() * 0.6), int(app.height() * 0.75))

        self.scope = QComboBox()
        self.scope.addItems(["All files", "Files in current list"])
        self.group = QComboBox()
        self.group.addItem("Group by split", "split")
        if self.session.folders:
            self.group.addItem("Group by folder", "folder")
        self.group.addItem("No grouping", "none")
        for w in (self.scope, self.group):
            w.currentIndexChanged.connect(lambda _i: self.rebuild())
        nxt = newButton("Next unreviewed", icon="review-next", slot=self.app.goto_next_unreviewed,
                        tip="Open the next file without a review mark (N)")
        top = QHBoxLayout()
        top.addWidget(self.scope)
        top.addWidget(self.group)
        top.addStretch(1)
        top.addWidget(nxt)

        legend = QHBoxLayout()
        self.counts = {}
        for key, role, text in ((None, "placeholder", "Unreviewed"), (REVIEW_OK, "ok", "OK"), (REVIEW_FLAG, "warn", "Flagged"),
                                (REVIEW_FIXED, "info", "Fixed")):
            sw = QLabel()
            sw.setFixedSize(14, 14)
            apply_style(sw, f"background:@{role}; border-radius:3px;")
            lab = QLabel()
            legend.addWidget(sw)
            legend.addWidget(lab)
            legend.addSpacing(12)
            self.counts[key] = (lab, text)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setFormat("%p% reviewed")
        legend.addStretch(1)
        legend.addWidget(self.progress)

        self.map = CoverageMap()
        self.map.activated.connect(lambda k: self.app.open_sample(k))
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(self.map)
        hint = QLabel("Every square is one file · click to open it · the outlined square is the current file")
        apply_style(hint, "color:@muted;")

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(legend)
        lay.addWidget(area, 1)
        lay.addWidget(hint)

        self.session.reviewChanged.connect(self._on_review)
        from PyQt6.QtCore import QTimer
        self._rebuild_soon = QTimer(self, singleShot=True, interval=400, timeout=self.rebuild)   # bursts -> one rebuild
        self.session.samplesRemoved.connect(lambda _k: self._rebuild_soon.start())
        self.session.samplesAdded.connect(lambda _k: self._rebuild_soon.start())
        self.app.file_dock.currentKeyChanged.connect(self.map.set_current)
        self.rebuild()

    def _samples(self):
        if self.scope.currentIndex() == 1:
            keys = set(self.app.file_dock.visible_keys())
            return [s for s in self.session.samples if s.key in keys]
        return list(self.session.samples)

    def rebuild(self):
        samples = self._samples()
        mode = self.group.currentData()
        groups: dict[str, list] = {}
        for s in samples:
            g = (s.split or "no split") if mode == "split" else ((s.folder or "(root)") if mode == "folder" else "All files")
            groups.setdefault(g, []).append(s.key)
        order = sorted(groups, key=lambda g: (SPLIT_ORDER.get(g, 9), g.lower())) if mode == "split" else sorted(groups, key=str.lower)
        rv = self.session.review
        sections = []
        for g in order:
            keys = groups[g]
            done = sum(1 for k in keys if rv.status(k))
            sections.append((f"{g} — {done:,}/{len(keys):,} reviewed", keys))
        by_key = self.session.by_key
        self.map.set_data(sections, rv.status, lambda k: by_key[k].display if k in by_key else k)
        self.map.set_current(self.app.current_key)
        self._update_counts(samples)

    def _update_counts(self, samples=None):
        samples = samples if samples is not None else self._samples()
        keys = [s.key for s in samples]
        c = self.session.review.counts(keys)
        n = len(keys)
        un = n - c[REVIEW_OK] - c[REVIEW_FLAG] - c[REVIEW_FIXED]
        for key, val in ((None, un), (REVIEW_OK, c[REVIEW_OK]), (REVIEW_FLAG, c[REVIEW_FLAG]),
                         (REVIEW_FIXED, c[REVIEW_FIXED])):
            lab, text = self.counts[key]
            lab.setText(f"{text} {val:,}")
        self.progress.setRange(0, max(n, 1))
        self.progress.setValue(n - un)

    def _on_review(self, keys):
        self.map.update_keys(keys)
        self._update_counts()

    def closeEvent(self, event):
        for sig, slot in ((self.session.reviewChanged, self._on_review),
                          (self.app.file_dock.currentKeyChanged, self.map.set_current)):
            try:
                sig.disconnect(slot)
            except TypeError:
                pass
        self.app.forget_dialog(self)
        super().closeEvent(event)
