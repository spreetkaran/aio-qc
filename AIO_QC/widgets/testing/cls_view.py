# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Model Lab viewer for classifier runs: one test image at a time with each model's answer (best class, confidence,
runner-ups, right / wrong against the image's class) in a band under the picture — one model, or two side by side
with zoom and pan linked. When the test set is a classification dataset (class folders, LabelMe flags) the image's
class can be changed right here (the image moves to the other class folder / its flags are rewritten) and the run is
re-scored.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSplitter,
                             QToolButton, QVBoxLayout, QWidget)

from AIO_QC.widgets.testing.annot_view import AnnotView, link_views, plot_name, save_plot_dialog, to_qimage


class ClsViewer(QDialog):
    """items: [run item index]; models: [mi] (one) or [a, b] (side by side); plotter(ii, mi, show_true, conf)."""
    labels_changed = pyqtSignal(list)                         # run item indices whose class changed

    def __init__(self, parent, run, items, models, start=0, plotter=None, can_edit=None, set_class=None):
        super().__init__(parent)
        self.setWindowTitle("Compare models" if len(models) > 1 else "Model answers")
        self.run, self.items, self.plotter = run, list(items), plotter
        self.can_edit, self.set_class = can_edit, set_class
        self.pos = -1
        self.changed = []
        from AIO_QC.utils.qt_utils import fit_size
        fit_size(self, "large")
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.prev_b = QToolButton()
        self.prev_b.setText("◀")
        self.prev_b.clicked.connect(lambda: self.go(self.pos - 1))
        self.next_b = QToolButton()
        self.next_b.setText("▶")
        self.next_b.clicked.connect(lambda: self.go(self.pos + 1))
        self.where = QLabel("")
        self.where.setTextFormat(Qt.TextFormat.RichText)
        self.truth = QCheckBox("Image's class")
        self.truth.setChecked(True)
        self.truth.setToolTip("Show the image's class under the answer, and mark the answer right / wrong")
        self.truth.setEnabled(bool(run.scored) or run.ground == "labels")
        self.truth.toggled.connect(lambda _: self._show())
        for w in (self.prev_b, self.next_b, self.where):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.truth)
        self.cls_box = QComboBox()
        self.cls_box.setToolTip("The image's class in the test set")
        for n in sorted(run.names.values(), key=str.lower):
            self.cls_box.addItem(str(n))
        self.apply_b = QPushButton("Change class")
        self.apply_b.setToolTip("Give the image this class: class folders move the image, LabelMe files get their "
                                "flags rewritten. The run is re-scored.")
        self.apply_b.clicked.connect(self._apply)
        self.cls_lbl = QLabel("Class")
        for w in (self.cls_lbl, self.cls_box, self.apply_b):
            top.addWidget(w)
        save = QPushButton("Save image…")
        save.clicked.connect(self._save)
        save.setVisible(plotter is not None)
        top.addWidget(save)
        lay.addLayout(top)
        panes = QSplitter()
        self.combos, self.views = [], []
        for m in models:
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            cb = QComboBox()
            for i, s in enumerate(run.specs):
                if run.preds[i] is not None:
                    cb.addItem(s.label, i)
            cb.setCurrentIndex(max(0, cb.findData(m)))
            cb.currentIndexChanged.connect(lambda _=0: self._show())
            view = AnnotView()
            v.addWidget(cb)
            v.addWidget(view, 1)
            self.combos.append(cb)
            self.views.append(view)
            panes.addWidget(w)
        if len(self.views) > 1:
            link_views(*self.views)
        lay.addWidget(panes, 1)
        hint = QLabel("Wheel = zoom · drag = pan · double-click = fit · ← / → next image · 1-9 = class, Enter = "
                      "change · Ctrl+S save")
        hint.setEnabled(False)
        lay.addWidget(hint)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self.go(self.pos - 1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self.go(self.pos + 1))
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self._save)
        for k in range(1, 10):
            QShortcut(QKeySequence(str(k)), self, activated=lambda k=k: self._pick(k - 1))
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=self._apply)
        self.go(start)

    def models(self):
        return [cb.currentData() for cb in self.combos if cb.currentData() is not None]

    def go(self, pos):
        if not self.items:
            return
        pos = max(0, min(len(self.items) - 1, pos))
        if pos == self.pos:
            return
        self.pos = pos
        self._show(fit=True)

    def _editable(self):
        ii = self.items[self.pos]
        return bool(self.set_class is not None and (self.can_edit is None or self.can_edit(ii)))

    def _show(self, fit=False):
        if not 0 <= self.pos < len(self.items):
            return
        ii = self.items[self.pos]
        it = self.run.items[ii]
        vid = it.meta.get("video")
        loc = f"{os.path.basename(vid)} · frame {it.meta.get('frame')}" if vid else os.path.basename(it.path)
        tags = list(it.tags or [])
        self.where.setText(f"<b>{self.pos + 1}</b> / {len(self.items)} · {loc}" +
                           (f" · class <b>{', '.join(tags)}</b>" if tags else ""))
        self.prev_b.setEnabled(self.pos > 0)
        self.next_b.setEnabled(self.pos < len(self.items) - 1)
        ed = self._editable()
        for w in (self.cls_lbl, self.cls_box, self.apply_b):
            w.setVisible(ed)
        if ed and tags:
            k = self.cls_box.findText(tags[0])
            if k >= 0:
                self.cls_box.setCurrentIndex(k)
        show_true = self.truth.isChecked() and self.truth.isEnabled()
        for cb, v in zip(self.combos, self.views):
            mi = cb.currentData()
            if mi is None or self.plotter is None:
                continue
            try:
                img = self.plotter(ii, mi, show_true, None)
            except Exception:                                   # noqa: BLE001 — unreadable image: empty view
                img = None
            v.set_image(to_qimage(img), keep_view=not fit)
            v.caption = ""
            v.update()

    def _pick(self, k):
        if self._editable() and k < self.cls_box.count():
            self.cls_box.setCurrentIndex(k)

    def _apply(self):
        if not self._editable():
            return
        ii = self.items[self.pos]
        new = self.cls_box.currentText()
        if new in (self.run.items[ii].tags or []):
            self.go(self.pos + 1)
            return
        try:
            self.set_class(ii, new)
        except ValueError as e:
            QMessageBox.warning(self, "Change class", str(e))
            return
        self.changed.append(ii)
        self.labels_changed.emit([ii])
        if self.pos < len(self.items) - 1:
            self.go(self.pos + 1)
        else:
            self._show()

    def _save(self):
        if self.plotter is None or not 0 <= self.pos < len(self.items):
            return
        import numpy as np
        ii = self.items[self.pos]
        ims = [self.plotter(ii, mi, self.truth.isChecked() and self.truth.isEnabled(), None) for mi in self.models()]
        if not ims or any(im is None for im in ims):
            save_plot_dialog(self, None, "")
            return
        if len(ims) == 2:
            import cv2
            h = max(a.shape[0] for a in ims)
            ims = [cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(250, 250, 250))
                   for a in ims]
            img = np.hstack([ims[0], np.full((h, max(4, h // 200), 3), 255, np.uint8), ims[1]])
        else:
            img = ims[0]
        name = plot_name(self.run, ii, self.models()[0])
        if len(ims) == 2:
            name = name.split("__")[0] + "__" + "_vs_".join(self.run.specs[m].label for m in self.models())
        save_plot_dialog(self, img, name)
