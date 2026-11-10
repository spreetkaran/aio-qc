# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Prediction Explorer ▸ sidebar: layers (labels + one row per model with its own confidence), match results, classes,
image filters. Every count is live (objects drawn at the current settings).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QScrollArea, QSizePolicy, QSlider, QToolButton, QVBoxLayout, QWidget,
)

from AIO_QC.core.evaluation import IOU_GRID
from AIO_QC.core.explorer import ERROR_TAGS, NA, TAG_HELP, TAG_NAMES, View
from AIO_QC.utils import get_color_icon, style as apply_style
from .paint import COLOR_MODES, DISAGREE_COLOR, tag_color

DISAGREE = "models disagree"
REVIEW = [("all", "All files"), ("flagged", "Flagged"), ("ok", "Marked OK"), ("fixed", "Fixed"),
          ("none", "Not reviewed")]
SEEN = [("all", "All images"), ("seen", "Trained on (a visible model saw it)"),
        ("unseen", "Held out (no visible model saw it)")]


class Section(QWidget):
    def __init__(self, title, body: QWidget, open_=True):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(2)
        self.btn = QToolButton()
        self.btn.setText(title)
        self.btn.setCheckable(True)
        self.btn.setChecked(open_)
        self.btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn.setArrowType(Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow)
        self.btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        apply_style(self.btn, "QToolButton{border:none; font-weight:700; text-align:left; padding:3px 0;}")
        self.btn.toggled.connect(self._toggle)
        self.extra = QHBoxLayout()
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self.btn, 1)
        head.addLayout(self.extra)
        lay.addLayout(head)
        self.body = body
        lay.addWidget(body)
        body.setVisible(open_)

    def _toggle(self, on):
        self.btn.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.body.setVisible(on)


def _link(text, slot, tip=""):
    b = QToolButton()
    b.setText(text)
    b.setAutoRaise(True)
    b.setToolTip(tip)
    apply_style(b, "QToolButton{color:@accent; border:none; padding:0 3px;}")
    b.clicked.connect(slot)
    return b


class CheckList(QListWidget):
    """Checkable list; changed() after a user toggle. None = everything checked."""
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setSizeAdjustPolicy(QListWidget.SizeAdjustPolicy.AdjustToContents)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.itemChanged.connect(lambda _i: None if self.signalsBlocked() else self.changed.emit())
        self.setUniformItemSizes(True)
        apply_style(self, "QListWidget{border:none; background:transparent;}")

    def set_items(self, items, checked=None):
        """items: [(value, text, color or None, tooltip)]; checked: values (None = all)."""
        self.blockSignals(True)
        self.clear()
        for value, text, color, tip in items:
            it = QListWidgetItem(text)
            it.setData(Qt.ItemDataRole.UserRole, value)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if checked is None or value in checked else Qt.CheckState.Unchecked)
            if color:
                it.setIcon(get_color_icon(QColor(color)))
            if tip:
                it.setToolTip(tip)
            self.addItem(it)
        self.blockSignals(False)
        self._fit()

    def _fit(self):
        n = self.count()
        h = self.sizeHintForRow(0) if n else 18
        self.setFixedHeight(min(n, 14) * h + 6 if n else 6)

    def values(self):
        """Checked values, or None when all are checked."""
        out, allv = [], True
        for i in range(self.count()):
            it = self.item(i)
            if it.isHidden():
                continue
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
            else:
                allv = False
        hidden_unchecked = any(self.item(i).isHidden() and self.item(i).checkState() != Qt.CheckState.Checked
                               for i in range(self.count()))
        return None if allv and not hidden_unchecked else out

    def set_all(self, on: bool, only=None):
        self.blockSignals(True)
        for i in range(self.count()):
            it = self.item(i)
            v = it.data(Qt.ItemDataRole.UserRole)
            want = on if only is None else (v in only)
            it.setCheckState(Qt.CheckState.Checked if want else Qt.CheckState.Unchecked)
        self.blockSignals(False)
        self.changed.emit()

    def set_texts(self, texts: dict):
        self.blockSignals(True)
        for i in range(self.count()):
            it = self.item(i)
            t = texts.get(it.data(Qt.ItemDataRole.UserRole))
            if t is not None:
                it.setText(t)
        self.blockSignals(False)

    def filter_text(self, q: str):
        q = q.lower().strip()
        for i in range(self.count()):
            it = self.item(i)
            it.setHidden(bool(q) and q not in str(it.data(Qt.ItemDataRole.UserRole)).lower())


class ModelRow(QWidget):
    changed = pyqtSignal()

    def __init__(self, mid, name, color, conf=0.25, visible=True):
        super().__init__()
        self.mid = mid
        g = QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(4)
        g.setVerticalSpacing(0)
        self.cb = QCheckBox(name)
        self.cb.setIcon(get_color_icon(QColor(color)))
        self.cb.setChecked(visible)
        self.cb.setToolTip(f"Show {name}'s predictions (dashed, {color})")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(round(conf * 100)))
        self.slider.setToolTip("Confidence threshold of this model")
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.0, 1.0)
        self.spin.setDecimals(2)
        self.spin.setSingleStep(0.05)
        self.spin.setValue(conf)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(62)
        g.addWidget(self.cb, 0, 0, 1, 2)
        g.addWidget(self.slider, 1, 0)
        g.addWidget(self.spin, 1, 1)
        self.slider.valueChanged.connect(self._from_slider)
        self.slider.sliderReleased.connect(self.changed.emit)
        self.spin.valueChanged.connect(self._from_spin)
        self.cb.toggled.connect(lambda _on: self.changed.emit())

    def _from_slider(self, v):
        self.spin.blockSignals(True)
        self.spin.setValue(v / 100)
        self.spin.blockSignals(False)
        if not self.slider.isSliderDown():
            self.changed.emit()
        else:
            self.changed.emit()                     # live while dragging (the panel debounces)

    def _from_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v * 100)))
        self.slider.blockSignals(False)
        self.changed.emit()

    def conf(self) -> float:
        return round(float(self.spin.value()), 2)

    def set_conf(self, v: float, emit=False):
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(float(v))
        self.slider.setValue(int(round(v * 100)))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)
        if emit:
            self.changed.emit()


class Sidebar(QScrollArea):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumWidth(290)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        self.lay = QVBoxLayout(host)
        self.lay.setContentsMargins(6, 4, 8, 4)
        self.lay.setSpacing(4)
        self.setWidget(host)
        self._quiet = False
        self.model_rows: list[ModelRow] = []

        # ---- layers
        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(0, 0, 0, 0)
        self.gt = QCheckBox("Labels (your annotations)")
        self.gt.setChecked(True)
        self.gt.setToolTip("Your labels (solid outlines)")
        ll.addWidget(self.gt)
        self.models_box = QVBoxLayout()
        self.models_box.setSpacing(6)
        ll.addLayout(self.models_box)
        form = QGridLayout()
        form.setContentsMargins(0, 4, 0, 0)
        self.color = QComboBox()
        for k, t in COLOR_MODES:
            self.color.addItem(t, k)
        self.color.setToolTip("Colour of the outlines: by match result (green correct, red not labelled, orange "
                              "missed, purple wrong class, amber poor fit), by class or by model")
        self.iou = QComboBox()
        for v in IOU_GRID:
            self.iou.addItem(f"{v:.2f}", float(v))
        self.iou.setCurrentIndex(int(np.argmin(np.abs(IOU_GRID - 0.5))))
        self.iou.setToolTip("Overlap a prediction needs with a same-class label to count as correct")
        form.addWidget(QLabel("Colour by"), 0, 0)
        form.addWidget(self.color, 0, 1)
        form.addWidget(QLabel("Match IoU ≥"), 1, 0)
        form.addWidget(self.iou, 1, 1)
        ll.addLayout(form)
        self.only = QCheckBox("Draw only the filtered objects")
        self.only.setChecked(True)
        self.only.setToolTip("With a class / result filter, draw just those objects (else every object of the "
                             "matching images)")
        self.names = QCheckBox("Class names on tiles")
        ll.addWidget(self.only)
        ll.addWidget(self.names)
        self.sec_layers = Section("Layers", lw)
        self.lay.addWidget(self.sec_layers)

        # ---- match results
        self.tags = CheckList()
        self.sec_tags = Section("Match result", self.tags)
        self.sec_tags.extra.addWidget(_link("all", lambda: self.tags.set_all(True)))
        self.sec_tags.extra.addWidget(_link("errors", lambda: self.tags.set_all(
            True, only=[TAG_NAMES[t] for t in ERROR_TAGS] + [DISAGREE]), "Only the objects with a problem"))
        self.sec_tags.extra.addWidget(_link("none", lambda: self.tags.set_all(False)))
        self.lay.addWidget(self.sec_tags)

        # ---- classes
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(0, 0, 0, 0)
        self.cls_search = QLineEdit()
        self.cls_search.setPlaceholderText("Find a class…")
        self.cls_search.setClearButtonEnabled(True)
        self.classes = CheckList()
        self.cls_search.textChanged.connect(self.classes.filter_text)
        cl.addWidget(self.cls_search)
        cl.addWidget(self.classes)
        self.sec_classes = Section("Classes  (labels · predictions)", cw)
        self.sec_classes.extra.addWidget(_link("all", lambda: self.classes.set_all(True)))
        self.sec_classes.extra.addWidget(_link("none", lambda: self.classes.set_all(False)))
        self.lay.addWidget(self.sec_classes)

        # ---- images
        iw = QWidget()
        il = QVBoxLayout(iw)
        il.setContentsMargins(0, 0, 0, 0)
        self.splits = CheckList()
        self.folders = CheckList()
        self.review = QComboBox()
        for k, t in REVIEW:
            self.review.addItem(t, k)
        self.seen = QComboBox()
        for k, t in SEEN:
            self.seen.addItem(t, k)
        self.seen.setToolTip("Uses each model's “Trained on” setting (Models & Environment)")
        self.split_lbl = QLabel("Split")
        self.folder_lbl = QLabel("Folder")
        for w in (self.split_lbl, self.splits, self.folder_lbl, self.folders, QLabel("Review"), self.review,
                  QLabel("Training"), self.seen):
            il.addWidget(w)
        self.sec_images = Section("Images", iw, open_=False)
        self.lay.addWidget(self.sec_images)
        self.lay.addStretch(1)

        for w in (self.gt, self.only, self.names):
            w.toggled.connect(self._emit)
        for w in (self.color, self.iou, self.review, self.seen):
            w.currentIndexChanged.connect(self._emit)
        for w in (self.tags, self.classes, self.splits, self.folders):
            w.changed.connect(self._emit)

    def _emit(self, *_):
        if not self._quiet:
            self.changed.emit()

    # ------------------------------------------------------------------ data
    def set_data(self, d, v: View):
        self._quiet = True
        for r in self.model_rows:
            r.setParent(None)
            r.deleteLater()
        self.model_rows = []
        for m in d.models:
            r = ModelRow(m, d.model_names.get(m, m), d.model_colors.get(m, "#ff2bd6"), v.conf_of(m), v.visible(m))
            r.changed.connect(self._emit)
            self.models_box.addWidget(r)
            self.model_rows.append(r)
        tag_items = [(TAG_NAMES[t], TAG_NAMES[t], tag_color(t).name(), TAG_HELP.get(t, "")) for t in
                     (0, 1, 2, 3, 4, 5, NA)]
        if len(d.models) >= 2:
            tag_items.append((DISAGREE, DISAGREE, DISAGREE_COLOR, "a prediction that another visible model does "
                                                                  "not make (same class, IoU ≥ 0.3)"))
        self.tags.set_items(tag_items, v.tags)
        self.classes.set_items([(l, l, None, "") for l in sorted(d.labels, key=str.lower)], v.classes)
        self.splits.set_items([(s, s or "(no split)", None, "") for s in d.splits], v.splits)
        self.splits.setVisible(len(d.splits) > 1)
        self.split_lbl.setVisible(len(d.splits) > 1)
        self.folders.set_items([(f, f or "(top folder)", None, "") for f in d.folders], v.folders)
        self.folders.setVisible(len(d.folders) > 1)
        self.folder_lbl.setVisible(len(d.folders) > 1)
        self.load(v)
        self._quiet = False

    def load(self, v: View):
        quiet, self._quiet = self._quiet, True
        self.gt.setChecked(bool(v.show_gt))
        for r in self.model_rows:
            r.cb.setChecked(v.visible(r.mid))
            r.set_conf(v.conf_of(r.mid))
        self.color.setCurrentIndex(max(0, self.color.findData(v.color)))
        self.iou.setCurrentIndex(int(np.argmin(np.abs(IOU_GRID - v.iou))))
        self.only.setChecked(bool(v.only_matching))
        self.names.setChecked(bool(v.tile_names))
        for lst, vals in ((self.tags, v.tags), (self.classes, v.classes), (self.splits, v.splits),
                          (self.folders, v.folders)):
            lst.blockSignals(True)
            for i in range(lst.count()):
                it = lst.item(i)
                on = vals is None or it.data(Qt.ItemDataRole.UserRole) in vals
                it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            lst.blockSignals(False)
        rv = "all" if not v.review else (v.review[0] if len(v.review) == 1 else "all")
        self.review.setCurrentIndex(max(0, self.review.findData(rv)))
        self.seen.setCurrentIndex(max(0, self.seen.findData(v.seen)))
        self._quiet = quiet

    def read_into(self, v: View) -> View:
        v.show_gt = self.gt.isChecked()
        v.models = {r.mid: {"visible": r.cb.isChecked(), "conf": r.conf()} for r in self.model_rows}
        v.color = self.color.currentData()
        v.iou = float(self.iou.currentData())
        v.only_matching = self.only.isChecked()
        v.tile_names = self.names.isChecked()
        v.tags = self.tags.values()
        v.classes = self.classes.values()
        v.splits = self.splits.values() if self.splits.isVisible() or self.splits.count() > 1 else None
        v.folders = self.folders.values() if self.folders.count() > 1 else None
        r = self.review.currentData()
        v.review = None if r == "all" else [r]
        v.seen = self.seen.currentData()
        return v

    def model_row(self, mid) -> ModelRow | None:
        return next((r for r in self.model_rows if r.mid == mid), None)

    def update_counts(self, st, d):
        self.tags.set_texts({k: f"{k}  ({n:,})" for k, n in st.tag_counts.items()})
        self.classes.set_texts({l: f"{l}  ({g:,} · {p:,})" for l, (g, p) in st.class_counts.items()})
        n_gt = int(st.g_show.sum()) if len(st.g_show) else 0
        self.gt.setText(f"Labels (your annotations)  ·  {n_gt:,}")
        for r in self.model_rows:
            mi = d.models.index(r.mid)
            n = int((st.p_show & (d.p_model == mi)).sum()) if len(st.p_show) else 0
            r.cb.setText(f"{d.model_names.get(r.mid, r.mid)}  ·  {n:,}")
