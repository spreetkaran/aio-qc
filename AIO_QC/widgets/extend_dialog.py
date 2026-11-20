# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Extend Model (Train Model ▸ Start from: my model + more images / + new classes; Improve Model ▸ Retrain): the plan,
fully visible before anything runs — core.extend does the work.

  · the class plan: the model's classes keep their ids (locked); the new data's classes are matched by name, the
    rest added at the end; per class: add / same as … / rename … / leave out — with the object counts, live
  · completeness: which of the model's classes the new labels have; the rest are pre-labelled by the model
  · the old training data (recommended: without it the model forgets), how big the copy is, time per epoch
  · two stages (the new head first, the backbone frozen; then everything at a low learning rate), the forgetting
    limit checked after training on the old validation images
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from AIO_QC import utils
from AIO_QC.core import extend as X

GOALS = {"images": "More images — the same classes", "classes": "New classes — the old ones kept"}


class ExtendDialog(QDialog):
    TITLE = "Extend Model"

    def __init__(self, parent, base: str, new_yaml: str, python: str = "", goal: str = "classes",
                 info: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(utils.newIcon("tool-train"))
        self.resize(980, 760)
        self.base, self.new_yaml, self.python = str(base), str(new_yaml), python
        self.info = info if info is not None else X.base_info(base, python)
        self.base_names = list(self.info.get("names") or [])
        self.new_labels = X.yaml_names(new_yaml) if new_yaml and Path(new_yaml).exists() else []
        self.actions: dict = {}
        self.result_plan: X.ExtendPlan | None = None
        self.counts_new = self._counts(new_yaml)
        lay = QVBoxLayout(self)
        head = QLabel(f"<b>{Path(base).name}</b> — {len(self.base_names)} class(es)"
                      + (f" · trained in {Path(self.info['run']).name}" if self.info.get("run") else "")
                      + (f"<br><span style='color:{utils.hexc('error')}'>Its classes could not be read: "
                         f"{self.info.get('error', '')[:300]}</span>" if not self.base_names else ""))
        head.setWordWrap(True)
        lay.addWidget(head)
        f = QFormLayout()
        self.goal = QComboBox()
        for k, t in GOALS.items():
            self.goal.addItem(t, k)
        self.goal.setCurrentIndex(max(0, self.goal.findData(goal)))
        self.goal.currentIndexChanged.connect(lambda _i: self._refresh())
        f.addRow("Goal", self.goal)
        self.old_yaml = QLineEdit(self.info.get("dataset_yaml") or "")
        self.old_yaml.setPlaceholderText("The data the model was trained on (dataset.yaml) — strongly recommended")
        self.old_yaml.textChanged.connect(lambda _t: self._refresh())
        orow = QHBoxLayout()
        orow.addWidget(self.old_yaml, 1)
        orow.addWidget(utils.newButton("Browse…", icon="open", slot=self._browse_old))
        f.addRow("Old training data", orow)
        f.addRow("New images", QLabel(new_yaml or "—"))
        lay.addLayout(f)

        # ---------------------------------------------------------------- class plan
        box = QGroupBox("Classes")
        bl = QVBoxLayout(box)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Class", "Id in the model", "New id", "What happens", "Objects (old)",
                                              "Objects (new)"])
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        bl.addWidget(self.table, 1)
        self.complete = QLabel("")
        self.complete.setWordWrap(True)
        self.complete.setTextFormat(Qt.TextFormat.RichText)
        bl.addWidget(self.complete)
        lay.addWidget(box, 1)

        # ---------------------------------------------------------------- how
        how = QGroupBox("How")
        hl = QFormLayout(how)
        self.prelabel = QCheckBox("The model labels its own classes on the new images")
        self.prelabel.setChecked(True)
        self.prelabel.setToolTip("Images labelled only with the new classes still show cars, people …: unlabelled, the "
                                 "model would learn them as background and forget them")
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.1, 0.95)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.5)
        self.conf.setPrefix("confidence ≥ ")
        self.review = QCheckBox("Let me review them before training (open the training copy in AIO QC)")
        self.review.setChecked(True)
        pr = QHBoxLayout()
        pr.addWidget(self.prelabel)
        pr.addWidget(self.conf)
        pr.addStretch(1)
        hl.addRow("Pre-labels", pr)
        hl.addRow("", self.review)
        self.two = QCheckBox("Two stages: the new head first (backbone frozen), then everything at a low learning rate")
        self.two.setChecked(True)
        self.e1 = QSpinBox()
        self.e1.setRange(1, 100)
        self.e1.setValue(5)
        self.e1.setPrefix("stage 1: ")
        self.e1.setSuffix(" epochs")
        self.frz = QSpinBox()
        self.frz.setRange(1, 30)
        self.frz.setValue(10)
        self.frz.setPrefix("freeze ")
        self.frz.setSuffix(" layers")
        self.lr2 = QDoubleSpinBox()
        self.lr2.setDecimals(4)
        self.lr2.setRange(0.0001, 0.02)
        self.lr2.setSingleStep(0.0005)
        self.lr2.setValue(0.001)
        self.lr2.setPrefix("stage 2 learning rate ")
        sr = QHBoxLayout()
        for w in (self.e1, self.frz, self.lr2):
            sr.addWidget(w)
        sr.addStretch(1)
        hl.addRow("Training", self.two)
        hl.addRow("", sr)
        self.limit = QDoubleSpinBox()
        self.limit.setRange(0.5, 20)
        self.limit.setValue(X.FORGET_LIMIT * 100)
        self.limit.setSuffix(" mAP points")
        self.limit.setToolTip("After training, every old class is scored on the old validation images, before and "
                              "after: a class that lost more than this is flagged")
        hl.addRow("Flag an old class that drops more than", self.limit)
        lay.addWidget(how)
        if X.is_cls_folder(new_yaml):                       # a classifier: every image has its class — no pre-labels
            self.prelabel.setChecked(False)
            for w in (self.prelabel, self.conf, self.review):
                w.setVisible(False)
            self.cls_note = QLabel("A classifier: each image already has its class, so nothing is pre-labelled. Its "
                                   "class ids follow the class folders' names; the old classes' part of its last "
                                   "layer is copied over by name.")
            self.cls_note.setWordWrap(True)
            hl.insertRow(0, self.cls_note)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.summary)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok = bb.button(QDialogButtonBox.StandardButton.Ok)
        self.ok.setText("Use this plan")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._refresh()

    # ------------------------------------------------------------------ data
    @staticmethod
    def _counts(yaml) -> dict:
        """{class name: objects} in the training split of a dataset.yaml (a quick read of its label files)."""
        from collections import Counter
        if yaml and Path(yaml).is_dir():                   # a classification folder: images per class folder
            return dict(Counter(c for _p, c in X.cls_images(yaml, "train")))
        if not yaml or not Path(yaml).is_file():
            return {}

        from AIO_QC.core.training import _label_path
        names = X.yaml_names(yaml)
        c = Counter()
        for img in X._yaml_images(yaml, "train")[:20000]:
            lp = _label_path(img)
            try:
                for ln in lp.read_text(encoding="utf-8", errors="replace").splitlines():
                    p = ln.split()
                    if p:
                        c[int(float(p[0]))] += 1
            except (OSError, ValueError):
                continue
        return {names[i] if i < len(names) else str(i): n for i, n in c.items()}

    def _browse_old(self):
        if X.is_cls_folder(self.new_yaml):
            p = QFileDialog.getExistingDirectory(self, "The classifier's training data (class folders)",
                                                 self.old_yaml.text())
        else:
            p, _f = QFileDialog.getOpenFileName(self, "The model's training data", self.old_yaml.text(),
                                                "Dataset (*.yaml *.yml)")
        if p:
            self.old_yaml.setText(p)

    def plan(self) -> X.ExtendPlan:
        rows, names, old_of, lm = self._classes()
        return X.ExtendPlan(base=self.base, base_names=self.base_names, names=names, old_of=old_of, label_map=lm,
                            new_yaml=self.new_yaml, old_yaml=self.old_yaml.text().strip(),
                            goal=self.goal.currentData(), prelabel=self.prelabel.isChecked(),
                            prelabel_conf=self.conf.value(), review=self.review.isChecked() and
                            self.prelabel.isChecked(), two_stage=self.two.isChecked(), stage1_epochs=self.e1.value(),
                            freeze=self.frz.value(), stage2_lr=self.lr2.value(), forget_limit=self.limit.value() / 100,
                            parent={"run": self.info.get("run"), "metrics": self.info.get("metrics")})

    def _classes(self):
        acts = dict(self.actions)
        if self.goal.currentData() == "images":            # the same classes: anything else is left out
            known = {X.norm(n) for n in self.base_names}
            for lab in self.new_labels:
                if X.norm(lab) not in known and not acts.get(lab):
                    acts[lab] = (X.DROP,)
        oy = self.old_yaml.text().strip()
        if getattr(self, "_old_counts_for", None) != oy:
            self._old_counts, self._old_counts_for = self._counts(oy) if oy else {}, oy
        old_counts = self._old_counts
        return X.plan_classes(self.base_names, self.new_labels, acts, counts_old=old_counts,
                              counts_new=self.counts_new)

    # ------------------------------------------------------------------ view
    def _refresh(self):
        rows, names, _old_of, lm = self._classes()
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            cells = [row.name, "" if row.old_id is None else str(row.old_id),
                     "—" if row.new_id is None else str(row.new_id), "", f"{row.old_images:,}" if row.old_id is not None
                     else "", f"{row.new_images:,}"]
            for c, t in enumerate(cells):
                it = QTableWidgetItem(t)
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if row.old_id is not None and c in (1, 2):
                    it.setToolTip("Locked: the model's own classes keep their ids")
                self.table.setItem(r, c, it)
            if row.source and row.old_id is None or (row.source and row.action in (X.MERGE, X.DROP)):
                self.table.setCellWidget(r, 3, self._action_box(row))
            else:
                word = {X.KEEP: "kept (its id is locked)" + (" — more examples from the new images"
                                                               if row.new_images else "")}.get(row.action, row.note)
                self.table.setItem(r, 3, QTableWidgetItem(word))
        # completeness: the model's classes in the new labels
        have = {X.norm(lab) for lab in self.new_labels if lm.get(lab) is not None and lm[lab] < len(self.base_names)}
        missing = [n for n in self.base_names if X.norm(n) not in have]
        tip = (f"The new labels have {len(self.base_names) - len(missing)} of the model's {len(self.base_names)} "
               "class(es)")
        if missing and X.is_cls_folder(self.new_yaml):
            tip += (" — fine for a classifier: the old data keeps teaching it " + ", ".join(missing[:8]) +
                    (" …" if len(missing) > 8 else ""))
        elif missing:
            tip += (" — the others (" + ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "") + ") are " +
                    ("pre-labelled by the model on the new images." if self.prelabel.isChecked() else
                     f"<span style='color:{utils.hexc('warn')}'>NOT labelled on the new images: the model learns "
                     "them as background there — turn pre-labels on</span>"))
        self.complete.setText(tip)
        P = self.plan()
        if getattr(self, "_est_key", None) != (P.old_yaml, P.new_yaml):
            self._est, self._est_key = X.estimate(P), (P.old_yaml, P.new_yaml)
        est = self._est
        n_train = est["old_train"] + est["new_train"]
        L = [f"Training copy: {est['old_train']:,} old + {est['new_train']:,} new training images · validation "
             f"{est['old_val']:,} old (kept apart: the before / after check) + {est['new_val']:,} new · hard links "
             f"(no extra disk space; about {est['bytes'] / 2**30:.1f} GB if they must be copied)"]
        t = X.time_estimate(self.info, n_train)
        if t:
            L.append(f"Speed: {t} (from the model's own run)")
        if P.new_classes:
            L.append(f"New classes: {', '.join(P.new_classes)} — ids {len(P.base_names)}…{len(P.names) - 1}; the "
                     "scoring head is enlarged, the old classes' part copied (it starts detecting exactly what it did)")
        if not P.old_yaml:
            L.append(f"<span style='color:{utils.hexc('warn')}'>Without the old training data the model sees only "
                     "the new images — it may forget the old classes (pre-labels help, the old data helps most)."
                     "</span>")
        self.summary.setText("<br>".join(L))
        self.ok.setEnabled(bool(self.base_names) and bool(P.new_classes or P.goal == "images"))

    def _action_box(self, row):
        cb = QComboBox()
        cb.addItem("Add as a new class", ("",))
        for n in self.base_names:
            cb.addItem(f"Same as “{n}” (merge)", (X.MERGE, n))
        cb.addItem("Rename…", (X.RENAME,))
        cb.addItem("Leave out", (X.DROP,))
        cur = self.actions.get(row.source)
        if cur:
            i = next((k for k in range(cb.count()) if cb.itemData(k)[0] == cur[0] and
                      (len(cur) < 2 or len(cb.itemData(k)) < 2 or cb.itemData(k)[1] == cur[1])), 0)
            cb.setCurrentIndex(i)
        cb.currentIndexChanged.connect(lambda _i, lab=row.source, cb=cb: self._act(lab, cb.currentData()))
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(cb)
        return w

    def _act(self, lab, data):
        kind = data[0] if data else ""
        if not kind:
            self.actions.pop(lab, None)
        elif kind == X.RENAME:
            name, ok = QInputDialog.getText(self, self.TITLE, f"New name for “{lab}”:", text=lab)
            if ok and name.strip():
                self.actions[lab] = (X.RENAME, name.strip())
        else:
            self.actions[lab] = tuple(data)
        self._refresh()

    def _accept(self):
        self.result_plan = self.plan()
        self.accept()
