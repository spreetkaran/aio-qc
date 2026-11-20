# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Testing (Model Lab) — step 2, the models.

A mixin of AIO_QC.widgets.testing.model_lab.ModelLab: the methods use the window's own attributes, set up in its __init__.
"""
import os
import time
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QFormLayout, QLabel, QPushButton, QTableWidgetItem, QAbstractItemView, QComboBox, QSpinBox, QGroupBox, QLineEdit, QFileDialog, QMessageBox, QMenu,
)
from AIO_QC.tools.frame_extractor.core.common import WEIGHTS_FILTER
from AIO_QC.core.diag import ignored as _ignored

from AIO_QC.widgets.testing.model_lab_common import (_item, _make_table, _set_shared_env, _shared_env, _swatch_icon, slot_hex)


class ModelsMixin:

    # ------------------------------------------------------------------ 2 models
    def _page_models(self):
        w, lay = self._page("Models", step=1, subtitle="Add the weights to test - two versions of the same model, different sizes, "
                            "or completely different models. Each keeps its colour in every chart, image and report.")
        self.m_table = _make_table(["", "Name", "Weights", "Runs on", "Image size", "FP16"], stretch_col=2)
        self.m_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                     QAbstractItemView.EditTrigger.EditKeyPressed)
        self.m_table.setToolTip("Double-click a name or image size to change it (0 = the size the model was "
                                "trained at).")
        self.m_table.itemChanged.connect(self._model_edited)
        lay.addWidget(self.m_table, 1)
        row = QHBoxLayout()
        for text, fn, tip in (("Add weights…", self._add_weights, ".pt / .onnx / .engine - several at once (also added "
                                                                  "to AIO QC's models)"),
                              ("Add AIO QC's models ▾", self._add_app_models, "The models of the open dataset "
                               "(Models & Environment) and every model added anywhere in AIO QC"),
                              ("Remove", self._remove_model, None), ("↑", lambda: self._move_model(-1), "Move up"),
                              ("↓", lambda: self._move_model(1), "Move down")):
            b = QPushButton(text)
            if tip:
                b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        box = QGroupBox("How .pt / .engine models run")
        f = QFormLayout(box)
        env_row = QHBoxLayout()
        self.env_edit = QLineEdit(_shared_env() or self.st.value("env_python", "") or self.model_cfg.get("env_python", ""))
        self.env_edit.editingFinished.connect(lambda: (_set_shared_env(self.env_edit.text().strip()),
                                                       self._sync_specs(), self._refresh_mapping_hint()))
        self.env_edit.setPlaceholderText("Python with ultralytics (the environment of Models & Environment)")
        self.env_edit.setToolTip(".onnx runs inside the app. .pt / .engine run in this Python environment "
                                 "(conda / venv with ultralytics + torch), on its GPU when it has one.")
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_env)
        env_row.addWidget(self.env_edit, 1)
        env_row.addWidget(b)
        f.addRow("Environment", env_row)
        self.device_edit = QComboBox()
        self.device_edit.setEditable(True)
        for d in ("auto", "cpu", "0", "1"):
            self.device_edit.addItem(d)
        self.device_edit.setCurrentText(str(self.st.value("device", "") or self.model_cfg.get("device", "auto")))
        self.device_edit.currentTextChanged.connect(lambda t: self.st.setValue("device", t.strip()))
        self.device_edit.setToolTip("auto = GPU when available. 0, 1 = CUDA device index.")
        f.addRow("Device", self.device_edit)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(int(self.st.value("batch", 8)))
        self.batch_spin.setToolTip("Images per forward pass. Larger is faster on a GPU (until memory runs out).")
        f.addRow("Batch size", self.batch_spin)
        lay.addWidget(box)
        self.map_lbl = QLabel("")
        self.map_lbl.setWordWrap(True)
        self.map_lbl.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.map_lbl)
        row = QHBoxLayout()
        self.check_btn = QPushButton("Check models")
        self.check_btn.setToolTip("Load every model once: backend, classes, and how its classes match the test data.")
        self.check_btn.clicked.connect(self._check_models)
        row.addWidget(self.check_btn)
        row.addStretch(1)
        nxt = QPushButton("Next: run  →")
        nxt.clicked.connect(lambda: self.nav.setCurrentRow(2))
        row.addWidget(nxt)
        lay.addLayout(row)
        return w

    def _browse_env(self):
        p, _ = QFileDialog.getOpenFileName(self, "Python executable of the environment", "",
                                           "Python (python python.exe python3);;All files (*)")
        if p:
            self.env_edit.setText(p)
            _set_shared_env(p)
            self._sync_specs()
            self._refresh_mapping_hint()

    def _add_weights(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Model weights", self.st.value("weights_dir", ""),
                                                WEIGHTS_FILTER)
        if paths:
            self.st.setValue("weights_dir", os.path.dirname(paths[0]))
            from AIO_QC.core import model_registry as R
            root = self.data.root if self.data is not None and self.data.qc else None
            rows = []
            for p in paths:
                e = R.register(p, root)                 # one registry: Image QC, the Frame Extractor, Testing
                rows.append(dict(path=p, label=e.get("name", ""), class_map=e.get("class_map") or {}))
            self._add_models(rows)

    def _registry_models(self):
        from AIO_QC.core import model_registry as R
        root = self.data.root if self.data is not None and self.data.qc else \
            (str(self.qc.session.root) if self.qc is not None and self.qc.session is not None else None)
        have = {os.path.normcase(os.path.abspath(self.m_table.item(r, 2).text())) for r in range(self.m_table.rowCount())}
        return [m for m in R.all_models(root) if os.path.isfile(m.get("path", "")) and
                os.path.normcase(os.path.abspath(m["path"])) not in have]

    def _add_app_models(self):
        ms = self._registry_models()
        if not ms:
            QMessageBox.information(self, "Testing", "Every model AIO QC knows is already in the list (add models in "
                                    "Image QC ▸ Models & Environment, or with Add weights…).")
            return
        m = QMenu(self)
        allb = m.addAction(f"All {len(ms)} models")
        m.addSeparator()
        for e in ms:
            a = m.addAction(f"{e.get('name') or os.path.basename(e['path'])}   ({'this dataset' if e.get('source') == 'dataset' else 'AIO QC'})")
            a.setData(e["path"])
        act = m.exec(self.m_table.mapToGlobal(self.m_table.rect().bottomLeft()))
        if act is None:
            return
        pick = ms if act is allb else [e for e in ms if e["path"] == act.data()]
        from AIO_QC.core import model_registry as R
        self._add_models([dict(path=e["path"], label=e.get("name", ""), imgsz=R.effective_imgsz(e),
                               class_map=e.get("class_map") or {}) for e in pick])

    def _add_models(self, dicts):
        have = {self.m_table.item(r, 2).text() for r in range(self.m_table.rowCount())}
        used = {self.m_table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in range(self.m_table.rowCount())}
        self.m_table.blockSignals(True)
        for d in dicts:
            p = os.path.abspath(d["path"])
            if p in have:
                continue
            slot = d.get("slot")
            if slot is None or slot in used:
                slot = next(s for s in range(64) if s not in used)
            used.add(slot)
            r = self.m_table.rowCount()
            self.m_table.insertRow(r)
            sw = _item("")
            sw.setIcon(_swatch_icon(slot_hex(slot), 14))
            sw.setData(Qt.ItemDataRole.UserRole, slot)
            self.m_table.setItem(r, 0, sw)
            name = QTableWidgetItem(d.get("label") or os.path.splitext(os.path.basename(p))[0])
            name.setData(Qt.ItemDataRole.UserRole, dict(d.get("class_map") or {}))   # AIO QC's class map
            if d.get("class_map"):
                name.setToolTip("Classes mapped to this dataset's labels (Image QC ▸ Models & Environment): " +
                                ", ".join(f"{k}→{v or 'ignored'}" for k, v in list(d["class_map"].items())[:12]))
            self.m_table.setItem(r, 1, name)
            self.m_table.setItem(r, 2, _item(p, tip=p))
            ext = os.path.splitext(p)[1].lower()
            self.m_table.setItem(r, 3, _item("in app (ONNX Runtime)" if ext == ".onnx" else "environment"))
            sz = QTableWidgetItem(str(d.get("imgsz", 0)))
            sz.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.m_table.setItem(r, 4, sz)
            half = QTableWidgetItem("")
            half.setFlags((half.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable)
            half.setCheckState(Qt.CheckState.Checked if d.get("half") else Qt.CheckState.Unchecked)
            half.setToolTip("Half precision on the GPU (.pt / .engine): faster, tiny accuracy change")
            self.m_table.setItem(r, 5, half)
            have.add(p)
        self.m_table.blockSignals(False)
        self._sync_specs()
        self._save_models()
        self._refresh_mapping_hint()

    def _remove_model(self):
        for r in sorted({i.row() for i in self.m_table.selectedIndexes()}, reverse=True):
            self.m_table.removeRow(r)
        self._sync_specs()
        self._save_models()

    def _move_model(self, d):
        rows = sorted({i.row() for i in self.m_table.selectedIndexes()})
        if len(rows) != 1:
            return
        r = rows[0]
        t = r + d
        if not 0 <= t < self.m_table.rowCount():
            return
        self.m_table.blockSignals(True)
        for c in range(self.m_table.columnCount()):
            a, b = self.m_table.takeItem(r, c), self.m_table.takeItem(t, c)
            self.m_table.setItem(r, c, b)
            self.m_table.setItem(t, c, a)
        self.m_table.blockSignals(False)
        self.m_table.selectRow(t)
        self._sync_specs()
        self._save_models()

    def _model_edited(self, _item):
        self._sync_specs()
        self._save_models()

    def _sync_specs(self):
        """Table -> ModelSpec list (+ colour slots); labels made unique."""
        from AIO_QC.core.testing import pipeline as P
        cfg = dict(self.model_cfg)
        cfg.update(env_python=self.env_edit.text().strip() if hasattr(self, "env_edit") else cfg.get("env_python", ""),
                   device=self.device_edit.currentText().strip() if hasattr(self, "device_edit") else "auto")
        specs, slots = [], []
        for r in range(self.m_table.rowCount()):
            try:
                imgsz = int(float(self.m_table.item(r, 4).text() or 0))
            except ValueError:
                imgsz = 0
            half = self.m_table.item(r, 5).checkState() == Qt.CheckState.Checked
            label = self.m_table.item(r, 1).text().strip()
            cmap = self.m_table.item(r, 1).data(Qt.ItemDataRole.UserRole) or None
            specs.append(P.spec_from_cfg(self.m_table.item(r, 2).text(), cfg, label=label, imgsz=imgsz, half=half,
                                         class_map=dict(cmap) if cmap else None))
            slots.append(int(self.m_table.item(r, 0).data(Qt.ItemDataRole.UserRole)))
        P.unique_labels(specs)
        self.specs, self.slots = specs, slots
        return specs

    def _save_models(self):
        """The model list is part of the project (autosaved)."""
        self._changed()

    def _refresh_mapping_hint(self):
        n = self.m_table.rowCount()
        pt = sum(1 for r in range(n) if not self.m_table.item(r, 2).text().lower().endswith(".onnx"))
        env = self.env_edit.text().strip()
        msg = []
        if pt and not env:
            msg.append("<span style='color:#d03b3b'>.pt / .engine models need an environment with ultralytics "
                       "(set it above or in Settings → Models).</span>")
        if self.data is not None and self.data.dataset.names:
            msg.append(f"Test data classes: {', '.join(list(self.data.dataset.names.values())[:20])}"
                       f"{'…' if len(self.data.dataset.names) > 20 else ''}. Model classes are matched by name "
                       "(order does not matter); others are ignored - 'Check models' shows the match.")
        self.map_lbl.setText("<br>".join(msg))

    def _check_models(self):
        specs = self._sync_specs()
        if not specs:
            return
        names = dict(self.data.dataset.names) if self.data is not None else {}

        def work(job):
            from AIO_QC.core.testing import metrics as M
            out = []
            for i, s in enumerate(specs):
                if job.cancelled():
                    break
                job.progress.emit((i, len(specs), f"Loading {s.label}…"))
                t0 = time.perf_counter()
                try:
                    det = s.load()
                    mn = dict(getattr(det, "names", None) or {})
                    _, matched, un = M.build_class_map(mn, names) if names else ({}, [], [])
                    out.append((s, getattr(det, "provider", ""), time.perf_counter() - t0, mn, matched, un, None))
                except Exception as e:
                    _ignored(e)
                    out.append((s, "", 0, {}, [], [], f"{type(e).__name__}: {e}"))
            return out

        def done(res):
            from html import escape
            lines = []
            for s, backend, secs, mn, matched, un, err in res:
                if err:
                    lines.append(f"<b>{escape(s.label)}</b>: <span style='color:#d03b3b'>{escape(err)}</span>")
                    continue
                txt = f"<b>{escape(s.label)}</b>: {escape(str(backend))}, {len(mn)} classes, loaded in {secs:.1f} s"
                if names:
                    txt += f" · {len(matched)} match the test data"
                    if un:
                        txt += f" · ignored: {escape(', '.join(map(str, un[:10])))}{'…' if len(un) > 10 else ''}"
                    if not matched:
                        txt += " · <span style='color:#d03b3b'>no class names in common - check the dataset classes</span>"
                lines.append(txt)
            self.map_lbl.setText("<br>".join(lines))
        self._start(work, done, button=self.check_btn, busy="Stop")
