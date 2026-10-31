# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
🔀 Label mapping (view-time).

One row per ORIGINAL label: tick "Apply", type/pick the mapped label. The mapping changes what
the viewer, the labels list, stats and every QC check see — files are untouched unless
"Write to files…" is used explicitly.
"""
from __future__ import annotations

from collections import Counter

from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QLineEdit, QMessageBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from AIO_QC.utils import theme_color, labelValidator, newButton, newIcon
from AIO_QC import utils
from AIO_QC.core.checks import classify_labels
from AIO_QC.core.mapping import load_mapping_file

COL_APPLY, COL_LABEL, COL_INST, COL_FILES, COL_TARGET, COL_RESULT = range(6)


class LabelMapDialog(QDialog):
    """
    The ONE place where labels are renamed / merged:
      * view-only mapping (files untouched) — "Apply (view only)"
      * permanent relabel — "Write to files" with a scope (current file / file list / all files)
    Other tools hand over here: 📝 Label Ambiguities (pre-filled suggestions), label-list context menus, Ctrl+R.
    """

    def __init__(self, app, prefill: dict | None = None, focus: str | None = None, scope: int = 2):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.setWindowTitle("Label Mapping")
        self.setWindowIcon(newIcon("tool-label-map"))
        self.resize(int(app.width() * 0.55), int(app.height() * 0.7))

        self.inst: Counter = self.session.label_totals(raw=True)
        self.files: Counter = self.session.raw_label_files()
        for name in self.session.base_adapter.class_names:      # YOLO classes with 0 instances
            self.inst.setdefault(name, 0)
        current = dict(self.session.label_map)
        self.targets = sorted(set(self.inst) | set(current.values()) | set(self.session.user_classes or []),
                              key=str.lower)

        info = QLabel("Map existing labels to the names you want to <b>view and check</b> with — e.g. "
                      "<code>license_plate → hsrp_white</code>. Several labels can map to one. "
                      "Files are <b>not</b> modified. Ticked rows are active; a row ticks itself when you "
                      "type a new name.")
        info.setWordWrap(True)
        if not self.session.index_complete:
            info.setText(info.text() + "<br><i>Label index still building — counts may be incomplete.</i>")

        # toolbar -----------------------------------------------------------------
        self.search = QLineEdit(placeholderText="Filter labels…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        bar = QHBoxLayout()
        bar.addWidget(self.search, 1)
        for text, slot, tip in (
            ("✨ Suggest", self.suggest, "Fill unmapped rows with format variants / close matches of the class list"),
            ("⇉ Set selected…", self.set_selected, "Map every selected row to one name"),
            ("📥 Import…", self.import_file, "JSON {old: new}, a config.py with CLASS_MAP, or 'old -> new' lines"),
            ("📤 Export…", self.export_file, "Save the ticked mapping as JSON"),
            ("↺ Clear all", self.clear_all, "Untick and empty every row"),
        ):
            b = newButton(text, slot=slot, tip=tip)
            bar.addWidget(b)

        # table -------------------------------------------------------------------
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Apply", "Existing label", "Instances", "Files", "→ Mapped label",
                                              "Result"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_TARGET, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        self._building = True
        for label in sorted(self.inst, key=lambda l: (-self.inst[l], l.lower())):
            self._add_row(label, current.get(label, ""))
        self._building = False

        # footer ------------------------------------------------------------------
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        foot = QHBoxLayout()
        foot.addWidget(self.summary, 1)
        self.scope = QComboBox()
        self.scope.addItems(["current file", "files in current list", "all files"])
        self.scope.setCurrentIndex(max(0, min(2, scope)))
        self.scope.setToolTip("Where 'Write to files' applies")
        bake = newButton("💾 Write to files…", slot=lambda: self.write_to_files(), tip="Permanently relabel the ticked rows in the annotation files (scope on the left)")
        disable = newButton("Disable mapping", slot=lambda: self._apply({}))
        cancel = newButton("Cancel", slot=self.reject)
        ok = newButton("✓ Apply (view only)", slot=lambda: self._apply(self.mapping()))
        ok.setDefault(True)
        foot.addWidget(QLabel("Write scope:"))
        foot.addWidget(self.scope)
        for b in (bake, disable, cancel, ok):
            foot.addWidget(b)

        lay = QVBoxLayout(self)
        lay.addWidget(info)
        lay.addLayout(bar)
        lay.addWidget(self.table, 1)
        lay.addLayout(foot)
        for src, dst in (prefill or {}).items():
            r = self._row_of(src)
            if r >= 0:
                self._set(r, dst)
        if prefill:
            info.setText(info.text() + f"<br><b>{len(prefill)} suggestion(s) pre-filled from 📝 Label Ambiguities "
                                       "— review, then apply or write.</b>")
        self._update_summary()
        if focus is not None:
            r = self._row_of(focus)
            if r >= 0:
                self.table.selectRow(r)
                self.table.scrollToItem(self.table.item(r, COL_LABEL))
                self.table.cellWidget(r, COL_TARGET).setFocus()

    # ------------------------------------------------------------------ rows
    def _add_row(self, label: str, target: str):
        r = self.table.rowCount()
        self.table.insertRow(r)
        chk = QTableWidgetItem()
        chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        chk.setCheckState(Qt.CheckState.Checked if target else Qt.CheckState.Unchecked)
        self.table.setItem(r, COL_APPLY, chk)
        name = QTableWidgetItem(label if label == label.strip() else repr(label))
        name.setData(Qt.ItemDataRole.UserRole, label)
        utils.update_label_colors(label, self.app.label_colors)
        name.setIcon(utils.get_color_icon(self.app.label_colors[label]))
        name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.table.setItem(r, COL_LABEL, name)
        for col, v in ((COL_INST, self.inst[label]), (COL_FILES, self.files[label])):
            it = QTableWidgetItem()
            it.setData(Qt.ItemDataRole.DisplayRole, int(v))
            it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(r, col, it)
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems([""] + [t for t in self.targets if t != label])
        combo.setCurrentText(target)
        combo.lineEdit().setPlaceholderText("keep as is")
        combo.setValidator(labelValidator())          # no leading whitespace → no ' plate' vs 'plate' variants
        combo.currentTextChanged.connect(lambda text, row_label=label: self._on_target_changed(row_label, text))
        self.table.setCellWidget(r, COL_TARGET, combo)
        res = QTableWidgetItem()
        res.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.table.setItem(r, COL_RESULT, res)

    def _row_of(self, label):
        for r in range(self.table.rowCount()):
            if self.table.item(r, COL_LABEL).data(Qt.ItemDataRole.UserRole) == label:
                return r
        return -1

    def _set(self, r, target: str, check: bool | None = None):
        combo = self.table.cellWidget(r, COL_TARGET)
        combo.blockSignals(True)
        combo.setCurrentText(target)
        combo.blockSignals(False)
        if check is None:
            check = bool(target.strip())
        self.table.item(r, COL_APPLY).setCheckState(Qt.CheckState.Checked if check else Qt.CheckState.Unchecked)
        self._update_summary()

    def _on_target_changed(self, label, text):
        r = self._row_of(label)
        if r >= 0:
            want = bool(text.strip()) and text.strip() != label
            self.table.item(r, COL_APPLY).setCheckState(Qt.CheckState.Checked if want else Qt.CheckState.Unchecked)
            self._update_summary()

    def _on_item_changed(self, item):
        if not self._building and item.column() == COL_APPLY:
            self._update_summary()

    def mapping(self) -> dict[str, str]:
        out = {}
        for r in range(self.table.rowCount()):
            if self.table.item(r, COL_APPLY).checkState() != Qt.CheckState.Checked:
                continue
            src = self.table.item(r, COL_LABEL).data(Qt.ItemDataRole.UserRole)
            dst = self.table.cellWidget(r, COL_TARGET).currentText().strip()
            if dst and dst != src:
                out[src] = dst
        return out

    def _update_summary(self):
        m = self.mapping()
        result = Counter()
        for label, n in self.inst.items():
            result[m.get(label, label)] += n
        sources = Counter(m.get(l, l) for l in self.inst)
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            src = self.table.item(r, COL_LABEL).data(Qt.ItemDataRole.UserRole)
            dst = m.get(src)
            item = self.table.item(r, COL_RESULT)
            if dst is None:
                item.setText("unchanged")
                item.setForeground(theme_color("faint"))
            else:
                merged = sources[dst] > 1
                new = dst not in self.inst
                item.setText(f"→ {dst} · {result[dst]:,} inst." + (" · merged" if merged else "") +
                             (" · new name" if new else ""))
                item.setForeground(theme_color("warn") if merged else theme_color("ok"))
        self.table.blockSignals(False)
        n_before = len([l for l in self.inst if self.inst[l]])
        n_after = len({m.get(l, l) for l in self.inst if self.inst[l]})
        self.summary.setText(f"<b>{len(m)}</b> label(s) mapped · {n_before} → <b>{n_after}</b> classes in view")

    def _filter(self, text):
        t = text.strip().lower()
        for r in range(self.table.rowCount()):
            src = self.table.item(r, COL_LABEL).data(Qt.ItemDataRole.UserRole).lower()
            dst = self.table.cellWidget(r, COL_TARGET).currentText().lower()
            self.table.setRowHidden(r, bool(t) and t not in src and t not in dst)

    # --------------------------------------------------------------- actions
    def suggest(self):
        truth = self.session.user_classes if not self.session.is_yolo else None
        findings = classify_labels(Counter({k: v for k, v in self.inst.items() if v}), truth)
        n = 0
        for f in findings:
            if f.kind == "unknown" or not f.suggestion:
                continue
            r = self._row_of(f.label)
            if r >= 0 and not self.table.cellWidget(r, COL_TARGET).currentText().strip():
                self._set(r, f.suggestion)
                n += 1
        ref = "the loaded class list" if truth else "more frequent labels"
        QMessageBox.information(self, "Suggest", f"Filled {n} row(s) from {ref}. Review them before applying."
                                if n else "No confident suggestions found.")

    def set_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            QMessageBox.information(self, "Set selected", "Select one or more rows first.")
            return
        target, ok = QInputDialog.getItem(self, "Map selected", f"Map {len(rows)} label(s) to:",
                                          self.targets, 0, True)
        if ok and target.strip():
            for r in rows:
                self._set(r, target.strip())

    def clear_all(self):
        for r in range(self.table.rowCount()):
            self._set(r, "", False)

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import mapping", str(self.session.root),
                                              "Mappings (*.json *.py *.txt *.csv);;All files (*)")
        if not path:
            return
        try:
            m = load_mapping_file(path)
        except Exception as exc:
            QMessageBox.warning(self, "Import", str(exc))
            return
        hit = 0
        for src, dst in m.items():
            r = self._row_of(src)
            if r >= 0:
                self._set(r, dst)
                hit += 1
        QMessageBox.information(self, "Import", f"{hit} of {len(m)} entries matched labels in this dataset.")

    def export_file(self):
        import json
        path, _ = QFileDialog.getSaveFileName(self, "Export mapping", str(self.session.root / "label_map.json"),
                                              "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.mapping(), f, indent=2, ensure_ascii=False)

    def write_to_files(self):
        m = self.mapping()
        if not m:
            QMessageBox.information(self, "Write to files", "Tick at least one mapped row.")
            return
        base = self.session.base_adapter
        bad = [d for d in set(m.values()) if not base.can_use_label(d)]
        if bad:
            QMessageBox.warning(self, "Write to files", "These targets are not valid for this dataset "
                                "(YOLO: must exist in the class map):\n" + "\n".join(bad))
            return
        scope = self.scope.currentIndex()
        if scope == 0:
            pool = [self.app.current_key] if self.app.current_key else []
        elif scope == 1:
            pool = self.app.file_dock.visible_keys()
        else:
            pool = list(self.session.raw_index)
        idx = self.session.raw_index
        keys = [k for k in pool if set(idx.get(k, {})) & set(m)]
        n_shapes = sum(idx[k][s] for k in keys for s in m if s in idx.get(k, {}))
        if not keys:
            QMessageBox.information(self, "Write to files", f"No file in scope ({self.scope.currentText()}) uses these labels.")
            return
        if QMessageBox.question(self, "Write to files",
                                f"Permanently relabel {n_shapes:,} shape(s) in {len(keys):,} file(s) "
                                f"({self.scope.currentText()})?\nThis edits the annotation files.") \
                != QMessageBox.StandardButton.Yes:
            return
        mapping = dict(m)

        def fn(ann):
            hit = False
            for s in ann.shapes:
                new = mapping.get(s.get("label"))
                if new is not None:
                    s["label"] = new
                    hit = True
            return hit
        changed, errors = self.session.modify_many(self, [(k, fn) for k in keys], "Writing labels...", raw=True)
        if errors:
            self.app.log_errors(errors)
        if scope == 2:   # every file relabelled: those entries no longer need a view mapping
            self.app.set_label_map({k: v for k, v in self.session.label_map.items() if k not in mapping})
        QMessageBox.information(self, "Write to files", f"Relabelled {len(changed):,} file(s)."
                                + (f"\n{len(errors)} error(s) — see Logs" if errors else ""))
        self.accept()

    def _apply(self, mapping):
        self.app.set_label_map(mapping)
        self.accept()
