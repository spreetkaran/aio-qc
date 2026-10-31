# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
✂️ Remove / re-order classes (GUI version of remove_class.py).

1. tick classes to remove        2. (YOLO) drag remaining classes into the new id order
3. 🔍 Preview plan  (full pre-scan, nothing written)   4. ✂️ Apply (optional backup) -> post-scan verify
"""
from __future__ import annotations

from AIO_QC.core.dataset import DETECTION_TYPES

from collections import Counter

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QDialog, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QMessageBox, QProgressDialog, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout,
)

from AIO_QC.utils import hexc, newButton
from AIO_QC.core.class_ops import apply_plan, plan_removal, verify_after


class ClassRemovalDialog(QDialog):
    SUPPORTED = DETECTION_TYPES          # needs shapes (not for classification datasets)
    TITLE, ICON = "Remove / Re-order Classes", "✂️"
    SUPPORTED = None

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.adapter = self.session.base_adapter            # always the REAL labels
        self.yolo = self.session.is_yolo
        self.plan = None
        self.setWindowTitle(self.TITLE)
        self.resize(int(app.width() * 0.6), int(app.height() * 0.75))

        counts: Counter = self.session.label_totals(raw=True)
        files: Counter = self.session.raw_label_files()
        head = []
        if self.yolo:
            cm = self.adapter.class_map
            self.names = list(cm.names)
            src = cm.source.name if cm.source else "?"
            head.append(f"YOLO class map: <b>{src}</b> ({len(self.names)} classes). Removed ids are dropped from "
                        "every label file, remaining ids are renumbered to the order on the right, and "
                        f"<b>{src}</b> is rewritten.")
            if not self.names:
                head.append(f"<span style='color:{hexc('error')}'>No class map — load classes.txt first.</span>")
        else:
            self.names = sorted(counts, key=lambda l: (-counts[l], l))
            head.append("LabelMe: shapes with the ticked labels are removed from every JSON.")
        if self.session.label_map:
            head.append("🔀 A view mapping is active — this tool works on the <b>original</b> labels.")
        if not self.session.index_complete:
            head.append("<i>Label index still building — counts may be incomplete (the preview re-counts).</i>")
        info = QLabel("<br>".join(head))
        info.setWordWrap(True)

        # left: classes --------------------------------------------------------------
        self.table = QTableWidget(len(self.names), 4)
        self.table.setHorizontalHeaderLabels(["Remove", "ID" if self.yolo else "#", "Class", "Instances / files"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for r, name in enumerate(self.names):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(r, 0, chk)
            self.table.setItem(r, 1, QTableWidgetItem(str(r)))
            self.table.setItem(r, 2, QTableWidgetItem(name))
            self.table.setItem(r, 3, QTableWidgetItem(f"{counts.get(name, 0):,} / {files.get(name, 0):,}"))
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(lambda *_: self._sync_order())
        left = QGroupBox("1 · Classes to remove")
        QVBoxLayout(left).addWidget(self.table)

        # right: new order (YOLO) ------------------------------------------------------
        body = QHBoxLayout()
        body.addWidget(left, 3)
        if self.yolo:
            right = QGroupBox("2 · New ID order (drag to reorder)")
            rl = QVBoxLayout(right)
            self.order = QListWidget()
            self.order.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self.order.model().rowsMoved.connect(lambda *_: self._update_preview())
            rl.addWidget(self.order, 1)
            btns = QHBoxLayout()
            for text, d in (("▲", -1), ("▼", 1)):
                b = newButton(text, slot=lambda d=d: self._move(d))
                btns.addWidget(b)
            reset = newButton("Reset order", slot=lambda: self._sync_order(reset=True))
            btns.addWidget(reset)
            rl.addLayout(btns)
            self.id_preview = QLabel()
            self.id_preview.setTextFormat(Qt.TextFormat.RichText)
            self.id_preview.setWordWrap(True)
            rl.addWidget(self.id_preview)
            body.addWidget(right, 2)

        # options + report -------------------------------------------------------------
        self.backup = QCheckBox("Back up affected files to .aio_qc/backups/ first (recommended)")
        self.backup.setChecked(True)
        self.report = QTextEdit(readOnly=True)
        self.report.setPlaceholderText("3 · Run 🔍 Preview plan — a full dry run; nothing is written.")
        self.report.setMaximumHeight(170)
        self.preview_btn = newButton("🔍 Preview plan (dry run)", slot=lambda: self.preview())
        self.apply_btn = newButton("✂️ Apply", slot=lambda: self.apply(), tip="Enabled after a successful preview of the current selection")
        self.apply_btn.setEnabled(False)
        close = newButton("Close", slot=self.close)
        foot = QHBoxLayout()
        foot.addWidget(self.backup, 1)
        for b in (self.preview_btn, self.apply_btn, close):
            foot.addWidget(b)

        lay = QVBoxLayout(self)
        lay.addWidget(info)
        lay.addLayout(body, 1)
        lay.addWidget(self.report)
        lay.addLayout(foot)
        self._sync_order(reset=True)

    # ------------------------------------------------------------------ state
    def removed(self) -> list[str]:
        return [self.table.item(r, 2).text() for r in range(self.table.rowCount())
                if self.table.item(r, 0).checkState() == Qt.CheckState.Checked]

    def _sync_order(self, reset=False):
        self.plan = None
        self.apply_btn.setEnabled(False)
        if not self.yolo:
            return
        removed = set(self.removed())
        current = [] if reset else [self.order.item(i).text() for i in range(self.order.count())]
        keep = [n for n in current if n not in removed] + \
               [n for n in self.names if n not in removed and n not in current]
        self.order.clear()
        self.order.addItems(keep)
        self._update_preview()

    def _move(self, d):
        r = self.order.currentRow()
        if r < 0 or not 0 <= r + d < self.order.count():
            return
        it = self.order.takeItem(r)
        self.order.insertItem(r + d, it)
        self.order.setCurrentRow(r + d)
        self._update_preview()

    def new_order(self) -> list[str]:
        return [self.order.item(i).text() for i in range(self.order.count())] if self.yolo else []

    def _update_preview(self):
        self.plan = None
        self.apply_btn.setEnabled(False)
        if not self.yolo:
            return
        new = {n: i for i, n in enumerate(self.new_order())}
        lines = []
        for i, n in enumerate(self.names):
            if n not in new:
                lines.append(f"<span style='color:{hexc('error')}'>{i} {n} → removed</span>")
            elif new[n] != i:
                lines.append(f"<span style='color:{hexc('warn')}'>{i} {n} → <b>{new[n]}</b></span>")
            else:
                lines.append(f"{i} {n} → {i}")
        self.id_preview.setText("<br>".join(lines))

    # ---------------------------------------------------------------- actions
    def _progress(self, title):
        dlg = QProgressDialog(title, None, 0, 100, self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(300)

        def cb(i, n):
            if i % 20 == 0 or i == n:
                dlg.setMaximum(max(n, 1))
                dlg.setValue(i)
                QApplication.processEvents()
        return dlg, cb

    def preview(self):
        removed = self.removed()
        if not removed and not (self.yolo and self.new_order() != self.names):
            QMessageBox.information(self, self.TITLE, "Tick classes to remove (or change the YOLO id order).")
            return
        if self.yolo and len(removed) == len(self.names):
            QMessageBox.warning(self, self.TITLE, "At least one class must remain.")
            return
        remove = {self.names.index(n) for n in removed} if self.yolo else set(removed)
        dlg, cb = self._progress("Scanning dataset…")
        try:
            plan = plan_removal(self.adapter, self.session.samples, remove,
                                self.new_order() if self.yolo else None, progress=cb)
        except ValueError as exc:
            dlg.close()
            QMessageBox.warning(self, self.TITLE, str(exc))
            return
        dlg.close()
        lines = [f"<b>Dry run — nothing written.</b>",
                 f"Shapes / rows removed: <b>{plan.rows_removed:,}</b> in {len(plan.affected):,} file(s)"]
        if self.yolo:
            lines.append(f"Files with renumbered ids only: {len(plan.remapped_files - set(plan.affected)):,}")
            lines.append(f"Class map → nc={len(plan.new_names)}: " +
                         ", ".join(f"{i}:{n}" for i, n in enumerate(plan.new_names)))
            per = Counter()
            for c in plan.affected.values():
                per.update(c)
            lines.append("Removed per class: " + ", ".join(f"{self.names[k]} ×{v:,}" for k, v in per.items()))
        else:
            per = Counter()
            for c in plan.affected.values():
                per.update(c)
            lines.append("Removed per label: " + ", ".join(f"{k} ×{v:,}" for k, v in per.items()))
        lines.append(f"Files to write: <b>{len(plan.files_to_write):,}</b>")
        if plan.errors:
            lines.append(f"<span style='color:{hexc('error')}'><b>{len(plan.errors)} blocking problem(s):</b><br>" +
                         "<br>".join(plan.errors[:10]) + "</span>")
        self.report.setHtml("<br>".join(lines))
        self.plan = plan
        self.apply_btn.setEnabled(not plan.errors and bool(plan.files_to_write or self.yolo))

    def apply(self):
        plan = self.plan
        if plan is None:
            return
        backup = self.backup.isChecked()
        msg = (f"Modify {len(plan.files_to_write):,} annotation file(s) in place"
               + (" and rewrite the class map" if self.yolo else "") + "?\n\n"
               + ("A backup of every affected file is made first." if backup
                  else "⚠ NO backup will be made."))
        if QMessageBox.question(self, self.TITLE, msg) != QMessageBox.StandardButton.Yes:
            return
        dlg, cb = self._progress("Applying…")
        backup_dir, errors = apply_plan(self.adapter, self.session.samples, plan, backup, progress=cb)
        dlg.close()
        root = self.session.root
        dtype = self.session.type
        # class map changed -> reopen the dataset, then verify against the new state
        self.app.open_dataset(root, dtype)
        new_session = self.app.session
        problems = verify_after(new_session.base_adapter, new_session.samples, plan) if new_session else []
        text = [f"Modified {len(plan.files_to_write) - len(errors):,} file(s)."]
        if backup_dir:
            text.append(f"Backup: {backup_dir}")
        text.append("Post-scan: ✅ clean" if not problems else f"Post-scan: ❌ {len(problems)} problem(s)")
        all_errors = errors + problems
        if all_errors:
            self.app.log_errors([f"[class removal] {e}" for e in all_errors])
            text.append("\n".join(all_errors[:10]))
        (QMessageBox.warning if all_errors else QMessageBox.information)(self.app, self.TITLE, "\n".join(text))
        self.close()

    def closeEvent(self, event):
        self.app.forget_dialog(self)
        super().closeEvent(event)
