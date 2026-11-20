# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Classes (QC ▸ Labels ▸ Remove / Re-order Classes, for classification datasets): every class with its images per
split, and what to do with it —

  · Rename …            LabelMe flags: the flag is renamed in every file · class folders: the folder is renamed
  · Merge into …        its images get the other class (folders: moved into the other folder)
  · Remove              LabelMe flags: the flag is removed from every file (images left without a class are listed
                        by Classification Check) · class folders: its images go to .aio_qc_trash/<class> (kept)
  · Move up / down      your class order — 1 … 9 and the Change class list follow it; exports use it for ids

Nothing is written until Apply; then it is one step of Undo (Ctrl+Z) — removed folders come back from the trash.
"""
from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMessageBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from AIO_QC.utils import newButton

KEEP, RENAME, MERGE, REMOVE = "keep", "rename", "merge", "remove"


class ClsClassesDialog(QDialog):
    TITLE = "Classes"

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.session = s = app.session
        self.setWindowTitle(f"{self.TITLE} — {s.type.title}")
        self.resize(760, 520)
        self.order = list(s.cls_classes())
        self.plan: dict[str, tuple] = {}             # class -> (action, target)
        totals = s.label_totals()
        self.counts = {c: totals.get(c, 0) for c in self.order}
        self.split_counts = {}
        for k, c in s.index.items():
            sp = s.by_key[k].split if k in s.by_key else ""
            for t in c:
                self.split_counts.setdefault(t, {}).setdefault(sp, 0)
                self.split_counts[t][sp] += 1
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Rename, merge or remove classes, and set their order. Nothing is written until "
                             "Apply — then one Ctrl+Z undoes it."))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "Class", "Images", "Change"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, slot in (("Rename…", self.rename), ("Merge into…", self.merge), ("Remove", self.remove),
                           ("Keep as is", self.keep), ("▲ Up", lambda: self.move(-1)), ("▼ Down", lambda: self.move(1)),
                           ("A → Z", self.sort_az), ("Most images first", self.sort_count)):
            row.addWidget(newButton(text, slot=slot))
        row.addStretch(1)
        lay.addLayout(row)
        self.summary = QLabel("")
        lay.addWidget(self.summary)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close)
        self.apply_btn = bb.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_btn.clicked.connect(self.apply)
        bb.rejected.connect(self.close)
        lay.addWidget(bb)
        self._fill()

    # ------------------------------------------------------------------ view
    def _fill(self, select: str | None = None):
        self.table.setRowCount(len(self.order))
        for r, c in enumerate(self.order):
            sp = self.split_counts.get(c, {})
            per = " · ".join(f"{k or 'all'} {v:,}" for k, v in sorted(sp.items())) if len(sp) > 1 else ""
            act, tgt = self.plan.get(c, (KEEP, ""))
            what = {KEEP: "", RENAME: f"→ rename to “{tgt}”", MERGE: f"→ merge into “{tgt}”",
                    REMOVE: "→ remove" + (" (images to .aio_qc_trash)" if self._folders else " (the flag)")}[act]
            cells = [str(r + 1) if r < 9 else "", c, f"{self.counts.get(c, 0):,}" + (f"  ({per})" if per else ""), what]
            for i, t in enumerate(cells):
                it = QTableWidgetItem(t)
                if i == 1:
                    it.setData(Qt.ItemDataRole.UserRole, c)
                self.table.setItem(r, i, it)
        if select in self.order:
            self.table.selectRow(self.order.index(select))
        n = len([1 for a, _t in self.plan.values() if a != KEEP])
        self.summary.setText(f"{n} change(s) planned" if n else "No changes planned — the order is saved on Apply")

    @property
    def _folders(self) -> bool:
        return self.session.type.value == "cls_folders"

    def _selected(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.order[r] for r in rows if r < len(self.order)]

    # ------------------------------------------------------------------ plan
    def rename(self):
        sel = self._selected()
        if len(sel) != 1:
            QMessageBox.information(self, self.TITLE, "Select one class to rename.")
            return
        c = sel[0]
        name, ok = QInputDialog.getText(self, self.TITLE, f"New name for “{c}”:", text=c)
        name = (name or "").strip()
        if not ok or not name or name == c:
            return
        if not self.session.base_adapter.can_use_label(name):
            QMessageBox.warning(self, self.TITLE, f"“{name}” cannot be a class name (no / \\ : * ? \" < > |).")
            return
        if name in self.order:
            self.plan[c] = (MERGE, name)            # renaming onto an existing class is a merge
        else:
            self.plan[c] = (RENAME, name)
        self._fill(c)

    def merge(self):
        sel = self._selected()
        if not sel:
            return
        others = [x for x in self.order if x not in sel]
        if not others:
            return
        tgt, ok = QInputDialog.getItem(self, self.TITLE, f"Merge {', '.join(sel)} into:", others, 0, False)
        if ok and tgt:
            for c in sel:
                self.plan[c] = (MERGE, tgt)
            self._fill(sel[0])

    def remove(self):
        for c in self._selected():
            self.plan[c] = (REMOVE, "")
        self._fill()

    def keep(self):
        for c in self._selected():
            self.plan.pop(c, None)
        self._fill()

    def move(self, d):
        sel = self._selected()
        if len(sel) != 1:
            return
        i = self.order.index(sel[0])
        j = max(0, min(len(self.order) - 1, i + d))
        self.order.insert(j, self.order.pop(i))
        self._fill(sel[0])

    def sort_az(self):
        self.order.sort(key=lambda n: (n.lower(), n))
        self._fill()

    def sort_count(self):
        self.order.sort(key=lambda n: (-self.counts.get(n, 0), n.lower()))
        self._fill()

    # ------------------------------------------------------------------ apply
    def apply(self):
        s = self.session
        changes = {c: v for c, v in self.plan.items() if v[0] != KEEP}
        final_order = []
        for c in self.order:
            act, tgt = changes.get(c, (KEEP, ""))
            name = tgt if act in (RENAME, MERGE) else (None if act == REMOVE else c)
            if name and name not in final_order:
                final_order.append(name)
        if changes:
            lines = [f"{c}: {a}" + (f" → {t}" if t else "") for c, (a, t) in changes.items()]
            n = sum(self.counts.get(c, 0) for c in changes)
            if QMessageBox.question(self, self.TITLE, f"Change {len(changes)} class(es) — {n:,} image(s)?\n\n" +
                                    "\n".join(lines[:15])) != QMessageBox.StandardButton.Yes:
                return
            done, errors = apply_class_changes(s, changes)
            if errors:
                self.app.log_errors([f"[classes] {e}" for e in errors])
            QMessageBox.information(self, self.TITLE, f"{len(done):,} image(s) changed (Ctrl+Z undoes it)."
                                    + (f"\n{len(errors)} could not be — see the log." if errors else ""))
        s.save_cls_settings(order=final_order)
        self.app.refresh_label_list()
        self.close()

    def closeEvent(self, e):
        try:
            self.app.forget_dialog(self)
        except Exception:
            pass
        super().closeEvent(e)


def apply_class_changes(s, changes: dict) -> tuple[list[str], list[str]]:
    """{class: (rename | merge | remove, target)} applied to a classification session — one undo step per class.
    Folders: renamed / merged by moving the images (the empty folder is removed); removed images go to
    .aio_qc_trash/<class>/… (Ctrl+Z moves them back)."""
    done, errors = [], []
    folders = s.type.value == "cls_folders"
    for c, (act, tgt) in changes.items():
        keys = [k for k in list(s.by_key) if c in s.tags_of(k)]
        if not keys and not folders:                  # a flag no image has set: still removed / renamed in files
            keys = [k for k, smp in s.by_key.items() if c in (s.base_adapter.flags_of(smp) or {})]
        if not keys:
            continue
        if act in (RENAME, MERGE):
            if folders:
                d, e = s.set_tags(keys, tags=[tgt], what=f"{c} → {tgt}")
            else:
                d, e = s.set_tags(keys, add=[tgt], remove=[c], drop=[c], what=f"{c} → {tgt}")
        else:
            if folders:
                with s.as_tool("Classes"):
                    d, e = s.trash_samples(keys, include_images=True)     # Undo bar / Ctrl+Z bring them back
            else:
                d, e = s.set_tags(keys, remove=[c], drop=[c], what=f"remove {c}")
        done += d
        errors += e
        if folders:
            _remove_empty_class_dirs(s, c)
    return done, errors


def _remove_empty_class_dirs(s, cls: str):
    for _split, base in getattr(s.base_adapter, "bases", []):
        d = Path(base) / cls
        if d.is_dir():
            for dp, _dn, _fn in sorted(os.walk(d), key=lambda x: -len(x[0])):
                try:
                    os.rmdir(dp)                          # only empty folders
                except OSError:
                    pass
