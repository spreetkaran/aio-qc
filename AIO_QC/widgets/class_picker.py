# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Change class (C) for classification datasets: type to search, ↑ ↓ and Enter, 1 … 9 for the first nine; a new
class can be created. One class per image: pick one. Multi-label: tick the classes the images should have — a
class only some of the selected images have shows partly ticked and stays as it is unless you click it.

    dlg = ClassPicker(parent, classes, current={"cat": 3, "dog": 1}, n=4, multi=False)
    if dlg.exec(): dlg.result -> ("set", ["cat"]) | ("change", add=[…], remove=[…])
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout,
)


class ClassPicker(QDialog):
    def __init__(self, parent, classes: list[str], current: dict | None = None, n: int = 1, multi: bool = False,
                 can_create: bool = True, counts: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Change class")
        self.multi, self.n, self.current = multi, max(1, n), dict(current or {})
        self.can_create = can_create
        self.result = None
        self.resize(380, 460)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"{'Classes' if multi else 'Class'} for {self.n:,} image(s)"
                             + (" — tick to add, untick to remove" if multi else "")))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Type to search" + (" — or a new class name" if can_create else ""))
        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._enter)
        lay.addWidget(self.search)
        self.list = QListWidget()
        self.list.itemActivated.connect(lambda _it: self._enter())
        lay.addWidget(self.list, 1)
        self.new_hint = QLabel("")
        lay.addWidget(self.new_hint)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._enter)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.classes = list(classes)
        counts = counts or {}
        for i, c in enumerate(self.classes):
            it = QListWidgetItem((f"{i + 1}  " if i < 9 else "    ") + c +
                                 (f"   ({counts[c]:,})" if counts.get(c) else ""))
            it.setData(Qt.ItemDataRole.UserRole, c)
            if multi:
                k = self.current.get(c, 0)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if k >= self.n else
                                 Qt.CheckState.PartiallyChecked if k else Qt.CheckState.Unchecked)
            self.list.addItem(it)
        if multi:
            self.list.itemClicked.connect(self._toggle)
        cur = max(self.current, key=self.current.get) if self.current else None
        if cur in self.classes:
            self.list.setCurrentRow(self.classes.index(cur))
        elif self.classes:
            self.list.setCurrentRow(0)
        self.search.setFocus()

    def _toggle(self, it):
        st = it.checkState()
        it.setCheckState(Qt.CheckState.Unchecked if st == Qt.CheckState.Checked else Qt.CheckState.Checked)

    def keyPressEvent(self, e):
        t = e.text()
        if t.isdigit() and t != "0" and not self.search.text():
            i = int(t) - 1
            if i < len(self.classes):
                it = self.list.item(i)
                if self.multi:
                    self._toggle(it)
                else:
                    self.result = ("set", [self.classes[i]])
                    self.accept()
                return
        if e.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up) and self.search.hasFocus():
            self.list.setFocus()
        super().keyPressEvent(e)

    def _filter(self, text):
        t = text.strip().lower()
        first = None
        for i in range(self.list.count()):
            it = self.list.item(i)
            hide = bool(t) and t not in it.data(Qt.ItemDataRole.UserRole).lower()
            it.setHidden(hide)
            if not hide and first is None:
                first = i
        if first is not None:
            self.list.setCurrentRow(first)
        exact = any(c.lower() == t for c in self.classes)
        self.new_hint.setText(f"Enter: new class “{text.strip()}”" if t and first is None and not exact
                              and self.can_create else "")

    def _enter(self):
        text = self.search.text().strip()
        visible = [self.list.item(i) for i in range(self.list.count()) if not self.list.item(i).isHidden()]
        if self.multi:
            add, remove = [], []
            for i in range(self.list.count()):
                it = self.list.item(i)
                c = it.data(Qt.ItemDataRole.UserRole)
                if it.checkState() == Qt.CheckState.Checked and self.current.get(c, 0) < self.n:
                    add.append(c)
                elif it.checkState() == Qt.CheckState.Unchecked and self.current.get(c, 0):
                    remove.append(c)
            if text and not visible and self.can_create:
                add.append(text)
            self.result = ("change", add, remove)
            self.accept()
            return
        exact = next((c for c in self.classes if c.lower() == text.lower()), None) if text else None
        if exact:
            self.result = ("set", [exact])
        elif text and not visible:                       # nothing matches: a new class
            if not self.can_create:
                return
            self.result = ("set", [text])
        else:
            it = self.list.currentItem()
            if it is None or it.isHidden():
                it = visible[0] if visible else None
            if it is None:
                return
            self.result = ("set", [it.data(Qt.ItemDataRole.UserRole)])
        self.accept()
