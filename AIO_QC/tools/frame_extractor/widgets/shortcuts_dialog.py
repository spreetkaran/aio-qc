# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QKeySequenceEdit, QLineEdit, QPushButton, QDialogButtonBox, QLabel, QWidget,
    QAbstractItemView,
)


class ShortcutsDialog(QDialog):
    """Settings ▸ Keyboard ▸ Customize: every command with its shortcut, editable. Conflicts are highlighted
    and must be resolved before saving. (The printable reference with mouse gestures is in the help center.)"""
    reference_requested = pyqtSignal()

    def __init__(self, parent, entries, defaults):
        """entries: list of (action_id, QAction, category); defaults: id -> default sequence string."""
        super().__init__(parent)
        self.setWindowTitle("Customize keyboard shortcuts")
        from AIO_QC.tools.frame_extractor.utils.qt_utils import preferred_size
        preferred_size(self, 720, 600)
        self.entries, self.defaults = entries, defaults
        lay = QVBoxLayout(self)

        kb = QWidget()
        kl = QVBoxLayout(kb)
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setToolTip('Filter by command name, category or key.')
        self.search.setPlaceholderText("Filter commands…")
        self.search.textChanged.connect(self._filter)
        top.addWidget(self.search, 1)
        for text, slot in (("Reset selected", self._reset_selected), ("Reset all", self._reset_all)):
            b = QPushButton(text)
            b.setToolTip('Reset shortcuts to the defaults (selected rows or all).')
            b.clicked.connect(slot)
            top.addWidget(b)
        kl.addLayout(top)
        self.table = QTableWidget(len(entries), 3)
        self.table.setHorizontalHeaderLabels(["Command", "Category", "Shortcut"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(2, 200)
        self.editors = {}
        for r, (aid, act, cat) in enumerate(entries):
            it = QTableWidgetItem(act.text().replace("\n", " ").replace("&&", "\0").replace("&", "").replace("\0", "&"))
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            it.setData(Qt.ItemDataRole.UserRole, aid)
            if not act.icon().isNull():
                it.setIcon(act.icon())
            self.table.setItem(r, 0, it)
            c = QTableWidgetItem(cat)
            c.setFlags(c.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 1, c)
            ed = QKeySequenceEdit(act.shortcut())
            ed.setToolTip('Click and press the new key combination; the clear button removes it. Conflicts are highlighted and must be resolved before saving.')
            try:
                ed.setClearButtonEnabled(True)
                ed.setMaximumSequenceLength(1)
            except AttributeError:
                pass
            ed.keySequenceChanged.connect(self._check_conflicts)
            self.table.setCellWidget(r, 2, ed)
            self.editors[aid] = ed
        kl.addWidget(self.table)
        self.conflict = QLabel()
        self.conflict.setStyleSheet("color: #d64545;")
        kl.addWidget(self.conflict)
        lay.addWidget(kb)
        ref = QPushButton("Show the shortcut reference (with mouse gestures)…")
        ref.clicked.connect(self.reference_requested.emit)
        kl.addWidget(ref, 0, Qt.AlignmentFlag.AlignLeft)

        self.bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        lay.addWidget(self.bb)
        self._check_conflicts()

    def _filter(self, text):
        t = text.lower()
        for r in range(self.table.rowCount()):
            aid = self.table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            hay = f"{self.table.item(r, 0).text()} {self.table.item(r, 1).text()} " \
                  f"{self.editors[aid].keySequence().toString()}".lower()
            self.table.setRowHidden(r, t not in hay)

    def _reset_selected(self):
        for r in {i.row() for i in self.table.selectedIndexes()}:
            aid = self.table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            self.editors[aid].setKeySequence(QKeySequence(self.defaults.get(aid, "")))

    def _reset_all(self):
        for aid, ed in self.editors.items():
            ed.setKeySequence(QKeySequence(self.defaults.get(aid, "")))

    def _check_conflicts(self, *_):
        seen, dup = {}, set()
        for aid, ed in self.editors.items():
            k = ed.keySequence().toString()
            if k:
                if k in seen:
                    dup.update({aid, seen[k]})
                seen.setdefault(k, aid)
        for r in range(self.table.rowCount()):
            aid = self.table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            self.table.item(r, 0).setForeground(QColor("#d64545") if aid in dup else self.palette().text().color())
        names = sorted({self.table.item(r, 0).text() for r in range(self.table.rowCount())
                        if self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) in dup})
        self.conflict.setText(("Conflicting shortcuts: " + ", ".join(names)) if names else "")
        self.bb.button(QDialogButtonBox.StandardButton.Save).setEnabled(not dup)

    def result_map(self):
        return {aid: ed.keySequence().toString() for aid, ed in self.editors.items()}
