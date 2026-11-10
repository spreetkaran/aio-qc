# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QLabel, QToolButton,
)


class BookmarksDock(QDockWidget):
    """In/Out range + bookmarks with editable notes. Double-click a row to jump."""
    jump_requested = pyqtSignal(int)
    note_changed = pyqtSignal(int, str)
    remove_requested = pyqtSignal(list)

    def __init__(self, parent, toolbar_actions=()):
        super().__init__("Marks", parent)
        self.setObjectName("MarksDock")
        self._updating = False
        self.time_text = lambda f: str(f)

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 6, 6)

        self.range_label = QLabel()
        self.range_label.setWordWrap(True)
        self.range_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.range_label)

        row = QHBoxLayout()
        for act in toolbar_actions:
            b = QToolButton()
            b.setDefaultAction(act)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            b.setAutoRaise(True)
            row.addWidget(b)
        row.addStretch(1)
        self.remove_btn = QToolButton()
        self.remove_btn.setText("Remove")
        self.remove_btn.setToolTip("Remove selected bookmarks (Del)")
        self.remove_btn.clicked.connect(self._remove_selected)
        row.addWidget(self.remove_btn)
        lay.addLayout(row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Frame", "Time", "Note"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                   QAbstractItemView.EditTrigger.EditKeyPressed)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        self.table.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self.table, 1)

        hint = QLabel("Double-click Frame/Time to jump, Note to edit.")
        hint.setStyleSheet("color: gray; font-size: 8pt;")
        lay.addWidget(hint)
        self.setWidget(body)

    def set_data(self, bookmarks, in_frame, out_frame):
        tt = self.time_text
        parts = [f"<b>In:</b> {tt(in_frame)} (#{in_frame})" if in_frame is not None else "<b>In:</b> start",
                 f"<b>Out:</b> {tt(out_frame)} (#{out_frame})" if out_frame is not None else "<b>Out:</b> end"]
        self.range_label.setText(" &nbsp; ".join(parts))
        self._updating = True
        self.table.setRowCount(0)
        for f, note in sorted(bookmarks.items()):
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, text in enumerate((str(f), tt(f), note or "")):
                it = QTableWidgetItem(text)
                it.setData(Qt.ItemDataRole.UserRole, f)
                if c < 2:
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, it)
        self._updating = False

    def _on_double_click(self, row, col):
        if col < 2:
            self.jump_requested.emit(self.table.item(row, 0).data(Qt.ItemDataRole.UserRole))

    def _on_item_changed(self, item):
        if not self._updating and item.column() == 2:
            self.note_changed.emit(item.data(Qt.ItemDataRole.UserRole), item.text())

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        frames = [self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows]
        if frames:
            self.remove_requested.emit(frames)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Delete:
            self._remove_selected()
        else:
            super().keyPressEvent(ev)
