# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QLabel
from AIO_QC.utils.fx_settings import fx_settings


class CommandPalette(QDialog):
    """Ctrl+K: type to find any command; Enter runs it. Recently used first."""

    def __init__(self, parent, entries):
        """entries: list of (action_id, QAction, category)."""
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.entries = entries
        self.store = fx_settings()
        self.recent = [r for r in (self.store.value("palette/recent", []) or []) if r]
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.edit = QLineEdit()
        self.edit.setToolTip("Type any part of a command's name, menu or description; ↑ / ↓ to choose, Enter to run, Esc to close.")
        self.edit.setPlaceholderText("Type a command…  (↑/↓ to move, Enter to run, Esc to close)")
        self.edit.textChanged.connect(self.refilter)
        self.edit.installEventFilter(self)
        lay.addWidget(self.edit)
        self.list = QListWidget()
        self.list.itemActivated.connect(self.run_item)
        lay.addWidget(self.list)
        self.hint = QLabel()
        self.hint.setStyleSheet("color: gray; font-size: 8pt;")
        lay.addWidget(self.hint)
        self.refilter("")
        if parent is not None:
            g = parent.geometry()
            self.resize(600, 420)
            self.move(g.x() + (g.width() - 600) // 2, g.y() + 90)

    @staticmethod
    def _label(act):
        return act.text().replace("\n", " ").replace("&&", "\0").replace("&", "").replace("\0", "&").strip()

    def refilter(self, text):
        words = text.lower().split()
        rows = []
        for aid, act, cat in self.entries:
            label = self._label(act)
            hay = f"{label} {cat} {act.toolTip()}".lower()
            if all(w in hay for w in words):
                score = (0 if label.lower().startswith(text.lower()) else 1,
                         self.recent.index(aid) if aid in self.recent else 99, label.lower())
                rows.append((score, aid, act, cat, label))
        rows.sort(key=lambda r: r[0])
        self.list.clear()
        for _, aid, act, cat, label in rows:
            sc = act.shortcut().toString()
            check = ("✓ " if act.isChecked() else "   ") if act.isCheckable() else ""
            it = QListWidgetItem(f"{check}{label}    —  {cat}" + (f"    [{sc}]" if sc else ""))
            it.setData(Qt.ItemDataRole.UserRole, aid)
            if act.icon() and not act.icon().isNull():
                it.setIcon(act.icon())
            if not act.isEnabled():
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.list.addItem(it)
        for i in range(self.list.count()):
            if self.list.item(i).flags() & Qt.ItemFlag.ItemIsEnabled:
                self.list.setCurrentRow(i)
                break
        self.hint.setText(f"{self.list.count()} command(s)")

    def eventFilter(self, obj, ev):
        from PyQt6.QtCore import QEvent
        if obj is self.edit and ev.type() == QEvent.Type.KeyPress:
            k = ev.key()
            if k in (Qt.Key.Key_Down, Qt.Key.Key_Up, Qt.Key.Key_PageDown, Qt.Key.Key_PageUp):
                self.list.keyPressEvent(ev)
                return True
            if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                it = self.list.currentItem()
                if it:
                    self.run_item(it)
                return True
        return super().eventFilter(obj, ev)

    def run_item(self, item):
        if not item.flags() & Qt.ItemFlag.ItemIsEnabled:
            return
        aid = item.data(Qt.ItemDataRole.UserRole)
        act = next(a for i, a, _ in self.entries if i == aid)
        self.recent = [aid] + [r for r in self.recent if r != aid][:14]
        self.store.setValue("palette/recent", self.recent)
        self.close()
        act.trigger()
