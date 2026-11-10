# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Command palette (Ctrl+K): type a few letters to run any menu action — every QC tool, review mark, view toggle,
Storage, Logs … Works with the QC Panel collapsed or hidden.

Fuzzy matching: all typed characters in order (word starts and consecutive runs score higher); recently used
commands come first for an empty query and win ties. Disabled actions are listed but greyed out and cannot run.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QVBoxLayout

from AIO_QC.utils import style as apply_style

ROLE = Qt.ItemDataRole.UserRole
RECENT_MAX = 12


def collect_actions(menubar) -> list[tuple[str, QAction]]:
    """(path such as 'QC ▸ Shapes ▸ Crop Gallery', action) for every runnable menu action, once each."""
    out, seen = [], set()

    def walk(menu: QMenu, path: list[str]):
        for act in menu.actions():
            if act.isSeparator() or not act.isVisible():
                continue
            sub = act.menu()
            if sub is not None:
                walk(sub, path + [_clean(act.text())])
                continue
            if id(act) in seen or not _clean(act.text()):
                continue
            seen.add(id(act))
            out.append((" ▸ ".join(path + [_clean(act.text())]), act))

    for act in menubar.actions():
        if act.menu() is not None:
            walk(act.menu(), [_clean(act.text())])
    return out


def _clean(text: str) -> str:
    return text.replace("&&", "\0").replace("&", "").replace("\0", "&").replace("…", "").strip()


def score(query: str, text: str) -> float | None:
    """Subsequence match score (higher is better) or None. Word starts and runs are rewarded."""
    if not query:
        return 0.0
    q, t = query.lower(), text.lower()
    pos, s, run, prev = 0, 0.0, 0, -2
    for ch in q:
        if ch == " ":
            continue
        i = t.find(ch, pos)
        if i < 0:
            return None
        run = run + 1 if i == prev + 1 else 1
        s += 1.0 + run * 0.5
        if i == 0 or not t[i - 1].isalnum():
            s += 2.0                                  # start of a word
        prev, pos = i, i + 1
    tail = t.rsplit("▸", 1)[-1]
    if q.replace(" ", "") in tail.replace(" ", ""):
        s += 6.0                                      # matches inside the command name itself
    return s - len(t) * 0.01


class CommandPalette(QDialog):
    def __init__(self, window, settings):
        super().__init__(window, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.window_, self.settings = window, settings
        self.setObjectName("palette")
        apply_style(self, "QDialog#palette { border: 1px solid @line; border-radius: 8px; }")
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Type a command — e.g. crop, dim, flag, storage, focus …")
        self.edit.setClearButtonEnabled(True)
        self.edit.textChanged.connect(self._fill)
        self.edit.installEventFilter(self)
        self.list = QListWidget()
        self.list.setUniformItemSizes(True)
        self.list.itemActivated.connect(self._run)
        self.list.itemClicked.connect(self._run)
        hint = QLabel("↑ ↓ choose · Enter run · Esc close")
        apply_style(hint, "color:@muted;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 6)
        lay.addWidget(self.edit)
        lay.addWidget(self.list, 1)
        lay.addWidget(hint)
        self.items: list[tuple[str, QAction]] = []

    def popup(self):
        self.items = collect_actions(self.window_.menuBar())
        w = min(640, max(420, self.window_.width() // 2))
        h = min(460, max(260, self.window_.height() // 2))
        top = self.window_.mapToGlobal(self.window_.rect().topLeft())
        self.setGeometry(top.x() + (self.window_.width() - w) // 2, top.y() + 70, w, h)
        self.edit.clear()
        self._fill("")
        self.show()
        self.raise_()
        self.activateWindow()
        self.edit.setFocus()

    def _recent(self) -> list[str]:
        try:
            return list(self.settings.value("palette/recent", []) or [])
        except Exception:
            return []

    def _fill(self, text: str):
        recent = self._recent()
        rank = {p: i for i, p in enumerate(recent)}
        scored = []
        for path, act in self.items:
            s = score(text.strip(), path)
            if s is None:
                continue
            bonus = (RECENT_MAX - rank[path]) * (0.6 if text.strip() else 100) if path in rank else 0.0
            scored.append((s + bonus + (0 if act.isEnabled() else -50), path, act))
        scored.sort(key=lambda x: -x[0])
        self.list.clear()
        for _s, path, act in scored[:200]:
            sc = act.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
            it = QListWidgetItem(act.icon(), path + (f"      {sc}" if sc else ""))
            it.setData(ROLE, (path, act))
            tip = act.toolTip()
            if tip and tip != _clean(act.text()):
                it.setToolTip(tip)
            if not act.isEnabled():
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.list.addItem(it)
        for r in range(self.list.count()):
            if self.list.item(r).flags() & Qt.ItemFlag.ItemIsEnabled:
                self.list.setCurrentRow(r)
                break

    def eventFilter(self, obj, e):
        if obj is self.edit and e.type() == QEvent.Type.KeyPress:
            k = e.key()
            if k in (Qt.Key.Key_Down, Qt.Key.Key_Up, Qt.Key.Key_PageDown, Qt.Key.Key_PageUp):
                self.list.keyPressEvent(e)
                return True
            if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                it = self.list.currentItem()
                if it is not None:
                    self._run(it)
                return True
            if k == Qt.Key.Key_Escape:
                self.close()
                return True
        return super().eventFilter(obj, e)

    def _run(self, item):
        path, act = item.data(ROLE)
        if not act.isEnabled():
            return
        recent = [p for p in self._recent() if p != path]
        self.settings.setValue("palette/recent", [path] + recent[:RECENT_MAX - 1])
        self.close()
        act.trigger()
