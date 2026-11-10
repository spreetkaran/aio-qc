# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Thumbnail strip around the current file (lazy, async, cached)."""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QAbstractItemView, QListView, QListWidget, QListWidgetItem

from AIO_QC.utils import theme_color, LRUCache
from .workers import ThumbSignals, ThumbTask

THUMB = 112
WINDOW = 15            # files shown before / after the current one


QWIDGETSIZE_MAX = 16777215
STRIP_THICKNESS = THUMB + 46   # height when horizontal, width when vertical


class Filmstrip(QListWidget):
    """
    NOTE: do not name any method `reset`, `clear`, `update`, ... — QListWidget already has
    those slots and Qt calls them internally with its own arguments.
    """
    keyActivated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setWrapping(False)
        self.setMovement(QListView.Movement.Static)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setIconSize(QSize(THUMB, THUMB))
        self.setSpacing(4)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.vertical = None
        self.set_vertical(False)
        self.itemClicked.connect(lambda it: self.keyActivated.emit(it.data(Qt.ItemDataRole.UserRole)))
        self.cache = LRUCache(capacity=600)
        self.signals = ThumbSignals()
        self.signals.loaded.connect(self._on_loaded)
        self.generation = 0
        self.session = None
        self._inflight: set[str] = set()
        self._placeholder = self._blank()

    def _blank(self) -> QIcon:
        pix = QPixmap(THUMB, THUMB)
        pix.fill(theme_color("placeholder"))
        return QIcon(pix)

    def set_vertical(self, vertical: bool):
        """Horizontal strip when docked top/bottom, vertical column when docked left/right."""
        if vertical == self.vertical:
            return
        self.vertical = vertical
        self.setFlow(QListView.Flow.TopToBottom if vertical else QListView.Flow.LeftToRight)
        on, off = Qt.ScrollBarPolicy.ScrollBarAsNeeded, Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        self.setHorizontalScrollBarPolicy(off if vertical else on)
        self.setVerticalScrollBarPolicy(on if vertical else off)
        # lock the thin side only; release the other one
        if vertical:
            self.setMinimumHeight(0)
            self.setMaximumHeight(QWIDGETSIZE_MAX)
            self.setFixedWidth(STRIP_THICKNESS)
        else:
            self.setMinimumWidth(0)
            self.setMaximumWidth(QWIDGETSIZE_MAX)
            self.setFixedHeight(STRIP_THICKNESS)
        self.doItemsLayout()
        if self.currentItem() is not None:
            self.scrollToItem(self.currentItem(), QAbstractItemView.ScrollHint.PositionAtCenter)

    def set_session(self, session):
        self.session = session
        self.generation += 1
        self.cache.clear()
        self._inflight.clear()
        self.clear()

    def show_around(self, keys: list[str], current: str | None):
        """keys: the visible neighbourhood (file-list order)."""
        if not self.isVisible() or self.session is None:
            return
        self.setUpdatesEnabled(False)
        self.clear()
        pool = QThreadPool.globalInstance()
        for k in keys:
            s = self.session.by_key.get(k)
            if s is None:
                continue
            it = QListWidgetItem(self._label(k, s))
            it.setData(Qt.ItemDataRole.UserRole, k)
            it.setToolTip(s.display)
            it.setSizeHint(QSize(THUMB + 12, THUMB + 38))
            pix = self.cache.get(k)
            it.setIcon(QIcon(self._decorate(pix, k)) if pix is not None else self._placeholder)
            self.addItem(it)
            if k == current:
                self.setCurrentItem(it)
            if pix is None and k not in self._inflight:
                self._inflight.add(k)
                pool.start(ThumbTask(self.signals, self.session.adapter, s, self.generation, THUMB))
        self.setUpdatesEnabled(True)
        if self.currentItem() is not None:
            self.scrollToItem(self.currentItem(), QAbstractItemView.ScrollHint.PositionAtCenter)

    def _label(self, key, sample) -> str:
        name = sample.name if len(sample.name) <= 16 else sample.name[:7] + "…" + sample.name[-7:]
        st = self.session.review.status(key)
        return {"ok": "✓ ", "flag": "⚑ ", "fixed": "🔧 "}.get(st, "") + name

    def _decorate(self, pix: QPixmap, key: str) -> QPixmap:
        st = self.session.review.status(key) if self.session else None
        if st is None:
            return pix
        out = QPixmap(pix)
        p = QPainter(out)
        p.setPen(QPen(theme_color({"ok": "ok", "fixed": "info"}.get(st, "warn")), 4))
        p.drawRect(out.rect().adjusted(2, 2, -2, -2))
        p.end()
        return out

    def _on_loaded(self, key, gen, img):
        self._inflight.discard(key)
        if gen != self.generation or img.isNull():
            return
        pix = QPixmap.fromImage(img)
        self.cache.put(key, pix)
        for i in range(self.count()):
            it = self.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == key:
                it.setIcon(QIcon(self._decorate(pix, key)))
                break

    def refresh_labels(self):
        if self.session is None:
            return
        for i in range(self.count()):
            it = self.item(i)
            k = it.data(Qt.ItemDataRole.UserRole)
            s = self.session.by_key.get(k)
            if s is not None:
                it.setText(self._label(k, s))
                pix = self.cache.get(k)
                if pix is not None:
                    it.setIcon(QIcon(self._decorate(pix, k)))
