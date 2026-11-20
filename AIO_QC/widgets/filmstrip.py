# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Thumbnail strip around the current file (lazy, async, cached)."""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QListView, QListWidget, QListWidgetItem, QStyle,
                             QStyledItemDelegate)

from AIO_QC.utils import theme_color, LRUCache
from .workers import ThumbSignals, ThumbTask

THUMB = 112
WINDOW = 15            # files shown before / after the current one


QWIDGETSIZE_MAX = 16777215
TILE_W, TILE_H = THUMB + 16, 63 + 40      # 3.0 mock-up: 16:9 thumbnail + the number under it
STRIP_THICKNESS = TILE_H + 22            # height when horizontal, width when vertical


class _TileDelegate(QStyledItemDelegate):
    """A rounded 16:9 thumbnail, the review dot (on the picture) and the file's number in the mono font; the
    current file has an accent frame on a soft accent tile."""

    def sizeHint(self, opt, idx):
        return QSize(TILE_W, TILE_H)

    def paint(self, p, opt, idx):
        from PyQt6.QtCore import QRect, QRectF
        from PyQt6.QtGui import QFontMetrics, QPainterPath
        from AIO_QC.widgets.design import mono, tokens
        t = tokens()
        r = opt.rect.adjusted(2, 2, -2, -2)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        sel = bool(opt.state & QStyle.StateFlag.State_Selected)
        hov = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        if sel:
            p.setPen(QPen(QColor(t["accent"]), 2))
            p.setBrush(QColor(t["accent_soft"]))
            p.drawRoundedRect(QRectF(r).adjusted(1, 1, -1, -1), 7, 7)
        elif hov:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(t["hover"]))
            p.drawRoundedRect(QRectF(r), 7, 7)
        img = QRectF(r.left() + 6, r.top() + 6, r.width() - 12, 63)
        clip = QPainterPath()
        clip.addRoundedRect(img, 4, 4)
        p.setClipPath(clip)
        p.fillRect(img, QColor(t["track"]))
        icon = idx.data(Qt.ItemDataRole.DecorationRole)
        if icon is not None and hasattr(icon, "pixmap"):
            pm = icon.pixmap(QSize(THUMB * 2, THUMB * 2))
            if not pm.isNull():
                sc = max(img.width() / pm.width(), img.height() / pm.height())     # fill the tile (centre crop)
                w, h = pm.width() * sc, pm.height() * sc
                p.drawPixmap(QRectF(img.center().x() - w / 2, img.center().y() - h / 2, w, h), pm,
                             QRectF(pm.rect()))
        p.setClipping(False)
        fs = self.parent()
        key = idx.data(Qt.ItemDataRole.UserRole)
        st = fs.session.review.status(key) if getattr(fs, "session", None) is not None and key else None
        if st is not None:                                # review mark: a dot inside the picture's corner
            d = 10
            p.setPen(QPen(QColor("#FFFFFF"), 1.6))
            p.setBrush(QColor(t[{"ok": "ok", "fixed": "info"}.get(st, "warn")]))
            p.drawEllipse(QRectF(img.right() - d - 5, img.top() + 5, d, d))
        sess = getattr(fs, "session", None)
        if sess is not None and key and sess.is_classification:  # a classifier image: its class on the picture
            tags = sess.tags_of(key)
            text = ", ".join(tags) if tags else "no class"
            cf = QFont(p.font())
            cf.setPixelSize(10)
            cf.setWeight(QFont.Weight.DemiBold)
            p.setFont(cf)
            fm = QFontMetrics(cf)
            text = fm.elidedText(text, Qt.TextElideMode.ElideRight, int(img.width()) - 26)
            w, h = fm.horizontalAdvance(text) + 18, fm.height() + 1
            chip = QRectF(img.left() + 3, img.bottom() - h - 3, w, h)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(20, 20, 24, 215))
            p.drawRoundedRect(chip, h / 2, h / 2)
            cols = fs.class_colors() if callable(getattr(fs, "class_colors", None)) else {}
            col = QColor(cols.get(tags[0])) if tags and tags[0] in cols else QColor(t["faint"])
            p.setBrush(col)
            p.drawEllipse(QRectF(chip.left() + 5, chip.center().y() - 3.5, 7, 7))
            p.setPen(QColor("white"))
            p.drawText(chip.adjusted(14, 0, -3, 0), int(Qt.AlignmentFlag.AlignVCenter), text)
        f = mono(12)
        if sel:
            f.setWeight(f.Weight.DemiBold)
        p.setFont(f)
        p.setPen(QColor(t["accent_text"] if sel else t["muted"]))
        text = str(idx.data(Qt.ItemDataRole.DisplayRole) or "")
        p.drawText(QRect(r.left(), int(img.bottom()) + 4, r.width(), r.bottom() - int(img.bottom()) - 4),
                   int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                   QFontMetrics(f).elidedText(text, Qt.TextElideMode.ElideLeft, r.width() - 8))
        p.restore()


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
        self.setItemDelegate(_TileDelegate(self))
        self.setMouseTracking(True)
        self.setFrameShape(QListView.Shape.NoFrame)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.vertical = None
        self.set_vertical(False)
        self.itemClicked.connect(lambda it: self.keyActivated.emit(it.data(Qt.ItemDataRole.UserRole)))
        self.cache = LRUCache(capacity=600)
        self.signals = ThumbSignals()
        self.signals.loaded.connect(self._on_loaded)
        self.generation = 0
        self.session = None
        self.class_colors = None              # () -> {class: QColor}: classification datasets show the class
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
            self.setFixedWidth(TILE_W + 22)
        else:
            self.setMinimumWidth(0)
            self.setMaximumWidth(QWIDGETSIZE_MAX)
            self.setFixedHeight(STRIP_THICKNESS)
        self.doItemsLayout()
        if self.currentItem() is not None:
            self.scrollToItem(self.currentItem(), QAbstractItemView.ScrollHint.PositionAtCenter)

    def set_session(self, session):
        self.session = session
        if session is not None:                     # class chips / review dots follow changes made elsewhere
            session.annotationsChanged.connect(lambda *_a: self.viewport().update())
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
        import os
        import re
        stem = os.path.splitext(sample.name)[0]
        m = re.search(r"(\d{2,})$", stem)                # frame-like names: just the number (0012)
        if m and len(stem) > 12:
            return m.group(1)
        return stem if len(stem) <= 16 else "…" + stem[-15:]

    def _decorate(self, pix: QPixmap, key: str) -> QPixmap:
        """3.0: the tile painter draws the review dot (inside the picture's corner); the pixmap stays as loaded."""
        return pix

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
