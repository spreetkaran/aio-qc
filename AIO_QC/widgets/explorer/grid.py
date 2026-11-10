# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The Prediction Explorer's image grid and object (patch) view — one virtual list view, two models.

Only the visible tiles are painted (QListView IconMode, uniform item sizes), images are decoded once per size
bucket WITHOUT overlays and the overlays are painted live, so moving a slider or switching a layer repaints
instantly — no re-decoding.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QAbstractListModel, QModelIndex, QRectF, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFontMetrics, QPainter, QPalette, QPen, QPixmap
from PyQt6.QtWidgets import QAbstractItemView, QListView, QStyle, QStyledItemDelegate

from AIO_QC.core.explorer import ERROR_TAGS, TAG_NAMES, decode_id
from AIO_QC.utils import LRUCache, theme_color
from ..card_gallery import draw_review_badge
from .paint import CropTask, Signals, ThumbTask, draw_objects, image_path, object_text, tag_color

FOOT = 36
GAP = 6


class ListModel(QAbstractListModel):
    """rows -> one int each (an image index, or an encoded object id in object mode)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = np.zeros(0, np.int64)

    def set_items(self, items):
        self.beginResetModel()
        self.items = np.asarray(items, np.int64)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.items)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.UserRole:
            return int(self.items[index.row()])
        return None


class TileDelegate(QStyledItemDelegate):
    def __init__(self, view):
        super().__init__(view)
        self.v = view

    def sizeHint(self, option, index):
        return self.v.cell_size()

    def paint(self, p: QPainter, option, index):
        item = index.data(Qt.ItemDataRole.UserRole)
        if item is None:
            return
        v = self.v
        r = option.rect.adjusted(3, 3, -3, -3)
        sel = bool(option.state & QStyle.StateFlag.State_Selected)
        cur = bool(option.state & QStyle.StateFlag.State_HasFocus)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(theme_color("accent") if sel else theme_color("line"), 3 if sel else 1))
        p.setBrush(option.palette.color(QPalette.ColorRole.Base))
        p.drawRoundedRect(QRectF(r), 7, 7)
        area = r.adjusted(GAP, GAP, -GAP, -FOOT)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme_color("placeholder"))
        p.drawRoundedRect(QRectF(area), 5, 5)
        if v.mode == "grid":
            self._paint_image(p, item, area)
            key = v.panel.data.keys[item]
            name = v.panel.display_name(key)
            chips = v.panel.tile_chips(item)
        else:
            self._paint_object(p, item, area)
            kind, row = decode_id(item)
            key, name, chips = v.panel.object_caption(kind, row)
        if cur and not sel:
            p.setPen(QPen(theme_color("accent"), 1.5, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(r).adjusted(1, 1, -1, -1), 7, 7)
        draw_review_badge(p, area.x() + 4, area.y() + 4, v.panel.session.review.status(key))
        # footer: name + chips
        fm = QFontMetrics(option.font)
        y = area.bottom() + 4
        x = r.x() + GAP
        for text, color in chips:
            w = fm.horizontalAdvance(text) + 10
            if x + w > r.right() - GAP:
                break
            c = QColor(color)
            bg = QColor(c)
            bg.setAlpha(50)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(bg)
            p.drawRoundedRect(QRectF(x, y, w, 15), 7, 7)
            p.setPen(c.darker(115) if option.palette.color(QPalette.ColorRole.Base).lightnessF() > 0.5 else c.lighter(120))
            p.drawText(QRectF(x, y, w, 15), Qt.AlignmentFlag.AlignCenter, text)
            x += w + 4
        p.setPen(option.palette.color(QPalette.ColorRole.Text))
        p.drawText(QRectF(r.x() + GAP, y + 15, r.width() - 2 * GAP, 16), Qt.AlignmentFlag.AlignLeft |
                   Qt.AlignmentFlag.AlignVCenter, fm.elidedText(name, Qt.TextElideMode.ElideMiddle, r.width() - 2 * GAP))
        p.restore()

    def _paint_image(self, p, i, area):
        v = self.v
        pix, info = v.thumb(i, area.size())
        if pix is None or pix.isNull():
            return
        sz = pix.deviceIndependentSize().toSize().scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x0 = area.x() + (area.width() - sz.width()) / 2
        y0 = area.y() + (area.height() - sz.height()) / 2
        p.drawPixmap(QRectF(x0, y0, sz.width(), sz.height()), pix, QRectF(pix.rect()))
        d = v.panel.data
        W = float(d.img_w[i] or (info[0] if info else 0) or 1)
        scale = sz.width() / W
        p.setClipRect(QRectF(x0, y0, sz.width(), sz.height()))
        draw_objects(p, v.panel.tile_objects(i), scale, x0, y0, **v.panel.paint_style(tile=True,
                                                                                         small=area.width() < 150))
        p.setClipping(False)

    def _paint_object(self, p, code, area):
        v = self.v
        pix, info = v.crop(code, area.size())
        if pix is None or pix.isNull() or info is None:
            return
        sz = pix.deviceIndependentSize().toSize().scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x0 = area.x() + (area.width() - sz.width()) / 2
        y0 = area.y() + (area.height() - sz.height()) / 2
        p.drawPixmap(QRectF(x0, y0, sz.width(), sz.height()), pix, QRectF(pix.rect()))
        cx, cy, k = info
        k *= sz.width() / max(1, pix.deviceIndependentSize().width())
        objs, focus = v.panel.patch_objects(code)
        p.setClipRect(QRectF(x0, y0, sz.width(), sz.height()))
        draw_objects(p, objs, k, x0 - cx * k, y0 - cy * k, selected=focus, fill_selected=False,
                     **v.panel.paint_style(tile=True, small=True))
        p.setClipping(False)

    def helpEvent(self, event, view, option, index):
        from PyQt6.QtWidgets import QToolTip
        item = index.data(Qt.ItemDataRole.UserRole)
        if item is None:
            return False
        QToolTip.showText(event.globalPos(), self.v.panel.tooltip(self.v.mode, item), view)
        return True


class ExplorerGrid(QListView):
    opened = pyqtSignal(int)                 # image index (grid) / encoded object (objects)
    selectionChangedSoon = pyqtSignal()

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.mode = "grid"
        self.tile_w = 220
        self.model_ = ListModel(self)
        self.setModel(self.model_)
        self.delegate = TileDelegate(self)
        self.setItemDelegate(self.delegate)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setUniformItemSizes(True)
        self.setSpacing(2)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.verticalScrollBar().setSingleStep(24)
        self.setMouseTracking(True)
        self.setUpdatesEnabled(True)
        self.cache = LRUCache(capacity=4000, max_bytes=600 * 1024 * 1024)
        self.sig = Signals()
        self.sig.done.connect(self._on_done)
        self.gen = 0
        self._inflight: set = set()
        self._prio = 0
        self._repaint = QTimer(self, singleShot=True, interval=25, timeout=self.viewport().update)
        self.doubleClicked.connect(self._open)
        self._apply_size()

    # ---- size
    def cell_size(self) -> QSize:
        w = self.tile_w
        h = int(w * (0.75 if self.mode == "grid" else 0.8)) + FOOT + GAP
        return QSize(w, h)

    def set_tile_width(self, w: int):
        self.want_w = int(max(110, min(w, 640)))
        self._fit_width()

    def _fit_width(self):
        """Tiles of about the wanted width that fill the row (no dead band on the right)."""
        avail = self.viewport().width() - 8
        want = getattr(self, "want_w", self.tile_w)
        cols = max(1, int(round(avail / (want + 4))))
        w = max(110, int(avail / cols) - 4)
        if w != self.tile_w:
            self.tile_w = w
            self._apply_size()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_width()

    def _apply_size(self):
        self.setGridSize(self.cell_size() + QSize(4, 4))
        self.doItemsLayout()

    # ---- content
    def set_items(self, mode: str, items, keep_scroll=False):
        sel = self.selected_items()
        cur = self.current_item()
        bar = self.verticalScrollBar().value()
        if mode != self.mode:
            self.mode = mode
            self._apply_size()
            keep_scroll = False
        self.model_.set_items(items)
        if keep_scroll:
            pos = {int(x): n for n, x in enumerate(self.model_.items)}
            sm = self.selectionModel()
            for x in sel:
                n = pos.get(int(x))
                if n is not None:
                    sm.select(self.model_.index(n), sm.SelectionFlag.Select)
            if cur is not None and cur in pos:
                sm.setCurrentIndex(self.model_.index(pos[cur]), sm.SelectionFlag.NoUpdate)
            QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(bar))
        else:
            self.scrollToTop()
        self.selectionChangedSoon.emit()

    def invalidate_images(self):
        """Files changed: drop decoded images."""
        self.gen += 1
        self._inflight.clear()
        self.cache.clear()
        self.viewport().update()

    def selected_items(self) -> list[int]:
        rows = sorted(i.row() for i in self.selectionModel().selectedIndexes()) if self.selectionModel() else []
        return [int(self.model_.items[r]) for r in rows if r < len(self.model_.items)]

    def current_item(self):
        i = self.currentIndex()
        return int(self.model_.items[i.row()]) if i.isValid() and i.row() < len(self.model_.items) else None

    def select_item(self, item):
        hits = np.nonzero(self.model_.items == item)[0]
        if len(hits):
            idx = self.model_.index(int(hits[0]))
            self.setCurrentIndex(idx)
            self.scrollTo(idx)

    def selectionChanged(self, a, b):
        super().selectionChanged(a, b)
        self.selectionChangedSoon.emit()

    def _open(self, index):
        item = index.data(Qt.ItemDataRole.UserRole)
        if item is not None:
            self.opened.emit(int(item))

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.currentIndex().isValid():
            self._open(self.currentIndex())
            return
        super().keyPressEvent(e)

    def wheelEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:           # Ctrl+wheel: tile size
            self.panel.zoom_tiles(1 if e.angleDelta().y() > 0 else -1)
            return
        super().wheelEvent(e)

    # ---- images
    @staticmethod
    def _bucket(size: QSize) -> QSize:
        b = 96
        return QSize((size.width() + b - 1) // b * b, (size.height() + b - 1) // b * b)

    def thumb(self, i: int, size: QSize):
        key = self.panel.data.keys[i]
        dpr = self.devicePixelRatioF()
        bs = self._bucket(QSize(int(size.width() * dpr), int(size.height() * dpr)))
        ck = ("t", key, bs.width(), bs.height())
        hit = self.cache.get(ck)
        if hit is not None:
            return hit
        smaller = self._any_cached(key)
        if ck not in self._inflight:
            self._inflight.add(ck)
            self._prio += 1
            QThreadPool.globalInstance().start(
                ThumbTask(self.sig, ck, image_path(self.panel.session, key), bs, self.gen), self._prio)
        return smaller if smaller is not None else (None, None)

    def _any_cached(self, key):
        for ck, val in reversed(self.cache.cache.items()):
            if ck[0] == "t" and ck[1] == key:
                return val
        return None

    def crop(self, code: int, size: QSize):
        dpr = self.devicePixelRatioF()
        bs = self._bucket(QSize(int(size.width() * dpr), int(size.height() * dpr)))
        ck = ("c", code, bs.width(), bs.height(), self.panel.data_version)
        hit = self.cache.get(ck)
        if hit is not None:
            return hit
        if ck not in self._inflight:
            self._inflight.add(ck)
            self._prio += 1
            key, box = self.panel.object_box(code)
            QThreadPool.globalInstance().start(
                CropTask(self.sig, ck, image_path(self.panel.session, key), box, bs, self.gen), self._prio)
        return None, None

    def _on_done(self, ck, gen, img, info):
        self._inflight.discard(ck)
        if gen != self.gen:
            return
        pix = QPixmap.fromImage(img) if not img.isNull() else QPixmap()
        pix.setDevicePixelRatio(self.devicePixelRatioF())
        self.cache.put(ck, (pix, info), max(1, pix.width() * pix.height() * 4))
        self._repaint.start()


def chips_for(counts, max_n=4) -> list:
    """Per-image tag counts -> footer chips (errors first)."""
    out = []
    for t in ERROR_TAGS:
        n = int(counts[t])
        if n:
            out.append((f"{n} {TAG_NAMES[t]}", tag_color(t).name()))
    return out[:max_n]


def object_chips(o, model_names) -> list:
    out = [(TAG_NAMES.get(o.tag, ""), tag_color(o.tag).name())]
    if o.kind == "pred":
        out.append((object_text(o), model_names and "#64748b" or "#64748b"))
    else:
        out.append((o.label, "#64748b"))
    return out
