# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Visual view for 🖼️ Duplicate Images: one row per duplicate group.

    [ info panel ][ thumb ][ thumb ][ thumb ] …  +N
      E0012 · exact          KEEP      ☑ marked    split chip · distance · file name
      3 images · train, val
      ⚠ leakage

Scales to thousands of groups: model/view (only visible rows paint), fixed row height, thumbnails decoded
lazily at tile size (QImageReader scaled decode) with newest-request-first priority, LRU-cached pixmaps.
Marks are the dialog's table check-states (single source of truth), so "Keep one per group", the table tab and
this view always agree.
"""
from __future__ import annotations

from PyQt6.QtCore import QAbstractListModel, QEvent, QModelIndex, QPointF, QRect, QRectF, QSize, Qt, QThreadPool, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPalette, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QAbstractItemView, QListView, QStyledItemDelegate, QToolTip

from AIO_QC.utils import LRUCache, theme_color
from .card_gallery import draw_review_badge, review_menu
from .workers import ThumbSignals, ThumbTask

THUMB = 112
CAPTION = 18
PAD = 10
INFO_W = 200
GAP = 10
BOX = 20            # tick-box size


class GroupModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.groups: list[dict] = []

    def set_groups(self, groups):
        self.beginResetModel()
        self.groups = groups
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.groups)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.UserRole:
            return self.groups[index.row()]
        return None


def _tiles(rect: QRect, n: int) -> tuple[list[QRect], int]:
    """Tile rects that fit in the row, and how many members do not fit."""
    x0 = rect.x() + INFO_W
    usable = rect.right() - x0 - PAD
    fit = max(1, usable // (THUMB + GAP))
    shown = min(n, fit if n <= fit else fit - 1)          # keep one slot for "+N"
    tiles = [QRect(x0 + k * (THUMB + GAP), rect.y() + PAD, THUMB, THUMB) for k in range(shown)]
    return tiles, n - shown


class GroupDelegate(QStyledItemDelegate):
    def __init__(self, view):
        super().__init__(view)
        self.view = view

    def sizeHint(self, option, index):
        return QSize(INFO_W + 4 * (THUMB + GAP), THUMB + CAPTION + 2 * PAD)

    # ------------------------------------------------------------------ paint
    def paint(self, painter: QPainter, option, index):
        g = index.data(Qt.ItemDataRole.UserRole)
        if g is None:
            return
        pal = option.palette
        r = option.rect
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = pal.color(QPalette.ColorRole.AlternateBase if index.row() % 2 else QPalette.ColorRole.Base)
        painter.fillRect(r, bg)
        text, muted = pal.color(QPalette.ColorRole.Text), theme_color("muted")

        # ---- info panel
        f = QFont(option.font)
        f.setBold(True)
        painter.setFont(f)
        painter.setPen(text)
        painter.drawText(QRect(r.x() + PAD, r.y() + PAD, INFO_W - 2 * PAD, 20), Qt.AlignmentFlag.AlignLeft,
                         f"{g['label']}")
        self._chip(painter, QRect(r.x() + PAD, r.y() + PAD + 24, 0, 18), g["kind"].upper(),
                   {"exact": theme_color("info"), "near": theme_color("accent"),
                    "same name": theme_color("warn")}.get(g["kind"], muted))
        painter.setFont(option.font)
        painter.setPen(muted)
        n_marked = sum(1 for m in g["members"] if self.view.is_marked(m))
        lines = [f"{len(g['members'])} images" + (f" · {n_marked} marked" if n_marked else ""),
                 ", ".join(g["splits"]) if g["splits"] else "no split"]
        if g["kind"] == "near":
            lines.append(f"distance ≤ {g['max_dist']} / 256 bits")
        for k, line in enumerate(lines):
            painter.drawText(QRect(r.x() + PAD, r.y() + PAD + 50 + k * 18, INFO_W - 2 * PAD, 18),
                             Qt.AlignmentFlag.AlignLeft, line)
        if g["leak"]:
            self._chip(painter, QRect(r.x() + PAD, r.y() + PAD + 52 + len(lines) * 18, 0, 18),
                       "⚠ LEAKAGE", theme_color("error"))

        # ---- thumbnails
        tiles, more = _tiles(r, len(g["members"]))
        any_marked = n_marked > 0
        fm = QFontMetrics(option.font)
        for tile, m in zip(tiles, g["members"]):
            self._thumb(painter, tile, m, pal)
            marked = self.view.is_marked(m)
            if marked:                                   # dim + red frame = will be trashed
                painter.fillRect(tile, QColor(0, 0, 0, 110))
                painter.setPen(QPen(theme_color("error"), 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(QRectF(tile).adjusted(1.5, 1.5, -1.5, -1.5), 6, 6)
            elif any_marked:                             # the survivors of a partially marked group
                self._chip(painter, QRect(tile.x() + 6, tile.y() + 6, 0, 18), "KEEP", theme_color("ok"))
            self._tickbox(painter, self._box_rect(tile), marked)
            if not any_marked or marked:            # KEEP chip sits in the same corner
                draw_review_badge(painter, tile.x() + 6, tile.y() + 6, self.view.dlg.session.review.status(m.key))
            split = self.view.split_of(m)
            if split:
                self._chip(painter, QRect(tile.x() + 6, tile.bottom() - 24, 0, 18), split, QColor(20, 20, 20, 190),
                           solid=True)
            if g["kind"] == "near" and m.cells[2] not in ("", 0):
                self._chip(painter, QRect(tile.right() - 44, tile.bottom() - 24, 0, 18), f"d {m.cells[2]}",
                           QColor(20, 20, 20, 190), solid=True)
            painter.setPen(text if not marked else muted)
            name = fm.elidedText(self.view.name_of(m), Qt.TextElideMode.ElideMiddle, THUMB)
            painter.drawText(QRect(tile.x(), tile.bottom() + 3, THUMB, CAPTION), Qt.AlignmentFlag.AlignHCenter, name)
        if more:
            last = tiles[-1] if tiles else QRect(r.x() + INFO_W, r.y() + PAD, 0, THUMB)
            box = QRect(last.right() + GAP, r.y() + PAD, THUMB, THUMB)
            painter.setPen(QPen(theme_color("line"), 1, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(box).adjusted(1, 1, -1, -1), 6, 6)
            painter.setPen(muted)
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, f"+{more}\nmore")
        # row separator
        painter.setPen(QPen(theme_color("line"), 1))
        painter.drawLine(r.bottomLeft(), r.bottomRight())
        painter.restore()

    def _thumb(self, painter, tile: QRect, member, pal):
        pix = self.view.thumb(member.key)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(theme_color("placeholder"))
        painter.drawRoundedRect(QRectF(tile), 6, 6)
        if pix is not None and not pix.isNull():
            sz = pix.size().scaled(tile.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRect(tile.x() + (tile.width() - sz.width()) // 2, tile.y() + (tile.height() - sz.height()) // 2,
                           sz.width(), sz.height())
            painter.drawPixmap(target, pix)
        painter.setPen(QPen(theme_color("line"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(tile).adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)

    @staticmethod
    def _box_rect(tile: QRect) -> QRect:
        return QRect(tile.right() - BOX - 6, tile.y() + 6, BOX, BOX)

    @staticmethod
    def _tickbox(painter, box: QRect, checked: bool):
        painter.setPen(QPen(QColor(255, 255, 255, 230), 2))
        painter.setBrush(theme_color("error") if checked else QColor(0, 0, 0, 120))
        painter.drawRoundedRect(QRectF(box), 4, 4)
        if checked:
            pen = QPen(QColor("white"), 2.4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            x, y = box.x(), box.y()
            painter.drawPolyline(QPolygonF([QPointF(x + 5, y + 10), QPointF(x + 9, y + 14), QPointF(x + 15, y + 6)]))

    @staticmethod
    def _chip(painter, at: QRect, text: str, color: QColor, solid=False):
        fm = painter.fontMetrics()
        w = fm.horizontalAdvance(text) + 12
        rect = QRectF(at.x(), at.y(), w, at.height() or 18)
        painter.setPen(Qt.PenStyle.NoPen)
        c = QColor(color)
        if not solid:
            c.setAlpha(45)
        painter.setBrush(c)
        painter.drawRoundedRect(rect, 9, 9)
        painter.setPen(QColor("white") if solid else color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # ------------------------------------------------------------------ input
    def editorEvent(self, event, model, option, index):
        if event.type() != QEvent.Type.MouseButtonRelease or event.button() != Qt.MouseButton.LeftButton:
            return False
        g = index.data(Qt.ItemDataRole.UserRole)
        tiles, _more = _tiles(option.rect, len(g["members"]))
        pos = event.position().toPoint()
        for tile, m in zip(tiles, g["members"]):
            if self._box_rect(tile).adjusted(-4, -4, 4, 4).contains(pos):
                self.view.toggle(m)
                return True
            if tile.contains(pos):
                self.view.open_member(m)
                return True
        return False

    def helpEvent(self, event, view, option, index):
        g = index.data(Qt.ItemDataRole.UserRole)
        if g is None:
            return False
        tiles, more = _tiles(option.rect, len(g["members"]))
        for tile, m in zip(tiles, g["members"]):
            if tile.contains(event.pos()):
                tip = (f"{self.view.name_of(m)}\nsplit: {self.view.split_of(m) or '—'}"
                       + (f"\ndistance: {m.cells[2]} / 256 bits" if g["kind"] == "near" else "")
                       + "\n\nclick: open in viewer · tick box: mark for trash")
                QToolTip.showText(event.globalPos(), tip, view)
                return True
        if more:
            QToolTip.showText(event.globalPos(), f"{more} more image(s) — see the Table tab", view)
            return True
        return False


class DupGalleryView(QListView):
    def __init__(self, dialog):
        super().__init__()
        self.dlg = dialog
        self.model_ = GroupModel(self)
        self.setModel(self.model_)
        self.setItemDelegate(GroupDelegate(self))
        self.setUniformItemSizes(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setMouseTracking(True)
        self.cache = LRUCache(capacity=2500)
        self.signals = ThumbSignals()
        self.signals.loaded.connect(self._on_thumb)
        self.generation = 0
        self._inflight: set[str] = set()
        self._priority = 0
        self._repaint = QTimer(self, singleShot=True, interval=30, timeout=self.viewport().update)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def _menu(self, pos):
        idx = self.indexAt(pos)
        g = idx.data(Qt.ItemDataRole.UserRole) if idx.isValid() else None
        if g is None:
            return
        tiles, _more = _tiles(self.visualRect(idx), len(g["members"]))
        for tile, m in zip(tiles, g["members"]):
            if tile.contains(pos):
                review_menu(self, self.dlg, m, self.viewport().mapToGlobal(pos))
                return

    # data -------------------------------------------------------------------------
    def set_groups(self, groups):
        self.model_.set_groups(groups)

    def is_marked(self, member) -> bool:
        return self.dlg.is_marked(member)

    def toggle(self, member):
        self.dlg.set_marked(member, not self.dlg.is_marked(member))
        self.viewport().update()

    def open_member(self, member):
        self.dlg.on_row_activated(member)

    def name_of(self, member) -> str:
        s = self.dlg.session.by_key.get(member.key)
        return s.name if s else member.key.rsplit("/", 1)[-1]

    def split_of(self, member) -> str:
        s = self.dlg.session.by_key.get(member.key)
        return s.split if s else ""

    # thumbnails -------------------------------------------------------------------
    def thumb(self, key):
        pix = self.cache.get(key)
        if pix is None and key not in self._inflight:
            s = self.dlg.session.by_key.get(key)
            if s is not None:
                self._inflight.add(key)
                self._priority += 1                   # newest first: what is on screen now
                QThreadPool.globalInstance().start(
                    ThumbTask(self.signals, self.dlg.session.adapter, s, self.generation, THUMB * 2), self._priority)
        return pix

    def _on_thumb(self, key, gen, img):
        self._inflight.discard(key)
        if gen != self.generation:
            return
        pix = QPixmap.fromImage(img) if not img.isNull() else QPixmap()
        pix.setDevicePixelRatio(1.0)
        self.cache.put(key, pix)
        self._repaint.start()

    def reset_thumbs(self):
        self.generation += 1
        self.cache.clear()
        self._inflight.clear()
