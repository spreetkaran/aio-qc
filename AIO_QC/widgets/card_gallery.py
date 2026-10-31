# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Reusable card grid for QC results that point at image regions.

    Card(row, panels=[Panel(key, focus_box, shapes=[Shape(points, color, dashed)])], chips=[(text, role)], lines=[...])

Each panel is a crop around `focus` (padded), with the shapes drawn on top (solid = labels, dashed = predictions).
Used by Model Check findings (1 panel: your box + the model's box), Duplicate Shapes (2 panels: kept | duplicate) and
Dim / Area (1 panel: the matching shape). Click a panel to open the image at that shape; tick the box to mark it
(marks = the dialog's table tick-boxes).

Scales to tens of thousands of cards: QListView in IconMode with uniform items (only visible cards paint),
crops decoded lazily with QImageReader clip rects (only the region is decoded) at card size, newest request
first, LRU cache.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtCore import (QAbstractListModel, QEvent, QModelIndex, QObject, QPointF, QRect, QRectF, QRunnable, QSize,
                          Qt, QThreadPool, QTimer, pyqtSignal)
from PyQt6.QtGui import (QColor, QFontMetrics, QImage, QImageIOHandler, QImageReader, QPainter, QPalette, QPen,
                         QPixmap, QPolygonF)
from PyQt6.QtWidgets import QAbstractItemView, QListView, QStyledItemDelegate, QToolTip

from AIO_QC.utils import LRUCache, theme_color

PANEL_W, PANEL_H = 176, 132
PAD = 8
CHIP_H = 20
LINE_H = 17
BOX = 20


@dataclass
class Shape:
    points: list                  # [(x, y), ...] (a polygon; rectangles as 4 corners)
    color: str
    dashed: bool = False
    width: float = 2.5


@dataclass
class Panel:
    key: str                      # sample key
    focus: tuple                  # (x1, y1, x2, y2) — region of interest
    shapes: list = field(default_factory=list)
    normalized: bool = False      # coordinates are 0..1 (YOLO) -> scaled by the image size
    caption: str = ""


@dataclass
class Card:
    row: object                   # ResultRow (marks, activation)
    panels: list
    chips: list = field(default_factory=list)     # [(text, theme role)]
    lines: list = field(default_factory=list)
    shape_index: int | None = None


def draw_review_badge(painter: QPainter, x: int, y: int, status: str | None):
    """Small ⚑ / ✓ badge (light disc under the icon: readable on any image)."""
    if status not in ("flag", "ok", "fixed"):
        return
    from AIO_QC.utils import newIcon
    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(255, 255, 255, 225))
    painter.drawEllipse(QRectF(x, y, 24, 24))
    newIcon({"flag": "review-flag", "ok": "review-ok", "fixed": "review-fixed"}[status]).paint(
        painter, QRect(x + 3, y + 3, 18, 18))
    painter.restore()


def review_menu(widget, dialog, row, global_pos):
    """Right-click on a card / tile: the same review actions as the result tables."""
    from PyQt6.QtWidgets import QMenu
    from AIO_QC.core.state import REVIEW_OK
    from AIO_QC.utils import newIcon
    m = QMenu(widget)
    m.addAction(newIcon("review-flag"), "Flag file", lambda: dialog.flag_files([row]))
    m.addAction(newIcon("review-ok"), "Mark file OK", lambda: dialog.session.set_review([row.key], REVIEW_OK))
    m.addAction(newIcon("review-clear"), "Clear review mark", lambda: dialog.session.set_review([row.key], None))
    m.addSeparator()
    m.addAction(newIcon("eye"), "Open", lambda: dialog.on_row_activated(row))
    m.exec(global_pos)


def box_points(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


# =============================================================================
# Crop rendering (thread pool)
# =============================================================================
class _Signals(QObject):
    done = pyqtSignal(object, int, QImage)


class PanelTask(QRunnable):
    def __init__(self, signals, ckey, panel: Panel, image_path, gen: int, scale: float):
        super().__init__()
        self.signals, self.ckey, self.panel, self.path, self.gen, self.scale = signals, ckey, panel, image_path, gen, scale

    def run(self):
        try:
            img = self._render()
        except Exception:
            img = QImage()
        self.signals.done.emit(self.ckey, self.gen, img)

    def _render(self) -> QImage:
        p = self.panel
        if self.path is None:
            return QImage()
        reader = QImageReader(str(self.path))
        reader.setAutoTransform(True)
        full = reader.size()
        rotated = reader.transformation() != QImageIOHandler.Transformation.TransformationNone
        src = None
        if rotated or not full.isValid():
            src = reader.read()
            if src.isNull():
                return QImage()
            W, H = src.width(), src.height()
        else:
            W, H = full.width(), full.height()
        sx, sy = (W, H) if p.normalized else (1.0, 1.0)
        x1, y1, x2, y2 = p.focus[0] * sx, p.focus[1] * sy, p.focus[2] * sx, p.focus[3] * sy
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        # pad to the panel aspect ratio, at least 35 % context
        tw, th = PANEL_W * self.scale, PANEL_H * self.scale
        cw, ch = max(bw * 1.7, 24), max(bh * 1.7, 24)
        if cw / ch < tw / th:
            cw = ch * tw / th
        else:
            ch = cw * th / tw
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rx1, ry1 = max(0, int(cx - cw / 2)), max(0, int(cy - ch / 2))
        rx2, ry2 = min(W, int(cx + cw / 2)), min(H, int(cy + ch / 2))
        rect = QRect(rx1, ry1, max(1, rx2 - rx1), max(1, ry2 - ry1))
        k = min(tw / rect.width(), th / rect.height(), 8.0)
        size = QSize(max(1, round(rect.width() * k)), max(1, round(rect.height() * k)))
        if src is not None:
            crop = src.copy(rect).scaled(size, Qt.AspectRatioMode.IgnoreAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
        else:
            reader.setClipRect(rect)
            reader.setScaledSize(size)
            crop = reader.read()
            if crop.isNull():
                return QImage()
        crop = crop.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(crop)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for s in p.shapes:
            pen = QPen(QColor(s.color), s.width * self.scale)
            if s.dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(QColor(0, 0, 0, 150))          # thin dark halo: visible on any background
            halo = QPen(QColor(0, 0, 0, 140), (s.width + 2) * self.scale)
            halo.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            poly = QPolygonF([QPointF((x * sx - rect.x()) * k, (y * sy - rect.y()) * k) for x, y in s.points])
            painter.setPen(halo)
            painter.drawPolygon(poly) if len(s.points) > 2 else painter.drawPolyline(poly)
            painter.setPen(pen)
            painter.drawPolygon(poly) if len(s.points) > 2 else painter.drawPolyline(poly)
        painter.end()
        return crop


# =============================================================================
# Model / delegate / view
# =============================================================================
class CardModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cards: list[Card] = []

    def set_cards(self, cards):
        self.beginResetModel()
        self.cards = list(cards)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.cards)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.UserRole:
            return self.cards[index.row()]
        return None


class CardDelegate(QStyledItemDelegate):
    def __init__(self, view):
        super().__init__(view)
        self.view = view

    def card_size(self) -> QSize:
        n = self.view.n_panels
        return QSize(n * PANEL_W + (n - 1) * 6 + 2 * PAD, PANEL_H + CHIP_H + 2 * LINE_H + 3 * PAD + 4)

    def sizeHint(self, option, index):
        return self.card_size()

    def _panel_rects(self, r: QRect, n: int):
        return [QRect(r.x() + PAD + i * (PANEL_W + 6), r.y() + PAD, PANEL_W, PANEL_H) for i in range(n)]

    @staticmethod
    def _box_rect(panel_rect: QRect) -> QRect:
        return QRect(panel_rect.right() - BOX - 6, panel_rect.y() + 6, BOX, BOX)

    def paint(self, painter: QPainter, option, index):
        card: Card = index.data(Qt.ItemDataRole.UserRole)
        if card is None:
            return
        pal = option.palette
        r = option.rect.adjusted(3, 3, -3, -3)
        marked = self.view.dialog.is_marked(card.row)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(theme_color("error") if marked else theme_color("line"), 2 if marked else 1))
        painter.setBrush(pal.color(QPalette.ColorRole.Base))
        painter.drawRoundedRect(QRectF(r), 8, 8)
        rects = self._panel_rects(r, len(card.panels))
        for i, (pr, panel) in enumerate(zip(rects, card.panels)):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(theme_color("placeholder"))
            painter.drawRoundedRect(QRectF(pr), 6, 6)
            pix = self.view.panel_pixmap(card, i)
            if pix is not None and not pix.isNull():
                sz = pix.deviceIndependentSize().toSize().scaled(pr.size(), Qt.AspectRatioMode.KeepAspectRatio)
                tgt = QRect(pr.x() + (pr.width() - sz.width()) // 2, pr.y() + (pr.height() - sz.height()) // 2,
                            sz.width(), sz.height())
                painter.drawPixmap(tgt, pix)
            if marked:
                painter.fillRect(pr, QColor(0, 0, 0, 90))
            if panel.caption:
                self._chip(painter, QRect(pr.x() + 6, pr.bottom() - 24, 0, 18), panel.caption, QColor(20, 20, 20, 200), True)
        if rects:
            draw_review_badge(painter, rects[0].x() + 6, rects[0].y() + 6,
                              self.view.dialog.session.review.status(card.row.key))
        if rects:
            self._tickbox(painter, self._box_rect(rects[-1]), marked)
        # chips
        x = r.x() + PAD
        y = r.y() + PAD + PANEL_H + 6
        for text, role in card.chips:
            x += self._chip(painter, QRect(x, y, 0, CHIP_H - 2), text, theme_color(role)) + 5
        # lines
        fm = QFontMetrics(option.font)
        painter.setFont(option.font)
        for k, line in enumerate(card.lines[:2]):
            painter.setPen(pal.color(QPalette.ColorRole.Text) if k == 0 else theme_color("muted"))
            painter.drawText(QRect(r.x() + PAD, y + CHIP_H + 2 + k * LINE_H, r.width() - 2 * PAD, LINE_H),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             fm.elidedText(line, Qt.TextElideMode.ElideMiddle, r.width() - 2 * PAD))
        painter.restore()

    @staticmethod
    def _chip(painter, at: QRect, text: str, color: QColor, solid=False) -> int:
        fm = painter.fontMetrics()
        w = fm.horizontalAdvance(text) + 12
        rect = QRectF(at.x(), at.y(), w, at.height() or 18)
        c = QColor(color)
        if not solid:
            c.setAlpha(45)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(c)
        painter.drawRoundedRect(rect, 9, 9)
        painter.setPen(QColor("white") if solid else color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        return int(w)

    @staticmethod
    def _tickbox(painter, box: QRect, checked: bool):
        painter.setPen(QPen(QColor(255, 255, 255, 230), 2))
        painter.setBrush(theme_color("error") if checked else QColor(0, 0, 0, 120))
        painter.drawRoundedRect(QRectF(box), 4, 4)
        if checked:
            pen = QPen(QColor("white"), 2.4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            x, y = box.x(), box.y()
            painter.drawPolyline(QPolygonF([QPointF(x + 5, y + 10), QPointF(x + 9, y + 14), QPointF(x + 15, y + 6)]))

    def editorEvent(self, event, model, option, index):
        if event.type() != QEvent.Type.MouseButtonRelease or event.button() != Qt.MouseButton.LeftButton:
            return False
        card: Card = index.data(Qt.ItemDataRole.UserRole)
        r = option.rect.adjusted(3, 3, -3, -3)
        rects = self._panel_rects(r, len(card.panels))
        pos = event.position().toPoint()
        if rects and self._box_rect(rects[-1]).adjusted(-4, -4, 4, 4).contains(pos):
            self.view.dialog.set_marked(card.row, not self.view.dialog.is_marked(card.row))
            self.view.viewport().update()
            return True
        for pr in rects:
            if pr.contains(pos):
                self.view.dialog.on_row_activated(card.row)
                return True
        return False

    def helpEvent(self, event, view, option, index):
        card: Card = index.data(Qt.ItemDataRole.UserRole)
        if card is None:
            return False
        QToolTip.showText(event.globalPos(), "\n".join(card.lines) +
                          "\n\nclick the image: open it · tick box: mark", view)
        return True


class CardView(QListView):
    """Grid of Cards. `dialog` must provide is_marked / set_marked / on_row_activated and a session."""

    def __init__(self, dialog, n_panels: int = 1):
        super().__init__()
        self.dialog, self.n_panels = dialog, n_panels
        self.model_ = CardModel(self)
        self.setModel(self.model_)
        self.delegate = CardDelegate(self)
        self.setItemDelegate(self.delegate)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setUniformItemSizes(True)
        self.setSpacing(4)
        self.setGridSize(self.delegate.card_size() + QSize(8, 8))
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setMouseTracking(True)
        self.cache = LRUCache(capacity=3000)
        self.signals = _Signals()
        self.signals.done.connect(self._on_done)
        self.gen = 0
        self._inflight: set = set()
        self._priority = 0
        self._repaint = QTimer(self, singleShot=True, interval=30, timeout=self.viewport().update)
        dialog.on_marks_changed(self._repaint.start)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def _menu(self, pos):
        idx = self.indexAt(pos)
        card = idx.data(Qt.ItemDataRole.UserRole) if idx.isValid() else None
        if card is not None:
            review_menu(self, self.dialog, card.row, self.viewport().mapToGlobal(pos))

    def set_cards(self, cards):
        self.gen += 1
        self._inflight.clear()
        self.cache.clear()
        self.model_.set_cards(cards)

    def _image_path(self, key):
        s = self.dialog.session.by_key.get(key)
        if s is None:
            return None
        if s.image_path is not None and s.image_path.exists():
            return s.image_path
        try:
            return self.dialog.session.adapter.load(s).image_path
        except Exception:
            return None

    def panel_pixmap(self, card: Card, i: int):
        ck = (id(card), i)
        pix = self.cache.get(ck)
        if pix is None and ck not in self._inflight:
            self._inflight.add(ck)
            self._priority += 1
            panel = card.panels[i]
            QThreadPool.globalInstance().start(
                PanelTask(self.signals, ck, panel, self._image_path(panel.key), self.gen, self.devicePixelRatioF()),
                self._priority)
        return pix

    def _on_done(self, ck, gen, img):
        self._inflight.discard(ck)
        if gen != self.gen:
            return
        pix = QPixmap.fromImage(img) if not img.isNull() else QPixmap()
        pix.setDevicePixelRatio(self.devicePixelRatioF())
        self.cache.put(ck, pix)
        self._repaint.start()
