# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Class Gallery — review a classification dataset one class at a time, as a wall of images: the odd ones stand out.

    left    every class with its colour dot, image count, how much of it is reviewed (bar) and ✓ when signed off
    centre  the class's images as tiles (64 … 320 px); Compare with … shows a second class beside it — drag tiles
            between the two (or onto a class on the left) to move them
    order   Most suspicious first: the classifier's findings (sure of another class, then another class, then its
            own class but unsure — from Model Check's cached predictions), then the least typical by look (vectors
            of a Visual Similarity run); Model: least sure first; Least typical first; Grouped by look; file order;
            random
    badges  ✓ / ⚑ review mark · "→ cat 93 %" the model's other class · "? 41 %" unsure · "≈ cat" a near-identical
            image in another class · the class (lists that mix classes)
    keys    1 … 9 that class · C pick a class · M the model's class · X flag (Shift+X with a note) · S OK · U clear
            Space: every tile on screen you did not mark becomes OK, then the next screen · Del trash ·
            Enter / double-click open in the viewer · hold P peek at full size · Ctrl+Z / Ctrl+Y undo / redo
    views   Confusions (the confusion matrix's cells as image lists), Look-alikes in other classes, Sample check
            (a random sample of the class; flag or move the wrong ones, then Sign off with the estimated number of
            wrong images in the whole class)

Built for big datasets: a uniform list view with a painted tile per item (no widget per image), thumbnails decoded at
tile size in the background (JPEG scaled decode), newest request first, bounded memory (LRU).
"""
from __future__ import annotations

import time
from datetime import datetime

from PyQt6.QtCore import (QAbstractListModel, QMimeData, QModelIndex, QObject, QPoint, QRect, QRectF, QRunnable,
                          QSize, Qt, QThreadPool, QTimer, pyqtSignal)
from PyQt6.QtGui import (QColor, QDrag, QFont, QFontMetrics, QImage, QImageReader, QKeySequence, QPainter, QPen,
                         QPixmap, QShortcut)
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QHBoxLayout, QInputDialog, QLabel,
                             QListView, QListWidget, QListWidgetItem, QMenu, QMessageBox, QSlider, QSplitter, QStyle,
                             QStyledItemDelegate, QToolButton, QVBoxLayout, QWidget)

from AIO_QC.core import cls_review as CR
from AIO_QC.core.dataset import CLASSIFICATION_TYPES
from AIO_QC.core.diag import ignored as _ignored
from AIO_QC.core.state import REVIEW_FIXED, REVIEW_FLAG, REVIEW_OK
from AIO_QC.utils import LRUCache, newButton, style as apply_style

MIME = "application/x-aioqc-image-keys"
NO_CLASS = "\x00none"                 # the list entry for images without a class
SORT_TIPS = {
    CR.SUSPICIOUS: "The classifier's findings first (sure of another class, another class, unsure), then the least "
                   "typical by look — needs a Model Check and / or a Visual Similarity run",
    CR.LEAST_SURE: "The images the classifier is least sure belong to their class first (needs a Model Check run)",
    CR.LEAST_TYPICAL: "The images that look least like the rest of the class first (needs a Visual Similarity run)",
    CR.BY_LOOK: "Look-alike images next to each other: odd ones stand out at group borders (Visual Similarity)",
    CR.FILE_ORDER: "As in the file list",
    CR.RANDOM: "Shuffled — for an unbiased look",
}
NAME_ROLE, COUNT_ROLE, DONE_ROLE, SIGN_ROLE = (Qt.ItemDataRole.UserRole + i for i in range(4))


def _tok():
    from AIO_QC.widgets.design import tokens
    return tokens()


# =============================================================================
# Thumbnails
# =============================================================================
class _Sig(QObject):
    loaded = pyqtSignal(str, int, QImage)


class ThumbTask(QRunnable):
    """The whole image at tile size (JPEG decodes 1/2 … 1/8 scale natively), EXIF rotation applied."""

    def __init__(self, sig, key, path, tile: int, gen: int):
        super().__init__()
        self.sig, self.key, self.path, self.tile, self.gen = sig, key, path, tile, gen

    def run(self):
        img = QImage()
        try:
            if self.path is not None:
                r = QImageReader(str(self.path))
                r.setAutoTransform(True)
                sz = r.size()
                if sz.isValid() and max(sz.width(), sz.height()) > self.tile * 2:
                    k = self.tile * 2 / max(sz.width(), sz.height())         # 2× for sharp tiles on HiDPI
                    r.setScaledSize(QSize(max(1, round(sz.width() * k)), max(1, round(sz.height() * k))))
                img = r.read()
        except Exception as exc:
            _ignored(exc)
            img = QImage()
        self.sig.loaded.emit(self.key, self.gen, img)


# =============================================================================
# Model + views
# =============================================================================
class KeyModel(QAbstractListModel):
    """Rows = image keys. Removing / inserting keeps the selection of the rows that stay."""

    def __init__(self, gallery):
        super().__init__(gallery)
        self.g = gallery
        self.keys: list[str] = []
        self.pos: dict[str, int] = {}

    def set_keys(self, keys):
        self.beginResetModel()
        self.keys = list(dict.fromkeys(keys))
        self.pos = {k: i for i, k in enumerate(self.keys)}
        self.endResetModel()

    def _reindex(self):
        self.pos = {k: i for i, k in enumerate(self.keys)}

    def remove(self, keys) -> int:
        rows = sorted((self.pos[k] for k in set(keys) if k in self.pos), reverse=True)
        if not rows:
            return 0
        if len(rows) > 300:
            gone = set(keys)
            self.set_keys([k for k in self.keys if k not in gone])
            return len(rows)
        for r in rows:
            self.beginRemoveRows(QModelIndex(), r, r)
            del self.keys[r]
            self.endRemoveRows()
        self._reindex()
        return len(rows)

    def insert_front(self, keys):
        new = [k for k in dict.fromkeys(keys) if k not in self.pos]
        if not new:
            return
        self.beginInsertRows(QModelIndex(), 0, len(new) - 1)
        self.keys[0:0] = new
        self.endInsertRows()
        self._reindex()

    def rename(self, ren: dict):
        hit = False
        for old, new in ren.items():
            i = self.pos.get(old)
            if i is not None and new not in self.pos:
                self.keys[i] = new
                hit = True
        if hit:
            self._reindex()
            self.dataChanged.emit(self.index(0), self.index(max(0, len(self.keys) - 1)))

    def changed(self, keys):
        for k in keys:
            i = self.pos.get(k)
            if i is not None:
                idx = self.index(i)
                self.dataChanged.emit(idx, idx)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.keys)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.keys):
            return None
        k = self.keys[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return k
        if role == Qt.ItemDataRole.DecorationRole:
            return self.g.pixmap_for(k)
        if role == Qt.ItemDataRole.DisplayRole:
            s = self.g.session.by_key.get(k)
            return s.name if s is not None else k
        if role == Qt.ItemDataRole.ToolTipRole:
            return self.g.tooltip(k)
        return None


class Grid(QListView):
    """One wall of tiles. Tiles drag out (onto the other grid or a class on the left); a grid that shows a class
    takes drops."""

    def __init__(self, gallery):
        super().__init__()
        self.g = gallery
        self.cls = ""                                    # the class shown ("" = a list of images)
        self.m = KeyModel(gallery)
        self.setViewMode(QListView.ViewMode.ListMode)    # list layout: O(1) per item
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setUniformItemSizes(True)
        self.setMouseTracking(True)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setItemDelegate(TileDelegate(gallery))
        self.setModel(self.m)
        self.doubleClicked.connect(lambda idx: gallery.open_key(self.m.keys[idx.row()]))
        self.selectionModel().selectionChanged.connect(lambda *_: gallery._update_status())

    def selected_keys(self) -> list[str]:
        rows = sorted({i.row() for i in self.selectionModel().selectedIndexes()})
        return [self.m.keys[r] for r in rows if r < len(self.m.keys)]

    def focusInEvent(self, e):
        self.g.active = self
        self.g._update_status()
        super().focusInEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_P and not e.isAutoRepeat():
            self.g.peek_press()
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_P and not e.isAutoRepeat():
            self.g.peek_release()
            return
        super().keyReleaseEvent(e)

    # ---- drag out, drop in
    def startDrag(self, actions):
        keys = self.selected_keys()
        if not keys:
            return
        mime = QMimeData()
        mime.setData(MIME, "\n".join(keys).encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(self.g.drag_pixmap(keys))
        drag.exec(Qt.DropAction.MoveAction)

    def _ok(self, e) -> bool:
        return e.mimeData().hasFormat(MIME) and e.source() is not self and bool(self.cls) and self.cls != NO_CLASS

    def dragEnterEvent(self, e):
        e.acceptProposedAction() if self._ok(e) else e.ignore()

    def dragMoveEvent(self, e):
        e.acceptProposedAction() if self._ok(e) else e.ignore()

    def dropEvent(self, e):
        if not self._ok(e):
            e.ignore()
            return
        keys = bytes(e.mimeData().data(MIME)).decode("utf-8").split("\n")
        src = e.source().cls if isinstance(e.source(), Grid) else ""
        e.acceptProposedAction()
        self.g.move_keys(keys, self.cls, source=src)


class ClassList(QListWidget):
    """The classes on the left: click shows one; tiles dropped on one move there."""

    def __init__(self, gallery):
        super().__init__()
        self.g = gallery
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setItemDelegate(ClassRowDelegate(self))
        self.setUniformItemSizes(True)
        self.setMouseTracking(True)

    def _target(self, e):
        it = self.itemAt(e.position().toPoint())
        name = it.data(NAME_ROLE) if it is not None else None
        return name if name and name != NO_CLASS else None

    def dragEnterEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasFormat(MIME) else e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(MIME) and self._target(e):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        name = self._target(e)
        if not name or not e.mimeData().hasFormat(MIME):
            e.ignore()
            return
        keys = bytes(e.mimeData().data(MIME)).decode("utf-8").split("\n")
        src = e.source().cls if isinstance(e.source(), Grid) else ""
        e.acceptProposedAction()
        self.g.move_keys(keys, name, source=src)


class ClassRowDelegate(QStyledItemDelegate):
    """dot · name · count, a thin bar of how much is reviewed, ✓ when signed off."""

    def sizeHint(self, opt, idx):
        return QSize(180, 38)

    def paint(self, p, opt, idx):
        t = _tok()
        r = opt.rect.adjusted(4, 2, -4, -2)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if opt.state & QStyle.StateFlag.State_Selected:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(t["accent_soft"]))
            p.drawRoundedRect(QRectF(r), 6, 6)
        elif opt.state & QStyle.StateFlag.State_MouseOver:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(t["hover"]))
            p.drawRoundedRect(QRectF(r), 6, 6)
        name = idx.data(NAME_ROLE)
        g = self.parent().g
        col = g.class_color(name) if name != NO_CLASS else QColor(t["faint"])
        cy = r.top() + 13
        p.setPen(QPen(col.darker(150), 1))
        p.setBrush(col)
        p.drawEllipse(QRectF(r.left() + 6, cy - 4.5, 9, 9))
        count = str(idx.data(COUNT_ROLE) or "")
        f = QFont(opt.font)
        fm = QFontMetrics(f)
        cw = fm.horizontalAdvance(count)
        sign = idx.data(SIGN_ROLE)
        sw = fm.horizontalAdvance("✓ ") if sign else 0
        p.setPen(QColor(t["muted"]))
        p.drawText(QRect(r.right() - cw - 6, r.top(), cw + 2, 26), int(Qt.AlignmentFlag.AlignVCenter), count)
        if sign:
            p.setPen(QColor(t["ok"]))
            p.drawText(QRect(r.right() - cw - 6 - sw, r.top(), sw, 26), int(Qt.AlignmentFlag.AlignVCenter), "✓")
        text = "No class" if name == NO_CLASS else str(name)
        num = g.key_number(name)
        if num:
            text = f"{num}  {text}"
        bold = bool(opt.state & QStyle.StateFlag.State_Selected)
        f.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
        p.setFont(f)
        p.setPen(QColor(t["text"]))
        x = r.left() + 22
        p.drawText(QRect(x, r.top(), r.right() - cw - sw - 12 - x, 26), int(Qt.AlignmentFlag.AlignVCenter),
                   QFontMetrics(f).elidedText(text, Qt.TextElideMode.ElideRight, r.right() - cw - sw - 12 - x))
        done = float(idx.data(DONE_ROLE) or 0.0)                    # reviewed share
        bar = QRectF(x, r.bottom() - 7, r.right() - 6 - x, 3)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(t["track"]))
        p.drawRoundedRect(bar, 1.5, 1.5)
        if done > 0:
            p.setBrush(QColor(t["ok"]))
            p.drawRoundedRect(QRectF(bar.left(), bar.top(), max(3.0, bar.width() * min(1.0, done)), 3), 1.5, 1.5)
        p.restore()


class TileDelegate(QStyledItemDelegate):
    """A tile: the image (fit), its review mark, the model's / look-alike badges and the name — painted directly."""

    def __init__(self, gallery):
        super().__init__(gallery)
        self.g = gallery

    def sizeHint(self, option, index):
        return self.g.grid_size

    def _chip(self, p, x, y, text, bg, fg=QColor("white"), right=False, f=None):
        fm = QFontMetrics(f or p.font())
        w, h = fm.horizontalAdvance(text) + 10, fm.height() + 2
        rr = QRectF(x - w if right else x, y, w, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(rr, h / 2, h / 2)
        p.setPen(fg)
        p.drawText(rr, int(Qt.AlignmentFlag.AlignCenter), text)
        return rr

    def paint(self, p: QPainter, opt, index):
        g, t = self.g, _tok()
        r = opt.rect
        tile = g.tile
        key = index.data(Qt.ItemDataRole.UserRole)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        sel = bool(opt.state & QStyle.StateFlag.State_Selected)
        if sel:
            p.setPen(QPen(QColor(t["accent"]), 2))
            p.setBrush(QColor(t["accent_soft"]))
            p.drawRoundedRect(QRectF(r).adjusted(2, 2, -2, -2), 7, 7)
        elif opt.state & QStyle.StateFlag.State_MouseOver:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(t["hover"]))
            p.drawRoundedRect(QRectF(r).adjusted(2, 2, -2, -2), 7, 7)
        th = g.tile_h
        box = QRect(r.x() + (r.width() - tile) // 2, r.y() + 6, tile, th)        # 4:3 — most photos
        pix = index.data(Qt.ItemDataRole.DecorationRole)
        img = QRect(box)
        if isinstance(pix, QPixmap) and not pix.isNull():
            dpr = pix.devicePixelRatio() or 1.0
            w, h = pix.width() / dpr, pix.height() / dpr
            s = min(tile / max(w, 1), th / max(h, 1))
            w, h = int(w * s), int(h * s)
            img = QRect(box.x() + (tile - w) // 2, box.y() + (th - h) // 2, w, h)
            p.drawPixmap(img, pix)
        f = QFont(opt.font)
        f.setPixelSize(11 if tile < 140 else 12)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        st = g.session.review.status(key) if key else None
        if st is not None:                                         # review mark, top-left
            col = QColor(t[{REVIEW_OK: "ok", REVIEW_FIXED: "info"}.get(st, "warn")])
            p.setPen(QPen(QColor("white"), 1.5))
            p.setBrush(col)
            c = QRectF(img.left() + 5, img.top() + 5, 16, 16)
            p.drawEllipse(c)
            p.setPen(QColor("white"))
            p.drawText(c, int(Qt.AlignmentFlag.AlignCenter), {REVIEW_OK: "✓", REVIEW_FIXED: "✓"}.get(st, "!"))
        y = img.top() + 5
        mv = g.model_badge(key)                                    # the classifier, top-right
        if mv is not None:
            text, tone = mv
            bg = QColor(t[tone]) if tone in t else QColor(40, 40, 46, 225)
            bg.setAlpha(235)
            rr = self._chip(p, img.right() - 4, y, text, bg, right=True, f=f)
            y = int(rr.bottom()) + 3
        la = g.lookalike_of.get(key)
        if la is not None:
            other = la[0]
            ocls = ", ".join(g.session.tags_of(other)) or "no class"
            self._chip(p, img.right() - 4, y, f"≈ {ocls}", QColor(t["info"]), right=True, f=f)
        if g.show_classes():                                       # lists that mix classes: the class, bottom-left
            tags = g.session.tags_of(key) if key else []
            text = ", ".join(tags) if tags else "no class"
            fm = QFontMetrics(f)
            text = fm.elidedText(text, Qt.TextElideMode.ElideRight, max(30, img.width() - 30))
            col = g.class_color(tags[0]) if tags else QColor(t["faint"])
            w, h = fm.horizontalAdvance(text) + 22, fm.height() + 2
            rr = QRectF(img.left() + 4, img.bottom() - h - 4, w, h)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(20, 20, 24, 210))
            p.drawRoundedRect(rr, h / 2, h / 2)
            p.setBrush(col)
            p.drawEllipse(QRectF(rr.left() + 6, rr.center().y() - 4, 8, 8))
            p.setPen(QColor("white"))
            p.drawText(rr.adjusted(17, 0, -4, 0), int(Qt.AlignmentFlag.AlignVCenter), text)
        if g.show_names:
            nf = QFont(opt.font)
            nf.setPixelSize(11)
            p.setFont(nf)
            p.setPen(QColor(t["accent_text"] if sel else t["muted"]))
            tr = QRect(r.x() + 4, box.bottom() + 3, r.width() - 8, r.bottom() - box.bottom() - 3)
            p.drawText(tr, int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                       QFontMetrics(nf).elidedText(str(index.data(Qt.ItemDataRole.DisplayRole) or ""),
                                                   Qt.TextElideMode.ElideMiddle, tr.width()))
        p.restore()


# =============================================================================
# The window
# =============================================================================
class ClassGalleryDialog(QDialog):
    TITLE, ICON = "Class Gallery", ""
    SUPPORTED = CLASSIFICATION_TYPES

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        s = self.session
        self.setWindowTitle(self.TITLE)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        from AIO_QC.utils.qt_utils import fit_size
        fit_size(self, "large")
        st = app.settings
        self.cache = LRUCache(capacity=3000, max_bytes=500 * 2 ** 20)
        self.sig = _Sig()
        self.sig.loaded.connect(self._on_thumb)
        self.gen = 0
        self._prio = 0
        self._inflight: set = set()
        self.show_names = st.value("cls_gallery/names", True, type=bool)
        self.preds: dict = {}                       # key -> [(class, conf)] (cached classifier predictions)
        self.model_name = ""
        self.lookalike_of: dict = {}                # key -> (other key, similarity)
        self.view = {"kind": "class", "cls": ""}
        self.sample: dict | None = None             # the running Sample check
        self.active: Grid | None = None
        self._peek = None
        self._peek_t = 0.0
        self._seed = int(time.time()) & 0xFFFF
        self._dirty: set = set()
        self._gone: set = set()
        self._sync = QTimer(self, singleShot=True, interval=0, timeout=self._sync_now)
        self._counts_timer = QTimer(self, singleShot=True, interval=250, timeout=self._fill_classes)

        # ---- top bar
        self.split_box = QComboBox()
        self.split_box.addItem("All splits", "")
        for sp in s.splits:
            self.split_box.addItem(sp, sp)
        self.split_box.setVisible(bool(s.splits))
        self.split_box.currentIndexChanged.connect(lambda _i: (self._fill_classes(), self._reload_view()))
        self.sort_box = QComboBox()
        for name in CR.SORTS:
            self.sort_box.addItem(name)
            self.sort_box.setItemData(self.sort_box.count() - 1, SORT_TIPS[name], Qt.ItemDataRole.ToolTipRole)
        self.sort_box.currentIndexChanged.connect(lambda _i: self._reload_view(resort=True))
        self.compare_box = QComboBox()
        self.compare_box.setToolTip("Show a second class beside this one — drag tiles between them")
        self.compare_box.currentIndexChanged.connect(lambda _i: self._set_compare())
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(64, 320)
        self.size_slider.setValue(int(st.value("cls_gallery/tile", 150)))
        self.size_slider.setFixedWidth(130)
        self.size_slider.setToolTip("Tile size — small tiles: more images per screen")
        self.size_slider.valueChanged.connect(self._resize_tiles)
        self.names_cb = QCheckBox("Names")
        self.names_cb.setChecked(self.show_names)
        self.names_cb.toggled.connect(self._toggle_names)
        self.model_lbl = QLabel("")
        apply_style(self.model_lbl, "color:@muted;")
        top = QHBoxLayout()
        for w in (self.split_box, QLabel("Order:"), self.sort_box, QLabel("Compare with:"), self.compare_box,
                  QLabel("Size:"), self.size_slider, self.names_cb):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.model_lbl)
        top.addWidget(newButton("Model Check…", slot=lambda: self.app.open_tool("model_check"),
                                tip="Run your classifier over the dataset once — its findings then sort and badge "
                                    "the tiles here"))
        top.addWidget(newButton("Visual Similarity…", slot=lambda: self.app.open_tool("similarity"),
                                tip="Compute look vectors once — for Least typical, Grouped by look and Look-alikes"))

        # ---- left: classes + views
        self.classes = ClassList(self)
        self.classes.currentItemChanged.connect(self._class_clicked)
        self.conf_btn = QToolButton()
        self.conf_btn.setText("Confusions")
        self.conf_btn.setToolTip("The classifier's mix-ups (class → the model's class) as image lists")
        self.conf_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.conf_menu = QMenu(self.conf_btn)
        self.conf_menu.aboutToShow.connect(self._fill_confusions)
        self.conf_btn.setMenu(self.conf_menu)
        self.look_btn = newButton("Look-alikes", slot=self.show_lookalikes,
                                  tip="Images of this class with a near-identical image in another class")
        self.sample_btn = newButton("Sample check…", slot=self.start_sample,
                                    tip="Review a random sample of this class, then sign it off with the estimated "
                                        "number of wrong images in the whole class")
        self.sign_btn = newButton("Sign off", slot=self.sign_off, tip="Record the sample's result for this class")
        self.sign_btn.setEnabled(False)
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        lt = QLabel("<b>Classes</b> — drop tiles on one to move them")
        lt.setWordWrap(True)
        left.addWidget(lt)
        left.addWidget(self.classes, 1)
        row1 = QHBoxLayout()
        row1.addWidget(self.conf_btn)
        row1.addWidget(self.look_btn)
        left.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(self.sample_btn)
        row2.addWidget(self.sign_btn)
        left.addLayout(row2)
        leftw = QWidget()
        leftw.setLayout(left)
        leftw.setMinimumWidth(190)

        # ---- centre: one or two grids
        self.grids = [Grid(self), Grid(self)]
        self.heads = [QLabel(""), QLabel("")]
        cols = []
        for gr, hd in zip(self.grids, self.heads):
            hd.setTextFormat(Qt.TextFormat.RichText)
            v = QVBoxLayout()
            v.setContentsMargins(0, 0, 0, 0)
            v.addWidget(hd)
            v.addWidget(gr, 1)
            w = QWidget()
            w.setLayout(v)
            cols.append(w)
        self.right_col = cols[1]
        self.right_col.hide()
        self.back_btn = newButton("◀ Back to the class", slot=self._back)
        self.back_btn.hide()
        centre = QSplitter(Qt.Orientation.Horizontal)
        centre.addWidget(cols[0])
        centre.addWidget(cols[1])
        self.active = self.grids[0]
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(leftw)
        split.addWidget(centre)
        split.setStretchFactor(1, 1)
        split.setSizes([220, 1000])
        self.centre = centre

        # ---- bottom
        self.status = QLabel("")
        self.status.setWordWrap(True)
        bar = QHBoxLayout()
        bar.addWidget(self.back_btn)
        bar.addWidget(self.status, 1)
        for text, slot, tip in (
                ("Class… (C)", self.pick_class, "Give the selected images a class"),
                ("Model's class (M)", self.use_model, "The selected images get the classifier's top class"),
                ("OK (S)", lambda: self.mark(REVIEW_OK), "Mark the selected images OK"),
                ("Flag (X)", lambda: self.mark(REVIEW_FLAG), "Flag the selected images"),
                ("Issues (I)", self.send_issues, "Set the selected images aside in the issues folder, by class "
                 "(issues/<class>/) — Ctrl+Z puts them back. Class-folder datasets."),
                ("Trash (Del)", self.trash, "Move the selected images to the dataset's trash (undo from the bar)"),
                ("Rest OK + next (Space)", self.accept_visible,
                 "Every image on screen you did not mark becomes OK, then the next screen")):
            b = newButton(text, slot=slot, tip=tip)
            if text.startswith("Issues"):
                self.issues_btn = b
                b.setVisible(self.session.type.value == "cls_folders")
            bar.addWidget(b)
        hint = QLabel("1 … 9 class · C pick · M model's class · X flag · Shift+X note · S OK · U clear · Space rest "
                      "OK + next · " + ("I issues · Shift+I issues + note · " if self.session.type.value ==
                                        "cls_folders" else "") +
                      "Del trash · Enter open · hold P peek · drag tiles onto a class · Ctrl+Z undo")
        hint.setWordWrap(True)
        apply_style(hint, "color:@faint;")
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(split, 1)
        lay.addLayout(bar)
        lay.addWidget(hint)
        self._apply_tile_size()
        self._keys()
        from PyQt6.QtWidgets import QPushButton
        for b in self.findChildren(QPushButton):          # Enter opens the image, never a button
            b.setAutoDefault(False)
            b.setDefault(False)

        s.annotationsChanged.connect(self._on_changed)
        s.reviewChanged.connect(self._on_review)
        s.samplesRemoved.connect(self._on_removed)
        s.samplesAdded.connect(self._on_changed)
        s.keysRenamed.connect(self._on_renamed)
        s.indexChanged.connect(self._counts_timer.start)
        self._fill_classes()
        self._fill_compare()
        first = next((self.classes.item(i) for i in range(self.classes.count())), None)
        if first is not None:
            self.classes.setCurrentItem(first)
        self.load_predictions()

    # ------------------------------------------------------------------ data
    def _split_ok(self, k) -> bool:
        sp = self.split_box.currentData()
        if not sp:
            return True
        x = self.session.by_key.get(k)
        return x is not None and x.split == sp

    def members(self, cls) -> list[str]:
        s = self.session
        if cls == NO_CLASS:
            return [x.key for x in s.samples if not s.tags_of(x.key) and self._split_ok(x.key)]
        return [x.key for x in s.samples if cls in s.tags_of(x.key) and self._split_ok(x.key)]

    def in_class(self, k, cls) -> bool:
        s = self.session
        if k not in s.by_key or not self._split_ok(k):
            return False
        tags = s.tags_of(k)
        return not tags if cls == NO_CLASS else cls in tags

    def class_color(self, name) -> QColor:
        cols = getattr(self.app, "label_colors", None)
        if cols is None:
            return QColor(128, 128, 128)
        if name not in cols:
            from AIO_QC import utils
            utils.update_label_colors(name, cols)
        return QColor(cols.get(name, QColor(128, 128, 128)))

    def key_number(self, name) -> int:
        """1 … 9: the key that gives the selected images this class (the Classes order)."""
        try:
            i = self.session.cls_classes().index(name)
        except ValueError:
            return 0
        return i + 1 if i < 9 else 0

    def show_classes(self) -> bool:
        return self.view["kind"] == "list" or self.session.cls_mode == "multi"

    def _vectors(self):
        emb = getattr(self.app, "embed_index", None) or {}
        rows = {k: r for (k, i), r in (emb.get("map") or {}).items() if i == -1}
        return rows, emb.get("vecs")

    def ordered(self, keys) -> list[str]:
        rows, vecs = self._vectors()
        mode = self.sort_box.currentText()
        return CR.order_keys(keys, mode, self.session.tags_of, self.preds, rows, vecs, seed=self._seed)

    def model_badge(self, key):
        p = self.preds.get(key)
        if not p:
            return None
        tier, _w, top, conf, own = CR.model_view(self.session.tags_of(key), p)
        if tier >= 2:
            return f"→ {top} {conf:.0%}", ("error" if tier == 3 else "warn")
        if tier == 1:
            return f"? {own:.0%}", "dark"
        return None

    def tooltip(self, k) -> str:
        s = self.session
        x = s.by_key.get(k)
        lines = [x.display if x is not None and hasattr(x, "display") else k,
                 "class: " + (", ".join(s.tags_of(k)) or "none")]
        p = self.preds.get(k)
        if p:
            lines.append(f"{self.model_name or 'classifier'}: " + " · ".join(f"{c} {v:.0%}" for c, v in p[:3]))
        la = self.lookalike_of.get(k)
        if la:
            o = s.by_key.get(la[0])
            lines.append(f"≈ {o.name if o else la[0]} ({', '.join(s.tags_of(la[0])) or 'no class'}) — "
                         f"{la[1]:.0%} alike")
        rv = s.review.get(k)
        if rv:
            lines.append(f"review: {rv.get('status')}" + (f" — {rv.get('note')}" if rv.get("note") else ""))
        return "\n".join(lines)

    # ------------------------------------------------------------------ predictions
    def load_predictions(self):
        from .cls_model_check_dialog import cached_predictions
        from .inference_service import FuncWorker
        samples = list(self.session.samples)
        self.model_lbl.setText("reading the classifier's results …")
        w = FuncWorker(lambda: cached_predictions(self.app, samples), self)
        w.done.connect(self._on_preds)
        w.failed.connect(lambda e: self.model_lbl.setText(f"classifier results: {e}"))
        self._preds_worker = w
        w.start()

    def _on_preds(self, res):
        name, preds = res
        self.model_name, self.preds = name, preds
        n = len(preds)
        if not name:
            self.model_lbl.setText("No classifier — add one for model sorting and badges")
        elif not n:
            self.model_lbl.setText(f"{name}: not run yet — Model Check fills it")
        else:
            self.model_lbl.setText(f"{name}: {n:,} image(s) checked")
        after = getattr(self, "_after_preds", None)
        self._after_preds = None
        if after is not None:
            after()
        elif self.sort_box.currentText() in (CR.SUSPICIOUS, CR.LEAST_SURE):
            self._reload_view(resort=True)
        else:
            for gr in self.grids:
                gr.viewport().update()

    # ------------------------------------------------------------------ left list
    def _fill_classes(self):
        s = self.session
        rv = s.review
        counts, done = {}, {}
        for x in s.samples:
            if not self._split_ok(x.key):
                continue
            tags = s.tags_of(x.key) or [NO_CLASS]
            ok = rv.status(x.key) is not None
            for c in tags:
                counts[c] = counts.get(c, 0) + 1
                done[c] = done.get(c, 0) + ok
        signed = (s.cls_settings().get("signoff") or {})
        cur = self.classes.currentItem().data(NAME_ROLE) if self.classes.currentItem() else self.view.get("cls")
        self.classes.blockSignals(True)
        self.classes.clear()
        names = [c for c in s.cls_classes() if counts.get(c)] + ([NO_CLASS] if counts.get(NO_CLASS) else [])
        names += [c for c in s.cls_classes() if not counts.get(c)]
        for c in names:
            it = QListWidgetItem("No class" if c == NO_CLASS else c)
            it.setData(NAME_ROLE, c)
            n = counts.get(c, 0)
            it.setData(COUNT_ROLE, f"{n:,}")
            it.setData(DONE_ROLE, (done.get(c, 0) / n) if n else 0.0)
            sg = signed.get(c)
            it.setData(SIGN_ROLE, bool(sg))
            tip = f"{it.text()}: {n:,} image(s), {done.get(c, 0):,} reviewed"
            if sg:
                tip += (f"\n✓ signed off {sg.get('when', '')}: {sg.get('wrong', 0)} wrong in a sample of "
                        f"{sg.get('checked', 0)} — about {sg.get('est', 0):,} wrong in the class "
                        f"({sg.get('est_lo', 0):,} … {sg.get('est_hi', 0):,})")
            it.setToolTip(tip)
            self.classes.addItem(it)
            if c == cur:
                self.classes.setCurrentItem(it)
        self.classes.blockSignals(False)

    def _fill_compare(self):
        self.compare_box.blockSignals(True)
        cur = self.compare_box.currentData()
        self.compare_box.clear()
        self.compare_box.addItem("—", "")
        for c in self.session.cls_classes():
            self.compare_box.addItem(c, c)
        i = self.compare_box.findData(cur)
        self.compare_box.setCurrentIndex(max(0, i))
        self.compare_box.blockSignals(False)

    def _class_clicked(self, it, _prev=None):
        if it is not None:
            self.show_class(it.data(NAME_ROLE))

    # ------------------------------------------------------------------ views
    def show_class(self, cls: str):
        """One class's images (the left grid) — and the compared class on the right."""
        self.view = {"kind": "class", "cls": cls}
        g = self.grids[0]
        g.cls = cls
        g.m.set_keys(self.ordered(self.members(cls)))
        g.scrollToTop()
        self.back_btn.hide()
        i = next((r for r in range(self.classes.count()) if self.classes.item(r).data(NAME_ROLE) == cls), -1)
        if i >= 0 and self.classes.currentRow() != i:
            self.classes.blockSignals(True)
            self.classes.setCurrentRow(i)
            self.classes.blockSignals(False)
        self._set_compare()
        self._update_heads()
        self._update_status()
        g.setFocus()

    def show_keys(self, title: str, keys, order: bool = False):
        """A list of images (a confusion cell, look-alikes, a sample) in the left grid."""
        self.view = {"kind": "list", "title": title, "cls": self.view.get("cls", ""), "keys": list(keys)}
        g = self.grids[0]
        g.cls = ""
        keys = [k for k in keys if k in self.session.by_key]
        g.m.set_keys(self.ordered(keys) if order else keys)
        g.scrollToTop()
        self._set_compare()                             # a confusion cell: the model's class beside it
        self.back_btn.setVisible(bool(self.view["cls"]))
        self._update_heads()
        self._update_status()
        g.setFocus()

    def _back(self):
        self.show_class(self.view.get("cls") or self._first_class())

    def _first_class(self):
        it = self.classes.item(0)
        return it.data(NAME_ROLE) if it is not None else ""

    def _set_compare(self):
        other = self.compare_box.currentData()
        g = self.grids[1]
        if not other or (self.view["kind"] == "class" and other == self.view["cls"]):
            self.right_col.hide()
            g.cls = ""
            g.m.set_keys([])
        else:
            g.cls = other
            g.m.set_keys(self.ordered(self.members(other)))
            self.right_col.show()
        self._update_heads()

    def _reload_view(self, resort: bool = False):
        if self.view["kind"] == "class":
            self.show_class(self.view["cls"])
        elif resort:
            self.show_keys(self.view["title"], self.grids[0].m.keys, order=True)

    def _update_heads(self):
        for gr, hd in zip(self.grids, self.heads):
            if gr is self.grids[0] and self.view["kind"] == "list":
                hd.setText(f"<b>{self.view['title']}</b> · {gr.m.rowCount():,} image(s)")
                continue
            name = "No class" if gr.cls == NO_CLASS else gr.cls
            hd.setText(f"<b>{name}</b> · {gr.m.rowCount():,} image(s)" if gr.cls else "")

    def _update_status(self):
        g = self.active or self.grids[0]
        sel = len(g.selectionModel().selectedIndexes()) if g.selectionModel() else 0
        keys = g.m.keys
        rv = self.session.review
        done = sum(1 for k in keys if rv.status(k) is not None)
        parts = [f"{len(keys):,} image(s) · {done:,} reviewed" + (f" ({done / len(keys):.0%})" if keys else "")]
        if sel:
            parts.append(f"{sel:,} selected")
        if self.sample is not None:
            parts.append(self._sample_text())
        self.status.setText(" · ".join(parts))
        self.sign_btn.setEnabled(self.sample is not None and self._sample_counts()[0] == len(self.sample["keys"]))

    # ------------------------------------------------------------------ thumbnails
    @property
    def tile(self) -> int:
        return self.size_slider.value()

    @property
    def tile_h(self) -> int:
        return round(self.tile * 0.75)

    def _apply_tile_size(self):
        t = self.tile
        self.grid_size = QSize(t + 14, self.tile_h + (28 if self.show_names else 12))
        for gr in self.grids:
            gr.setIconSize(QSize(t, t))
            gr.setGridSize(self.grid_size)
        self._blank = QPixmap(max(1, t), max(1, int(t * 0.75)))
        self._blank.fill(QColor(_tok()["track"]))

    def _resize_tiles(self, _v=None):
        self.app.settings.setValue("cls_gallery/tile", self.tile)
        self._apply_tile_size()
        self.gen += 1
        self.cache.clear()
        self._inflight.clear()
        for gr in self.grids:
            gr.viewport().update()

    def _toggle_names(self, on):
        self.show_names = bool(on)
        self.app.settings.setValue("cls_gallery/names", self.show_names)
        self._apply_tile_size()
        for gr in self.grids:
            gr.doItemsLayout()

    def pixmap_for(self, key) -> QPixmap:
        pix = self.cache.get(key)
        if pix is not None:
            return pix
        if key not in self._inflight:
            self._inflight.add(key)
            x = self.session.by_key.get(key)
            self._prio += 1                          # newest request first: what is on screen now
            QThreadPool.globalInstance().start(
                ThumbTask(self.sig, key, x.image_path if x is not None else None, self.tile, self.gen), self._prio)
        return self._blank

    def _on_thumb(self, key, gen, img):
        if gen != self.gen:
            return
        self._inflight.discard(key)
        if img.isNull():
            pix = QPixmap(self._blank)
            p = QPainter(pix)
            p.drawText(pix.rect(), int(Qt.AlignmentFlag.AlignCenter), "no image")
            p.end()
        else:
            pix = QPixmap.fromImage(img)
        self.cache.put(key, pix, cost=pix.width() * pix.height() * 4)
        for gr in self.grids:
            gr.m.changed([key])

    def drag_pixmap(self, keys) -> QPixmap:
        base = self.cache.get(keys[0])
        pm = QPixmap(72, 72)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if base is not None:
            p.drawPixmap(QRect(4, 4, 64, 64), base.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatioByExpanding))
        else:
            p.fillRect(QRect(4, 4, 64, 64), QColor(_tok()["track"]))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(_tok()["accent"]))
        p.drawEllipse(QRectF(44, 0, 28, 22))
        p.setPen(QColor("white"))
        p.drawText(QRectF(44, 0, 28, 22), int(Qt.AlignmentFlag.AlignCenter), str(len(keys)))
        p.end()
        return pm

    # ------------------------------------------------------------------ outside changes
    def _on_changed(self, keys):
        self._dirty.update(keys)
        self._sync.start()

    def _on_removed(self, keys):
        self._gone.update(keys)
        self._sync.start()

    def _on_renamed(self, ren: dict):
        for gr in self.grids:
            gr.m.rename(ren)
        for old, new in ren.items():
            self._gone.discard(old)
            if old in self.preds:
                self.preds[new] = self.preds.pop(old)
            if old in self.lookalike_of:
                self.lookalike_of[new] = self.lookalike_of.pop(old)
            pix = self.cache.get(old)
            if pix is not None:
                self.cache.put(new, pix, cost=pix.width() * pix.height() * 4)
            if self.sample is not None and old in self.sample["keys"]:
                self.sample["keys"].discard(old)
                self.sample["keys"].add(new)
            if self.view["kind"] == "list":
                self.view["keys"] = [new if k == old else k for k in self.view["keys"]]
        self._dirty.update(ren.values())
        self._sync.start()

    def _on_review(self, keys):
        for gr in self.grids:
            gr.m.changed(keys)
        self._counts_timer.start()
        self._update_status()

    def _sync_now(self):
        by = self.session.by_key
        gone, dirty = {k for k in self._gone if k not in by}, set(self._dirty)
        self._gone.clear()
        self._dirty.clear()
        for gr in self.grids:
            keep = gr.currentIndex().row()
            if gone:
                gr.m.remove(gone)
            if gr.cls:                                      # a class: images leave / join it
                out = [k for k in dirty if k in gr.m.pos and not self.in_class(k, gr.cls)]
                inn = [k for k in dirty if k not in gr.m.pos and self.in_class(k, gr.cls)]
                if out:
                    gr.m.remove(out)
                if inn and (gr is self.grids[0] or not self.right_col.isHidden()):
                    gr.m.insert_front(inn)
            gr.m.changed(dirty)
            n = gr.m.rowCount()
            if n and keep >= 0 and not gr.currentIndex().isValid():  # the keyboard stays where it was
                gr.setCurrentIndex(gr.m.index(min(keep, n - 1)))
        self._update_heads()
        self._counts_timer.start()
        self._update_status()

    # ------------------------------------------------------------------ actions
    def selected(self) -> list[str]:
        g = self.active or self.grids[0]
        keys = g.selected_keys()
        if not keys and g.currentIndex().isValid():
            keys = [g.m.keys[g.currentIndex().row()]]
        return keys

    def move_keys(self, keys, target: str, source: str = ""):
        """Give the images the class `target` (class folders: the files move). Multi-label: dragged out of a class
        means out of that class into the other one; elsewhere the class is added."""
        s = self.session
        keys = [k for k in keys if k in s.by_key]
        if not keys or not target or target == NO_CLASS:
            return
        if s.cls_mode == "multi":
            rem = [source] if source and source != NO_CLASS and source != target else []
            self.app._apply_classes(keys, add=[target], remove=rem,
                                    what=f"+{target}" + (f" −{source}" if rem else ""))
        else:
            if all(s.tags_of(k) == [target] for k in keys):
                return
            self.app._apply_classes(keys, tags=[target], what=f"class → {target}")

    def class_key(self, n: int):
        names = self.session.cls_classes()
        if n - 1 >= len(names):
            self.app.statusBar().showMessage(f"There is no class {n} (the dataset has {len(names)})", 4000)
            return
        keys = self.selected()
        if not keys:
            return
        name = names[n - 1]
        if self.session.cls_mode == "multi":
            have = all(name in self.session.tags_of(k) for k in keys)
            self.app._apply_classes(keys, add=() if have else [name], remove=[name] if have else (),
                                    what=("−" if have else "+") + name)
        else:
            self.move_keys(keys, name)

    def pick_class(self):
        from collections import Counter
        from .class_picker import ClassPicker
        s = self.session
        keys = self.selected()
        if not keys:
            return
        cur = Counter(t for k in keys for t in s.tags_of(k))
        dlg = ClassPicker(self, s.cls_classes(), cur, len(keys), multi=s.cls_mode == "multi",
                          counts=s.label_totals())
        if not dlg.exec() or dlg.result is None:
            return
        if dlg.result[0] == "set":
            name = dlg.result[1][0].strip()
            if name and s.base_adapter.can_use_label(name):
                self.app._apply_classes(keys, tags=[name], what=f"class → {name}")
        else:
            _k, add, remove = dlg.result
            if add or remove:
                self.app._apply_classes(keys, add=add, remove=remove,
                                        what=" ".join([f"+{a}" for a in add] + [f"−{r}" for r in remove]))

    def use_model(self):
        """The selected images get the classifier's top class (where it says another class)."""
        s = self.session
        by_cls: dict = {}
        for k in self.selected():
            p = self.preds.get(k)
            if p and p[0][0] not in s.tags_of(k):
                by_cls.setdefault(p[0][0], []).append(k)
        if not by_cls:
            self.app.statusBar().showMessage("The classifier agrees with the selected images (or has not seen them)",
                                             5000)
            return
        for cls, keys in by_cls.items():
            self.app._apply_classes(keys, tags=[cls], what=f"class → {cls} (model)")

    def mark(self, status, keys=None, note=None):
        keys = keys if keys is not None else self.selected()
        if not keys:
            return
        self.session.set_review(keys, status, note)
        self._advance()

    def _advance(self):
        """After marking one image with the keyboard: the next one."""
        g = self.active or self.grids[0]
        if len(g.selected_keys()) <= 1 and g.currentIndex().isValid():
            r = g.currentIndex().row() + 1
            if r < g.m.rowCount():
                g.setCurrentIndex(g.m.index(r))

    def visible_keys(self, g: Grid | None = None) -> list[str]:
        """The tiles fully on screen."""
        g = g or self.active or self.grids[0]
        g.doItemsLayout()                      # a pending re-flow (columns changed): measure what is really shown
        vp = g.viewport().rect()
        gh = max(1, self.grid_size.height())
        first = next((i for i in (g.indexAt(QPoint(x, y)) for x, y in ((4, 4), (gh // 2, gh // 2),
                                                                       (gh // 2, gh + 4))) if i.isValid()), None)
        r = max(0, first.row() - 64) if first is not None else 0
        out = []
        while r < g.m.rowCount():
            rect = g.visualRect(g.m.index(r))
            if rect.top() > vp.bottom():
                break
            if rect.top() >= vp.top() - 1 and rect.bottom() <= vp.bottom() + 1:
                out.append(g.m.keys[r])
            r += 1
        return out

    def accept_visible(self):
        """Space: every tile on screen without a mark becomes OK; then the next screen."""
        g = self.active or self.grids[0]
        rv = self.session.review
        keys = [k for k in self.visible_keys(g) if rv.status(k) is None]
        if keys:
            self.session.set_review(keys, REVIEW_OK)
        sb = g.verticalScrollBar()
        if sb.value() >= sb.maximum():              # the end: back to any image not reviewed yet (scrolled past)
            left = [i for i, k in enumerate(g.m.keys) if rv.status(k) is None]
            if left:
                idx = g.m.index(left[0])
                g.scrollTo(idx, QAbstractItemView.ScrollHint.PositionAtTop)
                g.setCurrentIndex(idx)
                self.app.statusBar().showMessage(f"{len(left):,} image(s) above are not reviewed yet — back to the "
                                                 "first of them", 6000)
            else:
                self.app.statusBar().showMessage("End of the list — every image on it is reviewed", 5000)
        else:                                       # the next screen starts with the first row not fully seen
            gh = max(1, self.grid_size.height())
            top, h = sb.value(), g.viewport().height()
            first_full = -(-top // gh)
            rows = max(1, (top + h) // gh - first_full)
            sb.setValue(min(sb.maximum(), (first_full + rows) * gh))
        self._update_status()

    def trash(self):
        keys = self.selected()
        if not keys:
            return
        if QMessageBox.question(self, self.TITLE, f"Move {len(keys):,} image(s) to the dataset's trash?\n\n"
                                "They can be brought back (Undo in the bar / File ▸ Trash).") != \
                QMessageBox.StandardButton.Yes:
            return
        done, errors = self.session.trash_samples(keys, include_images=True)
        if errors:
            self.app.log_errors([f"[Class Gallery] {e}" for e in errors])

    def send_issues(self, note=False):
        """I: the selected images to the issues folder (by class); the keyboard stays on the next tile."""
        keys = self.selected()
        if not keys or self.session.type.value != "cls_folders":
            return
        self.app.send_to_issues(note=True if note else None, keys=keys)
        self._sync_now()                                 # at once: the next I acts on the next tile

    def open_key(self, key):
        if key in self.session.by_key:
            self.app.open_sample(key)

    def undo(self):
        self.app.inline.undo()

    def redo(self):
        self.app.inline.redo()

    # ------------------------------------------------------------------ peek
    def peek_press(self):
        if self._peek is not None and self._peek.isVisible():
            self._peek.hide()
            return
        keys = self.selected()
        if not keys:
            return
        self._peek_t = time.monotonic()
        self._show_peek(keys[0])

    def peek_release(self):
        if self._peek is not None and time.monotonic() - self._peek_t > 0.35:      # held: closes on release
            self._peek.hide()

    def _show_peek(self, key):
        s = self.session
        if self._peek is None:
            self._peek = QLabel(self)
            self._peek.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._peek.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
            self._peek.mousePressEvent = lambda _e: self._peek.hide()
            apply_style(self._peek, "QLabel{background: rgba(12,12,14,235); color: white; border-radius: 10px;}")
        area = self.centre.geometry()
        self._peek.setGeometry(area)
        pair = [key] + ([self.lookalike_of[key][0]] if key in self.lookalike_of else [])
        w = (area.width() - 40) // len(pair)
        h = area.height() - 70
        canvas = QPixmap(area.width(), area.height())
        canvas.fill(Qt.GlobalColor.transparent)
        p = QPainter(canvas)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        for i, k in enumerate(pair):
            x = s.by_key.get(k)
            r = QImageReader(str(x.image_path)) if x is not None and x.image_path is not None else None
            img = QImage()
            if r is not None:
                r.setAutoTransform(True)
                sz = r.size()
                if sz.isValid() and (sz.width() > w or sz.height() > h):    # big photos: decode at screen size
                    k2 = min(w / sz.width(), h / sz.height())
                    r.setScaledSize(QSize(max(1, int(sz.width() * k2)), max(1, int(sz.height() * k2))))
                img = r.read()
                if not img.isNull():                     # fit the half / whole area — small images enlarged too
                    k2 = min(w / img.width(), h / img.height(), 8.0)
                    if abs(k2 - 1.0) > 0.02:
                        mode = Qt.TransformationMode.FastTransformation if k2 >= 3 else \
                            Qt.TransformationMode.SmoothTransformation          # tiny images: crisp pixels
                        img = img.scaled(max(1, int(img.width() * k2)), max(1, int(img.height() * k2)),
                                         Qt.AspectRatioMode.KeepAspectRatio, mode)
            x0 = 20 + i * w
            if not img.isNull():
                p.drawImage(QPoint(x0 + (w - img.width()) // 2, 16 + (h - img.height()) // 2), img)
            p.setPen(QColor("white"))
            cap = f"{x.name if x else k} — {', '.join(s.tags_of(k)) or 'no class'}"
            mb = self.preds.get(k)
            if mb:
                cap += f" · model: {mb[0][0]} {mb[0][1]:.0%}"
            p.drawText(QRect(x0, area.height() - 50, w, 40), int(Qt.AlignmentFlag.AlignCenter), cap)
        p.end()
        self._peek.setPixmap(canvas)
        self._peek.show()
        self._peek.raise_()

    # ------------------------------------------------------------------ Phase B / C views
    def _fill_confusions(self):
        m = self.conf_menu
        m.clear()
        if not self.preds:
            a = m.addAction("Run Model Check first — its results show here")
            a.triggered.connect(lambda: self.app.open_tool("model_check"))
            return
        keys = [x.key for x in self.session.samples if self._split_ok(x.key)]
        conf = CR.confusions(keys, self.session.tags_of, self.preds)
        if not conf:
            m.addAction("No mix-ups — the classifier agrees with every label").setEnabled(False)
            return
        for (g, p), ks in list(conf.items())[:25]:
            a = m.addAction(f"{g or 'no class'} → {p}   ({len(ks):,})")
            a.triggered.connect(lambda _c=False, g=g, p=p: self.show_confusion(g, p))

    def show_confusion(self, gt: str, pred: str):
        """The images of class `gt` the classifier calls `pred` (most confident first)."""
        keys = [x.key for x in self.session.samples if self._split_ok(x.key)]
        conf = CR.confusions(keys, self.session.tags_of, self.preds)
        ks = conf.get((gt, pred), [])
        if gt:
            self.view["cls"] = gt
        self.show_keys(f"{gt or 'no class'} → the model says {pred}", ks)
        self.compare_box.setCurrentIndex(max(0, self.compare_box.findData(pred)))

    def show_lookalikes(self, cls: str | None = None, threshold: float | None = None, wait: bool = False):
        rows, vecs = self._vectors()
        if vecs is None or not rows:
            QMessageBox.information(self, self.TITLE, "Look-alikes need the vectors of a Visual Similarity run "
                                                      "(Models ▸ Visual Similarity) — run it once on this dataset.")
            return
        cls = cls or self.view.get("cls") or self._first_class()
        thr = threshold if threshold is not None else float(self.app.settings.value("cls_gallery/alike", 0.95))
        mine = self.members(cls)
        mine_set = set(mine)
        others = [x.key for x in self.session.samples if x.key not in mine_set and self.session.tags_of(x.key)]
        if wait or len(mine) * max(1, len(others)) < 4_000_000:          # small: right away
            self._lookalikes_done(cls, thr, CR.lookalikes(mine, others, rows, vecs, thr))
            return
        from .inference_service import FuncWorker                       # big: off the window's thread
        self.look_btn.setEnabled(False)
        self.status.setText(f"Comparing {len(mine):,} image(s) of {cls} with {len(others):,} of other classes …")
        w = FuncWorker(lambda: CR.lookalikes(mine, others, rows, vecs, thr), self)
        w.done.connect(lambda found: self._lookalikes_done(cls, thr, found))
        w.failed.connect(lambda e: (self.look_btn.setEnabled(True), self.status.setText(f"Look-alikes: {e}")))
        self._look_worker = w
        w.start()

    def _lookalikes_done(self, cls, thr, found):
        self.look_btn.setEnabled(True)
        self.lookalike_of.update(found)
        self.view["cls"] = cls
        ks = sorted(found, key=lambda k: -found[k][1])
        self.show_keys(f"{cls}: near-identical to an image of another class (≥ {thr:.0%})", ks)
        if not ks:
            self.app.statusBar().showMessage(f"No image of {cls} has a near-identical image in another class", 5000)

    # ---- sample check + sign off
    def start_sample(self, cls: str | None = None, margin: float | None = None):
        cls = cls or self.view.get("cls") or self._first_class()
        if not cls or cls == NO_CLASS:
            return
        if self.sample is not None and self.sample["cls"] == cls and margin is None:     # carry on with it
            self.view["cls"] = cls
            self.show_keys(f"Sample of {cls} — flag (X) or move the wrong ones, Space for the rest",
                           [k for k in self.view.get("keys", []) if k in self.sample["keys"]] or
                           sorted(self.sample["keys"]))
            return
        mine = self.members(cls)
        if margin is None:
            from AIO_QC.core.sampling import sample_size
            opts = [(m, sample_size(len(mine), 0.95, m)) for m in (0.05, 0.10, 0.15)]
            labels = [f"± {m:.0%} — {n:,} images" for m, n in opts]
            choice, ok = QInputDialog.getItem(self, "Sample check", f"{cls}: {len(mine):,} images. Sample size "
                                              "(95 % sure the wrong share is within …):", labels, 1, False)
            if not ok:
                return
            margin = opts[labels.index(choice)][0]
        keys = CR.sample_for(mine, margin, seed=self._seed)
        self.sample = {"cls": cls, "keys": set(keys), "size": len(mine), "margin": margin}
        self.view["cls"] = cls
        self.show_keys(f"Sample of {cls} — flag (X) or move the wrong ones, Space for the rest", keys)

    def _sample_counts(self) -> tuple[int, int]:
        """(checked, wrong) of the running sample: wrong = flagged or no longer in the class."""
        s, smp = self.session, self.sample
        checked = wrong = 0
        for k in smp["keys"]:
            moved = k in s.by_key and smp["cls"] not in s.tags_of(k)
            st = s.review.status(k)
            if moved or st is not None:
                checked += 1
                if moved or st == REVIEW_FLAG:
                    wrong += 1
        return checked, wrong

    def _sample_text(self) -> str:
        c, w = self._sample_counts()
        n = len(self.sample["keys"])
        if c < n:
            return f"sample: {c:,} / {n:,} checked, {w:,} wrong so far"
        e = CR.signoff(c, w, self.sample["size"])
        return (f"sample done: {w:,} wrong of {c:,} → about {e['est']:,} wrong in {self.sample['cls']} "
                f"({e['est_lo']:,} … {e['est_hi']:,}) — Sign off records it")

    def sign_off(self):
        if self.sample is None:
            return
        c, w = self._sample_counts()
        if c < len(self.sample["keys"]):
            QMessageBox.information(self, self.TITLE, "Check every image of the sample first (Space marks the rest "
                                                      "OK).")
            return
        e = CR.signoff(c, w, self.sample["size"])
        e["when"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        e["margin"] = self.sample["margin"]
        s = self.session
        signed = dict(s.cls_settings().get("signoff") or {})
        signed[self.sample["cls"]] = e
        s.save_cls_settings(signoff=signed)
        cls = self.sample["cls"]
        self.sample = None
        self._fill_classes()
        self.app.statusBar().showMessage(f"{cls} signed off: about {e['est']:,} wrong image(s) "
                                         f"({e['est_lo']:,} … {e['est_hi']:,})", 8000)
        self.show_class(cls)

    def show_label(self, label: str, sort: str | None = None):
        """Open on a class (and an order) — used by Visual Similarity."""
        if sort:
            i = self.sort_box.findText(sort)
            if i >= 0:
                self.sort_box.blockSignals(True)
                self.sort_box.setCurrentIndex(i)
                self.sort_box.blockSignals(False)
        self.show_class(label)

    # ------------------------------------------------------------------ keys
    def _keys(self):
        def sc(seq, fn):
            q = QShortcut(QKeySequence(seq), self)
            q.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            q.activated.connect(fn)
        for n in range(1, 10):
            sc(str(n), lambda n=n: self.class_key(n))
        sc("C", self.pick_class)
        sc("M", self.use_model)
        sc("S", lambda: self.mark(REVIEW_OK))
        sc("X", lambda: self.mark(REVIEW_FLAG, note=None))
        sc("Shift+X", lambda: self._flag_with_note())
        sc("U", lambda: self.mark(None))
        sc("Space", self.accept_visible)
        sc("Del", self.trash)
        sc("I", self.send_issues)
        sc("Shift+I", lambda: self.send_issues(note=True))
        sc("Return", lambda: self.open_key((self.selected() or [None])[0]))
        sc("Enter", lambda: self.open_key((self.selected() or [None])[0]))
        sc("Ctrl+Z", self.undo)
        sc("Ctrl+Y", self.redo)
        sc("Ctrl+Shift+Z", self.redo)
        sc("Esc", lambda: self._peek.hide() if self._peek is not None and self._peek.isVisible() else self.close())

    def _flag_with_note(self):
        keys = self.selected()
        if not keys:
            return
        text, ok = QInputDialog.getText(self, "Flag", "Note:")
        if ok:
            self.mark(REVIEW_FLAG, keys, text.strip() or None)

    # ------------------------------------------------------------------ closing
    def closeEvent(self, e):
        self.gen += 1
        s = self.session
        for sig, slot in ((s.annotationsChanged, self._on_changed), (s.reviewChanged, self._on_review),
                          (s.samplesRemoved, self._on_removed), (s.samplesAdded, self._on_changed),
                          (s.keysRenamed, self._on_renamed), (s.indexChanged, self._counts_timer.start)):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        if hasattr(self.app, "forget_dialog"):
            self.app.forget_dialog(self)
        super().closeEvent(e)

