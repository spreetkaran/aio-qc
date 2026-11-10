# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
▦ Review Grid — whole images with their annotations, a page at a time, for the fast first pass.

The flow that makes it ~10× faster than one image at a time: look at the page, press X on the few bad ones,
then Space — every tile you did not mark becomes OK and the next page opens.

Keys   arrows move · S OK · X flag (Shift+X with a note) · U clear · Space accept the rest + next page ·
       Enter / PgDown next page (nothing marked) · Backspace / PgUp previous page · double-click open in the viewer
The file list is a snapshot taken when the grid opens (marking never reshuffles the pages). Images are decoded at
tile size in the background (JPEG scaled decode), the next page is prepared while you look at this one.
"""
from __future__ import annotations

import math
import time

from PyQt6.QtCore import QObject, QPointF, QRect, QRectF, QRunnable, QSize, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QImageIOHandler, QImageReader, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QHBoxLayout, QInputDialog, QLabel, QVBoxLayout, QWidget,
)

from AIO_QC.core.state import REVIEW_FIXED, REVIEW_FLAG, REVIEW_OK
from AIO_QC.utils import LRUCache, newButton, style as apply_style, theme_color

STATUS_ROLE = {REVIEW_OK: "success", REVIEW_FLAG: "warn", REVIEW_FIXED: "info"}
GRIDS = [(2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (6, 5)]


class _Signals(QObject):
    done = pyqtSignal(str, int, QImage)


class TileTask(QRunnable):
    """Decode at tile size + draw the annotations (QPainter on a QImage is fine off the GUI thread)."""

    def __init__(self, signals, adapter, sample, size: QSize, colors: dict, gen: int, tags: bool):
        super().__init__()
        self.sig, self.ad, self.s, self.size, self.colors, self.gen, self.tags = signals, adapter, sample, size, colors, gen, tags

    def run(self):
        img = QImage()
        try:
            ann = self.ad.load(self.s, light=True)
            path = ann.image_path or self.s.image_path
            r = QImageReader(str(path))
            r.setAutoTransform(True)
            full = r.size()
            tf = r.transformation()
            rotated = tf != QImageIOHandler.Transformation.TransformationNone
            if rotated and int(tf.value) & int(QImageIOHandler.Transformation.TransformationRotate90.value):
                full = QSize(full.height(), full.width())               # size as displayed
            if full.isValid() and not rotated:                         # scaled decode (JPEG: much faster)
                k = min(self.size.width() / full.width(), self.size.height() / full.height(), 1.0)
                if k < 1.0:
                    r.setScaledSize(QSize(max(1, round(full.width() * k)), max(1, round(full.height() * k))))
            img = r.read()
            if not img.isNull():
                img = img.scaled(self.size, Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation).convertToFormat(
                    QImage.Format.Format_ARGB32_Premultiplied)
                W, H = (1.0, 1.0) if ann.normalized else (float(ann.width or full.width() or 1),
                                                           float(ann.height or full.height() or 1))
                sx, sy = img.width() / W, img.height() / H
                p = QPainter(img)
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                f = QFont()
                f.setPointSizeF(7.5)
                f.setBold(True)
                p.setFont(f)
                for sh in ann.shapes:
                    pts = sh.get("points") or []
                    if not pts:
                        continue
                    col = QColor(self.colors.get(str(sh.get("label")), "#ff2bd6"))
                    p.setPen(QPen(col, 1.6))
                    q = [QPointF(float(a[0]) * sx, float(a[1]) * sy) for a in pts]
                    st = sh.get("shape_type") or "polygon"
                    if st == "rectangle" and len(q) == 2:
                        p.drawRect(QRectF(q[0], q[1]).normalized())
                    elif st == "circle" and len(q) == 2:
                        rr = math.hypot(q[1].x() - q[0].x(), q[1].y() - q[0].y())
                        p.drawEllipse(q[0], rr, rr)
                    elif len(q) == 1 or st in ("point", "points"):
                        for x in q:
                            p.drawEllipse(x, 2.5, 2.5)
                    elif st in ("line", "linestrip"):
                        p.drawPolyline(QPolygonF(q))
                    else:
                        p.drawPolygon(QPolygonF(q))
                    if self.tags:
                        top = min(q, key=lambda a: (a.y(), a.x()))
                        p.drawText(QPointF(top.x() + 2, max(9.0, top.y() - 2)), str(sh.get("label")))
                p.end()
        except Exception:
            img = QImage()
        self.sig.done.emit(self.s.key, self.gen, img)


class GridView(QWidget):
    activated = pyqtSignal(str)            # double-click -> open in the viewer
    focusMoved = pyqtSignal()

    def __init__(self, owner):
        super().__init__(owner)
        self.o = owner
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 300)
        self.focus = 0

    def cells(self) -> list[QRect]:
        cols, rows = self.o.grid
        gap = 6
        w = (self.width() - gap * (cols + 1)) / cols
        h = (self.height() - gap * (rows + 1)) / rows
        return [QRect(round(gap + c * (w + gap)), round(gap + r * (h + gap)), round(w), round(h))
                for r in range(rows) for c in range(cols)]

    def tile_size(self) -> QSize:
        c = self.cells()
        return QSize(max(32, c[0].width() - 8), max(32, c[0].height() - 26)) if c else QSize(200, 150)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        keys = self.o.page_keys()
        base, muted, line = theme_color("placeholder"), theme_color("muted"), theme_color("line")
        for i, cell in enumerate(self.cells()):
            if i >= len(keys):
                break
            key = keys[i]
            st = self.o.session.review.status(key)
            border = theme_color(STATUS_ROLE[st]) if st in STATUS_ROLE else line
            p.setPen(QPen(border, 3 if st in STATUS_ROLE else 1))
            p.setBrush(base)
            p.drawRoundedRect(QRectF(cell).adjusted(1.5, 1.5, -1.5, -1.5), 6, 6)
            img = self.o.images.get(key)
            area = cell.adjusted(4, 4, -4, -22)
            if img is not None and not img.isNull():
                x = area.x() + (area.width() - img.width()) // 2
                y = area.y() + (area.height() - img.height()) // 2
                p.drawImage(x, y, img)
            elif img is not None:
                p.setPen(muted)
                p.drawText(area, Qt.AlignmentFlag.AlignCenter, "no image")
            else:
                p.setPen(muted)
                p.drawText(area, Qt.AlignmentFlag.AlignCenter, "…")
            s = self.o.session.by_key.get(key)
            mark = {REVIEW_OK: "✓ ", REVIEW_FLAG: "⚑ ", REVIEW_FIXED: "🔧 "}.get(st, "")
            name = (mark + (s.display if s else key))
            tr = QRect(cell.x() + 6, cell.bottom() - 20, cell.width() - 12, 18)
            p.setPen(theme_color("text"))
            p.drawText(tr, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       p.fontMetrics().elidedText(name, Qt.TextElideMode.ElideMiddle, tr.width()))
            if i == self.focus and self.hasFocus():
                pen = QPen(theme_color("accent"), 2, Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(QRectF(cell).adjusted(-1, -1, 1, 1), 8, 8)
        p.end()

    def _cell_at(self, pos) -> int:
        for i, c in enumerate(self.cells()):
            if c.contains(pos):
                return i if i < len(self.o.page_keys()) else -1
        return -1

    def mousePressEvent(self, e):
        i = self._cell_at(e.position().toPoint())
        if i >= 0:
            self.focus = i
            self.setFocus()
            self.update()
            self.focusMoved.emit()

    def mouseDoubleClickEvent(self, e):
        i = self._cell_at(e.position().toPoint())
        if i >= 0:
            self.activated.emit(self.o.page_keys()[i])

    def keyPressEvent(self, e):
        cols, _rows = self.o.grid
        n = len(self.o.page_keys())
        k = e.key()
        mv = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1, Qt.Key.Key_Up: -cols, Qt.Key.Key_Down: cols}
        if k in mv and n:
            self.focus = min(max(self.focus + mv[k], 0), n - 1)
            self.update()
            self.focusMoved.emit()
            return
        if self.o.handle_key(e):
            return
        super().keyPressEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.o.tiles_resized()


class ReviewGridDialog(QDialog):
    TITLE, ICON = "Review Grid", "▦"
    SUPPORTED = None

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.setWindowTitle(self.TITLE)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(int(app.width() * 0.85), int(app.height() * 0.9))
        st = app.settings
        self.grid = tuple(st.value("grid/size", [3, 3])) if st.value("grid/size") else (3, 3)
        self.grid = (int(self.grid[0]), int(self.grid[1]))
        self.keys: list[str] = []
        self.page = 0
        self.images = LRUCache(capacity=400)
        self.gen = 0
        self.sig = _Signals()
        self.sig.done.connect(self._on_tile)
        self.stats = {"ok": 0, "flag": 0}
        self._inflight: set = set()
        self._started = time.monotonic()

        self.source = QComboBox()
        self.source.addItem("Unreviewed files in the current list", "unreviewed")
        self.source.addItem("All files in the current list", "list")
        self.source.addItem("Flagged / fixed files", "flagged")
        self.source.addItem("Every file in the dataset", "all")
        self.source.currentIndexChanged.connect(lambda _i: self._load_keys())
        self.size_box = QComboBox()
        for c, r in GRIDS:
            self.size_box.addItem(f"{c} × {r}", (c, r))
        j = self.size_box.findData(self.grid)
        self.size_box.setCurrentIndex(j if j >= 0 else 2)
        self.size_box.currentIndexChanged.connect(self._grid_changed)
        self.tags = QCheckBox("Label names")
        self.tags.setChecked(st.value("grid/tags", False, type=bool))
        self.tags.toggled.connect(self._restyle)
        top = QHBoxLayout()
        for w in (QLabel("Files:"), self.source, QLabel("Grid:"), self.size_box, self.tags):
            top.addWidget(w)
        top.addStretch(1)
        self.prev_btn = newButton("◀ Previous", slot=lambda: self.go(self.page - 1), tip="Previous page (PgUp)")
        self.next_btn = newButton("Next ▶", slot=lambda: self.go(self.page + 1), tip="Next page (Enter / PgDown)")
        self.accept_btn = newButton("✓ Rest OK + next (Space)", slot=self.accept_page,
                                    tip="Every unmarked tile on this page becomes OK, then the next page opens")
        for b in (self.prev_btn, self.next_btn, self.accept_btn):
            top.addWidget(b)
        self.view = GridView(self)
        self.view.activated.connect(lambda k: self.app.open_sample(k))
        self.info = QLabel("")
        apply_style(self.info, "color:@muted;")
        hint = QLabel("arrows move · S OK · X flag · Shift+X flag with a note · U clear · Space: rest OK + next page · "
                      "Enter next · Backspace previous · double-click opens in the viewer")
        apply_style(hint, "color:@faint;")
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(self.info, 1)
        bottom.addWidget(hint)
        lay.addLayout(bottom)
        self.session.reviewChanged.connect(self._review_changed)
        self.session.annotationsChanged.connect(self._annotations_changed)
        self._load_keys()

    # ------------------------------------------------------------------ data
    def load_custom(self, keys, title: str):
        """Review exactly these files (e.g. a Saved View)."""
        self._custom = (list(keys), title)
        i = self.source.findData("custom")
        if i < 0:
            self.source.addItem(title, "custom")
            i = self.source.count() - 1
        else:
            self.source.setItemText(i, title)
        self.source.blockSignals(True)
        self.source.setCurrentIndex(i)
        self.source.blockSignals(False)
        self._load_keys()

    def _load_keys(self):
        mode = self.source.currentData()
        s, dock = self.session, self.app.file_dock
        if mode == "custom":
            keys = [k for k in getattr(self, "_custom", ([], ""))[0] if k in s.by_key]
        elif mode == "all":
            keys = [x.key for x in s.samples]
        elif mode == "flagged":
            keys = s.keys_with_review({REVIEW_FLAG, REVIEW_FIXED}, rank=dock.model.row_of)
        else:
            keys = dock.visible_keys()
            if mode == "unreviewed":
                rv = s.review
                keys = [k for k in keys if rv.status(k) is None]
        self.keys = keys
        self.page = 0
        self.gen += 1
        self.images.clear()
        self._inflight.clear()
        self._request()
        self._update_info()
        self.view.focus = 0
        self.view.setFocus()
        self.view.update()

    def per_page(self) -> int:
        return self.grid[0] * self.grid[1]

    def pages(self) -> int:
        return max(1, math.ceil(len(self.keys) / self.per_page()))

    def page_keys(self, page: int | None = None) -> list[str]:
        p = self.page if page is None else page
        n = self.per_page()
        return self.keys[p * n:(p + 1) * n]

    def _colors(self) -> dict:
        return {k: c.name() for k, c in self.app.label_colors.items()}

    def _request(self):
        size = self.view.tile_size()
        colors, pool = self._colors(), QThreadPool.globalInstance()
        for page, prio in ((self.page, 10), (self.page + 1, 0)):              # this page first, then prefetch
            for k in self.page_keys(page):
                if k in self.images or k in self._inflight or k not in self.session.by_key:
                    continue
                self._inflight.add(k)
                pool.start(TileTask(self.sig, self.session.adapter, self.session.by_key[k], size, colors, self.gen,
                                    self.tags.isChecked()), prio)

    def _on_tile(self, key, gen, img):
        if gen != self.gen:
            return
        self._inflight.discard(key)
        self.images.put(key, img)
        if key in self.page_keys():
            self.view.update()

    def tiles_resized(self):
        self._restyle()

    def _restyle(self, *_):
        self.app.settings.setValue("grid/tags", self.tags.isChecked())
        self.gen += 1
        self.images.clear()
        self._inflight.clear()
        self._request()
        self.view.update()

    def _grid_changed(self, _i):
        self.grid = self.size_box.currentData()
        self.app.settings.setValue("grid/size", list(self.grid))
        first = self.page * self.per_page()
        self.page = 0
        self.view.focus = 0
        self.go(first // self.per_page())
        self._restyle()

    # ------------------------------------------------------------------ paging / marking
    def go(self, page: int):
        page = min(max(page, 0), self.pages() - 1)
        if page != self.page:
            self.page = page
            self.view.focus = 0
            self._request()
        self._update_info()
        self.view.update()

    def _focused_key(self) -> str | None:
        keys = self.page_keys()
        return keys[self.view.focus] if 0 <= self.view.focus < len(keys) else None

    def mark(self, status, note=None, key=None):
        key = key or self._focused_key()
        if key is None:
            return
        self.session.set_review([key], status, note if note is not None else
                                ("Review Grid" if status == REVIEW_FLAG else None))
        if status == REVIEW_OK:
            self.stats["ok"] += 1
        elif status == REVIEW_FLAG:
            self.stats["flag"] += 1
        keys = self.page_keys()
        if self.view.focus < len(keys) - 1:
            self.view.focus += 1
        self.view.update()
        self._update_info()

    def accept_page(self):
        rv = self.session.review
        todo = [k for k in self.page_keys() if rv.status(k) is None]
        if todo:
            self.session.set_review(todo, REVIEW_OK, None)
            self.stats["ok"] += len(todo)
        if self.page < self.pages() - 1:
            self.go(self.page + 1)
        else:
            self._update_info(done=True)
            self.view.update()

    def handle_key(self, e) -> bool:
        k, mods = e.key(), e.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if k == Qt.Key.Key_S:
            self.mark(REVIEW_OK)
        elif k == Qt.Key.Key_X:
            note = None
            if shift:
                note, ok = QInputDialog.getText(self, "Flag with a note", "What is wrong?")
                if not ok:
                    return True
            self.mark(REVIEW_FLAG, note or None)
        elif k == Qt.Key.Key_U:
            self.mark(None)
        elif k == Qt.Key.Key_Space:
            self.accept_page()
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_PageDown):
            self.go(self.page + 1)
        elif k in (Qt.Key.Key_Backspace, Qt.Key.Key_PageUp):
            self.go(self.page - 1)
        else:
            return False
        return True

    def keyPressEvent(self, e):
        if not self.handle_key(e):
            super().keyPressEvent(e)

    def _update_info(self, done=False):
        n = len(self.keys)
        mins = max((time.monotonic() - self._started) / 60, 1e-6)
        rate = (self.stats["ok"] + self.stats["flag"]) / mins
        self.info.setText(
            ("✅ Last page reviewed — " if done else "") +
            f"Page {self.page + 1:,} / {self.pages():,} · {n:,} file(s) · this session: {self.stats['ok']:,} OK, "
            f"{self.stats['flag']:,} flagged" + (f" · {rate:,.0f} files / min" if rate >= 1 else ""))
        self.prev_btn.setEnabled(self.page > 0)
        self.next_btn.setEnabled(self.page < self.pages() - 1)

    def _review_changed(self, keys):
        if set(keys) & set(self.page_keys()):
            self.view.update()

    def _annotations_changed(self, keys):
        hit = [k for k in keys if k in set(self.page_keys(self.page)) | set(self.page_keys(self.page + 1))]
        for k in hit:
            self.images.pop(k)
        if hit:
            self._request()

    def closeEvent(self, e):
        self.gen += 1
        for sig, slot in ((self.session.reviewChanged, self._review_changed),
                          (self.session.annotationsChanged, self._annotations_changed)):
            try:
                sig.disconnect(slot)
            except TypeError:
                pass
        if hasattr(self.app, "forget_dialog"):
            self.app.forget_dialog(self)
        super().closeEvent(e)
