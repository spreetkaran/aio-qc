# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The label picker for a new box / polygon or a relabel — a small popup next to the shape, as in professional
annotation tools: type to search, ↑ ↓ and Enter (or a click), 1 … 9 for the first nine classes, Esc to cancel. Every
label shows its colour and how often the dataset uses it; the last one used is selected. Where new labels are
allowed (LabelMe), typing a name that is not in the list offers to create it.

    label = pick_label(parent, "New bbox", names, current="car", colors={...}, counts={...}, allow_new=True,
                       at=QPoint(global), kind="box")      # -> str | None
"""
from __future__ import annotations

from PyQt6.QtCore import QEventLoop, QPoint, QRect, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QStyle, QStyledItemDelegate, QVBoxLayout)

from AIO_QC import utils

ROLE_NAME = Qt.ItemDataRole.UserRole
ROLE_NEW = Qt.ItemDataRole.UserRole + 1
ROLE_COLOR = Qt.ItemDataRole.UserRole + 2
ROLE_COUNT = Qt.ItemDataRole.UserRole + 3
ROLE_KEY = Qt.ItemDataRole.UserRole + 4
KIND_ICON = {"box": "shape-rectangle", "polygon": "shape-polygon", "relabel": "edit"}


class _Row(QStyledItemDelegate):
    """swatch · name ……… count  [key]"""

    def sizeHint(self, opt, index):
        return QSize(opt.rect.width(), 30)

    def paint(self, p, opt, index):
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = opt.rect
        sel = bool(opt.state & QStyle.StateFlag.State_Selected)
        hov = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        if sel or hov:
            bg = QColor(utils.hexc("accent"))
            bg.setAlpha(46 if sel else 20)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(bg)
            p.drawRoundedRect(r.adjusted(4, 1, -4, -1), 6, 6)
        x = r.left() + 12
        cy = r.center().y()
        ink, muted = QColor(utils.hexc("text")), QColor(utils.hexc("muted"))
        if index.data(ROLE_NEW):
            p.setPen(QColor(utils.hexc("accent")))
            f = QFont(opt.font)
            f.setWeight(QFont.Weight.DemiBold)
            p.setFont(f)
            p.drawText(QRect(x, r.top(), r.width() - 24, r.height()),
                       int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), f"+  Create “{index.data(ROLE_NAME)}”")
            p.restore()
            return
        col = index.data(ROLE_COLOR) or QColor("#888")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(col))
        p.drawRoundedRect(QRect(x, cy - 6, 12, 12), 3, 3)
        x += 22
        key = index.data(ROLE_KEY) or ""
        count = index.data(ROLE_COUNT)
        right = r.right() - 10
        fm = p.fontMetrics()
        if key:                                         # the key cap
            kw = max(18, fm.horizontalAdvance(key) + 10)
            kr = QRect(right - kw, cy - 9, kw, 18)
            p.setPen(QColor(utils.hexc("line")))
            p.setBrush(QColor(utils.hexc("base")))
            p.drawRoundedRect(kr, 4, 4)
            p.setPen(muted)
            p.drawText(kr, int(Qt.AlignmentFlag.AlignCenter), key)
            right = kr.left() - 8
        if count:
            t = f"{count:,}"
            p.setPen(muted)
            p.drawText(QRect(right - 60, r.top(), 60, r.height()),
                       int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), t)
            right -= fm.horizontalAdvance(t) + 10
        p.setPen(ink)
        name = fm.elidedText(index.data(ROLE_NAME), Qt.TextElideMode.ElideRight, max(20, right - x))
        p.drawText(QRect(x, r.top(), right - x, r.height()), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), name)
        p.restore()


class LabelPicker(QFrame):
    def __init__(self, parent, title: str, names, current=None, colors=None, counts=None, allow_new=True,
                 kind="box"):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("labelPicker")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        utils.style(self, "#labelPicker{background:@base; border:1px solid @line; border-radius:10px;}"
                          "QListWidget{border:none; background:transparent; outline:0;}")
        self.names = list(dict.fromkeys(str(n) for n in names))
        self.colors, self.counts = colors or {}, counts or {}
        self.allow_new = allow_new
        self.result = None
        self._loop = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 10, 8, 8)
        lay.setSpacing(8)
        head = QHBoxLayout()
        head.setContentsMargins(6, 0, 4, 0)
        ic = utils.IconLabel(KIND_ICON.get(kind, "edit"), 16)
        t = QLabel(title)
        f = t.font()
        f.setWeight(QFont.Weight.DemiBold)
        t.setFont(f)
        hint = QLabel("↵ choose · Esc cancel")
        utils.style(hint, "QLabel{color:@faint;}")
        head.addWidget(ic)
        head.addWidget(t)
        head.addStretch(1)
        head.addWidget(hint)
        lay.addLayout(head)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search or type a new label" if allow_new else "Search labels")
        self.search.setClearButtonEnabled(True)
        self.search.setMinimumHeight(32)
        self.search.textChanged.connect(self._fill)
        self.search.installEventFilter(self)
        lay.addWidget(self.search)
        self.list = QListWidget()
        self.list.setItemDelegate(_Row(self.list))
        self.list.setMouseTracking(True)
        self.list.setUniformItemSizes(True)
        self.list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.list.itemClicked.connect(self._choose)
        lay.addWidget(self.list)
        self.empty = QLabel("No labels yet — type one to create it" if allow_new else "No matching label")
        utils.style(self.empty, "QLabel{color:@muted; padding:6px;}")
        lay.addWidget(self.empty)
        self.setFixedWidth(300)
        self._current = current
        self._fill("")

    # ------------------------------------------------------------------
    def _fill(self, text: str):
        q = text.strip().lower()
        self.list.clear()
        shown = [n for n in self.names if q in n.lower()] if q else list(self.names)
        if q:                                            # names starting with the text first
            shown.sort(key=lambda n: (not n.lower().startswith(q), self.names.index(n)))
        for n in shown:
            i = self.names.index(n)
            it = QListWidgetItem(n)
            it.setData(ROLE_NAME, n)
            it.setData(ROLE_COLOR, self.colors.get(n))
            it.setData(ROLE_COUNT, self.counts.get(n, 0))
            it.setData(ROLE_KEY, str(i + 1) if i < 9 and not q else "")
            self.list.addItem(it)
        exact = any(n.lower() == q for n in self.names)
        if q and self.allow_new and not exact:
            it = QListWidgetItem(text.strip())
            it.setData(ROLE_NAME, text.strip())
            it.setData(ROLE_NEW, True)
            self.list.addItem(it)
        rows = self.list.count()
        self.list.setVisible(rows > 0)
        self.empty.setVisible(rows == 0)
        self.list.setFixedHeight(min(9, max(rows, 1)) * 30 + 4)
        if rows:
            pick = 0
            if not q and self._current in self.names:
                pick = shown.index(self._current)
            self.list.setCurrentRow(pick)
        self.adjustSize()

    def _choose(self, item=None):
        item = item or self.list.currentItem()
        if item is None:
            return
        self.result = item.data(ROLE_NAME)
        self.close()

    def eventFilter(self, obj, e):
        from PyQt6.QtCore import QEvent
        if obj is self.search and e.type() == QEvent.Type.KeyPress:
            k = e.key()
            if k in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                n = self.list.count()
                if n:
                    step = 1 if k == Qt.Key.Key_Down else -1
                    self.list.setCurrentRow((self.list.currentRow() + step) % n)
                return True
            if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._choose()
                return True
            if k == Qt.Key.Key_Escape:
                self.close()
                return True
            if not self.search.text() and Qt.Key.Key_1 <= k <= Qt.Key.Key_9:
                i = k - Qt.Key.Key_1
                if i < len(self.names):
                    self.result = self.names[i]
                    self.close()
                    return True
        return super().eventFilter(obj, e)

    # ------------------------------------------------------------------
    def popup_at(self, at: QPoint):
        self.adjustSize()
        scr = QApplication.screenAt(at) or QApplication.primaryScreen()
        av = scr.availableGeometry()
        x = min(max(at.x(), av.left() + 8), av.right() - self.width() - 8)
        y = at.y()
        if y + self.height() > av.bottom() - 8:
            y = max(av.top() + 8, at.y() - self.height())
        self.move(x, y)
        self.show()
        self.search.setFocus()

    def closeEvent(self, e):
        super().closeEvent(e)
        if self._loop is not None:
            self._loop.quit()

    def exec_at(self, at: QPoint):
        self.popup_at(at)
        self._loop = QEventLoop()
        self._loop.exec()
        self._loop = None
        return self.result


def pick_label(parent, title, names, current=None, colors=None, counts=None, allow_new=True, at=None,
               kind="box"):
    from PyQt6.QtGui import QCursor
    p = LabelPicker(parent, title, names, current, colors, counts, allow_new, kind)
    try:
        return p.exec_at(at if at is not None else QCursor.pos())
    finally:
        p.deleteLater()
