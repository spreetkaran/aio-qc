# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Small read-outs floating over the image canvas (3.0): the file (name · size · type · file size) top-left, the
review flag and its note top-right, the pointer position and the selected box / ROI bottom-right. They never take
the mouse (clicks go to the canvas) and keep to the corners as the view resizes. The status bar keeps the same
facts for those who look there.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from AIO_QC import utils
from AIO_QC.core.diag import ignored as _ignored

CHIP_CSS = ("QWidget#hudChip{background: rgba(15,17,20,0.74); border-radius: 6px;}"
            "QLabel{background: transparent; color: #DDE2E8;}")


class _Chip(QWidget):
    def __init__(self, parent, icon: str = "", readout: bool = False):
        super().__init__(parent)
        self.setObjectName("hudChip")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet(CHIP_CSS)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(9, 4, 10, 4)
        lay.setSpacing(6)
        self.icon = utils.IconLabel(utils.tintedIcon(icon, "#F7C873") if icon else "", 13)
        if not icon:                                 # (setVisible(True) on a widget without a parent yet opens it
            self.icon.hide()                         # as a window of its own for a moment — a focus flicker)
        self.main = QLabel("")
        f = self.main.font()
        f.setBold(True)
        self.main.setFont(f)
        from AIO_QC.widgets.design import mono
        if readout:                                   # x · y · box: keys, values and dots spaced properly
            self.extra = _Readout(mono(12), self)
        else:
            self.extra = QLabel("")
            self.extra.setTextFormat(Qt.TextFormat.PlainText)        # keeps the padding spaces of the values
            self.extra.setStyleSheet("color: #A7B0BB;")
            self.extra.setFont(mono(12))
        lay.addWidget(self.icon)
        lay.addWidget(self.main)
        lay.addWidget(self.extra)
        self.setVisible(False)

    def set(self, main: str, extra: str = "", color: str | None = None):
        if (main, extra, color) == getattr(self, "_last", None):
            return
        self._last = (main, extra, color)
        self.main.setText(main)
        self.main.setStyleSheet(f"color: {color};" if color else "")
        self.extra.setText(extra)
        self.extra.setVisible(bool(extra))
        self.main.setVisible(bool(main))
        on = bool(main or extra)
        self.setVisible(on)
        if on:
            self.extra.setMinimumWidth(0)
            self.setFixedSize(self.sizeHint())        # exactly its text (steady: the values are padded)

    sticky = False
    _sticky = 0

    def reset_width(self):
        self._sticky = 0


class _Dot(QWidget):
    """A separator dot with room on both sides, centred on the text's lower-case height (where a typed "·" sits a
    little off, and close to its neighbours)."""

    def __init__(self, font, color="#8A939E", parent=None):
        super().__init__(parent)
        from PyQt6.QtGui import QFontMetricsF
        self.fm = QFontMetricsF(font)
        self.color = color
        self.setFixedWidth(19)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def paintEvent(self, _e):
        from PyQt6.QtCore import QPointF
        from PyQt6.QtGui import QColor, QPainter
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self.color))
        fm = self.fm
        top = (self.height() - fm.height()) / 2                  # where a vertically centred label puts its text
        y = top + fm.ascent() - fm.xHeight() / 2
        p.drawEllipse(QPointF(self.width() / 2, y), 1.6, 1.6)


class _Segments(QWidget):
    """The details after the name — dimensions, type, file size — as separate pieces with dots between them."""

    def __init__(self, font, parent=None):
        super().__init__(parent)
        self._font = font
        self._parts = []
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(0)

    def setText(self, text: str):
        parts = [x.strip() for x in (text or "").split(" · ") if x.strip()]
        if parts == self._parts:
            return
        self._parts = parts
        while self.lay.count():
            it = self.lay.takeAt(0)
            if it.widget() is not None:
                it.widget().deleteLater()
        for i, part in enumerate(parts):
            if i:
                self.lay.addWidget(_Dot(self._font, parent=self))
            lab = QLabel(part, self)
            lab.setFont(self._font)
            lab.setStyleSheet("color: #A7B0BB; background: transparent;")
            self.lay.addWidget(lab)
        self.lay.invalidate()
        self.adjustSize()

    def text(self) -> str:
        return " · ".join(self._parts)


class _Readout(QWidget):
    """The pointer / selection read-out: "x 1,234 · y 567 · box 80 × 40 at 10, 20" drawn as keys in a quiet colour,
    values bright in the mono font (right-aligned in their padded width, so nothing jitters while the pointer moves)
    and evenly spaced dots between the pieces. Painted, not laid out: it changes on every mouse move."""
    KEY, VAL, DOT = "#8A939E", "#E3E7EC", "#6F7883"
    GAP_KEY, GAP_DOT = 5, 9

    def __init__(self, font, parent=None):
        super().__init__(parent)
        from PyQt6.QtGui import QFont, QFontMetricsF
        self.vfont = font
        self.kfont = QFont(font)
        self.vfm, self.kfm = QFontMetricsF(self.vfont), QFontMetricsF(self.kfont)
        self._parts: list = []
        self._text = ""
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    @staticmethod
    def _split(part: str):
        part = part.strip("\n")
        head, sep, rest = part.lstrip().partition(" ")
        if sep and head.isalpha():
            return head, rest                  # keep the value's padding spaces (a steady width)
        return "", part.strip()

    def setText(self, text: str):
        if text == self._text:
            return
        self._text = text
        self._parts = [self._split(p) for p in (text or "").split(" · ") if p.strip()]
        self.updateGeometry()
        self.setFixedSize(self.sizeHint())
        self.update()

    def text(self) -> str:
        return self._text

    def _width(self) -> float:
        w = 0.0
        for i, (k, v) in enumerate(self._parts):
            if i:
                w += 2 * self.GAP_DOT + 3
            if k:
                w += self.kfm.horizontalAdvance(k) + self.GAP_KEY
            w += self.vfm.horizontalAdvance(v)
        return w

    def sizeHint(self):
        from PyQt6.QtCore import QSize
        return QSize(int(self._width() + 1.5), int(self.vfm.height() + 2))

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, _e):
        from PyQt6.QtCore import QPointF
        from PyQt6.QtGui import QColor, QPainter
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h = self.height()
        base = (h - self.vfm.height()) / 2 + self.vfm.ascent()
        x = 0.0
        for i, (k, v) in enumerate(self._parts):
            if i:
                x += self.GAP_DOT
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(self.DOT))
                p.drawEllipse(QPointF(x + 1.5, base - self.vfm.xHeight() / 2), 1.5, 1.5)
                x += 3 + self.GAP_DOT
            if k:
                p.setFont(self.kfont)
                p.setPen(QColor(self.KEY))
                p.drawText(QPointF(x, base), k)
                x += self.kfm.horizontalAdvance(k) + self.GAP_KEY
            p.setFont(self.vfont)
            p.setPen(QColor(self.VAL))
            p.drawText(QPointF(x, base), v)
            x += self.vfm.horizontalAdvance(v)
        p.end()


class _InfoChip(QWidget):
    """The image's details as a small (i) icon; resting the mouse on it unfolds the details to the right (name ·
    size · type · file size) with an eased glide, and leaving folds them back the same way."""

    H = 28
    MS = 240

    def __init__(self, parent):
        super().__init__(parent)
        from PyQt6.QtCore import QEasingCurve, QVariantAnimation
        self.setObjectName("hudChip")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(CHIP_CSS)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.icon = utils.IconLabel(utils.tintedIcon("info", "#DDE2E8"), 15, self)
        self.icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.body = QWidget(self)                   # laid out at full width; the chip clips it while folded
        self.body.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lay = QHBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 12, 0)
        lay.setSpacing(0)
        self.main = QLabel("")
        f = self.main.font()
        f.setWeight(f.Weight.DemiBold)
        self.main.setFont(f)
        from AIO_QC.widgets.design import mono
        self.extra = _Segments(mono(12))
        self.name_dot = _Dot(mono(12), parent=self.body)        # name · dimensions · type · file size
        lay.addWidget(self.main)
        lay.addWidget(self.name_dot)
        lay.addWidget(self.extra)
        self._open = 0.0
        self.anim = QVariantAnimation(self)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.anim.valueChanged.connect(self._set_open)
        self.setFixedHeight(self.H)
        self.setVisible(False)
        self._full = ""
        # the cursor itself decides too: Enter / Leave go stale on Windows (after a native dialog, a drag and drop,
        # or with another window active) — the details did not unfold on hover until a click
        from .hover_watch import watch
        watch(self, on_enter=lambda: self._glide(1.0), on_leave=lambda: self._glide(0.0))

    def set(self, main: str, extra: str = "", color=None):
        self.main.setText(main)
        self.extra.setText(extra)
        self.extra.setVisible(bool(extra))
        self.name_dot.setVisible(bool(main and extra))
        self._full = " · ".join(x for x in (main, extra) if x)
        self.setToolTip("")
        self.body.adjustSize()
        on = bool(main or extra)
        self.setVisible(on)
        self._set_open(self._open)

    def _closed_w(self):
        return self.H

    def _open_w(self):
        return self.H + 2 + self.body.sizeHint().width()

    def _set_open(self, k):
        self._open = float(k)
        w = round(self._closed_w() + (self._open_w() - self._closed_w()) * self._open)
        self.setFixedWidth(max(self.H, w))
        self.icon.move((self.H - 15) // 2, (self.H - 15) // 2)
        self.body.setGeometry(self.H, 0, self.body.sizeHint().width(), self.H)

    def _glide(self, to):
        try:
            from AIO_QC.widgets import motion
            ms = motion.scaled(self.MS)
        except Exception as _exc:
            _ignored(_exc)
            ms = self.MS
        if self.anim.state() == self.anim.State.Running and self.anim.endValue() == float(to):
            return                                  # already gliding there (Enter and the watcher both asked)
        if self.anim.state() != self.anim.State.Running and abs(self._open - to) < 1e-6:
            return
        self.anim.stop()
        if ms <= 0:
            self._set_open(to)
            return
        self.anim.setDuration(int(ms * abs(to - self._open)) or 1)
        self.anim.setStartValue(self._open)
        self.anim.setEndValue(float(to))
        self.anim.start()

    def enterEvent(self, e):
        super().enterEvent(e)
        self._glide(1.0)

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self._glide(0.0)


class CanvasHud(QObject):
    MARGIN = 12
    top_offset = None             # fn() -> px taken at the top of the canvas (the floating hint card)

    def __init__(self, canvas, enabled: bool = True):
        super().__init__(canvas)
        self.canvas = canvas
        # children of the view, not of its viewport: a viewport scrolls its children along when the image is zoomed
        # or panned (QAbstractScrollArea::scrollContentsBy), which made the read-outs drift and jump
        self.file = _InfoChip(canvas)
        self.flag = _Chip(canvas, "review-flag")
        self.pos = _Chip(canvas, readout=True)
        self.enabled = enabled
        canvas.installEventFilter(self)
        canvas.viewport().installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show):
            self.place()
        return False

    def place(self):
        g = self.canvas.viewport().geometry()           # the image area inside the view (no scroll bars)
        m = self.MARGIN
        top = g.top() + m + (self.top_offset() if self.top_offset else 0)      # below a hint card, if any
        self.file.move(g.left() + m, top)
        self.flag.move(max(g.left() + m, g.right() + 1 - self.flag.width() - m), top)
        self.pos.move(max(g.left() + m, g.right() + 1 - self.pos.width() - m),
                      max(g.top() + m, g.bottom() + 1 - self.pos.height() - m))
        for c in (self.file, self.flag, self.pos):
            c.raise_()

    def set_enabled(self, on: bool):
        self.enabled = bool(on)
        if not on:
            for c in (self.file, self.flag, self.pos):
                c.setVisible(False)

    def set_file(self, name: str = "", details: str = ""):
        self.pos.reset_width()
        if self.enabled:
            self.file.set(name, details)
            self.place()

    def set_flag(self, status, note: str = ""):
        if not self.enabled:
            return
        if status == "flag":
            self.flag.set("Flagged" + (f" · {note}" if note else ""), "", "#F7C873")
        else:
            self.flag.set("")
        self.place()

    def set_pointer(self, text: str):
        if self.enabled:
            self.pos.set("", text)
            self.place()
