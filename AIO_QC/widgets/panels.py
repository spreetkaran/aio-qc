# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
3.0 side panels: dock title bars with a count badge, and row painters for the Labels, Annotations and File List
panels (count / id badges on the right, status dots, file names without the extension, shapes per file, a split
caption where the split changes). Painting only — the models, selection and keyboard behaviour stay as they were.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QPointF, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import (QDockWidget, QHBoxLayout, QLabel, QStyle, QStyledItemDelegate, QStyleOptionViewItem,
                             QToolButton, QWidget)

from AIO_QC import utils

COUNT_ROLE = Qt.ItemDataRole.UserRole + 11      # Labels: instances · Annotations: "#n"
HINT_ROLE = Qt.ItemDataRole.UserRole + 12       # Labels: "same as “car”?"


def _tok(role: str) -> QColor:
    return utils.theme_color(role)


def draw_badge(p: QPainter, right: int, cy: int, text: str, fill: QColor | None = None, fg: QColor | None = None,
               font: QFont | None = None) -> int:
    """A rounded count pill ending at x=right, centred on cy; returns its left x."""
    if font is not None:
        f = QFont(font)
    else:
        f = QFont(p.font())
        f.setPointSizeF(max(7.0, f.pointSizeF() * 0.85))
    fm = QFontMetrics(f)
    w = fm.horizontalAdvance(text) + 14
    h = fm.height() + 4
    r = QRectF(right - w, cy - h / 2, w, h)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(fill if fill is not None else _tok("pill"))
    p.drawRoundedRect(r, h / 2, h / 2)
    p.setFont(f)
    p.setPen(fg if fg is not None else _tok("muted"))
    p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
    p.restore()
    return int(r.left())


class BadgeDelegate(QStyledItemDelegate):
    """Default row (check box, icon, text) + a muted note after the text (HINT_ROLE) + a badge on the right
    (COUNT_ROLE). badge_plain=True draws the right text without a pill (Annotations' "#n")."""

    def __init__(self, parent=None, badge_plain: bool = False):
        super().__init__(parent)
        self.plain = badge_plain

    def sizeHint(self, opt, idx):
        s = super().sizeHint(opt, idx)
        return QSize(s.width() + 48, max(s.height(), 26))

    def paint(self, p, opt, idx):
        o = QStyleOptionViewItem(opt)
        self.initStyleOption(o, idx)
        badge = idx.data(COUNT_ROLE)
        hint = idx.data(HINT_ROLE)
        right = o.rect.right() - 8
        # reserve the badge's room so long names elide before it
        fm = QFontMetrics(o.font)
        reserve = (fm.horizontalAdvance(str(badge)) + 22) if badge not in (None, "") else 0
        o.rect = QRect(o.rect.left(), o.rect.top(), o.rect.width() - reserve, o.rect.height())
        style = o.widget.style() if o.widget is not None else None
        if style is not None:
            # full-width selection / hover background first, then the content in the narrowed rect
            bg = QStyleOptionViewItem(opt)
            self.initStyleOption(bg, idx)
            bg.text = ""
            bg.icon = bg.icon.__class__()
            bg.features &= ~QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
            style.drawPrimitive(QStyle.PrimitiveElement.PE_PanelItemViewItem, bg, p, o.widget)
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, o, p, o.widget)
        else:
            super().paint(p, o, idx)
        cy = opt.rect.center().y()
        if hint:
            text_rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, o, o.widget) if style else o.rect
            x = text_rect.left() + 4 + fm.horizontalAdvance(o.text) + 8
            f = QFont(o.font)
            f.setPointSizeF(max(7.0, f.pointSizeF() * 0.9))
            p.save()
            p.setFont(f)
            p.setPen(_tok("warn"))
            p.drawText(QRect(x, opt.rect.top(), max(0, right - reserve - x), opt.rect.height()),
                       int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), str(hint))
            p.restore()
        if badge not in (None, ""):
            if self.plain:
                p.save()
                p.setPen(_tok("muted"))
                p.drawText(QRect(right - 60, opt.rect.top(), 60, opt.rect.height()),
                           int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), str(badge))
                p.restore()
            else:
                draw_badge(p, right, cy, str(badge), font=o.font)


class FileRowDelegate(QStyledItemDelegate):
    """File List rows (3.0 mock-up): review status dot · name without the extension (mono) · shapes in the file,
    grouped by split — the first row of each split carries a header band (▾ TRAIN … 24) that folds the group."""

    STATUS = {"ok": "ok", "flag": "warn", "fixed": "info"}
    ROW, HEAD = 28, 28

    def __init__(self, dock, counts=None, classes=None):
        super().__init__(dock)
        self.dock = dock
        self._counts = counts           # () -> {key: Counter of labels} (session.index), or None
        self._classes = classes         # () -> True for a classification dataset: the row shows the class(es)
        self.split_sizes: dict[str, int] = {}
        m = dock.model
        for sig in (m.modelReset, m.rowsInserted, m.rowsRemoved):
            sig.connect(self._recount)
        self._recount()

    def _recount(self, *_a):
        from collections import Counter
        self.split_sizes = dict(Counter(s.split for s in self.dock.model.samples if s.split))

    # ---- helpers
    def _sample(self, idx):
        from AIO_QC.widgets.file_list import KEY_ROLE
        key = idx.data(KEY_ROLE)
        m = self.dock.model
        row = m.row_of(key) if key else -1
        return key, (m.samples[row] if 0 <= row < len(m.samples) else None)

    def _is_head(self, idx, s) -> bool:
        if s is None or not s.split:
            return False
        if idx.row() == 0:
            return True
        _pk, ps = self._sample(idx.sibling(idx.row() - 1, 0))
        return ps is None or ps.split != s.split

    def _folded(self, s) -> bool:
        return s is not None and s.split in self.dock.proxy.collapsed

    def sizeHint(self, opt, idx):
        _k, s = self._sample(idx)
        head = self._is_head(idx, s)
        h = (self.HEAD if head else 0) + (0 if head and self._folded(s) else self.ROW)
        return QSize(120, h)                      # never wider than the panel: names elide, no sideways scrolling

    def paint(self, p, opt, idx):
        key, s = self._sample(idx)
        model = self.dock.model
        r = QRect(opt.rect)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        head = self._is_head(idx, s)
        if head:
            hr = QRect(r.left(), r.top(), r.width(), self.HEAD)
            p.fillRect(hr, _tok("pill"))
            folded = self._folded(s)
            # chevron
            cx, cy = hr.left() + 18, hr.center().y()
            p.setPen(QPen(_tok("muted"), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            from PyQt6.QtGui import QPainterPath
            path = QPainterPath()
            if folded:
                path.moveTo(cx - 2, cy - 4)
                path.lineTo(cx + 2, cy)
                path.lineTo(cx - 2, cy + 4)
            else:
                path.moveTo(cx - 4, cy - 2)
                path.lineTo(cx, cy + 2)
                path.lineTo(cx + 4, cy - 2)
            p.drawPath(path)
            hf = QFont(opt.font)
            hf.setWeight(QFont.Weight.DemiBold)
            hf.setPixelSize(11)
            hf.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 108)
            p.setFont(hf)
            p.setPen(_tok("muted"))
            p.drawText(QRect(hr.left() + 32, hr.top(), hr.width() - 80, hr.height()),
                       int(Qt.AlignmentFlag.AlignVCenter), s.split.upper())
            n = self.split_sizes.get(s.split)
            if n:
                draw_badge(p, hr.right() - 10, cy, f"{n:,}", font=_mono_font(12))
            r = QRect(r.left(), r.top() + self.HEAD, r.width(), r.height() - self.HEAD)
            if folded or r.height() <= 0:
                p.restore()
                return
        # the file row
        ro = QStyleOptionViewItem(opt)
        ro.rect = r
        _row_bg(p, ro)
        rv = model.review.status(key) if model.review is not None and key else None
        cx, cy = r.left() + 20, r.center().y()
        if rv in self.STATUS:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_tok(self.STATUS[rv]))
            p.drawEllipse(QRectF(cx - 4, cy - 4, 8, 8))
        else:
            p.setPen(QPen(_tok("faint"), 1.3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(cx - 3.5, cy - 3.5, 7, 7))
        right = r.right() - 10
        right_text = ""
        if self._counts is not None and key is not None:
            c = (self._counts() or {}).get(key)
            if c is not None:
                n = sum(c.values())
                if self._classes is not None and self._classes():          # a classifier image: its class
                    names = sorted(c)
                    right_text = ("no class" if not names else names[0] if len(names) == 1 else
                                  f"{names[0]} +{len(names) - 1}")
                    right_text = QFontMetrics(opt.font).elidedText(right_text, Qt.TextElideMode.ElideRight, 120)
                else:
                    right_text = "no shapes" if not n else f"{n} shape{'s' if n != 1 else ''}"
        if right_text:
            f = QFont(opt.font)
            f.setPixelSize(12)
            p.setFont(f)
            p.setPen(_tok("muted"))
            tw = QFontMetrics(f).horizontalAdvance(right_text)
            p.drawText(QRect(right - tw - 4, r.top(), tw + 8, r.height()), int(Qt.AlignmentFlag.AlignRight) | int(Qt.AlignmentFlag.AlignVCenter),
                       right_text)
            right -= tw + 12
        name = os.path.splitext(s.name)[0] if s is not None else str(opt.text)
        mf = _mono_font(12.5)
        if opt.state & QStyle.StateFlag.State_Selected:
            mf.setWeight(QFont.Weight.Medium)
        p.setFont(mf)
        fg = idx.data(Qt.ItemDataRole.ForegroundRole)
        p.setPen(fg if isinstance(fg, QColor) else fg.color() if hasattr(fg, "color") else _tok("text"))
        x = r.left() + 36
        p.drawText(QRect(x, r.top(), max(0, right - x), r.height()), int(Qt.AlignmentFlag.AlignVCenter),
                   QFontMetrics(mf).elidedText(name, Qt.TextElideMode.ElideMiddle, max(0, right - x)))
        p.restore()

    def editorEvent(self, ev, model, opt, idx):
        """A click on a split header folds / opens the group (never selects the file under it)."""
        from PyQt6.QtCore import QEvent
        if ev.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
                         QEvent.Type.MouseButtonDblClick):
            _k, s = self._sample(idx)
            if self._is_head(idx, s) and ev.position().y() < opt.rect.top() + self.HEAD:
                if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                    self.dock.toggle_split(s.split)
                return True
        return super().editorEvent(ev, model, opt, idx)


class DockTitle(QWidget):
    """A dock's title bar: bold name, a count badge, optional buttons, then float / close."""
    HEIGHT = 40

    def __init__(self, dock: QDockWidget, title: str):
        super().__init__(dock)
        self.dock = dock
        self._wm = []                                   # float / close: shown while the header is hovered
        self.setObjectName("dockTitle")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        utils.style(self, "#dockTitle{background:@base; border-bottom:1px solid @line; border-top:1px solid @line;}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 0, 10, 0)
        lay.setSpacing(8)
        self.setFixedHeight(self.HEIGHT)                 # every panel header the same height, hovered or not
        self.title = QLabel(title)
        f = QFont(self.title.font())
        f.setWeight(QFont.Weight.DemiBold)
        self.title.setFont(f)
        self.count = QLabel("")
        self.count.setVisible(False)
        utils.style(self.count, "QLabel{color:@muted; background:@pill; border-radius:9px; padding:1px 7px;}")
        self.count.setFixedHeight(20)                      # a pill, not stretched to the header's height
        lay.addWidget(self.title, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.count, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addStretch(1)
        self.buttons = QHBoxLayout()
        self.buttons.setSpacing(2)
        lay.addLayout(self.buttons)
        for icon, tip, fn in (("restore-window", "Float / dock this panel",
                               lambda: dock.setFloating(not dock.isFloating())),
                              ("review-clear", "Close this panel", dock.close)):
            b = QToolButton()
            b.setAutoRaise(True)
            b.setIcon(utils.newIcon(icon))
            b.setProperty("aio_icon", icon)
            b.setIconSize(QSize(12, 12))
            b.setFixedSize(24, 24)
            b.setToolTip(tip)
            b.setAccessibleName(tip)
            b.clicked.connect(lambda _c=False, f=fn: f())
            sp = b.sizePolicy()
            sp.setRetainSizeWhenHidden(True)             # appearing on hover never moves or resizes anything
            b.setSizePolicy(sp)
            b.setVisible(False)                          # 3.0 mock-up: a clean header — shown while hovered
            self._wm.append(b)
            lay.addWidget(b)

    def add_help(self, page: str):
        """A quiet "?" in the header: this panel's page of the guide."""
        b = QToolButton()
        b.setAutoRaise(True)
        b.setIcon(utils.newIcon("question"))
        b.setProperty("aio_icon", "question")
        b.setIconSize(QSize(14, 14))
        b.setFixedSize(24, 24)
        tip = f"What is the {self.title.text()} panel? — its page of the guide"
        b.setToolTip(tip)
        b.setAccessibleName(tip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)

        def open_page(_c=False):
            from AIO_QC.widgets.help_center_bridge import open_help
            open_help(self.window(), page)
        b.clicked.connect(open_page)
        self.help_btn = b
        self.buttons.insertWidget(0, b)

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self.HEIGHT)

    def minimumSizeHint(self):
        # the name and the help "?" set the panel's narrowest width; float / close (shown on hover) may overlap
        # the count there rather than make every window wider
        need = 26 + self.title.minimumSizeHint().width() + (30 if getattr(self, "help_btn", None) else 0)
        return QSize(min(super().minimumSizeHint().width(), max(need, 96)), self.HEIGHT)

    def enterEvent(self, e):
        super().enterEvent(e)
        for b in self._wm:
            b.setVisible(True)

    def leaveEvent(self, e):
        super().leaveEvent(e)
        for b in self._wm:
            b.setVisible(False)

    def set_count(self, n):
        self.count.setText("" if n is None else f"{n:,}" if isinstance(n, int) else str(n))
        self.count.setVisible(n is not None)

    def add_widget(self, w):
        self.buttons.addWidget(w)
        return w

    def add_button(self, icon: str, tip: str, fn):
        b = QToolButton()
        b.setAutoRaise(True)
        b.setIcon(utils.newIcon(icon))
        b.setProperty("aio_icon", icon)
        b.setIconSize(QSize(16, 16))
        b.setFixedSize(24, 24)
        b.setToolTip(tip)
        b.setAccessibleName(tip)
        b.clicked.connect(lambda _c=False: fn())
        self.buttons.addWidget(b)
        return b


class EyeToggle(QToolButton):
    """An eye that closes: set_off(True) draws a slash across it while the eye dims, set_off(False) wipes the slash
    away — eased, one step per display frame (instant with Settings ▸ Animations ▸ Off). Drawn, not two icons."""

    def __init__(self, tip_on: str, tip_off: str, parent=None):
        super().__init__(parent)
        self.setAutoRaise(True)
        self.setFixedSize(24, 24)
        self.tips = (tip_on, tip_off)
        self.k = 0.0                     # 0 = open eye, 1 = struck through
        self.off = False
        self._anim = None
        self.setToolTip(tip_on)
        self.setAccessibleName(tip_on)

    def set_off(self, off: bool, animate: bool = True):
        off = bool(off)
        self.setToolTip(self.tips[1] if off else self.tips[0])
        self.setAccessibleName(self.toolTip())
        if off == self.off and self._anim is None:
            return
        self.off = off
        from PyQt6.QtCore import QEasingCurve
        from AIO_QC.widgets import motion
        from AIO_QC.widgets.mini_session import FrameAnimator
        if self._anim is not None:
            self._anim.stop()
        start, end = self.k, (1.0 if off else 0.0)
        ms = motion.scaled(260) if animate and self.isVisible() else 0
        if ms <= 0:
            self.k, self._anim = end, None
            self.update()
            return

        def step(t):
            self.k = start + (end - start) * t
            self.update()

        def done():
            self._anim = None
        self._anim = FrameAnimator(self, max(1, int(ms * abs(end - start)) or 1), step, done,
                                   QEasingCurve.Type.InOutCubic)
        self._anim.start()

    def paintEvent(self, _e):
        from PyQt6.QtGui import QPainterPath
        from PyQt6.QtGui import QIcon
        from PyQt6.QtWidgets import QStyle, QStyleOptionToolButton
        p = QPainter(self)
        opt = QStyleOptionToolButton()                       # the button itself: the app style sheet's hover /
        self.initStyleOption(opt)                            # pressed tone and radius, like its neighbours
        opt.icon, opt.text = QIcon(), ""
        self.style().drawComplexControl(QStyle.ComplexControl.CC_ToolButton, opt, p, self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        ink = QColor(_tok("text"))
        s = 16.0 / 256.0                                     # the eye-line icon's 256 grid, at 16 px
        ox, oy = (self.width() - 16) / 2, (self.height() - 16) / 2
        p.translate(ox, oy)
        p.scale(s, s)
        eye = QColor(ink)
        eye.setAlphaF(1.0 - 0.45 * self.k)
        pen = QPen(eye, 16, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        path.moveTo(24, 128)
        path.cubicTo(64, 56, 96, 56, 128, 56)
        path.cubicTo(160, 56, 192, 56, 232, 128)
        path.cubicTo(192, 200, 160, 200, 128, 200)
        path.cubicTo(96, 200, 64, 200, 24, 128)
        p.drawPath(path)
        r = 32 * (1.0 - 0.25 * self.k)                     # the pupil narrows a little as the eye closes
        p.drawEllipse(QPointF(128, 128), r, r)
        if self.k > 0.001:                                 # the slash, drawn from top-left as k grows
            k = self.k
            p.setPen(QPen(ink, 18, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(40, 40), QPointF(40 + 176 * k, 40 + 176 * k))
        p.end()

    def enterEvent(self, e):
        super().enterEvent(e)
        self.update()

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self.update()


HELP_PAGES = {   # panel -> its page of the guide (the "?" in its header). Image QC's panels have none: their names
    "Tables": "aio-data-studio", "Recipe": "aio-data-studio",      # say what they are (Files · Annotations · Labels)
}


def titled(dock: QDockWidget, title: str, help_page: str | None = None) -> DockTitle:
    t = DockTitle(dock, title)
    dock.setTitleBarWidget(t)
    page = help_page or HELP_PAGES.get(title)
    if page:
        t.add_help(page)
    return t


# ---------------------------------------------------------------------------------------------------- 3.0 rows
ROW_H = 30


def _mono_font(px=12):
    from AIO_QC.widgets.design import mono
    return mono(px)


def _row_bg(p: QPainter, opt, selected_bar: bool = True):
    """Hover / selection like the mock-up: a soft accent row with a 2 px accent bar on the left."""
    r = opt.rect
    sel = bool(opt.state & QStyle.StateFlag.State_Selected)
    hov = bool(opt.state & QStyle.StateFlag.State_MouseOver)
    if sel:
        p.fillRect(r, _tok("accent_soft"))
        if selected_bar:
            p.fillRect(QRect(r.left(), r.top(), 2, r.height()), _tok("accent"))
    elif hov:
        p.fillRect(r, _tok("pill"))


class LabelRowDelegate(QStyledItemDelegate):
    """Labels panel row: colour swatch · name · “same as …?” · count badge. No check box: the swatch is the
    visibility switch (click it, or Space) — filled = shown, hollow + faded name = hidden.
    Long names: the panel scrolls sideways, while the counts stay pinned to its right edge (always visible)."""

    SWATCH_ZONE = 34

    def __init__(self, view, colors):
        super().__init__(view)
        self.view = view
        self.colors = colors            # label -> QColor

    GAP = 14                            # between the end of the name (or hint) and the count
    FADE = 28                           # a name running under the count fades out over this many pixels
    SLACK = 10

    def _fonts(self, opt):
        f = QFont(opt.font)
        hf = QFont(f)
        hf.setPixelSize(max(10, f.pixelSize() - 1) if f.pixelSize() > 0 else 12)
        hf.setWeight(QFont.Weight.Medium)
        return f, hf

    def _parts(self, opt, idx):
        """(name, name width, hint, hint width, badge, badge width) in exact (fractional) pixels."""
        from PyQt6.QtGui import QFontMetricsF
        f, hf = self._fonts(opt)
        name = str(idx.data(Qt.ItemDataRole.DisplayRole) or "")
        hint = idx.data(HINT_ROLE)
        badge = idx.data(COUNT_ROLE)
        nw = QFontMetricsF(f).horizontalAdvance(name)
        hw = QFontMetricsF(hf).horizontalAdvance(str(hint)) if hint else 0.0
        bw = QFontMetrics(_mono_font(12)).horizontalAdvance(str(badge)) + 14 if badge not in (None, "") else 0.0
        return name, nw, hint, hw, badge, bw         # (bw exactly as draw_badge draws the pill)

    TEXT_X = 16 + 10 + 12               # after the swatch

    def sizeHint(self, opt, idx):
        import math
        _n, nw, hint, hw, _b, bw = self._parts(opt, idx)
        # scrolled fully right, the text ends a clear gap before the count (+ slack for rounding at any scaling)
        w = self.TEXT_X + nw + (10 + hw if hint else 0) + self.GAP + bw + 10 + self.SLACK
        return QSize(int(math.ceil(w)), ROW_H)

    def paint(self, p, opt, idx):
        from PyQt6.QtGui import QLinearGradient
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        vp = self.view.viewport().rect()
        row = QRect(vp.left(), opt.rect.top(), vp.width(), opt.rect.height())     # the visible part of the row
        o2 = QStyleOptionViewItem(opt)
        o2.rect = row
        _row_bg(p, o2, selected_bar=False)
        sel = bool(opt.state & QStyle.StateFlag.State_Selected)
        hov = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        bg = QColor(_tok("accent_soft") if sel else _tok("pill") if hov else _tok("surface"))
        r = opt.rect
        cy = r.center().y()
        label = idx.data(Qt.ItemDataRole.UserRole)
        shown = idx.data(Qt.ItemDataRole.CheckStateRole) != Qt.CheckState.Unchecked.value and \
            idx.data(Qt.ItemDataRole.CheckStateRole) != Qt.CheckState.Unchecked
        col = QColor(self.colors().get(label, _tok("muted"))) if callable(self.colors) else _tok("muted")
        name, nw, hint, hw, badge, bw = self._parts(opt, idx)
        f, hf = self._fonts(opt)
        # where the text may run: up to the count (pinned to the panel's right edge) and a gap
        limit = row.right() - 10 - (bw + self.GAP if bw else 0)
        sw = QRectF(r.left() + 16, cy - 5, 10, 10)
        if shown:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
        else:
            p.setPen(QPen(col, 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(sw, 3, 3)
        x = float(r.left() + self.TEXT_X)
        p.setFont(f)
        p.setPen(_tok("text") if shown else _tok("faint"))
        # never clipped mid-glyph: drawn whole (wide rect, no clip) — anything running under the count is
        # faded out below and covered by the count's own background
        p.drawText(QRectF(x, r.top(), nw + 40, r.height()),
                   int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), name)
        end = x + nw
        if hint:
            p.setFont(hf)
            p.setPen(_tok("warn"))
            hx = x + nw + 10
            p.drawText(QRectF(hx, r.top(), hw + 40, r.height()),
                       int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), str(hint))
            end = hx + hw
        if end > limit + 0.5:                         # the text runs on: fade it out before the count
            fade = QRectF(limit - self.FADE, r.top(), self.FADE, r.height())
            g = QLinearGradient(fade.left(), 0, fade.right(), 0)
            clear = QColor(bg)
            clear.setAlpha(0)
            g.setColorAt(0.0, clear)
            g.setColorAt(1.0, bg)
            p.fillRect(fade, g)
            p.fillRect(QRectF(limit, r.top(), row.right() + 1 - limit, r.height()), bg)
        if badge not in (None, ""):
            draw_badge(p, row.right() - 10, cy, str(badge), font=_mono_font(12))
        p.restore()

    def editorEvent(self, ev, model, opt, idx):
        from PyQt6.QtCore import QEvent
        if ev.type() in (QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick) and \
                ev.button() == Qt.MouseButton.LeftButton and ev.position().x() < opt.rect.left() + self.SWATCH_ZONE:
            if ev.type() == QEvent.Type.MouseButtonRelease:
                cur = idx.data(Qt.ItemDataRole.CheckStateRole)
                on = cur in (Qt.CheckState.Checked, Qt.CheckState.Checked.value)
                model.setData(idx, Qt.CheckState.Unchecked if on else Qt.CheckState.Checked,
                              Qt.ItemDataRole.CheckStateRole)
            return True
        if ev.type() == QEvent.Type.MouseButtonRelease:
            return False                      # a click elsewhere selects the row (never toggles)
        return super().editorEvent(ev, model, opt, idx)


class AnnotationRowDelegate(QStyledItemDelegate):
    """Annotations panel row: shape icon in the label colour · label · description (muted) · #n (mono, pinned to
    the panel's right edge). Long text scrolls sideways; text that would run under #n fades out just before it —
    glyphs are never cut."""

    GAP, FADE, TEXT_X = 14, 28, 16 + 16 + 12

    def __init__(self, view=None):
        super().__init__(view)
        self.view = view

    def _parts(self, opt, idx):
        from PyQt6.QtGui import QFontMetricsF
        text = str(idx.data(Qt.ItemDataRole.DisplayRole) or "")
        label, _sep, desc = text.partition(" · ")
        f = QFont(opt.font)
        f.setWeight(QFont.Weight.Medium)                 # measured at the selected weight: no jump on selection
        df = QFont(opt.font)
        badge = idx.data(COUNT_ROLE)
        lw = QFontMetricsF(f).horizontalAdvance(label)
        dw = QFontMetricsF(df).horizontalAdvance(" · " + desc) if desc else 0.0
        bw = QFontMetricsF(_mono_font(12)).horizontalAdvance(str(badge)) if badge not in (None, "") else 0.0
        return label, lw, desc, dw, badge, bw

    def sizeHint(self, opt, idx):
        import math
        _l, lw, _d, dw, _b, bw = self._parts(opt, idx)
        return QSize(int(math.ceil(self.TEXT_X + lw + dw + self.GAP + bw + 10 + 10)), ROW_H)

    def paint(self, p, opt, idx):
        from PyQt6.QtGui import QLinearGradient
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        view = self.view or opt.widget
        vp = view.viewport().rect() if view is not None else opt.rect
        row = QRect(vp.left(), opt.rect.top(), vp.width(), opt.rect.height())
        o2 = QStyleOptionViewItem(opt)
        o2.rect = row
        _row_bg(p, o2)
        sel = bool(opt.state & QStyle.StateFlag.State_Selected)
        hov = bool(opt.state & QStyle.StateFlag.State_MouseOver)
        bg = QColor(_tok("accent_soft") if sel else _tok("pill") if hov else _tok("surface"))
        r = opt.rect
        icon = idx.data(Qt.ItemDataRole.DecorationRole)
        if icon is not None and hasattr(icon, "paint"):
            icon.paint(p, QRect(r.left() + 16, r.center().y() - 8, 16, 16))
        label, lw, desc, dw, badge, bw = self._parts(opt, idx)
        limit = row.right() - 10 - (bw + self.GAP if bw else 0)
        f = QFont(opt.font)
        if sel:
            f.setWeight(QFont.Weight.Medium)
        p.setFont(f)
        x = float(r.left() + self.TEXT_X)
        fg = idx.data(Qt.ItemDataRole.ForegroundRole)
        p.setPen(fg.color() if hasattr(fg, "color") else fg if isinstance(fg, QColor) else _tok("text"))
        al = int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        p.drawText(QRectF(x, r.top(), lw + 40, r.height()), al, label)          # whole glyphs, never clipped
        end = x + lw
        if desc:
            p.setFont(QFont(opt.font))
            p.setPen(_tok("muted"))
            p.drawText(QRectF(end, r.top(), dw + 40, r.height()), al, " · " + desc)
            end += dw
        if end > limit + 0.5:                             # runs on under #n: fade it out before it
            fade = QRectF(limit - self.FADE, r.top(), self.FADE, r.height())
            g = QLinearGradient(fade.left(), 0, fade.right(), 0)
            clear = QColor(bg)
            clear.setAlpha(0)
            g.setColorAt(0.0, clear)
            g.setColorAt(1.0, bg)
            p.fillRect(fade, g)
            p.fillRect(QRectF(limit, r.top(), row.right() + 1 - limit, r.height()), bg)
        if badge not in (None, ""):
            mf = _mono_font(12)
            p.setFont(mf)
            p.setPen(_tok("muted"))
            p.drawText(QRectF(limit, r.top(), row.right() - 10 - limit, r.height()),
                       int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), str(badge))
        if sel:                                           # the accent bar stays at the panel's left edge
            p.fillRect(QRect(row.left(), r.top(), 2, r.height()), _tok("accent"))
        p.restore()
