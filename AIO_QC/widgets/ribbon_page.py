# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
3.0.1: the tools of one QC drawer tab, laid out like a professional ribbon.

    wide enough   large buttons: the icon on one top line for every tool, its name below in at most two lines
                  (never cut with "…"); groups divided by thin lines; the tab's question on the right
    narrower      compact buttons: icon + name on one line, three to a column, column after column
    narrower yet  the compact columns scroll sideways (wheel or drag the thin bar) — nothing is ever cut

    page = RibbonPage([[act, act], [act, (act, [sub_act, sub_act])]], question="Is it done …?",
                      labels={"screenshots": "Screenshots"})
A (action, [menu actions]) item is one button with a small menu (e.g. Hand-off ▸ Export / Import).
"""
from __future__ import annotations

from PyQt6.QtCore import QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPen
from PyQt6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QMenu, QScrollArea, QSizePolicy, QToolButton, \
    QWidget

ICON_L, ICON_S = 24, 16
LARGE_H = 74               # icon + two lines of text
SMALL_H = 24
ROWS = 3                   # compact buttons per column
PAGE_H = LARGE_H + 8 + 8   # + room for the thin sideways scroll bar (only shown when a page can't fit)


def _tok(k):
    from AIO_QC.widgets.design import tokens
    return tokens()[k]


def _clean(t: str) -> str:
    t = " ".join(t.split())                        # tool names may carry their own line breaks
    import re
    t = re.sub(r"&(?=\S)", "", t.replace("&&", "\0")).replace("\0", "&")       # mnemonics go, "A & B" stays
    return t.replace("…", "").replace("...", "").strip()


class RibbonButton(QToolButton):
    def __init__(self, action, label: str | None = None, menu_actions=None, parent=None):
        super().__init__(parent)
        self.setDefaultAction(action)
        self.setAutoRaise(True)
        self.label = _clean(label or action.iconText() or action.text())
        self.large = True
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        if menu_actions:
            m = QMenu(self)
            for a in menu_actions:
                m.addAction(a)
            self.setMenu(m)
            self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._sync_tip()
        action.changed.connect(self._sync_tip)

    def _sync_tip(self):
        a = self.defaultAction()
        if a is None:
            return
        tip = a.toolTip() or a.text()
        sc = a.shortcut().toString(a.shortcut().SequenceFormat.NativeText) if not a.shortcut().isEmpty() else ""
        self.setToolTip(f"{_clean(a.text())} ({sc}) — {tip}" if sc and sc not in tip else
                        (f"{_clean(a.text())} — {tip}" if tip and _clean(tip) != _clean(a.text()) else tip))

    def set_large(self, on: bool):
        if self.large != on:
            self.large = on
            self.updateGeometry()
            self.update()

    # ---- text
    def _bold(self) -> QFont:
        return QFont(self.font())

    def lines(self) -> list[str]:
        """One line if short, else two lines split at the space nearest the middle (no "…", ever)."""
        fm = QFontMetrics(self.font())
        t = self.label
        if fm.horizontalAdvance(t) <= 72 or " " not in t:
            return [t]
        best = None
        for i, ch in enumerate(t):
            if ch == " ":
                a, b = t[:i], t[i + 1:]
                w = max(fm.horizontalAdvance(a), fm.horizontalAdvance(b))
                if best is None or w < best[0]:
                    best = (w, [a, b])
        return best[1]

    def sizeHint(self):
        fm = QFontMetrics(self.font())
        extra = 10 if self.menu() is not None else 0
        if self.large:
            w = max(fm.horizontalAdvance(x) for x in self.lines()) + 18 + extra
            return QSize(max(58, w), LARGE_H)
        return QSize(8 + ICON_S + 7 + fm.horizontalAdvance(self.label) + 10 + extra, SMALL_H)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        on = self.isEnabled()
        if self.isDown() or (self.isChecked() and on):
            bg = _tok("accent_soft") if self.isChecked() and not self.isDown() else _tok("pressed")
        elif self.underMouse() and on:
            bg = _tok("hover")
        else:
            bg = None
        if bg:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(bg))
            p.drawRoundedRect(r, 6, 6)
        if self.hasFocus():
            p.setPen(QPen(QColor(_tok("accent")), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
        mode = QIcon.Mode.Normal if on else QIcon.Mode.Disabled
        ink = QColor(_tok("accent_text" if self.isChecked() else "text") if on else _tok("faint"))
        p.setPen(ink)
        p.setFont(self.font())
        fm = QFontMetrics(self.font())
        menu = self.menu() is not None
        if self.large:
            ir = QRect(0, 0, ICON_L, ICON_L)
            ir.moveCenter(self.rect().center())
            ir.moveTop(7)                                          # the same icon line for every tool
            self.icon().paint(p, ir, Qt.AlignmentFlag.AlignCenter, mode)
            y = ir.bottom() + 6
            for line in self.lines():
                p.drawText(QRect(0, y, self.width(), fm.height()), int(Qt.AlignmentFlag.AlignHCenter), line)
                y += fm.height() - 1
            if menu:
                self._chevron(p, ink, self.width() - 9, 14)
        else:
            ir = QRect(8, (self.height() - ICON_S) // 2, ICON_S, ICON_S)
            self.icon().paint(p, ir, Qt.AlignmentFlag.AlignCenter, mode)
            x = ir.right() + 8
            p.drawText(QRect(x, 0, self.width() - x, self.height()), int(Qt.AlignmentFlag.AlignVCenter), self.label)
            if menu:
                self._chevron(p, ink, self.width() - 8, self.height() / 2)

    @staticmethod
    def _chevron(p, ink, x, y):
        c = QColor(ink)
        c.setAlphaF(0.7)
        p.setPen(QPen(c, 1.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        from PyQt6.QtCore import QPointF
        p.drawPolyline([QPointF(x - 3, y - 1.5), QPointF(x, y + 1.5), QPointF(x + 3, y - 1.5)])

    def enterEvent(self, e):
        super().enterEvent(e)
        self.update()

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self.update()


class _Sep(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(13)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setPen(QPen(QColor(_tok("border")), 1))
        x = self.width() // 2
        p.drawLine(x, 8, x, self.height() - 8)


class RibbonPage(QWidget):
    """Groups of tools; large or compact by the width it has (see the module doc)."""

    def __init__(self, groups, question: str = "", labels: dict | None = None, extra=None, parent=None):
        super().__init__(parent)
        labels = labels or {}
        self.groups = []
        self.small_groups = set()
        for g in groups:
            if isinstance(g, tuple) and g and g[0] == "small":     # always compact: three to a column
                self.small_groups.add(len(self.groups))
                g = g[1]
            btns = []
            for item in g:
                act, sub = (item if isinstance(item, tuple) else (item, None))
                key = act.objectName() or act.text()
                btns.append(RibbonButton(act, labels.get(key), sub))
            if btns:
                self.groups.append(btns)
        self.extra = extra                           # e.g. the review progress label (shown after the tools)
        self.question = QLabel(question)
        self.question.setObjectName("ribbonQuestion")
        f = self.question.font()
        f.setItalic(True)
        self.question.setFont(f)
        self.question.setStyleSheet(f"color:{_tok('faint')};")
        if not question:                             # never show() before it has a parent (a stray window)
            self.question.hide()
        self.scroll = QScrollArea(self)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }"
                                  "QScrollBar:horizontal { height: 8px; }"
                                  "QScrollBar::handle:horizontal { margin: 2px 2px; border-radius: 2px; }")
        self.inner = QWidget()
        self.grid = QGridLayout(self.inner)
        self.grid.setContentsMargins(6, 3, 6, 3)
        self.grid.setHorizontalSpacing(2)
        self.grid.setVerticalSpacing(1)
        self.scroll.setWidget(self.inner)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 10, 0)
        outer.setSpacing(8)
        outer.addWidget(self.scroll, 1)
        outer.addWidget(self.question)
        self.mode = None
        self._seps = []
        self._layout("large")

    def _shown_groups(self):
        """The groups without the tools Simple menus leave out (their actions carry aio_simple_hidden)."""
        out = []
        for gi, g in enumerate(self.groups):
            vis = [b for b in g if not (b.defaultAction() is not None
                                        and (b.defaultAction().property("aio_simple_hidden")
                                             or b.defaultAction().property("aio_type_hidden")))]
            for b in g:
                if b not in vis:
                    b.setVisible(False)
            if vis:
                out.append((gi, vis))
        return out

    def refilter(self):
        self.mode = None
        self.relayout()

    def buttons(self):
        return [b for _gi, g in self._shown_groups() for b in g]

    def sizeHint(self):
        return QSize(self._large_width() + self.question.sizeHint().width() + 24, PAGE_H)

    def minimumSizeHint(self):
        return QSize(120, PAGE_H)

    def _large_width(self):
        w = 0
        shown = self._shown_groups()
        for gi, g in shown:
            if gi in self.small_groups:
                w += sum(max(_small_w(b) for b in g[c:c + ROWS]) for c in range(0, len(g), ROWS))
            else:
                w += sum(_large_w(b) for b in g) + 2 * (len(g) - 1)
        w += 13 * max(0, len(shown) - 1) + 12
        if self.extra is not None and self.extra.isVisibleTo(self):
            w += self.extra.sizeHint().width() + 8
        return w

    def _layout(self, mode):
        if mode == self.mode:
            return
        self.mode = mode
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w is not None and (w in self._seps or w is getattr(self, "_filler", None)):
                w.hide()                      # an old filler / divider left in place would cover the buttons
                w.setParent(None)
                w.deleteLater()
        self._seps = []
        for c in range(self.grid.columnCount()):
            self.grid.setColumnStretch(c, 0)
        col = 0
        large = mode == "large"
        for n, (gi, g) in enumerate(self._shown_groups()):
            if n:
                s = _Sep()
                self._seps.append(s)
                self.grid.addWidget(s, 0, col, ROWS, 1)
                col += 1
            small = (not large) or gi in self.small_groups
            for i, b in enumerate(g):
                b.set_large(not small)
                if not small:
                    self.grid.addWidget(b, 0, col, ROWS, 1, Qt.AlignmentFlag.AlignTop)
                    col += 1
                else:
                    self.grid.addWidget(b, i % ROWS, col + i // ROWS, Qt.AlignmentFlag.AlignLeft)
            if small:
                col += (len(g) + ROWS - 1) // ROWS
        for _gi, g in self._shown_groups():
            for b in g:
                if b.parentWidget() is not None:      # shown again only once inside the page: a button without a
                    b.setVisible(True)                # parent would open as a window of its own (a flash at start)
        if self.extra is not None:
            self.grid.addWidget(self.extra, 0, col, ROWS, 1, Qt.AlignmentFlag.AlignVCenter)
            col += 1
        self.grid.setColumnStretch(col, 1)
        self._filler = QWidget()
        self._filler.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.grid.addWidget(self._filler, 0, col)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.relayout()

    def showEvent(self, e):
        super().showEvent(e)
        self.relayout()

    def relayout(self):
        avail = self.width() - 10
        need = self._large_width()
        q = self.question.sizeHint().width() + 8 if self.question.text() else 0
        self.question.setVisible(bool(self.question.text()) and need + q <= avail)
        self._layout("large" if need <= avail else "small")


def _small_w(b: RibbonButton) -> int:
    was = b.large
    b.large = False
    w = b.sizeHint().width()
    b.large = was
    return w


def _large_w(b: RibbonButton) -> int:
    was = b.large
    b.large = True
    w = b.sizeHint().width()
    b.large = was
    return w
