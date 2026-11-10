# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Ribbon — the QC Panel as a slim tab row that gives the image its space back.

    collapsed (default)  only the tab row (~30 px). Clicking a tab — or resting the mouse on the tab row for a moment
                         — slides the tools down OVER the image (the canvas does not resize). The drawer has ONE
                         height for every tab, so switching tabs (click, or rest on another tab) just swaps the
                         tools in place, no new animation. It slides back up when a tool is launched, on Esc, on a
                         click elsewhere, on a second click on the open tab, or when the mouse stays away from the
                         tabs and the drawer for a moment.
    pinned (📌)           the tools stay open under the tabs, like a classic tool panel.
Double-clicking a tab toggles the pin too. The choice is remembered (QSettings "ribbon/pinned").

Animation: the drawer is an overlay INSIDE the main window (no OS popup window to resize every frame). For the slide
the tools are rendered once into a pixmap and a light "curtain" blits that pixmap at the eased offset — one cheap
paint per frame, ticked at the screen's refresh rate (FrameAnimator) — then the live tools are swapped in. Nothing is
re-laid out during the motion, so it stays perfectly smooth even on 144 / 240 Hz screens.

API kept compatible with the QTabWidget it replaces: addTab, setTabText, setTabToolTip, setTabIcon, setIconSize,
count, currentIndex, setCurrentIndex, widget.
"""
from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QEvent, QPoint, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QLinearGradient, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractButton, QApplication, QFrame, QHBoxLayout, QSizePolicy, QStackedWidget, QTabBar, QToolButton,
    QVBoxLayout, QWidget,
)

from AIO_QC.utils import style as apply_style
from .mini_session import FrameAnimator

OPEN_MS, CLOSE_MS, SWITCH_MS = 230, 170, 150
HOVER_OPEN_MS = 450           # resting on the tab row this long opens the drawer
HOVER_SWITCH_MS = 220         # resting on another tab (drawer open) switches to it
LEAVE_CLOSE_MS = 650          # mouse away from tabs + drawer this long closes it
SHADOW = 10


class _Curtain(QWidget):
    """Paints the snapshot of the tools at the animated offset (plus a soft shadow under it)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.pix: QPixmap | None = None
        self.t = 0.0

    def paintEvent(self, _):
        if self.pix is None:
            return
        p = QPainter(self)
        dpr = self.pix.devicePixelRatio() or 1.0
        h = self.pix.height() / dpr
        y = -(1.0 - self.t) * h                                   # slides down from under the tab row
        p.setOpacity(0.25 + 0.75 * self.t)
        p.drawPixmap(QPoint(0, round(y)), self.pix)
        bottom = y + h
        if bottom > 0:
            g = QLinearGradient(0, bottom, 0, bottom + SHADOW)
            g.setColorAt(0.0, QColor(0, 0, 0, round(70 * self.t)))
            g.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setOpacity(1.0)
            p.fillRect(QRect(0, round(bottom), self.width(), SHADOW), g)
        p.end()


class _Drawer(QWidget):
    """Overlay child of the main window: the live tools (frame) + the curtain used while animating."""

    def __init__(self, window):
        super().__init__(window)
        self.frame = QFrame(self)
        self.frame.setObjectName("ribbonDrop")
        apply_style(self.frame, "QFrame#ribbonDrop { background: palette(window); border: 1px solid @line; "
                                "border-top: none; border-bottom-left-radius: 6px; border-bottom-right-radius: 6px; }")
        self.lay = QVBoxLayout(self.frame)
        self.lay.setContentsMargins(6, 4, 6, 6)
        self.curtain = _Curtain(self)
        self.hide()

    def paintEvent(self, _):                                      # shadow under the live tools
        if not self.frame.isVisible():
            return
        p = QPainter(self)
        y = self.frame.height()
        g = QLinearGradient(0, y, 0, y + SHADOW)
        g.setColorAt(0.0, QColor(0, 0, 0, 70))
        g.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(QRect(0, y, self.width(), SHADOW), g)
        p.end()


class Ribbon(QWidget):
    pinnedChanged = pyqtSignal(bool)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.bar = QTabBar()
        self.bar.setDrawBase(False)
        self.bar.setExpanding(False)
        self.bar.setDocumentMode(True)
        self.stack = QStackedWidget(self)
        self.stack.hide()
        self.pin = QToolButton()
        self.pin.setCheckable(True)
        self.pin.setAutoRaise(True)
        self.pin.setText("📌")
        self.pin.toggled.connect(self.set_pinned)
        top = QHBoxLayout()
        top.setContentsMargins(4, 0, 4, 0)
        top.setSpacing(2)
        top.addWidget(self.bar)
        top.addStretch(1)
        top.addWidget(self.pin)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(0)
        self.lay.addLayout(top)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.drawer: _Drawer | None = None
        self._anim: FrameAnimator | None = None
        self._open = False                                    # target state of the drawer
        self.bar.setMouseTracking(True)
        self.bar.installEventFilter(self)
        self._hover_timer = QTimer(self, singleShot=True, timeout=self._hover_fire)
        self._leave_timer = QTimer(self, singleShot=True, interval=LEAVE_CLOSE_MS, timeout=self._leave_fire)
        self.bar.tabBarClicked.connect(self._on_tab_clicked)
        self.bar.tabBarDoubleClicked.connect(lambda _i: self.pin.toggle())
        self.bar.currentChanged.connect(self.stack.setCurrentIndex)
        pinned = bool(settings.value("ribbon/pinned", False, type=bool)) if settings is not None else False
        self._pinned = None
        self.pin.setChecked(pinned)
        self.set_pinned(pinned)

    # ------------------------------------------------------------------ QTabWidget-compatible API
    def addTab(self, widget, icon, title: str) -> int:
        self.stack.addWidget(widget)
        for b in widget.findChildren(QAbstractButton):           # launching a tool closes the drop-down
            if isinstance(b, QToolButton) and not b.isCheckable():
                b.clicked.connect(self._after_action)
        i = self.bar.addTab(icon, title)
        self._fit_height()
        return i

    def setTabText(self, i, text):
        self.bar.setTabText(i, text)

    def setTabToolTip(self, i, tip):
        self.bar.setTabToolTip(i, tip)

    def setTabIcon(self, i, icon):
        self.bar.setTabIcon(i, icon)

    def setIconSize(self, size: QSize):
        self.bar.setIconSize(size)

    def count(self) -> int:
        return self.bar.count()

    def currentIndex(self) -> int:
        return self.bar.currentIndex()

    def setCurrentIndex(self, i):
        self.bar.setCurrentIndex(i)

    def widget(self, i):
        return self.stack.widget(i)

    # ------------------------------------------------------------------ pin
    @property
    def pinned(self) -> bool:
        return bool(self._pinned)

    def set_pinned(self, on: bool):
        on = bool(on)
        if on == self._pinned:
            return
        self.close_drop(animate=False)
        from .anim import crossfade
        w = self.window()
        if w is not None and w is not self and w.isVisible():
            crossfade(w, 200)                       # the canvas grows / shrinks smoothly instead of jumping
        self._pinned = on
        if on:
            if self.drawer is not None:
                self.drawer.lay.removeWidget(self.stack)
            self.stack.setParent(self)
            self.lay.addWidget(self.stack)
            self.stack.show()
        else:
            self.lay.removeWidget(self.stack)
            self.stack.hide()
        self._fit_height()
        self.pin.blockSignals(True)
        self.pin.setChecked(on)
        self.pin.blockSignals(False)
        self.pin.setToolTip("Unpin: tools slide down over the image when a tab is clicked (more room for the image)"
                            if on else "Pin: keep the tools open under the tabs (double-click a tab does the same)")
        if self.settings is not None:
            self.settings.setValue("ribbon/pinned", on)
        self.pinnedChanged.emit(on)

    def _fit_height(self):
        """Never taller than needed — a window layout saved with the old, taller panel cannot stretch it."""
        row = max(self.bar.sizeHint().height(), self.pin.sizeHint().height())
        if self.pinned:
            pages = max((self.stack.widget(i).sizeHint().height() for i in range(self.stack.count())), default=0)
            self.setMaximumHeight(row + pages + 2)
        else:
            self.setMaximumHeight(row + 2)
        self.updateGeometry()

    # ------------------------------------------------------------------ drawer
    def is_open(self) -> bool:
        return self._open

    def _ensure_drawer(self) -> _Drawer:
        win = self.window()
        if self.drawer is None or self.drawer.parent() is not win:
            self.drawer = _Drawer(win)
        if self.stack.parent() is not self.drawer.frame:
            self.drawer.lay.addWidget(self.stack)
        return self.drawer

    def _geometry(self) -> QRect:
        """Drawer rectangle in main-window coordinates: under the tab row, as wide as the ribbon."""
        d = self.drawer
        # one height for every tab: switching tabs never changes the drawer's size
        h = max((self.stack.widget(i).sizeHint().height() for i in range(self.stack.count())), default=60) + 10
        top_left = self.mapTo(self.window(), QPoint(0, self.bar.geometry().bottom() + 1))
        w = max(self.width(), 320)
        d.frame.setGeometry(0, 0, w, h)
        d.curtain.setGeometry(0, 0, w, h + SHADOW)
        return QRect(top_left, QSize(w, h + SHADOW))

    def _snapshot(self) -> QPixmap:
        d = self.drawer
        d.frame.layout().activate()
        return d.frame.grab()

    def _on_tab_clicked(self, i: int):
        if i < 0 or self.pinned:
            return
        self._hover_timer.stop()
        if self._open and i == self.bar.currentIndex():
            self.close_drop()                                 # second click on the open tab: slide up
            return
        if self._open:
            self._switch(i)                                   # same drawer, other tools — no new animation
            return
        self.bar.setCurrentIndex(i)
        self.stack.setCurrentIndex(i)
        self.open_drop()

    def _switch(self, i: int):
        """Swap the tools shown in the open drawer in place (also mid-animation: the sliding picture is updated)."""
        if i == self.bar.currentIndex():
            return
        self.bar.setCurrentIndex(i)
        self.stack.setCurrentIndex(i)
        d = self.drawer
        if d is not None and not d.frame.isVisible() and d.curtain.isVisible():
            d.frame.show()
            d.curtain.pix = self._snapshot()
            d.frame.hide()
            d.curtain.update()

    # ---- keyboard: Alt+1… opens a tab with the focus on its first tool; arrows move, Enter runs, Esc closes
    def open_tab_keyboard(self, i: int):
        if not 0 <= i < self.count():
            return
        if self.pinned:
            self.setCurrentIndex(i)
        elif self._open:
            self._switch(i)
        else:
            self.bar.setCurrentIndex(i)
            self.stack.setCurrentIndex(i)
            self.open_drop()
        QTimer.singleShot(0 if self.pinned else OPEN_MS + 10, lambda: self._focus_first(i))

    def _page_buttons(self, i: int) -> list:
        from PyQt6.QtWidgets import QAbstractButton
        page = self.stack.widget(i)
        if page is None:
            return []
        return [b for b in page.findChildren(QAbstractButton) if b.isVisibleTo(page) and b.isEnabled()]

    def _focus_first(self, i: int):
        bs = self._page_buttons(i)
        for b in bs:
            if not b.property("_kbnav"):
                b.installEventFilter(self)
                b.setProperty("_kbnav", True)
            b.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if bs:
            bs[0].setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _nav(self, btn, key) -> bool:
        bs = self._page_buttons(self.stack.currentIndex())
        if btn not in bs:
            return False
        c = btn.mapTo(self.stack, btn.rect().center())
        best, score = None, None
        for b in bs:
            if b is btn:
                continue
            d = b.mapTo(self.stack, b.rect().center()) - c
            dx, dy = d.x(), d.y()
            ok = {Qt.Key.Key_Right: dx > 4, Qt.Key.Key_Left: dx < -4,
                  Qt.Key.Key_Down: dy > 4, Qt.Key.Key_Up: dy < -4}[key]
            if not ok:
                continue
            along, across = (abs(dx), abs(dy)) if key in (Qt.Key.Key_Left, Qt.Key.Key_Right) else (abs(dy), abs(dx))
            sc = along + 3 * across
            if score is None or sc < score:
                best, score = b, sc
        if best is not None:
            best.setFocus(Qt.FocusReason.TabFocusReason)
        return True

    # ---- hover: rest on the tab row to open, on another tab to switch; stay away to close
    def _hover_fire(self):
        if self.pinned or not self.isVisible():
            return
        i = self.bar.tabAt(self.bar.mapFromGlobal(QCursor.pos()))
        if i < 0 or QApplication.mouseButtons() != Qt.MouseButton.NoButton:
            return
        if self._open:
            self._switch(i)
        else:
            self.bar.setCurrentIndex(i)
            self.stack.setCurrentIndex(i)
            self.open_drop()

    def _in_zone(self, gp: QPoint) -> bool:
        """Tab row (the whole ribbon) or the drawer."""
        if QRect(self.mapToGlobal(QPoint(0, 0)), self.size()).contains(gp):
            return True
        d = self.drawer
        return d is not None and d.isVisible() and QRect(d.mapToGlobal(QPoint(0, 0)), d.frame.size()).contains(gp)

    def _leave_fire(self):
        if not self._open:
            return
        busy = QApplication.activePopupWidget() is not None or QApplication.mouseButtons() != Qt.MouseButton.NoButton
        if busy or self._in_zone(QCursor.pos()):
            if busy:
                self._leave_timer.start()                     # a combo list / a drag in progress: ask again later
            return
        self.close_drop()

    def open_drop(self, switching: bool = False):
        d = self._ensure_drawer()
        self._stop_anim()
        d.setGeometry(self._geometry())
        d.curtain.pix = None                                   # nothing painted until the snapshot is ready
        d.show()
        d.raise_()
        self.stack.show()
        d.frame.show()
        pix = self._snapshot()                                 # rendered once; no paint happens in between
        d.frame.hide()
        d.curtain.pix, d.curtain.t = pix, (0.45 if switching else 0.0)
        d.curtain.show()
        d.curtain.raise_()
        self._open = True
        QApplication.instance().installEventFilter(self)
        t0 = d.curtain.t

        def step(t):
            d.curtain.t = t0 + (1.0 - t0) * t
            d.curtain.update()

        def done():
            d.curtain.hide()
            d.curtain.pix = None
            d.frame.show()
            d.update()

        self._anim = FrameAnimator(self, SWITCH_MS if switching else OPEN_MS, step, done, QEasingCurve.Type.OutCubic)
        self._anim.start()

    def close_drop(self, animate: bool = True):
        d = self.drawer
        if not self._open or d is None:
            return
        self._open = False
        self._leave_timer.stop()
        self._hover_timer.stop()
        QApplication.instance().removeEventFilter(self)
        self._stop_anim()
        if not animate or not d.isVisible():
            d.hide()
            return
        pix = self._snapshot() if d.frame.isVisible() else d.curtain.pix
        t0 = d.curtain.t if not d.frame.isVisible() else 1.0
        d.frame.hide()
        d.curtain.pix, d.curtain.t = pix, t0
        d.curtain.show()

        def step(t):
            d.curtain.t = t0 * (1.0 - t)
            d.curtain.update()

        def done():
            d.hide()
            d.curtain.pix = None

        self._anim = FrameAnimator(self, max(60, round(CLOSE_MS * t0)), step, done, QEasingCurve.Type.InCubic)
        self._anim.start()

    def _stop_anim(self):
        if self._anim is not None and self._anim.running():
            self._anim.stop()
        self._anim = None

    def _after_action(self):
        if not self.pinned:
            self.close_drop()

    # ------------------------------------------------------------------ close on outside click / Esc, follow moves
    def eventFilter(self, obj, e):
        t = e.type()
        if t == QEvent.Type.KeyPress and obj is not self.bar and obj.property("_kbnav"):
            k = e.key()
            if k in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down) and self._nav(obj, k):
                return True
            if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                obj.click()
                return True
        if obj is self.bar and not self.pinned:                # hover on the tab row
            if t in (QEvent.Type.MouseMove, QEvent.Type.HoverMove, QEvent.Type.Enter):
                pos = e.position().toPoint() if hasattr(e, "position") else self.bar.mapFromGlobal(QCursor.pos())
                i = self.bar.tabAt(pos)
                if i >= 0 and (not self._open or i != self.bar.currentIndex()):
                    if not self._hover_timer.isActive() or getattr(self, "_hover_tab", None) != i:
                        self._hover_tab = i
                        self._hover_timer.start(HOVER_SWITCH_MS if self._open else HOVER_OPEN_MS)
                elif i < 0 or i == self.bar.currentIndex():
                    self._hover_timer.stop()
            elif t == QEvent.Type.Leave:
                self._hover_timer.stop()
        if self._open and t == QEvent.Type.MouseMove and hasattr(e, "globalPosition"):
            if self._in_zone(e.globalPosition().toPoint()):
                self._leave_timer.stop()
            elif not self._leave_timer.isActive():
                self._leave_timer.start()
        elif self._open and t == QEvent.Type.Leave and obj is self.window():
            if not self._leave_timer.isActive():
                self._leave_timer.start()                     # left the window altogether
        if self._open and t == QEvent.Type.MouseButtonPress:
            gp = e.globalPosition().toPoint()
            d = self.drawer
            inside = d is not None and QRect(d.mapToGlobal(QPoint(0, 0)), d.frame.size()).contains(gp)
            on_ribbon = QRect(self.mapToGlobal(QPoint(0, 0)), self.size()).contains(gp)
            popup = QApplication.activePopupWidget() is not None          # e.g. a combo list opened from the tools
            if not inside and not on_ribbon and not popup:
                self.close_drop()                                         # the click still reaches its target
        elif self._open and t == QEvent.Type.KeyPress and e.key() == Qt.Key.Key_Escape:
            self.close_drop()
            return True
        elif self._open and t == QEvent.Type.Resize and obj is self.window():
            self.drawer.setGeometry(self._geometry())
        return False

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._open and self.drawer is not None and not (self._anim and self._anim.running()):
            self.drawer.setGeometry(self._geometry())

    def hideEvent(self, e):
        super().hideEvent(e)
        self.close_drop(animate=False)                        # focus mode / panel hidden: no orphaned drawer
