# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Hover that does not depend on Qt's Enter / Leave events or on QApplication.mouseButtons().

Both go stale on Windows in ordinary situations — after a native dialog (the folder picker that opens a dataset), after
a drag and drop from Explorer, or while another of the app's windows is the active one — and stay stale
until the next click: Enter / Leave are then not delivered and mouseButtons() still reports the button of a click the
dialog consumed. Anything that opens on hover looked dead until it was clicked.

HoverWatch sees every real mouse move of the application (one event filter, a type check per event) and decides with
the cursor position itself and the buttons that move carries — always current. A watched widget gets
`entered()` / `left()` callbacks exactly once per crossing; the regular enterEvent / leaveEvent keep working too (the
watcher only adds the calls Qt dropped), so nothing changes where Qt's own events were fine.
"""
from __future__ import annotations

import weakref

from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication

_MOVES = (QEvent.Type.MouseMove, QEvent.Type.HoverMove, QEvent.Type.Enter, QEvent.Type.Leave,
          QEvent.Type.NonClientAreaMouseMove)


class HoverWatch(QObject):
    _inst = None

    @classmethod
    def instance(cls) -> "HoverWatch":
        if cls._inst is None:
            app = QApplication.instance()
            cls._inst = HoverWatch(app)
            app.installEventFilter(cls._inst)
            # out of the event path before Python tears the application down — an application-wide filter still
            # installed while the QApplication is destroyed at interpreter exit crashes the process
            app.aboutToQuit.connect(_uninstall)
            import atexit
            atexit.register(_uninstall)
        return cls._inst

    def __init__(self, parent):
        super().__init__(parent)
        self._items = []                     # [weakref(widget), on_enter, on_leave, on_move, inside]
        self.buttons = Qt.MouseButton.NoButton   # of the last real mouse event — never stale

    def watch(self, widget, on_enter=None, on_leave=None, on_move=None):
        """Call on_enter() / on_leave() when the cursor crosses `widget`'s visible rectangle, on_move(global pos)
        while it moves inside."""
        self._items.append([weakref.ref(widget), on_enter, on_leave, on_move, False])

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick):
            self.buttons = ev.buttons()
        elif t in _MOVES and self._items:
            if t in (QEvent.Type.MouseMove, QEvent.Type.HoverMove) and hasattr(ev, "buttons"):
                self.buttons = ev.buttons()
            self.update(QCursor.pos())
        return False

    def update(self, gp: QPoint):
        alive = []
        for item in self._items:
            w = item[0]()
            if w is None:
                continue
            alive.append(item)
            try:
                inside = w.isVisible() and QRect(w.mapToGlobal(QPoint(0, 0)), w.size()).contains(gp) and \
                    self._on_top(w, gp)
            except RuntimeError:                 # deleted under us
                continue
            if inside and not item[4]:
                item[4] = True
                if item[1] is not None:
                    item[1]()
            elif not inside and item[4]:
                item[4] = False
                if item[2] is not None:
                    item[2]()
            if inside and item[3] is not None:
                item[3](gp)
        self._items = alive

    @staticmethod
    def _on_top(w, gp) -> bool:
        """The cursor is over w itself (or one of its children), not over another window / an overlay above it."""
        top = QApplication.widgetAt(gp)
        while top is not None:
            if top is w:
                return True
            top = top.parentWidget()
        return False


def _uninstall():
    inst = HoverWatch._inst
    if inst is None:
        return
    HoverWatch._inst = None
    inst._items = []
    try:
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(inst)
    except RuntimeError:
        pass


def watch(widget, on_enter=None, on_leave=None, on_move=None):
    HoverWatch.instance().watch(widget, on_enter, on_leave, on_move)


def buttons():
    """The mouse buttons held, from the last real mouse event (QApplication.mouseButtons() can be stale)."""
    return HoverWatch.instance().buttons
