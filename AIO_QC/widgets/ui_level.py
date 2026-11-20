# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Make the app easy to start with, without taking anything away.

Simple menus (View ▸ Simple Menus, on for a new install, off for those who used the app before): the menus show the
everyday commands; the specialist ones (model tools, rules, text labels, hand-off …) are hidden from the menus and the
QC tabs. Nothing is removed — Ctrl+K still finds every command, its shortcut still works, and one click shows all.

    ui_level.install(window, advanced={"Model Check…", …} or callable(action) -> bool, reason=fn(action) -> str)

Explained buttons: every action's tooltip says what it does in plain words and its shortcut; while it is disabled it
also says why and what to do ("Open a dataset first — Ctrl+O"). Menus show these tooltips too.
"""
from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMenu

from AIO_QC.utils import app_settings
from AIO_QC.core.diag import ignored as _ignored

KEY = "ui/simple"
HIDDEN = "aio_simple_hidden"           # action property: left out of the menus by Simple menus (still runs, still listed)
BASE_TIP = "aio_base_tip"


class _Bus(QObject):
    changed = pyqtSignal(bool)


bus = _Bus()


def first_run() -> bool:
    """A new install: nothing opened before in any mode."""
    st = app_settings()
    from AIO_QC.utils.fx_settings import fx_settings
    return not (st.value("recent_datasets") or st.value("studio/recent") or st.value("geometry")
                or fx_settings().value("recent/files"))


def simple() -> bool:
    st = app_settings()
    if st.value(KEY) is None:
        st.setValue(KEY, first_run())         # decided once: new users start simple, existing users keep everything
    try:
        return st.value(KEY, False, type=bool)
    except (TypeError, ValueError):
        return False


def set_simple(on: bool):
    app_settings().setValue(KEY, bool(on))
    bus.changed.emit(bool(on))


def _clean(t: str) -> str:
    return " ".join(t.replace("&&", "\0").replace("&", "").replace("\0", "&").replace("-\n", "").split())


class _Level(QObject):
    def __init__(self, window, advanced, reason):
        super().__init__(window)
        self.win = window
        self.advanced = advanced
        self.reason = reason
        self._busy = False
        self.action = QAction("Simple Menus", window)
        self.action.setCheckable(True)
        self.action.setChecked(simple())
        self.action.setToolTip("Show the everyday commands only (Ctrl+K still finds every command)")
        self.action.toggled.connect(set_simple)
        bus.changed.connect(self._apply)
        self._hooked = set()
        self.refresh_all()
        self._apply(simple())

    # ---------------------------------------------------------------- simple menus
    def _is_advanced(self, act: QAction) -> bool:
        if act.property("aio_always") or act.isSeparator():
            return False
        a = self.advanced
        if callable(a):
            return bool(a(act))
        return _clean(act.text()).rstrip("…").strip() in a

    def _all(self, menu: QMenu) -> list:
        """The menu's full list of entries (kept the first time, so they can come back in their places)."""
        lst = getattr(menu, "_aio_all_actions", None)
        if lst is None:
            lst = list(menu.actions())
            menu._aio_all_actions = lst
        return lst

    def _hidden(self, act: QAction, on: bool) -> bool:
        if not on or act.isSeparator():
            return False
        sub = act.menu()
        if sub is not None:                     # a sub-menu goes when everything in it goes
            kids = [a for a in self._all(sub) if not a.isSeparator()]
            return bool(kids) and all(self._hidden(a, on) for a in kids)
        return self._is_advanced(act)

    def _apply(self, on: bool):
        """Take the specialist commands out of the menus (or put them back in their places). The actions stay
        enabled and visible — their shortcuts keep working, and the command palette still lists them."""
        if self.action.isChecked() != on:
            self.action.blockSignals(True)
            self.action.setChecked(on)
            self.action.blockSignals(False)
        menus = [a.menu() for a in self.win.menuBar().actions() if a.menu() is not None]
        seen = set()
        while menus:
            menu = menus.pop()
            if id(menu) in seen:
                continue
            seen.add(id(menu))
            full = self._all(menu)
            want = []
            for act in full:
                hide = self._hidden(act, on)
                act.setProperty(HIDDEN, hide)
                if not hide:
                    want.append(act)
                if act.menu() is not None:
                    menus.append(act.menu())
            if [a for a in menu.actions()] != want:
                for act in list(menu.actions()):
                    menu.removeAction(act)
                menu.addActions(want)
            menu.setToolTipsVisible(True)             # disabled entries say why; others say what they do
        hook = getattr(self.win, "apply_ui_level", None)
        if hook is not None:
            try:
                hook(on)
            except Exception as _exc:
                _ignored(_exc)

    # ---------------------------------------------------------------- tooltips
    def refresh_all(self):
        for act in self.win.findChildren(QAction):
            if id(act) not in self._hooked:
                self._hooked.add(id(act))
                act.changed.connect(self._on_changed)      # a QObject slot: let go of with the window
            self._tip(act)

    def _on_changed(self):
        act = self.sender()
        if isinstance(act, QAction):
            self._tip(act)

    def _tip(self, act: QAction):
        if self._busy:
            return
        from PyQt6 import sip
        if sip.isdeleted(act) or sip.isdeleted(self.win):
            return
        try:
            text = _clean(act.text())
        except RuntimeError:
            return
        if not text or act.isSeparator() or act.menu() is not None:
            return
        self._busy = True
        try:
            base = act.property(BASE_TIP)
            cur = act.toolTip()
            if base is None or (cur != act.property("aio_last_tip")):
                base = cur if cur and _clean(cur) != text else ""       # Qt's default tooltip is the text itself
                act.setProperty(BASE_TIP, base)
            sc = act.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
            tip = base or text
            if sc and sc not in tip:
                tip = f"{tip}  ({sc})"
            if not act.isEnabled():
                why = ""
                try:
                    why = self.reason(act) if self.reason else ""
                except Exception as _exc:
                    _ignored(_exc)
                    why = ""
                if why:
                    tip = f"{tip}\n\nNot available yet: {why}"
            act.setProperty("aio_last_tip", tip)
            act.setToolTip(tip)
        finally:
            self._busy = False


def install(window, advanced=(), reason=None):
    """-> the level helper; its .action is the View ▸ Simple Menus toggle (add it to a menu)."""
    lvl = _Level(window, set(advanced) if not callable(advanced) else advanced, reason)
    window._ui_level = lvl
    return lvl


def refresh(window):
    lvl = getattr(window, "_ui_level", None)
    if lvl is not None:
        lvl.refresh_all()
        lvl._apply(simple())
