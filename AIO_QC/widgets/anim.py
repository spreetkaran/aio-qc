# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Motion helpers — every animation runs on FrameAnimator (one step per display refresh, eased), so it is as smooth as
the screen allows and costs nothing when idle. Settings ▸ General ▸ Animations turns them all off (ENABLED = False): every helper
then applies the end state at once.

    fade(widget, True/False)               opacity in / out (child widgets and overlays)
    window_fade_in(dialog)                 a top-level window appears with a short fade
    crossfade(widget)                      a snapshot of `widget` fades out over its new layout (layout changes)
    toggle_dock(main, dock, visible)       a docked panel slides open / closed (its size is kept for next time)
    slide(widget, start QRect, end QRect)  geometry tween (cards, overlays)
"""
from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QRect, Qt
from PyQt6.QtWidgets import QDockWidget, QGraphicsOpacityEffect, QLabel, QWidget

from .mini_session import FrameAnimator, lerp_rect

ENABLED = True
_RUNNING: dict = {}          # (id(obj), kind) -> FrameAnimator (a new animation of the same kind replaces it)


def _run(obj, kind, ms, step, done=None, easing=QEasingCurve.Type.OutCubic):
    from PyQt6 import sip
    for k in [k for k, a in _RUNNING.items() if sip.isdeleted(a.timer)]:   # owner gone: its id may be reused
        _RUNNING.pop(k, None)
    key = (id(obj), kind)
    old = _RUNNING.pop(key, None)
    if old is not None:
        old.stop()

    def fin():
        _RUNNING.pop(key, None)
        if done is not None:
            done()
    a = FrameAnimator(obj, ms, step, fin, easing)
    _RUNNING[key] = a
    a.start()
    return a


def fade(widget: QWidget, show: bool, ms: int = 180, done=None):
    if not ENABLED:
        widget.setVisible(show)
        if done:
            done()
        return
    eff = widget.graphicsEffect()
    if not isinstance(eff, QGraphicsOpacityEffect):
        eff = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(eff)
    start = eff.opacity() if widget.isVisible() else 0.0
    end = 1.0 if show else 0.0
    if show:
        eff.setOpacity(start)
        widget.show()

    def step(t):
        try:
            eff.setOpacity(start + (end - start) * t)
        except RuntimeError:
            pass

    def fin():
        try:
            if not show:
                widget.hide()
            widget.setGraphicsEffect(None)          # no offscreen rendering once settled
        except RuntimeError:
            pass
        if done:
            done()
    _run(widget, "fade", ms, step, fin)


def hosts_opengl(win: QWidget) -> bool:
    """A window with an OpenGL surface (the Frame Extractor's GPU canvas). Window opacity makes a window layered on
    Windows, and a layered window cannot present OpenGL: the canvas fails and falls back to the CPU renderer."""
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    return bool(win.findChildren(QOpenGLWidget))


def window_fade_in(win: QWidget, ms: int = 160):
    if not ENABLED or not win.isWindow() or hosts_opengl(win):
        return
    if win.windowType() == Qt.WindowType.Dialog:
        from . import motion
        if motion.installed():                      # AppMotion already fades every dialog in: never twice
            return
    win.setWindowOpacity(0.0)

    def step(t):
        try:
            win.setWindowOpacity(t)
        except RuntimeError:
            pass
    _run(win, "wfade", ms, step, lambda: _safe(lambda: win.setWindowOpacity(1.0)))


def crossfade(widget: QWidget, ms: int = 220):
    """Call BEFORE changing the layout: the old look fades into the new one."""
    if not ENABLED or not widget.isVisible() or widget.width() < 2:
        return
    shot = widget.grab()
    lbl = QLabel(widget)
    lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    lbl.setPixmap(shot)
    lbl.setGeometry(0, 0, widget.width(), widget.height())
    eff = QGraphicsOpacityEffect(lbl)
    eff.setOpacity(1.0)
    lbl.setGraphicsEffect(eff)
    lbl.show()
    lbl.raise_()

    def step(t):
        try:
            eff.setOpacity(1.0 - t)
        except RuntimeError:
            pass
    _run(lbl, "xfade", ms, step, lambda: _safe(lbl.deleteLater), QEasingCurve.Type.InOutQuad)


def slide(widget: QWidget, start: QRect, end: QRect, ms: int = 220, done=None):
    if not ENABLED:
        widget.setGeometry(end)
        if done:
            done()
        return
    widget.setGeometry(start)
    _run(widget, "slide", ms, lambda t: _safe(lambda: widget.setGeometry(lerp_rect(start, end, t))), done)


def _safe(fn):
    try:
        fn()
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------------------------------- docks
_SIZES: dict = {}            # dock objectName -> last size along its dock axis


def _axis(main, dock):
    """The direction the dock grows in: across its side of the window — or, when other panels already share that
    side (stacked in the same column / row), along it, so the column itself does not jump."""
    area = main.dockWidgetArea(dock)
    vertical = area in (Qt.DockWidgetArea.TopDockWidgetArea, Qt.DockWidgetArea.BottomDockWidgetArea)
    shared = any(d is not dock and not d.isHidden() and not d.isFloating() and main.dockWidgetArea(d) == area
                 for d in main.findChildren(QDockWidget))
    if shared:
        vertical = not vertical
    return (Qt.Orientation.Vertical if vertical else Qt.Orientation.Horizontal), vertical


def _tabbed(main, dock) -> bool:
    return bool(main.tabifiedDockWidgets(dock))


def toggle_dock(main, dock: QDockWidget, visible: bool, ms: int = 210):
    """Show / hide a dock with a slide; tabbed or floating docks cross-fade instead."""
    if (not dock.isHidden()) == visible:              # isHidden: also right while the window is in the background
        if visible:
            dock.raise_()
        return
    if not ENABLED or not main.isVisible() or dock.isFloating() or _tabbed(main, dock) or main.dockWidgetArea(dock) == \
            Qt.DockWidgetArea.NoDockWidgetArea:
        if ENABLED and main.centralWidget() is not None:
            crossfade(main)
        dock.setVisible(visible)
        if visible:
            dock.raise_()
        return
    orient, vertical = _axis(main, dock)
    name = dock.objectName() or str(id(dock))
    cur = dock.height() if vertical else dock.width()
    if visible:
        target = _SIZES.get(name) or max(160, (dock.sizeHint().height() if vertical else dock.sizeHint().width()))
        dock.setVisible(True)
        main.resizeDocks([dock], [1], orient)
        start = 1
    else:
        _SIZES[name] = max(cur, 80)
        target, start = 1, cur

    def step(t):
        _safe(lambda: main.resizeDocks([dock], [max(1, int(round(start + (target - start) * t)))], orient))

    def fin():
        if not visible:
            _safe(lambda: dock.setVisible(False))
            _safe(lambda: main.resizeDocks([dock], [_SIZES[name]], orient))    # next show restores the size
    _run(dock, "dock", ms, step, fin, QEasingCurve.Type.OutCubic if visible else QEasingCurve.Type.InCubic)
