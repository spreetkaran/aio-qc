# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
3.0.1: one app window for the three modes, with the title bar merged into the window's own header.

The shell puts every mode window (Image QC, Frame Extractor, Data Studio) into one `FrameHost`: switching modes
changes the page inside the same window (a short cross-fade, no new window, no taskbar flicker). The OS title bar
is replaced by the menu bar: the AIO QC mark on the left, the mode switch, the gear and minimise / maximise / close
on the right. Drag the empty part of the menu bar to move the window, double-click it to maximise, drag any edge or
corner to resize — the same on Windows, macOS and Linux (X11 and Wayland; Qt's system move / resize where the
platform offers it, a manual fallback where it doesn't).

Settings ▸ General ▸ "Use the system title bar" brings the OS frame back (after a restart), e.g. for tiling window
managers.
"""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QGuiApplication, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QStackedLayout, QToolButton,
                             QVBoxLayout, QWidget)
from AIO_QC.core.diag import ignored as _ignored

SETTING = "ui/system_title_bar"

# ---------------------------------------------------------------------------------------------------- Windows
# A frameless Qt window on Windows has no caption / sizing styles, so Windows treats it as a plain popup: no
# minimise / maximise / restore animations, and clicking its taskbar button doesn't minimise it. The fix (the usual
# one for custom title bars): give the window back WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX |
# WS_SYSMENU — Windows then animates and handles the taskbar like any app — and answer WM_NCCALCSIZE so none of
# that frame is drawn (the whole window stays ours). Maximised, the client area is kept to the monitor's work area.
_WIN = sys.platform == "win32"
WM_NCCALCSIZE = 0x0083
_STYLES = 0x00C00000 | 0x00040000 | 0x00020000 | 0x00010000 | 0x00080000   # CAPTION THICKFRAME MIN MAX SYSMENU


# WM_NCCALCSIZE is answered from a plain Win32 window subclass (comctl32 SetWindowSubclass, a ctypes callback), not
# from a PyQt nativeEvent override: with that override the app crashed natively in PyQt's handling of the call on
# the window's first messages (restoreGeometry at start-up). The subclass never touches Qt.
# A start-up guard (shell/native_frame) turns the integration off by itself if a run ever ends while it is being set
# up; AIOQC_NATIVE_FRAME=0 turns it off, =1 tries it again.
NATIVE_STATE = "shell/native_frame"          # "" untested · "trying" · "ok" · "off"
WM_NCDESTROY = 0x0082
_SUB = {}                                    # ctypes pieces, made once: the callback must outlive every window


def _win_api():
    if _SUB:
        return _SUB
    import ctypes
    from ctypes import wintypes as W
    LRESULT = ctypes.c_ssize_t
    PROC = ctypes.WINFUNCTYPE(LRESULT, W.HWND, W.UINT, W.WPARAM, W.LPARAM, ctypes.c_size_t, ctypes.c_size_t)
    u, c = ctypes.WinDLL("user32", use_last_error=True), ctypes.WinDLL("comctl32", use_last_error=True)
    sigs = ((u, "GetWindowLongPtrW", [W.HWND, ctypes.c_int], ctypes.c_ssize_t),
            (u, "SetWindowLongPtrW", [W.HWND, ctypes.c_int, ctypes.c_ssize_t], ctypes.c_ssize_t),
            (u, "SetWindowPos", [W.HWND, W.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, W.UINT],
             W.BOOL),
            (u, "IsZoomed", [W.HWND], W.BOOL),
            (u, "MonitorFromWindow", [W.HWND, W.DWORD], W.HANDLE),
            (c, "SetWindowSubclass", [W.HWND, PROC, ctypes.c_size_t, ctypes.c_size_t], W.BOOL),
            (c, "RemoveWindowSubclass", [W.HWND, PROC, ctypes.c_size_t], W.BOOL),
            (c, "DefSubclassProc", [W.HWND, W.UINT, W.WPARAM, W.LPARAM], LRESULT))
    for lib, name, args, res in sigs:
        f = getattr(lib, name)
        f.argtypes, f.restype = args, res

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", W.DWORD), ("rcMonitor", W.RECT), ("rcWork", W.RECT), ("dwFlags", W.DWORD)]
    u.GetMonitorInfoW.argtypes = [W.HANDLE, ctypes.POINTER(MONITORINFO)]
    u.GetMonitorInfoW.restype = W.BOOL

    def proc(hwnd, msg, wp, lp, uid, _ref):
        try:
            if msg == WM_NCCALCSIZE and wp:
                if u.IsZoomed(hwnd):                                 # maximised: exactly the monitor's work area
                    mon = u.MonitorFromWindow(hwnd, 2)               # MONITOR_DEFAULTTONEAREST
                    mi = MONITORINFO()
                    mi.cbSize = ctypes.sizeof(MONITORINFO)
                    if mon and u.GetMonitorInfoW(mon, ctypes.byref(mi)):
                        r = W.RECT.from_address(lp)                  # NCCALCSIZE_PARAMS.rgrc[0]
                        r.left, r.top, r.right, r.bottom = (mi.rcWork.left, mi.rcWork.top, mi.rcWork.right,
                                                            mi.rcWork.bottom)
                return 0                                             # no system frame: the whole window is ours
            if msg == WM_NCDESTROY:
                c.RemoveWindowSubclass(hwnd, _SUB["cb"], uid)
        except Exception as _exc:                                            # noqa: BLE001 — never raise into Windows
            _ignored(_exc)
        return c.DefSubclassProc(hwnd, msg, wp, lp)

    class MARGINS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_int), ("r", ctypes.c_int), ("t", ctypes.c_int), ("b", ctypes.c_int)]
    _SUB.update(ctypes=ctypes, W=W, u=u, c=c, cb=PROC(proc), MARGINS=MARGINS)
    return _SUB


NATIVE_USER = "shell/native_frame_user"          # "off" only when YOU turned it off in Settings
NATIVE_STRIKES = "shell/native_frame_strikes"   # starts in a row that ended before the integration settled
NATIVE_MIGRATED = "shell/native_frame_v2"       # the old guard's automatic "off" was undone once
STRIKES_TO_OFF = 3
_CRASH_FRAME = None


def _crash_in_integration(dump: str) -> bool:
    """The native-crash dump points INTO this integration: a frame line of window_frame.py in one of its functions
    (not merely any crash during a window's show, which also passes through this file)."""
    global _CRASH_FRAME
    if _CRASH_FRAME is None:
        import re
        _CRASH_FRAME = re.compile(r'window_frame\.py", line \d+ in (proc|_win_native_frame|_apply_native_frame|'
                                  r'_win_api)\s*$', re.M)
    return bool(dump) and _CRASH_FRAME.search(dump) is not None


def native_frame_allowed() -> bool:
    """Windows taskbar / animation integration for the frameless window — on unless you switched it off.

    Safety net: when a run ends (crash or kill) before the integration has settled, that counts as a strike; it is
    switched off by itself only when the crash dump shows the crash INSIDE the integration, or after
    STRIKES_TO_OFF such starts in a row — and then the app says so on start, with a button to turn it back on.
    (Before 3.0.1 one unclean exit of any kind switched it off for good — that automatic "off" is undone once.)"""
    if not _WIN:
        return False
    env = os.environ.get("AIOQC_NATIVE_FRAME", "").strip()
    from AIO_QC.utils import app_settings
    st = app_settings()
    if env == "0":
        return False
    if env == "1":                                               # an explicit retry: on, whatever happened before
        st.setValue(NATIVE_STATE, "")
        st.setValue(NATIVE_USER, "")
        st.setValue(NATIVE_STRIKES, 0)
        st.setValue(NATIVE_MIGRATED, True)
        return True
    if not st.value(NATIVE_MIGRATED, False, type=bool):          # bring back what the old, over-eager guard took
        st.setValue(NATIVE_MIGRATED, True)
        if str(st.value(NATIVE_STATE, "") or "") in ("off", "trying") and \
                str(st.value(NATIVE_USER, "") or "") != "off":
            st.setValue(NATIVE_STATE, "")
            st.setValue(NATIVE_STRIKES, 0)
    if str(st.value(NATIVE_USER, "") or "") == "off":
        return False
    state = str(st.value(NATIVE_STATE, "") or "")
    if state == "auto-off":
        return False
    try:
        from AIO_QC.widgets.lifecycle import previous_crash
        dump = previous_crash()
    except Exception as _exc:
        _ignored(_exc)
        dump = ""
    inside = _crash_in_integration(dump)                         # the last run crashed IN it (setup or later)
    if state == "trying" or inside:                              # ... or ended before it settled
        strikes = int(st.value(NATIVE_STRIKES, 0) or 0) + (1 if state == "trying" else 0)
        st.setValue(NATIVE_STRIKES, strikes)
        if inside or strikes >= STRIKES_TO_OFF:
            st.setValue(NATIVE_STATE, "auto-off")
            from loguru import logger
            logger.warning("Windows taskbar integration switched off: " +
                           ("the last run crashed inside it" if inside else
                            f"{strikes} starts in a row ended before it settled") +
                           ". Settings ▸ General ▸ Windows taskbar and window animations turns it back on.")
            _NOTICE["auto_off"] = True
            return False
        st.setValue(NATIVE_STATE, "")
    return True


_NOTICE = {"auto_off": False}


def taskbar_notice(host):
    """Once, on start: say that the taskbar integration was switched off by itself, with a button to turn it on."""
    if not _NOTICE.get("auto_off"):
        return
    pages = [getattr(host, "current", None)] + list(getattr(host, "pages", []) or [])
    tm = next((getattr(p, "toast", None) for p in pages if getattr(p, "toast", None) is not None), None)
    if tm is None:
        return
    _NOTICE["auto_off"] = False
    try:
        def turn_on():
            from AIO_QC.utils import app_settings
            st = app_settings()
            st.setValue(NATIVE_STATE, "")
            st.setValue(NATIVE_STRIKES, 0)
            tm.show_message("<b>Turned back on</b><br>Taskbar minimise / restore works again from the next start.",
                            "success")
        tm.show_message("<b>Taskbar minimise / restore is off</b><br>AIO QC closed unexpectedly while setting it up "
                        "more than once, so it was switched off to be safe.", "warn",
                        actions=[("Turn it back on", turn_on)], sticky=True, key="taskbar-off")
    except Exception as _exc:
        _ignored(_exc)


def _win_native_frame(widget) -> bool:
    """Windows only: give the frameless top-level window its system styles back and answer WM_NCCALCSIZE (above)."""
    if not native_frame_allowed():
        return False
    from AIO_QC.utils import app_settings
    st = app_settings()
    first = True                                               # the guard, every start: cleared once the window
    if first:                                                  # has lived a while (or the app quit normally)
        st.setValue(NATIVE_STATE, "trying")
        st.sync()
    try:
        a = _win_api()
        W, u, c = a["W"], a["u"], a["c"]
        hwnd = W.HWND(int(widget.winId()))
        if not c.SetWindowSubclass(hwnd, a["cb"], 1, 0):
            raise OSError("SetWindowSubclass failed")
        style = u.GetWindowLongPtrW(hwnd, -16)                     # GWL_STYLE
        if style & _STYLES != _STYLES:
            u.SetWindowLongPtrW(hwnd, -16, style | _STYLES)
        try:                                                       # the system drop shadow, as for any window
            a["ctypes"].windll.dwmapi.DwmExtendFrameIntoClientArea(hwnd, a["ctypes"].byref(a["MARGINS"](0, 0, 0, 1)))
        except Exception as _exc:
            _ignored(_exc)
        u.SetWindowPos(hwnd, None, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020)   # FRAMECHANGED
    except Exception as exc:
        from loguru import logger
        logger.warning(f"Windows frame integration not available: {exc}")
        if first:
            st.setValue(NATIVE_STATE, "")
        return False
    if first:
        def settled():
            if str(st.value(NATIVE_STATE, "") or "") == "trying":
                st.setValue(NATIVE_STATE, "ok")
                st.setValue(NATIVE_STRIKES, 0)
        from AIO_QC.utils.qt_utils import later
        later(8000, widget, settled)
        app = QGuiApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(settled)
    return True


EDGE = 5                     # resize band (logical px) around a normal, frameless window


def use_system_frame() -> bool:
    from AIO_QC.utils import app_settings
    try:
        return app_settings().value(SETTING, False, type=bool)
    except Exception as _exc:
        _ignored(_exc)
        return False


def _tok(k):
    from AIO_QC.widgets.design import tokens
    return tokens()[k]


# ---------------------------------------------------------------------------------------------------- brand
def install_brand(win):
    """The AIO QC mark + name at the left of a mode window's menus — the same in every mode."""
    mb = win.menuBar()
    old = mb.cornerWidget(Qt.Corner.TopLeftCorner)
    if old is not None and old.objectName() == "brandCorner":
        return old
    from AIO_QC import utils
    from PyQt6.QtGui import QFont
    brand = QWidget()
    bl = QHBoxLayout(brand)
    bl.setContentsMargins(12, 0, 14, 0)
    bl.setSpacing(8)
    mark = utils.IconLabel("brand", 20)
    name = QLabel("AIO QC")
    f = QFont(name.font())
    f.setBold(True)
    name.setFont(f)
    bl.addWidget(mark)
    bl.addWidget(name)
    brand.setObjectName("brandCorner")
    brand.setToolTip("AIO QC")
    _show_menu_overflow(mb)
    mb.setCornerWidget(brand, Qt.Corner.TopLeftCorner)
    brand.show()
    return brand


def _show_menu_overflow(mb):
    """On a narrow window the menus that don't fit go behind the menu bar's overflow button — which the app's style
    otherwise draws blank, so Tools / Help seemed to vanish. Give it a visible chevron."""
    from PyQt6.QtWidgets import QToolButton
    ext = mb.findChild(QToolButton, "qt_menubar_ext_button")
    if ext is None:
        return
    from AIO_QC import utils
    ext.setIcon(utils.newIcon("chevron-right"))
    ext.setIconSize(QSize(12, 12))
    ext.setStyleSheet("QToolButton{padding:0; margin:0; border:none; min-width:12px;}"
                      "QToolButton::menu-indicator{image:none; width:0;}")
    ext.setToolTip("More menus")


# ---------------------------------------------------------------------------------------------------- controls
class _CtlButton(QToolButton):
    def __init__(self, icon, tip, close=False, parent=None):
        super().__init__(parent)
        self.close_btn = close
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(40, 30)
        self.setIconSize(QSize(16, 16))
        self.setToolTip(tip)
        self.setAccessibleName(tip)
        self._icon = icon
        self._hover = False
        self.set_icon(icon)

    def set_icon(self, icon):
        from AIO_QC import utils
        self._icon = icon
        self.setIcon(utils.tintedIcon(icon, "#FFFFFF") if (self.close_btn and self._hover) else utils.newIcon(icon))

    def enterEvent(self, e):
        self._hover = True
        self.set_icon(self._icon)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        self.set_icon(self._icon)
        super().leaveEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.isDown() or self._hover:
            c = QColor("#C42B1C") if self.close_btn else QColor(_tok("pressed" if self.isDown() else "hover"))
            if self.close_btn and self.isDown():
                c = c.darker(115)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawRoundedRect(self.rect().adjusted(1, 2, -1, -2), 6, 6)
        s = self.iconSize()
        r = QRect(0, 0, s.width(), s.height())
        r.moveCenter(self.rect().center())
        self.icon().paint(p, r)


class _FadeOverlay(QWidget):
    """The page before a mode switch, painted over the new one at a falling opacity (one blit per display frame;
    no graphics effect, so the cross-fade stays fluid on any machine)."""

    def __init__(self, parent, pix):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.pix = pix
        self.k = 1.0
        self.anim = None

    def run(self, ms, done):
        from AIO_QC.widgets.mini_session import FrameAnimator

        def step(t):
            self.k = 1.0 - t
            self.update()
        self.anim = FrameAnimator(self, max(1, ms), step, done)
        QTimer.singleShot(0, self.anim.start)      # after the new page painted once under it

    def stop(self):
        if self.anim is not None:
            self.anim.stop()

    def paintEvent(self, _e):
        if self.k <= 0:
            return
        p = QPainter(self)
        p.setOpacity(self.k)
        p.drawPixmap(0, 0, self.pix)


class BigLoader(QWidget):
    """A large loading ring (as in the reference) with a line of text, centred over the window."""

    SIZE = 56

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.text = ""
        self.k = 1.0
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        try:
            from AIO_QC.widgets.mini_session import refresh_hz
            self.timer.setInterval(max(2, int(1000 / refresh_hz(parent))))     # every display frame (60–240 Hz)
        except Exception as _exc:
            _ignored(_exc)
            self.timer.setInterval(8)
        self.timer.timeout.connect(self.update)
        self._fade = None
        self.hide()

    def set_text(self, t):
        self.text = t
        self.update()

    def start(self):
        import time
        self.k = 1.0
        self._t0 = time.monotonic()
        self.timer.start()

    def fade_out(self):
        from AIO_QC.widgets.mini_session import FrameAnimator

        def step(t):
            self.k = 1.0 - t
            self.update()

        def done():
            self.timer.stop()
            self.hide()
        self._fade = FrameAnimator(self, max(1, _ms(220)), step, done)
        self._fade.start()

    def paintEvent(self, _e):
        import time
        from PyQt6.QtCore import QRectF
        from AIO_QC.widgets.tasks import TURN_S, paint_ring
        p = QPainter(self)
        p.setOpacity(self.k)
        p.fillRect(self.rect(), QColor(_tok("window")))
        c = self.rect().center()
        s = self.SIZE
        angle = (time.monotonic() % TURN_S) / TURN_S * 360.0
        paint_ring(p, QRectF(c.x() - s / 2, c.y() - s / 2 - 14, s, s), angle)
        if self.text:
            p.setPen(QColor(_tok("muted")))
            f = p.font()
            f.setPixelSize(14)
            p.setFont(f)
            p.drawText(QRect(0, c.y() + s // 2 - 2, self.width(), 24), int(Qt.AlignmentFlag.AlignHCenter), self.text)


class WindowControls(QWidget):
    """Minimise · maximise / restore · close, at the right end of a header (the main window's menu bar, a tool
    window's header or title bar)."""

    def __init__(self, host, parent=None, minimise=True, maximise=True):
        super().__init__(parent)
        self.setObjectName("windowControls")
        self.host = host
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 2, 0)
        lay.setSpacing(0)
        self.min_btn = _CtlButton("win-min", "Minimise")
        self.max_btn = _CtlButton("win-max", "Maximise")
        self.close_btn = _CtlButton("win-close", "Close", close=True)
        self.min_btn.clicked.connect(lambda: _minimise(host))
        self.max_btn.clicked.connect(lambda: toggle_maximized(host))
        self.close_btn.clicked.connect(lambda: host.close())
        for b in (self.min_btn, self.max_btn, self.close_btn):
            lay.addWidget(b)                         # parented first: setVisible(True) on a parentless widget
        self.min_btn.setVisible(minimise)            # opens it as a window of its own for a moment
        self.max_btn.setVisible(maximise)
        self.sync()

    def sync(self):
        big = self.host.isMaximized() or self.host.isFullScreen()
        self.max_btn.set_icon("win-restore" if big else "win-max")
        self.max_btn.setToolTip("Restore" if big else "Maximise")


def _minimise(w):
    """The minimise button: a feature window goes to the shelf inside the main window (window_shelf), the main
    window and windows of their own to the taskbar / Dock."""
    from AIO_QC.widgets import window_shelf
    window_shelf.minimise(w)


def toggle_maximized(w):
    if w.isMaximized():
        w.showNormal()
    else:
        w.showMaximized()


# ---------------------------------------------------------------------------------------------------- host
class FrameHost(QWidget):
    """The one top-level window; the mode windows are its pages."""

    def __init__(self, shell=None):
        super().__init__(None)
        self.shell = shell
        self.custom = not use_system_frame()
        self.setObjectName("frameHost")
        self.setWindowTitle("AIO QC App")
        if self.custom:
            self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.stack = QStackedLayout()
        outer = QVBoxLayout(self)
        outer.setSpacing(0)
        outer.addLayout(self.stack)
        self._outer = outer
        self.pages = []
        self.controls = []
        self._overlay = None
        self._drag = None             # manual move fallback: (edges, press global pos, start geometry)
        self.grips = Grips(self) if self.custom else None
        self._update_margins()
        self._gl_ready = self._prepare_gl_surface()

    def _prepare_gl_surface(self) -> bool:
        """The Frame Extractor's video canvas (and the GPU image canvas) are OpenGL widgets. The first one added to a
        window makes Qt re-create that window with an OpenGL surface — the whole app window vanished and came back
        for a moment when a mode was prepared or opened. A tiny hidden OpenGL widget from the start gives the window
        that surface before it is ever shown, so nothing is re-created later. Only where OpenGL really works."""
        import os
        if os.environ.get("QT_QPA_PLATFORM", "").startswith(("offscreen", "minimal")):
            return False
        try:
            from PyQt6.QtGui import QOffscreenSurface, QOpenGLContext
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget
            ctx = QOpenGLContext()
            if not ctx.create():
                return False
            surf = QOffscreenSurface()
            surf.create()
            ok = ctx.makeCurrent(surf)
            if ok:
                ctx.doneCurrent()
            if not ok:
                return False
            gl = QOpenGLWidget(self)
            gl.setObjectName("glSurfaceKeeper")
            gl.setFixedSize(1, 1)
            gl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            gl.hide()
            self._gl_keeper = gl
            return True
        except Exception as _exc:
            _ignored(_exc)
            return False

    # ---- pages
    def add(self, win):
        """Embed a mode window (before it is ever shown): its menu bar becomes the title bar."""
        if win in self.pages:
            return win
        win.setWindowState(Qt.WindowState.WindowNoState)
        win.setParent(self, Qt.WindowType.Widget)
        mb = win.menuBar()
        mb.setNativeMenuBar(False)                # the menus live in the window on macOS too (same on every OS)
        install_brand(win)
        if self.custom:
            corner = mb.cornerWidget(Qt.Corner.TopRightCorner)
            ctl = WindowControls(self)
            if corner is not None and corner.layout() is not None:
                corner.layout().addWidget(ctl)
            else:
                box = QWidget(mb)
                h = QHBoxLayout(box)
                h.setContentsMargins(0, 0, 0, 0)
                h.setSpacing(4)
                if corner is not None:
                    corner.setParent(box)
                    h.addWidget(corner)
                    corner.show()
                h.addWidget(ctl)
                mb.setCornerWidget(box, Qt.Corner.TopRightCorner)
                box.show()
            self.controls.append(ctl)
            mb.installEventFilter(self)
            win._window_controls = ctl
        self.stack.addWidget(win)
        self.pages.append(win)
        ld = getattr(self, "_loader", None)
        if ld is not None and ld.isVisible():
            ld.raise_()                               # a page built under the loading ring stays under it
        win.windowTitleChanged.connect(lambda t, w=win: self._mirror_title(w))
        win.windowIconChanged.connect(lambda _i, w=win: self._mirror_title(w))
        return win

    def current(self):
        return self.stack.currentWidget()

    def remove(self, win):
        """Take a page out (a mode that was unloaded)."""
        if win in self.pages:
            self.pages.remove(win)
            ctl = getattr(win, "_window_controls", None)
            if ctl in self.controls:
                self.controls.remove(ctl)
            self.stack.removeWidget(win)

    def _mirror_title(self, w):
        if w is self.current():
            self.setWindowTitle(w.windowTitle() or "AIO QC App")
            icon = getattr(w, "_mode_icon", None)          # the mode's own icon (set by the shell)
            if icon is None or icon.isNull():
                icon = w.windowIcon()
            if not icon.isNull() and icon.cacheKey() != self.windowIcon().cacheKey():
                self.setWindowIcon(icon)

    def set_current(self, win, animate=True):
        """Show a page; a short cross-fade from the page before (the new page is already in place underneath)."""
        old = self.current()
        if win is old and win.isVisible():
            return
        animate = animate and old is not None and old is not win and self.isVisible() and _animations_on()
        if animate:
            self._fade_from(old)
        self.stack.setCurrentWidget(win)
        win.show()
        self._mirror_title(win)
        if self._overlay is not None:
            self._overlay.raise_()
        ld = getattr(self, "_loader", None)
        if ld is not None and ld.isVisible():
            ld.raise_()

    def _fade_from(self, old):
        self._end_fade()
        try:
            pix = old.grab()
        except Exception as _exc:
            _ignored(_exc)
            return
        ov = _FadeOverlay(self, pix)
        ov.setGeometry(old.geometry())
        ov.show()
        ov.raise_()
        self._overlay = ov
        ov.run(_ms(200), self._end_fade)

    def _end_fade(self):
        ov, self._overlay = self._overlay, None
        if ov is not None:
            ov.stop()
            ov.hide()
            ov.deleteLater()

    # ---- loading (a mode opened for the first time)
    def show_loading(self, text: str):
        """The big loading ring over the window while a mode is being prepared."""
        if getattr(self, "_loader", None) is None:
            self._loader = BigLoader(self)
        ld = self._loader
        ld.set_text(text)
        r = self.contentsRect().marginsRemoved(self._outer.contentsMargins())
        cur = self.current()
        mb = cur.menuBar() if cur is not None and hasattr(cur, "menuBar") else None
        if mb is not None and mb.isVisible():
            r.setTop(r.top() + mb.height())            # the header (and its mode switch) stays in view
        ld.setGeometry(r)
        ld.show()
        ld.raise_()
        ld.start()

    def hide_loading(self):
        ld = getattr(self, "_loader", None)
        if ld is not None and ld.isVisible():
            ld.fade_out()

    # ---- window state
    def toggle_maximized(self):
        toggle_maximized(self)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.grips is not None:
            self.grips.place()
        from PyQt6.QtWidgets import QApplication
        if self.isVisible() and not (self.isMaximized() or self.isFullScreen()) and \
                QApplication.mouseButtons() == Qt.MouseButton.NoButton and e.size().width() >= e.oldSize().width() >= 0:
            from AIO_QC.utils.qt_utils import later
            later(0, self, self._keep_on_screen)      # grown by its content (a mode's page), not by the user's drag

    def _keep_on_screen(self):
        """A window that grew (a page that needs more room) slides back inside the screen instead of running off its
        right or bottom edge."""
        scr = self.screen()
        if scr is None or self.isMaximized() or self.isFullScreen():
            return
        avail = scr.availableGeometry()
        g = self.frameGeometry()
        if g.width() > avail.width() or g.height() > avail.height():
            return
        x = min(max(g.x(), avail.left()), avail.right() + 1 - g.width())
        y = min(max(g.y(), avail.top()), avail.bottom() + 1 - g.height())
        if (x, y) != (g.x(), g.y()):
            self.move(self.pos() + QPoint(x - g.x(), y - g.y()))

    def showEvent(self, e):
        super().showEvent(e)
        if self.custom and _WIN and not getattr(self, "_native_pending", False):
            self._native_pending = True
            from AIO_QC.utils.qt_utils import later
            later(0, self, self._apply_native_frame)          # once the window is up, never inside its first show

    def _apply_native_frame(self):
        self._native_pending = False
        hwnd = int(self.winId())
        if getattr(self, "_native_hwnd", None) != hwnd and _win_native_frame(self):
            self._native_hwnd = hwnd                          # taskbar / animations like any Windows app
        elif _NOTICE.get("auto_off"):
            from AIO_QC.utils.qt_utils import later
            later(2500, self, lambda: taskbar_notice(self))

    def custom_minimise(self):
        """Minimise to the taskbar / Dock. macOS can't minimise a window without a title bar (a borderless window
        is not "miniaturizable"), so there the window takes the system title bar for the trip to the Dock and drops
        it again when it comes back."""
        if sys.platform == "darwin" and self.custom and not self.isMinimized():
            self._mac_geo = self.geometry() if not self.isMaximized() else None
            self._mac_was_max = self.isMaximized()
            self._mac_framed = True
            self.setWindowFlag(Qt.WindowType.FramelessWindowHint, False)      # re-creates the native window, hidden
            if self._mac_geo is not None:
                self.setGeometry(self._mac_geo)
            self.show()
            QTimer.singleShot(0, self.showMinimized)
            return
        self.showMinimized()

    def _mac_unframe(self):
        """Back from the Dock (macOS): borderless again, where it was."""
        if not getattr(self, "_mac_framed", False) or self.isMinimized():
            return
        self._mac_framed = False
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        if self._mac_was_max:
            self.showMaximized()
        else:
            if self._mac_geo is not None:
                self.setGeometry(self._mac_geo)
            self.show()
        self.raise_()
        self.activateWindow()

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() == QEvent.Type.WindowStateChange and getattr(self, "_mac_framed", False) and \
                not self.isMinimized():
            QTimer.singleShot(0, self._mac_unframe)
        if e.type() == QEvent.Type.WindowStateChange:
            self._update_margins()
            if self.grips is not None:
                self.grips.place()
            for c in self.controls:
                c.sync()
        elif e.type() in (QEvent.Type.ActivationChange, QEvent.Type.PaletteChange):
            self.update()

    def _update_margins(self):
        m = EDGE if self.custom and not (self.isMaximized() or self.isFullScreen()) else 0
        self._outer.setContentsMargins(m, m, m, m)

    def paintEvent(self, _e):
        if not self.custom:
            return
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(_tok("surface")))
        if not (self.isMaximized() or self.isFullScreen()):
            p.setPen(QPen(QColor(_tok("control" if self.isActiveWindow() else "border")), 1))
            p.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def closeEvent(self, e):
        sh = self.shell
        if sh is not None and not sh._closing:
            e.ignore()
            QTimer.singleShot(0, sh.quit_all)
            return
        self.save_geometry()
        super().closeEvent(e)

    # ---- geometry
    def save_geometry(self):
        try:
            from AIO_QC.utils import app_settings
            app_settings().setValue("shell/geometry", self.saveGeometry())
        except Exception as _exc:
            _ignored(_exc)

    def restore(self):
        """Last size and place — always inside a screen (a window bigger than the screen opens maximised)."""
        from AIO_QC.utils import app_settings
        geo = app_settings().value("shell/geometry")
        ok = False
        if geo:
            try:
                ok = self.restoreGeometry(geo)
            except Exception as _exc:
                _ignored(_exc)
                ok = False
        scr = (self.screen() or QGuiApplication.primaryScreen())
        avail = scr.availableGeometry() if scr is not None else QRect(0, 0, 1600, 900)
        if not ok:
            w, h = int(avail.width() * 0.86), int(avail.height() * 0.88)
            self.resize(max(w, 800), max(h, 560))
            self.move(avail.center() - QPoint(self.width() // 2, self.height() // 2))
        g = self.frameGeometry() if not self.custom else self.geometry()
        if g.width() > avail.width() or g.height() > avail.height():
            return "max"
        if not avail.intersects(g):
            self.move(avail.topLeft() + QPoint(40, 40))
        return "max" if self.isMaximized() else "normal"

    def show_restored(self):
        if self.restore() == "max":
            self.showMaximized()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    # ---- moving (menu bar) and resizing (edges)
    def _edges(self, pos: QPoint):
        if not self.custom or self.isMaximized() or self.isFullScreen():
            return Qt.Edge(0)
        r = self.rect()
        e = Qt.Edge(0)
        if pos.x() <= EDGE:
            e |= Qt.Edge.LeftEdge
        elif pos.x() >= r.width() - EDGE - 1:
            e |= Qt.Edge.RightEdge
        if pos.y() <= EDGE:
            e |= Qt.Edge.TopEdge
        elif pos.y() >= r.height() - EDGE - 1:
            e |= Qt.Edge.BottomEdge
        return e

    @staticmethod
    def _cursor(e):
        E, C = Qt.Edge, Qt.CursorShape
        if e in (E.LeftEdge | E.TopEdge, E.RightEdge | E.BottomEdge):
            return C.SizeFDiagCursor
        if e in (E.RightEdge | E.TopEdge, E.LeftEdge | E.BottomEdge):
            return C.SizeBDiagCursor
        if e & (E.LeftEdge | E.RightEdge):
            return C.SizeHorCursor
        if e & (E.TopEdge | E.BottomEdge):
            return C.SizeVerCursor
        return None

    def mouseMoveEvent(self, ev):
        if self._drag is not None:
            self._manual_drag(ev.globalPosition().toPoint())
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._drag = None
        super().mouseReleaseEvent(ev)

    def _manual_drag(self, gp: QPoint):
        edges, start, geo = self._drag
        d = gp - start
        if edges == Qt.Edge(0):                      # move
            self.move(geo.topLeft() + d)
            return
        g = QRect(geo)
        mn = self.minimumSizeHint().expandedTo(self.minimumSize())
        if edges & Qt.Edge.LeftEdge:
            g.setLeft(min(geo.left() + d.x(), geo.right() - mn.width()))
        if edges & Qt.Edge.RightEdge:
            g.setRight(max(geo.right() + d.x(), geo.left() + mn.width()))
        if edges & Qt.Edge.TopEdge:
            g.setTop(min(geo.top() + d.y(), geo.bottom() - mn.height()))
        if edges & Qt.Edge.BottomEdge:
            g.setBottom(max(geo.bottom() + d.y(), geo.top() + mn.height()))
        self.setGeometry(g)

    def eventFilter(self, obj, ev):
        """The menu bar's empty space is the title bar: drag = move, double-click = maximise / restore."""
        t = ev.type()
        if t in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick) and \
                ev.button() == Qt.MouseButton.LeftButton:
            pos = ev.position().toPoint()
            if obj.actionAt(pos) is not None or obj.childAt(pos) is not None:
                return False
            if t == QEvent.Type.MouseButtonDblClick:
                self.toggle_maximized()
                return True
            h = self.windowHandle()
            if h is None or not h.startSystemMove():
                if self.isMaximized():
                    return True
                self._drag = (Qt.Edge(0), ev.globalPosition().toPoint(), self.geometry())
                obj.grabMouse()
                self._grabber = obj
            return True
        if t == QEvent.Type.MouseMove and self._drag is not None and self._drag[0] == Qt.Edge(0):
            self._manual_drag(ev.globalPosition().toPoint())
            return True
        if t == QEvent.Type.MouseButtonRelease and self._drag is not None:
            self._drag = None
            g = getattr(self, "_grabber", None)
            if g is not None:
                g.releaseMouse()
                self._grabber = None
            return True
        return False


def _animations_on() -> bool:
    try:
        from AIO_QC.widgets import motion
        return motion.enabled()
    except Exception as _exc:
        _ignored(_exc)
        return True


def _ms(ms: int) -> int:
    try:
        from AIO_QC.widgets import motion
        return motion.scaled(ms)
    except Exception as _exc:
        _ignored(_exc)
        return ms


def system_title_setting():
    from AIO_QC.widgets.settings_dialog import Setting
    from AIO_QC.utils import app_settings
    return Setting("app.system_title_bar", "Use the system title bar", "bool",
                   get=lambda: app_settings().value(SETTING, False, type=bool),
                   set=lambda v: app_settings().setValue(SETTING, bool(v)), default=False, note="after a restart",
                   description="The app window and every feature window use their own header as the title bar "
                               "(drag it to move, double-click to maximise). Turn this on for the operating system's "
                               "title bars instead — e.g. with a tiling window manager.",
                   aliases=("title bar", "window frame", "frameless", "decorations"))


def native_frame_setting():
    """Windows: the taskbar button minimises / restores the app and the system window animations play."""
    from AIO_QC.widgets.settings_dialog import Setting
    from AIO_QC.utils import app_settings

    def get():
        st = app_settings()
        return str(st.value(NATIVE_USER, "") or "") != "off" and \
            str(st.value(NATIVE_STATE, "") or "") not in ("off", "auto-off")

    def put(v):
        st = app_settings()
        st.setValue(NATIVE_USER, "" if v else "off")             # your choice — never overridden by the safety net
        if v:
            st.setValue(NATIVE_STATE, "")
            st.setValue(NATIVE_STRIKES, 0)
    return Setting("app.native_frame", "Windows taskbar and window animations", "bool", get=get, set=put,
                   default=True, note="after a restart",
                   description="Clicking the taskbar button minimises and restores the window, and minimise / "
                               "maximise / restore play the system animations, as for any Windows app. Switched off "
                               "by itself only if the app crashes inside it (or fails to start with it three times "
                               "in a row) — the app then says so and offers to turn it back on.",
                   aliases=("taskbar", "minimise", "minimize", "animation", "snap"))



# ---------------------------------------------------------------------------------------------------- tool windows
# 3.0: every feature window of the three modes (scan windows, galleries, Settings, Testing, Preparation, the Frame
# Extractor's and Data Studio's dialogs …) gets the same merged title bar: its own header (or a slim title bar with its
# icon and title) carries minimise / maximise / close; drag it to move, double-click to maximise, drag an edge to
# resize. Done once, right before a window is first shown (show / exec / open), for every window of the app.

_SKIP = ("QFileDialog", "QMessageBox", "QInputDialog", "QColorDialog", "QFontDialog", "QProgressDialog",
         "QErrorMessage", "QWizard")


def _eligible(win) -> bool:
    try:
        if not win.isWindow() or win.property("_aio_chrome") or win.property("noChrome"):
            return False
        if win.windowType() not in (Qt.WindowType.Window, Qt.WindowType.Dialog):
            return False
        if win.windowFlags() & Qt.WindowType.FramelessWindowHint or isinstance(win, FrameHost):
            return False
        if win.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground):
            return False
        names = {c.__name__ for c in type(win).__mro__}
        if names & set(_SKIP):
            return False
        return use_system_frame() is False
    except RuntimeError:
        return False


class TitleBar(QWidget):
    """A slim title bar for a window without a header of its own: icon · title · controls."""

    def __init__(self, win):
        super().__init__(win)
        self.setObjectName("windowTitleBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(38)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 4, 0)
        lay.setSpacing(8)
        from AIO_QC import utils
        self.icon = QLabel()
        self.icon.setFixedSize(16, 16)
        self.title = QLabel()
        f = self.title.font()
        f.setWeight(f.Weight.DemiBold)
        self.title.setFont(f)
        lay.addWidget(self.icon)
        lay.addWidget(self.title, 1)
        self.layout_ = lay
        self.win = win
        win.windowTitleChanged.connect(lambda _t: self.sync())
        win.windowIconChanged.connect(lambda _i: self.sync())
        self.sync()
        _ = utils

    def sync(self):
        try:
            self.title.setText(self.win.windowTitle())
            ic = self.win.windowIcon()
            if ic.isNull():
                from PyQt6.QtWidgets import QApplication
                ic = QApplication.windowIcon()
            self.icon.setPixmap(ic.pixmap(16, 16) if not ic.isNull() else QPixmap())
            self.icon.setVisible(not ic.isNull())
        except RuntimeError:
            pass

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(_tok("surface")))
        p.setPen(QPen(QColor(_tok("border")), 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)


class _Grip(QWidget):
    """An invisible strip along one edge / corner of a frameless window: its own resize cursor (never inherited by
    the buttons and fields next to it, so the arrows can't stick after a window closes) and a press starts the
    system resize."""
    T = 5

    def __init__(self, win, edges):
        super().__init__(win)
        self.win, self.edges = win, edges
        self.setCursor(FrameHost._cursor(edges))
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._drag = None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            h = self.win.windowHandle()
            if h is None or not h.startSystemResize(self.edges):
                self._drag = (ev.globalPosition().toPoint(), self.win.geometry())   # manual fallback
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        start, geo = self._drag
        d = ev.globalPosition().toPoint() - start
        g = QRect(geo)
        mn = self.win.minimumSize().expandedTo(QSize(200, 120))
        if self.edges & Qt.Edge.LeftEdge:
            g.setLeft(min(geo.left() + d.x(), geo.right() - mn.width()))
        if self.edges & Qt.Edge.RightEdge:
            g.setRight(max(geo.right() + d.x(), geo.left() + mn.width()))
        if self.edges & Qt.Edge.TopEdge:
            g.setTop(min(geo.top() + d.y(), geo.bottom() - mn.height()))
        if self.edges & Qt.Edge.BottomEdge:
            g.setBottom(max(geo.bottom() + d.y(), geo.top() + mn.height()))
        self.win.setGeometry(g)

    def mouseReleaseEvent(self, ev):
        self._drag = None


class Grips:
    """The eight resize grips of a frameless window, kept on its edges (hidden while maximised / not resizable)."""

    def __init__(self, win):
        E = Qt.Edge
        self.win = win
        self.items = [_Grip(win, e) for e in (E.LeftEdge, E.RightEdge, E.TopEdge, E.BottomEdge,
                                              E.LeftEdge | E.TopEdge, E.RightEdge | E.TopEdge,
                                              E.LeftEdge | E.BottomEdge, E.RightEdge | E.BottomEdge)]
        self.place()

    def place(self):
        w, h, t = self.win.width(), self.win.height(), _Grip.T
        big = self.win.isMaximized() or self.win.isFullScreen()
        E = Qt.Edge
        for g in self.items:
            e = g.edges
            if e == E.LeftEdge:
                r = QRect(0, t, t, h - 2 * t)
            elif e == E.RightEdge:
                r = QRect(w - t, t, t, h - 2 * t)
            elif e == E.TopEdge:
                r = QRect(t, 0, w - 2 * t, t)
            elif e == E.BottomEdge:
                r = QRect(t, h - t, w - 2 * t, t)
            else:
                c = 2 * t
                r = QRect(0 if e & E.LeftEdge else w - c, 0 if e & E.TopEdge else h - c, c, c)
            g.setGeometry(r)
            g.setVisible(not big)
            g.raise_()


RADIUS = 8                   # feature windows' corners (Windows 11's own radius)


def _win11() -> bool:
    if not _WIN:
        return False
    try:
        return sys.getwindowsversion().build >= 22000
    except Exception as _exc:
        _ignored(_exc)
        return False


def round_corners(win) -> str:
    """Rounded corners for a frameless feature window — "dwm" (Windows 11 draws them: smooth, with its shadow),
    "mask" (elsewhere: the window is clipped to a rounded shape; the outline is drawn as a smooth arc over the
    edge) or "square" (maximised / full screen, as the OS does)."""
    square = win.isMaximized() or win.isFullScreen()
    if _win11():
        try:
            import ctypes
            from ctypes import wintypes
            pref = ctypes.c_int(1 if square else 2)                # DWMWCP_DONOTROUND / DWMWCP_ROUND
            hr = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(int(win.winId())), 33, ctypes.byref(pref), ctypes.sizeof(pref))   # CORNER_PREFERENCE
            if hr == 0:                                               # S_OK; otherwise: the mask below
                win.clearMask()
                return "square" if square else "dwm"
        except Exception as _exc:
            _ignored(_exc)
    if square:
        win.clearMask()
        return "square"
    from PyQt6.QtGui import QPainterPath, QRegion
    path = QPainterPath()
    path.addRoundedRect(QRectF(win.rect()), RADIUS, RADIUS)
    win.setMask(QRegion(path.toFillPolygon().toPolygon()))
    return "mask"


class _Border(QWidget):
    """The 1 px window outline, drawn over everything (never takes the mouse) — rounded like the window."""

    def __init__(self, win):
        super().__init__(win)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.win = win

    def paintEvent(self, _e):
        if self.win.isMaximized() or self.win.isFullScreen():
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(_tok("control" if self.win.isActiveWindow() else "border")), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), RADIUS - 0.5, RADIUS - 0.5)


class _Chrome(QObject):
    """Moves (drag areas), resizes (edges) and keeps the outline / controls in step for one decorated window."""

    def __init__(self, win, controls, drag_areas, resizable):
        super().__init__(win)
        self.win, self.controls, self.resizable = win, controls, resizable
        self.border = _Border(win)
        self.grips = Grips(win) if resizable else None
        self.drag = None
        win.installEventFilter(self)
        for a in drag_areas:
            a.installEventFilter(self)
        self.areas = set(drag_areas)

    def _edges(self, pos):
        w = self.win
        if not self.resizable or w.isMaximized() or w.isFullScreen():
            return Qt.Edge(0)
        r, e = w.rect(), Qt.Edge(0)
        if pos.x() <= EDGE:
            e |= Qt.Edge.LeftEdge
        elif pos.x() >= r.width() - EDGE - 1:
            e |= Qt.Edge.RightEdge
        if pos.y() <= EDGE:
            e |= Qt.Edge.TopEdge
        elif pos.y() >= r.height() - EDGE - 1:
            e |= Qt.Edge.BottomEdge
        return e

    def eventFilter(self, obj, ev):
        t = ev.type()
        w = self.win
        try:
            if obj is w:
                if t in (QEvent.Type.Resize, QEvent.Type.Show):
                    self.border.setGeometry(w.rect())
                    self.border.raise_()
                    self.corners = round_corners(w)
                    if self.grips is not None:
                        self.grips.place()
                elif t == QEvent.Type.WindowStateChange:
                    if self.controls is not None:
                        self.controls.sync()
                    self.corners = round_corners(w)
                    self.border.update()
                    if self.grips is not None:
                        self.grips.place()
                elif t == QEvent.Type.ActivationChange:
                    self.border.update()
                return False
            if obj in self.areas:
                if t in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick) and \
                        ev.button() == Qt.MouseButton.LeftButton:
                    child = obj.childAt(ev.position().toPoint())
                    if child is not None and (child.focusPolicy() != Qt.FocusPolicy.NoFocus or
                                              child.inherits("QAbstractButton")):
                        return False                      # a button / field in the header works as usual
                    if hasattr(obj, "actionAt") and obj.actionAt(ev.position().toPoint()) is not None:
                        return False
                    if t == QEvent.Type.MouseButtonDblClick:
                        if self.resizable:
                            toggle_maximized(w)
                        return True
                    h = w.windowHandle()
                    if h is not None:
                        h.startSystemMove()
                    return True
        except RuntimeError:
            pass
        return False


def decorate(win):
    """Give a feature window the merged title bar (once, before it is first shown)."""
    if not _eligible(win):
        return False
    from PyQt6.QtWidgets import QDialog, QMainWindow
    win.setProperty("_aio_chrome", True)
    resizable = not (QWidget.minimumSize(win) == QWidget.maximumSize(win) and QWidget.minimumWidth(win) > 0)
    modal = isinstance(win, QDialog) and (win.isModal() or win.windowModality() != Qt.WindowModality.NonModal)
    geo = QWidget.geometry(win)
    was_sized = win.testAttribute(Qt.WidgetAttribute.WA_Resized)
    was_moved = win.testAttribute(Qt.WidgetAttribute.WA_Moved)
    flags = win.windowFlags() | Qt.WindowType.FramelessWindowHint
    win.setWindowFlags(flags)
    # minimise for every window that is not modal: feature windows tuck into the shelf (fixed-size pop-ups such as
    # the command palette keep just their close button)
    controls = WindowControls(win, minimise=not modal and (resizable or not isinstance(win, QDialog) or
                                                           bool(win.windowFlags() & Qt.WindowType.WindowMinimizeButtonHint)),
                              maximise=resizable)
    areas = []
    header = None
    for c in win.findChildren(QWidget):
        if c.property("windowHeader"):
            header = c
            break
    if isinstance(win, QMainWindow):
        mb = win.menuWidget()
        if mb is not None and hasattr(mb, "actions") and mb.actions():
            corner = mb.cornerWidget(Qt.Corner.TopRightCorner)
            box = QWidget(mb)
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            if corner is not None:
                corner.setParent(box)
                h.addWidget(corner)
                corner.show()
            h.addWidget(controls)
            mb.setCornerWidget(box, Qt.Corner.TopRightCorner)
            areas.append(mb)
        else:
            bar = TitleBar(win)
            bar.layout_.addWidget(controls)
            win.setMenuWidget(bar)
            areas.append(bar)
        win.setContentsMargins(1, 1, 1, 1)
    elif header is not None and header.property("controlsLayout") is not None:
        lay = header.property("controlsLayout")
        lay.addWidget(controls, 0, Qt.AlignmentFlag.AlignTop)
        areas.append(header)
        win.setProperty("_aio_header", True)
    elif QWidget.layout(win) is not None:
        old = QWidget.layout(win)
        holder = QWidget(win)
        holder.setObjectName("windowBody")
        holder.setLayout(old)                       # takes the layout and its widgets
        outer = QVBoxLayout(win)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)
        bar = TitleBar(win)
        bar.layout_.addWidget(controls)
        outer.addWidget(bar)
        outer.addWidget(holder, 1)
        areas.append(bar)
        if QWidget.minimumHeight(win) > 0:
            QWidget.setMinimumHeight(win, QWidget.minimumHeight(win) + bar.height())
        geo.setHeight(geo.height() + bar.height())
    else:
        win.setWindowFlags(flags & ~Qt.WindowType.FramelessWindowHint)
        win.setProperty("_aio_chrome", None)
        controls.deleteLater()
        return False
    if was_sized:                                    # an explicit size / place stays (plus the title bar)
        QWidget.resize(win, geo.size())
    if was_moved:
        QWidget.move(win, geo.topLeft())
    win._aio_chrome = _Chrome(win, controls, areas, resizable)
    win._window_controls = controls
    return True


def install_tool_window_chrome():
    """Hook show / exec / open once for the whole app (Python callers — every feature window)."""
    from PyQt6.QtWidgets import QDialog
    if getattr(QWidget, "_aio_chrome_hook", False):
        return
    QWidget._aio_chrome_hook = True
    w_show = QWidget.show
    d_exec, d_open = QDialog.exec, QDialog.open

    def show(self):
        if self.isWindow() and not self.isVisible():
            from AIO_QC.widgets import window_shelf
            if window_shelf.is_shelved(self):         # tucked into the shelf: back, where it was
                window_shelf.bring_back(self)
                return None
            _try(self)
        r = w_show(self)
        if self.isWindow() and self.windowState() & Qt.WindowState.WindowMinimized and \
                not isinstance(self, FrameHost):
            from AIO_QC.widgets.window_shelf import unminimise
            unminimise(self)                          # opening a feature again brings its window back
        return r

    w_raise = QWidget.raise_

    def raise_(self):
        if self.isWindow() and self.windowState() & Qt.WindowState.WindowMinimized and \
                not isinstance(self, FrameHost):
            from AIO_QC.widgets.window_shelf import unminimise
            unminimise(self)
        return w_raise(self)

    def exec_(self):
        _try(self)
        return d_exec(self)

    def open_(self):
        _try(self)
        return d_open(self)

    QWidget.show = show
    QWidget.raise_ = raise_
    QDialog.exec = exec_
    QDialog.open = open_


def _target_screen(win):
    from PyQt6.QtGui import QCursor
    par = win.parentWidget()
    scr = par.window().screen() if par is not None and par.window() is not None else None
    return scr or QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()


def _make_scrollable(win):
    """The window's content can't shrink to the screen (long forms, many fields): it scrolls instead."""
    from PyQt6.QtWidgets import QFrame, QMainWindow, QScrollArea
    if win.property("_aio_scroll"):
        return False

    def area_for(body):
        sa = QScrollArea()
        sa.setObjectName("windowScroll")
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sa.setWidget(body)
        return sa
    if isinstance(win, QMainWindow):
        cw = win.centralWidget()
        if cw is None or isinstance(cw, QScrollArea):
            return False
        win.takeCentralWidget()
        win.setCentralWidget(area_for(cw))
    else:
        body = win.findChild(QWidget, "windowBody", Qt.FindChildOption.FindDirectChildrenOnly)
        outer = win.layout()
        if body is not None and outer is not None:                       # a decorated window: scroll its body
            i = outer.indexOf(body)
            outer.removeWidget(body)
            outer.insertWidget(i, area_for(body), 1)
        elif outer is not None:
            holder = QWidget()
            holder.setObjectName("windowBody")
            holder.setLayout(outer)
            lay = QVBoxLayout(win)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(area_for(holder))
        else:
            return False
    win.setProperty("_aio_scroll", True)
    return True


def fit_before_show(win):
    """Never bigger than the screen: a window (or dialog) whose content wants more room than the screen has opens
    at the screen's size with its content scrolling, centred on the window it belongs to."""
    if not win.isWindow() or win.isVisible() or win.windowType() not in (Qt.WindowType.Window, Qt.WindowType.Dialog):
        return
    scr = _target_screen(win)
    if scr is None:
        return
    avail = scr.availableGeometry()
    if avail.width() < 300 or avail.height() < 300:
        return
    maxw, maxh = int(avail.width() * 0.96), int(avail.height() * 0.94)
    from AIO_QC.utils.qt_utils import wrap_long_labels
    wrap_long_labels(win)                         # a long hint line never sets the window's width
    lay = QWidget.layout(win)
    if lay is not None:
        lay.activate()
    mins = QWidget.minimumSizeHint(win).expandedTo(QWidget.minimumSize(win))
    want = QWidget.size(win) if win.testAttribute(Qt.WidgetAttribute.WA_Resized) else QWidget.sizeHint(win)
    if mins.width() > maxw or mins.height() > maxh or want.height() > maxh and lay is not None and \
            lay.hasHeightForWidth():
        if _make_scrollable(win):
            if QWidget.minimumWidth(win) > maxw:
                QWidget.setMinimumWidth(win, min(maxw, 320))
            if QWidget.minimumHeight(win) > maxh:
                QWidget.setMinimumHeight(win, min(maxh, 240))
    if QWidget.minimumWidth(win) > maxw:
        QWidget.setMinimumWidth(win, maxw)
    if QWidget.minimumHeight(win) > maxh:
        QWidget.setMinimumHeight(win, maxh)
    if QWidget.maximumWidth(win) < 16777215 and QWidget.maximumWidth(win) > maxw:
        QWidget.setMaximumWidth(win, maxw)
    if QWidget.maximumHeight(win) < 16777215 and QWidget.maximumHeight(win) > maxh:
        QWidget.setMaximumHeight(win, maxh)
    w, h = min(max(want.width(), 1), maxw), min(max(want.height(), 1), maxh)
    if (w, h) != (want.width(), want.height()) or not win.testAttribute(Qt.WidgetAttribute.WA_Resized):
        QWidget.resize(win, w, h)
    g = QWidget.geometry(win)
    par = win.parentWidget()
    if not win.testAttribute(Qt.WidgetAttribute.WA_Moved) and par is not None and par.window() is not None:
        g.moveCenter(par.window().frameGeometry().center())
    x = min(max(g.x(), avail.left()), avail.right() + 1 - g.width())
    y = min(max(g.y(), avail.top()), avail.bottom() + 1 - g.height())
    if (x, y) != (QWidget.x(win), QWidget.y(win)) and (win.testAttribute(Qt.WidgetAttribute.WA_Moved) or par is not None):
        QWidget.move(win, x, y)


def _try(win):
    try:
        from AIO_QC.widgets import window_shelf
        window_shelf.guard(win)                       # minimised by anything -> the shelf, never lost
    except Exception as _exc:
        _ignored(_exc)
    try:
        decorate(win)
    except Exception as e:
        try:
            from loguru import logger
            logger.debug(f"[window chrome] {type(win).__name__}: {e}")
        except Exception as _exc:
            _ignored(_exc)
    try:
        if not isinstance(win, FrameHost) and not any(c.__name__ == "QMenu" for c in type(win).__mro__):
            fit_before_show(win)
    except Exception as e:                           # never block a window over its decoration
        try:
            from loguru import logger
            logger.debug(f"[window chrome] {type(win).__name__}: {e}")
        except Exception as _exc:
            _ignored(_exc)
