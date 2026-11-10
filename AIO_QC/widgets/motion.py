# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Motion engine (from the Frame Extractor) - shared by the whole app: ONE FrameClock and ONE AppMotion per process,
installed by AIO QC's main(). The setting is AIO QC's `ui/animations` (full / reduced / off); the older on/off
helpers in widgets/anim.py follow it.

Animation toolkit - one reusable, efficient engine for every motion in the app.

  FrameClock   a single precise timer shared by all animations. It runs ONLY while something moves and ticks
               at the refresh rate of the screen the animation is on - never faster (a 60 Hz screen gets 60
               updates, a 144 Hz screen 144) - passing the real elapsed time, so motion stays smooth and speed
               stays exact even when a frame is late.
  animate()    tween any value (float, tuple of floats, QColor, QPointF) with an easing curve. Re-animating the
               same key retargets smoothly from wherever it is now (no jumps when the user clicks fast).
  Follower     critically-damped "chase a moving target" (zoom, smooth scrolling, progress bars).
  helpers      fade_in / fade_out (widgets), fade_window (top-level windows), Transition (cross-fade + slide of
               a whole page: AnimatedStack, animate_tabs), SmoothProgressBar, smooth_scroll (wheel scrolling).

Motion can be reduced or turned off (Settings ▸ General ▸ Animations): everything then jumps straight to its end state, so
no code path depends on animations running.
"""
import math
import threading
import time

from PyQt6.QtCore import QObject, QTimer, Qt, QPointF, QEvent, QRect
from PyQt6.QtGui import QColor, QPainter, QGuiApplication
from PyQt6.QtWidgets import (QWidget, QGraphicsOpacityEffect, QGraphicsView, QProgressBar, QStackedWidget,
                             QAbstractScrollArea, QApplication)

# ------------------------------------------------------------------ settings
MODES = ("full", "reduced", "off")
_mode = None


def _coerce(m):
    """AIO QC stored a bool here before 2.1 (Settings ▸ General ▸ Animations)."""
    if isinstance(m, bool):
        return "full" if m else "off"
    m = str(m).strip().lower()
    return {"true": "full", "1": "full", "false": "off", "0": "off"}.get(m, m if m in MODES else "full")


def mode():
    global _mode
    if _mode is None:
        from AIO_QC.utils.qt_utils import app_settings
        _mode = _coerce(app_settings().value("ui/animations", "full"))
    return _mode


def set_mode(m):
    global _mode
    _mode = _coerce(m)
    from AIO_QC.utils.qt_utils import app_settings
    app_settings().setValue("ui/animations", _mode)
    try:
        from AIO_QC.widgets import anim as _legacy           # the FrameAnimator helpers follow the same switch
        _legacy.ENABLED = _mode != "off"
    except Exception:
        pass


def enabled():
    return mode() != "off"


def scaled(ms):
    """Duration for the current mode (reduced motion = shorter, gentler)."""
    return 0 if mode() == "off" else int(ms * (0.55 if mode() == "reduced" else 1.0))


# ------------------------------------------------------------------ easing
def linear(t):
    return t


def out_cubic(t):
    return 1 - (1 - t) ** 3


def out_quint(t):
    return 1 - (1 - t) ** 5


def in_out_cubic(t):
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def out_back(t, s=1.2):
    t -= 1
    return 1 + (s + 1) * t ** 3 + s * t ** 2


EASINGS = {"linear": linear, "out_cubic": out_cubic, "out_quint": out_quint, "in_out_cubic": in_out_cubic,
           "out_back": out_back}


# ------------------------------------------------------------------ clock
def screen_hz(widget=None):
    scr = None
    try:
        scr = widget.screen() if widget is not None else QGuiApplication.primaryScreen()
    except RuntimeError:
        pass
    hz = scr.refreshRate() if scr is not None else 60.0
    return hz if hz and 20 < hz < 1000 else 60.0


class FrameClock(QObject):
    """Shared animation timer. Runs only while subscribers exist and is paced to the fastest subscriber's screen
    refresh: every tick is scheduled at last_due + 1/hz (a precise single-shot timer re-armed each frame), so the
    average update rate equals the display rate exactly - not the next whole millisecond above or below it -
    and never exceeds it."""
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            if not _gui_thread():                # never create the clock on a worker thread (its timer would
                return _NO_CLOCK                 # live there and never fire)
            cls._inst = FrameClock(QApplication.instance())
        return cls._inst

    def __init__(self, parent=None):
        super().__init__(parent)
        self._subs = {}              # key -> (callback(dt) -> bool keep_running, hz)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._tick)
        self._last = self._due = 0.0
        self._period = 1 / 60.0

    def add(self, key, cb, hz=60.0):
        if not _gui_thread():            # Qt timers only run on the UI thread
            return
        self._subs[key] = (cb, hz)
        self._retime()
        if not self._timer.isActive():
            self._last = time.perf_counter()
            self._due = self._last
            self._arm(self._last)

    def remove(self, key):
        self._subs.pop(key, None)
        if not self._subs:
            self._timer.stop()
        else:
            self._retime()

    def has(self, key):
        return key in self._subs

    def _retime(self):
        hz = max((h for _, h in self._subs.values()), default=60.0)
        self._period = 1.0 / hz

    def _arm(self, now):
        self._due += self._period
        if self._due < now:                       # fell behind (busy UI thread): restart the cadence from now
            self._due = now + self._period
        self._timer.start(max(1, int(round((self._due - now) * 1000))))

    def _tick(self):
        now = time.perf_counter()
        dt, self._last = min(0.1, now - self._last), now
        try:
            for key, (cb, _) in list(self._subs.items()):
                try:
                    keep = cb(dt)
                except RuntimeError:         # the widget was deleted mid-animation
                    keep = False
                except Exception:            # a broken callback must never stop every other animation
                    keep = False
                    try:
                        import traceback
                        from loguru import logger
                        logger.error("Animation callback failed:\n" + traceback.format_exc())
                    except Exception:
                        pass
                if not keep:
                    self._subs.pop(key, None)
        finally:
            if self._subs:
                self._retime()
                self._arm(now)


def _alive(obj):
    """True while obj's C++ object exists. NOT a weak reference to the Python wrapper: PyQt hands out temporary
    wrappers for objects Qt created itself (e.g. QTabWidget's page stack from page.parentWidget()); such a wrapper
    is freed as soon as the caller drops it although the widget lives on - a weakref would wrongly report it
    dead and orphan the animation half-way (that left tab cross-fade snapshots stuck on screen)."""
    if obj is None:
        return True
    try:
        from PyQt6 import sip
        return not sip.isdeleted(obj)
    except Exception:
        return True


def _gui_thread():
    return threading.current_thread() is threading.main_thread()


class _NoClock:
    def add(self, *a, **k):
        pass

    def remove(self, *a, **k):
        pass

    def has(self, *a, **k):
        return False


_NO_CLOCK = _NoClock()


# ------------------------------------------------------------------ tween
def _lerp(a, b, k):
    if isinstance(a, QColor):
        return QColor.fromRgbF(*(a.getRgbF()[i] + (b.getRgbF()[i] - a.getRgbF()[i]) * k for i in range(4)))
    if isinstance(a, QPointF):
        return QPointF(a.x() + (b.x() - a.x()) * k, a.y() + (b.y() - a.y()) * k)
    if isinstance(a, (tuple, list)):
        return type(a)(x + (y - x) * k for x, y in zip(a, b))
    return a + (b - a) * k


class Tween:
    def __init__(self, key, setter, start, end, ms, ease, on_done, owner):
        self.key, self.setter, self.start, self.end = key, setter, start, end
        self.ms, self.ease, self.on_done = max(1, ms), EASINGS.get(ease, ease) if isinstance(ease, str) else ease, on_done
        self.t = 0.0
        self.value = start
        self.owner = owner              # strong reference for the tween's short life (see _alive)

    def finish(self):
        """Jump to the end state and run on_done (exactly once)."""
        _TWEENS.pop(self.key, None)
        if self.t >= 2.0:
            return
        self.t = 2.0
        try:
            self.setter(self.end)
        except RuntimeError:
            pass
        if self.on_done:
            try:
                self.on_done()
            except RuntimeError:
                pass

    def step(self, dt):
        if not _alive(self.owner):
            _TWEENS.pop(self.key, None)
            return False
        if self.t >= 2.0:                        # already finished
            return False
        self.t = min(1.0, self.t + dt * 1000.0 / self.ms)
        self.value = _lerp(self.start, self.end, self.ease(self.t))
        try:
            self.setter(self.end if self.t >= 1.0 else self.value)
        except RuntimeError:                 # target deleted
            _TWEENS.pop(self.key, None)
            return False
        if self.t >= 1.0:
            self.finish()
            return False
        return True


_TWEENS = {}


def animate(key, setter, start, end, ms=220, ease="out_cubic", on_done=None, widget=None, retarget=True):
    """Tween setter(value) from start to end. key identifies the animated property: animating the same key again
    continues from the current value (retarget). widget: whose screen sets the frame rate (and whose deletion
    stops the tween). Returns the Tween, or None when it completed immediately (animations off)."""
    old = _TWEENS.get(key)
    if old is not None and retarget:
        start = old.value
    ms = scaled(ms)
    if ms <= 0 or not _gui_thread():          # no motion, or called off the UI thread: end state right away
        _TWEENS.pop(key, None)
        FrameClock.get().remove(key)
        setter(end)
        if on_done:
            on_done()
        return None
    tw = Tween(key, setter, start, end, ms, ease, on_done, widget)
    _TWEENS[key] = tw
    FrameClock.get().add(key, tw.step, screen_hz(widget))
    return tw


def stop(key, finish=False):
    """Stop a tween; finish=True jumps to its end state and runs its on_done."""
    tw = _TWEENS.pop(key, None)
    FrameClock.get().remove(key)
    if tw is not None and finish:
        tw.finish()


def running(key):
    return key in _TWEENS


class Follower:
    """Critically-damped chase of a target that may keep moving (frame-rate independent):
    value += (target - value) * (1 - exp(-dt / tau)). setter(value) is called every frame until settled."""

    def __init__(self, setter, value=0.0, tau=0.07, eps=1e-3, widget=None, on_settled=None):
        self.setter, self.value, self.target = setter, value, value
        self.tau, self.eps, self.widget, self.on_settled = tau, eps, widget, on_settled
        self.key = ("follower", id(self))

    def set_target(self, target, immediate=False):
        self.target = target
        if immediate or not enabled() or not _gui_thread():
            self.value = target
            FrameClock.get().remove(self.key)
            self.setter(target)
            return
        if not FrameClock.get().has(self.key):
            FrameClock.get().add(self.key, self._step, screen_hz(self.widget))

    def _step(self, dt):
        tau = self.tau * (0.6 if mode() == "reduced" else 1.0)
        k = 1.0 - math.exp(-dt / max(1e-4, tau))
        self.value += (self.target - self.value) * k
        done = abs(self.target - self.value) <= self.eps
        if done:
            self.value = self.target
        self.setter(self.value)
        if done and self.on_settled:
            self.on_settled()
        return not done

    def stop(self):
        FrameClock.get().remove(self.key)


# ------------------------------------------------------------------ widget helpers
def fade_in(w, ms=180, start=0.0):
    """Fade a (non-OpenGL) widget in. The opacity effect is removed when done (no lasting paint cost)."""
    if not enabled() or w is None:
        if w is not None:
            w.show()
        return
    eff = QGraphicsOpacityEffect(w)
    eff.setOpacity(start)
    w.setGraphicsEffect(eff)
    w.show()

    def done():
        try:
            w.setGraphicsEffect(None)
        except RuntimeError:
            pass
    animate(("fade", id(w)), eff.setOpacity, start, 1.0, ms, "out_cubic", done, w, retarget=False)


def fade_out(w, ms=160, hide=True, on_done=None):
    if not enabled() or w is None:
        if w is not None and hide:
            w.hide()
        if on_done:
            on_done()
        return
    eff = QGraphicsOpacityEffect(w)
    eff.setOpacity(1.0)
    w.setGraphicsEffect(eff)

    def done():
        try:
            if hide:
                w.hide()
            w.setGraphicsEffect(None)
        except RuntimeError:
            pass
        if on_done:
            on_done()
    animate(("fade", id(w)), eff.setOpacity, 1.0, 0.0, ms, "out_cubic", done, w, retarget=False)


def fade_window(win, ms=170):
    """Top-level window / dialog: fade in through window opacity (composited by the OS - nearly free)."""
    if not enabled() or win is None:
        return
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    if win.findChildren(QOpenGLWidget):          # opacity makes the window layered (Windows): no OpenGL there
        return
    win.setWindowOpacity(0.0)

    def setter(v):
        try:
            win.setWindowOpacity(v)
        except RuntimeError:
            pass
    animate(("winfade", id(win)), setter, 0.0, 1.0, ms, "out_cubic", None, win, retarget=False)


class FadeShowFilter(QObject):
    """Install on a dialog / window: every time it is shown it fades in (works with exec())."""

    def __init__(self, win, ms=170):
        super().__init__(win)
        self.ms = ms
        win.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Show and obj.isWindow():
            fade_window(obj, self.ms)
        return False


def fade_on_show(win, ms=170):
    FadeShowFilter(win, ms)
    return win


_OVERLAYS = set()


class Transition(QWidget):
    """Overlay that shows a snapshot of the previous content fading and sliding away over the live new content -
    a cross-fade that costs one pixmap blit per frame (the page underneath is painted once)."""

    def __init__(self, host, pixmap, dx=0, dy=0, ms=230):
        super().__init__(host)
        self.host = host                 # keep the (possibly temporary) wrapper of the host alive
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.pm, self.dx, self.dy, self.k = pixmap, dx, dy, 0.0
        self._gone = False
        self.setGeometry(QRect(0, 0, host.width(), host.height()))
        self.raise_()
        self.show()
        _OVERLAYS.add(self)
        # watchdog: the snapshot is removed even if the animation could not run for any reason
        QTimer.singleShot(ms + 400, self._done)
        animate(("transition", id(host)), self._set, 0.0, 1.0, ms, "out_cubic", self._done, host, retarget=False)

    def _set(self, k):
        self.k = k
        self.update()

    def _done(self):
        if self._gone:
            return
        self._gone = True
        _OVERLAYS.discard(self)
        try:
            self.hide()
            self.deleteLater()
        except RuntimeError:
            pass

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setOpacity(max(0.0, 1.0 - self.k))
        p.drawPixmap(int(self.dx * self.k), int(self.dy * self.k), self.pm)
        p.end()


def transition(host, old_pixmap, direction=1, axis="x", ms=230):
    """Play a page transition over host (old snapshot slides 24 px and fades out)."""
    if not enabled() or old_pixmap is None or old_pixmap.isNull() or not host.isVisible():
        return
    stop(("transition", id(host)), finish=True)
    from PyQt6 import sip
    for ch in list(_OVERLAYS):
        if sip.isdeleted(ch):                       # its window was destroyed with it: just forget it
            _OVERLAYS.discard(ch)
            continue
        if ch.host is host or ch.parent() is host:
            ch._done()
    for ch in host.findChildren(Transition):
        ch._done()
    d = -24 * direction if mode() == "full" else 0
    Transition(host, old_pixmap, d if axis == "x" else 0, d if axis == "y" else 0, ms)


class AnimatedStack(QStackedWidget):
    """QStackedWidget whose page changes cross-fade and slide in the direction of travel."""

    def setCurrentIndex(self, i):
        old = self.currentIndex()
        if i == old or not enabled() or not self.isVisible() or self.currentWidget() is None:
            super().setCurrentIndex(i)
            return
        pm = self.currentWidget().grab()
        super().setCurrentIndex(i)
        transition(self, pm, 1 if i > old else -1)


def animate_tabs(tabs):
    """Cross-fade QTabWidget page changes made with the mouse (the old page is captured before the switch)."""
    state = {"pm": None, "from": tabs.currentIndex()}

    def pressed(i):
        if enabled() and i != tabs.currentIndex() and tabs.currentWidget() is not None:
            state["pm"], state["from"] = tabs.currentWidget().grab(), tabs.currentIndex()

    def changed(i):
        pm, state["pm"] = state["pm"], None
        page = tabs.currentWidget()
        if pm is not None and page is not None:
            host = page.parentWidget() or page
            transition(host, pm, 1 if i > state["from"] else -1)
        state["from"] = i
    tabs.tabBar().tabBarClicked.connect(pressed)
    tabs.currentChanged.connect(changed)


class SmoothProgressBar(QProgressBar):
    """setValue() glides to the new value (display-rate, ~90 ms) instead of jumping; resets are instant."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._target = 0
        self._f = Follower(lambda v: QProgressBar.setValue(self, int(round(v))), 0.0, tau=0.06, eps=0.5, widget=self)

    def setValue(self, v):
        v = int(v)
        jump = v < self._target or v <= self.minimum()
        self._target = v
        self._f.set_target(float(v), immediate=jump or not self.isVisible())

    def value(self):
        return self._target


class _Scroller:
    """Glide one scroll bar towards a wheel-driven target."""

    def __init__(self, bar, widget):
        self.bar = bar
        self.f = Follower(self._set, float(bar.value()), tau=0.075, eps=0.5, widget=widget)

    def _set(self, v):
        self.bar.setValue(int(round(v)))

    def wheel(self, px):
        f, bar = self.f, self.bar
        if not FrameClock.get().has(f.key):
            f.value = f.target = float(bar.value())      # resync after drags / programmatic scrolls
        f.set_target(max(bar.minimum(), min(bar.maximum(), f.target + px)))


def _scroll_wheel(area, ev, cache):
    """Handle a wheel event for a scroll area smoothly. Returns True when consumed."""
    if ev.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
        return False
    if not ev.pixelDelta().isNull():          # touchpads / precision mice already scroll smoothly
        return False
    ad = ev.angleDelta()
    horiz = (ad.x() != 0 and ad.y() == 0) or bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
    bar = area.horizontalScrollBar() if horiz else area.verticalScrollBar()
    if bar is None or bar.maximum() <= bar.minimum() or not bar.isEnabled():
        return False
    from PyQt6.QtWidgets import QAbstractItemView
    if isinstance(area, QAbstractItemView):
        mode_attr = area.horizontalScrollMode if horiz else area.verticalScrollMode
        if mode_attr() != QAbstractItemView.ScrollMode.ScrollPerPixel:
            (area.setHorizontalScrollMode if horiz else area.setVerticalScrollMode)(
                QAbstractItemView.ScrollMode.ScrollPerPixel)
            return False                      # this notch scrolls the classic way; the next ones glide
    delta = (ad.x() or ad.y()) if horiz else ad.y()
    lines = QApplication.wheelScrollLines() or 3
    px = -delta / 120.0 * max(bar.singleStep(), 20) * lines
    key = id(bar)
    sc = cache.get(key)
    if sc is None or sc.bar is not bar:
        sc = cache[key] = _Scroller(bar, area)
    sc.wheel(px)
    return True


class AppMotion(QObject):
    """Application-wide window behaviour, installed once (install()): every window is kept inside its screen,
    every dialog fades in, every scroll area glides on the mouse wheel. Costs one integer comparison per event
    when nothing applies."""
    _SHOW, _WHEEL = QEvent.Type.Show, QEvent.Type.Wheel

    def __init__(self, app):
        super().__init__(app)
        self._scrollers = {}
        app.installEventFilter(self)

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == self._WHEEL:
            if not enabled() or not isinstance(obj, QWidget):
                return False
            area = obj.parentWidget()
            if isinstance(area, QAbstractScrollArea) and area.viewport() is obj and \
                    not area.property("noSmoothScroll") and not isinstance(area, QGraphicsView):
                return _scroll_wheel(area, ev, self._scrollers)
            return False
        if t == self._SHOW and isinstance(obj, QWidget) and obj.isWindow():
            wt = obj.windowType()
            if wt in (Qt.WindowType.Window, Qt.WindowType.Dialog):
                from AIO_QC.utils.qt_utils import fit_to_screen
                fit_to_screen(obj)                       # every window stays inside its screen
                if enabled() and wt == Qt.WindowType.Dialog and not obj.property("noFade"):
                    fade_window(obj, 160)
        return False


def install(app):
    """Turn on app-wide window behaviour (keep windows on screen, dialog fades, smooth wheel scrolling).
    Call once after QApplication exists."""
    if getattr(app, "_app_motion", None) is None:
        app._app_motion = AppMotion(app)
        FrameClock.get()                         # created here, on the UI thread
    return app._app_motion


def installed():
    from PyQt6.QtWidgets import QApplication as _A
    app = _A.instance()
    return app is not None and getattr(app, "_app_motion", None) is not None


def smooth_scroll(*areas):
    """Per-widget opt-in (when install() is not used): glide wheel scrolling on these scroll areas."""
    for a in areas:
        if isinstance(a, QAbstractScrollArea):
            _ScrollFilter(a)


class _ScrollFilter(QObject):
    def __init__(self, area):
        super().__init__(area)
        self.area, self.cache = area, {}
        area.viewport().installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Wheel and enabled():
            return _scroll_wheel(self.area, ev, self.cache)
        return False
