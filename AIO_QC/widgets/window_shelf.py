# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Minimise, the same everywhere.

    A feature window (a scan, Search in Words, Claude Review, Preparation, Train, Testing, the Crop Viewer, Settings …)
    minimises into the SHELF: a row of chips at the bottom-left of the main AIO QC window. The window is only
    hidden — its job keeps running, and its chip shows the job's progress, then "Done" (with a short pulse). Click a
    chip (or open the feature again from its menu / shortcut) to bring the window back where it was; × closes it.
    No taskbar round trip, so nothing gets lost: a feature window belongs to the main window and has no taskbar button
    of its own, and on some systems a minimised one could not be found again.

    The main window (and a window of its own such as Help) minimises to the taskbar / Dock as usual.

    Whatever minimises a feature window — its button, Win+Down, a desktop's own title-bar button with "Use the system
    title bar" — ends in the shelf (guard). Opening a feature again always brings its window back: shown, never left
    minimised (bring_back, also used by the app-wide show hook).

    When a background job finishes while AIO QC is minimised or in the background, the app asks for attention: the
    taskbar button flashes (Windows), the Dock icon bounces (macOS), the window is marked urgent (Linux).
"""
from __future__ import annotations

import sys
import weakref

from PyQt6.QtCore import QEvent, QObject, QSize, Qt, QTimer, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QToolButton,
                             QWidget)

from AIO_QC.core.diag import ignored as _ignored

ATTENTION_MIN_S = 3.0                    # jobs shorter than this never ask for attention

_shelved: dict[int, "Chip"] = {}         # id(window) -> its chip


# ---------------------------------------------------------------------------------------------------- helpers
def _alive(w) -> bool:
    try:
        from PyQt6 import sip
        return w is not None and not sip.isdeleted(w)
    except Exception:
        return w is not None


def is_modal(w) -> bool:
    return isinstance(w, QDialog) and (w.isModal() or w.windowModality() != Qt.WindowModality.NonModal)


def root_of(w):
    """The main window a feature window belongs to: the top of its parent chain (None for a window of its own)."""
    p = w.parentWidget()
    if p is None:
        return None
    while p.parentWidget() is not None:
        p = p.parentWidget()
    r = p.window()
    return None if r is w else r


def is_shelved(w) -> bool:
    return id(w) in _shelved


def unminimise(w):
    st = w.windowState()
    if st & Qt.WindowState.WindowMinimized:
        w.setWindowState((st & ~Qt.WindowState.WindowMinimized) | Qt.WindowState.WindowActive)


def bring_back(w):
    """Show a window that was minimised, shelved or behind others — wherever it was, as it was (maximised stays)."""
    if not _alive(w):
        return
    chip = _shelved.get(id(w))
    if chip is not None:
        chip.shelf.release(w)
        return
    unminimise(w)
    if not w.isVisible():
        w.show()
    w.raise_()
    w.activateWindow()


def can_shelve(w) -> bool:
    if not _alive(w) or not w.isWindow() or is_modal(w):
        return False
    from AIO_QC.widgets.window_frame import FrameHost
    if isinstance(w, FrameHost):
        return False
    return root_of(w) is not None and not w.property("noShelf")


def minimise(w):
    """What a window's minimise button does."""
    custom = getattr(w, "custom_minimise", None)
    if custom is not None:
        custom()
    elif can_shelve(w):
        shelve(w)
    else:
        w.showMinimized()


def shelve(w) -> bool:
    if is_shelved(w):
        return True
    if not can_shelve(w):
        return False
    shelf = Shelf.for_host(root_of(w))
    if shelf is None:
        return False
    shelf.add(w)
    return True


# ---------------------------------------------------------------------------------------------------- the chip
class Chip(QFrame):
    """One tucked-away window: icon · title · its job's progress or 'Done' · ×. Click brings the window back."""
    clicked = pyqtSignal()

    def __init__(self, shelf, win):
        super().__init__(shelf)
        self.shelf, self.win = shelf, win
        self.done = False
        self.setObjectName("shelfChip")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 3, 2, 3)
        lay.setSpacing(6)
        self.icon = QLabel()
        ic = win.windowIcon()
        if not ic.isNull():
            self.icon.setPixmap(ic.pixmap(QSize(16, 16)))
        lay.addWidget(self.icon)
        self.title = QLabel("")
        f = self.title.font()
        f.setBold(True)
        self.title.setFont(f)
        self.state = QLabel("")
        self.state.setObjectName("shelfChipState")
        lay.addWidget(self.title)
        lay.addWidget(self.state)
        self.bar = QProgressBar(self)                        # a 2-px line along the chip's bottom edge
        self.bar.setObjectName("shelfChipBar")
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        self.bar.hide()
        self.close_btn = QToolButton()
        self.close_btn.setObjectName("shelfChipClose")
        self.close_btn.setText("×")
        self.close_btn.setAutoRaise(True)
        self.close_btn.setToolTip("Close this window")
        self.close_btn.clicked.connect(self._close)
        lay.addWidget(self.close_btn)
        for lab in (self.icon, self.title, self.state):
            lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedHeight(30)
        self._pulse = None
        self.refresh()

    TITLE_W, STATE_W = 150, 110

    def _title(self) -> str:
        t = (self.win.windowTitle() or type(self.win).__name__).replace(" — AIO QC App", "")
        return self.title.fontMetrics().elidedText(t, Qt.TextElideMode.ElideRight, self.TITLE_W)

    def _set_state(self, text: str):
        self.state.ensurePolished()
        t = self.state.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, self.STATE_W)
        if t != self.state.text():
            self.state.setText(t)
            self.state.setFixedWidth(self.state.fontMetrics().horizontalAdvance(t) + 6 if t else 0)
            self.state.setVisible(bool(t))
            self.shelf._place() if self.shelf.chips else None

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.bar.setGeometry(8, self.height() - 3, max(0, self.width() - 16), 2)

    def refresh(self):
        if not _alive(self.win):
            return
        self.title.ensurePolished()                       # measure with the font the style sheet gives it
        t = self._title()
        if t != self.title.text():
            self.title.setText(t)
            self.title.setFixedWidth(self.title.fontMetrics().horizontalAdvance(t) + 6)
        job = _running_job(self.win)
        if job is not None:
            frac = job.fraction
            self._set_state((f"{frac * 100:.0f} %" if frac is not None else "working …") +
                            (f" · {job.text}" if job.text else ""))
            self.bar.setVisible(frac is not None)
            if frac is not None:
                self.bar.setValue(int(frac * 1000))
            self.done = False
        elif self.done:
            self._set_state("✓ done")
            self.bar.hide()
        else:
            self._set_state("")
            self.bar.hide()
        job_txt = job.label() if job is not None else ("finished" if self.done else "")
        self.setToolTip(f"{self.win.windowTitle()}" + (f"\n{job_txt}" if job_txt else "") +
                        "\nClick to bring the window back · × closes it")

    def finished(self):
        self.done = True
        self.refresh()
        self.pulse()

    def pulse(self):
        from AIO_QC.widgets.design import tokens
        ok = QColor(tokens()["ok"])
        a = QVariantAnimation(self, startValue=0.0, endValue=1.0, duration=1400)

        def step(v):
            k = 1.0 - abs(1.0 - 2.0 * (v * 3 % 1.0))           # three soft blinks
            c = QColor(ok)
            c.setAlphaF(0.12 + 0.35 * k)
            self.setStyleSheet(f"#shelfChip {{ background: {c.name(QColor.NameFormat.HexArgb)}; }}")

        a.valueChanged.connect(step)
        a.finished.connect(lambda: self.setStyleSheet(""))
        a.start()
        self._pulse = a

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)

    def _close(self):
        w = self.win
        if not _alive(w):
            self.shelf.drop(w)
            return
        job = _running_job(w)
        if job is not None:
            from PyQt6.QtWidgets import QMessageBox
            if QMessageBox.question(self.shelf.host, w.windowTitle(),
                                    f"{w.windowTitle()} is still working ({job.label()}).\n\nClose it and stop the "
                                    "job?") != QMessageBox.StandardButton.Yes:
                return
        self.shelf.drop(w)
        w.close()


# ---------------------------------------------------------------------------------------------------- the shelf
class Shelf(QWidget):
    """The chips of one main window, at its bottom-left, above the status bar; follows the window's size."""
    BOTTOM = 32
    LEFT = 10

    @staticmethod
    def for_host(host):
        if not _alive(host):
            return None
        s = getattr(host, "_aio_shelf", None)
        if s is None or not _alive(s):
            s = host._aio_shelf = Shelf(host)
        return s

    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.setObjectName("windowShelf")
        self.chips: dict[int, Chip] = {}
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(6)
        self._timer = QTimer(self, interval=400, timeout=self._tick)
        host.installEventFilter(self)
        self.hide()

    # -- chips
    def add(self, w):
        geo = w.geometry()
        state = w.windowState() & ~Qt.WindowState.WindowMinimized
        chip = Chip(self, w)
        chip.geo, chip.state_before = geo, state
        chip.clicked.connect(lambda w=w: self.release(w))
        self.chips[id(w)] = chip
        _shelved[id(w)] = chip
        self._lay.addWidget(chip)
        _Watch.get().watch(w)
        w.hide()
        if w.windowState() & Qt.WindowState.WindowMinimized:
            w.setWindowState(state)
        self._place()
        self.show()
        self.raise_()
        self._timer.start()
        try:
            from AIO_QC.widgets.motion import fade_in
            fade_in(chip)
        except Exception as _exc:
            _ignored(_exc)

    def release(self, w, show: bool = True):
        chip = self.chips.pop(id(w), None)
        _shelved.pop(id(w), None)
        if chip is None:
            return
        chip.hide()
        chip.deleteLater()
        if show and _alive(w):
            w.setWindowState(chip.state_before)
            if not (chip.state_before & (Qt.WindowState.WindowMaximized | Qt.WindowState.WindowFullScreen)):
                w.setGeometry(chip.geo)
            if not w.isVisible():
                w.show()
            w.raise_()
            w.activateWindow()
            try:
                from AIO_QC.widgets import anim
                anim.window_fade_in(w)
            except Exception as _exc:
                _ignored(_exc)
        self._after_change()

    def drop(self, w):
        """The window is going away (closed / deleted): just remove its chip."""
        self.release(w, show=False)

    def _after_change(self):
        if not self.chips:
            self._timer.stop()
            self.hide()
        else:
            self._place()

    def _tick(self):
        for k, chip in list(self.chips.items()):
            if not _alive(chip.win):
                self.chips.pop(k, None)
                _shelved.pop(k, None)
                chip.deleteLater()
                continue
            chip.refresh()
        self._after_change()

    # -- geometry
    def _place(self):
        for c in self.chips.values():
            c.updateGeometry()
        self._lay.invalidate()
        self._lay.activate()
        h = self.host
        sz = self._lay.sizeHint()
        w = min(sz.width(), max(120, h.width() - 2 * self.LEFT))
        self.setGeometry(self.LEFT, max(0, h.height() - self.BOTTOM - sz.height()), w, sz.height())
        self.raise_()

    def eventFilter(self, obj, ev):
        if obj is self.host and ev.type() in (QEvent.Type.Resize, QEvent.Type.Show) and self.chips:
            QTimer.singleShot(0, self._place)
        return False

    def paintEvent(self, _e):
        pass


# ---------------------------------------------------------------------------------------------------- watching
class _Watch(QObject):
    """App-wide: shelved windows shown by any path (menu, shortcut, code) leave the shelf; closed ones drop out;
    feature windows minimised by anything (Win+Down, a desktop's title-bar button) go to the shelf."""
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None or not _alive(cls._inst):
            cls._inst = _Watch(QApplication.instance())
        return cls._inst

    def __init__(self, parent=None):
        super().__init__(parent)
        self._guarded = weakref.WeakSet()

    def watch(self, w):
        w.removeEventFilter(self)
        w.installEventFilter(self)

    def guard(self, w):
        if w in self._guarded:
            return
        self._guarded.add(w)
        self.watch(w)

    def eventFilter(self, obj, ev):
        t = ev.type()
        try:
            chip = _shelved.get(id(obj))
            if chip is not None and chip.win is obj:
                if t == QEvent.Type.Show:
                    QTimer.singleShot(0, lambda o=obj: _released_by_show(o))
                elif t in (QEvent.Type.Close, QEvent.Type.DeferredDelete):
                    chip.shelf.drop(obj)
            elif t == QEvent.Type.WindowStateChange and isinstance(obj, QWidget) and obj.isWindow() \
                    and obj.windowState() & Qt.WindowState.WindowMinimized:
                handles = getattr(obj, "handles_minimise", None)
                if not (handles is not None and handles()) and can_shelve(obj):
                    QTimer.singleShot(0, lambda o=obj: _shelve_minimised(o))
        except RuntimeError:
            pass
        return False


def _released_by_show(w):
    chip = _shelved.get(id(w))
    if chip is not None and _alive(w) and w.isVisible():
        chip.shelf.release(w)


def _shelve_minimised(w):
    if not _alive(w) or not (w.windowState() & Qt.WindowState.WindowMinimized) or is_shelved(w):
        return
    w.setWindowState(w.windowState() & ~Qt.WindowState.WindowMinimized)
    shelve(w)


def guard(w):
    """Called for every window before it is first shown."""
    try:
        if can_shelve(w):
            _Watch.get().guard(w)
    except Exception as _exc:
        _ignored(_exc)


# ---------------------------------------------------------------------------------------------------- jobs
def _running_job(w):
    try:
        from AIO_QC.widgets import tasks
        c = tasks.center()
        if c is None:
            return None
        for t in reversed(c.tasks):
            o = getattr(t, "owner", None)
            if _alive(o) and isinstance(o, QWidget) and (o is w or o.window() is w):
                return t
    except Exception as _exc:
        _ignored(_exc)
    return None


def _main_window():
    shell = getattr(QApplication.instance(), "shell", None)
    host = getattr(shell, "host", None)
    if _alive(host):
        return host
    return QApplication.activeWindow() or next((w for w in QApplication.topLevelWidgets()
                                                if w.isVisible() and w.parentWidget() is None), None)


def _on_job_finished(t):
    """A background job ended: its shelved window's chip says Done; an app in the background asks for attention."""
    try:
        if t.cancelled:
            return
        o = getattr(t, "owner", None)
        win = o.window() if _alive(o) and isinstance(o, QWidget) else None
        if win is not None:
            chip = _shelved.get(id(win))
            if chip is not None:
                chip.finished()
        if t.duration() < ATTENTION_MIN_S:
            return
        main = (root_of(win) if win is not None else None) or _main_window()
        away = QApplication.applicationState() != Qt.ApplicationState.ApplicationActive or \
            (main is not None and main.isMinimized())
        if away and main is not None:
            QApplication.alert(main, 0)              # taskbar flash · Dock bounce · urgent hint, until looked at
    except Exception as _exc:
        _ignored(_exc)


def install(app=None):
    """Once per app: job endings reach the shelf and the attention signal."""
    app = app or QApplication.instance()
    if app is None or getattr(app, "_aio_shelf_installed", False):
        return
    app._aio_shelf_installed = True
    from AIO_QC.widgets import tasks
    c = tasks.center()
    if c is not None:
        c.finished.connect(_on_job_finished)


# ---------------------------------------------------------------------------------------------------- macOS
MAC = sys.platform == "darwin"


def minimise_active():
    """Cmd+M (macOS): minimise the window in front — a feature window to the shelf, the main window to the Dock."""
    w = QApplication.activeWindow()
    if w is None or is_modal(w):
        return
    from AIO_QC.widgets.window_frame import FrameHost
    if isinstance(w, FrameHost):
        (getattr(w, "custom_minimise", None) or w.showMinimized)()
    else:
        minimise(w)

