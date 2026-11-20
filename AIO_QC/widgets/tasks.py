# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
One background-task indicator for the whole app.

Long jobs from any mode report here - Frame Extractor batch extraction, marker extraction, timeline analysis,
clip / annotated-video exports, Testing runs / test-set builds / reports, Image QC scans and Preparation runs, Data
Studio exports. Every mode window shows the same indicator in its status bar (installed by the shell): a loading
ring, then the latest job's text and progress; when everything has finished the ring turns into a green tick for a
moment. Click it (or the Background Tasks toolbar button) for the Background Tasks window: every job of this session,
running and finished, with progress, timings and Cancel.

    t = tasks.start("Batch extraction · 3 videos", cancel=fn)    # UI thread only
    t.update(0.42, "frame 1,204 / 2,880")                         # fraction 0..1 or None (busy)
    t.finish()                                                    # idempotent; also when the owner is deleted
"""
from __future__ import annotations

import time

from PyQt6.QtCore import QObject, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QConicalGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
                             QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
from AIO_QC.core.diag import ignored as _ignored


class Task:
    def __init__(self, center, title, cancel=None):
        self.center, self.title, self.cancel_fn = center, title, cancel
        self.owner = None                          # the job's window / object (the shelf shows its progress)
        self.fraction, self.text = None, ""
        self.t0 = time.monotonic()
        self.started = time.time()                # wall clock, for the viewer
        self.t1 = None
        self.done = False
        self.cancelled = False

    @property
    def state(self) -> str:
        return "running" if not self.done else ("cancelled" if self.cancelled else "done")

    def duration(self) -> float:
        return (self.t1 if self.t1 is not None else time.monotonic()) - self.t0

    def update(self, fraction=None, text=None):
        if self.done:
            return
        if fraction is not None:
            self.fraction = max(0.0, min(1.0, float(fraction)))
        if text is not None:
            self.text = str(text)
        self.center._touch()

    def finish(self):
        if not self.done:
            self.done = True
            self.t1 = time.monotonic()
            self.center._remove(self)

    def cancel(self):
        if self.cancel_fn is not None and not self.done:
            self.cancelled = True
            try:
                self.cancel_fn()
            except Exception as _exc:
                _ignored(_exc)

    def label(self):
        pct = f" {self.fraction * 100:.0f}%" if self.fraction is not None else ""
        return f"{self.title}{pct}" + (f" — {self.text}" if self.text else "")


class TaskCenter(QObject):
    changed = pyqtSignal()
    finished = pyqtSignal(object)                 # a Task that ended (done or cancelled)

    HISTORY = 300

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tasks: list[Task] = []
        self.history: list[Task] = []             # finished jobs of this session, newest last
        self._timer = QTimer(self, singleShot=True, interval=100, timeout=self.changed.emit)   # <= 10 updates/s

    def start(self, title, cancel=None, owner=None) -> Task:
        t = Task(self, title, cancel)
        self.tasks.append(t)
        # the job's QObject (given, or the one whose method cancels it) going away ends the task: a window closed
        # mid-job must not leave a task "running" forever
        owner = owner if owner is not None else getattr(cancel, "__self__", None)
        t.owner = owner
        if isinstance(owner, QObject):
            try:
                owner.destroyed.connect(lambda *_a, t=t: t.finish())
            except (RuntimeError, TypeError):
                pass
        self.changed.emit()
        return t

    def _touch(self):
        if not self._timer.isActive():
            self._timer.start()

    def _remove(self, t):
        ended = t in self.tasks
        if ended:
            self.tasks.remove(t)
            self.history.append(t)
            del self.history[:-self.HISTORY]
        self.changed.emit()
        if ended:
            self.finished.emit(t)

    def clear_history(self):
        self.history.clear()
        self.changed.emit()


def center() -> TaskCenter | None:
    app = QApplication.instance()
    if app is None:
        return None
    c = getattr(app, "_task_center", None)
    if c is None:
        c = TaskCenter(app)
        app._task_center = c
    return c


class _Null:
    def update(self, *a, **k):
        pass

    def finish(self):
        pass


def start(title, cancel=None, owner=None):
    c = center()
    return c.start(title, cancel, owner) if c is not None else _Null()


# ---------------------------------------------------------------------------------------------------- loading ring
BLUE, CYAN, GREEN = QColor("#2f7fe8"), QColor("#9be3f5"), QColor("#2fb344")
TURN_S = 0.9           # one turn of the arc
MORPH_S = 0.30         # ring: blue gradient -> plain green
TICK_S = 0.32          # the tick draws itself


def _clamp(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _ease(x):
    return 1 - (1 - x) ** 3


def paint_ring(p: QPainter, rect: QRectF, angle: float, done_k: float = 0.0, tick_k: float = 0.0, wobble: float = 0.0):
    """One frame. 3.0.1 (as in the reference): a thin, faint track ring with a brighter arc turning on it — the arc
    fades out along its tail. Ink = the theme's text colour (dark on light, light on dark). done_k morphs it into a
    plain green ring, tick_k draws the green tick (0..1 each)."""
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = min(rect.width(), rect.height())
    c = rect.center()
    pen_w = max(1.4, s * 0.11)
    r = s / 2 - pen_w / 2 - max(0.5, s * 0.04)
    spin_a = 1.0 - done_k
    if spin_a > 0:
        try:
            from AIO_QC.widgets.design import tokens
            ink = QColor(tokens()["text"])
        except Exception as _exc:
            _ignored(_exc)
            ink = QColor("#1B1F24")
        track = QColor(ink)
        track.setAlphaF(0.18 * spin_a)
        p.setPen(QPen(track, pen_w))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, r, r)
        span = 110.0                                            # the bright arc, in degrees
        head = -angle                                           # turning clockwise
        grad = QConicalGradient(c, head)
        lead, tail = QColor(ink), QColor(ink)
        lead.setAlphaF(0.95 * spin_a)
        tail.setAlphaF(0.0)
        grad.setColorAt(0.0, lead)
        grad.setColorAt(span / 360.0, tail)
        grad.setColorAt(1.0, tail)
        p.setPen(QPen(grad, pen_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        arc = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
        p.drawArc(arc, int(head * 16), int(span * 16))
    if done_k > 0:
        g = QColor(GREEN)
        g.setAlphaF(done_k)
        p.setPen(QPen(g, pen_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, r, r)
    if tick_k > 0:
        path = QPainterPath()
        pts = [(-0.42, 0.02), (-0.12, 0.32), (0.44, -0.30)]          # relative to the ring radius
        path.moveTo(c.x() + pts[0][0] * r, c.y() + pts[0][1] * r)
        for x, y in pts[1:]:
            path.lineTo(c.x() + x * r, c.y() + y * r)
        part = QPainterPath()
        n = 24
        steps = max(1, int(round(n * tick_k)))
        part.moveTo(path.pointAtPercent(0.0))
        for i in range(1, steps + 1):
            part.lineTo(path.pointAtPercent(min(1.0, tick_k * i / steps)))
        p.setPen(QPen(GREEN, pen_w * 1.05, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(part)


class LoadingRing(QWidget):
    """The loading animation: spin() while work runs, done() turns it into a green tick (then it stays a tick)."""

    def __init__(self, size: int = 16, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.mode = "spin"
        self._t_done = None
        self._timer = QTimer(self, timeout=self.update)                     # every display frame, only while it moves
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        try:
            from AIO_QC.widgets.mini_session import refresh_hz
            self._timer.setInterval(max(2, int(1000 / refresh_hz(self))))
        except Exception as _exc:
            _ignored(_exc)
            self._timer.setInterval(8)

    def spin(self):
        self.mode, self._t_done = "spin", None
        self._run()

    def done(self):
        if self.mode != "done":
            self.mode, self._t_done = "done", time.monotonic()
        self._run()

    def _run(self):
        if self.isVisible() and not self._timer.isActive():
            self._timer.start()
        self.update()

    def finished_drawing(self) -> bool:
        return self.mode == "done" and self._t_done is not None and \
            time.monotonic() - self._t_done > MORPH_S + TICK_S + 0.1

    def showEvent(self, e):
        super().showEvent(e)
        self._run()

    def hideEvent(self, e):
        super().hideEvent(e)
        self._timer.stop()

    def paintEvent(self, _e):
        now = time.monotonic()
        angle = (now % TURN_S) / TURN_S * 360.0
        dk = tk = 0.0
        if self.mode == "done" and self._t_done is not None:
            el = now - self._t_done
            dk = _ease(_clamp(el / MORPH_S))
            tk = _ease(_clamp((el - MORPH_S * 0.6) / TICK_S))
            if self.finished_drawing():
                self._timer.stop()
        p = QPainter(self)
        paint_ring(p, QRectF(self.rect()), angle, dk, tk, wobble=now * 3.1)
        p.end()


# ---------------------------------------------------------------------------------------------------- indicator
DONE_HOLD_MS = 1800      # the green tick stays this long after the last job ended


class TaskIndicator(QWidget):
    """Status bar: loading ring + the latest job's text (hidden when nothing runs). When the last job ends the ring
    becomes a green tick for a moment. Click: the Background Tasks window.

    `active_changed(bool)` says when it appears / goes, for a window that shares its place with something else (the
    Frame Extractor's left slot: the jobs replace the output-folder line while they run)."""
    active_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("taskIndicator")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 6, 0)
        lay.setSpacing(6)
        self.ring = LoadingRing(16)
        self.label = QLabel("")
        from PyQt6.QtWidgets import QSizePolicy
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.ring)
        lay.addWidget(self.label, 1)
        self.setFixedWidth(self.WIDTH)            # one fixed slot: changing job text never moves its neighbours
        self._full = ""
        self._was_running = False
        self._hold = QTimer(self, singleShot=True, interval=DONE_HOLD_MS, timeout=self._end_hold)
        self.c = center()
        self.c.changed.connect(self.refresh)
        self._active = False
        self.setVisible(False)
        self.refresh()

    def active(self) -> bool:
        """Showing a job (or its "Done" moment) — independent of a status message covering it."""
        return self._active

    def _show(self, on: bool):
        if self.isHidden() == on:
            self.setVisible(on)
        if on != self._active:
            self._active = on
            self.active_changed.emit(on)

    WIDTH = 260

    def text(self) -> str:
        return self._full

    def _set_text(self, t: str):
        self._full = t
        self._elide()

    def _elide(self):
        w = max(0, self.label.width() or (self.WIDTH - 34))
        self.label.setText(self.label.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, w))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def _front(self) -> bool:
        w = self.window()
        return w is not None and w.isVisible()

    def refresh(self):
        # isHidden(), not isVisible(): the indicator is in every mode's status bar, and a window in the background
        # is not visible — a job that ended meanwhile must still hide it there.
        ts = self.c.tasks
        if ts:
            self._hold.stop()
            self._was_running = True
            if not self._active or self.isHidden():
                self._show(True)
            self.ring.spin()
            last = ts[-1]
            more = f"  (+{len(ts) - 1} more)" if len(ts) > 1 else ""
            txt = last.label()
            self._set_text(f"{txt}{more}" if not more else f"{txt[:48]}{'…' if len(txt) > 48 else ''}{more}")
            self.setToolTip("Background tasks — click for the Background Tasks window:\n"
                            + "\n".join(f"• {t.label()}" for t in ts))
            return
        if self._was_running and self._front() and self._active:
            self._was_running = False
            last = self.c.history[-1] if self.c.history else None
            if last is not None and last.cancelled:
                self._set_text(f"Cancelled · {last.title}")
            else:
                self._set_text(f"Done · {last.title}" if last is not None else "Done")
            self.setToolTip("Every background task has finished — click for the list")
            self.ring.done()
            self._hold.start()
            return
        if self._hold.isActive():
            return
        self._was_running = False
        if self._active or not self.isHidden():
            self._show(False)

    def _end_hold(self):
        if not self.c.tasks:
            self._show(False)

    def showEvent(self, e):
        super().showEvent(e)
        if not self.c.tasks and not self._hold.isActive():
            self._show(False)
            return
        self.refresh()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            show_viewer(self.window())
        super().mousePressEvent(e)


def install(window):
    """Add the one indicator to a mode window's status bar (once)."""
    if getattr(window, "_task_indicator", None) is not None:
        return window._task_indicator
    ind = TaskIndicator(window)
    # first of the permanent widgets: its changing text only moves the message area, never the steady items to its
    # right (review mark, position, mode)
    sb = window.statusBar()
    from AIO_QC.widgets.status_fields import PaintStatusBar
    slot = getattr(window, "task_slot", None)
    if callable(slot):          # the window has a place of its own for it (Frame Extractor: the left slot)
        ind.setMinimumWidth(0)
        ind.setMaximumWidth(16777215)
        slot(ind)
    elif isinstance(sb, PaintStatusBar):
        sb.insertPermanentWidget(0, ind)
    else:                       # a bar with its own (non-permanent) items first: Qt only accepts it after them
        sb.addPermanentWidget(ind)
    window._task_indicator = ind
    return ind


# ---------------------------------------------------------------------------------------------------- viewer
def _dur(sec: float) -> str:
    if sec < 1:
        return f"{sec * 1000:.0f} ms"
    if sec < 60:
        return f"{sec:.1f} s"
    m, s_ = divmod(int(sec), 60)
    return f"{m} min {s_:02d} s" if m < 60 else f"{m // 60} h {m % 60:02d} min"


class _StatusCell(QWidget):
    def __init__(self, state: str):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 0, 0)
        if state == "running":
            w = LoadingRing(16)
        else:
            from AIO_QC.utils import IconLabel
            w = IconLabel("status-ok" if state == "done" else "status-warning", 16)
        lay.addWidget(w)
        lay.addStretch(1)


class TaskViewer(QDialog):
    """Every background job of this session: running ones first (live progress, Cancel), then the finished ones."""
    COLS = ["", "Task", "Progress", "Details", "Started", "Took", ""]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Background Tasks")
        self.setObjectName("taskViewer")
        from AIO_QC.utils.qt_utils import fit_size
        fit_size(self, "small")
        self.c = center()
        lay = QVBoxLayout(self)
        self.summary = QLabel("")
        lay.addWidget(self.summary)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        h = self.table.horizontalHeader()
        for i, mode in enumerate((QHeaderView.ResizeMode.Fixed, QHeaderView.ResizeMode.Stretch,
                                  QHeaderView.ResizeMode.ResizeToContents, QHeaderView.ResizeMode.Stretch,
                                  QHeaderView.ResizeMode.ResizeToContents, QHeaderView.ResizeMode.ResizeToContents,
                                  QHeaderView.ResizeMode.ResizeToContents)):
            h.setSectionResizeMode(i, mode)
        self.table.setColumnWidth(0, 30)
        lay.addWidget(self.table, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.clear_btn = bb.addButton("Clear finished", QDialogButtonBox.ButtonRole.ActionRole)
        self.clear_btn.clicked.connect(self.c.clear_history)
        bb.rejected.connect(self.close)
        lay.addWidget(bb)
        self._sig = None
        self.c.changed.connect(self.refresh)
        self._tick = QTimer(self, interval=1000, timeout=self._update_times)
        self.refresh()

    def showEvent(self, e):
        super().showEvent(e)
        self._tick.start()
        self.refresh()

    def hideEvent(self, e):
        super().hideEvent(e)
        self._tick.stop()

    def rows(self) -> list[Task]:
        return list(reversed(self.c.tasks)) + list(reversed(self.c.history))

    def refresh(self):
        rows = self.rows()
        run, fin = len(self.c.tasks), len(self.c.history)
        self.summary.setText(f"<b>{run}</b> running · <b>{fin}</b> finished this session" if run or fin else
                             "No background tasks yet in this session.")
        self.clear_btn.setEnabled(bool(fin))
        sig = tuple((id(t), t.state) for t in rows)
        if sig != self._sig:                           # rows came / went / changed state: rebuild
            self._sig = sig
            self.table.setRowCount(len(rows))
            for i, t in enumerate(rows):
                self.table.setCellWidget(i, 0, _StatusCell(t.state))
                for col in range(1, 6):
                    self.table.setItem(i, col, QTableWidgetItem(""))
                if t.state == "running" and t.cancel_fn is not None:
                    b = QPushButton("Cancel")
                    b.clicked.connect(lambda _c=False, t=t: t.cancel())
                    self.table.setCellWidget(i, 6, b)
                else:
                    self.table.removeCellWidget(i, 6)
                    self.table.setItem(i, 6, QTableWidgetItem(""))
        for i, t in enumerate(rows):
            vals = [t.title,
                    (f"{t.fraction * 100:.0f} %" if t.fraction is not None else "working…") if t.state == "running"
                    else {"done": "done", "cancelled": "cancelled"}[t.state],
                    t.text, time.strftime("%H:%M:%S", time.localtime(t.started)), _dur(t.duration())]
            for col, v in enumerate(vals, start=1):
                it = self.table.item(i, col)
                if it is not None and it.text() != v:
                    it.setText(v)
                    if col in (1, 3):
                        it.setToolTip(v)

    def _update_times(self):
        for i, t in enumerate(self.rows()):
            if t.state == "running":
                it = self.table.item(i, 5)
                if it is not None:
                    it.setText(_dur(t.duration()))


def show_viewer(parent=None):
    """The one Background Tasks window (non-modal)."""
    app = QApplication.instance()
    v = getattr(app, "_task_viewer", None)
    try:
        if v is not None:
            v.isVisible()
    except RuntimeError:                                    # deleted with its parent window
        v = None
    if v is None:
        v = TaskViewer(parent)
        app._task_viewer = v
    v.show()
    v.raise_()
    v.activateWindow()
    return v
