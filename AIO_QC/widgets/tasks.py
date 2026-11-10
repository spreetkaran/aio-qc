# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
One background-task indicator for the whole app.

Long jobs from any mode report here - Frame Extractor batch extraction, marker extraction, timeline analysis,
clip / annotated-video exports, Testing runs / test-set builds / reports, Image QC scans and Data Prep runs, Data
Studio exports. Every mode window shows the same indicator in its status bar (installed by the shell): a loading
ring, then the latest job's text and progress; when everything has finished the ring turns into a green tick for a
moment. Click it (or the Background Tasks toolbar button) for the Background Tasks window: every job of this session,
running and finished, with progress, timings and Cancel.

    t = tasks.start("Batch extraction · 3 videos", cancel=fn)    # UI thread only
    t.update(0.42, "frame 1,204 / 2,880")                         # fraction 0..1 or None (busy)
    t.finish()                                                    # idempotent; also when the owner is deleted
"""
from __future__ import annotations

import math
import time

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QConicalGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
                             QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)


class Task:
    def __init__(self, center, title, cancel=None):
        self.center, self.title, self.cancel_fn = center, title, cancel
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
            except Exception:
                pass

    def label(self):
        pct = f" {self.fraction * 100:.0f}%" if self.fraction is not None else ""
        return f"{self.title}{pct}" + (f" — {self.text}" if self.text else "")


class TaskCenter(QObject):
    changed = pyqtSignal()

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
        if t in self.tasks:
            self.tasks.remove(t)
            self.history.append(t)
            del self.history[:-self.HISTORY]
        self.changed.emit()

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
TURN_S = 1.1           # one turn of the gradient
MORPH_S = 0.30         # ring: blue gradient -> plain green
TICK_S = 0.32          # the tick draws itself


def _clamp(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _ease(x):
    return 1 - (1 - x) ** 3


def paint_ring(p: QPainter, rect: QRectF, angle: float, done_k: float = 0.0, tick_k: float = 0.0, wobble: float = 0.0):
    """One frame: a ring with a turning blue → cyan gradient and a soft wobbling glow; done_k morphs it into a plain
    green ring, tick_k draws the green tick (0..1 each)."""
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = min(rect.width(), rect.height())
    c = rect.center()
    pen_w = max(1.6, s * 0.15)
    r = s / 2 - pen_w / 2 - max(0.6, s * 0.06)
    spin_a = 1.0 - done_k
    if spin_a > 0:
        # glow: two translucent, slightly squashed loops turning the other way (the GIF's soft halo)
        for i, (col, k) in enumerate(((CYAN, 0.9), (QColor("#7fd8e8"), 0.7))):
            g = QColor(col)
            g.setAlphaF(0.28 * k * spin_a)
            p.save()
            p.translate(c)
            p.rotate(-angle * (0.6 + 0.3 * i) + 40 * i)
            dx = (0.5 + 0.5 * math.sin(wobble + i)) * s * 0.05
            p.setPen(QPen(g, pen_w * 1.25))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(dx, 0), r + s * 0.03, r - s * 0.02)
            p.restore()
        grad = QConicalGradient(c, -angle)
        b, cy = QColor(BLUE), QColor(CYAN)
        b.setAlphaF(spin_a)
        cy.setAlphaF(0.55 * spin_a)
        mid = QColor("#5aa9f0")
        mid.setAlphaF(spin_a)
        grad.setColorAt(0.0, b)
        grad.setColorAt(0.45, cy)
        grad.setColorAt(0.75, mid)
        grad.setColorAt(1.0, b)
        p.setPen(QPen(grad, pen_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, r, r)
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
        self._timer = QTimer(self, interval=33, timeout=self.update)        # ~30 fps, only while it moves

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
    becomes a green tick for a moment. Click: the Background Tasks window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("taskIndicator")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 0, 6, 0)
        lay.setSpacing(6)
        self.ring = LoadingRing(16)
        self.label = QLabel("")
        lay.addWidget(self.ring)
        lay.addWidget(self.label)
        self._was_running = False
        self._hold = QTimer(self, singleShot=True, interval=DONE_HOLD_MS, timeout=self._end_hold)
        self.c = center()
        self.c.changed.connect(self.refresh)
        self.setVisible(False)
        self.refresh()

    def text(self) -> str:
        return self.label.text()

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
            if self.isHidden():
                self.setVisible(True)
            self.ring.spin()
            last = ts[-1]
            more = f"  (+{len(ts) - 1} more)" if len(ts) > 1 else ""
            txt = last.label()
            self.label.setText(f"{txt[:60]}{'…' if len(txt) > 60 else ''}{more}")
            self.setToolTip("Background tasks — click for the Background Tasks window:\n"
                            + "\n".join(f"• {t.label()}" for t in ts))
            return
        if self._was_running and self._front() and not self.isHidden():
            self._was_running = False
            last = self.c.history[-1] if self.c.history else None
            if last is not None and last.cancelled:
                self.label.setText(f"Cancelled · {last.title[:50]}")
            else:
                self.label.setText(f"Done · {last.title[:50]}" if last is not None else "Done")
            self.setToolTip("Every background task has finished — click for the list")
            self.ring.done()
            self._hold.start()
            return
        if self._hold.isActive():
            return
        self._was_running = False
        if not self.isHidden():
            self.setVisible(False)

    def _end_hold(self):
        if not self.c.tasks:
            self.setVisible(False)

    def showEvent(self, e):
        super().showEvent(e)
        if not self.c.tasks and not self._hold.isActive():
            self.setVisible(False)
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
    window.statusBar().insertPermanentWidget(0, ind)
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
        self.resize(860, 420)
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
