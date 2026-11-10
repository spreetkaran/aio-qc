# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
One background-task indicator for the whole app.

Long jobs from any mode report here - Frame Extractor batch extraction, marker extraction, timeline analysis,
clip / annotated-video exports, Testing runs / test-set builds / reports, Image QC scans and Data Prep runs, Data
Studio exports. Every mode window shows the same indicator in its status bar (installed by the shell): a spinner,
how many jobs run, the latest one's progress; click for the list, each with Cancel when the job can be stopped.

    t = tasks.start("Batch extraction · 3 videos", cancel=fn)    # UI thread only
    t.update(0.42, "frame 1,204 / 2,880")                         # fraction 0..1 or None (busy)
    t.finish()                                                    # idempotent; also when the owner is deleted
"""
from __future__ import annotations

import time

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMenu, QToolButton


class Task:
    def __init__(self, center, title, cancel=None):
        self.center, self.title, self.cancel_fn = center, title, cancel
        self.fraction, self.text = None, ""
        self.t0 = time.monotonic()
        self.done = False

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
            self.center._remove(self)

    def cancel(self):
        if self.cancel_fn is not None:
            try:
                self.cancel_fn()
            except Exception:
                pass

    def label(self):
        pct = f" {self.fraction * 100:.0f}%" if self.fraction is not None else ""
        return f"{self.title}{pct}" + (f" — {self.text}" if self.text else "")


class TaskCenter(QObject):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tasks: list[Task] = []
        self._timer = QTimer(self, singleShot=True, interval=100, timeout=self.changed.emit)   # <= 10 updates/s

    def start(self, title, cancel=None) -> Task:
        t = Task(self, title, cancel)
        self.tasks.append(t)
        self.changed.emit()
        return t

    def _touch(self):
        if not self._timer.isActive():
            self._timer.start()

    def _remove(self, t):
        if t in self.tasks:
            self.tasks.remove(t)
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


def start(title, cancel=None):
    c = center()
    return c.start(title, cancel) if c is not None else _Null()


SPIN = "◐◓◑◒"


class TaskIndicator(QToolButton):
    """Status-bar button: hidden when nothing runs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoRaise(True)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setObjectName("taskIndicator")
        self._i = 0
        self._spin = QTimer(self, interval=250, timeout=self._tick)
        self.c = center()
        self.c.changed.connect(self.refresh)
        self.clicked.connect(self._menu)
        self.setVisible(False)
        self.refresh()

    def _tick(self):
        self._i = (self._i + 1) % len(SPIN)
        self.refresh()

    def refresh(self):
        ts = self.c.tasks
        on = bool(ts)
        if on != self.isVisible():
            self.setVisible(on)
        if not on:
            self._spin.stop()
            return
        if not self._spin.isActive():
            self._spin.start()
        last = ts[-1]
        more = f"  (+{len(ts) - 1} more)" if len(ts) > 1 else ""
        txt = last.label()
        self.setText(f"{SPIN[self._i]} {txt[:60]}{'…' if len(txt) > 60 else ''}{more}")
        self.setToolTip("Background tasks — click for the list:\n" + "\n".join(f"• {t.label()}" for t in ts))

    def _menu(self):
        m = QMenu(self)
        for t in list(self.c.tasks):
            sub = m.addMenu(t.label()[:90])
            a = sub.addAction("Cancel")
            a.setEnabled(t.cancel_fn is not None)
            a.triggered.connect(lambda _c=False, t=t: t.cancel())
        if not self.c.tasks:
            m.addAction("Nothing running").setEnabled(False)
        m.exec(self.mapToGlobal(self.rect().topLeft()))


def install(window):
    """Add the one indicator to a mode window's status bar (once)."""
    if getattr(window, "_task_indicator", None) is not None:
        return window._task_indicator
    ind = TaskIndicator(window)
    window.statusBar().addPermanentWidget(ind)
    window._task_indicator = ind
    return ind
