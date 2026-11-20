# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The one way long jobs look and behave, in every window:

    while it runs   a progress bar · what it is doing · “about 2 min left” · Stop — and an entry in Background Tasks
                    (the status-bar ring), so it can be followed and stopped from any window
    when it ends    one line saying what happened (✓ done · ⏹ stopped · ⚠ failed, in plain words) and the follow-ups
                    that fit: Undo, Open folder, Open, …

    bar = JobBar(self, "Convert to YOLO")
    bar.start("Converting", total=n, stop=self.cancel.set)       # stop: what Stop does (None: no Stop button)
    bar.step(i, n, "image 12 / 400")                             # from the UI thread
    bar.finish("1,204 images converted", undo=fn, open_path=out)  # or bar.fail(exc) / bar.stopped("…")

eta() is shared with the scan windows: a time-left estimate that waits for a few seconds of evidence and is not
shown for jobs that will be over in a moment.
"""
from __future__ import annotations

import time

from PyQt6.QtCore import QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from AIO_QC.utils import newButton, style as apply_style


def eta(started: float, done: int, total: int, now: float | None = None) -> str:
    """'about 3 min left' — '' while there is too little to go on (under 3 s or 2 %) or almost nothing is left."""
    now = time.monotonic() if now is None else now
    took = now - started
    if total <= 0 or done <= 0 or took < 3 or done / total < 0.02 or done >= total:
        return ""
    left = took * (total - done) / done
    if left < 5:
        return "a few seconds left"
    if left < 60:
        return f"about {int(round(left / 5.0) * 5)} s left"
    if left < 3600:
        return f"about {int(round(left / 60))} min left"
    return f"about {left / 3600:.1f} h left"


class JobBar(QWidget):
    stopRequested = pyqtSignal()

    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.title = title
        self.t0 = 0.0
        self._task = None
        self._stop = None
        self._undo = None
        self._open = None
        self._extra = []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        self.text = QLabel("")
        self.text.setWordWrap(True)
        apply_style(self.text, "color:@muted;")
        self.stop_btn = newButton("Stop", icon="stop", slot=self.request_stop,
                                  tip="Stop after the item in progress — what is done so far is kept")
        self.undo_btn = newButton("Undo", icon="undo", slot=self._do_undo)
        self.open_btn = newButton("Open folder", icon="open", slot=self._do_open)
        for b in (self.stop_btn, self.undo_btn, self.open_btn):
            b.setVisible(False)
        lay.addWidget(self.progress, 2)
        lay.addWidget(self.text, 3)
        self._extra_host = QHBoxLayout()
        lay.addLayout(self._extra_host)
        lay.addWidget(self.undo_btn)
        lay.addWidget(self.open_btn)
        lay.addWidget(self.stop_btn)

    # ------------------------------------------------------------------ running
    @property
    def running(self) -> bool:
        return self._task is not None

    def start(self, text: str = "Working", total: int = 0, stop=None):
        from AIO_QC.widgets import tasks
        self._clear_result()
        self.t0 = time.monotonic()
        self._stop = stop
        self.progress.setVisible(True)
        self.progress.setRange(0, max(0, int(total)))
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m" if total else "")
        self.stop_btn.setVisible(stop is not None)
        self.stop_btn.setEnabled(True)
        self.text.setText(text + " …")
        if self._task is not None:
            self._task.finish()
        self._task = tasks.start(f"{self.title} · {text.lower()}", cancel=self.request_stop if stop else None)

    def step(self, done: int, total: int, text: str = ""):
        if total > 0:
            if self.progress.maximum() != total:
                self.progress.setRange(0, total)
                self.progress.setFormat("%v / %m")
            self.progress.setValue(min(done, total))
        else:
            self.progress.setRange(0, 0)
        left = eta(self.t0, done, total)
        self.text.setText(" · ".join(x for x in (text, left) if x))
        if self._task is not None:
            self._task.update(done / total if total else None, text or None)

    def request_stop(self):
        if self._stop is None or not self.stop_btn.isEnabled():
            return
        self.stop_btn.setEnabled(False)
        self.text.setText("Stopping after the item in progress …")
        self._stop()
        self.stopRequested.emit()

    # ------------------------------------------------------------------ ended
    def _end(self):
        if self._task is not None:
            self._task.finish()
            self._task = None
        self.progress.setVisible(False)
        self.stop_btn.setVisible(False)

    def _took(self) -> str:
        s = time.monotonic() - self.t0
        return f"{s:.1f} s" if s < 60 else f"{int(s // 60)} min {int(s % 60)} s"

    def finish(self, text: str, undo=None, open_path=None, extra=()):
        """✓ text · Undo (when `undo` is given) · Open folder (when `open_path` is) · extra [(label, slot)]."""
        self._end()
        self.text.setText(f"✓ {text} — in {self._took()}")
        apply_style(self.text, "color:@ok;")
        self._show_result(undo, open_path, extra)

    def stopped(self, text: str = "Stopped — what was done so far is kept", undo=None, open_path=None):
        self._end()
        self.text.setText(f"⏹ {text}")
        apply_style(self.text, "color:@warn;")
        self._show_result(undo, open_path, ())

    def fail(self, err):
        from .errors import message
        self._end()
        self.text.setText(f"⚠ {message(err)}")
        apply_style(self.text, "color:@error;")

    def _show_result(self, undo, open_path, extra):
        self._undo, self._open = undo, open_path
        self.undo_btn.setVisible(undo is not None)
        self.undo_btn.setEnabled(True)
        self.open_btn.setVisible(open_path is not None)
        for label, slot in extra:
            b = newButton(label, slot=slot)
            self._extra_host.addWidget(b)
            self._extra.append(b)

    def _clear_result(self):
        self._undo = self._open = None
        self.undo_btn.setVisible(False)
        self.open_btn.setVisible(False)
        for b in self._extra:
            b.setParent(None)
            b.deleteLater()
        self._extra = []
        apply_style(self.text, "color:@muted;")

    def _do_undo(self):
        if self._undo is None:
            return
        fn, self._undo = self._undo, None
        self.undo_btn.setEnabled(False)
        msg = fn()
        self.text.setText(f"↶ {msg or 'Undone'}")
        apply_style(self.text, "color:@muted;")
        self.undo_btn.setVisible(False)

    def _do_open(self):
        if self._open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._open)))

    def close_job(self):
        """The window closes: the Background Tasks entry ends too."""
        if self._task is not None:
            self._task.finish()
            self._task = None
