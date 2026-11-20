# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Python's cycle collector, run on the GUI thread only.

Automatic collection runs on whichever thread happens to allocate when a threshold is crossed. With worker threads
(scans, conversions, readers) that can be a worker — and if the cycle it frees holds a widget that Python owns, the
widget is destroyed off the GUI thread. Qt then crashes later, somewhere unrelated (a deferred delete, a paint).
So automatic collection is switched off and the same generational thresholds are checked here every 100 ms on the GUI
thread (the approach pyqtgraph takes). Reference counting — which frees almost everything — is unaffected.

    install(app)     once (the shell does it); idempotent
    pause(True/False)  hold collection while something must not stutter (the zoom glide)
"""
from __future__ import annotations

import gc

from PyQt6.QtCore import QObject, QTimer

_INSTANCE: GuiThreadGC | None = None


class GuiThreadGC(QObject):
    def __init__(self, parent=None, interval_ms: int = 100):
        super().__init__(parent)
        self.threshold = gc.get_threshold()
        self.paused = False
        gc.disable()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check)
        self.timer.start(interval_ms)

    def check(self):
        if self.paused:
            return
        n0, n1, n2 = gc.get_count()
        t0, t1, t2 = self.threshold
        if n0 > t0:
            gc.collect(0)
            if n1 > t1:
                gc.collect(1)
                if n2 > t2:
                    gc.collect(2)

    def stop(self):
        self.timer.stop()
        gc.enable()


def install(app) -> GuiThreadGC:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = GuiThreadGC(app)
        _INSTANCE.destroyed.connect(lambda *_a: (gc.enable(), _forget()))
    return _INSTANCE


def _forget():
    global _INSTANCE
    _INSTANCE = None


def installed() -> bool:
    return _INSTANCE is not None


def pause(on: bool) -> bool:
    """Hold / resume collection; False when the collector is not installed (the caller handles gc itself)."""
    if _INSTANCE is None:
        return False
    _INSTANCE.paused = bool(on)
    return True
