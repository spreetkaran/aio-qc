# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Background scanner for outside changes (see core.watch). Emits snapshots; the GUI thread diffs and applies them."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from AIO_QC.widgets.lifecycle import RetainedThread

from AIO_QC.core.watch import snapshot
from AIO_QC.core.diag import ignored as _ignored


class DatasetWatcher(RetainedThread, QThread):
    snapshotReady = pyqtSignal(object, float)        # {path: sig}, seconds the scan took

    MIN_INTERVAL, MAX_INTERVAL = 3.0, 60.0

    def __init__(self, root: Path, exts: set[str], parent=None):
        super().__init__(parent)
        self.root, self.exts = Path(root), set(exts)
        self._wake = threading.Event()
        self._stop = False
        self.finished.connect(self.deleteLater)          # a stopped watcher goes away once its thread has ended

    def run(self):
        while not self._stop:
            t0 = time.monotonic()
            try:
                snap = snapshot(self.root, self.exts, lambda: self._stop)
            except Exception as _exc:
                _ignored(_exc)
                snap = None
            took = time.monotonic() - t0
            if self._stop:
                break
            if snap is not None:
                self.snapshotReady.emit(snap, took)
            # big datasets are scanned less often: ~10x the scan time, 3-60 s; poke() wakes us early
            self._wake.wait(min(self.MAX_INTERVAL, max(self.MIN_INTERVAL, took * 10)))
            self._wake.clear()

    def poke(self):
        """Scan now (e.g. the user just came back to AIO QC from another program)."""
        self._wake.set()

    def stop(self, wait_ms: int = 1000):
        """Ask it to stop: the scan gives up within a few thousand files, so this returns at once (it no longer
        waits for a full scan of a big dataset); a thread still busy after wait_ms ends by itself and is deleted."""
        self._stop = True
        self._wake.set()
        self.wait(wait_ms)
