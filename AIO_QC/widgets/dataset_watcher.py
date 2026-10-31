# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""Background scanner for outside changes (see core.watch). Emits snapshots; the GUI thread diffs and applies them."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from AIO_QC.core.watch import snapshot


class DatasetWatcher(QThread):
    snapshotReady = pyqtSignal(object, float)        # {path: sig}, seconds the scan took

    MIN_INTERVAL, MAX_INTERVAL = 3.0, 60.0

    def __init__(self, root: Path, exts: set[str], parent=None):
        super().__init__(parent)
        self.root, self.exts = Path(root), set(exts)
        self._wake = threading.Event()
        self._stop = False

    def run(self):
        while not self._stop:
            t0 = time.monotonic()
            try:
                snap = snapshot(self.root, self.exts)
            except Exception:
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

    def stop(self):
        self._stop = True
        self._wake.set()
        self.wait(5000)
