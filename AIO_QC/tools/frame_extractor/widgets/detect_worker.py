# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
import threading
from PyQt6.QtCore import QThread, pyqtSignal


class DetectWorker(QThread):
    """Runs the configured models on the paused frame for the canvas overlay.
    Models load lazily in this thread; requests are coalesced (latest wins)."""
    result = pyqtSignal(int, object)      # frame idx, list[list[Detection]] per model
    error = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._cond = threading.Condition()
        self._req = None
        self._quit = False

    def request(self, idx, bgr):
        with self._cond:
            self._req = (idx, bgr)
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._quit = True
            self._cond.notify()
        self.wait()

    def run(self):
        try:
            from AIO_QC.tools.frame_extractor.core.detector import load_detectors
            c = self.cfg
            from AIO_QC.tools.frame_extractor.widgets.models_dialog import detector_kwargs
            dets = load_detectors(c["models"], min(c["conf"], c["unc_low"]), **detector_kwargs(c))
        except Exception as e:
            self.error.emit(str(e))
            return
        while True:
            with self._cond:
                while self._req is None and not self._quit:
                    self._cond.wait()
                if self._quit:
                    return
                (idx, bgr), self._req = self._req, None
            try:
                self.result.emit(idx, [d.detect(bgr) for d in dets])
            except Exception as e:
                self.error.emit(str(e))
