# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
import threading
from PyQt6.QtCore import QThread, pyqtSignal

from AIO_QC.tools.frame_extractor.core.tracker import RoiTracker


class TrackWorker(QThread):
    """ROI tracking off the GUI thread. init() restarts on a frame; update()
    requests are coalesced (latest wins), so playback never waits on the tracker."""
    result = pyqtSignal(int, object)   # frame idx, (x1, y1, x2, y2) or None when lost
    error = pyqtSignal(str)

    def __init__(self, kind="auto"):
        super().__init__()
        self.kind = kind
        self._cond = threading.Condition()
        self._init = None
        self._update = None
        self._quit = False

    def init(self, idx, img, roi, scale=1.0):
        """img may be the full-res frame (scale 1) or the display-size playback frame (scale < 1);
        roi and results are always in full-resolution pixels."""
        with self._cond:
            self._init, self._update = (idx, img, tuple(roi), scale), None
            self._cond.notify()

    def update(self, idx, img, scale=1.0):
        with self._cond:
            self._update = (idx, img, scale)
            self._cond.notify()

    WORK_W = 960

    def _work(self, img, scale):
        """Resize any input (full-res or display-size) to one fixed working width, so the tracker
        always sees the same geometry. Returns (image, full-res -> work factor)."""
        import cv2
        full_w = img.shape[1] / max(scale, 1e-6)
        work_w = min(self.WORK_W, full_w)
        k = work_w / full_w
        tw, th = int(round(full_w * k)), int(round(img.shape[0] / max(scale, 1e-6) * k))
        if (img.shape[1], img.shape[0]) != (tw, th):
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA if tw < img.shape[1] else cv2.INTER_LINEAR)
        return img, k

    def stop(self):
        with self._cond:
            self._quit = True
            self._cond.notify()
        self.wait()

    def run(self):
        try:
            tracker = RoiTracker(self.kind)
        except Exception as e:
            self.error.emit(str(e))
            return
        while True:
            with self._cond:
                while self._init is None and self._update is None and not self._quit:
                    self._cond.wait()
                if self._quit:
                    return
                init, self._init = self._init, None
                upd = None if init else self._update
                if not init:
                    self._update = None
            try:
                if init:
                    img, k = self._work(init[1], init[3])
                    tracker.init(img, tuple(int(v * k) for v in init[2]))
                else:
                    img, k = self._work(upd[1], upd[2])
                    r = tracker.update(img)
                    self.result.emit(upd[0], tuple(int(round(v / k)) for v in r) if r else None)
            except Exception as e:
                self.error.emit(str(e))
                return
