# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""ROI tracking (OpenCV CSRT > KCF > MIL), run on a downscaled copy so it stays
fast on 4K. CSRT/KCF need opencv-contrib-python; MIL ships with opencv-python."""
import cv2

_FACTORIES = {
    "csrt": ("TrackerCSRT_create", "legacy.TrackerCSRT_create"),
    "kcf": ("TrackerKCF_create", "legacy.TrackerKCF_create"),
    "mil": ("TrackerMIL_create",),
}


def _factory(kind):
    for name in _FACTORIES[kind]:
        obj = cv2
        try:
            for part in name.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    return None


def available():
    return [k for k in _FACTORIES if _factory(k)]


class RoiTracker:
    def __init__(self, kind="auto", work_side=960, max_misses=5):
        kinds = available()
        if not kinds:
            raise RuntimeError("No OpenCV tracker available")
        self.kind = kinds[0] if kind == "auto" or kind not in kinds else kind
        self.work_side = work_side
        self.max_misses = max_misses
        self._t = None
        self._scale = 1.0
        self.roi = None
        self.misses = 0

    def _small(self, bgr):
        h, w = bgr.shape[:2]
        self._scale = min(1.0, self.work_side / max(h, w))
        if self._scale < 1.0:
            bgr = cv2.resize(bgr, (int(w * self._scale), int(h * self._scale)), interpolation=cv2.INTER_AREA)
        return bgr

    def init(self, bgr, roi):
        small = self._small(bgr)
        x1, y1, x2, y2 = roi
        s = self._scale
        self._t = _factory(self.kind)()
        self._t.init(small, (int(x1 * s), int(y1 * s), max(2, int((x2 - x1) * s)), max(2, int((y2 - y1) * s))))
        self.roi = tuple(int(v) for v in roi)
        self.misses = 0

    def update(self, bgr):
        """New ROI (x1, y1, x2, y2), the last ROI while briefly unsure, or None once lost."""
        if self._t is None:
            return None
        h, w = bgr.shape[:2]
        ok, (x, y, bw, bh) = self._t.update(self._small(bgr))
        if not ok or bw < 1 or bh < 1:
            self.misses += 1
            return self.roi if self.misses <= self.max_misses else None
        self.misses = 0
        s = self._scale
        x1, y1 = max(0, int(x / s)), max(0, int(y / s))
        x2, y2 = min(w, int((x + bw) / s)), min(h, int((y + bh) / s))
        if x2 - x1 < 2 or y2 - y1 < 2:
            self.misses += 1
            return self.roi if self.misses <= self.max_misses else None
        self.roi = (x1, y1, x2, y2)
        return self.roi
