# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Cheap per-frame metrics used for filtering and scene detection."""
import cv2
import numpy as np

_SHARP_MAX_SIDE = 640   # normalise resolution so thresholds mean the same at 720p and 4K


def _gray_small(bgr, max_side):
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr


def sharpness(bgr):
    """Variance of the Laplacian on a ~640px grayscale copy. Higher = sharper.
    Typical: < 50 very blurry, 100-300 soft, > 300 crisp (content dependent)."""
    g = _gray_small(bgr, _SHARP_MAX_SIDE)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def signature(bgr):
    """Tiny grayscale thumbnail used for near-duplicate comparison."""
    g = cv2.resize(_gray_small(bgr, 256), (64, 64), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (3, 3), 0).astype(np.int16)


def difference(sig_a, sig_b):
    """Mean absolute difference between signatures, in % (0 = identical)."""
    return float(np.mean(np.abs(sig_a - sig_b))) * 100.0 / 255.0


class SceneDetector:
    """HSV-histogram cut detector (Bhattacharyya distance between consecutive frames)."""

    def __init__(self, threshold=0.4, min_scene_len=10):
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self._prev = None
        self._since_cut = 0

    def update(self, bgr):
        small = cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        cut = False
        self._since_cut += 1
        if self._prev is not None and self._since_cut >= self.min_scene_len:
            d = cv2.compareHist(self._prev, hist, cv2.HISTCMP_BHATTACHARYYA)
            if d > self.threshold:
                cut = True
                self._since_cut = 0
        self._prev = hist
        return cut
