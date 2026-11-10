# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Duplicate detection for extracted frames (Qt-free).

Two separate questions:
  exact duplicate : the decoder produced the very same picture again (encoders repeat frames - common in CCTV
                    and screen recordings). Content hash of the pixels.
  near duplicate  : nothing that matters changed. Decided on an exposure-normalised 128 px greyscale copy:
                      - global change : mean difference over the whole picture
                      - local change  : the largest difference in any one of 16 x 9 cells
                    A frame is a near duplicate only if BOTH are below their thresholds. The local test is
                    what keeps a new car / plate in one corner of a 4K frame from being thrown away (it barely
                    moves the global average); exposure normalisation keeps auto-exposure flicker or a passing
                    cloud from counting as change.
Frames are compared with the last few saved frames (window), not just the previous one, so A -> B -> A
(a camera panning back) is caught too.
"""
import hashlib

import cv2
import numpy as np

PRESETS = {      # name: (global %, local %) - "exact" is handled separately
    "conservative": (1.0, 4.0),
    "balanced": (2.0, 8.0),
    "aggressive": (4.0, 15.0),
}
PRESET_HELP = {
    "exact": "Only bit-identical frames (repeated by the encoder). Never drops a frame that differs at all.",
    "conservative": "Near-identical only: sensor noise / compression changes. Any visible change keeps the frame.",
    "balanced": "Skips frames where nothing noticeable changed anywhere. Small objects (plates) still count.",
    "aggressive": "Skips frames with only minor changes. Small slow movements may be dropped.",
}
GRID = (16, 9)


class Signature:
    __slots__ = ("exact", "norm", "small")

    def __init__(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        h, w = g.shape[:2]
        self.exact = hashlib.blake2b(np.ascontiguousarray(img[::2, ::2]).tobytes(), digest_size=16).digest()
        tw, th = 128, max(GRID[1] * 2, int(round(128 * h / w / GRID[1])) * GRID[1])
        s = cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
        m, sd = float(s.mean()), float(s.std())
        # exposure normalisation: same mean / contrast for every frame
        self.norm = (s - m) / max(sd, 4.0) * 50.0 + 128.0
        self.small = None


def compare(a, b):
    """(exact, global %, local %) between two signatures."""
    if a.exact == b.exact:
        return True, 0.0, 0.0
    na, nb = a.norm, b.norm
    if na.shape != nb.shape:
        nb = cv2.resize(nb, (na.shape[1], na.shape[0]), interpolation=cv2.INTER_AREA)
    d = np.abs(na - nb)
    gx, gy = GRID
    h, w = d.shape
    cells = d[: h // gy * gy, : w // gx * gx].reshape(gy, h // gy, gx, w // gx).mean(axis=(1, 3))
    return False, float(d.mean()) * 100 / 255, float(cells.max()) * 100 / 255


class DedupFilter:
    """mode 'exact' | 'near'. is_duplicate(sig) -> (bool, reason)."""

    def __init__(self, mode="near", global_pct=2.0, local_pct=8.0, window=5):
        self.mode, self.g, self.l, self.window = mode, float(global_pct), float(local_pct), max(1, int(window))
        self.recent = []

    def is_duplicate(self, sig):
        for ref in reversed(self.recent):
            exact, g, l = compare(sig, ref)
            if exact:
                return True, "exact"
            if self.mode == "near" and g < self.g and l < self.l:
                return True, f"near (global {g:.1f}% local {l:.1f}%)"
        return False, ""

    def add(self, sig):
        self.recent.append(sig)
        if len(self.recent) > self.window:
            self.recent.pop(0)

    def reset(self):
        self.recent.clear()
