# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
import colorsys
import hashlib
from collections import OrderedDict

from PyQt6.QtGui import QColor


def generate_label_color(label: str) -> QColor:
    """Stable, visually distinct colour per label (md5 -> HSV)."""
    idx = int(hashlib.md5(label.encode()).hexdigest(), 16)
    hue = (idx % 360) / 360.0
    sat = min(0.70 + ((idx >> 8) % 20) / 100.0, 0.90)
    val = min(0.80 + ((idx >> 16) % 15) / 100.0, 0.95)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


def update_label_colors(label: str, label_dict: dict) -> None:
    """A new label gets the most distinct palette colour no other label uses (utils.colors)."""
    if label not in label_dict:
        from .colors import assign
        assign(label, label_dict)


def is_close(a: float, b: float, rel_tol: float = 0.01, abs_tol: float = 0.001) -> bool:
    return abs(a - b) < max(rel_tol * max(abs(a), abs(b)), abs_tol)


class LRUCache:
    """
    Main-thread-only LRU with an optional byte budget (workers hand results over
    via signals, so no locking is needed). 15 x 12MP pixmaps would be ~720 MB,
    hence the byte cap.
    """

    def __init__(self, capacity=20, max_bytes: int | None = None):
        self.cache = OrderedDict()
        self.costs = {}
        self.capacity = capacity
        self.max_bytes = max_bytes
        self.total = 0

    def __contains__(self, key):
        return key in self.cache

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value, cost: int = 0):
        self.pop(key)
        self.cache[key] = value
        self.costs[key] = cost
        self.total += cost
        while len(self.cache) > 1 and (len(self.cache) > self.capacity or
                                       (self.max_bytes and self.total > self.max_bytes)):
            old, _ = self.cache.popitem(last=False)
            self.total -= self.costs.pop(old, 0)

    def pop(self, key):
        if key in self.cache:
            del self.cache[key]
            self.total -= self.costs.pop(key, 0)

    def clear(self):
        self.cache.clear()
        self.costs.clear()
        self.total = 0
