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


def cost_of(value) -> int:
    """Bytes held by a cached value: QPixmap / QImage pixels, numpy arrays, and tuples / lists / dicts of them."""
    try:
        from PyQt6.QtGui import QImage, QPixmap
        if isinstance(value, QPixmap):
            return 0 if value.isNull() else value.width() * value.height() * max(1, value.depth()) // 8
        if isinstance(value, QImage):
            return 0 if value.isNull() else int(value.sizeInBytes())
    except Exception:
        pass
    nb = getattr(value, "nbytes", None)
    if isinstance(nb, int):
        return nb
    if isinstance(value, (tuple, list)):
        return sum(cost_of(v) for v in value)
    if isinstance(value, dict):
        return sum(cost_of(v) for v in value.values())
    return 0


class _Budget:
    """One memory budget shared by every image cache of the app (main image cache, thumbnails, crops, galleries):
    each cache keeps its own limits, and together they never hold more than `limit` bytes — the least recently
    used entries of the biggest caches go first. limit: ~25 % of the computer's memory, at most 2 GB."""

    def __init__(self):
        import weakref
        self.caches = weakref.WeakSet()
        self._limit = None

    @property
    def limit(self) -> int:
        if self._limit is None:
            from .memory import cache_budget
            self._limit = cache_budget(2 * 2 ** 30, share=0.25, floor=256 * 2 ** 20)
        return self._limit

    def total(self) -> int:
        return sum(c.total for c in list(self.caches))

    def enforce(self, keep=None):
        """Evict LRU entries (biggest cache first) until all caches fit; `keep`'s newest entry always stays."""
        over = self.total() - self.limit
        while over > 0:
            cands = [c for c in list(self.caches) if len(c.cache) > 1 or (c is not keep and len(c.cache) > 0)]
            if not cands:
                return
            c = max(cands, key=lambda x: x.total)
            if c.total <= 0:
                return
            over -= c._evict_one()

    def trim(self, fraction: float = 0.5):
        """Low memory: every cache drops its least recently used `fraction` (by bytes)."""
        for c in list(self.caches):
            target = c.total * (1 - fraction)
            while c.cache and c.total > target and len(c.cache) > 1:
                c._evict_one()


BUDGET = _Budget()


class LRUCache:
    """
    Main-thread-only LRU with an optional byte budget (workers hand results over
    via signals, so no locking is needed). 15 x 12MP pixmaps would be ~720 MB,
    hence the byte cap. Costs not given are measured (pixmaps, images, arrays), and all caches share one
    app-wide budget (BUDGET) on top of their own limits.
    """

    def __init__(self, capacity=20, max_bytes: int | None = None, shared: bool = True):
        self.cache = OrderedDict()
        self.costs = {}
        self.capacity = capacity
        self.max_bytes = max_bytes
        self.total = 0
        if shared:
            BUDGET.caches.add(self)

    def __contains__(self, key):
        return key in self.cache

    def __len__(self):
        return len(self.cache)

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value, cost: int = 0):
        self.pop(key)
        cost = int(cost) if cost else cost_of(value)
        self.cache[key] = value
        self.costs[key] = cost
        self.total += cost
        while len(self.cache) > 1 and (len(self.cache) > self.capacity or
                                       (self.max_bytes and self.total > self.max_bytes)):
            self._evict_one()
        if self in BUDGET.caches:
            BUDGET.enforce(keep=self)

    def _evict_one(self) -> int:
        if not self.cache:
            return 0
        old, _ = self.cache.popitem(last=False)
        c = self.costs.pop(old, 0)
        self.total -= c
        return c

    def pop(self, key):
        if key in self.cache:
            del self.cache[key]
            self.total -= self.costs.pop(key, 0)

    def clear(self):
        self.cache.clear()
        self.costs.clear()
        self.total = 0
