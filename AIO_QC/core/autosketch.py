# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Auto-sketch: marks drawn on a screenshot to point at chosen shapes — an arrow, a ring, a box — without a hand.

The output is a list of Sketch elements (the same dicts Sketch on Image stores and draws: {"t", "pts", "c", "w"}),
in image pixels, so a screenshot draws them exactly like a hand-made sketch.

Arrows come from free space: for each chosen shape, candidate directions all round are scored on a coarse grid of
where the image is "busy" (every shape of the file — also the hidden ones — the other marks and arrows already
placed), and on staying inside the image. The arrow starts in the emptiest direction, never crosses another arrow
when it can avoid it, and its head stops just outside the mark. Sizes follow the image (they read the same on a
640 px and a 6000 px image). Pure Python + numpy, no Qt: fast (well under a millisecond per arrow) and testable.
"""
from __future__ import annotations

import math

import numpy as np

STYLES = [("arrow_ring", "Arrow + ring"), ("arrow", "Arrow"), ("arrow_box", "Arrow + box"), ("ring", "Ring only")]
STYLE_KEYS = {k for k, _t in STYLES}
COLORS = ["#ff3b30", "#ffcc00", "#00c7be", "#007aff", "#af52de", "#ffffff"]
REF_PX = 1280                  # marks look on any image as they would on a view this many pixels across
GRID = 96                      # occupancy grid resolution (longer side)
ANGLES = 24                    # candidate arrow directions


def shape_box(points, shape_type: str | None) -> tuple[float, float, float, float] | None:
    """Axis-aligned box of a shape in image pixels (circle: centre + rim point)."""
    if not points:
        return None
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if shape_type == "circle" and len(pts) >= 2:
        r = float(np.hypot(*(pts[1] - pts[0])))
        cx, cy = pts[0]
        return cx - r, cy - r, cx + r, cy + r
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def unit(W: float, H: float) -> float:
    """Image pixels per 'screen pixel' of a REF_PX-wide view (≥ 0.5)."""
    return max(0.5, max(W, H) / REF_PX)


class _Grid:
    """Busy-ness of the image on a coarse grid: filled boxes add 1, outlines (rings / arrows) add more."""

    def __init__(self, W, H):
        self.W, self.H = float(W), float(H)
        self.s = GRID / max(W, H, 1.0)
        self.g = np.zeros((max(1, math.ceil(H * self.s)), max(1, math.ceil(W * self.s))), np.float32)

    def _ij(self, x, y):
        return (min(self.g.shape[0] - 1, max(0, int(y * self.s))), min(self.g.shape[1] - 1, max(0, int(x * self.s))))

    def add_box(self, b, w=1.0):
        x1, y1, x2, y2 = b
        i1, j1 = self._ij(x1, y1)
        i2, j2 = self._ij(x2, y2)
        self.g[i1:i2 + 1, j1:j2 + 1] += w

    def add_segment(self, a, b, w=3.0, n=None):
        n = n or max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) * self.s * 2))
        for t in np.linspace(0, 1, n):
            i, j = self._ij(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            self.g[max(0, i - 1):i + 2, max(0, j - 1):j + 2] += w / 3

    def cost(self, a, b, skip_box=None):
        """Mean busy-ness along the segment a→b, ignoring the target's own box."""
        n = max(4, int(math.hypot(b[0] - a[0], b[1] - a[1]) * self.s * 2))
        total = 0.0
        for t in np.linspace(0.08, 1.0, n):
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            if skip_box is not None and skip_box[0] <= x <= skip_box[2] and skip_box[1] <= y <= skip_box[3]:
                continue
            i, j = self._ij(x, y)
            total += float(self.g[i, j])
        return total / n


def _ring_rect(b, pad):
    """An ellipse round the box (1.2 × its half sizes plus a gap: the corners may peek out, as when circling by hand)."""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    rx = (b[2] - b[0]) / 2 * 1.2 + pad
    ry = (b[3] - b[1]) / 2 * 1.2 + pad
    return cx - rx, cy - ry, cx + rx, cy + ry


def _edge_point(mark, cx, cy, dx, dy, ellipse):
    """Where a ray from the centre in direction (dx, dy) leaves the mark (ellipse or box)."""
    rx, ry = (mark[2] - mark[0]) / 2, (mark[3] - mark[1]) / 2
    if ellipse:
        k = 1.0 / math.sqrt((dx / max(rx, 1e-6)) ** 2 + (dy / max(ry, 1e-6)) ** 2)
    else:
        kx = rx / abs(dx) if abs(dx) > 1e-9 else math.inf
        ky = ry / abs(dy) if abs(dy) > 1e-9 else math.inf
        k = min(kx, ky)
    return cx + dx * k, cy + dy * k


def _cross(p1, p2, p3, p4) -> bool:
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (o(p1, p2, p3) * o(p1, p2, p4) < 0) and (o(p3, p4, p1) * o(p3, p4, p2) < 0)


def plan(W: float, H: float, targets: list[tuple], others: list[tuple] | None = None, style: str = "arrow_ring",
         color: str = "#ff3b30", labels: list[str] | None = None, width: float | None = None,
         text_scale: float = 1.0) -> list[dict]:
    """Sketch elements for `targets` (boxes x1, y1, x2, y2 in image px) on a W × H image.
    others: boxes of the file's other shapes (hidden on the screenshot, but arrows still avoid covering them).
    labels: text written at each arrow's tail (None / "": no text); text_scale: its size × the arrow-based size
    (Ring only: the text goes above the ring)."""
    style = style if style in STYLE_KEYS else "arrow_ring"
    u = unit(W, H)
    w = float(width) if width else max(3.0, round(5.0 * u, 1))
    pad = 6.0 * u
    grid = _Grid(W, H)
    for b in list(others or []) + list(targets):
        grid.add_box(b, 1.0)
    els: list[dict] = []
    marks = []
    ellipse = style in ("arrow_ring", "ring")
    for b in targets:
        if style == "arrow":
            mark = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)       # arrows stop just outside the shape
        elif ellipse:
            mark = _ring_rect(b, pad)
        else:
            mark = (b[0] - 2 * pad, b[1] - 2 * pad, b[2] + 2 * pad, b[3] + 2 * pad)
        marks.append(mark)
        if style in ("arrow_ring", "ring"):
            els.append({"t": "ellipse", "pts": [[mark[0], mark[1]], [mark[2], mark[3]]], "c": color, "w": w})
            grid.add_segment((mark[0], mark[1]), (mark[2], mark[3]), 0.5)
        elif style == "arrow_box":
            els.append({"t": "rect", "pts": [[mark[0], mark[1]], [mark[2], mark[3]]], "c": color, "w": w})
    if style == "ring":
        if labels:
            fh = w * 5 * float(text_scale or 1.0)
            for i, mark in enumerate(marks):
                if i < len(labels) and labels[i]:
                    tw = fh * 0.62 * len(str(labels[i]).splitlines()[0])
                    tx = min(max((mark[0] + mark[2]) / 2 - tw / 2, 2 * w), W - tw - 2 * w)
                    ty = mark[1] - w if mark[1] - w - fh > 0 else min(H - 2 * w, mark[3] + fh)
                    els.append({"t": "text", "pts": [[tx, ty]], "c": color, "w": w, "fs": round(fh, 1),
                                "text": str(labels[i])})
        return els
    arrows: list[tuple] = []
    tails: list[tuple] = []
    margin = 4 * w
    # big shapes first: they have the fewest good directions
    order = sorted(range(len(targets)), key=lambda i: -(targets[i][2] - targets[i][0]) * (targets[i][3] - targets[i][1]))
    placed: dict[int, dict] = {}
    for i in order:
        b, mark = targets[i], marks[i]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        size = max(b[2] - b[0], b[3] - b[1])
        base_len = min(max(70.0 * u, 0.6 * size), 0.28 * min(W, H) + 40 * u)
        best = None
        for k in range(ANGLES):
            ang = -math.pi * 3 / 4 + 2 * math.pi * k / ANGLES        # start at up-left (a natural reading order)
            dx, dy = math.cos(ang), math.sin(ang)
            gap = 1.5 * w if ellipse else 0.0
            hx, hy = _edge_point(mark, cx, cy, dx, dy, ellipse)
            head = (hx + dx * gap, hy + dy * gap)
            for f in (1.0, 0.75, 0.5):
                L = base_len * f
                tail = (head[0] + dx * L, head[1] + dy * L)
                if not (margin <= tail[0] <= W - margin and margin <= tail[1] <= H - margin):
                    continue
                c = grid.cost(tail, head, skip_box=mark) * 4.0
                c += sum(6.0 for a in arrows if _cross(tail, head, a[0], a[1]))
                c += sum(3.0 for t in tails if math.hypot(t[0] - tail[0], t[1] - tail[1]) < 3 * L / 4)
                c += sum(5.0 for j, m in enumerate(marks) if j != i and m[0] <= tail[0] <= m[2] and m[1] <= tail[1] <= m[3])
                c += 0.15 * (1 - f)                                  # prefer the full length
                c += 0.02 * min(k, ANGLES - k)                       # and, all else equal, up-left
                if best is None or c < best[0]:
                    best = (c, tail, head)
                break                                                # the longest one that fits, per direction
        if best is None:                                             # tiny image / huge shape: from the centre up
            head = (cx, max(mark[1], margin * 2))
            best = (0, (cx, max(margin, head[1] - base_len)), head)
        _c, tail, head = best
        arrows.append((tail, head))
        tails.append(tail)
        grid.add_segment(tail, head, 3.0)
        el = {"t": "arrow", "pts": [[tail[0], tail[1]], [head[0], head[1]]], "c": color, "w": w}
        placed[i] = el
    texts: list[tuple] = []                                          # placed label rects: labels never overlap
    for i in range(len(targets)):
        if i in placed:
            els.append(placed[i])
            if labels and i < len(labels) and labels[i]:
                tail, head = placed[i]["pts"]
                fh = w * 5 * float(text_scale or 1.0)                # the text's pixel size
                lines = str(labels[i]).splitlines() or [""]
                text_w = fh * 0.62 * max(len(x) for x in lines)
                # text beyond the tail, on the side away from the head, kept inside the image
                tx = tail[0] - text_w if tail[0] < head[0] else tail[0]
                ty0 = tail[1] - w * 1.5 if tail[1] < head[1] else tail[1] + fh
                tx = min(max(tx, 2 * w), W - text_w - 2 * w)
                away = -1 if tail[1] < head[1] else 1                # move further from the head first
                spot = None
                for k in (0, 1, 2, 3, 4, -1, -2, -3, -4):
                    ty = min(max(ty0 + away * k * fh * 1.15, fh * 1.2), H - 2 * w)
                    r = (tx, ty - fh, tx + text_w, ty + fh * (0.25 + 1.2 * (len(lines) - 1)))
                    if not any(r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3] for q in texts):
                        spot = (ty, r)
                        break
                if spot is None:
                    ty = min(max(ty0, fh * 1.2), H - 2 * w)
                    spot = (ty, (tx, ty - fh, tx + text_w, ty + fh * (0.25 + 1.2 * (len(lines) - 1))))
                texts.append(spot[1])
                els.append({"t": "text", "pts": [[tx, spot[0]]], "c": color, "w": w, "fs": round(fh, 1),
                            "text": str(labels[i])})
    return els


# ====================================================================================================== choices
def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class ChoiceStore:
    """Which shapes to sketch, per file: <dataset>/.aio_qc/autosketch.json.

    Each choice keeps the shape's index AND its label and box, so a file edited later (a shape deleted, the list
    re-ordered, a box nudged) still finds the same shapes: same index if it still matches, else the best match of
    the same label (box IoU ≥ 0.5). A choice whose shape is gone is dropped."""

    VERSION = 1

    def __init__(self, root):
        from pathlib import Path
        self.root = Path(root)
        self.path = self.root / ".aio_qc" / "autosketch.json"
        self.data: dict[str, list] = {}
        self.dirty = False
        try:
            import json
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(d, dict) and isinstance(d.get("files"), dict):
                self.data = d["files"]
        except (OSError, ValueError):
            pass

    def rel(self, key) -> str:
        from pathlib import Path
        try:
            return Path(key).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return Path(key).as_posix()

    def set(self, key, shapes: list[dict], indices, texts: dict | None = None):
        """Remember `indices` of `shapes` (the file's shapes in image pixels) for `key`; texts: {index: own text}
        for shapes whose arrow says something else than the batch's text."""
        texts = texts or {}
        out = []
        for i in sorted(set(int(i) for i in indices)):
            if 0 <= i < len(shapes):
                b = shape_box(shapes[i].get("points"), shapes[i].get("shape_type"))
                if b is not None:
                    c = {"i": i, "label": str(shapes[i].get("label")), "box": [round(v, 2) for v in b]}
                    if texts.get(i):
                        c["text"] = str(texts[i])
                    out.append(c)
        rel = self.rel(key)
        if out:
            self.data[rel] = out
        else:
            self.data.pop(rel, None)
        self.dirty = True

    def resolve(self, key, shapes: list[dict], with_texts: bool = False):
        """The chosen indices in the file as it is now (with_texts: also {index: own text})."""
        saved = self.data.get(self.rel(key)) or []
        boxes = [shape_box(s.get("points"), s.get("shape_type")) for s in shapes]
        used: set[int] = set()
        out = []
        texts: dict[int, str] = {}
        for c in saved:
            i, label, box = int(c.get("i", -1)), str(c.get("label")), c.get("box")
            if (0 <= i < len(shapes) and i not in used and str(shapes[i].get("label")) == label
                    and boxes[i] is not None and (box is None or _iou(boxes[i], box) >= 0.5)):
                used.add(i)
                out.append(i)
                if c.get("text"):
                    texts[i] = str(c["text"])
                continue
            if box is None:
                continue
            best, best_iou = None, 0.5
            for j, s in enumerate(shapes):
                if j in used or boxes[j] is None or str(s.get("label")) != label:
                    continue
                v = _iou(boxes[j], box)
                if v >= best_iou:
                    best, best_iou = j, v
            if best is not None:
                used.add(best)
                out.append(best)
                if c.get("text"):
                    texts[best] = str(c["text"])
        return (sorted(out), texts) if with_texts else sorted(out)

    def count(self, key) -> int:
        return len(self.data.get(self.rel(key)) or [])

    def keys_with_choices(self) -> set[str]:
        return {(self.root / r).as_posix() for r in self.data}

    def flush(self):
        if not self.dirty:
            return
        import json
        from AIO_QC.core.io import atomic_write_text
        if self.data or self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.path, json.dumps({"version": self.VERSION, "files": self.data}, indent=1))
        self.dirty = False


def arrow_texts(shapes: list[dict], idx: list[int], template: str, own: dict | None = None,
                file: str | None = None) -> list[str] | None:
    """The text at each arrow: the shape's own text, else the batch template with its fields filled in."""
    from .sketch_text import expand
    own = own or {}
    if not template and not own:
        return None
    return [own.get(i) or (expand(template, label=str(shapes[i].get("label")), n=i + 1, file=file) if template else "")
            for i in idx]
