# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Collision-free placement of shape labels (Qt-free, screen pixels) — used by the canvas after every zoom / scroll.

place_anchored(items, bounds) -> [(anchor index, x, y)]      (what the canvas uses)
    items  [(anchors, w, h)]  anchors = points ON the shape's outline in screen pixels, best first (top-most vertex,
           left-most, right-most, bottom-most, then the rest); w, h = label size
    Each label is put right next to one of its anchors — its corner touching the point (above-right, above-left,
    right, left, below-right, below-left) — so it always sits on the shape itself, not on the empty corner of a
    bounding box. The anchor's connector dot is reserved too, so no label hides another shape's dot. When every
    spot near every anchor is taken, the label moves out in rows above / below its first anchor (the canvas then
    draws a leader line back to the anchor), or to the spot with the least overlap.

place(boxes, bounds) (bounding-box placement, kept for overlays):

    place(boxes, bounds) -> [(x, y)]   top-left of each label

boxes  [(bx, by, bw, bh, w, h)]  the shape's box on screen and the label's size, in placement order (most
       important first: the canvas passes the selected shape first, then small shapes before big ones)
bounds (left, top, right, bottom) labels are kept inside where possible (the visible part of the image), or None

For each label the first free spot wins: above the box (left-aligned) · inside its top-left corner (if the box is
big enough) · below the box · above (right-aligned) · left / right of the top edge · below (right-aligned) · above,
shifted one label-width right / left · then rows further above / below (4 each).
If all are taken, the spot with the least overlap is used. A coarse grid keeps it ~linear: thousands of labels in a
few milliseconds.
"""
from __future__ import annotations

CELL = 64.0
GAP = 2.0


def candidates(bx, by, bw, bh, w, h):
    out = [(bx, by - h - GAP)]
    if bw >= w + 4 and bh >= 2.2 * h:
        out.append((bx + 2, by + 2))
    out += [(bx, by + bh + GAP), (bx + bw - w, by - h - GAP), (bx - w - GAP, by), (bx + bw + GAP, by)]
    out += [(bx + bw - w, by + bh + GAP), (bx + w + GAP, by - h - GAP), (bx - w - GAP, by - h - GAP)]
    for k in range(1, 5):                                   # further rows above, then below
        out += [(bx, by - (h + GAP) * k - h - GAP), (bx, by + bh + GAP + (h + GAP) * k)]
    return out


def place(boxes, bounds=None):
    grid: dict = {}
    placed: list = []

    def cells(x1, y1, x2, y2):
        for gx in range(int(x1 // CELL), int(x2 // CELL) + 1):
            for gy in range(int(y1 // CELL), int(y2 // CELL) + 1):
                yield gx, gy

    def hits(x1, y1, x2, y2) -> bool:
        for c in cells(x1, y1, x2, y2):
            for k in grid.get(c, ()):
                a1, b1, a2, b2 = placed[k]
                if min(x2, a2) > max(x1, a1) and min(y2, b2) > max(y1, b1):
                    return True
        return False

    def overlap(x1, y1, x2, y2):
        seen, tot = set(), 0.0
        for c in cells(x1, y1, x2, y2):
            for k in grid.get(c, ()):
                if k in seen:
                    continue
                seen.add(k)
                a1, b1, a2, b2 = placed[k]
                ow, oh = min(x2, a2) - max(x1, a1), min(y2, b2) - max(y1, b1)
                if ow > 0 and oh > 0:
                    tot += ow * oh
        return tot

    res = []
    clamp = bounds is not None
    for bx, by, bw, bh, w, h in boxes:
        spots = []
        for x, y in candidates(bx, by, bw, bh, w, h):
            if clamp and bounds[2] - bounds[0] > w and bounds[3] - bounds[1] > h:
                x = min(max(x, bounds[0]), bounds[2] - w)
                y = min(max(y, bounds[1]), bounds[3] - h)
            spots.append((x, y))
        best = next(((x, y) for x, y in spots if not hits(x, y, x + w, y + h)), None)   # fast: first free spot
        if best is None:                                                               # crowded: least overlap
            best = min(spots[:6], key=lambda q: overlap(q[0], q[1], q[0] + w, q[1] + h))
        x, y = best
        k = len(placed)
        placed.append((x - 1, y - 1, x + w + 1, y + h + 1))
        for c in cells(*placed[-1]):
            grid.setdefault(c, []).append(k)
        res.append(best)
    return res


DOT = 5.0                       # connector dot (screen px) reserved around each used anchor


def anchored_candidates(anchors, w, h, gap=3.0, rows=4):
    """[(anchor index, x, y, near?)] in preference order."""
    out = []
    for i, (ax, ay) in enumerate(anchors):
        out += [(i, ax - 2, ay - h - gap, True), (i, ax - w + 2, ay - h - gap, True),
                (i, ax + gap + 1, ay - h / 2, True), (i, ax - w - gap - 1, ay - h / 2, True),
                (i, ax - 2, ay + gap, True), (i, ax - w + 2, ay + gap, True)]
    if anchors:
        ax, ay = anchors[0]
        for k in range(1, rows + 1):                       # further out: leader line back to the first anchor
            out += [(0, ax - 2, ay - h - gap - (h + GAP + 4) * k, False),
                    (0, ax - 2, ay + gap + (h + GAP + 4) * k, False),
                    (0, ax + gap + 8 * k + 4, ay - h / 2 - (h + GAP) * k, False),
                    (0, ax - w - gap - 8 * k - 4, ay - h / 2 - (h + GAP) * k, False)]
    return out


def place_anchored(items, bounds=None):
    grid: dict = {}
    placed: list = []

    def cells(x1, y1, x2, y2):
        for gx in range(int(x1 // CELL), int(x2 // CELL) + 1):
            for gy in range(int(y1 // CELL), int(y2 // CELL) + 1):
                yield gx, gy

    def add(r):
        k = len(placed)
        placed.append(r)
        for c in cells(*r):
            grid.setdefault(c, []).append(k)

    def overlap(x1, y1, x2, y2, first_only=False):
        seen, tot = set(), 0.0
        for c in cells(x1, y1, x2, y2):
            for k in grid.get(c, ()):
                if k in seen:
                    continue
                seen.add(k)
                a1, b1, a2, b2 = placed[k]
                ow, oh = min(x2, a2) - max(x1, a1), min(y2, b2) - max(y1, b1)
                if ow > 0 and oh > 0:
                    if first_only:
                        return 1.0
                    tot += ow * oh
        return tot

    # every shape's preferred anchor dot is reserved up front: labels never cover another shape's dot
    for anchors, _w, _h in items:
        if anchors:
            ax, ay = anchors[0]
            add((ax - DOT, ay - DOT, ax + DOT, ay + DOT))
    res = []
    clamp = bounds is not None
    crowded = len(items) > 600                     # thousands on screen: fewer spots each, still no overlaps where
    for anchors, w, h in items:                    # there is room (the canvas re-lays out when you zoom in)
        if not anchors:
            res.append((0, 0.0, 0.0))
            continue
        if crowded:
            anchors = anchors[:2]
        spots = []
        for i, x, y, near in anchored_candidates(anchors, w, h, rows=1 if crowded else 4):
            if clamp and bounds[2] - bounds[0] > w and bounds[3] - bounds[1] > h:
                cx = min(max(x, bounds[0]), bounds[2] - w)
                cy = min(max(y, bounds[1]), bounds[3] - h)
                if near and (abs(cx - x) > 1 or abs(cy - y) > 1):
                    continue                               # a near spot pushed off the image is not near any more
                x, y = cx, cy
            spots.append((i, x, y))
        if not spots:
            i, x, y, _n = anchored_candidates(anchors, w, h)[0]
            spots = [(i, x, y)]
        best = None
        for i, x, y in spots:                                   # first free spot (its own dot does not count)
            ax, ay = anchors[i]
            own = (ax - DOT, ay - DOT, ax + DOT, ay + DOT)
            if not _hits_except(grid, placed, cells, (x, y, x + w, y + h), own):
                best = (i, x, y)
                break
        if best is None:
            best = min(spots[:4 if crowded else 12], key=lambda q: overlap(q[1], q[2], q[1] + w, q[2] + h))
        i, x, y = best
        add((x - 1, y - 1, x + w + 1, y + h + 1))
        if i != 0:                                               # a different anchor is used: reserve its dot too
            ax, ay = anchors[i]
            add((ax - DOT, ay - DOT, ax + DOT, ay + DOT))
        res.append(best)
    return res


def _hits_except(grid, placed, cells, r, own) -> bool:
    x1, y1, x2, y2 = r
    seen = set()
    for c in cells(x1, y1, x2, y2):
        for k in grid.get(c, ()):
            if k in seen:
                continue
            seen.add(k)
            a = placed[k]
            if a == own:
                continue
            if min(x2, a[2]) > max(x1, a[0]) and min(y2, a[3]) > max(y1, a[1]):
                return True
    return False


def outline_anchors(points, kind: str, limit: int = 10):
    """Anchor points on a shape's outline (scene / any units), best first: top-most, left-most, right-most,
    bottom-most vertex, then the remaining vertices top to bottom (at most `limit`). Rectangles: their corners;
    circles (centre, rim point): top, left, right, bottom of the rim."""
    import math
    pts = [(float(x), float(y)) for x, y in points]
    if not pts:
        return []
    if kind == "circle" and len(pts) == 2:
        (cx, cy), (rx, ry) = pts
        r = math.hypot(rx - cx, ry - cy)
        return [(cx, cy - r), (cx - r, cy), (cx + r, cy), (cx, cy + r)]
    if kind == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        l, t, rr, b = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        return [(l, t), (rr, t), (l, b), (rr, b)]
    first = [min(pts, key=lambda p: (p[1], p[0])), min(pts, key=lambda p: (p[0], p[1])),
             max(pts, key=lambda p: (p[0], -p[1])), max(pts, key=lambda p: (p[1], -p[0]))]
    out = []
    for p in first + sorted(pts, key=lambda p: (p[1], p[0])):
        if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 1e-6 for q in out):
            out.append(p)
        if len(out) >= limit:
            break
    return out
