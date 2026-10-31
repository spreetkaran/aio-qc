# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Collision-free placement of shape labels (Qt-free, screen pixels) — used by the canvas after every zoom / scroll.

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
