# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image-region filter for Dim / Area (Qt-free, vectorised — millions of shapes in milliseconds).

A region is a rectangle in PIXELS of one exact image size (W × H). Only shapes of images with exactly that size take
part — on a fixed camera the same pixel rectangle is the same place in the scene; on other sizes it would not be.

    groups = size_groups(iw, ih, file_keys)               # [(W, H, files, shapes)] most common first
    m = region_mask(bx1, by1, bx2, by2, iw, ih, (W, H), (x1, y1, x2, y2), mode="inside")
    heat = coverage(bx1, by1, bx2, by2, sel, W, H)        # where annotations lie (for the canvas)
"""
from __future__ import annotations

import numpy as np

MODES = {
    "inside": "Fully inside the region",
    "center": "Centre inside the region",
    "mostly": "Mostly inside (≥ 50 % of the box)",
    "touch": "Touching the region (any overlap)",
}


def size_groups(iw: np.ndarray, ih: np.ndarray, file_ids: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Exact image sizes present: [(W, H, files, shapes)], most shapes first. Unknown sizes (0) are left out."""
    ok = (iw > 0) & (ih > 0)
    if not ok.any():
        return []
    code = iw[ok].astype(np.int64) * 1_000_000 + ih[ok].astype(np.int64)
    u, inv, shapes = np.unique(code, return_inverse=True, return_counts=True)
    fid = file_ids[ok].astype(np.int64)
    span = int(fid.max()) + 1
    pairs = np.unique(inv.astype(np.int64) * span + fid)                  # (group, file) once each
    files = np.bincount(pairs // span, minlength=len(u))
    out = [(int(c // 1_000_000), int(c % 1_000_000), int(f), int(s)) for c, f, s in zip(u, files, shapes)]
    out.sort(key=lambda g: (-g[3], -g[2]))
    return out


def in_size(iw, ih, size) -> np.ndarray:
    return (iw == size[0]) & (ih == size[1])


def region_mask(bx1, by1, bx2, by2, iw, ih, size, rect, mode: str = "inside", invert: bool = False) -> np.ndarray:
    """Shapes of `size` images that relate to `rect` (pixels) as `mode`. invert=True: the shapes of those images
    that do NOT — e.g. everything outside a region of interest. Other image sizes are never selected."""
    same = in_size(iw, ih, size)
    x1, y1, x2, y2 = rect
    if mode == "center":
        cx, cy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        hit = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
    elif mode == "touch":
        hit = (bx2 >= x1) & (bx1 <= x2) & (by2 >= y1) & (by1 <= y2)
    elif mode == "mostly":
        iw_ = np.clip(np.minimum(bx2, x2) - np.maximum(bx1, x1), 0, None)
        ih_ = np.clip(np.minimum(by2, y2) - np.maximum(by1, y1), 0, None)
        area = np.maximum((bx2 - bx1) * (by2 - by1), 1e-9)
        hit = iw_ * ih_ >= 0.5 * area
        thin = (bx2 - bx1 <= 0) | (by2 - by1 <= 0)                        # points / lines: centre decides
        if thin.any():
            cx, cy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
            hit = np.where(thin, (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2), hit)
    else:                                                                  # fully inside
        hit = (bx1 >= x1) & (bx2 <= x2) & (by1 >= y1) & (by2 <= y2)
    return same & (~hit if invert else hit)


def coverage(bx1, by1, bx2, by2, sel: np.ndarray, W: int, H: int, max_cells: int = 320) -> np.ndarray:
    """How many selected boxes cover each cell of a grid over a W × H image (difference array + 2-D prefix sum:
    O(shapes + cells), no per-box loops). -> float32 array (rows, cols)."""
    k = max(W, H) / max_cells
    cols, rows = max(1, int(round(W / k))), max(1, int(round(H / k)))
    grid = np.zeros((rows + 1, cols + 1), np.float32)
    if sel is not None and len(sel) and sel.any():
        sx, sy = cols / W, rows / H
        c1 = np.clip((bx1[sel] * sx).astype(np.int64), 0, cols - 1)
        r1 = np.clip((by1[sel] * sy).astype(np.int64), 0, rows - 1)
        c2 = np.clip(np.ceil(bx2[sel] * sx).astype(np.int64), c1 + 1, cols)
        r2 = np.clip(np.ceil(by2[sel] * sy).astype(np.int64), r1 + 1, rows)
        w, n = cols + 1, (rows + 1) * (cols + 1)
        bc = lambda r, c: np.bincount(r * w + c, minlength=n)              # integer counts: no weights needed
        grid = (bc(r1, c1) - bc(r1, c2) - bc(r2, c1) + bc(r2, c2)).reshape(rows + 1, w)
        grid = grid.cumsum(0).cumsum(1).astype(np.float32)
    return grid[:rows, :cols]


def clamp_rect(rect, W, H, min_size: float = 1.0):
    x1, y1, x2, y2 = rect
    x1, x2 = sorted((min(max(x1, 0.0), W), min(max(x2, 0.0), W)))
    y1, y2 = sorted((min(max(y1, 0.0), H), min(max(y2, 0.0), H)))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return (x1, y1, x2, y2)
