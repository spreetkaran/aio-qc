# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Geometry helpers (ported/optimised from dataset_analysis_v2.py).

Changes vs. the pipeline version
- Works in any coordinate space (pixel or YOLO-normalised): IoU is rasterised on
  a fixed-resolution grid over the pair's union bbox instead of a full-size mask.
- Exact analytic IoU for rectangle/rectangle pairs.
- Cheap upper-bound rejection before rasterising.
- Shape-type aware (rectangle / circle / polygon / line / point).
"""
from __future__ import annotations

import math

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

RASTER_RES = 256
VALID_POINT_COUNTS = {"polygon": (3, None), "rectangle": (2, 2), "circle": (2, 2),
                      "line": (2, 2), "point": (1, 1), "linestrip": (2, None), "points": (1, None)}


def as_array(points) -> np.ndarray | None:
    try:
        a = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[1] != 2:
        return None
    return a


def to_polygon(shape: dict) -> np.ndarray | None:
    """Area-bearing outline of a shape, or None for lines/points/invalid."""
    a = as_array(shape.get("points", []))
    if a is None or not np.isfinite(a).all():
        return None
    st = shape.get("shape_type") or "polygon"
    if st == "rectangle" and len(a) == 2:
        (x1, y1), (x2, y2) = a.min(0), a.max(0)
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if st == "circle" and len(a) == 2:
        (cx, cy), (px, py) = a
        r = math.hypot(px - cx, py - cy)
        t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)
    if st in ("polygon", None) and len(a) >= 3:
        return a
    return None


def polygon_area(a: np.ndarray | None) -> float:
    if a is None or len(a) < 3:
        return 0.0
    x, y = a[:, 0], a[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def bbox(a: np.ndarray) -> tuple[float, float, float, float]:
    mn, mx = a.min(0), a.max(0)
    return float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])


def _bbox_inter(b1, b2) -> float:
    w = min(b1[2], b2[2]) - max(b1[0], b2[0])
    h = min(b1[3], b2[3]) - max(b1[1], b2[1])
    return w * h if w > 0 and h > 0 else 0.0


def _raster_inter_union(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mn = np.minimum(a.min(0), b.min(0))
    mx = np.maximum(a.max(0), b.max(0))
    span = float(max(mx[0] - mn[0], mx[1] - mn[1]))
    if span <= 0:
        return 0.0, 0.0
    scale = RASTER_RES / span
    size = (int(math.ceil((mx[1] - mn[1]) * scale)) + 3, int(math.ceil((mx[0] - mn[0]) * scale)) + 3)
    ma, mb = np.zeros(size, np.uint8), np.zeros(size, np.uint8)
    shift = 4  # sub-pixel precision
    f = float(1 << shift)
    cv2.fillPoly(ma, [np.round((a - mn) * scale * f).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=shift)
    cv2.fillPoly(mb, [np.round((b - mn) * scale * f).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=shift)
    inter = int(np.count_nonzero(ma & mb))
    union = int(np.count_nonzero(ma | mb))
    return inter / scale ** 2, union / scale ** 2


def shape_iou(sa: dict, sb: dict, pa: np.ndarray | None = None, pb: np.ndarray | None = None,
              threshold: float = 0.0) -> float:
    """IoU of two shapes. Returns 0 early when it provably cannot reach `threshold`."""
    pa = to_polygon(sa) if pa is None else pa
    pb = to_polygon(sb) if pb is None else pb
    if pa is None or pb is None:
        return 0.0
    ba, bb = bbox(pa), bbox(pb)
    inter_bb = _bbox_inter(ba, bb)
    if inter_bb <= 0:
        return 0.0
    if sa.get("shape_type") == "rectangle" and sb.get("shape_type") == "rectangle":
        area_a = (ba[2] - ba[0]) * (ba[3] - ba[1])
        area_b = (bb[2] - bb[0]) * (bb[3] - bb[1])
        union = area_a + area_b - inter_bb
        return inter_bb / union if union > 0 else 0.0
    area_a, area_b = polygon_area(pa), polygon_area(pb)
    if max(area_a, area_b) <= 0:
        return 0.0
    if threshold > 0 and inter_bb / max(area_a, area_b) < threshold:
        return 0.0  # IoU <= inter/max(area) <= inter_bbox/max(area)
    if cv2 is None:
        return inter_bb / (area_a + area_b - inter_bb)
    inter, union = _raster_inter_union(pa, pb)
    return inter / union if union > 0 else 0.0


def containment(inner: dict, outer: dict) -> float:
    """Fraction of `inner` area lying inside `outer` (pipeline's line-inside-plate rule)."""
    pa, pb = to_polygon(inner), to_polygon(outer)
    if pa is None or pb is None or cv2 is None:
        return 0.0
    area = polygon_area(pa)
    if area <= 0 or _bbox_inter(bbox(pa), bbox(pb)) <= 0:
        return 0.0
    inter, _ = _raster_inter_union(pa, pb)
    return min(inter / area, 1.0)


def self_intersecting(a: np.ndarray) -> bool:
    n = len(a)
    if n < 4:
        return False

    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    def onseg(p, q, r):
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    for i in range(n):
        p, q = a[i], a[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # adjacent through wrap-around
            r, s = a[j], a[(j + 1) % n]
            o1, o2, o3, o4 = orient(p, q, r), orient(p, q, s), orient(r, s, p), orient(r, s, q)
            if o1 != o2 and o3 != o4:
                return True
            if (o1 == 0 and onseg(p, r, q)) or (o2 == 0 and onseg(p, s, q)) or \
               (o3 == 0 and onseg(r, p, s)) or (o4 == 0 and onseg(r, q, s)):
                return True
    return False


# Severity of each geometry issue code
GEOMETRY_ERRORS = {"malformed_points", "nan_or_inf", "invalid_point_count", "zero_area", "unknown_shape_type"}
GEOMETRY_WARNINGS = {"duplicate_consecutive_points", "duplicate_vertices", "self_intersection",
                     "tiny_area", "extreme_sliver", "coordinates_outside_image"}


def geometry_issues(shape: dict, width: float | None, height: float | None,
                    min_area: float = 2.0, normalized: bool = False) -> list[str]:
    st = shape.get("shape_type") or "polygon"
    if st not in VALID_POINT_COUNTS:
        return ["unknown_shape_type"]
    a = as_array(shape.get("points", []))
    if a is None:
        return ["malformed_points"]
    issues = []
    lo, hi = VALID_POINT_COUNTS[st]
    if len(a) < lo or (hi is not None and len(a) > hi):
        issues.append("invalid_point_count")
    if not np.isfinite(a).all():
        return sorted(set(issues + ["nan_or_inf"]))
    if st == "polygon" and len(a) >= 2:
        r = np.round(a, 6)
        if np.any(np.all(r[1:] == r[:-1], axis=1)):
            issues.append("duplicate_consecutive_points")
        elif len({tuple(x) for x in r}) < len(r):
            issues.append("duplicate_vertices")
        ring = a[np.r_[True, np.any(np.abs(np.diff(a, axis=0)) > 1e-9, axis=1)]]  # drop repeated vertices
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if len(ring) >= 4 and self_intersecting(ring):
            issues.append("self_intersection")
    poly = to_polygon(shape)
    if poly is not None:
        area = polygon_area(poly)
        if area <= 0 and "self_intersection" not in issues:
            issues.append("zero_area")
        elif not normalized and area < min_area:
            issues.append("tiny_area")
        bw, bh = np.ptp(poly[:, 0]), np.ptp(poly[:, 1])
        if bw > 0 and bh > 0 and max(bw / bh, bh / bw) > 1000:
            issues.append("extreme_sliver")
    if width and height:
        tol = 1e-3 if normalized else 1.0
        if (a[:, 0] < -tol).any() or (a[:, 1] < -tol).any() or \
           (a[:, 0] > width + tol).any() or (a[:, 1] > height + tol).any():
            issues.append("coordinates_outside_image")
    return sorted(set(issues))


def shape_metrics(shape: dict) -> tuple[float, float, float] | None:
    """(bbox width, bbox height, area) in the shape's coordinate space."""
    a = as_array(shape.get("points", []))
    if a is None or len(a) == 0 or not np.isfinite(a).all():
        return None
    poly = to_polygon(shape)
    ref = poly if poly is not None else a
    return float(np.ptp(ref[:, 0])), float(np.ptp(ref[:, 1])), polygon_area(poly)
