# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Pure QC checks. No Qt, no file writes. Each GUI tool is a thin view over these.
"""
from __future__ import annotations

import difflib
import os
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .dataset import Annotation, AnnotationError, DatasetType, Sample
from .geometry import GEOMETRY_ERRORS, geometry_issues, shape_iou, to_polygon
from pathlib import Path

from .io import verify_image

ERROR, WARNING = "Error", "Warning"


# =============================================================================
# Duplicates
# =============================================================================
def canonical_points(shape: dict, decimals: int) -> tuple | None:
    """
    Order-independent geometry key:
    - rectangles -> 4 corners (so a rectangle equals the identical 4-point polygon)
    - polygons   -> rotated to the smallest vertex and walked in one direction, so the
                    same outline drawn from another start point / reversed still matches
    """
    st = shape.get("shape_type") or "polygon"
    try:
        pts = [(round(float(p[0]), decimals), round(float(p[1]), decimals)) for p in shape.get("points", [])]
    except (TypeError, ValueError, IndexError):
        return None
    if not pts:
        return None
    if st == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        xa, xb, ya, yb = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
        pts, st = [(xa, ya), (xb, ya), (xb, yb), (xa, yb)], "polygon"
    if st == "polygon" and len(pts) >= 3:
        if len(pts) > 3 and pts[0] == pts[-1]:
            pts = pts[:-1]                       # explicitly closed ring
        i = pts.index(min(pts))
        fwd = pts[i:] + pts[:i]
        rev = [fwd[0]] + fwd[1:][::-1]
        pts = min(fwd, rev)
    return (st, tuple(pts))


def find_duplicate_pairs(shapes: list[dict], mode: str = "exact", iou_thr: float = 0.9,
                         same_class_only: bool = True, decimals: int = 2) -> list[tuple[int, int, float]]:
    """
    Returns (keep_index, duplicate_index, iou); every duplicate is paired with the
    FIRST shape of its cluster, so chains (A~B~C) never yield B-C pairs.
    exact: identical canonical geometry (start-vertex / direction / rect-vs-polygon independent).
    iou:   IoU >= iou_thr (exact for box pairs, rasterised for polygons).
    """
    out = []
    if mode == "exact":
        first: dict = {}
        for i, s in enumerate(shapes):
            geo = canonical_points(s, decimals)
            if geo is None:
                continue
            key = (geo, s.get("label") if same_class_only else None)
            if key in first:
                out.append((first[key], i, 1.0))
            else:
                first[key] = i
        return out
    polys = [to_polygon(s) for s in shapes]
    claimed: set[int] = set()
    for i in range(len(shapes)):
        if polys[i] is None or i in claimed:
            continue
        for j in range(i + 1, len(shapes)):
            if j in claimed or polys[j] is None:
                continue
            if same_class_only and shapes[i].get("label") != shapes[j].get("label"):
                continue
            iou = shape_iou(shapes[i], shapes[j], polys[i], polys[j], threshold=iou_thr)
            if iou >= iou_thr:
                out.append((i, j, iou))
                claimed.add(j)
    return out


# =============================================================================
# Label ambiguity / unknown labels
# =============================================================================
def normalize_label(label: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(label).strip().lower())


@dataclass
class LabelFinding:
    label: str
    kind: str          # "format variant" | "similar" | "unknown"
    suggestion: str
    score: float


def classify_labels(label_counts: Counter, true_classes: list[str] | None,
                    cutoff: float = 0.8) -> list[LabelFinding]:
    """
    With a class list: every label not exactly in it is a variant / similar / unknown.
    Without one: compare labels against *each other*; the rarer of two
    similar labels is flagged and the more frequent one suggested.
    """
    out = []
    if true_classes:
        truth = set(true_classes)
        norm = {normalize_label(c): c for c in true_classes}
        for label in sorted(label_counts):
            if label in truth:
                continue
            n = normalize_label(label)
            if n in norm:
                out.append(LabelFinding(label, "format variant", norm[n], 1.0))
                continue
            m = difflib.get_close_matches(n, list(norm), n=1, cutoff=cutoff)
            if m:
                ratio = difflib.SequenceMatcher(None, n, m[0]).ratio()
                out.append(LabelFinding(label, "similar", norm[m[0]], round(ratio, 3)))
            else:
                out.append(LabelFinding(label, "unknown", "", 0.0))
        return out
    ranked = [l for l, _ in label_counts.most_common()]
    for i, label in enumerate(ranked):
        n = normalize_label(label)
        best, best_score, kind = "", 0.0, ""
        for other in ranked[:i]:  # only more frequent labels are candidates
            no = normalize_label(other)
            if no == n:
                best, best_score, kind = other, 1.0, "format variant"
                break
            r = difflib.SequenceMatcher(None, n, no).ratio()
            if r >= cutoff and r > best_score:
                best, best_score, kind = other, r, "similar"
        if best:
            out.append(LabelFinding(label, kind, best, round(best_score, 3)))
    return out


# =============================================================================
# Outliers (per-class IQR on log values)
# =============================================================================
def iqr_bounds(values: list[float], k: float) -> tuple[float, float, float]:
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr, float(med)


def shape_measure(shape: dict, metric: str, img_w: float | None, img_h: float | None) -> float | None:
    poly = to_polygon(shape)
    if poly is None:
        return None
    w = float(np.ptp(poly[:, 0])); h = float(np.ptp(poly[:, 1]))
    if w <= 0 or h <= 0:
        return None
    if metric == "aspect ratio (w/h)":
        return w / h
    from .geometry import polygon_area
    area = polygon_area(poly)
    if img_w and img_h:  # scale invariant across image resolutions
        area /= (img_w * img_h)
        w /= img_w
        h /= img_h
    return {"area": area, "width": w, "height": h}.get(metric.split()[0])


# =============================================================================
# Validation
# =============================================================================
@dataclass
class Issue:
    severity: str
    category: str
    detail: str
    shape_index: int | None = None
    fix: str | None = None          # fix code understood by core.fixes (None = manual only)
    fix_arg: object = None


@dataclass
class ValidationOptions:
    """Validation = file & geometry INTEGRITY only (see the comment in validate_sample for what lives where)."""
    deep_image_check: bool = False
    geometry: bool = True
    min_area_px: float = 2.0


def validate_sample(adapter, sample: Sample, opts: ValidationOptions) -> list[Issue]:
    issues: list[Issue] = []
    try:
        ann: Annotation = adapter.load(sample)
    except AnnotationError as exc:
        cat = "Not LabelMe format" if "Not a LabelMe" in str(exc) else "Unreadable annotation"
        return [Issue(ERROR, cat, str(exc))]

    is_yolo = adapter.type in (DatasetType.YOLO, DatasetType.YOLO_SPLIT)

    # -- file-level -----------------------------------------------------------
    if is_yolo:
        if not ann.exists:
            issues.append(Issue(WARNING, "No label file (background)", "image has no .txt label"))
        img_path = sample.image_path
    else:
        for note in ann.notes:
            issues.append(Issue(WARNING, "Schema warning", note))
        if not ann.raw.get("imagePath"):
            issues.append(Issue(ERROR, "Missing imagePath", "imagePath is empty"))
        img_path = ann.image_path
        if img_path is None and not ann.image_data:
            issues.append(Issue(ERROR, "Image not found", str(ann.raw.get("imagePath"))))
        elif img_path is not None and ann.raw.get("imagePath"):
            declared = str(ann.raw["imagePath"]).replace("\\", "/")
            folder = sample.ann_path.parent
            if (folder / declared).resolve() != Path(img_path).resolve():
                rel = Path(os.path.relpath(img_path, folder)).as_posix()
                issues.append(Issue(WARNING, "imagePath mismatch",
                                    f"imagePath '{declared}' but image found at '{rel}'",
                                    fix="set_image_path", fix_arg=rel))

    # (empty annotations -> 📭 Empty tool; label names -> 📝 Label Ambiguities;
    #  duplicate shapes / YOLO lines -> 🔍 Duplicate Shapes; split leakage -> 🖼️ Duplicate Images)

    w, h = ann.width, ann.height
    if opts.deep_image_check and img_path is not None:
        ok, size, msg = verify_image(img_path)
        if not ok:
            issues.append(Issue(ERROR, "Image unreadable", msg or str(img_path)))
        elif not is_yolo and size and w and h and (int(w), int(h)) != size:
            issues.append(Issue(ERROR, "Size mismatch", f"JSON {int(w)}x{int(h)} vs image {size[0]}x{size[1]}",
                                fix="set_image_size", fix_arg=size))

    # -- shape-level ----------------------------------------------------------
    for i, s in enumerate(ann.shapes):
        label = str(s.get("label", ""))
        if is_yolo:
            if s.get("_error"):
                issues.append(Issue(ERROR, "Malformed label line", f"line {s['_line']}: {s['_error']}", i))
                continue
            cid = s["_class_id"]
            n_cls = len(adapter.class_map.names)   # raw class map (a view mapping may merge names)
            if n_cls and not 0 <= cid < n_cls:
                issues.append(Issue(ERROR, "Class ID out of range",
                                    f"line {s['_line']}: id {cid} (valid 0-{n_cls - 1})", i))
        elif not label.strip():
            issues.append(Issue(ERROR, "Empty label", "shape has no label", i))
        if opts.geometry:
            for code in geometry_issues(s, w, h, opts.min_area_px, normalized=ann.normalized):
                sev = ERROR if code in GEOMETRY_ERRORS else WARNING
                cat = "Coordinates outside image" if code == "coordinates_outside_image" else "Invalid geometry"
                fix = {"coordinates_outside_image": "clamp_to_image",
                       "duplicate_consecutive_points": "dedupe_vertices",
                       "duplicate_vertices": "dedupe_vertices"}.get(code)
                if fix == "clamp_to_image" and not (w and h):
                    fix = None
                elif fix == "clamp_to_image" and _entirely_outside(s, w, h):
                    fix, code = "delete_shape", "entirely outside image"
                issues.append(Issue(sev, cat, f"{label}: {code}", i, fix=fix,
                                    fix_arg=(w, h) if fix == "clamp_to_image" else None))
    return issues


def validate_dataset(adapter, samples: list[Sample]) -> list[tuple[str, Issue]]:
    """Dataset-level checks. Returns (path, Issue) pairs."""
    out: list[tuple[str, Issue]] = []
    is_yolo = adapter.type in (DatasetType.YOLO, DatasetType.YOLO_SPLIT)
    cm = getattr(adapter, "class_map", None)
    if is_yolo and cm is not None:
        src = str(cm.source) if cm.source else str(adapter.root)
        for p in cm.problems:
            out.append((src, Issue(ERROR, "Class map problem", p)))
    orphans = adapter.orphans(samples)
    for p in orphans["annotations_without_image"]:
        out.append((str(p), Issue(ERROR, "Label without image", p.name)))
    if adapter.type in (DatasetType.LABELME, DatasetType.CLS_LABELME):
        for p in orphans["images_without_annotation"]:
            out.append((str(p), Issue(WARNING, "Image without annotation", p.name)))
    return out


def _entirely_outside(shape: dict, w, h) -> bool:
    poly = to_polygon(shape)
    pts = poly if poly is not None else np.asarray(shape.get("points", []), dtype=float)
    if len(pts) == 0:
        return False
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    return x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h


# =============================================================================
# Stats (count_labelme_classes port + per split)
# =============================================================================
def color_band(pct: float) -> str:
    """Same bands as count_labelme_classes.print_summary."""
    if pct >= 20:
        return "high"
    if pct >= 10:
        return "good"
    if pct >= 2:
        return "low"
    return "rare"
