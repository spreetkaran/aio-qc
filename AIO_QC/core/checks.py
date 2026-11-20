# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Pure QC checks. No Qt, no file writes. Each GUI tool is a thin view over these.
"""
from __future__ import annotations

import difflib
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
# Overlapping shapes
# =============================================================================
OVERLAP_KINDS = {
    "double": "same object marked twice",          # IoU very high
    "overlap": "overlapping",                      # IoU above the threshold
    "inside": "one inside the other",              # a smaller shape mostly inside a bigger one
}


def find_overlaps(shapes: list[dict], iou_thr: float = 0.3, inside_thr: float = 0.85, double_thr: float = 0.85,
                  nested_other_class: bool = False, ignore: set | None = None,
                  labels: list | None = None) -> list[tuple]:
    """Every pair of shapes in one image that overlaps: (i, j, iou, inside, kind) with i < j, j the smaller.
    kind: "double" IoU ≥ double_thr · "overlap" IoU ≥ iou_thr · "inside" the smaller one ≥ inside_thr inside the
    bigger. A smaller shape of ANOTHER class inside a bigger one (a plate on a car, a wheel on a truck) is a part of it,
    not a mistake: reported only with nested_other_class. ignore: unordered class pairs that are expected.
    labels: only pairs where at least one shape has one of these labels."""
    from .geometry import containment, polygon_area
    polys = [to_polygon(s) for s in shapes]
    areas = [polygon_area(p) if p is not None else 0.0 for p in polys]
    want = set(labels) if labels else None
    out = []
    for a in range(len(shapes)):
        if polys[a] is None or areas[a] <= 0:
            continue
        for b in range(a + 1, len(shapes)):
            if polys[b] is None or areas[b] <= 0:
                continue
            la, lb = str(shapes[a].get("label")), str(shapes[b].get("label"))
            if want is not None and la not in want and lb not in want:
                continue
            if ignore and tuple(sorted((la, lb))) in ignore:
                continue
            iou = shape_iou(shapes[a], shapes[b], polys[a], polys[b])
            if iou <= 0:
                continue
            big, small = (a, b) if areas[a] >= areas[b] else (b, a)
            inside = containment(shapes[small], shapes[big]) if iou < double_thr else 1.0
            if iou >= double_thr:
                kind = "double"
            elif iou >= iou_thr:
                kind = "overlap"
            elif inside >= inside_thr and (la == lb or nested_other_class):
                kind = "inside"
            else:
                continue
            if kind == "overlap" and la != lb and inside >= inside_thr and not nested_other_class:
                continue                                    # a part on its object (plate on a small car)
            out.append((big, small, round(float(iou), 3), round(float(inside), 3), kind))
    return out


PAIR_KINDS = ("exact", "double", "overlap", "inside")


def find_shape_pairs(shapes: list[dict], iou_thr: float = 0.3, double_thr: float = 0.85, inside_thr: float = 0.85,
                     decimals: int = 2, same_class_only: bool = False, nested_other_class: bool = False,
                     ignore: set | None = None) -> list[tuple]:
    """Duplicate & Overlapping Shapes, one pass: (a, b, iou, inside, kind) per pair worth a look, no redundancy.
      exact    identical geometry after rounding to `decimals` (start vertex, direction and rectangle vs
               4-point polygon do not matter) — any class: two labels on one shape is a duplicate too
      double   marked twice: IoU ≥ double_thr
      overlap  IoU ≥ iou_thr
      inside   a shape at least inside_thr inside a bigger one of its class (other classes: a part, e.g. a plate
               on a car — reported only with nested_other_class)
    Duplicates (exact / double) are grouped: every copy is paired with the FIRST shape of its group (a = kept,
    b = the copy), so A~B~C gives A–B and A–C, never B–C; an overlap with a group is reported once (with the
    shape the scan meets first — the group's first shape when it overlaps too). Other pairs: a = the bigger shape, b = the smaller. `ignore`: unordered class
    pairs whose overlaps are expected (never applied to exact copies or marked-twice pairs of one class)."""
    n = len(shapes)
    label = [str(s.get("label")) for s in shapes]
    root = list(range(n))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    def join(i, j):                                   # the group's first shape stays its root
        ri, rj = find(i), find(j)
        if ri != rj:
            root[max(ri, rj)] = min(ri, rj)

    out = []
    first: dict = {}
    for i, s in enumerate(shapes):                    # exact copies
        geo = canonical_points(s, decimals)
        if geo is None:
            continue
        key = (geo, label[i] if same_class_only else None)
        if key in first:
            join(first[key], i)
            out.append((first[key], i, 1.0, 1.0, "exact"))
        else:
            first[key] = i
    pairs = find_overlaps(shapes, iou_thr, inside_thr, double_thr, nested_other_class=nested_other_class)
    doubles = sorted((p for p in pairs if p[4] == "double"), key=lambda p: (min(p[0], p[1]), max(p[0], p[1])))
    for a, b, iou, ins, _k in doubles:                # marked twice: grouped like exact copies
        i, j = min(a, b), max(a, b)
        if same_class_only and label[i] != label[j]:
            continue
        if find(i) == find(j):
            continue                                  # already in one group (an exact copy, or a chain)
        join(i, j)
        out.append((i, j, iou, ins, "double"))       # i: the earliest shape j copies (the group's first, mostly)
    reported = set()
    for a, b, iou, ins, kind in pairs:
        if kind == "double":
            continue
        if same_class_only and label[a] != label[b]:
            continue
        if ignore and tuple(sorted((label[a], label[b]))) in ignore:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue                                  # copies of one another: already a duplicate pair
        g = tuple(sorted((ra, rb)))
        if g in reported:
            continue                                  # the same overlap again through a copy
        reported.add(g)
        out.append((a, b, iou, ins, kind))
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

    # -- file-level: is it really LabelMe / YOLO? (pairing image <-> label: Image–Label Pairs) -------------
    from .formats import labelme_report, yolo_report
    shape_issue = {}                                  # shape index -> True: structure broken, no geometry checks
    if is_yolo:
        img_path = sample.image_path
        if ann.exists:
            text = "\n".join(" ".join(r) for r in (ann.raw or []))
            cm = getattr(adapter, "class_map", None)
            fr = yolo_report(text, len(cm.names) if cm is not None else 0, bool(getattr(cm, "kpt_shape", None)),
                             Path(sample.ann_path).name)
            if not fr.ok:
                return [Issue(ERROR, "Not YOLO format", f"looks like {fr.looks_like}")]
            for sev, text in fr.problems:
                issues.append(Issue(WARNING if sev == WARNING else ERROR, "Format warning", text))
            for i, sev, text in fr.items:
                cat = "Class ID out of range" if "class list has" in text else "Malformed label line"
                issues.append(Issue(ERROR if sev == "Error" else WARNING, cat, text, i))
                shape_issue[i] = True
    else:
        fr = labelme_report(ann.raw)
        raw_shapes = ann.raw.get("shapes") or []
        to_ann, k = {}, 0                             # raw shape index -> index among the loaded (object) shapes
        for j, sh in enumerate(raw_shapes):
            if isinstance(sh, dict):
                to_ann[j] = k
                k += 1
        for sev, text in fr.problems:
            issues.append(Issue(ERROR if sev == "Error" else WARNING,
                                "Schema error" if sev == "Error" else "Schema warning", text))
        for j, sev, text in fr.items:
            idx = to_ann.get(j)
            issues.append(Issue(ERROR if sev == "Error" else WARNING,
                                "Schema error" if sev == "Error" else "Schema warning",
                                f"shape {j}: {text}", idx, fix="delete_shape" if idx is not None and sev == "Error"
                                else None))
            if idx is not None and sev == "Error":
                shape_issue[idx] = True
        img_path = ann.image_path

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
        if shape_issue.get(i) or (is_yolo and s.get("_error")):
            continue                                  # reported above (format); no geometry for a broken shape
        if not is_yolo and not label.strip():
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
    # image <-> label pairing (files without their partner): the Image–Label Pairs check
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
