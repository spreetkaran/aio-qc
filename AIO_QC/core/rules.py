# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Relationship rules: how annotations relate to each other and to the image — configured once per dataset
(.aio_qc/rules.json), checked in seconds, no model needed. Rule = {"type", "enabled", "severity", …params}.

    rules = load_rules(root);  violations = check_image(rules, shapes, tags, W, H)
    suggestions = suggest(stats)  # mined from the dataset: rules that hold for >= 97 % of it, with evidence

Universal: every rule works on any classes (a class name, or * = any class) and any shape type; nothing is
specific to one kind of dataset. One rule per relationship (no overlapping rule types):
    part      Parent / part — the whole relationship of a part and its parent (wheel ⊂ car, head ⊂ person,
              nucleus ⊂ cell, text ⊂ sign, any object ⊂ region of interest):
              the part lies inside a parent (area share; points / lines: share of their points), is not larger
              (and not smaller) than a share of it, is turned like it (angle), each parent has min–max parts,
              parts of one parent do not overlap. Replaces the old inside / contains / siblings.
    touch     Must touch — every A overlaps some B without having to lie inside it (a person on a bicycle, a leaf
              on its stem, a label on its parcel).
    ratio     Not more than — #A ≤ per × #B in an image (no more heads than persons, ≤ 4 wheels per car).
    tilt      Tilt limit — polygons of a class turned at most N° from the image axes (minimum-area rectangle).
    never     Never together — at most one of these classes per image (shapes or classification tags). Replaces the
              old excludes (two classes) / exclusive (a list).
    Same-class duplicates are not a rule: Duplicate Shapes finds them (exact or IoU, with delete actions).
Saved rules of the old types are converted on load (upgrade_rules).

Geometry: shapes are compared by their real outline where it is convex (rectangles, 4-point quads, circles, most
polygons; cv2.intersectConvexConvex), else by their bounding boxes. Sizes for "larger than its parent" are the
long / short side of the minimum-area rectangle, so rotated objects compare correctly. Pairs are found
with vectorised bounding-box tests first; exact geometry only runs for pairs whose boxes touch.
"""
from __future__ import annotations

import json
import math
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .io import atomic_write_text

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# type -> (title, group, params with defaults, needs pixels)
TYPES = {
    "part": ("Parent / part", "objects", {"child": "", "parent": "", "need_parent": True, "min_frac": 0.9,
                                          "max_rel": 1.0, "min_rel": 0.0, "max_angle": 0.0, "min": 0, "max": 0,
                                          "apart_iou": 0.0}, False),
    "touch": ("Must touch", "objects", {"a": "", "b": "", "min_frac": 0.0}, False),
    "overlap": ("No overlap", "objects", {"a": "", "b": "", "max_iou": 0.5}, False),
    "count": ("Count per image", "image", {"label": "", "min": 0, "max": 1}, False),
    "ratio": ("Not more than", "image", {"a": "", "b": "", "per": 1.0}, False),
    "requires": ("Requires", "image", {"a": "", "b": ""}, False),
    "never": ("Never together", "image", {"labels": []}, False),
    "allowed": ("Allowed classes", "image", {"labels": []}, False),
    "size": ("Size limits", "object", {"label": "", "min_w": 0, "min_h": 0, "max_w": 0, "max_h": 0,
                                       "min_area_pct": 0.0, "max_area_pct": 0.0}, True),
    "aspect": ("Aspect ratio", "object", {"label": "", "min": 0.0, "max": 0.0}, True),
    "tilt": ("Tilt limit", "object", {"label": "", "max_deg": 30.0}, True),
    "shape_type": ("Shape type", "object", {"label": "", "types": ["rectangle"]}, False),
    "points": ("Vertex count", "object", {"label": "", "min": 3, "max": 0}, False),
    "border": ("Not at the border", "object", {"label": "", "margin_px": 1}, True),
    "attribute": ("Attribute set", "object", {"label": "", "flag": "", "value": "any"}, False),
    "group": ("Group must contain", "object", {"labels": []}, False),
    "one_class": ("One class per image", "classification", {}, False),
}
LEGACY = {"inside", "contains", "siblings", "excludes", "exclusive"}
ANY = "*"
META = ("id", "enabled", "severity", "held", "name")
WILDCARD = {"part": ("child", "parent"), "touch": ("a", "b"), "overlap": ("a", "b"), "count": ("label",),
            "size": ("label",), "aspect": ("label",), "tilt": ("label",), "shape_type": ("label",),
            "points": ("label",), "border": ("label",), "attribute": ("label",)}   # params that accept * (any class)


def is_any(lab) -> bool:
    return lab in ("", ANY, None)


def cls(lab, plural: bool = False) -> str:
    """A class for messages: '*' reads as 'object'."""
    return ("objects" if plural else "object") if is_any(lab) else str(lab)


def new_rule(rtype: str, **params) -> dict:
    title, _g, defaults, _px = TYPES[rtype]
    r = {"id": uuid.uuid4().hex[:8], "type": rtype, "enabled": True, "severity": "error"}
    r.update({k: (list(v) if isinstance(v, list) else v) for k, v in defaults.items()})
    r.update(params)
    return r


def signature(r: dict) -> str:
    """What a rule checks (without id / on-off / severity / evidence): equal signatures = the same rule."""
    d = dict(TYPES[r["type"]][2]) if r.get("type") in TYPES else {}
    d.update({k: v for k, v in r.items() if k not in META})
    return json.dumps(d, sort_keys=True)


def trigger_labels(r: dict) -> set | None:
    """Classes whose presence a file needs before this rule can fail in it; None = any file can fail it."""
    t, g = r["type"], r.get
    if t == "part":
        if is_any(g("child")) or (g("min") and is_any(g("parent"))):
            return None
        return {g("child")} | ({g("parent")} if g("min") else set())
    if t in ("touch", "ratio"):
        return None if is_any(g("a")) else {g("a")}
    if t == "overlap":
        return None if is_any(g("a")) or is_any(g("b")) else {g("a")}
    if t == "requires":
        return {g("a")}
    if t == "never":
        return set(g("labels") or []) or None
    if t in ("size", "aspect", "tilt", "shape_type", "points", "border", "attribute"):
        return {g("label")} if g("label") not in ("", ANY) else None
    if t == "count":
        return {g("label")} if g("label") not in ("", ANY) and not g("min") else None
    return None


def _pct(v) -> str:
    return f"{float(v) * 100:.0f} %"


def describe(r: dict) -> str:
    t = r["type"]
    g = r.get
    if r.get("name"):
        return str(r["name"])
    if t == "part":
        c, p = cls(g("child")), cls(g("parent"))
        parts = []
        if g("need_parent", True):
            parts.append(f"inside a {p} (≥ {_pct(g('min_frac', 0.9))} of it)")
        if g("max_rel"):
            parts.append(f"at most {_pct(g('max_rel'))} of its {p}'s size")
        if g("min_rel"):
            parts.append(f"at least {_pct(g('min_rel'))} of it")
        if g("max_angle"):
            parts.append(f"turned like it (± {float(g('max_angle')):g}°)")
        mn, mx = int(g("min", 0) or 0), int(g("max", 0) or 0)
        if mn or mx:
            rng = f"{mn}" if mn == mx else (f"at least {mn}" if not mx else f"{mn}–{mx}")
            parts.append(f"each {p} has {rng} {c}")
        if g("apart_iou"):
            parts.append(f"{c}s of one {p} do not overlap (IoU > {g('apart_iou')})")
        return f"{c} is part of {p}: " + ("; ".join(parts) or "no checks ticked")
    if t == "touch":
        mf = float(g("min_frac", 0) or 0)
        return f"every {cls(g('a'))} touches a {cls(g('b'))}" + (f" (≥ {_pct(mf)} of it on one)" if mf else "")
    if t == "ratio":
        per = float(g("per", 1) or 1)
        return (f"no more {g('a')} than {g('b')} in an image" if per == 1
                else f"at most {per:g} {g('a')} per {g('b')} in an image")
    if t == "tilt":
        return f"{cls(g('label'))} turned at most {float(g('max_deg', 30)):g}°"
    if t == "overlap":
        same = g("a") == g("b") or (is_any(g("a")) and is_any(g("b")))
        return (f"no two {cls(g('a'), True)} overlap (IoU > {g('max_iou')})" if same
                else f"{cls(g('a'))} and {cls(g('b'))} never overlap (IoU > {g('max_iou')})")
    if t == "count":
        lab = "objects" if g("label") in ("", ANY) else g("label")
        mx = g("max", 0)
        return (f"{g('min')}–{mx} {lab} per image" if mx else f"at least {g('min')} {lab} per image")
    if t == "requires":
        return f"an image with {g('a')} also has {g('b')}"
    if t == "never":
        labs = g("labels") or []
        return (f"{labs[0]} and {labs[1]} never in the same image" if len(labs) == 2
                else f"at most one of {', '.join(labs)} per image")
    if t == "allowed":
        return f"only these classes: {', '.join(g('labels') or [])}"
    if t == "size":
        parts = []
        if g("min_w") or g("min_h"):
            parts.append(f"≥ {g('min_w')}×{g('min_h')} px")
        if g("max_w") or g("max_h"):
            parts.append(f"≤ {g('max_w') or '∞'}×{g('max_h') or '∞'} px")
        if g("min_area_pct"):
            parts.append(f"≥ {g('min_area_pct')} % of the image")
        if g("max_area_pct"):
            parts.append(f"≤ {g('max_area_pct')} % of the image")
        return f"{g('label') or 'every object'} is " + (", ".join(parts) or "any size")
    if t == "aspect":
        return f"{cls(g('label'))} width/height between {g('min')} and {g('max') or '∞'}"
    if t == "shape_type":
        return f"{cls(g('label'))} is drawn as {' or '.join(g('types') or [])}"
    if t == "points":
        return f"{cls(g('label'))} polygons have {g('min')}–{g('max') or '∞'} vertices"
    if t == "border":
        return f"{g('label') or 'objects'} do not touch the image border ({g('margin_px')} px)"
    if t == "attribute":
        v = g("value", "any")
        return f"{cls(g('label'))} has the attribute “{g('flag')}”" + ("" if v == "any" else f" = {v}")
    if t == "group":
        return f"every group (group_id) contains {', '.join(g('labels') or [])}"
    if t == "one_class":
        return "exactly one class per image"
    return t


# =============================================================================
# Persistence (+ conversion of the old overlapping rule types)
# =============================================================================
def upgrade_rules(rules: list[dict]) -> tuple[list[dict], bool]:
    """Old rule types -> the merged ones. inside / contains / siblings of the same part + parent (and severity)
    become ONE part rule; excludes / exclusive become never. Exact duplicates are dropped. -> (rules, changed)"""
    out, changed, parts = [], False, {}
    for r in rules:
        t = r.get("type")
        keep = {k: r[k] for k in META if k in r}
        if t in ("inside", "contains", "siblings"):
            changed = True
            child, parent = r.get("child", ""), r.get("parent", "")
            sev = r.get("severity", "error")
            key = (child, parent, sev, bool(r.get("enabled", True)))
            p = parts.get(key)
            if p is None:
                p = new_rule("part", child=child, parent=parent, need_parent=False, max_rel=0.0,
                             min_frac=float(r.get("min_frac", 0.9)), **keep)
                parts[key] = p
                out.append(p)
            if t == "inside":
                p["need_parent"] = True
                p["max_rel"] = 1.0                    # a part is never larger than its whole
                p["min_frac"] = float(r.get("min_frac", p["min_frac"]))
            elif t == "contains":
                p["min"] = int(r.get("min", 1) or 0)
                mx = int(r.get("max", 99) or 0)
                p["max"] = 0 if mx >= 99 else mx
            else:
                p["apart_iou"] = float(r.get("max_iou", 0.1))
        elif t in ("excludes", "exclusive"):
            changed = True
            labs = [r.get("a", ""), r.get("b", "")] if t == "excludes" else list(r.get("labels") or [])
            out.append(new_rule("never", labels=[x for x in labs if x], **keep))
        elif t in TYPES:
            out.append(r)
        else:
            changed = True                            # unknown type: dropped
    seen, uniq = set(), []
    for r in out:
        s = signature(r) + r.get("severity", "error")
        if s in seen:
            changed = True
            continue
        seen.add(s)
        uniq.append(r)
    return uniq, changed


def load_rules(path) -> list[dict]:
    try:
        d = json.loads(open(path, encoding="utf-8").read())
    except (OSError, ValueError):
        return []
    rules, changed = upgrade_rules([r for r in d.get("rules", []) if isinstance(r, dict)])
    if changed:
        try:
            save_rules(path, rules)                   # converted once, for every tool that reads the file
        except OSError:
            pass
    return rules


def save_rules(path, rules: list[dict]):
    atomic_write_text(path, json.dumps({"version": 2, "rules": rules}, indent=1, ensure_ascii=False))


# =============================================================================
# Checking
# =============================================================================
@dataclass
class Violation:
    rule: dict
    text: str
    shapes: list = field(default_factory=list)       # shape indices involved (first = the one to show)
    fix: dict | None = None    # proposed correction: {"title", "op": "set" | "delete", "shape", "points", "type"}
                               # points in the shapes' own coordinates (pixels, or 0..1 when normalized)
    info: dict = field(default_factory=dict)   # what was measured: iou, inside, size, angle, count, … (measures())


def measures(info: dict) -> str:
    """The measured values of a violation, short: 'IoU 0.82', '107 % size · 87 % inside', '3 (expected 1–2)' …"""
    out = []
    if "size" in info:
        out.append(f"{info['size'] * 100:.0f} % size")
    if "inside" in info:
        out.append(f"{info['inside'] * 100:.0f} % inside")
    if "touch" in info:
        out.append(f"{info['touch'] * 100:.0f} % on")
    if "iou" in info:
        out.append(f"IoU {info['iou']:.2f}")
    if "angle" in info:
        out.append(f"{info['angle']:.0f}° off")
    if "tilt" in info:
        out.append(f"tilt {info['tilt']:.0f}°")
    if "count" in info:
        out.append(f"{info['count']}" + (f" (expected {info['expected']})" if info.get("expected") else ""))
    if "w" in info:
        out.append(f"{info['w']:.0f}×{info['h']:.0f} px" + (f" · {info['pct']:.2g} %" if "pct" in info else ""))
    if "aspect" in info:
        out.append(f"w/h {info['aspect']:.2f}")
    if "verts" in info:
        out.append(f"{info['verts']} vertices")
    return " · ".join(out)


def _boxes(shapes) -> np.ndarray:
    out = np.zeros((len(shapes), 4), np.float64)
    for i, s in enumerate(shapes):
        pts = s.get("points") or []
        if not pts:
            out[i] = (0, 0, 0, 0)
            continue
        a = np.asarray(pts, np.float64).reshape(-1, 2)
        if s.get("shape_type") == "circle" and len(a) >= 2:
            r = float(np.hypot(*(a[1] - a[0])))
            out[i] = (a[0][0] - r, a[0][1] - r, a[0][0] + r, a[0][1] + r)
        else:
            out[i] = (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
    return out


def _inter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def _touch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[n_a, n_b]: boxes overlap or touch (closed intervals — points and lines have zero-area boxes)."""
    return ((a[:, None, 0] <= b[None, :, 2]) & (b[None, :, 0] <= a[:, None, 2]) &
            (a[:, None, 1] <= b[None, :, 3]) & (b[None, :, 1] <= a[:, None, 3]))


def _area(a: np.ndarray) -> np.ndarray:
    return np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)


def _iou(a, b):
    inter = _inter(a, b)
    union = _area(a)[:, None] + _area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _frac_inside(child, parent):
    """[n_child, n_parent]: fraction of each child's box area inside each parent's box."""
    ca = _area(child)
    inter = _inter(child, parent)
    return np.where(ca[:, None] > 0, inter / np.maximum(ca[:, None], 1e-12), 0.0)


class _Geo:
    """Per-image geometry, computed lazily per shape: outline polygon, its area and convexity, and the long /
    short side of its minimum-area rectangle."""

    def __init__(self, shapes, boxes):
        self.shapes, self.boxes = shapes, boxes
        self._poly, self._dims = {}, {}

    def poly(self, i):
        """(N, 2) float32 outline of an area shape, convex flag, area — or (None, False, box area)."""
        if i in self._poly:
            return self._poly[i]
        s = self.shapes[i]
        st = s.get("shape_type") or "polygon"
        pts = np.asarray(s.get("points") or [], np.float32).reshape(-1, 2)
        if st == "rectangle" and len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[0], pts[1]
            x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
        elif st == "circle" and len(pts) >= 2:
            r = float(np.hypot(*(pts[1] - pts[0])))
            a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            poly = np.stack([pts[0][0] + r * np.cos(a), pts[0][1] + r * np.sin(a)], 1).astype(np.float32)
        elif st == "polygon" and len(pts) >= 3:
            poly = pts
        else:
            poly = None
        if poly is not None and cv2 is not None:
            area = abs(float(cv2.contourArea(poly)))
            convex = area > 0 and bool(cv2.isContourConvex(poly))
        else:
            b = self.boxes[i]
            area, convex = float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])), False
        self._poly[i] = (poly, convex, area)
        return self._poly[i]

    def frac_inside(self, c, p) -> float:
        """Share of shape c inside shape p: its area (exact for convex outlines, else bounding boxes), or for points
        and lines the share of their points."""
        bc, bp = self.boxes[c], self.boxes[p]
        pc, cc, ac = self.poly(c)
        pp, cp, _ap = self.poly(p)
        if pc is None:                                # point(s) / line: count the vertices inside the parent
            pts = np.asarray(self.shapes[c].get("points") or [], np.float32).reshape(-1, 2)
            if not len(pts):
                return 0.0
            if pp is not None and cv2 is not None:
                inside = [cv2.pointPolygonTest(pp, (float(x), float(y)), False) >= 0 for x, y in pts]
            else:
                inside = [bp[0] <= x <= bp[2] and bp[1] <= y <= bp[3] for x, y in pts]
            return float(sum(inside)) / len(pts)
        bi = max(0.0, min(bc[2], bp[2]) - max(bc[0], bp[0])) * max(0.0, min(bc[3], bp[3]) - max(bc[1], bp[1]))
        if bi <= 0:
            return 0.0
        if cv2 is not None and cc and cp and ac > 0:
            inter, _ = cv2.intersectConvexConvex(pc, pp)
            return float(min(1.0, max(0.0, inter) / ac))
        ab = max(1e-12, (bc[2] - bc[0]) * (bc[3] - bc[1]))
        return float(min(1.0, bi / ab))

    def angle(self, i) -> float | None:
        """Direction of the long side in degrees, 0..180 (None: no outline, or nearly square)."""
        poly, _c, _a = self.poly(i)
        if poly is None or cv2 is None or len(poly) < 3:
            return None
        (_cx, _cy), (w, h), ang = cv2.minAreaRect(poly)
        if min(w, h) <= 0 or max(w, h) / min(w, h) < 1.15:
            return None                               # a square has no direction
        return float((ang if w >= h else ang + 90.0) % 180.0)

    def clip_to(self, c, p) -> list | None:
        """Part c's outline moved inside parent p, vertex count kept: each vertex outside the parent goes to the
        nearest point of the parent's outline (convex parent: the result is inside). Rectangles: box ∩ parent's
        box. None when there is nothing sensible to propose."""
        s = self.shapes[c]
        st = s.get("shape_type") or "polygon"
        pts = np.asarray(s.get("points") or [], np.float64).reshape(-1, 2)
        bp = self.boxes[p]
        if st == "rectangle" and len(pts) >= 2:
            x1, y1 = max(pts[:, 0].min(), bp[0]), max(pts[:, 1].min(), bp[1])
            x2, y2 = min(pts[:, 0].max(), bp[2]), min(pts[:, 1].max(), bp[3])
            return [[float(x1), float(y1)], [float(x2), float(y2)]] if x2 > x1 and y2 > y1 else None
        pp, convex, _a = self.poly(p)
        if st not in ("polygon", "point", "points", "line", "linestrip") or not len(pts):
            return None
        if pp is None or not convex or cv2 is None:
            q = np.stack([np.clip(pts[:, 0], bp[0], bp[2]), np.clip(pts[:, 1], bp[1], bp[3])], 1)
            return [[float(x), float(y)] for x, y in q]
        out = []
        n = len(pp)
        for x, y in pts:
            if cv2.pointPolygonTest(pp, (float(x), float(y)), False) >= 0:
                out.append([float(x), float(y)])
                continue
            best, bd = None, 1e18                     # nearest point on the parent's edges
            for k in range(n):
                a, b = pp[k].astype(np.float64), pp[(k + 1) % n].astype(np.float64)
                ab = b - a
                tt = 0.0 if not ab.any() else float(np.clip(np.dot([x, y] - a, ab) / np.dot(ab, ab), 0, 1))
                q = a + tt * ab
                d = float((q[0] - x) ** 2 + (q[1] - y) ** 2)
                if d < bd:
                    best, bd = q, d
            out.append([float(best[0]), float(best[1])])
        return out

    def dims(self, i) -> tuple[float, float]:
        """(long side, short side): of the minimum-area rectangle for outlines (rotation-independent)."""
        if i in self._dims:
            return self._dims[i]
        poly, _c, _a = self.poly(i)
        if poly is not None and cv2 is not None and len(poly) >= 3:
            (_cx, _cy), (w, h), _ang = cv2.minAreaRect(poly)
        else:
            b = self.boxes[i]
            w, h = b[2] - b[0], b[3] - b[1]
        d = (float(max(w, h)), float(min(w, h)))
        self._dims[i] = d
        return d


def _angle_diff(a, b) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _check_part(r, ci, pi, geo, boxes, out, own=lambda pts: pts):
    """ci / pi: indices of the parts / possible parents (a shape is never its own parent).
    own(points in pixels) -> points in the shapes' own coordinates (for the proposed fixes)."""
    g = r.get
    child, parent = cls(g("child")), cls(g("parent"))
    min_frac, max_rel = float(g("min_frac", 0.9) or 0), float(g("max_rel", 0) or 0)
    min_rel, max_angle = float(g("min_rel", 0) or 0), float(g("max_angle", 0) or 0)
    need, mn, mx, apart = bool(g("need_parent", True)), int(g("min", 0) or 0), int(g("max", 0) or 0), \
        float(g("apart_iou", 0) or 0)
    assigned = defaultdict(list)                     # parent index -> its parts
    if ci and pi:
        touch = _touch(boxes[ci], boxes[pi])         # candidates by bounding boxes (vectorised)
        for a, c in enumerate(ci):
            cand = [pi[b] for b in np.nonzero(touch[a])[0] if pi[b] != c]
            best, frac = None, 0.0
            for p in cand:
                f = geo.frac_inside(c, p)
                if f > frac:
                    best, frac = p, f
            if best is not None and frac >= min_frac:
                assigned[best].append(c)
            problems, fix, clip, info = [], None, False, {}
            if best is not None and (max_rel or min_rel) and geo.poly(c)[0] is not None:
                (cl, cs), (pl, ps) = geo.dims(c), geo.dims(best)
                rl, rs = (cl / pl if pl else 0.0), (cs / ps if ps else 0.0)
                if max_rel and (rl > max_rel + 1e-6 or rs > max_rel + 1e-6):
                    info["size"] = max(rl, rs)
                    problems.append(f"larger than {parent} #{best} ({rl * 100:.0f} % × {rs * 100:.0f} % of it)")
                    clip = True
                elif min_rel and (rl < min_rel - 1e-6 and rs < min_rel - 1e-6):
                    info["size"] = max(rl, rs)
                    problems.append(f"too small for {parent} #{best} ({rl * 100:.0f} % × {rs * 100:.0f} %)")
            if best is not None and max_angle:
                ac, ap = geo.angle(c), geo.angle(best)
                if ac is not None and ap is not None and _angle_diff(ac, ap) > max_angle:
                    info["angle"] = _angle_diff(ac, ap)
                    problems.append(f"turned {_angle_diff(ac, ap):.0f}° against {parent} #{best}")
            if best is not None:
                info["inside"] = frac
            if need and frac < min_frac:
                problems.append(f"only {frac * 100:.0f} % inside {parent} #{best}" if best is not None
                                else f"not on any {parent}")
                clip = clip or best is not None
            if clip and best is not None:
                pts = geo.clip_to(c, best)
                if pts is not None:
                    fix = {"title": f"Fit inside {parent} #{best}", "op": "set", "shape": c, "points": own(pts)}
            elif need and best is None:
                fix = {"title": f"Delete {child} #{c}", "op": "delete", "shape": c}
            if problems:
                out.append(Violation(r, f"{child} #{c}: " + " and ".join(problems),
                                     [c] + ([best] if best is not None else []), fix, info))
    elif ci and need:
        for c in ci:
            out.append(Violation(r, f"{child} #{c}: not on any {parent} (the image has none)", [c]))
    if mn or mx:
        for p in pi:
            kids = assigned.get(p, [])
            n = len(kids)
            if n < mn or (mx and n > mx):
                exp = f"{mn}" if mn == mx else (f"at least {mn}" if not mx else f"{mn}–{mx}")
                out.append(Violation(r, f"{parent} #{p} has {n} {child} (expected {exp})", [p] + kids,
                                     info={"count": n, "expected": exp}))
    if apart:
        for p, kids in assigned.items():
            if len(kids) > 1:
                iou = _iou(boxes[kids], boxes[kids])
                np.fill_diagonal(iou, 0)
                a, b = np.unravel_index(np.argmax(iou), iou.shape)
                if iou[a, b] > apart:
                    out.append(Violation(r, f"{child} #{kids[a]} and #{kids[b]} on {parent} #{p} overlap "
                                            f"(IoU {iou[a, b]:.2f})", [kids[a], kids[b], p],
                                         _dup_fix(geo, kids[a], kids[b], iou[a, b]), {"iou": float(iou[a, b])}))


def _dup_fix(geo, i, j, iou) -> dict | None:
    """Two shapes that are almost the same object (IoU ≥ 0.7): propose deleting the smaller one."""
    if iou < 0.7:
        return None
    small = i if geo.poly(i)[2] <= geo.poly(j)[2] else j
    return {"title": f"Delete the smaller one (#{small})", "op": "delete", "shape": small}


def _retype_fix(shape: dict, allowed: set) -> dict | None:
    """rectangle <-> polygon, the only lossless-enough conversions (a polygon becomes its bounding box)."""
    st = shape.get("shape_type") or "polygon"
    pts = shape.get("points") or []
    if st == "polygon" and "rectangle" in allowed and len(pts) >= 3:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return {"title": "Make it a rectangle (its box)", "op": "set", "shape": None, "type": "rectangle",
                "points": [[min(xs), min(ys)], [max(xs), max(ys)]]}
    if st == "rectangle" and "polygon" in allowed and len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[0], pts[1]
        return {"title": "Make it a 4-point polygon", "op": "set", "shape": None, "type": "polygon",
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}
    return None


def check_image(rules: list[dict], shapes: list[dict], tags: list[str], W: float | None, H: float | None,
                normalized: bool = False) -> list[Violation]:
    """W, H: image size in px (needed only by size / aspect / border rules); normalized: shapes are 0..1."""
    rules = [r for r in rules if r.get("enabled", True)]
    if not rules:
        return []
    labels = [str(s.get("label", "")) for s in shapes]
    boxes = _boxes(shapes)
    if normalized and W and H and len(boxes):
        px = boxes * np.array([W, H, W, H])
    else:
        px = boxes
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, l in enumerate(labels):
        by_label[l].append(i)
    present = Counter(labels)
    present.update(tags)
    out: list[Violation] = []
    sel = lambda lab: list(range(len(shapes))) if is_any(lab) else by_label.get(lab, [])

    def members(lab, other):
        """Shapes of class `lab` (* = any class but `other`, so 'any object inside a region' skips the regions)."""
        if not is_any(lab):
            return by_label.get(lab, [])
        return [i for i, l in enumerate(labels) if is_any(other) or l != other]
    geo = None

    for r in rules:
        t, g = r["type"], r.get
        if t == "part":
            ci, pi = members(g("child"), g("parent")), members(g("parent"), g("child"))
            if not ci and not (g("min") and pi):
                continue
            if geo is None:
                if normalized and W and H:          # exact geometry in pixels (aspect matters for shapes)
                    sh = [dict(s, points=[[p[0] * W, p[1] * H] for p in (s.get("points") or [])]) for s in shapes]
                    geo = _Geo(sh, px)
                else:
                    geo = _Geo(shapes, boxes)
            own = (lambda pts: [[x / W, y / H] for x, y in pts]) if normalized and W and H else (lambda pts: pts)
            _check_part(r, ci, pi, geo, geo.boxes, out, own)
        elif t == "touch":
            ai, bi = members(g("a"), g("b")), members(g("b"), g("a"))
            if not ai:
                continue
            if geo is None:
                geo = _Geo(shapes, boxes) if not (normalized and W and H) else _Geo(
                    [dict(s, points=[[p[0] * W, p[1] * H] for p in (s.get("points") or [])]) for s in shapes], px)
            need = float(g("min_frac", 0) or 0)
            touch = _touch(geo.boxes[ai], geo.boxes[bi]) if bi else np.zeros((len(ai), 0), bool)
            for x, a in enumerate(ai):
                cand = [bi[y] for y in np.nonzero(touch[x])[0] if bi[y] != a]
                best = max((geo.frac_inside(a, b) for b in cand), default=0.0)
                if best <= 0 or best < need:
                    out.append(Violation(r, f"{labels[a]} #{a} " + (f"touches no {cls(g('b'))}" if best <= 0 else
                                         f"has only {best * 100:.0f} % on a {cls(g('b'))}"), [a],
                                         info={"touch": best}))
        elif t == "ratio":
            na, nb = present.get(g("a"), 0), present.get(g("b"), 0)
            per = float(g("per", 1) or 1)
            if na > per * nb + 1e-9:
                out.append(Violation(r, f"{na} {g('a')} but {nb} {g('b')}", by_label.get(g("a"), [])[:1],
                                     info={"count": na, "expected": f"≤ {per * nb:g}"}))
        elif t == "tilt":
            if geo is None:
                geo = _Geo(shapes, boxes) if not (normalized and W and H) else _Geo(
                    [dict(s, points=[[p[0] * W, p[1] * H] for p in (s.get("points") or [])]) for s in shapes], px)
            lim = float(g("max_deg", 30) or 0)
            for k in sel(g("label")):
                a = geo.angle(k)
                if a is not None and lim and min(a, 180 - a) > lim:
                    out.append(Violation(r, f"{labels[k]} #{k} turned {min(a, 180 - a):.0f}°", [k],
                                         info={"tilt": min(a, 180 - a)}))
        elif t == "overlap":
            ai, bi = sel(g("a")), sel(g("b"))
            if not ai or not bi:
                continue
            iou = _iou(boxes[ai], boxes[bi])
            same = g("a") == g("b") or (is_any(g("a")) and is_any(g("b")))
            if same:
                iou = np.triu(iou, 1)
            iou[np.asarray(ai)[:, None] == np.asarray(bi)[None, :]] = 0      # never a shape with itself
            seen = set()
            for x, y in zip(*np.nonzero(iou > g("max_iou", 0.5))):
                pair = (min(ai[x], bi[y]), max(ai[x], bi[y]))
                if pair in seen:                                            # * on one side: each pair once
                    continue
                seen.add(pair)
                fix = None
                if labels[ai[x]] == labels[bi[y]]:
                    if geo is None:
                        geo = _Geo(shapes, boxes)
                    fix = _dup_fix(geo, ai[x], bi[y], iou[x, y])
                out.append(Violation(r, f"{labels[ai[x]]} #{ai[x]} and {labels[bi[y]]} #{bi[y]} overlap "
                                        f"(IoU {iou[x, y]:.2f})", [ai[x], bi[y]], fix, {"iou": float(iou[x, y])}))
        elif t == "count":
            lab = g("label")
            n = len(shapes) if lab in ("", ANY) else present.get(lab, 0)
            mx = g("max", 0)
            if n < g("min", 0) or (mx and n > mx):
                out.append(Violation(r, f"{n} {lab if lab not in ('', ANY) else 'objects'} "
                                        f"(expected {g('min')}–{mx or '∞'})", sel(lab)[:1],
                                     info={"count": n, "expected": f"{g('min')}–{mx or '∞'}"}))
        elif t == "requires":
            if present.get(g("a")) and not present.get(g("b")):
                out.append(Violation(r, f"has {g('a')} but no {g('b')}", by_label.get(g("a"), [])[:1]))
        elif t == "never":
            hit = [l for l in (g("labels") or []) if present.get(l)]
            if len(hit) > 1:
                out.append(Violation(r, f"{' and '.join(hit)} in the same image",
                                     [by_label[l][0] for l in hit if by_label.get(l)][:2]))
        elif t == "allowed":
            ok = set(g("labels") or [])
            bad = sorted(l for l in present if l not in ok)
            if ok and bad:
                out.append(Violation(r, f"class(es) not allowed: {', '.join(bad)}",
                                     [i for i, l in enumerate(labels) if l in bad][:1]))
        elif t == "size":
            if W is None or H is None:
                continue
            for k in sel(g("label")):
                w, h = px[k, 2] - px[k, 0], px[k, 3] - px[k, 1]
                pct = 100.0 * w * h / max(W * H, 1)
                why = []
                if (g("min_w") and w < g("min_w")) or (g("min_h") and h < g("min_h")):
                    why.append(f"{w:.0f}×{h:.0f} px too small")
                if (g("max_w") and w > g("max_w")) or (g("max_h") and h > g("max_h")):
                    why.append(f"{w:.0f}×{h:.0f} px too large")
                if g("min_area_pct") and pct < g("min_area_pct"):
                    why.append(f"{pct:.2f} % of the image")
                if g("max_area_pct") and pct > g("max_area_pct"):
                    why.append(f"{pct:.1f} % of the image")
                if why:
                    out.append(Violation(r, f"{labels[k]} #{k}: " + ", ".join(why), [k],
                                         info={"w": float(w), "h": float(h), "pct": float(pct)}))
        elif t == "aspect":
            for k in sel(g("label")):
                w, h = px[k, 2] - px[k, 0], px[k, 3] - px[k, 1]
                if h <= 0:
                    continue
                ar = w / h
                if (g("min") and ar < g("min")) or (g("max") and ar > g("max")):
                    out.append(Violation(r, f"{labels[k]} #{k}: width/height {ar:.2f}", [k], info={"aspect": float(ar)}))
        elif t == "shape_type":
            allowed = set(g("types") or [])
            for k in sel(g("label")):
                st = shapes[k].get("shape_type") or "polygon"
                if allowed and st not in allowed:
                    out.append(Violation(r, f"{labels[k]} #{k} is a {st}", [k], _retype_fix(shapes[k], allowed)))
        elif t == "points":
            for k in sel(g("label")):
                if (shapes[k].get("shape_type") or "polygon") != "polygon":
                    continue
                n = len(shapes[k].get("points") or [])
                if n < g("min", 3) or (g("max") and n > g("max")):
                    out.append(Violation(r, f"{labels[k]} #{k} has {n} vertices", [k], info={"verts": n}))
        elif t == "border":
            if W is None or H is None:
                continue
            m = g("margin_px", 1)
            for k in sel(g("label")):
                x1, y1, x2, y2 = px[k]
                if x1 <= m or y1 <= m or x2 >= W - m or y2 >= H - m:
                    out.append(Violation(r, f"{labels[k]} #{k} touches the image border", [k]))
        elif t == "attribute":
            v = g("value", "any")
            for k in sel(g("label")):
                flags = shapes[k].get("flags") or {}
                if g("flag") not in flags or (v != "any" and str(flags.get(g("flag"))).lower() != str(v).lower()):
                    out.append(Violation(r, f"{labels[k]} #{k}: attribute “{g('flag')}” "
                                            f"{'missing' if g('flag') not in flags else '= ' + str(flags.get(g('flag')))}", [k]))
        elif t == "group":
            need = set(g("labels") or [])
            groups = defaultdict(list)
            for i, s in enumerate(shapes):
                if s.get("group_id") is not None:
                    groups[s.get("group_id")].append(i)
            for gid, members in groups.items():
                miss = need - {labels[i] for i in members}
                if need and miss:
                    out.append(Violation(r, f"group {gid} has no {', '.join(sorted(miss))}", members[:1]))
        elif t == "one_class":
            n = len(tags)
            if n != 1:
                out.append(Violation(r, "no class" if n == 0 else f"{n} classes: {', '.join(tags)}", []))
    for v in out:                                       # a fix without a target acts on the violation's shape
        if v.fix is not None and v.fix.get("shape") is None and v.shapes:
            v.fix["shape"] = v.shapes[0]
    return out


# =============================================================================
# Mining suggestions from the dataset
# =============================================================================
class _Res:
    """Bounded sample of values (reservoir): percentiles of millions of shapes from a few thousand numbers."""
    __slots__ = ("vals", "n", "cap", "rng")

    def __init__(self, cap=4000):
        import random
        self.vals, self.n, self.cap, self.rng = [], 0, cap, random.Random(0)

    def add(self, v):
        self.n += 1
        if len(self.vals) < self.cap:
            self.vals.append(float(v))
        else:
            j = self.rng.randrange(self.n)
            if j < self.cap:
                self.vals[j] = float(v)

    def q(self, p) -> float | None:
        return float(np.percentile(self.vals, p)) if self.vals else None


class RuleStats:
    """Feed every image (or a large sample) once; then suggest() (whole rules) or recommend() / rank() (the choices
    in a rule form: likely classes and values, each with its evidence)."""

    def __init__(self):
        self.images = 0
        self.m = defaultdict(lambda: defaultdict(_Res))   # label -> metric -> sample (w, h, area_pct, aspect, tilt, verts)
        self.pm = defaultdict(lambda: defaultdict(_Res))  # (a, b) -> metric -> sample (frac, rel_max, rel_min, angle,
        #                                                   touch_frac, iou, ratio)
        self.flags = defaultdict(Counter)                  # label -> attribute -> instances having it
        self.flag_vals = defaultdict(lambda: defaultdict(Counter))   # label -> attribute -> value -> n
        self.border = Counter()                            # label -> instances touching the image border
        self.sized = Counter()                             # label -> instances measured in pixels
        self.img_with = Counter()                 # label -> images containing it
        self.pair_with = Counter()                # (a, b) -> images containing both
        self.inst = Counter()                     # label -> instances
        self.inside = Counter()                   # (child, parent) -> child instances inside some parent
        self.not_larger = Counter()               # (child, parent) -> of those, not larger than that parent
        self.touching = Counter()                 # (a, b) -> a instances overlapping some b
        self.at_most = Counter()                  # (a, b) -> images with a where #a <= #b
        self.kids = defaultdict(Counter)          # (parent, child) -> Counter(#children per parent)
        self.max_per_img = defaultdict(Counter)   # label -> Counter(count per image, images with it)
        self.types = defaultdict(Counter)         # label -> Counter(shape_type)
        self.tags_per_img = Counter()

    def add(self, shapes, tags, W=None, H=None, normalized=False):
        """W, H: image size (pixel measures need it); normalized: shapes are 0..1 (YOLO)."""
        self.images += 1
        if normalized and W and H:
            shapes = [dict(s, points=[[p[0] * W, p[1] * H] for p in (s.get("points") or [])]) for s in shapes]
        labels = [str(s.get("label", "")) for s in shapes]
        present = set(labels) | set(tags)
        self.tags_per_img[len(tags)] += 1
        for l in present:
            self.img_with[l] += 1
        pl = sorted(present)
        for i, a in enumerate(pl):
            for b in pl[i + 1:]:
                self.pair_with[(a, b)] += 1
        c = Counter(labels)
        for l, n in c.items():
            self.inst[l] += n
            self.max_per_img[l][n] += 1
        for a, na in c.items():
            for b, nb in c.items():
                if a != b and na <= nb:
                    self.at_most[(a, b)] += 1
        for s in shapes:
            self.types[str(s.get("label", ""))][s.get("shape_type") or "polygon"] += 1
        for l, n in c.items():                                 # per image counts of every class (incl. 0 later)
            self.m[l]["count"].add(n)
        if not shapes:
            return
        boxes = _boxes(shapes)
        wh = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
        dims = np.stack([wh.max(1), wh.min(1)], 1)         # long, short side
        idx = defaultdict(list)
        for i, l in enumerate(labels):
            idx[l].append(i)
        geo = _Geo(shapes, boxes)
        for i, (s, l) in enumerate(zip(shapes, labels)):    # per object
            mm = self.m[l]
            w, h = wh[i]
            if h > 0:
                mm["aspect"].add(w / h)
            st = s.get("shape_type") or "polygon"
            if st == "polygon":
                mm["verts"].add(len(s.get("points") or []))
                a = geo.angle(i)
                if a is not None:
                    mm["tilt"].add(min(a, 180 - a))
            for k, v in (s.get("flags") or {}).items():
                self.flags[l][k] += 1
                self.flag_vals[l][k][str(v).lower()] += 1
            if W and H:
                self.sized[l] += 1
                mm["w"].add(w)
                mm["h"].add(h)
                mm["area_pct"].add(100.0 * w * h / max(W * H, 1))
                x1, y1, x2, y2 = boxes[i]
                if x1 <= 1 or y1 <= 1 or x2 >= W - 1 or y2 >= H - 1:
                    self.border[l] += 1
        for l, li in idx.items():                           # same class: how much do they overlap?
            if len(li) > 1:
                iou = np.triu(_iou(boxes[li], boxes[li]), 1)
                self.pm[(l, l)]["iou"].add(float(iou.max()))
        for a, na in c.items():
            for b, nb in c.items():
                if a != b:
                    self.pm[(a, b)]["ratio"].add(na / max(nb, 1))
        if len(c) < 2:
            return
        for child, ci in idx.items():
            for parent, pi in idx.items():
                if child == parent:
                    continue
                frac = _frac_inside(boxes[ci], boxes[pi])
                self.touching[(child, parent)] += int((frac.max(axis=1) > 0).sum())
                best = frac.argmax(axis=1)
                ok = frac.max(axis=1) >= 0.9
                pmx = self.pm[(child, parent)]
                for a in range(len(ci)):
                    f = float(frac[a].max())
                    if f > 0:
                        pmx["touch_frac"].add(f)
                    if f >= 0.5:                               # mostly on a parent: how far in, how big, how turned
                        c_, p_ = ci[a], pi[best[a]]
                        pmx["frac"].add(f)
                        (cl, cs), (pl, ps) = dims[c_], dims[p_]
                        if pl > 0 and ps > 0:
                            pmx["rel_max"].add(max(cl / pl, cs / ps))
                            pmx["rel_min"].add(min(cl / pl, cs / ps))
                        ac, ap = geo.angle(c_), geo.angle(p_)
                        if ac is not None and ap is not None:
                            pmx["angle"].add(_angle_diff(ac, ap))
                if len(ci) and len(pi):
                    pmx["iou"].add(float(_iou(boxes[ci], boxes[pi]).max()))
                self.inside[(child, parent)] += int(ok.sum())
                if ok.any():
                    cd, pdm = dims[np.asarray(ci)[ok]], dims[np.asarray(pi)[best[ok]]]
                    self.not_larger[(child, parent)] += int(((cd <= pdm * 1.0001).all(axis=1)).sum())
                assign = np.zeros(len(pi), int)               # parts per parent (each part counted once)
                for a in np.nonzero(ok)[0]:
                    assign[best[a]] += 1
                self.kids[(parent, child)].update(int(x) for x in assign)
                kid_idx = defaultdict(list)                       # siblings on one parent: how much they overlap
                for a in np.nonzero(ok)[0]:
                    kid_idx[best[a]].append(ci[a])
                for kids in kid_idx.values():
                    if len(kids) > 1:
                        iou = np.triu(_iou(boxes[kids], boxes[kids]), 1)
                        self.pm[(child, parent)]["apart"].add(float(iou.max()))


@dataclass
class Suggestion:
    rule: dict
    evidence: str
    strength: float           # share of the data that already follows it


def suggest(st: RuleStats, min_support: int = 20, hold: float = 0.97) -> list[Suggestion]:
    """Rules that already hold for >= `hold` of the data. One Parent / part rule per pair (inside + relative size
    + parts per parent together); same-class overlaps are left to Duplicate Shapes."""
    out = []
    labels = [l for l, n in st.img_with.items() if n >= min_support]
    parts, one_each = set(), set()           # pairs with a part rule · … whose parents hold exactly one part
    for (child, parent), n_in in st.inside.items():
        tot = st.inst[child]
        if tot < min_support or n_in / tot < hold or st.inst[parent] < min_support:
            continue
        parts.add((child, parent))
        rule = new_rule("part", child=child, parent=parent)
        ev = [f"{n_in:,} / {tot:,} {child} lie inside a {parent}"]
        small = st.not_larger[(child, parent)] / max(n_in, 1)
        if small < hold:
            rule["max_rel"] = 0.0
        else:
            ev.append(f"{small * 100:.1f} % no larger than it")
        dist = st.kids[(parent, child)]
        total_p = sum(dist.values())
        if total_p >= min_support and dist.get(0, 0) / total_p <= 1 - hold:
            cum, hi = 0, 0
            for k in sorted(dist):                           # 99.5th percentile of the count
                cum += dist[k]
                hi = k
                if cum / total_p >= 0.995:
                    break
            lo = min(k for k in dist if k >= 1)
            share = sum(v for k, v in dist.items() if lo <= k <= hi) / total_p
            if share >= hold:
                rule["min"], rule["max"] = lo, hi
                ev.append(f"{share * 100:.1f} % of {total_p:,} {parent} have {lo}–{hi}")
        if rule["max"] == 1:
            one_each.add((child, parent))
        out.append(Suggestion(rule, " · ".join(ev), n_in / tot))
    # max count per image
    for l in labels:
        dist = st.max_per_img[l]
        imgs = sum(dist.values())
        share_one = dist.get(1, 0) / imgs
        if share_one >= hold:                                # also when it already always holds: guards new data
            out.append(Suggestion(new_rule("count", label=l, min=0, max=1, severity="warn"),
                                  f"{share_one * 100:.1f} % of images with {l} have exactly one", share_one))
    # touching without lying inside
    for (a, b), n in st.touching.items():
        tot = st.inst[a]
        if (a, b) in parts or tot < min_support or st.inst[b] < min_support or n / tot < hold:
            continue
        parts.add((a, b))                                    # implies "requires" as well
        out.append(Suggestion(new_rule("touch", a=a, b=b, severity="warn"), f"{n:,} / {tot:,} {a} touch a {b}",
                              n / tot))
    # no more A than B per image
    implied = set(parts)
    for (a, b), n in st.at_most.items():
        imgs = st.img_with[a]
        if a in labels and b in labels and imgs >= min_support and n / imgs >= hold and (a, b) not in one_each \
                and st.max_per_img[a].get(1, 0) / max(1, sum(st.max_per_img[a].values())) < hold:
            implied.add((a, b))
            out.append(Suggestion(new_rule("ratio", a=a, b=b, per=1.0, severity="warn"),
                                  f"{n / imgs * 100:.1f} % of {imgs:,} images with {a} have no more {a} than {b}",
                                  n / imgs))
    # co-occurrence (not when a part / touch / ratio rule already implies it)
    for a in labels:
        for b in labels:
            if a == b or (a, b) in implied:
                continue
            both = st.pair_with[tuple(sorted((a, b)))]
            if st.img_with[a] >= min_support and both / st.img_with[a] >= hold:
                out.append(Suggestion(new_rule("requires", a=a, b=b), f"{both:,} / {st.img_with[a]:,} images with {a} "
                                      f"also have {b}", both / st.img_with[a]))
    # shape types
    for l, tc in st.types.items():
        tot = sum(tc.values())
        t, n = tc.most_common(1)[0]
        if tot >= min_support and n / tot >= 0.99:
            out.append(Suggestion(new_rule("shape_type", label=l, types=[t], severity="warn"),
                                  f"all {tot:,} {l} are {t}s" if n == tot else f"{n:,} / {tot:,} {l} are {t}s",
                                  n / tot))
    # classification
    if st.images and st.tags_per_img and sum(n for k, n in st.tags_per_img.items() if k > 0) / st.images >= 0.5:
        share = st.tags_per_img.get(1, 0) / st.images
        if share >= hold:
            out.append(Suggestion(new_rule("one_class"), f"{share * 100:.1f} % of images have exactly one class", share))
    out.sort(key=lambda s: (-s.strength, s.rule["type"]))
    return out


# =============================================================================
# Choices for a rule form, detected from the dataset
# =============================================================================
@dataclass
class Hint:
    value: object             # what to put in the field
    text: str                 # the evidence, e.g. "99.5 % of 1,204 wheel are ≤ 100 % of their car"


def _up(v, step):
    return math.ceil(v / step - 1e-9) * step


def _down(v, step):
    return math.floor(v / step + 1e-9) * step


def _n(x) -> str:
    return f"{x:,}"


LABEL_PARAMS = ("child", "parent", "a", "b", "label")


def rank(st: RuleStats, rtype: str, param: str, params: dict, top: int = 6) -> list[tuple[str, str]]:
    """Likely classes for a class field, best first: [(class, why)] — from what the data already does."""
    g = params.get
    have = lambda l: l in st.inst or l in st.img_with
    out = []
    if rtype == "part" and param == "parent" and have(g("child")):
        c = g("child")
        for (ch, p), n in st.inside.items():
            if ch == c and n:
                out.append((p, f"{n / max(st.inst[c], 1) * 100:.0f} % of {c} lie inside one", n))
    elif rtype == "part" and param == "child" and have(g("parent")):
        p = g("parent")
        for (ch, pa), n in st.inside.items():
            if pa == p and n:
                out.append((ch, f"{n / max(st.inst[ch], 1) * 100:.0f} % of {ch} lie inside a {p}", n))
    elif rtype == "touch" and param == "b" and have(g("a")):
        a = g("a")
        for (x, b), n in st.touching.items():
            if x == a and n:
                out.append((b, f"{n / max(st.inst[a], 1) * 100:.0f} % of {a} touch one", n))
    elif rtype in ("requires", "ratio", "overlap") and param == "b" and have(g("a")):
        a = g("a")
        for b in st.img_with:
            if b == a:
                continue
            both = st.pair_with[tuple(sorted((a, b)))]
            if both:
                out.append((b, f"in {both / max(st.img_with[a], 1) * 100:.0f} % of the images with {a}", both))
    elif param in LABEL_PARAMS:                         # any other class field: the most common classes
        for l, n in st.inst.items():
            out.append((l, f"{_n(n)} instances", n))
    out.sort(key=lambda x: -x[2])
    return [(l, why) for l, why, _n_ in out[:top]]


def recommend(st: RuleStats, rtype: str, params: dict) -> dict[str, Hint]:
    """Suggested values for the fields of a rule, given the classes already chosen — each with its evidence.
    Limits are set so ~99.5 % of the data passes (the rest is what the rule should find), rounded to friendly
    steps; nothing is suggested without enough data (≥ 10 values)."""
    g = params.get
    h: dict[str, Hint] = {}
    enough = lambda r: r is not None and len(r.vals) >= 10
    if rtype == "part":
        c, p = g("child"), g("parent")
        if st.inst.get(c) and st.inst.get(p) and st.inside.get((c, p), 0) / st.inst[c] < 0.5:
            share = st.inside.get((c, p), 0) / st.inst[c]
            alt = [l for l, _w in rank(st, "part", "parent", params, top=2) if l != p]
            return {"_note": Hint(None, f"Only {share * 100:.0f} % of {_n(st.inst[c])} {c} lie inside a {p} — "
                                        + (f"is {p} the right parent? Most {c} lie inside a {alt[0]}." if alt else
                                           f"{c} is rarely inside any other class."))}
        pm = st.pm.get((c, p)) if (c, p) in st.pm else None
        if pm is not None:
            fr = pm.get("frac")
            if enough(fr):
                lo = max(0.5, min(0.95, _down(fr.q(0.5), 0.05)))
                h["min_frac"] = Hint(round(lo, 2), f"99.5 % of {_n(fr.n)} {c} on a {p} are ≥ {fr.q(0.5) * 100:.0f} % "
                                                   "inside it")
            rmax = pm.get("rel_max")
            if enough(rmax):
                q = rmax.q(99.5)
                v = 1.0 if q <= 1.0 else round(_up(q, 0.05), 2)
                h["max_rel"] = Hint(v, f"99.5 % of {c} are ≤ {q * 100:.0f} % of their {p}'s size")
            rmin = pm.get("rel_min")
            if enough(rmin) and rmin.q(0.5) > 0.05:
                h["min_rel"] = Hint(round(_down(rmin.q(0.5), 0.05), 2), f"99.5 % of {c} are ≥ "
                                                                        f"{rmin.q(0.5) * 100:.0f} % of their {p}")
            ang = pm.get("angle")
            if enough(ang):
                h["max_angle"] = Hint(float(max(5, _up(ang.q(99.5), 5))), f"99.5 % of {c} are within "
                                                                          f"{ang.q(99.5):.0f}° of their {p}'s angle")
            ap = pm.get("apart")
            if enough(ap):
                h["apart_iou"] = Hint(round(min(0.9, max(0.05, _up(ap.q(99.5), 0.05))), 2),
                                      f"{c} on one {p} overlap at most IoU {ap.q(99.5):.2f} (99.5 %)")
        dist = st.kids.get((p, c))
        total = sum(dist.values()) if dist else 0
        if total >= 10:
            ks = sorted(dist)
            cum, hi = 0, ks[-1]
            for k in ks:
                cum += dist[k]
                if cum / total >= 0.995:
                    hi = k
                    break
            lo = next((k for k in ks if dist[k] / total >= 0.005), 0)
            if hi > 0:
                h["min"] = Hint(int(lo), f"99.5 % of {_n(total)} {p} have ≥ {lo} {c}" if lo else
                                f"{dist.get(0, 0) / total * 100:.1f} % of {_n(total)} {p} have no {c}")
                h["max"] = Hint(int(hi), f"99.5 % of {p} have ≤ {hi} {c}")
        inside = st.inside.get((c, p), 0)
        if st.inst.get(c):
            share = inside / st.inst[c]
            h["need_parent"] = Hint(share >= 0.9, f"{share * 100:.1f} % of {_n(st.inst[c])} {c} lie inside a {p}")
    elif rtype == "touch":
        a, b = g("a"), g("b")
        tf = st.pm[(a, b)].get("touch_frac") if (a, b) in st.pm else None
        if enough(tf):
            h["min_frac"] = Hint(round(_down(tf.q(0.5), 0.05), 2),
                                 f"99.5 % of the {a} touching a {b} have ≥ {tf.q(0.5) * 100:.0f} % on it")
    elif rtype == "overlap":
        a, b = g("a"), g("b")
        io = st.pm[(a, b)].get("iou") if (a, b) in st.pm else None
        if enough(io):
            h["max_iou"] = Hint(round(min(0.95, max(0.05, _up(io.q(99.5), 0.05))), 2),
                                f"99.5 % of images: {a} and {b} overlap at most IoU {io.q(99.5):.2f}")
    elif rtype == "count":
        l = g("label")
        cnt = st.m[l].get("count") if l in st.m else None
        if enough(cnt):
            share = st.img_with[l] / max(st.images, 1)
            h["min"] = Hint(1 if share >= 0.995 else 0, f"{l} is in {share * 100:.1f} % of {_n(st.images)} images")
            h["max"] = Hint(int(max(1, math.ceil(cnt.q(99.5)))), f"99.5 % of the images with {l} have ≤ "
                                                                  f"{math.ceil(cnt.q(99.5))}")
    elif rtype == "ratio":
        a, b = g("a"), g("b")
        rt = st.pm[(a, b)].get("ratio") if (a, b) in st.pm else None
        if enough(rt):
            h["per"] = Hint(float(max(1.0, _up(rt.q(99.5), 0.5))), f"99.5 % of images have ≤ {rt.q(99.5):.2g} "
                                                                   f"{a} per {b}")
    elif rtype in ("size", "aspect", "tilt", "points", "border", "shape_type", "attribute"):
        l = g("label")
        mm = st.m.get(l, {})
        if rtype == "size":
            for k, src, q, rnd in (("min_w", "w", 0.5, _down), ("min_h", "h", 0.5, _down),
                                   ("max_w", "w", 99.5, _up), ("max_h", "h", 99.5, _up)):
                r = mm.get(src)
                if enough(r):
                    v = int(rnd(r.q(q), 2 if r.q(q) < 50 else 10))
                    h[k] = Hint(max(0, v), f"{'99.5 % are ' + ('≥' if q < 50 else '≤')} {r.q(q):.0f} px")
            r = mm.get("area_pct")
            if enough(r):
                h["min_area_pct"] = Hint(round(max(0.0, r.q(0.5) * 0.9), 4), f"99.5 % cover ≥ {r.q(0.5):.3g} %")
                h["max_area_pct"] = Hint(round(min(100.0, r.q(99.5) * 1.1), 2), f"99.5 % cover ≤ {r.q(99.5):.3g} %")
        elif rtype == "aspect":
            r = mm.get("aspect")
            if enough(r):
                h["min"] = Hint(round(_down(r.q(0.5), 0.05), 2), f"99.5 % have width/height ≥ {r.q(0.5):.2f}")
                h["max"] = Hint(round(_up(r.q(99.5), 0.05), 2), f"99.5 % have width/height ≤ {r.q(99.5):.2f}")
        elif rtype == "tilt":
            r = mm.get("tilt")
            if enough(r):
                h["max_deg"] = Hint(float(max(5, _up(r.q(99.5), 5))), f"99.5 % of {l} polygons are turned ≤ "
                                                                       f"{r.q(99.5):.0f}°")
        elif rtype == "points":
            r = mm.get("verts")
            if enough(r):
                h["min"] = Hint(int(r.q(0.5)), f"99.5 % of {l} polygons have ≥ {r.q(0.5):.0f} vertices")
                h["max"] = Hint(int(math.ceil(r.q(99.5))), f"99.5 % have ≤ {math.ceil(r.q(99.5))}")
        elif rtype == "border" and st.sized.get(l):
            share = st.border[l] / st.sized[l]
            h["margin_px"] = Hint(1, f"{share * 100:.1f} % of {_n(st.sized[l])} {l} touch the border today")
        elif rtype == "shape_type" and st.types.get(l):
            tc = st.types[l]
            tot = sum(tc.values())
            keep = [t_ for t_, n in tc.most_common() if n / tot >= 0.01]
            h["types"] = Hint(keep, ", ".join(f"{n / tot * 100:.1f} % {t_}" for t_, n in tc.most_common(3)))
        elif rtype == "attribute" and st.flags.get(l):
            fl, n = st.flags[l].most_common(1)[0]
            h["flag"] = Hint(fl, f"{n / max(st.inst[l], 1) * 100:.0f} % of {l} have “{fl}”")
            vals = st.flag_vals[l][fl]
            v, nv = vals.most_common(1)[0]
            if v in ("true", "false") and nv / max(sum(vals.values()), 1) >= 0.97:
                h["value"] = Hint(v, f"{nv / sum(vals.values()) * 100:.1f} % are {v}")
    elif rtype == "allowed":
        labs = sorted(l for l, n in st.inst.items() if n >= 3) or sorted(st.img_with)
        h["labels"] = Hint(labs, f"the {len(labs)} classes in the data (≥ 3 instances)")
    elif rtype == "never":
        labs = sorted(st.img_with, key=lambda l: -st.img_with[l])[:30]
        apart = [(a, b) for i, a in enumerate(labs) for b in labs[i + 1:]
                 if st.img_with[a] >= 10 and st.img_with[b] >= 10 and not st.pair_with[tuple(sorted((a, b)))]]
        if apart:
            a, b = apart[0]
            h["labels"] = Hint([a, b], "never together in the data: " + ", ".join(f"{x} + {y}" for x, y in apart[:4]))
    return h
