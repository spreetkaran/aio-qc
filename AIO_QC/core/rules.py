# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Relationship rules: how annotations relate to each other and to the image — configured once per dataset
(.aio_qc/rules.json), checked in seconds, no model needed. Rule = {"type", "enabled", "severity", …params}.

    rules = load_rules(root);  violations = check_image(rules, shapes, tags, W, H)
    suggestions = suggest(stats)  # mined from the dataset: rules that hold for >= 97 % of it, with evidence

Geometry uses axis-aligned bounding boxes of the shapes (fast, vectorised); "inside" means a fraction of the
child's box area lies inside the parent's box.
"""
from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .io import atomic_write_text

# type -> (title, group, params with defaults, needs pixels)
TYPES = {
    "inside": ("Must be inside", "objects", {"child": "", "parent": "", "min_frac": 0.9}, False),
    "contains": ("Must contain", "objects", {"parent": "", "child": "", "min": 1, "max": 99, "min_frac": 0.9}, False),
    "overlap": ("No overlap", "objects", {"a": "", "b": "", "max_iou": 0.5}, False),
    "siblings": ("Children apart", "objects", {"parent": "", "child": "", "max_iou": 0.1}, False),
    "count": ("Count per image", "image", {"label": "", "min": 0, "max": 1}, False),
    "requires": ("Requires", "image", {"a": "", "b": ""}, False),
    "excludes": ("Never together", "image", {"a": "", "b": ""}, False),
    "allowed": ("Allowed classes", "image", {"labels": []}, False),
    "size": ("Size limits", "object", {"label": "", "min_w": 0, "min_h": 0, "max_w": 0, "max_h": 0,
                                       "min_area_pct": 0.0, "max_area_pct": 0.0}, True),
    "aspect": ("Aspect ratio", "object", {"label": "", "min": 0.0, "max": 0.0}, True),
    "shape_type": ("Shape type", "object", {"label": "", "types": ["rectangle"]}, False),
    "points": ("Vertex count", "object", {"label": "", "min": 3, "max": 0}, False),
    "border": ("Not at the border", "object", {"label": "", "margin_px": 1}, True),
    "attribute": ("Attribute set", "object", {"label": "", "flag": "", "value": "any"}, False),
    "group": ("Group must contain", "object", {"labels": []}, False),
    "one_class": ("One class per image", "classification", {}, False),
    "exclusive": ("Classes exclusive", "classification", {"labels": []}, False),
}
ANY = "*"


def new_rule(rtype: str, **params) -> dict:
    title, _g, defaults, _px = TYPES[rtype]
    r = {"id": uuid.uuid4().hex[:8], "type": rtype, "enabled": True, "severity": "error"}
    r.update({k: (list(v) if isinstance(v, list) else v) for k, v in defaults.items()})
    r.update(params)
    return r


def describe(r: dict) -> str:
    t = r["type"]
    g = r.get
    if t == "inside":
        return f"every {g('child')} lies inside a {g('parent')} (≥ {g('min_frac', 0.9) * 100:.0f} % of its box)"
    if t == "contains":
        mx = g("max", 99)
        rng = f"{g('min')}" if g("min") == mx else (f"at least {g('min')}" if mx >= 99 else f"{g('min')}–{mx}")
        return f"every {g('parent')} contains {rng} {g('child')}"
    if t == "overlap":
        same = g("a") == g("b")
        return (f"no two {g('a')} overlap (IoU > {g('max_iou')})" if same
                else f"{g('a')} and {g('b')} never overlap (IoU > {g('max_iou')})")
    if t == "siblings":
        return f"the {g('child')}s inside one {g('parent')} do not overlap (IoU > {g('max_iou')})"
    if t == "count":
        lab = "objects" if g("label") in ("", ANY) else g("label")
        mx = g("max", 0)
        return (f"{g('min')}–{mx} {lab} per image" if mx else f"at least {g('min')} {lab} per image")
    if t == "requires":
        return f"an image with {g('a')} also has {g('b')}"
    if t == "excludes":
        return f"{g('a')} and {g('b')} never in the same image"
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
        return f"{g('label')} width/height between {g('min')} and {g('max') or '∞'}"
    if t == "shape_type":
        return f"{g('label')} is drawn as {' or '.join(g('types') or [])}"
    if t == "points":
        return f"{g('label')} polygons have {g('min')}–{g('max') or '∞'} vertices"
    if t == "border":
        return f"{g('label') or 'objects'} do not touch the image border ({g('margin_px')} px)"
    if t == "attribute":
        v = g("value", "any")
        return f"{g('label')} has the attribute “{g('flag')}”" + ("" if v == "any" else f" = {v}")
    if t == "group":
        return f"every group (group_id) contains {', '.join(g('labels') or [])}"
    if t == "one_class":
        return "exactly one class per image"
    if t == "exclusive":
        return f"at most one of {', '.join(g('labels') or [])} per image"
    return t


# =============================================================================
# Persistence
# =============================================================================
def load_rules(path) -> list[dict]:
    try:
        d = json.loads(open(path, encoding="utf-8").read())
        return [r for r in d.get("rules", []) if r.get("type") in TYPES]
    except (OSError, ValueError):
        return []


def save_rules(path, rules: list[dict]):
    atomic_write_text(path, json.dumps({"version": 1, "rules": rules}, indent=1, ensure_ascii=False))


# =============================================================================
# Checking
# =============================================================================
@dataclass
class Violation:
    rule: dict
    text: str
    shapes: list = field(default_factory=list)       # shape indices involved (first = the one to show)


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
    sel = lambda lab: list(range(len(shapes))) if lab in ("", ANY) else by_label.get(lab, [])

    for r in rules:
        t, g = r["type"], r.get
        if t == "inside":
            ci, pi = by_label.get(g("child"), []), by_label.get(g("parent"), [])
            if not ci:
                continue
            ok = (_frac_inside(boxes[ci], boxes[pi]).max(axis=1) >= g("min_frac", 0.9)) if pi else np.zeros(len(ci), bool)
            for k, good in zip(ci, ok):
                if not good:
                    out.append(Violation(r, f"{g('child')} #{k} is not inside any {g('parent')}", [k]))
        elif t in ("contains", "siblings"):
            pi, ci = by_label.get(g("parent"), []), by_label.get(g("child"), [])
            if not pi:
                continue
            frac = _frac_inside(boxes[ci], boxes[pi]) if ci else np.zeros((0, len(pi)))
            inside = frac >= g("min_frac", 0.9)
            for j, p in enumerate(pi):
                kids = [ci[i] for i in np.nonzero(inside[:, j])[0]] if ci else []
                if t == "contains":
                    n, mx = len(kids), g("max", 99)
                    if n < g("min", 1) or (mx and n > mx):
                        out.append(Violation(r, f"{g('parent')} #{p} contains {n} {g('child')} "
                                                f"(expected {g('min')}{'' if mx >= 99 else f'–{mx}'})", [p] + kids))
                elif len(kids) > 1:
                    iou = _iou(boxes[kids], boxes[kids])
                    np.fill_diagonal(iou, 0)
                    a, b = np.unravel_index(np.argmax(iou), iou.shape)
                    if iou[a, b] > g("max_iou", 0.1):
                        out.append(Violation(r, f"{g('child')} #{kids[a]} and #{kids[b]} inside {g('parent')} #{p} "
                                                f"overlap (IoU {iou[a, b]:.2f})", [kids[a], kids[b], p]))
        elif t == "overlap":
            ai, bi = by_label.get(g("a"), []), by_label.get(g("b"), [])
            if not ai or not bi:
                continue
            iou = _iou(boxes[ai], boxes[bi])
            same = g("a") == g("b")
            if same:
                iou = np.triu(iou, 1)
            for x, y in zip(*np.nonzero(iou > g("max_iou", 0.5))):
                out.append(Violation(r, f"{g('a')} #{ai[x]} and {g('b')} #{bi[y]} overlap (IoU {iou[x, y]:.2f})",
                                     [ai[x], bi[y]]))
        elif t == "count":
            lab = g("label")
            n = len(shapes) if lab in ("", ANY) else present.get(lab, 0)
            mx = g("max", 0)
            if n < g("min", 0) or (mx and n > mx):
                out.append(Violation(r, f"{n} {lab if lab not in ('', ANY) else 'objects'} "
                                        f"(expected {g('min')}–{mx or '∞'})", sel(lab)[:1]))
        elif t == "requires":
            if present.get(g("a")) and not present.get(g("b")):
                out.append(Violation(r, f"has {g('a')} but no {g('b')}", by_label.get(g("a"), [])[:1]))
        elif t == "excludes":
            if present.get(g("a")) and present.get(g("b")):
                out.append(Violation(r, f"{g('a')} and {g('b')} together", (by_label.get(g("a"), []) +
                                                                             by_label.get(g("b"), []))[:2]))
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
                    out.append(Violation(r, f"{labels[k]} #{k}: " + ", ".join(why), [k]))
        elif t == "aspect":
            for k in by_label.get(g("label"), []):
                w, h = px[k, 2] - px[k, 0], px[k, 3] - px[k, 1]
                if h <= 0:
                    continue
                ar = w / h
                if (g("min") and ar < g("min")) or (g("max") and ar > g("max")):
                    out.append(Violation(r, f"{labels[k]} #{k}: width/height {ar:.2f}", [k]))
        elif t == "shape_type":
            allowed = set(g("types") or [])
            for k in by_label.get(g("label"), []):
                st = shapes[k].get("shape_type") or "polygon"
                if allowed and st not in allowed:
                    out.append(Violation(r, f"{labels[k]} #{k} is a {st}", [k]))
        elif t == "points":
            for k in by_label.get(g("label"), []):
                if (shapes[k].get("shape_type") or "polygon") != "polygon":
                    continue
                n = len(shapes[k].get("points") or [])
                if n < g("min", 3) or (g("max") and n > g("max")):
                    out.append(Violation(r, f"{labels[k]} #{k} has {n} vertices", [k]))
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
            for k in by_label.get(g("label"), []):
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
        elif t == "exclusive":
            hit = [l for l in (g("labels") or []) if present.get(l)]
            if len(hit) > 1:
                out.append(Violation(r, f"several of: {', '.join(hit)}", []))
    return out


# =============================================================================
# Mining suggestions from the dataset
# =============================================================================
class RuleStats:
    """Feed every image (or a large sample) once; then suggest()."""

    def __init__(self):
        self.images = 0
        self.img_with = Counter()                 # label -> images containing it
        self.pair_with = Counter()                # (a, b) -> images containing both
        self.inst = Counter()                     # label -> instances
        self.inside = Counter()                   # (child, parent) -> child instances inside some parent
        self.kids = defaultdict(Counter)          # (parent, child) -> Counter(#children per parent)
        self.max_per_img = defaultdict(Counter)   # label -> Counter(count per image, images with it)
        self.types = defaultdict(Counter)         # label -> Counter(shape_type)
        self.tags_per_img = Counter()
        self.same_overlap = Counter()             # label -> images with two of it at IoU > 0.7

    def add(self, shapes, tags):
        self.images += 1
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
        for s in shapes:
            self.types[str(s.get("label", ""))][s.get("shape_type") or "polygon"] += 1
        if len(c) < 2 and not any(n > 1 for n in c.values()):
            return
        boxes = _boxes(shapes)
        idx = defaultdict(list)
        for i, l in enumerate(labels):
            idx[l].append(i)
        for child, ci in idx.items():
            for parent, pi in idx.items():
                if child == parent:
                    continue
                frac = _frac_inside(boxes[ci], boxes[pi])
                self.inside[(child, parent)] += int((frac.max(axis=1) >= 0.9).sum())
                per_parent = (frac >= 0.9).sum(axis=0)
                self.kids[(parent, child)].update(int(x) for x in per_parent)
        for l, li in idx.items():
            if len(li) > 1:
                iou = np.triu(_iou(boxes[li], boxes[li]), 1)
                if (iou > 0.7).any():
                    self.same_overlap[l] += 1


@dataclass
class Suggestion:
    rule: dict
    evidence: str
    strength: float           # share of the data that already follows it


def suggest(st: RuleStats, min_support: int = 20, hold: float = 0.97) -> list[Suggestion]:
    out = []
    labels = [l for l, n in st.img_with.items() if n >= min_support]
    # containment + per-parent child counts
    for (child, parent), n_in in st.inside.items():
        tot = st.inst[child]
        if tot >= min_support and n_in / tot >= hold and st.inst[parent] >= min_support:
            out.append(Suggestion(new_rule("inside", child=child, parent=parent), f"{n_in:,} / {tot:,} {child} lie inside "
                                  f"a {parent}", n_in / tot))
            dist = st.kids[(parent, child)]                  # children per parent instance
            total_p = sum(dist.values())
            if total_p >= min_support and dist.get(0, 0) / total_p <= 1 - hold:
                cum, hi = 0, 0
                for k in sorted(dist):                       # 99.5th percentile of the count
                    cum += dist[k]
                    hi = k
                    if cum / total_p >= 0.995:
                        break
                lo = min(k for k in dist if k >= 1)
                share = sum(v for k, v in dist.items() if lo <= k <= hi) / total_p
                if share >= hold:
                    out.append(Suggestion(new_rule("contains", parent=parent, child=child, min=lo, max=hi),
                                          f"{share * 100:.1f} % of {total_p:,} {parent} contain {lo}–{hi} {child}", share))
    # max count per image
    for l in labels:
        dist = st.max_per_img[l]
        imgs = sum(dist.values())
        share_one = dist.get(1, 0) / imgs
        if share_one >= hold:                                # also when it already always holds: guards new data
            out.append(Suggestion(new_rule("count", label=l, min=0, max=1, severity="warn"),
                                  f"{share_one * 100:.1f} % of images with {l} have exactly one", share_one))
    # co-occurrence
    for a in labels:
        for b in labels:
            if a == b:
                continue
            both = st.pair_with[tuple(sorted((a, b)))]
            if st.img_with[a] >= min_support and both / st.img_with[a] >= hold and \
                    not any(s.rule["type"] == "inside" and s.rule["child"] == a and s.rule["parent"] == b for s in out):
                out.append(Suggestion(new_rule("requires", a=a, b=b), f"{both:,} / {st.img_with[a]:,} images with {a} "
                                      f"also have {b}", both / st.img_with[a]))
    # shape types
    for l, tc in st.types.items():
        tot = sum(tc.values())
        t, n = tc.most_common(1)[0]
        if tot >= min_support and n / tot >= 0.99 and len(tc) > 1:
            out.append(Suggestion(new_rule("shape_type", label=l, types=[t], severity="warn"),
                                  f"{n:,} / {tot:,} {l} are {t}s", n / tot))
        elif tot >= min_support and len(tc) == 1:
            out.append(Suggestion(new_rule("shape_type", label=l, types=[t], severity="warn"),
                                  f"all {tot:,} {l} are {t}s", 1.0))
    # duplicates of the same class
    for l in labels:
        if st.inst[l] >= min_support and st.same_overlap[l] / max(st.img_with[l], 1) < 1 - hold:
            out.append(Suggestion(new_rule("overlap", a=l, b=l, max_iou=0.7, severity="warn"),
                                  f"{l} rarely overlap each other ({st.same_overlap[l]:,} images)",
                                  1 - st.same_overlap[l] / max(st.img_with[l], 1)))
    # classification
    if st.images and st.tags_per_img and sum(n for k, n in st.tags_per_img.items() if k > 0) / st.images >= 0.5:
        share = st.tags_per_img.get(1, 0) / st.images
        if share >= hold:
            out.append(Suggestion(new_rule("one_class"), f"{share * 100:.1f} % of images have exactly one class", share))
    out.sort(key=lambda s: (-s.strength, s.rule["type"]))
    return out
