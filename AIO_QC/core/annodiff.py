# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
What changed between two versions of an annotation file (Qt-free) — for history previews and batch merges.

    a = parse(bytes_a, "labelme" | "yolo", class_names)     b = parse(bytes_b, …)
    d = diff(a, b)  ->  {"added": [...], "removed": [...], "relabelled": [(old, new)], "reshaped": [...],
                          "unchanged": n, "flags": (old, new) | None}
    summary(d) -> "2 added · 1 relabelled (car → truck) · 1 reshaped"
Shapes are matched by overlap (boxes relative to the image, so pixel / normalised files compare alike): same label
first, then any label (= relabelled); a matched shape whose points moved more than a hair is "reshaped".
"""
from __future__ import annotations

import json
from collections import Counter

TOL = 0.002          # relative coordinate tolerance for "unchanged"


def parse(data: bytes | None, kind: str, names=None) -> dict:
    """-> {"shapes": [(label, type, bbox rel, points rel)], "flags": set}. Unreadable -> {"error": text}."""
    if data is None:
        return {"shapes": [], "flags": set(), "missing": True}
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"error": "not UTF-8", "shapes": [], "flags": set()}
    out = []
    if kind == "yolo":
        for line in text.splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            try:
                cid = int(float(p[0]))
                v = [float(x) for x in p[1:]]
            except ValueError:
                continue
            label = names[cid] if names and 0 <= cid < len(names) else f"class_{cid}"
            if len(v) == 4 or (len(v) > 4 and (len(v) - 4) % 3 == 0 and len(v) % 2 == 1):
                cx, cy, w, h = v[:4]
                pts = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy + h / 2)]
                st = "rectangle"
            else:
                pts = list(zip(v[0::2], v[1::2]))
                st = "polygon"
            out.append((label, st, _bbox(pts), pts))
        return {"shapes": out, "flags": set()}
    try:
        d = json.loads(text)
    except ValueError as exc:
        return {"error": f"invalid JSON ({exc})", "shapes": [], "flags": set()}
    W = float(d.get("imageWidth") or 1) or 1.0
    H = float(d.get("imageHeight") or 1) or 1.0
    for s in d.get("shapes") or []:
        try:
            pts = [(float(a[0]) / W, float(a[1]) / H) for a in s.get("points") or []]
        except (TypeError, ValueError, IndexError):
            continue
        if not pts:
            continue
        out.append((str(s.get("label")), s.get("shape_type") or "polygon", _bbox(pts), pts))
    flags = {k for k, v in (d.get("flags") or {}).items() if v is True}
    return {"shapes": out, "flags": flags}


def _bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(a[2] - a[0], 1e-9) * max(a[3] - a[1], 1e-9) + max(b[2] - b[0], 1e-9) * max(b[3] - b[1], 1e-9) - inter
    return inter / ua if ua > 0 else 0.0


def _same_pts(p, q):
    return len(p) == len(q) and all(abs(a[0] - b[0]) <= TOL and abs(a[1] - b[1]) <= TOL for a, b in zip(p, q))


def diff(a: dict, b: dict) -> dict:
    A, B = a.get("shapes", []), b.get("shapes", [])
    used_a, used_b = set(), set()
    pairs = []
    # exact geometry first (cheap, the common case), then overlap with the same label, then any label
    for i, sa in enumerate(A):
        for j, sb in enumerate(B):
            if j not in used_b and sa[1] == sb[1] and _same_pts(sa[3], sb[3]):
                used_a.add(i)
                used_b.add(j)
                pairs.append((i, j))
                break
    for same_label in (True, False):
        cand = []
        for i, sa in enumerate(A):
            if i in used_a:
                continue
            for j, sb in enumerate(B):
                if j in used_b or (same_label and sa[0] != sb[0]):
                    continue
                v = _iou(sa[2], sb[2])
                if v >= 0.5:
                    cand.append((v, i, j))
        for _v, i, j in sorted(cand, reverse=True):
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
    out = {"added": [B[j][0] for j in range(len(B)) if j not in used_b],
           "removed": [A[i][0] for i in range(len(A)) if i not in used_a],
           "relabelled": [], "reshaped": [], "unchanged": 0, "flags": None}
    for i, j in pairs:
        sa, sb = A[i], B[j]
        changed = False
        if sa[0] != sb[0]:
            out["relabelled"].append((sa[0], sb[0]))
            changed = True
        if not _same_pts(sa[3], sb[3]):
            out["reshaped"].append(sb[0])
            changed = True
        if not changed:
            out["unchanged"] += 1
    fa, fb = a.get("flags", set()), b.get("flags", set())
    if fa != fb:
        out["flags"] = (sorted(fa), sorted(fb))
    return out


def summary(d: dict) -> str:
    parts = []
    if d["added"]:
        parts.append(f"{len(d['added'])} added ({_top(d['added'])})")
    if d["removed"]:
        parts.append(f"{len(d['removed'])} removed ({_top(d['removed'])})")
    if d["relabelled"]:
        c = Counter(f"{a} → {b}" for a, b in d["relabelled"])
        parts.append(f"{len(d['relabelled'])} relabelled ({', '.join(f'{k}' + (f' ×{n}' if n > 1 else '') for k, n in c.most_common(3))})")
    if d["reshaped"]:
        parts.append(f"{len(d['reshaped'])} reshaped")
    if d["flags"] is not None:
        parts.append(f"flags {', '.join(d['flags'][0]) or '—'} → {', '.join(d['flags'][1]) or '—'}")
    return " · ".join(parts) or "no annotation changes"


def is_change(d: dict) -> bool:
    return bool(d["added"] or d["removed"] or d["relabelled"] or d["reshaped"] or d["flags"] is not None)


def _top(labels):
    c = Counter(labels)
    return ", ".join(f"{k}" + (f" ×{n}" if n > 1 else "") for k, n in c.most_common(3)) + (" …" if len(c) > 3 else "")
