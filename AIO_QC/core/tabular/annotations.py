# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The image dataset as a table (Qt-free): one row per annotation shape — and one row per file without shapes — so
Data Studio can analyse annotations like any other data (class balance per split, box sizes, aspect ratios, review
coverage, attributes …).

    build(samples, adapter, review=None, progress=None, cancel=None) -> pandas DataFrame
"""
from __future__ import annotations

import math

COLUMNS = ["file", "split", "folder", "label", "shape_type", "x1", "y1", "x2", "y2", "width", "height", "area",
           "cx", "cy", "rel_width", "rel_height", "rel_area", "aspect", "image_width", "image_height", "points",
           "group_id", "description", "flags", "review", "shape_index"]


def _box(sh, W, H, normalized):
    pts = sh.get("points") or []
    try:
        xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs:
        return None
    if (sh.get("shape_type") or "polygon") == "circle" and len(xs) == 2:
        r = math.hypot(xs[1] - xs[0], ys[1] - ys[0])
        x1, y1, x2, y2 = xs[0] - r, ys[0] - r, xs[0] + r, ys[0] + r
    else:
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if normalized:
        x1, x2, y1, y2 = x1 * W, x2 * W, y1 * H, y2 * H
    return x1, y1, x2, y2, len(xs)


def build(samples, adapter, review=None, progress=None, cancel=None):
    import pandas as pd
    rows = []
    samples = list(samples)
    for i, s in enumerate(samples, 1):
        if cancel is not None and cancel.is_set():
            break
        base = {"file": s.key, "split": s.split or "", "folder": s.folder or "",
                "review": (review.status(s.key) if review is not None else None) or ""}
        try:
            ann = adapter.load(s, light=True)
        except Exception:
            rows.append({**base, "label": None, "shape_type": "(unreadable)"})
            continue
        W = H = None
        if ann.width and ann.height and not ann.normalized:
            W, H = float(ann.width), float(ann.height)
        else:
            try:
                wh = adapter.image_size(s, ann)
            except Exception:
                wh = None
            if wh:
                W, H = float(wh[0]), float(wh[1])
        shapes = ann.shapes or []
        tags = list(getattr(ann, "tags", []) or [])
        if not shapes and not tags:
            rows.append({**base, "label": None, "shape_type": "(no shapes)", "image_width": W, "image_height": H})
        for t in tags:                                             # classification datasets: image-level classes
            rows.append({**base, "label": str(t), "shape_type": "image class", "image_width": W, "image_height": H})
        for j, sh in enumerate(shapes):
            r = {**base, "label": str(sh.get("label", "")), "shape_type": sh.get("shape_type") or "polygon",
                 "image_width": W, "image_height": H, "group_id": sh.get("group_id"),
                 "description": sh.get("description") if isinstance(sh.get("description"), str) else None,
                 "flags": ",".join(sorted(k for k, v in (sh.get("flags") or {}).items() if v is True)) or None,
                 "shape_index": j}
            b = _box(sh, W or 1.0, H or 1.0, bool(ann.normalized)) if (W and H) or not ann.normalized else None
            if b is not None:
                x1, y1, x2, y2, npts = b
                w, h = x2 - x1, y2 - y1
                r.update(x1=round(x1, 2), y1=round(y1, 2), x2=round(x2, 2), y2=round(y2, 2), width=round(w, 2),
                         height=round(h, 2), area=round(w * h, 2), cx=round((x1 + x2) / 2, 2),
                         cy=round((y1 + y2) / 2, 2), points=npts, aspect=round(w / h, 4) if h > 0 else None)
                if W and H:
                    r.update(rel_width=round(w / W, 6), rel_height=round(h / H, 6), rel_area=round(w * h / (W * H), 8))
            rows.append(r)
        if progress is not None and (i % 200 == 0 or i == len(samples)):
            progress(i, len(samples))
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("shape_index", "points", "group_id"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df
