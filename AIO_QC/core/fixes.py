# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
User-confirmed repairs for validation issues. Nothing here runs automatically:
the Validation dialog lists every proposed fix and applies only the checked ones.
"""
from __future__ import annotations

FIX_DESCRIPTIONS = {
    "clamp_to_image": "Clamp points into the image",
    "dedupe_vertices": "Remove repeated vertices",
    "set_image_path": "Point imagePath to the found image",
    "set_image_size": "Update imageWidth / imageHeight",
    "delete_shape": "Delete the shape (entirely outside the image)",
}


def describe(fix: str, shape: dict | None, arg) -> str:
    """Human 'before → after' text for the confirmation table."""
    if fix == "set_image_path":
        return f"imagePath → {arg!r}"
    if fix == "set_image_size" and arg:
        return f"imageWidth×imageHeight → {arg[0]}×{arg[1]}"
    if fix == "clamp_to_image" and shape and arg:
        out = [p for p in shape.get("points", []) if not (0 <= p[0] <= arg[0] and 0 <= p[1] <= arg[1])]
        return f"{len(out)} point(s) moved to the image border"
    if fix == "dedupe_vertices" and shape:
        pts = shape.get("points", [])
        return f"{len(pts)} → {len(_dedupe(pts))} vertices"
    return FIX_DESCRIPTIONS.get(fix, fix)


def _dedupe(points):
    out = []
    for p in points:
        q = [round(float(p[0]), 6), round(float(p[1]), 6)]
        if q not in [[round(float(a[0]), 6), round(float(a[1]), 6)] for a in out]:
            out.append(p)
    return out


def apply_fix(ann, fix: str, shape_index: int | None, arg) -> bool:
    """Mutates `ann` (as loaded, unscaled). Returns True if something changed."""
    shape = ann.shapes[shape_index] if shape_index is not None and 0 <= shape_index < len(ann.shapes) else None
    if fix == "clamp_to_image" and shape is not None:
        w, h = (1.0, 1.0) if ann.normalized else (arg or (ann.width, ann.height))
        if not (w and h):
            return False
        changed = False
        for p in shape.get("points", []):
            x, y = min(max(float(p[0]), 0.0), float(w)), min(max(float(p[1]), 0.0), float(h))
            if (x, y) != (p[0], p[1]):
                p[0], p[1] = x, y
                changed = True
        if changed:
            shape["_geom_dirty"] = True
        return changed
    elif fix == "dedupe_vertices" and shape is not None:
        pts = shape.get("points", [])
        new = _dedupe(pts)
        if len(new) != len(pts) and len(new) >= 3:
            shape["points"] = new
            shape["_geom_dirty"] = True
            return True
    elif fix == "set_image_path" and not ann.normalized and arg:
        if ann.raw.get("imagePath") != arg:
            ann.raw["imagePath"] = arg
            return True
    elif fix == "set_image_size" and not ann.normalized and arg:
        if (ann.raw.get("imageWidth"), ann.raw.get("imageHeight")) != tuple(arg):
            ann.raw["imageWidth"], ann.raw["imageHeight"] = int(arg[0]), int(arg[1])
            return True
    elif fix == "delete_shape" and shape is not None:
        del ann.shapes[shape_index]
        return True
    return False
