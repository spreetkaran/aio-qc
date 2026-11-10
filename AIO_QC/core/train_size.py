# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Training-size check (Qt-free): how big is every object when the image is shrunk to the training size?

Ultralytics-style training resizes each image so its LONGER side equals `imgsz` (aspect kept, then letterboxed), so
an object of w × h pixels in a W × H image becomes (w·s) × (h·s) with s = imgsz / max(W, H). Detectors learn little
from objects only a few pixels across (YOLO's finest stride is 8 px) — and scale augmentation (`scale`, default 0.5)
shrinks them further at random (down to 1 − scale).

Per object:
    too_small   the shorter side at training size is below `min_side` px (or the area below `min_area` px²)
    at_risk     fine at training size, but below `min_side` once scale augmentation shrinks the image
    needed      the imgsz at which this object's shorter side reaches `min_side`
Per class: counts, the median size at training size, and the imgsz that would keep 95 % of the class above the
limit (rounded up to a multiple of 32) — with a tiling hint when that is far beyond what is practical.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

TOO_SMALL, AT_RISK = "too small", "at risk with augmentation"
SKIP_TYPES = {"point", "line", "linestrip", "points"}


@dataclass
class Obj:
    key: str
    idx: int
    label: str
    w: float                 # pixels in the original image
    h: float
    img_long: float          # longer side of the image
    verdict: str = ""

    def at(self, imgsz: int) -> tuple[float, float]:
        s = imgsz / self.img_long if self.img_long else 0.0
        return self.w * s, self.h * s

    def needed(self, min_side: float) -> float:
        """imgsz at which the shorter side reaches min_side."""
        short = min(self.w, self.h)
        return math.inf if short <= 0 else min_side * self.img_long / short


def shape_wh(shape: dict, W: float, H: float, normalized: bool) -> tuple[float, float] | None:
    """Width / height in image pixels of a shape with an area (None for points / lines)."""
    st = shape.get("shape_type") or "polygon"
    pts = shape.get("points") or []
    if st in SKIP_TYPES or len(pts) < 2:
        return None
    try:
        xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
    except (TypeError, ValueError, IndexError):
        return None
    if st == "circle":
        r = math.hypot(xs[1] - xs[0], ys[1] - ys[0])
        w = h = 2 * r
        if normalized:
            w, h = 2 * r * W, 2 * r * H         # a normalised radius is rare; treat it per axis
    else:
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if normalized:
            w, h = w * W, h * H
    if w <= 0 or h <= 0:
        return None
    return w, h


def classify(o: Obj, imgsz: int, min_side: float, min_area: float, aug_scale: float) -> str:
    tw, th = o.at(imgsz)
    short = min(tw, th)
    if short < min_side or (min_area > 0 and tw * th < min_area):
        return TOO_SMALL
    if aug_scale > 0:
        g = max(0.0, 1.0 - aug_scale)
        if short * g < min_side or (min_area > 0 and tw * th * g * g < min_area):
            return AT_RISK
    return ""


def round_up(v: float, step: int = 32) -> int:
    return int(math.ceil(v / step) * step)


def class_summary(objs: list[Obj], imgsz: int, min_side: float, share: float = 0.95) -> list[dict]:
    """One dict per class: n, too_small, at_risk, median short side at imgsz, imgsz for `share` of the class."""
    by = defaultdict(list)
    for o in objs:
        by[o.label].append(o)
    out = []
    for lab, os_ in by.items():
        shorts = sorted(min(o.at(imgsz)) for o in os_)
        need = sorted(o.needed(min_side) for o in os_)
        q = need[min(len(need) - 1, max(0, int(math.ceil(share * len(need))) - 1))]
        out.append({"label": lab, "n": len(os_),
                    "too_small": sum(1 for o in os_ if o.verdict == TOO_SMALL),
                    "at_risk": sum(1 for o in os_ if o.verdict == AT_RISK),
                    "median_short": shorts[len(shorts) // 2] if shorts else 0.0,
                    "min_short": shorts[0] if shorts else 0.0,
                    "imgsz_needed": round_up(q) if math.isfinite(q) else None})
    out.sort(key=lambda d: (-(d["too_small"] / max(1, d["n"])), d["label"]))
    return out


def advice(row: dict, imgsz: int) -> str:
    need = row["imgsz_needed"]
    if need is None:
        return "degenerate boxes"
    if need <= imgsz:
        return "fine at this size"
    if need <= 1.5 * imgsz:
        return f"train at imgsz {need}"
    if need <= 2.5 * imgsz:
        return f"imgsz {need}, or crops / tiles of the region"
    return f"needs imgsz {need} — use tiled training + SAHI inference, or a second-stage crop model"
