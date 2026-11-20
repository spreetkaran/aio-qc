# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
LabelMe -> YOLO (detection or segmentation), Qt-free. Used by Convert to YOLO and by Export & Split.

    sv = survey(samples, adapter)                       # what is there: per label objects / images / shape types
    st = convert(samples_by_split, adapter, root, out, Options(task="det", classes=["car", "person"]))

Shapes, per task:
    rectangle            det: its box                       seg: 4-point polygon
    polygon              det: bounding box                  seg: the polygon
    circle               det: its box                       seg: polygon (16-64 points, about one every 8 px)
    oriented rectangle   det: axis-aligned bounding box     seg: its 4 corners
    mask                 det: its box                       seg: the mask's outline (traced)
    line / linestrip / point / points: not an area -> skipped (counted, with the reason)
Shapes with the same group_id AND label are one object drawn in pieces (e.g. split by an occluder): det -> one box
around all of them, seg -> one polygon (the pieces joined by zero-width bridges, as YOLO wants one polygon per object).
Optional per Options.merge_groups.

Polygons are clipped to the image (Sutherland-Hodgman), boxes clamped; what is left without area is skipped
("outside the image" / "no area"). Only the listed classes are written, ids in list order; other labels are counted
as "not included". Images without an included object get an empty label file (a background image) or are left out.

Output (layout "flat"): out/images/... + out/labels/....txt; (layout "splits"): out/<split>/images|labels/...
Sub-folders are kept (split folder names dropped); names are unique by file stem, case-insensitive, so a.jpg and
a.png never share labels/a.txt. Images are hard-linked when possible (no extra space), else copied; a JSON whose image
only exists as embedded imageData gets it decoded. Writes dataset.yaml (names quoted, nc), classes.txt,
conversion_report.txt and a manifest (.aioqc_yolo.json) that lets a later run replace exactly what this one wrote.
"""
from __future__ import annotations

import base64
import io as _io
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .io import atomic_write_text, image_size
from AIO_QC.core.diag import ignored as _ignored

MANIFEST = ".aioqc_yolo.json"
TASKS = {"det": "Detection (boxes)", "seg": "Segmentation (polygons)"}
AREA_TYPES = {"rectangle", "polygon", "circle", "oriented rectangle", "mask"}
NOT_AREA = {"line": "lines have no area", "linestrip": "lines have no area", "point": "points have no area",
            "points": "points have no area"}
_STRIP = {"train", "val", "valid", "validation", "test", "images", "labels"}


TRAINER_FORMATS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp", ".dng", ".mpo", ".pfm"}
BROKEN = {"bad coordinates", "too few points", "mask could not be read"}


@dataclass
class Options:
    task: str = "det"                          # det | seg
    classes: list = field(default_factory=list)   # output classes; their order = the YOLO ids
    layout: str = "flat"                       # flat (images/ + labels/) | splits (<split>/images …)
    link: str = "hardlink"                     # hardlink | copy
    background: str = "keep"                   # images without an included object: keep (empty .txt) | skip
    merge_groups: bool = True
    precision: int = 6
    workers: int = 0                           # 0: automatic
    rename: dict = field(default_factory=dict)   # dataset label -> output class (several labels -> one = merged)
    fold_case: bool = False                    # "Car", "car " and "CAR" are the same class
    min_area: float = 0.0                      # px²: smaller objects are not written
    left_out: str = "drop"                     # an image with a label not written: drop those shapes | skip image
    broken: str = "drop"                       # an image with a broken shape: drop the shape | skip image
    parts: list = field(default_factory=list)  # [(part class, [whole classes], min share inside 0-1)]
    orphans: str = "drop"                      # a part not inside a whole: drop it | skip image
    fix_images: bool = True                    # re-encode formats YOLO trainers can't read (→ PNG)
    verify: bool = True                        # read every written label back
    folders: str = "keep"                      # keep (sub-folders as in the dataset) | flat (one folder per split)
    naming: dict = field(default_factory=dict)  # flat: core.naming.Scheme as a dict (default: the folder prefix)
    name_by: str = "annotation"                # output names after the annotation file (LabelMe) | the image (YOLO,
                                               # COCO: one JSON for all images)


def _norm_label(x: str) -> str:
    return " ".join(str(x).split()).lower()


class LabelIds:
    """dataset label -> (YOLO id, output class) through the rename map (and case / space folding), or None."""

    def __init__(self, classes, rename=None, fold=False):
        self.idx = {c: i for i, c in enumerate(classes)}
        self.rename = dict(rename or {})
        self.fold = fold
        self.folded = {}
        if fold:
            for c in classes:
                self.folded.setdefault(_norm_label(c), c)
            for src, out in self.rename.items():
                if out in self.idx:
                    self.folded.setdefault(_norm_label(src), out)
        self._memo = {}

    def get(self, lab):
        r = self._memo.get(lab, 0)
        if r != 0:
            return r
        out = self.rename.get(lab, lab)
        if out not in self.idx and self.fold:
            out = self.folded.get(_norm_label(lab), out)
        r = (self.idx[out], out) if out in self.idx else None
        self._memo[lab] = r
        return r


def _ids(cid):
    if isinstance(cid, LabelIds):
        return cid.get
    return lambda lab: (cid[lab], lab) if lab in cid else None


# ====================================================================================================== geometry
def _num_pts(pts):
    out = []
    for p in pts or []:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        out.append([x, y])
    return out


def _circle(pts):
    (cx, cy), (px, py) = pts[0], pts[1]
    r = math.hypot(px - cx, py - cy)
    n = max(16, min(64, int(2 * math.pi * r / 8)))
    return [[cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)] for i in range(n)]


def _mask_outline(shape, pts):
    """LabelMe 'mask': points = [[x1, y1], [x2, y2]] + base64 PNG of the box. -> outline polygons in image pixels."""
    try:
        import cv2
        import numpy as np
        raw = base64.b64decode(shape.get("mask") or "")
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            return []
        if arr.ndim == 3:
            arr = arr[..., -1] if arr.shape[2] == 4 else arr.max(axis=2)
        m = (arr > 0).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x0, y0 = min(pts[0][0], pts[1][0]), min(pts[0][1], pts[1][1])
        out = []
        for c in sorted(cs, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(c) < 1:
                continue
            out.append([[float(p[0][0]) + x0, float(p[0][1]) + y0] for p in c])
        return out
    except Exception as _exc:
        _ignored(_exc)
        return []


def shape_polygons(shape: dict, task: str):
    """-> (list of polygons / [box as 2 points], None) or (None, reason). Pixel coordinates, not clipped."""
    st = (shape.get("shape_type") or "polygon").lower()
    if st in NOT_AREA:
        return None, NOT_AREA[st]
    pts = _num_pts(shape.get("points"))
    if pts is None:
        return None, "bad coordinates"
    if st == "rectangle":
        if len(pts) < 2:
            return None, "too few points"
        (x1, y1), (x2, y2) = pts[0], pts[1]
        return [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], None
    if st == "circle":
        if len(pts) < 2:
            return None, "too few points"
        return [_circle(pts)], None
    if st == "mask":
        if len(pts) < 2:
            return None, "too few points"
        if task == "seg":
            polys = _mask_outline(shape, pts)
            return (polys, None) if polys else (None, "mask could not be read")
        (x1, y1), (x2, y2) = pts[0], pts[1]
        return [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], None
    if st in ("polygon", "oriented rectangle"):
        if len(pts) < 3:
            return None, "too few points"
        return [pts], None
    return None, f"unknown shape type '{st}'"


def clip_polygon(poly, W, H):
    """Sutherland-Hodgman against the image rectangle."""
    def clip(pts, inside, cross):
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            ci, pi = inside(cur), inside(prev)
            if ci:
                if not pi:
                    out.append(cross(prev, cur))
                out.append(cur)
            elif pi:
                out.append(cross(prev, cur))
        return out

    def x_at(a, b, x):
        t = (x - a[0]) / (b[0] - a[0])
        return [x, a[1] + t * (b[1] - a[1])]

    def y_at(a, b, y):
        t = (y - a[1]) / (b[1] - a[1])
        return [a[0] + t * (b[0] - a[0]), y]

    p = list(poly)
    for inside, cross in ((lambda q: q[0] >= 0, lambda a, b: x_at(a, b, 0.0)),
                          (lambda q: q[0] <= W, lambda a, b: x_at(a, b, float(W))),
                          (lambda q: q[1] >= 0, lambda a, b: y_at(a, b, 0.0)),
                          (lambda q: q[1] <= H, lambda a, b: y_at(a, b, float(H)))):
        if not p:
            break
        p = clip(p, inside, cross)
    out = []
    for q in p:                                      # drop repeated points
        if not out or abs(out[-1][0] - q[0]) > 1e-9 or abs(out[-1][1] - q[1]) > 1e-9:
            out.append(q)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) < 1e-9 and abs(out[0][1] - out[-1][1]) < 1e-9:
        out.pop()
    return out


def poly_area(p):
    return abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))) / 2


def bridge(polys):
    """Several polygons -> one, joined by zero-width bridges at their closest points (YOLO: one polygon per object)."""
    polys = [p for p in polys if len(p) >= 3]
    if not polys:
        return []
    merged = list(polys[0])
    rest = polys[1:]
    while rest:
        best = None
        for k, P in enumerate(rest):
            for i, a in enumerate(merged):
                for j, b in enumerate(P):
                    d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                    if best is None or d < best[0]:
                        best = (d, k, i, j)
        _d, k, i, j = best
        P = rest.pop(k)
        merged = merged[:i + 1] + P[j:] + P[:j + 1] + merged[i:]
    return merged


def objects(shapes, W, H, cid, task: str, merge_groups: bool = True, precision: int = 6, min_area: float = 0.0,
            parts=(), stats: dict | None = None):
    """-> (lines, written Counter[class], dropped Counter[reason], not_included Counter[label], merged int).
    cid: {label: id} or a LabelIds (rename / fold). parts: [(part class, [whole classes], min share inside)] — a part
    not (enough) inside a whole of its classes is dropped ("part outside its whole"). stats (optional) receives
    "broken" (shapes that could not be read) for the per-image policies."""
    ids = _ids(cid)
    written, dropped, excluded = Counter(), Counter(), Counter()
    groups: dict = defaultdict(list)                 # (class, group_id) or (class, None, i) -> polygons
    order = []
    broken = 0
    for i, s in enumerate(shapes):
        lab = str(s.get("label", ""))
        hit = ids(lab)
        if hit is None:
            excluded[lab] += 1
            continue
        polys, why = shape_polygons(s, task)
        if polys is None:
            dropped[why] += 1
            broken += why in BROKEN or why.startswith("unknown shape")
            continue
        gid = s.get("group_id")
        key = (hit[1], gid) if (merge_groups and gid is not None) else (hit[1], None, i)
        if key not in groups:
            order.append(key)
        groups[key].extend(polys)
    merged = sum(1 for k in order if len(groups[k]) > 1 and k[1] is not None)
    objs = []                                        # (class, clipped polygons)
    for key in order:
        clipped = [c for c in (clip_polygon(p, W, H) for p in groups[key]) if len(c) >= 3 and poly_area(c) > 1e-6]
        if not clipped:
            raw = [q for p in groups[key] for q in p]
            inside = any(0 <= q[0] <= W and 0 <= q[1] <= H for q in raw)
            dropped["no area" if inside else "outside the image"] += 1
            broken += inside
            continue
        if min_area > 0:
            area = _box_area(clipped) if task == "det" else sum(poly_area(c) for c in clipped)
            if area < min_area:
                dropped["smaller than the minimum area"] += 1
                continue
        objs.append((key[0], clipped))
    if parts:
        objs = _keep_parts(objs, parts, dropped)
    lines = []
    fmt = f"{{:.{precision}f}}"
    idx = {}
    for name, clipped in objs:
        if name not in idx:
            idx[name] = ids_for(cid, name)
        if task == "det":
            xs = [q[0] for p in clipped for q in p]
            ys = [q[1] for p in clipped for q in p]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                dropped["no area"] += 1
                continue
            vals = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
        else:
            poly = clipped[0] if len(clipped) == 1 else bridge(clipped)
            vals = tuple(v for q in poly for v in (min(max(q[0] / W, 0.0), 1.0), min(max(q[1] / H, 0.0), 1.0)))
        lines.append(f"{idx[name]} " + " ".join(fmt.format(v) for v in vals))
        written[name] += 1
    if stats is not None:
        stats["broken"] = broken
    return lines, written, dropped, excluded, merged


def ids_for(cid, name):
    if isinstance(cid, LabelIds):
        return cid.idx[name]
    return cid[name]


def _box_area(polys):
    xs = [q[0] for p in polys for q in p]
    ys = [q[1] for p in polys for q in p]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def inside_share(part, whole) -> float:
    """Share of the part's area inside the whole (polygon lists, pixels): raster on their common box."""
    pa = sum(poly_area(p) for p in part)
    if pa <= 0:
        return 0.0
    xs = [q[0] for P in (part, whole) for p in P for q in p]
    ys = [q[1] for P in (part, whole) for p in P for q in p]
    x0, y0 = min(xs), min(ys)
    bx = (min(q[0] for p in part for q in p), min(q[1] for p in part for q in p),
          max(q[0] for p in part for q in p), max(q[1] for p in part for q in p))
    wx = (min(q[0] for p in whole for q in p), min(q[1] for p in whole for q in p),
          max(q[0] for p in whole for q in p), max(q[1] for p in whole for q in p))
    if bx[2] <= wx[0] or wx[2] <= bx[0] or bx[3] <= wx[1] or wx[3] <= bx[1]:
        return 0.0
    try:
        import cv2
        import numpy as np
    except ImportError:                          # box overlap as a stand-in
        iw = max(0.0, min(bx[2], wx[2]) - max(bx[0], wx[0]))
        ih = max(0.0, min(bx[3], wx[3]) - max(bx[1], wx[1]))
        return iw * ih / max((bx[2] - bx[0]) * (bx[3] - bx[1]), 1e-9)
    span = max(max(xs) - x0, max(ys) - y0, 1.0)
    k = min(4.0, 512.0 / span)                   # at most 512 px across: fast and plenty precise
    W, H = int(math.ceil((max(xs) - x0) * k)) + 2, int(math.ceil((max(ys) - y0) * k)) + 2
    a, b = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    cv2.fillPoly(a, [np.round((np.asarray(p) - [x0, y0]) * k).astype(np.int32) for p in part], 1)
    cv2.fillPoly(b, [np.round((np.asarray(p) - [x0, y0]) * k).astype(np.int32) for p in whole], 1)
    n = int(a.sum())
    return float(np.logical_and(a, b).sum()) / n if n else 0.0


def _keep_parts(objs, parts, dropped):
    rules = defaultdict(list)                    # part class -> [(whole classes, min share)]
    for part, wholes, share in parts:
        rules[part].append((set(wholes), float(share)))
    out = []
    for name, polys in objs:
        if name not in rules:
            out.append((name, polys))
            continue
        ok = False
        for wholes, share in rules[name]:
            for other, wp in objs:
                if other in wholes and wp is not polys and inside_share(polys, wp) >= share - 1e-9:
                    ok = True
                    break
            if ok:
                break
        if ok:
            out.append((name, polys))
        else:
            dropped["part outside its whole"] += 1
    return out


# ====================================================================================================== survey
@dataclass
class Survey:
    objects: Counter = field(default_factory=Counter)            # label -> shapes
    images: Counter = field(default_factory=Counter)             # label -> images with it
    types: dict = field(default_factory=lambda: defaultdict(Counter))   # label -> shape_type -> count
    groups: int = 0                                              # objects drawn in several pieces
    files: int = 0
    empty: int = 0                                               # files without shapes
    missing_image: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    per_sample: dict = field(default_factory=dict)               # key -> (labels Counter, boxes_norm)

    def suggested_task(self) -> str:
        t = Counter()
        for c in self.types.values():
            t.update(c)
        poly = t["polygon"] + t["mask"] + t["circle"]
        return "seg" if poly > t["rectangle"] + t["oriented rectangle"] else "det"


def survey(samples, adapter, progress=None, cancel=None, workers: int = 0) -> Survey:
    """One pass over the annotations (core.parallel: read by a few threads when the files wait on storage)."""
    from .parallel import ordered_map
    sv = Survey()
    n = len(samples)

    def read(s):
        ann = adapter.load(s, light=True)
        img = ann.image_path or s.image_path
        return ann, (img is None or not Path(img).is_file()) and not _has_image_data(s.ann_path)

    for i, (s, res) in enumerate(ordered_map(read, samples, workers)):
        if i % 500 == 0:
            if cancel is not None and cancel.is_set():
                break
            if progress is not None:
                progress(i, n)
        sv.files += 1
        if isinstance(res, Exception):
            sv.errors.append(f"{s.name}: {res}")
            continue
        ann, missing = res
        if missing:
            sv.missing_image.append(s.name)
        _survey_one(sv, s, ann)
    if progress is not None:
        progress(n, n)
    return sv


def _survey_one(sv: Survey, s, ann):
    W, H = (1.0, 1.0) if ann.normalized else (ann.width or 0, ann.height or 0)
    labs, boxes, gkeys = Counter(), [], Counter()
    for sh in ann.shapes:
        lab = str(sh.get("label", ""))
        st = (sh.get("shape_type") or "polygon").lower()
        sv.objects[lab] += 1
        sv.types[lab][st] += 1
        labs[lab] += 1
        if sh.get("group_id") is not None:
            gkeys[(lab, sh.get("group_id"))] += 1
        pts = _num_pts(sh.get("points")) or []
        if W and H and len(pts) >= 2:
            xs, ys = [p[0] / W for p in pts], [p[1] / H for p in pts]
            boxes.append((min(xs), min(ys), max(xs), max(ys), lab))
    sv.groups += sum(1 for v in gkeys.values() if v > 1)
    for lab in labs:
        sv.images[lab] += 1
    if not labs:
        sv.empty += 1
    sv.per_sample[s.key] = (labs, boxes)


def _has_image_data(json_path) -> bool:
    if json_path is None:
        return False
    try:
        with open(json_path, "rb") as f:
            head = f.read(1 << 20)
        i = head.find(b'"imageData"')
        if i < 0:
            return False
        j = head.find(b":", i)
        rest = head[j + 1:j + 64].lstrip()
        return rest[:1] == b'"' and rest[1:2] != b'"'
    except OSError:
        return False


# ====================================================================================================== convert
def _rel(root: Path, img: Path) -> Path:
    try:
        parts = list(Path(img).relative_to(root).parts)
    except ValueError:
        parts = [Path(img).name]
    dirs = [p for p in parts[:-1] if p.lower() not in _STRIP]
    return Path(*dirs, parts[-1]) if dirs else Path(parts[-1])


def _place(src, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "linked"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copied"


def _yaml_str(s: str) -> str:
    return json.dumps(str(s), ensure_ascii=False)    # a JSON string is a valid double-quoted YAML scalar


def dataset_yaml(out: Path, splits: list[str], classes: list[str], task: str, source: str = "") -> str:
    q = lambda p: str(p).replace("\\", "/")          # noqa: E731
    lines = [f"# YOLO {TASKS[task].split(' (')[0].lower()} dataset" + (f" converted from {source}" if source else "")
             + f" by AIO QC, {time.strftime('%Y-%m-%d')}",
             "# The folders below are relative to 'path'. Moved the dataset? Change 'path' (or delete the line: "
             "Ultralytics then uses this file's folder).",
             f"path: {_yaml_str(q(Path(out).resolve()))}"]
    if splits == [""]:
        lines += ["train: images", "val: images      # no separate validation set (make one with a new split)"]
    else:
        for sp in ("train", "val", "test"):
            if sp in splits:
                lines.append(f"{sp}: {sp}/images")
        if "val" not in splits and "train" in splits:
            lines.append("val: train/images      # no validation split in the source")
    lines += ["", f"nc: {len(classes)}", "names:"] + [f"  {i}: {_yaml_str(c)}" for i, c in enumerate(classes)]
    return "\n".join(lines) + "\n"


def previous_output(out) -> dict | None:
    """The manifest of an earlier conversion into this folder (what it created), or None."""
    p = Path(out) / MANIFEST
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        return m if isinstance(m, dict) and isinstance(m.get("created"), list) else None
    except (OSError, ValueError):
        return None


def add_created(out, rel: str):
    """Record one more thing this conversion created (so a later run removes it too)."""
    m = previous_output(out)
    if m is not None and rel not in m["created"]:
        m["created"].append(rel)
        atomic_write_text(Path(out) / MANIFEST, json.dumps(m, indent=1, ensure_ascii=False))


def remove_previous(out) -> int:
    """Delete exactly what the earlier conversion created (its folders and files) — nothing else in the folder."""
    out = Path(out)
    m = previous_output(out)
    if m is None:
        return 0
    n = 0
    for rel in m["created"]:
        p = (out / rel).resolve()
        if out.resolve() not in p.parents:           # never outside the folder
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            n += 1
        elif p.exists():
            p.unlink()
            n += 1
    for name in (MANIFEST, ".aio_qc_provenance.json"):   # the record describes what was just removed
        try:
            (out / name).unlink()
        except OSError:
            pass
    return n


def _named_after(s, by: str) -> Path:
    if by == "image" and s.image_path:
        return Path(s.image_path)
    return Path(s.ann_path) if s.ann_path else Path(s.image_path)


def naming_items(samples_by_split: dict, root, by: str = "annotation") -> tuple[list, dict]:
    """core.naming items for a flattened conversion (one per sample, named after its JSON — or its image), and
    key → sample."""
    from . import naming as N
    items, by_key = [], {}
    root = Path(root)
    for split, samples in samples_by_split.items():
        for s in samples:
            rel = _rel(root, _named_after(s, by)).as_posix()
            key = (split, s.key)
            items.append(N.Item(key, rel, split=split, path=str(s.image_path or s.ann_path or ""),
                                extra_exts=(".txt",)))
            by_key[key] = s
    return items, by_key


def name_jobs(samples_by_split: dict, root: Path, out: Path, opts: Options):
    """[(split, sample, image dest without extension, label dest)] and how many files got a new name (flat:
    the naming engine — every name verified unique per split, the rest numbered; renamed.csv written)."""
    jobs = []
    if opts.folders == "flat":
        from . import naming as N
        items, by = naming_items(samples_by_split, root, opts.name_by)
        sc = N.Scheme.from_dict(opts.naming) if opts.naming else N.Scheme(preset="folder")
        P = N.plan(items, sc, flat=True, root_len=len(str(out.resolve())) + len("/train/images"))
        if not P.ok:
            P = N.resolve(P, root_len=len(str(out.resolve())) + len("/train/images"))
        for it in P.items:
            split = it.key[0]
            base = out / split if split else out
            stem = P.new[it.key]
            jobs.append((split, by[it.key], base / "images" / stem, base / "labels" / (stem + ".txt")))
        n = sum(1 for it in P.items if P.new[it.key] != it.stem)
        if n:
            import csv
            with open(out / "renamed.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["original", "new"])
                for it in P.items:
                    if P.new[it.key] != it.stem:
                        w.writerow([it.rel, (it.key[0] + "/" if it.key[0] else "") + P.new[it.key]])
        return jobs, n
    taken = set()
    for split, samples in samples_by_split.items():
        base = out / split if split else out
        for s in samples:
            rel = _rel(root, _named_after(s, opts.name_by))
            stem, folder = rel.stem, rel.parent
            k, cand = 0, stem
            while (str(folder).lower(), split, cand.lower()) in taken:
                k += 1
                cand = f"{stem}_{k}"
            taken.add((str(folder).lower(), split, cand.lower()))
            jobs.append((split, s, base / "images" / folder / cand, base / "labels" / folder / (cand + ".txt")))
    return jobs, 0


def convert(samples_by_split: dict, adapter, root, out, opts: Options, progress=None, cancel=None,
            source_name: str = "") -> dict:
    """samples_by_split: {"": [Sample]} for layout flat, {"train": [...], "val": [...], ...} for splits."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cid = LabelIds(opts.classes, opts.rename, opts.fold_case)
    st = {"images": Counter(), "objects": Counter(), "background": 0, "left_out_background": 0,
          "dropped": Counter(), "not_included": Counter(), "merged": 0, "skipped": [], "placed": Counter(),
          "size_mismatch": 0, "decoded": 0, "left_out_images": Counter(), "renamed": Counter(), "fixed": Counter()}
    # 1. destination names (unique by folder + stem, case-insensitive — named after the JSON: unique in its folder;
    # YOLO pairs image and label by name; the image keeps its own extension, set when it is placed)
    jobs, renamed_csv = name_jobs(samples_by_split, root, out, opts)
    if renamed_csv:
        st["renamed_files"] = renamed_csv
    total = len(jobs)

    def one(job):
        split, s, dimg, dlab = job
        if cancel is not None and cancel.is_set():
            return None
        try:
            ann = adapter.load(s, light=True)
        except Exception as exc:
            return {"skip": f"{s.name}: could not read ({exc})"}
        src = ann.image_path or s.image_path
        decoded = None
        if src is None or not Path(src).is_file():
            from .export import _embedded_image
            decoded = _embedded_image(s.ann_path) if opts.name_by == "annotation" else None   # LabelMe imageData
            if decoded is None:
                return {"skip": f"{s.name}: image not found (no image file, no imageData)"}
        W, H = (ann.width, ann.height) if ann.width and ann.height else (None, None)
        real = image_size(src) if decoded is None else None
        mismatch = False
        if real and W and H and (real[0], real[1]) not in ((int(W), int(H)), (int(H), int(W))):
            mismatch = True                            # JSON size kept: the points were drawn in that space
        if not (W and H):
            if real:
                W, H = real
            elif decoded is not None:
                try:
                    from PIL import Image
                    W, H = Image.open(_io.BytesIO(decoded[0])).size
                except Exception as _exc:
                    _ignored(_exc)
                    W = H = None
        if not (W and H):
            return {"skip": f"{s.name}: unknown image size"}
        shapes = ann.shapes
        if ann.normalized:
            shapes = [dict(x, points=[[p[0] * W, p[1] * H] for p in x.get("points") or []]) for x in shapes]
        info = {}
        min_area = opts.min_area
        if min_area and real and real[0] * real[1] and float(W) * float(H) != real[0] * real[1]:
            min_area *= float(W) * float(H) / (real[0] * real[1])   # px² of the real image, in the JSON's space
        lines, written, dropped, excluded, merged = objects(shapes, float(W), float(H), cid, opts.task,
                                                            opts.merge_groups, opts.precision, min_area,
                                                            opts.parts, info)
        renamed = Counter()
        for sh in shapes:
            lab = str(sh.get("label", ""))
            hit = cid.get(lab)
            if hit is not None and hit[1] != lab:
                renamed[f"{lab} → {hit[1]}"] += 1
        why = None
        if excluded and opts.left_out == "skip":
            why = "has a label that is not written"
        elif info.get("broken") and opts.broken == "skip":
            why = "has a broken shape"
        elif dropped.get("part outside its whole") and opts.orphans == "skip":
            why = "has a part outside its whole"
        if why:
            return {"left_out": why, "dropped": dropped, "excluded": excluded}
        if not lines and opts.background == "skip":
            return {"left_out": True, "dropped": dropped, "excluded": excluded}
        ext = (decoded[1] if decoded is not None else Path(src).suffix).lower()
        fix = None
        if opts.fix_images and ext not in TRAINER_FORMATS:
            fix, ext = "re-encoded to PNG (format YOLO trainers can't read)", ".png"
        dimg = dimg.with_name(dimg.name + ext)
        dimg.parent.mkdir(parents=True, exist_ok=True)
        if fix:
            _reencode(decoded[0] if decoded is not None else src, dimg)
            how = "re-encoded"
            _stamp_image(dimg, s, "re-encoded as PNG")
        elif decoded is not None:
            dimg.write_bytes(decoded[0])
            how = "decoded from imageData"
            _stamp_image(dimg, s, "decoded from the annotation's imageData")
        elif ext in (".jpg", ".jpeg") and opts.link == "hardlink" and _jpeg_truncated(src):
            shutil.copy2(src, dimg)                    # trainers rewrite such JPEGs in place: never the original
            how, fix = "copied", "JPEG without end marker: copied, not linked (trainers repair it in place)"
        else:
            how = _place(src, dimg, opts.link)
        dlab.parent.mkdir(parents=True, exist_ok=True)
        with open(dlab, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        return {"split": split, "how": how, "written": written, "dropped": dropped, "excluded": excluded,
                "merged": merged, "background": not lines, "mismatch": mismatch, "renamed": renamed, "fix": fix,
                "label": dlab, "image": dimg}

    from .parallel import ordered_map
    done = 0
    pairs = []
    # in order, a bounded number in flight (a million images never become a million futures); threads only when the
    # files wait on storage (core.parallel)
    for job, r in ordered_map(one, jobs, opts.workers):
        done += 1
        if progress is not None and (done % 50 == 0 or done == total):
            progress(done, total)
        if r is None:
            st["cancelled"] = True
            break
        if isinstance(r, Exception):
            st["skipped"].append(f"{job[1].name}: {r}")
            continue
        if "skip" in r:
            st["skipped"].append(r["skip"])
            continue
        st["dropped"].update(r["dropped"])
        st["not_included"].update(r["excluded"])
        if r.get("left_out"):
            if r["left_out"] is True:
                st["left_out_background"] += 1
            else:
                st["left_out_images"][r["left_out"]] += 1
            continue
        st["renamed"].update(r["renamed"])
        if r["fix"]:
            st["fixed"][r["fix"]] += 1
        pairs.append((r["label"], r["image"]))
        st["images"][r["split"] or "all"] += 1
        st["placed"][r["how"]] += 1
        st["objects"].update(r["written"])
        st["merged"] += r["merged"]
        st["background"] += r["background"]
        st["size_mismatch"] += r["mismatch"]
    if st.get("cancelled"):
        return st
    if opts.verify:
        st["verify"] = verify(pairs, len(opts.classes), opts.task, opts.classes)
    splits = [sp for sp in samples_by_split if samples_by_split[sp]]
    atomic_write_text(out / "dataset.yaml", dataset_yaml(out, splits, opts.classes, opts.task, source_name))
    atomic_write_text(out / "classes.txt", "\n".join(opts.classes) + "\n")
    atomic_write_text(out / "conversion_report.txt", report_text(st, opts, source_name))
    created = (["images", "labels"] if splits == [""] else splits) + ["dataset.yaml", "classes.txt",
                                                                       "conversion_report.txt"] + \
        (["renamed.csv"] if st.get("renamed_files") else [])
    atomic_write_text(out / MANIFEST, json.dumps({"tool": "AIO QC Convert to YOLO", "task": opts.task,
                                                  "classes": opts.classes, "source": source_name,
                                                  "time": time.strftime("%Y-%m-%d %H:%M:%S"), "created": created},
                                                 indent=1, ensure_ascii=False))
    from . import provenance as PV
    from dataclasses import asdict
    PV.record(out, "convert", "Convert & Split", inputs=[root], input_fingerprints=False,
              settings=dict(asdict(opts), format="yolo_seg" if opts.task == "seg" else "yolo_det"),
              counts={"images": dict(st.get("images") or {}), "objects": sum((st.get("objects") or {}).values())},
              o=PV.make_origin("conversion", "Convert & Split", detail=f"to YOLO {opts.task}"))
    return st


def _jpeg_truncated(path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(-2, 2)
            return f.read() != b"\xff\xd9"
    except OSError:
        return False


def _stamp_image(path: Path, s, what: str):
    """An image this conversion made (not a link or copy of the original): its provenance, embedded."""
    from . import provenance as PV
    if PV.enabled():
        PV.embed_file(path, PV.image_record(PV.make_origin("conversion", "Convert & Split", detail=what),
                                            source=Path(str(s.image_path or s.ann_path or "")).name))


def _reencode(src, dst: Path):
    """Any image Pillow (or OpenCV) reads -> PNG, EXIF orientation applied (LabelMe's points are drawn on that)."""
    try:
        from PIL import Image, ImageOps
        im = Image.open(_io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "RGBA", "L", "I;16"):
            im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
        im.save(dst, "PNG")
        return
    except Exception as _exc:
        _ignored(_exc)
    import cv2
    import numpy as np
    buf = np.frombuffer(src, np.uint8) if isinstance(src, (bytes, bytearray)) else np.fromfile(str(src), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("unreadable image")
    ok, enc = cv2.imencode(".png", img)
    dst.write_bytes(enc.tobytes())


def verify(pairs, nc: int, task: str, names=None) -> dict:
    """Read every written label back: values, ids, coordinates; image present. -> counts + problems."""
    objs, problems = Counter(), []
    for lab, img in pairs:
        if not Path(img).is_file():
            problems.append(f"{Path(lab).name}: its image is missing")
        try:
            text = Path(lab).read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"{Path(lab).name}: {exc}")
            continue
        for n, ln in enumerate(text.splitlines(), 1):
            v = ln.split()
            if not v:
                continue
            try:
                c, xs = int(v[0]), [float(x) for x in v[1:]]
            except ValueError:
                problems.append(f"{Path(lab).name}:{n}: not numbers")
                continue
            bad = (not 0 <= c < nc) or any(not 0.0 <= x <= 1.0 for x in xs) or \
                  (len(xs) != 4 if task == "det" else (len(xs) < 6 or len(xs) % 2))
            if bad:
                problems.append(f"{Path(lab).name}:{n}: invalid line")
                continue
            objs[names[c] if names else c] += 1
    return {"files": len(pairs), "objects": objs, "problems": problems}


def report_text(st: dict, opts: Options, source: str = "") -> str:
    L = ["AIO QC — LabelMe to YOLO", "=" * 60, f"Source : {source}", f"Task   : {TASKS[opts.task]}",
         f"Classes: {len(opts.classes)} (ids in this order)"]
    L += [f"  {i:3d}  {c}  ({st['objects'].get(c, 0):,} objects)" for i, c in enumerate(opts.classes)]
    L += ["", "Images : " + ", ".join(f"{k} {v:,}" for k, v in st["images"].items()),
          f"  background (no object) : {st['background']:,}"]
    if st["left_out_background"]:
        L.append(f"  left out (no object)   : {st['left_out_background']:,}")
    for why, n in st.get("left_out_images", Counter()).most_common():
        L.append(f"  left out, {why} : {n:,}")
    if opts.folders == "flat":
        L.append("Folders: flattened (one images/ and labels/ per split)" +
                 (f" — {st['renamed_files']:,} file(s) renamed so no two share a name; renamed.csv lists them"
                  if st.get("renamed_files") else " — every name was already unique"))
    if st["merged"]:
        L.append(f"Objects drawn in pieces, merged (same group_id) : {st['merged']:,}")
    if st.get("renamed"):
        L += ["", "Written under another class name:"] + [f"  {k}: {v:,}" for k, v in st["renamed"].most_common()]
    if opts.parts:
        L += ["", "Parts kept only inside a whole:"] + [f"  {p} inside {' / '.join(w)} (>= {100 * float(r):.0f} %)"
                                                       for p, w, r in opts.parts]
    if st.get("fixed"):
        L += ["", "Images:"] + [f"  {k}: {v:,}" for k, v in st["fixed"].most_common()]
    if st["not_included"]:
        L += ["", "Labels not included (not written):"] + [f"  {k}: {v:,}" for k, v in st["not_included"].most_common()]
    if st["dropped"]:
        L += ["", "Shapes not written:"] + [f"  {k}: {v:,}" for k, v in st["dropped"].most_common()]
    if st["size_mismatch"]:
        L += ["", f"{st['size_mismatch']:,} JSON(s) give an image size other than the image file's; the JSON size "
                  "was used (the points were drawn in it). Check those images (Image Quality ▸ size mismatch)."]
    v = st.get("verify")
    if v is not None:
        same = v["objects"] == +st["objects"] and not v["problems"]
        L += ["", f"Verified: {v['files']:,} label files read back — "
                  + ("every object as written." if same else f"{len(v['problems'])} problem(s)")]
        L += [f"  {x}" for x in v["problems"][:100]]
    if st["skipped"]:
        L += ["", f"Skipped ({len(st['skipped']):,}):"] + [f"  {x}" for x in st["skipped"][:500]]
    return "\n".join(L) + "\n"
