# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Writing label corrections back (Model Lab quick fixes) and per-dataset exclusions - Qt-free.

save_labels(item, boxes, cls, origin, names, ...) rewrites the item's label file:
  * YOLO txt  - original lines are kept verbatim for boxes that still exist (segmentation polygons, extra
                columns survive); only the class id changes when you re-classed one; new boxes are appended
  * LabelMe   - original shapes are kept (polygons, flags, descriptions ...) with the new label when re-classed;
                deleted ones are dropped; new boxes become rectangles
  origin[k] = index of box k in the ORIGINAL file (as parsed by core.lab.dataset), or -1 for a new box.
An image without a label file gets one (YOLO next to the dataset's other labels, or LabelMe when the dataset
is LabelMe). A file you edit counts as reviewed ground truth (pre-label fingerprints no longer match).

Exclusions ("leave this image out of scoring") live in the test set manifest, or for plain datasets in
~/.frame_extractor/lab/exclusions/<hash of the dataset root>.json - the dataset folder is never touched.
"""
import hashlib
import json
import os

import numpy as np

from AIO_QC.core.testing import dataset as D


# ------------------------------------------------------------------ where the labels are
def label_target(item, data_fmt="yolo", testset=None):
    """(path, fmt) of the label file for an item - existing, or where a new one belongs."""
    lp = item.label_path or ""
    if lp and os.path.isfile(lp):
        return lp, ("labelme" if lp.lower().endswith(".json") else "yolo")
    stem = os.path.splitext(item.path)[0]
    if os.path.isfile(stem + ".json"):
        return stem + ".json", "labelme"
    if testset is not None:
        rec = _testset_record(testset, item.path)
        if rec is not None:
            return testset.label_path(rec, "labelme" if data_fmt == "labelme" else "yolo"), \
                ("labelme" if data_fmt == "labelme" else "yolo")
    if data_fmt == "labelme":
        return stem + ".json", "labelme"
    return (D.yolo_label_path(item.path) or stem + ".txt"), "yolo"


def _testset_record(ts, path):
    ap = os.path.abspath(path)
    for rec in ts.items:
        if os.path.abspath(ts.abspath(rec)) == ap:
            return rec
    return None


# ------------------------------------------------------------------ writing
def _yolo_entries(path):
    """The label lines that core.lab.dataset.parse_yolo turns into boxes, in the same order."""
    out = []
    try:
        lines = open(path, encoding="utf-8").read().split("\n")
    except OSError:
        return out
    for line in lines:
        v = line.split()
        if len(v) < 5:
            continue
        try:
            int(float(v[0]))
            nums = [float(x) for x in v[1:]]
        except ValueError:
            continue
        if len(nums) in (4, 5) or (len(nums) >= 6 and len(nums) % 2 == 0):
            out.append(line)
    return out


def _labelme_shapes(d):
    """Indices of the shapes that core.lab.dataset.parse_labelme turns into boxes (same order)."""
    idx = []
    for k, s in enumerate(d.get("shapes", [])):
        pts = np.asarray(s.get("points") or [], np.float32).reshape(-1, 2)
        st = s.get("shape_type") or "polygon"
        if not len(pts) or st in ("point", "line", "linestrip", "points"):
            continue
        if st == "circle" and len(pts) == 2:
            r = float(np.hypot(*(pts[1] - pts[0])))
            b = [pts[0][0] - r, pts[0][1] - r, pts[0][0] + r, pts[0][1] + r]
        else:
            b = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
        if b[2] - b[0] < 1 or b[3] - b[1] < 1:
            continue
        idx.append(k)
    return idx


def _atomic_write(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".fxtmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def save_labels(item, boxes, cls, origin, names, data_fmt="yolo", testset=None):
    """Write the corrected labels of one image; updates the item in place. Returns the label path."""
    path, fmt = label_target(item, data_fmt, testset)
    w, h = item.w, item.h
    if not w or not h:
        w, h = D.image_size(item.path)
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    cls = np.asarray(cls, int).reshape(-1)
    if fmt == "yolo":
        orig = _yolo_entries(path) if os.path.isfile(path) else []
        lines = []
        for b, c, o in zip(boxes, cls, origin):
            if 0 <= o < len(orig):
                v = orig[o].split()
                v[0] = str(int(c))
                lines.append(" ".join(v))
            else:
                x1, y1, x2, y2 = (float(t) for t in b)
                lines.append(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} "
                             f"{(y2 - y1) / h:.6f}")
        _atomic_write(path, "\n".join(lines) + ("\n" if lines else ""))
    else:
        try:
            d = json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else None
        except (OSError, ValueError):
            d = None
        if d is None:
            d = dict(version="5.0.1", flags={}, shapes=[], imagePath=os.path.basename(item.path), imageData=None,
                     imageHeight=int(h), imageWidth=int(w))
        shapes = d.get("shapes", [])
        box_shapes = _labelme_shapes(d)
        used = set(box_shapes)
        keep_other = [s for k, s in enumerate(shapes) if k not in used]      # points / lines are left alone
        new_shapes = []
        for b, c, o in zip(boxes, cls, origin):
            label = str(names.get(int(c), int(c)))
            if 0 <= o < len(box_shapes):
                s = dict(shapes[box_shapes[o]])
                s["label"] = label
            else:
                x1, y1, x2, y2 = (float(t) for t in b)
                s = dict(label=label, points=[[x1, y1], [x2, y2]], group_id=None, description="",
                         shape_type="rectangle", flags={})
            new_shapes.append(s)
        d["shapes"] = new_shapes + keep_other
        if isinstance(d.get("flags"), dict):
            d["flags"].pop("prelabel", None)
        _atomic_write(path, json.dumps(d, indent=2, ensure_ascii=False))
    item.boxes, item.cls, item.label_path = boxes, cls, path
    item.reviewed = True
    if item.meta.get("excluded") == "unreviewed pre-labels":
        item.meta.pop("excluded", None)
    if testset is not None:
        testset.mark_reviewed([path])
    return path


# ------------------------------------------------------------------ exclusions
def _excl_file(root):
    from AIO_QC.core.state import STATE_DIR
    own = os.path.join(os.path.abspath(root), STATE_DIR)
    if os.path.isdir(own):                 # a dataset AIO QC knows: "left out of scoring" travels with the dataset
        return os.path.join(own, "testing_excluded.json")
    from AIO_QC.core.fx_storage import testing_home
    d = str(testing_home() / "exclusions")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, hashlib.sha1(os.path.abspath(root).encode()).hexdigest()[:16] + ".json")


def _own(root):
    from AIO_QC.core.state import STATE_DIR
    return os.path.isdir(os.path.join(os.path.abspath(root), STATE_DIR))


def load_exclusions(root, testset=None):
    if testset is not None:
        return set(testset.manifest.get("excluded") or [])
    try:
        raw = json.load(open(_excl_file(root), encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if _own(root):                         # stored relative to the dataset (it may move)
        return {os.path.abspath(os.path.join(root, p)) for p in raw}
    return set(raw)


def set_excluded(item, on, root, testset=None):
    """Leave an image out of scoring (on) or bring it back. Persists; updates the item."""
    key = os.path.abspath(item.path)
    ex = load_exclusions(root, testset)
    (ex.add if on else ex.discard)(key)
    if testset is not None:
        testset.manifest["excluded"] = sorted(ex)
        testset.save()
    else:
        rows = sorted(os.path.relpath(p, root).replace(os.sep, "/") for p in ex) if _own(root) else sorted(ex)
        _atomic_write(_excl_file(root), json.dumps(rows, indent=1))
    if on:
        item.meta["excluded"] = "excluded by you"
    elif item.meta.get("excluded") == "excluded by you":
        item.meta.pop("excluded", None)


def apply_exclusions(dataset, root, testset=None):
    ex = load_exclusions(root, testset)
    for it in dataset.items:
        if os.path.abspath(it.path) in ex:
            it.meta["excluded"] = "excluded by you"
    return len(ex)
