# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Frame Extractor output folder -> Image QC dataset (Qt-free).

The Frame Extractor writes frames with optional pre-annotations next to them: LabelMe .json (opens in Image QC as
it is), or YOLO .txt + classes.txt, or nothing. Image QC opens LabelMe folders and YOLO images/ + labels/ layouts,
so a folder of frames with YOLO .txt beside them, or with no labels at all, gets LabelMe files added next to each
frame first (YOLO boxes converted with the names from classes.txt; frames without labels get an empty file, ready to
label). The .txt files are left untouched; nothing is moved or overwritten.

    inspect(folder)              -> {"kind": "labelme" | "yolo" | "images" | "empty", "images", "labels", "missing"}
    prepare_labelme(folder, ...) -> number of LabelMe files written
"""
from __future__ import annotations

import json
import os

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SKIP = {"_rejected", ".fx_resume", "cache", "reports", "__pycache__"}


def _frames(folder):
    for r, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith("."))
        for f in sorted(files):
            if f.lower().endswith(IMAGE_EXTS):
                yield os.path.join(r, f)


CLASS_FOLDERS_MARK = ".aio_class_folders"      # the Frame Extractor sorted frames into class folders (a classifier)


def inspect(folder) -> dict:
    """kind: labelme | yolo | images | empty — or, for a classifier's pre-labels, cls_folders (frames sorted into
    class folders) / cls_labelme (LabelMe files whose flags are the classes, no shapes)."""
    imgs = list(_frames(folder))
    mark = _class_folders_root(folder)
    if imgs and mark is not None:                   # the class-folder dataset is the folder holding the marker
        loose = sum(1 for p in imgs if os.path.normcase(os.path.dirname(os.path.abspath(p))) ==
                    os.path.normcase(os.path.abspath(mark)))
        n = sum(1 for p in imgs if os.path.abspath(p).startswith(os.path.abspath(mark) + os.sep))
        return {"kind": "cls_folders", "images": n, "labels": n - loose, "missing": loose, "root": mark}
    jsons = [os.path.splitext(p)[0] + ".json" for p in imgs]
    have = [j for j in jsons if os.path.isfile(j)]
    js = len(have)
    tx = sum(1 for p in imgs if os.path.isfile(os.path.splitext(p)[0] + ".txt"))
    kind = "empty" if not imgs else "labelme" if js and js >= tx else "yolo" if tx else "images"
    if kind == "labelme" and _flag_classes(have[:60]):
        kind = "cls_labelme"
    return {"kind": kind, "images": len(imgs), "labels": max(js, tx), "missing": len(imgs) - js}


def _class_folders_root(folder, depth=2):
    """The folder (this one, or one or two levels down) where the Frame Extractor sorted frames into class folders."""
    if os.path.isfile(os.path.join(folder, CLASS_FOLDERS_MARK)):
        return str(folder)
    if depth <= 0:
        return None
    try:
        subs = sorted(e.path for e in os.scandir(folder) if e.is_dir() and not e.name.startswith(".")
                      and e.name not in SKIP)
    except OSError:
        return None
    for d in subs:
        r = _class_folders_root(d, depth - 1)
        if r is not None:
            return r
    return None


def _flag_classes(jsons) -> bool:
    """LabelMe files with no shapes whose flags name a class (a classifier's pre-labels)."""
    flagged = 0
    for j in jsons:
        try:
            with open(j, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if d.get("shapes"):
            return False
        flagged += any(v is True for v in (d.get("flags") or {}).values())
    return flagged > 0


def _classes(folder, img_dir):
    for d in (img_dir, folder, os.path.dirname(folder)):
        p = os.path.join(d, "classes.txt")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
    return []


def prepare_labelme(folder, progress=None, cancel=None) -> int:
    from AIO_QC.core.testing.dataset import image_size, parse_yolo
    imgs = list(_frames(folder))
    n = 0
    cls_cache = {}
    for i, p in enumerate(imgs):
        if cancel is not None and cancel():
            break
        stem = os.path.splitext(p)[0]
        js = stem + ".json"
        if os.path.exists(js):
            continue
        w, h = image_size(p)
        shapes = []
        txt = stem + ".txt"
        if os.path.isfile(txt) and w and h:
            d = os.path.dirname(p)
            names = cls_cache.setdefault(d, _classes(folder, d))
            boxes, cls = parse_yolo(txt, w, h)
            for (x1, y1, x2, y2), c in zip(boxes if boxes is not None else [], cls if cls is not None else []):
                label = names[int(c)] if 0 <= int(c) < len(names) else str(int(c))
                shapes.append({"label": label, "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                               "group_id": None, "description": "", "shape_type": "rectangle", "flags": {}})
        data = {"version": "5.4.1", "flags": {}, "shapes": shapes, "imagePath": os.path.basename(p),
                "imageData": None, "imageHeight": int(h or 0), "imageWidth": int(w or 0)}
        from . import provenance as PV
        PV.stamp_labelme(data, o=PV.make_origin("conversion", "Frames to Image QC",
                                                detail="YOLO pre-labels as LabelMe" if shapes else ""))
        tmp = js + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, js)
        n += 1
        if progress and i % 200 == 0:
            progress(i + 1, len(imgs))
    return n
