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


def inspect(folder) -> dict:
    imgs = list(_frames(folder))
    js = sum(1 for p in imgs if os.path.isfile(os.path.splitext(p)[0] + ".json"))
    tx = sum(1 for p in imgs if os.path.isfile(os.path.splitext(p)[0] + ".txt"))
    kind = "empty" if not imgs else "labelme" if js and js >= tx else "yolo" if tx else "images"
    return {"kind": kind, "images": len(imgs), "labels": max(js, tx), "missing": len(imgs) - js}


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
        tmp = js + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, js)
        n += 1
        if progress and i % 200 == 0:
            progress(i + 1, len(imgs))
    return n
