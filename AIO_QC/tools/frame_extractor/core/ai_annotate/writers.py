# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Writing and reading annotations next to the images: YOLO txt (class ids in the project's class order, classes.txt,
data.yaml) and/or LabelMe (5.11.4 / 6.3.1) — Qt-free.

YOLO has no place for read text, confidence or image flags; those go to the run's side table (and a meta.jsonl
in each packaged part), so nothing the AI said is lost.
"""
from __future__ import annotations

import json
import os

from AIO_QC.tools.frame_extractor.core.autoannotate import labelme_io as LM
from AIO_QC.core.diag import ignored as _ignored


def find_classes_txt(folder: str, levels=3) -> str | None:
    d = os.path.abspath(folder)
    for _ in range(levels + 1):
        p = os.path.join(d, "classes.txt")
        if os.path.isfile(p):
            return p
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return None


def read_classes(path: str | None) -> list[str]:
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except OSError:
        return []


def _size(img_path):
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.size
    except Exception as _exc:
        _ignored(_exc)
        from AIO_QC.tools.frame_extractor.core.imageio import imread
        im = imread(img_path)
        return (im.shape[1], im.shape[0]) if im is not None else (0, 0)


def read_labels(img_path: str, with_meta=False):
    """[(label, (x1, y1, x2, y2) px)] from the image's LabelMe JSON, else its YOLO txt."""
    stem = os.path.splitext(img_path)[0]
    out = []
    if os.path.exists(stem + ".json"):
        try:
            with open(stem + ".json", encoding="utf-8") as f:
                d = json.load(f)
            for s in d.get("shapes") or []:
                pts = s.get("points") or []
                if len(pts) < 2:
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                out.append((str(s.get("label", "")), (min(xs), min(ys), max(xs), max(ys))))
            return out
        except (OSError, ValueError, TypeError, IndexError):
            return []
    if os.path.exists(stem + ".txt"):
        names = read_classes(find_classes_txt(os.path.dirname(img_path)))
        w, h = _size(img_path)
        try:
            with open(stem + ".txt", encoding="utf-8") as f:
                for ln in f:
                    p = ln.split()
                    if len(p) < 5:
                        continue
                    c = int(float(p[0]))
                    vals = [float(v) for v in p[1:]]
                    label = names[c] if 0 <= c < len(names) else str(c)
                    if len(vals) == 4:
                        cx, cy, bw, bh = vals
                        out.append((label, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w,
                                            (cy + bh / 2) * h)))
                    else:
                        xs, ys = vals[0::2], vals[1::2]
                        out.append((label, (min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h)))
        except (OSError, ValueError):
            return []
    return out


def labels_of(img_path: str) -> list[str]:
    return [lab for lab, _ in read_labels(img_path)]


def has_labels_file(img_path: str) -> bool:
    stem = os.path.splitext(img_path)[0]
    return os.path.exists(stem + ".json") or os.path.exists(stem + ".txt")


def write_classes(folders, names):
    for d in folders:
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "classes.txt.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(names) + "\n")
        os.replace(tmp, os.path.join(d, "classes.txt"))


def write_data_yaml(folder, names, train="images", val="images"):
    # no `path:` — ultralytics then reads train/val relative to this file, so a moved or unzipped part still works
    lines = [f"train: {train}", f"val: {val}",
             f"nc: {len(names)}", "names:"] + [f"  {i}: {json.dumps(n)}" for i, n in enumerate(names)]
    with open(os.path.join(folder, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _yolo_line(ci, s, w, h):
    if s.get("polygon"):
        pts = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in s["polygon"])
        return f"{ci} {pts}\n"
    x1, y1, x2, y2 = s["box"]
    return f"{ci} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n"


def _lm_shape(s, write_conf, origin=None):
    sh = _lm_shape_plain(s, write_conf)
    if origin is not None:
        from AIO_QC.core import provenance as PV
        if PV.enabled():
            sh[PV.KEY] = PV.shape_record(origin, score=s.get("conf"))
    return sh


def _lm_shape_plain(s, write_conf):
    desc = []
    if s.get("text"):
        desc.append(str(s["text"]))
    if write_conf and s.get("conf") is not None:
        desc.append(f"conf={float(s['conf']):.3f}")
    d = " ".join(desc)
    if s.get("polygon"):
        return {"label": s["label"], "points": [[round(x, 2), round(y, 2)] for x, y in s["polygon"]],
                "group_id": None, "description": d, "shape_type": "polygon", "flags": {}, "mask": None}
    return LM.rectangle(s["label"], *s["box"], description=d)


def write(img_path: str, w: int, h: int, shapes: list, out_cfg: dict, classes: list[str], flags: dict | None = None,
          origin=None):
    """Write the image's annotation file(s). Returns what was done: written | skipped | merged. origin: the
    provenance of the shapes (the model, each shape's confidence)."""
    fmt = out_cfg.get("format", "yolo")
    merge = out_cfg.get("merge_with_existing", "add")
    stem = os.path.splitext(img_path)[0]
    did = "written"
    if fmt in ("labelme", "both"):
        jp = stem + ".json"
        existing = None
        if os.path.exists(jp) and merge != "replace":
            if merge == "skip":
                did = "skipped"
            else:
                try:
                    with open(jp, encoding="utf-8") as f:
                        existing = json.load(f)
                except (OSError, ValueError):
                    existing = None
        if did != "skipped":
            new = [_lm_shape(s, out_cfg.get("write_confidence", True), origin) for s in shapes]
            if existing:
                have = {s.get("label") for s in existing.get("shapes") or []}
                existing["shapes"] = list(existing.get("shapes") or []) + [s for s in new if s["label"] not in have]
                existing.setdefault("flags", {}).update(flags or {})
                existing["version"] = out_cfg.get("labelme_version", existing.get("version", "6.3.1"))
                LM.write(jp, existing, origin=origin)
                did = "merged"
            else:
                LM.write(jp, LM.document(img_path, jp, w, h, new, out_cfg.get("labelme_version", "6.3.1"),
                                         bool(out_cfg.get("image_data")), flags), origin=origin)
    if fmt in ("yolo", "both"):
        tp = stem + ".txt"
        if os.path.exists(tp) and merge == "skip":
            return "skipped"
        lines = []
        have = set()
        if os.path.exists(tp) and merge == "add":
            with open(tp, encoding="utf-8") as f:
                lines = [ln if ln.endswith("\n") else ln + "\n" for ln in f if ln.strip()]
            have = {int(float(ln.split()[0])) for ln in lines if ln.split()}
            did = "merged" if lines else did
        for s in shapes:
            if s["label"] in classes:
                ci = classes.index(s["label"])
                if ci not in have:
                    lines.append(_yolo_line(ci, s, w, h))
        tmp = f"{tp}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(tmp, tp)
    return did
