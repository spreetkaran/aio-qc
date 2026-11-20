# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Clean export: writes the (mapped) dataset into train / val / [test] in a chosen format.

    export(samples_by_split, adapter, out_dir, fmt, class_names, mode="hardlink", progress=None, cancel=None)

Formats: yolo_det (boxes) · yolo_seg (polygons) · labelme · coco · cls_folders (<split>/<class>/<image>) · masks ·
crops · the export_more formats (VOC, CVAT, YOLO OBB, DOTA, YOLO pose).
Images are hard-linked when possible (no extra space), otherwise copied. Sub-folders are kept (existing split folder
names stripped); a name clash gets a numeric suffix instead of overwriting. The split "" is one set without split
folders (out/images …).

Large datasets (core.parallel): when files turn out to wait on storage (a network share, large copies), the annotations
are read ahead by a few threads (in order) and the images placed by a pool while the next ones are written; on a warm
local disk it stays in one thread, where Python is fastest.

ClassView(adapter, classes, rename, fold, left_out, background) wraps an adapter so that every format writes only the
chosen classes, under their output names (Convert & Split's class list); an image it leaves out raises LeftOut, which
export() counts in stats["left_out_images"] instead of "skipped".
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from collections import Counter
from dataclasses import replace as _replace
from pathlib import Path

from .io import atomic_write_text, image_size
from AIO_QC.core.diag import ignored as _ignored

FORMATS = {
    "yolo_det": "YOLO detection (boxes)",
    "yolo_seg": "YOLO segmentation (polygons)",
    "labelme": "LabelMe JSON",
    "coco": "COCO JSON",
    "cls_folders": "Class folders (classification)",
    "cls_csv": "Classification table (CSV: file, classes, split)",
    "crops": "Crops of every shape → class folders (a classifier from boxes)",
    "flags_folders": "Image flags → class folders (classification from a LabelMe dataset's flags)",
    "masks": "Segmentation masks (PNG: class index + instances)",
}
from .export_more import MORE as _MORE  # noqa: E402
FORMATS.update(_MORE)
CLS_FORMATS = ("cls_folders", "cls_csv", "flags_folders")
_STRIP = {"train", "val", "valid", "validation", "test", "images", "labels"}


from . import provenance as PV  # noqa: E402
from .parallel import Placer, default_workers, ordered_map  # noqa: E402

WORKERS = default_workers()


class LeftOut(Exception):
    """An image a rule leaves out (not an error): counted by its reason."""


class ClassView:
    """The adapter, as the chosen classes: labels renamed / merged (rename, case folding), other labels dropped —
    or the image left out (left_out="skip"); an image left without a written object is left out when
    background="skip". Everything else passes through to the adapter."""

    def __init__(self, adapter, classes, rename=None, fold=False, left_out="drop", background="keep",
                 tags: bool = False):
        from .yolo_convert import LabelIds
        self._adapter = adapter
        self.ids = LabelIds(list(classes), rename or {}, fold)
        self.left_out, self.background, self.tags = left_out, background, tags      # tags: a classifier's classes

    def __getattr__(self, name):
        return getattr(self._adapter, name)

    def load(self, sample, *a, **kw):
        ann = self._adapter.load(sample, *a, **kw)
        keep, excluded = [], 0
        for x in ann.shapes:
            lab = str(x.get("label", ""))
            hit = self.ids.get(lab)
            if hit is None:
                excluded += 1
                continue
            if hit[1] == lab:
                keep.append(x)
            else:                                       # written under another name: one more change on record
                y = dict(x, label=hit[1])
                if PV.enabled():
                    y[PV.KEY] = PV.bump(x.get(PV.KEY), ["label"], PV.make_origin("conversion", "Convert & Split"))
                keep.append(y)
        tags = list(ann.tags or [])
        if self.tags:
            mapped = []
            for c in tags:
                hit = self.ids.get(str(c))
                if hit is None:
                    excluded += 1
                else:
                    mapped.append(hit[1])
            tags = list(dict.fromkeys(mapped))
        if excluded and self.left_out == "skip":
            raise LeftOut("has a class that is not written")
        if not keep and (not tags or not self.tags) and self.background == "skip":
            raise LeftOut("no object of a written class" if not self.tags else "no written class")
        changed = excluded or tags != list(ann.tags or []) or any(a is not b for a, b in zip(keep, ann.shapes))
        return _replace(ann, shapes=keep, tags=tags) if changed else ann


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def _rel(root: Path, img: Path, cls: str | None = None) -> Path:
    try:
        parts = list(Path(img).relative_to(root).parts)
    except ValueError:
        parts = [Path(img).name]
    dirs = [p for p in parts[:-1] if p.lower() not in _STRIP]
    if cls is not None and dirs and dirs[0] == cls:                 # classification: the class folder is re-added
        dirs = dirs[1:]
    return Path(*dirs, parts[-1]) if dirs else Path(parts[-1])


def _embedded_image(json_path):
    """(bytes, ext) of a LabelMe JSON's imageData, verified — or None."""
    if json_path is None or Path(json_path).suffix.lower() != ".json":
        return None
    try:
        from .prep.common import decode_image_data, read_json
        d = read_json(json_path).get("imageData")
        return decode_image_data(d) if isinstance(d, str) and d.strip() else None
    except Exception as _exc:
        _ignored(_exc)
        return None


def _unique_stem(path: Path, taken: set) -> Path:
    """Unique by folder + stem, case-insensitive: a.jpg and a.png must not share labels/a.txt (YOLO pairs by name)."""
    k, cand = 0, path
    while (str(cand.parent).lower(), cand.stem.lower()) in taken:
        k += 1
        cand = path.with_name(f"{path.stem}_{k}{path.suffix}")
    taken.add((str(cand.parent).lower(), cand.stem.lower()))
    return cand


def _unique(path: Path, taken: set) -> Path:
    if str(path) not in taken:
        taken.add(str(path))
        return path
    i = 1
    while True:
        p = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if str(p) not in taken:
            taken.add(str(p))
            return p
        i += 1


def _poly(s: dict) -> list[list[float]]:
    pts = [list(map(float, p)) for p in s.get("points") or []]
    st = s.get("shape_type") or "polygon"
    if st == "rectangle" and len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[0], pts[1]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if st == "circle" and len(pts) >= 2:
        (cx, cy), (px, py) = pts[0], pts[1]
        r = math.hypot(px - cx, py - cy)
        return [[cx + r * math.cos(a * math.pi / 8), cy + r * math.sin(a * math.pi / 8)] for a in range(16)]
    return pts


def _bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _flat_names(samples_by_split, root: Path, out: Path, naming: dict) -> dict:
    """{(split, key): new stem} — unique per split (case ignored, by name without extension), verified with the
    naming engine; renamed.csv lists every change."""
    from . import naming as N
    items = []
    for split, samples in samples_by_split.items():
        for s in samples:
            main = s.image_path or s.ann_path
            if main is None:
                continue
            items.append(N.Item((split, s.key), _rel(root, Path(main)).as_posix(), split=split, path=str(main),
                                extra_exts=(".txt", ".json")))
    sc = N.Scheme.from_dict(naming or {"preset": "folder"})
    rl = len(str(Path(out).resolve())) + len("/train/images")
    P = N.plan(items, sc, flat=True, root_len=rl)
    if not P.ok:
        P = N.resolve(P, root_len=rl)
    changed = [it for it in P.items if P.new[it.key] != it.stem]
    if changed:
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(Path(out) / "renamed.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["original (without extension)", "new (without extension)"])
            for it in changed:
                old = it.rel[:-len(it.suffix)] if it.suffix else it.rel
                w.writerow([old, f"{it.key[0]}/{P.new[it.key]}" if it.key[0] else P.new[it.key]])
    return {it.key: P.new[it.key] for it in P.items}


def export(samples_by_split: dict, adapter, root: Path, out: Path, fmt: str, class_names: list[str],
           mode: str = "hardlink", progress=None, cancel=None, is_classification: bool = False,
           naming: dict | None = None, crop: dict | None = None, workers: int = WORKERS) -> dict:
    """samples_by_split: {"train": [Sample], "val": [...], "test": [...]} (or {"": [...]}: one set) -> stats dict.
    naming (a core.naming.Scheme as a dict): flatten — no sub-folders, names made unique per split with it
    (renamed.csv)."""
    out = Path(out)
    root = Path(root)
    flat_names = _flat_names(samples_by_split, root, out, naming) if naming is not None and fmt not in CLS_FORMATS \
        and fmt != "crops" else None
    if fmt == "flags_folders":                      # the image flags of a LabelMe (detection) dataset as classes
        fmt = "cls_folders"
        tags_from_flags = True
    else:
        tags_from_flags = False
    crop = dict({"pad": 0.1, "square": False, "min": 8}, **(crop or {}))
    table = [] if fmt == "cls_csv" else None
    cid = {c: i for i, c in enumerate(class_names)}
    conv = PV.make_origin("conversion", "Convert & Split", detail=f"to {FORMATS.get(fmt, fmt)}")
    stats = {"images": Counter(), "objects": Counter(), "skipped": [], "unknown_labels": Counter(), "links": Counter(),
             "left_out_images": Counter()}
    placer = Placer(workers)
    taken: set = set()
    total = sum(len(v) for v in samples_by_split.values())
    done = 0
    coco = {}
    writer = None
    if fmt in _MORE:
        from . import export_more as XM
        kpts = None
        if fmt == "yolo_pose":
            boxes, kpts = XM.keypoint_names([x for v in samples_by_split.values() for x in v], adapter)
            class_names = [c for c in class_names if c not in kpts] or boxes
            cid = {c: i for i, c in enumerate(class_names)}
            stats["keypoints"] = list(kpts)
        writer = XM.Writer(fmt, out, class_names, kpts)
    for split, samples in samples_by_split.items():
        if not samples:
            continue
        if fmt == "coco":
            coco[split] = {"images": [], "annotations": [], "categories": [
                {"id": i + 1, "name": c, "supercategory": ""} for i, c in enumerate(class_names)]}
        for s, ann in ordered_map(lambda x: adapter.load(x, light=True), samples, workers):
            if cancel is not None and cancel.is_set():
                placer.close()
                stats["links"].update(placer.counts)
                stats["cancelled"] = True
                return stats
            done += 1
            if progress is not None and done % 25 == 0:
                progress(done, total)
            if isinstance(ann, LeftOut):
                stats["left_out_images"][str(ann)] += 1
                continue
            if isinstance(ann, Exception):
                stats["skipped"].append(f"{s.name}: {ann}")
                continue
            img = ann.image_path or s.image_path
            if tags_from_flags and not ann.tags and isinstance(ann.raw, dict):
                from .dataset import flag_value
                ann.tags = [str(k) for k, v in (ann.raw.get("flags") or {}).items() if flag_value(v)[0]]
            if fmt in ("cls_folders", "cls_csv") and (img is None or not Path(img).is_file()):
                # LabelMe flags dataset whose JSONs carry the only copy of the image (imageData): decode it
                emb = _embedded_image(s.ann_path)
                if emb is None:
                    stats["skipped"].append(f"{s.name}: image not found (no image file, no imageData)")
                    continue
                raw, x = emb
                for cls in (ann.tags or ["_unlabelled"]):
                    dst = _unique(out / split / cls / (Path(s.ann_path).stem + x), taken)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(raw)
                    stats["links"]["decoded from imageData"] += 1
                    stats["objects"][cls] += 1
                stats["images"][split] += 1
                continue
            if img is None or not Path(img).is_file():
                stats["skipped"].append(f"{s.name}: image not found")
                continue
            img = Path(img)
            if fmt == "cls_folders":
                # every class the image has (multi-label flags) gets its copy (hard link: no extra space)
                for cls in (ann.tags or ["_unlabelled"]):
                    dst = _unique(out / split / cls / _rel(root, img, cls), taken)
                    placer.put(link_or_copy, img, dst, mode)
                    stats["objects"][cls] += 1
                stats["images"][split] += 1
                continue
            if fmt == "cls_csv":
                dst = _unique(out / split / "images" / _rel(root, img), taken)
                placer.put(link_or_copy, img, dst, mode)
                table.append((dst.relative_to(out).as_posix(), ";".join(ann.tags), split))
                stats["objects"].update(ann.tags or ["(no class)"])
                stats["images"][split] += 1
                continue
            if fmt == "crops":
                n = _write_crops(img, ann, out / split, _rel(root, img), crop, cid, taken, stats)
                stats["images"][split] += bool(n)
                continue
            W, H = (ann.width, ann.height) if ann.width and ann.height else (image_size(img) or (None, None))
            if not W or not H:
                stats["skipped"].append(f"{s.name}: unknown image size")
                continue
            shapes = ann.shapes
            if ann.normalized:
                shapes = [dict(x, points=[[p[0] * W, p[1] * H] for p in x.get("points") or []]) for x in shapes]
            rel = _rel(root, img)
            if flat_names is not None:
                rel = Path(flat_names.get((split, s.key), img.stem) + img.suffix)
            if fmt in ("yolo_det", "yolo_seg", "masks") or writer is not None:
                dst_img = _unique_stem(out / split / "images" / rel, taken)
            elif fmt == "coco":
                dst_img = _unique(out / split / "images" / rel, taken)
            else:
                dst_img = _unique(out / split / rel, taken)
            placer.put(link_or_copy, img, dst_img, mode)
            stats["images"][split] += 1
            if writer is not None:
                writer.add(split, dst_img, shapes, W, H, ann.tags, stats)
            elif fmt in ("yolo_det", "yolo_seg"):
                from .yolo_convert import objects
                lines, written, dropped, excluded, _merged = objects(shapes, float(W), float(H), cid,
                                                                     "det" if fmt == "yolo_det" else "seg")
                stats["objects"].update(written)
                stats["unknown_labels"].update(excluded)
                stats.setdefault("dropped", Counter()).update(dropped)
                lbl = out / split / "labels" / dst_img.relative_to(out / split / "images").with_suffix(".txt")
                lbl.parent.mkdir(parents=True, exist_ok=True)
                lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            elif fmt == "masks":
                from .prep.masks import shapes_mask, write_instance_png, write_mask_png
                sem, inst, n = shapes_mask(shapes, int(W), int(H), cid)
                rel_png = dst_img.relative_to(out / split / "images").with_suffix(".png")
                write_mask_png(out / split / "masks" / rel_png, sem, len(class_names))
                write_instance_png(out / split / "instances" / rel_png, inst)
                if PV.enabled():
                    rec = PV.image_record(conv, source=img.name, classes=len(class_names))
                    PV.embed_file(out / split / "masks" / rel_png, rec)
                    PV.embed_file(out / split / "instances" / rel_png, rec)
                for x in shapes:
                    lab = str(x.get("label", ""))
                    if lab in cid:
                        stats["objects"][lab] += 1
                    else:
                        stats["unknown_labels"][lab] += 1
            elif fmt == "labelme":
                flags = {c: c in ann.tags for c in class_names} if is_classification else {}
                flags.update({t: True for t in ann.tags})
                data = {"version": "5.0.0", "flags": flags,
                        "shapes": [dict({"label": str(x.get("label", "")), "points": x.get("points") or [],
                                         "group_id": x.get("group_id"), "description": x.get("description", ""),
                                         "shape_type": x.get("shape_type") or "polygon", "flags": x.get("flags") or {}},
                                        **({PV.KEY: x[PV.KEY]} if isinstance(x.get(PV.KEY), dict) else {}))
                                   for x in shapes],
                        "imagePath": dst_img.name, "imageData": None, "imageHeight": int(H), "imageWidth": int(W)}
                raw = ann.raw if isinstance(ann.raw, dict) else {}
                PV.derive_labelme(data, raw.get(PV.KEY), conv, source=Path(str(s.ann_path or img)).name)
                for x in shapes:
                    stats["objects"][str(x.get("label", ""))] += 1
                atomic_write_text(dst_img.with_suffix(".json"), json.dumps(data, ensure_ascii=False, indent=2))
            elif fmt == "coco":
                c = coco[split]
                image_id = len(c["images"]) + 1
                c["images"].append({"id": image_id, "file_name": dst_img.relative_to(out / split / "images").as_posix(),
                                    "width": int(W), "height": int(H)})
                for x in shapes:
                    lab = str(x.get("label", ""))
                    if lab not in cid:
                        stats["unknown_labels"][lab] += 1
                        continue
                    poly = _poly(x)
                    if len(poly) < 2:
                        continue
                    x1, y1, x2, y2 = _bbox(poly)
                    seg = [[round(v, 2) for p in poly for v in p]] if len(poly) >= 3 else []
                    area = abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                                   for i in range(len(poly)))) / 2 if len(poly) >= 3 else (x2 - x1) * (y2 - y1)
                    c["annotations"].append({"id": len(c["annotations"]) + 1, "image_id": image_id,
                                             "category_id": cid[lab] + 1, "bbox": [round(x1, 2), round(y1, 2),
                                                                                    round(x2 - x1, 2), round(y2 - y1, 2)],
                                             "area": round(area, 2), "segmentation": seg, "iscrowd": 0})
                    if isinstance(x.get(PV.KEY), dict):
                        c["annotations"][-1][PV.KEY] = x[PV.KEY]
                    stats["objects"][lab] += 1
    placer.close()
    stats["links"].update(placer.counts)
    stats["skipped"] += placer.errors
    if writer is not None:
        writer.finish(stats, root.name)
    for split, c in coco.items():
        atomic_write_text(out / "annotations" / (f"instances_{split}.json" if split else "instances.json"),
                          json.dumps(c, ensure_ascii=False))
    if fmt == "masks":
        from .prep.masks import write_labelmap
        out.mkdir(parents=True, exist_ok=True)
        write_labelmap(out, class_names)
        atomic_write_text(out / "classes.txt", "background\n" + "\n".join(class_names) + "\n")
    if table is not None:
        with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "labels", "split"])
            w.writerows(table)
        atomic_write_text(out / "classes.txt", "\n".join(class_names) + "\n")
    if fmt in ("yolo_det", "yolo_seg"):
        from .yolo_convert import dataset_yaml
        splits = [""] if stats["images"].get("") else [sp for sp in ("train", "val", "test") if stats["images"].get(sp)]
        atomic_write_text(out / "dataset.yaml", dataset_yaml(out, splits, class_names,
                                                             "det" if fmt == "yolo_det" else "seg", root.name))
        atomic_write_text(out / "classes.txt", "\n".join(class_names) + "\n")
    if progress is not None:
        progress(total, total)
    PV.record(out, "convert", "Convert & Split", inputs=[root], input_fingerprints=False, o=conv,
              settings={"format": fmt, "classes": list(class_names), "link": mode,
                        **({"naming": naming} if naming else {}), **({"crop": crop} if fmt == "crops" else {})},
              counts={"images": dict(stats["images"]), "objects": sum(stats["objects"].values()),
                      "left_out": sum(stats["left_out_images"].values()), "skipped": len(stats["skipped"])})
    return stats


def _write_crops(img: Path, ann, base: Path, rel: Path, opt: dict, cid: dict, taken: set, stats: dict) -> int:
    """Every shape cut out (its box + pad %, square if asked) → <base>/<label>/<image stem>_<n>.<ext>."""
    import cv2
    import numpy as np
    shapes = [x for x in ann.shapes if x.get("points")]
    if not shapes:
        return 0
    data = np.fromfile(str(img), np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if im is None:
        stats["skipped"].append(f"{img.name}: image could not be read")
        return 0
    H, W = im.shape[:2]
    n = 0
    for i, x in enumerate(shapes):
        lab = str(x.get("label", ""))
        if cid and lab not in cid:
            stats["unknown_labels"][lab] += 1
            continue
        pts = x["points"]
        if ann.normalized:
            pts = [[p[0] * W, p[1] * H] for p in pts]
        if x.get("shape_type") == "circle" and len(pts) == 2:
            r = ((pts[1][0] - pts[0][0]) ** 2 + (pts[1][1] - pts[0][1]) ** 2) ** 0.5
            pts = [[pts[0][0] - r, pts[0][1] - r], [pts[0][0] + r, pts[0][1] + r]]
        x1, y1, x2, y2 = _bbox(pts)
        w, h = x2 - x1, y2 - y1
        if min(w, h) < opt["min"]:
            stats.setdefault("dropped", Counter())[f"{lab} smaller than {opt['min']} px"] += 1
            continue
        px, py = w * opt["pad"], h * opt["pad"]
        x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py
        if opt["square"]:
            side = max(x2 - x1, y2 - y1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            x1, y1, x2, y2 = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2
        a, b, c, d = int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))
        if c - a < 2 or d - b < 2:
            continue
        safe = "".join("_" if ch in '\\/:*?"<>|' else ch for ch in lab).strip() or "_"
        ext = img.suffix.lower() if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp") else ".png"
        dst = _unique(base / safe / rel.parent / f"{img.stem}_{i:03d}{ext}", taken)
        dst.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(ext, im[b:d, a:c])
        if ok:
            data = buf.tobytes()
            if PV.enabled():                              # the crop says which image and which object it is
                r = x.get(PV.KEY) if isinstance(x.get(PV.KEY), dict) else {}
                data = PV.embed_bytes(data, PV.image_record(PV.make_origin("conversion", "Convert & Split",
                                                                           detail="crop of an object"),
                                                            source=img.name, label=lab, object=r.get("id"),
                                                            box=[int(a), int(b), int(c), int(d)]))
            with open(dst, "wb") as f:
                f.write(data)
            stats["objects"][lab] += 1
            stats["links"]["crops"] += 1
            n += 1
    return n


def write_reports(out: Path, report: dict, assign: dict, settings: dict, stats: dict, items=None):
    """split_report.txt · feature_summary.csv · feature_map.csv · split_manifest.json (as the pipeline's split
    script writes)."""
    d = Path(out) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    if items:
        from .split import write_feature_map
        write_feature_map(d / "feature_map.csv", items, assign)
    with open(d / "feature_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature_type", "feature", "images", "train", "val", "test", "status"])
        for r in report["features"]:
            w.writerow([r["kind"], r["value"], r["images"], r["train"], r["val"], r["test"], r["status"]])
    manifest = {"settings": settings, "counts": report["splits"], "groups": report["groups"],
                "export": {"images": dict(stats["images"]), "objects": dict(stats["objects"]),
                           "skipped": stats["skipped"][:200], "unknown_labels": dict(stats["unknown_labels"])},
                "splits": {sp: sorted(k for k, v in assign.items() if v == sp) for sp in ("train", "val", "test")}}
    atomic_write_text(d / "split_manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False))
    total = max(report["total"], 1)
    lines = ["=" * 64, "DATASET EXPORT / SPLIT REPORT", "=" * 64, f"Total images : {report['total']:,}"]
    for sp in ("train", "val", "test"):
        n = report["splits"].get(sp, 0)
        if n:
            lines.append(f"{sp.capitalize():6s} images : {n:,} ({100 * n / total:.1f} %)")
    g = report["groups"]
    lines += ["", f"Groups kept together : {g['multi']:,} (largest {g['largest']}) · split across sets: {g['split_across']}",
              "", "PER CLASS (images)", "-" * 64]
    for c, v in report["classes"].items():
        lines.append(f"{c:28s} train={v.get('train', 0):6d} val={v.get('val', 0):6d} test={v.get('test', 0):6d}")
    flags = [r for r in report["features"] if r["status"] != "ok"]
    lines += ["", "FEATURE NOTES", "-" * 64] + ([f"{r['kind']}:{r['value']} -> {r['status']} ({r['images']} images)"
                                                 for r in flags] or ["none"])
    if stats["skipped"]:
        lines += ["", f"SKIPPED ({len(stats['skipped'])})", "-" * 64] + stats["skipped"][:50]
    atomic_write_text(d / "split_report.txt", "\n".join(lines) + "\n")


def report_text(stats: dict, fmt: str, class_names: list[str], source: str = "") -> str:
    """export_report.txt: what was written, left out and skipped."""
    lines = ["=" * 64, f"AIO QC CONVERT & SPLIT — {FORMATS.get(fmt, fmt)}", "=" * 64,
             f"Source  : {source}", f"Written : {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for sp, n in stats["images"].items():
        lines.append(f"{(sp or 'images').capitalize():8s}: {n:,} image(s)")
    if class_names:
        lines += ["", "CLASSES (in order)", "-" * 64] + [f"{i:4d}  {c}  ({stats['objects'].get(c, 0):,} objects)"
                                                        for i, c in enumerate(class_names)]
    lo = stats.get("left_out_images") or {}
    if lo:
        lines += ["", "LEFT OUT BY YOUR RULES", "-" * 64] + [f"{n:,}  {why}" for why, n in lo.items()]
    if stats["unknown_labels"]:
        lines += ["", "LABELS NOT WRITTEN", "-" * 64] + [f"{n:,}  {k}" for k, n in
                                                         sorted(stats["unknown_labels"].items(), key=lambda kv: -kv[1])]
    if stats.get("dropped"):
        lines += ["", "OBJECTS NOT WRITTEN", "-" * 64] + [f"{n:,}  {k}" for k, n in stats["dropped"].items()]
    if stats["skipped"]:
        lines += ["", f"SKIPPED ({len(stats['skipped']):,})", "-" * 64] + stats["skipped"][:500]
    return "\n".join(lines) + "\n"


def write_manifest(out: Path, fmt: str, class_names: list[str], source: str, created: list[str]):
    """The same manifest the YOLO converter writes: a later run into the folder replaces exactly this, Undo removes
    it — and nothing else in the folder."""
    from .yolo_convert import MANIFEST
    atomic_write_text(Path(out) / MANIFEST, json.dumps(
        {"tool": "AIO QC Convert & Split", "task": fmt, "format": fmt, "classes": list(class_names), "source": source,
         "time": time.strftime("%Y-%m-%d %H:%M:%S"), "created": sorted(created)}, indent=1, ensure_ascii=False))
