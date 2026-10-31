# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Clean export: writes the (mapped) dataset into train / val / [test] in a chosen format.

    export(samples_by_split, adapter, out_dir, fmt, class_names, mode="hardlink", progress=None, cancel=None)

Formats: yolo_det (boxes) · yolo_seg (polygons) · labelme · coco · cls_folders (<split>/<class>/<image>).
Images are hard-linked when possible (no extra space), otherwise copied. Sub-folders are kept (existing split folder
names stripped); a name clash gets a numeric suffix instead of overwriting.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

from .io import atomic_write_text, image_size

FORMATS = {
    "yolo_det": "YOLO detection (boxes)",
    "yolo_seg": "YOLO segmentation (polygons)",
    "labelme": "LabelMe JSON",
    "coco": "COCO JSON",
    "cls_folders": "Class folders (classification)",
}
_STRIP = {"train", "val", "valid", "validation", "test", "images", "labels"}


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
    except Exception:
        return None


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


def export(samples_by_split: dict, adapter, root: Path, out: Path, fmt: str, class_names: list[str],
           mode: str = "hardlink", progress=None, cancel=None, is_classification: bool = False) -> dict:
    """samples_by_split: {"train": [Sample], "val": [...], "test": [...]} -> stats dict."""
    out = Path(out)
    root = Path(root)
    cid = {c: i for i, c in enumerate(class_names)}
    stats = {"images": Counter(), "objects": Counter(), "skipped": [], "unknown_labels": Counter(), "links": Counter()}
    taken: set = set()
    total = sum(len(v) for v in samples_by_split.values())
    done = 0
    coco = {}
    for split, samples in samples_by_split.items():
        if not samples:
            continue
        if fmt == "coco":
            coco[split] = {"images": [], "annotations": [], "categories": [
                {"id": i + 1, "name": c, "supercategory": ""} for i, c in enumerate(class_names)]}
        for s in samples:
            if cancel is not None and cancel.is_set():
                stats["cancelled"] = True
                return stats
            done += 1
            if progress is not None and done % 25 == 0:
                progress(done, total)
            try:
                ann = adapter.load(s, light=True)
            except Exception as exc:
                stats["skipped"].append(f"{s.name}: {exc}")
                continue
            img = ann.image_path or s.image_path
            if fmt == "cls_folders" and (img is None or not Path(img).is_file()):
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
                    stats["links"][link_or_copy(img, dst, mode)] += 1
                    stats["objects"][cls] += 1
                stats["images"][split] += 1
                continue
            W, H = (ann.width, ann.height) if ann.width and ann.height else (image_size(img) or (None, None))
            if not W or not H:
                stats["skipped"].append(f"{s.name}: unknown image size")
                continue
            shapes = ann.shapes
            if ann.normalized:
                shapes = [dict(x, points=[[p[0] * W, p[1] * H] for p in x.get("points") or []]) for x in shapes]
            rel = _rel(root, img)
            if fmt in ("yolo_det", "yolo_seg", "coco"):
                dst_img = _unique(out / split / "images" / rel, taken)
            else:
                dst_img = _unique(out / split / rel, taken)
            stats["links"][link_or_copy(img, dst_img, mode)] += 1
            stats["images"][split] += 1
            if fmt in ("yolo_det", "yolo_seg"):
                lines = []
                for x in shapes:
                    lab = str(x.get("label", ""))
                    if lab not in cid:
                        stats["unknown_labels"][lab] += 1
                        continue
                    poly = _poly(x)
                    if len(poly) < 2:
                        continue
                    if fmt == "yolo_det" or len(poly) < 3:
                        x1, y1, x2, y2 = _bbox(poly)
                        x1, x2 = max(0.0, x1), min(float(W), x2)
                        y1, y2 = max(0.0, y1), min(float(H), y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        lines.append(f"{cid[lab]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                    else:
                        lines.append(f"{cid[lab]} " + " ".join(f"{min(max(p[0] / W, 0), 1):.6f} {min(max(p[1] / H, 0), 1):.6f}"
                                                               for p in poly))
                    stats["objects"][lab] += 1
                lbl = out / split / "labels" / dst_img.relative_to(out / split / "images").with_suffix(".txt")
                lbl.parent.mkdir(parents=True, exist_ok=True)
                lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            elif fmt == "labelme":
                data = {"version": "5.0.0", "flags": {t: True for t in ann.tags},
                        "shapes": [{"label": str(x.get("label", "")), "points": x.get("points") or [],
                                    "group_id": x.get("group_id"), "description": x.get("description", ""),
                                    "shape_type": x.get("shape_type") or "polygon", "flags": x.get("flags") or {}}
                                   for x in shapes],
                        "imagePath": dst_img.name, "imageData": None, "imageHeight": int(H), "imageWidth": int(W)}
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
                    stats["objects"][lab] += 1
    for split, c in coco.items():
        atomic_write_text(out / "annotations" / f"instances_{split}.json", json.dumps(c, ensure_ascii=False))
    if fmt in ("yolo_det", "yolo_seg"):
        q = lambda p: str(p).replace("\\", "/")
        lines = [f"path: {q(out.resolve())}"]
        for split in ("train", "val", "test"):
            if stats["images"].get(split):
                lines.append(f"{split}: {split}/images")
        lines += ["", "names:"] + [f"  {i}: {c}" for i, c in enumerate(class_names)]
        atomic_write_text(out / "dataset.yaml", "\n".join(lines) + "\n")
    if progress is not None:
        progress(total, total)
    return stats


def write_reports(out: Path, report: dict, assign: dict, settings: dict, stats: dict):
    """split_report.txt · feature_summary.csv · split_manifest.json (as the pipeline's split script writes)."""
    d = Path(out) / "reports"
    d.mkdir(parents=True, exist_ok=True)
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
