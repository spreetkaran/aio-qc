# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
More Convert & Split formats (Qt-free), one writer each — export() places the images, a writer writes the labels:

    voc         Pascal VOC XML: <split>/images/<image> + <split>/Annotations/<stem>.xml (one <object> per shape, its
                box; difficult / truncated / occluded from the shape's flags) + <split>/ImageSets/Main/<split>.txt +
                labels.txt — opens in desktop box tools and every VOC reader
    cvat        CVAT for images 1.1: <split>/images/<image> + annotations.xml (one file, every split as a subset):
                box, polygon, polyline, points, ellipse (circle); shape flags as attributes; image classes as tags
    yolo_obb    YOLO oriented boxes: "class x1 y1 x2 y2 x3 y3 x4 y4" (0…1). A rectangle keeps its corners, a 4-point
                polygon its points, any other area shape gets its smallest enclosing rotated box
    dota        DOTA: <split>/images + <split>/labelTxt/<stem>.txt — "x1 y1 … x4 y4 class difficult" in pixels
    yolo_pose   YOLO pose: "class cx cy w h  x y v …" — each box with the points of its group (same group id; or the
                image's only box), in one keypoint order for the whole dataset; dataset.yaml gets kpt_shape and,
                when names say left / right, flip_idx
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from .io import atomic_write_text

MORE = {
    "voc": "Pascal VOC XML (boxes)",
    "cvat": "CVAT for images 1.1 XML",
    "yolo_obb": "YOLO oriented boxes (OBB)",
    "dota": "DOTA (oriented boxes, pixel corners)",
    "yolo_pose": "YOLO pose (boxes + keypoints)",
}
AREA = {"rectangle", "polygon", "circle", "mask", "oriented rectangle"}
TRUE = {"1", "true", "yes", "y"}


def _pts(x) -> list:
    return [[float(p[0]), float(p[1])] for p in x.get("points") or []]


def _area_poly(x) -> list:
    from .export import _poly
    return _poly(x)


def _bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _clipf(v, hi):
    return min(max(v, 0.0), float(hi))


def _flag(x, *names) -> bool:
    f = x.get("flags") or {}
    for n in names:
        v = f.get(n)
        if v is True or str(v).strip().lower() in TRUE:
            return True
    return False


def quad(x) -> list | None:
    """Four corners of a shape's oriented box (pixels), or None for shapes without an area."""
    st = x.get("shape_type") or "polygon"
    pts = _pts(x)
    if st not in AREA or len(pts) < 2:
        return None
    if st == "rectangle":
        (x1, y1), (x2, y2) = pts[0], pts[1]
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if len(pts) == 4 and st != "circle":
        return pts
    import cv2
    import numpy as np
    poly = _area_poly(x)
    if len(poly) < 3:
        return None
    return cv2.boxPoints(cv2.minAreaRect(np.asarray(poly, np.float32))).tolist()


def flip_idx(names: list[str]) -> list[int] | None:
    """Index of each keypoint's mirror (left ↔ right) — None when no names say left / right."""
    low = [n.lower() for n in names]
    out, found = [], False
    for i, n in enumerate(low):
        j = i
        for a, b in (("left", "right"), ("l_", "r_"), ("_l", "_r")):
            for s, t in ((a, b), (b, a)):
                m = n.replace(s, t)
                if s in n and m != n and m in low:
                    j = low.index(m)
                    break
            if j != i:
                break
        found = found or j != i
        out.append(j)
    return out if found else None


def keypoint_names(samples, adapter) -> tuple[list[str], list[str]]:
    """(box classes, keypoint names) in first-seen order — a pass over the dataset before a pose export."""
    boxes, kpts = [], []
    for s in samples:
        try:
            ann = adapter.load(s, light=True)
        except Exception:
            continue
        for x in ann.shapes:
            lab = str(x.get("label", ""))
            tgt = kpts if (x.get("shape_type") or "") == "point" else boxes
            if lab not in tgt:
                tgt.append(lab)
    return boxes, kpts


class Writer:
    def __init__(self, fmt: str, out: Path, class_names: list[str], kpt_names: list[str] | None = None):
        self.fmt, self.out, self.names = fmt, Path(out), list(class_names)
        self.cid = {c: i for i, c in enumerate(self.names)}
        self.kpts = list(kpt_names or [])
        self.kid = {k: i for i, k in enumerate(self.kpts)}
        self.sets = defaultdict(list)                     # VOC ImageSets
        self.cvat = ET.Element("annotations")
        ET.SubElement(self.cvat, "version").text = "1.1"
        self._cvat_meta = ET.SubElement(self.cvat, "meta")
        self.n_img = 0

    # -------------------------------------------------------------------------------------------- per image
    def add(self, split: str, dst_img: Path, shapes: list, W: float, H: float, tags: list, stats: dict):
        getattr(self, "_" + self.fmt)(split, dst_img, shapes, float(W), float(H), list(tags or []), stats)

    def _label_file(self, split, dst_img, folder, ext) -> Path:
        rel = dst_img.relative_to(self.out / split / "images").with_suffix(ext)
        p = self.out / split / folder / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _known(self, x, stats) -> str | None:
        lab = str(x.get("label", ""))
        if lab not in self.cid:
            stats["unknown_labels"][lab] += 1
            return None
        return lab

    def _voc(self, split, dst_img, shapes, W, H, tags, stats):
        a = ET.Element("annotation")
        ET.SubElement(a, "folder").text = "images"
        ET.SubElement(a, "filename").text = dst_img.name
        ET.SubElement(a, "path").text = str(dst_img)
        ET.SubElement(ET.SubElement(a, "source"), "database").text = "Unknown"
        sz = ET.SubElement(a, "size")
        for k, v in (("width", W), ("height", H), ("depth", 3)):
            ET.SubElement(sz, k).text = str(int(v))
        ET.SubElement(a, "segmented").text = "0"
        for x in shapes:
            lab = self._known(x, stats)
            if lab is None:
                continue
            poly = _area_poly(x) if (x.get("shape_type") or "polygon") in AREA else []
            if len(poly) < 2:
                stats.setdefault("dropped", Counter())["no area (points, lines)"] += 1
                continue
            x1, y1, x2, y2 = _bbox(poly)
            x1, x2, y1, y2 = _clipf(x1, W), _clipf(x2, W), _clipf(y1, H), _clipf(y2, H)
            if x2 - x1 < 1 or y2 - y1 < 1:
                stats.setdefault("dropped", Counter())["smaller than a pixel"] += 1
                continue
            o = ET.SubElement(a, "object")
            ET.SubElement(o, "name").text = lab
            ET.SubElement(o, "pose").text = "Unspecified"
            ET.SubElement(o, "truncated").text = str(int(_flag(x, "truncated")))
            ET.SubElement(o, "occluded").text = str(int(_flag(x, "occluded")))
            ET.SubElement(o, "difficult").text = str(int(_flag(x, "difficult")))
            bb = ET.SubElement(o, "bndbox")
            for k, v in (("xmin", x1), ("ymin", y1), ("xmax", x2), ("ymax", y2)):
                ET.SubElement(bb, k).text = str(int(round(v)))
            stats["objects"][lab] += 1
        ET.indent(a)
        p = self._label_file(split, dst_img, "Annotations", ".xml")
        atomic_write_text(p, ET.tostring(a, encoding="unicode") + "\n")
        self.sets[split].append(p.relative_to(self.out / split / "Annotations").with_suffix("").as_posix())

    def _cvat(self, split, dst_img, shapes, W, H, tags, stats):
        im = ET.SubElement(self.cvat, "image", id=str(self.n_img), name=dst_img.relative_to(self.out).as_posix(),
                           subset=split or "default", width=str(int(W)), height=str(int(H)))
        self.n_img += 1
        f2 = lambda v: f"{v:.2f}"                                    # noqa: E731
        for t in tags:
            ET.SubElement(im, "tag", label=str(t), source="manual")
        for x in shapes:
            lab = self._known(x, stats)
            if lab is None:
                continue
            st, pts = x.get("shape_type") or "polygon", _pts(x)
            if not pts:
                continue
            common = {"label": lab, "source": "manual", "occluded": str(int(_flag(x, "occluded"))), "z_order": "0"}
            if x.get("group_id") is not None:
                common["group_id"] = str(x.get("group_id"))
            ptxt = ";".join(f"{f2(a)},{f2(b)}" for a, b in pts)
            if st == "rectangle" and len(pts) >= 2:
                x1, y1, x2, y2 = _bbox(pts[:2])
                el = ET.SubElement(im, "box", xtl=f2(x1), ytl=f2(y1), xbr=f2(x2), ybr=f2(y2), **common)
            elif st == "circle" and len(pts) >= 2:
                r = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
                el = ET.SubElement(im, "ellipse", cx=f2(pts[0][0]), cy=f2(pts[0][1]), rx=f2(r), ry=f2(r), **common)
            elif st in ("polygon", "mask", "oriented rectangle") and len(pts) >= 3:
                el = ET.SubElement(im, "polygon", points=ptxt, **common)
            elif st in ("linestrip", "line") and len(pts) >= 2:
                el = ET.SubElement(im, "polyline", points=ptxt, **common)
            elif st in ("point", "points"):
                el = ET.SubElement(im, "points", points=ptxt, **common)
            else:
                stats.setdefault("dropped", Counter())[f"{st} with {len(pts)} point(s)"] += 1
                continue
            for k, v in (x.get("flags") or {}).items():
                if k != "occluded":
                    ET.SubElement(el, "attribute", name=str(k)).text = str(v).lower() if isinstance(v, bool) else str(v)
            stats["objects"][lab] += 1

    def _obb_lines(self, shapes, W, H, stats, norm: bool):
        lines = []
        for x in shapes:
            lab = self._known(x, stats)
            if lab is None:
                continue
            q = quad(x)
            if q is None:
                stats.setdefault("dropped", Counter())["no area (points, lines)"] += 1
                continue
            q = [[_clipf(a, W), _clipf(b, H)] for a, b in q]
            if norm:
                lines.append(" ".join([str(self.cid[lab])] + [f"{v:.6f}" for a, b in q for v in (a / W, b / H)]))
            else:
                name = re.sub(r"\s+", "-", lab)
                lines.append(" ".join(f"{v:.1f}" for p in q for v in p) + f" {name} {int(_flag(x, 'difficult'))}")
            stats["objects"][lab] += 1
        return lines

    def _yolo_obb(self, split, dst_img, shapes, W, H, tags, stats):
        lines = self._obb_lines(shapes, W, H, stats, True)
        atomic_write_text(self._label_file(split, dst_img, "labels", ".txt"), "\n".join(lines) + ("\n" if lines else ""))

    def _dota(self, split, dst_img, shapes, W, H, tags, stats):
        lines = self._obb_lines(shapes, W, H, stats, False)
        atomic_write_text(self._label_file(split, dst_img, "labelTxt", ".txt"), "\n".join(lines) + ("\n" if lines else ""))

    def _yolo_pose(self, split, dst_img, shapes, W, H, tags, stats):
        boxes, points = [], defaultdict(list)
        for x in shapes:
            st = x.get("shape_type") or "polygon"
            if st == "point":
                points[x.get("group_id")].append(x)
            elif st in AREA:
                boxes.append(x)
        lines = []
        used = set()
        for b in boxes:
            lab = self._known(b, stats)
            if lab is None:
                continue
            poly = _area_poly(b)
            if len(poly) < 2:
                continue
            x1, y1, x2, y2 = _bbox(poly)
            x1, x2, y1, y2 = _clipf(x1, W), _clipf(x2, W), _clipf(y1, H), _clipf(y2, H)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            g = b.get("group_id")
            mine = points.get(g, []) if g is not None else (points.get(None, []) if len(boxes) == 1 else [])
            kp = [[0.0, 0.0, 0]] * len(self.kpts)
            for p in mine:
                i = self.kid.get(str(p.get("label", "")))
                pts = _pts(p)
                if i is None or not pts:
                    continue
                vis = 1 if _flag(p, "occluded", "hidden") else 2
                kp = kp[:i] + [[_clipf(pts[0][0], W) / W, _clipf(pts[0][1], H) / H, vis]] + kp[i + 1:]
                used.add(id(p))
            lines.append(" ".join([str(self.cid[lab]), f"{(x1 + x2) / 2 / W:.6f}", f"{(y1 + y2) / 2 / H:.6f}",
                                   f"{(x2 - x1) / W:.6f}", f"{(y2 - y1) / H:.6f}"] +
                                  [f"{a:.6f} {b:.6f} {v}" for a, b, v in kp]))
            stats["objects"][lab] += 1
        left = sum(1 for ps in points.values() for p in ps if id(p) not in used)
        if left:
            stats.setdefault("dropped", Counter())["points not tied to a box (give them the box's group id)"] += left
        atomic_write_text(self._label_file(split, dst_img, "labels", ".txt"), "\n".join(lines) + ("\n" if lines else ""))

    # -------------------------------------------------------------------------------------------- the end
    def finish(self, stats: dict, source: str = ""):
        out = self.out
        out.mkdir(parents=True, exist_ok=True)
        splits = [sp for sp in ("train", "val", "test") if stats["images"].get(sp)]
        one = bool(stats["images"].get(""))               # one set, no split folders
        if self.fmt == "voc":
            for sp, stems in self.sets.items():
                atomic_write_text(out / sp / "ImageSets" / "Main" / f"{sp or 'trainval'}.txt", "\n".join(stems) + "\n")
            atomic_write_text(out / "labels.txt", "\n".join(self.names) + "\n")
        elif self.fmt == "cvat":
            task = ET.SubElement(self._cvat_meta, "task")
            ET.SubElement(task, "name").text = source or out.name
            ET.SubElement(task, "size").text = str(self.n_img)
            ET.SubElement(task, "mode").text = "annotation"
            labs = ET.SubElement(task, "labels")
            for c in self.names:
                lb = ET.SubElement(labs, "label")
                ET.SubElement(lb, "name").text = c
                ET.SubElement(lb, "type").text = "any"
            ET.SubElement(task, "subsets").text = "\n".join(splits + (["default"] if one else []))
            ET.indent(self.cvat)
            atomic_write_text(out / "annotations.xml", '<?xml version="1.0" encoding="utf-8"?>\n' +
                              ET.tostring(self.cvat, encoding="unicode") + "\n")
        elif self.fmt == "dota":
            atomic_write_text(out / "classes.txt", "\n".join(re.sub(r"\s+", "-", c) for c in self.names) + "\n")
        elif self.fmt in ("yolo_obb", "yolo_pose"):
            from .yolo_convert import _yaml_str
            q = str(out.resolve()).replace("\\", "/")
            lines = [f"# YOLO {'oriented boxes' if self.fmt == 'yolo_obb' else 'pose'} dataset by AIO QC"
                     + (f", from {source}" if source else ""), f"path: {_yaml_str(q)}"]
            for sp in splits:
                lines.append(f"{sp}: {sp}/images")
            if one:
                lines += ["train: images", "val: images      # no separate validation set (make one with a new split)"]
            if "val" not in splits and "train" in splits:
                lines.append("val: train/images      # no validation split in the source")
            if self.fmt == "yolo_pose":
                lines += ["", f"kpt_shape: [{len(self.kpts)}, 3]      # x, y, visible (0 missing · 1 hidden · 2 seen)"]
                fi = flip_idx(self.kpts)
                if fi:
                    lines.append(f"flip_idx: {fi}")
                lines += ["# keypoints, in order: " + ", ".join(self.kpts)]
            lines += ["", f"nc: {len(self.names)}", "names:"] + [f"  {i}: {_yaml_str(c)}" for i, c in enumerate(self.names)]
            atomic_write_text(out / "dataset.yaml", "\n".join(lines) + "\n")
            atomic_write_text(out / "classes.txt", "\n".join(self.names) + "\n")
