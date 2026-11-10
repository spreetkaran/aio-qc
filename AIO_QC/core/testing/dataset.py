# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Ground truth for evaluation: YOLO and LabelMe datasets, auto-detected.

  YOLO     images/…/x.jpg  +  labels/…/x.txt   (Ultralytics layout: the last 'images' path part -> 'labels')
           or x.txt next to x.jpg; class names from data.yaml / dataset.yaml / classes.txt (searched upwards);
           segmentation polygons are reduced to their bounding box; a data.yaml can be opened directly
           (its val / test / train split is resolved).
  LabelMe  x.json next to the image (rectangle / polygon / circle -> box; points and lines ignored).

Image sizes come from the file header (no decoding) and are EXIF-orientation corrected, exactly as YOLO
does - normalised YOLO boxes refer to the rotated image. Labels are read with a thread pool, so tens of
thousands of images load in seconds.

An image without a label file is "unlabelled" (excluded from accuracy metrics) - different from an image
with an empty label file (labelled, no objects: every detection on it is a false detection).
"""
import hashlib
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from AIO_QC.tools.frame_extractor.core.common import IMG_EXTS
from AIO_QC.tools.frame_extractor.core.imageio import imread
SKIP_DIRS = {"_rejected", ".fx_resume", "__pycache__", ".git", "cache", "reports", "negatives_blurry"}


@dataclass
class ImageItem:
    path: str
    w: int = 0
    h: int = 0
    boxes: np.ndarray = None            # (n, 4) xyxy pixels; None = unlabelled
    cls: np.ndarray = None              # (n,) dataset class ids
    label_path: str = ""
    reviewed: bool = True               # False: pre-annotations nobody has checked yet
    meta: dict = field(default_factory=dict)   # video, frame, time, source ...

    @property
    def labelled(self):
        return self.boxes is not None


@dataclass
class Dataset:
    items: list
    names: dict
    fmt: str
    root: str = ""
    warnings: list = field(default_factory=list)

    @property
    def n_labelled(self):
        return sum(1 for it in self.items if it.labelled)

    def gts(self, only_reviewed=True):
        """(boxes, cls) per labelled item, plus the items they belong to."""
        items = [it for it in self.items if it.labelled and (it.reviewed or not only_reviewed)]
        return items, [(it.boxes, it.cls) for it in items]


# ------------------------------------------------------------------ image size from the header
def _exif_orientation(data):
    """Orientation tag from an APP1 Exif payload (bytes after 'Exif\\0\\0')."""
    try:
        bo = "<" if data[:2] == b"II" else ">"
        off = struct.unpack(bo + "I", data[4:8])[0]
        n = struct.unpack(bo + "H", data[off:off + 2])[0]
        for i in range(n):
            e = off + 2 + 12 * i
            tag, typ = struct.unpack(bo + "HH", data[e:e + 4])
            if tag == 0x0112:
                return struct.unpack(bo + "H", data[e + 8:e + 10])[0]
    except (struct.error, IndexError):
        pass
    return 1


def image_size(path):
    """(w, h) after EXIF rotation, read from the header; (0, 0) if unknown."""
    try:
        with open(path, "rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return struct.unpack(">II", head[16:24])
            if head[:2] == b"BM":
                w, h = struct.unpack("<ii", head[18:26])
                return abs(w), abs(h)
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                f.seek(12)
                chunk = f.read(30)
                kind = chunk[:4]
                if kind == b"VP8 ":
                    w, h = struct.unpack("<HH", chunk[14:18])
                    return w & 0x3FFF, h & 0x3FFF
                if kind == b"VP8L":
                    b = chunk[9:13]
                    return 1 + (((b[1] & 0x3F) << 8) | b[0]), 1 + (((b[3] & 0xF) << 10) | (b[2] << 2) | ((b[1] & 0xC0) >> 6))
                if kind == b"VP8X":
                    return 1 + int.from_bytes(chunk[12:15], "little"), 1 + int.from_bytes(chunk[15:18], "little")
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                orient, size = 1, None
                while True:
                    m = f.read(2)
                    if len(m) < 2 or m[0] != 0xFF:
                        break
                    while m[1] == 0xFF:                        # padding
                        m = m[:1] + f.read(1)
                    marker = m[1]
                    if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                        continue
                    ln = struct.unpack(">H", f.read(2))[0]
                    if marker == 0xE1:
                        seg = f.read(ln - 2)
                        if seg[:6] == b"Exif\x00\x00":
                            orient = _exif_orientation(seg[6:])
                        continue
                    if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        seg = f.read(ln - 2)
                        h, w = struct.unpack(">HH", seg[1:5])
                        size = (w, h)
                        break
                    if marker == 0xDA:
                        break
                    f.seek(ln - 2, 1)
                if size:
                    return (size[1], size[0]) if orient in (5, 6, 7, 8) else size
    except (OSError, struct.error):
        pass
    try:
        img = imread(path)        # applies EXIF orientation, like YOLO
        if img is not None:
            return img.shape[1], img.shape[0]
    except Exception:
        pass
    return 0, 0


# ------------------------------------------------------------------ names
def _mini_yaml(text):
    """Tiny YAML subset (top-level 'key: value', 'names:' as inline list / dict or block list / mapping) -
    used only when PyYAML is not installed."""
    import ast
    out, lines = {}, text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].split("#", 1)[0].rstrip()
        i += 1
        if not line or line[0] in " \t-" or ":" not in line:
            continue
        key, rest = line.split(":", 1)
        rest = rest.strip()
        if rest:
            try:
                out[key.strip()] = ast.literal_eval(rest)
            except (ValueError, SyntaxError):
                if rest.startswith("[") and rest.endswith("]"):          # [a, b] with bare words
                    out[key.strip()] = [x.strip().strip("'\"") for x in rest[1:-1].split(",") if x.strip()]
                else:
                    out[key.strip()] = rest.strip("'\"")
            continue
        block = {}
        while i < len(lines) and (lines[i].startswith((" ", "\t", "-")) or not lines[i].strip()):
            sub = lines[i].split("#", 1)[0].strip()
            i += 1
            if sub.startswith("- "):
                block[len(block)] = sub[2:].strip().strip("'\"")
            elif ":" in sub:
                k, v = sub.split(":", 1)
                try:
                    block[int(k)] = v.strip().strip("'\"")
                except ValueError:
                    block[k.strip()] = v.strip().strip("'\"")
        out[key.strip()] = block
    return out


def load_yaml(path):
    text = open(path, encoding="utf-8").read()
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        return _mini_yaml(text)


def read_names_yaml(path):
    """names from a YOLO data.yaml (dict or list form); None if absent."""
    try:
        d = load_yaml(path)
    except Exception:
        return None
    names = d.get("names") if isinstance(d, dict) else None
    if isinstance(names, dict):
        try:
            return {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError):
            return None
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return None


def find_names(start_dir, levels=3):
    """Search start_dir and its parents for data.yaml / dataset.yaml / *.yaml with names / classes.txt."""
    d = os.path.abspath(start_dir)
    for _ in range(levels + 1):
        for fn in ("data.yaml", "dataset.yaml", "data.yml"):
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                n = read_names_yaml(p)
                if n:
                    return n, p
        for fn in ("classes.txt", os.path.join("labels", "classes.txt")):
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                names = [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
                if names:
                    return {i: n for i, n in enumerate(names)}, p
        try:
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith((".yaml", ".yml")):
                    n = read_names_yaml(os.path.join(d, fn))
                    if n:
                        return n, os.path.join(d, fn)
        except OSError:
            pass
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None, ""


# ------------------------------------------------------------------ discovery
def list_images(root, recursive=True):
    out = []
    if os.path.isfile(root):
        return [root] if root.lower().endswith(IMG_EXTS) else []
    for r, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        out += [os.path.join(r, f) for f in sorted(files) if f.lower().endswith(IMG_EXTS)]
        if not recursive:
            break
    return out


def yolo_label_path(img_path):
    """Ultralytics img2label_paths: last /images/ -> /labels/, extension -> .txt; else next to the image."""
    sep = os.sep
    p = os.path.abspath(img_path)
    sa, sb = f"{sep}images{sep}", f"{sep}labels{sep}"
    if sa in p:
        i = p.rfind(sa)
        cand = p[:i] + sb + p[i + len(sa):]
        cand = os.path.splitext(cand)[0] + ".txt"
        if os.path.isfile(cand):
            return cand
    side = os.path.splitext(p)[0] + ".txt"
    return side if os.path.isfile(side) else ""


def labelme_path(img_path):
    p = os.path.splitext(img_path)[0] + ".json"
    return p if os.path.isfile(p) else ""


def detect_format(images):
    """'yolo' | 'labelme' | 'none' from a sample of images."""
    sample = images[:: max(1, len(images) // 200)][:200]
    y = sum(1 for p in sample if yolo_label_path(p))
    lm = sum(1 for p in sample if labelme_path(p))
    if not y and not lm:
        return "none"
    return "yolo" if y >= lm else "labelme"


# ------------------------------------------------------------------ parsers
def parse_yolo(txt, w, h):
    """(boxes xyxy px, cls) from a YOLO txt (detection or segmentation lines)."""
    boxes, cls = [], []
    try:
        lines = open(txt, encoding="utf-8").read().split("\n")
    except OSError:
        return None, None
    for line in lines:
        v = line.split()
        if len(v) < 5:
            continue
        try:
            c = int(float(v[0]))
            nums = [float(x) for x in v[1:]]
        except ValueError:
            continue
        if len(nums) == 4 or len(nums) == 5:                     # box (+ optional conf)
            cx, cy, bw, bh = nums[:4]
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        elif len(nums) >= 6 and len(nums) % 2 == 0:              # polygon
            xs, ys = nums[0::2], nums[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        boxes.append([x1 * w, y1 * h, x2 * w, y2 * h])
        cls.append(c)
    b = np.array(boxes, np.float32).reshape(-1, 4)
    if len(b):
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b, np.array(cls, int)


def parse_labelme(path):
    """(boxes, names list, (w, h), flags) from a LabelMe JSON."""
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None, None, (0, 0), {}
    boxes, names = [], []
    for s in d.get("shapes", []):
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
        boxes.append(b)
        names.append(str(s.get("label", "")))
    return (np.array(boxes, np.float32).reshape(-1, 4), names,
            (int(d.get("imageWidth") or 0), int(d.get("imageHeight") or 0)), d.get("flags") or {})


def file_sha1(path):
    try:
        return hashlib.sha1(open(path, "rb").read()).hexdigest()
    except OSError:
        return ""


# ------------------------------------------------------------------ loading
def load_folder(root, fmt="auto", names=None, recursive=True, images=None, prelabel_hashes=None,
                progress=None, cancel=None, workers=8, label_paths=None):
    """Load a folder (or an explicit image list) with its labels.
    prelabel_hashes: {label path: sha1} of pre-annotations written by FrameExtractor - unchanged ones are
    marked unreviewed. label_paths: {image path: label path or ""} - used instead of looking the labels up
    (AIO QC's dataset, whose annotations may live in their own folder). Returns a Dataset."""
    if root and root.lower().endswith((".yaml", ".yml")) and images is None:
        return load_data_yaml(root, progress=progress, cancel=cancel)
    images = list(images) if images is not None else list_images(root, recursive)
    warnings = []
    fmt = detect_format(images) if fmt == "auto" else fmt
    if names is None and fmt == "yolo" and label_paths is None:
        lab_dirs = {os.path.dirname(yolo_label_path(p)) for p in images[:50] if yolo_label_path(p)}
        for d in sorted(lab_dirs) + [root or ""]:
            if d:
                names, src = find_names(d)
                if names:
                    break
        if not names:
            warnings.append("No class names found (data.yaml / classes.txt): models are matched by class id")
    lm_names = {}

    def one(p):
        if cancel is not None and cancel():
            return None
        w, h = image_size(p)
        it = ImageItem(path=p, w=w, h=h)
        if fmt == "yolo":
            lp = label_paths.get(p, "") if label_paths is not None else yolo_label_path(p)
            if lp and w and h:
                it.boxes, it.cls = parse_yolo(lp, w, h)
                it.label_path = lp
        elif fmt == "labelme":
            lp = label_paths.get(p, "") if label_paths is not None else labelme_path(p)
            if lp:
                b, nm, (jw, jh), flags = parse_labelme(lp)
                if b is not None:
                    it.boxes, it.label_path = b, lp
                    it.meta["label_names"] = nm
                    if (not w or not h) and jw and jh:
                        it.w, it.h = jw, jh
        if it.label_path and prelabel_hashes and prelabel_hashes.get(os.path.abspath(it.label_path)):
            it.reviewed = file_sha1(it.label_path) != prelabel_hashes[os.path.abspath(it.label_path)]
        return it

    items = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, it in enumerate(ex.map(one, images)):
            if it is None:
                break
            items.append(it)
            if progress and (i % 200 == 0 or i == len(images) - 1):
                progress(i + 1, len(images))
    if fmt == "labelme":
        # class ids: given names first (e.g. the test set's), then new names alphabetically
        names = dict(names or {})
        known = {v: k for k, v in names.items()}
        for nm in sorted({n for it in items for n in it.meta.get("label_names", [])} - set(known)):
            k = max(names) + 1 if names else 0
            names[k] = nm
            known[nm] = k
        for it in items:
            if it.boxes is not None:
                it.cls = np.array([known[n] for n in it.meta.pop("label_names", [])], int)
    bad = sum(1 for it in items if not it.w)
    if bad:
        warnings.append(f"{bad} image(s) could not be read")
    return Dataset(items=items, names=dict(names or {}), fmt=fmt, root=root, warnings=warnings)


def load_data_yaml(path, split="val", progress=None, cancel=None):
    """A YOLO data.yaml: images of one split (val, else test, else train)."""
    d = load_yaml(path)
    base = d.get("path") or os.path.dirname(os.path.abspath(path))
    if not os.path.isabs(base):
        base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), base))
    entry = d.get(split) or d.get("val") or d.get("test") or d.get("train")
    entries = entry if isinstance(entry, list) else [entry] if entry else []
    images = []
    for e in entries:
        p = e if os.path.isabs(e) else os.path.join(base, e)
        if os.path.isdir(p):
            images += list_images(p)
        elif os.path.isfile(p) and p.lower().endswith(".txt"):                 # list file
            for line in open(p, encoding="utf-8"):
                q = line.strip()
                if q:
                    images.append(q if os.path.isabs(q) else os.path.normpath(os.path.join(base, q)))
    names = read_names_yaml(path)
    ds = load_folder(base, "yolo", names=names, images=images, progress=progress, cancel=cancel)
    ds.root = path
    return ds


def write_yolo_labels(path, w, h, boxes, cls):
    with open(path, "w", encoding="utf-8") as f:
        for (x1, y1, x2, y2), c in zip(boxes, cls):
            f.write(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
