# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Add data to an existing dataset, matched by IMAGE CONTENT (not file names) — Qt-free, shared by Data Prep and the
command line (aio-qc add-data / add-classes).

The dataset: LabelMe (image + JSON side by side, any nesting) or YOLO (images/ + labels/, flat or split: train/images
or images/train …, detection or segmentation). The new data: one or more folders of LabelMe or YOLO pairs (each
detected on its own; a YOLO folder's class list is read by name). Any mix converts: LabelMe → YOLO (the same
conversion as Convert to YOLO: groups, circles, lines …), YOLO → LabelMe (rectangles / polygons), boxes ↔ polygons
for a detection / segmentation dataset.

    o = Options(target, [new folder, …], add_new=True, on_match="skip")      Add new data
    o = Options(target, [source], add_new=False, on_match="merge")          Add classes by image
    P = plan(o, progress, cancel)        nothing changes; P.rows / P.details say what would happen
    st = run(P, progress, cancel)        logged (RunLog): Preparation ▸ Undo removes / restores exactly that

Matching: byte-identical images are the same image, whatever their names and folders. Only images whose byte size
also occurs on the other side are read (a 3-sample fingerprint, then the full hash) — a dataset of millions is
listed, not read. Images repeated inside the new data count once.

  · new images (add_new): copied (or hard-linked) into the dataset — the split you choose (all to train, a balanced
    split like the dataset's own shares or yours — frames of one video stay together, rare classes in train — or the
    new data's own train / val / test); names that clash with the dataset or each other are renamed with the naming
    engine (core.naming: a preset or your pattern, every name verified); the annotation converted to the dataset's
    format, class ids by name; classes the dataset does not have yet: added to its class list, dropped, or the image
    left out; images without an annotation: left out or added as background; every image decoded first (optional)
  · images the dataset already has (on_match): skip — the annotations compared and the differing ones listed;
    replace — the dataset's annotation becomes the new one; merge — objects of chosen classes (by default the classes
    the dataset does not have) are added to the dataset's annotation of that image (every copy of it), the class
    list extended. A class both have in one image: keep the dataset's, replace them, or add only the objects that
    do not overlap (IoU). Dataset images the source does not have are listed — they would lack the new classes
  · a report of every decision (…_report.csv beside the run log)
"""
from __future__ import annotations

import csv
import math
import os
import re
import shutil
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import common as C

SPLITS = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
WORKERS = max(4, min(32, (os.cpu_count() or 4) * 2))
LABELME_VERSION = "5.5.0"


@dataclass
class Options:
    target: str
    sources: list
    add_new: bool = True                  # images the dataset does not have are added
    on_match: str = "skip"                # the dataset has the image: skip | replace | merge
    merge_classes: list | None = None     # merge: these classes (None: those the dataset does not have)
    conflict: str = "keep"                # merge, a class both have in one image: keep | replace | union
    iou: float = 0.6                      # union: an object counts as already there from this overlap
    split_mode: str = "train"             # a dataset with splits: train | balanced | source
    val: float | None = None              # balanced: None = the dataset's own shares
    test: float | None = None
    new_classes: str = "add"              # YOLO dataset, classes it lacks: add | drop | skip (the image)
    rename: dict = field(default_factory=dict)      # label in the new data -> class in the dataset
    unlabelled: str = "skip"              # new images without an annotation: skip | add (background)
    placement: str = "mirror"             # LabelMe dataset: mirror (the new data's sub-folders) | flat
    subfolder: str = ""                   # LabelMe dataset: under this folder (and the split)
    link: str = "copy"                    # copy | hardlink
    check_images: bool = True             # every image to add fully decoded first
    naming: dict = field(default_factory=dict)      # core.naming.Scheme as a dict (default: number suffix)
    seed: int = 42
    workers: int = WORKERS


# ====================================================================================================== scanning
class Img:
    """One image of a dataset / of the new data."""
    __slots__ = ("path", "size", "split", "ann", "rel", "dir_rel", "labels", "nobj", "bad", "w", "h", "ds")

    def __init__(self, path, size, split, ann, rel, dir_rel, ds):
        self.path, self.size, self.split, self.ann, self.rel, self.dir_rel, self.ds = \
            path, size, split, ann, rel, dir_rel, ds
        self.labels, self.nobj, self.bad, self.w, self.h = (), 0, "", 0, 0


@dataclass
class Dataset:
    root: Path
    fmt: str = "labelme"                  # labelme | yolo
    items: list = field(default_factory=list)
    classes: list = field(default_factory=list)     # YOLO
    class_files: list = field(default_factory=list)
    task: str = "det"                     # YOLO: det | seg
    layout: str = "flat"                  # YOLO: flat | split_first (train/images) | images_first (images/train)
    split_dirs: dict = field(default_factory=dict)  # canonical split -> its folder name ("valid" …)
    image_dirs: dict = field(default_factory=dict)  # YOLO: canonical split ("" flat) -> images folder
    orphans: list = field(default_factory=list)     # annotation files without their image
    ambiguous: list = field(default_factory=list)   # (json, [images]) LabelMe
    names: dict = field(default_factory=lambda: defaultdict(set))   # naming: dest key -> name keys taken
    label_dirs: list = field(default_factory=list)

    @property
    def splits(self) -> list:
        return sorted({it.split for it in self.items if it.split}, key=["train", "val", "test"].index)


def _key(name) -> str:
    return " ".join(str(name).split()).casefold()


def _walk(root: Path, skip=()):
    """[(dirpath, rel parts, [(name, size)])] — images, .json and .txt files; hidden folders, AIO QC's log and
    issues folders and `skip` left out."""
    from AIO_QC.core.issues_folder import is_issues_dir
    skip_n = [C.norm(s) for s in skip if s]
    out, stack = [], [(str(root), ())]
    while stack:
        d, parts = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        files, subs = [], []
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if e.name.startswith(".") or e.name == C.LOG_DIR or any(C.inside(e.path, s) for s in skip_n):
                            continue
                        if not is_issues_dir(e.path):
                            subs.append((e.path, parts + (e.name,)))
                        continue
                    if e.name.startswith("."):           # hidden: AIO QC's own records, OS files
                        continue
                    x = os.path.splitext(e.name)[1].lower()
                    if x in C.IMAGE_EXTS or x in (".json", ".txt"):
                        files.append((e.name, e.stat().st_size if x in C.IMAGE_EXTS else 0))
                except OSError:
                    continue
        out.append((d, parts, files))
        stack.extend(sorted(subs, reverse=True))
    return out


def _is_yolo(walked) -> bool:
    has_lab = any(any(p.lower() == "labels" for p in parts) and any(n.lower().endswith(".txt") and
                                                                    n.lower() != "classes.txt" for n, _s in files)
                  for _d, parts, files in walked)
    has_img = any(any(p.lower() == "images" for p in parts) and any(os.path.splitext(n)[1].lower() in C.IMAGE_EXTS
                                                                    for n, _s in files)
                  for _d, parts, files in walked)
    return has_lab and has_img


def scan(root, role="target", skip=(), cancel=None) -> Dataset:
    """Every image of a dataset with its annotation file (or None), split, size."""
    root = Path(root).resolve()
    walked = _walk(root, skip)
    D = Dataset(root)
    D.fmt = "yolo" if _is_yolo(walked) else "labelme"
    if D.fmt == "yolo" or (role == "target" and _has_class_list(root) and not any(
            n.lower().endswith(".json") for _d, _p, fs in walked for n, _s in fs)):
        D.fmt = "yolo"
        _scan_yolo(D, walked, cancel)
    else:
        _scan_labelme(D, walked, cancel)
    return D


def _has_class_list(root: Path) -> bool:
    return any((root / n).is_file() for n in ("classes.txt", "data.yaml", "dataset.yaml"))


def _split_of(parts) -> str:
    return next((SPLITS[p.lower()] for p in parts if p.lower() in SPLITS), "")


def _scan_yolo(D: Dataset, walked, cancel):
    from AIO_QC.core.class_map import find_class_map
    labels = set()
    for d, parts, files in walked:
        if any(p.lower() == "labels" for p in parts):
            D.label_dirs.append(Path(d))
            for n, _s in files:
                if n.lower().endswith(".txt") and n.lower() != "classes.txt":
                    labels.add(os.path.normcase(os.path.join(d, n)))
    used = set()
    dirs = defaultdict(Counter)
    first_split_at, first_images_at = None, None
    for d, parts, files in walked:
        if cancel is not None and cancel.is_set():
            raise C.Cancelled()
        low = [p.lower() for p in parts]
        if "images" not in low:
            continue
        i = len(low) - 1 - low[::-1].index("images")
        split = _split_of(parts)
        si = next((k for k, p in enumerate(low) if p in SPLITS), None)
        if si is not None:
            D.split_dirs.setdefault(SPLITS[low[si]], parts[si])
            first_split_at, first_images_at = si, i
        lab_dir = Path(D.root, *parts[:i], "labels", *parts[i + 1:])
        img_dir = Path(D.root, *parts[:i + 1], *(parts[i + 1:i + 2] if si == i + 1 else ()))
        for n, size in files:
            if os.path.splitext(n)[1].lower() not in C.IMAGE_EXTS:
                continue
            lab = lab_dir / (os.path.splitext(n)[0] + ".txt")
            nk = os.path.normcase(str(lab))
            has = nk in labels
            if has:
                used.add(nk)
            p = Path(d, n)
            rel = "/".join(parts + (n,))
            D.items.append(Img(p, size, split, lab if has else None, rel, "/".join(parts), D))
            dirs[split][img_dir] += 1
            D.names[split].add(_name_key(os.path.splitext(n)[0]))
    D.orphans = sorted(labels - used)[:5000]
    D.image_dirs = {sp: c.most_common(1)[0][0] for sp, c in dirs.items()}
    if first_split_at is not None:
        D.layout = "split_first" if first_split_at < first_images_at else "images_first"
    cm = find_class_map(D.root, tuple(D.label_dirs))
    D.classes = list(cm.names)
    D.class_files = _class_files(D.root, D.label_dirs)
    D.task = _task_of(D)


def _name_key(stem: str) -> str:
    from AIO_QC.core.naming import name_key
    return name_key(stem)


def _class_files(root: Path, label_dirs) -> list:
    """Every class list the dataset has (all are kept in step when classes are added)."""
    out = []
    for p in sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml")):
        try:
            if re.search(r"(?m)^names\s*:", p.read_text(encoding="utf-8", errors="replace")):
                out.append(p)
        except OSError:
            pass
    for d in [root, root / "labels", *label_dirs]:
        for n in ("classes.txt", "obj.names"):
            p = d / n
            if p.is_file() and p not in out:
                out.append(p)
    return out


def _task_of(D: Dataset) -> str:
    for it in [x for x in D.items if x.ann is not None][:300]:
        try:
            for line in it.ann.read_text(encoding="utf-8", errors="replace").splitlines()[:20]:
                if len(line.split()) > 5:
                    return "seg"
        except OSError:
            pass
    return "det"


def _image_path_of(json_path) -> str:
    try:
        return C.ip_basename(str(C.read_json(json_path).get("imagePath") or ""))
    except Exception:
        return ""


def _scan_labelme(D: Dataset, walked, cancel):
    for d, parts, files in walked:
        if cancel is not None and cancel.is_set():
            raise C.Cancelled()
        imgs, jsons = defaultdict(list), {}
        for n, size in files:
            st, x = os.path.splitext(n)
            if x.lower() in C.IMAGE_EXTS:
                imgs[st.lower()].append((n, size))
            elif x.lower() == ".json" and not n.startswith(("_", ".")):
                jsons[st.lower()] = n
        split = SPLITS.get(parts[0].lower(), "") if parts else ""
        if split:
            D.split_dirs.setdefault(split, parts[0])
        dir_rel = "/".join(parts)
        paired = {}
        for k, cands in imgs.items():
            jn = jsons.get(k)
            if jn is None:
                continue
            if len(cands) == 1:
                paired[cands[0][0]] = jn
            else:
                ip = _image_path_of(os.path.join(d, jn)).lower()
                hit = [c for c in cands if c[0].lower() == ip]
                if len(hit) == 1:
                    paired[hit[0][0]] = jn
                else:
                    D.ambiguous.append((os.path.join(d, jn), [os.path.join(d, c[0]) for c in cands]))
            jsons.pop(k, None)
        by_name = {n.lower(): n for cs in imgs.values() for n, _s in cs}
        for k, jn in list(jsons.items()):                    # x.json pointing at an image of another name
            ip = _image_path_of(os.path.join(d, jn)).lower()
            n = by_name.get(ip)
            if n and n not in paired:
                paired[n] = jn
                jsons.pop(k)
        D.orphans.extend(os.path.join(d, jn) for jn in jsons.values())
        for cands in imgs.values():
            for n, size in cands:
                jn = paired.get(n)
                D.items.append(Img(Path(d, n), size, split, Path(d, jn) if jn else None,
                                   "/".join(parts + (n,)), dir_rel, D))
                D.names[dir_rel.lower()].add(_name_key(os.path.splitext(n)[0]))
        for jn in paired.values():
            D.names[dir_rel.lower()].add(_name_key(os.path.splitext(jn)[0]))


# ====================================================================================================== annotations
class Obj:
    """One object: label, "box" / "poly" points in pixels, and its LabelMe shape when it came from one (kept as it
    is wherever the result is LabelMe)."""
    __slots__ = ("label", "kind", "pts", "shape")

    def __init__(self, label, kind, pts, shape=None):
        self.label, self.kind, self.pts, self.shape = label, kind, pts, shape


def image_size(it: Img, data=None) -> tuple[int, int]:
    if it.w and it.h:
        return it.w, it.h
    if isinstance(data, dict):
        try:
            w, h = int(data.get("imageWidth") or 0), int(data.get("imageHeight") or 0)
            if w > 0 and h > 0:
                it.w, it.h = w, h
                return w, h
        except (TypeError, ValueError):
            pass
    from AIO_QC.core import io as cio
    wh = cio.image_size(str(it.path))
    if not wh:
        raise ValueError(f"cannot read the size of {it.path}")
    it.w, it.h = wh
    return wh


def read(it: Img, names=None):
    """-> ("labelme", JSON dict) | ("yolo", [(class id, [values])]) | None (no annotation file)."""
    if it.ann is None:
        return None
    if it.ds.fmt == "labelme":
        data = C.read_json(it.ann)
        if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
            raise ValueError("not a LabelMe file (no shapes list)")
        return "labelme", data
    rows = []
    for ln in it.ann.read_text(encoding="utf-8", errors="replace").splitlines():
        p = ln.split()
        if not p:
            continue
        try:
            cid = int(float(p[0]))
            vals = [float(v) for v in p[1:]]
        except ValueError:
            raise ValueError(f"not a YOLO line: {ln[:60]}") from None
        if len(vals) < 4 or (len(vals) > 4 and len(vals) % 2):
            raise ValueError(f"not a YOLO line: {ln[:60]}")
        rows.append((cid, vals))
    return "yolo", rows


def _class_name(names, cid) -> str:
    return names[cid] if 0 <= cid < len(names) else f"class_{cid}"


def objects(it: Img, ann, rename=None) -> list:
    """The annotation as Objs (labels renamed by `rename`, keys compared without case / spaces)."""
    if ann is None:
        return []
    ren = {_key(k): v for k, v in (rename or {}).items()}
    out = []
    if ann[0] == "labelme":
        for s in ann[1]["shapes"]:
            if not isinstance(s, dict):
                continue
            lab = str(s.get("label", ""))
            lab = ren.get(_key(lab), lab)
            pts = s.get("points") or []
            kind = "box" if s.get("shape_type") == "rectangle" else "poly"
            if lab != s.get("label"):
                s = dict(s, label=lab)
            out.append(Obj(lab, kind, pts, s))
        return out
    W, H = image_size(it)
    names = it.ds.classes
    for cid, v in ann[1]:
        lab = _class_name(names, cid)
        lab = ren.get(_key(lab), lab)
        if len(v) == 4:
            cx, cy, w, h = v
            out.append(Obj(lab, "box", [[(cx - w / 2) * W, (cy - h / 2) * H], [(cx + w / 2) * W, (cy + h / 2) * H]]))
        else:
            out.append(Obj(lab, "poly", [[v[i] * W, v[i + 1] * H] for i in range(0, len(v), 2)]))
    return out


def _bbox(o: Obj):
    pts = o.pts
    try:
        if o.shape is not None and o.shape.get("shape_type") == "circle" and len(pts) >= 2:
            (cx, cy), (ex, ey) = pts[0], pts[1]
            r = math.hypot(ex - cx, ey - cy)
            return cx - r, cy - r, cx + r, cy + r
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError, IndexError):
        return None


def _iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def to_yolo(objs, W, H, ids: dict, task: str) -> tuple[list, Counter]:
    """YOLO lines for the objects whose label `ids` has ({key(label): id}); -> (lines, dropped Counter)."""
    from AIO_QC.core import yolo_convert as Y
    lines, dropped = [], Counter()
    shapes = []
    for o in objs:
        cid = ids.get(_key(o.label))
        if cid is None:
            dropped["class not in the dataset"] += 1
            continue
        if o.shape is not None:
            shapes.append(dict(o.shape, label=str(cid)))
            continue
        pts = o.pts
        if task == "det" or o.kind == "box":
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x1, x2 = max(0.0, min(xs)), min(float(W), max(xs))
            y1, y2 = max(0.0, min(ys)), min(float(H), max(ys))
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                dropped["no area"] += 1
                continue
            if task == "det":
                vals = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
            else:
                vals = (x1 / W, y1 / H, x2 / W, y1 / H, x2 / W, y2 / H, x1 / W, y2 / H)
        else:
            vals = tuple(min(max(v / d, 0.0), 1.0) for p in pts for v, d in ((p[0], W), (p[1], H)))
        lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in vals))
    if shapes:
        ids_by_str = {str(i): i for i in set(ids.values())}
        ln, _w, drop, _ex, _m = Y.objects(shapes, W, H, ids_by_str, task)
        lines += ln
        dropped.update(drop)
    return lines, dropped


def to_shapes(objs) -> list:
    out = []
    for o in objs:
        if o.shape is not None:
            out.append(o.shape)
        else:
            out.append({"label": o.label, "points": [[round(float(x), 2), round(float(y), 2)] for x, y in o.pts],
                        "group_id": None, "description": "",
                        "shape_type": "rectangle" if o.kind == "box" else "polygon", "flags": {}})
    return out


def _yolo_canon(lines) -> list:
    out = []
    for ln in lines:
        p = ln.split()
        try:
            out.append((int(float(p[0])),) + tuple(float(v) for v in p[1:]))
        except (ValueError, IndexError):
            out.append((ln,))
    return out


def _lm_canon(shapes) -> list:
    out = []
    for s in shapes:
        try:
            pts = tuple(float(v) for p in (s.get("points") or []) for v in p)
        except (TypeError, ValueError):
            pts = ()
        out.append((str(s.get("label", "")), str(s.get("shape_type") or "polygon")) + pts)
    return out


def _same_rows(a: list, b: list, tol: float, head: int) -> bool:
    """Two annotations hold the same objects: one-to-one, the first `head` fields equal, every coordinate within
    `tol` (a tolerance, not rounding: 0.4996 and 0.5004 are the same box; rounding to a grid would split them)."""
    if len(a) != len(b):
        return False
    left = list(b)
    for r in a:
        hit = next((i for i, q in enumerate(left) if len(q) == len(r) and q[:head] == r[:head]
                    and all(abs(x - y) <= tol for x, y in zip(r[head:], q[head:]))), None)
        if hit is None:
            return False
        left.pop(hit)
    return True


def merge(target_objs, source_objs, classes: set, conflict: str, iou: float):
    """Objects of `classes` (keys) from the source added to the target's -> (objects, Counter added, Counter replaced,
    Counter kept-as-they-were)."""
    src = [o for o in source_objs if _key(o.label) in classes]
    have = defaultdict(list)
    for o in target_objs:
        have[_key(o.label)].append(o)
    out = list(target_objs)
    added, replaced, kept = Counter(), Counter(), Counter()
    by_cls = defaultdict(list)
    for o in src:
        by_cls[_key(o.label)].append(o)
    for k, objs in by_cls.items():
        if not have[k]:
            out += objs
            added[objs[0].label] += len(objs)
        elif conflict == "replace":
            out = [o for o in out if _key(o.label) != k] + objs
            replaced[objs[0].label] += len(have[k])
            added[objs[0].label] += len(objs)
        elif conflict == "union":
            boxes = [_bbox(t) for t in have[k]]
            new = [o for o in objs if all(_iou(_bbox(o), b) < iou for b in boxes)]
            out += new
            if new:
                added[objs[0].label] += len(new)
            if len(new) < len(objs):
                kept[objs[0].label] += len(objs) - len(new)
        else:
            kept[objs[0].label] += len(objs)
    return out, added, replaced, kept


# ====================================================================================================== plan
@dataclass
class Plan:
    opts: Options
    target: Dataset = None
    sources: list = field(default_factory=list)
    adds: list = field(default_factory=list)          # dicts: item, split, dst_img, dst_ann, new stem
    updates: list = field(default_factory=list)       # dicts: target item, source item, change summary
    classes_after: list = field(default_factory=list)
    new_classes: list = field(default_factory=list)   # appended to a YOLO dataset's class list
    merge_keys: set = field(default_factory=set)
    report: list = field(default_factory=list)        # [status, source, dataset file, destination, note]
    stats: Counter = field(default_factory=Counter)
    details: dict = field(default_factory=lambda: defaultdict(list))
    rows: list = field(default_factory=list)

    @property
    def actions(self):
        return self.adds + self.updates

    def summary_rows(self):
        s, o = self.stats, self.opts
        rows = [("New data: images found", s["source_images"])]
        if s["dups_in_sources"]:
            rows.append(("    repeated in the new data (counted once)", s["dups_in_sources"]))
        rows.append(("Already in the dataset (same image)", s["matched"]))
        if o.on_match == "skip" and s["matched"]:
            rows.append(("    annotation differs (listed, not changed)", s["match_diff"]))
        if o.on_match == "replace":
            rows.append(("    annotation replaced by the new one", s["replaced"]))
        if o.on_match == "merge":
            rows += [("    dataset images that gain objects", s["merged_images"]),
                     ("    objects added", s["objects_added"])]
            if s["objects_replaced"]:
                rows.append(("    objects replaced (same class)", s["objects_replaced"]))
            if s["objects_kept"]:
                rows.append(("    objects not added (class already there)", s["objects_kept"]))
            rows.append(("Dataset images missing the added classes", s["target_unmatched"]))
        if o.add_new:
            rows.append(("New images → added", len(self.adds)))
            if s["renamed"]:
                rows.append(("    renamed (name taken by another image)", s["renamed"]))
            for sp in ("train", "val", "test", ""):
                if s[f"to_{sp}"] and self.target.splits:
                    rows.append((f"    → {sp or 'the dataset'}", s[f"to_{sp}"]))
            if s["background"]:
                rows.append(("    without objects (background)", s["background"]))
        else:
            rows.append(("Images the dataset doesn't have (not added)", s["not_in_target"]))
        if self.new_classes:
            rows.append(("Classes added to the class list", len(self.new_classes)))
        for k, label in (("no_annotation", "Left out: no annotation"),
                         ("unknown_skipped", "Left out: a class the dataset lacks"),
                         ("objects_dropped", "Objects dropped (class not in the dataset)"),
                         ("unreadable", "Left out: image unreadable"),
                         ("invalid", "Left out: annotation unreadable"),
                         ("orphans", "Annotation files without their image"),
                         ("ambiguous", "Left out: JSON matching several images")):
            if s[k]:
                rows.append((label, s[k]))
        return rows


def _say(progress, d, t, msg):
    if progress is not None:
        progress(d, t, msg)


def plan(opts: Options, progress=None, cancel=None) -> Plan:
    P = Plan(opts)
    tgt = Path(opts.target).resolve()
    srcs = [Path(s).resolve() for s in opts.sources if str(s).strip()]
    if not tgt.is_dir():
        raise ValueError(f"The dataset folder does not exist: {tgt}")
    if not srcs:
        raise ValueError("Choose the folder(s) with the new data.")
    for s in srcs:
        if not s.is_dir():
            raise ValueError(f"Not a folder: {s}")
        if s == tgt or C.inside(s, tgt):
            raise ValueError(f"{s} is inside the dataset — every image would match itself. Choose data outside it.")
        if C.inside(tgt, s):
            raise ValueError(f"The dataset is inside {s} — choose the new data's own folder.")
    _say(progress, 0, 0, "Listing the dataset")
    T = scan(tgt, "target", cancel=cancel)
    P.target = T
    for s in srcs:
        _say(progress, 0, 0, f"Listing {s.name}")
        P.sources.append(scan(s, "source", skip=[tgt], cancel=cancel))
    for D in P.sources:                            # a YOLO folder without a class list: the dataset's ids
        if D.fmt == "yolo" and not D.classes and T.fmt == "yolo" and T.classes:
            D.classes = list(T.classes)
            P.details["New data without a class list — the dataset's class ids assumed"].append(str(D.root))
    S = [it for D in P.sources for it in D.items]
    P.stats["source_images"] = len(S)
    P.stats["orphans"] = sum(len(D.orphans) for D in P.sources)
    P.stats["ambiguous"] = sum(len(D.ambiguous) for D in P.sources)
    for D in P.sources:
        for o in D.orphans[:100]:
            P.report.append(["orphan annotation", str(o), "", "", "no image of the same name"])
            P.details["Annotation files without their image"].append(str(o))
        for j, cands in D.ambiguous[:100]:
            P.report.append(["ambiguous", " | ".join(cands), "", "", f"{Path(j).name} matches several images"])
            P.details["JSONs matching several images (left out)"].append(j)
    if not S:
        raise ValueError("No images found in the new data.")

    # ---------------------------------------------------------------- read the new annotations (labels, validity)
    def parse(it):
        if cancel is not None and cancel.is_set():
            raise C.Cancelled()
        try:
            ann = read(it)
            objs = objects(it, ann, opts.rename) if ann is not None and ann[0] == "labelme" else None
            if ann is None:
                it.labels = ()
            elif objs is not None:
                it.labels = tuple(o.label for o in objs)
            else:
                ren = {_key(k): v for k, v in opts.rename.items()}
                it.labels = tuple(ren.get(_key(n), n) for n in (_class_name(it.ds.classes, c) for c, _v in ann[1]))
            it.nobj = len(it.labels)
        except Exception as exc:
            it.bad = f"{type(exc).__name__}: {exc}"[:200]
    _say(progress, 0, len(S), "Reading the new annotations")
    with ThreadPoolExecutor(opts.workers) as ex:
        list(ex.map(parse, S))
    for D in P.sources:
        if D.fmt == "yolo" and not D.classes:
            P.details["New data without a class list (class_0, class_1 … as names)"].append(str(D.root))

    # ---------------------------------------------------------------- the same images (byte-identical)
    _say(progress, 0, 0, "Finding the images the dataset already has")
    sizes = {it.size for it in S}
    T_c = [it for it in T.items if it.size in sizes]
    pool = [("s", i) for i in range(len(S))] + [("t", i) for i in range(len(T_c))]
    obj = {"s": S, "t": T_c}
    groups = C.exact_groups(pool, lambda k: obj[k[0]][k[1]].size, lambda k: str(obj[k[0]][k[1]].path), None,
                            opts.workers, cancel, lambda d, t: _say(progress, d, t, "Comparing same-size images"))
    match = {}                 # source index -> [target items]
    dup_of = {}                # source index -> source index kept
    for g in groups:
        ts = [T_c[i] for k, i in g if k == "t"]
        ss = sorted((i for k, i in g if k == "s"), key=lambda i: str(S[i].path).lower())   # first by path kept
        if ts:
            for i in ss:
                match[i] = ts
        else:
            for i in ss[1:]:
                dup_of[i] = ss[0]
    P.stats["matched"] = len(match)
    P.stats["dups_in_sources"] = len(dup_of)
    for i, j in list(dup_of.items())[:200]:
        P.details["The same image more than once in the new data (counted once)"].append(
            f"{S[i].path}  =  {S[j].path}")
    for i, j in dup_of.items():
        P.report.append(["duplicate in the new data", str(S[i].path), "", "", f"same image as {S[j].path}"])
    copies = Counter(len(v) for v in match.values() if len(v) > 1)
    if copies:
        P.details["Images the dataset has more than once (all copies get the change)"].append(
            ", ".join(f"{n:,} images × {k}" for k, n in sorted(copies.items())))
        leaks = sum(1 for v in match.values() if len({t.split for t in v}) > 1)
        if leaks:
            P.details["The same image in more than one split of the dataset (leakage)"].append(f"{leaks:,} images")

    # ---------------------------------------------------------------- classes
    T_keys = {_key(c): i for i, c in enumerate(T.classes)}
    new_cls = []                                   # appended (YOLO dataset)

    def need_class(label):
        k = _key(label)
        if k not in T_keys and k not in {_key(c) for c in new_cls}:
            new_cls.append(str(label).strip())

    # ---------------------------------------------------------------- matched images
    if opts.on_match == "merge":
        _plan_merge(P, S, match, T, T_keys, need_class, progress, cancel)
    else:
        _plan_matched(P, S, match, T, T_keys, new_cls, need_class, progress, cancel)

    # ---------------------------------------------------------------- new images
    new_idx = [i for i in range(len(S)) if i not in match and i not in dup_of]
    P.stats["not_in_target"] = len(new_idx)
    if opts.add_new:
        _plan_adds(P, S, new_idx, T, T_keys, new_cls, need_class, progress, cancel)
    P.new_classes = new_cls if T.fmt == "yolo" else []
    P.classes_after = T.classes + P.new_classes
    if P.new_classes:
        P.details["Classes added to the dataset's class list"] = [
            f"{len(T.classes) + i}: {c}" for i, c in enumerate(P.new_classes)]
    P.rows = P.summary_rows()
    return P


def _plan_matched(P, S, match, T, T_keys, new_cls, need_class, progress, cancel):
    """skip / replace: the annotations of every matched image compared (in parallel)."""
    o = P.opts
    todo = [(S[i], t) for i, ts in match.items() for t in ts]
    done = [0]
    lock = threading.Lock()

    def cmp(pair):
        if cancel is not None and cancel.is_set():
            raise C.Cancelled()
        s, t = pair
        with lock:
            done[0] += 1
            if done[0] % 200 == 0:
                _say(progress, done[0], len(todo), "Comparing the annotations of the images the dataset has")
        if s.bad:
            return None, f"annotation unreadable: {s.bad}"
        try:
            return _same(s, t, T, T_keys, o), ""
        except Exception as exc:
            return None, f"could not compare: {type(exc).__name__}: {exc}"
    with ThreadPoolExecutor(o.workers) as ex:
        res = list(ex.map(cmp, todo))
    for (s, t), (same, note) in zip(todo, res):
        note = note or ("same annotation" if same else "different annotation")
        if same is False:
            P.stats["match_diff"] += 1
            if len(P.details["Same image, DIFFERENT annotation"]) < 500:
                P.details["Same image, DIFFERENT annotation"].append(f"{s.path}  ↔  {t.path}")
            if o.on_match == "replace":
                unknown = sorted({lab for lab in s.labels if _key(lab) not in T_keys and
                                  _key(lab) not in {_key(c) for c in new_cls}}) if T.fmt == "yolo" else []
                if unknown and o.new_classes == "skip":
                    note = "not replaced: classes the dataset lacks (" + ", ".join(unknown) + ")"
                else:
                    if unknown and o.new_classes == "add":
                        for lab in unknown:
                            need_class(lab)
                    P.updates.append({"target": t, "source": s, "kind": "replace"})
                    P.stats["replaced"] += 1
                    note = "annotation replaced by the new one"
        P.report.append(["already in the dataset", str(s.path), str(t.path), "", note])


def _target_view(s, t, T, T_keys, o, rename=True):
    """(dataset annotation as Objs, new annotation as Objs) for one matched pair."""
    ta = read(t)
    if ta is not None and ta[0] == "labelme":
        image_size(t, ta[1])
    tobj = objects(t, ta)
    sa = read(s)
    if sa is not None and sa[0] == "labelme":
        image_size(s, sa[1])
    if s.w and not t.w:
        t.w, t.h = s.w, s.h
    if t.w and not s.w:
        s.w, s.h = t.w, t.h
    sobj = objects(s, sa, o.rename if rename else None)
    return ta, tobj, sobj


def _same(s, t, T, T_keys, o) -> bool:
    ta, tobj, sobj = _target_view(s, t, T, T_keys, o)
    if T.fmt == "yolo":
        W, H = image_size(t)
        ids = dict(T_keys)
        for ob in sobj:                                 # a class the dataset lacks makes it different
            ids.setdefault(_key(ob.label), len(ids) + 10_000)
        tol = 1.0 / max(W or 0, H or 0, 1000)                  # about one pixel (a thousandth at most)
        a, b = _yolo_canon(to_yolo(tobj, W, H, ids, T.task)[0]), _yolo_canon(to_yolo(sobj, W, H, ids, T.task)[0])
        if any(isinstance(r[0], str) for r in a + b):
            return sorted(map(str, a)) == sorted(map(str, b))
        return _same_rows(a, b, tol, 1)
    return _same_rows(_lm_canon(to_shapes(tobj)), _lm_canon(to_shapes(sobj)), 1.0, 2)


def _plan_merge(P, S, match, T, T_keys, need_class, progress, cancel):
    o = P.opts
    pairs = [(i, t) for i, ts in match.items() for t in ts if not S[i].bad]
    for i in [i for i in match if S[i].bad]:
        P.stats["invalid"] += 1
        P.report.append(["annotation unreadable", str(S[i].path), "", "", S[i].bad])
    src_labels = Counter(lab for i in match for lab in S[i].labels)
    if o.merge_classes:
        want = {_key(c) for c in o.merge_classes}
        missing = sorted({c for c in o.merge_classes if _key(c) not in {_key(x) for x in src_labels}})
        if missing:
            P.details["Chosen classes the matched new data does not have"].append(", ".join(missing))
    elif T.fmt == "yolo":
        want = {_key(c) for c in src_labels if _key(c) not in T_keys}
    else:                                           # LabelMe: classes no matched dataset image has
        have = set()
        for n, (i, t) in enumerate(pairs):
            if n % 500 == 0:
                _say(progress, n, len(pairs), "Reading the dataset's classes")
            try:
                have |= {_key(ob.label) for ob in objects(t, read(t))}
            except Exception:
                pass
        want = {_key(c) for c in src_labels if _key(c) not in have}
    P.merge_keys = want
    spell = {}
    for c, _n in src_labels.most_common():
        spell.setdefault(_key(c), c)
    P.details["Classes brought over"] = [f"{spell.get(k, k)} ({src_labels[spell.get(k, k)]:,} objects in the matched "
                                         "new data)" for k in sorted(want)] or ["(none — every class is there)"]
    for k in sorted(want):
        if T.fmt == "yolo" and k not in T_keys and k in spell:
            need_class(spell[k])
    gained, objs_added = set(), Counter()
    for n, (i, t) in enumerate(pairs):
        if cancel is not None and cancel.is_set():
            raise C.Cancelled()
        if n % 200 == 0:
            _say(progress, n, len(pairs), "Merging the annotations of the images the dataset has")
        s = S[i]
        try:
            _ta, tobj, sobj = _target_view(s, t, T, T_keys, o)
            _out, added, replaced, kept = merge(tobj, sobj, want, o.conflict, o.iou)
        except Exception as exc:
            P.report.append(["could not merge", str(s.path), str(t.path), "", f"{type(exc).__name__}: {exc}"])
            P.stats["invalid"] += 1
            continue
        P.stats["objects_replaced"] += sum(replaced.values())
        P.stats["objects_kept"] += sum(kept.values())
        if added or replaced:
            P.updates.append({"target": t, "source": s, "kind": "merge"})
            gained.add(id(t))
            objs_added.update(added)
            P.report.append(["classes added", str(s.path), str(t.path), "",
                             ", ".join(f"{k} +{v}" for k, v in added.items())
                             + ("; replaced " + ", ".join(f"{k} {v}" for k, v in replaced.items()) if replaced else "")])
        else:
            P.report.append(["already complete", str(s.path), str(t.path), "",
                             "nothing to add" + (" (that class was already there)" if kept else "")])
    P.stats["merged_images"] = len(gained)
    P.stats["objects_added"] = sum(objs_added.values())
    if objs_added:
        P.details["Objects added per class"] = [f"{k}: {v:,}" for k, v in objs_added.most_common()]
    matched_t = {id(t) for ts in match.values() for t in ts}
    un = [t for t in T.items if id(t) not in matched_t]
    P.stats["target_unmatched"] = len(un) if want else 0
    if want and un:
        P.details["Dataset images the source does not have — they will NOT have the added classes (a model learns "
                  "those objects are background there: label them, or leave them out)"] = [str(t.path)
                                                                                            for t in un[:500]]
        for t in un:
            P.report.append(["dataset image not in the source", "", str(t.path), "", "not given the added classes"])


def _plan_adds(P, S, new_idx, T, T_keys, new_cls, need_class, progress, cancel):
    o = P.opts
    keep = []
    for i in new_idx:
        s = S[i]
        if s.bad:
            P.stats["invalid"] += 1
            P.details["Annotations that could not be read (left out)"].append(f"{s.ann}: {s.bad}")
            P.report.append(["annotation unreadable", str(s.path), "", "", s.bad])
            continue
        if s.ann is None and o.unlabelled != "add":
            P.stats["no_annotation"] += 1
            P.report.append(["no annotation", str(s.path), "", "", "left out (images without an annotation)"])
            continue
        if T.fmt == "yolo":
            unknown = sorted({lab for lab in s.labels if _key(lab) not in T_keys and
                              _key(lab) not in {_key(c) for c in new_cls}})
            if unknown and o.new_classes == "skip":
                P.stats["unknown_skipped"] += 1
                P.report.append(["class not in the dataset", str(s.path), "", "", "left out: " + ", ".join(unknown)])
                continue
            if unknown and o.new_classes == "add":
                for lab in unknown:
                    need_class(lab)
            if unknown and o.new_classes == "drop":
                P.stats["objects_dropped"] += sum(1 for lab in s.labels if _key(lab) not in T_keys)
        keep.append(s)
    if o.check_images and keep:
        def chk(s):
            if cancel is not None and cancel.is_set():
                raise C.Cancelled()
            ok, why, wh = C.verify_file(s.path)
            if ok and wh and not s.w:
                s.w, s.h = wh
            return ok, why
        _say(progress, 0, len(keep), "Checking that every new image opens")
        with ThreadPoolExecutor(o.workers) as ex:
            res = list(ex.map(chk, keep))
        good = []
        for s, (ok, why) in zip(keep, res):
            if ok:
                good.append(s)
            else:
                P.stats["unreadable"] += 1
                P.details["Images that could not be read (left out)"].append(f"{s.path}: {why}")
                P.report.append(["unreadable image", str(s.path), "", "", why])
        keep = good
    if not keep:
        return
    split = _assign_splits(P, keep, T)
    _name_and_place(P, keep, split, T)


def _assign_splits(P, keep, T) -> dict:
    """id(item) -> canonical split ("" for a dataset without splits)."""
    o = P.opts
    splits = T.splits
    if not splits:
        return {id(s): "" for s in keep}
    base = "train" if "train" in splits else splits[0]
    if o.split_mode == "source":
        return {id(s): (s.split if s.split in ("train", "val", "test") else base) for s in keep}
    if o.split_mode != "balanced":
        return {id(s): base for s in keep}
    from AIO_QC.core import split as SP
    n = Counter(it.split for it in T.items)
    tot = sum(n.values()) or 1
    val = o.val if o.val is not None else n["val"] / tot
    test = o.test if o.test is not None else n["test"] / tot
    if val + test >= 0.9:
        raise ValueError("Validation and test shares leave too little for training.")
    keys = [str(k) for k in range(len(keep))]
    links = []
    seq = {}
    for k, s in zip(keys, keep):
        sid = SP.sequence_id(Path(s.rel).stem, s.dir_rel + "|" + str(s.ds.root))
        if sid:
            if sid in seq:
                links.append((seq[sid], k))
            else:
                seq[sid] = k
    groups = SP.build_groups(keys, links)
    items = [SP.SplitItem(k, SP.detection_features(Counter(s.labels), []), groups[k]) for k, s in zip(keys, keep)]
    sp = SP.plan_split(items, val=val, test=test, seed=o.seed)
    P.details["Split of the new images"].append(
        f"val {val:.0%}, test {test:.0%} ({'the dataset' if o.val is None else 'yours'}) — frames of one video stay "
        f"together ({len(set(groups.values())):,} groups), classes seen in ≤ 3 images stay in train")
    return {id(s): sp.assign.get(k, base) for k, s in zip(keys, keep)}


def _src_rel(s: Img, many: bool) -> str:
    """The folders that tell the new files apart (images/ labels/ and the split dropped; the source's name first when
    there are several sources) — for the naming tokens."""
    parts = [p for p in s.dir_rel.split("/") if p and p.lower() not in ("images", "labels") and p.lower() not in
             SPLITS]
    if many:
        parts = [s.ds.root.name] + parts
    return "/".join(parts + [s.path.name])


def _yolo_img_dir(T: Dataset, split: str) -> Path:
    if split in T.image_dirs:
        return T.image_dirs[split]
    name = T.split_dirs.get(split, split)
    if not split or T.layout == "flat":
        return T.image_dirs.get("", T.root / "images")
    if T.layout == "images_first":
        return T.root / "images" / name
    return T.root / name / "images"


def _labels_for(img_dir: Path) -> Path:
    parts = list(img_dir.parts)
    i = len(parts) - 1 - [p.lower() for p in parts][::-1].index("images")
    parts[i] = "labels"
    return Path(*parts)


def _lm_dest_dir(P, s: Img, split: str, T: Dataset) -> str:
    o = P.opts
    parts = []
    if split:
        parts.append(T.split_dirs.get(split, split))
    parts += [p for p in re.split(r"[\\/]+", o.subfolder.strip()) if p and p not in (".", "..")]
    if o.placement == "mirror":
        parts += [p for p in s.dir_rel.split("/") if p and p.lower() not in ("images", "labels") and
                  p.lower() not in SPLITS]
    return "/".join(parts)


def _name_and_place(P, keep, split, T):
    from AIO_QC.core import naming as N
    o = P.opts
    many = len(P.sources) > 1
    items, dest = [], {}
    for k, s in enumerate(keep):
        sp = split[id(s)]
        d = sp if T.fmt == "yolo" else _lm_dest_dir(P, s, sp, T)
        dest[k] = d
        items.append(N.Item(k, _src_rel(s, many), split=d, path=str(s.path),
                            extra_exts=(".txt",) if T.fmt == "yolo" else (".json",)))
    existing = {k.lower(): v for k, v in T.names.items()}
    by_dest = defaultdict(Counter)
    for it in items:
        by_dest[it.split.lower()][it.nk] += 1
    touch = {it.key for it in items if by_dest[it.split.lower()][it.nk] > 1 or it.nk in existing.get(it.split.lower(),
                                                                                                        ())}
    sc = N.Scheme.from_dict(o.naming) if o.naming else N.Scheme(preset="suffix")
    longest = max((len(d) for d in dest.values()), default=0)
    root_len = len(str(T.root)) + 1 + longest + (len("/images") if T.fmt == "yolo" else 0)
    NP = N.plan(items, sc, flat=True, existing=existing, root_len=root_len, touch=touch)
    if not NP.ok:
        NP = N.resolve(NP, existing=existing, root_len=root_len)
    if not NP.ok:
        bad = sorted(NP.problems.items())[:5]
        raise ValueError("Some names could not be made unique: " + "; ".join(f"{keep[k].path.name}: {', '.join(v)}"
                                                                             for k, v in bad))
    for it in NP.items:
        s = keep[it.key]
        stem = NP.new[it.key]
        sp = split[id(s)]
        if T.fmt == "yolo":
            idir = _yolo_img_dir(T, sp)
            dst_img, dst_ann = idir / (stem + s.path.suffix), _labels_for(idir) / (stem + ".txt")
        else:
            d = T.root.joinpath(*[p for p in dest[it.key].split("/") if p])
            dst_img, dst_ann = d / (stem + s.path.suffix), d / (stem + ".json")
        renamed = stem != s.path.stem
        P.stats["renamed"] += renamed
        P.stats[f"to_{sp}"] += 1
        P.stats["background"] += not s.labels
        P.adds.append({"item": s, "split": sp, "dst_img": dst_img, "dst_ann": dst_ann, "renamed": renamed})
        P.report.append(["added" if not renamed else "added, renamed", str(s.path), "", str(dst_img),
                         (f"was {s.path.name}; " if renamed else "") + (sp or "")])
        if renamed and len(P.details["Renamed (the name was taken by a different image)"]) < 1000:
            P.details["Renamed (the name was taken by a different image)"].append(f"{s.path.name}  →  "
                                                                                  f"{dst_img.relative_to(T.root)}")


# ====================================================================================================== run
def _class_ids(P) -> dict:
    return {_key(c): i for i, c in enumerate(P.classes_after)}


def _new_annotation(P, s: Img, base: Img | None, kind: str):
    """The bytes of the annotation to write: for an added image (base None) or a matched dataset image (replace /
    merge). None: nothing to write (YOLO: never — an empty file is background)."""
    o, T = P.opts, P.target
    sa = read(s)
    if sa is not None and sa[0] == "labelme":
        image_size(s, sa[1])
    sobj = objects(s, sa, o.rename)
    tdata = None
    if base is not None:
        ta = read(base)
        if ta is not None and ta[0] == "labelme":
            image_size(base, ta[1])
            tdata = ta[1]
        if s.w and not base.w:
            base.w, base.h = s.w, s.h
        tobj = objects(base, ta)
        objs = sobj if kind == "replace" else merge(tobj, sobj, P.merge_keys, o.conflict, o.iou)[0]
        ref = base
    else:
        objs, ref = sobj, s
    if T.fmt == "yolo":
        W, H = image_size(ref)
        lines, _d = to_yolo(objs, W, H, _class_ids(P), T.task)
        return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    W, H = image_size(ref)
    if tdata is not None:
        data = dict(tdata, shapes=to_shapes(objs))
    elif sa is not None and sa[0] == "labelme":
        data = dict(sa[1], shapes=to_shapes(objs))
    else:
        if base is None and not objs and s.ann is None:
            return None                       # a background LabelMe image: the image alone
        data = {"version": LABELME_VERSION, "flags": {}, "shapes": to_shapes(objs), "imagePath": "",
                "imageData": None, "imageHeight": H, "imageWidth": W}
    data["imagePath"] = (base.path.name if base is not None else None) or data.get("imagePath")
    return data


def _write_class_lists(P, log):
    T = P.target
    names = P.classes_after
    files = T.class_files or [T.root / "classes.txt"]
    for f in files:
        try:
            orig = log.keep_original(f) if f.exists() else None
            if f.suffix.lower() in (".yaml", ".yml"):
                text = _yaml_with_names(f.read_text(encoding="utf-8", errors="replace"), names)
            else:
                text = "\n".join(names) + "\n"
            C.write_atomic(str(f), text.encode("utf-8"))
            if orig:
                log.rewritten(f, orig, f"class list: + {', '.join(P.new_classes)}")
            else:
                log.created(f, "class list")
        except OSError as exc:
            log.error("class list", str(f), str(f), exc)
            raise


def _yaml_with_names(text: str, names: list) -> str:
    """The YAML with its names / nc replaced — everything else (comments, paths) as it was."""
    from AIO_QC.core.yolo_convert import _yaml_str
    block = "names:\n" + "".join(f"  {i}: {_yaml_str(n)}\n" for i, n in enumerate(names))
    lines = text.splitlines(keepends=True)
    out, i, done = [], 0, False
    while i < len(lines):
        ln = lines[i]
        if not done and re.match(r"names\s*:", ln):
            out.append(block)
            done = True
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i][:1] in (" ", "\t", "-")):
                if not lines[i].strip() and i + 1 < len(lines) and lines[i + 1][:1] not in (" ", "\t", "-"):
                    break
                i += 1
            continue
        if re.match(r"nc\s*:", ln):
            out.append(f"nc: {len(names)}\n")
        else:
            out.append(ln)
        i += 1
    if not done:
        out.append(("\n" if out and not out[-1].endswith("\n") else "") + block)
    res = "".join(out)
    try:
        import yaml
        d = yaml.safe_load(res)
        got = d.get("names")
        got = [got[k] for k in sorted(got)] if isinstance(got, dict) else got
        if [str(x) for x in got] != [str(x) for x in names]:
            raise ValueError("names did not come out right")
    except ImportError:
        pass
    except Exception:                      # an unusual layout: the whole file rewritten (the original is kept)
        import yaml
        d = yaml.safe_load(text) or {}
        d["names"] = {i: n for i, n in enumerate(names)}
        d["nc"] = len(names)
        res = yaml.safe_dump(d, sort_keys=False, allow_unicode=True)
    return res


def run(P: Plan, progress=None, cancel=None) -> dict:
    o, T = P.opts, P.target
    log = C.RunLog(T.root / C.LOG_DIR, "add_data", keep_originals=True)
    done, errors = Counter(), []
    from .. import provenance as PV
    origin = PV.make_origin("import", "Add new data" if o.add_new else "Add classes")
    lock = threading.Lock()
    try:
        if P.new_classes:
            _write_class_lists(P, log)
            done["classes"] = len(P.new_classes)
        n_add, n_upd = len(P.adds), len(P.updates)
        prog = [0]

        def tick(label, total):
            with lock:
                prog[0] += 1
                if progress is not None and (prog[0] % 25 == 0 or prog[0] == total):
                    progress(prog[0], total, label)

        def add(a):
            if cancel is not None and cancel.is_set():
                raise C.Cancelled()
            s, di, da = a["item"], a["dst_img"], a["dst_ann"]
            made = []
            try:
                for t in (di, da):
                    if t.exists():
                        raise FileExistsError(f"{t} already exists (changed since the plan — plan again)")
                content = _new_annotation(P, s, None, "add")
                di.parent.mkdir(parents=True, exist_ok=True)
                how = "copy"
                if o.link == "hardlink":
                    try:
                        os.link(s.path, di)
                        how = "link"
                    except OSError:
                        shutil.copy2(s.path, di)
                else:
                    shutil.copy2(s.path, di)
                made.append(di)
                log.created(di, how, str(s.path), "image")
                if content is not None:
                    if isinstance(content, dict):
                        content["imagePath"] = di.name
                        PV.stamp_labelme(content, o=origin, old=None,
                                         source={"added_from": Path(str(s.path)).parent.name,
                                                 "image": Path(str(s.path)).name})
                        content = C.dump_json(content)
                    da.parent.mkdir(parents=True, exist_ok=True)
                    C.write_atomic(str(da), content)
                    made.append(da)
                    log.created(da, "annotation", str(s.ann or ""), "annotation")
                with lock:
                    done["added"] += 1
                    done[how] += 1
            except C.Cancelled:
                raise
            except Exception as exc:
                for t in made:                         # never half a pair (both were free before)
                    try:
                        t.unlink()
                    except OSError:
                        pass
                log.error("add", str(s.path), str(di), exc)
                with lock:
                    errors.append(f"{s.path}: {type(exc).__name__}: {exc}")
            tick("Adding the new images", n_add)

        with ThreadPoolExecutor(min(8, o.workers)) as ex:
            list(ex.map(add, P.adds))
        prog[0] = 0
        for u in P.updates:
            if cancel is not None and cancel.is_set():
                raise C.Cancelled()
            t, s = u["target"], u["source"]
            ann = t.ann
            try:
                content = _new_annotation(P, s, t, u["kind"])
                if isinstance(content, dict):
                    PV.stamp_labelme(content, ann if ann is not None else None, o=origin)
                    content = C.dump_json(content)
                if ann is None:
                    ann = (_labels_for(t.path.parent) / (t.path.stem + ".txt")) if T.fmt == "yolo" else \
                        t.path.with_suffix(".json")
                    ann.parent.mkdir(parents=True, exist_ok=True)
                    C.write_atomic(str(ann), content)
                    log.created(ann, f"annotation ({u['kind']})", str(s.path), "annotation")
                else:
                    orig = log.keep_original(ann)
                    C.write_atomic(str(ann), content)
                    log.rewritten(ann, orig, f"{u['kind']} from {s.path}")
                done["updated"] += 1
            except Exception as exc:
                log.error(u["kind"], str(s.path), str(ann), exc)
                errors.append(f"{t.path}: {type(exc).__name__}: {exc}")
            tick("Updating the dataset's annotations", n_upd)
    except C.Cancelled:
        errors.append("stopped by you — what was done so far is in the log (and can be undone)")
    finally:
        log.close()
    report = Path(str(log.base) + "_report.csv")
    try:
        with open(report, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["status", "new data file", "dataset file", "added as", "note"])
            w.writerows(P.report)
    except OSError as exc:
        errors.append(f"report: {exc}")
    return {"done": done, "errors": errors, "log": str(log.csv_path), "undo": str(log.jsonl_path),
            "report": str(report), "t": time.time()}
