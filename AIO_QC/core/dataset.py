# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Dataset abstraction.

All formats are exposed to the GUI and QC checks through the same objects:

    Sample      one image/annotation pair (key = unique str)
    Annotation  LabelMe-style shape dicts: {"label", "points", "shape_type", ...}

Supported layouts
-----------------
LABELME     root/**/*.json (+ images next to them or via imagePath)
YOLO        root/images/**  +  root/labels/**      (also images/train, labels/train ...)
YOLO_SPLIT  root/<split>/images/**  +  root/<split>/labels/**   (train/val/test/...)
"""
from __future__ import annotations

import os
import sys

import copy
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .class_map import ClassMap, find_class_map
from .io import (IMAGE_EXTENSIONS, any_file, atomic_write_text, dump_labelme, image_size, load_json, load_json_light,
                 natural_key, walk_files)
from AIO_QC.core.diag import ignored as _ignored

SPLIT_ORDER = {"train": 0, "val": 1, "valid": 1, "validation": 1, "test": 2}
SPLIT_NAMES = frozenset(SPLIT_ORDER)          # the ONLY folder names treated as splits (case-insensitive)


def is_split_name(name: str) -> bool:
    return name.lower() in SPLIT_NAMES


VAL_NAMES = frozenset({"val", "valid", "validation"})
CONTAINERS = frozenset({"images", "labels"})


class SplitResolver:
    """
    Dataset-level split rule. Folders are splits only when
      * they are siblings under the same parent folder, and
      * that parent has BOTH `train` and `val` (or `valid` / `validation`) — `test` is optional.
    A lone `train/`, `val/` or `test/`, or train / val under different parents, are ordinary folders.
    If qualifying parents exist at several depths, the shallowest level wins (every qualifying parent at that
    level counts, e.g. cam1/{train,val} and cam2/{train,val}).
    """

    def __init__(self, dir_parts_list):
        children: dict[tuple, set] = {}
        for parts in dir_parts_list:
            for i, d in enumerate(parts):
                if is_split_name(d):
                    children.setdefault(tuple(parts[:i]), set()).add(d.lower())
        qualifying = [p for p, names in children.items() if "train" in names and names & VAL_NAMES]
        depth = min((len(p) for p in qualifying), default=None)
        self.parents = {p for p in qualifying if len(p) == depth}
        # explain the decision (shown when the dataset opens)
        splits = sorted({n for p in self.parents for n in children[p]}, key=lambda n: (SPLIT_ORDER.get(n, 9), n))
        deeper = sorted({"/".join(p + (n,)) for p in qualifying if p not in self.parents for n in children[p]})
        unpaired = sorted({"/".join(p + (n,)) for p, names in children.items() if p not in qualifying for n in names})
        fmt = lambda xs: ", ".join(xs[:4]) + (" …" if len(xs) > 4 else "")
        parts = []
        if splits:
            where = "dataset root" if self.parents == {()} else ", ".join("/".join(p) for p in sorted(self.parents))
            parts.append(f"Splits: {', '.join(splits)} (in {where})")
        if unpaired:
            parts.append(f"normal folders — no train + val pair beside them: {fmt(unpaired)}")
        if deeper:
            parts.append(f"normal folders — deeper than the split level: {fmt(deeper)}")
        self.note = "; ".join(parts)

    def __call__(self, parts) -> tuple[str, str]:
        """relative directory parts -> (split, source folder)."""
        parts = tuple(parts)
        split_i = next((i for i, d in enumerate(parts) if is_split_name(d) and parts[:i] in self.parents), None)
        split = parts[split_i].lower() if split_i is not None else ""
        folder = next((d for i, d in enumerate(parts) if i != split_i and d.lower() not in CONTAINERS), "")
        return split, folder


class PathOrder:
    """Dataset order — the file list's order, one rule for discovery and for files that appear later:
    per folder its files, then its sub-folders, both in natural order (frame_2 before frame_10); split folders
    (the ones SplitResolver accepted) as train → val → test. Keys of folder paths are cached (few distinct)."""
    FILE = -1                                  # a file sorts before the sub-folders next to it

    def __init__(self, parents=()):
        self.parents = set(parents or ())
        self._dirs: dict[tuple, tuple] = {}

    def dirs(self, parts) -> tuple:
        parts = tuple(parts)
        k = self._dirs.get(parts)
        if k is None:
            k = tuple((SPLIT_ORDER.get(d.lower(), 9) if is_split_name(d) and parts[:i] in self.parents else 9,
                       natural_key(d)) for i, d in enumerate(parts))
            self._dirs[parts] = k
        return k

    def key(self, parts, name: str) -> tuple:
        return self.dirs(parts) + ((self.FILE, natural_key(name)),)

    def arrange(self, items: list, parts_of) -> list:
        """Walk order (already natural per folder) -> dataset order: only split folders move, so it is a stable
        sort by the folder key (nearly sorted input: close to linear), and none at all without splits."""
        if self.parents:
            items.sort(key=lambda x: self.dirs(parts_of(x)))
        return items


LABELME_REQUIRED = {"shapes", "imagePath"}
LABELME_EXPECTED = {"version", "shapes", "imagePath", "imageData"}


class AnnotationError(Exception):
    """Annotation exists but cannot be interpreted."""


class DatasetType(str, Enum):
    LABELME = "labelme"
    YOLO = "yolo"
    YOLO_SPLIT = "yolo_split"
    CLS_FOLDERS = "cls_folders"        # classification: <class>/<image> (optionally under train/val/test)
    CLS_LABELME = "cls_labelme"        # classification: LabelMe JSON image flags
    COCO = "coco"                      # COCO JSON file(s), opened in place (core.coco_dataset)

    @property
    def title(self) -> str:
        return {"labelme": "LabelMe", "yolo": "YOLO (images/labels)", "yolo_split": "YOLO split / sub-folders",
                "cls_folders": "Classification (class folders)",
                "cls_labelme": "Classification (LabelMe flags)", "coco": "COCO JSON"}[self.value]

    @property
    def is_classification(self) -> bool:
        return self in (DatasetType.CLS_FOLDERS, DatasetType.CLS_LABELME)


DETECTION_TYPES = {DatasetType.LABELME, DatasetType.YOLO, DatasetType.YOLO_SPLIT, DatasetType.COCO}   # shapes
FILE_TYPES = {DatasetType.LABELME, DatasetType.YOLO, DatasetType.YOLO_SPLIT}    # one annotation file per image
CLASSIFICATION_TYPES = {DatasetType.CLS_FOLDERS, DatasetType.CLS_LABELME}


@dataclass(eq=False, **({"slots": True} if sys.version_info >= (3, 10) else {}))
class Sample:                    # slots: ~2.5× less memory per file (matters at millions of files)
    key: str
    name: str
    ann_path: Path | None
    image_path: Path | None
    split: str = ""              # train / val / test only ("" when the dataset has no splits)
    folder: str = ""             # source sub-folder (any other nesting) — for filters / slices, never a split

    @property
    def display(self) -> str:
        return f"[{self.split}] {self.name}" if self.split else self.name


@dataclass
class Annotation:
    shapes: list[dict]
    width: float | None = None
    height: float | None = None
    normalized: bool = False
    raw: Any = None
    image_path: Path | None = None
    image_data: str | None = None
    exists: bool = True
    light: bool = False          # loaded without imageData (read-only): must never be saved
    notes: list[str] = field(default_factory=list)
    tags: list = field(default_factory=list)       # image-level classes (classification datasets)
    orig_tags: list | None = None                  # tags before label mapping
    count_shapes: bool = True                      # classification with flags: shapes in the file are not classes
    flag_issues: list = field(default_factory=list)   # [(kind, flag, value)] — LabelMe-flag classification

    def label_counts(self) -> Counter:
        """Shape labels + image-level classes."""
        c = Counter(str(s.get("label", "")) for s in self.shapes) if self.count_shapes else Counter()
        c.update(self.tags)
        return c

    def raw_label_counts(self) -> Counter:
        """Like label_counts, but with the ORIGINAL labels when a label mapping is active."""
        c = Counter(str(s.get("_orig_label", s.get("label", ""))) for s in self.shapes) if self.count_shapes \
            else Counter()
        c.update(self.orig_tags if self.orig_tags is not None else self.tags)
        return c

    def scaled(self, w: float, h: float) -> "Annotation":
        """Normalized -> pixel copy (for YOLO display / pixel-space checks)."""
        if not self.normalized:
            return self
        shapes = []
        for s in self.shapes:
            s2 = dict(s)
            s2["points"] = [[p[0] * w, p[1] * h] for p in s.get("points", [])]
            shapes.append(s2)
        out = copy.copy(self)
        out.shapes, out.width, out.height, out.normalized = shapes, float(w), float(h), False
        return out


def is_labelme(data: Any) -> tuple[bool, list[str]]:
    """Schema check (from count_labelme_classes.labelme_json) -> (valid, missing keys)."""
    if not isinstance(data, dict):
        return False, ["<root is not an object>"]
    missing = sorted(LABELME_EXPECTED - data.keys())
    if not LABELME_REQUIRED <= data.keys() or not isinstance(data.get("shapes"), list):
        return False, missing or ["shapes (not a list)"]
    return True, missing


def _split_key(name: str) -> tuple:
    return (SPLIT_ORDER.get(name.lower(), 9), name.lower())


# =============================================================================
# LabelMe
# =============================================================================
class LabelMeAdapter:
    type = DatasetType.LABELME

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.class_map = ClassMap()          # LabelMe has no intrinsic class map
        self.split_note = ""
        self._img_size_cache: dict[str, tuple[int, int] | None] = {}

    @property
    def class_names(self) -> list[str]:
        return self.class_map.names

    def discover(self) -> list[Sample]:
        paths = list(walk_files(self.root, {".json"}))
        dirs = [p.relative_to(self.root).parts[:-1] for p in paths]
        resolve = SplitResolver(dirs)
        self.split_note = resolve.note
        self._dirs, self._resolver = set(dirs), resolve      # kept for sample_for_path (live updates)
        self._path_order = PathOrder(resolve.parents)
        pairs = self._path_order.arrange(list(zip(paths, dirs)), lambda x: x[1])
        return [self._sample(p, d, resolve) for p, d in pairs]

    @staticmethod
    def _sample(p: Path, dirs, resolve) -> Sample:
        split, folder = resolve(dirs)
        return Sample(key=str(p), name=p.name, ann_path=p, image_path=None, split=split, folder=folder)

    def order_key(self, sample: Sample) -> tuple:
        """Sort key reproducing discovery order (PathOrder)."""
        order = getattr(self, "_path_order", None) or PathOrder()
        try:
            parts = Path(sample.ann_path).relative_to(self.root).parts
        except ValueError:
            return ((9, natural_key(str(sample.ann_path))),)
        return order.key(parts[:-1], parts[-1])

    def sample_for_path(self, path) -> tuple[Sample | None, bool]:
        """
        A file that appeared after discovery -> (sample, structural). structural=True means the new file changes the
        split layout (e.g. completes a train + val pair): the dataset must be reloaded instead of patched.
        """
        p = Path(path)
        if p.suffix.lower() != ".json" or not hasattr(self, "_resolver"):
            return None, False
        try:
            dirs = p.relative_to(self.root).parts[:-1]
        except ValueError:
            return None, False
        if any(d.startswith(".") for d in dirs):
            return None, False
        if dirs not in self._dirs:
            trial = SplitResolver(list(self._dirs) + [dirs])
            if trial.parents != self._resolver.parents:
                return None, True
            self._dirs.add(dirs)
        return self._sample(p, dirs, self._resolver), False

    def resolve_image(self, sample: Sample, data: dict | None) -> Path | None:
        if sample.image_path is not None and sample.image_path.exists():
            return sample.image_path
        folder = sample.ann_path.parent
        candidates = []
        ip = (data or {}).get("imagePath")
        if ip:
            ip = str(ip).replace("\\", "/")
            candidates += [folder / ip, folder / Path(ip).name]
        candidates += [sample.ann_path.with_suffix(ext) for ext in (".jpg", ".png", ".jpeg", ".bmp", ".webp")]
        for c in candidates:
            if c.is_file():
                # abspath = string normalisation (collapses '..'); resolve() hit the file system per component
                # (~70 µs) and ran for every file during indexing
                sample.image_path = Path(os.path.abspath(c))
                return sample.image_path
        return None

    def load(self, sample: Sample, image_size: tuple[int, int] | None = None, light: bool = False) -> Annotation:
        """light=True: skip the embedded base64 image (fast, read-only)."""
        try:
            data = load_json_light(sample.ann_path) if light else load_json(sample.ann_path)
        except Exception as exc:
            raise AnnotationError(f"Unreadable JSON: {exc}") from exc
        ok, missing = is_labelme(data)
        if not ok:
            from .formats import labelme_report
            raise AnnotationError(f"Not a LabelMe JSON — looks like {labelme_report(data).looks_like}")
        shapes = [s for s in data["shapes"] if isinstance(s, dict)]
        ann = Annotation(
            shapes=shapes,
            width=data.get("imageWidth") or None,
            height=data.get("imageHeight") or None,
            raw=data,
            image_path=self.resolve_image(sample, data),
            image_data=data.get("imageData") or None,
            light=light,
        )
        if len(shapes) != len(data["shapes"]):
            ann.notes.append(f"{len(data['shapes']) - len(shapes)} non-object entries in 'shapes'")
        if missing:
            ann.notes.append(f"Missing keys: {', '.join(missing)}")
        return ann

    def save(self, sample: Sample, ann: Annotation) -> None:
        if ann.light:
            raise AnnotationError("internal: refusing to save a light (imageData-less) annotation")
        data = ann.raw
        data["shapes"] = [{k: v for k, v in s.items() if not k.startswith("_")} for s in ann.shapes]
        from . import provenance
        provenance.stamp_labelme(data, sample.ann_path)          # what made / changed each shape (no person)
        dump_labelme(sample.ann_path, data)

    def image_size(self, sample: Sample, ann: Annotation | None = None) -> tuple[int, int] | None:
        if ann is not None and ann.width and ann.height:
            return int(ann.width), int(ann.height)
        if sample.key not in self._img_size_cache:
            p = ann.image_path if ann else sample.image_path
            self._img_size_cache[sample.key] = image_size(p) if p else None
        return self._img_size_cache[sample.key]

    def can_use_label(self, label: str) -> bool:
        return bool(label.strip())

    def orphans(self, samples: list[Sample]) -> dict[str, list[Path]]:
        """Images with no JSON next to them (same folder + stem)."""
        json_keys = {(s.ann_path.parent, s.ann_path.stem) for s in samples}
        linked = {os.path.abspath(s.image_path) for s in samples if s.image_path}
        imgs = [p for p in walk_files(self.root, IMAGE_EXTENSIONS)
                if (p.parent, p.stem) not in json_keys and os.path.abspath(p) not in linked]
        return {"images_without_annotation": imgs, "annotations_without_image": []}


# =============================================================================
# YOLO (converted and split)
# =============================================================================
TRUE_WORDS = {"true", "yes", "y", "1", "on"}
FALSE_WORDS = {"false", "no", "n", "0", "off", ""}


def flag_value(v) -> tuple[bool, bool]:
    """A LabelMe flag value -> (is it set, was it written as a proper true / false). 1, "true", "yes" … count as set
    (and are reported, and written back as true / false when the file is saved)."""
    if v is True or v is False:
        return v, True
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v != 0, False
    if isinstance(v, str):
        w = v.strip().lower()
        return w in TRUE_WORDS, False
    return False, False


class LabelMeClassAdapter(LabelMeAdapter):
    """Classification with LabelMe image flags: the flags that are set are the image's classes. Shapes a file may
    also hold are kept as they are but are not classes. Values like 1 or "true" count as set and are reported."""
    type = DatasetType.CLS_LABELME

    def __init__(self, root: str | Path):
        super().__init__(root)
        self._flag_names: set[str] = set()

    @property
    def class_names(self) -> list[str]:
        return sorted(self._flag_names, key=lambda n: (n.lower(), n))

    def load(self, sample: Sample, image_size=None, light: bool = False) -> Annotation:
        ann = super().load(sample, image_size, light=light)
        ann.count_shapes = False
        raw = ann.raw if isinstance(ann.raw, dict) else {}
        flags = raw.get("flags")
        if not isinstance(flags, dict):
            if ann.exists:
                ann.flag_issues.append(("no_flags", "", None if flags is None else flags))
            flags = {}
        self._flag_names.update(str(k) for k in flags)
        tags = []
        for k, v in flags.items():
            on, clean = flag_value(v)
            if not clean:
                ann.flag_issues.append(("value", str(k), v))
            if on:
                tags.append(str(k))
        ann.tags = tags
        return ann

    def save(self, sample: Sample, ann: Annotation, drop=()) -> None:
        flags = {str(k): k in ann.tags for k in (ann.raw.get("flags") or {}) if k not in drop}   # real true / false
        for t in ann.tags:
            flags[t] = True
        ann.raw["flags"] = flags
        super().save(sample, ann)

    def set_class(self, sample: Sample, new_class: str) -> Sample:
        return self.set_tags(sample, [new_class])

    def set_tags(self, sample: Sample, tags: list[str], drop=()) -> Sample:
        """The image's classes (flags set true); `drop`: flag names removed from the file altogether."""
        ann = self.load(sample)
        ann.tags = [t for t in dict.fromkeys(tags) if t and t not in drop]
        self.save(sample, ann, drop=drop)
        self._flag_names.update(ann.tags)
        return sample

    def forget_names(self, names):
        self._flag_names.difference_update(names)

    def flags_of(self, sample: Sample) -> dict | None:
        """The file's flags exactly as written (None: no flags key) — for undo."""
        ann = LabelMeAdapter.load(self, sample)
        f = ann.raw.get("flags") if isinstance(ann.raw, dict) else None
        return dict(f) if isinstance(f, dict) else None

    def write_flags(self, sample: Sample, flags: dict | None) -> None:
        """Put a file's flags back exactly (undo / redo)."""
        ann = LabelMeAdapter.load(self, sample)
        if flags is None:
            ann.raw.pop("flags", None)
        else:
            ann.raw["flags"] = dict(flags)
        LabelMeAdapter.save(self, sample, ann)
        self._flag_names.update(str(k) for k in (flags or {}))

    def rename_class(self, sample: Sample, mapping: dict) -> bool:
        """Flags renamed / merged / removed (mapping old -> new name, or None = removed). True when the file changed."""
        ann = self.load(sample)
        flags = dict(ann.raw.get("flags") or {}) if isinstance(ann.raw, dict) else {}
        if not any(k in mapping for k in flags):
            return False
        out = {}
        for k, v in flags.items():
            on, _c = flag_value(v)
            if k in mapping:
                new = mapping[k]
                if new is None:
                    continue
                out[new] = bool(out.get(new)) or on
            else:
                out[k] = bool(out.get(k)) or on
        ann.raw["flags"] = out
        ann.tags = [k for k, v in out.items() if v]
        LabelMeAdapter.save(self, sample, ann)
        for old, new in mapping.items():
            self._flag_names.discard(old)
            if new:
                self._flag_names.add(new)
        return True


class ClassFolderAdapter:
    """
    Classification as folders: <class>/<image>, or <split>/<class>/<image> when train + val (+ test) folders exist.
    Deeper folders inside a class folder are kept as the sample's folder. The class of an image = its class folder;
    changing it MOVES the image (set_class). There are no annotation files (ann_path is None).
    """
    type = DatasetType.CLS_FOLDERS

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._img_size_cache: dict = {}
        self._classes: list[str] = []
        self.split_note = ""
        self.bases = self._bases()

    def _bases(self) -> list[tuple[str, Path]]:
        out = class_folder_splits(self.root)
        if out:
            self.split_note = "splits: " + ", ".join(n for n, _d in out)
            return out
        return [("", self.root)]

    @property
    def class_names(self) -> list[str]:
        """The class folders that still exist (one renamed, merged or removed in the app drops out)."""
        alive = [c for c in self._classes if any((b / c).is_dir() for _s, b in self.bases)]
        if len(alive) != len(self._classes):
            self._classes = alive
        return list(alive)

    def _issues_dirs(self) -> set[str]:
        """Top-level folders AIO QC made for set-aside images (core.issues_folder): never classes."""
        from .issues_folder import is_issues_dir
        return {d.name for d in _subdirs(self.root) if is_issues_dir(d)}

    def discover(self) -> list[Sample]:
        out, classes = [], set()
        self._skip = self._issues_dirs()
        for split, base in self.bases:
            for img in walk_files(base, IMAGE_EXTENSIONS):
                parts = img.relative_to(base).parts
                if len(parts) > 1 and base == self.root and (parts[0].lower() in ("runs", "labels")
                                                             or parts[0] in self._skip):
                    continue                          # training runs / a stray labels folder / issues: not classes
                cls = parts[0] if len(parts) > 1 else ""
                if cls:
                    classes.add(cls)
                out.append(Sample(key=str(img), name=img.name, ann_path=None, image_path=img, split=split,
                                  folder="/".join(parts[1:-1])))
        self._classes = sorted(classes, key=str.lower)
        return out

    def _base_of(self, path: Path) -> tuple[int, str, Path] | None:
        for i, (split, base) in enumerate(self.bases):
            try:
                path.relative_to(base)
                return i, split, base
            except ValueError:
                continue
        return None

    def class_of(self, sample: Sample) -> str:
        b = self._base_of(Path(sample.image_path))
        if b is None:
            return ""
        parts = Path(sample.image_path).relative_to(b[2]).parts
        return parts[0] if len(parts) > 1 else ""

    def load(self, sample: Sample, image_size=None, light: bool = False) -> Annotation:
        cls = self.class_of(sample)
        ann = Annotation(shapes=[], image_path=Path(sample.image_path), raw=None, light=light,
                         tags=[cls] if cls else [])
        if not cls:
            ann.notes.append("the image is not inside a class folder")
        return ann

    def save(self, sample: Sample, ann: Annotation) -> None:
        raise AnnotationError("class-folder datasets have no annotation files — use Change class (moves the image)")

    def set_class(self, sample: Sample, new_class: str) -> Sample:
        """Move the image into another class folder (same split, same sub-folders). Returns the new sample."""
        src = Path(sample.image_path)
        b = self._base_of(src)
        if b is None:
            raise AnnotationError("image is outside the dataset folders")
        _i, split, base = b
        parts = src.relative_to(base).parts
        rest = parts[1:] if len(parts) > 1 else parts
        dst = base.joinpath(new_class, *rest)
        if dst.exists():
            raise AnnotationError(f"{dst.name} already exists in class '{new_class}'")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
        if new_class not in self._classes:
            self._classes = sorted(set(self._classes) | {new_class}, key=str.lower)
        return Sample(key=str(dst), name=dst.name, ann_path=None, image_path=dst, split=split,
                      folder="/".join(rest[:-1]))

    def order_key(self, sample: Sample) -> tuple:
        p = Path(sample.image_path)
        b = self._base_of(p)
        if b is None:
            return (len(self.bases), (9, natural_key(str(p))))
        parts = p.relative_to(b[2]).parts
        return (b[0],) + PathOrder().key(parts[:-1], parts[-1])

    def sample_for_path(self, path) -> tuple[Sample | None, bool]:
        p = Path(path)
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            return None, False
        b = self._base_of(p)
        if b is None:
            return None, False
        _i, split, base = b
        parts = p.relative_to(base).parts
        if any(x.startswith(".") for x in parts[:-1]):
            return None, False
        if len(parts) > 1 and base == self.root:
            from .issues_folder import is_issues_dir
            if parts[0] in (getattr(self, "_skip", None) or set()) or is_issues_dir(base / parts[0]):
                return None, False                    # an image set aside in the issues folder
        cls = parts[0] if len(parts) > 1 else ""
        if cls and cls not in self._classes:            # a new class folder changes the class list -> reload
            return None, True
        return Sample(key=str(p), name=p.name, ann_path=None, image_path=p, split=split,
                      folder="/".join(parts[1:-1])), False

    def image_size(self, sample: Sample, ann: Annotation | None = None) -> tuple[int, int] | None:
        if sample.key not in self._img_size_cache:
            self._img_size_cache[sample.key] = image_size(sample.image_path) if sample.image_path else None
        return self._img_size_cache[sample.key]

    def can_use_label(self, label: str) -> bool:
        return bool(label.strip()) and not any(c in label for c in '/\\:*?"<>|')

    def orphans(self, samples: list[Sample]) -> dict[str, list[Path]]:
        return {"images_without_annotation": [Path(s.image_path) for s in samples if not self.class_of(s)],
                "annotations_without_image": []}


CLS_SPLITS = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


def _subdirs(root: Path) -> list[Path]:
    """The visible sub-folders, sorted — from one directory listing (os.scandir knows file / folder without a disk
    call per entry: a folder of 100,000 images costs one listing, not 100,000 stat calls)."""
    try:
        with os.scandir(root) as it:
            names = [e.name for e in it if not e.name.startswith(".") and _is_dir(e)]
    except OSError:
        return []
    return [Path(root) / n for n in sorted(names)]


def _is_dir(e) -> bool:
    try:
        return e.is_dir()
    except OSError:
        return False


def class_folder_splits(root: Path) -> list[tuple[str, Path]]:
    """train / val / test folders that each hold class folders (any of them — train + test without val too).
    [] when the root's own sub-folders are the classes."""
    subs = _subdirs(root)
    splits = [(CLS_SPLITS[d.name.lower()], d) for d in subs if d.name.lower() in CLS_SPLITS]
    if not splits:
        return []
    from .issues_folder import is_issues_dir
    others = [d for d in subs if d.name.lower() not in CLS_SPLITS and d.name.lower() not in ("labels", "runs")
              and not is_issues_dir(d) and _shallow_images(d)]
    if others:                                    # "train" next to "cat" and "dog": train is a class
        return []
    order = {"train": 0, "val": 1, "test": 2}
    seen, out = set(), []
    for name, d in sorted(splits, key=lambda x: (order[x[0]], x[1].name)):
        if name not in seen and _subdirs(d):
            seen.add(name)
            out.append((name, d))
    return out


def _shallow_images(d: Path, limit: int = 400) -> bool:
    n = 0
    for _dp, _dn, fns in os.walk(d):
        for f in fns:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                return True
            n += 1
            if n > limit:
                return False
    return False


def _folder_profile(d: Path, limit: int = 300) -> tuple[int, int]:
    """(images, annotation files) among the first files of a folder tree."""
    imgs = anns = 0
    for _dp, dns, fns in os.walk(d):
        dns[:] = [x for x in dns if not x.startswith(".")]
        for f in fns:
            e = os.path.splitext(f)[1].lower()
            if e in IMAGE_EXTENSIONS:
                imgs += 1
            elif e in (".json", ".txt", ".xml"):
                anns += 1
            if imgs + anns >= limit:
                return imgs, anns
    return imgs, anns


def _looks_like_class_folders(root: Path) -> bool:
    """>= 2 sub-folders with images where annotation files are rare (a stray notes.txt is fine) — at the root, or
    inside train / val / test."""
    if not root.is_dir() or (root / "images").is_dir():
        return False
    splits = class_folder_splits(root)
    bases = [d for _n, d in splits] or [root]
    imgs = anns = 0
    n_cls = 0
    for base in bases:
        from .issues_folder import is_issues_dir
        for d in _subdirs(base)[:200]:
            if d.name.lower() in ("images", "labels", "runs") or is_issues_dir(d):
                continue
            i, a = _folder_profile(d, 120)
            if i:
                n_cls += 1
                imgs += i
                anns += a
    return n_cls >= 2 and anns <= 0.2 * imgs


def labelme_flags_profile(root: Path, probe: int = 60) -> tuple[int, int, int]:
    """Among the first LabelMe JSONs: (how many, with image flags, without shapes)."""
    n = flagged = shapeless = 0
    for p in walk_files(root, {".json"}):
        try:
            d = load_json_light(p)
        except Exception as _exc:
            _ignored(_exc)
            continue
        if not isinstance(d, dict) or "shapes" not in d:
            continue
        n += 1
        flags = d.get("flags")
        flagged += isinstance(flags, dict) and bool(flags)
        shapeless += not d.get("shapes")
        if n >= probe:
            break
    return n, flagged, shapeless


def _looks_like_labelme_flags(root: Path, probe: int = 60) -> bool:
    """The JSONs carry image flags and (mostly) no shapes: classification."""
    n, flagged, shapeless = labelme_flags_profile(root, probe)
    return n > 0 and flagged > 0 and shapeless / n >= 0.5


def _yolo_children(root: Path) -> list[Path]:
    return [c for c in _subdirs(root) if (c / "images").is_dir() and (c / "labels").is_dir()]


def find_yolo_splits(root: Path) -> list[Path]:
    """
    <split>/images + labels children of root that form a split set: train AND val (valid / validation) must both be
    present (test optional). Otherwise there are no splits: an empty list.
    """
    cands = [c for c in _yolo_children(root) if is_split_name(c.name)]
    names = {c.name.lower() for c in cands}
    if "train" not in names or not names & VAL_NAMES:
        return []
    return sorted(cands, key=lambda p: _split_key(p.name))


class YoloAdapter:
    def __init__(self, root: str | Path, split_layout: bool):
        self.root = Path(root)
        self.type = DatasetType.YOLO_SPLIT if split_layout else DatasetType.YOLO
        if split_layout:
            # split folders first; other <dir>/images+labels siblings are loaded too, as source folders (no split)
            split_dirs = set(find_yolo_splits(self.root))
            others = sorted(c for c in _yolo_children(self.root) if c not in split_dirs)
            self.bases: list[tuple[str, Path]] = [(p.name, p) for p in find_yolo_splits(self.root)] + \
                                                 [(f"folder:{c.name}", c) for c in others]
        else:
            self.bases = [("", self.root)]
        self.class_map: ClassMap = find_class_map(self.root, tuple(b / "labels" for _, b in self.bases))
        self.split_note = ""
        self.name_to_id = {n: i for i, n in enumerate(self.class_map.names)}
        self._img_size_cache: dict[str, tuple[int, int] | None] = {}

    @property
    def class_names(self) -> list[str]:
        return self.class_map.names

    @property
    def has_yaml(self) -> bool:
        return self.class_map.kind == "yaml"

    def set_class_names(self, names: list[str], source: Path | None = None) -> None:
        """Use a user-loaded classes.txt as the class map (only when no dataset.yaml exists)."""
        if self.has_yaml:
            raise ValueError(f"Class names come from {self.class_map.source.name}; remove it to use classes.txt")
        self.class_map = ClassMap(names=list(names), source=Path(source) if source else None, kind="user")
        if len(set(names)) != len(names):
            self.class_map.problems.append("classes file contains duplicate class names")
        self.name_to_id = {n: i for i, n in enumerate(self.class_map.names)}

    def discover(self) -> list[Sample]:
        out = []
        notes = []
        self._bases_state = []          # (base_name, images_root, labels_root, resolver | None, dir set) — live updates
        self._path_orders = {}
        for base_name, base in self.bases:
            images_root, labels_root = base / "images", base / "labels"
            imgs = list(walk_files(images_root, IMAGE_EXTENSIONS))
            rels = [img.relative_to(images_root).parts[:-1] for img in imgs]
            # root/images/<split>/… : resolve inside images/. Split layout: the base folder IS the split.
            resolve = SplitResolver(rels) if base_name == "" else None
            if resolve is not None and resolve.note:
                notes.append(resolve.note)
            self._bases_state.append((base_name, images_root, labels_root, resolve, set(rels)))
            pairs = PathOrder(resolve.parents if resolve is not None else ()).arrange(list(zip(imgs, rels)),
                                                                                     lambda x: x[1])
            for img, dparts in pairs:
                out.append(self._sample(base_name, images_root, labels_root, img, dparts, resolve))
        self._finish_discover_notes(notes)
        return out

    @staticmethod
    def _sample(base_name, images_root, labels_root, img: Path, dparts, resolve) -> Sample:
        rel = img.relative_to(images_root)
        if resolve is not None:
            split, folder = resolve(dparts)
        else:
            split = "" if base_name.startswith("folder:") else base_name.lower()
            folder = next((d for d in dparts if d.lower() not in CONTAINERS), "")
            if base_name.startswith("folder:"):
                folder = folder or base_name[len("folder:"):]
        return Sample(key=str(img), name=img.name, ann_path=labels_root / rel.with_suffix(".txt"),
                      image_path=img, split=split, folder=folder)

    def order_key(self, sample: Sample) -> tuple:
        """Discovery order: base (split / folder) order, then the images walk (files before sub-folders, by name)."""
        img = Path(sample.image_path)
        for i, (_n, images_root, _l, resolve, _d) in enumerate(getattr(self, "_bases_state", [])):
            try:
                parts = img.relative_to(images_root).parts
            except ValueError:
                continue
            orders = self.__dict__.setdefault("_path_orders", {})
            order = orders.get(i)
            if order is None:
                order = orders[i] = PathOrder(resolve.parents if resolve is not None else ())
            return (i,) + order.key(parts[:-1], parts[-1])
        return (len(getattr(self, "_bases_state", [])), (9, natural_key(str(img))))

    def sample_for_path(self, path) -> tuple[Sample | None, bool]:
        """An image that appeared after discovery -> (sample, structural). See LabelMeAdapter.sample_for_path."""
        p = Path(path)
        if p.suffix.lower() not in IMAGE_EXTENSIONS or not hasattr(self, "_bases_state"):
            return None, False
        for base_name, images_root, labels_root, resolve, dirs_seen in self._bases_state:
            try:
                dparts = p.relative_to(images_root).parts[:-1]
            except ValueError:
                continue
            if any(d.startswith(".") for d in dparts):
                return None, False
            if resolve is not None and dparts not in dirs_seen:
                trial = SplitResolver(list(dirs_seen) + [dparts])
                if trial.parents != resolve.parents:
                    return None, True
                dirs_seen.add(dparts)
            return self._sample(base_name, images_root, labels_root, p, dparts, resolve), False
        # under the root but in no known images/ folder: a new split / sub-folder layout -> reload
        try:
            rel = p.relative_to(self.root).parts
        except ValueError:
            return None, False
        return None, "images" in [x.lower() for x in rel[:-1]]

    def _finish_discover_notes(self, notes):
        if self.type == DatasetType.YOLO_SPLIT:
            names = [n for n, _b in self.bases if n and not n.startswith("folder:")]
            others = [n[len("folder:"):] for n, _b in self.bases if n.startswith("folder:")]
            notes.append((f"Splits: {', '.join(names)}" if names else
                          "No split set (needs train + val side by side)") +
                         (f"; loaded as normal folders: {', '.join(others)}" if others else ""))
        self.split_note = "; ".join(notes)

    # -- parsing -------------------------------------------------------------
    def label_for(self, cid: int) -> str:
        names = self.class_map.names
        return names[cid] if 0 <= cid < len(names) else f"class_{cid}"

    def _row_to_shape(self, parts: list[str], row: int, line_no: int) -> dict:
        shape = {"label": "<invalid>", "points": [], "shape_type": "invalid", "group_id": None,
                 "_row": row, "_line": line_no, "_class_id": None, "_error": None}
        try:
            f = float(parts[0])
            if not f.is_integer():
                raise ValueError
            cid = int(f)
        except ValueError:
            shape["_error"] = f"non-integer class id {parts[0]!r}"
            return shape
        shape["_class_id"] = cid
        shape["label"] = self.label_for(cid)
        try:
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            shape["_error"] = "non-numeric coordinate"
            return shape
        n = len(vals)
        if n == 4 or (self.class_map.kpt_shape and n > 4):
            cx, cy, w, h = vals[:4]
            shape["points"] = [[cx - w / 2, cy - h / 2], [cx + w / 2, cy + h / 2]]
            shape["shape_type"] = "rectangle"
        elif n >= 6 and n % 2 == 0:
            shape["points"] = [[vals[i], vals[i + 1]] for i in range(0, n, 2)]
            shape["shape_type"] = "polygon"
        else:
            shape["_error"] = f"unexpected value count ({n})"
        return shape

    def load(self, sample: Sample, image_size: tuple[int, int] | None = None, light: bool = False) -> Annotation:
        p = sample.ann_path
        if p is None or not p.is_file():
            ann = Annotation([], 1.0, 1.0, True, raw=[], image_path=sample.image_path, exists=False)
            return ann.scaled(*image_size) if image_size else ann
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as exc:
            raise AnnotationError(f"Unreadable label file: {exc}") from exc
        rows, shapes = [], []
        for ln, line in enumerate(text.splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            shapes.append(self._row_to_shape(parts, len(rows), ln))
            rows.append(parts)
        ann = Annotation(shapes, 1.0, 1.0, True, raw=rows, image_path=sample.image_path)
        return ann.scaled(*image_size) if image_size else ann

    def save(self, sample: Sample, ann: Annotation) -> None:
        """Lossless: untouched rows keep their original tokens; only class ids change."""
        lines = []
        for s in ann.shapes:
            label = s.get("label")
            if s.get("_row") is None:                        # shape added in this session
                if label not in self.name_to_id:
                    raise ValueError(f"'{label}' is not in the YOLO class map")
                lines.append(" ".join([str(self.name_to_id[label])] + self._geometry_tokens(s, ann)))
                continue
            parts = list(ann.raw[s["_row"]])
            cid = s.get("_class_id")
            original = self.label_for(cid) if cid is not None else "<invalid>"
            if label != original:
                if label not in self.name_to_id:
                    raise ValueError(f"'{label}' is not in the YOLO class map; add it to the YAML first")
                parts[0] = str(self.name_to_id[label])
            if s.get("_geom_dirty"):
                extra = parts[5:] if s.get("shape_type") == "rectangle" and len(parts) > 5 else []   # pose kpts
                parts = [parts[0]] + self._geometry_tokens(s, ann) + extra
            lines.append(" ".join(parts))
        text = "\n".join(lines) + ("\n" if lines else "")
        if not lines and not sample.ann_path.exists():
            return
        atomic_write_text(sample.ann_path, text)
        from . import provenance                                 # YOLO text has no room: one log line per save
        try:
            rel = os.path.relpath(sample.ann_path, self.root)
        except ValueError:
            rel = Path(sample.ann_path).name
        provenance.log_yolo(self.root, rel, provenance.yolo_counts(
            [list(r) for r in (ann.raw or [])], [ln.split() for ln in lines]))

    @staticmethod
    def _geometry_tokens(s: dict, ann: Annotation) -> list[str]:
        """Only for shapes whose points were edited; everything else keeps its original tokens."""
        sx, sy = (1.0, 1.0) if ann.normalized else (1.0 / ann.width, 1.0 / ann.height)
        pts = [(float(p[0]) * sx, float(p[1]) * sy) for p in s["points"]]
        if s.get("shape_type") == "rectangle":
            (x1, y1), (x2, y2) = pts[0], pts[1]
            vals = [(x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1), abs(y2 - y1)]
        else:
            vals = [v for p in pts for v in p]
        return [f"{v:.6f}".rstrip("0").rstrip(".") if v else "0" for v in vals]

    def image_size(self, sample: Sample, ann: Annotation | None = None) -> tuple[int, int] | None:
        if sample.key not in self._img_size_cache:
            self._img_size_cache[sample.key] = image_size(sample.image_path)
        return self._img_size_cache[sample.key]

    def can_use_label(self, label: str) -> bool:
        return label in self.name_to_id

    def orphans(self, samples: list[Sample]) -> dict[str, list[Path]]:
        have = {s.ann_path for s in samples}
        labels_wo_img = []
        for _, base in self.bases:
            for p in walk_files(base / "labels", {".txt"}):
                if p.name in ("classes.txt", "obj.names", "classes.names"):
                    continue
                if p not in have:
                    labels_wo_img.append(p)
        imgs_wo_label = [s.image_path for s in samples if not s.ann_path.exists()]
        return {"images_without_annotation": imgs_wo_label, "annotations_without_image": labels_wo_img}


# =============================================================================
# Detection / factory
# =============================================================================
def detect_dataset_types(root: str | Path) -> list[DatasetType]:
    root = Path(root)
    found = []
    if (root / "images").is_dir() and (root / "labels").is_dir():
        found.append(DatasetType.YOLO)
    if find_yolo_splits(root) or _yolo_children(root):      # split set, or per-folder images + labels
        found.append(DatasetType.YOLO_SPLIT)
    folders = _looks_like_class_folders(root)
    from .coco_dataset import find_coco_files
    coco = bool(find_coco_files(root)) if root.is_dir() else False
    n = 0
    if any_file(root, {".json"}, skip_dirs={"labels"}):
        n, flagged, shapeless = labelme_flags_profile(root)
        if n and flagged and shapeless / n >= 0.5 and not folders:
            found.append(DatasetType.CLS_LABELME)          # offered first: the JSONs (mostly) only carry image flags
        if n or (not folders and not coco):
            found.append(DatasetType.LABELME)
        if n and flagged and DatasetType.CLS_LABELME not in found:
            found.append(DatasetType.CLS_LABELME)          # boxes AND image flags: their flags as classes, too
    if coco:                                               # after LabelMe when LabelMe JSONs are there too (a stray
        at = len(found) if n else next((i for i, t in enumerate(found)  # coco.json in a LabelMe folder)
                                         if t in (DatasetType.LABELME, DatasetType.CLS_LABELME)), len(found))
        found.insert(at, DatasetType.COCO)
    if folders:
        found.insert(0, DatasetType.CLS_FOLDERS)          # images dominate: a stray JSON / notes.txt does not matter
    return found


def open_dataset(root: str | Path, dtype: DatasetType):
    if dtype == DatasetType.COCO:
        from .coco_dataset import CocoAdapter
        return CocoAdapter(root)
    if dtype == DatasetType.LABELME:
        return LabelMeAdapter(root)
    if dtype == DatasetType.CLS_LABELME:
        return LabelMeClassAdapter(root)
    if dtype == DatasetType.CLS_FOLDERS:
        return ClassFolderAdapter(root)
    return YoloAdapter(root, split_layout=(dtype == DatasetType.YOLO_SPLIT))


def new_box_shape(label: str, xyxy_px, ann: Annotation, image_w: float, image_h: float) -> dict:
    """Rectangle shape for `ann` from a pixel box (normalised automatically for YOLO)."""
    x1, y1, x2, y2 = map(float, xyxy_px)
    if ann.normalized:
        x1, x2, y1, y2 = x1 / image_w, x2 / image_w, y1 / image_h, y2 / image_h
    return {"label": label, "points": [[x1, y1], [x2, y2]], "group_id": None, "description": "",
            "shape_type": "rectangle", "flags": {}, "_row": None, "_class_id": None, "_geom_dirty": True,
            "_new": True}
