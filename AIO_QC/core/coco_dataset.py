# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
COCO JSON as a dataset type — opened, checked and fixed IN PLACE (no conversion).

Layouts found by themselves:
    annotations/instances_<split>.json  +  <split>/ or images/<split>/ or <split>2017/      (COCO, Detectron2, MMDet)
    <split>/_annotations.coco.json with the images next to it                              (Roboflow)
    annotations.json / instances.json  +  images/  (or the images next to the file)         (CVAT, Label Studio, …)
Every COCO file is a split (named after the file or its folder).

Shapes: bbox → rectangle, polygon segmentation → polygon (an object in several pieces: one shape per piece, the
same group id), RLE masks (iscrowd) → read-only outlines traced from the mask (kept exactly as written unless you
edit them). Saving rewrites only that image's annotations in the file — untouched objects keep every field they
had (ids, area, iscrowd, attributes, keypoints, extra keys); edited ones get a new bbox / segmentation / area;
new labels become new categories. Writes are atomic; a batch (class removal, relabelling many) writes each file
once at the end.

Every sample shares its COCO file, so the session asks the adapter (shared_ann) for what per-file tools do with
separate files: per-image snapshots for Undo (unit_bytes / write_unit), removal (remove_samples) and renaming an
image (relocate).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from .dataset import AnnotationError, Annotation, DatasetType, IMAGE_EXTENSIONS, Sample
from . import provenance as PV
from .io import atomic_write_text, image_size as _img_size, natural_key

SPLIT_WORDS = (("train", "train"), ("val", "val"), ("valid", "val"), ("test", "test"))


_LABELME_KEYS = (b'"shapes"', b'"imagePath"', b'"imageData"')
COCO_NAME_HINTS = ("coco", "instances", "annotation", "train", "val", "test", "person_keypoints", "panoptic")
MAX_PROBES_PER_DIR = 12


def is_coco_file(p: Path) -> bool:
    """A COCO file: a JSON object with "images" and "annotations". A LabelMe file is told apart from its first 4 KB
    (no need to read its embedded image); a COCO candidate from its first 256 KB."""
    try:
        if p.stat().st_size < 20:
            return False
        with open(p, "rb") as f:
            head = f.read(4096)
            if any(k in head for k in _LABELME_KEYS):         # a LabelMe file: stop here
                return False
            head += f.read((1 << 18) - len(head))
    except OSError:
        return False
    s = head.lstrip()
    if re.search(rb'"images"\s*:\s*\[\s*\]', head):          # no images: nothing to open
        return False
    return s[:1] == b"{" and b'"images"' in head and (b'"annotations"' in head or b'"categories"' in head)


def _json_files(d: Path) -> list[Path]:
    try:
        with os.scandir(d) as it:
            names = [e.name for e in it if e.name.lower().endswith(".json") and not e.name.startswith(".")]
    except OSError:
        return []
    return [Path(d) / n for n in sorted(names)]


def find_coco_files(root: Path) -> list[Path]:
    """COCO files in the root, annotations/ and the root's sub-folders. A folder with many JSONs is a LabelMe folder:
    only files whose name looks like a COCO file are looked at there, at most MAX_PROBES_PER_DIR per folder — opening a
    big LabelMe dataset no longer reads every JSON."""
    root = Path(root)
    if not root.is_dir():
        return []
    from .dataset import _subdirs
    dirs = [root] + ([root / "annotations"] if (root / "annotations").is_dir() else []) + \
        [d for d in _subdirs(root) if d.name != "annotations"]
    seen, out = set(), []
    for d in dirs:
        files = _json_files(d)
        if len(files) > MAX_PROBES_PER_DIR:               # many JSONs: LabelMe — only COCO-looking names
            hint = re.compile("|".join(COCO_NAME_HINTS), re.IGNORECASE)
            files = [p for p in files if hint.search(p.name)][:MAX_PROBES_PER_DIR]
        for p in files:
            k = os.path.normcase(os.path.abspath(p))
            if k not in seen and is_coco_file(p):
                seen.add(k)
                out.append(p)
    return out


def _split_of(p: Path) -> str:
    for word in (p.stem.lower(), p.parent.name.lower()):
        for w, sp in SPLIT_WORDS:
            if re.search(rf"(^|[_\-.]){w}(\d*|$)", word) or word.startswith(w):
                return sp
    return ""


def _load(p: Path) -> dict:
    try:
        import orjson
        return orjson.loads(p.read_bytes())
    except ImportError:
        return json.loads(p.read_text(encoding="utf-8"))


def _dump(d: dict) -> str:
    try:
        import orjson
        return orjson.dumps(d).decode("utf-8")
    except ImportError:
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


class CocoFile:
    def __init__(self, path: Path, root: Path):
        self.path = Path(path)
        self.data = _load(self.path)
        if not isinstance(self.data, dict):
            raise AnnotationError(f"{self.path.name} is not a COCO file")
        self.split = _split_of(self.path)
        self.cats = {int(c["id"]): str(c.get("name", c["id"])) for c in self.data.get("categories") or []}
        self.anns = defaultdict(list)                     # image id -> [ann dicts] (the file's own objects)
        for a in self.data.get("annotations") or []:
            self.anns[a.get("image_id")].append(a)
        self.images = {im.get("id"): im for im in self.data.get("images") or []}
        self.base = self._image_base(root)
        self.dirty = False
        self.next_ann = max((int(a.get("id", 0)) for a in self.data.get("annotations") or []), default=0) + 1

    def _image_base(self, root: Path) -> Path:
        """The folder the file names are relative to — the first that holds the images it names."""
        names = [im.get("file_name", "") for im in list(self.images.values())[:20] if im.get("file_name")]
        here, stem = self.path.parent, self.path.stem
        split_dirs = [w for w, _s in SPLIT_WORDS if w in stem.lower()]
        cands = [here, root, root / "images", here.parent / "images", here.parent]
        m = re.search(r"instances_(\w+)", stem)
        if m:
            split_dirs.insert(0, m.group(1))
        for w in split_dirs:
            cands += [root / w / "images", root / w, root / "images" / w, root / f"{w}2017", root / f"{w}2014"]
        for c in cands:
            if names and sum((c / n).is_file() for n in names) >= max(1, len(names) // 2):
                return c
        for c in cands:
            if names and any((c / Path(n).name).is_file() for n in names):
                return c
        return here

    def image_path(self, im: dict) -> Path:
        fn = str(im.get("file_name", ""))
        p = self.base / fn
        if p.is_file() or not fn:
            return p
        q = self.base / Path(fn).name
        return q if q.is_file() else p

    def write(self):
        self.data["annotations"] = [a for anns in self.anns.values() for a in anns]
        self.data["categories"] = [c for c in self.data.get("categories") or []] or \
            [{"id": k, "name": v, "supercategory": ""} for k, v in self.cats.items()]
        atomic_write_text(self.path, _dump(self.data))
        self.dirty = False


def _poly_area(pts) -> float:
    return abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                   for i in range(len(pts)))) / 2 if len(pts) >= 3 else 0.0


class CocoAdapter:
    type = DatasetType.COCO
    shared_ann = True                        # every sample of a file shares it (see the module doc)

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._img_size_cache: dict = {}
        self.files: dict[str, CocoFile] = {}
        self.by_key: dict[str, tuple[CocoFile, object]] = {}
        self._batch = 0
        self.notes: list[str] = []

    # ------------------------------------------------------------------ discover
    @property
    def class_names(self) -> list[str]:
        names = {}
        for f in self.files.values():
            for k, v in sorted(f.cats.items()):
                names.setdefault(v, k)
        return list(names)

    def discover(self) -> list[Sample]:
        out = []
        self.files.clear()
        self.by_key.clear()
        for p in find_coco_files(self.root):
            try:
                f = CocoFile(p, self.root)
            except Exception as exc:
                self.notes.append(f"{p.name}: {exc}")
                continue
            self.files[str(p)] = f
            for iid, im in f.images.items():
                img = f.image_path(im)
                key = str(img)
                if key in self.by_key:                    # the same image in two files: the first one wins
                    self.notes.append(f"{img.name} is in {Path(self.by_key[key][0].path).name} and {p.name}")
                    continue
                self.by_key[key] = (f, iid)
                try:
                    rel = img.relative_to(self.root)
                    folder = "/".join(rel.parts[:-1])
                except ValueError:
                    folder = ""
                out.append(Sample(key=key, name=img.name, ann_path=p, image_path=img, split=f.split, folder=folder))
        out.sort(key=self.order_key)
        return out

    def order_key(self, sample: Sample) -> tuple:
        return (sample.split or "", natural_key(str(sample.key)))

    def sample_for_path(self, path) -> tuple[Sample | None, bool]:
        p = Path(path)
        if p.suffix.lower() == ".json" and str(p) in self.files:
            return None, True                            # the COCO file changed outside: reload
        return None, False

    # ------------------------------------------------------------------ read
    def _entry(self, sample):
        hit = self.by_key.get(sample.key)
        if hit is None:
            raise AnnotationError(f"{sample.name} is not in a COCO file")
        return hit

    def load(self, sample: Sample, image_size=None, light: bool = False) -> Annotation:
        f, iid = self._entry(sample)
        im = f.images[iid]
        W, H = float(im.get("width") or 0), float(im.get("height") or 0)
        shapes = []
        for a in f.anns.get(iid, []):
            label = f.cats.get(int(a.get("category_id", -1)), str(a.get("category_id")))
            seg = a.get("segmentation")
            base = {"label": label, "group_id": None, "description": "", "flags": {}, "_ann_id": a.get("id"),
                    "_geom_dirty": False}
            if isinstance(a.get("provenance"), dict):
                base["provenance"] = a["provenance"]
            if isinstance(seg, list) and seg and isinstance(seg[0], list):
                parts = [s for s in seg if len(s) >= 6]
                for n, s in enumerate(parts):
                    pts = [[float(s[i]), float(s[i + 1])] for i in range(0, len(s) - 1, 2)]
                    shapes.append(dict(base, points=pts, shape_type="polygon", _part=n,
                                       group_id=a.get("id") if len(parts) > 1 else None))
                if parts:
                    continue
            if isinstance(seg, dict) and seg.get("counts") is not None and not light:
                try:
                    from .prep.importers import mask_polygons, rle_decode
                    h, w = (seg.get("size") or [H, W])[:2]
                    polys = mask_polygons(rle_decode(seg["counts"], int(h), int(w)))
                    for n, pts in enumerate(polys):
                        shapes.append(dict(base, points=pts, shape_type="polygon", _part=n, _rle=True,
                                           group_id=a.get("id") if len(polys) > 1 else None,
                                           description="mask (RLE): edit to turn it into a polygon"))
                    if polys:
                        continue
                except Exception:
                    pass
            bb = a.get("bbox")
            if bb and len(bb) == 4:
                x, y, w, h = map(float, bb)
                shapes.append(dict(base, points=[[x, y], [x + w, y + h]], shape_type="rectangle", _part=0))
        ann = Annotation(shapes, W or None, H or None, False, raw={"file": str(f.path), "image_id": iid},
                         image_path=f.image_path(im), light=light)
        if image_size and (not W or not H):
            ann.width, ann.height = image_size
        return ann

    def image_size(self, sample: Sample, ann: Annotation | None = None) -> tuple[int, int] | None:
        f, iid = self._entry(sample)
        im = f.images[iid]
        if im.get("width") and im.get("height"):
            return int(im["width"]), int(im["height"])
        if sample.key not in self._img_size_cache:
            self._img_size_cache[sample.key] = _img_size(sample.image_path)
        return self._img_size_cache[sample.key]

    def can_use_label(self, label: str) -> bool:
        return bool(str(label).strip())

    # ------------------------------------------------------------------ write
    def _cat_id(self, f: CocoFile, label: str) -> int:
        for k, v in f.cats.items():
            if v == label:
                return k
        k = max(f.cats, default=0) + 1
        f.cats[k] = label
        f.data.setdefault("categories", []).append({"id": k, "name": label, "supercategory": ""})
        return k

    def save(self, sample: Sample, ann: Annotation) -> None:
        f, iid = self._entry(sample)
        old = {a.get("id"): a for a in f.anns.get(iid, [])}
        groups: dict = {}
        order = []
        for sh in ann.shapes:
            aid = sh.get("_ann_id")
            k = aid if aid in old else ("new", id(sh))
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].append(sh)
        new_anns = []
        for k in order:
            shs = groups[k]
            label = str(shs[0].get("label", ""))
            cid = self._cat_id(f, label)
            orig = old.get(k) if not isinstance(k, tuple) else None
            touched = orig is None or any(s.get("_geom_dirty") for s in shs) or \
                len(shs) != len([1 for s in (orig.get("segmentation") or [[]]) if isinstance(s, list)]
                                if isinstance(orig.get("segmentation"), list) else shs)
            if orig is not None and not touched:
                a = dict(orig)
                if PV.enabled() and orig.get("category_id") != cid:
                    a[PV.KEY] = PV.bump(orig.get(PV.KEY), ["label"])
                a["category_id"] = cid                    # a relabel keeps everything else as written
                new_anns.append(a)
                continue
            a = copy.deepcopy(orig) if orig is not None else {"id": f.next_ann, "image_id": iid, "iscrowd": 0}
            if orig is None:
                f.next_ann += 1
            pts_all = [s["points"] for s in shs if s.get("points")]
            xs = [p[0] for pts in pts_all for p in pts]
            ys = [p[1] for pts in pts_all for p in pts]
            if not xs:
                continue
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            a["category_id"] = cid
            a["bbox"] = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
            polys = [pts for s, pts in zip(shs, pts_all) if s.get("shape_type") != "rectangle" and len(pts) >= 3]
            if polys:
                a["segmentation"] = [[round(v, 2) for p in pts for v in p] for pts in polys]
                a["area"] = round(sum(_poly_area(p) for p in polys), 2)
                a["iscrowd"] = 0
            else:
                a["segmentation"] = [] if not isinstance(a.get("segmentation"), dict) or shs[0].get("_geom_dirty") \
                    else a["segmentation"]
                a["area"] = round((x2 - x1) * (y2 - y1), 2)
            if PV.enabled():                              # the shape's record (a new object: made now)
                carried = next((s.get(PV.KEY) for s in shs if isinstance(s.get(PV.KEY), dict)), None)
                if orig is None:
                    a[PV.KEY] = carried if carried and carried.get("id") else PV.shape_record()
                else:
                    a[PV.KEY] = PV.bump(orig.get(PV.KEY), ["geometry"] + (["label"] if orig.get("category_id") != cid
                                                                          else []))
            new_anns.append(a)
        f.anns[iid] = new_anns
        f.dirty = True
        if not self._batch:
            f.write()

    @contextlib.contextmanager
    def batch(self):
        """Many saves, each file written once at the end."""
        self._batch += 1
        try:
            yield
        finally:
            self._batch -= 1
            if not self._batch:
                for f in self.files.values():
                    if f.dirty:
                        f.write()

    # ------------------------------------------------------------------ per image (undo, removal, rename)
    def unit_bytes(self, sample: Sample) -> bytes:
        """One image's annotations (and its image entry) as bytes — what Undo keeps for it."""
        f, iid = self._entry(sample)
        return json.dumps({"image": f.images.get(iid), "anns": f.anns.get(iid, [])}, sort_keys=True).encode()

    def unit_sig(self, sample: Sample):
        return hashlib.blake2b(self.unit_bytes(sample), digest_size=12).hexdigest()

    def write_unit(self, sample: Sample, data: bytes | None) -> None:
        f, iid = self._entry(sample)
        d = json.loads(data.decode()) if data else {"anns": []}
        f.anns[iid] = list(d.get("anns") or [])
        f.next_ann = max([f.next_ann] + [int(a.get("id", 0)) + 1 for a in f.anns[iid]])
        f.dirty = True
        if not self._batch:
            f.write()

    def remove_samples(self, samples) -> dict:
        """Take images out of their COCO files (entries + their objects). -> a record for put_back()."""
        rec = []
        for smp in samples:
            hit = self.by_key.pop(smp.key, None)
            if hit is None:
                continue
            f, iid = hit
            im = f.images.pop(iid, None)
            anns = f.anns.pop(iid, [])
            f.data["images"] = [x for x in f.data.get("images") or [] if x.get("id") != iid]
            f.dirty = True
            rec.append({"file": str(f.path), "key": smp.key, "image": im, "anns": anns})
        if not self._batch:
            for f in self.files.values():
                if f.dirty:
                    f.write()
        return {"coco": rec}

    def put_back(self, record) -> list[str]:
        keys = []
        for r in record.get("coco") or []:
            f = self.files.get(r["file"])
            if f is None or r["image"] is None:
                continue
            iid = r["image"].get("id")
            f.images[iid] = r["image"]
            f.data.setdefault("images", []).append(r["image"])
            f.anns[iid] = r["anns"]
            self.by_key[r["key"]] = (f, iid)
            f.dirty = True
            keys.append(r["key"])
        for f in self.files.values():
            if f.dirty:
                f.write()
        return keys

    def relocate(self, sample: Sample, new_image: Path) -> Sample:
        """The image file was renamed / moved: its file_name follows."""
        f, iid = self._entry(sample)
        im = f.images[iid]
        try:
            im["file_name"] = Path(new_image).relative_to(f.base).as_posix()
        except ValueError:
            im["file_name"] = str(new_image)
        self.by_key.pop(sample.key, None)
        key = str(new_image)
        self.by_key[key] = (f, iid)
        f.dirty = True
        if not self._batch:
            f.write()
        return Sample(key=key, name=Path(new_image).name, ann_path=sample.ann_path, image_path=Path(new_image),
                      split=sample.split, folder=sample.folder)

    def orphans(self, samples) -> dict[str, list[Path]]:
        listed = {os.path.normcase(str(s.image_path)) for s in samples}
        bases = {f.base for f in self.files.values()}
        imgs = []
        for b in bases:
            if not b.is_dir():
                continue
            for p in sorted(b.iterdir()):
                if p.suffix.lower() in IMAGE_EXTENSIONS and os.path.normcase(str(p)) not in listed:
                    imgs.append(p)
        missing = [Path(s.image_path) for s in samples if not Path(s.image_path).is_file()]
        return {"images_without_annotation": imgs, "annotations_without_image": missing}
