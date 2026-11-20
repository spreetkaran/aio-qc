# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image Augmentation — the app side of core.augment_ops (the engine): recipes on disk, and writing an augmented copy of
a dataset.

Recipes (plain JSON, app-wide): recipe_names() · load_recipe(name) · save_recipe(recipe) · delete_recipe(name) ·
import_file(path) (a recipe .json, or the older imgaug pipeline's aug_parameters.yaml).

Generating files — for trainers that need files, or to review the result in AIO QC (training with Train Model needs
none: the recipe runs in memory there):
    items = scan(samples, root, fmt)                       what can be augmented (paths relative to the dataset)
    copies = copies_per_item(items, base, …)               how many augmented copies of each image
    job = Job(root, out, fmt, recipe, …); run(job, items, copies, progress, cancel) -> report
Built for large datasets: a pool of processes, a bounded queue (memory stays flat at any size), every file written
atomically, a per-image seed (the same settings give the same files, on any number of workers), and resumable — what
is already there is skipped.
"""
from __future__ import annotations

import base64
import json
import math
import os
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import augment_ops as G
from .appdirs import data_home
from AIO_QC.core.diag import ignored as _ignored

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SUFFIX = "__aug"


# ====================================================================================================== recipes
def recipe_dir() -> Path:
    d = data_home() / "augment"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(name: str) -> str:
    s = "".join(c if c.isalnum() or c in " -_.()&" else "_" for c in str(name)).strip(" .")
    return s[:80] or "recipe"


def recipe_names() -> list[str]:
    out = []
    for p in sorted(recipe_dir().glob("*.json"), key=lambda p: p.name.lower()):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")).get("name") or p.stem)
        except (OSError, ValueError):
            pass
    return out


def _recipe_path(name: str) -> Path:
    return recipe_dir() / f"{_slug(name)}.json"


def load_recipe(name: str) -> dict | None:
    """A saved recipe, or a built-in preset of that name."""
    p = _recipe_path(name)
    if p.is_file():
        try:
            return G.clean(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None
    pre = G.presets().get(name)
    return G.clean(pre[0]) if pre else None


def save_recipe(recipe: dict) -> Path:
    r = G.clean(recipe)
    p = _recipe_path(r["name"])
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
    return p


def delete_recipe(name: str) -> bool:
    try:
        _recipe_path(name).unlink()
        return True
    except OSError:
        return False


def all_names() -> list[str]:
    """Saved recipes first, then the built-in presets not shadowed by one."""
    saved = recipe_names()
    return saved + [n for n in G.presets() if n not in saved]


def import_file(path) -> dict:
    """A recipe .json of ours, or the older pipeline's aug_parameters.yaml (its four apply_* sections)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    data = None
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
            data = yaml.safe_load(text)
        except ImportError:
            from .testing.dataset import load_yaml
            data = load_yaml(p)
    if not isinstance(data, dict):
        raise ValueError("not a recipe (a JSON / YAML mapping is expected)")
    if "ops" in data:
        r = G.clean(data)
    elif any(k.startswith("apply_") for k in data):
        r = G.from_legacy(data)
        r["name"] = p.stem
    else:
        raise ValueError("neither an AIO QC recipe nor an aug_parameters.yaml of the older pipeline")
    return r


# ====================================================================================================== what to augment
@dataclass
class Item:
    ann: str            # label file relative to the dataset ("" when the image has none)
    img: str            # image relative to the dataset ("" for LabelMe: found from the JSON)
    split: str = ""
    key: str = ""


def scan(samples, root: Path, fmt: str) -> tuple[list[Item], int]:
    """-> (items, files outside the dataset folder — left out)."""
    root = Path(os.path.abspath(root))
    items, outside = [], 0
    for s in samples:
        try:
            ann = os.path.relpath(s.ann_path, root) if s.ann_path is not None else ""
            img = os.path.relpath(s.image_path, root) if s.image_path is not None else ""
        except ValueError:                                 # another drive
            outside += 1
            continue
        if ann.startswith("..") or img.startswith(".."):
            outside += 1
            continue
        if fmt != "labelme" and not img:
            continue
        items.append(Item(ann.replace("\\", "/"), img.replace("\\", "/"), s.split or "", s.key))
    return items, outside


def copies_per_item(items, base: int, only_keys: set | None = None, splits: set | None = None,
                    class_keys: dict | None = None, balance: bool = False, cap: int = 5) -> list[int]:
    """How many augmented copies each image gets. `splits`: only these (others: 0); `only_keys`: only these images;
    `balance` with class_keys {class: keys}: images holding a rare class get more — up to `cap` × — so the classes
    come closer (factor of an image = the largest of its classes' most-common-count / own-count)."""
    factor: dict = {}
    if balance and class_keys:
        counts = {c: len(k) for c, k in class_keys.items() if k}
        top = max(counts.values()) if counts else 0
        for c, keys in class_keys.items():
            if not keys:
                continue
            f = min(float(cap), top / counts[c])
            for k in keys:
                if f > factor.get(k, 1.0):
                    factor[k] = f
    out = []
    for it in items:
        if (splits is not None and it.split and it.split not in splits) or \
                (only_keys is not None and it.key not in only_keys):
            out.append(0)
        else:
            out.append(int(round(base * factor.get(it.key, 1.0))))
    return out


@dataclass
class Job:
    root: str
    out: str
    fmt: str                              # "yolo" | "labelme" | "images" (class folders: no label files)
    recipe: dict
    seed: int = 0
    originals: bool = True                # the untouched images (and labels) into the output too
    link: str = "hardlink"                # originals: hardlink when the drive allows, else copy
    obb: bool = False                     # YOLO: 4-point rows are oriented boxes (stay 4 points)
    quality: int = 92                     # JPEG quality of the written images
    max_side: int = 0                     # resize the longer side to this before augmenting (0: as it is)
    workers: int = 0                      # 0: one per CPU (minus one)
    names: list = field(default_factory=list)
    reference: bool = False               # YOLO, originals not copied: dataset.yaml lists their folders as well
    paste: dict | None = None             # objects cut from other images pasted in first (see PASTE)


# Objects cut out of other images of the dataset and pasted in before the effects run (the older pipeline's "paste
# crop on image"): more instances per image, rare classes in new places. Only where there are files (not in memory
# while training — Ultralytics has its own copy-paste for segmentation).
PASTE = {"p": 0.5,                # chance an image gets pasted objects
         "count": [1, 3],         # how many
         "scale": [0.7, 1.2],     # their size, relative to the size they had in their own image
         "classes": None,         # only these classes (None: any)
         "overlap": 0.0,          # how much a pasted object may cover an existing one (share of the smaller box)
         "boxes": False,          # boxes too: a rectangular cut-out with its background (polygons are cut exactly)
         "feather": 1.5}          # soft edge, px


def clean_paste(d: dict | None) -> dict | None:
    if not d:
        return None
    out = dict(PASTE)
    try:
        out["p"] = min(max(float(d.get("p", out["p"])), 0.0), 1.0)
        a, b = d.get("count", out["count"])
        out["count"] = [max(1, min(int(a), int(b))), max(1, min(20, max(int(a), int(b))))]
        a, b = d.get("scale", out["scale"])
        out["scale"] = [max(0.05, min(float(a), float(b))), min(5.0, max(float(a), float(b)))]
        out["overlap"] = min(max(float(d.get("overlap", 0.0)), 0.0), 1.0)
        out["feather"] = min(max(float(d.get("feather", 1.5)), 0.0), 20.0)
    except (TypeError, ValueError):
        pass
    out["boxes"] = bool(d.get("boxes", False))
    cl = d.get("classes")
    out["classes"] = [str(c) for c in cl] if cl else None
    return out if out["p"] > 0 else None


def is_obb(root: Path, items, probe: int = 300) -> bool:
    """YOLO: do the label files hold oriented boxes (every polygon row has exactly 4 points)?"""
    seen = 0
    for it in items[:: max(1, len(items) // probe)][:probe]:
        if not it.ann:
            continue
        try:
            for ln in (Path(root) / it.ann).read_text(encoding="utf-8", errors="replace").splitlines():
                n = len(ln.split()) - 1
                if n == 4 or n <= 0:
                    continue
                if n != 8:
                    return False
                seen += 1
        except OSError:
            pass
    return seen > 0


def estimate_bytes(root: Path, items, copies, originals: bool, link: str, probe: int = 200) -> int:
    """Roughly what the output needs on disk (image sizes sampled)."""
    idx = [i for i, it in enumerate(items) if it.img or it.ann]
    if not idx:
        return 0
    sizes = []
    for i in idx[:: max(1, len(idx) // probe)][:probe]:
        it = items[i]
        p = Path(root) / (it.img or it.ann)
        try:
            sizes.append(p.stat().st_size if it.img else max(p.stat().st_size // 8, 200_000))
        except OSError:
            pass
    avg = (sum(sizes) / len(sizes)) if sizes else 300_000
    n = sum(copies) + (len(items) if originals and link != "hardlink" else 0)
    return int(n * avg * 1.15)


# ====================================================================================================== workers
_CFG: dict = {}
_AUG = None


def _init(cfg: dict, pooled: bool = False):
    global _CFG, _AUG
    _CFG = cfg
    _DONORS.clear()
    if pooled:                                             # a pool process: ends with the app, low priority
        try:
            from . import procs
            procs.worker_init(threads=1, background=True)
        except Exception as _exc:
            _ignored(_exc)
    _AUG = G.Augmenter(cfg["recipe"], keep_size=False)
    try:
        import cv2
        cv2.setNumThreads(1)                               # the pool is the parallelism
        cv2.ocl.setUseOpenCL(False)
    except Exception as _exc:
        _ignored(_exc)


def _imread(path: Path):
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)           # (non-ASCII paths on Windows)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if len(data) else None


def _imwrite(path: Path, img, quality: int) -> bool:
    import cv2
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext in (".jpg", ".jpeg") else \
        ([cv2.IMWRITE_WEBP_QUALITY, int(quality)] if ext == ".webp" else [])
    if ext in (".jpg", ".jpeg") and img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    _write_bytes(path, buf.tobytes())
    return True


def _write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _aug_name(rel: str, k: int, ext: str | None = None) -> str:
    p = Path(rel)
    return str(p.with_name(f"{p.stem}{SUFFIX}{k}{ext or p.suffix}")).replace("\\", "/")


def _yolo_rows(text: str, w: int, h: int, obb: bool):
    """-> (shapes in pixels, rows that cannot be augmented)."""
    shapes, odd = [], 0
    for ln in text.splitlines():
        parts = ln.split()
        if not parts:
            continue
        try:
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            odd += 1
            continue
        n = len(vals)
        if n == 4:
            cx, cy, bw, bh = vals
            shapes.append({"cls": parts[0], "shape_type": "rectangle",
                           "points": [[(cx - bw / 2) * w, (cy - bh / 2) * h], [(cx + bw / 2) * w, (cy + bh / 2) * h]]})
        elif n >= 6 and n % 2 == 0:
            shapes.append({"cls": parts[0], "shape_type": "obb" if (obb and n == 8) else "polygon",
                           "points": [[vals[i] * w, vals[i + 1] * h] for i in range(0, n, 2)]})
        else:
            odd += 1
    return shapes, odd


def _yolo_text(shapes, w: int, h: int) -> str:
    def f(v):
        return f"{min(max(v, 0.0), 1.0):.6f}".rstrip("0").rstrip(".") or "0"
    lines = []
    for s in shapes:
        P = s["points"]
        if s["shape_type"] == "rectangle":
            (x1, y1), (x2, y2) = P[0], P[1]
            vals = [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, abs(x2 - x1) / w, abs(y2 - y1) / h]
        else:
            vals = [v for x, y in P for v in (x / w, y / h)]
        lines.append(" ".join([s["cls"]] + [f(v) for v in vals]))
    return "\n".join(lines) + ("\n" if lines else "")


def _labelme_image(folder: Path, ann: Path, data: dict):
    """(image array, the image's file name) of a LabelMe file: its imagePath, a file of the same name, or the
    picture embedded in the JSON."""
    cands = []
    ip = data.get("imagePath")
    if ip:
        ip = str(ip).replace("\\", "/")
        cands += [folder / ip, folder / Path(ip).name]
    cands += [ann.with_suffix(e) for e in IMAGE_EXTS]
    for c in cands:
        if c.is_file():
            return _imread(c), c.name
    if data.get("imageData"):
        import cv2
        raw = np.frombuffer(base64.b64decode(data["imageData"]), np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED), (Path(str(ip)).name if ip else ann.stem + ".jpg")
    return None, None


def _resize(img, max_side: int):
    import cv2
    h, w = img.shape[:2]
    if max_side and max(w, h) > max_side:
        f = max_side / float(max(w, h))
        return cv2.resize(img, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA), f
    return img, 1.0


def load_item(root: Path, fmt: str, ann_rel: str, img_rel: str, obb: bool = False, max_side: int = 0):
    """One image of a dataset with its shapes in pixels.
    -> (image or None, shapes, LabelMe data or None, the image's file name, YOLO rows that are not shapes)."""
    root = Path(root)
    ann_p = (root / ann_rel) if ann_rel else None
    data, odd = None, 0
    if fmt == "labelme":
        data = json.loads(ann_p.read_text(encoding="utf-8"))
        img, img_name = _labelme_image(ann_p.parent, ann_p, data)
    else:
        img, img_name = _imread(root / img_rel), Path(img_rel).name
    if img is None:
        return None, [], data, img_name, 0
    img, f = _resize(img, int(max_side))
    h, w = img.shape[:2]
    shapes = []
    if fmt == "labelme":
        for s in data.get("shapes") or []:
            s2 = dict(s)
            s2["points"] = [[float(p[0]) * f, float(p[1]) * f] for p in (s.get("points") or [])]
            shapes.append(s2)
    elif fmt == "yolo" and ann_p is not None and ann_p.is_file():
        shapes, odd = _yolo_rows(ann_p.read_text(encoding="utf-8", errors="replace"), w, h, obb)
    return img, shapes, data, img_name, odd


def _class_of(s: dict) -> str:
    return str(s.get("label") if s.get("label") is not None else s.get("cls", ""))


def _bbox_of(s: dict):
    P = np.asarray(s.get("points") or [], np.float64).reshape(-1, 2)
    if len(P) < 2:
        return None
    if (s.get("shape_type") or "polygon") == "circle":
        r = float(np.hypot(*(P[1] - P[0])))
        return P[0][0] - r, P[0][1] - r, P[0][0] + r, P[0][1] + r
    return float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max())


def _to_channels(patch, like):
    """A cut-out in the colour layout of the image it goes into (grey / BGR / BGRA)."""
    import cv2
    if patch.ndim == 3 and patch.shape[2] == 4:
        patch = patch[..., :3]
    if like.ndim == 2:
        return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    return patch


def paste_objects(img, shapes, cfg: dict, rng, donor, n_donors: int, names: dict | None = None):
    """Objects cut out of other images, pasted into `img` (a copy is returned) where they do not cover what is
    labelled already. donor(i) -> (image, shapes) of the i-th possible source (or None). -> (image, shapes, pasted)."""
    import cv2
    cfg = clean_paste(cfg)
    if cfg is None or n_donors <= 0 or img is None or rng.random() >= cfg["p"]:
        return img, shapes, 0
    H, W = img.shape[:2]
    want = set(cfg["classes"]) if cfg["classes"] else None
    kinds = ("polygon", "rectangle", "obb") if cfg["boxes"] else ("polygon",)
    out = img.copy()
    shapes = [dict(s) for s in shapes]
    boxes = [b for b in (_bbox_of(s) for s in shapes) if b is not None]
    n = int(rng.integers(cfg["count"][0], cfg["count"][1] + 1))
    done = 0
    for _ in range(n):
        placed = False
        for _try in range(6):                              # a few sources until one has a usable object
            got = donor(int(rng.integers(0, n_donors)))
            if got is None or got[0] is None:
                continue
            d_img, d_shapes = got
            dh, dw = d_img.shape[:2]
            cands = []
            for s in d_shapes:
                st = s.get("shape_type") or "polygon"
                c = _class_of(s)
                label = (names or {}).get(c, c)
                if st not in kinds or (want is not None and c not in want and label not in want):
                    continue
                P = np.asarray(s.get("points") or [], np.float64).reshape(-1, 2)
                if st == "rectangle" and len(P) >= 2:
                    x1, y1, x2, y2 = _bbox_of(s)
                    P = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64)
                if len(P) < 3:
                    continue
                P[:, 0] = np.clip(P[:, 0], 0, dw)
                P[:, 1] = np.clip(P[:, 1], 0, dh)
                if np.ptp(P[:, 0]) >= 12 and np.ptp(P[:, 1]) >= 12:
                    cands.append((s, st, P))
            if not cands:
                continue
            s, st, P = cands[int(rng.integers(0, len(cands)))]
            x1, y1 = int(math.floor(P[:, 0].min())), int(math.floor(P[:, 1].min()))
            x2, y2 = int(math.ceil(P[:, 0].max())), int(math.ceil(P[:, 1].max()))
            crop = d_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            mask = np.zeros(crop.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.round(P - [x1, y1]).astype(np.int32)], 255)
            # the size it had in its own image (relative), times the chosen factor; never larger than 80 % of this one
            f = float(rng.uniform(*cfg["scale"])) * (H / float(dh))
            f = min(f, 0.8 * W / crop.shape[1], 0.8 * H / crop.shape[0])
            cw, ch = int(round(crop.shape[1] * f)), int(round(crop.shape[0] * f))
            if cw < 8 or ch < 8:
                continue
            interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
            crop = _to_channels(cv2.resize(crop, (cw, ch), interpolation=interp), out)
            mask = cv2.resize(mask, (cw, ch), interpolation=interp)
            for _pos in range(25):                         # a free place
                px, py = int(rng.integers(0, W - cw + 1)), int(rng.integers(0, H - ch + 1))
                nb = (px, py, px + cw, py + ch)
                ok = True
                for b in boxes:
                    iw, ih = min(nb[2], b[2]) - max(nb[0], b[0]), min(nb[3], b[3]) - max(nb[1], b[1])
                    if iw > 0 and ih > 0:
                        small = min(cw * ch, max((b[2] - b[0]) * (b[3] - b[1]), 1e-6))
                        if iw * ih / small > cfg["overlap"]:
                            ok = False
                            break
                if not ok:
                    continue
                a = mask.astype(np.float32) / 255.0
                if cfg["feather"] > 0:
                    a = np.minimum(a, cv2.GaussianBlur(a, (0, 0), cfg["feather"]) * 1.0)   # soft, never grows
                region = out[py:py + ch, px:px + cw]
                tgt = region[..., :3] if (region.ndim == 3 and region.shape[2] == 4) else region
                aa = a if tgt.ndim == 2 else a[..., None]
                tgt[:] = np.clip(crop.astype(np.float32) * aa + tgt.astype(np.float32) * (1 - aa), 0, 255
                                 ).astype(np.uint8)
                s2 = dict(s)
                Q = (P - [x1, y1]) * f + [px, py]
                if st == "rectangle":
                    s2["points"] = [[float(px), float(py)], [float(px + cw), float(py + ch)]]
                else:
                    s2["points"] = [[float(x), float(y)] for x, y in Q]
                shapes.append(s2)
                boxes.append(nb)
                placed = True
                done += 1
                break
            if placed:
                break
    return (out if done else img), shapes, done


_DONORS: dict = {}


def _donor(i: int):
    """The i-th source image of this run for pasting (a few kept in memory per worker)."""
    cfg = _CFG
    if i in _DONORS:
        return _DONORS[i]
    ann_rel, img_rel = cfg["donors"][i]
    try:
        img, shapes, _d, _n, _o = load_item(Path(cfg["root"]), cfg["fmt"], ann_rel, img_rel, cfg["obb"],
                                            int(cfg["max_side"]))
    except Exception as _exc:
        _ignored(_exc)
        img, shapes = None, []
    if len(_DONORS) >= 24:
        _DONORS.pop(next(iter(_DONORS)))
    _DONORS[i] = (img, shapes) if img is not None else None
    return _DONORS[i]


def _work(batch) -> dict:
    """[(ann, img, copies)] -> counters. Never raises: a bad file is one error line."""
    cfg, aug = _CFG, _AUG
    root, out, fmt = Path(cfg["root"]), Path(cfg["out"]), cfg["fmt"]
    res = {"written": 0, "skipped": 0, "shapes_in": 0, "shapes_out": 0, "errors": [], "odd_rows": 0, "images": 0,
           "pasted": 0}
    for ann_rel, img_rel, n in batch:
        try:
            res["images"] += 1
            out_rel_img_dir = Path(ann_rel).parent if fmt == "labelme" else Path(img_rel).parent
            # resume: what is already there (image, and its label when there is one) is not made again
            todo = []
            for k in range(1, n + 1):
                if fmt == "labelme":
                    done = (out / _aug_name(ann_rel, k)).is_file()
                elif fmt == "yolo" and ann_rel:
                    done = (out / _aug_name(ann_rel, k)).is_file() and (out / _aug_name(img_rel, k)).is_file()
                else:
                    done = (out / _aug_name(img_rel, k)).is_file()
                if done:
                    res["skipped"] += 1
                else:
                    todo.append(k)
            if not todo:
                continue
            img, shapes, data, img_name, odd = load_item(root, fmt, ann_rel, img_rel, cfg["obb"], cfg["max_side"])
            res["odd_rows"] += odd
            if img is None:
                res["errors"].append(f"{ann_rel or img_rel}: the image cannot be read")
                continue
            ext = Path(img_name).suffix.lower()
            ext = ext if ext in IMAGE_EXTS else ".jpg"
            base_seed = zlib.crc32((ann_rel or img_rel).encode("utf-8")) ^ (int(cfg["seed"]) * 1_000_003 & 0xFFFFFFFF)
            paste, donors = cfg.get("paste"), cfg.get("donors") or []
            for k in todo:
                seed_k = (base_seed + k * 7919) & 0xFFFFFFFF
                src_img, src_shapes = img, shapes
                if paste and donors:                       # objects from other images first: the effects cover both
                    src_img, src_shapes, n_p = paste_objects(img, shapes, paste, np.random.default_rng(seed_k ^ 0x5EED),
                                                             _donor, len(donors))
                    res["pasted"] += n_p
                o_img, o_shapes, _log = aug.apply(src_img, src_shapes, seed=seed_k)
                oh, ow = o_img.shape[:2]
                res["shapes_in"] += len(src_shapes)
                res["shapes_out"] += len(o_shapes)
                name_k = f"{Path(img_name).stem}{SUFFIX}{k}{ext}"
                if not _imwrite(out / out_rel_img_dir / name_k, o_img, cfg["quality"]):
                    res["errors"].append(f"{ann_rel or img_rel}: the image could not be written as {ext}")
                    continue
                from . import provenance as PV
                aug_o = PV.make_origin("augmentation", "Image Augmentation", timeless=True)   # same seed: same bytes
                PV.embed_file(out / out_rel_img_dir / name_k, PV.image_record(aug_o, source=Path(img_name).name,
                                                                              copy=k, seed=seed_k))
                if fmt == "labelme":
                    d2 = dict(data)
                    if PV.enabled():                       # each object: one more change on its record
                        o_shapes = [dict(sh, **{PV.KEY: PV.bump(sh.get(PV.KEY), ["geometry"], aug_o)})
                                    if isinstance(sh, dict) else sh for sh in o_shapes]
                        PV.derive_labelme(d2, data.get(PV.KEY) if isinstance(data, dict) else None, aug_o,
                                          source=Path(img_name).name, copy=k)
                    d2["shapes"] = o_shapes
                    d2["imagePath"], d2["imageData"] = name_k, None
                    d2["imageHeight"], d2["imageWidth"] = int(oh), int(ow)
                    _write_bytes(out / _aug_name(ann_rel, k),
                                 json.dumps(d2, indent=2, ensure_ascii=False).encode("utf-8"))
                elif fmt == "yolo" and ann_rel:            # (an image without a label file stays without one)
                    _write_bytes(out / _aug_name(ann_rel, k), _yolo_text(o_shapes, ow, oh).encode("utf-8"))
                res["written"] += 1
        except Exception as exc:
            _ignored(exc)
            res["errors"].append(f"{ann_rel or img_rel}: {type(exc).__name__}: {exc}")
    return res


# ====================================================================================================== the run
def _copy_originals(job: Job, items, cancel) -> int:
    from .export import link_or_copy
    root, out, n = Path(job.root), Path(job.out), 0
    for it in items:
        if cancel is not None and cancel.is_set():
            break
        rels = [it.ann, it.img]
        if job.fmt == "labelme" and it.ann and not it.img:  # the image a JSON points to
            try:
                d = json.loads((root / it.ann).read_text(encoding="utf-8"))
                ip = str(d.get("imagePath") or "").replace("\\", "/")
                for c in ([Path(it.ann).parent / ip, Path(it.ann).parent / Path(ip).name] if ip else []) + \
                        [Path(it.ann).with_suffix(e) for e in IMAGE_EXTS]:
                    if (root / c).is_file():
                        rels.append(str(c))
                        break
            except (OSError, ValueError):
                pass
        for rel in rels:
            if rel and (root / rel).is_file() and not (out / rel).exists():
                try:
                    link_or_copy(root / rel, out / rel, job.link)
                    n += 1
                except OSError:
                    pass
    return n


def _yaml_data(p: Path) -> dict | None:
    try:
        try:
            import yaml
            d = yaml.safe_load(p.read_text(encoding="utf-8", errors="replace"))
        except ImportError:
            from .testing.dataset import load_yaml
            d = load_yaml(p)
        return d if isinstance(d, dict) else None
    except Exception as _exc:
        _ignored(_exc)
        return None


def _referencing_yaml(job: Job, p: Path, aug_splits: set) -> str | None:
    """A dataset.yaml for an output that does not hold the originals: every split points at the original folders
    (absolute), and the augmented splits list the new folder as well — or only the new one when the originals are
    left out. Nothing is copied, and val / test stay the real images."""
    d = _yaml_data(p)
    if not d or not any(k in d for k in ("train", "val", "test")):
        return None
    root, out = Path(job.root), Path(job.out)
    base = Path(str(d.get("path") or "."))
    if not base.is_absolute():
        base = (p.parent / base)
    if not base.exists():
        base = p.parent
    q = lambda v: json.dumps(str(v).replace("\\", "/"), ensure_ascii=False)     # noqa: E731
    lines = [f"# made by AIO QC Image Augmentation from {p.name}: absolute paths (the originals stay where they are)",
             f"path: {q(out)}"]
    for split in ("train", "val", "test"):
        v = d.get(split)
        if not v:
            continue
        entries = []
        for e in (v if isinstance(v, (list, tuple)) else [v]):
            ep = Path(str(e))
            ep = ep if ep.is_absolute() else (base / ep)
            ep = Path(os.path.abspath(ep))
            mine = None
            try:
                mine = out / ep.relative_to(Path(os.path.abspath(root)))
            except ValueError:
                pass
            if split in aug_splits and mine is not None and mine.exists():
                if job.reference:
                    entries.append(ep)
                entries.append(mine)
            else:
                entries.append(ep)
        lines.append(f"{split}: {q(entries[0])}" if len(entries) == 1 else
                     f"{split}:\n" + "\n".join(f"  - {q(e)}" for e in entries))
    names = d.get("names")
    if isinstance(names, dict):
        lines.append("names:")
        lines += [f"  {int(k) if str(k).isdigit() else json.dumps(str(k))}: {json.dumps(str(v), ensure_ascii=False)}"
                  for k, v in names.items()]
    elif isinstance(names, (list, tuple)):
        lines.append("names:")
        lines += [f"  {i}: {json.dumps(str(v), ensure_ascii=False)}" for i, v in enumerate(names)]
    if "nc" in d:
        lines.append(f"nc: {d['nc']}")
    for k in ("kpt_shape", "flip_idx"):
        if k in d:
            lines.append(f"{k}: {json.dumps(d[k])}")
    return "\n".join(lines) + "\n"


def _copy_meta(job: Job, aug_splits: set | None = None):
    """dataset.yaml / classes.txt of a YOLO dataset: with the originals in the output, the same file with `path:`
    pointing at the new folder; without them, one that refers to the original folders (see _referencing_yaml)."""
    root, out = Path(job.root), Path(job.out)
    for p in list(root.glob("*.yaml")) + list(root.glob("*.yml")) + list(root.glob("classes.txt")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            if p.suffix in (".yaml", ".yml") and not job.originals:
                ref = _referencing_yaml(job, p, aug_splits or {"train"})
                if ref is not None:
                    (out / p.name).write_text(ref, encoding="utf-8")
                    continue
            if p.suffix in (".yaml", ".yml"):
                lines = text.splitlines()
                if any(ln.startswith("path:") for ln in lines):
                    lines = [f"path: {json.dumps(str(out))}" if ln.startswith("path:") else ln for ln in lines]
                else:
                    lines.insert(0, f"path: {json.dumps(str(out))}")
                text = "\n".join(lines) + "\n"
            (out / p.name).write_text(text, encoding="utf-8")
        except OSError:
            pass


def run(job: Job, items, copies, progress=None, cancel=None) -> dict:
    """Write the augmented dataset. progress(done images, total, report so far); cancel: threading.Event.
    -> report {written, skipped, shapes_in, shapes_out, errors, originals, seconds, cancelled, out}."""
    t0 = time.time()
    out = Path(job.out)
    out.mkdir(parents=True, exist_ok=True)
    paste = clean_paste(job.paste) if job.fmt in ("yolo", "labelme") else None
    donors = []
    if paste:                                              # sources for pasting: a spread sample of the dataset
        pool = [(it.ann, it.img) for it in items if it.ann]
        donors = pool[:: max(1, len(pool) // 20000)][:20000]
    cfg = {"root": job.root, "out": job.out, "fmt": job.fmt, "recipe": job.recipe, "seed": job.seed,
           "obb": job.obb, "quality": job.quality, "max_side": job.max_side, "paste": paste, "donors": donors}
    todo = [(it.ann, it.img, n) for it, n in zip(items, copies) if n > 0]
    aug_splits = {it.split for it, n in zip(items, copies) if n > 0 and it.split}
    total = len(todo)
    rep = {"written": 0, "skipped": 0, "shapes_in": 0, "shapes_out": 0, "errors": [], "odd_rows": 0, "images": 0,
           "pasted": 0, "originals": 0, "cancelled": False, "out": str(out), "total": total}
    workers = job.workers or max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, min(workers, 61, total or 1))          # (Windows: at most 61 handles per wait)
    chunk = 8 if total > 2000 else max(1, min(8, total // (workers * 4) or 1))
    batches = (todo[i:i + chunk] for i in range(0, total, chunk))

    def merge(r):
        for k in ("written", "skipped", "shapes_in", "shapes_out", "odd_rows", "images", "pasted"):
            rep[k] += r[k]
        if len(rep["errors"]) < 200:
            rep["errors"].extend(r["errors"][:200 - len(rep["errors"])])
        rep["n_errors"] = rep.get("n_errors", 0) + len(r["errors"])
        if progress is not None:
            progress(rep["images"], total, rep)

    if total:
        if workers == 1:
            _init(cfg)
            for b in batches:
                if cancel is not None and cancel.is_set():
                    rep["cancelled"] = True
                    break
                merge(_work(b))
        else:
            import multiprocessing as mp
            from . import procs
            # fresh processes ("spawn", as on Windows): a forked copy of a process whose OpenCV has already used
            # its threads (the preview has) can hang in its first OpenCV call
            ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     initializer=_init, initargs=(cfg, True))
            pending = set()
            try:
                for b in batches:
                    if cancel is not None and cancel.is_set():
                        rep["cancelled"] = True
                        break
                    pending.add(ex.submit(_work, b))
                    if len(pending) >= workers * 3:          # bounded: memory does not grow with the dataset
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for f in done:
                            merge(f.result())
                while pending and not rep["cancelled"]:
                    done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    for f in done:
                        merge(f.result())
                    if cancel is not None and cancel.is_set():
                        rep["cancelled"] = True
            finally:
                if rep["cancelled"]:
                    for f in pending:
                        f.cancel()
                    procs.terminate_pool(ex, grace=2.0)
                else:
                    ex.shutdown(wait=True)
    if not rep["cancelled"]:
        if job.originals:
            rep["originals"] = _copy_originals(job, items, cancel)
        if job.fmt == "yolo":
            _copy_meta(job, aug_splits)
    rep["seconds"] = round(time.time() - t0, 1)
    try:                                                   # what made these files (to repeat or explain them)
        (out / ".aio_qc").mkdir(exist_ok=True)              # (not in the dataset itself: it is not a label file)
        (out / ".aio_qc" / "augmentation.json").write_text(json.dumps(
            {"recipe": G.clean(job.recipe), "job": {k: v for k, v in asdict(job).items() if k != "recipe"},
             "report": {k: v for k, v in rep.items() if k != "errors"}, "errors": rep["errors"][:200],
             "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    from . import provenance as PV
    PV.record(out, "augment", "Image Augmentation", inputs=[job.root], settings={"recipe": G.clean(job.recipe),
              "copies": copies, "seed": job.seed, "format": job.fmt},
              counts={k: rep[k] for k in ("written", "images", "shapes_in", "shapes_out", "pasted", "originals")},
              o=PV.make_origin("augmentation", "Image Augmentation"), cancelled=rep["cancelled"] or None)
    return rep
