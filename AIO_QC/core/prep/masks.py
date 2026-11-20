# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Segmentation masks (Qt-free): PNG masks ⇄ polygons.

Reading (Import annotations ▸ Segmentation masks): one mask per image, matched by name (img001.jpg ↔ img001.png,
also img001_mask.png / img001_gtFine_labelIds.png …). What a pixel value means is found by itself and can be set:

    class index         0 = background, 1 … n = classes  (Pascal VOC SegmentationClass, ADE20K, most tools);
                        255 = "void / border" is ignored
    colour              RGB masks: a colour per class (CamVid, CVAT / Roboflow exports) — from a labelmap.txt
                        ("name:r,g,b::"), else each colour found becomes a class
    instance            every distinct value is one object (VOC SegmentationObject, a *_instance.png); its class from
                        the matching class mask when there is one (or Cityscapes ids: class * 1000 + instance)
Each region becomes a polygon (holes are outlined with the region; tiny specks below `min_area` px are dropped).

Writing (Convert & Split ▸ Masks): <split>/masks/<image>.png (class index, palette PNG — 0 = background,
class i = i + 1, in your class order) and <split>/instances/<image>.png (16-bit: object n = n), labelmap.txt and
classes.txt; polygons, boxes, circles and rotated boxes are filled; overlapping objects: the later one on top.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

VOID = 255
SUFFIXES = ("", "_mask", "_masks", "_label", "_labels", "_seg", "_gtFine_labelIds", "_gtFine_instanceIds",
            "_instance", "_instanceIds", "_labelIds", "-mask")
PALETTE_VOC = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
               (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
               (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]


def palette(n: int) -> list[tuple[int, int, int]]:
    """The VOC colour map (bit-interleaved), as long as needed."""
    out = []
    for i in range(max(n, 1)):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        out.append((r, g, b))
    return out


def read_labelmap(path) -> tuple[list[str], dict]:
    """labelmap.txt ("name:r,g,b:parts:actions", CVAT / Roboflow), classes.txt (a name per line) or a YOLO yaml.
    -> (names in order, {(r, g, b): name})."""
    if not path or not os.path.isfile(path):
        return [], {}
    p = str(path)
    if p.lower().endswith((".yaml", ".yml")):
        from AIO_QC.core.class_map import parse_yaml_class_map
        return list(parse_yaml_class_map(Path(p)).names), {}
    names, colors = [], {}
    for ln in Path(p).read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ":" in ln:
            name, rest = ln.split(":", 1)
            rgb = rest.split(":", 1)[0]
            try:
                c = tuple(int(v) for v in rgb.split(","))
                if len(c) == 3:
                    colors[c] = name.strip()
            except ValueError:
                pass
            names.append(name.strip())
        else:
            names.append(ln)
    return names, colors


def _read(path):
    import cv2
    import numpy as np
    data = np.fromfile(str(path), np.uint8)
    m = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None, False
    if m.ndim == 3:
        if m.shape[2] == 4:
            m = m[:, :, :3]
        if (m[:, :, 0] == m[:, :, 1]).all() and (m[:, :, 1] == m[:, :, 2]).all():
            return m[:, :, 0], False
        return m[:, :, ::-1].copy(), True                     # BGR -> RGB
    # a palette PNG decodes to its colours in OpenCV only when IMREAD_COLOR: read the indices with PIL if present
    return m, False


def _read_indexed(path):
    """Palette PNGs (VOC): the INDEX of each pixel (not its colour)."""
    try:
        from PIL import Image
        import numpy as np
        im = Image.open(path)
        if im.mode == "P":
            return np.asarray(im), False
    except Exception:
        pass
    return _read(path)


def find_mask(masks_dir: Path, rel_img: str, index: dict) -> Path | None:
    stem = os.path.splitext(rel_img)[0]
    for sfx in SUFFIXES:
        for cand in (stem + sfx, os.path.basename(stem) + sfx):
            hit = index.get(cand.replace("\\", "/").lower())
            if hit:
                return hit
    return None


def index_masks(masks_dir) -> dict:
    out = {}
    for dp, _dn, fns in os.walk(masks_dir):
        for f in fns:
            if f.lower().endswith((".png", ".bmp", ".tif", ".tiff")):
                p = Path(dp, f)
                rel = os.path.splitext(os.path.relpath(p, masks_dir))[0].replace("\\", "/").lower()
                out.setdefault(rel, p)
                out.setdefault(os.path.basename(rel), p)
    return out


def sniff(masks_dir, sample: int = 20) -> dict:
    """What the masks look like: {"mode": index | color | instance, "values": Counter, "colors": n, "n": masks}."""
    idx = index_masks(masks_dir)
    files = sorted(set(idx.values()))[:sample]
    vals, colors, rgb, big = Counter(), set(), 0, 0
    for p in files:
        m, is_rgb = _read_indexed(p)
        if m is None:
            continue
        import numpy as np
        if is_rgb:
            rgb += 1
            u = np.unique(m.reshape(-1, 3), axis=0)
            colors |= {tuple(int(v) for v in c) for c in u[:512]}
        else:
            u = np.unique(m)
            vals.update(int(v) for v in u)
            big += int(u.max()) > 255 if len(u) else 0
    mode = "color" if rgb > len(files) // 2 else ("instance" if big or any(v > 1000 for v in vals) else "index")
    return {"mode": mode, "values": vals, "colors": len(colors), "n": len(set(idx.values()))}


def mask_to_objs(mask, mode: str, names: list[str], colors: dict, class_mask=None, min_area: float = 16.0,
                 ignore=(VOID,)):
    """-> [(label, polygon points)] for one mask."""
    import numpy as np
    from .importers import mask_polygons
    out = []
    if mode == "color":
        flat = mask.reshape(-1, 3)
        uniq = np.unique(flat, axis=0)
        voc = {c: i for i, c in enumerate(palette(256))}
        for c in uniq:
            c = tuple(int(v) for v in c)
            if c == (0, 0, 0) and c not in colors:
                continue
            if c not in colors and c in voc:                # a VOC palette mask read as colours: its index
                i = voc[c]
                if i in (0, VOID):
                    continue
                label = names[i - 1] if 0 < i <= len(names) else f"class_{i}"
            else:
                label = colors.get(c) or f"color_{c[0]}_{c[1]}_{c[2]}"
            if label.lower() in ("background", "_background_", "bg", "void"):
                continue
            region = np.all(mask == np.array(c, mask.dtype), axis=2)
            out += [(label, p) for p in mask_polygons(region, min_area)]
        return out
    for v in np.unique(mask):
        v = int(v)
        if v == 0 or v in ignore:
            continue
        region = mask == v
        if mode == "instance":
            if v >= 1000:                                   # Cityscapes: class * 1000 + instance
                cid = v // 1000
            elif class_mask is not None:
                cc = class_mask[region]
                cc = cc[(cc != 0) & (cc != VOID)]
                cid = int(np.bincount(cc.ravel()).argmax()) if cc.size else 0
            else:
                cid = 1
            label = names[cid - 1] if 0 < cid <= len(names) else f"class_{cid}"
            polys = mask_polygons(region, min_area)
            if polys:                                        # one object: its biggest piece
                out.append((label, max(polys, key=len)))
            continue
        label = names[v - 1] if 0 < v <= len(names) else f"class_{v}"
        if label.lower() in ("background", "_background_", "bg", "void"):
            continue
        out += [(label, p) for p in mask_polygons(region, min_area)]
    return out


def read_masks(masks_dir, images_dir, mode: str = "auto", labelmap: str = "", class_masks_dir: str = "",
               min_area: float = 16.0, notes=None, progress=None, cancel=None):
    """Records (core.prep.importers) for every image that has a mask. names: labelmap / classes.txt; with a
    VOC-style list whose first name is background, that name is class 0."""
    from .importers import Obj, Record
    from . import common as C
    masks_dir, images_dir = Path(masks_dir), Path(images_dir)
    names, colors = read_labelmap(labelmap)
    if names and names[0].lower() in ("background", "_background_", "bg"):
        names = names[1:]                                     # value 0 = background, 1 = the next name
    if mode == "auto":
        mode = sniff(masks_dir)["mode"]
    idx = index_masks(masks_dir)
    cidx = index_masks(class_masks_dir) if class_masks_dir else {}
    imgs, _j, _z, _o, _s = C.scan([str(images_dir)], jsons=False)
    recs, order, missing = [], [], 0
    import cv2  # noqa: F401  (fail early when OpenCV is missing)
    for i, img in enumerate(sorted(imgs)):
        if cancel is not None and cancel.is_set():
            raise C.Cancelled()
        if progress is not None and i % 50 == 0:
            progress(i, len(imgs), "Reading masks")
        rel = os.path.relpath(img, images_dir)
        mp = find_mask(masks_dir, rel, idx)
        if mp is None:
            missing += 1
            continue
        m, is_rgb = _read_indexed(mp)
        if m is None:
            if notes is not None:
                notes.append(f"unreadable mask: {mp}")
            continue
        cm = None
        if mode == "instance" and cidx:
            cp = find_mask(Path(class_masks_dir), rel, cidx)
            if cp is not None:
                cm, _rgb = _read_indexed(cp)
        r = Record(rel, float(m.shape[1]), float(m.shape[0]))
        for label, pts in mask_to_objs(m, "color" if is_rgb else mode, names, colors, cm, min_area):
            r.objs.append(Obj(label, "polygon", pts))
            if label not in order:
                order.append(label)
        recs.append(r)
    if notes is not None and missing:
        notes.append(f"{missing:,} image(s) have no mask (not imported)")
    if names:
        order = [n for n in names if n in order] + [o for o in order if o not in names]
    return recs, order


# ====================================================================================================== writing
def shapes_mask(shapes, W: int, H: int, cid: dict, normalized: bool = False):
    """-> (class index mask uint8 (0 = background), instance mask uint16, objects written)."""
    import cv2
    import math
    import numpy as np
    sem = np.zeros((H, W), np.uint8)
    inst = np.zeros((H, W), np.uint16)
    n = 0
    for sh in shapes:
        lab = str(sh.get("label", ""))
        if lab not in cid:
            continue
        pts = sh.get("points") or []
        if normalized:
            pts = [[p[0] * W, p[1] * H] for p in pts]
        st = sh.get("shape_type") or "polygon"
        if st == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif st == "circle" and len(pts) == 2:
            r = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            pts = [[pts[0][0] + r * math.cos(2 * math.pi * k / 32), pts[0][1] + r * math.sin(2 * math.pi * k / 32)]
                   for k in range(32)]
        if len(pts) < 3 or st in ("line", "linestrip", "point", "points"):
            continue
        poly = np.round(np.asarray(pts, np.float64)).astype(np.int32)
        n += 1
        cv2.fillPoly(sem, [poly], int(cid[lab]) + 1)
        cv2.fillPoly(inst, [poly], n)
    return sem, inst, n


def write_mask_png(path: Path, sem, names_n: int):
    """A palette PNG (class i shows in the VOC colour i; the pixel value IS the class index)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        im = Image.fromarray(sem, mode="P")
        pal = [v for c in palette(256) for v in c]
        im.putpalette(pal)
        im.save(str(path))
        return
    except ImportError:
        pass
    import cv2
    ok, buf = cv2.imencode(".png", sem)
    if ok:
        buf.tofile(str(path))


def write_instance_png(path: Path, inst):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", inst)
    if ok:
        buf.tofile(str(path))


def write_labelmap(out: Path, names: list[str]):
    pal = palette(len(names) + 1)
    lines = ["# label:color_rgb:parts:actions", "background:0,0,0::"]
    lines += [f"{n}:{pal[i + 1][0]},{pal[i + 1][1]},{pal[i + 1][2]}::" for i, n in enumerate(names)]
    (out / "labelmap.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_rel(rel: Path) -> Path:
    return Path(re.sub(r"[<>:\"|?*]", "_", str(rel)))
