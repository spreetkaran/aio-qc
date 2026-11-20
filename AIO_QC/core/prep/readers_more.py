# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
More annotation readers for Import annotations (Qt-free) — each returns (Records, class order) like the others:

    kitti         a folder of KITTI label .txt files (label_2): type … left top right bottom …; DontCare skipped
    via           VGG Image Annotator JSON (VIA 1 / 2 export or project): rect, polygon, polyline, circle, ellipse,
                  point; the class from the region attribute you name (else the first one); file attributes → flags
    supervisely   a Supervisely project folder (meta.json + <dataset>/ann/*.json): rectangle, polygon, point,
                  line, bitmap (traced); image tags → whole-image classes
    createml      CreateML JSON (Apple): [{"image", "annotations": [{"label", "coordinates": centre x/y, w/h}]}]
    openimages    Open Images CSV (ImageID, LabelName, XMin, XMax, YMin, YMax — 0…1); LabelName ids turned into
                  names with class-descriptions(-boxable).csv when it sits next to it
    datumaro      Datumaro JSON (annotations/<subset>.json): bbox, polygon, points, polyline, mask (RLE), label
    ls_brush      Label Studio brush masks (its own RLE) — read by read_labelstudio through decode_ls_rle()
"""
from __future__ import annotations

import base64
import csv
import json
import os
import zlib
from pathlib import Path

from .importers import Obj, Record, _ellipse, mask_polygons


# ====================================================================================================== KITTI
def read_kitti(folder, notes=None):
    recs, order = [], []
    skipped = 0
    for p in sorted(Path(folder).rglob("*.txt")):
        r = Record(p.stem)                                      # the image: same name (png / jpg) — found by name
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            t = ln.split()
            if len(t) < 8:
                continue
            lab = t[0]
            if lab.lower() == "dontcare":
                skipped += 1
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in t[4:8])
            except ValueError:
                continue
            flags = {}
            try:
                if float(t[1]) > 0.5:
                    flags["truncated"] = True
                if int(float(t[2])) >= 2:
                    flags["occluded"] = True
            except ValueError:
                pass
            r.objs.append(Obj(lab, "rectangle", [[x1, y1], [x2, y2]], flags))
            if lab not in order:
                order.append(lab)
        recs.append(r)
    if notes is not None and skipped:
        notes.append(f"{skipped:,} DontCare region(s) skipped")
    return recs, order


def looks_like_kitti(folder) -> bool:
    from itertools import islice
    for p in islice(Path(folder).rglob("*.txt"), 10):
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()[:3]:
            t = ln.split()
            if len(t) in (15, 16) and not _num(t[0]):
                return True
    return False


def _num(s) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# ====================================================================================================== VIA
def _via_entries(d):
    if isinstance(d, dict) and "_via_img_metadata" in d:
        return list(d["_via_img_metadata"].values())
    if isinstance(d, dict):
        return [v for v in d.values() if isinstance(v, dict) and "filename" in v]
    return []


def read_via(path, notes=None, attr: str = ""):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    recs, order = [], []
    for e in _via_entries(d):
        r = Record(str(e.get("filename", "")))
        regions = e.get("regions") or []
        if isinstance(regions, dict):                            # VIA 1
            regions = list(regions.values())
        for reg in regions:
            sa, ra = reg.get("shape_attributes") or {}, reg.get("region_attributes") or {}
            lab = ""
            keys = [attr] if attr and attr in ra else list(ra)
            for k in keys:
                v = ra.get(k)
                if isinstance(v, str) and v.strip():
                    lab = v.strip()
                    break
                if isinstance(v, dict):                          # checkbox / dropdown: the ticked option
                    on = [o for o, t in v.items() if t]
                    if on:
                        lab = on[0]
                        break
            lab = lab or "object"
            n = sa.get("name")
            if n == "rect":
                x, y, w, h = (float(sa.get(k, 0)) for k in ("x", "y", "width", "height"))
                o = Obj(lab, "rectangle", [[x, y], [x + w, y + h]])
            elif n in ("polygon", "polyline"):
                pts = [[float(x), float(y)] for x, y in zip(sa.get("all_points_x") or [], sa.get("all_points_y") or [])]
                o = Obj(lab, "polygon" if n == "polygon" else "linestrip", pts)
            elif n == "circle":
                cx, cy, rr = float(sa.get("cx", 0)), float(sa.get("cy", 0)), float(sa.get("r", 0))
                o = Obj(lab, "circle", [[cx, cy], [cx + rr, cy]])
            elif n == "ellipse":
                kind, pts = _ellipse(float(sa.get("cx", 0)), float(sa.get("cy", 0)), float(sa.get("rx", 0)),
                                     float(sa.get("ry", 0)), float(sa.get("theta", 0) or 0) * 57.29578)
                o = Obj(lab, kind, pts)
            elif n == "point":
                o = Obj(lab, "point", [[float(sa.get("cx", 0)), float(sa.get("cy", 0))]])
            else:
                continue
            r.objs.append(o)
            if lab not in order:
                order.append(lab)
        for k, v in (e.get("file_attributes") or {}).items():
            if isinstance(v, str) and v.strip():
                r.tags.append(v.strip())
            elif isinstance(v, dict):
                r.tags += [o for o, t in v.items() if t]
        recs.append(r)
    return recs, order


# ====================================================================================================== Supervisely
def _sly_bitmap(bm) -> list:
    import cv2
    import numpy as np
    raw = zlib.decompress(base64.b64decode(bm["data"]))
    im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        return []
    m = im[:, :, 3] if im.ndim == 3 and im.shape[2] == 4 else (im if im.ndim == 2 else im[:, :, 0])
    ox, oy = bm.get("origin") or [0, 0]
    return [[[x + ox, y + oy] for x, y in poly] for poly in mask_polygons(m > 0)]


def read_supervisely(folder, notes=None):
    folder = Path(folder)
    recs, order = [], []
    for ann in sorted(folder.rglob("ann/*.json")):
        try:
            d = json.loads(ann.read_text(encoding="utf-8"))
        except ValueError:
            continue
        name = ann.name[:-5] if ann.name.endswith(".json") else ann.stem       # "img.jpg.json" -> "img.jpg"
        img_rel = os.path.relpath(ann.parent.parent / "img" / name, folder)
        size = d.get("size") or {}
        r = Record(img_rel, float(size.get("width") or 0), float(size.get("height") or 0))
        for ob in d.get("objects") or []:
            lab = str(ob.get("classTitle", "object"))
            g = ob.get("geometryType")
            ext = (ob.get("points") or {}).get("exterior") or []
            if g == "rectangle" and len(ext) >= 2:
                r.objs.append(Obj(lab, "rectangle", [list(map(float, ext[0])), list(map(float, ext[1]))]))
            elif g == "polygon" and len(ext) >= 3:
                r.objs.append(Obj(lab, "polygon", [list(map(float, p)) for p in ext]))
            elif g == "line" and len(ext) >= 2:
                r.objs.append(Obj(lab, "linestrip", [list(map(float, p)) for p in ext]))
            elif g == "point" and ext:
                r.objs.append(Obj(lab, "point", [list(map(float, ext[0]))]))
            elif g == "bitmap" and ob.get("bitmap"):
                try:
                    for poly in _sly_bitmap(ob["bitmap"]):
                        r.objs.append(Obj(lab, "polygon", poly))
                except Exception as exc:
                    if notes is not None:
                        notes.append(f"{name}: bitmap not readable ({exc})")
            else:
                continue
            if lab not in order:
                order.append(lab)
        for t in d.get("tags") or []:
            n = str(t.get("name", ""))
            if n:
                r.tags.append(n)
        recs.append(r)
    return recs, order


def looks_like_supervisely(folder) -> bool:
    f = Path(folder)
    return (f / "meta.json").is_file() and next(f.rglob("ann/*.json"), None) is not None


# ====================================================================================================== CreateML
def read_createml(path, notes=None):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    recs, order = [], []
    for e in d if isinstance(d, list) else []:
        r = Record(str(e.get("image", "")))
        for a in e.get("annotations") or []:
            c = a.get("coordinates") or {}
            x, y, w, h = (float(c.get(k, 0)) for k in ("x", "y", "width", "height"))
            lab = str(a.get("label", "object"))
            r.objs.append(Obj(lab, "rectangle", [[x - w / 2, y - h / 2], [x + w / 2, y + h / 2]]))
            if lab not in order:
                order.append(lab)
        recs.append(r)
    return recs, order


# ====================================================================================================== Open Images
def read_openimages(path, notes=None):
    path = Path(path)
    names = {}
    for cand in list(path.parent.glob("*class-descriptions*.csv")):
        with open(cand, encoding="utf-8", errors="replace") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    names[row[0]] = row[1]
    recs, order = {}, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            iid = row.get("ImageID") or row.get("image_id")
            if not iid:
                continue
            r = recs.get(iid)
            if r is None:
                r = recs[iid] = Record(iid, norm=True)
            lab = names.get(row.get("LabelName", ""), row.get("LabelName", "object"))
            try:
                x1, x2, y1, y2 = (float(row[k]) for k in ("XMin", "XMax", "YMin", "YMax"))
            except (KeyError, ValueError):
                continue
            flags = {k.lower(): True for k in ("IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside")
                     if str(row.get(k, "0")) == "1"}
            r.objs.append(Obj(lab, "rectangle", [[x1, y1], [x2, y2]], flags))
            if lab not in order:
                order.append(lab)
    if notes is not None:
        notes.append("Open Images boxes are 0…1: scaled to each image's size" +
                     ("" if names else " · class ids kept (put class-descriptions-boxable.csv next to the file "
                                        "for names)"))
    return list(recs.values()), order


def looks_like_openimages(path) -> bool:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.readline()
    except OSError:
        return False
    return "ImageID" in head and "XMin" in head


# ====================================================================================================== Datumaro
def read_datumaro(path, notes=None):
    from .importers import rle_decode
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = [str(x.get("name")) for x in ((d.get("categories") or {}).get("label") or {}).get("labels") or []]
    recs, order = [], list(labels)
    for it in d.get("items") or []:
        im = it.get("image") or {}
        H, W = (im.get("size") or [0, 0])[:2]
        r = Record(str(im.get("path") or it.get("id", "")), float(W or 0), float(H or 0))
        for a in it.get("annotations") or []:
            li = a.get("label_id")
            lab = labels[li] if isinstance(li, int) and 0 <= li < len(labels) else str(li)
            t = a.get("type")
            if t == "bbox":
                x, y, w, h = map(float, a.get("bbox") or [0, 0, 0, 0])
                r.objs.append(Obj(lab, "rectangle", [[x, y], [x + w, y + h]], group=a.get("group") or None))
            elif t in ("polygon", "polyline", "points"):
                p = a.get("points") or []
                pts = [[float(p[i]), float(p[i + 1])] for i in range(0, len(p) - 1, 2)]
                kind = {"polygon": "polygon", "polyline": "linestrip", "points": "point"}[t]
                if kind == "point":
                    for q in pts:
                        r.objs.append(Obj(lab, "point", [q], group=a.get("group") or None))
                else:
                    r.objs.append(Obj(lab, kind, pts, group=a.get("group") or None))
            elif t == "mask" and a.get("rle"):
                rle = a["rle"]
                try:
                    h, w = rle.get("size") or [H, W]
                    for poly in mask_polygons(rle_decode(rle["counts"], int(h), int(w))):
                        r.objs.append(Obj(lab, "polygon", poly))
                except Exception as exc:
                    if notes is not None:
                        notes.append(f"{r.ref}: mask not readable ({exc})")
            elif t == "label":
                r.tags.append(lab)
        recs.append(r)
    return recs, order


def looks_like_datumaro(d) -> bool:
    return isinstance(d, dict) and "items" in d and "categories" in d


def via_attributes(path) -> list[str]:
    """The region attribute names of a VIA file, the most used first."""
    from collections import Counter
    c = Counter()
    for e in _via_entries(json.loads(Path(path).read_text(encoding="utf-8"))):
        regions = e.get("regions") or []
        for reg in regions.values() if isinstance(regions, dict) else regions:
            c.update(k for k, v in (reg.get("region_attributes") or {}).items() if v)
    return [k for k, _n in c.most_common()]


def looks_like_via(d) -> bool:
    return bool(_via_entries(d)) and all("regions" in e for e in _via_entries(d)[:5])


def looks_like_createml(d) -> bool:
    return isinstance(d, list) and bool(d) and isinstance(d[0], dict) and "image" in d[0] and "annotations" in d[0]


# ====================================================================================================== Label Studio brush
def decode_ls_rle(rle, h: int, w: int):
    """Label Studio brush RLE (its own bit-packed format) -> uint8 mask (h, w)."""
    import numpy as np
    bits = "".join(f"{b:08b}" for b in bytes(int(v) & 0xFF for v in rle))
    pos = [0]

    def read(n):
        v = int(bits[pos[0]:pos[0] + n] or "0", 2)
        pos[0] += n
        return v
    num = read(32)
    word = read(5) + 1
    sizes = [read(4) + 1 for _ in range(4)]
    out = np.zeros(num, np.uint8)
    i = 0
    while i < num:
        x = read(1)
        j = i + 1 + read(sizes[read(2)])
        if x:
            out[i:j] = read(word)
            i = j
        else:
            while i < j and i < num:
                out[i] = read(word)
                i += 1
    return (out.reshape(h, w, 4)[:, :, 3] > 0).astype(np.uint8)
