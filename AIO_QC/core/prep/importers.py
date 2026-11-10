# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Import annotations from other tools (Qt-free) into a NEW folder as LabelMe or YOLO — sources are never touched.

Readers (auto-detected):
    COCO          instances JSON: bbox -> rectangle, polygon segmentation -> polygon(s), RLE masks -> polygon(s)
                  (traced from the mask), iscrowd kept as a flag; keypoints are not imported (counted)
    Pascal VOC    a folder of .xml files (labelImg & co.): bndbox -> rectangle, difficult / truncated / occluded ->
                  flags; coordinates taken as written (labelImg convention)
    CVAT          "CVAT for images 1.1" XML: box (rotated boxes -> 4-point polygon), polygon, polyline, points,
                  ellipse (-> circle when round, else polygon), mask (RLE -> polygon); attributes -> flags /
                  description; "CVAT for video 1.1" tracks too (frame n -> the n-th image, natural name order)
    Label Studio  JSON export: rectanglelabels (rotation too), polygonlabels, keypointlabels, ellipselabels; a
                  textarea on the same region becomes the shape's description (OCR transcriptions); the latest
                  not-cancelled annotation of each task is used

Every reader yields the same neutral records, so one writer serves all of them:
    Record(image ref, width, height, [Obj(label, kind, points (pixels), flags, attrs, description)])

Images are found by relative path, then by file name (case-insensitive), then by the name without Label Studio's
upload prefix ("1a2b3c4d-photo.jpg" -> "photo.jpg"). Images that cannot be found are listed and skipped.

Writers:
    labelme       <out>/<same sub-folders>/<image> + <stem>.json (imageData null)
    yolo_det      <out>/images/… + <out>/labels/….txt (class cx cy w h), data.yaml + classes.txt
    yolo_seg      as yolo_det, polygons as "class x y x y …" (boxes become 4-point polygons)
Images are hard-linked when possible (no extra space), else copied.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import common as C

FORMATS = {"coco": "COCO JSON", "voc": "Pascal VOC XML (folder)", "cvat": "CVAT XML",
           "labelstudio": "Label Studio JSON"}
OUTPUTS = {"labelme": "LabelMe (JSON per image)", "yolo_det": "YOLO detection (boxes)",
           "yolo_seg": "YOLO segmentation (polygons)"}
LS_PREFIX = re.compile(r"^[0-9a-f]{8}-(.+)$", re.I)


@dataclass
class Obj:
    label: str
    kind: str                     # rectangle | polygon | linestrip | point | circle
    points: list                  # pixels
    flags: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)
    description: str = ""


@dataclass
class Record:
    ref: str                      # image reference as written in the source (path / name / URL)
    width: float = 0
    height: float = 0
    objs: list = field(default_factory=list)


# ================================================================================================ detection
def detect(src) -> str | None:
    p = Path(src)
    if p.is_dir():
        for x in list(p.rglob("*.xml"))[:20]:
            try:
                root = _iter_root(x)
            except Exception:
                continue
            if root == "annotation":
                return "voc"
            if root == "annotations":
                return "cvat"
        return None
    if p.suffix.lower() == ".xml":
        r = _iter_root(p)
        return {"annotation": "voc", "annotations": "cvat"}.get(r)
    if p.suffix.lower() == ".json":
        with open(p, "rb") as f:
            head = f.read(1 << 16).lstrip()
        if head[:1] == b"[":
            return "labelstudio"
        d = C.read_json(p)
        if isinstance(d, dict) and "images" in d and "annotations" in d:
            return "coco"
        if isinstance(d, list):
            return "labelstudio"
    return None


def _iter_root(path) -> str:
    for _ev, el in ET.iterparse(str(path), events=("start",)):
        return el.tag
    return ""


# ================================================================================================ COCO
def rle_decode(counts, h: int, w: int):
    """COCO RLE (uncompressed list or compressed string) -> uint8 mask (h, w)."""
    import numpy as np
    if isinstance(counts, str) or isinstance(counts, bytes):
        s = counts.encode() if isinstance(counts, str) else counts
        cnts, p = [], 0
        while p < len(s):
            x, k, more = 0, 0, 1
            while more:
                c = s[p] - 48
                x |= (c & 0x1F) << (5 * k)
                more = c & 0x20
                p += 1
                k += 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)
            if len(cnts) > 2:
                x += cnts[-2]
            cnts.append(x)
    else:
        cnts = [int(c) for c in counts]
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in cnts:
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return flat.reshape((w, h)).T                       # column-major


def mask_polygons(mask, min_area: float = 4.0) -> list[list]:
    import cv2
    import numpy as np
    m = np.ascontiguousarray(mask.astype(np.uint8))
    cs, _h = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < min_area:
            continue
        a = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(a) >= 3:
            out.append([[float(x), float(y)] for x, y in a])
    return out


def read_coco(path, use_masks=True, notes=None) -> tuple[list[Record], list[str]]:
    d = C.read_json(path)
    if not isinstance(d, dict):
        raise ValueError("not a COCO JSON")
    cats = {c["id"]: str(c.get("name", c["id"])) for c in d.get("categories") or []}
    order = [cats[k] for k in sorted(cats)]
    recs = {}
    for im in d.get("images") or []:
        recs[im["id"]] = Record(str(im.get("file_name", "")), float(im.get("width") or 0),
                                float(im.get("height") or 0))
    kp = 0
    for a in d.get("annotations") or []:
        r = recs.get(a.get("image_id"))
        if r is None:
            continue
        lab = cats.get(a.get("category_id"), str(a.get("category_id")))
        flags = {"iscrowd": True} if a.get("iscrowd") else {}
        seg = a.get("segmentation")
        made = False
        if use_masks and isinstance(seg, list) and seg:
            for poly in seg:
                if isinstance(poly, list) and len(poly) >= 6:
                    r.objs.append(Obj(lab, "polygon", [[float(poly[i]), float(poly[i + 1])]
                                                        for i in range(0, len(poly) - 1, 2)], dict(flags)))
                    made = True
        elif use_masks and isinstance(seg, dict) and seg.get("counts") is not None:
            h, w = (seg.get("size") or [r.height, r.width])[:2]
            try:
                for poly in mask_polygons(rle_decode(seg["counts"], int(h), int(w))):
                    r.objs.append(Obj(lab, "polygon", poly, dict(flags)))
                    made = True
            except Exception as exc:
                if notes is not None:
                    notes.append(f"{r.ref}: mask not readable ({exc}) — box used")
        if not made and a.get("bbox"):
            x, y, w, h = [float(v) for v in a["bbox"][:4]]
            r.objs.append(Obj(lab, "rectangle", [[x, y], [x + w, y + h]], flags))
        if a.get("keypoints"):
            kp += 1
    if kp and notes is not None:
        notes.append(f"{kp:,} keypoint set(s) not imported (boxes / polygons only)")
    return list(recs.values()), order


# ================================================================================================ VOC
def _f(el, tag, default=0.0):
    x = el.find(tag)
    try:
        return float(x.text) if x is not None and x.text is not None else default
    except ValueError:
        return default


def read_voc(folder, notes=None) -> tuple[list[Record], list[str]]:
    files = [Path(folder)] if Path(folder).is_file() else sorted(Path(folder).rglob("*.xml"))
    recs, order = [], []
    for x in files:
        try:
            root = ET.parse(x).getroot()
        except ET.ParseError as exc:
            if notes is not None:
                notes.append(f"{x.name}: not readable ({exc})")
            continue
        if root.tag != "annotation":
            continue
        fn = (root.findtext("filename") or "").strip() or x.with_suffix(".jpg").name
        rel_dir = os.path.relpath(x.parent, folder) if Path(folder).is_dir() else ""
        size = root.find("size")
        r = Record(os.path.join(rel_dir, fn) if rel_dir not in ("", ".") else fn,
                   _f(size, "width") if size is not None else 0, _f(size, "height") if size is not None else 0)
        for ob in root.findall("object"):
            lab = (ob.findtext("name") or "").strip()
            bb = ob.find("bndbox")
            if not lab or bb is None:
                continue
            if lab not in order:
                order.append(lab)
            flags = {k: True for k in ("difficult", "truncated", "occluded") if (ob.findtext(k) or "0").strip() == "1"}
            r.objs.append(Obj(lab, "rectangle", [[_f(bb, "xmin"), _f(bb, "ymin")], [_f(bb, "xmax"), _f(bb, "ymax")]],
                              flags))
        recs.append(r)
    return recs, order


# ================================================================================================ CVAT
TEXT_ATTRS = {"text", "description", "value", "transcription", "ocr"}     # -> description; all others kept as attributes


def _attrs(el) -> tuple[dict, dict, str]:
    flags, attrs, desc = {}, {}, ""
    for a in el.findall("attribute"):
        name, val = a.get("name", ""), (a.text or "").strip()
        if val.lower() in ("true", "false"):
            flags[name] = val.lower() == "true"
        elif name.lower() in TEXT_ATTRS and val:
            desc = val
        elif val:
            attrs[name] = val
    if el.get("occluded") == "1":
        flags["occluded"] = True
    return flags, attrs, desc


def _pts(s: str) -> list:
    out = []
    for pair in (s or "").split(";"):
        if "," in pair:
            x, y = pair.split(",")[:2]
            out.append([float(x), float(y)])
    return out


def _rot_box(x1, y1, x2, y2, deg):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        dx, dy = x - cx, y - cy
        out.append([cx + dx * ca - dy * sa, cy + dx * sa + dy * ca])
    return out


def _ellipse(cx, cy, rx, ry, rot=0.0) -> tuple[str, list]:
    if abs(rx - ry) <= 0.5:
        return "circle", [[cx, cy], [cx + rx, cy]]
    a = math.radians(rot)
    pts = []
    for i in range(24):
        t = 2 * math.pi * i / 24
        x, y = rx * math.cos(t), ry * math.sin(t)
        pts.append([cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)])
    return "polygon", pts


def _cvat_shape(el, lab) -> Obj | None:
    flags, attrs, desc = _attrs(el)
    t = el.tag
    if t == "box":
        x1, y1, x2, y2 = (float(el.get(k, 0)) for k in ("xtl", "ytl", "xbr", "ybr"))
        rot = float(el.get("rotation", 0) or 0)
        if abs(rot) > 0.01:
            return Obj(lab, "polygon", _rot_box(x1, y1, x2, y2, rot), flags, attrs, desc)
        return Obj(lab, "rectangle", [[x1, y1], [x2, y2]], flags, attrs, desc)
    if t == "polygon":
        return Obj(lab, "polygon", _pts(el.get("points")), flags, attrs, desc)
    if t == "polyline":
        return Obj(lab, "linestrip", _pts(el.get("points")), flags, attrs, desc)
    if t == "points":
        pts = _pts(el.get("points"))
        return Obj(lab, "point" if len(pts) == 1 else "linestrip", pts, flags, attrs, desc) if pts else None
    if t == "ellipse":
        kind, pts = _ellipse(float(el.get("cx", 0)), float(el.get("cy", 0)), float(el.get("rx", 0)),
                             float(el.get("ry", 0)), float(el.get("rotation", 0) or 0))
        return Obj(lab, kind, pts, flags, attrs, desc)
    if t == "mask":
        try:
            left, top = int(float(el.get("left", 0))), int(float(el.get("top", 0)))
            w, h = int(float(el.get("width", 0))), int(float(el.get("height", 0)))
            import numpy as np
            runs = [int(v) for v in (el.get("rle") or "").split(",") if v.strip()]
            flat = np.zeros(w * h, np.uint8)
            pos, val = 0, 0
            for c in runs:
                if val:
                    flat[pos:pos + c] = 1
                pos += c
                val ^= 1
            polys = mask_polygons(flat.reshape(h, w))
            if polys:
                big = max(polys, key=len)
                return Obj(lab, "polygon", [[x + left, y + top] for x, y in big], flags, attrs, desc)
        except Exception:
            return None
    return None


def read_cvat(path, images_dir=None, notes=None) -> tuple[list[Record], list[str]]:
    root = ET.parse(path).getroot()
    order = [(lb.findtext("name") or "").strip() for lb in root.iter("label") if lb.findtext("name")]
    order = list(dict.fromkeys(o for o in order if o))
    recs = []
    for im in root.findall("image"):
        r = Record(im.get("name", ""), float(im.get("width") or 0), float(im.get("height") or 0))
        for el in im:
            lab = el.get("label")
            if not lab:
                continue
            o = _cvat_shape(el, lab)
            if o is not None and o.points:
                r.objs.append(o)
            if lab not in order:
                order.append(lab)
        recs.append(r)
    tracks = root.findall("track")
    if tracks:
        frames: dict[int, Record] = {}
        for tr in tracks:
            lab = tr.get("label", "")
            if lab and lab not in order:
                order.append(lab)
            for el in tr:
                if el.get("outside") == "1":
                    continue
                o = _cvat_shape(el, lab)
                if o is None or not o.points:
                    continue
                o.flags = {**o.flags}
                n = int(el.get("frame", 0))
                frames.setdefault(n, Record(f"#frame{n}")).objs.append(o)
        names = []
        if images_dir:
            imgs, _j, _z, _o, _s = C.scan([images_dir], jsons=False)
            from ..sequences import natural_key
            names = sorted(imgs, key=lambda p: natural_key(os.path.relpath(p, images_dir)))
        for n, r in sorted(frames.items()):
            if n < len(names):
                r.ref = os.path.relpath(names[n], images_dir)
            recs.append(r)
        if notes is not None:
            notes.append(f"video tracks: frame n = the n-th image of the images folder in name order "
                         f"({len(frames):,} frame(s), {len(names):,} image(s))")
    return recs, order


# ================================================================================================ Label Studio
def _ls_image(task) -> str:
    data = task.get("data") or {}
    for k in ("image", "img", "url", "image_url"):
        if isinstance(data.get(k), str):
            return data[k]
    for v in data.values():
        if isinstance(v, str) and os.path.splitext(v.split("?")[0])[1].lower() in C.IMAGE_EXTS:
            return v
    return ""


def read_labelstudio(path, notes=None) -> tuple[list[Record], list[str]]:
    tasks = C.read_json(path)
    if isinstance(tasks, dict):
        tasks = [tasks]
    order, recs, skipped = [], [], Counter()
    for t in tasks or []:
        anns = [a for a in (t.get("annotations") or []) if not a.get("was_cancelled")]
        if not anns:
            continue
        ann = max(anns, key=lambda a: (a.get("updated_at") or a.get("created_at") or "", a.get("id") or 0))
        r = Record(_ls_image(t))
        regions, texts = {}, {}
        for res in ann.get("result") or []:
            typ, v = res.get("type", ""), res.get("value") or {}
            W, H = float(res.get("original_width") or 0), float(res.get("original_height") or 0)
            if W and H:
                r.width, r.height = W, H
            if typ == "textarea":
                txt = v.get("text")
                texts[res.get("id")] = "\n".join(txt) if isinstance(txt, list) else str(txt or "")
                continue
            labels = next((v[k] for k in v if k.endswith("labels") and isinstance(v[k], list)), None)
            if not labels:
                if typ not in ("choices",):
                    skipped[typ] += 1
                continue
            if not (W and H):
                skipped["no image size"] += 1
                continue
            lab = str(labels[0])
            if lab not in order:
                order.append(lab)
            if typ == "rectanglelabels":
                x, y, w, h = (float(v.get(k, 0)) * s / 100 for k, s in (("x", W), ("y", H), ("width", W),
                                                                        ("height", H)))
                rot = float(v.get("rotation", 0) or 0)
                if abs(rot) > 0.01:              # Label Studio rotates around the top-left corner
                    a = math.radians(rot)
                    ca, sa = math.cos(a), math.sin(a)
                    pts = [[x + dx * ca - dy * sa, y + dx * sa + dy * ca] for dx, dy in ((0, 0), (w, 0), (w, h), (0, h))]
                    o = Obj(lab, "polygon", pts)
                else:
                    o = Obj(lab, "rectangle", [[x, y], [x + w, y + h]])
            elif typ == "polygonlabels":
                o = Obj(lab, "polygon", [[float(p[0]) * W / 100, float(p[1]) * H / 100] for p in v.get("points") or []])
            elif typ == "keypointlabels":
                o = Obj(lab, "point", [[float(v.get("x", 0)) * W / 100, float(v.get("y", 0)) * H / 100]])
            elif typ == "ellipselabels":
                kind, pts = _ellipse(float(v.get("x", 0)) * W / 100, float(v.get("y", 0)) * H / 100,
                                     float(v.get("radiusX", 0)) * W / 100, float(v.get("radiusY", 0)) * H / 100,
                                     float(v.get("rotation", 0) or 0))
                o = Obj(lab, kind, pts)
            else:
                skipped[typ] += 1
                continue
            regions[res.get("id")] = o
            r.objs.append(o)
        for rid, txt in texts.items():
            if rid in regions and txt:
                regions[rid].description = txt
        recs.append(r)
    if skipped and notes is not None:
        notes.append("not imported: " + ", ".join(f"{k} ×{n}" for k, n in skipped.most_common()))
    return recs, order


# ================================================================================================ images
class ImageIndex:
    """Find an image by relative path, then name, then name without Label Studio's upload prefix."""

    def __init__(self, folder):
        self.folder = os.path.abspath(folder)
        imgs, _j, _z, _o, _s = C.scan([folder], jsons=False)
        self.by_rel = {C.norm(os.path.relpath(p, folder)): p for p in imgs}
        self.by_name = defaultdict(list)
        for p in imgs:
            self.by_name[os.path.basename(p).lower()].append(p)
        self.n = len(imgs)

    def find(self, ref: str) -> tuple[str | None, str]:
        """(path, how) — how: '' exact · 'name' · 'ambiguous' (several with that name) · 'missing'."""
        if not ref:
            return None, "missing"
        ref = ref.split("?")[0].replace("\\", "/")
        if ref.startswith(("http://", "https://", "s3://", "gs://", "/data/")):
            ref = ref.rsplit("/", 1)[-1]
        p = self.by_rel.get(C.norm(ref.lstrip("/")))
        if p:
            return p, ""
        name = os.path.basename(ref).lower()
        for cand in (name, (LS_PREFIX.match(name).group(1) if LS_PREFIX.match(name) else None)):
            if cand and self.by_name.get(cand):
                hits = self.by_name[cand]
                return hits[0], "name" if len(hits) == 1 else "ambiguous"
        return None, "missing"


# ================================================================================================ plan / run
def read_any(fmt, src, images_dir, use_masks=True, notes=None):
    if fmt == "coco":
        return read_coco(src, use_masks, notes)
    if fmt == "voc":
        return read_voc(src, notes)
    if fmt == "cvat":
        return read_cvat(src, images_dir, notes)
    if fmt == "labelstudio":
        return read_labelstudio(src, notes)
    raise ValueError(f"unknown format {fmt}")


def plan(src, images_dir, out, fmt="auto", output="labelme", use_masks=True, link=True, progress=None, cancel=None):
    from .tasks import Plan
    src, images_dir, out = os.path.abspath(src), os.path.abspath(images_dir), os.path.abspath(out)
    if C.inside(out, images_dir) or (os.path.isdir(src) and C.inside(out, src)):
        raise ValueError("Choose an output folder outside the source folders.")
    if os.path.isdir(out) and any(os.scandir(out)):
        raise ValueError("The output folder must be new or empty.")
    if fmt == "auto":
        fmt = detect(src)
        if fmt is None:
            raise ValueError("Could not tell the format — choose it.")
    notes = []
    if progress is not None:
        progress(0, 0, f"Reading {FORMATS[fmt]}")
    recs, order = read_any(fmt, src, images_dir, use_masks, notes)
    if cancel is not None and cancel.is_set():
        raise C.Cancelled()
    idx = ImageIndex(images_dir)
    P = Plan("import", opts={"fmt": fmt, "output": output, "out": out, "images": images_dir, "link": link,
                             "order": order})
    found, by_name, missing, ambiguous, dup = 0, 0, 0, 0, 0
    kinds = Counter()
    used = set()
    for r in recs:
        path, how = idx.find(r.ref)
        if path is None:
            missing += 1
            P.details["Images not found (skipped)"].append(r.ref or "(no image reference)")
            continue
        if path in used:
            dup += 1
            P.details["Listed twice (merged)"].append(r.ref)
            prev = next(a for a in P.actions if a[1] == path)
            prev[3]["rec"].objs.extend(r.objs)
            continue
        used.add(path)
        found += 1
        by_name += how == "name"
        if how == "ambiguous":
            ambiguous += 1
            P.details["Several images with this name — the first one was used"].append(f"{r.ref} → {path}")
        for o in r.objs:
            kinds[o.kind] += 1
        rel = os.path.relpath(path, images_dir)
        P.actions.append(("import", path, rel, {"rec": r}))
    labels = Counter(o.label for a in P.actions for o in a[3]["rec"].objs)
    for lab in labels:
        if lab not in order:
            order.append(lab)
    P.rows = [("Format", FORMATS[fmt]), ("Images in the annotation file", len(recs)), ("Images found", found),
              ("    found by file name only", by_name), ("    not found (skipped)", missing),
              ("Shapes", sum(kinds.values()))] + \
             [(f"    {k}", n) for k, n in kinds.most_common()] + \
             [("Classes", len(labels))] + [(f"    {lab}", labels[lab]) for lab in order if labels[lab]] + \
             [("Write as", OUTPUTS[output])]
    for n in notes:
        P.details["Notes"].append(n)
    for lab in order:
        P.details["Classes (in this order" + (" = YOLO ids)" if output != "labelme" else ")")].append(lab)
    return P


def _place_image(src, dst, link):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if link:
        try:
            os.link(src, dst)
            return "linked"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copied"


def _clip(v, hi):
    return min(max(v, 0.0), hi) if hi else v


def to_labelme(rec: Record, image_name: str, W, H) -> dict:
    shapes = []
    for o in rec.objs:
        sh = {"label": o.label, "points": [[round(_clip(x, W), 2), round(_clip(y, H), 2)] for x, y in o.points],
              "group_id": None, "description": o.description or "", "shape_type": o.kind, "flags": dict(o.flags)}
        if o.attrs:
            sh["attributes"] = dict(o.attrs)
        shapes.append(sh)
    return {"version": "5.4.1", "flags": {}, "shapes": shapes, "imagePath": image_name, "imageData": None,
            "imageHeight": int(H), "imageWidth": int(W)}


def to_yolo(rec: Record, W, H, ids: dict, seg: bool) -> tuple[str, int]:
    lines, skipped = [], 0
    for o in rec.objs:
        pts = o.points
        if o.kind == "circle" and len(pts) == 2:
            r = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            cx, cy = pts[0]
            pts = [[cx + r * math.cos(2 * math.pi * i / 16), cy + r * math.sin(2 * math.pi * i / 16)] for i in range(16)]
        elif o.kind == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif o.kind in ("point", "linestrip") or len(pts) < 3:
            skipped += 1
            continue
        xs = [_clip(p[0], W) / W for p in pts]
        ys = [_clip(p[1], H) / H for p in pts]
        if seg:
            lines.append(" ".join([str(ids[o.label])] + [f"{v:.6f}" for xy in zip(xs, ys) for v in xy]))
        else:
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                skipped += 1
                continue
            lines.append(f"{ids[o.label]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return "\n".join(lines) + ("\n" if lines else ""), skipped


def run(P, progress=None, cancel=None) -> dict:
    from .. import io as cio
    o = P.opts
    out, output = o["out"], o["output"]
    os.makedirs(out, exist_ok=True)
    log = C.RunLog(Path(out) / C.LOG_DIR, "import", keep_originals=False)
    st = {"done": 0, "shapes": 0, "skipped": 0, "errors": [], "linked": 0, "copied": 0,
          "log": str(log.csv_path), "undo": str(log.jsonl_path)}
    ids = {lab: i for i, lab in enumerate(o["order"])}
    yolo = output != "labelme"
    try:
        for i, (_k, img, rel, ex) in enumerate(P.actions, 1):
            if cancel is not None and cancel.is_set():
                st["errors"].append("stopped by you")
                break
            rec: Record = ex["rec"]
            try:
                W, H = rec.width, rec.height
                if not (W and H):
                    wh = cio.image_size(img)
                    if not wh:
                        raise ValueError("image size unknown")
                    W, H = wh
                if yolo:
                    dimg = os.path.join(out, "images", rel)
                    dlab = os.path.join(out, "labels", os.path.splitext(rel)[0] + ".txt")
                else:
                    dimg = os.path.join(out, rel)
                    dlab = os.path.splitext(dimg)[0] + ".json"
                if os.path.exists(dimg) or os.path.exists(dlab):
                    raise FileExistsError(f"{dimg} already exists")
                how = _place_image(img, dimg, o["link"])
                st[how] += 1
                log.created(dimg, how, img, "image")
                if yolo:
                    text, skipped = to_yolo(rec, W, H, ids, output == "yolo_seg")
                    st["skipped"] += skipped
                    os.makedirs(os.path.dirname(dlab), exist_ok=True)
                    C.write_atomic(dlab, text.encode("utf-8"))
                else:
                    C.write_atomic(dlab, C.dump_json(to_labelme(rec, os.path.basename(dimg), W, H)))
                log.created(dlab, f"{len(rec.objs)} shape(s)", "", "annotation")
                st["shapes"] += len(rec.objs)
                st["done"] += 1
            except Exception as exc:
                log.error("import", img, "", exc)
                st["errors"].append(f"{img}: {type(exc).__name__}: {exc}")
            if progress is not None and (i % 25 == 0 or i == len(P.actions)):
                progress(i, len(P.actions), "Importing")
        if yolo:
            names = o["order"]
            yaml = (f"path: {out}\ntrain: images\nval: images\n\nnames:\n" +
                    "".join(f"  {i}: {json.dumps(n, ensure_ascii=False)}\n" for i, n in enumerate(names)))
            for fn, txt in (("data.yaml", yaml), ("classes.txt", "\n".join(names) + "\n")):
                p = os.path.join(out, fn)
                C.write_atomic(p, txt.encode("utf-8"))
                log.created(p, "class list", "", "classes")
    finally:
        log.close()
    return st
