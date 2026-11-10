# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Fix EXIF rotation (Qt-free): phone / camera photos often store the pixels sideways plus an EXIF "rotate me" tag.
Viewers, labelme and Ultralytics apply the tag; OpenCV-based tools, some annotation tools and older trainers do not —
so the same box lands on a different part of the picture depending on the program.

The fix makes the file unambiguous: the rotation is baked into the pixels and the tag reset to 1 (other EXIF data
and the colour profile are kept). The labels are brought into the same, upright frame:

    labels drawn on the upright view (what labelme / Ultralytics show)   -> unchanged
    labels drawn on the raw, un-rotated pixels                           -> every point rotated / mirrored with it

Which frame the labels are in is decided per file ("auto"): a LabelMe JSON whose imageWidth / imageHeight equal the
RAW size of a 90°/270° photo was drawn on the raw pixels; equal to the upright size -> the upright view. When the
sizes cannot tell (180° / mirrored, square images, YOLO .txt without a size) the upright view is assumed, which is
what labelme and Ultralytics use. "raw" / "upright" force one frame for every file.

Every rewritten file is kept (compressed) in the run log first, so a run can be undone exactly.
"""
from __future__ import annotations

import os
import tempfile

from . import common as C

ROTATED = {5, 6, 7, 8}
ORIENT_TEXT = {2: "mirrored", 3: "rotated 180°", 4: "flipped", 5: "transposed", 6: "rotated 90°",
               7: "transversed", 8: "rotated 270°"}


def map_point(o: int, x: float, y: float, W: float, H: float) -> tuple[float, float]:
    """A point of the RAW W × H image -> the same point in the upright (EXIF-applied) image."""
    if o == 2:
        return W - x, y
    if o == 3:
        return W - x, H - y
    if o == 4:
        return x, H - y
    if o == 5:
        return y, x
    if o == 6:
        return H - y, x
    if o == 7:
        return H - y, W - x
    if o == 8:
        return y, W - x
    return x, y


def upright_size(o: int, W, H):
    return (H, W) if o in ROTATED else (W, H)


def labels_frame(o: int, raw_wh, ann_wh, policy: str = "auto") -> str:
    """'raw' | 'upright' | 'unknown' (sizes contradict both)."""
    if policy in ("raw", "upright"):
        return policy
    if not ann_wh or not ann_wh[0] or not ann_wh[1]:
        return "upright"
    W, H = raw_wh
    aw, ah = int(round(float(ann_wh[0]))), int(round(float(ann_wh[1])))
    if o in ROTATED and W != H:
        if (aw, ah) == (W, H):
            return "raw"
        if (aw, ah) == (H, W):
            return "upright"
        return "unknown"
    if (aw, ah) != (W, H):
        return "unknown"
    return "upright"


def exif_info(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            try:
                o = int(im.getexif().get(0x0112, 1) or 1)
            except Exception:
                o = 1
            return im.width, im.height, o
    except Exception:
        return None


def yolo_txt_for(img: str) -> str | None:
    """images/…/x.jpg -> labels/…/x.txt (the YOLO convention), if it exists."""
    p = img.replace("\\", "/")
    i = p.rfind("/images/")
    if i < 0:
        return None
    t = os.path.splitext(p[:i] + "/labels/" + p[i + 8:])[0] + ".txt"
    t = os.path.normpath(t)
    return t if os.path.isfile(t) else None


def json_for(img: str) -> str | None:
    j = os.path.splitext(img)[0] + ".json"
    return j if os.path.isfile(j) else None


# ------------------------------------------------------------------------------------------------- plan
def plan_files(items, policy="auto", progress=None, cancel=None):
    """items: [(image, json or None, yolo txt or None)] -> Plan (only images with an orientation tag ≠ 1)."""
    from .tasks import Plan, _pmap
    P = Plan("exif", opts={"policy": policy})
    infos = _pmap(lambda it: exif_info(it[0]), list(items), progress, "Reading EXIF", cancel)
    n_raw = n_up = n_unknown = 0
    for (img, js, txt), info in zip(items, infos):
        if info is None:
            P.details["Unreadable images (skipped)"].append(img)
            continue
        W, H, o = info
        if o in (0, 1) or o not in ORIENT_TEXT:
            continue
        ann_wh = None
        if js:
            d = C.read_json(js)
            if isinstance(d, dict):
                ann_wh = (d.get("imageWidth"), d.get("imageHeight"))
        frame = labels_frame(o, (W, H), ann_wh, policy)
        if frame == "unknown":
            n_unknown += 1
            P.details["Label size matches neither orientation — skipped (check with Image Quality)"].append(
                f"{img}  (image {W}×{H}, labels {ann_wh[0]}×{ann_wh[1]}, EXIF {ORIENT_TEXT[o]})")
            continue
        P.actions.append(("fix", img, None, {"o": o, "raw": (W, H), "frame": frame, "json": js, "txt": txt}))
        if frame == "raw" and (js or txt):
            n_raw += 1
            P.details["Labels rotated with the image (drawn on the raw pixels)"].append(img)
        else:
            n_up += 1
            P.details["Pixels made upright, labels unchanged (already upright)"].append(img)
    P.rows = [("Images checked", len(items)), ("With an EXIF rotation tag", len(P.actions) + n_unknown),
              ("    labels rotated with the image", n_raw), ("    labels already upright", n_up),
              ("    skipped: label size fits neither", n_unknown)]
    return P


def plan_folder(root, policy="auto", progress=None, cancel=None):
    images, _j, _z, _o, _s = C.scan([root], jsons=False, cancel=cancel)
    return plan_files([(p, json_for(p), yolo_txt_for(p)) for p in images], policy, progress, cancel)


# ------------------------------------------------------------------------------------------------- run
def bake(path: str) -> tuple[int, int]:
    """Rewrite an image with its EXIF rotation applied to the pixels and the tag reset (other EXIF + ICC kept)."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        fmt = im.format or "JPEG"
        icc = im.info.get("icc_profile")
        sub = None
        if fmt == "JPEG":
            try:
                from PIL import JpegImagePlugin
                sub = JpegImagePlugin.get_sampling(im)
            except Exception:
                sub = None
        exif = im.getexif()
        out = ImageOps.exif_transpose(im)
        if out is im:
            out = im.copy()
    exif[0x0112] = 1
    kw = {"exif": exif.tobytes()}
    if icc:
        kw["icc_profile"] = icc
    if fmt == "JPEG":
        kw["quality"] = 95
        if sub is not None and sub >= 0:
            kw["subsampling"] = sub
        if out.mode not in ("RGB", "L", "CMYK"):
            out = out.convert("RGB")
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".aioqc_", suffix=os.path.splitext(path)[1], dir=d)
    os.close(fd)
    try:
        out.save(tmp, format=fmt, **kw)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return out.width, out.height


def rotate_labelme(d: dict, o: int, W: int, H: int) -> int:
    n = 0
    for sh in d.get("shapes") or []:
        pts = sh.get("points")
        if isinstance(pts, list):
            new = []
            for p in pts:
                x, y = map_point(o, float(p[0]), float(p[1]), W, H)
                new.append([round(x, 2), round(y, 2)])
            if (sh.get("shape_type") or "") == "rectangle" and len(new) == 2:      # keep top-left / bottom-right
                (x1, y1), (x2, y2) = new
                new = [[min(x1, x2), min(y1, y2)], [max(x1, x2), max(y1, y2)]]
            sh["points"] = new
            n += 1
    d["imageWidth"], d["imageHeight"] = upright_size(o, W, H)
    return n


def rotate_yolo(text: str, o: int) -> str:
    """Normalised YOLO lines (detection: c cx cy w h · segmentation: c x y x y …) into the upright frame."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            out.append(line)
            continue
        if len(vals) == 4:
            cx, cy = map_point(o, vals[0], vals[1], 1.0, 1.0)
            w, h = (vals[3], vals[2]) if o in ROTATED else (vals[2], vals[3])
            out.append(" ".join([parts[0]] + [f"{v:.6f}" for v in (cx, cy, w, h)]))
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            pts = []
            for i in range(0, len(vals), 2):
                pts += map_point(o, vals[i], vals[i + 1], 1.0, 1.0)
            out.append(" ".join([parts[0]] + [f"{v:.6f}" for v in pts]))
        else:
            raise ValueError(f"unsupported YOLO line ({len(vals)} values) — pose / OBB labels are not rotated")
    return "\n".join(out) + ("\n" if out else "")


def run(P, log_dir, progress=None, cancel=None) -> dict:
    log = C.RunLog(log_dir, "exif_fix", keep_originals=True)
    st = {"done": 0, "labels": 0, "errors": [], "changed": [], "log": str(log.csv_path), "undo": str(log.jsonl_path)}
    try:
        for i, (_k, img, _t, ex) in enumerate(P.actions, 1):
            if cancel is not None and cancel.is_set():
                st["errors"].append("stopped by you")
                break
            o, (W, H) = ex["o"], ex["raw"]
            try:
                # labels first (all prepared before anything is written: a bad label file leaves the image alone)
                new_json = new_txt = None
                if ex["frame"] == "raw":
                    if ex.get("json"):
                        d = C.read_json(ex["json"])
                        if not isinstance(d, dict):
                            raise ValueError("label file is not a LabelMe JSON")
                        rotate_labelme(d, o, W, H)
                        new_json = C.dump_json(d)
                    if ex.get("txt"):
                        with open(ex["txt"], encoding="utf-8") as f:
                            new_txt = rotate_yolo(f.read(), o).encode("utf-8")
                orig = log.keep_original(img)
                bake(img)
                log.rewritten(img, orig, f"EXIF {ORIENT_TEXT[o]} baked into the pixels")
                st["changed"].append(img)
                for path, raw in ((ex.get("json"), new_json), (ex.get("txt"), new_txt)):
                    if raw is not None:
                        orig = log.keep_original(path)
                        C.write_atomic(path, raw)
                        log.rewritten(path, orig, f"points {ORIENT_TEXT[o]} with the image")
                        st["changed"].append(path)
                        st["labels"] += 1
                st["done"] += 1
            except Exception as exc:
                log.error("exif", img, "", exc)
                st["errors"].append(f"{img}: {type(exc).__name__}: {exc}")
            if progress is not None and (i % 10 == 0 or i == len(P.actions)):
                progress(i, len(P.actions), "Fixing rotation")
    finally:
        log.close()
    return st
