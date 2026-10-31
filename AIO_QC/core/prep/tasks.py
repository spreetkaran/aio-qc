# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
The smaller Data Prep tools (Qt-free). Each has plan(...) -> Plan (nothing changes) and run(plan) -> stats (logged,
undoable — see common.RunLog).

Unzip         every .zip under a folder (nested zips too, optional) -> a folder named after it; already-extracted
              archives skipped; corrupt / password-protected archives reported; zip-slip and zip-bomb guarded;
              optionally the archive goes to _extracted_zips afterwards.
Orphans       in place: pairs stay, JSONs without an image and images without a JSON move to sibling folders
              (sub-folder structure kept); pairing by same stem + folder, else imagePath; stem clashes reported.
Integrity     every image fully decoded (truncated JPEGs, 0-byte files, wrong signatures …), every JSON parsed and
              checked as LabelMe (shapes list, points numeric, imageWidth / imageHeight vs the real image, imagePath
              that leads nowhere) -> bad files (with their partner, optional) to a quarantine folder, structure kept.
Exact dupes   byte-identical images in a folder tree (whatever their names); kept copy: annotated > original name
              (fewest _1 / _2 suffixes) > oldest > A–Z; other copies + their JSON aside (or deleted); reports copies
              whose annotations DIFFER from the kept one.
Resize        large images downscaled (longest side / height limit / factor), JPEG / PNG / keep format, quality;
              LabelMe points and imageWidth / imageHeight scaled with them (imageData dropped); YOLO .txt copied
              unchanged (normalised); output folder with the same structure — sources untouched.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import common as C

WORKERS = max(4, min(32, (os.cpu_count() or 4) * 2))


@dataclass
class Plan:
    tool: str
    actions: list = field(default_factory=list)
    rows: list = field(default_factory=list)            # [(label, count)] summary
    details: dict = field(default_factory=lambda: defaultdict(list))
    opts: dict = field(default_factory=dict)


def _pmap(fn, items, progress, label, cancel, workers=WORKERS):
    lock = threading.Lock()
    done = [0]
    n = len(items)

    def w(it):
        if cancel is not None and cancel.is_set():
            raise C.Cancelled()
        r = fn(it)
        with lock:
            done[0] += 1
            if progress is not None and (done[0] % 25 == 0 or done[0] == n):
                progress(done[0], n, label)
        return r
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(w, items))


# =============================================================================================================== unzip
MAX_RATIO = 200          # uncompressed / compressed — beyond this an archive is treated as a zip bomb
MAX_TOTAL = 200 * 1024 ** 3


def unzip_plan(root, nested=True, progress=None, cancel=None) -> Plan:
    P = Plan("unzip", opts={"root": str(root), "nested": nested})
    _i, _j, zips, _o, sizes = C.scan([root], images=False, jsons=False, zips=True, cancel=cancel)
    skipped = 0
    for z in zips:
        d = Path(z).with_suffix("")
        if d.exists() and not d.is_dir():
            d = Path(z).parent / (Path(z).stem + "_unzipped")
        if d.is_dir() and any(d.iterdir()):
            skipped += 1
            P.details["Already extracted (skipped)"].append(str(z))
            continue
        P.actions.append((z, str(d)))
    P.rows = [("Archives found", len(zips)), ("To extract", len(P.actions)), ("Already extracted", skipped),
              ("Compressed size", C.human(sum(sizes.get(z, 0) for z, _d in P.actions)))]
    return P


def _extract(z, d, log):
    with zipfile.ZipFile(z) as zf:
        infos = zf.infolist()
        total = sum(i.file_size for i in infos)
        comp = sum(i.compress_size for i in infos) or 1
        if total > MAX_TOTAL or total / comp > MAX_RATIO and total > 1024 ** 3:
            raise ValueError(f"suspicious archive ({C.human(total)} from {C.human(comp)}) — not extracted")
        if any(i.flag_bits & 0x1 for i in infos):
            raise ValueError("password-protected")
        base = Path(d).resolve()
        for i in infos:
            name = i.filename.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/") or re.match(r"^[A-Za-z]:", name):
                raise ValueError(f"unsafe path inside: {i.filename}")
            t = (base / name).resolve()
            if not str(t).startswith(str(base)):
                raise ValueError(f"unsafe path inside: {i.filename}")
        base.mkdir(parents=True, exist_ok=True)
        log.write({"op": "create_dir", "dst": str(base)})
        zf.extractall(base)                             # CRCs are checked while extracting
        return len(infos), total, None


def unzip_run(P: Plan, move_archives=False, progress=None, cancel=None) -> dict:
    root = Path(P.opts["root"])
    log = C.RunLog(root / C.LOG_DIR, "unzip", keep_originals=False)
    st = {"extracted": 0, "files": 0, "bytes": 0, "errors": [], "nested": 0, "log": str(log.csv_path),
          "undo": str(log.jsonl_path)}
    queue = list(P.actions)
    seen = set()
    try:
        while queue:
            def one(a):
                z, d = a
                if cancel is not None and cancel.is_set():
                    raise C.Cancelled()
                try:
                    n, b, _ = _extract(z, d, log)
                    log.row("extract", z, d, f"{n} files, {C.human(b)}")
                    if move_archives:
                        t = C.unique_path(Path(z).parent / "_extracted_zips" / Path(z).name)
                        t.parent.mkdir(exist_ok=True)
                        shutil.move(z, t)
                        log.moved(z, t, "archive set aside")
                    return a, n, b, None
                except zipfile.BadZipFile:
                    return a, 0, 0, "corrupt / not a zip"
                except C.Cancelled:
                    raise
                except Exception as exc:
                    return a, 0, 0, str(exc)
            res = _pmap(one, queue, progress, "Extracting", cancel, workers=min(8, WORKERS))
            nxt = []
            for (z, d), n, b, err in res:
                seen.add(C.norm(z))
                if err:
                    st["errors"].append(f"{z}: {err}")
                    log.error("extract", z, d, err)
                    if os.path.isdir(d) and not any(os.scandir(d)):
                        os.rmdir(d)
                    continue
                st["extracted"] += 1
                st["files"] += n
                st["bytes"] += b
                if P.opts.get("nested"):
                    _i, _j, inner, _o, _s = C.scan([d], images=False, jsons=False, zips=True)
                    for iz in inner:
                        if C.norm(iz) not in seen:
                            nxt.append((iz, str(Path(iz).with_suffix(""))))
            st["nested"] += len(nxt)
            queue = [a for a in nxt if not (Path(a[1]).is_dir() and any(Path(a[1]).iterdir()))]
    except C.Cancelled:
        st["errors"].append("stopped by you")
    finally:
        log.close()
    return st


# =============================================================================================================== orphans
def orphans_plan(root, out=None, use_image_path=True, progress=None, cancel=None) -> Plan:
    root = os.path.abspath(root)
    out = os.path.abspath(out or root)
    jdir, idir = os.path.join(out, "jsons_without_images"), os.path.join(out, "images_without_jsons")
    P = Plan("orphans", opts={"root": root, "out": out})
    images, jsons, _z, _o, _s = C.scan([root], skip=[jdir, idir], cancel=cancel)
    index = defaultdict(list)
    for p in images:
        index[(C.norm(os.path.dirname(p)), C.stem(p).lower())].append(p)
    matched = set()

    def via_ip(js):
        try:
            ip = C.read_json(js).get("imagePath")
        except Exception:
            return None
        if not isinstance(ip, str) or not ip.strip():
            return None
        c = os.path.normpath(os.path.join(os.path.dirname(js), ip.replace("\\", "/")))
        return c if os.path.isfile(c) and C.ext(c).lower() in C.IMAGE_EXTS else None
    lonely = []
    for js in jsons:
        hits = index.get((C.norm(os.path.dirname(js)), C.stem(js).lower()), [])
        if hits:
            matched.update(C.norm(h) for h in hits)
        else:
            lonely.append(js)
    if use_image_path and lonely:
        refs = _pmap(via_ip, lonely, progress, "Reading imagePath", cancel)
    else:
        refs = [None] * len(lonely)
    orphan_j = []
    for js, ref in zip(lonely, refs):
        if ref:
            matched.add(C.norm(ref))
        else:
            orphan_j.append(js)
    orphan_i = [p for p in images if C.norm(p) not in matched]
    taken = set()
    for js in orphan_j:
        P.actions.append((js, str(C.unique_path(Path(jdir) / os.path.relpath(js, root), taken)), "JSON without image"))
    for p in orphan_i:
        P.actions.append((p, str(C.unique_path(Path(idir) / os.path.relpath(p, root), taken)), "image without JSON"))
    for (folder, s), imgs in index.items():
        if len(imgs) > 1:
            P.details["Several images share one name (all count as matched)"].append(
                f"{os.path.relpath(folder, root)}: {', '.join(os.path.basename(i) for i in imgs)}")
    P.rows = [("Images", len(images)), ("JSONs", len(jsons)), ("Matched pairs (stay in place)", len(jsons) - len(orphan_j)),
              ("JSONs without image → jsons_without_images", len(orphan_j)),
              ("Images without JSON → images_without_jsons", len(orphan_i))]
    P.details["JSONs without image"] = [os.path.relpath(j, root) for j in orphan_j]
    P.details["Images without JSON"] = [os.path.relpath(i, root) for i in orphan_i]
    return P


def move_run(P: Plan, root, copy=False, progress=None, cancel=None) -> dict:
    """Generic: P.actions = [(src, dst, note)] — move (or copy) each, logged."""
    log = C.RunLog(Path(root) / C.LOG_DIR, P.tool, keep_originals=False)
    st = {"done": 0, "errors": [], "log": str(log.csv_path), "undo": str(log.jsonl_path)}
    touched = set()
    lock = threading.Lock()

    def one(a):
        src, dst, note = a
        try:
            if os.path.exists(dst):
                raise FileExistsError(f"{dst} already exists")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if copy:
                shutil.copy2(src, dst)
                log.created(dst, note, src, "copy")
            else:
                shutil.move(src, dst)
                log.moved(src, dst, note)
            with lock:
                touched.add(os.path.dirname(src))
            return None
        except Exception as exc:
            log.error("move", src, dst, exc)
            return f"{src}: {exc}"
    try:
        for err in _pmap(one, P.actions, progress, "Copying" if copy else "Moving", cancel):
            if err:
                st["errors"].append(err)
            else:
                st["done"] += 1
    except C.Cancelled:
        st["errors"].append("stopped by you")
    finally:
        log.close()
    return st


# =============================================================================================================== integrity
def integrity_plan(root, quarantine=None, check_images=True, check_jsons=True, with_partner=True, progress=None,
                   cancel=None) -> Plan:
    root = os.path.abspath(root)
    q = os.path.abspath(quarantine or (root.rstrip("\\/") + "_quarantine"))
    P = Plan("integrity", opts={"root": root, "quarantine": q})
    images, jsons, _z, _o, _s = C.scan([root], skip=[q], cancel=cancel)
    bad = {}                                            # path -> reason
    sizes = {}
    if check_images:
        for p, (ok, why, wh) in zip(images, _pmap(C.verify_file, images, progress, "Decoding images", cancel)):
            if not ok:
                bad[p] = why
            elif wh:
                sizes[C.norm(p)] = wh

    def check_json(p):
        try:
            d = C.read_json(p)
        except Exception as exc:
            return f"invalid JSON ({type(exc).__name__})"
        if not isinstance(d, dict):
            return "not a LabelMe JSON (top level is not an object)"
        sh = d.get("shapes")
        if not isinstance(sh, list):
            return "not a LabelMe JSON (no 'shapes' list)"
        for i, s in enumerate(sh):
            if not isinstance(s, dict) or not isinstance(s.get("points"), list):
                return f"shape #{i}: no points list"
            try:
                for pt in s["points"]:
                    float(pt[0]), float(pt[1])
            except (TypeError, ValueError, IndexError):
                return f"shape #{i}: points are not numbers"
            if s.get("label") in (None, ""):
                return f"shape #{i}: empty label"
        ip = d.get("imagePath")
        img = None
        if isinstance(ip, str) and ip.strip():
            c = os.path.normpath(os.path.join(os.path.dirname(p), ip.replace("\\", "/")))
            if os.path.isfile(c):
                img = c
        if img is None:
            for x in C.IMAGE_EXTS:
                c = os.path.join(os.path.dirname(p), C.stem(p) + x)
                if os.path.isfile(c):
                    img = c
                    break
        wh = sizes.get(C.norm(img)) if img else None
        try:
            W, H = int(d.get("imageWidth") or 0), int(d.get("imageHeight") or 0)
        except (TypeError, ValueError):
            return "imageWidth / imageHeight are not numbers"
        if wh and W and H and (W, H) != tuple(wh):
            return f"size {W}×{H} in the JSON, image is {wh[0]}×{wh[1]}"
        if img is None and not (isinstance(d.get("imageData"), str) and d["imageData"].strip()):
            return "no image file and no imageData"
        return None
    if check_jsons:
        for p, why in zip(jsons, _pmap(check_json, jsons, progress, "Checking JSONs", cancel)):
            if why:
                bad[p] = why
    taken = set()
    moved = set()
    by_stem = defaultdict(list)
    for p in images + jsons:
        by_stem[(C.norm(os.path.dirname(p)), C.stem(p).lower())].append(p)
    reasons = Counter()
    for p, why in sorted(bad.items()):
        group = [p] + ([x for x in by_stem[(C.norm(os.path.dirname(p)), C.stem(p).lower())] if x != p]
                       if with_partner else [])
        for x in group:
            if C.norm(x) in moved:
                continue
            moved.add(C.norm(x))
            note = why if x == p else f"partner of {os.path.basename(p)} ({why})"
            P.actions.append((x, str(C.unique_path(Path(q) / os.path.relpath(x, root), taken)), note))
        reasons[why.split(" (")[0].split(":")[0]] += 1
        P.details["Bad files"].append(f"{os.path.relpath(p, root)}  —  {why}")
    P.rows = [("Images checked", len(images) if check_images else 0), ("JSONs checked", len(jsons) if check_jsons else 0),
              ("Bad files", len(bad)), ("Files to quarantine (with partners)", len(P.actions))]
    P.rows += [(f"    {k}", v) for k, v in reasons.most_common(12)]
    return P


# =============================================================================================================== exact dupes
SUFFIX = re.compile(r"^(.*)_(\d+)$")


def _suffix_depth(s):
    d = 0
    while SUFFIX.match(s):
        s = SUFFIX.match(s).group(1)
        d += 1
    return d


def dupes_plan(root, dup_dir=None, progress=None, cancel=None) -> Plan:
    root = os.path.abspath(root)
    dd = os.path.abspath(dup_dir or (root.rstrip("\\/") + "_duplicates"))
    P = Plan("dupes", opts={"root": root, "dup_dir": dd})
    images, jsons, _z, _o, sizes = C.scan([root], skip=[dd], cancel=cancel)
    js = defaultdict(list)
    for j in jsons:
        js[(C.norm(os.path.dirname(j)), C.stem(j).lower())].append(j)
    partner = {}
    for p in images:
        hits = js.get((C.norm(os.path.dirname(p)), C.stem(p).lower()))
        if hits:
            partner[p] = hits[0]
    groups = C.exact_groups(images, lambda p: sizes.get(p), lambda p: p, None, WORKERS, cancel,
                            lambda d, t: progress(d, t, "Comparing same-size images") if progress else None)

    def rank(p):
        try:
            mt = os.path.getmtime(p)
        except OSError:
            mt = 0
        return (0 if p in partner else 1, _suffix_depth(C.stem(p)), mt, os.path.basename(p).lower())

    def lab(j):
        try:
            return C.label_digest(C.read_json(j))
        except Exception:
            return None
    kinds = Counter()
    taken = set()
    nbytes = 0
    for g in groups:
        g.sort(key=rank)
        keep = g[0]
        for p in g[1:]:
            j = partner.get(p)
            if not j:
                kind = "this copy has no JSON"
            elif keep not in partner:
                kind = "kept copy has no JSON"
            else:
                kind = "identical annotations" if lab(j) == lab(partner[keep]) else "DIFFERENT annotations (review)"
            kinds[kind] += 1
            s = None
            for f in [p] + ([j] if j else []):
                t = C.unique_path(Path(dd) / os.path.relpath(f, root), taken) if s is None else \
                    Path(dd) / os.path.relpath(os.path.dirname(f), root) / (s + C.ext(f))
                s = s or t.stem
                P.actions.append((f, str(t), f"{kind}; kept: {os.path.relpath(keep, root)}"))
                nbytes += sizes.get(f, 0) or (os.path.getsize(f) if os.path.exists(f) else 0)
            sec = "Duplicates with DIFFERENT annotations — review" if kind.startswith("DIFF") else "Other duplicates"
            P.details[sec].append(f"{os.path.relpath(p, root)}  =  {os.path.relpath(keep, root)}")
    P.rows = [("Images checked", len(images)), ("Images that have copies", len(groups)),
              ("Duplicate copies (exact same bytes)", sum(kinds.values()))]
    P.rows += [(f"    {k}", v) for k, v in kinds.most_common()]
    P.rows.append(("Space they take (images + JSONs)", C.human(nbytes)))
    return P


# =============================================================================================================== resize
@dataclass
class ResizeOptions:
    src: str
    dst: str
    mode: str = "long"            # long (longest side ≤ limit) · height (height ≥ limit → ×factor) · factor (all)
    limit: int = 1920
    factor: float = 0.5
    fmt: str = "keep"             # keep · jpg · png
    quality: int = 90
    copy_small: bool = True       # images that need no resize: copy as they are (else re-encode to fmt)
    with_annotations: bool = True


def resize_plan(o: ResizeOptions, progress=None, cancel=None) -> Plan:
    src, dst = os.path.abspath(o.src), os.path.abspath(o.dst)
    if C.inside(dst, src) or C.norm(dst) == C.norm(src):
        raise ValueError("Choose an output folder outside the source folder.")
    P = Plan("resize", opts={"o": o})
    images, jsons, _z, _o, sizes = C.scan([src], cancel=cancel)
    from .. import io as cio

    def info(p):
        return cio.image_size(p)
    whs = _pmap(info, images, progress, "Reading image sizes", cancel)
    jby = defaultdict(list)
    for j in jsons:
        jby[(C.norm(os.path.dirname(j)), C.stem(j).lower())].append(j)
    txts = []
    if o.with_annotations:
        for dp, _dn, fn in os.walk(src):
            if C.LOG_DIR in dp:
                continue
            txts += [os.path.join(dp, f) for f in fn if f.lower().endswith(".txt") or f.lower() in
                     ("data.yaml", "dataset.yaml")]
    n_resize = 0
    before = after = 0
    for p, wh in zip(images, whs):
        if wh is None:
            P.details["Unreadable images (skipped)"].append(os.path.relpath(p, src))
            continue
        w, h = wh
        if o.mode == "long":
            s = min(1.0, o.limit / max(w, h))
        elif o.mode == "height":
            s = o.factor if h >= o.limit else 1.0
        else:
            s = o.factor
        x = C.ext(p) if o.fmt == "keep" else (".jpg" if o.fmt == "jpg" else ".png")
        t = os.path.join(dst, os.path.relpath(os.path.dirname(p), src), C.stem(p) + x)
        j = jby.get((C.norm(os.path.dirname(p)), C.stem(p).lower()), [None])[0] if o.with_annotations else None
        P.actions.append(("img", p, t, {"scale": s, "wh": (w, h), "json": j}))
        n_resize += s < 1.0
        before += w * h
        after += int(w * s) * int(h * s)
    paired = {a[3]["json"] for a in P.actions if a[3].get("json")}
    for j in jsons if o.with_annotations else []:
        if j not in paired:
            P.actions.append(("copy", j, os.path.join(dst, os.path.relpath(j, src)), {}))
    for t in txts:
        P.actions.append(("copy", t, os.path.join(dst, os.path.relpath(t, src)), {}))
    P.rows = [("Images", len(images)), ("Downscaled", n_resize), ("Copied / re-encoded as they are",
                                                                  len(images) - n_resize),
              ("Pixels after / before", f"{after / max(1, before):.0%}"),
              ("LabelMe JSONs scaled with their image", len(paired)),
              ("Other annotation files copied (YOLO .txt, data.yaml, lone JSONs)",
               sum(1 for a in P.actions if a[0] == "copy"))]
    return P


def resize_run(P: Plan, progress=None, cancel=None) -> dict:
    o: ResizeOptions = P.opts["o"]
    log = C.RunLog(Path(o.dst) / C.LOG_DIR, "resize", keep_originals=False)
    st = {"done": 0, "resized": 0, "errors": [], "bytes": [0, 0], "log": str(log.csv_path), "undo": str(log.jsonl_path)}
    lock = threading.Lock()

    def one(a):
        kind, src, t, ex = a
        try:
            if os.path.exists(t):
                raise FileExistsError(f"{t} already exists")
            os.makedirs(os.path.dirname(t), exist_ok=True)
            if kind == "copy":
                shutil.copy2(src, t)
                log.created(t, "copied", src, "copy")
                return None
            s = ex["scale"]
            same_fmt = C.SAME_FORMAT.get(C.ext(src).lower(), C.ext(src).lower()) == \
                C.SAME_FORMAT.get(C.ext(t).lower(), C.ext(t).lower())
            if s >= 1.0 and same_fmt and o.copy_small:
                shutil.copy2(src, t)
                note = "copied (no resize needed)"
            else:
                _encode(src, t, s, o)
                note = f"{ex['wh'][0]}×{ex['wh'][1]} → ×{s:.3g}"
            log.created(t, note, src, "image")
            with lock:
                st["bytes"][0] += os.path.getsize(src)
                st["bytes"][1] += os.path.getsize(t)
                st["resized"] += s < 1.0
            j = ex.get("json")
            if j:
                jt = os.path.join(os.path.dirname(t), C.stem(t) + ".json")
                if os.path.exists(jt):
                    raise FileExistsError(f"{jt} already exists")
                d = C.read_json(j)
                if isinstance(d, dict):
                    nw, nh = _scaled_size(ex["wh"], s)
                    sx, sy = nw / ex["wh"][0], nh / ex["wh"][1]
                    for sh in d.get("shapes") or []:
                        pts = sh.get("points")
                        if isinstance(pts, list):
                            sh["points"] = [[round(float(p[0]) * sx, 2), round(float(p[1]) * sy, 2)] for p in pts]
                    d["imageWidth"], d["imageHeight"] = nw, nh
                    d["imagePath"] = os.path.basename(t)
                    d["imageData"] = None
                C.write_atomic(jt, C.dump_json(d))
                log.created(jt, "points scaled", j, "json")
            return None
        except Exception as exc:
            log.error(kind, src, t, exc)
            return f"{src}: {exc}"
    try:
        for err in _pmap(one, P.actions, progress, "Resizing", cancel, workers=max(2, (os.cpu_count() or 2))):
            if err:
                st["errors"].append(err)
            else:
                st["done"] += 1
    except C.Cancelled:
        st["errors"].append("stopped by you")
    finally:
        log.close()
    return st


def _scaled_size(wh, s):
    return max(1, int(round(wh[0] * s))), max(1, int(round(wh[1] * s)))


def _encode(src, dst, s, o: ResizeOptions):
    """Decode (EXIF orientation applied, like the viewer), resize with area filter, encode."""
    import cv2
    import numpy as np
    img = None
    try:
        from PIL import Image, ImageOps
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode not in ("RGB", "RGBA", "L"):
                im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            img = np.asarray(im)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA if img.shape[2] == 4 else cv2.COLOR_RGB2BGR)
    except ImportError:
        img = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("cannot decode")
    h, w = img.shape[:2]
    if s < 1.0:
        nw, nh = _scaled_size((w, h), s)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x = C.ext(dst).lower()
    if x in (".jpg", ".jpeg"):
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(o.quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    elif x == ".png":
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    elif x == ".webp":
        ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, int(o.quality)])
    else:
        ok, buf = cv2.imencode(x, img)
    if not ok:
        raise ValueError(f"cannot encode {x}")
    C.write_atomic(dst, buf.tobytes())
