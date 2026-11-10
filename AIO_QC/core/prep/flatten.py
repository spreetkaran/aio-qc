# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Combine & Flatten (Qt-free): one or more nested folders of images + LabelMe JSONs -> one flat dataset folder,
without duplicate images.

Pairing, per JSON (first hit wins):
    a) imagePath, resolved relative to the JSON      b) same stem, same folder
    c) the imagePath file name anywhere (if unique)  d) the JSON's stem anywhere (if unique)
Then:
    pair                -> flat folder; imagePath = the image's new name; imageData null (optional: kept)
    JSON w/o image, imageData decodes -> the image is RESTORED from imageData next to it (fully verified)
    other JSON w/o image (not LabelMe / invalid / no or corrupt imageData) -> <orphans>/jsons
    image without JSON  -> <orphans>/images  (or flat folder, as background images)
    second JSON pointing at an image another JSON already uses -> <orphans>/duplicates
Exact duplicate images (by content, whatever their names — files, embedded imageData and what the flat folder
already holds) are added once. Kept copy: already in the flat folder > annotated > real file > restored from
imageData > first. Every other copy goes to <orphans>/duplicates WITH its own JSON, unchanged; the plan says whether
its labels are identical to the kept copy's or DIFFERENT (review). Name clashes of different images get _1, _2 ….
Everything is logged (CSV + undo log); empty source folders are removed after a move.
"""
from __future__ import annotations

import os
import shutil
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import common as C

R_NOT_LABELME = "not a LabelMe JSON"
R_INVALID = "unreadable / invalid JSON"
R_NO_DATA = "LabelMe, no image and no imageData"
R_BAD_DATA = "LabelMe, imageData could not be decoded"
D_SAME = "same image, identical annotations"
D_DIFF = "same image, DIFFERENT annotations (review)"
D_NO_CMP = "same image, kept copy has no annotation"
D_NO_JSON = "same image, this copy has no annotation"


@dataclass
class Options:
    sources: list
    dst: str
    orphans: str = ""                     # default <dst>_orphans
    copy: bool = True                     # copy (sources untouched) or move
    null_image_data: bool = True          # imageData -> null in the flat folder
    restore_images: bool = True           # JSON without image file: rebuild the image from imageData
    dedupe: bool = True
    background_images: bool = False       # images without JSON -> flat folder (negatives) instead of orphans
    remove_empty_dirs: bool = True
    workers: int = max(4, min(32, (os.cpu_count() or 4) * 2))

    def orph(self):
        return self.orphans or (self.dst.rstrip("\\/") + "_orphans")


@dataclass
class Plan:
    opts: Options
    actions: list = field(default_factory=list)       # (kind, src, target, extra)
    stats: Counter = field(default_factory=Counter)
    how: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    dup_kinds: Counter = field(default_factory=Counter)
    details: dict = field(default_factory=lambda: defaultdict(list))   # section -> [text]
    other: Counter = field(default_factory=Counter)

    def summary_rows(self):
        s = self.stats
        rows = [("Image + JSON pairs → flat folder", s["pairs"])]
        rows += [(f"    matched by {k}", v) for k, v in self.how.most_common()]
        rows += [("    already clean (moved / copied without rewrite)", s["clean"]),
                 ("Images restored from imageData → flat folder", s["restore"]),
                 ("Images without JSON → " + ("flat folder (background)" if self.opts.background_images
                                              else "orphans/images"), s["orphan_img"]),
                 ("JSONs without image → orphans/jsons", s["orphan_json"])]
        rows += [(f"    {k}", v) for k, v in self.reasons.most_common()]
        rows += [("Duplicate images (exact same bytes) → orphans/duplicates", s["dups"])]
        rows += [(f"    {k}", v) for k, v in self.dup_kinds.most_common()]
        rows += [("imagePath corrected", s["ip_fixed"]), ("Renamed (different images, same name)", s["renamed"])]
        if self.other:
            rows.append(("Other files (left in place)", sum(self.other.values())))
        return rows


def _inspect(path):
    try:
        data = C.read_json(path)
    except Exception:
        return {"kind": "invalid"}
    if not isinstance(data, dict) or "shapes" not in data:
        return {"kind": "notlabelme"}
    ip = data.get("imagePath")
    idata = data.get("imageData")
    has = isinstance(idata, str) and bool(idata.strip())
    return {"kind": "labelme", "ip": ip if isinstance(ip, str) and ip.strip() else None,
            "null": "imageData" in data and idata is None, "has": has, "data_ext": C.peek_image_data(idata),
            "data_size": C.b64_size(idata) if has else None, "label": C.label_digest(data)}


def plan(opts: Options, progress=None, cancel=None) -> Plan:
    P = Plan(opts)
    dst, orph = os.path.abspath(opts.dst), os.path.abspath(opts.orph())
    o_img, o_json, o_dup = (os.path.join(orph, n) for n in ("images", "jsons", "duplicates"))
    srcs = [os.path.abspath(s) for s in opts.sources]
    for s in srcs:
        if C.norm(s) == C.norm(dst):
            raise ValueError("The flat folder must be different from the source folders.")
        if C.inside(s, dst) or C.inside(s, orph):
            raise ValueError(f"{s} is inside the output / orphan folder.")
    if C.norm(orph) == C.norm(dst):
        raise ValueError("The orphan folder must be different from the flat folder.")
    say = progress or (lambda *_: None)
    say(0, 0, "Scanning folders")
    images, jsons, _z, other, sizes = C.scan(srcs, skip=[dst, orph], cancel=cancel)
    P.other = other
    by_key = {C.norm(p): p for p in images}
    by_name, by_stem = defaultdict(list), defaultdict(list)
    for p in images:
        by_name[os.path.basename(p).lower()].append(p)
        by_stem[C.stem(p).lower()].append(p)

    # ---- inspect JSONs (threads: orjson / file reads release the GIL much of the time)
    infos = []
    lock = threading.Lock()
    done = [0]

    def insp(p):
        if cancel is not None and cancel.is_set():
            raise C.Cancelled()
        r = _inspect(p)
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                say(done[0], len(jsons), "Reading JSONs")
        return r
    with ThreadPoolExecutor(opts.workers) as ex:
        infos = list(ex.map(insp, jsons))

    claimed, second, pairs, restorable, orphan_jsons = {}, [], [], [], []
    for jp, info in zip(jsons, infos):
        if info["kind"] == "invalid":
            orphan_jsons.append((jp, R_INVALID))
            continue
        if info["kind"] == "notlabelme":
            orphan_jsons.append((jp, R_NOT_LABELME))
            continue
        jdir, ip = os.path.dirname(jp), info["ip"]
        img = how = None
        if ip:
            k = C.norm(os.path.join(jdir, ip.replace("\\", os.sep).replace("/", os.sep)))
            if k in by_key:
                img, how = by_key[k], "imagePath"
        if img is None:
            jn = C.norm(jdir)
            for p in by_stem.get(C.stem(jp).lower(), []):
                if C.norm(os.path.dirname(p)) == jn:
                    img, how = p, "same name, same folder"
                    break
        if img is None and ip:
            hits = by_name.get(C.ip_basename(ip).lower(), [])
            if len(hits) == 1:
                img, how = hits[0], "imagePath name, other folder"
        if img is None:
            hits = by_stem.get(C.stem(jp).lower(), [])
            if len(hits) == 1:
                img, how = hits[0], "same name, other folder"
        if img is not None:
            k = C.norm(img)
            if k in claimed:
                second.append((jp, info) + claimed[k])
                continue
            claimed[k] = (jp, info)
            pairs.append((jp, img, info))
            P.how[how] += 1
        elif info["has"] and info["data_ext"] and opts.restore_images:
            restorable.append((jp, info))
        elif info["has"]:
            orphan_jsons.append((jp, R_BAD_DATA if not info["data_ext"] else R_NO_DATA + " (restore off)"))
        else:
            orphan_jsons.append((jp, R_NO_DATA))
    orphan_images = [p for p in images if C.norm(p) not in claimed]

    # ---- exact duplicates
    cands = []
    for folder, kind in ((dst, "existing"), (o_img, "existing_orphan")):
        if os.path.isdir(folder):
            with os.scandir(folder) as it:
                for e in it:
                    if e.is_file() and C.ext(e.name).lower() in C.IMAGE_EXTS:
                        js = os.path.join(dst, C.stem(e.name) + ".json") if kind == "existing" else None
                        cands.append({"kind": kind, "img": e.path, "json": js if js and os.path.isfile(js) else None,
                                      "size": e.stat().st_size, "label": None})
    n_exist = len(cands)
    for jp, img, info in pairs:
        cands.append({"kind": "pair", "img": img, "json": jp, "size": sizes.get(img), "label": info["label"]})
    for jp, info in restorable:
        cands.append({"kind": "restore", "img": None, "json": jp, "size": info["data_size"], "label": info["label"]})
    for p in orphan_images:
        cands.append({"kind": "orphan_img", "img": p, "json": None, "size": sizes.get(p), "label": None})
    dup_of = {}
    if opts.dedupe:
        say(0, 0, "Finding duplicate images")
        rank_k = {"existing": 0, "existing_orphan": 0, "pair": 1, "restore": 2, "orphan_img": 3}

        def digest_embedded(c):
            import hashlib
            try:
                return hashlib.sha256(C.b64_bytes(C.read_json(c["json"]).get("imageData") or "")).hexdigest()
            except Exception:
                return None
        idx = list(range(len(cands)))
        groups = C.exact_groups(idx, lambda i: cands[i]["size"], lambda i: cands[i]["img"],
                                lambda i: digest_embedded(cands[i]), opts.workers, cancel,
                                lambda d, t: say(d, t, "Comparing same-size images"))
        for g in groups:
            g.sort(key=lambda i: (0 if cands[i]["kind"] == "existing" else 1, 0 if cands[i]["json"] else 1,
                                  0 if cands[i]["kind"] == "existing_orphan" else 1, rank_k[cands[i]["kind"]], i))
            for i in g[1:]:
                dup_of[i] = g[0]

    def label_of(c):
        if c["label"] is None and c["json"] and c["kind"] == "existing":
            c["label"] = (_inspect(c["json"]) or {}).get("label")
        return c["label"]

    def dup_kind(c, keep):
        if c["json"] is None:
            return D_NO_JSON
        if keep["json"] is None:
            return D_NO_CMP
        return D_SAME if label_of(c) == label_of(keep) else D_DIFF

    def rel(p):
        for s in srcs:
            if p and C.inside(p, s):
                return os.path.relpath(p, s) if len(srcs) == 1 else os.path.join(os.path.basename(s),
                                                                                    os.path.relpath(p, s))
        return os.path.basename(p or "?") + " (already in the output)"

    # ---- names + actions
    n_dst, n_oi, n_oj, n_dup = C.Names(dst), C.Names(o_img), C.Names(o_json), C.Names(o_dup)
    A = P.actions

    def plan_dup(files, keep, kind):
        s = n_dup.get(C.stem(files[0]))
        A.append(("dup", files[0], None, {"files": [(p, os.path.join(o_dup, s + C.ext(p))) for p in files],
                                          "of": rel(keep["img"] or keep["json"]), "kind": kind}))
        P.dup_kinds[kind] += 1
        P.stats["dups"] += 1
        sec = "Duplicates with DIFFERENT annotations — review which to keep" if kind == D_DIFF else "Other duplicates"
        P.details[sec].append(f"{rel(files[0])}  =  {rel(keep['img'] or keep['json'])}")

    ci = n_exist
    for jp, img, info in pairs:
        if ci in dup_of:
            plan_dup([jp, img], cands[dup_of[ci]], dup_kind(cands[ci], cands[dup_of[ci]]))
        else:
            s = n_dst.get(C.stem(img))
            img_dst = os.path.join(dst, s + C.ext(img))
            new = os.path.basename(img_dst)
            if info["ip"] != new:
                P.stats["ip_fixed"] += 1
                P.details["imagePath corrected"].append(f"{rel(jp)}: {info['ip']!r} → {new!r}")
            clean = info["ip"] == new and (info["null"] or not opts.null_image_data)
            P.stats["clean"] += clean
            P.stats["pairs"] += 1
            A.append(("pair_clean" if clean else "pair", jp, os.path.join(dst, s + ".json"),
                      {"img_src": img, "img_dst": img_dst}))
        ci += 1
    for jp, info in restorable:
        if ci in dup_of:
            plan_dup([jp], cands[dup_of[ci]], dup_kind(cands[ci], cands[dup_of[ci]]))
        else:
            ip = info["ip"]
            base = C.stem(C.ip_basename(ip)) if ip else C.stem(jp)
            x = info["data_ext"]
            if ip:
                ie = C.ext(C.ip_basename(ip)).lower()
                if C.SAME_FORMAT.get(ie, ie) == x:
                    x = C.ext(C.ip_basename(ip))
            s = n_dst.get(base)
            P.stats["restore"] += 1
            A.append(("restore", jp, os.path.join(dst, s + ".json"), {"img_dst": os.path.join(dst, s + x)}))
        ci += 1
    for p in orphan_images:
        if ci in dup_of:
            plan_dup([p], cands[dup_of[ci]], D_NO_JSON)
        else:
            if opts.background_images:
                s = n_dst.get(C.stem(p))
                A.append(("orphan_img", p, os.path.join(dst, s + C.ext(p)), {"note": "background image"}))
            else:
                s = n_oi.get(C.stem(p))
                A.append(("orphan_img", p, os.path.join(o_img, s + C.ext(p)), {}))
            P.stats["orphan_img"] += 1
        ci += 1
    for jp, info, first_jp, first_info in second:
        kind = D_SAME if info["label"] == first_info["label"] else D_DIFF
        s = n_dup.get(C.stem(jp))
        A.append(("dup", jp, None, {"files": [(jp, os.path.join(o_dup, s + ".json"))], "of": rel(first_jp),
                                    "kind": kind + " (second JSON for one image)"}))
        P.dup_kinds[kind + " (second JSON for one image)"] += 1
        P.stats["dups"] += 1
    for jp, reason in orphan_jsons:
        s = n_oj.get(C.stem(jp))
        A.append(("orphan_json", jp, os.path.join(o_json, s + ".json"), {"reason": reason}))
        P.reasons[reason] += 1
        P.stats["orphan_json"] += 1
        P.details["JSONs without image"].append(f"{rel(jp)}  ({reason})")
    P.stats["renamed"] = n_dst.renamed + n_oi.renamed + n_oj.renamed
    P.stats["images"], P.stats["jsons"] = len(images), len(jsons)
    return P


def run(P: Plan, progress=None, cancel=None) -> dict:
    o = P.opts
    dst, orph = os.path.abspath(o.dst), os.path.abspath(o.orph())
    for d in (dst,):
        os.makedirs(d, exist_ok=True)
    log = C.RunLog(os.path.join(dst, C.LOG_DIR), "flatten", keep_originals=not o.copy)
    transfer = shutil.copy2 if o.copy else shutil.move
    done, errors, late = Counter(), [], []
    touched = set()
    lock = threading.Lock()
    size = [0, 0]

    def put(src, tgt, note, action):
        if os.path.exists(tgt):
            raise FileExistsError(f"{tgt} already exists")
        os.makedirs(os.path.dirname(tgt), exist_ok=True)
        transfer(src, tgt)
        (log.created(tgt, note, src, "copy") if o.copy else log.moved(src, tgt, note, action))

    def one(a):
        if cancel is not None and cancel.is_set():
            raise C.Cancelled()
        kind, src, tgt, ex = a
        try:
            if kind in ("pair", "pair_clean", "restore"):
                for t in (ex["img_dst"], tgt):
                    if os.path.exists(t):
                        raise FileExistsError(f"{t} already exists")
            if kind == "pair_clean":
                put(ex["img_src"], ex["img_dst"], "paired", "image")
                put(src, tgt, "already clean", "json")
            elif kind in ("pair", "restore"):
                data = C.read_json(src)
                if kind == "restore":
                    try:
                        raw, _x = C.decode_image_data(data.get("imageData") or "")
                    except ValueError as exc:
                        return "late", a, str(exc)
                    C.write_atomic(ex["img_dst"], raw)
                    log.created(ex["img_dst"], f"decoded from imageData of {src}", src, "restore")
                else:
                    put(ex["img_src"], ex["img_dst"], "paired", "image")
                before = os.path.getsize(src)
                data["imagePath"] = os.path.basename(ex["img_dst"])
                if o.null_image_data:
                    data["imageData"] = None
                out = C.dump_json(data)
                del data
                C.write_atomic(tgt, out)
                with lock:
                    size[0] += before
                    size[1] += len(out)
                if o.copy:
                    log.created(tgt, "imagePath fixed" + (", imageData → null" if o.null_image_data else ""), src,
                                "json")
                else:
                    orig = log.keep_original(src)
                    os.remove(src)
                    log.moved_rewritten(src, tgt, orig, "imagePath fixed" +
                                        (", imageData → null" if o.null_image_data else ""))
            elif kind == "dup":
                for fs, ft in ex["files"]:
                    if os.path.exists(ft):
                        raise FileExistsError(f"{ft} already exists")
                for fs, ft in ex["files"]:
                    put(fs, ft, f"{ex['kind']}; kept: {ex['of']}", "duplicate")
            else:
                put(src, tgt, ex.get("reason") or ex.get("note") or "orphan image", kind)
            with lock:
                done[kind.replace("_clean", "")] += 1
                touched.add(os.path.dirname(src))
                if "img_src" in ex:
                    touched.add(os.path.dirname(ex["img_src"]))
                for fs, _t in ex.get("files", ()):
                    touched.add(os.path.dirname(fs))
            return "ok", a, None
        except C.Cancelled:
            raise
        except Exception as exc:
            log.error(kind, src, tgt, exc)
            return "error", a, f"{type(exc).__name__}: {exc}"

    n = len(P.actions)
    k = 0
    try:
        with ThreadPoolExecutor(o.workers) as ex:
            for status, a, err in ex.map(one, P.actions):
                k += 1
                if status == "late":
                    late.append((a[1], err))
                elif status == "error":
                    errors.append(f"{a[1]}: {err}")
                if progress is not None and (k % 25 == 0 or k == n):
                    progress(k, n, "Copying files" if o.copy else "Moving files")
        names = C.Names(os.path.join(orph, "jsons"))
        for s, err in late:                       # imageData that did not decode after all -> orphans/jsons
            t = os.path.join(orph, "jsons", names.get(C.stem(s)) + ".json")
            try:
                put(s, t, f"{R_BAD_DATA}: {err}", "orphan_json")
                done["orphan_json"] += 1
            except Exception as exc:
                errors.append(f"{s}: {exc}")
        removed = 0
        if not o.copy and o.remove_empty_dirs:
            for s in o.sources:
                removed += C.remove_empty_dirs([d for d in touched if C.inside(d, s)], s, (dst, orph))
    except C.Cancelled:
        errors.append("stopped by you — what was done so far is in the log (and can be undone)")
    finally:
        log.close()
    return {"done": done, "errors": errors, "late": late, "removed_dirs": locals().get("removed", 0),
            "json_bytes": tuple(size), "log": str(log.csv_path), "undo": str(log.jsonl_path)}
