# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Auto-Annotate: videos -> unique frames -> LabelMe files -> the frames you need -> parts ready to upload (Qt-free).

Five steps, each optional and each re-runnable on what the earlier ones left in the run folder:

    frames    videos (or an image folder) -> frames/<video>/...            skips repeats within / across videos
    similar   frames that look like one already kept -> similar/...        nothing is deleted
    annotate  the model -> one LabelMe file next to every frame             5.11.4 or 6.3.1, imageData optional
    rules     frames without the classes you need -> extra_images/...      or kept as they are
    package   package/<folder>/part_001/ (+ part_001.zip)                   hard links (default) or copies

The run folder:

    <run>/frames/           the working set: images + their .json (and .txt)
    <run>/similar/          set aside by "similar"     (same sub-folders as frames/)
    <run>/extra_images/     set aside by "rules"
    <run>/package/          built by "package" (rebuilt each time)
    <run>/reports/          summary.json + summary.html
    <run>/.autoannotate/    state.json (what ran, with which settings), files.db (what this run wrote and moved)

Running a step again first puts back what that step and the later ones set aside, so a new threshold or rule is
applied to the whole set again. Files are only ever moved inside the run folder; annotation files the run did not
write, or that were edited after it wrote them, are left alone unless "replace" is chosen. Stopping is always safe:
the next run continues (extraction from its checkpoints, annotation from the files already written).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

from AIO_QC.tools.frame_extractor.core.autoannotate import labelme_io as LM
from AIO_QC.tools.frame_extractor.core.autoannotate.recipe import (Recipe, SIMILAR_PRESETS, STAGES,
                                                                       STAGE_TITLES)
from AIO_QC.core.diag import ignored as _ignored

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".webm", ".wmv", ".mpeg", ".mpg", ".m4v", ".ts", ".mts",
              ".m2ts", ".3gp")
FRAMES, SIMILAR, EXTRA, PACKAGE, REPORTS, META = ("frames", "similar", "extra_images", "package", "reports",
                                                    ".autoannotate")
REJECTED = "rejected"
PACKAGE_MARK = ".autoannotate-package"
UNMATCHED = "_unmatched"


class Stopped(Exception):
    """The user stopped the run, or a limit (budget) did (everything done so far is kept)."""


class StepError(Exception):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


class NeedsConfirmation(Exception):
    """Re-extracting with different settings would replace the previous frames (and everything made from them)."""

    def __init__(self, message, count):
        super().__init__(message)
        self.count = count


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _patterns(text: str):
    return [p.strip() for p in re.split(r"[;,\n]", text or "") if p.strip()]


def _matches(name: str, pats) -> bool:
    n = name.lower()
    return any(fnmatch.fnmatch(n, p.lower() if any(c in p for c in "*?[") else f"*{p.lower()}*") for p in pats)


def find_videos(folder: str, recursive=True, include="", exclude="") -> list[str]:
    inc, exc = _patterns(include), _patterns(exclude)
    out = []
    if os.path.isfile(folder):
        return [folder] if folder.lower().endswith(VIDEO_EXTS) else []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for f in files:
            if f.startswith(".") or not f.lower().endswith(VIDEO_EXTS):
                continue
            if inc and not _matches(f, inc) or exc and _matches(f, exc):
                continue
            out.append(os.path.join(root, f))
        if not recursive:
            break
    return sorted(out, key=lambda p: natural_key(os.path.relpath(p, folder)))


def find_images(folder: str, recursive=True, include="", exclude="", skip=()) -> list[str]:
    inc, exc = _patterns(include), _patterns(exclude)
    out = []
    if not os.path.isdir(folder):
        return out
    skip = {os.path.normcase(os.path.abspath(s)) for s in skip}
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".")
                         and os.path.normcase(os.path.abspath(os.path.join(root, d))) not in skip)
        for f in files:
            if f.startswith(".") or not f.lower().endswith(IMAGE_EXTS) or ".tmp" in f:
                continue
            if inc and not _matches(f, inc) or exc and _matches(f, exc):
                continue
            out.append(os.path.join(root, f))
        if not recursive:
            break
    return sorted(out, key=lambda p: natural_key(os.path.relpath(p, folder)))


def _key(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _siblings(img_path: str):
    """The image and the files that belong to it (.json / .txt with the same name)."""
    stem = os.path.splitext(img_path)[0]
    return [img_path] + [p for p in (stem + ".json", stem + ".txt") if os.path.exists(p)]


def _link(src, dst, copy=False):
    """A hard link (no extra disk space), or a copy when asked or when the drive has no hard links."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


class Context:
    """How a step reports back: progress(done, total, text), log(text), and cancelled() to stop."""

    def __init__(self, progress=None, log=None, cancel: threading.Event | None = None):
        self._progress, self._log = progress, log
        self.cancel = cancel or threading.Event()
        self.stage = ""

    def progress(self, done, total, text=""):
        if self._progress:
            self._progress(self.stage, int(done), int(total), text)

    def log(self, text):
        if self._log:
            self._log(self.stage, text)

    def check(self):
        if self.cancel.is_set():
            raise Stopped()


# ====================================================================================================== workspace
class Workspace:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.frames = os.path.join(self.root, FRAMES)
        self.similar = os.path.join(self.root, SIMILAR)
        self.extra = os.path.join(self.root, EXTRA)
        self.package = os.path.join(self.root, PACKAGE)
        self.reports = os.path.join(self.root, REPORTS)
        self.rejected = os.path.join(self.root, REJECTED)
        self.meta = os.path.join(self.root, META)
        self._db = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ state
    @property
    def state_path(self):
        return os.path.join(self.meta, "state.json")

    def state(self) -> dict:
        try:
            with open(self.state_path, encoding="utf-8") as f:
                st = json.load(f)
            return st if isinstance(st, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_state(self, st: dict):
        os.makedirs(self.meta, exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_path)

    def stage_record(self, stage) -> dict:
        return (self.state().get("stages") or {}).get(stage) or {}

    def set_stage(self, stage, **rec):
        st = self.state()
        st.setdefault("stages", {})[stage] = dict(rec, at=time.time())
        self.save_state(st)

    def stale(self, stage) -> bool:
        """A later step that ran before an earlier one was run again (its results may be out of date)."""
        me = self.stage_record(stage)
        if me.get("status") != "done":
            return False
        for s in STAGES[:STAGES.index(stage)]:
            r = self.stage_record(s)
            if r.get("status") == "done" and r.get("at", 0) > me.get("at", 0):
                return True
        return False

    # ------------------------------------------------------------------ files.db
    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            os.makedirs(self.meta, exist_ok=True)
            self._db = sqlite3.connect(os.path.join(self.meta, "files.db"), timeout=60, check_same_thread=False)
            from AIO_QC.core.sqlite_util import configure
            configure(self._db, os.path.join(self.meta, "files.db"))
            self._db.execute("CREATE TABLE IF NOT EXISTS moves (id INTEGER PRIMARY KEY, stage TEXT, src TEXT, "
                             "dst TEXT)")
            self._db.execute("CREATE TABLE IF NOT EXISTS anns (img TEXT PRIMARY KEY, key TEXT, mtime INTEGER, "
                             "size INTEGER, boxes INTEGER, labels TEXT)")
            self._db.execute("CREATE TABLE IF NOT EXISTS hashes (img TEXT PRIMARY KEY, mtime INTEGER, "
                             "size INTEGER, h INTEGER, sig BLOB, sharp REAL)")
            self._db.execute("CREATE TABLE IF NOT EXISTS ai_meta (name TEXT PRIMARY KEY, data TEXT)")
            self._db.commit()
        return self._db

    def close(self):
        if self._db is not None:
            try:
                self._db.commit()
                self._db.close()
            except sqlite3.Error:
                pass
            self._db = None

    def rel(self, p):
        return os.path.relpath(os.path.abspath(p), self.root).replace(os.sep, "/")

    def abs(self, rel):
        return os.path.join(self.root, *rel.split("/"))

    def move(self, stage, src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            os.remove(dst)
        os.replace(src, dst) if os.path.splitdrive(src)[0] == os.path.splitdrive(dst)[0] else shutil.move(src, dst)
        with self._lock:
            self.db.execute("INSERT INTO moves (stage, src, dst) VALUES (?,?,?)", (stage, self.rel(src),
                                                                                   self.rel(dst)))

    def restore(self, stages) -> int:
        """Put back what these steps set aside (newest first). Returns the number of files moved back."""
        stages = list(stages)
        if not stages:
            return 0
        q = ",".join("?" * len(stages))
        rows = self.db.execute(f"SELECT id, src, dst FROM moves WHERE stage IN ({q}) ORDER BY id DESC",
                               stages).fetchall()
        n = 0
        for rid, src, dst in rows:
            s, d = self.abs(src), self.abs(dst)
            if os.path.exists(d) and not os.path.exists(s):
                os.makedirs(os.path.dirname(s), exist_ok=True)
                os.replace(d, s)
                n += 1
            self.db.execute("DELETE FROM moves WHERE id=?", (rid,))
        self.db.commit()
        for top in (self.similar, self.extra, self.rejected):
            _prune_empty(top)
        return n

    def moved_count(self, stage) -> int:
        try:
            return self.db.execute("SELECT COUNT(*) FROM moves WHERE stage=? AND dst LIKE '%'", (stage,)
                                   ).fetchone()[0]
        except sqlite3.Error:
            return 0

    def frame_images(self) -> list[str]:
        return find_images(self.frames)

    def counts(self) -> dict:
        """What the run folder holds now (for the window's step list)."""
        imgs = self.frame_images()
        jsons = sum(1 for p in imgs if os.path.exists(os.path.splitext(p)[0] + ".json")
                    or os.path.exists(os.path.splitext(p)[0] + ".txt"))
        return {"frames": len(imgs), "annotated": jsons, "similar": len(find_images(self.similar)),
                "extra": len(find_images(self.extra)), "rejected": len(find_images(self.rejected)),
                "parts": len([1 for r, d, f in os.walk(self.package) for x in f if x.endswith(".zip")])
                if os.path.isdir(self.package) else 0}


def _prune_empty(top):
    if not os.path.isdir(top):
        return
    for root, dirs, files in os.walk(top, topdown=False):
        if root != top and not os.listdir(root):
            try:
                os.rmdir(root)
            except OSError:
                pass


# ====================================================================================================== 1 frames
def frames_key(r: Recipe, source: str) -> str:
    keys = ("source", "mode", "step", "min_sharpness", "skip_repeats", "across_videos", "ext", "quality",
            "name_template", "import_mode")
    return _key({k: getattr(r, k) for k in keys} | {"src": os.path.normcase(os.path.abspath(source))})


def _video_folders(videos, source):
    """One output folder per video, named after the video (parent folders added only when two videos share a name)."""
    names, out = {}, {}
    for v in videos:
        names.setdefault(os.path.splitext(os.path.basename(v))[0], []).append(v)
    for stem, vs in names.items():
        for v in vs:
            if len(vs) == 1:
                out[v] = stem
            else:
                rel = os.path.relpath(os.path.splitext(v)[0], source if os.path.isdir(source) else
                                      os.path.dirname(source))
                out[v] = re.sub(r"[\\/]+", "__", rel)
    return out


def extract_settings(r: Recipe, ws: Workspace):
    from AIO_QC.tools.frame_extractor.core.extract import ExtractSettings
    return ExtractSettings(
        mode=r.mode, step=max(1, int(r.step)), min_sharpness=float(r.min_sharpness),
        dedup=r.skip_repeats != "off", dedup_mode="exact" if r.skip_repeats == "exact" else "near",
        cross_dedup=bool(r.across_videos), hash_db=os.path.join(ws.meta, "hashes.db"),
        ext=r.ext, quality=int(r.quality), name_template=r.name_template or "{video}_frame_{frame}",
        on_exists="overwrite", resume=True, preannot="none", models=[])


def stage_frames(ws: Workspace, r: Recipe, ctx: Context, source: str, replace_previous=False) -> dict:
    if not source or not os.path.exists(source):
        raise StepError("frames", f"The {'video' if r.source == 'videos' else 'image'} folder was not found: "
                                  f"{source or '(none chosen)'}")
    key = frames_key(r, source)
    prev = ws.stage_record("frames")
    have = ws.frame_images() + find_images(ws.similar) + find_images(ws.extra) + find_images(ws.rejected)
    if have and prev.get("key") and prev["key"] != key:
        if not replace_previous:
            raise NeedsConfirmation(
                f"The frame settings (or the source folder) changed since the last run. Extracting again replaces "
                f"the {len(have):,} frames of that run and everything made from them (annotations, set-aside "
                f"frames, the package).", len(have))
        _clear_run(ws)
    elif have and not prev.get("key"):
        if not replace_previous:
            raise NeedsConfirmation(
                f"The run folder already has {len(have):,} frames that this run did not extract. Extracting "
                f"replaces them and everything made from them.", len(have))
        _clear_run(ws)
    ws.set_stage("frames", status="running", key=key)
    ws.restore(["similar", "annotate", "rules"])  # frames set aside earlier belong to the set again
    if r.source == "images":
        stats = _import_images(ws, r, ctx, source)
    else:
        stats = _extract_videos(ws, r, ctx, source)
    stats["frames"] = len(ws.frame_images())
    return stats


def _clear_run(ws: Workspace):
    for d in (ws.frames, ws.similar, ws.extra, ws.rejected):
        if os.path.isdir(d):
            shutil.rmtree(d)
    _clear_package(ws)
    ws.db.execute("DELETE FROM moves")
    ws.db.execute("DELETE FROM anns")
    ws.db.execute("DELETE FROM hashes")
    ws.db.execute("DELETE FROM ai_meta")
    ws.db.commit()
    h = os.path.join(ws.meta, "hashes.db")
    for p in (h, h + "-wal", h + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    st = ws.state()
    st["stages"] = {}
    ws.save_state(st)


def _import_images(ws, r, ctx, source):
    imgs = find_images(source, r.recursive, r.include, r.exclude, skip=[ws.root])
    if not imgs:
        raise StepError("frames", "No images found in that folder (check the name filters).")
    n = linked = 0
    for i, p in enumerate(imgs):
        ctx.check()
        dst = os.path.join(ws.frames, os.path.relpath(p, source))
        if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(p)):
            if r.import_mode == "link":
                _link(p, dst)
                linked += 1
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(p, dst)
            n += 1
        if i % 50 == 0:
            ctx.progress(i + 1, len(imgs), f"{i + 1:,} / {len(imgs):,} images")
    ctx.progress(len(imgs), len(imgs), "")
    return {"images_found": len(imgs), "added": n}


def _extract_videos(ws, r, ctx, source):
    from AIO_QC.tools.frame_extractor.core.batch import run_batch
    from AIO_QC.tools.frame_extractor.core.extract import Job
    videos = find_videos(source, r.recursive, r.include, r.exclude)
    if not videos:
        raise StepError("frames", "No videos found in that folder (check the name filters).")
    folders = _video_folders(videos, source)
    jobs = [Job(os.path.abspath(v), os.path.join(ws.frames, folders[v])) for v in videos]
    s = extract_settings(r, ws)
    from AIO_QC.core import procs
    workers = int(r.workers) or max(1, min(len(jobs), max(1, procs.cpu_budget() // 2), 6))
    prog = {}
    t0 = time.monotonic()

    def on_event(kind, ji, payload):
        if kind == "progress":
            prog[ji] = payload
            done = sum(p[0] for p in prog.values())
            total = sum(p[1] for p in prog.values()) or 1
            saved = sum(p[2] for p in prog.values())
            ctx.progress(done, total, f"{saved:,} frames saved · {len([1 for p in prog.values() if p[0] >= p[1]])}"
                                      f" / {len(jobs)} videos")
        elif kind == "error":
            ctx.log(f"{os.path.basename(jobs[ji].video_path)}: {payload}")

    ctx.log(f"{len(videos)} video(s), {workers} worker(s)")
    res = run_batch(jobs, s, workers=workers, on_event=on_event, cancel=ctx.cancel.is_set)
    errors = {os.path.basename(jobs[i].video_path): v for i, v in res.items() if isinstance(v, str)}
    stats = {"videos": len(videos), "seconds": round(time.monotonic() - t0, 1)}
    for v in res.values():
        if isinstance(v, dict):
            for k in ("saved", "duplicates", "exact_duplicates", "cross_duplicates", "blurry", "frames"):
                if isinstance(v.get(k), (int, float)) and not isinstance(v.get(k), bool):
                    stats[k] = stats.get(k, 0) + v[k]
    stats["saved"] = stats.get("saved", 0)
    if errors:
        stats["errors"] = errors
    if ctx.cancel.is_set() or any(isinstance(v, dict) and v.get("cancelled") for v in res.values()):
        raise Stopped()
    if errors and len(errors) == len(videos):
        raise StepError("frames", "No video could be read: " + "; ".join(f"{k}: {v}" for k, v in
                                                                            list(errors.items())[:3]))
    return stats


# ====================================================================================================== 2 similar
def _signature(path):
    import cv2
    from AIO_QC.tools.frame_extractor.core.hashdb import phash, thumb_sig
    from AIO_QC.tools.frame_extractor.core.imageio import imread
    img = imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if img is None:
        return None
    sharp = float(cv2.Laplacian(img, cv2.CV_32F).var())
    return phash(img), thumb_sig(img), sharp


def stage_similar(ws: Workspace, r: Recipe, ctx: Context) -> dict:
    import numpy as np
    from AIO_QC.tools.frame_extractor.core.hashdb import _popcount64, thumb_compare
    ws.restore(["similar", "annotate", "rules"])
    imgs = ws.frame_images()
    if not imgs:
        raise StepError("similar", "There are no frames yet — run Frames first.")
    dist, gpct, lpct = SIMILAR_PRESETS.get(r.similar, SIMILAR_PRESETS["balanced"])
    cache = {row[0]: row[1:] for row in ws.db.execute("SELECT img, mtime, size, h, sig, sharp FROM hashes")}
    sigs = [None] * len(imgs)
    todo = []
    for i, p in enumerate(imgs):
        st = os.stat(p)
        c = cache.get(ws.rel(p))
        if c and c[0] == st.st_mtime_ns and c[1] == st.st_size:
            sigs[i] = (int(c[2]) & ((1 << 64) - 1), c[3], c[4])
        else:
            todo.append(i)
    done = len(imgs) - len(todo)
    ctx.progress(done, len(imgs), "Reading frames")
    with ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4)))) as ex:
        for k in range(0, len(todo), 64):
            ctx.check()
            chunk = todo[k:k + 64]
            for i, sig in zip(chunk, ex.map(lambda j: _signature(imgs[j]), chunk)):
                sigs[i] = sig
                if sig is not None:
                    st = os.stat(imgs[i])
                    h = sig[0] - (1 << 64) if sig[0] >= (1 << 63) else sig[0]
                    ws.db.execute("INSERT OR REPLACE INTO hashes VALUES (?,?,?,?,?,?)",
                                  (ws.rel(imgs[i]), st.st_mtime_ns, st.st_size, h, sig[1], sig[2]))
            ws.db.commit()
            done += len(chunk)
            ctx.progress(done, len(imgs), "Reading frames")
    kept_h = np.zeros(0, np.uint64)
    kept = []                                    # (index into imgs, sig, sharpness)
    out = set()
    unreadable = 0
    for i, sig in enumerate(sigs):
        if i % 200 == 0:
            ctx.check()
            ctx.progress(i, len(imgs), "Comparing")
        if sig is None:
            unreadable += 1
            continue
        h, ts, sharp = sig
        match = None
        if len(kept_h):
            d = _popcount64(kept_h ^ np.uint64(h))
            idx = np.nonzero(d <= dist)[0]
            for j in idx[np.argsort(d[idx], kind="stable")]:
                g, l = thumb_compare(ts, kept[j][1])
                if g < gpct and l < lpct:
                    match = int(j)
                    break
        if match is None:
            kept.append((i, ts, sharp))
            kept_h = np.append(kept_h, np.uint64(h))
        elif r.prefer_sharper and sharp > kept[match][2] * 1.15 + 1.0:
            out.add(kept[match][0])              # the new frame is clearly sharper: it takes the place
            kept[match] = (i, ts, sharp)
            kept_h[match] = np.uint64(h)
        else:
            out.add(i)
    moved = 0
    for n, i in enumerate(sorted(out)):
        if n % 100 == 0:
            ctx.progress(n, len(out), "Setting aside")
        for p in _siblings(imgs[i]):
            ws.move("similar", p, os.path.join(ws.similar, os.path.relpath(p, ws.frames)))
        moved += 1
    ws.db.commit()
    ctx.progress(len(imgs), len(imgs), "")
    return {"frames": len(imgs), "similar": moved, "kept": len(imgs) - moved - unreadable,
            **({"unreadable": unreadable} if unreadable else {})}


# ====================================================================================================== 3 annotate
def annotate_key(r: Recipe, model_sig: str, imgsz: int) -> str:
    keys = ("conf", "iou", "max_det", "classes", "min_box", "labelme_version", "image_data", "description",
            "also_yolo", "half", "output_format")
    return _key({k: getattr(r, k) for k in keys} | {"model": model_sig, "imgsz": imgsz})


def model_signature(path):
    st = os.stat(path)
    return f"{os.path.normcase(os.path.abspath(path))}|{st.st_size}|{st.st_mtime_ns}"


def output_classes(r: Recipe, names: dict) -> list[str]:
    """The labels written, in the model's class order (unique) — classes.txt for the YOLO files."""
    out = []
    for cid in sorted(names):
        n = r.output_name(cid, names[cid])
        if n and n not in out:
            out.append(n)
    return out


def load_model(r: Recipe, env_python="", info=None):
    """(detector, names, imgsz) — the image size is the model's own unless the setup sets one."""
    from AIO_QC.tools.frame_extractor.core.autoannotate.model_info import info as model_info
    from AIO_QC.tools.frame_extractor.core.detector import load_detector
    if not r.model or not os.path.isfile(r.model):
        raise StepError("annotate", f"Model file not found: {r.model or '(none chosen)'}")
    info = info or model_info(r.model)
    ext = os.path.splitext(r.model)[1].lower()
    fixed = ext == ".onnx" and info.get("imgsz") and not info.get("dynamic_size")
    imgsz = int(info["imgsz"]) if fixed else int(r.imgsz) or int(info.get("imgsz") or 0)   # a fixed ONNX input wins
    if ext != ".onnx" and not env_python:
        import importlib.util
        if importlib.util.find_spec("ultralytics") is None:
            raise StepError("annotate", "A .pt model runs in a Python environment that has ultralytics. Choose one "
                                        "in Settings ▸ Models (Image QC ▸ Models & Environment), or export the model "
                                        "to ONNX once:  yolo export model=<model>.pt format=onnx")
    lo = min([float(r.conf)] + [e["conf"] for c in (r.classes or {}) for e in [r.class_entry(c)] if e["conf"]])
    try:
        det = load_detector(r.model, conf=lo, iou=float(r.iou), imgsz=imgsz, device=r.device or "auto",
                            env_python=env_python, half=bool(r.half), max_det=int(r.max_det))
    except Exception as exc:
        raise StepError("annotate", f"The model could not be loaded: {exc}") from exc
    names = dict(det.names or {}) or dict(info.get("names") or {})
    used = det.imgsz[0] if getattr(det, "imgsz", None) else imgsz
    det.conf = lo
    return det, names, int(used or imgsz or 640)


def shapes_for(rows, r: Recipe, names: dict, w: int, h: int, origin=None):
    """Model rows (x1, y1, x2, y2, score, class) -> (LabelMe shapes, YOLO rows). origin: each shape's provenance
    record (the model, its score)."""
    shapes, yolo = [], []
    classes = output_classes(r, names) if r.writes_yolo() else []
    for x1, y1, x2, y2, score, c in rows:
        c = int(c)
        label = r.output_name(c, names.get(c, str(c)))
        if label is None:
            continue
        e = r.class_entry(c)
        if score < (e["conf"] if e["conf"] else float(r.conf)):
            continue
        x1, x2 = sorted((max(0.0, min(float(x1), w)), max(0.0, min(float(x2), w))))
        y1, y2 = sorted((max(0.0, min(float(y1), h)), max(0.0, min(float(y2), h))))
        if x2 - x1 < max(1, int(r.min_box)) or y2 - y1 < max(1, int(r.min_box)):
            continue
        shapes.append(LM.rectangle(label, x1, y1, x2, y2, f"conf={score:.3f}" if r.description == "conf" else ""))
        if origin is not None:
            from AIO_QC.core import provenance as PV
            if PV.enabled():
                shapes[-1][PV.KEY] = PV.shape_record(origin, score=score)
        if r.writes_yolo():
            yolo.append((classes.index(label), x1, y1, x2, y2))
    return shapes, yolo


def _image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception as _exc:
        _ignored(_exc)
        import cv2
        from AIO_QC.tools.frame_extractor.core.imageio import imread
        img = imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if img is None:
            return None
        full = imread(path)
        return (full.shape[1], full.shape[0]) if full is not None else None


def plan_todo(ws, imgs, key, existing, ext, foreign="keep"):
    """Which images to annotate: (todo, up to date, edited by you, not written by this run).
    foreign="process": files this run did not write are annotated too (the AI merges into them)."""
    recs = {row[0]: row[1:] for row in ws.db.execute("SELECT img, key, mtime, size FROM anns")}
    todo, skipped, kept_yours, kept_other = [], 0, 0, 0
    for p in imgs:
        out = os.path.splitext(p)[0] + ext
        rec = recs.get(ws.rel(p))
        if os.path.exists(out) and existing != "replace":
            st = os.stat(out)
            ours_unchanged = rec is not None and rec[1] == st.st_mtime_ns and rec[2] == st.st_size
            if existing == "keep":
                skipped += 1
                continue
            if rec is None:
                if foreign == "process":
                    todo.append(p)
                    continue
                kept_other += 1                  # a file this run did not write: never overwritten
                continue
            if not ours_unchanged:
                kept_yours += 1                  # written by this run, then edited: yours now
                continue
            if rec[0] == key:
                skipped += 1                     # already done with these settings
                continue
        todo.append(p)
    return todo, skipped, kept_yours, kept_other


def stage_annotate(ws: Workspace, r: Recipe, ctx: Context, env_python="", detector=None) -> dict:
    if r.annotator == "ai":
        from AIO_QC.tools.frame_extractor.core.ai_annotate.stage import stage_ai
        return stage_ai(ws, r, ctx, env_python)
    ws.restore(["annotate", "rules"])
    imgs = ws.frame_images()
    if not imgs:
        raise StepError("annotate", "There are no frames yet — run Frames first (or choose an image folder).")
    own = detector is None
    if own:
        ctx.progress(0, 0, "Loading the model")
        det, names, imgsz = load_model(r, env_python)
    else:
        det, names, imgsz = detector, dict(detector.names or {}), int(detector.imgsz[0])
    key = annotate_key(r, model_signature(r.model) if r.model and os.path.isfile(r.model) else "given", imgsz)
    ws.set_stage("annotate", status="running", key=key)
    todo, skipped, kept_yours, kept_other = plan_todo(ws, imgs, key, r.existing,
                                                      ".json" if r.writes_labelme() else ".txt")
    classes = output_classes(r, names)
    from AIO_QC.core import provenance as PV
    prov = PV.make_origin("model", "Auto-Annotate", model_path=r.model if r.model and os.path.isfile(r.model) else None,
                          model=os.path.basename(r.model or ""))
    if r.writes_yolo():
        from AIO_QC.tools.frame_extractor.core.ai_annotate.writers import write_classes
        write_classes({os.path.dirname(p) for p in imgs} | {ws.frames}, classes)
    per_class, empty, boxes, failed = {}, 0, 0, []
    total = len(todo)
    ctx.log(f"model {os.path.basename(r.model)} · image size {imgsz} · {len(names)} classes · {total:,} frames "
            f"to annotate ({skipped:,} up to date)")
    t0 = time.monotonic()
    remote = not hasattr(det, "_postprocess")
    B = max(1, int(r.batch)) * (4 if remote else 2)
    pool = ThreadPoolExecutor(max_workers=4)

    def read(paths):
        from AIO_QC.tools.frame_extractor.core.imageio import imread
        return list(pool.map(imread, paths))

    try:
        nxt = None if remote or not todo else pool.submit(read, todo[:B])
        for k in range(0, total, B):
            ctx.check()
            chunk = todo[k:k + B]
            if remote:
                res = det.detect_files(chunk, conf=det.conf, batch=max(1, int(r.batch)))
                frames = [None] * len(chunk)
            else:
                frames = nxt.result()
                nxt = pool.submit(read, todo[k + B:k + 2 * B]) if k + B < total else None
                ok = [f for f in frames if f is not None]
                it = iter(det.detect_batch(ok, det.conf) if ok else [])
                res = [next(it) if f is not None else None for f in frames]
            for p, f, rows in zip(chunk, frames, res):
                if rows is None:
                    failed.append(os.path.basename(p))
                    continue
                if f is not None:
                    h, w = f.shape[:2]
                else:
                    sz = _image_size(p)
                    if sz is None:
                        failed.append(os.path.basename(p))
                        continue
                    w, h = sz
                shapes, yolo = shapes_for(list(map(tuple, rows)), r, names, w, h, origin=prov)
                jp = os.path.splitext(p)[0] + ".json"
                if r.writes_labelme():
                    LM.write(jp, LM.document(p, jp, w, h, shapes, r.labelme_version, r.image_data), origin=prov)
                if r.writes_yolo():
                    LM.write_yolo(os.path.splitext(p)[0] + ".txt", w, h, yolo)
                st = os.stat(jp if r.writes_labelme() else os.path.splitext(p)[0] + ".txt")
                cnt = {}
                for sh in shapes:
                    cnt[sh["label"]] = cnt.get(sh["label"], 0) + 1
                ws.db.execute("INSERT OR REPLACE INTO anns VALUES (?,?,?,?,?,?)",
                              (ws.rel(p), key, st.st_mtime_ns, st.st_size, len(shapes), json.dumps(cnt)))
            ws.db.commit()
            done = min(total, k + B)
            el = time.monotonic() - t0
            rate = done / el if el > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            ctx.progress(done, total, f"{done:,} / {total:,} frames · {rate:.1f}/s"
                                      + (f" · {_dur(eta)} left" if done < total and rate else ""))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        if own and hasattr(det, "close"):
            try:
                det.close()
            except Exception as _exc:
                _ignored(_exc)
    # totals over every frame in the set (written now or earlier with the same settings)
    rows = {row[0]: row[1] for row in ws.db.execute("SELECT img, labels FROM anns WHERE key=?", (key,))}
    for p in imgs:
        try:
            cnt = json.loads(rows.get(ws.rel(p)) or "null")
        except ValueError:
            cnt = None
        if not isinstance(cnt, dict):
            continue
        for lab, c in cnt.items():
            per_class[lab] = per_class.get(lab, 0) + int(c)
        boxes += sum(int(c) for c in cnt.values())
        empty += not cnt
    out = {"frames": len(imgs), "annotated": total - len(failed), "up_to_date": skipped, "boxes": boxes,
           "frames_without_boxes": empty, "classes": dict(sorted(per_class.items())), "image_size": imgsz,
           "labelme_version": r.labelme_version, "image_data": bool(r.image_data),
           "seconds": round(time.monotonic() - t0, 1)}
    if kept_yours:
        out["edited_kept"] = kept_yours
    if kept_other:
        out["not_ours_kept"] = kept_other
    if failed:
        out["unreadable"] = len(failed)
        ctx.log("could not read: " + ", ".join(failed[:10]) + (" …" if len(failed) > 10 else ""))
    return out


def _dur(s):
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


# ====================================================================================================== 4 rules
def keeps(r: Recipe, labels: list[str]) -> bool:
    if r.keep_when == "always":
        return True
    if r.keep_when == "detections":
        return len(labels) >= max(1, int(r.min_boxes))
    want = set(r.keep_classes or [])
    have = set(labels)
    if r.keep_when == "all_of":
        return bool(want) and want <= have
    return bool(want & have)


def stage_rules(ws: Workspace, r: Recipe, ctx: Context) -> dict:
    ws.restore(["rules"])
    imgs = ws.frame_images()
    if not imgs:
        raise StepError("rules", "There are no frames yet — run Frames first.")
    from AIO_QC.tools.frame_extractor.core.ai_annotate.writers import has_labels_file, labels_of
    have_json = [p for p in imgs if has_labels_file(p)]
    if not have_json and r.keep_when != "always":
        raise StepError("rules", "The frames have no annotation files yet — run Annotate first.")
    kept = aside = no_json = 0
    per_class = {}
    for i, p in enumerate(imgs):
        if i % 200 == 0:
            ctx.check()
            ctx.progress(i, len(imgs), "Checking frames")
        if not has_labels_file(p):
            no_json += 1
            labels = []
        else:
            labels = labels_of(p)
        if keeps(r, labels):
            kept += 1
            for lab in labels:
                per_class[lab] = per_class.get(lab, 0) + 1
        elif r.others == "move":
            for f in _siblings(p):
                ws.move("rules", f, os.path.join(ws.extra, os.path.relpath(f, ws.frames)))
            aside += 1
        else:
            kept += 1
    ws.db.commit()
    ctx.progress(len(imgs), len(imgs), "")
    return {"frames": len(imgs), "kept": kept, "set_aside": aside, "kept_without_annotation": no_json
            if r.keep_when == "always" else 0, "classes_kept": dict(sorted(per_class.items()))}


# ====================================================================================================== 5 package
def group_of(r: Recipe, rel_in_frames: str) -> str:
    if r.group_by == "video":
        parts = rel_in_frames.split("/")
        return parts[0] if len(parts) > 1 else UNMATCHED
    if r.group_by == "pattern" and r.group_pattern:
        m = re.search(r.group_pattern, os.path.basename(rel_in_frames))
        if not m:
            return UNMATCHED
        g = m.group(1) if m.re.groups and m.group(1) is not None else m.group(0)
        return re.sub(r'[<>:"/\\|?*]+', "_", g).strip(". ") or UNMATCHED
    return ""


def part_name(r: Recipe, n: int, start: int, total_parts: int) -> str:
    if r.part_names == "start":
        return f"part_{start}"
    return f"part_{n:0{max(3, len(str(total_parts)))}d}"


def plan_package(ws: Workspace, r: Recipe) -> dict:
    """{group: [[images of part 1], [part 2], ...]} — what package would build (also the window's preview)."""
    groups = {}
    for p in ws.frame_images():
        groups.setdefault(group_of(r, os.path.relpath(p, ws.frames).replace(os.sep, "/")), []).append(p)
    out = {}
    for g in sorted(groups, key=natural_key):
        files = sorted(groups[g], key=lambda p: natural_key(os.path.basename(p)))
        n = int(r.split_size)
        out[g] = [files[i:i + n] for i in range(0, len(files), n)] if n > 0 else [files]
    return out


def _clear_package(ws: Workspace):
    if os.path.isdir(ws.package):
        if os.path.exists(os.path.join(ws.package, PACKAGE_MARK)) or not os.listdir(ws.package):
            shutil.rmtree(ws.package)
        else:
            raise StepError("package", f"{ws.package} holds files this run did not make — move them first.")


def _zip(folder, zpath):
    tmp = zpath + ".tmp"
    base = os.path.dirname(folder)
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for root, dirs, files in os.walk(folder):
            dirs.sort()
            for f in sorted(files, key=natural_key):
                p = os.path.join(root, f)
                arc = os.path.relpath(p, base).replace(os.sep, "/")
                comp = zipfile.ZIP_STORED if f.lower().endswith(IMAGE_EXTS) else zipfile.ZIP_DEFLATED
                z.write(p, arc, compress_type=comp)
    os.replace(tmp, zpath)
    return os.path.getsize(zpath)


def stage_package(ws: Workspace, r: Recipe, ctx: Context) -> dict:
    plan = plan_package(ws, r)
    if not plan:
        raise StepError("package", "There are no frames to package — run Frames first.")
    _clear_package(ws)
    copy = r.package_files == "copy"

    def put(src, dst):
        _link(src, dst, copy)
    os.makedirs(ws.package, exist_ok=True)
    open(os.path.join(ws.package, PACKAGE_MARK), "w").close()
    from AIO_QC.tools.frame_extractor.core.ai_annotate.writers import read_classes, write_data_yaml
    classes_src = os.path.join(ws.frames, "classes.txt")
    names = read_classes(classes_src) if os.path.exists(classes_src) else []
    split = r.yolo_layout == "split"
    meta = {row[0]: row[1] for row in ws.db.execute("SELECT name, data FROM ai_meta")}
    jobs, rows = [], []
    total = sum(len(f) for parts in plan.values() for f in parts)
    done = 0
    for g, parts in plan.items():
        gdir = os.path.join(ws.package, g) if g else ws.package
        for n, files in enumerate(parts, 1):
            ctx.check()
            start = sum(len(x) for x in parts[:n - 1])
            pdir = os.path.join(gdir, part_name(r, n, start, len(parts))) if int(r.split_size) > 0 else gdir
            count = 0
            used = set()
            any_txt = False
            metas = []
            idir = os.path.join(pdir, "images") if split else pdir
            ldir = os.path.join(pdir, "labels") if split else pdir
            for img in files:
                rel = os.path.relpath(img, ws.frames)
                flat = os.path.basename(rel)
                if flat.lower() in used:                 # same name in two sub-folders: keep both, named by path
                    flat = re.sub(r"[\\/]+", "__", rel)
                used.add(flat.lower())
                stem = os.path.splitext(img)[0]
                if r.package_images:
                    put(img, os.path.join(idir, flat))
                if os.path.exists(stem + ".json"):
                    jdst = os.path.join(idir, os.path.splitext(flat)[0] + ".json")
                    if flat == os.path.basename(rel):
                        put(stem + ".json", jdst)
                    else:                                # renamed: the copy points at the renamed image
                        try:
                            with open(stem + ".json", encoding="utf-8") as f:
                                doc = json.load(f)
                            doc["imagePath"] = flat
                            from AIO_QC.core import provenance as PV
                            PV.derive_labelme(doc, doc.get(PV.KEY), PV.make_origin("conversion", "Auto-Annotate"),
                                              source=rel, renamed_to=flat)
                            LM.write(jdst, doc, stamp=False)
                        except (OSError, ValueError):
                            put(stem + ".json", jdst)
                if os.path.exists(stem + ".txt"):
                    put(stem + ".txt", os.path.join(ldir, os.path.splitext(flat)[0] + ".txt"))
                    any_txt = True
                m = meta.get(os.path.basename(img))
                if m:
                    metas.append(json.dumps({"image": flat, **json.loads(m)}, ensure_ascii=False))
                count += 1
                done += 1
                if done % 100 == 0:
                    ctx.progress(done, total, "Making the parts")
            if any_txt and names:
                put(classes_src, os.path.join(ldir, "classes.txt"))
                if split or ldir == pdir:
                    write_data_yaml(pdir, names, train="images" if split else ".", val="images" if split else ".")
            if metas:
                os.makedirs(pdir, exist_ok=True)
                with open(os.path.join(pdir, "meta.jsonl"), "w", encoding="utf-8") as f:
                    f.write("\n".join(metas) + "\n")
            rows.append({"folder": g, "part": os.path.basename(pdir) if int(r.split_size) > 0 else "",
                         "files": count, "path": os.path.relpath(pdir, ws.root).replace(os.sep, "/")})
            if r.zip_parts and pdir != ws.package:
                jobs.append((pdir, pdir + ".zip", rows[-1]))
    ctx.progress(total, total, "")
    if jobs:
        ctx.progress(0, len(jobs), "Zipping")
        done_z = 0
        with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 2))) as ex:
            futs = {ex.submit(_zip, a, b): (a, b, row) for a, b, row in jobs}
            for fu in futs:
                ctx.check()
                a, b, row = futs[fu]
                row["zip"] = os.path.relpath(b, ws.root).replace(os.sep, "/")
                row["zip_mb"] = round(fu.result() / 2 ** 20, 1)
                done_z += 1
                ctx.progress(done_z, len(jobs), "Zipping")
        if not r.keep_folders:
            for a, _, _ in jobs:
                shutil.rmtree(a, ignore_errors=True)
    import csv
    with open(os.path.join(ws.package, "parts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["folder", "part", "files", "path", "zip", "zip_mb"])
        w.writeheader()
        w.writerows(rows)
    per_folder = {}
    for row in rows:
        per_folder[row["folder"] or "(all)"] = per_folder.get(row["folder"] or "(all)", 0) + row["files"]
    return {"files": total, "parts": len(rows), "files_as": "copies" if copy else "hard links", "zips": len(jobs) if r.zip_parts else 0,
            "folders": dict(per_folder)}


# ====================================================================================================== run
def run(root: str, r: Recipe, stages=None, source="", ctx: Context | None = None, env_python="",
        replace_previous=False, detector=None) -> dict:
    """Run the chosen steps in order. Returns {stage: stats}; a stop or failure keeps what was done (and says so in
    the run's state)."""
    ctx = ctx or Context()
    ws = Workspace(root)
    os.makedirs(ws.root, exist_ok=True)
    stages = [s for s in STAGES if s in (stages if stages is not None else r.stages)]
    problems = r.problems() if stages else ["Choose at least one step."]
    problems = [p for p in problems if _problem_applies(p, stages)]
    if problems:
        raise StepError(stages[0] if stages else "frames", " ".join(problems))
    st = ws.state()
    st.update(recipe=r.to_dict(), source=source or st.get("source", ""), updated=time.time())
    ws.save_state(st)
    results = {}
    try:
        for s in stages:
            ctx.stage = s
            ctx.check()
            t0 = time.monotonic()
            ws.set_stage(s, status="running", key=ws.stage_record(s).get("key"))
            try:
                if s == "frames":
                    stats = stage_frames(ws, r, ctx, source, replace_previous)
                elif s == "similar":
                    stats = stage_similar(ws, r, ctx)
                elif s == "annotate":
                    stats = stage_annotate(ws, r, ctx, env_python, detector)
                elif s == "rules":
                    stats = stage_rules(ws, r, ctx)
                else:
                    stats = stage_package(ws, r, ctx)
            except Stopped:
                ws.set_stage(s, status="stopped", key=ws.stage_record(s).get("key"))
                raise
            except NeedsConfirmation:
                ws.set_stage(s, status="waiting", key=ws.stage_record(s).get("key"))
                raise
            except StepError as exc:
                ws.set_stage(s, status="failed", error=str(exc), key=ws.stage_record(s).get("key"))
                raise
            except Exception as exc:
                ws.set_stage(s, status="failed", error=f"{type(exc).__name__}: {exc}",
                             key=ws.stage_record(s).get("key"))
                raise StepError(s, f"{STAGE_TITLES[s]}: {type(exc).__name__}: {exc}") from exc
            stats.setdefault("seconds", round(time.monotonic() - t0, 1))
            results[s] = stats
            ws.set_stage(s, status="done", stats=stats, key=ws.stage_record(s).get("key")
                         or _key(r.to_dict()))
            ctx.log(f"{STAGE_TITLES[s]} done")
    finally:
        try:
            write_report(ws, r)
        except Exception as _exc:
            _ignored(_exc)
        ws.close()
    return results


def _problem_applies(p: str, stages) -> bool:
    for s, title in (("annotate", "Annotate:"), ("rules", "Keep rule:"), ("package", "Package:")):
        if p.startswith(title):
            return s in stages
    return True


# ====================================================================================================== report
def write_report(ws: Workspace, r: Recipe) -> str:
    os.makedirs(ws.reports, exist_ok=True)
    st = ws.state()
    counts = ws.counts()
    data = {"run_folder": ws.root, "source": st.get("source", ""), "setup": r.to_dict(), "now": counts,
            "steps": {s: st.get("stages", {}).get(s, {}) for s in STAGES if s in st.get("stages", {})}}
    with open(os.path.join(ws.reports, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    from html import escape

    def table(d):
        rows = "".join(f"<tr><th>{escape(str(k).replace('_', ' '))}</th><td>{_fmt(v)}</td></tr>" for k, v in d.items())
        return f"<table>{rows}</table>"

    def _fmt(v):
        if isinstance(v, dict):
            return table(v) if v else "—"
        if isinstance(v, float):
            return f"{v:,.1f}"
        if isinstance(v, int) and not isinstance(v, bool):
            return f"{v:,}"
        return escape(str(v))

    parts = [f"<h1>Auto-Annotate</h1><p class=m>{escape(ws.root)}<br>source: {escape(str(data['source']))}</p>",
             "<h2>The run folder now</h2>", table({"frames": counts["frames"], "annotated": counts["annotated"],
                                                   "similar (set aside)": counts["similar"],
                                                   "extra_images (set aside)": counts["extra"],
                                                   "zip files": counts["parts"]})]
    for s, rec in data["steps"].items():
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(rec.get("at", 0))) if rec.get("at") else ""
        parts.append(f"<h2>{escape(STAGE_TITLES[s])} <span class=m>{escape(rec.get('status', ''))} · {when}</span>"
                     f"</h2>")
        if rec.get("error"):
            parts.append(f"<p class=e>{escape(rec['error'])}</p>")
        if rec.get("stats"):
            parts.append(table(rec["stats"]))
    css = ("body{font:14px/1.45 Inter,Segoe UI,system-ui,sans-serif;color:#1B1F24;background:#F6F7F9;max-width:900px;"
           "margin:24px auto;padding:0 16px}h1{font-size:22px}h2{font-size:16px;margin-top:22px}.m{color:#6B7480;"
           "font-weight:400;font-size:13px}.e{color:#C62828}table{border-collapse:collapse;background:#fff;"
           "border:1px solid #E3E6EA;border-radius:6px}th,td{padding:4px 10px;text-align:left;vertical-align:top;"
           "border-bottom:1px solid #EEF1F4}th{color:#4A5360;font-weight:500;white-space:nowrap}td table{border:0}"
           "td{font-variant-numeric:tabular-nums}")
    path = os.path.join(ws.reports, "summary.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><meta charset=utf-8><title>Auto-Annotate summary</title><style>{css}</style>"
                + "".join(parts))
    return path


# ====================================================================================================== preview
def sample_frames(r: Recipe, source: str, n=4):
    """A few frames to try the model on before a run: spread over the first videos (or images)."""
    out = []
    if r.source == "images":
        from AIO_QC.tools.frame_extractor.core.imageio import imread
        imgs = find_images(source, r.recursive, r.include, r.exclude)
        for p in imgs[::max(1, len(imgs) // n)][:n]:
            f = imread(p)
            if f is not None:
                out.append((os.path.basename(p), f))
        return out
    from AIO_QC.tools.frame_extractor.core.sources import open_source
    vids = find_videos(source, r.recursive, r.include, r.exclude)[:n]
    per = max(1, -(-n // max(1, len(vids))))
    for v in vids:
        src = open_source(v, "auto", False)
        try:
            fc = max(1, src.frame_count)
            for k in range(per):
                if len(out) >= n:
                    break
                idx = int(fc * (k + 1) / (per + 1))
                src.seek(idx)
                ok, fr = src.read()
                fr = fr if ok else None
                if fr is not None:
                    out.append((f"{os.path.basename(v)} · frame {idx}", fr))
        finally:
            try:
                src.release()
            except Exception as _exc:
                _ignored(_exc)
    return out


def preview(r: Recipe, frames, env_python="", detector=None):
    """[(title, image, shapes)] with the setup's classes, thresholds and box filter applied."""
    det, names, _ = (detector, dict(detector.names or {}), 0) if detector is not None else load_model(r, env_python)
    imgs = [f for _, f in frames]
    res = det.detect_batch(imgs, det.conf) if imgs else []
    out = []
    for (title, f), rows in zip(frames, res):
        h, w = f.shape[:2]
        shapes, _ = shapes_for(list(map(tuple, rows)), r, names, w, h)
        out.append((title, f, shapes))
    return out, det, names
