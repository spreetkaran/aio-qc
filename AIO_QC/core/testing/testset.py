# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Test sets: a frozen, reusable set of frames to evaluate models on - so a new model next month is measured on
exactly the same images as today's.

    <test set>/
        testset.json        manifest: sources, settings, items (file, video, frame, time, size, sharpness)
        data.yaml           YOLO-compatible (path: ., val: images, names) - open it in any YOLO tool
        images/             frames taken from the videos  (<video>_<frame>.jpg)
        labels/             ground truth in YOLO format (optional - add it any time, e.g. after labelling)
        cache/              prediction cache (runner)
        reports/            generated reports

Sources: any number of videos (sampled: every N seconds / every N frames / N evenly spaced / all frames,
optional time range) and image folders (referenced in place, or copied). Unique frames: a fast check inside
each video against the last saved frames, then a global check against every frame already in the set (all
videos), so a scene repeating an hour later or in another video is kept once. Same similarity rules as
batch extraction (exposure-normalised; whole-picture AND local change).

Labels: YOLO txt in labels/ or LabelMe json next to the image; pre-annotations written by FrameExtractor are
fingerprinted and stay "unreviewed" (excluded from accuracy by default) until somebody edits them.
"""
import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from AIO_QC.core.testing import dataset as D
from AIO_QC.tools.frame_extractor.core.dedup import PRESETS as _DEDUP_PRESETS
from AIO_QC.tools.frame_extractor.core.imageio import imread

MANIFEST = "testset.json"
VERSION = 1
DEFAULTS = dict(sampling="interval", interval_s=1.0, every_n=30, count=100, max_per_video=0,
                unique="balanced", min_sharpness=0.0, image_ext=".jpg", quality=95, copy_folders=False)
# same similarity rules as batch extraction: (whole-picture %, local-cell %); "exact" = identical pixels only
UNIQUE_PRESETS = {"off": None, "exact": (0.0, 0.0), **_DEDUP_PRESETS}
GSIG = (64, 36)          # global signature: exposure-normalised 64x36 grey, 16x9 cells of 4x4 px


# ------------------------------------------------------------------ global uniqueness
def global_sig(img):
    from AIO_QC.tools.frame_extractor.core.hashdb import phash
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    t = cv2.resize(cv2.resize(g, (256, 144), interpolation=cv2.INTER_AREA), GSIG, interpolation=cv2.INTER_AREA)
    t = t.astype(np.float32)
    n = np.clip((t - t.mean()) / max(float(t.std()), 4.0) * 50.0 + 128.0, 0, 255).astype(np.uint8)
    exact = hashlib.blake2b(np.ascontiguousarray(img[::2, ::2]).tobytes(), digest_size=12).digest()
    return np.uint64(phash(img)), n, exact


class UniqueIndex:
    """All frames kept so far; is_dup() checks a new frame against every one of them (vectorised)."""

    def __init__(self, preset="balanced", hash_dist=12):
        self.limits = UNIQUE_PRESETS.get(preset)
        self.exact_only = preset == "exact"
        self.hash_dist = hash_dist
        self.h = np.zeros(0, np.uint64)
        self.thumbs = np.zeros((0, GSIG[1], GSIG[0]), np.uint8)
        self._cap = 0
        self.exact = set()
        self.n = 0

    def is_dup(self, sig):
        if self.limits is None:
            return False
        h, thumb, exact = sig
        if exact in self.exact:
            return True
        if self.exact_only or not self.n:
            return False
        from AIO_QC.tools.frame_extractor.core.hashdb import _popcount64
        d = _popcount64(self.h[:self.n] ^ h)
        cand = np.nonzero(d <= self.hash_dist)[0]
        if not len(cand):
            return False
        g_lim, l_lim = self.limits
        diff = np.abs(self.thumbs[cand].astype(np.int16) - thumb.astype(np.int16)).astype(np.float32)
        glob = diff.mean(axis=(1, 2)) * 100 / 255
        cells = diff.reshape(len(cand), 9, 4, 16, 4).mean(axis=(2, 4)).max(axis=(1, 2)) * 100 / 255
        return bool(((glob < g_lim) & (cells < l_lim)).any())

    def add(self, sig):
        h, thumb, exact = sig
        self.exact.add(exact)
        if self.n >= self._cap:
            self._cap = max(256, self._cap * 2)
            nh = np.zeros(self._cap, np.uint64)
            nt = np.zeros((self._cap, GSIG[1], GSIG[0]), np.uint8)
            nh[:self.n], nt[:self.n] = self.h[:self.n], self.thumbs[:self.n]
            self.h, self.thumbs = nh, nt
        self.h[self.n], self.thumbs[self.n] = h, thumb
        self.n += 1


# ------------------------------------------------------------------ sampling
def target_frames(src, settings, start_s=None, end_s=None):
    """Frame indices to take from one video."""
    n = src.frame_count
    a = src.index_of(start_s) if start_s else 0
    b = min(n - 1, src.index_of(end_s)) if end_s else n - 1
    if b < a:
        return []
    mode = settings["sampling"]
    if mode == "all":
        idx = list(range(a, b + 1))
    elif mode == "every_n":
        idx = list(range(a, b + 1, max(1, int(settings["every_n"]))))
    elif mode == "count":
        k = max(1, int(settings["count"]))
        idx = sorted(set(np.linspace(a, b, k).round().astype(int).tolist()))
    else:                                                    # interval in seconds (exact for VFR)
        t0, t1, step = src.time_of(a), src.time_of(b), max(1e-3, float(settings["interval_s"]))
        idx = sorted(set(min(b, max(a, src.index_of(t))) for t in np.arange(t0, t1 + 1e-9, step)))
    cap = int(settings.get("max_per_video") or 0)
    if cap and len(idx) > cap:                               # keep the spread, not the first N
        idx = [idx[i] for i in np.linspace(0, len(idx) - 1, cap).round().astype(int)]
    return idx


def iter_frames(src, indices, cancel=None):
    """Yield (idx, frame) for sorted indices: sequential decode for dense picks, seeks for sparse ones."""
    keys = getattr(src, "keyframes", None)
    gop = float(np.diff(keys).mean()) if keys and len(keys) > 1 else 60.0
    pos = None
    for idx in indices:
        if cancel is not None and cancel():
            return
        if pos is not None and 0 <= idx - pos <= max(4.0, 1.5 * gop):
            while pos < idx:                                 # cheaper to decode through the gap
                if not src.grab():
                    return
                pos += 1
        else:
            src.seek(idx)
            pos = idx
        ok, frame = src.read()
        if not ok:
            return
        pos = idx + 1
        yield idx, frame


def _safe_stem(path, taken):
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in os.path.splitext(os.path.basename(path))[0])
    stem = stem.strip("._") or "video"
    if stem in taken and taken[stem] != os.path.abspath(path):
        stem = f"{stem}_{hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:6]}"
    taken[stem] = os.path.abspath(path)
    return stem


# ------------------------------------------------------------------ the test set
class TestSet:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.manifest = dict(version=VERSION, name=os.path.basename(self.root), created=time.time(),
                             settings=dict(DEFAULTS), sources=[], items=[], names={}, prelabels={}, stems={})
        self._lock = threading.Lock()

    # ---- files
    @property
    def images_dir(self):
        return os.path.join(self.root, "images")

    @property
    def labels_dir(self):
        return os.path.join(self.root, "labels")

    @property
    def cache_dir(self):
        return os.path.join(self.root, "cache")

    @property
    def reports_dir(self):
        return os.path.join(self.root, "reports")

    @staticmethod
    def is_testset(path):
        return os.path.isfile(os.path.join(path, MANIFEST))

    @classmethod
    def open(cls, root):
        ts = cls(root)
        with open(os.path.join(ts.root, MANIFEST), encoding="utf-8") as f:
            m = json.load(f)
        m["names"] = {int(k): v for k, v in (m.get("names") or {}).items()}
        m.setdefault("prelabels", {})
        m.setdefault("stems", {})
        ts.manifest.update(m)
        return ts

    @classmethod
    def create(cls, root, name=None, settings=None):
        ts = cls(root)
        os.makedirs(ts.images_dir, exist_ok=True)
        os.makedirs(ts.labels_dir, exist_ok=True)
        if name:
            ts.manifest["name"] = name
        ts.manifest["settings"].update(settings or {})
        ts.save()
        return ts

    def save(self):
        m = dict(self.manifest)
        m["names"] = {str(k): v for k, v in self.manifest["names"].items()}
        tmp = os.path.join(self.root, MANIFEST + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=1)
        os.replace(tmp, os.path.join(self.root, MANIFEST))          # atomic: never a half-written manifest
        self._write_data_yaml()

    def _write_data_yaml(self):
        names = self.manifest["names"]
        lines = ["# written by AIO QC - this test set opens in any YOLO tool", "path: .",
                 "train: images", "val: images", "names:"]
        lines += [f"  {k}: {json.dumps(v)}" for k, v in sorted(names.items())] if names else ["  {}"]
        with open(os.path.join(self.root, "data.yaml"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        if names:
            with open(os.path.join(self.labels_dir, "classes.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(names[k] for k in sorted(names)) + "\n")

    @property
    def items(self):
        return self.manifest["items"]

    @property
    def names(self):
        return self.manifest["names"]

    def set_names(self, names):
        self.manifest["names"] = {int(k): str(v) for k, v in names.items()}
        self.save()

    def abspath(self, item):
        f = item["file"]
        return f if os.path.isabs(f) else os.path.join(self.root, f)

    def image_paths(self):
        return [self.abspath(it) for it in self.items]

    # ---- building
    def _unique_index(self, preset):
        """Index of every image already in the set (rebuilt from disk when adding to an existing set)."""
        ui = UniqueIndex(preset)
        if ui.limits is None:
            return ui
        for it in self.items:
            img = imread(self.abspath(it))
            if img is not None:
                ui.add(global_sig(img))
        return ui

    def add_sources(self, sources, settings=None, progress=None, cancel=None, workers=0):
        """sources: [{type: 'video'|'folder', path, start, end}]. progress(done, total, text).
        Returns stats dict. The manifest is saved even when cancelled (what was added stays)."""
        s = dict(self.manifest["settings"])
        s.update(settings or {})
        self.manifest["settings"] = s
        cancel = cancel or (lambda: False)
        progress = progress or (lambda *a: None)
        os.makedirs(self.images_dir, exist_ok=True)
        uniq = self._unique_index(s["unique"]) if self.items else UniqueIndex(s["unique"])
        from AIO_QC.tools.frame_extractor.core.dedup import DedupFilter, Signature
        from AIO_QC.tools.frame_extractor.core.metrics import sharpness
        stats = dict(candidates=0, added=0, duplicates=0, blurry=0, failed=[], seconds=0.0)
        t0 = time.time()
        videos = [x for x in sources if x.get("type", "video") == "video"]
        folders = [x for x in sources if x.get("type") == "folder"]

        # plan: frame indices per video (cheap, header only)
        plans = []
        from AIO_QC.tools.frame_extractor.core.sources import open_source
        for v in videos:
            try:
                src = open_source(v["path"], "auto", hw_accel=False)
                idx = target_frames(src, s, v.get("start"), v.get("end"))
                plans.append((v, idx, src.fps, src.frame_count))
                src.release()
            except Exception as e:
                stats["failed"].append(f"{os.path.basename(v['path'])}: {e}")
        # one plan per video: the same video twice (or overlapping ranges) would make two threads write the same
        # file names at once - their frames are merged into one plan instead
        merged = {}
        for v, idx, fps, n in plans:
            k = os.path.normcase(os.path.abspath(v["path"]))
            if k in merged:
                merged[k] = (merged[k][0], sorted(set(merged[k][1]) | set(idx)), fps, n)
            else:
                merged[k] = (v, list(idx), fps, n)
        source_plans, plans = plans, list(merged.values())
        total = sum(len(p[1]) for p in plans) + sum(1 for _ in folders)
        done = [0]
        lock = threading.Lock()
        g = UNIQUE_PRESETS.get(s["unique"])

        def work(plan):
            """Decode + local filters in a worker thread; returns candidate records (file written)."""
            v, idx, fps, n = plan
            out = []
            stem = _safe_stem(v["path"], self.manifest["stems"])
            local = None
            if g is not None:
                local = DedupFilter("exact" if s["unique"] == "exact" else "near", g[0], g[1], window=5)
            try:
                src = open_source(v["path"], "auto", hw_accel=False)
            except Exception as e:
                return out, f"{os.path.basename(v['path'])}: {e}"
            try:
                for k, frame in iter_frames(src, idx, cancel):
                    with lock:
                        done[0] += 1
                        if done[0] % 5 == 0:
                            progress(done[0], total, f"{os.path.basename(v['path'])} · frame {k}")
                    sharp = sharpness(frame)
                    if s["min_sharpness"] and sharp < s["min_sharpness"]:
                        out.append(dict(skip="blurry"))
                        continue
                    if local is not None:
                        sig = Signature(frame)
                        if local.is_duplicate(sig)[0]:
                            out.append(dict(skip="duplicate"))
                            continue
                        local.add(sig)
                    fn = f"{stem}_{k:07d}{s['image_ext']}"
                    from AIO_QC.tools.frame_extractor.core.imageio import imwrite
                    imwrite(os.path.join(self.images_dir, fn), frame, s["image_ext"], s["quality"])
                    out.append(dict(file=f"images/{fn}", source=os.path.abspath(v["path"]), video=os.path.abspath(v["path"]),
                                    frame=int(k), time=round(float(src.time_of(k)), 6), w=int(frame.shape[1]),
                                    h=int(frame.shape[0]), sharp=round(float(sharp), 1),
                                    _sig=global_sig(frame) if g is not None else None))
            finally:
                src.release()
            return out, None

        from AIO_QC.core.procs import cpu_budget
        n_workers = workers or max(1, min(len(plans), cpu_budget("balanced"), 4))
        existing = {it["file"] for it in self.items}
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            # results are merged in source order -> the same input always gives the same test set
            for recs, err in ex.map(work, plans):
                if err:
                    stats["failed"].append(err)
                for r in recs:
                    stats["candidates"] += 1
                    if r.get("skip"):
                        stats["duplicates" if r["skip"] == "duplicate" else "blurry"] += 1
                        continue
                    sig = r.pop("_sig")
                    if r["file"] in existing:                    # re-adding the same video: same file
                        continue
                    if sig is not None and uniq.is_dup(sig):
                        stats["duplicates"] += 1
                        try:
                            os.remove(os.path.join(self.root, r["file"]))
                        except OSError:
                            pass
                        continue
                    if sig is not None:
                        uniq.add(sig)
                    self.items.append(r)
                    existing.add(r["file"])
                    stats["added"] += 1
        for f in folders:
            if cancel():
                break
            self._add_folder(f, s, uniq, stats, progress, cancel)
            done[0] += 1
        for v, idx, fps, n in source_plans:
            self.manifest["sources"].append(dict(type="video", path=os.path.abspath(v["path"]), frames=n, fps=fps,
                                                 start=v.get("start"), end=v.get("end"), sampled=len(idx)))
        for f in folders:
            self.manifest["sources"].append(dict(type="folder", path=os.path.abspath(f["path"]),
                                                 copied=bool(s["copy_folders"])))
        stats["seconds"] = round(time.time() - t0, 2)
        stats["cancelled"] = bool(cancel())
        self.save()
        progress(total, total, "done")
        return stats

    def _add_folder(self, f, s, uniq, stats, progress, cancel):
        from AIO_QC.tools.frame_extractor.core.metrics import sharpness
        imgs = D.list_images(f["path"])
        existing = {self.abspath(it) for it in self.items}
        check = UNIQUE_PRESETS.get(s["unique"]) is not None
        for i, p in enumerate(imgs):
            if cancel():
                return
            if i % 20 == 0:
                progress(i, len(imgs), f"{os.path.basename(f['path'])} · {i}/{len(imgs)}")
            stats["candidates"] += 1
            if os.path.abspath(p) in existing:
                continue
            w, h = D.image_size(p)
            rec = dict(source=os.path.abspath(f["path"]), w=w, h=h)
            if check or s["min_sharpness"]:
                img = imread(p)
                if img is None:
                    stats["failed"].append(p)
                    continue
                if s["min_sharpness"] and sharpness(img) < s["min_sharpness"]:
                    stats["blurry"] += 1
                    continue
                if check:
                    sig = global_sig(img)
                    if uniq.is_dup(sig):
                        stats["duplicates"] += 1
                        continue
                    uniq.add(sig)
            if s["copy_folders"]:
                rel = os.path.relpath(p, f["path"]).replace(os.sep, "_")
                dst = os.path.join(self.images_dir, rel)
                shutil.copy2(p, dst)
                for lab in (D.yolo_label_path(p), D.labelme_path(p)):
                    if lab and lab.endswith(".txt"):
                        os.makedirs(self.labels_dir, exist_ok=True)
                        shutil.copy2(lab, os.path.join(self.labels_dir, os.path.splitext(rel)[0] + ".txt"))
                    elif lab:
                        shutil.copy2(lab, os.path.splitext(dst)[0] + ".json")
                rec["file"] = "images/" + rel
            else:
                rec["file"] = os.path.abspath(p)
            self.items.append(rec)
            stats["added"] += 1
        if not self.names:
            names, _ = D.find_names(f["path"])
            if names:
                self.manifest["names"] = names

    def remove_items(self, files):
        drop = set(files)
        for it in self.items:
            if it["file"] in drop and not os.path.isabs(it["file"]):
                for p in (self.abspath(it), self.label_path(it, "yolo"), self.label_path(it, "labelme")):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        self.manifest["items"] = [it for it in self.items if it["file"] not in drop]
        self.save()

    # ---- labels
    def label_path(self, item, fmt="yolo"):
        p = self.abspath(item)
        if fmt == "labelme":
            return os.path.splitext(p)[0] + ".json"
        if os.path.isabs(item["file"]):
            return D.yolo_label_path(p) or os.path.splitext(p)[0] + ".txt"
        return os.path.join(self.labels_dir, os.path.splitext(os.path.basename(p))[0] + ".txt")

    def dataset(self, fmt="auto", progress=None, cancel=None):
        """Ground truth for every item (Dataset; unlabelled items have boxes None)."""
        ds = D.load_folder(self.root, fmt, names=dict(self.names) or None, images=self.image_paths(),
                           prelabel_hashes=self.manifest["prelabels"], progress=progress, cancel=cancel)
        for it, rec in zip(ds.items, self.items):
            it.meta.update({k: rec[k] for k in ("video", "frame", "time", "source") if k in rec})
            if not it.w:
                it.w, it.h = rec.get("w", 0), rec.get("h", 0)
        if ds.names and not self.names:
            self.set_names(ds.names)
        return ds

    def write_prelabels(self, preds_by_file, names, fmt="yolo", only_unlabelled=True):
        """Pre-annotations for labelling: preds_by_file {item file: (n, 6) array in test-set class ids}.
        Fingerprinted so they count as ground truth only after a person has edited them."""
        from AIO_QC.tools.frame_extractor.core.detector import Detection, write_labelme
        if not self.names and names:
            self.manifest["names"] = dict(names)
        written = 0
        os.makedirs(self.labels_dir, exist_ok=True)
        for it in self.items:
            pr = preds_by_file.get(it["file"])
            if pr is None:
                continue
            lp = self.label_path(it, fmt)
            if only_unlabelled and (os.path.isfile(self.label_path(it, "yolo")) or
                                    os.path.isfile(self.label_path(it, "labelme"))):
                continue
            w, h = it.get("w") or 0, it.get("h") or 0
            if not w:
                w, h = D.image_size(self.abspath(it))
            if fmt == "labelme":
                dets = [Detection(*map(float, b[:4]), float(b[4]), int(b[5]), self.names.get(int(b[5]), str(int(b[5]))))
                        for b in pr]
                write_labelme(lp, os.path.basename(self.abspath(it)), w, h, dets, flags={"prelabel": True})
            else:
                D.write_yolo_labels(lp, w, h, pr[:, :4], pr[:, 5].astype(int))
            self.manifest["prelabels"][os.path.abspath(lp)] = D.file_sha1(lp)
            written += 1
        self.save()
        return written

    def mark_reviewed(self, label_paths=None):
        """Accept pre-annotations as ground truth (all, or the given label files)."""
        if label_paths is None:
            self.manifest["prelabels"] = {}
        else:
            for p in label_paths:
                self.manifest["prelabels"].pop(os.path.abspath(p), None)
        self.save()

    def summary(self):
        vids = {it.get("video") for it in self.items if it.get("video")}
        return dict(name=self.manifest["name"], images=len(self.items), videos=len(vids),
                    folders=sum(1 for s in self.manifest["sources"] if s["type"] == "folder"),
                    names=self.names, root=self.root)
