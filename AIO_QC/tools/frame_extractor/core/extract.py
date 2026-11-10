# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Batch frame extraction (Qt-free, runs inside worker processes and the CLI).

Pipeline per frame (only the stages you enable cost anything):
    decode -> ROI crop -> [scene cut?] -> candidate selection
           -> blur filter (rejects may be sampled as negatives)
           -> near-duplicate filter
           -> [model mining: keep only uncertain / empty / disagreeing / chosen-class frames]
           -> save image [+ LabelMe JSON / YOLO txt pre-annotations]

Extra stages:
    cross-video dedup : pHash vs. everything already extracted into the dataset (core.hashdb)
    temporal mining   : primary model on consecutive frames; frames where a tracked object
                        vanishes ("flicker") or appears only briefly ("blip") are saved
    ROI tracking      : the crop follows the object (core.tracker); such jobs are never split
    resume            : checkpoints in <output>/.fx_resume let a crashed/cancelled run continue

Candidate selection:
    "step"     : every Nth frame.
    "sharpest" : the sharpest frame of every window of N frames.
Both use a grid anchored at the job start, so a video split into chunks across
workers yields the same frames as a single pass (scene-cut restarts and
duplicate checks are chunk-local, so those can differ right at chunk edges).
"""
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict, field, fields


from AIO_QC.tools.frame_extractor.core.sources import open_source, parse_time
from AIO_QC.tools.frame_extractor.core.metrics import sharpness, SceneDetector
from AIO_QC.tools.frame_extractor.core.imageio import imread
from AIO_QC.tools.frame_extractor.core import frame_sources

NEGATIVE_DIR = "negatives_blurry"


@dataclass
class ExtractSettings:
    mode: str = "step"            # "step" | "sharpest"
    step: int = 1                 # N
    min_sharpness: float = 0.0    # 0 = blur filter off
    keep_blurry_every: int = 0    # save 1 of every K rejected-blurry frames as negatives (0 = off)
    dedup: bool = False
    dedup_mode: str = "near"      # "exact": only bit-identical repeats | "near": also near-identical frames
    dedup_threshold: float = 2.0  # near: whole-frame change (%) below which ...
    dedup_local: float = 8.0      # ... AND largest local (16x9 cell) change (%) below which a frame is a duplicate
    dedup_window: int = 5         # compared with the last N saved frames (catches A -> B -> A)
    scene_detect: bool = False
    scene_threshold: float = 0.4
    ext: str = ".jpg"
    quality: int = 95             # 0-100, mapped per format (core.imageio)
    name_template: str = "{video}_frame_{frame}"   # core.naming tokens; "/" makes sub-folders
    on_exists: str = "overwrite"  # overwrite | suffix | skip
    backend: str = "auto"         # auto | opencv | pyav
    hw_accel: bool = True
    # --- models / mining
    models: list = field(default_factory=list)   # .onnx / .pt paths
    det_conf: float = 0.25
    det_iou: float = 0.45
    det_imgsz: int = 0            # 0 = from model
    det_device: str = "auto"      # auto | cpu
    mine: list = field(default_factory=list)     # subset of detector.MINING_CRITERIA
    unc_low: float = 0.25
    unc_high: float = 0.60
    mine_classes: list = field(default_factory=list)
    disagree_iou: float = 0.5
    preannot: str = "none"        # none | labelme | yolo
    det_threads: int = 0          # set by the scheduler (CPU threads per worker)
    # --- temporal consistency (mine "temporal"): primary model on every stride-th frame
    temporal_stride: int = 1
    temporal_gap: int = 2         # object missing / present for at most this many samples
    # --- cross-video dedup
    cross_dedup: bool = False
    cross_dedup_dist: int = 6     # max Hamming distance of the 64-bit pHash (0-64)
    hash_db: str = ""             # "" = <output root>/.frameextractor_hashes.db
    # --- ROI tracking
    roi_track: bool = False
    roi_tracker: str = "auto"     # auto | csrt | kcf | mil
    # --- robustness
    resume: bool = True           # continue interrupted jobs from their last checkpoint
    # --- model runtime (Settings -> Models): .pt/.engine run in this Python environment
    env_python: str = ""
    det_half: bool = False
    det_sahi: bool = False        # sliced inference (small objects in large frames); needs sahi in the env
    sahi_slice: int = 640
    sahi_overlap: float = 0.2

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def save_preset(self, path):
        d = self.to_dict()
        for k in ("det_threads", "hash_db", "env_python"):
            d.pop(k, None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load_preset(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class Job:
    video_path: str
    output_dir: str
    start: str = ""               # time string; "" = beginning
    end: str = ""                 # time string; "" = end of video
    roi: tuple = None             # (x1, y1, x2, y2) in full-res video pixels
    chunk: int = 0                # this worker's part ...
    chunks: int = 1               # ... of this many equal frame ranges
    frames: list = None           # explicit picks: [frame, roi | None] (timeline markers); ignores start/end


def _imwrite(path, img, ext, quality=95):
    from AIO_QC.tools.frame_extractor.core.imageio import imwrite
    imwrite(path, img, ext, quality)   # any format, quality, non-ASCII Windows paths


def _clamp_roi(roi, w, h):
    if not roi:
        return None
    x1, y1, x2, y2 = (int(v) for v in roi)
    x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
    y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError(f"ROI {roi} is empty for a {w}x{h} video")
    return x1, y1, x2, y2


def resolve_range(src, job, step=1):
    """Full [start, end] of the job, then this chunk's [lo, hi] sub-range.
    Chunk edges are snapped to the sampling grid so no window straddles two chunks."""
    last = src.frame_count - 1
    start = src.index_of(parse_time(job.start)) if job.start else 0
    end = src.index_of(parse_time(job.end)) if job.end else last
    start, end = max(0, min(start, last)), max(0, min(end, last))
    if end < start:
        raise ValueError("End time is before start time")
    n = end - start + 1
    k, parts = int(job.chunk), max(1, int(job.chunks))
    edge = lambda i: end + 1 if i >= parts else start + (n * i // parts) // step * step
    return start, end, edge(k), edge(k + 1) - 1


RESUME_DIR = ".fx_resume"
_KEY_EXCLUDE = {"det_threads", "hw_accel", "resume", "backend", "hash_db"}


def _job_key(job, s):
    d = {k: v for k, v in s.to_dict().items() if k not in _KEY_EXCLUDE}
    d["job"] = [os.path.abspath(job.video_path), job.start, job.end,
                list(job.roi) if job.roi else None, job.chunk, job.chunks]
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


class _Checkpoint:
    """Atomic JSON checkpoint per job/chunk; only valid for identical settings."""

    def __init__(self, path, key):
        self.path, self.key = path, key

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                st = json.load(f)
            return st if st.get("key") == self.key else None
        except (OSError, ValueError):
            return None

    def save(self, **state):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(state, key=self.key), f)
        os.replace(tmp, self.path)


class TemporalMonitor:
    """Follows primary-model detections over consecutive samples (IoU tracker)
    and reports the samples where the model is temporally inconsistent:
      flicker : a tracked object is missing for 1..gap samples, then comes back
      blip    : an object appears for at most `gap` samples and disappears
    Both usually mean a missed or a false detection - valuable training frames."""

    def __init__(self, gap=2, iou=0.3):
        from collections import deque
        self.gap, self.iou = max(1, int(gap)), iou
        self.tracks = []
        self.n = 0
        self.buffer = deque(maxlen=2 * self.gap + 2)
        self.emitted = set()

    def step(self, idx, img, dets):
        n = self.n
        self.n += 1
        self.buffer.append((n, idx, img, dets))
        flagged, used = {}, set()
        for t in self.tracks:
            best, bj = 0.0, -1
            for j, d in enumerate(dets):
                if j in used or d.name != t["name"]:
                    continue
                v = t["box"].iou(d)
                if v >= self.iou and v > best:
                    best, bj = v, j
            if bj >= 0:
                used.add(bj)
                missed = n - t["last"] - 1
                if 1 <= missed <= self.gap:
                    for m in range(t["last"] + 1, n):
                        flagged[m] = "flicker"
                t.update(box=dets[bj], last=n, hits=t["hits"] + 1)
                t["seen"].append(n)
        for j, d in enumerate(dets):
            if j not in used:
                self.tracks.append(dict(name=d.name, box=d, last=n, first=n, hits=1, seen=[n]))
        alive = []
        for t in self.tracks:
            if n - t["last"] > self.gap:
                if t["hits"] <= self.gap and t["first"] > 0:
                    for m in t["seen"]:
                        flagged.setdefault(m, "blip")
            else:
                alive.append(t)
        self.tracks = alive
        out = []
        for bn, bidx, bimg, bdets in self.buffer:
            if bn in flagged and bn not in self.emitted:
                self.emitted.add(bn)
                out.append((bidx, bimg, bdets, flagged[bn]))
        return out


def detector_kwargs(s):
    return dict(iou=s.det_iou, imgsz=s.det_imgsz, device=s.det_device, threads=s.det_threads,
                env_python=s.env_python, half=s.det_half,
                sahi={"slice": s.sahi_slice, "overlap": s.sahi_overlap} if s.det_sahi else None)


def extract_frames(job, s, progress=None, cancelled=lambda: False, detectors=None):
    """Save exactly the picked frames (timeline markers). Filters don't apply - the user chose them;
    models only add pre-annotations; the dataset hash DB is updated so later auto runs skip them."""
    from AIO_QC.tools.frame_extractor.core.markers import MarkerExtractor
    picks = sorted({int(f[0] if isinstance(f, (list, tuple)) else f):
                    (tuple(f[1]) if isinstance(f, (list, tuple)) and len(f) > 1 and f[1] else job.roi)
                    for f in job.frames}.items())
    if s.models and detectors is None and s.preannot != "none":
        from AIO_QC.tools.frame_extractor.core.detector import load_detectors
        detectors = load_detectors(s.models, s.det_conf, **detector_kwargs(s))
    hdb = (s.hash_db or os.path.join(os.path.dirname(os.path.dirname(job.output_dir)),
                                     ".frameextractor_hashes.db")) if s.cross_dedup else None
    mx = MarkerExtractor(job.video_path, job.output_dir, s.backend, s.ext, detectors, s.preannot, s.det_conf, hdb,
                         quality=s.quality, name_template=s.name_template, on_exists=s.on_exists)
    stats = dict(saved=0, frames=0, failed=0, backend=mx.src.backend, vfr=mx.src.is_vfr, markers=len(picks))
    try:
        for i, (idx, roi) in enumerate(picks):
            if cancelled():
                stats["cancelled"] = True
                break
            try:
                mx.save(idx, roi)
                stats["saved"] += 1
            except Exception:
                stats["failed"] += 1
            stats["frames"] += 1
            if progress:
                progress(i + 1, len(picks), stats["saved"])
    finally:
        mx.close()
    return stats


def extract(job, settings, progress=None, cancelled=lambda: False, detectors=None):
    """Run one job (or one chunk of it). progress(done, total, saved) is called
    periodically. Returns a stats dict."""
    s = settings
    if job.frames:
        return extract_frames(job, s, progress, cancelled, detectors)
    src = open_source(job.video_path, s.backend, s.hw_accel)
    csv_file = hdb = None
    try:
        step = max(1, int(s.step))
        start, end, lo, hi = resolve_range(src, job, step)
        roi = _clamp_roi(job.roi, src.width, src.height)
        if s.roi_track and not roi:
            raise ValueError("ROI tracking needs an ROI (set one for this job)")

        mining = [m for m in s.mine if m] if s.models else []
        temporal = "temporal" in mining
        frame_mining = [m for m in mining if m != "temporal"]
        if s.models and detectors is None and (mining or s.preannot != "none"):
            from AIO_QC.tools.frame_extractor.core.detector import load_detectors
            detectors = load_detectors(s.models, min(s.det_conf, s.unc_low if "uncertain" in mining else 1.0),
                                       **detector_kwargs(s))
        use_models = bool(detectors) and (mining or s.preannot != "none")

        name = os.path.splitext(os.path.basename(job.video_path))[0]
        video_key = os.path.abspath(job.video_path)
        os.makedirs(job.output_dir, exist_ok=True)
        neg_dir = os.path.join(job.output_dir, NEGATIVE_DIR)
        total = max(0, hi - lo + 1)

        # ---------- resume
        ckpt = _Checkpoint(os.path.join(job.output_dir, RESUME_DIR, f"{name}_{lo}-{hi}.json"), _job_key(job, s))
        state = ckpt.load() if s.resume else None
        if state and state.get("done"):
            st = dict(state["stats"], resumed="already complete")
            if progress:
                progress(total, total, st.get("saved", 0))
            return st
        stats = dict(saved=0, duplicates=0, cross_duplicates=0, blurry=0, negatives=0, scenes=1,
                     mined_out=0, frames=0, backend=src.backend, vfr=src.is_vfr)
        first, phase = lo, start
        if state:
            stats = dict(state["stats"])
            first, phase = state["next_idx"], state["phase"]
            roi = tuple(state["roi"]) if state.get("roi") else roi
            stats["resumed_from"] = first

        if use_models and s.preannot == "yolo" and detectors[0].names:
            with open(os.path.join(job.output_dir, "classes.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(detectors[0].names[i] for i in sorted(detectors[0].names)) + "\n")
        log = None
        if use_models and mining:
            log_path = os.path.join(job.output_dir, f"mining_{name}_{lo}-{hi}.csv")
            new = not (state and os.path.exists(log_path))
            csv_file = open(log_path, "a" if not new else "w", newline="", encoding="utf-8")
            log = csv.writer(csv_file)
            if new:
                log.writerow(["frame", "time_s", "file", "reasons", "detections", "min_conf"])
        if s.cross_dedup:
            from AIO_QC.tools.frame_extractor.core.hashdb import HashDB, phash, default_db_path
            hdb = HashDB(s.hash_db or default_db_path(os.path.dirname(os.path.dirname(job.output_dir))),
                         s.cross_dedup_dist)

        need_all_pixels = s.mode == "sharpest" or s.scene_detect or temporal or s.roi_track
        scene = SceneDetector(s.scene_threshold) if s.scene_detect else None
        tracker = None
        if s.roi_track:
            from AIO_QC.tools.frame_extractor.core.tracker import RoiTracker
            tracker = RoiTracker(s.roi_tracker)
        monitor = TemporalMonitor(s.temporal_gap) if temporal else None
        from AIO_QC.tools.frame_extractor.core.dedup import DedupFilter, Signature
        pos_dd = DedupFilter(s.dedup_mode, s.dedup_threshold, s.dedup_local, s.dedup_window)
        neg_dd = DedupFilter(s.dedup_mode, s.dedup_threshold, s.dedup_local, s.dedup_window)
        last_saved = [state.get("last_saved") if state else None]
        if state and s.dedup and last_saved[0]:
            p = os.path.join(job.output_dir, last_saved[0])
            if os.path.exists(p):
                img = imread(p)
                if img is not None:
                    pos_dd.add(Signature(img))
        saved_idx = set()
        det_cache = {}
        from AIO_QC.tools.frame_extractor.core.naming import Namer
        namer = Namer(s.name_template, s.on_exists, job.video_path, src.fps, src.time_of)
        if state:
            namer.counter = int(state.get("counter", 0))

        def save(idx, img, folder, kind="batch", reason=""):
            """Write via the naming template; returns the path relative to the job folder (None = skipped)."""
            base = folder or job.output_dir
            path, _ = namer.path(base, idx, s.ext, img, kind, reason, roi)
            if path is None:
                stats["skipped_existing"] = stats.get("skipped_existing", 0) + 1
                return None
            _imwrite(path, img, s.ext, s.quality)
            frame_sources.record(path, job.video_path, idx, src.time_of(idx))
            return os.path.relpath(path, job.output_dir).replace(os.sep, "/")

        def run_models(idx, img, primary_only=False):
            c = det_cache.get(idx)
            if c is None or (not primary_only and len(c) < len(detectors)):
                c0 = c[0] if c else detectors[0].detect(img)
                c = [c0] if primary_only else [c0] + [d.detect(img) for d in detectors[1:]]
                det_cache[idx] = c
                while len(det_cache) > 8:
                    det_cache.pop(next(iter(det_cache)))
            return c

        def cross_dup(idx, img):
            if hdb is None:
                return None, False
            from AIO_QC.tools.frame_extractor.core.hashdb import thumb_sig
            h, ts_ = phash(img), thumb_sig(img)
            if hdb.find(h, video_key, idx, ts_, s.dedup_threshold, s.dedup_local):
                stats["cross_duplicates"] += 1
                return (h, ts_), True
            return (h, ts_), False

        def write_outputs(idx, img, per_model, reasons, h=None):
            fname = save(idx, img, None, "mined" if reasons else "batch", "+".join(reasons))
            if fname is None:
                return
            saved_idx.add(idx)
            last_saved[0] = fname
            stats["saved"] += 1
            if hdb is not None:
                from AIO_QC.tools.frame_extractor.core.hashdb import thumb_sig
                hh, ts_ = h if h is not None else (phash(img), thumb_sig(img))
                hdb.add(hh, video_key, idx, fname, ts_)
            if use_models and per_model is not None:
                ih, iw = img.shape[:2]
                stem = os.path.join(job.output_dir, os.path.splitext(fname)[0])
                final = [d for d in per_model[0] if d.conf >= s.det_conf]
                if s.preannot == "labelme":
                    from AIO_QC.tools.frame_extractor.core.detector import write_labelme
                    write_labelme(stem + ".json", os.path.basename(fname), iw, ih, final,
                                  flags={r: True for r in reasons})
                elif s.preannot == "yolo":
                    from AIO_QC.tools.frame_extractor.core.detector import write_yolo
                    write_yolo(stem + ".txt", iw, ih, final)
                if log is not None:
                    alld = [d for dets in per_model for d in dets]
                    log.writerow([idx, f"{src.time_of(idx):.3f}", fname, "|".join(reasons), len(alld),
                                  f"{min((d.conf for d in alld), default=0):.3f}"])
            for r in reasons:
                stats[f"mined_{r}"] = stats.get(f"mined_{r}", 0) + 1

        def handle_blurry(idx, img):
            stats["blurry"] += 1
            k = int(s.keep_blurry_every)
            if k <= 0 or (stats["blurry"] - 1) % k != 0:
                return
            if s.dedup:
                sig = Signature(img)
                if neg_dd.is_duplicate(sig)[0]:
                    return
                neg_dd.add(sig)
            if save(idx, img, neg_dir, "negative") is not None:
                stats["negatives"] += 1

        def accept(idx, img):
            if idx in saved_idx:
                return
            sig = None
            if s.dedup:
                sig = Signature(img)
                dup, why = pos_dd.is_duplicate(sig)
                if dup:
                    stats["duplicates"] += 1
                    if why == "exact":
                        stats["exact_duplicates"] = stats.get("exact_duplicates", 0) + 1
                    return
            if mining and not frame_mining:      # temporal-only mining: grid frames aren't kept
                stats["mined_out"] += 1
                return
            h, dup = cross_dup(idx, img)
            if dup:
                return
            per_model, reasons = None, []
            if use_models:
                per_model = run_models(idx, img)
                if frame_mining:
                    from AIO_QC.tools.frame_extractor.core.detector import mining_reasons
                    reasons = mining_reasons(per_model, frame_mining, s.det_conf, s.unc_low, s.unc_high,
                                             s.mine_classes, s.disagree_iou, detectors)
                    if not reasons:
                        stats["mined_out"] += 1
                        return
            if s.dedup:
                pos_dd.add(sig)
            write_outputs(idx, img, per_model, reasons, h)

        def temporal_save(idx, img, dets, reason):
            # hard examples: no blur / local-dedup filter (blur often *is* why the model failed)
            if idx in saved_idx:
                return
            h, dup = cross_dup(idx, img)
            if not dup:
                write_outputs(idx, img, [dets], [reason], h)

        def is_blurry(img, sharp=None):
            if s.min_sharpness <= 0:
                return False
            return (sharpness(img) if sharp is None else sharp) < s.min_sharpness

        best = [None]      # sharpest mode: running best (sharpness, idx, image)

        def flush_window():
            if best[0] is not None:
                accept(best[0][1], best[0][2])
            best[0] = None

        snapshot = None
        t_ckpt = time.monotonic()
        t_last = 0.0
        tracker_ready = False
        if first <= hi:
            src.seek(first)
        for idx in range(first, hi + 1):
            on_grid = (idx - phase) % step == 0
            if on_grid:
                # safe point: window closed, everything before idx fully handled
                if s.mode == "sharpest":
                    flush_window()
                if s.resume:
                    snapshot = dict(next_idx=idx, phase=phase, roi=list(roi) if roi else None,
                                    stats=dict(stats), last_saved=last_saved[0], done=False, counter=namer.counter)
                    if time.monotonic() - t_ckpt > 3.0:
                        ckpt.save(**snapshot)
                        t_ckpt = time.monotonic()
            if cancelled():
                stats["cancelled"] = True
                break
            if need_all_pixels or on_grid:
                ok, frame = src.read()
            else:
                ok, frame = src.grab(), None   # decode only: no conversion/copy
            if not ok:
                break
            stats["frames"] += 1

            if frame is not None:
                if tracker is not None:
                    if not tracker_ready:
                        tracker.init(frame, roi)
                        tracker_ready = True
                    else:
                        r = tracker.update(frame)
                        if r is None:
                            stats.setdefault("track_lost_at", idx)
                        else:
                            roi = r
                crop = frame[roi[1]:roi[3], roi[0]:roi[2]] if roi else frame
                if scene is not None and scene.update(crop) and idx != first:
                    stats["scenes"] += 1
                    pos_dd.reset()                 # new shot: never a duplicate of the previous one
                    phase = idx
                    on_grid = True
                if monitor is not None and (idx - lo) % max(1, s.temporal_stride) == 0:
                    dets = [d for d in run_models(idx, crop, primary_only=True)[0] if d.conf >= s.det_conf]
                    kept = crop.copy() if roi else crop
                    for fi, fimg, fdets, reason in monitor.step(idx, kept, dets):
                        temporal_save(fi, fimg, fdets, reason)
                if s.mode == "step":
                    if on_grid:
                        if is_blurry(crop):
                            handle_blurry(idx, crop)
                        else:
                            accept(idx, crop)
                else:
                    if on_grid:
                        flush_window()
                    sh = sharpness(crop)
                    if is_blurry(crop, sh):
                        handle_blurry(idx, crop)
                    elif best[0] is None or sh > best[0][0]:
                        best[0] = (sh, idx, crop.copy() if roi else crop)

            now = time.monotonic()
            if progress and now - t_last > 0.25:
                progress(idx - lo + 1, total, stats["saved"])
                t_last = now

        if stats.get("cancelled"):
            if s.resume and snapshot:
                ckpt.save(**snapshot)        # resume from the last safe point
        else:
            if s.mode == "sharpest":
                flush_window()
            if s.resume:
                ckpt.save(next_idx=hi + 1, phase=phase, roi=list(roi) if roi else None,
                          stats=stats, last_saved=last_saved[0], done=True, counter=namer.counter)
        if progress:
            progress(total if not stats.get("cancelled") else stats["frames"], total, stats["saved"])
        return stats
    finally:
        if csv_file:
            csv_file.close()
        if hdb is not None:
            hdb.close()
        src.release()


def job_process(job_id, job, settings, queue, cancel_event):
    """Entry point for worker processes: reports through a multiprocessing queue."""
    t0 = time.monotonic()

    def progress(done, total, saved):
        queue.put(("progress", job_id, done, total, saved, time.monotonic() - t0))

    try:
        queue.put(("started", job_id))
        stats = extract(job, settings, progress, cancel_event.is_set)
        queue.put(("done", job_id, stats))
    except Exception as e:
        queue.put(("error", job_id, f"{type(e).__name__}: {e}"))
