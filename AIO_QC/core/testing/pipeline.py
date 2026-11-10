# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Headless Model Lab pipeline - the same steps the Model Lab window runs, usable from the command line
(`aio-qc eval ...`) and from scripts:

    data   = open_data(path)                       # test set folder, data.yaml, or a YOLO / LabelMe image folder
    run, _ = evaluate(data, specs, cache)          # every model on every image (cached, resumable) + metrics
    vids   = analyse_videos([...], specs, cache)   # label-free behaviour on whole videos
    make_report("report.html", "detailed", run, vids)

Nothing here imports Qt.
"""
import math
import os
import threading
from dataclasses import dataclass, field

from AIO_QC.tools.frame_extractor.core.imageio import imread

from AIO_QC.core.testing import dataset as D
from AIO_QC.core.testing.evaluation import LabRun
from AIO_QC.core.testing.report import ReportBuilder, write_exports, write_report
from AIO_QC.core.testing.runner import (
    ModelSpec,
    PredCache,
    default_cache_dir,
    image_key,
    load_predictions,
    run_on_images,
    run_on_video,
)
from AIO_QC.core.testing.testset import TestSet
from AIO_QC.core.testing import videoeval as V


@dataclass
class Data:
    dataset: D.Dataset
    root: str
    testset: TestSet = None
    label: str = ""
    keys: list = field(default_factory=list)
    qc: bool = False            # the dataset open in Image QC (AIO QC's file list, labels and class names)

    @property
    def images(self):
        return list(zip(self.keys, [it.path for it in self.dataset.items]))


def open_data(path, fmt="auto", split="val", progress=None, cancel=None):
    """Test set folder / data.yaml / image folder -> Data (ground truth loaded; unlabelled images allowed)."""
    path = os.path.abspath(path)
    ts = None
    if os.path.isdir(path) and TestSet.is_testset(path):
        ts = TestSet.open(path)
        ds = ts.dataset(fmt, progress=progress, cancel=cancel)
        root, label = ts.root, ts.manifest.get("name") or os.path.basename(path)
    elif os.path.isfile(path) and path.lower().endswith((".yaml", ".yml")):
        ds = D.load_data_yaml(path, split)
        root, label = os.path.dirname(path), os.path.basename(os.path.dirname(path))
    elif os.path.isdir(path):
        ds = D.load_folder(path, fmt, progress=progress, cancel=cancel)
        root, label = path, os.path.basename(path)
    else:
        raise FileNotFoundError(f"Not a test set, data.yaml or image folder: {path}")
    if not ds.items and ts is None:          # an empty test set is fine: frames get added next
        raise ValueError(f"No images found in {path}")
    from AIO_QC.core.testing.labels_io import apply_exclusions
    apply_exclusions(ds, root, ts)
    keys = [image_key(it.path, root) for it in ds.items]
    return Data(ds, root, ts, label, keys)


def _labelme_image(ann_path):
    """AIO QC's rule for a LabelMe JSON's image: its imagePath (relative to the JSON, or just the file name next to
    it), else an image with the JSON's name."""
    import json as _json
    folder = os.path.dirname(ann_path)
    cands = []
    try:
        with open(ann_path, encoding="utf-8") as f:
            head = f.read(1 << 16)
        ip = None
        try:
            ip = _json.loads(head).get("imagePath") if head.rstrip().endswith("}") else None
        except ValueError:
            ip = None
        if ip is None:
            import re
            m = re.search(r'"imagePath"\s*:\s*"((?:[^"\\]|\\.)*)"', head)
            ip = _json.loads('"' + m.group(1) + '"') if m else None
        if ip:
            ip = str(ip).replace("\\", "/")
            cands += [os.path.join(folder, ip), os.path.join(folder, os.path.basename(ip))]
    except OSError:
        pass
    stem = os.path.splitext(ann_path)[0]
    cands += [stem + e for e in (".jpg", ".png", ".jpeg", ".bmp", ".webp", ".JPG", ".PNG", ".JPEG")]
    return next((os.path.abspath(c) for c in cands if os.path.isfile(c)), "")


def open_qc(root, fmt, pairs, names=None, label="", progress=None, cancel=None):
    """The dataset open in Image QC as test data: AIO QC's own file list (image, annotation) - read on the UI
    thread and passed in as plain strings - and the dataset's class names; nothing is searched again."""
    root = os.path.abspath(str(root))
    images, label_paths = [], {}
    for i, (img, ann) in enumerate(pairs):
        if cancel is not None and cancel():
            break
        if not img and ann and fmt == "labelme":
            img = _labelme_image(ann)
        if not img:
            continue
        images.append(img)
        label_paths[img] = ann if ann and os.path.isfile(ann) else ""
        if progress and i % 5000 == 0:
            progress(i, len(pairs), "Listing the dataset")
    ds = D.load_folder(root, fmt, names=names, images=images, label_paths=label_paths, progress=progress,
                       cancel=cancel)
    ds.root = root
    if not ds.items:
        raise ValueError("The open dataset has no images.")
    from AIO_QC.core.testing.labels_io import apply_exclusions
    apply_exclusions(ds, root, None)
    from AIO_QC.core.model_registry import qc_image_key
    keys = [qc_image_key(it.path, root) for it in ds.items]
    return Data(ds, root, None, label or os.path.basename(root), keys, qc=True)


def cache_for(data=None):
    """Prediction cache: inside a test set (travels with it); for the dataset open in Image QC the dataset's own
    .aio_qc/predictions.db (shared with Model Check); else the one in AIO QC's data folder."""
    if data is not None and getattr(data, "qc", False):
        from AIO_QC.core.model_registry import shared_cache_path
        p = shared_cache_path(data.root)
        p.parent.mkdir(parents=True, exist_ok=True)
        return PredCache(str(p))
    folder = data.testset.cache_dir if data is not None and data.testset is not None else default_cache_dir()
    os.makedirs(folder, exist_ok=True)
    return PredCache(os.path.join(folder, "predictions.db"))


def spec_from_cfg(path, cfg=None, label=None, **over):
    """ModelSpec from a weights path and the app's model settings (env, device, image size, half, SAHI)."""
    cfg = cfg or {}
    sahi = {"slice": cfg.get("sahi_slice", 640), "overlap": cfg.get("sahi_overlap", 0.2)} if cfg.get("sahi") else None
    kw = dict(path=os.path.abspath(path), label=label or "", imgsz=int(cfg.get("imgsz") or 0),
              half=bool(cfg.get("half")), sahi=sahi, env_python=cfg.get("env_python", "") or "",
              device=str(cfg.get("device") or "auto"))
    kw.update(over)
    return ModelSpec(**kw)


def unique_labels(specs):
    """Distinct display labels (two 'best.pt' from different runs -> 'best (run12)', 'best (run15)')."""
    seen = {}
    for s in specs:
        seen.setdefault(s.label, []).append(s)
    for label, group in seen.items():
        if len(group) < 2:
            continue
        for s in group:
            parent = os.path.basename(os.path.dirname(os.path.dirname(s.path))) if \
                os.path.basename(os.path.dirname(s.path)) == "weights" else os.path.basename(os.path.dirname(s.path))
            s.label = f"{label} ({parent})"
        if len({s.label for s in group}) < len(group):
            for n, s in enumerate(group, 1):
                s.label = f"{label} #{n}"
    return specs


def evaluate(data, specs, cache=None, only_reviewed=True, ground="auto", reference=None, op_conf=None, batch=8,
             progress=None, cancel=None, log=None):
    """Run what is missing, then score. Returns (LabRun, infos)."""
    cache = cache or cache_for(data)
    infos = run_on_images(specs, data.images, cache, progress=progress, cancel=cancel, batch=batch, log=log)
    ok = [s for s in specs if not (infos.get(s.key) or {}).get("error") or cache.keys_for(s.key)]
    run = LabRun.from_cache(data.dataset, ok, cache, data.keys, infos, only_reviewed=only_reviewed, ground=ground,
                            reference=reference, op_conf=op_conf, source=data.root)
    return run, infos


VIDEO_CONF_FLOOR = 0.05      # label-free video analysis never needs the 0.001 floor of mAP: 20-50x less data
VIDEO_MAX_DET = 100
MAX_SAMPLES = 20000          # per video: a 10-hour video is sampled every ~1.8 s instead of every 0.2 s


def video_plan(path, every_s=0.2, start=None, end=None, max_samples=MAX_SAMPLES):
    """(stride in frames, effective seconds between samples, number of samples) - long videos get a coarser step
    so analysis time and memory stay bounded."""
    from AIO_QC.tools.frame_extractor.core.sources import open_source
    src = open_source(path, "auto", hw_accel=False)
    fps = src.fps or 30.0
    dur = src.duration()
    src.release()
    span = max(0.0, (end if end else dur) - (start or 0.0))
    if span / max(every_s, 1e-9) + 1 > max_samples:      # long video: at most max_samples samples, both ends included
        stride = max(1, math.ceil(fps * span / max(1, max_samples - 1) - 1e-9))
    else:
        stride = max(1, int(round(fps * every_s)))
    return stride, stride / fps, min(max_samples, int(span * fps / stride) + 1)


def video_stride(path, every_s=0.2):
    return video_plan(path, every_s)[0]


def video_spec(spec):
    """The model as used for video analysis: higher confidence floor, fewer boxes (cached separately)."""
    import dataclasses
    return dataclasses.replace(spec, conf=max(spec.conf, VIDEO_CONF_FLOOR), max_det=min(spec.max_det, VIDEO_MAX_DET))


def analyse_videos(videos, specs, cache=None, every_s=0.2, op_conf=0.25, reference=None, gap=2, batch=8,
                   progress=None, cancel=None, log=None, max_samples=MAX_SAMPLES):
    """videos: [path | {path, start, end}] -> [(path, analysis dict)] (label-free behaviour, agreement, events).
    progress(video index, done, total, text) - throttled to ~10 updates/s."""
    import time as _time
    cache = cache or cache_for()
    out = []
    vspecs = [video_spec(s) for s in specs]
    for vi, v in enumerate(videos):
        v = v if isinstance(v, dict) else dict(path=v)
        if cancel and cancel():
            break
        stride, step_s, n = video_plan(v["path"], every_s, v.get("start"), v.get("end"), max_samples)
        if log and step_s > every_s * 1.05:
            log(f"{os.path.basename(v['path'])}: long video - a frame every {step_s:.2f} s ({n:,} samples)")
        last = [0.0]

        def prog(_mi, done, total, _ms, text, vi=vi):
            now = _time.monotonic()
            if progress and (now - last[0] > 0.1 or done >= total):
                last[0] = now
                progress(vi, done, total, text)
        infos, idx, times, keys = run_on_video(vspecs, v["path"], cache, stride, v.get("start"), v.get("end"),
                                               prog, cancel, batch, log)
        good = [i for i, s in enumerate(vspecs) if not infos[s.key].get("error")]
        if not good or (cancel and cancel()):
            continue
        preds, names = [], []
        for i in good:
            p, nm = load_predictions(cache, vspecs[i], keys)
            preds.append(p)
            names.append(nm or infos[vspecs[i].key].get("names") or {})
        ref = good.index(reference) if reference is not None and reference in good else None
        a = V.analyse_video([(specs[i].label, nm) for i, nm in zip(good, names)], preds, times, idx,
                            op_conf=op_conf, gap=gap, reference=ref)
        a["specs"] = [specs[i].key for i in good]           # the user's model keys (colours, labels)
        a["speed_ms"] = [infos[vspecs[i].key].get("speed_ms") for i in good]
        a["step_s"] = step_s
        out.append((v["path"], a))
    return out


VIDEO_RUN_MAX_FRAMES = 6000     # sampled video frames scored in a videos-only run (spread evenly over all videos)


def video_run(videos_out, specs, cache, ground="auto", reference=None, op_conf=None, max_frames=VIDEO_RUN_MAX_FRAMES):
    """A LabRun over the frames sampled by analyse_videos - so a run with only videos fills the same result views as
    a test set. Scored against the reference model (no labels exist): with 2+ models and ground 'auto' /
    'reference'; otherwise unscored (plots and behaviour only). Predictions come from the cache (nothing runs).
    Items are video frames: ImageItem(path=video, meta={video, frame, time, source='video'})."""
    from AIO_QC.core.testing.dataset import ImageItem
    from AIO_QC.core.testing.evaluation import LabRun
    from AIO_QC.core.testing.runner import video_key
    vspecs = [video_spec(s) for s in specs]
    total = sum(len(a.get("frames") or []) for _, a in videos_out)
    if not total:
        return None
    step = max(1.0, total / max(1, max_frames))
    pick = {int(k * step) for k in range(int(total / step))} if step > 1.0 else None   # even spread, all videos
    items, keys_per_model = [], [[] for _ in specs]
    g = 0
    for path, a in videos_out:
        frames, times = a.get("frames") or [], a.get("times") or []
        w = h = 0
        try:
            from AIO_QC.tools.frame_extractor.core.sources import open_source
            src = open_source(path, "auto", hw_accel=False)
            w, h = src.width, src.height
            src.release()
        except Exception:
            pass
        vk = video_key(path)
        done_keys = set(a.get("specs") or [])
        for fr, t in zip(frames, times):
            g += 1
            if pick is not None and (g - 1) not in pick:
                continue
            items.append(ImageItem(path=path, w=w, h=h, meta=dict(video=path, frame=int(fr), time=float(t),
                                                                   source="video")))
            for mi, s_ in enumerate(specs):
                keys_per_model[mi].append(f"{vk}#{int(fr)}" if s_.key in done_keys else None)
    preds_raw, model_names = [], []
    for mi, vs in enumerate(vspecs):
        ks = keys_per_model[mi]
        got = cache.get_many(vs.key, [k for k in ks if k])
        preds_raw.append([got[k][0] if k and k in got else None for k in ks])
        model_names.append((cache.model(vs.key) or {}).get("names") or {})
    if ground == "labels":
        ground = "none"                       # sampled video frames have no labels
    if ground in ("auto", "reference"):
        ground = "reference" if len(specs) > 1 and reference is not None else "none"
    infos = {}
    for _, an in videos_out:                  # speed per model (from the video analysis)
        for k, ms in zip(an.get("specs") or [], an.get("speed_ms") or []):
            if ms and k not in infos:
                infos[k] = dict(speed_ms=ms)
    run = LabRun(items, {}, specs, preds_raw, model_names, infos, ground, reference if ground == "reference" else None,
                 op_conf, source="videos")
    run.video_step = step
    return run


_READERS = {}                   # thread id -> (video path, open source): one decoder per thread, ALWAYS released
_READERS_LOCK = threading.Lock()


def release_reader(tid=None):
    """Close the video decoder kept open by a thread (default: the calling one). Worker jobs call this when they
    end, viewers when they close - decoders (and their GPU / memory) never pile up."""
    tid = threading.get_ident() if tid is None else tid
    with _READERS_LOCK:
        cur = _READERS.pop(tid, None)
    if cur is not None:
        try:
            cur[1].release()
        except Exception:
            pass


def open_readers():
    with _READERS_LOCK:
        return len(_READERS)


def _video_frame(path, frame):
    """One frame of a video, keeping the calling thread's decoder open between calls (galleries read many frames
    of one video). CPU decoding only: the Model Lab never competes with the player for the GPU decoder."""
    from AIO_QC.tools.frame_extractor.core.sources import open_source
    tid = threading.get_ident()
    with _READERS_LOCK:
        cur = _READERS.get(tid)
    if cur is None or cur[0] != path:
        release_reader(tid)
        src = open_source(path, "auto", hw_accel=False)
        with _READERS_LOCK:
            _READERS[tid] = cur = (path, src)
    src = cur[1]
    try:
        src.seek(int(frame))
        ok, f = src.read()
        return f if ok else None
    except Exception:
        release_reader(tid)
        return None


def frame_getter(x):
    """Image item (dataset.ImageItem; a sampled video frame when meta['source'] == 'video') or {video, frame}
    -> BGR image (or None)."""
    if isinstance(x, dict):
        return _video_frame(x["video"], x["frame"])
    meta = getattr(x, "meta", None) or {}
    if meta.get("source") == "video":
        return _video_frame(meta["video"], meta["frame"])
    return imread(getattr(x, "path", x))


def make_report(path, kind="summary", run=None, videos=None, title=None, slots=None, pair=None, exports=True,
                version=""):
    """Write the HTML report (+ metrics.json / per_class.csv / per_image.csv next to it). Returns written paths."""
    if run is None and not videos:
        raise ValueError("Nothing to report: evaluate a dataset or analyse a video first")
    if kind == "video" and not videos:
        kind = "summary"
    if kind in ("summary", "detailed", "comparison") and run is None:
        kind = "video"
    title = title or ("Model comparison" if kind == "comparison" else "Model evaluation")
    rb = ReportBuilder(run, title, slots=slots, frame_getter=frame_getter, video=videos or [], version=version)
    paths = [write_report(path, rb.build(kind, pair))]
    if exports and run is not None and run.scored:
        paths += write_exports(run, os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]
                                                 + "_data"))
    return paths


def prelabel(testset, spec, cache, conf=0.25, fmt="yolo", only_unlabelled=True, progress=None, cancel=None,
             log=None):
    """Pre-annotate a test set with one model (for review in any labelling tool). Returns files written."""
    data_keys = [image_key(p, testset.root) for p in testset.image_paths()]
    infos = run_on_images([spec], list(zip(data_keys, testset.image_paths())), cache, progress=progress,
                          cancel=cancel, log=log)
    info = infos[spec.key]
    if info.get("error") and not cache.keys_for(spec.key):
        raise RuntimeError(info["error"])
    preds, names = load_predictions(cache, spec, data_keys)
    from AIO_QC.core.testing import metrics as M
    target = dict(testset.names) or dict(names)
    mp, _, _ = M.build_class_map(names, target)
    by_file = {}
    for it, p in zip(testset.items, preds):
        if p is None:
            continue
        p = M.remap(p, mp)
        by_file[it["file"]] = p[p[:, 4] >= conf]
    return testset.write_prelabels(by_file, target, fmt, only_unlabelled)
