# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Command line interface (no Qt needed).

  aio-qc extract INPUT [INPUT ...] -o OUT [options]
  aio-qc probe VIDEO [VIDEO ...]
  aio-qc testset OUT --add VIDEO|FOLDER ... [--every 1s] [--unique balanced]
  aio-qc eval DATA --model A.pt --model B.onnx [--report summary|detailed|comparison|video]
  aio-qc clip VIDEO START END -o OUT.mp4 [--exact]

INPUT may be video files or folders (use -r to recurse). Settings can come
from a JSON preset (saved from the GUI's batch dialog or with --save-preset);
explicit flags override the preset.

Examples
  # 1 sharp frame per second, skip near-duplicates, 4 workers (long videos are split)
  aio-qc extract D:/videos -o D:/frames --mode sharpest --every 1s --dedup 2 -w 4

  # mine frames where two YOLO models disagree or are unsure, write LabelMe pre-annotations
  aio-qc extract cam01.mp4 -o out --every 5 --model plate.onnx --model vehicle.onnx \
      --mine uncertain,disagree --preannot labelme
"""
import argparse
import os
import sys
import time

from AIO_QC.tools.frame_extractor.core.extract import ExtractSettings, Job
from AIO_QC.tools.frame_extractor.core.batch import run_batch
from AIO_QC.tools.frame_extractor.core.sources import open_source, format_time, HAVE_PYAV

from AIO_QC.tools.frame_extractor.core.common import VIDEO_EXTS


def _collect(inputs, recursive):
    out = []
    for p in inputs:
        if os.path.isdir(p):
            walker = os.walk(p) if recursive else [(p, [], os.listdir(p))]
            for root, _, files in walker:
                out += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(VIDEO_EXTS)]
        elif os.path.isfile(p):
            out.append(p)
        else:
            print(f"warning: not found: {p}", file=sys.stderr)
    return out


def _roi(text):
    v = [int(float(x)) for x in text.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("ROI must be x1,y1,x2,y2")
    return tuple(v)


def build_parser():
    p = argparse.ArgumentParser(prog="aio-qc",
                                description="AIO QC — Frame Extractor command line (run aio-qc without a command for the app).")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extract frames from videos",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Examples", 1)[1])
    e.add_argument("inputs", nargs="+", help="video files and/or folders")
    e.add_argument("-o", "--output", required=True, help="output root (frames go to OUT/<video>/batch_extract)")
    e.add_argument("-r", "--recursive", action="store_true", help="search folders recursively")
    e.add_argument("--preset", help="JSON preset with extraction settings")
    e.add_argument("--save-preset", help="write the effective settings to this JSON and continue")
    g = e.add_argument_group("sampling")
    g.add_argument("--mode", choices=["step", "sharpest"])
    g.add_argument("-n", "--every", help="N frames, or a duration like 0.5s / 2s")
    g.add_argument("--start", default="", help="start time (HH:MM:SS.mmm or seconds)")
    g.add_argument("--end", default="", help="end time")
    g.add_argument("--roi", type=_roi, help="crop x1,y1,x2,y2 (full-resolution pixels)")
    g.add_argument("--frames", help="save exactly these frames: '10,250,1200' or @file (one per line, "
                                    "optionally 'frame x1,y1,x2,y2'); ignores sampling/filters")
    g = e.add_argument_group("filters")
    g.add_argument("--min-sharpness", type=float)
    g.add_argument("--keep-blurry", type=int, metavar="K", help="keep 1 in K blurry rejects as negatives")
    g.add_argument("--dedup", nargs="?", const="balanced", metavar="PRESET",
                   help="skip duplicate frames: exact | conservative | balanced (default) | aggressive, "
                        "or a number = whole-picture %% change (old behaviour, local limit 8%%)")
    g.add_argument("--dedup-local", type=float, metavar="PCT", help="largest change allowed in any 16x9 cell (%%)")
    g.add_argument("--dedup-window", type=int, metavar="N", help="compare with the last N saved frames (5)")
    g.add_argument("--scenes", type=float, metavar="THR", help="scene-cut detection threshold (e.g. 0.4)")
    g.add_argument("--cross-dedup", type=int, metavar="BITS",
                   help="skip frames within BITS (0-64, ~6) pHash distance of anything already in the dataset")
    g.add_argument("--hash-db", help="hash database (default OUT/.frameextractor_hashes.db)")
    g.add_argument("--track-roi", nargs="?", const="auto", choices=["auto", "csrt", "kcf", "mil"],
                   help="let the --roi follow the object through the video")
    g = e.add_argument_group("models / mining")
    g.add_argument("--model", action="append", help=".onnx or .pt YOLO model (repeat for several)")
    g.add_argument("--conf", type=float)
    g.add_argument("--iou", type=float)
    g.add_argument("--imgsz", type=int)
    g.add_argument("--device", help="auto | cpu | 0 (GPU index, .pt models) | mps")
    g.add_argument("--env", help="Python environment (folder or python exe) with ultralytics for .pt models")
    g.add_argument("--half", action="store_true", help="FP16 on GPU (.pt models)")
    g.add_argument("--sahi", nargs="?", const=640, type=int, metavar="SLICE",
                   help="sliced inference for small objects (.pt models, needs sahi in the env)")
    g.add_argument("--mine", help="comma list of: uncertain,empty,disagree,classes,temporal")
    g.add_argument("--temporal-stride", type=int, help="run the model every Nth frame for temporal mining (1)")
    g.add_argument("--temporal-gap", type=int, help="max frames an object may vanish/appear to be flagged (2)")
    g.add_argument("--uncertain", help="uncertainty band LOW,HIGH (default 0.25,0.6)")
    g.add_argument("--classes", help="comma list of class names/ids for --mine classes")
    g.add_argument("--preannot", choices=["none", "labelme", "yolo"])
    g = e.add_argument_group("output / performance")
    g.add_argument("--format", choices=["jpg", "png", "webp", "avif", "jp2", "tif", "bmp"])
    g.add_argument("--quality", type=int, metavar="1-100", help="image quality for lossy formats (default 95)")
    g.add_argument("--name", metavar="TEMPLATE",
                   help="file-name template, e.g. '{video}_{frame:06d}', '{date}/{video}_{time}' "
                        "(tokens: video frame time tc sec ms counter kind reason roi w h fps date datetime vdate)")
    g.add_argument("--on-exists", choices=["overwrite", "suffix", "skip"])
    g.add_argument("--backend", choices=["auto", "opencv", "pyav"])
    g.add_argument("--no-hw", action="store_true", help="disable hardware decoding")
    g.add_argument("-w", "--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    g.add_argument("--no-split", action="store_true", help="never split one video across workers")
    g.add_argument("--no-resume", action="store_true",
                   help="ignore checkpoints and redo everything (default: continue interrupted/finished jobs)")
    g.add_argument("--min-chunk", type=float, default=60.0, help="min seconds per split chunk")

    pr = sub.add_parser("probe", help="show decoder / timing info for videos")
    pr.add_argument("videos", nargs="+")
    pr.add_argument("--backend", choices=["auto", "opencv", "pyav"], default="auto")

    ts = sub.add_parser("testset", help="build / extend a test set: unique frames from videos and image folders",
                        description="Frames from many videos (and image folders), duplicates removed inside each "
                                    "video and across all of them. The folder opens in any YOLO tool (data.yaml).")
    ts.add_argument("out", help="test set folder (created, or extended when it exists)")
    ts.add_argument("--add", nargs="+", required=True, metavar="SRC", help="videos and/or image folders")
    ts.add_argument("--every", default="1s", help="sampling: '1s' every N seconds, '30f' every N frames, "
                                                  "'100n' N evenly spaced per video, 'all' (default 1s)")
    ts.add_argument("--max-per-video", type=int, default=0)
    ts.add_argument("--unique", default="balanced", choices=["off", "exact", "conservative", "balanced", "aggressive"])
    ts.add_argument("--min-sharpness", type=float, default=0.0)
    ts.add_argument("--copy-folders", action="store_true", help="copy folder images into the test set")
    ts.add_argument("--prelabel", metavar="MODEL", help="pre-annotate new images with this model (marked unreviewed)")
    ts.add_argument("--env", default="", help="Python / conda env for .pt models")

    ev = sub.add_parser("eval", help="test YOLO weights: accuracy, speed, confusion matrix, HTML report",
                        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
DATA: a test set folder, a YOLO data.yaml, or a folder of images with YOLO txt / LabelMe json labels.
      Without labels, models are compared with each other (--reference) and analysed for stability.
Examples
  aio-qc eval testsets/gate -m old.pt -m new.pt --env yolo --report comparison
  aio-qc eval data.yaml -m best.onnx --report detailed -o report.html
  aio-qc eval --video cam1.mp4 --video cam2.mp4 -m a.pt -m b.pt --report video""")
    ev.add_argument("data", nargs="?", help="test set / data.yaml / image folder")
    ev.add_argument("-m", "--model", action="append", required=True, help="weights (.pt .onnx .engine); repeat")
    ev.add_argument("--video", action="append", default=[], help="also analyse this video (label-free); repeat")
    ev.add_argument("--env", default="", help="Python / conda env for .pt / .engine models")
    ev.add_argument("--imgsz", type=int, default=0)
    ev.add_argument("--device", default="auto")
    ev.add_argument("--half", action="store_true")
    ev.add_argument("--batch", type=int, default=8)
    ev.add_argument("--conf", type=float, default=None, help="operating threshold (default: best F1 per model)")
    ev.add_argument("--reference", type=int, default=None, help="index of the model used as reference ground truth")
    ev.add_argument("--include-unreviewed", action="store_true", help="also score unreviewed pre-labels")
    ev.add_argument("--every", type=float, default=0.2, help="video sampling step in seconds (default 0.2)")
    ev.add_argument("--report", default="summary", choices=["summary", "detailed", "comparison", "video"])
    ev.add_argument("-o", "--output", help="report path (default: <test set>/reports/<type>_<time>.html)")
    ev.add_argument("--title", default=None)
    ev.add_argument("--no-exports", action="store_true", help="skip metrics.json / CSV files")

    cl = sub.add_parser("clip", help="cut a clip: fast (no re-encode, starts on a keyframe) or exact")
    cl.add_argument("video")
    cl.add_argument("start", help="seconds or h:mm:ss.ms")
    cl.add_argument("end", help="seconds or h:mm:ss.ms")
    cl.add_argument("-o", "--output", required=True)
    cl.add_argument("--exact", action="store_true", help="re-encode for frame-exact cuts")
    cl.add_argument("--no-audio", action="store_true")
    return p


def _secs(t):
    from AIO_QC.tools.frame_extractor.core.sources import parse_time
    return parse_time(str(t).rstrip("s"))


class _Bar:
    """One-line progress for the terminal."""
    def __init__(self):
        self.t = 0.0

    def __call__(self, done, total, text=""):
        now = time.monotonic()
        if now - self.t < 0.2 and done < total:
            return
        self.t = now
        pct = done / total * 100 if total else 0
        sys.stderr.write(f"\r  {pct:5.1f}%  {text[:70]:70s}")
        sys.stderr.flush()
        if total and done >= total:
            sys.stderr.write("\n")


def cmd_testset(a):
    from AIO_QC.core.testing.testset import TestSet
    every = a.every.strip().lower()
    st = dict(unique=a.unique, min_sharpness=a.min_sharpness, max_per_video=a.max_per_video, copy_folders=a.copy_folders)
    if every == "all":
        st["sampling"] = "all"
    elif every.endswith("f"):
        st.update(sampling="every_n", every_n=int(every[:-1]))
    elif every.endswith("n"):
        st.update(sampling="count", count=int(every[:-1]))
    else:
        st.update(sampling="interval", interval_s=float(every.rstrip("s")))
    ts = TestSet.open(a.out) if TestSet.is_testset(a.out) else TestSet.create(a.out, settings=st)
    sources = []
    for p in a.add:
        if os.path.isdir(p):
            sources.append(dict(type="folder", path=os.path.abspath(p)))
        elif os.path.isfile(p):
            sources.append(dict(type="video", path=os.path.abspath(p)))
        else:
            print(f"warning: not found: {p}", file=sys.stderr)
    bar = _Bar()
    try:
        stats = ts.add_sources(sources, st, progress=bar)
    except KeyboardInterrupt:
        print("\nInterrupted - frames added so far are kept.")
        return 130
    print(f"\n{ts.root}: +{stats['added']} images ({stats['duplicates']} duplicates, {stats['blurry']} blurry skipped) "
          f"in {stats['seconds']:.1f}s - {len(ts.items)} images in total")
    if a.prelabel:
        from AIO_QC.core.testing import pipeline as P
        spec = P.spec_from_cfg(a.prelabel, dict(env_python=a.env))
        n = P.prelabel(ts, spec, P.cache_for(P.Data(None, ts.root, ts)), progress=lambda mi, d, t, ms, txt: bar(d, t, txt))
        print(f"pre-annotations written for {n} images (unreviewed until edited)")
    return 0


def cmd_eval(a):
    from AIO_QC.tools.frame_extractor import __version__
    from AIO_QC.core.testing import pipeline as P
    if not a.data and not a.video:
        print("error: give a DATA folder / data.yaml and/or --video", file=sys.stderr)
        return 2
    cfg = dict(env_python=a.env, imgsz=a.imgsz, device=a.device, half=a.half)
    specs = P.unique_labels([P.spec_from_cfg(m, cfg) for m in a.model])
    for s in specs:
        if not os.path.isfile(s.path):
            print(f"error: weights not found: {s.path}", file=sys.stderr)
            return 2
    bar = _Bar()
    run = data = None
    cache = None
    t0 = time.monotonic()
    try:
        if a.data:
            data = P.open_data(a.data)
            cache = P.cache_for(data)
            print(f"{data.label}: {len(data.dataset.items)} images, {data.dataset.n_labelled} labelled, "
                  f"{len(data.dataset.names)} classes")
            run, infos = P.evaluate(data, specs, cache, only_reviewed=not a.include_unreviewed,
                                    reference=a.reference, op_conf=a.conf, batch=a.batch,
                                    progress=lambda mi, d, t, ms, txt: bar(d, t, txt), log=lambda m: print(m, file=sys.stderr))
            for s in specs:
                if infos[s.key].get("error"):
                    print(f"  {s.label}: {infos[s.key]['error']}", file=sys.stderr)
            rows = run.summary_table()
            if rows:
                print(f"\n{'#':>2} {'model':28s} {'mAP50-95':>9} {'mAP50':>7} {'P':>6} {'R':>6} {'F1':>6} {'conf':>5} {'ms/img':>7}")
                for r in rows:
                    print(f"{r['rank']:>2} {r['model'][:28]:28s} {r['map5095']:9.3f} {r['map50']:7.3f} {r['precision']:6.3f} "
                          f"{r['recall']:6.3f} {r['f1']:6.3f} {r['op_conf']:5.2f} "
                          + (f"{r['speed_ms']:7.1f}" if r['speed_ms'] else "      -"))
            for kind, text in run.verdict():
                print(f"  - {text}")
        vids = []
        if a.video:
            cache = cache or P.cache_for()
            vids = P.analyse_videos(a.video, specs, cache, every_s=a.every, op_conf=a.conf or 0.25,
                                    reference=a.reference, batch=a.batch,
                                    progress=lambda vi, d, t, txt: bar(d, t, txt), log=lambda m: print(m, file=sys.stderr))
            for path, v in vids:
                print(f"\n{os.path.basename(path)}: {len(v['times'])} samples")
                for m in v["models"]:
                    print(f"  {m['label'][:28]:28s} {m['detections']:7d} detections  stability {m['stability'] * 100:5.1f}%  "
                          f"flicker {m['flicker']}  blips {m['blip']}")
                for pr in v["pairs"]:
                    print(f"  agreement {pr['a']} vs {pr['b']}: {pr['mean'] * 100:.1f}%")
    except KeyboardInterrupt:
        print("\nInterrupted - finished predictions are cached; run again to continue.")
        return 130
    out = a.output
    name = f"{a.report}_{time.strftime('%Y%m%d_%H%M%S')}.html"
    if not out:
        base = data.testset.reports_dir if data is not None and data.testset is not None else os.getcwd()
        out = os.path.join(base, name)
    elif os.path.isdir(out) or out.endswith(("/", os.sep)) or not os.path.splitext(out)[1]:
        out = os.path.join(out, name)          # -o a folder (existing, or given without a file extension)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    paths = P.make_report(out, a.report, run, vids, a.title, exports=not a.no_exports, version=__version__)
    print(f"\nReport: {paths[0]}" + (f"  (+ {len(paths) - 1} data files)" if len(paths) > 1 else "")
          + f"  · {time.monotonic() - t0:.1f}s")
    return 0


def cmd_clip(a):
    from AIO_QC.core.testing.annotate import export_clip
    from AIO_QC.tools.frame_extractor.core.preview_pipe import find_ffmpeg
    ff = find_ffmpeg()
    if not ff:
        print("error: ffmpeg not found", file=sys.stderr)
        return 2
    bar = _Bar()
    r = export_clip(a.video, a.output, _secs(a.start), _secs(a.end), ff, mode="exact" if a.exact else "fast",
                    audio=not a.no_audio, progress=lambda d, t, txt="": bar(d, t, txt))
    note = "" if a.exact else f" (starts on the keyframe at {r['start']:.3f}s)"
    print(f"{r['path']}{note}")
    return 0


def _settings_from_args(a):
    s = ExtractSettings.load_preset(a.preset) if a.preset else ExtractSettings()
    if a.mode: s.mode = a.mode
    if a.min_sharpness is not None: s.min_sharpness = a.min_sharpness
    if a.keep_blurry is not None: s.keep_blurry_every = a.keep_blurry
    if a.dedup is not None:
        from AIO_QC.tools.frame_extractor.core.dedup import PRESETS as _DP
        s.dedup = True
        if a.dedup == "exact":
            s.dedup_mode = "exact"
        elif a.dedup in _DP:
            s.dedup_mode, (s.dedup_threshold, s.dedup_local) = "near", _DP[a.dedup]
        else:
            try:
                v = float(a.dedup)
            except ValueError:
                raise SystemExit(f"--dedup: unknown preset '{a.dedup}'")
            s.dedup, s.dedup_mode, s.dedup_threshold = v > 0, "near", v
    if a.dedup_local is not None: s.dedup_local = a.dedup_local
    if a.dedup_window is not None: s.dedup_window = a.dedup_window
    if a.scenes is not None: s.scene_detect, s.scene_threshold = a.scenes > 0, a.scenes or s.scene_threshold
    if a.model: s.models = [os.path.abspath(m) for m in a.model]
    if a.conf is not None: s.det_conf = a.conf
    if a.iou is not None: s.det_iou = a.iou
    if a.imgsz is not None: s.det_imgsz = a.imgsz
    if a.device: s.det_device = a.device
    if a.mine is not None: s.mine = [m.strip() for m in a.mine.split(",") if m.strip()]
    if a.uncertain:
        s.unc_low, s.unc_high = (float(x) for x in a.uncertain.split(","))
    if a.classes: s.mine_classes = [c.strip() for c in a.classes.split(",")]
    if a.preannot: s.preannot = a.preannot
    if a.format: s.ext = "." + a.format
    if a.quality is not None: s.quality = max(1, min(100, a.quality))
    if a.name:
        from AIO_QC.tools.frame_extractor.core.naming import validate
        errors, warnings = validate(a.name)
        if errors:
            raise SystemExit("--name: " + "; ".join(errors))
        for w in warnings:
            print("note: " + w, file=sys.stderr)
        s.name_template = a.name
    if a.on_exists: s.on_exists = a.on_exists
    if a.backend: s.backend = a.backend
    if a.no_hw: s.hw_accel = False
    if a.cross_dedup is not None: s.cross_dedup, s.cross_dedup_dist = a.cross_dedup > 0, a.cross_dedup
    if a.hash_db: s.hash_db = os.path.abspath(a.hash_db)
    if a.track_roi: s.roi_track, s.roi_tracker = True, a.track_roi
    if a.temporal_stride: s.temporal_stride = a.temporal_stride
    if a.temporal_gap: s.temporal_gap = a.temporal_gap
    if a.no_resume: s.resume = False
    if a.env:
        from AIO_QC.tools.frame_extractor.core.env_manager import python_in_env
        py = python_in_env(a.env)
        if py is None:
            raise SystemExit(f"--env: no python found in {a.env}")
        s.env_python = str(py)
    if a.half: s.det_half = True
    if a.sahi: s.det_sahi, s.sahi_slice = True, a.sahi
    return s


def _parse_frames(text):
    if not text:
        return None
    lines = open(text[1:], encoding="utf-8").read().splitlines() if text.startswith("@") else text.split(",")
    out = []
    for line in lines:
        parts = line.replace(";", " ").split()
        if not parts:
            continue
        roi = [int(float(v)) for v in parts[1].split(",")] if len(parts) > 1 else None
        out.append([int(parts[0]), roi])
    return out


def _every_frames(text, path):
    """'5' -> 5 frames; '0.5s' -> frames at this video's fps."""
    if text is None:
        return None
    if text.endswith("s"):
        src = open_source(path, "opencv", False)
        try:
            return max(1, int(round(float(text[:-1]) * src.fps)))
        finally:
            src.release()
    return max(1, int(text))


def cmd_probe(a):
    for v in a.videos:
        try:
            s = open_source(v, a.backend, True)
        except Exception as e:
            print(f"{v}: ERROR {e}")
            continue
        print(f"{v}\n  backend {s.backend}{' +GPU' if s.hw_accel else ''} | {s.width}x{s.height} | "
              f"{s.fps:.3f} fps | {s.frame_count} frames | {format_time(s.duration())} | "
              f"VFR: {s.is_vfr} | rotation: {s.rotation}")
        s.release()
    if not HAVE_PYAV:
        print("(PyAV not installed: VFR/rotation detection limited - pip install av)")
    return 0


def cmd_extract(a):
    videos = _collect(a.inputs, a.recursive)
    if not videos:
        print("No videos found.", file=sys.stderr)
        return 2
    base = _settings_from_args(a)
    if base.cross_dedup and not base.hash_db:
        from AIO_QC.tools.frame_extractor.core.hashdb import default_db_path
        base.hash_db = default_db_path(os.path.abspath(a.output))
    if base.roi_track and not a.roi:
        print("--track-roi needs --roi", file=sys.stderr)
        return 2
    if a.save_preset:
        base.save_preset(a.save_preset)
        print(f"Preset saved to {a.save_preset}")
    # --every may be in seconds -> per-video step; group videos by resulting settings
    jobs, settings_for = [], []
    for v in videos:
        s = ExtractSettings.from_dict(base.to_dict())
        n = _every_frames(a.every, v)
        if n:
            s.step = n
        name = os.path.splitext(os.path.basename(v))[0]
        frames = _parse_frames(a.frames)
        sub = "markers" if frames else "batch_extract"
        jobs.append(Job(os.path.abspath(v), os.path.join(os.path.abspath(a.output), name, sub),
                        a.start, a.end, a.roi, frames=frames))
        settings_for.append(s)

    print(f"{len(jobs)} video(s), {a.workers} worker(s), mode={base.mode}"
          + (f", models={len(base.models)} mine={base.mine or '-'} preannot={base.preannot}" if base.models else ""))
    t0 = time.monotonic()
    state = {}
    failures = 0
    finished_jobs = [0]

    def on_event(kind, ji, payload):
        nonlocal failures
        name = os.path.basename(jobs_run[ji].video_path)
        if kind == "progress":
            state[ji] = payload
            done = sum(p[0] for p in state.values())
            total = sum(p[1] for p in state.values()) or 1
            saved = sum(p[2] for p in state.values())
            el = time.monotonic() - t0
            fps = done / el if el else 0
            eta = format_time((total - done) / fps)[:8] if fps else "--:--:--"
            sys.stdout.write(f"\r  [{finished_jobs[0]}/{len(jobs_run)} videos]  {100 * done / total:5.1f}% of started"
                             f"  {fps:6.0f} fps  saved {saved:6d}  ETA {eta} ")
            sys.stdout.flush()
        elif kind == "done":
            finished_jobs[0] += 1
            keys = ("saved", "duplicates", "cross_duplicates", "blurry", "negatives", "mined_out", "scenes")
            extra = ", ".join(f"{k} {v}" for k, v in payload.items() if k.startswith("mined_") and k != "mined_out")
            note = " (already complete)" if payload.get("resumed") else (
                f" (resumed at frame {payload['resumed_from']})" if payload.get("resumed_from") else "")
            print(f"\r  done  {name}{note}: " + ", ".join(f"{k} {payload[k]}" for k in keys if payload.get(k))
                  + (f", {extra}" if extra else "") + " " * 20)
        else:
            finished_jobs[0] += 1
            failures += 1
            print(f"\r  ERROR {name}: {payload}" + " " * 20)

    # run each distinct settings group (usually just one)
    groups = {}
    for j, s in zip(jobs, settings_for):
        groups.setdefault(repr(s.to_dict()), (s, []))[1].append(j)
    try:
        for s, jobs_run in groups.values():
            state.clear()
            finished_jobs[0] = 0
            run_batch(jobs_run, s, a.workers, not a.no_split, a.min_chunk, on_event)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    print(f"Finished in {format_time(time.monotonic() - t0)[:8]}"
          + (f" with {failures} error(s)" if failures else ""))
    return 1 if failures else 0


def main(argv=None):
    from AIO_QC.core.fx_storage import migrate_files
    migrate_files()                    # once: the standalone app's data -> AIO QC's data folder
    a = build_parser().parse_args(argv)
    return dict(probe=cmd_probe, extract=cmd_extract, testset=cmd_testset, eval=cmd_eval, clip=cmd_clip)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
