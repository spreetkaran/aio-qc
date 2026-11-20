# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
aio-qc autoannotate — the Auto-Annotate steps without the window (a GPU server, a scheduled job).

  aio-qc autoannotate RUN --videos D:/batch_12 --setup project.aasetup.json
  aio-qc autoannotate RUN --videos D:/batch_12 --model best.pt --every 3 --keep-classes trailer --split 200
  aio-qc autoannotate RUN --images D:/crops --ai-config damage.yaml --only frames,annotate   # AI annotate
  aio-qc autoannotate RUN --only frames --videos D:/batch_12          # just the unique frames
  aio-qc autoannotate RUN --only annotate,rules --conf 0.4            # again on the frames already there
  aio-qc autoannotate RUN --status                                   # what the run folder holds

RUN is the run folder (created when needed). A setup saved from the window holds every choice; the options below
override it. Stopping (Ctrl+C) is safe: run the same command again to continue.

Exit codes: 0 done · 1 error · 2 the frame settings changed (add --fresh to replace the previous frames) · 130 stopped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from AIO_QC.tools.frame_extractor.core.autoannotate import engine as E
from AIO_QC.tools.frame_extractor.core.autoannotate.recipe import Recipe, STAGES, STAGE_TITLES
from AIO_QC.core.diag import ignored as _ignored


def add_parser(sub):
    p = sub.add_parser("autoannotate", help="videos -> unique frames -> LabelMe files -> parts to upload",
                       formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__.split("\n\n", 1)[1])
    p.add_argument("run", help="the run folder")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--videos", help="folder of videos (or one video)")
    src.add_argument("--images", help="folder of images instead of videos")
    p.add_argument("--setup", help="a setup saved from the window (.aasetup.json)")
    p.add_argument("--save-setup", help="write the effective setup to this file and continue")
    p.add_argument("--only", help=f"steps to run, comma-separated: {','.join(STAGES)} (default: the setup's)")
    p.add_argument("--fresh", action="store_true", help="the frame settings changed: replace the previous frames")
    p.add_argument("--status", action="store_true", help="show what the run folder holds and exit")
    p.add_argument("--env", default=None, help="Python of the environment that runs .pt models (default: the app's)")
    g = p.add_argument_group("frames")
    g.add_argument("--every", type=int, help="every Nth frame (or the sharpest of every N with --mode sharpest)")
    g.add_argument("--mode", choices=["step", "sharpest"])
    g.add_argument("--include", help="only videos / images whose names match, e.g. '*cam1*;*cam2*'")
    g.add_argument("--exclude", help="leave out names that match")
    g.add_argument("--repeats", choices=["off", "exact", "near"], help="skip repeated frames within a video")
    g.add_argument("--across-videos", action="store_true", default=None, help="also skip frames seen in another "
                                                                               "video or an earlier run")
    g.add_argument("--min-sharpness", type=float)
    g.add_argument("--workers", type=int)
    g.add_argument("--similar", choices=["strict", "balanced", "loose"], help="how alike 'similar' frames are")
    g = p.add_argument_group("annotate")
    g.add_argument("--ai-config", help="annotate with AI annotate: the project's config.yaml (made in the window)")
    g.add_argument("--model", help=".onnx or .pt (your trained model)")
    g.add_argument("--format", choices=["labelme", "yolo", "both"], help="files written by the model annotator")
    g.add_argument("--conf", type=float)
    g.add_argument("--iou", type=float)
    g.add_argument("--imgsz", type=int, help="0 = the model's own size (default)")
    g.add_argument("--device", help="auto, cpu, 0, 1 …")
    g.add_argument("--batch", type=int)
    g.add_argument("--labelme", choices=["6.3.1", "5.11.4"], help="LabelMe version written")
    g.add_argument("--image-data", action="store_true", default=None, help="embed the image in each JSON")
    g.add_argument("--yolo", action="store_true", default=None, help="also write YOLO .txt + classes.txt")
    g.add_argument("--rename", action="append", default=[], metavar="ID=NAME",
                   help="write a model class under another name (repeatable); NAME '-' leaves it out")
    g.add_argument("--existing", choices=["update", "keep", "replace"])
    g = p.add_argument_group("keep rule")
    g.add_argument("--keep-when", choices=["always", "detections", "any_of", "all_of"])
    g.add_argument("--keep-classes", help="comma-separated labels for any_of / all_of")
    g.add_argument("--others", choices=["move", "keep"], help="frames that fail: move to extra_images/, or keep")
    g = p.add_argument_group("package")
    g.add_argument("--group-by", choices=["none", "video", "pattern"])
    g.add_argument("--pattern", help="regular expression on the file name that names the folder")
    g.add_argument("--split", type=int, help="files per part (0 = no parts)")
    g.add_argument("--part-names", choices=["number", "start"])
    g.add_argument("--no-zip", action="store_true")
    g.add_argument("--yolo-layout", choices=["beside", "split"], help="YOLO files next to the images, or images/ + "
                                                                      "labels/ in each part")
    g.add_argument("--copy", action="store_true", help="copy the files into the parts (default: hard links, which "
                                                       "take no extra disk space)")
    return p


def _recipe(a) -> Recipe:
    if a.setup:
        r = Recipe.load(a.setup)
    else:                                     # continue with the run folder's last setup
        r = Recipe.from_dict(E.Workspace(a.run).state().get("recipe") or {})
    if a.images:
        r.source = "images"
    elif a.videos:
        r.source = "videos"
    for opt, attr in (("every", "step"), ("mode", "mode"), ("include", "include"), ("exclude", "exclude"),
                      ("repeats", "skip_repeats"), ("across_videos", "across_videos"),
                      ("min_sharpness", "min_sharpness"), ("workers", "workers"), ("similar", "similar"),
                      ("model", "model"), ("conf", "conf"), ("iou", "iou"), ("imgsz", "imgsz"), ("device", "device"),
                      ("batch", "batch"), ("labelme", "labelme_version"), ("image_data", "image_data"),
                      ("yolo", "also_yolo"), ("existing", "existing"), ("keep_when", "keep_when"),
                      ("others", "others"), ("group_by", "group_by"), ("pattern", "group_pattern"),
                      ("split", "split_size"), ("part_names", "part_names"), ("format", "output_format"),
                      ("yolo_layout", "yolo_layout")):
        v = getattr(a, opt)
        if v is not None:
            setattr(r, attr, v)
    if a.model:
        r.model = os.path.abspath(a.model)
        r.annotator = "model"
    if a.ai_config:
        r.ai_config = os.path.abspath(a.ai_config)
        r.annotator = "ai"
    if a.pattern and a.group_by is None:
        r.group_by = "pattern"
    if a.keep_classes is not None:
        r.keep_classes = [c.strip() for c in a.keep_classes.split(",") if c.strip()]
        if a.keep_when is None and r.keep_when in ("always", "detections"):
            r.keep_when = "any_of"
    if a.no_zip:
        r.zip_parts = False
    if a.copy:
        r.package_files = "copy"
    for item in a.rename:
        cid, _, name = item.partition("=")
        e = dict(r.classes.get(cid.strip(), {}))
        if name.strip() == "-":
            e["keep"] = False
        else:
            e.update(keep=True, name=name.strip())
        r.classes[cid.strip()] = e
    if a.only:
        bad = [s for s in a.only.split(",") if s.strip() and s.strip() not in STAGES]
        if bad:
            raise SystemExit(f"unknown step(s): {', '.join(bad)} (use {', '.join(STAGES)})")
        r.stages = [s for s in STAGES if s in {x.strip() for x in a.only.split(",")}]
    return r


def _status(root):
    ws = E.Workspace(root)
    c = ws.counts()
    print(f"{ws.root}\n  frames {c['frames']:,} · annotated {c['annotated']:,} · similar {c['similar']:,} · "
          f"extra_images {c['extra']:,} · zip files {c['parts']:,}")
    for s in STAGES:
        rec = ws.stage_record(s)
        if rec:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(rec.get("at", 0)))
            print(f"  {STAGE_TITLES[s]:<15} {rec.get('status', ''):<8} {when}"
                  f"{'  (older than an earlier step: run it again)' if ws.stale(s) else ''}"
                  f"{'  ' + rec['error'] if rec.get('error') else ''}")
    ws.close()
    return 0


def cmd_autoannotate(a) -> int:
    if a.status:
        return _status(a.run)
    r = _recipe(a)
    if a.save_setup:
        r.save(a.save_setup)
    source = a.videos or a.images or (E.Workspace(a.run).state().get("source") or "")
    env = a.env
    if env is None:
        try:
            from AIO_QC.utils.qt_utils import app_settings
            env = app_settings().value("inference/python", "") or ""
        except Exception as _exc:
            _ignored(_exc)
            env = ""
    last = {"t": 0.0, "stage": ""}
    tty = sys.stderr.isatty()

    def progress(stage, done, total, text):
        now = time.monotonic()
        if stage != last["stage"] or now - last["t"] > (1.0 if tty else 10.0):
            last.update(t=now, stage=stage)
            pct = f"{100 * done / total:5.1f}%" if total else "  …  "
            line = f"[{STAGE_TITLES.get(stage, stage)}] {pct} {text}"
            print(f"\r{line}".ljust(100) if tty else line, end="" if tty else "\n", file=sys.stderr, flush=True)

    def log(stage, text):
        line = f"[{STAGE_TITLES.get(stage, stage)}] {text}"
        print(f"\r{line}".ljust(100) if tty else line, file=sys.stderr, flush=True)

    ctx = E.Context(progress, log)
    try:
        res = E.run(a.run, r, source=source, ctx=ctx, env_python=env, replace_previous=a.fresh)
    except KeyboardInterrupt:
        ctx.cancel.set()
        print("\nstopped — run the same command again to continue", file=sys.stderr)
        return 130
    except E.Stopped as exc:
        print(f"\n{exc or 'stopped'} — run the same command again to continue", file=sys.stderr)
        return 130
    except E.NeedsConfirmation as exc:
        print(f"\n{exc}\nAdd --fresh to replace them.", file=sys.stderr)
        return 2
    except E.StepError as exc:
        msg, title = str(exc), STAGE_TITLES.get(exc.stage, exc.stage)
        print(f"\n{msg if msg.startswith(title) or ':' in msg[:24] else f'{title}: {msg}'}", file=sys.stderr)
        return 1
    print(file=sys.stderr)
    for s, st in res.items():
        flat = ", ".join(f"{k.replace('_', ' ')} {v:,}" for k, v in st.items()
                         if isinstance(v, int) and not isinstance(v, bool))
        print(f"{STAGE_TITLES[s]}: {flat}")
    print(f"summary: {os.path.join(os.path.abspath(a.run), 'reports', 'summary.html')}")
    return 0
