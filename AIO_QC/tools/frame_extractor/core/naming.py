# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
File naming templates for extracted frames (Qt-free).

A template is a relative path without extension. Tokens (optionally with a format spec,
e.g. {frame:06d}, {sec:.2f}); "/" creates sub-folders:

  {video}     video file name without extension          cam01_2024-05-02
  {frame}     frame index                                 2503   ({frame:06d} -> 002503)
  {time}      position HH-MM-SS-mmm                       00-01-23-450
  {tc}        SMPTE timecode HH-MM-SS-FF (drop-frame ;)   00-01-23-13
  {sec}       seconds (float)                             83.45  ({sec:.3f})
  {ms}        milliseconds (int)                          83450
  {counter}   running number per job                      17     ({counter:05d})
  {kind}      frame | roi | marker | batch | negative | mined
  {reason}    mining reason(s), e.g. uncertain+flicker    (empty when none)
  {roi}       ROI as x1-y1-x2-y2, or "full"
  {w} {h}     saved image size
  {fps}       frame rate (rounded)
  {date} {datetime}      extraction date 20250928 / 20250928-143005
  {vdate}     video file's modified date                  20250502

Characters that are illegal in file names are replaced, so any template is safe on Windows.
"""
import datetime as _dt
import os
import re
import string

from AIO_QC.tools.frame_extractor.core.timecode import frames_to_smpte

DEFAULT_TEMPLATE = "{video}_frame_{frame}"
PRESETS = {
    "Video + frame (default)": "{video}_frame_{frame}",
    "Video + padded frame": "{video}_{frame:06d}",
    "Video + time": "{video}_{time}",
    "Video + timecode": "{video}_{tc}",
    "Folder per video": "{video}/{frame:06d}",
    "Dated folders": "{date}/{video}_{frame:06d}",
    "Sequential (dataset style)": "{video}_{counter:05d}",
    "By kind": "{kind}/{video}_{frame:06d}",
    "Mining reason": "{reason}/{video}_{frame:06d}",
}
TOKENS = {
    "video": "video file name", "frame": "frame index", "time": "HH-MM-SS-mmm", "tc": "SMPTE timecode",
    "sec": "seconds (float)", "ms": "milliseconds", "counter": "running number", "kind": "frame/roi/marker/…",
    "reason": "mining reason", "roi": "ROI x1-y1-x2-y2", "w": "width", "h": "height", "fps": "frame rate",
    "date": "today YYYYMMDD", "datetime": "now YYYYMMDD-HHMMSS", "vdate": "video file date",
}
UNIQUE_TOKENS = ("frame", "time", "tc", "sec", "ms", "counter")
ON_EXISTS = {"overwrite": "Overwrite", "suffix": "Keep both (add _1, _2 …)", "skip": "Skip (keep the existing file)"}
_BAD = re.compile(r'[<>:"\\|?*\x00-\x1f]')


def fields(template):
    try:
        return [f for _, f, _, _ in string.Formatter().parse(template) if f is not None]
    except ValueError as e:
        raise ValueError(f"template syntax: {e}")


def validate(template):
    """(errors, warnings) lists."""
    errors, warnings = [], []
    t = (template or "").strip()
    if not t:
        return ["the template is empty"], []
    try:
        names = [f.split(":")[0].split("!")[0] for f in fields(t)]
    except ValueError as e:
        return [str(e)], []
    unknown = sorted({n for n in names if n not in TOKENS})
    if unknown:
        errors.append("unknown token(s): " + ", ".join("{" + n + "}" for n in unknown))
    if t.startswith(("/", "\\")) or ".." in t.replace("\\", "/").split("/"):
        errors.append("must be a relative path inside the output folder")
    if not errors and not any(n in UNIQUE_TOKENS for n in names):
        warnings.append("no {frame}/{time}/{counter}: names would collide — _{frame} will be appended")
    if not errors:
        try:
            render(t, NameContext(video="v", frame=1))
        except (ValueError, KeyError) as e:
            errors.append(f"format spec: {e}")
    return errors, warnings


class NameContext:
    def __init__(self, video="", frame=0, sec=None, fps=30.0, counter=0, kind="frame", reason="", roi=None,
                 w=0, h=0, video_path=None, now=None):
        self.video, self.frame, self.fps = video, int(frame), fps or 30.0
        self.sec = float(sec) if sec is not None else frame / self.fps
        self.counter, self.kind, self.reason, self.roi = counter, kind, reason or "", roi
        self.w, self.h, self.video_path = w, h, video_path
        self.now = now or _dt.datetime.now()

    def values(self):
        ms = int(round(self.sec * 1000))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, rem = divmod(rem, 1000)
        vdate = ""
        if self.video_path and os.path.exists(self.video_path):
            vdate = _dt.datetime.fromtimestamp(os.path.getmtime(self.video_path)).strftime("%Y%m%d")
        return dict(
            video=self.video, frame=self.frame, time=f"{h:02d}-{m:02d}-{s:02d}-{rem:03d}",
            tc=frames_to_smpte(self.frame, self.fps).replace(":", "-").replace(";", "-"),
            sec=round(self.sec, 6), ms=ms, counter=self.counter, kind=self.kind, reason=self.reason,
            roi="-".join(str(int(v)) for v in self.roi) if self.roi else "full", w=self.w, h=self.h,
            fps=int(round(self.fps)), date=self.now.strftime("%Y%m%d"),
            datetime=self.now.strftime("%Y%m%d-%H%M%S"), vdate=vdate)


def _clean(part):
    part = _BAD.sub("_", part).strip().strip(".")
    return re.sub(r"_+", "_", part) or "_"


def render(template, ctx):
    """Relative path without extension, e.g. 'cam01/002503'."""
    t = (template or DEFAULT_TEMPLATE).strip()
    names = [f.split(":")[0].split("!")[0] for f in fields(t)]
    if not any(n in UNIQUE_TOKENS for n in names):
        t += "_{frame}"
    text = t.format(**ctx.values())
    parts = [p for p in re.split(r"[\\/]+", text) if p not in ("", ".", "..")]
    return "/".join(_clean(p) for p in parts) or f"frame_{ctx.frame}"


def target_path(out_dir, rel_stem, ext, on_exists="overwrite"):
    """Absolute path for rel_stem + ext following the collision policy; None = skip."""
    path = os.path.join(out_dir, *rel_stem.split("/")) + ext
    if on_exists == "overwrite" or not os.path.exists(path):
        return path
    if on_exists == "skip":
        return None
    base = path[:-len(ext)]
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"


class Namer:
    """Per-job naming state (counter) + writing with the collision policy."""

    def __init__(self, template=DEFAULT_TEMPLATE, on_exists="overwrite", video_path="", fps=30.0,
                 time_of=None, counter_start=1):
        self.template = template or DEFAULT_TEMPLATE
        self.on_exists = on_exists
        self.video_path = video_path
        self.video = os.path.splitext(os.path.basename(video_path))[0]
        self.fps, self.time_of = fps, time_of
        self.counter = counter_start - 1
        self.now = _dt.datetime.now()

    def path(self, out_dir, frame, ext, img=None, kind="frame", reason="", roi=None):
        """Returns (absolute path or None when skipped, relative stem)."""
        self.counter += 1
        h, w = (img.shape[:2] if img is not None else (0, 0))
        ctx = NameContext(self.video, frame, self.time_of(frame) if self.time_of else None, self.fps,
                          self.counter, kind, reason, roi, w, h, self.video_path, self.now)
        rel = render(self.template, ctx)
        return target_path(out_dir, rel, ext, self.on_exists), rel
