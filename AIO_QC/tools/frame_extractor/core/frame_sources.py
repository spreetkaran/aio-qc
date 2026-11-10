# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Where an extracted frame came from (Qt-free) - so Image QC can offer "Show source video in Frame Extractor".

Every frame the Frame Extractor writes (Save frame, markers, batch / CLI extraction) appends one JSON line
{"file": <path relative to the folder>, "video": <absolute video path>, "frame": n, "time": seconds} to
<folder>/.fx_sources-<pid>.jsonl (one file per process: parallel batch workers never interleave writes). The
files are hidden, tiny (~150 bytes a frame) and travel with the folder.

    record(image_path, video, frame, t)   after a frame was written (never raises)
    lookup(image_path, recent_videos=())  -> {"video", "frame", "time"} or None: the image's folder and up to three
                                             parents are searched (templates like {video}/{frame} write into
                                             sub-folders); Testing test sets (testset.json) are understood too; else
                                             the default name "<video>_frame_<n>" is matched against known videos.
"""
import glob
import json
import os
import re
import threading

PREFIX = ".fx_sources-"
_lock = threading.Lock()
_cache = {}                         # folder -> (mtimes signature, {relative file: record})


def record(image_path, video, frame, t=None):
    try:
        folder = os.path.dirname(os.path.abspath(image_path))
        row = {"file": os.path.basename(image_path), "video": os.path.abspath(video), "frame": int(frame)}
        if t is not None:
            row["time"] = round(float(t), 3)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with _lock, open(os.path.join(folder, f"{PREFIX}{os.getpid()}.jsonl"), "a", encoding="utf-8") as f:
            f.write(line)
    except (OSError, TypeError, ValueError):
        pass


def _index(folder):
    files = glob.glob(os.path.join(folder, PREFIX + "*.jsonl"))
    if not files:
        return {}
    sig = tuple(sorted((f, os.path.getmtime(f), os.path.getsize(f)) for f in files))
    hit = _cache.get(folder)
    if hit is not None and hit[0] == sig:
        return hit[1]
    idx = {}
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        idx[os.path.normcase(r["file"])] = r
                    except (ValueError, KeyError):
                        continue
        except OSError:
            continue
    if len(_cache) > 256:
        _cache.clear()
    _cache[folder] = (sig, idx)
    return idx


def _testset_lookup(image_path):
    folder = os.path.dirname(os.path.abspath(image_path))
    for d in (folder, os.path.dirname(folder)):
        p = os.path.join(d, "testset.json")
        if not os.path.isfile(p):
            continue
        try:
            m = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            return None
        name = os.path.basename(image_path)
        for r in m.get("items", []) or []:
            if os.path.basename(str(r.get("file", ""))) == name and r.get("video"):
                return {"video": r["video"], "frame": int(r.get("frame") or 0), "time": r.get("time")}
    return None


_NAME = re.compile(r"^(?P<video>.+?)_frame_(?P<frame>\d+)(?:_roi)?(?:_\d+)?$")


def lookup(image_path, recent_videos=()):
    image_path = os.path.abspath(image_path)
    folder = os.path.dirname(image_path)
    d = folder
    for _ in range(4):
        idx = _index(d)
        if idx:
            rel = os.path.normcase(os.path.relpath(image_path, d))
            r = idx.get(rel) or (idx.get(os.path.normcase(os.path.basename(image_path))) if d == folder else None)
            if r is not None and os.path.isfile(r.get("video", "")):
                return {"video": r["video"], "frame": int(r["frame"]), "time": r.get("time")}
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    r = _testset_lookup(image_path)
    if r is not None and os.path.isfile(r["video"]):
        return r
    m = _NAME.match(os.path.splitext(os.path.basename(image_path))[0])
    if m:
        stem = m.group("video")
        for v in recent_videos or ():
            if os.path.splitext(os.path.basename(v))[0] == stem and os.path.isfile(v):
                return {"video": os.path.abspath(v), "frame": int(m.group("frame")), "time": None}
    return None
