# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Per-video session (bookmarks, in/out points, ROI) stored in the user's home,
keyed by the video's absolute path, so it survives output-dir changes."""
import hashlib
import json
import os

from AIO_QC.core.fx_storage import fx_home

SESSION_DIR = str(fx_home() / "sessions")          # AIO QC data folder (moved from ~/.frame_extractor)


def _path(video_path):
    key = hashlib.sha1(os.path.abspath(video_path).lower().encode("utf-8")).hexdigest()[:16]
    return os.path.join(SESSION_DIR, f"{key}.json")


def load(video_path):
    try:
        with open(_path(video_path), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    return {
        "bookmarks": {int(k): v for k, v in data.get("bookmarks", {}).items()},
        "in": data.get("in"),
        "out": data.get("out"),
        "roi": tuple(data["roi"]) if data.get("roi") else None,
        "markers": {int(k): v for k, v in data.get("markers", {}).items()},
        "marker_opts": data.get("marker_opts", {}),
    }


def save(video_path, bookmarks, in_frame, out_frame, roi, markers=None, marker_opts=None):
    os.makedirs(SESSION_DIR, exist_ok=True)
    data = {
        "video": os.path.abspath(video_path),
        "bookmarks": {str(k): v for k, v in sorted(bookmarks.items())},
        "in": in_frame,
        "out": out_frame,
        "roi": list(roi) if roi else None,
        "markers": {str(k): v for k, v in sorted((markers or {}).items())},
        "marker_opts": marker_opts or {},
    }
    tmp = _path(video_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, _path(video_path))
