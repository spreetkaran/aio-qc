# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Testing projects (.fxlab, formerly the Frame Extractor's Model Lab): one JSON file that remembers a whole evaluation set-up - test data, videos, models
(with their colours), environment, run options, report settings - plus a summary of the last result for the
start page. Predictions live in the cache, so reopening a project shows its results again in seconds.

Qt-free. The open session is also autosaved (~/.frame_extractor/lab/last.fxlab).
"""
import json
import os
import time

EXT = ".fxlab"
VERSION = 1


def lab_dir():
    from AIO_QC.core.fx_storage import testing_home
    d = str(testing_home())
    os.makedirs(d, exist_ok=True)
    return d


AUTOSAVE = os.path.join(lab_dir(), "last" + EXT)
_RECENT = os.path.join(lab_dir(), "recent.json")


def dataset_project(root):
    """The Testing project that belongs to a dataset open in Image QC: <dataset>/.aio_qc/testing.fxlab (part of
    the dataset's AIO QC session; any other .fxlab still opens with Project ▸ Open)."""
    from AIO_QC.core.state import state_dir
    from pathlib import Path
    return str(state_dir(Path(root)) / ("testing" + EXT))


def save(path, state):
    state = dict(state)
    state["version"] = VERSION
    state["saved"] = time.time()
    state.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)
    return path


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict) or "models" not in d:
        raise ValueError(f"Not a Testing project: {path}")
    return d


def recent(limit=12):
    """[{path, name, saved, summary}] newest first; missing files are dropped."""
    try:
        rows = json.load(open(_RECENT, encoding="utf-8"))
    except (OSError, ValueError):
        rows = []
    out = []
    for r in rows:
        p = r.get("path", "")
        if p and os.path.isfile(p) and os.path.abspath(p) != os.path.abspath(AUTOSAVE):
            out.append(r)
    return out[:limit]


def add_recent(path, state):
    path = os.path.abspath(path)
    rows = [r for r in recent(50) if os.path.abspath(r.get("path", "")) != path]
    rows.insert(0, dict(path=path, name=state.get("name") or os.path.splitext(os.path.basename(path))[0],
                        saved=time.time(), summary=state.get("summary") or {}))
    tmp = _RECENT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows[:30], f, indent=1)
    os.replace(tmp, _RECENT)


def remove_recent(path):
    path = os.path.abspath(path)
    rows = [r for r in recent(50) if os.path.abspath(r.get("path", "")) != path]
    with open(_RECENT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)


def summarize(run, videos=None):
    """Headline of a result for the start page."""
    s = dict(date=time.time())
    if run is not None and run.scored:
        rows = run.summary_table()
        if rows:
            s.update(best=rows[0]["model"], map5095=round(rows[0]["map5095"], 4), map50=round(rows[0]["map50"], 4),
                     models=len(run.specs), images=len(run.eval_idx),
                     ground="reference" if run.ground == "reference" else "labels")
    if videos:
        s["videos"] = len(videos)
    return s
