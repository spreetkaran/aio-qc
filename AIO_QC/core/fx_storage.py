# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Where the Frame Extractor and Testing keep their files inside AIO QC, and the one-time move from the standalone
FrameExtractor's ~/.frame_extractor (Qt-free; the settings part is utils/fx_settings.py).

    fx_home()        <AIO QC data>/frame_extractor     per-video sessions (markers, in/out, ROI), frame search index
    testing_home()   <AIO QC data>/testing             prediction cache, project autosave + recents, exclusions
    migrate_files()  ~/.frame_extractor/{sessions,search} -> fx_home(), ~/.frame_extractor/lab/* -> testing_home();
                     once (a marker in the data folder), file by file, never overwriting; logs stay where they were
                     (AIO QC has one log). A note in the old folder says where everything went.

The paths are pure (no side effects), so worker processes and the CLI can import them freely; migrate_files() is
called by the UI when the Frame Extractor or Testing opens for the first time, and by the CLI.
"""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

from AIO_QC.core.appdirs import data_home

MARKER = ".frame_extractor_migrated"
_lock = threading.Lock()
_done = [False]


def legacy_home() -> Path:
    return Path(os.path.expanduser("~")) / ".frame_extractor"


def fx_home() -> Path:
    return data_home() / "frame_extractor"


def testing_home() -> Path:
    return data_home() / "testing"


def _merge(src: Path, dst: Path, moved: list, kept: list):
    """Move src into dst; where dst already has the same name, keep dst's (never overwrite) and leave src."""
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append(str(dst))
        return
    if src.is_dir() and dst.is_dir():
        for c in list(src.iterdir()):
            _merge(c, dst / c.name, moved, kept)
        try:
            src.rmdir()
        except OSError:
            pass
    else:
        kept.append(str(src))


def migrate_files(old: Path | None = None) -> dict:
    """One-time move of the standalone FrameExtractor's data. Returns {'moved': [...], 'kept': [...]}."""
    with _lock:
        if _done[0]:
            return {"moved": [], "kept": [], "skipped": "already done"}
        old = old or legacy_home()
        marker = data_home() / MARKER
        if marker.is_file() or not old.is_dir():
            _done[0] = True
            return {"moved": [], "kept": [], "skipped": "nothing to do"}
        moved, kept = [], []
        plan = [(old / "sessions", fx_home() / "sessions"), (old / "search", fx_home() / "search")]
        lab = old / "lab"
        if lab.is_dir():
            plan += [(c, testing_home() / c.name) for c in lab.iterdir()]
        for src, dst in plan:
            if not src.exists():
                continue
            try:
                _merge(src, dst, moved, kept)
            except OSError as exc:                       # locked file, other drive, ...: leave it, report it
                kept.append(f"{src} ({exc})")
        try:
            data_home().mkdir(parents=True, exist_ok=True)
            marker.write_text("moved from " + str(old) + "\n", encoding="utf-8")
            (old / "MOVED_TO_AIO_QC.txt").write_text(
                "The Frame Extractor is now part of AIO QC.\n"
                f"Sessions and the search index moved to: {fx_home()}\n"
                f"Model Lab data (Testing) moved to:      {testing_home()}\n"
                "Logs stay here; AIO QC writes its own log.\n", encoding="utf-8")
        except OSError:
            pass
        _done[0] = True
        return {"moved": moved, "kept": kept}
