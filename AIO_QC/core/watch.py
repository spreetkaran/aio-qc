# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Detect changes made to a dataset OUTSIDE AIO QC (another tool, a script, git, a file manager).

    snap = snapshot(root, exts)                 # {path: "mtime_ns|size"} — fast os.scandir walk, hidden dirs skipped
    tracker = ChangeTracker(snap)               # baseline
    changes = tracker.step(snapshot(...))       # only changes that were stable for two scans
    tracker.note_own(paths)                     # AIO QC's own writes / trash: never reported as outside changes

Read-only by design: the app reacts to changes by re-reading files, never by writing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

GONE = "gone"
META_NAMES = {"classes.txt"}
META_SUFFIXES = {".yaml", ".yml"}


def snapshot(root: str | Path, exts: set[str]) -> dict[str, str]:
    """Signature of every relevant file (plus class-list / YAML files, which change the dataset's meaning)."""
    out: dict[str, str] = {}
    stack = [str(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    name = e.name
                    if name.startswith("."):
                        continue
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                            continue
                        low = name.lower()
                        ext = os.path.splitext(low)[1]
                        if ext in exts or low in META_NAMES or ext in META_SUFFIXES:
                            st = e.stat()
                            out[e.path] = f"{st.st_mtime_ns}|{st.st_size}"
                    except OSError:
                        continue
        except OSError:
            continue
    return out


def file_sig(path: str | Path) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        return GONE


def is_meta(path: str) -> bool:
    p = Path(path)
    return p.name.lower() in META_NAMES or p.suffix.lower() in META_SUFFIXES


@dataclass
class Changes:
    added: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    modified: list = field(default_factory=list)

    def __bool__(self):
        return bool(self.added or self.removed or self.modified)

    @property
    def count(self) -> int:
        return len(self.added) + len(self.removed) + len(self.modified)


class ChangeTracker:
    def __init__(self, base: dict[str, str]):
        self.base = dict(base)
        self.pending: dict[str, str] = {}

    def note_own(self, paths):
        """AIO QC wrote / moved these files: take their current state as the new baseline."""
        for p in paths:
            if p is None:
                continue
            p = str(p)
            sig = file_sig(p)
            if sig == GONE:
                self.base.pop(p, None)
            else:
                self.base[p] = sig
            self.pending.pop(p, None)

    def step(self, snap: dict[str, str]) -> Changes:
        ch = Changes()
        differing = {p for p in snap if snap[p] != self.base.get(p)} | {p for p in self.base if p not in snap}
        for p in list(self.pending):                       # went back to the baseline: nothing happened
            if p not in differing:
                del self.pending[p]
        for p in differing:
            new = snap.get(p, GONE)
            if self.pending.get(p) != new:                 # first sighting (or still changing): wait one scan
                self.pending[p] = new
                continue
            del self.pending[p]
            old = self.base.get(p)
            if new == GONE:
                self.base.pop(p, None)
                ch.removed.append(p)
            else:
                self.base[p] = new
                (ch.modified if old is not None else ch.added).append(p)
        return ch
