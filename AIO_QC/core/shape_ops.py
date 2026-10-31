# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Bulk shape operations that stay correct when files change between the scan and the action.

A scan records each shape's FINGERPRINT (label + type + exact points). At action time every file is loaded fresh and
each target is re-identified: same index and same fingerprint, or — if shapes before it were added / removed — the
same fingerprint elsewhere in the file. A target that no longer exists is skipped and reported, never guessed.

    fp = fingerprint(shape)
    found, missing = resolve(shapes, [(index, fp), …])
    backup = Backup(root, "dimarea")  ·  backup.save(path)  ·  backup.written(path)  ·  backup.restore()   (undo)
"""
from __future__ import annotations

import os
import time
from pathlib import Path


def fingerprint(shape: dict) -> str:
    label = str(shape.get("_orig_label", shape.get("label", "")))
    pts = shape.get("points") or []
    coords = ",".join(f"{float(p[0]):.4f}:{float(p[1]):.4f}" for p in pts)
    return f"{label}|{shape.get('shape_type') or 'polygon'}|{coords}"


def resolve(shapes: list[dict], targets: list[tuple[int, str]]) -> tuple[list[int], list[tuple[int, str]]]:
    """Indices (into `shapes`) of the targets that still exist, and the targets that do not."""
    fps = [fingerprint(s) for s in shapes]
    used: set[int] = set()
    found, missing = [], []
    for idx, fp in sorted(targets):                     # exact positions first, then relocated ones
        if 0 <= idx < len(fps) and idx not in used and fps[idx] == fp:
            used.add(idx)
            found.append(idx)
    for idx, fp in sorted(targets):
        if idx in found:
            continue
        j = next((j for j, f in enumerate(fps) if f == fp and j not in used), None)
        if j is None:
            missing.append((idx, fp))
        else:
            used.add(j)
            found.append(j)
    return sorted(found), missing


def file_sig(path) -> tuple | None:
    try:
        st = os.stat(path)
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


class Backup:
    """Originals of the files an operation rewrites, kept in the compact store (<root>/.aio_qc/backups.sqlite) + undo."""

    def __init__(self, root, what: str, title: str = ""):
        from .backups import BackupStore
        self.root = Path(root)
        self.store = BackupStore(self.root)
        self.what = what
        self.op: int | None = None                      # created on the first save (no empty operations)
        self.title = title or f"{what} {time.strftime('%Y-%m-%d %H:%M')}"
        self.saved: set[str] = set()
        self.after: dict[str, tuple] = {}              # original path -> signature right after our write

    @property
    def label(self) -> str:
        return f"{self.title} ({len(self.saved):,} file(s), .aio_qc/backups.sqlite)"

    def save(self, path) -> None:
        path = Path(path)
        if str(path) in self.saved or not path.is_file():
            return
        if self.op is None:
            self.op = self.store.begin(self.what, self.title)
        self.store.save(self.op, path)
        self.saved.add(str(path))

    def written(self, path) -> None:
        self.after[str(path)] = file_sig(path)

    def close(self):
        self.store.close()

    def restore(self) -> tuple[list[str], list[str]]:
        """Put the originals back — only files nobody changed after our write. -> (restored paths, skipped paths)"""
        restored, skipped = [], []
        for orig in sorted(self.saved):
            if orig not in self.after:
                continue                                 # saved but never written (failed): nothing to undo
            if file_sig(orig) != self.after[orig]:
                skipped.append(orig)
                continue
            if self.store.restore(self.op, orig) != "missing":
                restored.append(orig)
            else:
                skipped.append(orig)
        self.store.close()
        return restored, skipped
