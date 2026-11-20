# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The issues folder of a class-folder dataset (Qt-free): images set aside while reviewing — one key (I), one step of
Undo — kept by the class they were filed under, so nothing about them is lost:

    <dataset>/issues/<class>/<image>              (default; inside the dataset, never read as a class)
    <dataset>/issues/<split>/<class>/<image>      (datasets with train / val / test)
    <dataset>_issues/<class>/<image>              (next to the dataset)
    <your folder>/<dataset name>/<class>/<image>  (a folder of your choice, e.g. another drive)

The folder carries a marker file (.aio_qc_issues) so AIO QC never reads it as a class, and an issues_log.csv with
every image sent there and back (when, from where, by which tool, the note). It is itself a class-folder dataset:
open it to review the set-aside images in the Class Gallery, then put them back (into the class folder they are in
now — so a class fixed in the issues folder is fixed on the way back).
"""
from __future__ import annotations

import csv
import errno
import os
import shutil
import time
from pathlib import Path

MARK = ".aio_qc_issues"
NAME = "issues"
NO_CLASS = "_no_class"
LOG = "issues_log.csv"
PLACES = ("dataset", "beside", "custom")


def is_issues_dir(p) -> bool:
    """A folder AIO QC made for set-aside images (never a class or a split)."""
    try:
        return os.path.isfile(os.path.join(str(p), MARK))
    except OSError:
        return False


def location(root, place: str = "dataset", custom: str = "") -> Path:
    """The issues folder of the dataset at root (see the module text)."""
    root = Path(root).resolve()
    if place == "beside":
        return root.parent / f"{root.name}_{NAME}"
    if place == "custom" and str(custom or "").strip():
        return Path(custom).expanduser().resolve() / root.name
    return root / NAME


def ensure(folder: Path, root) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mark = folder / MARK
    if not mark.exists():
        mark.write_text(f"AIO QC issues folder of {Path(root).resolve()}\n", encoding="utf-8")
    return folder


def _safe(name: str) -> str:
    return "".join("_" if c in '<>:"/\\|?*' else c for c in str(name)).strip(" .") or NO_CLASS


def destination(folder: Path, cls: str, split: str, name: str, taken=None) -> Path:
    """Where an image of this class (and split) goes; a free name (name_1.jpg …) when one is there already."""
    d = Path(folder)
    if split:
        d = d / _safe(split)
    d = d / _safe(cls or NO_CLASS)
    p = d / name
    stem, ext = os.path.splitext(name)
    n = 1
    while p.exists() or (taken is not None and str(p) in taken):
        p = d / f"{stem}_{n}{ext}"
        n += 1
    return p


def move(src, dst):
    """Rename when on the same drive (instant, any size); copy + delete across drives."""
    os.makedirs(os.path.dirname(str(dst)), exist_ok=True)
    try:
        os.replace(str(src), str(dst))
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.move(str(src), str(dst))


def log(folder: Path, rows, root):
    """Append [(action, source, destination, class, split, note, tool)] to issues_log.csv (paths of the dataset are
    written relative to it)."""
    p = Path(folder) / LOG
    new = not p.exists()
    root = str(Path(root).resolve())

    def rel(x):
        x = str(x)
        return os.path.relpath(x, root) if x.startswith(root + os.sep) else x
    try:
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "action", "from", "to", "class", "split", "note", "tool"])
            t = time.strftime("%Y-%m-%d %H:%M:%S")
            for action, src, dst, cls, split, note, tool in rows:
                w.writerow([t, action, rel(src), rel(dst), cls, split, note or "", tool or ""])
    except OSError:
        pass                                    # the log is a record, never a reason to fail a move


def contents(folder: Path, image_exts) -> list[Path]:
    """Every image in the issues folder."""
    out = []
    folder = Path(folder)
    if not folder.is_dir():
        return out
    for dp, dn, fn in os.walk(folder):
        dn[:] = [d for d in dn if not d.startswith(".")]
        for f in fn:
            if os.path.splitext(f)[1].lower() in image_exts:
                out.append(Path(dp) / f)
    return out


def return_plan(folder: Path, images, bases: dict, root) -> list[tuple[Path, Path, str, str]]:
    """Where each image in the issues folder goes back to: (image, destination, class, split) — the class folder
    it is in NOW (so a class fixed in the issues folder is fixed on the way back). bases: {split: base folder}
    ("" for a dataset without splits). Images directly in the issues folder are skipped."""
    folder = Path(folder)
    plan, taken = [], set()
    for img in images:
        parts = Path(img).relative_to(folder).parts
        split = ""
        if len(parts) >= 3 and parts[0] in bases and "" not in bases:
            split, parts = parts[0], parts[1:]
        if len(parts) < 2:
            continue
        cls = parts[0]
        base = Path(bases.get(split) or bases.get("") or root)
        dst = base / (cls if cls != NO_CLASS else "") / Path(*parts[1:])
        dst = _free(dst, taken)
        taken.add(str(dst))
        plan.append((Path(img), dst, "" if cls == NO_CLASS else cls, split))
    return plan


def _free(p: Path, taken) -> Path:
    stem, ext = os.path.splitext(p.name)
    n, q = 1, p
    while q.exists() or str(q) in taken:
        q = p.with_name(f"{stem}_{n}{ext}")
        n += 1
    return q


def prune_empty(folder: Path):
    """Remove empty class / split folders left in the issues folder (never the folder itself or its log)."""
    folder = Path(folder)
    for dp, _dn, _fn in sorted(os.walk(folder), key=lambda x: -len(x[0])):
        if Path(dp) == folder:
            continue
        try:
            os.rmdir(dp)
        except OSError:
            pass
