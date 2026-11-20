# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Renaming files in the dataset itself (Name Clashes ▸ Rename): an image and its label file get the new name
together, in their own folders (YOLO: images/… and labels/… alike); a LabelMe JSON's imagePath is pointed at the
renamed image. All or nothing per file, nothing is ever overwritten, an upper / lower case-only rename goes via a
temporary name (Windows). Qt-free; the session carries review marks, notes, issues, cached results and backups over
to the new names (widgets.session.rename_samples).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path


def items_for(samples, root) -> list:
    """naming.Item per sample: named after its image (or its label file when there is no image)."""
    from .naming import Item
    root = os.path.abspath(str(root))
    out = []
    for s in samples:
        main = s.image_path if s.image_path is not None else s.ann_path
        if main is None:
            continue
        try:
            rel = os.path.relpath(str(main), root)
        except ValueError:
            rel = Path(main).name
        if rel.startswith(".."):
            rel = Path(main).name
        rel = rel.replace(os.sep, "/")
        # the folder that distinguishes it: below images/ (YOLO) — "images/train/camA/x.jpg" → camA
        parts = rel.split("/")
        low = [p.lower() for p in parts]
        if "images" in low[:-1]:
            i = low.index("images")
            parts = parts[i + 1:]
            if parts and parts[0].lower() in ("train", "val", "valid", "test") and len(parts) > 1:
                parts = parts[1:]
        elif parts and parts[0].lower() in ("train", "val", "valid", "test") and len(parts) > 1:
            parts = parts[1:]
        extra = tuple(dict.fromkeys(Path(p).suffix for p in (s.ann_path,) if p is not None and p != main))
        it = Item(s.key, "/".join(parts), split=s.split or "", path=str(main), extra_exts=extra)
        it.parent_l = os.path.dirname(os.path.normcase(str(main)))       # unique within its own folder
        out.append(it)
    return out


def pairs_for(sample, new_stem: str, image_only: bool = False) -> list[tuple[Path, Path]]:
    """[(old, new)] for the files of one sample (image_only: COCO — the annotation file is shared, not renamed).
    Raises ValueError when a name is already taken."""
    out = []
    files = [p for p in (sample.image_path, None if image_only else sample.ann_path) if p is not None]
    if sample.image_path is None and sample.ann_path is not None and not image_only:   # the image next to its JSON
        a = Path(sample.ann_path)
        files += [a.with_suffix(x) for x in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
                  if a.with_suffix(x).is_file()][:1]
    for f in dict.fromkeys(files):
        f = Path(f)
        if not f.exists():
            continue
        t = f.with_name(new_stem + f.suffix)
        if t == f:
            continue
        same = os.path.normcase(str(t)) == os.path.normcase(str(f))
        if not same and (t.exists() or os.path.lexists(t)):
            raise ValueError(f"'{t.name}' already exists there — not overwritten")
        out.append((f, t))
    return out


def move(a: Path, b: Path):
    if os.path.normcase(str(a)) == os.path.normcase(str(b)) and str(a) != str(b):
        tmp = a.with_name(a.name + ".aioqc-tmp")       # case-only: Windows needs a step in between
        os.replace(a, tmp)
        os.replace(tmp, b)
    else:
        os.replace(a, b)


def apply(pairs) -> None:
    """All or nothing for one sample's files."""
    done = []
    try:
        for a, b in pairs:
            move(a, b)
            done.append((a, b))
    except OSError:
        for a, b in reversed(done):
            try:
                move(b, a)
            except OSError:
                pass
        raise


def write_csv(path, log, root) -> None:
    """renamed.csv: every renamed file, old → new (relative to the dataset) — to trace a file back."""
    root = os.path.abspath(str(root))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["original", "new"])
        for e in log:
            for a, b in e["pairs"]:
                w.writerow([os.path.relpath(a, root).replace(os.sep, "/"), os.path.relpath(b, root).replace(os.sep, "/")])
