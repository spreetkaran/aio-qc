# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Save flagged files: copy them to a folder, or write their names to a text file (Review ▸ Save Flagged Files…).

    items = [FlaggedItem(display, [ann_path, image_path, …], status, note, ts), …]
    copy_items(items, root, out, keep_structure=True)  -> stats   (never overwrites; LabelMe imagePath kept valid)
    write_list(items, root, path, mode="relative")      -> lines written

Structure kept: paths relative to the dataset root are reproduced (YOLO images/ + labels/ stay paired, splits stay
splits). Flat: every file of an item gets the same unique stem (a.jpg + a.txt -> a_1.jpg + a_1.txt on a clash).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

OUTSIDE = "_outside_dataset"
LIST_MODES = {"relative": "Paths relative to the dataset", "absolute": "Full paths", "names": "File names only"}


@dataclass
class FlaggedItem:
    display: str
    files: list                      # [annotation, image, …] — existing files only are copied
    status: str = "flag"
    note: str = ""
    ts: str = ""
    copied: list = field(default_factory=list)


def _rel(root: Path, p: Path) -> Path:
    try:
        return Path(os.path.abspath(p)).relative_to(os.path.abspath(root))
    except ValueError:
        return Path(OUTSIDE) / Path(p).name


def _free(path: Path, taken: set) -> bool:
    return str(path) not in taken and not path.exists()


def _targets(item: FlaggedItem, root: Path, out: Path, keep_structure: bool, taken: set) -> list[tuple[Path, Path]]:
    files = [Path(f) for f in dict.fromkeys(item.files) if f is not None and Path(f).is_file()]
    if not files:
        return []
    if keep_structure:
        pairs = [(f, out / _rel(root, f)) for f in files]
        if all(_free(d, taken) for _s, d in pairs):
            return pairs
        # clash (exporting into a folder that already holds them): same numbered stem for every file of the item
        n = 1
        while True:
            cand = [(s, d.with_name(f"{d.stem}_{n}{d.suffix}")) for s, d in pairs]
            if all(_free(d, taken) for _s, d in cand):
                return cand
            n += 1
    stem = files[0].stem
    n = 0
    while True:
        st = stem if n == 0 else f"{stem}_{n}"
        cand = [(f, out / f"{st}{f.suffix}") for f in files]
        if len({d for _s, d in cand}) == len(cand) and all(_free(d, taken) for _s, d in cand):
            return cand
        n += 1


def _fix_labelme(json_dst: Path, image_dst: Path):
    """The copied JSON must point at the copied image (relative), everything else unchanged."""
    try:
        data = json.loads(json_dst.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(data, dict) or "imagePath" not in data:
        return
    new = Path(os.path.relpath(image_dst, json_dst.parent)).as_posix()
    if data.get("imagePath") != new:
        data["imagePath"] = new
        json_dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_items(items: list[FlaggedItem], root, out, keep_structure: bool = True, notes_csv: bool = True,
               progress=None, cancel=None) -> dict:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    taken: set = set()
    stats = {"items": 0, "files": 0, "missing": [], "errors": [], "cancelled": False}
    for i, it in enumerate(items, 1):
        if cancel is not None and cancel.is_set():
            stats["cancelled"] = True
            break
        pairs = _targets(it, root, out, keep_structure, taken)
        if not pairs:
            stats["missing"].append(it.display)
            continue
        try:
            for src, dst in pairs:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                taken.add(str(dst))
                it.copied.append(dst)
                stats["files"] += 1
            jsons = [d for _s, d in pairs if d.suffix.lower() == ".json"]
            imgs = [d for _s, d in pairs if d.suffix.lower() != ".json" and d.suffix.lower() != ".txt"]
            if jsons and imgs:
                _fix_labelme(jsons[0], imgs[0])
            stats["items"] += 1
        except OSError as exc:
            stats["errors"].append(f"{it.display}: {exc}")
        if progress is not None and i % 20 == 0:
            progress(i, len(items))
    if notes_csv and stats["items"]:
        with open(out / "flagged_files.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "status", "note", "marked_at", "copied_to"])
            for it in items:
                if it.copied:
                    w.writerow([it.display, it.status, it.note, it.ts,
                                " | ".join(Path(p).relative_to(out).as_posix() for p in it.copied)])
    return stats


def list_line(item: FlaggedItem, root: Path, mode: str) -> str:
    p = Path(item.files[0])
    if mode == "absolute":
        return os.path.abspath(p)
    if mode == "names":
        return p.name
    r = _rel(root, p)
    return r.as_posix() if r.parts[0] != OUTSIDE else os.path.abspath(p)


def write_list(items: list[FlaggedItem], root, path, mode: str = "relative", with_notes: bool = False) -> int:
    root = Path(root)
    lines = []
    for it in items:
        if not it.files:
            continue
        line = list_line(it, root, mode)
        if with_notes and it.note:
            line += "\t" + it.note.replace("\t", " ").replace("\n", " ")
        lines.append(line)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(tmp, path)
    return len(lines)
