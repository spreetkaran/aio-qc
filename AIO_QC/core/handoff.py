# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Hand-off batches (Qt-free): send flagged files to an annotator, merge the corrected files back safely.

    export_batch(root, items, out, kind, names, title)       -> stats       (Review ▸ Hand-off ▸ Export Batch…)
    m = load_manifest(batch_dir)
    rows = compare(root, m, batch_dir, kind, names)           -> [MergeRow]  (Review ▸ Hand-off ▸ Import Batch…)
    apply_rows(root, rows, title)                             -> (written, errors, backup op)

The batch folder is self-describing and works without AIO QC on the annotator's side:
    batch.json        manifest — per file: dataset paths, where it is in the batch, a fingerprint (sha256) of the
                      annotation as exported, the reviewer's note and the findings
    <dataset layout>  images + annotations (YOLO images/ + labels/ stay paired, splits stay splits; LabelMe imagePath
                      points at the copied image) + the class list for YOLO
    _hints/           (optional) the image with the flagged regions drawn — what to look at
    notes.csv, README.txt

Merge = three-way, per file, by fingerprint (base = as exported, theirs = returned, mine = the dataset now):
    ready      they changed it, nobody changed it here since       -> written on merge
    conflict   they changed it AND it changed here since export    -> you choose (take theirs / keep mine)
    unchanged  they returned it as it was                          -> nothing to do
    same       the dataset already has exactly their version       -> nothing to do
    missing    the file is not in the returned batch
    invalid    unreadable JSON / YOLO class id outside the class list / different image size / dataset file gone
Every written file is backed up first (one backup operation, undo in Backups); LabelMe files keep the dataset's
imagePath / imageData / size so a returned batch never re-points or bloats the dataset.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import annodiff
from .flagged_export import FlaggedItem, _fix_labelme, _rel, _targets

FORMAT = "aio-qc-batch"
MANIFEST = "batch.json"
STATES = {"ready": "Ready to merge", "conflict": "Conflict — changed on both sides", "unchanged": "Not changed",
          "same": "Already merged", "missing": "Not returned", "invalid": "Invalid"}


@dataclass
class BatchItem:
    key: str
    display: str
    ann_path: str | None
    image_path: str | None
    note: str = ""
    status: str = ""
    issues: list = field(default_factory=list)        # ["Overlap: car / truck IoU 0.93", …]
    marks: list = field(default_factory=list)         # [(x1, y1, x2, y2, text)] in image pixels — hint image


def sha(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def _read(p) -> bytes | None:
    try:
        return Path(p).read_bytes() if p is not None and Path(p).is_file() else None
    except OSError:
        return None


def _posix(root: Path, p) -> str:
    return _rel(root, Path(p)).as_posix()


# ---------------------------------------------------------------------------------------------------------- export
def export_batch(root, items: list[BatchItem], out, kind: str, names=None, title: str = "", hints: bool = True,
                 class_files=(), backup_base=True, progress=None, cancel=None) -> dict:
    """Copy the files (dataset layout kept), write the manifest. The exported annotations are also stored in the
    dataset's backup store (one operation) so a later merge can show what the annotator changed."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / MANIFEST).exists():
        raise FileExistsError(f"{out} already holds a batch — choose an empty folder")
    store, op = None, None
    if backup_base:
        from .backups import BackupStore
        store = BackupStore(root)
        op = store.begin("handoff", f"Hand-off batch “{title or out.name}” · as exported")
    taken: set = set()
    entries, stats = [], {"items": 0, "files": 0, "hints": 0, "missing": [], "errors": [], "cancelled": False}
    try:
        for i, it in enumerate(items, 1):
            if cancel is not None and cancel.is_set():
                stats["cancelled"] = True
                break
            fi = FlaggedItem(it.display, [it.ann_path, it.image_path])
            pairs = _targets(fi, root, out, True, taken)
            if not pairs:
                stats["missing"].append(it.display)
                continue
            try:
                ann_dst = img_dst = None
                for src, dst in pairs:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    taken.add(str(dst))
                    stats["files"] += 1
                    if it.ann_path is not None and os.path.abspath(src) == os.path.abspath(it.ann_path):
                        ann_dst = dst
                    elif it.image_path is not None and os.path.abspath(src) == os.path.abspath(it.image_path):
                        img_dst = dst
                if ann_dst is None and it.ann_path is not None and kind == "yolo":
                    # YOLO background image: no label file yet — where the annotator should create it
                    ann_dst = out / _rel(root, Path(it.ann_path))
                if ann_dst is not None and img_dst is not None and ann_dst.suffix.lower() == ".json" \
                        and ann_dst.is_file():
                    _fix_labelme(ann_dst, img_dst)
                base = _read(it.ann_path)
                if store is not None and base is not None:
                    store.save(op, it.ann_path)
                hint = None
                if hints and it.marks and img_dst is not None:
                    hint = _hint(img_dst, it.marks, out / "_hints" / (Path(img_dst).stem + "_hint.jpg"))
                    stats["hints"] += hint is not None
                entries.append({
                    "key": _posix(root, it.key), "ann": _posix(root, it.ann_path) if it.ann_path else None,
                    "image": _posix(root, it.image_path) if it.image_path else None,
                    "batch_ann": ann_dst.relative_to(out).as_posix() if ann_dst is not None else None,
                    "batch_image": img_dst.relative_to(out).as_posix() if img_dst is not None else None,
                    "hint": hint.relative_to(out).as_posix() if hint is not None else None,
                    "base_sha256": sha(base), "note": it.note, "status": it.status, "issues": list(it.issues)[:50],
                })
                stats["items"] += 1
            except OSError as exc:
                stats["errors"].append(f"{it.display}: {exc}")
            if progress is not None and i % 20 == 0:
                progress(i, len(items))
    finally:
        if store is not None:
            store.close()
    for cf in class_files:                              # YOLO: data.yaml / classes.txt so the editor knows the names
        try:
            if Path(cf).is_file() and not (out / Path(cf).name).exists():
                shutil.copy2(cf, out / Path(cf).name)
        except OSError:
            pass
    if kind == "yolo" and names and not any((out / n).exists() for n in ("classes.txt", "data.yaml")):
        (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    try:                                                # the labelling rules travel with the files
        from .guidelines import GuideStore
        g = GuideStore(root)
        if g.scopes():
            g.export_html(out / "GUIDELINES.html", title=f"Labelling guidelines — {Path(root).name}")
            stats["guidelines"] = len(g.scopes())
    except Exception:
        pass
    manifest = {"format": FORMAT, "version": 1, "id": uuid.uuid4().hex, "title": title or out.name,
                "created": time.time(), "user": _user(), "dataset": os.path.abspath(root), "kind": kind,
                "classes": list(names or []), "backup_op": op, "items": entries}
    (out / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "notes.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["annotation", "image", "note", "findings", "hint"])
        for e in entries:
            w.writerow([e["batch_ann"] or "", e["batch_image"] or "", e["note"], " | ".join(e["issues"]),
                        e["hint"] or ""])
    (out / "README.txt").write_text(_README.format(title=manifest["title"], n=len(entries),
                                                   when=time.strftime("%Y-%m-%d %H:%M"), user=manifest["user"]),
                                    encoding="utf-8")
    stats["manifest"] = manifest
    return stats


_README = """Hand-off batch: {title}
{n} files · exported {when} by {user} with AIO QC

What to do
  1. Open this folder in your annotation tool (LabelMe, or any YOLO editor) — it has the dataset's own layout.
  2. Fix the files listed in notes.csv. The note says what the reviewer wants; _hints/ shows the regions to look at;
     GUIDELINES.html (if present) has the labelling rules with good / bad examples.
  3. Save in place. Do not rename or move files, and keep batch.json — it is how the files are merged back.
  4. Return the whole folder (or a zip of it).

The reviewer merges it with Review ▸ Hand-off ▸ Import Batch. Files changed on both sides since the export are
shown as conflicts and never overwritten silently.
"""


def _user() -> str:
    from .history import user
    return user()


def _hint(img_path: Path, marks, dst: Path) -> Path | None:
    """The image with the flagged regions drawn (downscaled to 1600 px; best effort)."""
    from .thumbs import render
    data = render(img_path, [tuple(m[:5]) + ((255, 64, 160),) for m in marks], max_side=1600, quality=88)
    if data is None:
        return None
    n = 1
    while dst.exists():
        dst = dst.with_name(f"{dst.stem}_{n}{dst.suffix}")
        n += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return dst


# ---------------------------------------------------------------------------------------------------------- import
def find_batch(path) -> Path | None:
    """A batch folder, a folder containing exactly one batch, or a zip (extracted next to it)."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".zip":
        import zipfile
        dst = p.with_suffix("")
        n = 1
        while dst.exists() and not (dst / MANIFEST).exists() and not any(dst.glob(f"*/{MANIFEST}")):
            dst = p.with_name(f"{p.stem}_{n}")
            n += 1
        if not dst.exists():
            with zipfile.ZipFile(p) as z:
                for m in z.infolist():                    # no path escapes
                    t = (dst / m.filename).resolve()
                    if not str(t).startswith(str(dst.resolve())):
                        raise ValueError(f"unsafe path in zip: {m.filename}")
                z.extractall(dst)
        p = dst
    if p.is_file() and p.name == MANIFEST:
        p = p.parent
    if (p / MANIFEST).is_file():
        return p
    found = list(p.glob(f"*/{MANIFEST}")) if p.is_dir() else []
    return found[0].parent if len(found) == 1 else None


def load_manifest(batch_dir) -> dict:
    m = json.loads((Path(batch_dir) / MANIFEST).read_text(encoding="utf-8"))
    if m.get("format") != FORMAT:
        raise ValueError("not an AIO QC hand-off batch (batch.json has a different format)")
    return m


@dataclass
class MergeRow:
    entry: dict
    state: str
    target: Path | None                    # the dataset annotation file
    data: bytes | None = None              # what will be written (theirs, LabelMe image fields taken from mine)
    theirs: str = ""                       # what the annotator changed (base -> returned)
    mine: str = ""                         # what changed here since the export (base -> now)
    reason: str = ""
    choice: str = ""                       # "theirs" | "mine" — merge decision (ready -> theirs by default)


def _labelme_keep_mine(ret: bytes, cur: bytes | None) -> tuple[bytes | None, str]:
    try:
        r = json.loads(ret.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return None, f"returned file is not valid JSON ({exc})"
    if not isinstance(r, dict) or not isinstance(r.get("shapes", []), list):
        return None, "returned file is not a LabelMe annotation"
    if cur is not None:
        try:
            c = json.loads(cur.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            c = None
        if isinstance(c, dict):
            for k in ("imageWidth", "imageHeight"):
                if c.get(k) and r.get(k) and int(r[k]) != int(c[k]):
                    return None, f"image size differs ({r.get('imageWidth')}×{r.get('imageHeight')} returned, " \
                                 f"{c.get('imageWidth')}×{c.get('imageHeight')} in the dataset)"
            for k in ("imagePath", "imageData", "imageWidth", "imageHeight"):
                if k in c:
                    r[k] = c[k]
                else:
                    r.pop(k, None)
    return (json.dumps(r, ensure_ascii=False, indent=2) + "\n").encode("utf-8"), ""


def _yolo_check(ret: bytes, n_classes: int) -> str:
    try:
        text = ret.decode("utf-8")
    except UnicodeDecodeError:
        return "returned file is not UTF-8 text"
    for ln, line in enumerate(text.splitlines(), 1):
        p = line.split()
        if not p:
            continue
        try:
            cid = int(p[0])
            vals = [float(v) for v in p[1:]]
        except ValueError:
            return f"line {ln}: not numbers"
        if len(vals) < 4:
            return f"line {ln}: too few values"
        if n_classes and not 0 <= cid < n_classes:
            return f"line {ln}: class id {cid} is not in the class list (0–{n_classes - 1})"
    return ""


def compare(root, manifest: dict, batch_dir, kind: str | None = None, names=None, progress=None) -> list[MergeRow]:
    root, batch_dir = Path(root), Path(batch_dir)
    kind = kind or manifest.get("kind", "labelme")
    names = list(names) if names else list(manifest.get("classes") or [])
    op = manifest.get("backup_op")
    store = None
    if op is not None:
        from .backups import BackupStore
        store = BackupStore(root)
        if not store.exists():
            store.close()
            store = None
    rows = []
    try:
        items = manifest.get("items") or []
        for i, e in enumerate(items, 1):
            rows.append(_compare_one(root, batch_dir, e, kind, names, store, op))
            if progress is not None and i % 50 == 0:
                progress(i, len(items))
    finally:
        if store is not None:
            store.close()
    return rows


def _compare_one(root, batch_dir, e, kind, names, store, op) -> MergeRow:
    target = root / e["ann"] if e.get("ann") and not os.path.isabs(e["ann"]) else None
    if target is None:
        return MergeRow(e, "invalid", None, reason="the annotation was outside the dataset folder")
    if e.get("batch_ann") is None:
        return MergeRow(e, "missing", target, reason="no annotation in the batch")
    ret = _read(batch_dir / e["batch_ann"])
    cur = _read(target)
    base_sha = e.get("base_sha256")
    if ret is None:
        return MergeRow(e, "missing", target, reason="the annotation file is not in the returned batch")
    if cur is None and base_sha is not None:
        return MergeRow(e, "invalid", target, reason="the dataset file was removed since the export")
    image = root / e["image"] if e.get("image") and not os.path.isabs(e["image"]) else None
    if image is not None and not image.is_file() and kind == "yolo":
        return MergeRow(e, "invalid", target, reason="the image is no longer in the dataset")
    # what will be written
    if kind == "labelme":
        data, err = _labelme_keep_mine(ret, cur)
    else:
        err = _yolo_check(ret, len(names))
        data = ret
    if err:
        return MergeRow(e, "invalid", target, reason=err)
    base = None
    if store is not None and base_sha is not None:
        got = store.original_bytes(op, target)
        base = got[0] if got else None
    parsed = {}

    def P(b):
        k = id(b)
        if k not in parsed:
            parsed[k] = annodiff.parse(b, kind, names)
        return parsed[k]

    ret_sha, cur_sha = sha(ret), sha(cur)
    their_d = annodiff.diff(P(base), P(ret)) if base is not None or base_sha is None else None
    if ret_sha == base_sha or (their_d is not None and not annodiff.is_change(their_d) and base is not None):
        return MergeRow(e, "unchanged", target, data, "no changes", reason="returned as exported")
    if cur is not None and (cur_sha == ret_sha or data == cur):
        return MergeRow(e, "same", target, data, reason="the dataset already has this version")
    theirs = annodiff.summary(their_d) if their_d is not None else \
        "changed (" + annodiff.summary(annodiff.diff(P(cur), P(ret))) + " vs now)"
    if cur_sha == base_sha:
        return MergeRow(e, "ready", target, data, theirs, choice="theirs")
    mine = annodiff.summary(annodiff.diff(P(base), P(cur))) if base is not None else "changed here since export"
    if base is not None and not annodiff.is_change(annodiff.diff(P(base), P(cur))):
        # only bytes changed here (formatting / imageData) — annotations are the same: safe to take theirs
        return MergeRow(e, "ready", target, data, theirs, "formatting only", choice="theirs")
    return MergeRow(e, "conflict", target, data, theirs, mine, reason="changed here AND by the annotator",
                    choice="")


def apply_rows(root, rows: list[MergeRow], title: str = "") -> tuple[list[MergeRow], list[str], int | None]:
    """Write every row whose choice is "theirs" (backup first, atomic replace). -> (written, errors, backup op)."""
    from .backups import BackupStore
    todo = [r for r in rows if r.choice == "theirs" and r.data is not None and r.target is not None
            and r.state in ("ready", "conflict")]
    if not todo:
        return [], [], None
    store = BackupStore(root)
    written, errors = [], []
    try:
        op = store.begin("batch", f"Merge of batch “{title}” · {time.strftime('%Y-%m-%d %H:%M')}")
        for r in todo:
            try:
                if r.target.is_file():
                    store.save(op, r.target)
                r.target.parent.mkdir(parents=True, exist_ok=True)
                tmp = r.target.with_name(f".{r.target.name}.aioqc_merge")
                tmp.write_bytes(r.data)
                if r.target.exists():
                    try:
                        shutil.copystat(r.target, tmp)
                    except OSError:
                        pass
                os.replace(tmp, r.target)
                written.append(r)
            except OSError as exc:
                errors.append(f"{r.entry.get('ann')}: {exc}")
    finally:
        store.close()
    return written, errors, op
