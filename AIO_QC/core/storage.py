# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Space used by AIO QC next to a dataset, and the housekeeping that keeps it small (Qt-free).

    <root>/.aio_qc/cache.sqlite       labels, image hashes, model / OCR / integrity results   (rebuildable)
    <root>/.aio_qc/backups.sqlite     originals of rewritten files                            (undo / safety)
    <root>/.aio_qc/edit_sessions/     editor sessions (work copies while open; a small manifest after)
    <root>/.aio_qc/exports/           ONNX / TensorRT exports made in Models & Environment
    <root>/.aio_qc/…                  review marks, issues, rules, settings (small)
    <root>/.aio_qc_trash/             files moved by QC deletes (recoverable)

housekeeping() runs in the background after a dataset is indexed: old backup folders (≤ 3.3) are compacted into
backups.sqlite, backups / finished sessions past the retention are removed, unused model results are dropped, the
cache forgets files that left the dataset, and leftover temp files are cleaned.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .backups import BackupStore, folder_bytes, legacy_folders, migrate_legacy
from .io import TRASH_DIR_NAME
from .state import STATE_DIR

DEFAULTS = {
    "backup_days": 30,        # delete backups older than this (0 = keep forever)
    "session_days": 14,       # delete finished editor-session manifests older than this
    "result_days": 90,        # drop cached model / OCR results not used for this long (0 = keep)
    "auto": True,             # run housekeeping automatically after indexing
}
_TEMP_SUFFIXES = (".aioqc_undo", ".migrating", ".tmp", ".partial")


@dataclass
class Part:
    key: str
    title: str
    bytes: int
    detail: str = ""
    items: list = field(default_factory=list)     # sub-entries (backup operations, cache signatures, …)


def _fmt_age(ts: float) -> str:
    d = (time.time() - ts) / 86400
    return "today" if d < 1 else f"{d:.0f} day(s) ago"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def usage(root, cache=None) -> list[Part]:
    """Breakdown of the space AIO QC uses for this dataset (fast: no walk of the dataset itself)."""
    root = Path(root)
    sd = root / STATE_DIR
    parts: list[Part] = []
    accounted = set()

    # cache
    db = sd / "cache.sqlite"
    files = [p for p in sd.glob("cache.sqlite*") if p.is_file()] if sd.is_dir() else []
    size = sum(p.stat().st_size for p in files)
    accounted.update(p.name for p in files)
    items, detail = [], ""
    if cache is not None and db.exists():
        try:
            tb = cache.table_bytes()
            detail = " · ".join(f"{k} {v[0]:,}" for k, v in tb.items() if k != "paths")
            items = cache.signatures()
        except Exception:
            pass
    parts.append(Part("cache", "Cache (rebuilt automatically when needed)", size, detail, items))

    # backups
    store = BackupStore(root)
    ops = store.ops() if store.exists() else []
    size = store.disk_size() if store.exists() else 0
    if sd.is_dir():
        accounted.update(p.name for p in sd.glob("backups.sqlite*"))
    orig = sum(o["original_bytes"] for o in ops)
    parts.append(Part("backups", "Backups (undo / originals of rewritten files)", size,
                      f"{len(ops)} operation(s) · {sum(o['files'] for o in ops):,} file(s)"
                      + (f" · originals {human(orig)}" if orig else ""), ops))
    store.close()
    legacy = legacy_folders(root)
    if legacy:
        accounted.add("backups")
        lb = [(f, folder_bytes(f)) for f in legacy]
        parts.append(Part("legacy", "Old-style backup folders (full copies — compact them)", sum(b for _f, b in lb),
                          f"{len(lb)} folder(s)", [{"path": str(f), "bytes": b} for f, b in lb]))

    # editor sessions
    sess = sd / "edit_sessions"
    if sess.is_dir():
        accounted.add("edit_sessions")
        items = []
        for d in sorted(sess.iterdir()):
            if d.is_dir():
                items.append({"path": str(d), "bytes": _unique_bytes(d), "active": _session_active(d),
                              "mtime": d.stat().st_mtime})
        parts.append(Part("sessions", "Editor sessions", sum(i["bytes"] for i in items),
                          f"{sum(i['active'] for i in items)} open · {sum(not i['active'] for i in items)} finished",
                          items))

    # exports (models)
    ex = sd / "exports"
    if ex.is_dir():
        accounted.add("exports")
        parts.append(Part("exports", "Model exports (ONNX / TensorRT)", folder_bytes(ex), str(ex)))

    # everything else in .aio_qc
    other = 0
    if sd.is_dir():
        for p in sd.iterdir():
            if p.name in accounted:
                continue
            other += folder_bytes(p) if p.is_dir() else p.stat().st_size
    parts.append(Part("state", "Review marks, issues, rules, settings", other))

    trash = root / TRASH_DIR_NAME
    if trash.is_dir():
        parts.append(Part("trash", "Trash (files moved by QC deletes)", folder_bytes(trash), str(trash)))
    return parts


def _unique_bytes(path: Path) -> int:
    """Bytes a folder really adds (hard-linked dataset images cost nothing)."""
    total = 0
    for dirpath, _d, files in os.walk(path):
        for f in files:
            try:
                st = os.lstat(os.path.join(dirpath, f))
            except OSError:
                continue
            if st.st_nlink <= 1 and not os.path.islink(os.path.join(dirpath, f)):
                total += st.st_size
    return total


def _session_active(d: Path) -> bool:
    import json
    try:
        return bool(json.loads((d / "session.json").read_text(encoding="utf-8")).get("active"))
    except (OSError, ValueError):
        return False


def clean_sessions(root, older_than_days: float) -> int:
    """Remove finished editor-session folders older than N days (0 = all finished ones). -> removed."""
    sess = Path(root) / STATE_DIR / "edit_sessions"
    if not sess.is_dir():
        return 0
    cutoff = time.time() - older_than_days * 86400
    n = 0
    for d in sess.iterdir():
        if d.is_dir() and not _session_active(d) and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


def clean_temp(root) -> int:
    sd = Path(root) / STATE_DIR
    n = 0
    if sd.is_dir():
        for p in sd.iterdir():
            if p.is_file() and p.name.endswith(_TEMP_SUFFIXES) and time.time() - p.stat().st_mtime > 3600:
                try:
                    p.unlink()
                    n += 1
                except OSError:
                    pass
    return n


def dir_bytes(root) -> int:
    sd = Path(root) / STATE_DIR
    return _unique_bytes(sd) if sd.is_dir() else 0


def housekeeping(root, cache=None, cfg: dict | None = None, alive_paths=None, keep_ops=frozenset()) -> dict:
    """Keep .aio_qc small. `alive_paths`: every image + annotation path of the dataset (cache orphan prune; only
    when the cache clearly holds more files than the dataset). -> report {freed, notes}"""
    cfg = {**DEFAULTS, **(cfg or {})}
    root = Path(root)
    before = dir_bytes(root)
    notes = []
    if legacy_folders(root):
        n, freed = migrate_legacy(root)
        if n:
            notes.append(f"compacted {n} old backup folder(s) ({human(freed)}) into backups.sqlite")
    if cfg["backup_days"] > 0:
        store = BackupStore(root)
        n = store.prune(cfg["backup_days"], keep=set(keep_ops))
        store.close()
        if n:
            notes.append(f"removed {n} backup(s) older than {cfg['backup_days']} days")
    if cfg["session_days"] > 0:
        n = clean_sessions(root, cfg["session_days"])
        if n:
            notes.append(f"removed {n} finished editor session(s)")
    if cache is not None:
        try:
            if cfg["result_days"] > 0:
                gone = cache.prune_sigs(cfg["result_days"])
                if gone:
                    notes.append(f"dropped {len(gone)} cached result set(s) unused for {cfg['result_days']} days")
            if alive_paths is not None:
                alive = [p for p in alive_paths if p is not None]
                if cache.file_count() > len(alive) * 1.05 + 100:
                    n = cache.prune(alive)
                    if n:
                        notes.append(f"cache forgot {n:,} row(s) of files no longer in the dataset")
        except Exception as exc:              # the cache is an optimisation: never fail housekeeping for it
            notes.append(f"cache maintenance skipped ({exc})")
    clean_temp(root)
    after = dir_bytes(root)
    return {"freed": max(0, before - after), "notes": notes, "size": after}
