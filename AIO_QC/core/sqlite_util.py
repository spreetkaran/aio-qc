# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
SQLite where the data lives. The dataset's own databases (.aio_qc/cache.sqlite, history, backups, …) sit next to the
dataset — often on a network share or a USB drive. SQLite's WAL mode keeps its index in a memory-mapped "-shm" file,
which network file systems do not support (SQLite's own documentation) and removable drives handle badly: reads of
that mapping fail with "Windows fatal exception: page error" (EXCEPTION_IN_PAGE_ERROR) and can corrupt the database.
There the classic rollback journal is used instead (no memory mapping); local disks keep WAL (faster).

    journal_mode(path)      -> "WAL" | "DELETE"
    configure(db, path)     PRAGMA journal_mode (+ no memory-mapped I/O off local disks); returns the mode used
    storage_kind(path)      -> "local" | "network" | "removable"
"""
from __future__ import annotations

import os
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path
from AIO_QC.core.diag import ignored as _ignored

NETWORK_FS = {"nfs", "nfs4", "cifs", "smb", "smb2", "smb3", "smbfs", "sshfs", "fuse.sshfs", "9p", "davfs", "fuse.davfs",
              "afpfs", "glusterfs", "fuse.glusterfs", "ceph", "fuse.ceph", "lustre", "fuse.rclone", "fuse.s3fs",
              "gcsfuse", "fuse.gcsfuse", "webdav", "vboxsf", "drvfs", "9p2000"}
REMOVABLE_FS = {"vfat", "exfat", "msdos", "fuseblk"}


def storage_kind(path) -> str:
    try:
        p = os.path.abspath(str(path))
    except (TypeError, ValueError):
        return "local"
    if os.environ.get("AIOQC_SQLITE_STORAGE") in ("local", "network", "removable"):
        return os.environ["AIOQC_SQLITE_STORAGE"]       # an override (and for tests)
    if sys.platform == "win32":
        return _windows_kind(p)
    if sys.platform == "darwin":
        return "removable" if p.startswith("/Volumes/") else "local"
    return _linux_kind(p)


def _windows_kind(p: str) -> str:
    if p.startswith("\\\\"):
        return "network"                                 # \\server\share\…
    drive = os.path.splitdrive(p)[0]
    if not drive:
        return "local"
    return _windows_drive(drive.upper() + "\\")


@lru_cache(maxsize=64)
def _windows_drive(root: str) -> str:
    try:
        import ctypes
        t = ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(root))
    except Exception as _exc:
        _ignored(_exc)
        return "local"
    return {4: "network", 2: "removable"}.get(t, "local")    # DRIVE_REMOTE, DRIVE_REMOVABLE


def _linux_kind(p: str) -> str:
    best, fstype, dev = "", "", ""
    try:
        with open("/proc/mounts", encoding="utf-8", errors="replace") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1].replace("\\040", " ")
                if (p == mnt or p.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best):
                    best, fstype, dev = mnt, parts[2], parts[0]
    except OSError:
        return "local"
    if fstype in NETWORK_FS or dev.startswith("//") or (":" in dev and dev.split(":", 1)[0] and "/" not in
                                                        dev.split(":", 1)[0]):
        return "network"
    if fstype in REMOVABLE_FS and best.startswith(("/media/", "/run/media/", "/mnt/")):
        return "removable"
    return "local"


def journal_mode(path) -> str:
    return "WAL" if storage_kind(path) == "local" else "DELETE"


def configure(db: sqlite3.Connection, path) -> str:
    """Set the journal for where `path` is; switching an existing WAL database back needs it unshared — if that
    fails the database simply stays as it is."""
    mode = journal_mode(Path(path).parent if Path(path).suffix else path)
    try:
        got = db.execute(f"PRAGMA journal_mode={mode}").fetchone()
        mode = str(got[0]).upper() if got else mode
    except sqlite3.Error:
        pass
    if mode != "WAL":
        try:
            db.execute("PRAGMA mmap_size=0")             # no memory-mapped reads either
        except sqlite3.Error:
            pass
    return mode
