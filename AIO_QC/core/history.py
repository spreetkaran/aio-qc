# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Per-file audit trail (Qt-free): <root>/.aio_qc/history.sqlite

Every write AIO QC makes, every review mark, every outside change it notices and every revert becomes one row:
    when · who (OS user) · tool · action · what · backup operation (if the original was stored)
Rows are tiny (a few dozen bytes) and written in one transaction per batch, so marking 10 000 files costs one commit.
Saved versions of a file are the originals in the backup store (core.backups): each operation that rewrote the file
stored the version BEFORE it — "revert to this version" restores exactly those bytes (after storing the current file
first, so a revert can itself be reverted).
"""
from __future__ import annotations

import getpass
import os
import sqlite3
import threading
import time
from pathlib import Path

ACTIONS = {"edit": "edited", "review": "review mark", "external": "changed outside AIO QC", "revert": "reverted",
           "undo": "undone", "trash": "moved to trash", "class": "class changed", "batch": "merged from batch",
           "move": "moved"}


def user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USERNAME") or os.environ.get("USER") or "?"


class HistoryLog:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / ".aio_qc" / "history.sqlite"
        self._db = None
        self._lock = threading.RLock()
        self.user = user()
        r = os.path.abspath(str(self.root))
        self._prefixes = tuple(sorted({r.rstrip("/\\") + s for s in ("/", "\\")}, key=len, reverse=True))

    @property
    def db(self):
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False, isolation_level=None)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, rel TEXT, user TEXT, tool TEXT,
                    action TEXT, what TEXT, op INTEGER);
                CREATE INDEX IF NOT EXISTS events_rel ON events(rel, ts);
                CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
            """)
            self._db = db
        return self._db

    def rel(self, key) -> str:
        k = os.path.abspath(str(key))
        for pre in self._prefixes:
            if k.startswith(pre):
                return k[len(pre):].replace("\\", "/")
        return k.replace("\\", "/")

    def log(self, keys, tool: str, action: str, what: str = "", op: int | None = None, ts: float | None = None):
        keys = [k for k in keys if k]
        if not keys:
            return
        now = ts or time.time()
        rows = [(now, self.rel(k), self.user, tool, action, what[:500], op) for k in keys]
        with self._lock:
            try:
                db = self.db
                db.execute("BEGIN")
                db.executemany("INSERT INTO events(ts, rel, user, tool, action, what, op) VALUES (?,?,?,?,?,?,?)", rows)
                db.execute("COMMIT")
            except sqlite3.Error:
                try:
                    self.db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass

    def for_file(self, key, limit: int = 500) -> list[dict]:
        if not self.path.exists():
            return []
        with self._lock:
            rows = self.db.execute("SELECT ts, user, tool, action, what, op FROM events WHERE rel=? ORDER BY ts DESC, "
                                   "id DESC LIMIT ?", (self.rel(key), limit)).fetchall()
        return [{"ts": r[0], "user": r[1], "tool": r[2], "action": r[3], "what": r[4], "op": r[5]} for r in rows]

    def since(self, ts: float, actions=None) -> list[dict]:
        if not self.path.exists():
            return []
        q = "SELECT ts, rel, user, tool, action, what FROM events WHERE ts > ?"
        args: list = [ts]
        if actions:
            q += " AND action IN (%s)" % ",".join("?" * len(actions))
            args += list(actions)
        with self._lock:
            rows = self.db.execute(q + " ORDER BY ts", args).fetchall()
        return [{"ts": r[0], "rel": r[1], "user": r[2], "tool": r[3], "action": r[4], "what": r[5]} for r in rows]

    def close(self):
        with self._lock:
            if self._db is not None:
                try:
                    self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    pass
                self._db.close()
                self._db = None


def versions(root, key) -> list[dict]:
    """Saved versions of a file in the backup store, newest first: [{op, kind, title, created, size}]."""
    from .backups import BackupStore
    store = BackupStore(root)
    if not store.exists():
        return []
    rel = store.rel(key)
    try:
        with store._lock:
            rows = store.db.execute("""SELECT o.id, o.kind, o.title, o.created, i.size FROM items i JOIN ops o
                                       ON o.id = i.op WHERE i.rel = ? ORDER BY o.created DESC""", (rel,)).fetchall()
    finally:
        store.close()
    return [{"op": r[0], "kind": r[1], "title": r[2], "created": r[3], "size": r[4]} for r in rows]


def version_bytes(root, op: int, key) -> bytes | None:
    from .backups import BackupStore
    store = BackupStore(root)
    try:
        got = store.original_bytes(op, key)
    finally:
        store.close()
    return got[0] if got else None


def revert(root, key, op: int) -> tuple[int | None, str]:
    """Restore version `op` of `key`; the current file is stored first (new op "revert"). -> (new op, how)."""
    from .backups import BackupStore
    store = BackupStore(root)
    try:
        new_op = None
        if Path(key).is_file():
            new_op = store.begin("revert", f"Before revert of {Path(key).name} · {time.strftime('%Y-%m-%d %H:%M')}")
            store.save(new_op, key)
        how = store.restore(op, key)
    finally:
        store.close()
    return new_op, how
