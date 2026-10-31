# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Persistent per-dataset cache: <root>/.aio_qc/cache.sqlite

Everything is keyed by (relative path, mtime_ns, size) of the file it was derived from, so an entry is
reused only while that file is unchanged — no manual invalidation needed.

    labels   annotation file  -> raw label counts / parse error
    images   image file       -> width, height, sha256, pHash, lighting
    preds    image file + signature (model / OCR settings / image check) -> results

Space (schema 2, ~5x smaller than schema 1):
  * every path is stored ONCE (files table) and referenced by an integer id; signatures likewise (sigs table);
  * results are compact JSON compressed with zlib; sha256 as 32 raw bytes;
  * WAL capped (journal_size_limit) and truncated on close; auto_vacuum=INCREMENTAL returns freed pages to disk;
  * prune(alive) drops rows of files that left the dataset; drop_sig() / prune_sigs() drop unused signatures.
A schema-1 cache is migrated on open (predictions are kept — they can be expensive to recompute).

Thread-safe (one connection, a lock, WAL journal). Writes are batched: call flush() or rely on the
periodic commit every `commit_every` puts.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import zlib
from collections import Counter
from pathlib import Path

from .state import state_dir

SCHEMA_VERSION = 2
WAL_LIMIT = 32 * 1024 * 1024          # bytes the WAL file may keep after a checkpoint
_DUMPS = dict(separators=(",", ":"), ensure_ascii=False)


def file_sig(path) -> tuple[int, int] | None:
    """(mtime_ns, size) or None when the file does not exist."""
    try:
        st = os.stat(path)
    except (OSError, TypeError):
        return None
    return st.st_mtime_ns, st.st_size


def _pack(obj) -> bytes:
    return zlib.compress(json.dumps(obj, **_DUMPS).encode("utf-8"), 6)


def _unpack(blob):
    if blob is None:
        return None
    if isinstance(blob, str):                         # never expected in schema 2; tolerate
        return json.loads(blob)
    return json.loads(zlib.decompress(blob).decode("utf-8"))


SCHEMA = """
    CREATE TABLE IF NOT EXISTS meta  (k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, rel TEXT NOT NULL UNIQUE);
    CREATE TABLE IF NOT EXISTS sigs  (id INTEGER PRIMARY KEY, sig TEXT NOT NULL UNIQUE, used INTEGER);
    CREATE TABLE IF NOT EXISTS labels(fid INTEGER PRIMARY KEY, mtime INTEGER, size INTEGER, counts TEXT, error TEXT);
    CREATE TABLE IF NOT EXISTS images(fid INTEGER PRIMARY KEY, mtime INTEGER, size INTEGER, w INTEGER, h INTEGER,
                                      sha BLOB, phash BLOB, light TEXT);
    CREATE TABLE IF NOT EXISTS preds (sid INTEGER, fid INTEGER, mtime INTEGER, size INTEGER, data BLOB,
                                      PRIMARY KEY (sid, fid)) WITHOUT ROWID;
"""


class DatasetCache:
    def __init__(self, root: Path, commit_every: int = 500):
        self.root = Path(root)
        self.path = state_dir(self.root) / "cache.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pending = 0
        self.commit_every = commit_every
        self.hits = self.misses = 0
        self._root_str = os.path.normcase(os.path.abspath(str(self.root))) + os.sep
        self._fid: dict[str, int] = {}                # rel -> file id (bounded, see _file_id)
        self._sid: dict[str, int] = {}
        self._used: set[int] = set()
        self.migrated = 0
        self.db = self._open()

    # ------------------------------------------------------------------ open / migrate
    def _connect(self, path) -> sqlite3.Connection:
        db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        db.execute("PRAGMA auto_vacuum=INCREMENTAL")  # only takes effect on a new database (set before tables)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute(f"PRAGMA journal_size_limit={WAL_LIMIT}")
        return db

    def _schema(self, db) -> int:
        try:
            row = db.execute("SELECT v FROM meta WHERE k='schema'").fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0

    def _open(self) -> sqlite3.Connection:
        db = self._connect(self.path)
        ver = self._schema(db)
        if ver == SCHEMA_VERSION:
            return db
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if ver == 0 and not tables:
            db.executescript(SCHEMA)
            db.execute("INSERT OR REPLACE INTO meta VALUES ('schema', ?)", (str(SCHEMA_VERSION),))
            db.commit()
            return db
        db.close()
        return self._migrate(ver)

    def _migrate(self, ver: int) -> sqlite3.Connection:
        """Old cache -> new file (atomic swap). Schema 1 rows are converted; anything else starts empty."""
        tmp = self.path.with_name("cache.sqlite.migrating")
        for p in (tmp, Path(str(tmp) + "-wal"), Path(str(tmp) + "-shm")):
            try:
                p.unlink()
            except OSError:
                pass
        new = self._connect(tmp)
        new.executescript(SCHEMA)
        n = 0
        if ver == 1:
            old = sqlite3.connect(str(self.path), timeout=30)
            fid, sid = {}, {}

            def f_id(rel):
                i = fid.get(rel)
                if i is None:
                    i = fid[rel] = new.execute("INSERT INTO files(rel) VALUES (?)", (rel,)).lastrowid
                return i

            try:
                for rel, mt, sz, counts, err in old.execute("SELECT rel, mtime, size, counts, error FROM labels"):
                    new.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?)",
                                (f_id(rel), mt, sz, None if counts is None else
                                 json.dumps(json.loads(counts), **_DUMPS), err))
                    n += 1
                for rel, mt, sz, w, h, sha, ph, light in old.execute(
                        "SELECT rel, mtime, size, w, h, sha, phash, light FROM images"):
                    new.execute("INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?)",
                                (f_id(rel), mt, sz, w, h, bytes.fromhex(sha) if sha else None, ph, light))
                    n += 1
                for model, rel, mt, sz, data in old.execute("SELECT model, rel, mtime, size, data FROM preds"):
                    s = sid.get(model)
                    if s is None:
                        s = sid[model] = new.execute("INSERT INTO sigs(sig, used) VALUES (?, ?)",
                                                     (model, int(time.time()))).lastrowid
                    new.execute("INSERT OR REPLACE INTO preds VALUES (?,?,?,?,?)",
                                (s, f_id(rel), mt, sz, _pack(json.loads(data))))
                    n += 1
                ns = old.execute("SELECT v FROM meta WHERE k='labels_ns'").fetchone()
                if ns:
                    new.execute("INSERT OR REPLACE INTO meta VALUES ('labels_ns', ?)", ns)
            except sqlite3.Error:
                pass                                  # a damaged old cache: keep what was converted
            finally:
                old.close()
        new.execute("INSERT OR REPLACE INTO meta VALUES ('schema', ?)", (str(SCHEMA_VERSION),))
        new.commit()
        new.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        new.close()
        for suffix in ("-wal", "-shm"):
            try:
                Path(str(self.path) + suffix).unlink()
            except OSError:
                pass
        os.replace(tmp, self.path)
        for suffix in ("-wal", "-shm"):
            try:
                Path(str(tmp) + suffix).unlink()
            except OSError:
                pass
        self.migrated = n
        return self._connect(self.path)

    # ------------------------------------------------------------------ utils
    def rel(self, path) -> str:
        """String-only relative key (no filesystem resolve(): ~50x faster)."""
        p = os.path.abspath(str(path))
        n = os.path.normcase(p)
        if n.startswith(self._root_str):
            p = p[len(self._root_str):]
        return p.replace(os.sep, "/")

    def _file_id(self, rel: str, create: bool) -> int | None:
        """Caller holds the lock."""
        i = self._fid.get(rel)
        if i is not None:
            return i
        row = self.db.execute("SELECT id FROM files WHERE rel=?", (rel,)).fetchone()
        if row is not None:
            i = row[0]
        elif create:
            i = self.db.execute("INSERT INTO files(rel) VALUES (?)", (rel,)).lastrowid
        else:
            return None
        if len(self._fid) > 300_000:                  # bounded: ~30 MB of keys at most
            self._fid.clear()
        self._fid[rel] = i
        return i

    def _sig_id(self, sig: str, create: bool) -> int | None:
        """Caller holds the lock. Marks the signature as used today (pruning of unused signatures)."""
        i = self._sid.get(sig)
        if i is None:
            row = self.db.execute("SELECT id FROM sigs WHERE sig=?", (sig,)).fetchone()
            if row is not None:
                i = row[0]
            elif create:
                i = self.db.execute("INSERT INTO sigs(sig, used) VALUES (?, ?)", (sig, int(time.time()))).lastrowid
            else:
                return None
            self._sid[sig] = i
        if i not in self._used:
            self._used.add(i)
            self.db.execute("UPDATE sigs SET used=? WHERE id=?", (int(time.time()), i))
        return i

    def _tick(self):
        self._pending += 1
        if self._pending >= self.commit_every:
            self.db.commit()
            self._pending = 0

    def flush(self):
        with self._lock:
            if self._pending:
                self.db.commit()
                self._pending = 0

    def close(self):
        """Commit, return freed pages to the file system and truncate the WAL (nothing left behind)."""
        with self._lock:
            try:
                self.db.commit()
                self.db.execute("PRAGMA incremental_vacuum")
                self.db.commit()
                self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self.db.close()

    def clear(self):
        with self._lock:
            self.db.executescript("DELETE FROM labels; DELETE FROM images; DELETE FROM preds; DELETE FROM sigs; "
                                  "DELETE FROM files;")
            self.db.commit()
            self._fid.clear()
            self._sid.clear()
            self._used.clear()
            self.db.execute("VACUUM")
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def stats(self) -> dict:
        with self._lock:
            return {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("labels", "images", "preds")}

    def file_count(self) -> int:
        with self._lock:
            return self.db.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    def disk_size(self) -> int:
        return sum(Path(str(self.path) + s).stat().st_size for s in ("", "-wal", "-shm")
                   if Path(str(self.path) + s).exists())

    # ------------------------------------------------------------------ space management
    def prune(self, alive_paths) -> int:
        """Remove every row of files that are no longer part of the dataset. -> rows removed."""
        alive = {self.rel(p) for p in alive_paths if p is not None}
        with self._lock:
            dead = [i for i, rel in self.db.execute("SELECT id, rel FROM files") if rel not in alive]
            n = 0
            for k in range(0, len(dead), 900):
                chunk = dead[k:k + 900]
                q = ",".join("?" * len(chunk))
                for sql in (f"DELETE FROM labels WHERE fid IN ({q})", f"DELETE FROM images WHERE fid IN ({q})",
                            f"DELETE FROM preds WHERE fid IN ({q})", f"DELETE FROM files WHERE id IN ({q})"):
                    n += self.db.execute(sql, chunk).rowcount
            self.db.commit()
            self._fid.clear()
            if n:
                self.db.execute("PRAGMA incremental_vacuum")
                self.db.commit()
            return n

    def signatures(self) -> list[dict]:
        """[{sig, rows, bytes, used}] — results per model / OCR setting / check (Storage window)."""
        with self._lock:
            return [{"sig": s, "rows": n or 0, "bytes": b or 0, "used": u} for s, n, b, u in self.db.execute(
                "SELECT s.sig, COUNT(p.fid), SUM(LENGTH(p.data)), s.used FROM sigs s "
                "LEFT JOIN preds p ON p.sid = s.id GROUP BY s.id ORDER BY 3 DESC")]

    def drop_sig(self, sig: str) -> int:
        with self._lock:
            row = self.db.execute("SELECT id FROM sigs WHERE sig=?", (sig,)).fetchone()
            if row is None:
                return 0
            n = self.db.execute("DELETE FROM preds WHERE sid=?", row).rowcount
            self.db.execute("DELETE FROM sigs WHERE id=?", row)
            self.db.commit()
            self._sid.pop(sig, None)
            self.db.execute("PRAGMA incremental_vacuum")
            self.db.commit()
            return n

    def prune_sigs(self, older_than_days: float, keep_prefixes=("integrity|", "iq|")) -> list[str]:
        """Drop results of signatures not used for N days (old model versions, abandoned OCR settings)."""
        limit = time.time() - older_than_days * 86400
        with self._lock:
            old = [s for s, u in self.db.execute("SELECT sig, used FROM sigs")
                   if (u or 0) < limit and not s.startswith(tuple(keep_prefixes))]
        for s in old:
            self.drop_sig(s)
        return old

    def table_bytes(self) -> dict:
        """Approximate payload per table (labels / images / predictions) for the Storage window."""
        with self._lock:
            q = {"labels": "SELECT COUNT(*), SUM(LENGTH(counts)) + 24 * COUNT(*) FROM labels",
                 "images": "SELECT COUNT(*), SUM(LENGTH(sha)) + SUM(LENGTH(phash)) + 40 * COUNT(*) FROM images",
                 "predictions": "SELECT COUNT(*), SUM(LENGTH(data)) + 24 * COUNT(*) FROM preds",
                 "paths": "SELECT COUNT(*), SUM(LENGTH(rel)) * 2 + 16 * COUNT(*) FROM files"}
            return {k: tuple(v or 0 for v in self.db.execute(sql).fetchone()) for k, sql in q.items()}

    # ----------------------------------------------------------------- labels
    def ensure_label_namespace(self, signature: str):
        """Cached label names depend on the YOLO class map: a different map invalidates them."""
        with self._lock:
            row = self.db.execute("SELECT v FROM meta WHERE k='labels_ns'").fetchone()
            if row is None or row[0] != signature:
                self.db.execute("DELETE FROM labels")
                self.db.execute("INSERT OR REPLACE INTO meta VALUES ('labels_ns', ?)", (signature,))
                self.db.commit()

    def load_all_labels(self) -> dict:
        """Bulk snapshot {rel: (mtime, size, counts_json, error)} for a fast full-dataset index."""
        with self._lock:
            return {r[0]: r[1:] for r in self.db.execute(
                "SELECT f.rel, l.mtime, l.size, l.counts, l.error FROM labels l JOIN files f ON f.id = l.fid")}

    def lookup_labels(self, snapshot: dict, ann_path):
        """Like get_labels() but against a snapshot from load_all_labels() (no SQL per file)."""
        row = snapshot.get(self.rel(ann_path))
        sig = file_sig(ann_path) or (0, -1)
        if row is None or (row[0], row[1]) != sig:
            self.misses += 1
            return None
        self.hits += 1
        return (Counter(json.loads(row[2])) if row[2] is not None else None), row[3]

    def get_labels(self, ann_path):
        """-> (Counter | None, error | None) on a hit, or None on a miss. Missing file => sig (0, -1)."""
        sig = file_sig(ann_path) or (0, -1)
        with self._lock:
            row = self.db.execute("SELECT l.mtime, l.size, l.counts, l.error FROM labels l JOIN files f "
                                  "ON f.id = l.fid WHERE f.rel=?", (self.rel(ann_path),)).fetchone()
        if row is None or (row[0], row[1]) != sig:
            self.misses += 1
            return None
        self.hits += 1
        return (Counter(json.loads(row[2])) if row[2] is not None else None), row[3]

    def put_labels(self, ann_path, counts: Counter | None, error: str | None):
        sig = file_sig(ann_path) or (0, -1)
        with self._lock:
            fid = self._file_id(self.rel(ann_path), True)
            self.db.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?)",
                            (fid, sig[0], sig[1], json.dumps(dict(counts), **_DUMPS) if counts is not None else None,
                             error))
            self._tick()

    # ----------------------------------------------------------------- images
    def get_image(self, img_path) -> dict | None:
        sig = file_sig(img_path)
        if sig is None:
            return None
        with self._lock:
            row = self.db.execute("SELECT i.mtime, i.size, i.w, i.h, i.sha, i.phash, i.light FROM images i "
                                  "JOIN files f ON f.id = i.fid WHERE f.rel=?", (self.rel(img_path),)).fetchone()
        if row is None or (row[0], row[1]) != sig:
            self.misses += 1
            return None
        self.hits += 1
        return {"w": row[2], "h": row[3], "sha": row[4].hex() if row[4] else None, "phash": row[5], "light": row[6]}

    def put_image(self, img_path, **fields):
        """Merge fields (w, h, sha, phash, light) into the row for this exact file version."""
        sig = file_sig(img_path)
        if sig is None:
            return
        cur = self.get_image(img_path) or {}
        cur.update({k: v for k, v in fields.items() if v is not None})
        ph, sha = cur.get("phash"), cur.get("sha")
        with self._lock:
            fid = self._file_id(self.rel(img_path), True)
            self.db.execute("INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?)",
                            (fid, sig[0], sig[1], cur.get("w"), cur.get("h"),
                             bytes.fromhex(sha) if isinstance(sha, str) else sha,
                             bytes(ph) if ph is not None else None, cur.get("light")))
            self._tick()

    # ------------------------------------------------------------------ preds
    def get_preds(self, model_sig: str, img_path):
        sig = file_sig(img_path)
        if sig is None:
            return None
        with self._lock:
            sid = self._sig_id(model_sig, False)
            if sid is None:
                self.misses += 1
                return None
            row = self.db.execute("SELECT p.mtime, p.size, p.data FROM preds p JOIN files f ON f.id = p.fid "
                                  "WHERE p.sid=? AND f.rel=?", (sid, self.rel(img_path))).fetchone()
        if row is None or (row[0], row[1]) != sig:
            self.misses += 1
            return None
        self.hits += 1
        return _unpack(row[2])

    def put_preds(self, model_sig: str, img_path, data):
        sig = file_sig(img_path)
        if sig is None:
            return
        blob = _pack(data)                            # compress outside the lock
        with self._lock:
            sid = self._sig_id(model_sig, True)
            fid = self._file_id(self.rel(img_path), True)
            self.db.execute("INSERT OR REPLACE INTO preds VALUES (?,?,?,?,?)", (sid, fid, sig[0], sig[1], blob))
            self._tick()
