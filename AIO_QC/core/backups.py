# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Compact, crash-safe backup store for every file AIO QC rewrites: <root>/.aio_qc/backups.sqlite

One SQLite file instead of folders of full copies:
  * each original is stored zlib-compressed (JSON / YOLO text shrinks 5–20×)
  * a LabelMe `imageData` (the base64 image inside the JSON — usually 95 % of the file) is NOT stored when it can be
    rebuilt: either it IS the image file's bytes (checked by SHA-256; exact restore) or it can be re-encoded from the
    image file the way that labelme version does it ("regen"; same pixels). When there is no image file to rebuild
    from — or `EXACT_IMAGE_DATA` is on — the decoded bytes go to a shared, de-duplicated blob table.
  * restoring puts the base64 back at its exact character position, so files come back byte-for-byte (exact) or
    differ only in the re-encoded image bytes (regen)
  * transactional (a crash never corrupts earlier backups), incremental vacuum (deleting backups frees disk)

    store = BackupStore(root)
    op = store.begin("dimarea", "Dim/Area delete")      # one operation = one undo unit
    store.save(op, path)                                 # before the first write, once per file
    store.restore(op, path)                              # undo  -> "exact" | "regen" | "no-image" | "missing"
    store.ops() · store.delete(op) · store.prune(days) · store.export(op, folder) · store.import_folder(...)
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
import zlib
from pathlib import Path

from .labelme_image import KEY as IMG_KEY, encode_b64, image_file_of

DB_NAME = "backups.sqlite"
EXACT_IMAGE_DATA = False          # app setting: keep embedded images byte-exact even when an image file exists
_MIN_INLINE = 256                 # shorter imageData strings are simply kept in the compressed text
_locks: dict[str, threading.RLock] = {}


def _lock_for(path: Path) -> threading.RLock:
    return _locks.setdefault(str(path), threading.RLock())


def _sha(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


class BackupStore:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / ".aio_qc" / DB_NAME
        self._db: sqlite3.Connection | None = None
        self._lock = _lock_for(self.path)

    # ------------------------------------------------------------------ connection
    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False, isolation_level=None)
            db.execute("PRAGMA auto_vacuum=INCREMENTAL")         # only effective on a new file
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("PRAGMA journal_size_limit=8388608")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS ops(id INTEGER PRIMARY KEY, kind TEXT, title TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS items(op INTEGER, rel TEXT, size INTEGER, mtime REAL, data BLOB,
                    img_mode TEXT, img_pos INTEGER, img_sha BLOB, img_ver TEXT, PRIMARY KEY(op, rel));
                CREATE TABLE IF NOT EXISTS blobs(sha BLOB PRIMARY KEY, data BLOB, refs INTEGER) WITHOUT ROWID;
            """)
            self._db = db
        return self._db

    def close(self):
        with self._lock:
            if self._db is not None:
                try:
                    self._db.execute("PRAGMA incremental_vacuum")
                    self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    pass
                self._db.close()
                self._db = None

    def exists(self) -> bool:
        return self.path.is_file()

    # ------------------------------------------------------------------ paths
    def rel(self, path) -> str:
        p = os.path.abspath(str(path))
        r = os.path.abspath(str(self.root))
        try:
            rel = os.path.relpath(p, r)
        except ValueError:                  # other drive (Windows)
            return p
        return p if rel.startswith("..") else rel.replace(os.sep, "/")

    def abs(self, rel: str) -> Path:
        return Path(rel) if os.path.isabs(rel) else self.root / rel

    # ------------------------------------------------------------------ write
    def begin(self, kind: str, title: str = "") -> int:
        with self._lock:
            cur = self.db.execute("INSERT INTO ops(kind, title, created) VALUES (?,?,?)", (kind, title, time.time()))
            return int(cur.lastrowid)

    def has(self, op: int, path) -> bool:
        with self._lock:
            return self.db.execute("SELECT 1 FROM items WHERE op=? AND rel=?", (op, self.rel(path))).fetchone() \
                is not None

    def save(self, op: int, path, target=None, exact: bool | None = None) -> bool:
        """Back up `path` (once per operation). `target`: where the file belongs when `path` is a copy elsewhere
        (legacy import). Returns False when it was already saved or does not exist."""
        src = Path(path)
        target = Path(target) if target is not None else src
        rel = self.rel(target)
        with self._lock:
            if self.db.execute("SELECT 1 FROM items WHERE op=? AND rel=?", (op, rel)).fetchone():
                return False
            try:
                raw = src.read_bytes()
                mtime = src.stat().st_mtime
            except OSError:
                return False
            row = _pack(raw, target, EXACT_IMAGE_DATA if exact is None else exact)
            data, mode, pos, sha, ver, blob = row
            db = self.db
            db.execute("BEGIN")
            try:
                if blob is not None:
                    db.execute("INSERT INTO blobs(sha, data, refs) VALUES (?,?,1) "
                               "ON CONFLICT(sha) DO UPDATE SET refs = refs + 1", (sha, blob))
                db.execute("INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?)",
                           (op, rel, len(raw), mtime, data, mode, pos, sha, ver))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            return True

    # ------------------------------------------------------------------ read / restore
    def files(self, op: int) -> list[str]:
        with self._lock:
            return [r[0] for r in self.db.execute("SELECT rel FROM items WHERE op=? ORDER BY rel", (op,))]

    def original_bytes(self, op: int, path=None, rel: str | None = None) -> tuple[bytes, str] | None:
        rel = rel if rel is not None else self.rel(path)
        with self._lock:
            row = self.db.execute("SELECT data, img_mode, img_pos, img_sha, img_ver FROM items WHERE op=? AND rel=?",
                                  (op, rel)).fetchone()
            if row is None:
                return None
            blob = None
            if row[1] == "blob":
                b = self.db.execute("SELECT data FROM blobs WHERE sha=?", (row[3],)).fetchone()
                blob = b[0] if b else None
        return _unpack(row, self.abs(rel), blob)

    def restore(self, op: int, path, dest=None) -> str:
        """Write the original back (atomically) to `dest` (default: where it came from)."""
        got = self.original_bytes(op, path)
        if got is None:
            return "missing"
        data, how = got
        dest = Path(dest) if dest is not None else Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".aioqc_undo")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        return how

    def export(self, op: int, folder, progress=None) -> int:
        """Write the originals of one operation into `folder` (same relative layout) for manual inspection."""
        folder = Path(folder)
        rels = self.files(op)
        for i, rel in enumerate(rels):
            got = self.original_bytes(op, rel=rel)
            if got is None:
                continue
            out = folder / ("_external/" + Path(rel).name if os.path.isabs(rel) else rel)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(got[0])
            if progress:
                progress(i + 1, len(rels))
        return len(rels)

    # ------------------------------------------------------------------ management
    def ops(self) -> list[dict]:
        if not self.exists():
            return []
        with self._lock:
            rows = self.db.execute("""
                SELECT o.id, o.kind, o.title, o.created, COUNT(i.rel), COALESCE(SUM(i.size), 0),
                       COALESCE(SUM(LENGTH(i.data)), 0)
                FROM ops o LEFT JOIN items i ON i.op = o.id GROUP BY o.id ORDER BY o.created DESC""").fetchall()
            blob_bytes = dict(self.db.execute("""
                SELECT i.op, SUM(LENGTH(b.data) * 1.0 / b.refs) FROM items i JOIN blobs b ON b.sha = i.img_sha
                WHERE i.img_mode = 'blob' GROUP BY i.op""").fetchall())
        return [{"id": r[0], "kind": r[1], "title": r[2], "created": r[3], "files": r[4], "original_bytes": r[5],
                 "stored_bytes": int(r[6] + blob_bytes.get(r[0], 0))} for r in rows]

    def delete(self, op: int):
        with self._lock:
            db = self.db
            db.execute("BEGIN")
            for (sha,) in db.execute("SELECT img_sha FROM items WHERE op=? AND img_mode='blob'", (op,)).fetchall():
                db.execute("UPDATE blobs SET refs = refs - 1 WHERE sha=?", (sha,))
            db.execute("DELETE FROM blobs WHERE refs <= 0")
            db.execute("DELETE FROM items WHERE op=?", (op,))
            db.execute("DELETE FROM ops WHERE id=?", (op,))
            db.execute("COMMIT")
            db.execute("PRAGMA incremental_vacuum")

    def prune(self, older_than_days: float, keep: set[int] = frozenset()) -> int:
        """Delete operations older than N days (never the ones in `keep`, e.g. an open undo). -> count."""
        if older_than_days <= 0 or not self.exists():
            return 0
        cutoff = time.time() - older_than_days * 86400
        with self._lock:
            old = [r[0] for r in self.db.execute("SELECT id FROM ops WHERE created < ?", (cutoff,))]
        n = 0
        for op in old:
            if op not in keep:
                self.delete(op)
                n += 1
        return n

    def disk_size(self) -> int:
        return sum(p.stat().st_size for p in self.path.parent.glob(DB_NAME + "*") if p.is_file())

    def compact(self):
        with self._lock:
            self.db.execute("PRAGMA incremental_vacuum")
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def import_folder(self, folder, kind: str, title: str = "", created: float | None = None,
                      remove: bool = True) -> int:
        """Pull a legacy backup folder (<.aio_qc>/backups/<op>/<rel path>) into the store; the folder is deleted
        after every file was stored. -> files imported."""
        folder = Path(folder)
        op = self.begin(kind, title or folder.name)
        if created is not None:
            with self._lock:
                self.db.execute("UPDATE ops SET created=? WHERE id=?", (created, op))
        n = 0
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                rel = p.relative_to(folder)
                target = self.root / rel if rel.parts[0] != "_external" else p
                self.save(op, p, target=target)
                n += 1
        if remove:
            shutil.rmtree(folder, ignore_errors=True)
        return n


# ---------------------------------------------------------------------- packing
def _pack(raw: bytes, target: Path, exact: bool):
    """-> (zlib data, img_mode, img_pos, img_sha, img_ver, blob bytes | None)"""
    if target.suffix.lower() == ".json" and len(raw) > _MIN_INLINE * 2:
        try:
            text = raw.decode("utf-8")
            d = json.loads(text)
        except (UnicodeDecodeError, ValueError):
            d = None
        b64 = d.get(IMG_KEY) if isinstance(d, dict) else None
        if isinstance(b64, str) and len(b64) >= _MIN_INLINE:
            pos = text.find(b64)
            ok = pos > 0 and text[pos - 1] == '"' and text.find(b64, pos + 1) < 0
            decoded = None
            if ok:
                try:
                    decoded = base64.b64decode(b64, validate=True)
                    ok = base64.b64encode(decoded).decode("ascii") == b64    # restorable exactly
                except (ValueError, TypeError):
                    ok = False
            if ok:
                stripped = text[:pos] + text[pos + len(b64):]
                sha = _sha(decoded)
                ver = str(d.get("version") or "")
                img = image_file_of(target, d)
                mode, blob = "regen", None
                if img is not None:
                    try:
                        if _sha(img.read_bytes()) == sha:
                            mode = "file"
                    except OSError:
                        img = None
                if mode != "file" and (exact or img is None):
                    mode, blob = "blob", decoded
                return zlib.compress(stripped.encode("utf-8"), 6), mode, pos, sha, ver, blob
    return zlib.compress(raw, 6), None, None, None, None, None


def _unpack(row, target: Path, blob: bytes | None) -> tuple[bytes, str]:
    data, mode, pos, sha, ver = row
    raw = zlib.decompress(data)
    if not mode:
        return raw, "exact"
    text = raw.decode("utf-8")
    b64, how = None, "exact"
    if mode == "blob" and blob is not None:
        b64 = base64.b64encode(blob).decode("ascii")
    else:
        try:
            img = image_file_of(target, json.loads(text))
        except ValueError:
            img = None
        if img is not None:
            try:
                file_bytes = img.read_bytes()
                if _sha(file_bytes) == sha:
                    b64 = base64.b64encode(file_bytes).decode("ascii")
                else:
                    b64 = encode_b64(img, ver or "5")
                    how = "exact" if _sha(base64.b64decode(b64)) == sha else "regen"
            except Exception:
                b64 = None
    if b64 is None:                               # nothing to rebuild from: imageData becomes null
        return (text[:pos - 1] + "null" + text[pos + 1:]).encode("utf-8"), "no-image"
    return (text[:pos] + b64 + text[pos:]).encode("utf-8"), how


# ---------------------------------------------------------------------- legacy folders
def legacy_folders(root) -> list[Path]:
    """Backup folders written by AIO QC ≤ 3.3 (full copies)."""
    base = Path(root) / ".aio_qc" / "backups"
    return sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


def folder_bytes(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def migrate_legacy(root, progress=None) -> tuple[int, int]:
    """Import every legacy backup folder into backups.sqlite. -> (folders, bytes freed)"""
    store = BackupStore(root)
    folders = legacy_folders(root)
    freed = 0
    for i, f in enumerate(folders):
        size = folder_bytes(f)
        name = f.name
        kind = name.split("_")[0] if not name.startswith("class_removal") else "class_removal"
        try:
            created = f.stat().st_mtime
        except OSError:
            created = None
        store.import_folder(f, kind, name, created=created)
        freed += size
        if progress:
            progress(i + 1, len(folders))
    if folders:
        try:
            (Path(root) / ".aio_qc" / "backups").rmdir()
        except OSError:
            pass
    store.close()
    return len(folders), freed
