# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Shared pieces of the Data Prep tools (Qt-free): scanning, JSON / imageData helpers, content hashing, unique names,
and the run log every tool writes so a run can be undone.

RunLog: every file operation of a run is recorded twice —
    <log dir>/<tool>_<stamp>.csv     for people (action, from, to, note, status)
    <log dir>/<tool>_<stamp>.jsonl   for Undo: move · copy / create (with the result's sha256) · move_rewrite (the
                                     source JSON's original bytes kept, zlib-compressed, under …_originals/) ·
                                     rewrite (a file changed in place, its original bytes kept the same way)
undo_run(jsonl) reverses it newest-first: moved files go back, created files are removed only if they are unchanged
since the run, rewritten files get their original bytes back (in-place rewrites only if untouched since). Nothing is overwritten on the way back.
"""
from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import json
import os
import re
import shutil
import threading
import time
import zlib
from collections import defaultdict
from pathlib import Path

try:
    import orjson  # type: ignore
except ImportError:  # pragma: no cover
    orjson = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".avif", ".jfif"}
SAME_FORMAT = {".jpeg": ".jpg", ".tiff": ".tif", ".jfif": ".jpg"}
SAMPLE = 64 * 1024
LOG_DIR = "_aioqc_prep"


class Cancelled(Exception):
    pass


def norm(p) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))


def inside(child, parent) -> bool:
    c, p = norm(child), norm(parent)
    return c == p or c.startswith(p.rstrip(os.sep) + os.sep)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def stem(p) -> str:
    return os.path.splitext(os.path.basename(str(p)))[0]


def ext(p) -> str:
    return os.path.splitext(str(p))[1]


def ip_basename(ip: str) -> str:
    return ip.replace("\\", "/").rstrip("/").split("/")[-1]


# ---------------------------------------------------------------------------------------------------- scanning
def scan(roots, skip=(), images=True, jsons=True, zips=False, cancel=None):
    """Recursive scan of one or more folders -> (images, jsons, zips, other Counter, sizes {path: bytes}).
    `skip`: folders never entered (outputs of the tool itself). Hidden folders and our log folders are skipped."""
    from collections import Counter
    skip_n = [norm(s) for s in skip if s]
    imgs, js, zs, sizes, other = [], [], [], {}, Counter()
    for root in roots:
        stack = [str(root)]
        while stack:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            d = stack.pop()
            try:
                it = os.scandir(d)
            except OSError:
                continue
            with it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if e.name.startswith(".") or e.name == LOG_DIR:
                                continue
                            if not any(inside(e.path, s) for s in skip_n):
                                stack.append(e.path)
                            continue
                        if not e.is_file():
                            continue
                        x = os.path.splitext(e.name)[1].lower()
                        if images and x in IMAGE_EXTS:
                            imgs.append(e.path)
                            sizes[e.path] = e.stat().st_size
                        elif jsons and x == ".json" and not e.name.startswith("_"):
                            js.append(e.path)
                        elif zips and x == ".zip":
                            zs.append(e.path)
                            sizes[e.path] = e.stat().st_size
                        elif not e.name.endswith((".tmp", ".csv", ".jsonl")):
                            other[x or "(no ext)"] += 1
                    except OSError:
                        continue
    imgs.sort()
    js.sort()
    zs.sort()
    return imgs, js, zs, other, sizes


# ---------------------------------------------------------------------------------------------------- JSON
def read_json(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    if orjson is not None:
        try:
            return orjson.loads(raw)
        except orjson.JSONDecodeError:
            pass
    return json.loads(raw.decode("utf-8"))


def dump_json(data) -> bytes:
    """Like LabelMe writes it (2-space indent, UTF-8)."""
    if orjson is not None:
        try:
            return orjson.dumps(data, option=orjson.OPT_INDENT_2)
        except TypeError:
            pass
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def write_atomic(path, raw: bytes):
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".aioqc_tmp"
    with open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, path)


def label_digest(data: dict) -> str:
    """Fingerprint of an annotation without imagePath / imageData — to REPORT whether two copies carry the same
    labels (never used to decide what is a duplicate)."""
    lab = {k: v for k, v in data.items() if k not in ("imagePath", "imageData")}
    s = json.dumps(lab, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------------- imageData
_B64_WS = re.compile(r"\s+")


def sniff(head: bytes) -> str | None:
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head.startswith(b"BM"):
        return ".bmp"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return ".tif"
    return None


def _payload(s: str) -> str:
    if s.startswith("data:"):
        c = s.find(",")
        if c != -1:
            return s[c + 1:]
    return s


def peek_image_data(s) -> str | None:
    """Image extension from the first bytes of base64 imageData (no full decode)."""
    if not isinstance(s, str) or not s.strip():
        return None
    head = _B64_WS.sub("", _payload(s[:240]))[:32]
    head = head[: len(head) // 4 * 4]
    try:
        return sniff(base64.b64decode(head))
    except (binascii.Error, ValueError):
        return None


def b64_size(s: str) -> int:
    """Exact decoded size of base64 text without decoding it."""
    p = _payload(s)
    n = len(p) - sum(p.count(c) for c in ("\n", "\r", " ", "\t"))
    tail = p.rstrip()
    return n * 3 // 4 - (2 if tail.endswith("==") else 1 if tail.endswith("=") else 0)


def b64_bytes(s: str) -> bytes:
    p = _payload(s)
    try:
        return base64.b64decode(p, validate=True)
    except (binascii.Error, ValueError):
        try:
            return base64.b64decode(_B64_WS.sub("", p), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"invalid base64 ({exc})")


def decode_image_data(s: str) -> tuple[bytes, str]:
    """imageData -> (bytes, ext), fully verified (every pixel decoded). Raises ValueError."""
    raw = b64_bytes(s)
    x = sniff(raw[:16])
    if not raw or x is None:
        raise ValueError("decoded data is not a recognised image")
    ok, why = verify_bytes(raw)
    if not ok:
        raise ValueError(f"image data is corrupt ({why})")
    return raw, x


# ---------------------------------------------------------------------------------------------------- image checks
def verify_bytes(raw: bytes) -> tuple[bool, str]:
    """Full decode of image bytes (Pillow if present, else OpenCV) + truncated-JPEG check."""
    if raw[:3] == b"\xff\xd8\xff" and b"\xff\xd9" not in raw[-4096:]:
        return False, "truncated JPEG (no end marker)"
    try:
        import io

        from PIL import Image
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            if im.size[0] < 1 or im.size[1] < 1:
                return False, "empty image"
        return True, ""
    except ImportError:
        pass
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:160]
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        return (img is not None and img.size > 0), ("" if img is not None else "cannot be decoded")
    except Exception as exc:
        return False, str(exc)[:160]


def verify_file(path) -> tuple[bool, str, tuple | None]:
    """(ok, why, (w, h)) after a full decode."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        return False, f"unreadable ({exc.strerror})", None
    if not raw:
        return False, "empty file (0 bytes)", None
    if sniff(raw[:16]) is None and ext(path).lower() not in (".avif",):
        return False, "not an image (unknown file signature)", None
    ok, why = verify_bytes(raw)
    if not ok:
        return False, why, None
    try:
        from .. import io as cio
        return True, "", cio.image_size(path)
    except Exception:
        return True, "", None


# ---------------------------------------------------------------------------------------------------- hashing
def file_sha(path, chunk=1 << 20) -> str | None:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def sample_sha(path, size) -> str | None:
    """SHA-256 of 64 KB from the start, middle and end (the whole file when small) — only rules images OUT."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            if size <= 3 * SAMPLE:
                h.update(f.read())
            else:
                for off in (0, size // 2 - SAMPLE // 2, size - SAMPLE):
                    f.seek(off)
                    h.update(f.read(SAMPLE))
    except OSError:
        return None
    return h.hexdigest()


def exact_groups(items, size_of, path_of, digest_of=None, workers=16, cancel=None, progress=None) -> list[list]:
    """Groups of byte-identical items. size_of(it) -> bytes; path_of(it) -> file path, or None for content that is
    not a plain file (embedded imageData), which digest_of(it) hashes. Size first, then a 3-sample fingerprint, then
    the full hash — only collisions are ever read."""
    from concurrent.futures import ThreadPoolExecutor
    by_size = defaultdict(list)
    for it in items:
        s = size_of(it)
        if s is not None:
            by_size[s].append(it)
    same = [(s, it) for s, g in by_size.items() if len(g) > 1 for it in g]
    if not same:
        return []
    total = len(same)
    done = [0]
    lock = threading.Lock()

    def tick():
        with lock:
            done[0] += 1
            if progress is not None and done[0] % 50 == 0:
                progress(done[0], total)

    def quick(si):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        s, it = si
        p = path_of(it)
        r = sample_sha(p, s) if p is not None else (digest_of(it) if digest_of else None)
        tick()
        return r
    with ThreadPoolExecutor(workers) as ex:
        qd = list(ex.map(quick, same))
    by_q = defaultdict(list)
    for (s, it), d in zip(same, qd):
        if d is not None:
            by_q[(s, d)].append(it)
    cands = [(k[0], k[1], it) for k, g in by_q.items() if len(g) > 1 for it in g]

    def full(sdi):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        s, d, it = sdi
        p = path_of(it)
        if p is None or s <= 3 * SAMPLE:
            return d                        # already a full hash (embedded content / small file)
        return file_sha(p)
    with ThreadPoolExecutor(workers) as ex:
        fd = list(ex.map(full, cands))
    groups = defaultdict(list)
    for (s, _d, it), d in zip(cands, fd):
        if d is not None:
            groups[(s, d)].append(it)
    return [g for g in groups.values() if len(g) > 1]


# ---------------------------------------------------------------------------------------------------- names
class Names:
    """Unique (case-insensitive) stems inside one output folder: a, a_1, a_2 …"""

    def __init__(self, folder=None):
        self.taken = set()
        self.renamed = 0
        if folder and os.path.isdir(folder):
            with os.scandir(folder) as it:
                for e in it:
                    if e.is_file():
                        self.taken.add(stem(e.name).lower())

    def get(self, s: str, sep="_") -> str:
        s = s or "unnamed"
        cand, n = s, 1
        while cand.lower() in self.taken:
            cand, n = f"{s}{sep}{n}", n + 1
        self.taken.add(cand.lower())
        self.renamed += cand != s
        return cand


def unique_path(p: Path, taken: set | None = None) -> Path:
    """p, or p with _1, _2 … so it neither exists nor is reserved."""
    p = Path(p)
    cand, n = p, 1
    while cand.exists() or (taken is not None and norm(cand) in taken):
        cand, n = p.with_name(f"{p.stem}_{n}{p.suffix}"), n + 1
    if taken is not None:
        taken.add(norm(cand))
    return cand


# ---------------------------------------------------------------------------------------------------- run log
class RunLog:
    def __init__(self, log_dir, tool: str, keep_originals: bool = True):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.base = self.dir / f"{tool}_{stamp}"
        n = 1
        while self.base.with_suffix(".jsonl").exists():
            self.base = self.dir / f"{tool}_{stamp}_{n}"
            n += 1
        self.csv_path = self.base.with_suffix(".csv")
        self.jsonl_path = self.base.with_suffix(".jsonl")
        self.orig_dir = self.dir / (self.base.name + "_originals")
        self.keep_originals = keep_originals
        self._lock = threading.Lock()
        self._csv_f = open(self.csv_path, "w", newline="", encoding="utf-8-sig")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(["action", "from", "to", "note", "status"])
        self._j = open(self.jsonl_path, "w", encoding="utf-8")
        self._n = 0
        self.write({"op": "run", "tool": tool, "when": time.time()})

    def write(self, rec: dict):
        with self._lock:
            self._j.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def row(self, action, src, dst, note="", status="ok"):
        with self._lock:
            self._csv.writerow([action, src or "", dst or "", note, status])

    def keep_original(self, src) -> str | None:
        """Store a file's bytes before it is rewritten / removed (compressed). -> stored path."""
        if not self.keep_originals:
            return None
        with self._lock:
            self._n += 1
            n = self._n
        self.orig_dir.mkdir(parents=True, exist_ok=True)
        dst = self.orig_dir / f"{n:07d}.z"
        with open(src, "rb") as f:
            dst.write_bytes(zlib.compress(f.read(), 6))
        return str(dst)

    # high-level records
    def moved(self, src, dst, note="", action="move"):
        self.write({"op": "move", "src": str(src), "dst": str(dst)})
        self.row(action, src, dst, note)

    def created(self, dst, note="", src="", action="create"):
        self.write({"op": "create", "dst": str(dst), "sha": file_sha(dst)})
        self.row(action, src, dst, note)

    def moved_rewritten(self, src, dst, original, note=""):
        """src (a JSON) is gone, dst holds a rewritten copy; `original` = keep_original(src) result."""
        self.write({"op": "move_rewrite", "src": str(src), "dst": str(dst), "orig": original, "sha": file_sha(dst)})
        self.row("rewrite", src, dst, note)

    def rewritten(self, path, original, note=""):
        """`path` was rewritten in place; `original` = keep_original(path) result (taken BEFORE the write)."""
        self.write({"op": "rewrite", "src": str(path), "orig": original, "sha": file_sha(path)})
        self.row("rewrite", path, path, note)

    def deleted(self, src, original, note=""):
        self.write({"op": "delete", "src": str(src), "orig": original})
        self.row("delete", src, "", note)

    def error(self, action, src, dst, msg):
        self.row(action, src, dst, "", f"ERROR: {msg}")

    def close(self):
        with self._lock:
            for f in (self._csv_f, self._j):
                try:
                    f.close()
                except Exception:
                    pass


def list_runs(folder) -> list[Path]:
    d = Path(folder) / LOG_DIR
    return sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True) if d.is_dir() else []


def undo_run(jsonl, progress=None) -> dict:
    """Reverse a logged run (newest operation first). Never overwrites: a file that came back / changed since the
    run is reported and left alone."""
    recs = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    st = {"restored": 0, "removed": 0, "skipped": [], "errors": []}
    ops = [r for r in recs if r.get("op") != "run"]
    for i, r in enumerate(reversed(ops), 1):
        op = r.get("op")
        try:
            if op == "move":
                src, dst = r["src"], r["dst"]
                if not os.path.exists(dst):
                    st["skipped"].append(f"{dst}: no longer there")
                elif os.path.exists(src):
                    st["skipped"].append(f"{src}: a file is back at the original place")
                else:
                    os.makedirs(os.path.dirname(src), exist_ok=True)
                    shutil.move(dst, src)
                    st["restored"] += 1
            elif op == "create":
                dst = r["dst"]
                if not os.path.exists(dst):
                    continue
                if r.get("sha") and file_sha(dst) != r["sha"]:
                    st["skipped"].append(f"{dst}: changed after the run — kept")
                    continue
                os.remove(dst)
                st["removed"] += 1
            elif op == "create_dir":
                d = r["dst"]
                if os.path.isdir(d):
                    shutil.rmtree(d)
                    st["removed"] += 1
            elif op == "rewrite":                       # rewritten in place: back only if untouched since
                src, orig = r["src"], r.get("orig")
                if orig is None or not os.path.exists(orig):
                    st["skipped"].append(f"{src}: original not kept")
                elif os.path.exists(src) and r.get("sha") and file_sha(src) != r["sha"]:
                    st["skipped"].append(f"{src}: changed after the run — kept")
                else:
                    write_atomic(src, zlib.decompress(Path(orig).read_bytes()))
                    st["restored"] += 1
            elif op in ("move_rewrite", "delete"):
                src, orig = r["src"], r.get("orig")
                if orig is None or not os.path.exists(orig):
                    st["skipped"].append(f"{src}: original not kept")
                    continue
                if os.path.exists(src):
                    st["skipped"].append(f"{src}: a file is back at the original place")
                    continue
                write_atomic(src, zlib.decompress(Path(orig).read_bytes()))
                st["restored"] += 1
                dst = r.get("dst")
                if dst and os.path.exists(dst):
                    if r.get("sha") and file_sha(dst) != r["sha"]:
                        st["skipped"].append(f"{dst}: changed after the run — kept (the original is back too)")
                    else:
                        os.remove(dst)
        except Exception as exc:
            st["errors"].append(f"{r.get('src') or r.get('dst')}: {exc}")
        if progress is not None and i % 50 == 0:
            progress(i, len(ops))
    try:
        Path(jsonl).rename(Path(jsonl).with_suffix(".undone.jsonl"))
    except OSError:
        pass
    return st


def remove_empty_dirs(dirs, root, protected=()) -> int:
    """Remove folders (and their parents up to `root`) that are now empty."""
    rn = norm(root)
    cand = set()
    for d in dirs:
        while True:
            dn = norm(d)
            if dn == rn or not dn.startswith(rn + os.sep) or dn in cand:
                break
            cand.add(dn)
            d = os.path.dirname(d)
    n = 0
    for d in sorted(cand, key=len, reverse=True):
        if any(inside(d, p) or inside(p, d) for p in protected):
            continue
        try:
            os.rmdir(d)
            n += 1
        except OSError:
            pass
    return n
