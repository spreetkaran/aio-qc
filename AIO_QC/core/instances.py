# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Shape instances for the Crop Gallery, stored column-wise so millions of instances stay cheap (Qt-free).

    store = InstanceStore()
    store.add_rows(rows)                      # rows from collect_file / iter_collect (streaming)
    order = store.sorted_ids("Smallest first", alive_only=True)   # numpy argsort, no Python objects per instance
    inst = store.get(iid)                     # a small Instance object, only for tiles on screen / selections

Memory: ~40 bytes per instance + its outline (8 bytes per point) instead of a Python object with lists (~1 KB).
Collection: pure-Python outline maths (no numpy per shape), files without the label are skipped with the label
index, and big collections run in a process pool (spawn; falls back to one thread whenever that is not possible).
"""
from __future__ import annotations

import math
import os
import random
from array import array
from dataclasses import dataclass

import numpy as np

SORTS = ["File order", "Smallest first", "Largest first", "Narrowest first", "Widest first", "Random"]
POOL_MIN_FILES = 40_000            # below this a process pool costs more than it saves (start-up ~1–2 s)
CHUNK = 2_000


@dataclass
class Instance:
    key: str
    index: int                    # shape index in the file
    label: str
    image: object                 # path str | None
    points: list                  # outline in annotation coordinates [(x, y), …]
    normalized: bool
    area: float                   # bbox area in annotation coordinates (normalised for YOLO) — sorting only
    aspect: float
    iid: int = -1                 # id in the InstanceStore


# ---------------------------------------------------------------------------- geometry (pure Python, fast)
def outline(shape: dict) -> list[float] | None:
    """Flat [x0, y0, x1, y1, …] outline: rectangle -> 4 corners, circle -> 24-gon, others as drawn."""
    pts = shape.get("points") or []
    try:
        flat = [float(c) for p in pts for c in (p[0], p[1])]
    except (TypeError, ValueError, IndexError):
        return None
    if not flat or not all(math.isfinite(v) for v in flat):
        return None
    st = shape.get("shape_type") or "polygon"
    if st == "rectangle" and len(flat) == 4:
        x1, x2 = sorted((flat[0], flat[2]))
        y1, y2 = sorted((flat[1], flat[3]))
        return [x1, y1, x2, y1, x2, y2, x1, y2]
    if st == "circle" and len(flat) == 4:
        cx, cy, px, py = flat
        r = math.hypot(px - cx, py - cy)
        out = []
        for i in range(24):
            t = 2 * math.pi * i / 24
            out += [cx + r * math.cos(t), cy + r * math.sin(t)]
        return out
    return flat


def collect_file(adapter, sample, label=None, indices=None) -> list[tuple]:
    """Rows (key, index, label, image, normalized, flat outline, area, aspect) of `label` (or exactly `indices`)."""
    ann = adapter.load(sample, light=True)
    image = ann.image_path or sample.image_path
    image = str(image) if image is not None else None
    out = []
    for i, sh in enumerate(ann.shapes):
        lab = sh.get("label")
        if indices is not None:
            if i not in indices:
                continue
        elif lab != label:
            continue
        flat = outline(sh)
        if not flat:
            continue
        xs, ys = flat[0::2], flat[1::2]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        out.append((sample.key, i, lab if indices is not None else label, image, bool(ann.normalized), flat,
                    w * h, (w / h) if h > 0 else 0.0))
    return out


def _chunk_job(adapter, samples, label):
    """Process-pool job: one chunk of files -> (rows, errors)."""
    rows, errors = [], []
    for s in samples:
        try:
            rows.extend(collect_file(adapter, s, label))
        except Exception as exc:
            errors.append(f"{s.name}: {exc}")
    return rows, errors


def iter_collect(adapter, samples, label, stop=lambda: False, workers: int | None = None):
    """Yield (files done, rows, errors) while collecting; big jobs use a process pool."""
    n = len(samples)
    workers = workers if workers is not None else max(1, min(8, (os.cpu_count() or 2) - 1))
    if n >= POOL_MIN_FILES and workers > 1:
        try:
            yield from _pool_collect(adapter, samples, label, stop, workers)
            return
        except _PoolUnavailable as exc:
            done = exc.done
            samples = [s for i, s in enumerate(samples) if i // CHUNK not in done]
    rows, errors, k = [], [], 0
    for k, s in enumerate(samples, 1):
        if stop():
            return
        try:
            rows.extend(collect_file(adapter, s, label))
        except Exception as exc:
            errors.append(f"{s.name}: {exc}")
        if len(rows) >= 400 or k % 500 == 0:
            yield k, rows, errors
            rows, errors = [], []
    yield k, rows, errors


class _PoolUnavailable(Exception):
    def __init__(self, done: set):
        super().__init__("process pool unavailable")
        self.done = done


def _pool_collect(adapter, samples, label, stop, workers):
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    chunks = [samples[i:i + CHUNK] for i in range(0, len(samples), CHUNK)]
    done_chunks: set[int] = set()
    files = 0
    try:
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    except Exception:
        raise _PoolUnavailable(done_chunks)
    try:
        pending, nxt = {}, 0
        while nxt < len(chunks) or pending:
            while nxt < len(chunks) and len(pending) < workers * 2:      # bounded queue: memory stays flat
                pending[ex.submit(_chunk_job, adapter, chunks[nxt], label)] = nxt
                nxt += 1
            finished, _ = wait(list(pending), timeout=0.25, return_when=FIRST_COMPLETED)
            if stop():
                return
            for f in finished:
                ci = pending.pop(f)
                try:
                    rows, errors = f.result()
                except Exception:
                    if not done_chunks:          # the pool never worked (e.g. cannot spawn): fall back to a thread
                        raise _PoolUnavailable(done_chunks)
                    rows, errors = _chunk_job(adapter, chunks[ci], label)       # one failed chunk: do it here
                done_chunks.add(ci)
                files += len(chunks[ci])
                yield files, rows, errors
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------- columnar store
class InstanceStore:
    def __init__(self):
        self.keys: list[str] = []
        self.key_id: dict[str, int] = {}
        self.images: list = []
        self.norm = bytearray()
        self.rank = array("q")                  # file position in the dataset (for "File order")
        self.labels: list[str] = []
        self.label_id: dict[str, int] = {}
        self.kid = array("i")
        self.idx = array("i")
        self.lab = array("i")
        self.area = array("f")
        self.aspect = array("f")
        self.off = array("q", [0])              # outline of instance i: pts[off[i]:off[i+1]]
        self.pts = array("f")
        self.alive = bytearray()
        self.per_key = array("i")               # alive instances per file
        self.files = 0                          # files with at least one alive instance
        self.n_alive = 0
        self.rank_of = None                     # callable key -> dataset position (set by the owner)

    def __len__(self):
        return len(self.kid)

    def _kid(self, key: str, image, normalized: bool) -> int:
        k = self.key_id.get(key)
        if k is None:
            k = self.key_id[key] = len(self.keys)
            self.keys.append(key)
            self.images.append(image)
            self.norm.append(1 if normalized else 0)
            self.rank.append(self.rank_of(key) if self.rank_of else k)
            self.per_key.append(0)
        return k

    def _lid(self, label: str) -> int:
        i = self.label_id.get(label)
        if i is None:
            i = self.label_id[label] = len(self.labels)
            self.labels.append(label)
        return i

    def add_rows(self, rows) -> tuple[int, int]:
        """Append rows; -> (first id, last id + 1)."""
        start = len(self.kid)
        for key, index, label, image, normalized, flat, area, aspect in rows:
            k = self._kid(key, image, normalized)
            self.kid.append(k)
            self.idx.append(index)
            self.lab.append(self._lid(label))
            self.area.append(area)
            self.aspect.append(aspect)
            self.pts.extend(flat)
            self.off.append(len(self.pts))
            self.alive.append(1)
            if self.per_key[k] == 0:
                self.files += 1
            self.per_key[k] += 1
        self.n_alive += len(self.kid) - start
        return start, len(self.kid)

    def kill_keys(self, keys) -> int:
        """Mark every instance of these files dead (edited / removed files). -> killed"""
        kids = {self.key_id[k] for k in keys if k in self.key_id}
        if not kids:
            return 0
        kid = np.frombuffer(self.kid, dtype=np.int32)
        alive = np.frombuffer(self.alive, dtype=np.uint8)
        hit = np.nonzero(np.isin(kid, np.fromiter(kids, dtype=np.int32)) & (alive == 1))[0]
        for i in hit:
            self.alive[i] = 0
        for k in kids:
            if self.per_key[k]:
                self.files -= 1
            self.per_key[k] = 0
        self.n_alive -= len(hit)
        return len(hit)

    def kill_ids(self, ids) -> None:
        for i in ids:
            if self.alive[i]:
                self.alive[i] = 0
                k = self.kid[i]
                self.per_key[k] -= 1
                if self.per_key[k] == 0:
                    self.files -= 1
                self.n_alive -= 1

    def get(self, i: int) -> Instance:
        k = self.kid[i]
        flat = self.pts[self.off[i]:self.off[i + 1]]
        return Instance(self.keys[k], self.idx[i], self.labels[self.lab[i]], self.images[k],
                        list(zip(flat[0::2], flat[1::2])), bool(self.norm[k]), self.area[i], self.aspect[i], i)

    def key_of(self, i: int) -> str:
        return self.keys[self.kid[i]]

    def label_of(self, i: int) -> str:
        return self.labels[self.lab[i]]

    def alive_ids(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        a = np.frombuffer(self.alive, dtype=np.uint8)[start:stop]
        return np.nonzero(a)[0].astype(np.int64) + start

    def sorted_ids(self, mode: str, seed: int = 0) -> np.ndarray:
        ids = self.alive_ids()
        if not len(ids):
            return ids
        if mode == "File order":
            rank = np.frombuffer(self.rank, dtype=np.int64)[np.frombuffer(self.kid, dtype=np.int32)[ids]]
            idx = np.frombuffer(self.idx, dtype=np.int32)[ids]
            return ids[np.lexsort((idx, rank))]
        if mode in ("Smallest first", "Largest first"):
            v = np.frombuffer(self.area, dtype=np.float32)[ids]
            return ids[np.argsort(-v if mode == "Largest first" else v, kind="stable")]
        if mode in ("Narrowest first", "Widest first"):
            v = np.frombuffer(self.aspect, dtype=np.float32)[ids]
            return ids[np.argsort(-v if mode == "Widest first" else v, kind="stable")]
        if mode == "Random":
            rng = np.random.default_rng(seed or len(ids))
            return ids[rng.permutation(len(ids))]
        return ids

    def needs_compaction(self) -> bool:
        return len(self.kid) > 50_000 and self.n_alive < len(self.kid) * 0.6

    def compacted(self) -> tuple["InstanceStore", np.ndarray]:
        """A copy without dead instances, and old id -> new id (-1 = dropped)."""
        new = InstanceStore()
        new.rank_of = self.rank_of
        remap = np.full(len(self.kid), -1, dtype=np.int64)
        rows = []
        for i in self.alive_ids():
            k = self.kid[i]
            rows.append((self.keys[k], self.idx[i], self.labels[self.lab[i]], self.images[k], bool(self.norm[k]),
                         self.pts[self.off[i]:self.off[i + 1]], self.area[i], self.aspect[i]))
            remap[i] = len(rows) - 1
        new.add_rows(rows)
        return new, remap


def shuffle_seed() -> int:
    return random.randrange(1, 2 ** 31)
