# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Adaptive reading / writing for very large datasets (Qt-free).

Threads help when each file waits on storage (a network share, a USB disk, a cold HDD, multi-MB images to copy) and
hurt when it does not: reading a small JSON from a warm local disk is Python work under the GIL, and handing it to a
thread costs more than it saves. So both helpers start in the calling thread, time the first calls, and switch to a
pool (in chunks, a bounded number in flight) only when the files turn out to be slow.

    for item, result in ordered_map(load, samples, workers=4):    # results in order; exceptions returned, not raised
        ...
    placer = Placer(workers=4); placer.put(src, dst, fn); …; placer.close()   # fire-and-forget writes, counted
"""
from __future__ import annotations

import itertools
import os
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

SLOW_MS = 0.5            # a file that takes longer than this waits on storage (or is real work): worth a thread


def default_workers() -> int:
    return min(8, max(2, (os.cpu_count() or 2) + 2))


def _safe(fn, x):
    try:
        return fn(x)
    except Exception as exc:                            # noqa: BLE001 — returned to the caller, per item
        return exc


def ordered_map(fn, items, workers: int = 0, probe: int = 48, chunk: int = 64, window: int = 8,
                slow_ms: float = SLOW_MS):
    """Yield (item, fn(item) or its exception) in order. Sequential while fn is fast; a thread pool (chunks of
    `chunk`, `window` chunks in flight) once the first `probe` calls average more than `slow_ms`."""
    items = items if isinstance(items, (list, tuple)) else list(items)
    workers = workers or default_workers()
    n = len(items)
    spent, i = 0.0, 0
    while i < min(probe, n):
        t = time.perf_counter()
        r = _safe(fn, items[i])
        spent += time.perf_counter() - t
        yield items[i], r
        i += 1
    if i >= n:
        return
    if workers <= 1 or spent / i * 1000 < slow_ms or n - i < 2 * chunk:
        for x in items[i:]:
            yield x, _safe(fn, x)
        return

    def many(part):
        return [_safe(fn, x) for x in part]
    ex = ThreadPoolExecutor(workers, thread_name_prefix="aioqc-read")
    q = deque()
    try:
        parts = (items[k:k + chunk] for k in range(i, n, chunk))
        for part in itertools.islice(parts, window):
            q.append((part, ex.submit(many, part)))
        while q:
            part, f = q.popleft()
            nxt = next(parts, None)
            if nxt is not None:
                q.append((nxt, ex.submit(many, nxt)))
            yield from zip(part, f.result())
    finally:
        for _p, f in q:
            f.cancel()
        ex.shutdown(wait=True)


class Placer:
    """Writes (link / copy …) done in the calling thread while they are quick, by a pool once they are slow (copies
    of large images, a network target). put() returns at once; close() waits. fn returns a label to count
    (e.g. "link" / "copy"); an OSError is collected in .errors."""

    def __init__(self, workers: int = 0, probe: int = 32, batch: int = 16, slow_ms: float = SLOW_MS):
        self.workers = workers or default_workers()
        self.probe, self.batch, self.slow_ms = probe, batch, slow_ms
        self.ex = None
        self.q = deque()
        self.pending = []
        self.counts = Counter()
        self.errors = []
        self._n, self._spent = 0, 0.0

    def _run(self, jobs):
        out = []
        for fn, args, name in jobs:
            try:
                out.append((fn(*args), None))
            except OSError as exc:
                out.append((None, f"{name}: could not be written ({exc})"))
        return out

    def _take(self, res):
        for label, err in res:
            if err:
                self.errors.append(err)
            else:
                self.counts[label] += 1

    def put(self, fn, *args, name: str = ""):
        job = (fn, args, name or str(args[-1] if args else ""))
        if self.ex is None:
            t = time.perf_counter()
            self._take(self._run([job]))
            self._spent += time.perf_counter() - t
            self._n += 1
            if self._n == self.probe and self.workers > 1 and self._spent / self._n * 1000 >= self.slow_ms:
                self.ex = ThreadPoolExecutor(self.workers, thread_name_prefix="aioqc-write")
            return
        self.pending.append(job)
        if len(self.pending) >= self.batch:
            self.q.append(self.ex.submit(self._run, self.pending))
            self.pending = []
            if len(self.q) > 4 * self.workers:
                self._take(self.q.popleft().result())

    def close(self):
        if self.pending:
            if self.ex is not None:
                self.q.append(self.ex.submit(self._run, self.pending))
            else:
                self._take(self._run(self.pending))
            self.pending = []
        while self.q:
            self._take(self.q.popleft().result())
        if self.ex is not None:
            self.ex.shutdown(wait=True)
            self.ex = None

    @property
    def threaded(self) -> bool:
        return self.ex is not None
