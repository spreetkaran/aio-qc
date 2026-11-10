# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Streaming detection analysis for video review - Qt-free; one worker thread per video.

The player shows saved detections for the exact frame on screen; this keeps them ready ahead of the playhead:

  * Look-ahead: frames from the playhead to `lead_s` seconds ahead are analysed first, in batches, every model on the
    same decoded frame (decoded once, CPU only, with the player's own decoder backend so frame numbers match).
  * Modes:
      best      every frame; the player waits (buffers) when the playhead reaches a frame not analysed yet
      balanced  (default) the frame step adapts to the measured speed so the look-ahead never runs dry; frames in
                between are interpolated from the analysed frames on both sides; idle time backfills them
      fixed     every `stride`-th frame (+ interpolation), then backfill
  * Priority: a paused frame that is not analysed yet is analysed next (request_exact); a seek moves the look-ahead.
  * Everything is saved in Testing's prediction cache (same model keys, video keys '<video key>#<frame>'), so a
    replay, Testing's video analysis or the annotated export never run the same frame twice, and a stopped analysis
    resumes where it stopped.
  * Failure isolation: a model that can't load or fails keeps its error (status()); the other models go on. A model
    running out of GPU memory retries with smaller batches.

    a = StreamAnalyser(specs, path, cache_path, mode="balanced")
    a.start(); a.set_playhead(idx, playing=True, speed=1.0)
    per_model, how = a.detections(idx)        # {model key: (n, 6) array}, "exact" | "interpolated" | "held" | None
    a.status()                                # stride, coverage, speed, ahead, phase, errors
    a.stop()
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

MODES = ("balanced", "best", "fixed")
MAX_STRIDE = 8


# ------------------------------------------------------------------ interpolation (pure)
def _iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def interpolate(da, db, t, min_iou=0.2):
    """Boxes between two analysed frames: t in [0, 1] from da (at 0) to db (at 1).
    Same-class boxes are paired greedily - overlapping pairs first (IoU), then a fast mover whose centre moved less
    than its own diagonal - and their corners / confidence blended; an unpaired box is shown only on its own half of
    the gap (it appears or disappears in the middle)."""
    da = np.asarray(da, np.float32).reshape(-1, 6)
    db = np.asarray(db, np.float32).reshape(-1, 6)
    pairs, used_a, used_b = [], set(), set()
    if len(da) and len(db):
        iou = _iou_matrix(da, db)
        ca = np.column_stack([(da[:, 0] + da[:, 2]) / 2, (da[:, 1] + da[:, 3]) / 2])
        cb = np.column_stack([(db[:, 0] + db[:, 2]) / 2, (db[:, 1] + db[:, 3]) / 2])
        diag = (np.hypot(da[:, 2] - da[:, 0], da[:, 3] - da[:, 1])[:, None]
                + np.hypot(db[:, 2] - db[:, 0], db[:, 3] - db[:, 1])[None, :]) / 2 + 1e-6
        dn = np.hypot(ca[:, None, 0] - cb[None, :, 0], ca[:, None, 1] - cb[None, :, 1]) / diag
        sim = np.where(iou >= min_iou, 1.0 + iou, np.where(dn < 1.0, 1.0 - dn, 0.0))
        sim[da[:, 5][:, None] != db[:, 5][None, :]] = 0.0
        for flat in np.argsort(-sim, axis=None):
            i, j = divmod(int(flat), sim.shape[1])
            if sim[i, j] <= 0:
                break
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
    rows = [da[i] * (1 - t) + db[j] * t for i, j in pairs]
    for r in rows:
        r[5] = round(float(r[5]))
    if t <= 0.5:
        rows += [da[i] for i in range(len(da)) if i not in used_a]
    else:
        rows += [db[j] for j in range(len(db)) if j not in used_b]
    return np.asarray(rows, np.float32).reshape(-1, 6)


def adaptive_stride(need_fps, have_fps, current=1, margin=1.15, max_stride=MAX_STRIDE):
    """Smallest frame step N so that analysing every N-th frame keeps up with playback: the analyser does have_fps
    frames per second, playback consumes need_fps, so N >= need_fps / have_fps (with a margin). It only gets finer
    one step at a time, and only with 30 % headroom, so it doesn't oscillate."""
    if have_fps <= 0 or need_fps <= 0:
        return current
    want = max(1, min(max_stride, math.ceil(need_fps * margin / have_fps)))
    if want < current:                       # be careful going finer: one step at a time, with 30 % headroom
        want = current - 1 if need_fps * 1.3 / have_fps <= current - 1 else current
    return max(1, min(max_stride, want))


# ------------------------------------------------------------------ analyser
class StreamAnalyser:
    def __init__(self, specs, path, cache_path, mode="balanced", stride=1, lead_s=3.0, batch=8, backend="auto",
                 start=None, end=None, log=None, cache=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.specs, self.path, self.cache_path = list(specs), path, cache_path
        self.mode, self.fixed_stride = mode, max(1, int(stride))
        self.lead_s, self.batch, self.backend = float(lead_s), max(1, int(batch)), backend
        self.start_s, self.end_s = start, end
        self.log = log or (lambda m: None)
        self._cache = cache
        self._cond = threading.Condition()
        self._quit = False
        self._thread = None
        self._store = {s.key: {} for s in self.specs}         # model key -> {frame: (n, 6) array}
        self._errors: dict[str, str] = {}
        self._names: dict[str, dict] = {}
        self._playhead, self._playing, self._speed = 0, False, 1.0
        self._exact: list[int] = []
        self._seek_gen = 0
        self.stride = 1 if mode == "best" else self.fixed_stride if mode == "fixed" else 1
        self.fps = 25.0
        self.first = self.last = 0
        self.frame_count = 0
        self.infer_fps = 0.0                                  # analysed frames per second (all models, with decode)
        self.phase = "starting"
        self.error = None
        self.ready = threading.Event()                        # opened (fps / range known)
        self.done = threading.Event()

    # ---- control (any thread)
    def start(self):
        self._thread = threading.Thread(target=self._run, name="stream-analysis", daemon=True)
        self._thread.start()
        return self

    def stop(self, wait=5.0):
        with self._cond:
            self._quit = True
            self._cond.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(wait)

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def set_playhead(self, idx, playing=False, speed=1.0):
        with self._cond:
            jump = abs(int(idx) - self._playhead) > max(2, int(self.fps * 0.5))
            self._playhead, self._playing, self._speed = int(idx), bool(playing), max(0.05, float(speed))
            if jump:
                self._seek_gen += 1
            self._cond.notify_all()

    def request_exact(self, idx):
        """Analyse this frame next (a paused frame without its own detections)."""
        idx = int(idx)
        if self.has(idx):
            return
        with self._cond:
            if idx not in self._exact:
                self._exact.append(idx)
            self._cond.notify_all()

    # ---- reads (any thread)
    def has(self, idx):
        ok = [k for k in self._store if k not in self._errors]
        return bool(ok) and all(idx in self._store[k] for k in ok)

    def names(self, key):
        return self._names.get(key, {})

    def detections(self, idx):
        """({model key: (n, 6) array}, how) for frame idx. how: exact | interpolated | held | None (nothing near)."""
        out, how = {}, "exact"
        gap = max(2, self.stride) * 2 + 1
        for k, frames in self._store.items():
            if k in self._errors:
                continue
            a = frames.get(idx)
            if a is not None:
                out[k] = a
                continue
            lo = next((i for i in range(idx - 1, idx - gap - 1, -1) if i in frames), None)
            hi = next((i for i in range(idx + 1, idx + gap + 1) if i in frames), None)
            if lo is not None and hi is not None:
                out[k] = interpolate(frames[lo], frames[hi], (idx - lo) / (hi - lo))
                how = "interpolated" if how in ("exact", "interpolated") else how
            elif lo is not None and idx - lo <= max(1, self.stride):
                out[k] = frames[lo]
                how = "held"
            else:
                how = None
        return (out, how) if out else ({}, None)

    def ready_until(self, idx, limit=None):
        """Last frame from idx on whose detections can be shown (exact or interpolated) - the buffer's end
        (looked at up to `limit` frames ahead; default 4 × the look-ahead)."""
        ok = [k for k in self._store if k not in self._errors]
        if not ok:
            return idx
        step = max(1, self.stride)
        end = min(self.last, idx + (limit or max(64, int(self.lead_s * self.fps * 4))))
        j = idx
        while j <= end:
            if all(j in self._store[k] for k in ok):
                j += 1
                continue
            nxt = next((i for i in range(j + 1, min(end, j + step) + 1) if all(i in self._store[k] for k in ok)),
                       None)
            if nxt is None:
                break
            j = nxt
        return j - 1

    def coverage(self):
        ok = [k for k in self._store if k not in self._errors]
        if not ok or not self.frame_count:
            return 0.0
        n = self.last - self.first + 1
        return min(len(self._store[k]) for k in ok) / max(1, n)

    def status(self):
        ahead = self.ready_until(self._playhead) - self._playhead
        return dict(phase=self.phase, mode=self.mode, stride=self.stride, coverage=self.coverage(),
                    infer_fps=self.infer_fps, fps=self.fps, ahead_frames=max(0, ahead),
                    ahead_s=max(0, ahead) / max(1e-6, self.fps), errors=dict(self._errors), error=self.error)

    # ---- worker
    def _open_cache(self):
        if self._cache is not None:
            return self._cache, False
        from AIO_QC.core.testing.runner import PredCache
        return PredCache(self.cache_path), True

    def _run(self):
        from AIO_QC.core.testing.runner import video_key
        from AIO_QC.tools.frame_extractor.core.sources import open_source
        cache = src = None
        own = False
        try:
            cache, own = self._open_cache()
            src = open_source(self.path, self.backend, hw_accel=False)        # CPU decoding only
            self.fps = float(getattr(src, "fps", 0) or 25.0)
            self.frame_count = int(src.frame_count)
            a = src.index_of(self.start_s) if self.start_s else 0
            b = min(src.frame_count - 1, src.index_of(self.end_s)) if self.end_s else src.frame_count - 1
            self.first, self.last = int(a), int(b)
            with self._cond:
                self._playhead = max(self._playhead, self.first)
            self.vk = video_key(self.path)
            self._load_cached(cache)
            self.ready.set()
            dets = self._load_models(cache)
            if not dets:
                self.phase = "failed"
                self.error = "no model could be loaded: " + "; ".join(self._errors.values())
                return
            self._loop(src, cache, dets)
        except Exception as e:                                      # never take the app down
            self.error = f"{type(e).__name__}: {e}"
            self.phase = "failed"
            self.log(f"video analysis of {self.path}: {self.error}")
        finally:
            self.ready.set()
            self.done.set()
            try:
                if src is not None:
                    src.release()
            except Exception:
                pass
            if own and cache is not None:
                cache.close()

    def _load_cached(self, cache):
        prefix = f"{self.vk}#"
        for s in self.specs:
            keys = [k for k in cache.keys_for(s.key) if k.startswith(prefix)]
            if keys:
                got = cache.get_many(s.key, keys)
                st = self._store[s.key]
                for k, (arr, _ms) in got.items():
                    st[int(k[len(prefix):])] = arr
                m = cache.model(s.key) or {}
                self._names[s.key] = m.get("names") or {}

    def _load_models(self, cache):
        dets = {}
        for s in self.specs:
            if self._complete(s.key):
                continue                                            # fully cached: no need to load it at all
            try:
                d = s.load()
                dets[s.key] = d
                self._names[s.key] = dict(getattr(d, "names", None) or {})
                cache.set_model(s.key, s, self._names[s.key], dict(backend=getattr(d, "provider", "")))
            except Exception as e:
                self._errors[s.key] = f"{s.label}: {type(e).__name__}: {e}"
                self.log(self._errors[s.key])
        if not dets and all(self._complete(s.key) or s.key in self._errors for s in self.specs) and \
                len(self._errors) < len(self.specs):
            dets = {"__complete__": None}
        return dets

    def _complete(self, key):
        st = self._store[key]
        return len(st) >= self.last - self.first + 1 and all(i in st for i in range(self.first, self.last + 1))

    def _missing(self, i):
        return any(i not in self._store[k] for k in self._store if k not in self._errors)

    def _next_batch(self):
        """(frames to analyse, why) - exact requests first, then the look-ahead window, then backfill."""
        with self._cond:
            while self._exact:
                i = self._exact.pop(0)
                if self.first <= i <= self.last and self._missing(i):
                    return [i], "exact"
            ph, playing, speed = self._playhead, self._playing, self._speed
        step = self.stride
        lead = max(self.batch, int(self.lead_s * self.fps * (speed if playing else 1.0)))
        start = max(self.first, ph)
        grid = start - ((start - self.first) % step)
        todo = [i for i in range(grid, min(self.last, start + lead) + 1, step) if i >= start and self._missing(i)]
        if todo:
            return todo[:self.batch], "ahead"
        # backfill: everything missing, nearest the playhead first (forward, then from the start)
        for lo, hi in ((start, self.last), (self.first, start - 1)):
            todo = []
            for i in range(lo, hi + 1):
                if self._missing(i):
                    todo.append(i)
                    if len(todo) >= self.batch:
                        break
            if todo:
                return todo, "backfill"
        return [], "done"

    def _loop(self, src, cache, dets):
        pos = None                                     # decoder position: the next frame read() returns
        pending = {k: [] for k in dets if k != "__complete__"}
        batch_cap = {k: self.batch for k in pending}
        t_last_flush = time.monotonic()
        while True:
            with self._cond:
                if self._quit:
                    break
            frames, why = self._next_batch()
            if not frames:
                self.phase = "done"
                self._flush(cache, pending)
                with self._cond:
                    if self._quit:
                        break
                    self._cond.wait(0.5)
                continue
            self.phase = why
            t0 = time.perf_counter()
            imgs = []
            for i in frames:
                if pos != i:
                    if pos is None or i < pos or i - pos > 48:
                        src.seek(i)
                    else:
                        while pos < i:
                            if not src.grab():
                                break
                            pos += 1
                ok, img = src.read()
                pos = i + 1
                if not ok:
                    self.last = min(self.last, i - 1)      # the file ends earlier than its header said
                    break
                imgs.append((i, img))
                with self._cond:
                    if self._quit:
                        break
            if not imgs:
                continue
            for k, d in dets.items():
                if k == "__complete__" or k in self._errors:
                    continue
                todo = [(i, im) for i, im in imgs if i not in self._store[k]]
                if not todo:
                    continue
                res = self._infer(k, d, [im for _, im in todo], batch_cap)
                if res is None:
                    continue
                st = self._store[k]
                for (i, _), r in zip(todo, res):
                    st[i] = np.asarray(r, np.float32).reshape(-1, 6)
                    pending[k].append((f"{self.vk}#{i}", st[i], 0.0))
            dt = time.perf_counter() - t0
            rate = len(imgs) / max(dt, 1e-6)
            self.infer_fps = rate if not self.infer_fps else 0.7 * self.infer_fps + 0.3 * rate
            self._adapt()
            if time.monotonic() - t_last_flush > 2.0 or sum(len(v) for v in pending.values()) > 256:
                self._flush(cache, pending)
                t_last_flush = time.monotonic()
        self._flush(cache, pending)

    def _infer(self, key, det, frames, batch_cap):
        """Run one model on frames; smaller batches on out-of-memory; None = the model failed (recorded)."""
        from AIO_QC.core.testing.runner import _infer
        spec = next(s for s in self.specs if s.key == key)
        out, i = [], 0
        while i < len(frames):
            n = batch_cap[key]
            try:
                out += _infer(det, frames[i:i + n], spec.conf)
                i += n
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if ("memory" in msg.lower() or "alloc" in msg.lower()) and n > 1:
                    batch_cap[key] = max(1, n // 2)
                    self.log(f"{spec.label}: out of memory - batch {batch_cap[key]}")
                    continue
                self._errors[key] = f"{spec.label}: {msg}"
                self.log(self._errors[key])
                return None
        return out

    def _adapt(self):
        if self.mode != "balanced":
            return
        with self._cond:
            playing, speed = self._playing, self._speed
        if not playing:
            return
        need = self.fps * speed
        new = adaptive_stride(need, self.infer_fps, self.stride)
        ahead = self.ready_until(self._playhead) - self._playhead
        if ahead < self.fps * self.lead_s * 0.25 and new <= self.stride and self.stride < MAX_STRIDE \
                and self.infer_fps < need * 1.05:
            new = self.stride + 1                          # the buffer is running dry: coarser now
        if new != self.stride:
            self.log(f"video analysis: every {new}{'st' if new == 1 else 'nd' if new == 2 else 'rd' if new == 3 else 'th'}"
                     f" frame ({self.infer_fps:.0f} analysed / s, playback needs {need:.0f} / s)")
            self.stride = new

    def _flush(self, cache, pending):
        for k, rows in pending.items():
            if rows:
                try:
                    cache.put_many(k, rows)
                except Exception as e:                    # a full disk must not stop the review
                    self.log(f"prediction cache: {e}")
                pending[k] = []
