# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Running models: batched inference with a persistent prediction cache.

  * one load per model; frames are read ahead on a background thread while the model works on the
    previous batch; in-app ONNX models with a dynamic batch axis run one forward pass per batch, environment
    models (.pt / .engine) receive whole batches through shared memory (no disk)
  * every prediction is stored in a SQLite cache keyed by (model fingerprint, image key): re-running is
    instant, a crash or cancel resumes where it stopped, changing thresholds never re-runs a model.
    The model fingerprint is the weights' content (size + first/last 4 MB) plus the settings that change
    predictions (image size, NMS IoU, confidence floor, max detections, FP16, SAHI) - renaming or moving
    the file keeps its cache
  * predictions are stored at a low confidence floor (default 0.001, NMS IoU 0.7, 300 boxes - the
    `yolo val` settings) so mAP and every operating threshold can be computed afterwards
  * speed = median ms per image of the inference call (pre/post-processing included, image loading
    excluded, first batch ignored as warm-up) - the same measurement for every backend
  * video sources: frames are decoded once per batch and handed to all models
"""
import hashlib
import json
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from AIO_QC.tools.frame_extractor.core.detector import load_detector
from AIO_QC.tools.frame_extractor.core.imageio import imread


# ------------------------------------------------------------------ model specs
def file_fingerprint(path, chunk=4 * 2 ** 20):
    st = os.stat(path)
    h = hashlib.sha1(str(st.st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if st.st_size > 2 * chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return h.hexdigest()


@dataclass
class ModelSpec:
    path: str
    label: str = ""
    imgsz: int = 0
    conf: float = 0.001          # confidence floor stored in the cache
    iou: float = 0.7             # NMS IoU (yolo val default)
    max_det: int = 300
    half: bool = False
    sahi: dict = None
    env_python: str = ""
    device: str = "auto"
    color: str = ""
    class_map: dict = None       # {model class id: dataset label} (AIO QC's class map; "" = ignore) - not in the key
    _fp: str = field(default="", repr=False)

    def __post_init__(self):
        self.label = self.label or os.path.splitext(os.path.basename(self.path))[0]

    @property
    def key(self):
        if not self._fp:
            self._fp = file_fingerprint(self.path)
        params = dict(imgsz=int(self.imgsz), conf=round(float(self.conf), 6), iou=round(float(self.iou), 4),
                      max_det=int(self.max_det), half=bool(self.half), sahi=self.sahi or None)
        return hashlib.sha1((self._fp + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:16]

    def to_dict(self):
        d = asdict(self)
        d.pop("_fp", None)
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "_fp"})

    def load(self):
        return load_detector(self.path, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device,
                             env_python=self.env_python, half=self.half, sahi=self.sahi, max_det=self.max_det)


# ------------------------------------------------------------------ cache
class PredCache:
    """SQLite store of predictions: float32 (n, 6) arrays in model class ids + ms per image."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self.con = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute("CREATE TABLE IF NOT EXISTS models (key TEXT PRIMARY KEY, path TEXT, label TEXT, "
                         "names TEXT, spec TEXT, info TEXT, updated REAL)")
        self.con.execute("CREATE TABLE IF NOT EXISTS preds (model TEXT, image TEXT, data BLOB, ms REAL, "
                         "PRIMARY KEY (model, image)) WITHOUT ROWID")
        self.con.commit()

    def close(self):
        with self._lock:
            self.con.close()

    def set_model(self, key, spec, names=None, info=None):
        with self._lock:
            old = self.model(key) or {}
            names = names if names is not None else old.get("names")
            info = {**(old.get("info") or {}), **(info or {})}
            self.con.execute("INSERT OR REPLACE INTO models VALUES (?,?,?,?,?,?,?)",
                             (key, spec.path, spec.label, json.dumps({str(k): v for k, v in (names or {}).items()}),
                              json.dumps(spec.to_dict()), json.dumps(info), time.time()))
            self.con.commit()

    def model(self, key):
        with self._lock:
            r = self.con.execute("SELECT path, label, names, spec, info FROM models WHERE key=?", (key,)).fetchone()
        if not r:
            return None
        return dict(path=r[0], label=r[1], names={int(k): v for k, v in json.loads(r[2] or "{}").items()},
                    spec=json.loads(r[3] or "{}"), info=json.loads(r[4] or "{}"))

    def keys_for(self, model):
        with self._lock:
            return {r[0] for r in self.con.execute("SELECT image FROM preds WHERE model=?", (model,))}

    def put_many(self, model, rows):
        """rows: [(image key, (n, 6) array, ms)]"""
        with self._lock:
            self.con.executemany("INSERT OR REPLACE INTO preds VALUES (?,?,?,?)",
                                 [(model, k, np.asarray(a, np.float32).reshape(-1, 6).tobytes(), float(ms))
                                  for k, a, ms in rows])
            self.con.commit()

    def get_many(self, model, keys):
        """{image key: (array, ms)} for the keys present."""
        out = {}
        keys = list(keys)
        with self._lock:
            for i in range(0, len(keys), 900):
                part = keys[i:i + 900]
                q = f"SELECT image, data, ms FROM preds WHERE model=? AND image IN ({','.join('?' * len(part))})"
                for k, blob, ms in self.con.execute(q, [model] + part):
                    out[k] = (np.frombuffer(blob, np.float32).reshape(-1, 6).copy(), ms)
        return out

    def speed_ms(self, model):
        with self._lock:
            ms = [r[0] for r in self.con.execute("SELECT ms FROM preds WHERE model=? AND ms > 0", (model,))]
        return float(np.median(ms)) if ms else None

    def clear(self, model=None):
        with self._lock:
            if model:
                self.con.execute("DELETE FROM preds WHERE model=?", (model,))
                self.con.execute("DELETE FROM models WHERE key=?", (model,))
            else:
                self.con.execute("DELETE FROM preds")
                self.con.execute("DELETE FROM models")
            self.con.commit()


def image_key(path, root=None):
    """Stable key: path relative to the test set root, or absolute path + size + mtime for loose files."""
    if root:
        try:
            rel = os.path.relpath(path, root)
            if not rel.startswith(".."):
                return rel.replace(os.sep, "/")
        except ValueError:
            pass
    try:
        st = os.stat(path)
        return f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        return os.path.abspath(path)


def video_key(path):
    st = os.stat(path)
    return hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]


# ------------------------------------------------------------------ prefetch
def _prefetch(paths, batch, cancel, depth=2):
    """Background reader: yields (start index, [frames or None])."""
    q = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def reader():
        try:
            for i in range(0, len(paths), batch):
                if stop.is_set() or (cancel and cancel()):
                    break
                q.put((i, [imread(p) for p in paths[i:i + batch]]))
        finally:
            q.put(None)
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is None:
                return
            yield item
    finally:
        stop.set()
        while t.is_alive():                     # unblock a reader waiting on a full queue
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            t.join(0.05)


def _infer(det, frames, conf, cancel=None):
    if hasattr(det, "detect_batch"):
        try:
            return det.detect_batch(frames, conf, cancel=cancel)
        except TypeError:
            return det.detect_batch(frames, conf)
    from AIO_QC.tools.frame_extractor.core.detector import dets_to_array
    return [dets_to_array(det.detect(f, conf)) for f in frames]


# ------------------------------------------------------------------ images
def run_on_images(specs, images, cache, progress=None, cancel=None, batch=8, log=None):
    """specs: [ModelSpec]; images: [(key, path)]. Runs what is missing from the cache.
    progress(model_index, done, total, ms_per_image, text). Returns {model key: info dict}."""
    cancel = cancel or (lambda: False)
    progress = progress or (lambda *a: None)
    log = log or (lambda m: None)
    out = {}
    for mi, spec in enumerate(specs):
        mk = spec.key
        info = dict(label=spec.label, key=mk, error=None, ran=0, failed=0)
        out[mk] = info
        have = cache.keys_for(mk)
        todo = [(k, p) for k, p in images if k not in have]
        cached = cache.model(mk)
        if not todo:
            info["names"] = (cached or {}).get("names") or {}
            info["speed_ms"] = cache.speed_ms(mk)
            progress(mi, len(images), len(images), info["speed_ms"], f"{spec.label}: cached")
            continue
        if cancel():
            break
        progress(mi, len(images) - len(todo), len(images), None, f"{spec.label}: loading model…")
        try:
            t0 = time.perf_counter()
            det = spec.load()
            info["load_s"] = round(time.perf_counter() - t0, 2)
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
            log(f"{spec.label}: {info['error']}")
            progress(mi, 0, len(images), None, f"{spec.label}: {info['error']}")
            continue
        names = dict(getattr(det, "names", None) or {})
        info["names"] = names
        info["backend"] = getattr(det, "provider", "")
        used = getattr(det, "imgsz", None)                  # the size it really ran at (shared-cache matching)
        cache.set_model(mk, spec, names, dict(backend=info["backend"], load_s=info["load_s"],
                                              imgsz_used=list(used) if isinstance(used, (tuple, list)) else used))
        keys, paths = [k for k, _ in todo], [p for _, p in todo]
        done = len(images) - len(todo)
        first = True
        pending = []
        for start, frames in _prefetch(paths, batch, cancel):
            if cancel():
                break
            ok = [i for i, f in enumerate(frames) if f is not None]
            info["failed"] += len(frames) - len(ok)
            if not ok:
                continue
            t0 = time.perf_counter()
            try:
                res = _infer(det, [frames[i] for i in ok], spec.conf)
            except Exception as e:
                if "cancelled" in str(e):
                    break
                info["error"] = f"{type(e).__name__}: {e}"
                log(f"{spec.label}: {info['error']}")
                break
            ms = (time.perf_counter() - t0) * 1000 / len(ok)
            pending += [(keys[start + i], r, 0.0 if first else ms) for i, r in zip(ok, res)]
            first = False
            done += len(frames)
            info["ran"] += len(ok)
            if len(pending) >= 64:
                cache.put_many(mk, pending)
                pending = []
            progress(mi, done, len(images), ms, f"{spec.label}: {done}/{len(images)}")
        if pending:
            cache.put_many(mk, pending)
        info["speed_ms"] = cache.speed_ms(mk)
        cache.set_model(mk, spec, names, dict(speed_ms=info["speed_ms"]))
    return out


def load_predictions(cache, spec_or_key, keys):
    """Aligned list of (n, 6) arrays (None where missing) + names for one model."""
    mk = spec_or_key if isinstance(spec_or_key, str) else spec_or_key.key
    got = cache.get_many(mk, keys)
    m = cache.model(mk) or {}
    return [got[k][0] if k in got else None for k in keys], m.get("names") or {}


# ------------------------------------------------------------------ videos
def frame_range(src, start=None, end=None):
    """Seconds (None = open) -> first and last frame index of an opened source."""
    a = src.index_of(start) if start else 0
    b = min(src.frame_count - 1, src.index_of(end)) if end else src.frame_count - 1
    return a, b


def video_frames(path, stride=1, start=None, end=None, cancel=None):
    """Yield (frame index, time, frame) every `stride` frames (sequential decode, exact timestamps)."""
    from AIO_QC.tools.frame_extractor.core.sources import open_source
    src = open_source(path, "auto", hw_accel=False)
    try:
        a, b = frame_range(src, start, end)
        if a:
            src.seek(a)
        if stride >= 48:                 # sparse samples: seek to each (decodes ~half a GOP, not every frame)
            for i in range(a, b + 1, stride):
                if cancel is not None and cancel():
                    return
                if i != a:
                    src.seek(i)
                ok, frame = src.read()
                if not ok:
                    return
                yield i, src.time_of(i), frame
            return
        i = a
        while i <= b:
            if cancel is not None and cancel():
                return
            if (i - a) % stride == 0:
                ok, frame = src.read()
                if not ok:
                    return
                yield i, src.time_of(i), frame
            else:
                if not src.grab():
                    return
            i += 1
    finally:
        src.release()


def run_on_video(specs, path, cache, stride=1, start=None, end=None, progress=None, cancel=None, batch=8,
                 log=None):
    """Every model on the same decoded frames (decode once). Cache keys: '<video key>#<frame>'.
    Returns ({model key: info}, frame indices, times)."""
    cancel = cancel or (lambda: False)
    progress = progress or (lambda *a: None)
    log = log or (lambda m: None)
    vk = video_key(path)
    from AIO_QC.tools.frame_extractor.core.sources import open_source
    src = open_source(path, "auto", hw_accel=False)
    a, b = frame_range(src, start, end)
    idx_all = list(range(a, b + 1, max(1, stride)))
    times = [src.time_of(i) for i in idx_all]
    src.release()
    keys_all = [f"{vk}#{i}" for i in idx_all]
    infos, dets, todo_models = {}, {}, []
    for spec in specs:
        mk = spec.key
        info = dict(label=spec.label, key=mk, error=None, ran=0)
        infos[mk] = info
        missing = set(keys_all) - cache.keys_for(mk)
        if not missing:
            m = cache.model(mk) or {}
            info["names"], info["speed_ms"] = m.get("names") or {}, cache.speed_ms(mk)
            continue
        try:
            progress(len(todo_models), 0, len(idx_all), None, f"{spec.label}: loading model…")
            d = spec.load()
            dets[mk] = d
            info["names"] = dict(getattr(d, "names", None) or {})
            info["backend"] = getattr(d, "provider", "")
            cache.set_model(mk, spec, info["names"], dict(backend=info["backend"]))
            todo_models.append((spec, missing))
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
            log(f"{spec.label}: {info['error']}")
    if todo_models:
        need = set().union(*[m for _, m in todo_models])
        first = {s.key: True for s, _ in todo_models}
        buf, done = [], 0
        pending = {s.key: [] for s, _ in todo_models}

        def flush(buf):
            frames = [f for _, f in buf]
            for spec, missing in todo_models:
                if infos[spec.key].get("error"):
                    continue
                sel = [j for j, (k, _) in enumerate(buf) if k in missing]
                if not sel:
                    continue
                t0 = time.perf_counter()
                try:
                    res = _infer(dets[spec.key], [frames[j] for j in sel], spec.conf)
                except Exception as e:
                    infos[spec.key]["error"] = f"{type(e).__name__}: {e}"
                    log(f"{spec.label}: {infos[spec.key]['error']}")
                    continue
                ms = (time.perf_counter() - t0) * 1000 / len(sel)
                pending[spec.key] += [(buf[j][0], r, 0.0 if first[spec.key] else ms) for j, r in zip(sel, res)]
                first[spec.key] = False
                infos[spec.key]["ran"] += len(sel)
                if len(pending[spec.key]) >= 128:
                    cache.put_many(spec.key, pending[spec.key])
                    pending[spec.key] = []
        for i, t, frame in video_frames(path, stride, start, end, cancel):
            done += 1
            k = f"{vk}#{i}"
            if k in need:
                buf.append((k, frame))
            if len(buf) >= batch:
                flush(buf)
                buf = []
            if done % 4 == 0:
                progress(0, done, len(idx_all), None, f"{os.path.basename(path)} · frame {i}")
        if buf and not cancel():
            flush(buf)
        for mk, rows in pending.items():
            if rows:
                cache.put_many(mk, rows)
        for spec, _ in todo_models:
            infos[spec.key]["speed_ms"] = cache.speed_ms(spec.key)
            cache.set_model(spec.key, spec, infos[spec.key].get("names"), dict(speed_ms=infos[spec.key]["speed_ms"]))
    return infos, idx_all, times, keys_all


def default_cache_dir():
    from AIO_QC.core.fx_storage import testing_home
    d = str(testing_home())
    os.makedirs(d, exist_ok=True)
    return d
