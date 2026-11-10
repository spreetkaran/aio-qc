# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Video review: what happens over time, and which frames are worth a closer look - Qt-free.

From the saved detections of one video (frame -> (n, 6) array per model, exact frames only):

    summarize(store, first, last, thr)          per-model detections-per-frame (-1 = not analysed)
    find_events(store, names, first, last, fps, thr, band)
        disagree   two models see different things (a box of one has no same-class box of the other at IoU ≥ 0.5)
        flicker    a model loses an object for one frame (present before and after, missing in between)
        uncertain  a box just below the threshold (thr - band ≤ conf < thr): the model hesitates
      events closer than `merge_s` are merged into one (at its strongest frame); a stretch longer than `max_span_s`
      is cut into several, so a long disagreement still offers several frames -> [Event]
    pick_hard_frames(events, kinds, max_n, min_gap_frames)   spread-out frames for annotation
    export_frames(...)                          those frames + LabelMe pre-labels + their video provenance

Class names are compared by name (models trained with different class orders compare correctly).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

KINDS = ("disagree", "flicker", "uncertain")
KIND_TEXT = {"disagree": "Models disagree", "flicker": "Flicker", "uncertain": "Uncertain"}


@dataclass
class Event:
    frame: int
    kind: str
    score: float
    text: str
    start: int
    end: int


def _iou(a, b):
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


def _labels(arr, names):
    return np.array([names.get(int(c), str(int(c))) for c in arr[:, 5]], object) if len(arr) else np.zeros(0, object)


def _unmatched(a, la, b, lb, thr=0.5):
    """How many boxes of a have no same-name box in b at IoU ≥ thr."""
    if not len(a):
        return 0
    if not len(b):
        return len(a)
    iou = _iou(a, b)
    iou[la[:, None] != lb[None, :]] = 0
    return int((iou.max(axis=1) < thr).sum())


def summarize(store: dict, first: int, last: int, thr: float) -> dict:
    """{model key: int16 array (last - first + 1)} - boxes ≥ thr per frame, -1 where not analysed."""
    n = max(0, last - first + 1)
    out = {}
    for k, frames in store.items():
        a = np.full(n, -1, np.int16)
        for i, arr in list(frames.items()):
            if first <= i <= last:
                a[i - first] = int((arr[:, 4] >= thr).sum()) if len(arr) else 0
        out[k] = a
    return out


def find_events(store: dict, names: dict, first: int, last: int, fps: float, thr: float = 0.25, band: float = 0.1,
                merge_s: float = 0.5, kinds=KINDS, max_span_s: float = 2.0) -> list:
    keys = [k for k in store if store[k]]
    raw = []                                                 # (frame, kind, score, text)
    frames_all = sorted(set().union(*[set(store[k]) for k in keys])) if keys else []
    frames_all = [i for i in frames_all if first <= i <= last]

    def kept(k, i):
        arr = store[k].get(i)
        if arr is None:
            return None, None
        arr = arr[arr[:, 4] >= thr] if len(arr) else arr.reshape(-1, 6)
        return arr, _labels(arr, names.get(k, {}))
    for i in frames_all:
        have = {k: kept(k, i) for k in keys}
        have = {k: v for k, v in have.items() if v[0] is not None}
        if "disagree" in kinds and len(have) >= 2:
            ks = list(have)
            worst, pair = 0, None
            for x in range(len(ks)):
                for y in range(x + 1, len(ks)):
                    (a, la), (b, lb) = have[ks[x]], have[ks[y]]
                    u = _unmatched(a, la, b, lb) + _unmatched(b, lb, a, la)
                    if u > worst:
                        worst, pair = u, (ks[x], ks[y])
            if worst:
                raw.append((i, "disagree", float(worst), pair))
        if "uncertain" in kinds:
            n_unc = 0
            for k in keys:
                arr = store[k].get(i)
                if arr is not None and len(arr):
                    n_unc += int(((arr[:, 4] >= max(0.0, thr - band)) & (arr[:, 4] < thr)).sum())
            if n_unc:
                raw.append((i, "uncertain", float(n_unc), None))
        if "flicker" in kinds:
            for k in keys:
                cur, lc = have.get(k, (None, None))
                prev, lp = kept(k, i - 1)
                nxt, ln = kept(k, i + 1)
                if cur is None or prev is None or nxt is None or not len(prev) or not len(nxt):
                    continue
                iou = _iou(prev, nxt)
                iou[lp[:, None] != ln[None, :]] = 0
                stable = [j for j in range(len(prev)) if iou[j].max() >= 0.5]
                if not stable:
                    continue
                lost = _unmatched(prev[stable], lp[stable], cur, lc)
                if lost:
                    raw.append((i, "flicker", float(lost), k))
    gap = max(1, int(round(merge_s * (fps or 25))))
    span = max(gap, int(round(max_span_s * (fps or 25))))      # a long stretch becomes several events
    events = []
    for kind in kinds:
        rows = sorted((r for r in raw if r[1] == kind), key=lambda r: r[0])
        seg = []
        for r in rows:
            if seg and (r[0] - seg[-1][0] > gap or r[0] - seg[0][0] >= span):
                events.append(_merge(seg, names, store))
                seg = []
            seg.append(r)
        if seg:
            events.append(_merge(seg, names, store))
    return sorted(events, key=lambda e: e.frame)


def _merge(seg, names, store):
    best = max(seg, key=lambda r: r[2])
    kind = best[1]
    if kind == "disagree":
        text = f"{int(best[2])} box(es) differ"
    elif kind == "flicker":
        text = f"{int(best[2])} object(s) lost for a frame"
    else:
        text = f"{int(best[2])} box(es) just below the threshold"
    return Event(frame=int(best[0]), kind=kind, score=float(best[2]), text=text, start=int(seg[0][0]),
                 end=int(seg[-1][0]))


def pick_hard_frames(events, kinds=KINDS, max_n=50, min_gap=12) -> list:
    """Strongest events first, at least min_gap frames apart, then in time order."""
    cand = sorted((e for e in events if e.kind in kinds), key=lambda e: (-e.score, e.frame))
    picked = []
    for e in cand:
        if len(picked) >= max_n:
            break
        if all(abs(e.frame - p.frame) >= min_gap for p in picked):
            picked.append(e)
    return sorted(picked, key=lambda e: e.frame)


def load_store(specs, path, cache_path) -> tuple[dict, dict]:
    """Saved detections of one video from the prediction cache: ({model key: {frame: arr}}, {model key: names})."""
    from AIO_QC.core.testing.runner import PredCache, video_key
    vk = video_key(path)
    prefix = f"{vk}#"
    cache = PredCache(cache_path)
    try:
        store, names = {}, {}
        for s in specs:
            keys = [k for k in cache.keys_for(s.key) if k.startswith(prefix)]
            got = cache.get_many(s.key, keys) if keys else {}
            store[s.key] = {int(k[len(prefix):]): a for k, (a, _ms) in got.items()}
            names[s.key] = (cache.model(s.key) or {}).get("names") or {}
        return store, names
    finally:
        cache.close()


def export_frames(path, frames, out_dir, store=None, names=None, label_model=None, thr=0.25, backend="auto",
                  ext=".png", progress=None, cancel=None) -> list:
    """Write each frame (exact decode) and, with label_model, LabelMe pre-labels from that model's detections ≥ thr.
    Each image remembers its video and frame (Image QC ▸ Show Source Video). Returns the image paths."""
    from AIO_QC.tools.frame_extractor.core.detector import Detection, write_labelme
    from AIO_QC.tools.frame_extractor.core.frame_sources import record
    from AIO_QC.tools.frame_extractor.core.imageio import imwrite
    from AIO_QC.tools.frame_extractor.core.sources import open_source
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    src = open_source(path, backend, hw_accel=False)
    out = []
    try:
        for n, i in enumerate(sorted(frames)):
            if cancel is not None and cancel():
                break
            src.seek(i)
            ok, img = src.read()
            if not ok:
                continue
            name = f"{stem}_{i:06d}{ext}"
            p = os.path.join(out_dir, name)
            imwrite(p, img, ext)
            record(p, path, i, src.time_of(i))
            if label_model and store is not None and label_model in store and i in store[label_model]:
                arr = store[label_model][i]
                nm = (names or {}).get(label_model, {})
                dets = [Detection(float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), int(a[5]),
                                  nm.get(int(a[5]), str(int(a[5])))) for a in arr if a[4] >= thr]
                h, w = img.shape[:2]
                write_labelme(os.path.splitext(p)[0] + ".json", name, w, h, dets)
            out.append(p)
            if progress is not None:
                progress(n + 1, len(frames))
    finally:
        src.release()
    return out
