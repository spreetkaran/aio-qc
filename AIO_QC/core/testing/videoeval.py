# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Label-free video analysis: how does a model behave on footage nobody has annotated?

Per model, at its operating threshold, over the sampled frames:
  * detections per class over time and mean confidence (timeline)
  * stability (IoU tracker, class-aware, tolerant to `gap` missing samples):
        flicker - a tracked object disappears for 1..gap samples and comes back   (missed detections)
        blip    - an object exists for at most `gap` samples                       (likely false detections)
        fragmentation = tracks / (tracks that would exist without flicker)
    stability score = share of detections not involved in a flicker / blip
  * confidence histogram
Across models:
  * agreement per frame (F1 of class-aware box matching at IoU 0.5) -> timeline + mean, plus the frames where
    models disagree most
  * "vs reference": one model's detections act as ground truth for the others -> every accuracy metric and a
    confusion matrix *relative to the reference* (clearly not true accuracy, but very useful to compare a
    candidate model with the production one)
All classes are put in one namespace by name (models trained with different class orders line up).
"""
import numpy as np

from AIO_QC.core.testing import metrics as M


def unify_names(models_names):
    """[{id: name}] per model -> (union {uid: name}, [mapping model id -> uid])."""
    union, by_norm, maps = {}, {}, []
    for names in models_names:
        mp = {}
        for k, v in sorted((names or {}).items()):
            n = M._norm(v)
            if n not in by_norm:
                by_norm[n] = len(union)
                union[len(union)] = str(v)
            mp[int(k)] = by_norm[n]
        maps.append(mp)
    return union, maps


class Tracker:
    """Greedy IoU tracker over consecutive samples (class-aware), recording flicker / blip events."""

    def __init__(self, gap=2, iou=0.3):
        self.gap, self.iou = max(1, int(gap)), iou
        self.tracks = []          # dict(box, cls, last, first, hits, frames)
        self.flicker = []         # (sample index, cls, box)
        self.blip = []
        self.n_tracks = 0
        self.n = 0

    def step(self, dets):
        """dets: (k, 6) array at the operating threshold."""
        n = self.n
        self.n += 1
        used = np.zeros(len(dets), bool)
        if len(dets) and self.tracks:
            tb = np.array([t["box"] for t in self.tracks], np.float32)
            iou = M.box_iou(tb, dets[:, :4])
            same = np.array([t["cls"] for t in self.tracks])[:, None] == dets[:, 5][None, :].astype(int)
            iou = np.where(same, iou, 0)
            order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
            matched_t = set()
            for ti, di in order:
                if iou[ti, di] < self.iou:
                    break
                if ti in matched_t or used[di]:
                    continue
                matched_t.add(ti)
                used[di] = True
                t = self.tracks[ti]
                missed = n - t["last"] - 1
                if 1 <= missed <= self.gap:
                    for m in range(t["last"] + 1, n):
                        self.flicker.append((m, t["cls"], t["box"]))
                t.update(box=dets[di, :4].tolist(), last=n, hits=t["hits"] + 1)
        for di in np.nonzero(~used)[0]:
            self.tracks.append(dict(box=dets[di, :4].tolist(), cls=int(dets[di, 5]), last=n, first=n, hits=1))
            self.n_tracks += 1
        alive = []
        for t in self.tracks:
            if n - t["last"] > self.gap:
                self._close(t)
            else:
                alive.append(t)
        self.tracks = alive

    def _close(self, t, final=False):
        if t["hits"] <= self.gap and t["first"] > 0 and not (final and t["last"] >= self.n - 1):
            for m in range(t["first"], t["last"] + 1):
                self.blip.append((m, t["cls"], t["box"]))

    def finish(self):
        for t in self.tracks:
            self._close(t, final=True)
        self.tracks = []


def analyse_video(models, preds, times, frames, op_conf=0.25, gap=2, reference=None, agree_iou=0.5):
    """models: [(label, names)] ; preds: [[(n, 6) array or None per frame] per model] (model class ids);
    times / frames: per sample. op_conf: float or list per model. reference: index of the reference model.
    Returns a dict ready for charts / reports."""
    n_models = len(models)
    confs = op_conf if isinstance(op_conf, (list, tuple)) else [op_conf] * n_models
    union, maps = unify_names([nm for _, nm in models])
    nc = max(1, len(union))
    unified = [[M.remap(p if p is not None else np.zeros((0, 6), np.float32), maps[i]) for p in preds[i]]
               for i in range(n_models)]
    out = dict(names=union, times=list(map(float, times)), frames=list(map(int, frames)), models=[], pairs=[],
               events=[], reference=None)
    for i, (label, _) in enumerate(models):
        c = confs[i]
        counts = np.zeros((len(times), nc), np.int32)
        mean_conf = np.zeros(len(times), np.float32)
        tracker = Tracker(gap)
        all_conf = []
        n_det = 0
        for f, p in enumerate(unified[i]):
            k = p[p[:, 4] >= c] if len(p) else p
            n_det += len(k)
            if len(k):
                np.add.at(counts[f], k[:, 5].astype(int), 1)
                mean_conf[f] = float(k[:, 4].mean())
            all_conf.append(p[:, 4] if len(p) else np.zeros(0))
            tracker.step(k)
        tracker.finish()
        unstable = len(tracker.flicker) + len(tracker.blip)
        hist, edges = np.histogram(np.concatenate(all_conf) if all_conf else np.zeros(0), bins=20, range=(0, 1))
        out["models"].append(dict(
            label=label, op_conf=c, detections=int(n_det), frames_with_dets=int((counts.sum(1) > 0).sum()),
            counts=counts, mean_conf=mean_conf, tracks=tracker.n_tracks, flicker=len(tracker.flicker),
            blip=len(tracker.blip), stability=float(1 - unstable / max(1, n_det + len(tracker.flicker))),
            per_class={union[k]: int(counts[:, k].sum()) for k in range(nc) if counts[:, k].sum()},
            conf_hist=hist.tolist(), conf_edges=edges.tolist(),
            flicker_frames=sorted({m for m, _, _ in tracker.flicker}),
            blip_frames=sorted({m for m, _, _ in tracker.blip})))
        for kind, evs in (("flicker", tracker.flicker), ("blip", tracker.blip)):
            for m, cls, box in evs:
                out["events"].append(dict(sample=int(m), frame=int(frames[m]), time=float(times[m]), model=label,
                                          type=kind, cls=union.get(cls, str(cls)), box=[float(v) for v in box]))
    # agreement between every pair of models
    for a in range(n_models):
        for b in range(a + 1, n_models):
            ag = np.array([M.agreement(unified[a][f], unified[b][f], confs[a], confs[b], agree_iou)[0]
                           for f in range(len(times))], np.float32)
            worst = np.argsort(ag)[:20]
            out["pairs"].append(dict(a=models[a][0], b=models[b][0], agreement=ag, mean=float(ag.mean()) if len(ag) else 1.0,
                                     worst_frames=[int(frames[w]) for w in worst if ag[w] < 1.0]))
            for w in worst:
                if ag[w] < 0.5:
                    out["events"].append(dict(sample=int(w), frame=int(frames[w]), time=float(times[w]),
                                              model=f"{models[a][0]} vs {models[b][0]}", type="disagree",
                                              cls="", box=None, score=float(ag[w])))
    out["events"].sort(key=lambda e: (e["frame"], e["type"]))
    out["preds"] = unified                         # per model, per sample, in the unified class ids
    out["model_names"] = [union] * n_models
    if reference is not None and 0 <= reference < n_models and n_models > 1:
        ref = unified[reference]
        gts = [(r[r[:, 4] >= confs[reference]][:, :4], r[r[:, 4] >= confs[reference]][:, 5].astype(int)) for r in ref]
        res = []
        for i, (label, _) in enumerate(models):
            if i == reference:
                continue
            res.append(M.evaluate(label, union, gts, unified[i], op_conf=confs[i]))
        out["reference"] = dict(label=models[reference][0], results=res)
    return out
