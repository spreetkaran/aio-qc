# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Prediction Explorer data — every label and every model prediction of a dataset in flat numpy arrays (Qt-free).

Built once by a Model Check run (or loaded from .aio_qc/explorer.npz), then every slider / filter / sort is a few
vectorised numpy passes, so it stays interactive on millions of boxes.

Matching
    Each model is matched against the labels ON ITS OWN (COCO greedy, most confident first, same class), for every
    IoU of the grid at once. Greedy matching is prefix-consistent in confidence, so a confidence threshold is a mask:
    no re-matching when a slider moves. Comparing models is therefore fair — model B never "steals" a label from A.

Tags (what the drawing colours and the filters use)
    predictions   correct · wrong class · poor fit · duplicate · not labelled (no label near it: an unlabeled
                  object or a false alarm)
    labels        correct · wrong class · poor fit · missed · not checked (no visible model predicts that class)
    A label is "correct" when ANY visible model matches it; its status otherwise is the most specific reason.

Geometry
    boxes are pixel xyxy; polygons (labels that are not rectangles, segmentation / OBB predictions) and keypoints
    live in CSR arrays (start/end per object into one flat point array) — memory stays proportional to the points.

Edits
    replace_gt() swaps the labels of some images (after an accept / relabel / delete): the new labels are appended,
    the old ones are marked dead, and the predictions of those images are re-matched. No full rebuild.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

import numpy as np

from .evaluation import IOU_GRID
from .geometry import to_polygon
from .model_check import iou_matrix

FORMAT = 1

# ---- tags -------------------------------------------------------------------------------------------------------
OK, CLASS, FIT, DUPE, EXTRA, MISSED, NA, HIDDEN = range(8)
TAG_NAMES = {OK: "correct", CLASS: "wrong class", FIT: "poor fit", DUPE: "duplicate", EXTRA: "not labelled",
             MISSED: "missed", NA: "not checked", HIDDEN: "below threshold"}
TAG_HELP = {
    OK: "label and prediction agree (same class, overlap ≥ match IoU)",
    CLASS: "a prediction of ANOTHER class sits on the label — wrong label, or a model confusion",
    FIT: "same class but a loose overlap — sloppy box (or a sloppy model)",
    DUPE: "a second prediction on an object that is already matched — a model issue",
    EXTRA: "a prediction with no label near it — an unlabeled object, or a false alarm",
    MISSED: "a label no visible model finds — a doubtful / wrong label, or a very hard object",
    NA: "no visible model predicts this class — nothing to compare",
}
ERROR_TAGS = (CLASS, FIT, DUPE, EXTRA, MISSED)
PRED_TAGS = (OK, CLASS, FIT, DUPE, EXTRA)
GT_TAGS = (OK, CLASS, FIT, MISSED, NA)
TAG_BY_NAME = {v: k for k, v in TAG_NAMES.items()}

SORTS = [
    ("errors", "Most errors"), ("missed", "Most missed labels"), ("extra", "Most unlabelled detections"),
    ("class", "Most wrong classes"), ("fit", "Worst box fit"), ("uncertain", "Most uncertain"),
    ("disagree", "Models disagree most"), ("objects", "Most objects"), ("low_conf", "Lowest confidence"),
    ("name", "File name"), ("random", "Random"), ("scan", "Scan order"),
]
SORT_KEYS = [k for k, _t in SORTS]


def _empty(dtype, *shape):
    return np.zeros((0, *shape) if shape else 0, dtype)


# =================================================================================================================
# Data
# =================================================================================================================
@dataclass
class ExplorerData:
    keys: list
    labels: list
    models: list                       # model ids (order = p_model index)
    model_names: dict                  # id -> display name
    model_colors: dict                 # id -> "#rrggbb"
    splits: list
    folders: list
    img_w: np.ndarray
    img_h: np.ndarray
    img_split: np.ndarray
    img_folder: np.ndarray
    img_seen: np.ndarray               # [n_models, N] the model trained on this image
    cov: np.ndarray                    # [n_models, n_labels] the model predicts this label
    img_g0: np.ndarray                 # labels of image i: rows img_g0[i]:img_g1[i]
    img_g1: np.ndarray
    img_p0: np.ndarray
    img_p1: np.ndarray
    g_img: np.ndarray
    g_cls: np.ndarray
    g_idx: np.ndarray                  # shape index in the file
    g_box: np.ndarray
    g_alive: np.ndarray
    g_pt0: np.ndarray                  # polygon points (labels that are not plain rectangles)
    g_pt1: np.ndarray
    g_pts: np.ndarray
    p_img: np.ndarray
    p_cls: np.ndarray
    p_conf: np.ndarray
    p_model: np.ndarray
    p_box: np.ndarray
    p_pt0: np.ndarray                  # segmentation / OBB outline
    p_pt1: np.ndarray
    p_pts: np.ndarray
    p_k0: np.ndarray                   # keypoints (x, y, conf)
    p_k1: np.ndarray
    p_kps: np.ndarray
    match: np.ndarray                  # [T, n_pred] global label row or -1 (per-model greedy)
    p_bsame: np.ndarray                # best IoU with a live label of the same class
    p_bother: np.ndarray               # ... of another class
    pr_g: np.ndarray                   # label <-> prediction overlaps (IoU ≥ 0.1)
    pr_p: np.ndarray
    pr_iou: np.ndarray
    pr_same: np.ndarray
    pp_a: np.ndarray                   # prediction <-> prediction of ANOTHER model, same class, IoU ≥ 0.3
    pp_b: np.ndarray
    pp_iou: np.ndarray
    meta: dict = field(default_factory=dict)
    _key_index: dict | None = None
    _label_index: dict | None = None

    # ---- lookup
    def __len__(self):
        return len(self.keys)

    @property
    def n_models(self):
        return len(self.models)

    def index_of(self, key) -> int | None:
        if self._key_index is None:
            self._key_index = {k: i for i, k in enumerate(self.keys)}
        return self._key_index.get(key)

    def label_id(self, label: str, create: bool = False) -> int | None:
        if self._label_index is None:
            self._label_index = {l: i for i, l in enumerate(self.labels)}
        i = self._label_index.get(label)
        if i is None and create:
            i = len(self.labels)
            self.labels.append(label)
            self._label_index[label] = i
            self.cov = np.concatenate([self.cov, np.zeros((self.cov.shape[0], 1), bool)], 1)
        return i

    def gt_rows(self, i) -> np.ndarray:
        r = np.arange(self.img_g0[i], self.img_g1[i])
        return r[self.g_alive[r]]

    def pred_rows(self, i) -> np.ndarray:
        return np.arange(self.img_p0[i], self.img_p1[i])

    def g_poly(self, g) -> np.ndarray | None:
        a, b = self.g_pt0[g], self.g_pt1[g]
        return self.g_pts[a:b] if b > a else None

    def p_poly(self, p) -> np.ndarray | None:
        a, b = self.p_pt0[p], self.p_pt1[p]
        return self.p_pts[a:b] if b > a else None

    def p_keypoints(self, p) -> np.ndarray | None:
        a, b = self.p_k0[p], self.p_k1[p]
        return self.p_kps[a:b] if b > a else None


# =================================================================================================================
# Building
# =================================================================================================================
def _shape_geometry(shape, sx, sy):
    """(box xyxy px, polygon px or None) of an area-bearing shape, else None."""
    poly = to_polygon(shape)
    if poly is None:
        return None
    poly = poly * (sx, sy)
    x1, y1 = poly.min(0)
    x2, y2 = poly.max(0)
    st = shape.get("shape_type") or "polygon"
    keep = None if st == "rectangle" else poly.astype(np.float32)
    if keep is not None and len(keep) == 4:
        xs, ys = sorted(set(np.round(keep[:, 0], 3))), sorted(set(np.round(keep[:, 1], 3)))
        if len(xs) == 2 and len(ys) == 2:
            keep = None                                          # an axis-aligned 4-point polygon is a box
    return (float(x1), float(y1), float(x2), float(y2)), keep


class Builder:
    """Collects one image at a time (worker thread); every CHUNK images the new block is matched (vectorised), so
    the cost is spread over the scan. finish() -> ExplorerData."""
    CHUNK = 5000

    def __init__(self, models, model_names=None, model_colors=None, coverage=None, meta=None):
        self.models = list(models)
        self.midx = {m: i for i, m in enumerate(self.models)}
        self.model_names = dict(model_names or {})
        self.model_colors = dict(model_colors or {})
        self.coverage = {k: set(v) for k, v in (coverage or {}).items()}       # label -> model ids
        self.meta = dict(meta or {})
        self.keys, self.labels, self.lab = [], [], {}
        self.splits, self.sidx, self.folders, self.fidx = [], {}, [], {}
        self.W, self.H, self.S, self.F, self.seen = [], [], [], [], []
        self.G = {k: [] for k in ("img", "cls", "idx", "box", "pt0", "pt1")}
        self.P = {k: [] for k in ("img", "cls", "conf", "model", "box", "pt0", "pt1", "k0", "k1")}
        self.gpts, self.ppts, self.kps = [], [], []
        self.n_gpts = self.n_ppts = self.n_kps = 0
        self.done = (0, 0, 0)                                    # images, labels, predictions already matched
        self.blocks = []

    def _label(self, name):
        i = self.lab.get(name)
        if i is None:
            i = self.lab[name] = len(self.labels)
            self.labels.append(name)
        return i

    def add(self, key, split, folder, w, h, shapes, normalized, preds_by_model, seen=None):
        """preds_by_model: {model id: [(label, conf, (x1, y1, x2, y2)[, polygon (k, 2)[, keypoints (k, 3)]])]}
        (labels already mapped to dataset names; unmapped classes left out)."""
        i = len(self.keys)
        self.keys.append(key)
        self.W.append(float(w or 0))
        self.H.append(float(h or 0))
        if split not in self.sidx:
            self.sidx[split] = len(self.splits)
            self.splits.append(split)
        if folder not in self.fidx:
            self.fidx[folder] = len(self.folders)
            self.folders.append(folder)
        self.S.append(self.sidx[split])
        self.F.append(self.fidx[folder])
        self.seen.append([bool((seen or {}).get(m, False)) for m in self.models])
        sx, sy = (w or 1.0, h or 1.0) if normalized else (1.0, 1.0)
        G = self.G
        for j, s in enumerate(shapes or []):
            pts = s.get("points") or ()
            if (s.get("shape_type") == "rectangle" and len(pts) == 2):            # fast path (most labels)
                try:
                    (ax, ay), (bx, by) = pts
                    box = (min(ax, bx) * sx, min(ay, by) * sy, max(ax, bx) * sx, max(ay, by) * sy)
                    box = tuple(map(float, box))
                except (TypeError, ValueError):
                    continue
                if not all(np.isfinite(box)):
                    continue
                poly = None
            else:
                geo = _shape_geometry(s, sx, sy)
                if geo is None:
                    continue
                box, poly = geo
            G["img"].append(i)
            G["cls"].append(self._label(str(s.get("label", ""))))
            G["idx"].append(j)
            G["box"].append(box)
            G["pt0"].append(self.n_gpts)
            if poly is not None:
                self.gpts.append(poly)
                self.n_gpts += len(poly)
            G["pt1"].append(self.n_gpts)
        P = self.P
        for mid, preds in (preds_by_model or {}).items():
            mi = self.midx.get(mid)
            if mi is None:
                continue
            for p in preds or ():
                P["img"].append(i)
                P["cls"].append(self._label(str(p[0])))
                P["conf"].append(float(p[1]))
                P["model"].append(mi)
                P["box"].append(tuple(map(float, p[2])))
                P["pt0"].append(self.n_ppts)
                poly = p[3] if len(p) > 3 else None
                if poly is not None and len(poly) >= 3:
                    a = np.asarray(poly, np.float32).reshape(-1, 2)
                    self.ppts.append(a)
                    self.n_ppts += len(a)
                P["pt1"].append(self.n_ppts)
                P["k0"].append(self.n_kps)
                kp = p[4] if len(p) > 4 else None
                if kp is not None and len(kp):
                    a = np.asarray(kp, np.float32).reshape(-1, 3)
                    self.kps.append(a)
                    self.n_kps += len(a)
                P["k1"].append(self.n_kps)
        if len(self.keys) - self.done[0] >= self.CHUNK:
            self.flush()

    def flush(self):
        """Match the images added since the last flush."""
        i0, g0, p0 = self.done
        i1, g1, p1 = len(self.keys), len(self.G["img"]), len(self.P["img"])
        if i1 <= i0:
            return
        G, P = self.G, self.P
        gi = np.array(G["img"][g0:g1], np.int64) - i0
        pi = np.array(P["img"][p0:p1], np.int64) - i0
        r = _match_block(i1 - i0, gi, np.array(G["cls"][g0:g1], np.int64),
                         np.array(G["box"][g0:g1], np.float32).reshape(-1, 4), pi,
                         np.array(P["cls"][p0:p1], np.int64), np.array(P["conf"][p0:p1], np.float32),
                         np.array(P["model"][p0:p1], np.int64), np.array(P["box"][p0:p1], np.float32).reshape(-1, 4),
                         len(self.models))
        r["match"] = np.where(r["match"] >= 0, r["match"] + g0, -1).astype(np.int32)
        r["pr_g"] = (r["pr_g"] + g0).astype(np.int32)
        r["pr_p"] = (r["pr_p"] + p0).astype(np.int32)
        for k in ("pp_a", "pp_b"):
            if k in r:
                r[k] = (r[k] + p0).astype(np.int32)
        self.blocks.append(r)
        self.done = (i1, g1, p1)

    def finish(self) -> ExplorerData:
        self.flush()
        N, M = len(self.keys), len(self.models)
        G, P = self.G, self.P
        g_img = np.array(G["img"], np.int32)
        p_img = np.array(P["img"], np.int32)
        p_cls = np.array(P["cls"], np.int32)
        p_model = np.array(P["model"], np.int16)
        for label in self.coverage:
            self._label(label)
        cov = np.zeros((M, len(self.labels)), bool)
        for label, mids in self.coverage.items():
            for mid in mids:
                if mid in self.midx:
                    cov[self.midx[mid], self.lab[label]] = True
        if not self.coverage and len(p_cls):                     # unknown: a model covers what it predicted
            cov[p_model, p_cls] = True
        T = len(IOU_GRID)

        def cat(k, dtype, shape=()):
            parts = [b[k] for b in self.blocks if k in b]
            return np.concatenate(parts, axis=-1).astype(dtype) if parts else _empty(dtype, *shape)
        match = np.concatenate([b["match"] for b in self.blocks], axis=1) if self.blocks else \
            np.zeros((T, 0), np.int32)
        d = ExplorerData(
            keys=self.keys, labels=self.labels, models=self.models, model_names=self.model_names,
            model_colors=self.model_colors, splits=self.splits, folders=self.folders,
            img_w=np.array(self.W, np.float32), img_h=np.array(self.H, np.float32),
            img_split=np.array(self.S, np.int16), img_folder=np.array(self.F, np.int32),
            img_seen=np.array(self.seen, bool).reshape(N, M).T.copy() if N else np.zeros((M, 0), bool),
            cov=cov,
            img_g0=_starts(g_img, N), img_g1=_ends(g_img, N), img_p0=_starts(p_img, N), img_p1=_ends(p_img, N),
            g_img=g_img, g_cls=np.array(G["cls"], np.int32), g_idx=np.array(G["idx"], np.int32),
            g_box=np.array(G["box"], np.float32).reshape(-1, 4), g_alive=np.ones(len(g_img), bool),
            g_pt0=np.array(G["pt0"], np.int64), g_pt1=np.array(G["pt1"], np.int64),
            g_pts=np.concatenate(self.gpts).astype(np.float32) if self.gpts else _empty(np.float32, 2),
            p_img=p_img, p_cls=p_cls, p_conf=np.array(P["conf"], np.float32),
            p_model=p_model, p_box=np.array(P["box"], np.float32).reshape(-1, 4),
            p_pt0=np.array(P["pt0"], np.int64), p_pt1=np.array(P["pt1"], np.int64),
            p_pts=np.concatenate(self.ppts).astype(np.float32) if self.ppts else _empty(np.float32, 2),
            p_k0=np.array(P["k0"], np.int64), p_k1=np.array(P["k1"], np.int64),
            p_kps=np.concatenate(self.kps).astype(np.float32) if self.kps else _empty(np.float32, 3),
            match=match.astype(np.int32), p_bsame=cat("bsame", np.float32), p_bother=cat("bother", np.float32),
            pr_g=cat("pr_g", np.int32), pr_p=cat("pr_p", np.int32), pr_iou=cat("pr_iou", np.float32),
            pr_same=cat("pr_same", bool), pp_a=cat("pp_a", np.int32), pp_b=cat("pp_b", np.int32),
            pp_iou=cat("pp_iou", np.float32), meta=self.meta)
        self.blocks = []
        return d


def _starts(img, n):
    return np.searchsorted(img, np.arange(n), "left").astype(np.int64) if n else _empty(np.int64)


def _ends(img, n):
    return np.searchsorted(img, np.arange(n), "right").astype(np.int64) if n else _empty(np.int64)


PP_MIN_CONF = 0.1          # agreement between models is measured for detections from this confidence
_COMBO_CHUNK = 4_000_000   # pair candidates per numpy pass (memory bound)


def _combos(a_img, b_img, n):
    """All (a, b) row pairs of the same image; a_img / b_img sorted local image ids in [0, n)."""
    cnt_b = np.bincount(b_img, minlength=n)
    start_b = np.cumsum(cnt_b) - cnt_b
    cnt = cnt_b[a_img]
    total = int(cnt.sum())
    if not total:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    ai = np.repeat(np.arange(len(a_img), dtype=np.int64), cnt)
    first = np.cumsum(cnt) - cnt
    offs = np.arange(total, dtype=np.int64) - np.repeat(first, cnt)
    bi = np.repeat(start_b[a_img], cnt) + offs
    return ai, bi


def _pair_iou(A, B):
    ix = np.clip(np.minimum(A[:, 2], B[:, 2]) - np.maximum(A[:, 0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, 3], B[:, 3]) - np.maximum(A[:, 1], B[:, 1]), 0, None)
    inter = ix * iy
    ua = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-12), 0.0).astype(np.float32)


def _overlaps(a_img, a_box, b_img, b_box, n, min_iou, keep=None):
    """(a, b, iou) of same-image pairs with IoU ≥ min_iou, in chunks (bounded memory)."""
    A, B, U = [], [], []
    if not len(a_img) or not len(b_img):
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    cnt_b = np.bincount(b_img, minlength=n)
    per_a = cnt_b[a_img]
    bounds = [0]
    acc = 0
    for j, c in enumerate(np.add.reduceat(per_a, np.arange(0, len(per_a), 4096)) if len(per_a) else []):
        acc += int(c)
        if acc >= _COMBO_CHUNK:
            bounds.append(min(len(a_img), (j + 1) * 4096))
            acc = 0
    bounds.append(len(a_img))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:
            continue
        ai, bi = _combos(a_img[lo:hi], b_img, n)
        ai += lo
        if keep is not None and len(ai):
            k = keep(ai, bi)
            ai, bi = ai[k], bi[k]
        if not len(ai):
            continue
        u = _pair_iou(a_box[ai], b_box[bi])
        k = u >= min_iou
        A.append(ai[k]), B.append(bi[k]), U.append(u[k])
    if not A:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(A), np.concatenate(B), np.concatenate(U)


def _match_block(n, g_img, g_cls, g_box, p_img, p_cls, p_conf, p_model, p_box, n_models, pp=True):
    """Matching of one block of images (local ids 0..n-1, rows sorted by image). Every model on its own, COCO
    greedy (most confident first) at every IoU of the grid — vectorised across images: round r handles the r-th
    most confident prediction of every (image, model) at once (they never compete for the same label)."""
    T, nG, nP = len(IOU_GRID), len(g_img), len(p_img)
    match = np.full((T, nP), -1, np.int64)
    bsame = np.zeros(nP, np.float32)
    bother = np.zeros(nP, np.float32)
    pg, pp_, piou = _overlaps(p_img, p_box, g_img, g_box, n, 0.1)        # (pred, label, iou)
    same = p_cls[pg] == g_cls[pp_] if len(pg) else np.zeros(0, bool)
    if len(pg):
        np.maximum.at(bsame, pg[same], piou[same])
        np.maximum.at(bother, pg[~same], piou[~same])
    if len(pg) and same.any():
        sp, sg, su = pg[same], pp_[same], piou[same]
        o = np.lexsort((-p_conf, p_model, p_img))                       # rank inside (image, model)
        grp = np.r_[True, (p_img[o][1:] != p_img[o][:-1]) | (p_model[o][1:] != p_model[o][:-1])] if nP else []
        gstart = np.maximum.accumulate(np.where(grp, np.arange(nP), 0)) if nP else []
        rank = np.empty(nP, np.int64)
        rank[o] = np.arange(nP) - gstart
        for m in range(n_models):
            k = p_model[sp] == m
            if not k.any():
                continue
            a, g, u = sp[k], sg[k], su[k]
            r = rank[a]
            o2 = np.lexsort((-u, a, r))
            a, g, u, r = a[o2], g[o2], u[o2], r[o2]
            cuts = np.searchsorted(r, np.arange(int(r.max()) + 2))
            taken = np.zeros((T, nG), bool)
            for ri in range(len(cuts) - 1):
                lo, hi = cuts[ri], cuts[ri + 1]
                if hi <= lo:
                    continue
                aa, gg, uu = a[lo:hi], g[lo:hi], u[lo:hi]
                for t in range(T):
                    valid = (uu >= IOU_GRID[t]) & ~taken[t, gg]
                    idx = np.nonzero(valid)[0]
                    if not len(idx):
                        if IOU_GRID[t] > uu.max():
                            break                                   # higher thresholds: nothing either
                        continue
                    _u, first = np.unique(aa[idx], return_index=True)
                    ch = idx[first]
                    match[t, aa[ch]] = gg[ch]
                    taken[t, gg[ch]] = True
    out = {"match": match, "bsame": bsame, "bother": bother, "pr_p": pg, "pr_g": pp_, "pr_iou": piou,
           "pr_same": same}
    if pp and n_models >= 2 and nP:
        strong = np.nonzero(p_conf >= PP_MIN_CONF)[0]
        si, sm = p_img[strong], p_model[strong]

        def keep(ai, bi):
            return (bi > ai) & (sm[ai] != sm[bi]) & (p_cls[strong][ai] == p_cls[strong][bi])
        a, b, u = _overlaps(si, p_box[strong], si, p_box[strong], n, 0.3, keep)
        out.update(pp_a=strong[a], pp_b=strong[b], pp_iou=u)
    return out


def _rematch(d: ExplorerData, imgs):
    """Re-match the predictions of some images against their (new) labels."""
    imgs = sorted(set(int(i) for i in imgs))
    gr = [d.gt_rows(i) for i in imgs]
    pr = [d.pred_rows(i) for i in imgs]
    G = np.concatenate(gr) if gr else np.zeros(0, np.int64)
    P = np.concatenate(pr) if pr else np.zeros(0, np.int64)
    gl = np.repeat(np.arange(len(imgs)), [len(x) for x in gr])
    pl = np.repeat(np.arange(len(imgs)), [len(x) for x in pr])
    if len(d.pr_g):                                            # drop old overlaps of these predictions
        keep = ~np.isin(d.pr_p, P)
        d.pr_g, d.pr_p, d.pr_iou, d.pr_same = d.pr_g[keep], d.pr_p[keep], d.pr_iou[keep], d.pr_same[keep]
    if not len(P):
        return
    r = _match_block(len(imgs), gl, d.g_cls[G].astype(np.int64), d.g_box[G], pl, d.p_cls[P].astype(np.int64),
                     d.p_conf[P], d.p_model[P].astype(np.int64), d.p_box[P], d.n_models, pp=False)
    d.match[:, P] = np.where(r["match"] >= 0, G[np.maximum(r["match"], 0)], -1)
    d.p_bsame[P], d.p_bother[P] = r["bsame"], r["bother"]
    if len(r["pr_g"]):
        d.pr_g = np.concatenate([d.pr_g, G[r["pr_g"]].astype(np.int32)])
        d.pr_p = np.concatenate([d.pr_p, P[r["pr_p"]].astype(np.int32)])
        d.pr_iou = np.concatenate([d.pr_iou, r["pr_iou"]])
        d.pr_same = np.concatenate([d.pr_same, r["pr_same"]])


def replace_gt(d: ExplorerData, updates: dict):
    """updates: {key: (shapes, normalized, w, h)} -> the labels of these images are replaced and re-matched."""
    imgs, G, pts = [], {k: [] for k in ("img", "cls", "idx", "box", "pt0", "pt1")}, []
    base_g, base_pt = len(d.g_img), len(d.g_pts)
    n_pt = base_pt
    for key, (shapes, normalized, w, h) in updates.items():
        i = d.index_of(key)
        if i is None:
            continue
        imgs.append(i)
        d.g_alive[d.img_g0[i]:d.img_g1[i]] = False
        if w:
            d.img_w[i], d.img_h[i] = w, h
        W, H = float(d.img_w[i] or 1), float(d.img_h[i] or 1)
        sx, sy = (W, H) if normalized else (1.0, 1.0)
        start = base_g + len(G["img"])
        for j, s in enumerate(shapes or []):
            geo = _shape_geometry(s, sx, sy)
            if geo is None:
                continue
            box, poly = geo
            G["img"].append(i)
            G["cls"].append(d.label_id(str(s.get("label", "")), create=True))
            G["idx"].append(j)
            G["box"].append(box)
            G["pt0"].append(n_pt)
            if poly is not None:
                pts.append(poly)
                n_pt += len(poly)
            G["pt1"].append(n_pt)
        d.img_g0[i], d.img_g1[i] = start, base_g + len(G["img"])
    if not imgs:
        return []
    d.g_img = np.concatenate([d.g_img, np.array(G["img"], np.int32)])
    d.g_cls = np.concatenate([d.g_cls, np.array(G["cls"], np.int32)])
    d.g_idx = np.concatenate([d.g_idx, np.array(G["idx"], np.int32)])
    d.g_box = np.concatenate([d.g_box, np.array(G["box"], np.float32).reshape(-1, 4)])
    d.g_alive = np.concatenate([d.g_alive, np.ones(len(G["img"]), bool)])
    d.g_pt0 = np.concatenate([d.g_pt0, np.array(G["pt0"], np.int64)])
    d.g_pt1 = np.concatenate([d.g_pt1, np.array(G["pt1"], np.int64)])
    if pts:
        d.g_pts = np.concatenate([d.g_pts] + pts).astype(np.float32)
    if len(d.pr_g):                                           # overlaps of the replaced labels are dropped
        keep = d.g_alive[d.pr_g]
        d.pr_g, d.pr_p, d.pr_iou, d.pr_same = d.pr_g[keep], d.pr_p[keep], d.pr_iou[keep], d.pr_same[keep]
    _rematch(d, imgs)
    return imgs


def compact(d: ExplorerData) -> ExplorerData:
    """Drop dead label rows (after many edits; before saving)."""
    if d.g_alive.all():
        return d
    alive = np.nonzero(d.g_alive)[0]
    order = alive[np.lexsort((alive, d.g_img[alive]))]
    remap = np.full(len(d.g_img), -1, np.int64)
    remap[order] = np.arange(len(order))
    pts, n, p0, p1 = [], 0, [], []
    for g in order:
        a, b = d.g_pt0[g], d.g_pt1[g]
        p0.append(n)
        if b > a:
            pts.append(d.g_pts[a:b])
            n += b - a
        p1.append(n)
    d.g_img, d.g_cls, d.g_idx, d.g_box = d.g_img[order], d.g_cls[order], d.g_idx[order], d.g_box[order]
    d.g_alive = np.ones(len(order), bool)
    d.g_pt0, d.g_pt1 = np.array(p0, np.int64), np.array(p1, np.int64)
    d.g_pts = np.concatenate(pts).astype(np.float32) if pts else _empty(np.float32, 2)
    d.match = np.where(d.match >= 0, remap[np.maximum(d.match, 0)], -1).astype(np.int32)
    if len(d.pr_g):
        d.pr_g = remap[d.pr_g].astype(np.int32)
    d.img_g0, d.img_g1 = _starts(d.g_img, len(d)), _ends(d.g_img, len(d))
    return d


# =================================================================================================================
# Saving
# =================================================================================================================
_ARRAYS = ("img_w", "img_h", "img_split", "img_folder", "img_seen", "cov", "img_g0", "img_g1", "img_p0", "img_p1",
           "g_img", "g_cls", "g_idx", "g_box", "g_alive", "g_pt0", "g_pt1", "g_pts", "p_img", "p_cls", "p_conf",
           "p_model", "p_box", "p_pt0", "p_pt1", "p_pts", "p_k0", "p_k1", "p_kps", "match", "p_bsame", "p_bother",
           "pr_g", "pr_p", "pr_iou", "pr_same", "pp_a", "pp_b", "pp_iou")


def save(d: ExplorerData, path):
    """Write d (dead label rows dropped in the written copy; d itself is left as it is — row ids stay valid)."""
    import copy
    d = compact(copy.copy(d))
    info = {"format": FORMAT, "keys": d.keys, "labels": d.labels, "models": d.models, "model_names": d.model_names,
            "model_colors": d.model_colors, "splits": d.splits, "folders": d.folders, "meta": d.meta}
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, info=np.frombuffer(json.dumps(info).encode(), np.uint8),
                        **{k: getattr(d, k) for k in _ARRAYS})
    os.replace(tmp, path)


def load(path) -> ExplorerData | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            info = json.loads(bytes(z["info"]).decode())
            if info.get("format") != FORMAT:
                return None
            arrays = {k: z[k] for k in _ARRAYS}
    except (OSError, ValueError, KeyError):
        return None
    return ExplorerData(keys=info["keys"], labels=info["labels"], models=info["models"],
                        model_names=info["model_names"], model_colors=info["model_colors"], splits=info["splits"],
                        folders=info["folders"], meta=info.get("meta") or {}, **arrays)


# =================================================================================================================
# A view: what is visible, filtered and sorted
# =================================================================================================================
@dataclass
class View:
    models: dict = field(default_factory=dict)       # id -> {"visible": bool, "conf": float}
    show_gt: bool = True
    iou: float = 0.5
    classes: list | None = None                       # label names; None = all
    tags: list | None = None                          # tag names; None = all
    splits: list | None = None
    folders: list | None = None
    review: list | None = None                        # "flagged" / "ok" / "fixed" / "none"
    seen: str = "all"                                 # all / seen / unseen (by a visible model)
    search: str = ""
    keys: list | None = None                          # an explicit selection (e.g. from the Analysis)
    keys_title: str = ""
    sort: str = "errors"
    desc: bool = True
    only_matching: bool = True                        # draw only the boxes the class / tag filter selects
    band: float = 0.15                                # "uncertain": |conf - threshold| < band
    seed: int = 0
    color: str = "match"                              # drawing colours: match result / class / model
    tile_names: bool = False                          # class names on the grid tiles

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, v: dict) -> "View":
        known = {k: v[k] for k in cls.__dataclass_fields__ if k in (v or {})}
        return cls(**known)

    def conf_of(self, mid, default=0.25) -> float:
        return float((self.models.get(mid) or {}).get("conf", default))

    def visible(self, mid) -> bool:
        return bool((self.models.get(mid) or {}).get("visible", True))


@dataclass
class State:
    vis: np.ndarray            # [n_models] visible
    conf: np.ndarray           # [n_models] threshold
    live: np.ndarray           # predictions drawn (visible model, conf ≥ its threshold)
    p_tag: np.ndarray          # tag per prediction (HIDDEN when not live)
    p_gt: np.ndarray           # matched label row or -1
    p_iou: np.ndarray          # IoU with the matched label (0 when unmatched)
    p_disagree: np.ndarray     # live and some other visible model has no live same-class box on it
    g_tag: np.ndarray          # tag per label row (dead rows: NA)
    g_show: np.ndarray         # label rows drawn (alive, class filter, tag filter)
    p_show: np.ndarray         # prediction rows drawn
    counts: np.ndarray         # [N, 8] per image: number of drawn objects per tag (labels + predictions)
    order: np.ndarray          # image indices in view order
    min_iou: np.ndarray        # [N] worst matched IoU (1 = none)
    min_conf: np.ndarray       # [N] lowest live confidence (1 = none)
    uncertain: np.ndarray      # [N]
    disagree: np.ndarray       # [N]
    class_counts: dict         # label -> (labels drawn, predictions drawn)
    tag_counts: dict           # tag -> objects drawn
    n_images: int              # images matching every filter


def evaluate(d: ExplorerData, v: View, review: dict | None = None) -> State:
    """review: {key: "flag" | "ok" | "fixed"} (only needed with a review filter)."""
    M, N = d.n_models, len(d)
    vis = np.array([v.visible(m) for m in d.models], bool) if M else np.zeros(0, bool)
    conf = np.array([v.conf_of(m) for m in d.models], np.float32) if M else np.zeros(0, np.float32)
    nP, nG = len(d.p_img), len(d.g_img)
    live = vis[d.p_model] & (d.p_conf >= conf[d.p_model]) if nP else np.zeros(0, bool)
    ti = int(np.argmin(np.abs(IOU_GRID - v.iou)))
    m = d.match[ti] if nP else np.zeros(0, np.int32)
    ok = live & (m >= 0)
    if ok.any():
        ok[ok] = d.g_alive[m[ok]]
    p_gt = np.where(ok, m, -1)
    matched = np.zeros(nG, bool)
    matched[p_gt[ok]] = True
    p_iou = np.zeros(nP, np.float32)
    if ok.any():
        gb, pb = d.g_box[p_gt[ok]], d.p_box[ok]
        ix = np.clip(np.minimum(gb[:, 2], pb[:, 2]) - np.maximum(gb[:, 0], pb[:, 0]), 0, None)
        iy = np.clip(np.minimum(gb[:, 3], pb[:, 3]) - np.maximum(gb[:, 1], pb[:, 1]), 0, None)
        inter = ix * iy
        ua = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]) + (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1]) - inter
        p_iou[ok] = inter / np.maximum(ua, 1e-9)
    # prediction tags
    p_tag = np.full(nP, HIDDEN, np.int8)
    p_tag[ok] = OK
    fp = live & ~ok
    s, o = d.p_bsame, d.p_bother
    p_tag[fp & (s >= v.iou)] = DUPE
    p_tag[fp & (s < v.iou) & (o >= v.iou)] = CLASS
    p_tag[fp & (s < v.iou) & (o < v.iou) & (s >= 0.1)] = FIT
    p_tag[fp & (s < 0.1) & (o < v.iou)] = EXTRA
    # label tags
    covered = (d.cov[vis].any(0) if M and vis.any() else np.zeros(len(d.labels), bool))
    g_tag = np.full(nG, NA, np.int8)
    if nG:
        cov_g = covered[d.g_cls] & d.g_alive
        g_tag[cov_g] = MISSED
        if len(d.pr_g):
            lp = live[d.pr_p] & d.g_alive[d.pr_g]
            best_same = np.zeros(nG, np.float32)
            best_other = np.zeros(nG, np.float32)
            sm = lp & d.pr_same
            om = lp & ~d.pr_same
            np.maximum.at(best_same, d.pr_g[sm], d.pr_iou[sm])
            np.maximum.at(best_other, d.pr_g[om], d.pr_iou[om])
            g_tag[cov_g & (best_same >= 0.1)] = FIT
            g_tag[cov_g & (best_same < v.iou) & (best_other >= v.iou)] = CLASS
        g_tag[matched] = OK
    # disagreement between visible models
    p_dis = np.zeros(nP, bool)
    if M >= 2 and vis.sum() >= 2 and nP:
        has = np.zeros((nP, M), bool)
        if len(d.pp_a):
            both = live[d.pp_a] & live[d.pp_b]
            a, b = d.pp_a[both], d.pp_b[both]
            has[a, d.p_model[b]] = True
            has[b, d.p_model[a]] = True
        has[np.arange(nP), d.p_model] = True
        p_dis = live & ~has[:, vis].all(1)
    # filters on objects
    cls_ok_g = np.ones(nG, bool)
    cls_ok_p = np.ones(nP, bool)
    if v.classes is not None:
        want = np.zeros(len(d.labels), bool)
        for c in v.classes:
            i = d.label_id(c)
            if i is not None:
                want[i] = True
        cls_ok_g, cls_ok_p = want[d.g_cls] if nG else cls_ok_g, want[d.p_cls] if nP else cls_ok_p
    tag_ok_g = np.ones(nG, bool)
    tag_ok_p = np.ones(nP, bool)
    if v.tags is not None:
        want_t = np.zeros(9, bool)
        dis_wanted = "models disagree" in v.tags
        for t in v.tags:
            if t in TAG_BY_NAME:
                want_t[TAG_BY_NAME[t]] = True
        tag_ok_g = want_t[g_tag] if nG else tag_ok_g
        tag_ok_p = (want_t[p_tag] | (dis_wanted & p_dis)) if nP else tag_ok_p
    g_base = d.g_alive & bool(v.show_gt) if nG else np.zeros(0, bool)
    g_sel = g_base & cls_ok_g & tag_ok_g
    p_sel = live & cls_ok_p & tag_ok_p
    obj_filter = v.classes is not None or v.tags is not None
    g_show = g_sel if (v.only_matching and obj_filter) else g_base
    p_show = p_sel if (v.only_matching and obj_filter) else live
    # per-image counts (of the drawn objects)
    counts = np.zeros((N, 8), np.int32)
    for t in range(8):
        if nG:
            gm = g_show & (g_tag == t)
            if gm.any():
                counts[:, t] += np.bincount(d.g_img[gm], minlength=N)
        if nP:
            pm = p_show & (p_tag == t)
            if pm.any():
                counts[:, t] += np.bincount(d.p_img[pm], minlength=N)
    min_iou = np.ones(N, np.float32)
    min_conf = np.ones(N, np.float32)
    unc = np.zeros(N, np.int32)
    dis = np.zeros(N, np.int32)
    if nP:
        t = ok & p_show
        np.minimum.at(min_iou, d.p_img[t], p_iou[t])
        np.minimum.at(min_conf, d.p_img[p_show], d.p_conf[p_show])
        near = vis[d.p_model] & (np.abs(d.p_conf - conf[d.p_model]) < v.band) & cls_ok_p
        if near.any():
            unc = np.bincount(d.p_img[near], minlength=N).astype(np.int32)
        if p_dis.any():
            dm = p_dis & p_show
            dis = np.bincount(d.p_img[dm], minlength=N).astype(np.int32) if dm.any() else dis
    # images
    img_ok = np.ones(N, bool)
    if obj_filter:
        hit = np.zeros(N, bool)
        if nG:
            hit[d.g_img[g_sel]] = True
        if nP:
            hit[d.p_img[p_sel]] = True
        img_ok &= hit
    if v.splits is not None:
        want_s = np.array([s in v.splits for s in d.splits], bool)
        img_ok &= want_s[d.img_split] if N else img_ok
    if v.folders is not None:
        want_f = np.array([f in v.folders for f in d.folders], bool)
        img_ok &= want_f[d.img_folder] if N else img_ok
    if v.seen != "all" and M and vis.any():
        s_any = d.img_seen[vis].any(0)
        img_ok &= s_any if v.seen == "seen" else ~s_any
    if v.review is not None:
        review = review or {}
        want_r = set(v.review)
        st = np.array([{"flag": "flagged", "ok": "ok", "fixed": "fixed"}.get(review.get(k), "none") in want_r
                       for k in d.keys], bool)
        img_ok &= st
    if v.search:
        q = v.search.lower()
        img_ok &= np.array([q in k.lower() for k in d.keys], bool)
    if v.keys is not None:
        sel = np.zeros(N, bool)
        idx = [d.index_of(k) for k in v.keys]
        sel[[i for i in idx if i is not None]] = True
        img_ok &= sel
    idx = np.nonzero(img_ok)[0]
    order = _sort(d, idx, v, counts, min_iou, min_conf, unc, dis)
    # sidebar counts (objects drawn, ignoring the class filter for the class list, the tag filter for the tags)
    class_counts = {}
    if len(d.labels):
        gl = np.bincount(d.g_cls[g_base & tag_ok_g], minlength=len(d.labels)) if nG else np.zeros(len(d.labels), int)
        pl = np.bincount(d.p_cls[live & tag_ok_p], minlength=len(d.labels)) if nP else np.zeros(len(d.labels), int)
        class_counts = {l: (int(gl[i]), int(pl[i])) for i, l in enumerate(d.labels)}
    tag_counts = {}
    for t in range(7):
        n = int((g_base & cls_ok_g & (g_tag == t)).sum()) if nG else 0
        n += int((live & cls_ok_p & (p_tag == t)).sum()) if nP else 0
        tag_counts[TAG_NAMES[t]] = n
    tag_counts["models disagree"] = int((p_dis & cls_ok_p).sum())
    return State(vis, conf, live, p_tag, p_gt, p_iou, p_dis, g_tag, g_show, p_show, counts, order, min_iou,
                 min_conf, unc, dis, class_counts, tag_counts, len(idx))


def _sort(d, idx, v, counts, min_iou, min_conf, unc, dis):
    if not len(idx):
        return idx.astype(np.int64)
    k = v.sort
    errs = counts[:, list(ERROR_TAGS)].sum(1)
    if k == "scan":
        return idx if not v.desc else idx[::-1]
    if k == "random":
        rng = np.random.default_rng(v.seed)
        return idx[rng.permutation(len(idx))]
    if k == "name":
        names = np.array([d.keys[i].lower() for i in idx], object)
        o = np.argsort(names, kind="stable")
        return idx[o[::-1]] if v.desc else idx[o]
    key = {"errors": errs, "missed": counts[:, MISSED], "extra": counts[:, EXTRA], "class": counts[:, CLASS],
           "fit": -min_iou, "uncertain": unc, "disagree": dis, "objects": counts.sum(1),
           "low_conf": -min_conf}.get(k, errs)[idx].astype(float)
    tie = -errs[idx].astype(float)                                 # ties: more errors first, then scan order
    o = np.lexsort((idx, tie, -key if v.desc else key))
    return idx[o]


# =================================================================================================================
# One image: the objects to draw
# =================================================================================================================
@dataclass
class Obj:
    kind: str                  # "gt" | "pred"
    row: int                   # global row (label row / prediction row)
    label: str
    box: tuple
    tag: int
    conf: float = 1.0
    model: str = ""
    poly: np.ndarray | None = None
    kps: np.ndarray | None = None
    shape_index: int | None = None
    iou: float = 0.0
    partner: int = -1          # matched label row (prediction) / matching prediction row (label)
    disagree: bool = False


def objects(d: ExplorerData, st: State, i: int, everything: bool = False) -> list[Obj]:
    """Drawn objects of image i (labels first). everything=True: also hidden predictions and filtered-out ones."""
    out = []
    gr = d.gt_rows(i)
    partner = {}
    pr = d.pred_rows(i)
    for p in pr:
        g = st.p_gt[p]
        if g >= 0 and (g not in partner or d.p_conf[p] > d.p_conf[partner[g]]):
            partner[g] = p
    for g in gr:
        if not everything and not st.g_show[g]:
            continue
        out.append(Obj("gt", int(g), d.labels[d.g_cls[g]], tuple(map(float, d.g_box[g])), int(st.g_tag[g]),
                       poly=d.g_poly(g), shape_index=int(d.g_idx[g]), partner=int(partner.get(g, -1))))
    for p in pr:
        if not everything and not st.p_show[p]:
            continue
        out.append(Obj("pred", int(p), d.labels[d.p_cls[p]], tuple(map(float, d.p_box[p])), int(st.p_tag[p]),
                       float(d.p_conf[p]), d.models[d.p_model[p]], d.p_poly(p), d.p_keypoints(p),
                       iou=float(st.p_iou[p]), partner=int(st.p_gt[p]), disagree=bool(st.p_disagree[p])))
    return out


def suggestion(d: ExplorerData, st: State, o: Obj, iou_thr: float) -> tuple | None:
    """What accepting this object would do: ("add", label, box) / ("relabel", g, label) / ("reshape", g, box) /
    ("delete", g) — None when there is nothing to do."""
    if o.kind == "pred":
        if o.tag == EXTRA:
            return ("add", o.label, o.box)
        i = d.p_img[o.row]
        gr = d.gt_rows(i)
        if not len(gr):
            return ("add", o.label, o.box)
        ious = iou_matrix(np.array([o.box], float), d.g_box[gr].astype(float))[0]
        if o.tag == CLASS:
            other = d.g_cls[gr] != d.p_cls[o.row]
            if (ious * other).max() >= iou_thr:
                g = int(gr[int(np.argmax(ious * other))])
                return ("relabel", g, o.label)
        if o.tag == FIT:
            same = d.g_cls[gr] == d.p_cls[o.row]
            if (ious * same).max() >= 0.1:
                g = int(gr[int(np.argmax(ious * same))])
                return ("reshape", g, o.box)
        return None
    if o.tag == MISSED:
        return ("delete", o.row)
    return None


def fix_op(d: ExplorerData, sug: tuple):
    """A suggestion -> (key, session.apply_fixes op with relative boxes)."""
    kind = sug[0]
    if kind == "add":
        return None
    g = sug[1]
    i = int(d.g_img[g])
    W, H = float(d.img_w[i] or 1), float(d.img_h[i] or 1)
    gb = d.g_box[g]
    rel = (float(gb[0]) / W, float(gb[1]) / H, float(gb[2]) / W, float(gb[3]) / H)
    if kind == "relabel":
        return d.keys[i], ("relabel", int(d.g_idx[g]), rel, sug[2])
    if kind == "reshape":
        b = sug[2]
        return d.keys[i], ("reshape", int(d.g_idx[g]), rel, (b[0] / W, b[1] / H, b[2] / W, b[3] / H))
    if kind == "delete":
        return d.keys[i], ("delete", int(d.g_idx[g]), rel)
    return None


def add_op(d: ExplorerData, i: int, label: str, box) -> tuple:
    W, H = float(d.img_w[i] or 1), float(d.img_h[i] or 1)
    return d.keys[i], ("add", label, (box[0] / W, box[1] / H, box[2] / W, box[3] / H))


def op_for(d: ExplorerData, st: State, o: Obj, iou_thr: float):
    """(key, op) that accepts object o, or None."""
    s = suggestion(d, st, o, iou_thr)
    if s is None:
        return None
    if s[0] == "add":
        return add_op(d, int(d.p_img[o.row]), s[1], s[2])
    return fix_op(d, s)


def bulk_ops(d: ExplorerData, st: State, imgs, kinds: set, min_conf: float, iou_thr: float) -> dict:
    """{key: [op]} accepting the visible model's suggestions of these images: kinds ⊆ {"add", "relabel",
    "reshape"} (deleting labels is never done in bulk). Per label only the most confident suggestion counts."""
    out: dict = {}
    for i in imgs:
        used_g = set()
        cands = []
        for p in d.pred_rows(i):
            if not st.p_show[p] or d.p_conf[p] < min_conf or st.p_tag[p] not in (EXTRA, CLASS, FIT):
                continue
            cands.append(p)
        cands.sort(key=lambda p: -d.p_conf[p])
        added = []
        for p in cands:
            o = Obj("pred", int(p), d.labels[d.p_cls[p]], tuple(map(float, d.p_box[p])), int(st.p_tag[p]),
                    float(d.p_conf[p]), d.models[d.p_model[p]])
            s = suggestion(d, st, o, iou_thr)
            if s is None or s[0] not in kinds:
                continue
            if s[0] == "add":
                b = np.array([s[2]], float)
                if added and iou_matrix(b, np.array(added, float)).max() >= 0.5:
                    continue                                       # two models found the same unlabeled object
                added.append(s[2])
                k, op = add_op(d, int(i), s[1], s[2])
            else:
                if s[1] in used_g:
                    continue
                used_g.add(s[1])
                k, op = fix_op(d, s)
            out.setdefault(k, []).append(op)
    return out


# =================================================================================================================
# Patches (objects as crops)
# =================================================================================================================
def patch_rows(d: ExplorerData, st: State, source: str, imgs=None, sort: str = "class") -> np.ndarray:
    """Encoded object ids in display order: label rows as g, prediction rows as -(p + 1).
    source: "gt" | "pred" | "errors" (every drawn object with an error tag)."""
    img_ok = None
    if imgs is not None:
        img_ok = np.zeros(len(d), bool)
        img_ok[np.asarray(imgs, np.int64)] = True
    parts = []
    if source in ("gt", "errors"):
        g = st.g_show.copy()
        if source == "errors":
            g &= np.isin(st.g_tag, ERROR_TAGS)
        if img_ok is not None:
            g &= img_ok[d.g_img]
        gi = np.nonzero(g)[0]
        parts.append((gi.astype(np.int64), d.g_cls[gi], np.ones(len(gi), np.float32), st.g_tag[gi], d.g_img[gi]))
    if source in ("pred", "errors"):
        p = st.p_show.copy()
        if source == "errors":
            p &= np.isin(st.p_tag, ERROR_TAGS)
        if img_ok is not None:
            p &= img_ok[d.p_img]
        pi = np.nonzero(p)[0]
        parts.append((-(pi.astype(np.int64) + 1), d.p_cls[pi], d.p_conf[pi], st.p_tag[pi], d.p_img[pi]))
    if not parts:
        return np.zeros(0, np.int64)
    ids = np.concatenate([x[0] for x in parts])
    cls = np.concatenate([x[1] for x in parts])
    conf = np.concatenate([x[2] for x in parts])
    tag = np.concatenate([x[3] for x in parts])
    img = np.concatenate([x[4] for x in parts])
    if sort == "conf":
        o = np.lexsort((img, conf))
    elif sort == "tag":
        o = np.lexsort((conf, tag))
    elif sort == "image":
        o = np.lexsort((ids, img))
    else:                                                          # class, then least confident first
        o = np.lexsort((conf, cls))
    return ids[o]


def decode_id(code: int) -> tuple[str, int]:
    return ("gt", int(code)) if code >= 0 else ("pred", int(-code - 1))


# =================================================================================================================
# Saved views
# =================================================================================================================
def views_path(root) -> str:
    return os.path.join(str(root), ".aio_qc", "explorer_views.json")


def load_views(root) -> dict:
    try:
        with open(views_path(root), encoding="utf-8") as f:
            v = json.load(f)
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def save_views(root, views: dict):
    p = views_path(root)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(views, f, indent=1)
    os.replace(tmp, p)


def data_path(root) -> str:
    return os.path.join(str(root), ".aio_qc", "explorer.npz")


def new_seed() -> int:
    return random.randrange(1, 2 ** 31)


# =================================================================================================================
# Thresholds: confidence histogram and precision / recall of one model (optionally one class)
# =================================================================================================================
HIST_BINS = 50


def curves(d: ExplorerData, model: int, cls: int | None, iou: float, bins: int = HIST_BINS) -> dict:
    """{"tp", "fp": per-bin counts over confidence 0..1, "npos": labels the model could find,
        "conf", "precision", "recall": the curve (predictions most confident first), "ap", "best_conf", "best_f1"}
    TP = matched to a live label at this IoU (the model's own greedy matching)."""
    ti = int(np.argmin(np.abs(IOU_GRID - iou)))
    sel = d.p_model == model if len(d.p_model) else np.zeros(0, bool)
    if cls is not None:
        sel = sel & (d.p_cls == cls)
    p = np.nonzero(sel)[0]
    m = d.match[ti][p] if len(p) else np.zeros(0, np.int32)
    tp = (m >= 0) & d.g_alive[np.maximum(m, 0)] if len(p) else np.zeros(0, bool)
    conf = d.p_conf[p]
    edges = np.linspace(0, 1, bins + 1)
    h_tp = np.histogram(conf[tp], edges)[0]
    h_fp = np.histogram(conf[~tp], edges)[0]
    g = d.g_alive.copy()
    if len(g):
        g &= d.cov[model][d.g_cls] if model < d.cov.shape[0] else False
        if cls is not None:
            g &= d.g_cls == cls
    npos = int(g.sum())
    out = {"tp": h_tp, "fp": h_fp, "edges": edges, "npos": npos, "n": int(len(p)),
           "conf": np.zeros(0), "precision": np.zeros(0), "recall": np.zeros(0), "ap": float("nan"),
           "best_conf": float("nan"), "best_f1": float("nan")}
    if not len(p):
        return out
    o = np.argsort(-conf, kind="stable")
    t = tp[o].astype(np.float64)
    ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    rec = ctp / npos if npos else np.zeros_like(ctp)
    out.update(conf=conf[o], precision=prec, recall=rec)
    if npos:
        env = np.maximum.accumulate(prec[::-1])[::-1]
        grid = np.linspace(0, 1, 101)
        idx = np.searchsorted(rec, grid, side="left")
        out["ap"] = float(np.mean(np.where(idx < len(env), env[np.minimum(idx, len(env) - 1)], 0.0)))
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        k = int(np.argmax(f1))
        out["best_conf"], out["best_f1"] = float(conf[o][k]), float(f1[k])
    return out


def at_threshold(c: dict, thr: float) -> tuple[float, float, int]:
    """(precision, recall, predictions kept) of a curve at confidence ≥ thr."""
    n = int(np.searchsorted(-c["conf"], -thr, side="right")) if len(c["conf"]) else 0
    if not n:
        return float("nan"), 0.0 if c["npos"] else float("nan"), 0
    return float(c["precision"][n - 1]), float(c["recall"][n - 1]) if c["npos"] else float("nan"), n


# =================================================================================================================
# Embedding map: the label objects that Visual Similarity already embedded, placed on a 2-D map
# =================================================================================================================
def parse_embed_id(eid: str):
    """'<rel>|<shape_type>:<x:y,x:y…>|<image sig>' -> (rel, points (k, 2)) or None (whole-image ids)."""
    try:
        rel, geom, _sig = eid.rsplit("|", 2)
        st, pts = geom.split(":", 1)
        xy = np.array([[float(a) for a in p.split(":")] for p in pts.split(",") if p], np.float32)
    except ValueError:
        return None
    if st == "image" or not len(xy):
        return None
    if st == "circle" and len(xy) == 2:
        r = float(np.hypot(*(xy[1] - xy[0])))
        xy = np.array([xy[0] - r, xy[0] + r], np.float32)
    return rel, xy


def embedded_rows(d: ExplorerData, root, ids, max_n: int = 50_000, seed: int = 0):
    """Match cached object embeddings to label rows of the explorer (same file, same box) ->
    (cache positions, label rows). At most max_n (random sample, reproducible)."""
    by_img: dict = {}
    for k in d.keys:
        try:
            rel = os.path.relpath(k, str(root)).replace("\\", "/")
        except ValueError:
            rel = k
        by_img[rel] = d.index_of(k)
    pos, rows = [], []
    order = np.arange(len(ids))
    if len(ids) > max_n * 3:
        order = np.random.default_rng(seed).permutation(len(ids))
    for j in order:
        got = parse_embed_id(str(ids[j]))
        if got is None:
            continue
        i = by_img.get(got[0])
        if i is None:
            continue
        gr = d.gt_rows(i)
        if not len(gr):
            continue
        xy = got[1]
        W, H = float(d.img_w[i] or 1), float(d.img_h[i] or 1)
        if xy.max() <= 1.5 and max(W, H) > 4:                        # normalised (YOLO) points
            xy = xy * (W, H)
        box = np.array([[xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()]], float)
        ious = iou_matrix(box, d.g_box[gr].astype(float))[0]
        b = int(np.argmax(ious))
        if ious[b] >= 0.9:
            pos.append(int(j))
            rows.append(int(gr[b]))
            if len(pos) >= max_n:
                break
    return np.array(pos, np.int64), np.array(rows, np.int64)


def project(vecs: np.ndarray, method: str = "pca", seed: int = 0, progress=None) -> tuple[np.ndarray, str]:
    """(n, 2) map coordinates scaled to 0..1 + the method actually used. pca (instant) · tsne (scikit-learn) ·
    umap (umap-learn), falling back to PCA when the library is missing."""
    x = np.asarray(vecs, np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    x = x - x.mean(0)
    used = "pca"
    k = min(50, x.shape[1], max(2, len(x) - 1))
    if len(x) >= 3:
        _u, _s, vt = np.linalg.svd(x[: min(len(x), 20_000)], full_matrices=False)
        red = x @ vt[:k].T
    else:
        red = np.zeros((len(x), 2), np.float32)
    xy = red[:, :2]
    if method == "umap" and len(x) >= 10:
        try:
            import umap                                               # optional
            xy = umap.UMAP(n_components=2, random_state=seed).fit_transform(red)
            used = "umap"
        except Exception:
            method = "tsne"
    if method == "tsne" and len(x) >= 10:
        try:
            from sklearn.manifold import TSNE                         # optional
            xy = TSNE(n_components=2, init="pca", random_state=seed,
                      perplexity=float(min(30, max(5, len(x) // 100)))).fit_transform(red)
            used = "tsne"
        except Exception:
            pass
    xy = np.asarray(xy, np.float32)
    if len(xy):
        lo, hi = xy.min(0), xy.max(0)
        xy = (xy - lo) / np.maximum(hi - lo, 1e-9)
    return xy, used
