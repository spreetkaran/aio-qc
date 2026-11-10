# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Detection metrics, computed the way `yolo val` (Ultralytics) computes them, so numbers here match what
YOLO users already know:

  * predictions at conf >= 0.001 (configurable), matched to ground truth per IoU threshold 0.50:0.05:0.95,
    class-aware, highest-IoU pairs first (Ultralytics' match_predictions, including its tie behaviour)
  * AP: precision envelope, 101-point interpolation + trapezoid integration (Ultralytics "interp")
  * P / R / F1 reported at the confidence that maximises the (smoothed) mean F1 over classes
  * confusion matrix: class-agnostic IoU matching (> 0.45 by default) at an operating confidence,
    with a background row / column (missed objects / false detections)

On top of that, an error analysis at the operating point (IoU 0.5, greedy by confidence like COCO):
  false detections are split into  wrong class | poor localisation | duplicate | background
  missed objects into              missed | missed but found below the threshold
and recall by object size (COCO areas: small < 32², medium < 96², large).

Everything is numpy and vectorised; per-image work is small matrices only.
Predictions are float32 arrays (N, 6): x1, y1, x2, y2, conf, cls - already in dataset class ids.
"""
from dataclasses import dataclass, field

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
EPS = 1e-16
ERROR_TYPES = ("cls", "loc", "dup", "bkg")             # false detections
MISS_TYPES = ("missed", "missed_low_conf")             # false negatives
ERROR_LABELS = {"cls": "Wrong class", "loc": "Poor localisation", "dup": "Duplicate",
                "bkg": "Background (nothing there)", "missed": "Missed", "missed_low_conf": "Missed (found below threshold)",
                "tp": "Correct"}
SIZE_EDGES = (32 ** 2, 96 ** 2)                        # COCO area thresholds (pixels²)
SIZE_NAMES = ("small", "medium", "large")


# ------------------------------------------------------------------ boxes
def box_iou(a, b):
    """IoU matrix (len(a), len(b)) for xyxy boxes."""
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def size_bucket(boxes):
    """0 small / 1 medium / 2 large per box (COCO areas)."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return np.searchsorted(np.asarray(SIZE_EDGES, np.float32), area, side="right")


# ------------------------------------------------------------------ matching (Ultralytics match_predictions)
def match_predictions(pred_cls, true_cls, iou, thresholds=IOU_THRESHOLDS):
    """(n_pred, n_thr) bool: prediction is a true positive at each IoU threshold."""
    correct = np.zeros((len(pred_cls), len(thresholds)), dtype=bool)
    if not len(pred_cls) or not len(true_cls):
        return correct
    iou = iou * (true_cls[:, None] == pred_cls[None, :])      # (n_true, n_pred), zero for other classes
    for i, t in enumerate(thresholds):
        matches = np.array(np.nonzero(iou >= t)).T             # (k, 2): true idx, pred idx
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


# ------------------------------------------------------------------ AP (Ultralytics ap_per_class)
def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_ap(recall, precision):
    """AP, precision envelope, recall points (101-point interpolation, trapezoid integration)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    ap = trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, n_classes):
    """tp (n, n_thr) bool, conf (n,), pred_cls (n,), target_cls (all GT classes).
    Returns dict with per-class arrays for classes that have ground truth."""
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    px = np.linspace(0, 1, 1000)
    n_thr = tp.shape[1]
    ap = np.zeros((nc, n_thr))
    p_curve, r_curve = np.zeros((nc, 1000)), np.zeros((nc, 1000))
    prec_at_rec = np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        m = pred_cls == c
        n_l, n_p = nt[ci], int(m.sum())
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[m]).cumsum(0)
        tpc = tp[m].cumsum(0)
        recall = tpc / (n_l + EPS)
        r_curve[ci] = np.interp(-px, -conf[m], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-px, -conf[m], precision[:, 0], left=1)
        for j in range(n_thr):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_at_rec[ci] = np.interp(px, mrec, mpre)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + EPS)
    sm = smooth(f1_curve.mean(0), 0.1) if nc else np.zeros(1)
    best = int(sm.argmax()) if nc else 0                       # Ultralytics' index: P / R / F1 reported here
    # operating threshold: the HIGH end of the best-F1 plateau - same F1, fewer false detections (a flat F1 curve
    # would otherwise put the threshold at 0.00)
    op = int(np.nonzero(sm >= sm[best] - 1e-4)[0].max()) if nc else 0
    return dict(classes=unique_classes.astype(int), nt=nt.astype(int), ap=ap, px=px,
                p_curve=p_curve, r_curve=r_curve, f1_curve=f1_curve, prec_at_rec=prec_at_rec,
                best_index=best, best_conf=float(px[op]) if nc else 0.25,
                p=p_curve[:, best] if nc else np.zeros(0), r=r_curve[:, best] if nc else np.zeros(0),
                f1=f1_curve[:, best] if nc else np.zeros(0))


# ------------------------------------------------------------------ confusion matrix (Ultralytics)
def confusion_update(matrix, nc, gt_boxes, gt_cls, pred, conf_thr, iou_thr=0.45):
    """matrix (nc+1, nc+1) indexed [true, predicted]; index nc = background."""
    # '>=' (Ultralytics uses '>' with its fixed 0.25): the operating threshold here is a chosen value and must
    # agree with the error analysis, which keeps detections AT the threshold
    pred = pred[pred[:, 4] >= conf_thr] if len(pred) else pred
    gt_cls = np.asarray(gt_cls, int)
    if not len(gt_cls):
        for dc in pred[:, 5].astype(int):
            matrix[nc, dc] += 1                                  # false detection on background
        return
    if not len(pred):
        for gc in gt_cls:
            matrix[gc, nc] += 1                                  # missed
        return
    det_cls = pred[:, 5].astype(int)
    iou = box_iou(gt_boxes, pred[:, :4])
    x = np.nonzero(iou > iou_thr)
    if x[0].shape[0]:
        matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    else:
        matches = np.zeros((0, 3))
    n = matches.shape[0] > 0
    m0, m1 = matches[:, 0].astype(int), matches[:, 1].astype(int)
    for i, gc in enumerate(gt_cls):
        j = m0 == i
        if n and j.sum() == 1:
            matrix[gc, det_cls[m1[j]][0]] += 1
        else:
            matrix[gc, nc] += 1
    for i, dc in enumerate(det_cls):
        if not (m1 == i).any():
            matrix[nc, dc] += 1


# ------------------------------------------------------------------ error analysis (operating point)
def classify_errors(gt_boxes, gt_cls, pred, conf_thr, iou_thr=0.5):
    """Per image at the operating point. Returns
       pred_type: list per kept prediction ('tp' | error type), kept prediction indices,
       gt_type: list per GT ('tp' | 'missed' | 'missed_low_conf'), gt_match: matched pred index or -1."""
    gt_cls = np.asarray(gt_cls, int)
    keep = np.nonzero(pred[:, 4] >= conf_thr)[0] if len(pred) else np.zeros(0, int)
    keep = keep[np.argsort(-pred[keep, 4])] if len(keep) else keep
    P = pred[keep]
    ng = len(gt_cls)
    gt_match = -np.ones(ng, int)
    pred_type = []
    iou = box_iou(gt_boxes, P[:, :4]) if ng and len(P) else np.zeros((ng, len(P)), np.float32)
    for k in range(len(P)):
        c = int(P[k, 5])
        col = iou[:, k] if ng else np.zeros(0)
        same = gt_cls == c
        cand = np.where(same & (gt_match < 0) & (col >= iou_thr), col, -1.0) if ng else np.zeros(0)
        if ng and cand.max() >= iou_thr:
            g = int(cand.argmax())
            gt_match[g] = k
            pred_type.append("tp")
            continue
        if not ng:
            pred_type.append("bkg")
            continue
        best_same = float(col[same].max()) if same.any() else 0.0
        best_other = float(col[~same].max()) if (~same).any() else 0.0
        if best_same >= iou_thr:
            pred_type.append("dup")
        elif best_other >= iou_thr:
            pred_type.append("cls")
        elif best_same >= 0.1:
            pred_type.append("loc")
        elif best_other >= 0.1:
            pred_type.append("cls")
        else:
            pred_type.append("bkg")
    gt_type = []
    low = pred[pred[:, 4] < conf_thr] if len(pred) else pred
    low_iou = box_iou(gt_boxes, low[:, :4]) if ng and len(low) else None
    for g in range(ng):
        if gt_match[g] >= 0:
            gt_type.append("tp")
        elif low_iou is not None and ((low_iou[g] >= iou_thr) & (low[:, 5].astype(int) == gt_cls[g])).any():
            gt_type.append("missed_low_conf")
        else:
            gt_type.append("missed")
    return pred_type, keep, gt_type, gt_match


# ------------------------------------------------------------------ result
@dataclass
class ModelResult:
    name: str
    names: dict                         # dataset class id -> name
    n_images: int = 0
    n_gt: int = 0
    classes: np.ndarray = None          # classes with ground truth
    nt: np.ndarray = None               # GT count per class
    ap: np.ndarray = None               # (n_classes, 10)
    p: np.ndarray = None                # per class at best conf
    r: np.ndarray = None
    f1: np.ndarray = None
    px: np.ndarray = None
    p_curve: np.ndarray = None
    r_curve: np.ndarray = None
    f1_curve: np.ndarray = None
    prec_at_rec: np.ndarray = None
    best_conf: float = 0.25
    op_conf: float = 0.25               # operating threshold used for CM / errors
    confusion: np.ndarray = None        # (nc+1, nc+1) [true, predicted]
    errors: dict = field(default_factory=dict)        # type -> count
    errors_by_class: dict = field(default_factory=dict)  # class -> {type: count}
    recall_by_size: dict = field(default_factory=dict)   # size -> (found, total)
    per_image: list = field(default_factory=list)       # dict per image (see evaluate)
    pred_count: dict = field(default_factory=dict)       # class -> predictions at op conf
    unscored_classes: list = field(default_factory=list)  # predicted at op conf but absent from GT
    speed_ms: float = None
    extra: dict = field(default_factory=dict)

    # ---- headline numbers
    @property
    def map50(self):
        return float(self.ap[:, 0].mean()) if self.ap is not None and len(self.ap) else 0.0

    @property
    def map5095(self):
        return float(self.ap.mean()) if self.ap is not None and len(self.ap) else 0.0

    @property
    def mp(self):
        return float(self.p.mean()) if self.p is not None and len(self.p) else 0.0

    @property
    def mr(self):
        return float(self.r.mean()) if self.r is not None and len(self.r) else 0.0

    @property
    def mf1(self):
        return float(self.f1.mean()) if self.f1 is not None and len(self.f1) else 0.0

    def tp_fp_fn(self):
        """At the operating point (error analysis counts)."""
        tp = int(self.errors.get("tp", 0))
        fp = int(sum(self.errors.get(k, 0) for k in ERROR_TYPES))
        fn = int(sum(self.errors.get(k, 0) for k in MISS_TYPES))
        return tp, fp, fn

    def op_prf(self):
        tp, fp, fn = self.tp_fp_fn()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    def class_rows(self):
        """Per-class table rows (classes with GT)."""
        rows = []
        for i, c in enumerate(self.classes if self.classes is not None else []):
            e = self.errors_by_class.get(int(c), {})
            rows.append(dict(cls=int(c), name=self.names.get(int(c), str(c)), gt=int(self.nt[i]),
                             p=float(self.p[i]), r=float(self.r[i]), f1=float(self.f1[i]),
                             ap50=float(self.ap[i, 0]), ap=float(self.ap[i].mean()),
                             tp=int(e.get("tp", 0)),
                             fp=int(sum(e.get(k, 0) for k in ERROR_TYPES)),
                             fn=int(sum(e.get(k, 0) for k in MISS_TYPES))))
        return rows

    def summary(self):
        p, r, f1 = self.op_prf()
        return dict(model=self.name, images=self.n_images, objects=self.n_gt, map50=self.map50,
                    map5095=self.map5095, precision=self.mp, recall=self.mr, f1=self.mf1,
                    best_conf=self.best_conf, op_conf=self.op_conf, op_precision=p, op_recall=r, op_f1=f1,
                    speed_ms=self.speed_ms)


def evaluate(name, names, gts, preds, op_conf=None, iou_cm=0.45, iou_err=0.5, keep_per_image=True):
    """gts: list of (boxes (n,4), cls (n,)) per image; preds: list of (m, 6) arrays (dataset class ids).
    op_conf: operating threshold for confusion matrix / errors (None = this model's best-F1 confidence)."""
    nc = max([max(names) + 1 if names else 0] + [int(g[1].max()) + 1 for g in gts if len(g[1])] +
             [int(p[:, 5].max()) + 1 for p in preds if len(p)])
    tps, confs, pcls, tcls = [], [], [], []
    for (gb, gc), pr in zip(gts, preds):
        gc = np.asarray(gc, int)
        tcls.append(gc)
        if not len(pr):
            continue
        iou = box_iou(gb, pr[:, :4]) if len(gc) else np.zeros((0, len(pr)), np.float32)
        tps.append(match_predictions(pr[:, 5].astype(int), gc, iou))
        confs.append(pr[:, 4])
        pcls.append(pr[:, 5].astype(int))
    tp = np.concatenate(tps) if tps else np.zeros((0, len(IOU_THRESHOLDS)), bool)
    conf = np.concatenate(confs) if confs else np.zeros(0)
    pc = np.concatenate(pcls) if pcls else np.zeros(0, int)
    tc = np.concatenate(tcls) if tcls else np.zeros(0, int)
    res = ModelResult(name=name, names=dict(names), n_images=len(gts), n_gt=int(len(tc)))
    if len(tc):
        a = ap_per_class(tp, conf, pc, tc, nc)
        res.classes, res.nt, res.ap = a["classes"], a["nt"], a["ap"]
        res.p, res.r, res.f1 = a["p"], a["r"], a["f1"]
        res.px, res.p_curve, res.r_curve = a["px"], a["p_curve"], a["r_curve"]
        res.f1_curve, res.prec_at_rec, res.best_conf = a["f1_curve"], a["prec_at_rec"], a["best_conf"]
    else:
        res.classes, res.nt, res.ap = np.zeros(0, int), np.zeros(0, int), np.zeros((0, 10))
        res.p = res.r = res.f1 = np.zeros(0)
    res.op_conf = float(res.best_conf if op_conf is None else op_conf)
    analyse(res, gts, preds, nc, iou_cm, iou_err, keep_per_image)
    return res


def analyse(res, gts, preds, nc=None, iou_cm=0.45, iou_err=0.5, keep_per_image=True):
    """(Re)compute everything that depends on the operating threshold - cheap, so the UI can re-run it
    when the user moves the threshold slider."""
    if nc is None:
        nc = res.confusion.shape[0] - 1 if res.confusion is not None else (max(res.names) + 1 if res.names else 1)
    cm = np.zeros((nc + 1, nc + 1), np.int64)
    errors = {k: 0 for k in ("tp",) + ERROR_TYPES + MISS_TYPES}
    by_class = {}
    size_found, size_total = np.zeros(3, int), np.zeros(3, int)
    pred_count = {}
    per_image = []
    gt_classes = set()
    for idx, ((gb, gc), pr) in enumerate(zip(gts, preds)):
        gc = np.asarray(gc, int)
        gt_classes.update(gc.tolist())
        confusion_update(cm, nc, gb, gc, pr, res.op_conf, iou_cm)
        pt, keep, gtt, _ = classify_errors(np.asarray(gb, np.float32).reshape(-1, 4), gc,
                                           pr if len(pr) else np.zeros((0, 6), np.float32), res.op_conf, iou_err)
        for t, k in zip(pt, keep):
            c = int(pr[k, 5])
            errors[t] += 1
            by_class.setdefault(c, {}).setdefault(t, 0)
            by_class[c][t] += 1
            pred_count[c] = pred_count.get(c, 0) + 1
        for t, c in zip(gtt, gc):
            if t != "tp":
                errors[t] += 1
                by_class.setdefault(int(c), {}).setdefault(t, 0)
                by_class[int(c)][t] += 1
        if len(gc):
            sb = size_bucket(gb)
            for s, t in zip(sb, gtt):
                size_total[s] += 1
                size_found[s] += t == "tp"
        if keep_per_image:
            fp = sum(t != "tp" for t in pt)
            fn = sum(t != "tp" for t in gtt)
            per_image.append(dict(i=idx, tp=sum(t == "tp" for t in pt), fp=fp, fn=fn, n_gt=len(gc),
                                  pred_type=pt, keep=keep.tolist(), gt_type=gtt))
    res.confusion = cm
    res.errors = errors
    res.errors_by_class = by_class
    res.recall_by_size = {SIZE_NAMES[i]: (int(size_found[i]), int(size_total[i])) for i in range(3)}
    res.pred_count = pred_count
    res.unscored_classes = sorted(c for c in pred_count if c not in gt_classes)
    res.per_image = per_image
    return res


# ------------------------------------------------------------------ class mapping
def _norm(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def build_class_map(model_names, data_names):
    """model class id -> dataset class id (or -1 = not in the dataset).
    Matches by normalised name ('Trailer_Text' == 'trailer text'); if the dataset has no names, by id.
    Returns (mapping dict, matched list[(model_name, data_name)], unmatched model names)."""
    model_names = {int(k): str(v) for k, v in (model_names or {}).items()}
    if not data_names:
        return {k: k for k in model_names}, [(v, v) for v in model_names.values()], []
    by_norm = {_norm(v): int(k) for k, v in data_names.items()}
    mapping, matched, unmatched = {}, [], []
    for k, v in model_names.items():
        d = by_norm.get(_norm(v))
        mapping[k] = -1 if d is None else d
        (matched.append((v, data_names[d])) if d is not None else unmatched.append(v))
    if not model_names:                       # model without names: assume same ids
        mapping = {k: k for k in data_names}
    return mapping, matched, unmatched


def remap(pred, mapping):
    """Apply a model->dataset class mapping; drops predictions of unmapped classes."""
    if pred is None or not len(pred):
        return np.zeros((0, 6), np.float32)
    lut_max = int(max(max(mapping) if mapping else 0, pred[:, 5].max())) + 1
    lut = -np.ones(lut_max, int)
    for k, v in mapping.items():
        if 0 <= k < lut_max:
            lut[k] = v
    out = pred.copy()
    out[:, 5] = lut[pred[:, 5].astype(int)]
    return out[out[:, 5] >= 0]


# ------------------------------------------------------------------ model vs model (no labels)
def agreement(pred_a, pred_b, conf_a, conf_b, iou_thr=0.5):
    """Per-frame agreement between two models at their operating thresholds: F1 of matching A's boxes to
    B's (class-aware, IoU >= iou_thr). 1.0 = identical detections, both empty = 1.0."""
    a = pred_a[pred_a[:, 4] >= conf_a] if len(pred_a) else pred_a
    b = pred_b[pred_b[:, 4] >= conf_b] if len(pred_b) else pred_b
    if not len(a) and not len(b):
        return 1.0, 0, 0, 0
    if not len(a) or not len(b):
        return 0.0, 0, len(a), len(b)
    iou = box_iou(a[:, :4], b[:, :4]) * (a[:, 5][:, None] == b[:, 5][None, :])
    used_b = np.zeros(len(b), bool)
    m = 0
    for i in np.argsort(-a[:, 4]):
        row = np.where(used_b, -1, iou[i])
        j = int(row.argmax())
        if row[j] >= iou_thr:
            used_b[j] = True
            m += 1
    return 2 * m / (len(a) + len(b)), m, len(a) - m, len(b) - m
