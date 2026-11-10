# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Model-assisted QC: compare model detections with the annotations and rank likely label errors.

Findings (per image):
    class mismatch       GT box overlaps a confident prediction of a DIFFERENT class (and no same-class one)
    missing annotation   confident prediction with no GT box nearby  -> object probably not labelled
    missed by model      GT box with no matching prediction           -> doubtful / wrong / very hard label
    poor box fit         same class, but IoU below the match threshold -> sloppy box

Ultralytics is imported lazily; only ModelRunner needs it. match() is pure Python/numpy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .geometry import to_polygon

MISMATCH, MISSING, MISSED, POOR_FIT = "class mismatch", "missing annotation", "missed by model", "poor box fit"


@dataclass
class Finding:
    kind: str
    gt_index: int | None          # shape index (None for missing annotation)
    gt_label: str
    pred_label: str
    conf: float
    iou: float
    box: tuple | None             # pred box xyxy (pixels) for overlays / "add as annotation"
    score: float                  # ranking key: higher = more likely a real label error
    model: str = ""               # id of the model that produced the prediction
    gt_box: tuple | None = None   # the labelled box (pixels) — for crops / cards


def gt_boxes(shapes, width: float, height: float, normalized: bool):
    """[(shape_index, label, xyxy)] in pixels for every shape with an area."""
    out = []
    sx, sy = (width, height) if normalized else (1.0, 1.0)
    for i, s in enumerate(shapes):
        poly = to_polygon(s)
        if poly is None:
            continue
        x1, y1 = poly.min(0)
        x2, y2 = poly.max(0)
        out.append((i, str(s.get("label", "")), (x1 * sx, y1 * sy, x2 * sx, y2 * sy)))
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match(gt, preds, iou_thr=0.5, conf_thr=0.25, missing_conf=0.5, mismatch_conf=0.4,
          near_iou=0.3, agnostic=False, covered: set | None = None, mismatch_ok=None) -> list[Finding]:
    """
    gt:    [(shape_index, label, xyxy)]
    preds: [(label, conf, xyxy[, model_id])]  (already mapped to dataset label names)
    covered:     labels at least one model can predict; GT of other labels is never "missed by model"
    mismatch_ok: fn(gt_label, pred) -> bool; e.g. only when the predicting model also knows gt_label
    """
    preds = [p for p in preds if p[1] >= conf_thr]
    G = np.array([g[2] for g in gt], dtype=float).reshape(-1, 4)
    P = np.array([p[2] for p in preds], dtype=float).reshape(-1, 4)
    M = iou_matrix(G, P)
    same = np.array([[agnostic or g[1] == p[0] for p in preds] for g in gt], dtype=bool).reshape(len(gt), len(preds))
    findings: list[Finding] = []
    used_p, used_g = set(), set()

    # 1) one-to-one same-class matching, most confident predictions first
    for j in sorted(range(len(preds)), key=lambda j: -preds[j][1]):
        cand = [(M[i, j], i) for i in range(len(gt)) if i not in used_g and same[i, j] and M[i, j] >= iou_thr]
        if cand:
            _, i = max(cand)
            used_g.add(i)
            used_p.add(j)

    # 2) unmatched GT: class mismatch / poor fit / missed
    for i, (idx, label, _box) in enumerate(gt):
        if i in used_g:
            continue
        free = [j for j in range(len(preds)) if j not in used_p]
        other = [j for j in free if not same[i, j] and M[i, j] >= iou_thr and preds[j][1] >= mismatch_conf
                 and (mismatch_ok is None or mismatch_ok(label, preds[j]))]
        if other:
            j = max(other, key=lambda j: preds[j][1] * M[i, j])
            used_p.add(j)
            findings.append(Finding(MISMATCH, idx, label, preds[j][0], preds[j][1], float(M[i, j]),
                                    tuple(preds[j][2]), preds[j][1] * M[i, j] + 1.0, _mid(preds[j])))
            continue
        sloppy = [j for j in free if same[i, j] and near_iou <= M[i, j] < iou_thr]
        if sloppy:
            j = max(sloppy, key=lambda j: M[i, j])
            used_p.add(j)
            findings.append(Finding(POOR_FIT, idx, label, preds[j][0], preds[j][1], float(M[i, j]),
                                    tuple(preds[j][2]), (1 - M[i, j]) * preds[j][1], _mid(preds[j])))
            continue
        if covered is not None and not agnostic and label not in covered:
            continue                                  # no model knows this class: nothing to say
        best = float(M[i].max()) if len(preds) else 0.0
        findings.append(Finding(MISSED, idx, label, "", 0.0, best, None, 0.5 * (1 - best)))

    # 3) confident predictions far from every GT box -> probably unlabeled object
    for j, p in enumerate(preds):
        plabel, conf, box = p[0], p[1], p[2]
        if j in used_p or conf < missing_conf:
            continue
        if len(gt) and M[:, j].max() >= near_iou:
            continue
        findings.append(Finding(MISSING, None, "", plabel, conf, float(M[:, j].max()) if len(gt) else 0.0,
                                tuple(box), conf + 0.5, _mid(p)))
    return findings


def _mid(p) -> str:
    return p[3] if len(p) > 3 else ""


# =============================================================================
# Multiple models
# =============================================================================
def model_signature(spec: dict) -> str:
    """Everything that changes a model's raw detections -> prediction-cache key."""
    try:
        st = os.stat(spec["path"])
        ver = f"{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        ver = "missing"
    sahi = spec.get("sahi")
    sahi_key = f"sahi{sahi.get('slice')}/{sahi.get('overlap')}" if sahi else "std"
    return f"{os.path.abspath(spec['path'])}|{ver}|{spec.get('imgsz')}|{sahi_key}|half{int(bool(spec.get('half')))}"


def coverage(class_maps: dict[str, dict[str, str]]) -> dict[str, set]:
    """dataset label -> ids of the models that predict it."""
    out: dict[str, set] = {}
    for mid, cmap in class_maps.items():
        for label in cmap.values():
            if label:
                out.setdefault(label, set()).add(mid)
    return out


def merge_predictions(by_model: dict, class_maps: dict[str, dict[str, str]], nms_iou: float = 0.6) -> list[tuple]:
    """
    by_model:   {model_id: [[x1, y1, x2, y2, conf, cls], ...]}
    class_maps: {model_id: {"cls_id": dataset_label or ""}}  ("" = ignore that model class)
    -> [(label, conf, (x1, y1, x2, y2), model_id)] after cross-model, per-label NMS
    """
    cand = []
    for mid, boxes in by_model.items():
        cmap = class_maps.get(mid, {})
        for x1, y1, x2, y2, conf, cls in boxes or []:
            label = cmap.get(str(int(cls)), "")
            if label:
                cand.append((label, float(conf), (x1, y1, x2, y2), mid))
    if len({c[3] for c in cand}) <= 1:
        return cand                                        # single model: its own NMS already ran
    cand.sort(key=lambda c: -c[1])
    kept: list[tuple] = []
    for c in cand:
        clash = [k for k in kept if k[0] == c[0]]
        if clash:
            ious = iou_matrix(np.array([c[2]], float), np.array([k[2] for k in clash], float))[0]
            if (ious >= nms_iou).any():
                continue
        kept.append(c)
    return kept


# =============================================================================
# Ultralytics wrapper (optional dependency)
# =============================================================================
class ModelRunner:
    def __init__(self, weights: str, device: str = "", imgsz: int = 640, conf_min: float = 0.05):
        try:
            from ultralytics import YOLO  # noqa: WPS433 (lazy import)
        except ImportError as exc:
            raise RuntimeError("Model check needs the 'ultralytics' package:  pip install ultralytics") from exc
        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.names = {int(k): str(v) for k, v in dict(self.model.names).items()}
        self.device = device or None
        self.imgsz = int(imgsz)
        self.conf_min = float(conf_min)
        st = os.stat(self.weights)
        # everything that changes raw predictions is part of the cache key
        self.signature = f"{os.path.abspath(self.weights)}|{st.st_mtime_ns}|{st.st_size}|{self.imgsz}|{self.conf_min}"

    def predict(self, image_path) -> dict:
        """-> {"w", "h", "boxes": [[x1, y1, x2, y2, conf, cls], ...]} in original-image pixels."""
        r = self.model.predict(str(image_path), imgsz=self.imgsz, conf=self.conf_min, device=self.device,
                               verbose=False)[0]
        h, w = r.orig_shape[:2]
        boxes = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            boxes = [[*map(float, b), float(c), int(k)] for b, c, k in zip(xyxy, conf, cls)]
        elif getattr(r, "obb", None) is not None and len(r.obb):
            xyxy = r.obb.xyxy.cpu().numpy()
            conf = r.obb.conf.cpu().numpy()
            cls = r.obb.cls.cpu().numpy().astype(int)
            boxes = [[*map(float, b), float(c), int(k)] for b, c, k in zip(xyxy, conf, cls)]
        return {"w": int(w), "h": int(h), "boxes": boxes}
