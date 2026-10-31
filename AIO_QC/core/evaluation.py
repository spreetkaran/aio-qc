# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Dataset-level GT vs prediction analysis (Qt-free, vectorised).

Build once after a Model Check run, then every threshold / group change is a numpy count:

  * COCO-style greedy matching (predictions in descending confidence) is prefix-consistent: the
    matches of predictions with conf >= c do not depend on lower-confidence predictions. So we match
    ONCE per IoU threshold, store the matched GT per prediction, and any conf threshold is a mask.
  * AP50 / AP50-95 from the same arrays (101-point interpolation, per class).
  * Confusion matrix at (conf, IoU): diagonal from the stored matches, off-diagonal from stored
    cross-class candidate pairs.

Held-out support: every prediction knows whether its model was trained on that image ("seen"), and
every GT knows whether the models covering its label saw the image (1 seen, 0 unseen, -1 mixed).
Out-of-fold mode re-matches with only predictions from models that did NOT train on the image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model_check import iou_matrix

IOU_GRID = np.round(np.concatenate([[0.1, 0.2, 0.3, 0.4], np.arange(0.5, 0.951, 0.05)]), 2)
COCO = IOU_GRID >= 0.5
BACKGROUND = "background"


# -----------------------------------------------------------------------------
# Which images did a model train on?
# -----------------------------------------------------------------------------
def load_image_list(path: str | Path) -> set[str]:
    """Training list (txt with one image path per line, ultralytics style) -> set of file names and stems."""
    out = set()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip().replace("\\", "/")
        if line and not line.startswith("#"):
            name = line.rsplit("/", 1)[-1]
            out.add(name)
            out.add(name.rsplit(".", 1)[0])
    return out


def make_seen_fn(trained_on: dict | None):
    """
    trained_on: {"mode": "all" | "train" | "none" | "list", "path": txt}  (default "all")
      all   -> the model was trained on this whole dataset (typical in-sample QC)
      train -> trained on the train split only: val/test are held-out
      none  -> held-out model (never saw this dataset)
      list  -> trained on the images in a list file (k-fold / custom subsets)
    """
    mode = (trained_on or {}).get("mode", "all")
    if mode == "none":
        return lambda sample: False
    if mode == "train":
        return lambda sample: (sample.split or "").lower() in ("", "train")
    if mode == "list":
        names = load_image_list(trained_on["path"])

        def seen(sample):
            img = str(sample.image_path.name if sample.image_path else sample.name)
            return img in names or img.rsplit(".", 1)[0] in names
        return seen
    return lambda sample: True


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
@dataclass
class Record:
    key: str
    split: str
    gt: list            # [(shape_index, label, (x1, y1, x2, y2), seen_state)]
    preds: list         # [(label, conf, (x1, y1, x2, y2), model_id, seen_bool)]
    w: float | None = None          # image size (slices, box quality)
    h: float | None = None
    folder: str = ""                # first relative folder (source) — slices
    light: str | None = None        # lighting class if known (from the Stats image profile cache)


@dataclass
class EvalData:
    labels: list[str]
    splits: list[str]
    keys: list[str]
    # GT arrays
    gt_cls: np.ndarray = field(default=None)
    gt_img: np.ndarray = field(default=None)
    gt_idx: np.ndarray = field(default=None)          # shape index in the file
    gt_seen: np.ndarray = field(default=None)         # 1 seen / 0 unseen / -1 mixed
    gt_oof: np.ndarray = field(default=None)          # at least one covering model did NOT see the image
    # prediction arrays
    p_cls: np.ndarray = field(default=None)
    p_conf: np.ndarray = field(default=None)
    p_img: np.ndarray = field(default=None)
    p_model: np.ndarray = field(default=None)
    p_seen: np.ndarray = field(default=None)
    p_box: np.ndarray = field(default=None)
    match_all: np.ndarray = field(default=None)       # [n_iou, n_pred] global GT index or -1
    match_oof: np.ndarray = field(default=None)       # same, matching with unseen predictions only
    img_split: np.ndarray = field(default=None)
    cross: np.ndarray = field(default=None)           # [(gt, pred, iou)] different-class overlaps >= 0.1
    pairs: np.ndarray = field(default=None)           # [(gt, pred, iou, same_class)] all overlaps >= 0.1
    g_box: np.ndarray = field(default=None)
    img_w: np.ndarray = field(default=None)
    img_h: np.ndarray = field(default=None)
    img_folder: list = field(default_factory=list)
    img_light: list = field(default_factory=list)
    models: list[str] = field(default_factory=list)


def _greedy(M_same: np.ndarray, order: np.ndarray, n_gt: int) -> np.ndarray:
    """All IoU thresholds at once. Returns [n_iou, n_pred_local] matched local GT index or -1."""
    T = len(IOU_GRID)
    out = np.full((T, M_same.shape[1]), -1, dtype=np.int32)
    if n_gt == 0 or M_same.shape[1] == 0:
        return out
    taken = np.zeros((T, n_gt), dtype=bool)
    thr = IOU_GRID[:, None]
    for j in order:
        col = M_same[:, j]
        if col.max() < IOU_GRID[0]:
            continue                                   # overlaps nothing: FP at every threshold
        valid = (col[None, :] >= thr) & ~taken
        masked = np.where(valid, col[None, :], -1.0)
        best = masked.argmax(1)
        hit = masked[np.arange(T), best] >= 0
        out[hit, j] = best[hit]
        taken[np.nonzero(hit)[0], best[hit]] = True
    return out


def build(records: list[Record], labels: list[str], models: list[str] | None = None) -> EvalData:
    lab = {l: i for i, l in enumerate(labels)}
    splits = sorted({r.split for r in records}, key=lambda s: (s not in ("train", ""), s))
    sidx = {s: i for i, s in enumerate(splits)}
    models = list(models or sorted({p[3] for r in records for p in r.preds}))
    midx = {m: i for i, m in enumerate(models)}
    G = {k: [] for k in ("cls", "img", "idx", "seen", "oof")}
    P = {k: [] for k in ("cls", "conf", "img", "model", "seen", "box")}
    m_all, m_oof, pairs, g_boxes = [], [], [], []
    img_w, img_h, img_folder, img_light = [], [], [], []
    g_off = p_off = 0
    img_split = []
    for ri, r in enumerate(records):
        img_split.append(sidx[r.split])
        img_w.append(float(r.w or 0))
        img_h.append(float(r.h or 0))
        img_folder.append(r.folder or "")
        img_light.append(r.light)
        gt = [g for g in r.gt if g[1] in lab]
        pr = [p for p in r.preds if p[0] in lab]
        for g in gt:
            g_boxes.append(g[2])
            G["cls"].append(lab[g[1]]); G["img"].append(ri); G["idx"].append(g[0])
            G["seen"].append(g[3]); G["oof"].append(g[3] != 1)
        for p in pr:
            P["cls"].append(lab[p[0]]); P["conf"].append(p[1]); P["img"].append(ri)
            P["model"].append(midx.get(p[3], -1)); P["seen"].append(bool(p[4])); P["box"].append(p[2])
        n_g, n_p = len(gt), len(pr)
        if n_p:
            Gb = np.array([g[2] for g in gt], float).reshape(-1, 4)
            Pb = np.array([p[2] for p in pr], float).reshape(-1, 4)
            M = iou_matrix(Gb, Pb)
            gc = np.array([lab[g[1]] for g in gt], int)
            pc = np.array([lab[p[0]] for p in pr], int)
            same = (gc[:, None] == pc[None, :]) if n_g else np.zeros((0, n_p), bool)
            Ms = np.where(same, M, 0.0)
            order = np.argsort(-np.array([p[1] for p in pr]), kind="stable")
            local = _greedy(Ms, order, n_g)
            m_all.append(np.where(local >= 0, local + g_off, -1))
            unseen = np.array([not p[4] for p in pr], bool)
            if unseen.any():
                local_oof = np.full_like(local, -1)
                idxs = np.nonzero(unseen)[0]
                sub = _greedy(Ms[:, idxs], np.argsort(-np.array([pr[i][1] for i in idxs]), kind="stable"), n_g)
                local_oof[:, idxs] = sub
                m_oof.append(np.where(local_oof >= 0, local_oof + g_off, -1))
            else:
                m_oof.append(np.full_like(local, -1))
            if n_g:
                gi, pj = np.nonzero(M >= 0.1)
                for a, b in zip(gi, pj):
                    pairs.append((a + g_off, b + p_off, M[a, b], float(same[a, b])))
        g_off += n_g
        p_off += n_p
    T = len(IOU_GRID)
    d = EvalData(labels, splits, [r.key for r in records], models=models)
    d.gt_cls = np.array(G["cls"], int); d.gt_img = np.array(G["img"], int); d.gt_idx = np.array(G["idx"], int)
    d.gt_seen = np.array(G["seen"], int); d.gt_oof = np.array(G["oof"], bool)
    d.p_cls = np.array(P["cls"], int); d.p_conf = np.array(P["conf"], float); d.p_img = np.array(P["img"], int)
    d.p_model = np.array(P["model"], int); d.p_seen = np.array(P["seen"], bool)
    d.p_box = np.array(P["box"], float).reshape(-1, 4)
    d.match_all = np.concatenate(m_all, axis=1) if m_all else np.zeros((T, 0), np.int32)
    d.match_oof = np.concatenate(m_oof, axis=1) if m_oof else np.zeros((T, 0), np.int32)
    d.img_split = np.array(img_split, int)
    d.pairs = np.array(pairs, float).reshape(-1, 4)
    d.cross = d.pairs[d.pairs[:, 3] == 0][:, :3] if len(d.pairs) else np.zeros((0, 3))
    d.g_box = np.array(g_boxes, float).reshape(-1, 4)
    d.img_w, d.img_h = np.array(img_w, float), np.array(img_h, float)
    d.img_folder, d.img_light = img_folder, img_light
    return d


# -----------------------------------------------------------------------------
# Views
# -----------------------------------------------------------------------------
@dataclass
class View:
    """Which part of the data a metric looks at."""
    split: str | None = None        # None = all splits
    seen: str = "all"               # "all" | "seen" | "unseen"
    oof: bool = False               # out-of-fold: only predictions from models that did not train on the image


def masks(d: EvalData, v: View):
    gm = np.ones(len(d.gt_cls), bool)
    pm = np.ones(len(d.p_cls), bool)
    if v.split is not None and v.split in d.splits:
        s = d.splits.index(v.split)
        gm &= d.img_split[d.gt_img] == s if len(d.gt_img) else gm
        pm &= d.img_split[d.p_img] == s if len(d.p_img) else pm
    if v.oof:
        gm &= d.gt_oof
        pm &= ~d.p_seen
    elif v.seen == "seen":
        gm &= d.gt_seen == 1
        pm &= d.p_seen
    elif v.seen == "unseen":
        gm &= d.gt_seen == 0
        pm &= ~d.p_seen
    return gm, pm


def _ti(iou: float) -> int:
    return int(np.argmin(np.abs(IOU_GRID - iou)))


def _ap(conf, tp, npos) -> float:
    if npos == 0 or len(conf) == 0:
        return float("nan") if npos == 0 else 0.0
    o = np.argsort(-conf, kind="stable")
    t = tp[o].astype(float)
    ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
    rec = ctp / npos
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]          # precision envelope
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, grid, side="left")
    return float(np.mean(np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)))


def report(d: EvalData, conf: float, iou: float, v: View) -> list[dict]:
    """Per-class report card (+ an 'all' micro row) at the given thresholds."""
    gm, pm = masks(d, v)
    match = d.match_oof if v.oof else d.match_all
    ti = _ti(iou)
    rows = []
    tot = dict(gt=0, pred=0, tp=0)
    for c, label in enumerate(d.labels):
        g = gm & (d.gt_cls == c)
        p = pm & (d.p_cls == c)
        npos = int(g.sum())
        # matches pointing at GT outside the view (other split / seen-group) don't count
        m = match[ti][p]
        valid_gt = np.zeros(len(d.gt_cls), bool)
        valid_gt[g] = True
        tp_all = (m >= 0) & valid_gt[np.maximum(m, 0)]
        above = d.p_conf[p] >= conf
        tp = int((tp_all & above).sum())
        npred = int(above.sum())
        fp, fn = npred - tp, npos - tp
        prec = tp / npred if npred else float("nan")
        rec = tp / npos if npos else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if npred and npos and (prec + rec) else float("nan")
        # AP over all confidences
        aps = []
        for t in np.nonzero(COCO)[0]:
            mt = match[t][p]
            aps.append(_ap(d.p_conf[p], (mt >= 0) & valid_gt[np.maximum(mt, 0)], npos))
        # best F1 threshold
        best_conf = float("nan")
        if npos and p.any():
            o = np.argsort(-d.p_conf[p], kind="stable")
            ctp = np.cumsum(tp_all[o])
            k = np.arange(1, len(o) + 1)
            f1c = 2 * ctp / (k + npos)
            best_conf = float(d.p_conf[p][o][int(np.argmax(f1c))])
        rows.append(dict(label=label, gt=npos, pred=npred, tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                         ap50=aps[0] if aps else float("nan"),
                         ap5095=float(np.nanmean(aps)) if aps and not np.all(np.isnan(aps)) else float("nan"),
                         ratio=npred / npos if npos else float("nan"), best_conf=best_conf))
        tot["gt"] += npos
        tot["pred"] += npred
        tot["tp"] += tp
    P = tot["tp"] / tot["pred"] if tot["pred"] else float("nan")
    R = tot["tp"] / tot["gt"] if tot["gt"] else float("nan")
    valid = [r for r in rows if r["gt"]]
    rows.append(dict(label="ALL (micro)", gt=tot["gt"], pred=tot["pred"], tp=tot["tp"], fp=tot["pred"] - tot["tp"],
                     fn=tot["gt"] - tot["tp"], precision=P, recall=R,
                     f1=2 * P * R / (P + R) if tot["pred"] and tot["gt"] and (P + R) else float("nan"),
                     ap50=float(np.nanmean([r["ap50"] for r in valid])) if valid else float("nan"),
                     ap5095=float(np.nanmean([r["ap5095"] for r in valid])) if valid else float("nan"),
                     ratio=tot["pred"] / tot["gt"] if tot["gt"] else float("nan"), best_conf=float("nan")))
    return rows


def confusion(d: EvalData, conf: float, iou: float, v: View):
    """
    (matrix [n+1, n+1] of counts, cells {(gt_i, pred_j): [("gt", gt_global) | ("pred", p_global)]})
    Rows = GT label (last row = background → false positives), columns = predicted label
    (last column = background → missed GT). Cells keep their members for drill-down.
    """
    n = len(d.labels)
    B = n
    gm, pm = masks(d, v)
    match = (d.match_oof if v.oof else d.match_all)[_ti(iou)]
    live_p = pm & (d.p_conf >= conf)
    mat = np.zeros((n + 1, n + 1), int)
    cells: dict = {}
    matched_gt = np.zeros(len(d.gt_cls), bool)
    matched_p = np.zeros(len(d.p_cls), bool)
    for j in np.nonzero(live_p & (match >= 0))[0]:
        gi = match[j]
        if gm[gi]:
            matched_gt[gi] = matched_p[j] = True
            c = d.gt_cls[gi]
            mat[c, c] += 1
            cells.setdefault((c, c), []).append(("gt", int(gi)))
    # cross-class: best remaining pair, greedy by IoU
    if len(d.cross):
        cand = [(iu, int(g), int(p)) for g, p, iu in d.cross
                if iu >= iou and gm[int(g)] and live_p[int(p)] and not matched_gt[int(g)] and not matched_p[int(p)]]
        for iu, g, p in sorted(cand, reverse=True):
            if matched_gt[g] or matched_p[p]:
                continue
            matched_gt[g] = matched_p[p] = True
            a, b = d.gt_cls[g], d.p_cls[p]
            mat[a, b] += 1
            cells.setdefault((a, b), []).append(("gt", g))
    for g in np.nonzero(gm & ~matched_gt)[0]:
        a = d.gt_cls[g]
        mat[a, B] += 1
        cells.setdefault((a, B), []).append(("gt", int(g)))
    for p in np.nonzero(live_p & ~matched_p)[0]:
        b = d.p_cls[p]
        mat[B, b] += 1
        cells.setdefault((B, b), []).append(("pred", int(p)))
    return mat, cells


# -----------------------------------------------------------------------------
# Shared matching state at (conf, IoU, view)
# -----------------------------------------------------------------------------
@dataclass
class State:
    gm: np.ndarray            # GT in view
    live: np.ndarray          # predictions in view with conf >= threshold
    matched_gt: np.ndarray
    matched_p: np.ndarray
    gt_of_p: np.ndarray       # matched GT per prediction (-1)


def state(d: EvalData, conf: float, iou: float, v: View) -> State:
    gm, pm = masks(d, v)
    match = (d.match_oof if v.oof else d.match_all)[_ti(iou)]
    live = pm & (d.p_conf >= conf)
    gt_of_p = np.where(live & (match >= 0), match, -1)
    ok = gt_of_p >= 0
    ok[ok] = gm[gt_of_p[ok]]
    gt_of_p = np.where(ok, gt_of_p, -1)
    matched_gt = np.zeros(len(d.gt_cls), bool)
    matched_gt[gt_of_p[ok]] = True
    return State(gm, live, matched_gt, ok, gt_of_p)


# -----------------------------------------------------------------------------
# Error-type breakdown (TIDE-style, counts at the current thresholds)
# -----------------------------------------------------------------------------
ERROR_TYPES = [
    ("cls", "Classification", "right place, other class", "a wrong class label — or a model confusion"),
    ("loc", "Localization", "right class, loose overlap", "sloppy / inconsistent boxes"),
    ("both", "Class + location", "other class, loose overlap", "wrong label on a poorly drawn box, or clutter"),
    ("dupe", "Duplicate", "extra box on an already matched object", "a model issue — not a label problem"),
    ("bkg", "Background", "no label anywhere near", "an unlabeled object (missing annotation) or a false alarm"),
    ("miss", "Missed", "label with no prediction near it", "a doubtful / wrong label, or a very hard object"),
]


def error_breakdown(d: EvalData, conf: float, iou: float, v: View) -> dict:
    """Every unmatched prediction and every unmatched label gets exactly one error type."""
    st = state(d, conf, iou, v)
    n = len(d.labels)
    per = np.zeros((n, len(ERROR_TYPES)), int)
    col = {k: i for i, (k, *_r) in enumerate(ERROR_TYPES)}
    members = {}
    fp = np.nonzero(st.live & ~st.matched_p)[0]
    if len(fp):
        best_same = np.zeros(len(d.p_cls))
        best_other = np.zeros(len(d.p_cls))
        if len(d.pairs):
            g, p, iu, same = d.pairs[:, 0].astype(int), d.pairs[:, 1].astype(int), d.pairs[:, 2], d.pairs[:, 3] > 0
            ing = st.gm[g]
            np.maximum.at(best_same, p[same & ing], iu[same & ing])
            np.maximum.at(best_other, p[~same & ing], iu[~same & ing])
        for j in fp:
            s, o = best_same[j], best_other[j]
            if s >= iou:
                t = "dupe"
            elif o >= iou:
                t = "cls"
            elif s >= 0.1:
                t = "loc"
            elif o >= 0.1:
                t = "both"
            else:
                t = "bkg"
            per[d.p_cls[j], col[t]] += 1
            members.setdefault((int(d.p_cls[j]), t), []).append(("pred", int(j)))
    fn = np.nonzero(st.gm & ~st.matched_gt)[0]
    if len(fn):
        near = np.zeros(len(d.gt_cls), bool)
        if len(d.pairs):
            g, p = d.pairs[:, 0].astype(int), d.pairs[:, 1].astype(int)
            near[g[st.live[p]]] = True
        for gi in fn:
            if not near[gi]:
                per[d.gt_cls[gi], col["miss"]] += 1
                members.setdefault((int(d.gt_cls[gi]), "miss"), []).append(("gt", int(gi)))
    tp = np.bincount(d.gt_cls[st.matched_gt], minlength=n) if n else np.zeros(0, int)
    n_pred = np.bincount(d.p_cls[st.live], minlength=n) if n else np.zeros(0, int)
    n_gt = np.bincount(d.gt_cls[st.gm], minlength=n) if n else np.zeros(0, int)
    return {"per_class": per, "tp": tp, "pred": n_pred, "gt": n_gt, "members": members}


# -----------------------------------------------------------------------------
# Box quality on matched pairs
# -----------------------------------------------------------------------------
IOU_BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


def box_quality(d: EvalData, conf: float, iou: float, v: View) -> list[dict]:
    """
    Per class: IoU distribution of matched pairs and systematic edge offsets.
    Offsets are signed, as % of the GT size, positive = the LABEL extends further than the model on that side
    (label looser), negative = the label is tighter than the model.
    """
    st = state(d, conf, iou, v)
    js = np.nonzero(st.gt_of_p >= 0)[0]
    out = []
    groups = [(c, d.labels[c]) for c in range(len(d.labels))] + [(-1, "ALL")]
    if not len(js):
        return [dict(label=l, n=0) for _c, l in groups]
    gi = st.gt_of_p[js]
    G, P = d.g_box[gi], d.p_box[js]
    gw = np.maximum(G[:, 2] - G[:, 0], 1e-9)
    gh = np.maximum(G[:, 3] - G[:, 1], 1e-9)
    ix = np.clip(np.minimum(G[:, 2], P[:, 2]) - np.maximum(G[:, 0], P[:, 0]), 0, None)
    iy = np.clip(np.minimum(G[:, 3], P[:, 3]) - np.maximum(G[:, 1], P[:, 1]), 0, None)
    inter = ix * iy
    pa = (P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1])
    ious = inter / np.maximum(gw * gh + pa - inter, 1e-9)
    off = np.stack([(P[:, 0] - G[:, 0]) / gw, (P[:, 1] - G[:, 1]) / gh,
                    (G[:, 2] - P[:, 2]) / gw, (G[:, 3] - P[:, 3]) / gh], 1) * 100
    area_ratio = (gw * gh) / np.maximum(pa, 1e-9)
    cls = d.gt_cls[gi]
    for c, label in groups:
        m = np.ones(len(js), bool) if c < 0 else cls == c
        k = int(m.sum())
        if not k:
            out.append(dict(label=label, n=0))
            continue
        hist = np.histogram(ious[m], bins=IOU_BINS)[0]
        med = np.median(off[m], 0)
        out.append(dict(label=label, n=k, mean_iou=float(ious[m].mean()), median_iou=float(np.median(ious[m])),
                        loose=float((ious[m] < 0.75).mean()), left=float(med[0]), top=float(med[1]),
                        right=float(med[2]), bottom=float(med[3]),
                        area=float(np.median(area_ratio[m])), hist=hist.tolist(),
                        worst=[(int(gi[i]), float(ious[i])) for i in np.nonzero(m)[0][np.argsort(ious[m])[:200]]]))
    return out


# -----------------------------------------------------------------------------
# Slice analysis
# -----------------------------------------------------------------------------
def _size_bucket(rel):
    return np.select([rel < 0.03, rel < 0.08, rel < 0.2], ["tiny (<3% side)", "small (3–8%)", "medium (8–20%)"],
                     "large (≥20%)")


def _aspect_bucket(ar):
    return np.select([ar < 0.5, ar <= 2, ar <= 4], ["tall (<0.5)", "square-ish (0.5–2)", "wide (2–4)"], "very wide (>4)")


def _res_bucket(m):
    return np.select([m <= 0, m < 480, m < 720, m < 1080], ["unknown", "<480 px", "480–719 px", "720–1079 px"], "≥1080 px")


def _box_slices(boxes, img, d) -> dict[str, np.ndarray]:
    W, H = d.img_w[img], d.img_h[img]
    bw = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-9)
    bh = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-9)
    known = (W > 0) & (H > 0)
    rel = np.where(known, np.sqrt(bw * bh / np.maximum(W * H, 1e-9)), -1)
    size = np.where(known, _size_bucket(rel), "unknown")
    tol = 0.01 * np.maximum(W, H)
    edge = known & ((boxes[:, 0] <= tol) | (boxes[:, 1] <= tol) | (boxes[:, 2] >= W - tol) | (boxes[:, 3] >= H - tol))
    out = {"Object size": size, "Aspect ratio (w/h)": _aspect_bucket(bw / bh),
           "Position": np.where(edge, "touching the image edge", np.where(known, "inside", "unknown")),
           "Image resolution (short side)": _res_bucket(np.minimum(W, H)),
           "Split": np.array([d.splits[s] or "(none)" for s in d.img_split[img]], dtype=object)}
    folders = np.array(d.img_folder, dtype=object)
    if len({f for f in d.img_folder}) > 1:
        out["Source folder"] = folders[img]
    lights = np.array([l or "unknown" for l in d.img_light], dtype=object)
    if any(l for l in d.img_light):
        out["Lighting"] = lights[img]
    return out


def slices(d: EvalData, conf: float, iou: float, v: View, label: str | None = None) -> list[dict]:
    """Recall per slice of the labels, precision per slice of the predictions (optionally one class)."""
    st = state(d, conf, iou, v)
    gsel = st.gm.copy()
    psel = st.live.copy()
    if label is not None and label in d.labels:
        c = d.labels.index(label)
        gsel &= d.gt_cls == c
        psel &= d.p_cls == c
    gi, pj = np.nonzero(gsel)[0], np.nonzero(psel)[0]
    if not len(gi) and not len(pj):
        return []
    gs = _box_slices(d.g_box[gi], d.gt_img[gi], d) if len(gi) else {}
    ps = _box_slices(d.p_box[pj], d.p_img[pj], d) if len(pj) else {}
    tp_g = st.matched_gt[gi]
    tp_p = st.matched_p[pj]
    total_rec = float(tp_g.mean()) if len(gi) else float("nan")
    rows = []
    for dim in dict.fromkeys(list(gs) + list(ps)):
        gv = gs.get(dim, np.array([]))
        pv = ps.get(dim, np.array([]))
        for val in sorted(set(gv.tolist()) | set(pv.tolist()), key=str):
            gmask = gv == val if len(gv) else np.zeros(0, bool)
            pmask = pv == val if len(pv) else np.zeros(0, bool)
            ng, npd = int(gmask.sum()), int(pmask.sum())
            rec = float(tp_g[gmask].mean()) if ng else float("nan")
            prec = float(tp_p[pmask].mean()) if npd else float("nan")
            rows.append(dict(dimension=dim, value=str(val), gt=ng, recall=rec, pred=npd, precision=prec,
                             delta=rec - total_rec if ng and total_rec == total_rec else float("nan"),
                             members=[("gt", int(gi[i])) for i in np.nonzero(gmask & ~tp_g)[0][:2000]]))
    return rows
