# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Improve Model (Qt-free): the loop train → check → label more → retrain, kept per dataset so it survives restarts.

    st = load(root)                          .aio_qc/improve.json — the run being improved, its model, the pool …
    spots = weak_spots(eval_data, conf, iou) where the new model is weak, in plain words, and what to do about it
    record_check(st, run, spots)             one line of history per check: is each round better than the last?
    keys = hardest_images(eval_data, conf, iou, n)   the dataset files with the most model / label disagreement
    stubs = label_stubs(images, kind)        empty label files so new images can be labelled where they are

A weak spot is a class whose recall or precision is low (with enough labels to tell); its advice follows its main
error type: missed objects → more examples / check the labels, class confusions → look at the pairs, loose boxes →
tighter boxes, false alarms → label what is really there or add negatives.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from .state import state_dir

FILE = "improve.json"
WEAK = 0.7                       # recall or precision below this: a weak spot
MIN_LABELS = 5                   # fewer labels than this: too few to judge (reported as such)
STEPS = ("train", "check", "label", "retrain")

ADVICE = {
    "miss": "the model misses many of them — label more images that show them (Pick with model uncertainty), and "
            "check that the labels it misses are real",
    "cls": "it confuses them with another class — compare the two in Crop Gallery; fix wrong labels, or add "
           "examples that tell them apart",
    "loc": "it finds them but its boxes do not fit the labels — check that the labels are drawn tightly and the "
           "same way everywhere",
    "both": "wrong class AND a loose box — usually clutter or a poorly drawn label: fix those labels first",
    "bkg": "it reports many that are not labelled — some may be real objects nobody labelled (Model Check ▸ "
           "Findings ▸ add box), the rest are false alarms: add images without them",
    "dupe": "it reports the same object twice — a model setting (NMS IoU), not a label problem",
}


def path(root) -> Path:
    return state_dir(root) / FILE


def load(root) -> dict:
    try:
        d = json.loads(path(root).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save(root, st: dict) -> None:
    p = path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def _f(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def weak_spots(d, conf: float = 0.25, iou: float = 0.5, view=None) -> dict:
    """-> {"overall": {precision, recall, f1, ap50, ap5095, gt, pred}, "classes": [{label, gt, precision, recall,
    ap5095, main_error, weak, advice}] weakest first, "weak": [labels], "few": [labels with too few labels]}"""
    from .evaluation import ERROR_TYPES, View, error_breakdown, report
    v = view or View()
    rows = report(d, conf, iou, v)
    eb = error_breakdown(d, conf, iou, v)
    keys = [k for k, *_r in ERROR_TYPES]
    per = eb["per_class"]
    out, weak, few = [], [], []
    for i, r in enumerate(rows[:-1]):
        if not r["gt"] and not r["pred"]:
            continue
        errs = {keys[j]: int(per[i, j]) for j in range(len(keys))} if i < len(per) else {}
        main = max(errs, key=errs.get) if errs and max(errs.values()) > 0 else ""
        if not main and (r["fn"] or r["fp"]):        # unmatched, but near other objects: the plain reading
            main = "miss" if r["fn"] >= r["fp"] else "bkg"
        rec, prec = r["recall"], r["precision"]
        too_few = r["gt"] < MIN_LABELS
        is_weak = not too_few and ((rec == rec and rec < WEAK) or (prec == prec and prec < WEAK))
        if too_few:
            advice = f"only {r['gt']} label(s) — too few to judge or to learn from: label more of them"
            few.append(r["label"])
        elif is_weak:
            advice = ADVICE.get(main, "check its labels")
            weak.append(r["label"])
        else:
            advice = "good"
        out.append({"label": r["label"], "gt": r["gt"], "pred": r["pred"], "precision": _f(prec), "recall": _f(rec),
                    "ap5095": _f(r["ap5095"]), "main_error": main, "errors": errs, "weak": is_weak,
                    "few": too_few, "advice": advice})
    order = {True: 0, False: 1}
    out.sort(key=lambda c: (order[c["weak"] or c["few"]], c["recall"] if c["recall"] is not None else 1.0))
    a = rows[-1]
    overall = {"precision": _f(a["precision"]), "recall": _f(a["recall"]), "f1": _f(a["f1"]), "ap50": _f(a["ap50"]),
               "ap5095": _f(a["ap5095"]), "gt": a["gt"], "pred": a["pred"]}
    return {"overall": overall, "classes": out, "weak": weak, "few": few, "conf": conf, "iou": iou}


def weak_spots_cls(m: dict) -> dict:
    """A classifier's check (Model Check for classification: its metrics()) in the shape weak_spots() returns —
    per class precision / recall over images, the class it is confused with most, top-1 / top-5 overall."""
    conf = m.get("confusion") or {}
    out, weak, few = [], [], []
    for label, c in (m.get("classes") or {}).items():
        mixed = {p: n for p, n in (conf.get(label) or {}).items() if p != label}
        other = max(mixed, key=mixed.get) if mixed else ""
        rec, prec = c.get("recall"), c.get("precision")
        too_few = c.get("support", 0) < MIN_LABELS
        is_weak = not too_few and ((rec is not None and rec < WEAK) or (prec is not None and prec < WEAK))
        if too_few:
            advice = f"only {c.get('support', 0)} image(s) — too few to judge or to learn from: add more of them"
            few.append(label)
        elif is_weak:
            advice = (f"it is confused with {other} — open Class Gallery ▸ Confusions ({label} → {other}): fix "
                      "wrong classes, add images that tell the two apart") if other else \
                "it is often called something else — Class Gallery ▸ Most suspicious first shows those images"
            weak.append(label)
        else:
            advice = "good"
        out.append({"label": label, "gt": c.get("support", 0), "pred": c.get("tp", 0) + c.get("fp", 0),
                    "precision": _f(prec), "recall": _f(rec), "ap5095": None, "main_error": "cls" if mixed else "",
                    "errors": {"cls": c.get("fn", 0)}, "confused_with": other, "weak": is_weak, "few": too_few,
                    "advice": advice})
    order = {True: 0, False: 1}
    out.sort(key=lambda c: (order[c["weak"] or c["few"]], c["recall"] if c["recall"] is not None else 1.0))
    ps = [c["precision"] for c in out if c["precision"] is not None]
    rs = [c["recall"] for c in out if c["recall"] is not None]
    p, r = (sum(ps) / len(ps) if ps else None), (sum(rs) / len(rs) if rs else None)
    overall = {"precision": _f(p), "recall": _f(r), "f1": _f(2 * p * r / (p + r)) if p and r else None,
               "ap50": None, "ap5095": None, "top1": _f(m.get("top1")), "top5": _f(m.get("top5")),
               "gt": m.get("n", 0), "pred": m.get("n", 0)}
    return {"overall": overall, "classes": out, "weak": weak, "few": few, "classifier": True}


def hardest_cls(recs, n: int = 50000) -> list[tuple[str, int]]:
    """A classifier's misclassified images, the surest mistakes first: [(file key, 1)]."""
    wrong = [(r["pred"][0][1], r["key"]) for r in recs if r.get("gt") and r.get("pred")
             and r["pred"][0][0] not in r["gt"]]
    return [(k, 1) for _c, k in sorted(wrong, key=lambda t: -t[0])[:n]]


def summary_text(spots: dict) -> str:
    """Plain words for the top of the check step."""
    o = spots.get("overall") or {}
    head = "Overall: " + " · ".join(f"{name} {o[k]:.2f}" for k, name in (("top1", "top-1 accuracy"),
                                                                       ("precision", "precision"),
                                                                       ("recall", "recall"),
                                                                       ("ap5095", "mAP50-95")) if o.get(k) is not None)
    weak = spots.get("weak") or []
    few = spots.get("few") or []
    if not weak and not few:
        return head + " — no weak class at these thresholds."
    parts = []
    if weak:
        parts.append(f"weak: {', '.join(weak[:6])}" + (f" (+{len(weak) - 6})" if len(weak) > 6 else ""))
    if few:
        parts.append(f"too few labels: {', '.join(few[:6])}")
    return head + " — " + "; ".join(parts) + "."


def hardest_images(d, conf: float = 0.25, iou: float = 0.5, n: int = 50, view=None) -> list[tuple[str, int]]:
    """[(file key, disagreements)] — most first: every unmatched prediction and every unmatched label counts once."""
    import numpy as np
    from .evaluation import View, state
    st = state(d, conf, iou, view or View())
    imgs = np.concatenate([d.gt_img[st.gm & ~st.matched_gt], d.p_img[st.live & ~st.matched_p]]).astype(int)
    count = Counter(imgs.tolist())
    return [(d.keys[i], c) for i, c in count.most_common(n)]


def record_check(st: dict, run: str, model_path: str, spots: dict, n_images: int) -> dict:
    """Append one round to the history (a re-check of the same model replaces its line)."""
    hist = [h for h in st.get("history", []) if h.get("model") != model_path]
    o = spots.get("overall") or {}
    hist.append({"when": time.strftime("%Y-%m-%d %H:%M"), "run": run, "model": model_path, "images": n_images,
                 "precision": o.get("precision"), "recall": o.get("recall"), "ap5095": o.get("ap5095"),
                 "top1": o.get("top1"), "weak": list(spots.get("weak") or [])})
    st["history"] = hist[-50:]
    st["last_check"] = {"model": model_path, "spots": spots}
    st.setdefault("done", {})["check"] = time.time()
    return st


def trend(st: dict) -> str:
    """'mAP50-95 0.41 → 0.47 (+0.06) since the last round' (a classifier: top-1 accuracy) — or ''."""
    hist = st.get("history", [])
    key, name = ("top1", "top-1") if hist and hist[-1].get("top1") is not None else ("ap5095", "mAP50-95")
    h = [x for x in hist if x.get(key) is not None]
    if len(h) < 2:
        return ""
    a, b = h[-2][key], h[-1][key]
    word = "better" if b > a + 0.005 else "worse" if b < a - 0.005 else "about the same"
    return f"{name} {a:.2f} → {b:.2f} ({b - a:+.2f}) since the round before — {word}"


def label_stubs(images, kind: str = "labelme") -> list[str]:
    """An empty label file next to each new image (LabelMe JSON, or a YOLO txt in the matching labels folder) so
    it can be labelled in place. Existing label files are left alone. -> the label files written."""
    from .io import image_size
    out = []
    for img in images:
        img = Path(img)
        if kind == "labelme":
            j = img.with_suffix(".json")
            if j.exists():
                continue
            wh = image_size(img) or (0, 0)
            from . import provenance as PV
            doc = {"version": "5.4.1", "flags": {}, "shapes": [], "imagePath": img.name,
                   "imageData": None, "imageHeight": int(wh[1]), "imageWidth": int(wh[0])}
            PV.stamp_labelme(doc, o=PV.make_origin("rule", "Improve Model", detail="empty label file to fill in"))
            j.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            out.append(str(j))
        else:
            parts = list(img.parts)
            if "images" not in parts:
                continue
            i = len(parts) - 1 - parts[::-1].index("images")
            t = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            if t.exists():
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text("", encoding="utf-8")
            out.append(str(t))
    return out
