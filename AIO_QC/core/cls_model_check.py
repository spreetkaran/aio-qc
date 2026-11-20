# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Model Check for classification datasets (Qt-free): a classifier's top classes for every image against its label.

    per image   {"key", "split", "gt": [classes], "pred": [(class, conf), …]  (top 5, mapped to dataset classes)}
    findings(…) likely mislabels (the model is sure of another class), wrong but unsure, the model unsure,
                images without a class (with the model's suggestion)
    metrics(…)  accuracy (top-1 / top-5), per class precision / recall / F1 / support, the confusion matrix

Multi-label datasets: a prediction is right when its top class is one of the image's classes.
"""
from __future__ import annotations

from collections import Counter, defaultdict

LIKELY, WRONG, UNSURE, NO_CLASS = "Likely mislabel", "Model disagrees", "Model unsure", "No class — model suggests"


def map_probs(probs, cmap: dict, names: dict) -> list[tuple[str, float]]:
    """[[model class id, conf]] -> [(dataset class, conf)], highest first; a class the map leaves out is skipped,
    two model classes on one dataset class add up."""
    acc = defaultdict(float)
    for k, c in probs or []:
        lab = cmap.get(str(k)) if cmap else names.get(str(k), str(k))
        if lab:
            acc[lab] += float(c)
    return sorted(acc.items(), key=lambda x: -x[1])


def findings(recs, sure: float = 0.8, unsure: float = 0.5) -> list[dict]:
    out = []
    for r in recs:
        pred = r["pred"]
        if not pred:
            continue
        top, conf = pred[0]
        gt = r["gt"]
        if not gt:
            out.append(dict(r, kind=NO_CLASS, suggest=top, conf=conf, why=f"{top} {conf:.0%}"))
        elif top not in gt:
            gconf = next((c for k, c in pred if k in gt), 0.0)
            kind = LIKELY if conf >= sure else WRONG
            out.append(dict(r, kind=kind, suggest=top, conf=conf,
                            why=f"labelled {', '.join(gt)} ({gconf:.0%}) — the model says {top} ({conf:.0%})"))
        elif conf < unsure:
            alt = next(((k, c) for k, c in pred[1:]), None)
            out.append(dict(r, kind=UNSURE, suggest=top, conf=conf,
                            why=f"{top} only {conf:.0%}" + (f" — next {alt[0]} {alt[1]:.0%}" if alt else "")))
    order = {LIKELY: 0, NO_CLASS: 1, WRONG: 2, UNSURE: 3}
    out.sort(key=lambda f: (order[f["kind"]], -f["conf"]))
    return out


def metrics(recs) -> dict:
    """{"n", "top1", "top5", "classes": {c: {"tp", "fp", "fn", "support", "precision", "recall", "f1"}},
    "confusion": {gt: Counter(pred)}} — over images that have a class and a prediction."""
    n = top1 = top5 = 0
    cls = defaultdict(lambda: Counter())
    conf = defaultdict(Counter)
    for r in recs:
        if not r["gt"] or not r["pred"]:
            continue
        n += 1
        top = r["pred"][0][0]
        gt = r["gt"]
        ok = top in gt
        top1 += ok
        top5 += any(k in gt for k, _c in r["pred"][:5])
        conf[gt[0] if ok or len(gt) == 1 else gt[0]][top] += 1
        for g in gt:
            cls[g]["support"] += 1
        if ok:
            cls[top]["tp"] += 1
        else:
            cls[top]["fp"] += 1
            for g in gt:
                cls[g]["fn"] += 1
    out = {}
    for c, v in cls.items():
        p = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else 0.0
        rc = v["tp"] / v["support"] if v["support"] else 0.0
        out[c] = {"tp": v["tp"], "fp": v["fp"], "fn": v["fn"], "support": v["support"], "precision": p, "recall": rc,
                  "f1": 2 * p * rc / (p + rc) if p + rc else 0.0}
    return {"n": n, "top1": top1 / n if n else 0.0, "top5": top5 / n if n else 0.0, "classes": out,
            "confusion": {k: dict(v) for k, v in conf.items()}}
