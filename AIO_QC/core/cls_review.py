# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Reviewing a classification dataset fast (Qt-free; used by the Class Gallery):

    model_view(…)        what the classifier says about one image: sure of another class / another class / its own
                         class but unsure / fine — with the numbers for a tile badge
    order_keys(…)        the review orders: most suspicious first (model findings first, then the least typical by
                         look), the model least sure of the image's own class first, least typical first, grouped by
                         look, file order, random
    confusions(…)        {(class, model's class): [images]} — the cells of the confusion matrix, as image lists
    lookalikes(…)        images that have a near-identical image in ANOTHER class (vectors of a Visual Similarity run)
    sample_for(…)        a random sample of a class big enough for an error estimate (± margin, 95 %)
    signoff(…)           the estimate from a checked sample: wrong share with its Wilson interval, in images
"""
from __future__ import annotations

import random

import numpy as np

SURE, UNSURE = 0.8, 0.5
FILE_ORDER, SUSPICIOUS, LEAST_SURE, LEAST_TYPICAL, BY_LOOK, RANDOM = (
    "File order", "Most suspicious first", "Model: least sure first", "Least typical first", "Grouped by look",
    "Random")
SORTS = [SUSPICIOUS, LEAST_SURE, LEAST_TYPICAL, BY_LOOK, FILE_ORDER, RANDOM]
NEEDS_MODEL = {LEAST_SURE}
NEEDS_LOOK = {LEAST_TYPICAL, BY_LOOK}


def model_view(gt, pred, sure: float = SURE, unsure: float = UNSURE) -> tuple[int, float, str, float, float]:
    """(tier, within, top class, its confidence, confidence in the image's own class).
    tier 3: the model is sure of another class · 2: another class · 1: its own class, unsure · 0: fine / no
    prediction. `within` orders images inside a tier (higher = look first)."""
    if not pred:
        return 0, 0.0, "", 0.0, 0.0
    top, conf = pred[0]
    gt = list(gt or [])
    own = next((c for k, c in pred if k in gt), 0.0)
    if not gt or top not in gt:
        return (3 if conf >= sure else 2), float(conf), top, float(conf), float(own)
    if own < unsure:
        return 1, 1.0 - own, top, float(conf), float(own)
    return 0, 1.0 - own, top, float(conf), float(own)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def atypicality(keys, rows: dict, vecs) -> dict:
    """{key: 0 … 1} for the keys that have a vector — 1 = the least typical of them (rank, so it does not depend on
    the model's similarity scale)."""
    have = [k for k in keys if k in rows]
    if len(have) < 3 or vecs is None:
        return {}
    v = _unit(vecs[[rows[k] for k in have]])
    c = v.mean(0)
    c /= max(float(np.linalg.norm(c)), 1e-9)
    typ = v @ c
    rank = np.empty(len(have), dtype=np.float64)
    rank[np.argsort(typ, kind="stable")] = np.arange(len(have))          # 0 = least typical
    return {k: 1.0 - r / (len(have) - 1) for k, r in zip(have, rank)}


def order_keys(keys, mode: str, tags_of, preds: dict | None = None, rows: dict | None = None, vecs=None,
               seed: int = 0, sure: float = SURE, unsure: float = UNSURE) -> list:
    """The keys in a review order. Keys a sort cannot rank (no prediction / no vector) keep file order, last."""
    keys = list(keys)
    preds = preds or {}
    rows = rows or {}
    if mode == RANDOM:
        rnd = random.Random(seed)
        keys = keys[:]
        rnd.shuffle(keys)
        return keys
    if mode == SUSPICIOUS:
        look = atypicality(keys, rows, vecs)

        def score(k):
            t, w, *_ = model_view(tags_of(k), preds.get(k), sure, unsure)
            lk = look.get(k, 0.0)
            return (t, w + 0.5 * lk if t <= 1 else w, lk)
        return sorted(keys, key=score, reverse=True)                     # stable: ties keep file order
    if mode == LEAST_SURE:
        have = [k for k in keys if preds.get(k)]
        rest = [k for k in keys if not preds.get(k)]
        return sorted(have, key=lambda k: model_view(tags_of(k), preds[k], sure, unsure)[4]) + rest
    if mode == LEAST_TYPICAL:
        look = atypicality(keys, rows, vecs)
        return sorted([k for k in keys if k in look], key=lambda k: -look[k]) + [k for k in keys if k not in look]
    if mode == BY_LOOK:
        from AIO_QC.core.embeddings import look_order
        have = [k for k in keys if k in rows]
        if len(have) < 3 or vecs is None:
            return keys
        order = look_order(_unit(vecs[[rows[k] for k in have]]), seed=seed)
        return [have[i] for i in order] + [k for k in keys if k not in rows]
    return keys


def confusions(keys, tags_of, preds: dict) -> dict:
    """{(class, model's top class): [keys, most confident first]} for images whose top class is not one of theirs
    ("" = the image has no class)."""
    out: dict = {}
    for k in keys:
        p = preds.get(k)
        if not p:
            continue
        top, conf = p[0]
        gt = list(tags_of(k) or [])
        if top in gt:
            continue
        for g in gt or [""]:
            out.setdefault((g, top), []).append((conf, k))
    return {pair: [k for _c, k in sorted(v, key=lambda t: -t[0])] for pair, v in
            sorted(out.items(), key=lambda kv: -len(kv[1]))}


def lookalikes(keys, others, rows: dict, vecs, threshold: float = 0.95, chunk: int = 1024) -> dict:
    """{key: (other key, similarity)} — for each image of `keys` the most similar image among `others` (images of
    other classes) when at least `threshold` alike."""
    a_keys = [k for k in keys if k in rows]
    b_keys = [k for k in others if k in rows]
    if not a_keys or not b_keys or vecs is None:
        return {}
    A = _unit(vecs[[rows[k] for k in a_keys]])
    B = _unit(vecs[[rows[k] for k in b_keys]])
    best = np.full(len(a_keys), -2.0, dtype=np.float32)
    arg = np.zeros(len(a_keys), dtype=np.int64)
    step_b = max(1, (64 * 2 ** 20) // max(1, 4 * chunk))         # ~64 MB of similarities at a time
    for i in range(0, len(a_keys), chunk):
        a = A[i:i + chunk]
        for j in range(0, len(b_keys), step_b):
            s = a @ B[j:j + step_b].T
            m = s.argmax(1)
            v = s[np.arange(len(a)), m]
            better = v > best[i:i + chunk]
            best[i:i + chunk][better] = v[better]
            arg[i:i + chunk][better] = m[better] + j
    return {k: (b_keys[int(arg[n])], float(best[n])) for n, k in enumerate(a_keys) if best[n] >= threshold}


def sample_for(keys, margin: float = 0.05, confidence: float = 0.95, seed: int = 0) -> list:
    """A random sample big enough to estimate the share of wrong images to ± margin."""
    from AIO_QC.core.sampling import sample_size
    keys = list(keys)
    n = sample_size(len(keys), confidence, margin)
    return random.Random(seed).sample(keys, n) if n < len(keys) else keys


def signoff(checked: int, wrong: int, population: int, confidence: float = 0.95) -> dict:
    """{"p", "lo", "hi"} (share of wrong images, with its interval) and the same in images ("est", "est_lo",
    "est_hi") for the whole class."""
    from AIO_QC.core.sampling import wilson
    p, lo, hi = wilson(wrong, checked, confidence)
    return {"checked": checked, "wrong": wrong, "size": population, "p": p, "lo": lo, "hi": hi,
            "est": round(p * population), "est_lo": int(lo * population), "est_hi": int(round(hi * population))}
