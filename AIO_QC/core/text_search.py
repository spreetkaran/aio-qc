# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Search in words + class descriptions (Qt-free, numpy only) — on the vectors of a text-aware vision model (CLIP /
SigLIP / SigLIP2), the same vectors Visual Similarity caches.

Search          a query ("truck at night in rain") becomes a vector; images / objects are ranked by cosine
                similarity. One matrix-vector product: instant on millions once the images are embedded.
Class           one line per class ("trailer_text: painted container number on a trailer side"), kept in
descriptions    <dataset>/.aio_qc/class_descriptions.json (shared with Claude Review). Each object is scored against
                every description; scores are standardised PER DESCRIPTION (z over all objects) so a description
                that is simply "louder" for the model does not win everywhere. Findings:
                    other     its own description scores clearly below another class's
                    poor      it matches its own description far worse than the rest of its class
                Plus per class: how often the model agrees with the label, and which class it confuses it with —
                a low agreement means the description (or the class itself) is hard to tell apart.
Raw cosine values differ per model (SigLIP ≈ 0.05–0.2, CLIP ≈ 0.2–0.35): only ranks and standardised scores are
used, never fixed similarity thresholds.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DESC_FILE = "class_descriptions.json"
DEFAULT_TEMPLATE = "a photo of {}."


# ---------------------------------------------------------------------------------------------- descriptions
def default_description(name: str) -> str:
    return " ".join(str(name).replace("_", " ").replace("-", " ").split())


def load_descriptions(root) -> dict:
    """{"template": str, "classes": {name: description}} — missing file -> defaults."""
    p = Path(root) / ".aio_qc" / DESC_FILE
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d, dict) and isinstance(d.get("classes"), dict):
            return {"template": str(d.get("template", DEFAULT_TEMPLATE)),
                    "classes": {str(k): str(v) for k, v in d["classes"].items()}}
    except (OSError, ValueError):
        pass
    return {"template": DEFAULT_TEMPLATE, "classes": {}}


def save_descriptions(root, data: dict):
    p = Path(root) / ".aio_qc" / DESC_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"template": data.get("template", DEFAULT_TEMPLATE),
                               "classes": dict(sorted((data.get("classes") or {}).items()))},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def descriptions_for(names, data: dict) -> list[str]:
    cl = data.get("classes") or {}
    return [cl.get(n) or default_description(n) for n in names]


def phrase(text: str, template: str | None) -> str:
    text = " ".join(str(text).split())
    if not template or "{}" not in template:
        return text
    return template.replace("{}", text)


# ---------------------------------------------------------------------------------------------- search
def rank(vecs: np.ndarray, tvec: np.ndarray, valid: np.ndarray | None = None, top: int = 200,
         allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(indices best first, cosine scores) of the top items for one text vector."""
    if not len(vecs):
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    t = np.asarray(tvec, np.float32).reshape(-1)
    s = vecs.astype(np.float32, copy=False) @ t
    mask = np.ones(len(s), bool) if valid is None else valid.copy()
    if allowed is not None:
        mask &= allowed
    s = np.where(mask, s, -np.inf)
    k = int(min(max(top, 1), int(mask.sum())))
    if k <= 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    part = np.argpartition(-s, k - 1)[:k]
    order = part[np.argsort(-s[part], kind="stable")]
    return order, s[order]


def relative(scores: np.ndarray, all_scores_median: float | None = None) -> np.ndarray:
    """Scores re-expressed 0…100 inside the result list (the top hit = 100, the median of all items = 0)."""
    if not len(scores):
        return scores
    lo = float(all_scores_median) if all_scores_median is not None else float(scores.min())
    hi = float(scores.max())
    return np.clip((scores - lo) / max(hi - lo, 1e-9) * 100, 0, 100)


def split_counts(hit_splits: list[str], all_splits: list[str]) -> list[tuple[str, int, int]]:
    """[(split, hits, items in that split)] — hits per split next to the split's size (coverage in words)."""
    h, a = Counter(hit_splits), Counter(all_splits)
    order = [s for s in ("train", "val", "valid", "test", "") if s in a] + sorted(s for s in a if s not in
                                                                                    ("train", "val", "valid", "test", ""))
    return [(s, h.get(s, 0), a[s]) for s in order]


# ---------------------------------------------------------------------------------------------- descriptions check
@dataclass
class DescFinding:
    i: int
    kind: str              # other | poor
    label: str
    suggest: str
    score: float
    evidence: str


@dataclass
class DescReport:
    findings: list = field(default_factory=list)
    agreement: dict = field(default_factory=dict)      # class -> share of its objects the model puts in it
    confused: dict = field(default_factory=dict)       # class -> (other class, share)
    counts: dict = field(default_factory=dict)
    close_texts: list = field(default_factory=list)    # [(a, b, cosine)] descriptions too alike


DESC_KINDS = {"other": "Looks like another class", "poor": "Unlike its class description"}


def check_descriptions(vecs: np.ndarray, labels: list[str], names: list[str], tmat: np.ndarray,
                       valid: np.ndarray | None = None, margin: float = 1.0, poor_z: float = 2.5) -> DescReport:
    """vecs [n, d] items, labels per item, names [c] classes in the order of tmat [c, d] (description vectors)."""
    n, c = len(labels), len(names)
    rep = DescReport()
    if n == 0 or c < 1:
        return rep
    valid = np.ones(n, bool) if valid is None else valid
    code = {nm: j for j, nm in enumerate(names)}
    own = np.fromiter((code.get(lab, -1) for lab in labels), dtype=np.int64, count=n)
    use = valid & (own >= 0)
    S = vecs.astype(np.float32) @ tmat.astype(np.float32).T                     # [n, c]
    if use.sum() < 2:
        return rep
    mu, sd = S[use].mean(0), S[use].std(0)
    Z = (S - mu) / np.where(sd > 1e-9, sd, 1.0)
    pred = np.argmax(Z, axis=1) if c > 1 else np.zeros(n, np.int64)
    rows = np.nonzero(use)[0]
    zown = Z[rows, own[rows]]
    # per class: agreement and the class it is mistaken for
    for j, nm in enumerate(names):
        m = rows[own[rows] == j]
        rep.counts[nm] = int(len(m))
        if not len(m):
            continue
        p = pred[m]
        rep.agreement[nm] = float((p == j).mean()) if c > 1 else 1.0
        other = Counter(int(x) for x in p if x != j)
        if other:
            b, cnt = other.most_common(1)[0]
            rep.confused[nm] = (names[b], cnt / len(m))
    # within-class robust z of the own score (for "poor")
    zin = np.zeros(n, np.float32)
    for j in range(c):
        m = rows[own[rows] == j]
        if len(m) < 5:
            continue
        v = Z[m, j]
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826 or 1e-6
        zin[m] = (v - med) / mad
    for r, zo in zip(rows, zown):
        j = int(own[r])
        b = int(pred[r])
        if c > 1 and b != j and Z[r, b] - zo >= margin and zo < 0:
            rep.findings.append(DescFinding(int(r), "other", labels[r], names[b], float(Z[r, b] - zo),
                                            f"matches '{names[b]}' ({Z[r, b]:+.1f}) far better than its own "
                                            f"'{labels[r]}' ({zo:+.1f}) — scores standardised per description"))
        elif zin[r] <= -poor_z:
            rep.findings.append(DescFinding(int(r), "poor", labels[r], "", float(-zin[r]),
                                            f"matches '{labels[r]}' much worse than the rest of its class "
                                            f"({abs(zin[r]):.1f} robust SDs below)"))
    rep.findings.sort(key=lambda f: -f.score)
    # descriptions that are nearly the same text for the model
    T = tmat.astype(np.float32)
    if c > 1:
        G = T @ T.T
        iu = np.triu_indices(c, 1)
        vals = G[iu]
        thr = float(np.median(vals) + 2 * (np.std(vals) or 0)) if len(vals) > 2 else 0.95
        for a, b, v in zip(iu[0], iu[1], vals):
            if v >= max(thr, 0.9):
                rep.close_texts.append((names[a], names[b], float(v)))
    return rep
