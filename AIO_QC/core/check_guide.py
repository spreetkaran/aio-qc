# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Which QC checks to run first on this dataset — and then which next (Qt-free; widgets.check_guide shows it).

    sig = Signals(...)                    # cheap facts gathered when the dataset opens (+ a small sample of files)
    recs = recommend(sig, done)           # [Rec(tool, score, why)] best first; done = {tool: files flagged}

Order of importance, in short: files that cannot be read → broken images → label names that disagree → empty
files → splits / balance → duplicates (frames of a video) → relationship rules the data clearly follows → text
labels → models → finishing (sampling, fixing flagged files). A tool that already ran moves to “done” with what it
found; the next one is always the highest-scoring tool not run yet. Every reason is written from this dataset's own
facts (“3 files could not be read”, “‘Car ’ and ‘car’ look like one class”).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

FRAME_RE = re.compile(r"(?:frame|img|image|_|-)?\d{3,}$", re.IGNORECASE)


@dataclass
class Signals:
    files: int = 0
    unreadable: int = 0                     # annotation files that could not be parsed
    broken_images: int = 0                  # integrity sweep (may still be running: -1 = unknown)
    empty: int = 0                          # files without any shape / class
    labels: Counter = field(default_factory=Counter)        # instances per class
    class_list: list = field(default_factory=list)          # loaded / dataset classes ([] = none)
    splits: list = field(default_factory=list)
    frame_like: float = 0.0                 # share of file names that end in a frame number
    multi_class: float = 0.0                # share of sampled images with ≥ 2 classes
    parts: list = field(default_factory=list)               # [(child, parent, share inside)] from the sample
    texts: int = 0                          # sampled shapes with a text (description)
    polygons: float = 0.0                   # share of sampled shapes that are polygons
    models: int = 0                         # models set up for this dataset
    rules: int = 0                          # relationship rules saved
    flagged: int = 0
    reviewed: float = 0.0                   # share of files with a review mark
    classification: bool = False


@dataclass
class Rec:
    tool: str
    score: float
    why: str
    done: bool = False
    found: int | None = None                # files the tool flagged when it ran


def _norm(l: str) -> str:
    return re.sub(r"[\s_\-]+", "", l.strip().lower())


def label_variants(labels) -> list[list[str]]:
    """Class names that differ only by case, spaces, '-' or '_' (“Car ”, “car”, “c-ar”)."""
    groups: dict[str, list[str]] = {}
    for l in labels:
        groups.setdefault(_norm(str(l)), []).append(str(l))
    return [sorted(g) for g in groups.values() if len(g) > 1]


def frame_share(names) -> float:
    names = list(names)
    if not names:
        return 0.0
    import os
    return sum(1 for n in names if FRAME_RE.search(os.path.splitext(n)[0])) / len(names)


def _n(x) -> str:
    return f"{x:,}"


def recommend(s: Signals, done: dict | None = None) -> list[Rec]:
    """Every useful check for this dataset, best first; tools in `done` ({tool: files flagged}) are marked done
    and listed after the ones still to run."""
    done = done or {}
    r: dict[str, Rec] = {}

    def add(tool, score, why):
        if tool not in r or r[tool].score < score:
            r[tool] = Rec(tool, score, why)

    # ---- the files themselves
    if s.unreadable:
        add("validation", 100, f"{_n(s.unreadable)} annotation file(s) could not be read — fix these first")
    else:
        add("validation", 72, "file & geometry integrity: pairing, schema, points, image size — the base for "
                              "every other check")
    if s.broken_images > 0:
        add("image_quality", 96, f"{_n(s.broken_images)} image(s) are broken (corrupt / truncated)")
    else:
        add("image_quality", 34, "corrupt or rotated images, blur and exposure")
    # ---- label names
    variants = label_variants(s.labels)
    unknown = sorted(l for l in s.labels if s.class_list and l not in s.class_list)
    if variants:
        g = variants[0]
        add("ambiguity", 92, f"{' / '.join(repr(x) for x in g[:3])} look like one class"
                             + (f" (+{len(variants) - 1} more)" if len(variants) > 1 else ""))
    if unknown:
        add("ambiguity", 90, f"{len(unknown)} label(s) not in the class list: {', '.join(unknown[:3])}"
                             + (" …" if len(unknown) > 3 else ""))
    if not variants and not unknown and len(s.labels) > 1:
        add("ambiguity", 30, "misspelt / unknown label names")
    if s.empty:
        add("empty", 58 + min(20, 100 * s.empty / max(s.files, 1)),
            f"{_n(s.empty)} file(s) have no shapes — missing work or true negatives?")
    # ---- balance and splits
    rare = sorted((l for l, n in s.labels.items() if n < 20), key=lambda l: s.labels[l])
    if s.splits:
        add("split_health", 62 if rare else 52, f"splits {', '.join(s.splits)}: every class in every split, "
                                                "no leakage between them")
    add("stats", 50 if rare else 40, (f"rare classes: {', '.join(rare[:3])} (< 20 instances)" if rare else
                                      f"{len(s.labels)} classes — check the balance"))
    # ---- duplicates
    if s.frame_like >= 0.5:
        add("image_dups", 66, f"{s.frame_like * 100:.0f} % of the names look like video frames — near-identical "
                              "images inflate the data and leak across splits")
        add("sequence", 54, "consecutive frames: objects missing for a frame, label flips")
    else:
        add("image_dups", 36, "exact / near-duplicate images")
    if not s.classification:
        add("duplicates", 46, "the same object annotated twice")
        add("outliers", 38, "shapes whose size is unusual for their class")
        # ---- relationships
        if s.parts and not s.rules:
            c, p, sh = s.parts[0]
            add("rules", 60, f"{sh * 100:.0f} % of {c} lie inside a {p} — let Relationship Rules suggest the rules "
                             "your data follows, then find the files that break them")
        elif s.multi_class >= 0.3 and not s.rules:
            add("rules", 44, f"{s.multi_class * 100:.0f} % of images have several classes — rules for how they "
                             "relate")
        elif s.rules:
            add("rules", 56, f"{s.rules} relationship rule(s) saved — check the data against them")
    # ---- text, models
    if s.texts:
        add("text_labels", 55, f"{_n(s.texts)}+ shapes carry a text — check formats without OCR")
    if s.models:
        add("model_check", 64, f"{s.models} model(s) set up — where models and labels disagree")
    # ---- finishing
    if s.flagged:
        add("fix_editor", 30 + min(25, s.flagged / 10), f"{_n(s.flagged)} flagged file(s) — fix them in an editor")
    add("sampling", 24 if s.reviewed < 0.05 else 18, "review a random sample to estimate the error rate")
    out = list(r.values())
    for x in out:
        if x.tool in done:
            x.done, x.found = True, done[x.tool]
    out.sort(key=lambda x: (x.done, -x.score))
    return out


def next_check(recs: list[Rec]) -> Rec | None:
    return next((x for x in recs if not x.done), None)
