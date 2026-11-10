# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Split health (Qt-free): are train / val / test fit to train and evaluate on?

    rep = analyze(files, hashes, params)
        files   [(key, split, Counter labels)]     hashes  {key: phash ndarray} (optional — leakage check)
    rep = {"splits": [...], "table": {split: {"images", "instances", "empty", "cls_img": Counter, "cls_inst": Counter}},
           "findings": [Finding], "moves": [Move], "groups": [[keys]] (leaking groups), "js": {split: divergence}}

Findings (each with a severity and a plain-language "why it matters"):
    missing in train        a class that val/test have but train does not — the model never learns it
    missing in split        a class train has but val/test do not — it is never measured
    too few                 fewer than `min_images` images of a class in a split — its metric is noise
    share skew              a class is 2× over/under-represented in a split compared with train (and not tiny)
    distribution drift      Jensen–Shannon divergence of a split's class mix vs train above `js_warn`
    split size              val or test holds < 5 % of the images (or more than train)
    leakage                 near-identical images in different splits — the score is inflated by memorised images
Suggested moves (a plan, nothing is moved until applied):
    leakage   every member of a leaking group goes to the group's majority split (ties: train)
    missing / too few in val·test   images with that class move from train, preferring images that hold few other
              classes and are not part of a leaking group, never taking train below `min_images` for any class
    missing in train   half of the val/test images with that class move to train
Moves keep the dataset's own layout: the split folder name in each file's path is swapped (YOLO images/ + labels/
together, LabelMe JSON + image together, class folders keep the class). core.split_health.plan_paths does that.
"""
from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

VAL_NAMES = frozenset({"val", "valid", "validation"})


def norm(split: str) -> str:
    s = (split or "").lower()
    return "val" if s in VAL_NAMES else s


@dataclass
class Finding:
    severity: str            # error | warn | info
    kind: str
    split: str
    cls: str
    text: str
    why: str
    keys: tuple = ()


@dataclass
class Move:
    key: str
    src: str
    dst: str
    reason: str


def js_divergence(p: Counter, q: Counter) -> float:
    """Jensen–Shannon divergence (base 2, 0..1) of two count distributions."""
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    if not keys or not sp or not sq:
        return 0.0
    out = 0.0
    for k in keys:
        a, b = p.get(k, 0) / sp, q.get(k, 0) / sq
        m = (a + b) / 2
        if a:
            out += 0.5 * a * math.log2(a / m)
        if b:
            out += 0.5 * b * math.log2(b / m)
    return max(0.0, min(1.0, out))


def analyze(files, hashes=None, min_images: int = 10, js_warn: float = 0.05, skew: float = 2.0,
            hamming: int = 10, val_share: float = 0.15) -> dict:
    by_split: dict[str, list] = defaultdict(list)
    labels_of = {}
    for key, split, counts in files:
        sp = norm(split)
        by_split[sp].append(key)
        labels_of[key] = counts or Counter()
    order = {"train": 0, "val": 1, "test": 2}
    splits = sorted(by_split, key=lambda s: (order.get(s, 9), s))
    table = {}
    for sp in splits:
        ci, cn, empty = Counter(), Counter(), 0
        for k in by_split[sp]:
            c = labels_of[k]
            if not c:
                empty += 1
            ci.update({lab: 1 for lab in c})
            cn.update(c)
        table[sp] = {"images": len(by_split[sp]), "instances": sum(cn.values()), "empty": empty,
                     "cls_img": ci, "cls_inst": cn}
    rep = {"splits": splits, "table": table, "findings": [], "moves": [], "groups": [], "js": {}}
    F = rep["findings"]
    total = sum(t["images"] for t in table.values()) or 1
    if len(splits) < 2 or "train" not in table:
        F.append(Finding("info", "no splits", "", "", "No train / val / test split found",
                         "Split health compares splits; Export & Split can build them."))
        return rep
    tr = table["train"]
    classes = sorted(set().union(*(t["cls_img"] for t in table.values())), key=str.lower)
    for sp in splits:
        t = table[sp]
        share = t["images"] / total
        if sp != "train" and share < 0.05:
            F.append(Finding("warn", "split size", sp, "", f"{sp} holds only {share:.1%} of the images ({t['images']:,})",
                             "A very small evaluation split gives unstable scores."))
        if sp != "train" and t["images"] > tr["images"]:
            F.append(Finding("warn", "split size", sp, "", f"{sp} is larger than train ({t['images']:,} vs "
                             f"{tr['images']:,})", "Usually train holds most of the data."))
    for sp in splits:
        if sp == "train":
            continue
        t = table[sp]
        rep["js"][sp] = d = js_divergence(t["cls_inst"], tr["cls_inst"])
        if d > js_warn:
            F.append(Finding("warn" if d < 3 * js_warn else "error", "distribution drift", sp, "",
                             f"{sp} class mix differs from train (JS divergence {d:.3f})",
                             "Scores on this split will not reflect the data the model was trained on."))
        n_tr, n_sp = sum(tr["cls_inst"].values()) or 1, sum(t["cls_inst"].values()) or 1
        for c in classes:
            a, b = tr["cls_img"].get(c, 0), t["cls_img"].get(c, 0)
            keys = tuple(k for k in by_split[sp] if labels_of[k].get(c))[:2000] if b else ()
            if b and not a:
                F.append(Finding("error", "missing in train", sp, c, f"'{c}' is in {sp} ({b} images) but not in train",
                                 "The model never sees this class while training.", keys))
            elif a and not b:
                F.append(Finding("error" if a >= min_images else "warn", "missing in split", sp, c,
                                 f"'{c}' has no images in {sp} (train has {a})",
                                 "This class is never measured — its errors are invisible."))
            elif b < min_images:
                F.append(Finding("warn", "too few", sp, c, f"'{c}': only {b} image(s) in {sp}",
                                 f"With fewer than {min_images} images the class score is mostly noise.", keys))
            elif a:
                r = (t["cls_inst"].get(c, 0) / n_sp) / max(tr["cls_inst"].get(c, 0) / n_tr, 1e-9)
                if (r >= skew or r <= 1 / skew) and t["cls_inst"].get(c, 0) >= 5:
                    F.append(Finding("info", "share skew", sp, c,
                                     f"'{c}' is {r:.1f}× as common in {sp} as in train" if r > 1 else
                                     f"'{c}' is {1 / r:.1f}× rarer in {sp} than in train",
                                     "The split weighs this class differently from training.", keys))
    # ---- leakage: near-identical images across splits
    split_of = {k: norm(sp) for k, sp, _c in files}
    in_group = set()
    if hashes:
        from .profile import near_duplicate_pairs, union_find_groups
        ks = [k for k in hashes if k in split_of]
        pairs = near_duplicate_pairs([hashes[k] for k in ks], hamming)
        for g in union_find_groups(len(ks), [(i, j) for i, j, _d in pairs]):
            members = [ks[i] for i in g]
            sps = Counter(split_of[k] for k in members)
            if len(sps) > 1:
                rep["groups"].append(members)
                in_group.update(members)
                best = max(sps, key=lambda s: (sps[s], s == "train"))
                for k in members:
                    if split_of[k] != best:
                        rep["moves"].append(Move(k, split_of[k], best, f"near-duplicate of {sps[best]} {best} image(s)"))
        if rep["groups"]:
            n = sum(len(g) for g in rep["groups"])
            F.append(Finding("error", "leakage", "", "", f"{len(rep['groups']):,} groups of near-identical images span "
                             f"splits ({n:,} images)", "Evaluation includes images the model has effectively "
                             "trained on — scores are inflated.", tuple(k for g in rep["groups"] for k in g)[:5000]))
    # ---- moves for coverage
    moving = {m.key for m in rep["moves"]}
    rarity = {c: tr["cls_img"].get(c, 0) for c in classes}
    tr_img = Counter(tr["cls_img"])

    def pick(cands, need, protect_train):
        out = []
        cands = sorted(cands, key=lambda k: (len(labels_of[k]), sum(1 / (1 + rarity.get(c, 0)) for c in labels_of[k]), k))
        for k in cands:
            if len(out) >= need:
                break
            if k in moving or k in in_group:
                continue
            if protect_train and any(tr_img[c] - 1 < min_images for c in labels_of[k]):
                continue
            out.append(k)
            if protect_train:
                tr_img.subtract({c: 1 for c in labels_of[k]})
        return out

    for f in list(F):
        if f.kind in ("missing in split", "too few") and f.split != "train":
            have = table[f.split]["cls_img"].get(f.cls, 0)
            want = max(min_images, math.ceil((tr["cls_img"].get(f.cls, 0) + have) * val_share))
            need = max(0, want - have)
            avail = tr_img.get(f.cls, 0) - min_images
            need = min(need, max(0, avail))
            cands = [k for k in by_split["train"] if labels_of[k].get(f.cls)]
            for k in pick(cands, need, True):
                rep["moves"].append(Move(k, "train", f.split, f"gives {f.split} images of '{f.cls}'"))
                moving.add(k)
        elif f.kind == "missing in train":
            cands = [k for k in by_split[f.split] if labels_of[k].get(f.cls)]
            for k in pick(cands, max(1, len(cands) // 2), False):
                rep["moves"].append(Move(k, f.split, "train", f"gives train images of '{f.cls}'"))
                moving.add(k)
    return rep


# ---------------------------------------------------------------------------------------------------- moving
def split_dirs(root, samples) -> dict[str, str]:
    """split (normalised) -> the folder name the dataset uses for it ("valid", "Val", …)."""
    root = Path(root)
    seen: dict[str, Counter] = defaultdict(Counter)
    for s in samples:
        sp = norm(s.split)
        if not sp:
            continue
        p = Path(s.ann_path or s.image_path)
        try:
            parts = p.relative_to(root).parts[:-1]
        except ValueError:
            continue
        for part in parts:
            if norm(part) == sp:
                seen[sp][part] += 1
                break
    return {sp: c.most_common(1)[0][0] for sp, c in seen.items()}


def swap_split(root, path, src: str, dst_name: str) -> Path | None:
    """The same path with its (first) `src` split folder renamed to `dst_name`; None when the path has none."""
    root = Path(root)
    p = Path(path)
    try:
        parts = list(p.relative_to(root).parts)
    except ValueError:
        return None
    for i, part in enumerate(parts[:-1]):
        if norm(part) == norm(src):
            parts[i] = dst_name
            return root.joinpath(*parts)
    return None


def plan_paths(root, sample, image_path, dst: str, dirs: dict) -> list[tuple[Path, Path]]:
    """[(src, dst)] for the files of one sample. Raises ValueError when it cannot be moved safely."""
    name = dirs.get(norm(dst))
    if name is None:
        raise ValueError(f"no '{dst}' folder in this dataset")
    out = []
    for f in dict.fromkeys(p for p in (sample.ann_path, image_path) if p is not None):
        f = Path(f)
        if not f.exists():
            continue
        t = swap_split(root, f, sample.split, name)
        if t is None:
            continue                        # shared file outside the split folders: stays (LabelMe imagePath fixed)
        if t.exists():
            raise ValueError(f"{t.relative_to(root)} already exists")
        out.append((f, t))
    if not out:
        raise ValueError("its files are not inside a split folder")
    return out


def fix_image_path(json_path: Path, image_path: Path):
    """After a LabelMe JSON moved without its image (or vice versa): imagePath relative to the new place."""
    import json
    from .io import atomic_write_text
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    new = Path(os.path.relpath(image_path, json_path.parent)).as_posix()
    if isinstance(data, dict) and data.get("imagePath") not in (None, new):
        data["imagePath"] = new
        atomic_write_text(json_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
