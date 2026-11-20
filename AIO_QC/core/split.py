# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Split planner (train / val / optional test) — adapted from the pipeline's 02_split_dataset.py, made scalable.

    items = [SplitItem(key, features, group), …]        features: {kind: [values]} per image
    plan = plan_split(items, val=0.10, test=0.0, seed=42)
    plan.assign[key] -> "train" | "val" | "test";  plan.report -> per feature / class / group diagnostics

Rules (as in the script): groups (duplicates, frames of one video, one OCR identity …) are indivisible; features seen
in <= rare_max images are reserved for train; every feature keeps >= 1 example in train; per-feature caps keep val /
test representative without draining train; selection by information gain under these hard constraints.
Scalable: LAZY greedy (a priority queue; only the top candidate is re-scored — scores only fall as a split fills),
O(n log n) instead of re-scoring every group for every pick. Then a REPAIR pass (the script's swap improvement, aimed):
a feature still missing from val / test gets a train group swapped in for a val / test group of the same size, when
that keeps every cap, train coverage and the features val / test already had.

    items = build_items(samples, adapter, cache, ...)   # features + groups from the dataset (Convert & Split)
    write_reports(folder, items, plan, settings)        # split_report.txt, feature_summary.csv, feature_map.csv, json
"""
from __future__ import annotations

import heapq
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from AIO_QC.core.diag import ignored as _ignored


@dataclass
class SplitItem:
    key: str
    features: dict                  # kind -> list of values
    group: str = ""                 # same group = same split
    labels: Counter = field(default_factory=Counter)

    def keys(self) -> list[str]:
        return [f"{k}:{v}" for k, vs in self.features.items() for v in set(vs)]


# =============================================================================
# Features
# =============================================================================
def _bucket(v, edges, names):
    for e, n in zip(edges, names):
        if v < e:
            return n
    return names[-1]


def detection_features(labels: Counter, boxes_norm, lighting: str | None = None, focus=None) -> dict:
    """labels: label -> count; boxes_norm: [(x1, y1, x2, y2) 0..1] of the objects (may be empty).
    focus: the classes that drive the balance (combination, object count, size, position) — e.g. the "whole" classes
    when others are parts of them (a text line on a plate); every class still counts as a class feature."""
    f: dict[str, list] = {"class": sorted(labels) or ["(background)"]}
    if focus is not None:                       # boxes may carry their label as a 5th value
        labels = Counter({k: v for k, v in labels.items() if k in focus})
        boxes_norm = [b for b in boxes_norm if len(b) < 5 or b[4] in focus]
    n = sum(labels.values())
    f["objects"] = ["0" if n == 0 else str(n) if n <= 3 else "4+"]
    f["combination"] = ["+".join(sorted(labels)) if labels else "(empty)"]
    if boxes_norm:
        areas = [max(0.0, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes_norm]
        f["size"] = [_bucket(max(areas), [0.002, 0.01, 0.05, 0.20], ["tiny", "small", "medium", "large", "very large"])]
        cx = sum((b[0] + b[2]) / 2 for b in boxes_norm) / len(boxes_norm)
        cy = sum((b[1] + b[3]) / 2 for b in boxes_norm) / len(boxes_norm)
        f["position"] = [_bucket(cy, [0.33, 0.66], ["top", "middle", "bottom"]) + "-" +
                         _bucket(cx, [0.33, 0.66], ["left", "centre", "right"])]
    if lighting:
        f["lighting"] = [lighting]
    return f


def classification_features(tags: list[str], folder: str = "", lighting: str | None = None) -> dict:
    f = {"class": sorted(tags) or ["(none)"]}
    if folder:
        f["folder"] = [folder.split("/")[0]]
    if lighting:
        f["lighting"] = [lighting]
    return f


_FRAME = re.compile(r"(.*?)(?:[_\-. ]?(?:frame|frm|f|img|image)?[_\-. ]?\d{2,})$", re.I)


def sequence_id(name_stem: str, folder: str) -> str | None:
    """'cam1_clip03_frame_0042' -> 'cam1_clip03' (frames of one video share it); None if there is no frame number."""
    m = _FRAME.match(name_stem)
    if not m or not m.group(1):
        return None
    return f"{folder}|{m.group(1).rstrip('_-. ').lower()}"


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p.setdefault(x, x)
        while p != self.p[p]:
            self.p[p] = self.p[self.p[p]]
            p = self.p[p]
        self.p[x] = p
        return p

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_groups(keys: list[str], links: list[tuple[str, str]]) -> dict[str, str]:
    """links: pairs that must stay together (duplicates, same sequence, same identity) -> key -> group id."""
    uf = _UF()
    for k in keys:
        uf.find(k)
    for a, b in links:
        uf.union(a, b)
    return {k: uf.find(k) for k in keys}


# =============================================================================
# Planning
# =============================================================================
@dataclass
class SplitPlan:
    assign: dict
    report: dict


class _Keep(dict):
    def __missing__(self, _k):
        return 1


KEEP = _Keep()                                   # feature -> images train must keep (default 1); set by _caps


def _caps(avail: Counter, total: int, target: int, rare_max: int, floor_f=1.0, ceil_f=1.25, power=0.5, soft=()):
    """Per-feature ceiling for a split: its proportional share × 1.0–1.25 (more tolerance for abundant features), so
    the target ratio is reachable while no feature is over-represented. Rare features (<= rare_max images): 0 —
    except in `soft` kinds (too fine-grained to protect, e.g. class combinations when nearly every image has its own):
    there they only must keep an example in train."""
    caps, reserved = {}, set()
    keep = KEEP
    for key, a in avail.items():
        kind = key.split(":", 1)[0]
        if a <= rare_max and kind == "class" and "class" in soft:
            caps[key] = min(1, max(0, a - 2))            # last resort: one image out, two kept in train
            keep[key] = 2
            continue
        if a <= rare_max and kind in soft:
            caps[key] = a                                # a combination of its own: no train copy possible
            keep[key] = 0
            continue
        if a <= rare_max:
            caps[key] = 0
            reserved.add(key)
            continue
        prop = target * a / max(total, 1)
        ab = a / max(total, 1)
        factor = floor_f + (ceil_f - floor_f) * min(1.0, ab ** power)
        caps[key] = max(1, min(int(math.ceil(prop * factor)), target))
    return caps, reserved


def _select(units, unit_feats, target, caps, avail, used, reserved, rng):
    """
    Greedy by information gain under hard constraints. Units with the same feature signature (and size) always score
    the same, so the priority queue holds SIGNATURES: the best one gives units until another becomes better.
    Re-scoring scales with the number of distinct signatures, not with the number of images.
    used: feature counts already placed outside train (val + test).
    """
    counts: dict = {}
    chosen: list[int] = []
    size = 0
    buckets: dict[tuple, list[int]] = defaultdict(list)
    sig_feats: dict[tuple, Counter] = {}
    for ui, uf in enumerate(unit_feats):
        sig = (len(units[ui]),) + tuple(sorted(uf.items()))
        buckets[sig].append(ui)
        sig_feats[sig] = uf
    for q in buckets.values():
        rng.shuffle(q)
    # per signature, precomputed: (feature, add, cap, limit = most this split may take, rarity weight) + its bonus
    pre: dict[tuple, tuple] = {}
    for sig, uf in sig_feats.items():
        if any(k in reserved for k in uf):
            continue                                   # never allowed: not queued at all
        rows = []
        for k, add in uf.items():
            cap = caps.get(k, 0)
            lim = avail[k] - used[k] - KEEP.get(k, 1)
            rows.append((k, add, cap, min(cap, lim), 1.0 + 4.0 * (1.0 / math.sqrt(max(avail[k], 1))),
                         1.65 / max(cap, 1)))
        pre[sig] = (rows, min(len({k.split(":", 1)[0] for k in uf}), 8) * 0.8)

    def safe(sig):
        if size + sig[0] > target:
            return False
        for k, add, _cap, lim, _w, _f in pre[sig][0]:
            if counts.get(k, 0) + add > lim:
                return False
        return True

    def score(sig):
        rows, bonus = pre[sig]
        s = bonus
        for k, add, cap, _lim, w, f in rows:
            c = counts.get(k, 0)
            rem = cap - c
            if rem <= 0:
                continue
            s += add * w * (0.35 + f * rem)
            if c == 0:
                s += 4.0
        return s - max(0, sig[0] - (target - size)) * 2.0

    heap = [(-score(sig), rng.random(), sig) for sig in pre]
    heapq.heapify(heap)
    while heap and size < target:
        _neg, tie, sig = heapq.heappop(heap)
        q = buckets[sig]
        if not q or not safe(sig):
            continue                                   # caps only fill up: never safe again
        s = score(sig)
        nxt = -heap[0][0] if heap else -math.inf
        slack = 0.03 * abs(nxt) + 1e-9                 # near-greedy: within 3 % of the best counts as best (fast)
        if s < nxt - slack:
            heapq.heappush(heap, (-s, tie, sig))       # stale: re-queue with its real score
            continue
        while q and size < target and safe(sig) and score(sig) >= nxt - slack:
            ui = q.pop()
            chosen.append(ui)
            size += len(units[ui])
            for k, add in sig_feats[sig].items():
                counts[k] = counts.get(k, 0) + add
        if q and safe(sig):
            heapq.heappush(heap, (-score(sig), tie, sig))
    return chosen, Counter(counts)


def soft_kinds(items, rare_max: int, share: float = 0.2) -> set:
    """Kinds whose rare values cover more than `share` of the images (a combination per image, …): protecting them
    would keep most images out of val. Classes are always protected."""
    if not items:
        return set()
    avail = Counter(k for it in items for k in it.keys())
    hit = Counter()
    for it in items:
        for kind in {k.split(":", 1)[0] for k in it.keys() if avail[k] <= rare_max}:
            hit[kind] += 1
    return {k for k, n in hit.items() if n / len(items) > share and k != "class"}


def plan_split(items: list[SplitItem], val: float = 0.10, test: float = 0.0, seed: int = 42,
               rare_max: int = 3, repair: bool = True) -> SplitPlan:
    soft = soft_kinds(items, rare_max)
    plan = _plan(items, val, test, seed, rare_max, repair, soft)
    kinds = {k.split(":", 1)[0] for it in items for k in it.keys()}
    for relax in (kinds - {"class"}, kinds):        # far short of the target: protect less, step by step
        short = [sp for sp, t in plan.report["targets"].items() if t and plan.report["splits"].get(sp, 0) < 0.5 * t]
        if not short or not (relax - soft):
            continue
        plan2 = _plan(items, val, test, seed, rare_max, repair, relax, fill=relax == kinds)
        if sum(plan2.report["splits"].get(sp, 0) for sp in short) > sum(plan.report["splits"].get(sp, 0)
                                                                         for sp in short):
            plan2.report["relaxed"] = sorted(relax)
            soft, plan = relax, plan2
    plan.report["soft"] = sorted(soft)
    return plan


def _fill(units, unit_feats, taken, target, caps, avail, used, counts, rng, soft) -> list[int]:
    """Last resort when the caps leave a split far short (a small dataset, many classes per image): add groups at
    random beyond the caps — never a reserved feature, and train keeps its examples (two of a rare class)."""
    size = sum(len(units[c]) for c in taken)
    order = [i for i in range(len(units)) if i not in taken]
    rng.shuffle(order)
    order.sort(key=lambda i: len(units[i]))
    out = []
    for ui in order:
        if size >= target:
            break
        if size + len(units[ui]) > target:
            continue
        ok = True
        for k, add in unit_feats[ui].items():
            if caps.get(k, 0) == 0 and k.split(":", 1)[0] not in soft:
                ok = False
                break
            if KEEP.get(k, 1) == 2 and counts[k] + add > caps.get(k, 0):        # a rare class: its one image at most
                ok = False
                break
            if avail[k] - (used[k] + counts[k] + add) < KEEP.get(k, 1):
                ok = False
                break
        if ok:
            out.append(ui)
            size += len(units[ui])
            counts.update(unit_feats[ui])
    return out


def _plan(items, val, test, seed, rare_max, repair, soft, fill=False) -> SplitPlan:
    KEEP.clear()
    rng = random.Random(seed)
    filled = {}
    total = len(items)
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_group[it.group or it.key].append(i)
    units = list(by_group.values())
    rng.shuffle(units)
    feats = [it.keys() for it in items]
    unit_feats = [Counter(k for i in u for k in feats[i]) for u in units]
    avail = Counter(k for f in feats for k in f)
    where = ["train"] * len(units)                 # unit -> split
    used = Counter()
    report_caps, reserved_all, targets = {}, set(), {}
    split_counts: dict[str, Counter] = {}
    for split, ratio in (("val", val), ("test", test)):
        if ratio <= 0:
            continue
        target = int(round(total * ratio))
        targets[split] = target
        caps, reserved = _caps(avail, total, target, rare_max, soft=soft)
        reserved_all |= reserved
        report_caps[split] = caps
        pool = [i for i in range(len(units)) if where[i] == "train"]
        sub_units = [units[i] for i in pool]
        sub_feats = [unit_feats[i] for i in pool]
        chosen, counts = _select(sub_units, sub_feats, target, caps, avail, used, reserved, rng)
        if fill and sum(len(sub_units[c]) for c in chosen) < 0.7 * target:
            extra = _fill(sub_units, sub_feats, set(chosen), target, caps, avail, used, counts, rng, soft)
            filled[split] = len(extra)
            chosen += extra
        for ci in chosen:
            where[pool[ci]] = split
        used.update(counts)
        split_counts[split] = counts
    repaired = 0
    if repair:
        for split in split_counts:
            repaired += _repair(split, units, unit_feats, where, split_counts, report_caps[split], avail,
                                reserved_all, rare_max)
    assign = {}
    for ui, u in enumerate(units):
        for i in u:
            assign[items[i].key] = where[ui]
    rep = make_report(items, assign, avail, report_caps, rare_max, len(units), soft)
    rep["targets"] = targets
    rep["repaired"] = repaired
    rep["filled"] = filled
    return SplitPlan(assign, rep)


def _repair(split, units, unit_feats, where, split_counts, caps, avail, reserved, rare_max, budget=400) -> int:
    """Swap train groups in for same-size groups of `split` so features missing there get covered, without breaking
    a cap, train coverage (>= 1 left in train) or a feature the split already had (aimed version of the script's swap
    passes; bounded)."""
    counts = split_counts[split]
    other = Counter()
    for sp, c in split_counts.items():
        if sp != split:
            other.update(c)
    missing = sorted((k for k, a in avail.items() if a > rare_max and caps.get(k, 0) > 0 and counts[k] == 0
                      and k not in reserved), key=lambda k: avail[k])
    if not missing:
        return 0
    inside = defaultdict(list)                     # size -> units of this split
    for ui, sp in enumerate(where):
        if sp == split:
            inside[len(units[ui])].append(ui)
    by_feat = defaultdict(list)                    # feature -> train units with it
    for ui, sp in enumerate(where):
        if sp == "train":
            for k in unit_feats[ui]:
                if k in missing:
                    by_feat[k].append(ui)
    swaps = 0
    tries = 0
    for k in missing:
        if counts[k] > 0:                          # covered by an earlier swap
            continue
        for tu in sorted(by_feat.get(k, ()), key=lambda u: len(units[u]))[:40]:
            if where[tu] != "train":
                continue
            tf = unit_feats[tu]
            if any(f in reserved for f in tf):
                continue
            done = False
            for vu in inside.get(len(units[tu]), ())[:200]:
                tries += 1
                if tries > budget * 50:
                    return swaps
                vf = unit_feats[vu]
                new = counts - vf
                new.update(tf)
                ok = True
                for f in set(tf) | set(vf):
                    n = new[f]
                    if f in tf and n > caps.get(f, 0):
                        ok = False
                        break
                    if avail[f] - (other[f] + n) < KEEP.get(f, 1):           # train keeps its examples
                        ok = False
                        break
                    if counts[f] > 0 and n == 0 and avail[f] > rare_max and caps.get(f, 0) > 0:
                        ok = False                                   # would lose a feature the split had
                        break
                if not ok:
                    continue
                where[tu], where[vu] = split, "train"
                inside[len(units[tu])].remove(vu)
                inside[len(units[tu])].append(tu)
                counts.subtract(vf)
                counts.update(tf)
                for f in list(counts):
                    if counts[f] <= 0:
                        del counts[f]
                for f in vf:                       # the swapped-out unit can now help another missing feature
                    if f in missing:
                        by_feat[f].append(vu)
                swaps += 1
                done = True
                break
            if done or swaps >= budget:
                break
    return swaps


def make_report(items, assign, avail=None, caps=None, rare_max=3, n_groups=None, soft=()) -> dict:
    """Per split / class / feature counts + status (also for existing splits: caps={})."""
    splits = Counter(assign.values())
    avail = avail or Counter(k for it in items for k in it.keys())
    per = defaultdict(Counter)                      # feature -> split -> images
    cls = defaultdict(Counter)                      # class -> split -> images
    inst = defaultdict(Counter)                     # class -> split -> instances
    for it in items:
        sp = assign.get(it.key, "train")
        for k in it.keys():
            per[k][sp] += 1
        for c in it.features.get("class", []):
            cls[c][sp] += 1
        for c, n in it.labels.items():
            inst[c][sp] += n
    rows = []
    for k, a in sorted(avail.items()):
        kind, value = k.split(":", 1)
        c = per[k]
        if a <= rare_max and kind in soft:
            status = ("ok" if c["train"] >= min(a, 2) else "MISSING IN TRAIN") if kind == "class" else "ok"
        elif a <= rare_max:
            status = "train only (rare)" if c["val"] + c["test"] == 0 else "RARE IN VAL/TEST"
        elif c["train"] == 0:
            status = "MISSING IN TRAIN"
        elif "val" in splits and c["val"] == 0:
            status = "missing in val"
        else:
            status = "ok"
        rows.append({"kind": kind, "value": value, "images": a, "train": c["train"], "val": c["val"],
                     "test": c["test"], "status": status})
    groups = Counter(it.group or it.key for it in items)
    leaks = 0
    gs = defaultdict(set)
    for it in items:
        gs[it.group or it.key].add(assign.get(it.key, "train"))
    leaks = sum(1 for s in gs.values() if len(s) > 1)
    return {"total": len(items), "splits": dict(splits), "features": rows,
            "classes": {c: dict(v) for c, v in sorted(cls.items())},
            "instances": {c: dict(v) for c, v in sorted(inst.items())},
            "groups": {"count": len(groups), "multi": sum(1 for n in groups.values() if n > 1),
                       "largest": max(groups.values(), default=0), "split_across": leaks},
            "caps": {s: dict(c) for s, c in (caps or {}).items()}}


# =============================================================================
# Items from a dataset (Convert & Split, every format)
# =============================================================================
def build_items(samples, adapter, cache=None, *, per_sample=None, classification=False, dups=True, near=True,
                video=True, lighting=True, measure=True, idents=(), focus=None, near_threshold=6, progress=None,
                cancel=None) -> list[SplitItem]:
    """
    One SplitItem per sample: features (class, combination, object count, size, position, lighting) and a group
    (exact / near duplicates, frames of one video, one OCR identity — always in the same split).
    per_sample: {key: (labels Counter, boxes_norm)} already known (Convert's survey) — saves reading the annotations.
    measure: images the cache knows nothing about are looked at once (reduced decode: lighting + perceptual hash) and
    the result kept in the cache. idents: [(key, text)] identities (e.g. OCR reads).
    """
    from pathlib import Path
    items, links = [], []
    shas: dict = {}
    hashes: list = []
    to_measure: list = []
    n = len(samples)
    for i, x in enumerate(samples):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("cancelled")
        ann = None
        if per_sample is not None and x.key in per_sample:
            labels, boxes = per_sample[x.key]
            img = x.image_path
        else:
            try:
                ann = adapter.load(x, light=True)
            except Exception:
                ann = None
            img = (ann.image_path if ann else None) or x.image_path
            labels, boxes = Counter(), []
            if ann is not None:
                labels = Counter(ann.tags) if classification else ann.label_counts()
                if not classification and ann.shapes:
                    W, H = (1.0, 1.0) if ann.normalized else (ann.width or 0, ann.height or 0)
                    if not (W and H):
                        sz = adapter.image_size(x, ann)
                        W, H = sz if sz else (0, 0)
                    if W and H:
                        for sh in ann.shapes:
                            pts = sh.get("points") or []
                            if len(pts) >= 2:
                                try:
                                    xs, ys = [p[0] / W for p in pts], [p[1] / H for p in pts]
                                except (TypeError, IndexError):
                                    continue
                                boxes.append((min(xs), min(ys), max(xs), max(ys), str(sh.get("label", ""))))
        meta = None
        if cache is not None and img is not None:
            try:
                meta = cache.get_image(img)
            except Exception as _exc:
                _ignored(_exc)
                meta = None
        light = (meta or {}).get("light") if lighting else None
        if classification:
            feats = classification_features(list(labels), getattr(x, "folder", ""), light)
        else:
            feats = detection_features(labels, boxes, light, focus)
        it = SplitItem(x.key, feats, labels=Counter(labels))
        items.append(it)
        if img is not None and measure and ((lighting and not light) or (near and not (meta or {}).get("phash"))):
            to_measure.append((len(items) - 1, img))
        if dups and meta and meta.get("sha"):
            shas.setdefault(meta["sha"], []).append(x.key)
        if near and meta and meta.get("phash"):
            hashes.append((x.key, meta["phash"]))
        if video:
            p = Path(img or x.key)
            sid = sequence_id(p.stem, str(p.parent))
            if sid:
                links.append(("seq:" + sid, x.key))
        if progress is not None and i % 500 == 0:
            progress(i, n + len(to_measure))
    if to_measure:
        _measure(items, to_measure, cache, lighting, near, hashes, progress, cancel, n)
    for ks in shas.values():
        links += [(ks[0], k) for k in ks[1:]]
    if near and len(hashes) > 1:
        try:
            import numpy as np
            from AIO_QC.core.profile import near_duplicate_pairs
            arr = [np.frombuffer(h, dtype=np.uint8) if isinstance(h, (bytes, bytearray, memoryview)) else h
                   for _k, h in hashes]
            for a, b, _d in near_duplicate_pairs(arr, near_threshold):
                links.append((hashes[a][0], hashes[b][0]))
        except Exception as _exc:
            _ignored(_exc)
    for key, text in idents or ():
        links.append(("id:" + text, key))
    keys = [it.key for it in items]
    groups = build_groups(keys + sorted({a for a, _b in links if a.startswith(("seq:", "id:"))}), links)
    for it in items:
        it.group = groups[it.key]
    return items


def _measure(items, todo, cache, lighting, near, hashes, progress, cancel, n0):
    """Reduced decode of the images the cache did not know: lighting + perceptual hash (kept in the cache)."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    from AIO_QC.core import profile as P

    def one(job):
        idx, img = job
        if cancel is not None and cancel.is_set():
            return idx, img, None, None
        try:
            g = P.read_gray_reduced(img)
        except Exception as _exc:
            _ignored(_exc)
            g = None
        if g is None:
            return idx, img, None, None
        return idx, img, P.lighting(g)[0], P.phash(g)
    with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 2))) as ex:
        for j, (idx, img, light, h) in enumerate(ex.map(one, todo, chunksize=8)):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            if progress is not None and j % 100 == 0:
                progress(n0 + j, n0 + len(todo))
            if light is None:
                continue
            it = items[idx]
            if lighting and "lighting" not in it.features:
                it.features["lighting"] = [light]
            if near and h is not None:
                hashes.append((it.key, h.tobytes()))
            if cache is not None:
                try:
                    cache.put_image(img, light=light, phash=h.tobytes())
                except Exception as _exc:
                    _ignored(_exc)
    if cache is not None:
        try:
            cache.flush()
        except Exception as _exc:
            _ignored(_exc)


# =============================================================================
# Reports
# =============================================================================
def write_reports(folder, items, assign: dict, report: dict, settings: dict, extra: list[str] | None = None):
    """split_report.txt · feature_summary.csv · feature_map.csv (per image) · split_manifest.json."""
    import csv
    import json
    from pathlib import Path
    from AIO_QC.core.io import atomic_write_text
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "feature_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature_type", "feature", "images", "train", "val", "test", "status"])
        for r in report["features"]:
            w.writerow([r["kind"], r["value"], r["images"], r["train"], r["val"], r["test"], r["status"]])
    write_feature_map(d / "feature_map.csv", items, assign)
    manifest = {"settings": settings, "counts": report["splits"], "targets": report.get("targets", {}),
                "groups": report["groups"], "repaired_swaps": report.get("repaired", 0),
                "splits": {sp: sorted(k for k, v in assign.items() if v == sp) for sp in ("train", "val", "test")}}
    atomic_write_text(d / "split_manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False))
    atomic_write_text(d / "split_report.txt", report_text(report, extra))


def write_feature_map(path, items, assign: dict):
    import csv
    kinds = sorted({k for it in items for k in it.features})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "split", "group"] + kinds)
        for it in sorted(items, key=lambda z: z.key):
            w.writerow([it.key, assign.get(it.key, "train"), it.group or ""]
                       + ["|".join(map(str, it.features.get(k, []))) for k in kinds])


def report_text(report: dict, extra: list[str] | None = None) -> str:
    total = max(report["total"], 1)
    L = ["=" * 64, "DATASET SPLIT REPORT", "=" * 64, f"Total images : {report['total']:,}"]
    for sp in ("train", "val", "test"):
        n = report["splits"].get(sp, 0)
        if n or sp in report.get("targets", {}):
            t = report.get("targets", {}).get(sp)
            L.append(f"{sp.capitalize():6s} images : {n:,} ({100 * n / total:.1f} %)"
                     + (f"   target {t:,}" + (" — fewer: rare features and per-feature caps held some back"
                                              if n < 0.9 * t else "") if t else ""))
    g = report["groups"]
    L += ["", f"Groups kept together : {g['multi']:,} (largest {g['largest']} images) · split across sets: "
              f"{g['split_across']}"]
    if report.get("soft"):
        L.append(f"Not protected (too fine-grained: most images have a rare value): {', '.join(report['soft'])}")
    if report.get("relaxed"):
        L.append("Relaxed to reach the target (most images hold a rare feature): "
                 + ("rare classes may give ONE image to val / test, keeping at least two in train"
                    if "class" in report["relaxed"] else "only classes stay protected"))
    if any((report.get("filled") or {}).values()):
        L.append("Filled beyond the per-feature caps to reach the target: "
                 + ", ".join(f"{k} +{v}" for k, v in report["filled"].items() if v)
                 + " (a small dataset with many classes per image: val is less evenly spread)")
    if report.get("repaired"):
        L.append(f"Repair swaps         : {report['repaired']} (features missing in val / test brought in)")
    L += ["", "PER CLASS (images)", "-" * 64]
    for c, v in report["classes"].items():
        L.append(f"{c:28s} train={v.get('train', 0):6d} val={v.get('val', 0):6d} test={v.get('test', 0):6d}")
    flags = [r for r in report["features"] if r["status"] != "ok"]
    L += ["", "FEATURE NOTES", "-" * 64] + ([f"{r['kind']}:{r['value']} -> {r['status']} ({r['images']} images)"
                                             for r in flags] or ["none"])
    L += ["", "Rules: duplicates / near-duplicates / frames of one video stay in one split; features seen in <= rare "
              "images stay in train; every feature keeps an example in train; per-feature caps keep val / test "
              "representative. feature_map.csv lists every image with its features and split."]
    if extra:
        L += [""] + extra
    return "\n".join(L) + "\n"
