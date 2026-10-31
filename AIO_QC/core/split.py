# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Split planner (train / val / optional test) — adapted from the pipeline's 02_split_dataset.py, made scalable.

    items = [SplitItem(key, features, group), …]        features: {kind: [values]} per image
    plan = plan_split(items, val=0.10, test=0.0, seed=42)
    plan.assign[key] -> "train" | "val" | "test";  plan.report -> per feature / class / group diagnostics

Rules (as in the script): groups (duplicates, frames of one video, one OCR identity …) are indivisible; features seen
in <= rare_max images are reserved for train; every feature keeps >= 1 example in train; per-feature caps keep val /
test representative without draining train; selection by information gain under these hard constraints.
Scalable: LAZY greedy (a priority queue; only the top candidate is re-scored — scores only fall as a split fills),
O(n log n) instead of re-scoring every group for every pick.
"""
from __future__ import annotations

import heapq
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field


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


def detection_features(labels: Counter, boxes_norm, lighting: str | None = None) -> dict:
    """labels: label -> count; boxes_norm: [(x1, y1, x2, y2) 0..1] of the objects (may be empty)."""
    f: dict[str, list] = {"class": sorted(labels) or ["(none)"]}
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


def _caps(avail: Counter, total: int, target: int, rare_max: int, floor_f=1.0, ceil_f=1.25, power=0.5):
    """Per-feature ceiling for a split: its proportional share × 1.0–1.25 (more tolerance for abundant features), so
    the target ratio is reachable while no feature is over-represented. Rare features (<= rare_max images): 0."""
    caps, reserved = {}, set()
    for key, a in avail.items():
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
    counts = Counter()
    chosen: list[int] = []
    size = 0
    rarity = {k: 1.0 / math.sqrt(max(a, 1)) for k, a in avail.items()}
    buckets: dict[tuple, list[int]] = defaultdict(list)
    sig_feats: dict[tuple, Counter] = {}
    for ui, uf in enumerate(unit_feats):
        sig = (len(units[ui]),) + tuple(sorted(uf.items()))
        buckets[sig].append(ui)
        sig_feats[sig] = uf
    for q in buckets.values():
        rng.shuffle(q)

    def safe(sig):
        if size + sig[0] > target:
            return False
        for k, add in sig_feats[sig].items():
            if k in reserved or counts[k] + add > caps.get(k, 0) or avail[k] - (used[k] + counts[k] + add) < 1:
                return False
        return True

    def score(sig):
        s = 0.0
        for k, add in sig_feats[sig].items():
            cap = caps.get(k, 0)
            rem = cap - counts[k]
            if rem <= 0:
                continue
            s += add * (1.0 + 4.0 * rarity[k]) * (0.35 + 1.65 * rem / max(cap, 1))
            if counts[k] == 0:
                s += 4.0
        s += min(len({k.split(":", 1)[0] for k in sig_feats[sig]}), 8) * 0.8
        return s - max(0, sig[0] - (target - size)) * 2.0

    heap = [(-score(sig), rng.random(), sig) for sig in buckets]
    heapq.heapify(heap)
    while heap and size < target:
        _neg, tie, sig = heapq.heappop(heap)
        q = buckets[sig]
        if not q or not safe(sig):
            continue                                   # caps only fill up: never safe again
        s = score(sig)
        nxt = -heap[0][0] if heap else -math.inf
        if s < nxt - 1e-9:
            heapq.heappush(heap, (-s, tie, sig))       # stale: re-queue with its real score
            continue
        while q and size < target and safe(sig) and score(sig) >= nxt - 1e-9:
            ui = q.pop()
            chosen.append(ui)
            size += len(units[ui])
            counts.update(sig_feats[sig])
        if q and safe(sig):
            heapq.heappush(heap, (-score(sig), tie, sig))
    return chosen, counts


def plan_split(items: list[SplitItem], val: float = 0.10, test: float = 0.0, seed: int = 42,
               rare_max: int = 3) -> SplitPlan:
    rng = random.Random(seed)
    total = len(items)
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_group[it.group or it.key].append(i)
    units = list(by_group.values())
    rng.shuffle(units)
    feats = [it.keys() for it in items]
    unit_feats = [Counter(k for i in u for k in feats[i]) for u in units]
    avail = Counter(k for f in feats for k in f)
    assign = {it.key: "train" for it in items}
    used = Counter()
    report_caps = {}
    taken = set()
    for split, ratio in (("val", val), ("test", test)):
        if ratio <= 0:
            continue
        target = int(round(total * ratio))
        caps, reserved = _caps(avail, total, target, rare_max)
        report_caps[split] = caps
        pool = [i for i in range(len(units)) if i not in taken]
        sub_units = [units[i] for i in pool]
        sub_feats = [unit_feats[i] for i in pool]
        chosen, counts = _select(sub_units, sub_feats, target, caps, avail, used, reserved, rng)
        for ci in chosen:
            taken.add(pool[ci])
            for i in sub_units[ci]:
                assign[items[i].key] = split
        used.update(counts)
    return SplitPlan(assign, make_report(items, assign, avail, report_caps, rare_max, len(units)))


def make_report(items, assign, avail=None, caps=None, rare_max=3, n_groups=None) -> dict:
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
        if a <= rare_max:
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
