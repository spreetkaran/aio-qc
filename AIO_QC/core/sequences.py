# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Sequence consistency (Qt-free): frames that come from one video / burst are checked against their neighbours.
Works for any dataset whose frames can be ordered — nothing here knows about a particular project.

1. Sequences
   * files are grouped by folder (and split) + the name with its LAST number removed: "cam1_frame_000123.jpg" ->
     group "cam1_frame_#", frame 123; "20240101_101530_0042.json" -> "20240101_101530_#", 42
   * inside a group, frames are ordered by that number; the usual step is learned (e.g. every 5th frame) and the
     group is cut where the step jumps far beyond it (a pause / another recording) -> sequences
   * optional: also cut where the picture itself changes completely (scene cut), from tiny thumbnails
   * alternative mode "folder = one sequence" for names without numbers (natural name order)

2. Tracks — boxes of consecutive frames are linked (class-agnostic, so label flips are visible): best IoU first,
   with a centre-distance fallback for small / fast objects; a track may skip up to `max_gap` frames.

3. Findings (each with the reason in plain words):
   missing      present before and after, absent here (gap of ≤ max_gap frames) — with the interpolated box
   label_flip   a short run of another label inside a track whose neighbours agree
   size_jump    box much larger / smaller than in BOTH neighbouring frames (one-frame spike)
   shift_jump   box far from where the neighbours put it (one-frame spike)
   attr_flip    a stored attribute (description / text / flags) differs from both neighbours, or is missing
   one_frame    an object that exists in a single frame of a long, steady sequence (junk / duplicate?)

Boxes are compared in image-relative coordinates, so mixed resolutions and YOLO / LabelMe behave the same.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

_NUM = re.compile(r"(\d+)(?!.*\d)")          # the last run of digits in a name


@dataclass
class FrameInfo:
    key: str
    group: str                    # sequence group (folder + name pattern)
    n: float                      # frame number (or position)
    boxes: list = field(default_factory=list)   # [(shape_idx, label, (x1, y1, x2, y2) relative, attrs)]
    thumb: object = None          # optional tiny grey thumbnail (numpy) for scene-cut detection


@dataclass
class Finding:
    key: str
    code: str
    severity: str                 # warn | info
    label: str
    message: str
    shape_index: int | None = None
    box: tuple | None = None      # relative (x1, y1, x2, y2) — e.g. the interpolated box of a missing object
    seq: str = ""
    frame: int = 0                # position inside the sequence
    track: int = -1


CODES = {
    "missing": "Missing in this frame",
    "label_flip": "Label differs from the track",
    "size_jump": "Size spike",
    "shift_jump": "Position spike",
    "attr_flip": "Attribute differs",
    "one_frame": "Only in this frame",
}


def split_name(stem: str) -> tuple[str, float | None]:
    """("cam1_frame_#", 123) — the last number becomes the frame number."""
    m = _NUM.search(stem)
    if not m:
        return stem, None
    return stem[:m.start()] + "#" + stem[m.end():], float(int(m.group(1)))


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def build_sequences(frames: list[FrameInfo], mode: str = "auto", gap_factor: float = 3.0, min_len: int = 3,
                    scene_cut: float | None = None) -> list[list[FrameInfo]]:
    """Group + order + cut. mode: 'auto' (name numbers) | 'folder' (one sequence per group, name order)."""
    groups: dict[str, list[FrameInfo]] = defaultdict(list)
    for f in frames:
        groups[f.group].append(f)
    out = []
    for g, fs in groups.items():
        if mode == "folder" or any(f.n is None for f in fs):
            fs.sort(key=lambda f: natural_key(f.key))
            for i, f in enumerate(fs):
                f.n = float(i)
            runs = [fs]
        else:
            fs.sort(key=lambda f: f.n)
            diffs = [b.n - a.n for a, b in zip(fs, fs[1:]) if b.n > a.n]
            step = sorted(diffs)[len(diffs) // 2] if diffs else 1.0
            runs, cur = [], [fs[0]] if fs else []
            for a, b in zip(fs, fs[1:]):
                if b.n - a.n > step * gap_factor or b.n == a.n:
                    runs.append(cur)
                    cur = []
                cur.append(b)
            if cur:
                runs.append(cur)
        for run in runs:
            if scene_cut is not None:
                out.extend(_cut_scenes(run, scene_cut, min_len))
            elif len(run) >= min_len:
                out.append(run)
    return out


def _cut_scenes(run, threshold, min_len):
    import numpy as np
    parts, cur = [], [run[0]]
    for a, b in zip(run, run[1:]):
        if a.thumb is not None and b.thumb is not None:
            d = float(np.mean(np.abs(a.thumb.astype(np.float32) - b.thumb.astype(np.float32)))) / 255.0
            if d > threshold:
                parts.append(cur)
                cur = []
        cur.append(b)
    parts.append(cur)
    return [p for p in parts if len(p) >= min_len]


# ---------------------------------------------------------------------- tracking
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _area(b) -> float:
    return max(b[2] - b[0], 1e-9) * max(b[3] - b[1], 1e-9)


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _link_score(a, b, iou_min: float) -> float:
    """> 0 when b plausibly continues a: IoU, or (small / fast objects) a close centre and a similar size."""
    v = iou(a, b)
    if v >= iou_min:
        return 1.0 + v
    (ax, ay), (bx, by) = _center(a), _center(b)
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    ratio = _area(a) / _area(b)
    if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= 0.6 * size and 0.4 <= ratio <= 2.5:
        return 0.5 + v
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))       # one box (mostly) inside the other: a loose /
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))       # tight box of the same object
    if ix * iy >= 0.6 * min(_area(a), _area(b)):
        return 0.3 + v
    return 0.0


def track(seq: list[FrameInfo], iou_min: float = 0.3, max_gap: int = 1) -> list[dict]:
    """Tracks: [{t: (shape_idx, label, box, attrs)}] keyed by the frame position in `seq`."""
    tracks: list[dict] = []
    last: list[tuple[int, tuple]] = []             # per track: (last t, last box)
    for t, f in enumerate(seq):
        cand = []
        for ti, (lt, lb) in enumerate(last):
            if t - lt > max_gap + 1:
                continue
            for bi, (_si, _lab, box, _at) in enumerate(f.boxes):
                sc = _link_score(lb, box, iou_min)
                if sc > 0:
                    cand.append((sc - 0.05 * (t - lt - 1), ti, bi))
        cand.sort(reverse=True)
        used_t, used_b = set(), set()
        for _sc, ti, bi in cand:
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            tracks[ti][t] = f.boxes[bi]
            last[ti] = (t, f.boxes[bi][2])
        for bi, b in enumerate(f.boxes):
            if bi not in used_b:
                tracks.append({t: b})
                last.append((t, b[2]))
    return tracks


# ---------------------------------------------------------------------- checks
def check_sequence(seq: list[FrameInfo], name: str, iou_min=0.3, max_gap=1, size_factor=1.8, shift_factor=0.6,
                   check_attrs=True, min_track=3) -> list[Finding]:
    out: list[Finding] = []
    tracks = track(seq, iou_min, max_gap)
    n = len(seq)
    for tid, tr in enumerate(tracks):
        ts = sorted(tr)
        labels = Counter(tr[t][1] for t in ts)
        major, major_n = labels.most_common(1)[0]
        # ---- missing frames (gap inside the track)
        for a, b in zip(ts, ts[1:]):
            if 1 < b - a <= max_gap + 1 and len(ts) >= 2:
                ba, bb = tr[a][2], tr[b][2]
                if iou(ba, bb) < 0.1 and _link_score(ba, bb, iou_min) <= 0:
                    continue
                for t in range(a + 1, b):
                    w = (t - a) / (b - a)
                    box = tuple(ba[i] + (bb[i] - ba[i]) * w for i in range(4))
                    out.append(Finding(seq[t].key, "missing", "warn", tr[a][1],
                                       f"'{tr[a][1]}' is in the frame before and after this one, but not here",
                                       None, box, name, t, tid))
        # ---- one-frame objects in steady sequences
        if len(ts) == 1 and n >= 5:
            t = ts[0]
            if 0 < t < n - 1:
                si, lab, box, _at = tr[t]
                out.append(Finding(seq[t].key, "one_frame", "info", lab,
                                   f"'{lab}' exists only in this frame of a {n}-frame sequence — junk, duplicate, "
                                   "or missing in the other frames?", si, box, name, t, tid))
        if len(ts) < min_track:
            continue
        # ---- label flips: a short minority run inside a track whose neighbours agree
        if len(labels) > 1 and major_n >= 2:
            for i, t in enumerate(ts):
                si, lab, box, _at = tr[t]
                if lab == major:
                    continue
                prev_lab = tr[ts[i - 1]][1] if i > 0 else None
                next_lab = tr[ts[i + 1]][1] if i + 1 < len(ts) else None
                run_ok = (prev_lab == major or next_lab == major)
                if run_ok and labels[lab] <= max(2, len(ts) // 5):
                    out.append(Finding(seq[t].key, "label_flip", "warn", lab,
                                       f"labelled '{lab}' here, '{major}' in {major_n} of {len(ts)} frames of the "
                                       "same object", si, box, name, t, tid))
        # ---- one-frame size / position spikes (both neighbours agree with each other)
        for i in range(1, len(ts) - 1):
            p, t, q = ts[i - 1], ts[i], ts[i + 1]
            bp, bt, bq = tr[p][2], tr[t][2], tr[q][2]
            si, lab = tr[t][0], tr[t][1]
            ap, at_, aq = _area(bp), _area(bt), _area(bq)
            if max(ap, aq) / min(ap, aq) < 1.5:                          # neighbours are consistent
                ref = (ap * aq) ** 0.5
                r = at_ / ref
                if r >= size_factor or r <= 1 / size_factor:
                    out.append(Finding(seq[t].key, "size_jump", "warn", lab,
                                       f"box is {r:.1f}× the size it has in the frames before and after"
                                       if r > 1 else f"box is {1 / r:.1f}× smaller than in the frames before and after",
                                       si, bt, name, t, tid))
                    continue
                w = (t - p) / (q - p)
                cx = _center(bp)[0] + (_center(bq)[0] - _center(bp)[0]) * w
                cy = _center(bp)[1] + (_center(bq)[1] - _center(bp)[1]) * w
                tx, ty = _center(bt)
                size = (ref ** 0.5)
                d = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
                if d > shift_factor * size and iou(bp, bq) > 0.3:
                    out.append(Finding(seq[t].key, "shift_jump", "warn", lab,
                                       f"box is {d / size:.1f} box-sizes away from where the frames before and after "
                                       "put it", si, bt, name, t, tid))
            # ---- attribute flips
            if check_attrs:
                ap_, at2, aq_ = tr[p][3] or {}, tr[t][3] or {}, tr[q][3] or {}
                for k in set(ap_) | set(at2):
                    v, vp, vq = at2.get(k), ap_.get(k), aq_.get(k)
                    if vp is not None and vp == vq and v != vp:
                        here = "missing here" if v is None else f"{v!r} here"
                        out.append(Finding(seq[t].key, "attr_flip", "info" if k == "flags" else "warn", lab,
                                           f"{k}: {here}, {vp!r} in the frames before and after", si, bt,
                                           name, t, tid))
    return out


def attrs_of(shape: dict) -> dict:
    """Attributes worth comparing across frames: text-like fields, flags, group id."""
    out = {}
    for k in ("description", "text", "ocr", "value", "attributes"):
        v = shape.get(k)
        if v not in (None, "", {}, []):
            out[k] = v if not isinstance(v, (dict, list)) else repr(v)
    flags = shape.get("flags")
    if isinstance(flags, dict):
        on = sorted(str(k) for k, v in flags.items() if v is True)
        if on:
            out["flags"] = ",".join(on)
    return out                     # group ids are per-image numbering: not comparable across frames
