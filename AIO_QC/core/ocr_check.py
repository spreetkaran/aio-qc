# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
OCR Check analysis (Qt-free). OCR runs once (cached); everything here is re-evaluated instantly from the reads
when thresholds, validators or class settings change.

Per text box -> findings:
    no text · low confidence · rule failed (validators) · truncated · loose box · several lines · text mismatch
    (vs a stored transcription) · too small · read error
Across the dataset -> identity leakage (same text in several splits) and frequent identities.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .validators import identity_key, normalize

KINDS = {  # kind -> (title, severity role)
    "no_text": ("no text", "error"),
    "low_conf": ("low confidence", "warn"),
    "rule": ("rule failed", "error"),
    "truncated": ("truncated", "error"),
    "loose": ("loose box", "warn"),
    "lines": ("several lines", "warn"),
    "mismatch": ("text mismatch", "error"),
    "small": ("too small", "muted"),
    "error": ("read error", "muted"),
}
DEFAULT_THRESHOLDS = {"min_conf": 0.6, "trunc_tol": 0.08, "loose": 0.3, "min_h": 8, "pad": 0.15, "min_id_len": 4}
DEFAULT_CLASS = {"validators": [], "lines": "single", "normalize": {"upper": True, "remove": " -._", "strip": True},
                 "text_field": ""}


@dataclass
class Read:
    key: str
    idx: int                     # shape index in the file
    label: str
    split: str
    box: tuple                   # (x1, y1, x2, y2) image pixels
    res: dict                    # server result: text, conf, lines | skip | error
    gt: str | None = None        # stored transcription, if the class has a text field
    img_wh: tuple = (0, 0)


@dataclass
class Finding:
    read: Read
    kinds: list                  # [kind, …] (first = most important)
    text: str                    # normalised read
    detail: str
    passed: str = ""             # validator that accepted the text
    cer: float | None = None


def cer(a: str, b: str) -> float:
    """Character error rate of a vs reference b (Levenshtein / len(b))."""
    if not b:
        return 0.0 if not a else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(b)


def _geometry(r: Read, thr: dict, lines_mode: str) -> list[tuple[str, str]]:
    """Box-vs-text checks (locate mode only: needs the text line boxes)."""
    lines = r.res.get("lines") or []
    if not lines:
        return []
    x1, y1, x2, y2 = r.box
    bh, bw = max(y2 - y1, 1e-6), max(x2 - x1, 1e-6)
    out = []
    lx1 = min(l[0] for l in lines)
    ly1 = min(l[1] for l in lines)
    lx2 = max(l[2] for l in lines)
    ly2 = max(l[3] for l in lines)
    over = max(x1 - lx1, lx2 - x2, y1 - ly1, ly2 - y2, 0.0) / bh
    W, H = r.img_wh
    pad = thr["pad"] * bh
    touches = (lx1 <= max(0, x1 - pad) + 1 and x1 - pad > 1) or (W and lx2 >= min(W, x2 + pad) - 1 and x2 + pad < W - 1)
    if over > thr["trunc_tol"] or touches:
        side = ("left" if x1 - lx1 >= max(lx2 - x2, y1 - ly1, ly2 - y2) else "right" if lx2 - x2 >= max(y1 - ly1, ly2 - y2)
                else "top" if y1 - ly1 >= ly2 - y2 else "bottom")
        out.append(("truncated", f"text runs {over * 100:.0f}% of the box height past the {side} edge"
                                 + (" and continues beyond the view" if touches else "")))
    area = sum(max(0.0, l[2] - l[0]) * max(0.0, l[3] - l[1]) for l in lines)
    fill = min(1.0, area / (bw * bh))
    if fill < thr["loose"]:
        out.append(("loose", f"text fills {fill * 100:.0f}% of the box"))
    if lines_mode == "single" and len(lines) >= 2:
        ys = sorted((l[1], l[3]) for l in lines)
        stacked = any(ys[i + 1][0] > (ys[i][0] + ys[i][1]) / 2 for i in range(len(ys) - 1))
        if stacked:
            out.append(("lines", f"{len(lines)} text lines in a single-line box"))
    return out


def evaluate(r: Read, cls_cfg: dict, thr: dict, validators: dict) -> Finding | None:
    """None = the box is fine. Validators: {name: Validator}."""
    res = r.res or {}
    if "skip" in res:
        return Finding(r, ["small"], "", f"{res['skip']} ({res.get('h', '?')} px high)")
    if "error" in res:
        return Finding(r, ["error"], "", res["error"])
    raw = res.get("text") or ""
    text = normalize(raw, cls_cfg.get("normalize"))
    conf = float(res.get("conf") or 0.0)
    kinds, details, passed, c = [], [], "", None
    if not text:
        kinds.append("no_text")
        details.append("nothing readable")
    else:
        if conf < thr["min_conf"]:
            kinds.append("low_conf")
            details.append(f"confidence {conf:.2f}")
        names = cls_cfg.get("validators") or []
        if names:
            reasons = []
            for n in names:
                v = validators.get(n)
                if v is None:
                    reasons.append(f"{n}: missing")
                    continue
                ok, why = v(text)
                if ok:
                    passed = v.title
                    break
                reasons.append(f"{v.title}: {why}" if why else v.title)
            if not passed:
                kinds.append("rule")
                details.append("; ".join(reasons))
        if r.gt is not None and r.gt.strip():
            gt = normalize(r.gt, cls_cfg.get("normalize"))
            c = cer(text, gt)
            if c > 0:
                kinds.append("mismatch")
                details.append(f"stored “{gt}” (CER {c:.2f})")
    geometry = _geometry(r, thr, cls_cfg.get("lines", "single")) if thr.get("geometry", True) else []
    for k, d in geometry:        # locate mode only: in read mode the "line" is just the crop
        kinds.append(k)
        details.append(d)
    if not kinds:
        return None
    order = list(KINDS)
    kinds.sort(key=order.index)
    return Finding(r, kinds, text, " · ".join(details), passed, c)


@dataclass
class Summary:
    per_class: dict = field(default_factory=dict)      # label -> Counter(reads, readable, passed, low, …)
    lengths: dict = field(default_factory=dict)        # label -> Counter(len)
    chars: Counter = field(default_factory=Counter)
    leaks: list = field(default_factory=list)          # [(identity, {split: [keys]})]
    frequent: list = field(default_factory=list)       # [(identity, n_files)]


def summarize(reads: list[Read], cfg: dict, thr: dict, validators: dict, frequent_min: int = 20) -> Summary:
    s = Summary()
    ids: dict[str, dict] = defaultdict(lambda: defaultdict(set))
    for r in reads:
        cc = {**DEFAULT_CLASS, **cfg.get(r.label, {})}
        c = s.per_class.setdefault(r.label, Counter())
        c["boxes"] += 1
        res = r.res or {}
        if "skip" in res or "error" in res:
            c["skipped"] += 1
            continue
        text = normalize(res.get("text") or "", cc["normalize"])
        conf = float(res.get("conf") or 0.0)
        if not text:
            c["empty"] += 1
            continue
        c["readable"] += 1
        c["confident"] += conf >= thr["min_conf"]
        s.lengths.setdefault(r.label, Counter())[len(text)] += 1
        s.chars.update(text)
        names = cc.get("validators") or []
        if names:
            c["validated"] += 1
            if any((validators.get(n) and validators[n](text)[0]) for n in names):
                c["passed"] += 1
        if conf >= thr["min_conf"] and len(text) >= thr["min_id_len"]:
            ids[identity_key(text)][r.split or "(no split)"].add(r.key)
    for ident, by_split in ids.items():
        files = set().union(*by_split.values())
        real_splits = [sp for sp in by_split if sp != "(no split)"]
        if len(real_splits) > 1:
            s.leaks.append((ident, {k: sorted(v) for k, v in by_split.items()}))
        if len(files) >= frequent_min:
            s.frequent.append((ident, len(files)))
    s.leaks.sort(key=lambda x: -sum(len(v) for v in x[1].values()))
    s.frequent.sort(key=lambda x: -x[1])
    return s
