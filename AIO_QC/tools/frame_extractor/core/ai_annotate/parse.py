# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
From a model's answer to boxes in image pixels (Qt-free): JSON found in any wrapping, the five box conventions,
text checks (ISO 6346, Luhn), and the accept rules (a tiny, safe expression language — no eval).
"""
from __future__ import annotations

import json
import re


def extract_json(text: str):
    """The JSON object (or list) in an answer — bare, in ```json fences, or surrounded by words."""
    if text is None:
        raise ValueError("empty answer")
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in the answer")
    s = min(starts)
    depth, in_str, esc = 0, False, False
    for i in range(s, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                chunk = t[s:i + 1]
                try:
                    return json.loads(chunk)
                except ValueError:
                    return json.loads(re.sub(r",\s*([}\]])", r"\1", chunk))   # trailing commas
    raise ValueError("the JSON in the answer is cut off (raise max_tokens)")


def to_pixels(box, fmt: str, sent_w: int, sent_h: int, scale_x: float, scale_y: float, off_x=0.0, off_y=0.0):
    """A model box -> (x1, y1, x2, y2) in pixels of the original image.
    sent_w/h: the image the model saw; scale: original px per sent px; off: where that image (crop) sits."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"a box needs 4 numbers, got {box!r}")
    a, b, c, d = (float(v) for v in box)
    if fmt == "yxyx_1000":
        y1, x1, y2, x2 = a, b, c, d
        x1, x2, y1, y2 = x1 / 1000 * sent_w, x2 / 1000 * sent_w, y1 / 1000 * sent_h, y2 / 1000 * sent_h
    elif fmt == "xyxy_1000":
        x1, y1, x2, y2 = a / 1000 * sent_w, b / 1000 * sent_h, c / 1000 * sent_w, d / 1000 * sent_h
    elif fmt == "xyxy_norm":
        x1, y1, x2, y2 = a * sent_w, b * sent_h, c * sent_w, d * sent_h
    elif fmt == "yxyx_norm":
        x1, y1, x2, y2 = b * sent_w, a * sent_h, d * sent_w, c * sent_h
    elif fmt == "xyxy_px":
        x1, y1, x2, y2 = a, b, c, d
    else:
        raise ValueError(f"unknown box format {fmt}")
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (off_x + x1 * scale_x, off_y + y1 * scale_y, off_x + x2 * scale_x, off_y + y2 * scale_y)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def inside(child, parent, tol=0.85) -> bool:
    """At least `tol` of the child's area lies in the parent."""
    ix = max(0.0, min(child[2], parent[2]) - max(child[0], parent[0]))
    iy = max(0.0, min(child[3], parent[3]) - max(child[1], parent[1]))
    a = (child[2] - child[0]) * (child[3] - child[1])
    return a > 0 and ix * iy / a >= tol


def nms(dets, thr):
    """dets: [dict(box, conf, label)] -> kept, highest confidence first, per label."""
    out = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if all(o["label"] != d["label"] or iou(o["box"], d["box"]) <= thr for o in out):
            out.append(d)
    return out


# ------------------------------------------------------------------------------------------------ text checks
def iso6346_ok(text: str) -> bool:
    """Owner (3 letters) + category (U/J/Z) + 6 digits + check digit."""
    t = re.sub(r"[\s\-\[\]()|]", "", (text or "").upper())
    if not re.fullmatch(r"[A-Z]{3}[UJZ]\d{7}", t):
        return False
    vals, v = {}, 10
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if v % 11 == 0:
            v += 1
        vals[ch] = v
        v += 1
    total = sum((vals[c] if c.isalpha() else int(c)) * 2 ** i for i, c in enumerate(t[:10]))
    return total % 11 % 10 == int(t[10])


def luhn_ok(text: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", text or "")]
    if len(d) < 2:
        return False
    s = 0
    for i, n in enumerate(reversed(d)):
        if i % 2:
            n *= 2
            n = n - 9 if n > 9 else n
        s += n
    return s % 10 == 0


def text_ok(text, pattern="", check="none") -> bool:
    if text is None or str(text).strip() == "":
        return False
    t = str(text).strip()
    if pattern and not re.search(pattern, t):
        return False
    if check == "iso6346":
        return iso6346_ok(t)
    if check == "luhn":
        return luhn_ok(t)
    return True


# ------------------------------------------------------------------------------------------------ accept rules
_RULE = re.compile(r"^\s*(len\(\s*(\w+)\s*\)|count\(\s*([\w\- .]+?)\s*\)|(boxes)|(\w+))\s*(>=|<=|==|!=|>|<)\s*(.+?)"
                   r"\s*$")


def _value(raw: str):
    r = raw.strip()
    low = r.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return float(r)
    except ValueError:
        return r.strip("'\"")


def rule_ok(rule: str, fields: dict, counts: dict) -> tuple[bool, str]:
    """(passes, why) — `photorealistic >= 4`, `same_viewpoint == true`, `len(artifacts) <= 1`, `count(x) >= 1`."""
    m = _RULE.match(str(rule))
    if not m:
        return False, f"rule not understood: {rule}"
    if m.group(2):
        v = fields.get(m.group(2))
        left = len(v) if isinstance(v, (list, tuple, str, dict)) else 0
    elif m.group(3):
        left = counts.get(m.group(3).strip(), 0)
    elif m.group(4):
        left = sum(counts.values())
    else:
        left = fields.get(m.group(5))
        if left is None:
            return False, f"{m.group(5)} missing in the answer"
    op, right = m.group(6), _value(m.group(7))
    try:
        if isinstance(right, bool) or isinstance(left, bool):
            lv = left if isinstance(left, bool) else str(left).strip().lower() in ("true", "yes", "1")
            ok = {"==": lv == right, "!=": lv != right}.get(op)
            if ok is None:
                return False, f"{rule}: true/false can only be compared with == or !="
        elif isinstance(right, float):
            lf = float(left)
            ok = {">=": lf >= right, "<=": lf <= right, ">": lf > right, "<": lf < right, "==": lf == right,
                  "!=": lf != right}[op]
        else:
            ok = {"==": str(left) == right, "!=": str(left) != right}.get(op, False)
    except (TypeError, ValueError):
        return False, f"{rule}: the answer had {left!r}"
    return ok, "" if ok else f"{rule} (was {left!r})"
