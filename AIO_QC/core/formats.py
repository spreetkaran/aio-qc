# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Is this file really a LabelMe JSON / a YOLO label file — and if not, what is it?

labelme_report(data)  -> FormatReport for a parsed JSON: ok (it is LabelMe), what it looks like when it is not (COCO,
                         Label Studio, VIA, CVAT, Supervisely, Create ML, Labelbox, a JSON list …), file-level
                         problems (wrong types of imagePath / imageWidth / flags …) and per-shape problems (shape is
                         not an object, label not text, points not [x, y] pairs, wrong point count for its
                         shape_type, unknown shape_type, group_id not a number …).
yolo_report(text, …)  -> FormatReport for a .txt: ok, what it looks like when it is not YOLO (class NAMES instead of
                         ids — a class list, KITTI, DOTA; pixel coordinates — not normalised; comma separated; JSON /
                         XML in a .txt; binary), per-line problems (class id, value count, values outside 0…1, zero
                         size boxes, odd polygon coordinates) and whether rows mix boxes and polygons.

Both are pure (no Qt, no file system) — Validation and Image–Label Pairs use them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SHAPE_POINTS = {                     # shape_type -> (min points, max points or None)
    "polygon": (3, None), "rectangle": (2, 2), "circle": (2, 2), "line": (2, 2), "point": (1, 1),
    "linestrip": (2, None), "points": (1, None), "mask": (2, 2), "oriented rectangle": (4, 4),
}
CLASS_LIST_NAMES = {"classes.txt", "obj.names", "classes.names", "labels.txt", "predefined_classes.txt"}


@dataclass
class FormatReport:
    ok: bool                                        # it IS the expected format (maybe with problems)
    looks_like: str = ""                            # when not ok: what the file seems to be
    problems: list = field(default_factory=list)    # file-level: (severity, text)
    items: list = field(default_factory=list)       # per shape / line: (index, severity, text)

    @property
    def summary(self) -> str:
        if not self.ok:
            return f"not this format — looks like {self.looks_like}" if self.looks_like else "not this format"
        n = len(self.problems) + len(self.items)
        return "ok" if not n else f"{n} problem(s)"


# ======================================================================================================== LabelMe
def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def _other_json(data) -> str:
    """What a JSON that is not LabelMe looks like (empty: nothing recognisable)."""
    if isinstance(data, list):
        if not data:
            return "an empty JSON list"
        first = data[0] if isinstance(data[0], dict) else {}
        if {"data"} <= first.keys() and ({"annotations"} & first.keys() or {"predictions"} & first.keys()):
            return "a Label Studio export (a list of tasks)"
        if "image" in first and "annotations" in first:
            return "a Create ML annotation file"
        if "Labeled Data" in first or "DataRow ID" in first or "data_row" in first:
            return "a Labelbox export"
        if {"label", "points"} <= first.keys():
            return "a bare list of shapes (no LabelMe header)"
        return "a JSON list (not a LabelMe object)"
    if not isinstance(data, dict):
        return f"a JSON {type(data).__name__}"
    keys = set(data)
    if {"images", "annotations"} <= keys or {"images", "categories"} <= keys:
        return "a COCO dataset file"
    if {"objects", "size"} <= keys:
        return "a Supervisely annotation"
    if "data" in keys and ("annotations" in keys or "predictions" in keys):
        return "a Label Studio task"
    if "_via_img_metadata" in keys or "_via_settings" in keys:
        return "a VIA project"
    vals = list(data.values())[:5]
    if vals and all(isinstance(v, dict) and {"filename", "regions"} <= v.keys() for v in vals):
        return "a VIA annotation file"
    if {"version", "tags", "shapes"} & keys == {"tags"} or "meta" in keys and "tags" in keys:
        return "a CVAT / generic tags file"
    if "shapes" in keys and not isinstance(data.get("shapes"), list):
        return "LabelMe-like, but 'shapes' is not a list"
    if "shapes" not in keys and "imagePath" in keys:
        return "LabelMe-like, but without 'shapes'"
    if {"info", "licenses"} & keys:
        return "a COCO-style file without images / annotations"
    if not keys:
        return "an empty JSON object"
    return "another JSON (keys: " + ", ".join(sorted(map(str, keys))[:6]) + ("…" if len(keys) > 6 else "") + ")"


def labelme_report(data) -> FormatReport:
    """Strict LabelMe check of a parsed JSON."""
    if not isinstance(data, dict) or not isinstance(data.get("shapes"), list) or "imagePath" not in data:
        return FormatReport(False, _other_json(data))
    r = FormatReport(True)
    ip = data.get("imagePath")
    if not isinstance(ip, str):
        r.problems.append(("Error", f"imagePath is {type(ip).__name__}, not text"))
    elif not ip.strip():
        r.problems.append(("Error", "imagePath is empty"))
    if "version" not in data:
        r.problems.append(("Warning", "no 'version' (LabelMe writes its version in every file)"))
    elif not isinstance(data["version"], str):
        r.problems.append(("Warning", "'version' is not text"))
    for k in ("imageHeight", "imageWidth"):
        v = data.get(k)
        if v is None:
            r.problems.append(("Warning", f"no '{k}'"))
        elif not _num(v) or v <= 0 or float(v) != int(v):
            r.problems.append(("Error", f"'{k}' is {v!r}, not a positive whole number"))
    if "imageData" in data and data["imageData"] is not None and not isinstance(data["imageData"], str):
        r.problems.append(("Error", "'imageData' is neither empty nor base64 text"))
    if "flags" in data and not isinstance(data["flags"], (dict, type(None))):
        r.problems.append(("Warning", "'flags' is not an object"))
    for i, s in enumerate(data["shapes"]):
        r.items.extend((i, sev, t) for sev, t in _shape_problems(s))
    return r


def _shape_problems(s) -> list:
    if not isinstance(s, dict):
        return [("Error", f"shape is a {type(s).__name__}, not an object")]
    out = []
    label = s.get("label")
    if "label" not in s:
        out.append(("Error", "no 'label'"))
    elif not isinstance(label, str):
        out.append(("Error", f"label is {type(label).__name__} {label!r}, not text"))
    pts = s.get("points")
    if not isinstance(pts, list):
        out.append(("Error", "'points' is missing or not a list"))
        return out
    bad = [j for j, p in enumerate(pts) if not (isinstance(p, (list, tuple)) and len(p) == 2 and _num(p[0])
                                                and _num(p[1]))]
    if bad:
        out.append(("Error", f"point {bad[0]} is not an [x, y] pair of numbers" +
                    (f" (and {len(bad) - 1} more)" if len(bad) > 1 else "")))
    st = s.get("shape_type") or "polygon"            # LabelMe treats a missing type as polygon
    if not isinstance(st, str):
        out.append(("Error", f"shape_type is {st!r}, not text"))
    elif st not in SHAPE_POINTS:
        out.append(("Warning", f"unknown shape_type '{st}'"))
    else:
        lo, hi = SHAPE_POINTS[st]
        n = len(pts)
        if n < lo or (hi is not None and n > hi):
            want = f"{lo}" if lo == hi else (f"at least {lo}" if hi is None else f"{lo}–{hi}")
            out.append(("Error", f"{st} with {n} point(s) (needs {want})"))
    g = s.get("group_id")
    if g is not None and not (_num(g) and float(g) == int(g)):
        out.append(("Warning", f"group_id is {g!r}, not a whole number"))
    if "flags" in s and not isinstance(s["flags"], (dict, type(None))):
        out.append(("Warning", "shape 'flags' is not an object"))
    return out


# =========================================================================================================== YOLO
_NUM = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def yolo_report(text: str, n_classes: int = 0, kpt: bool = False, name: str = "") -> FormatReport:
    """Strict YOLO check of a label file's text (detection, segmentation, pose; normalised 0…1 coordinates)."""
    if name.lower() in CLASS_LIST_NAMES:
        return FormatReport(False, "a class list (one name per line), not labels")
    if "\x00" in text:
        return FormatReport(False, "a binary file")
    s = text.lstrip()
    if s[:1] and s[:1] in "{[":
        return FormatReport(False, "JSON saved as .txt")
    if s.startswith("<"):
        return FormatReport(False, "XML (e.g. Pascal VOC) saved as .txt")
    rows = [(ln, line.split()) for ln, line in enumerate(text.splitlines(), 1) if line.strip()]
    if not rows:
        return FormatReport(True)                    # empty = a background image; that is valid YOLO
    if all("," in " ".join(p) for _ln, p in rows) and all(len(p) <= 2 for _ln, p in rows):
        return FormatReport(False, "comma-separated values (CSV), not space-separated YOLO")
    first_num = [_NUM.match(p[0]) is not None for _ln, p in rows]
    if not any(first_num):
        lens = {len(p) for _ln, p in rows}
        if lens == {1} or all(len(p) <= 3 for _ln, p in rows):
            return FormatReport(False, "a list of class names, not labels")
        if any(len(p) in (15, 16) for _ln, p in rows):
            return FormatReport(False, "KITTI labels (class name first, pixel coordinates)")
        return FormatReport(False, "labels with class names instead of class ids")
    if sum(not f for f in first_num) == 0 and all(len(p) in (9, 10) and not _NUM.match(p[8]) for _ln, p in rows):
        return FormatReport(False, "DOTA labels (8 pixel coordinates + class name)")
    r = FormatReport(True)
    big = 0
    seen = set()
    vals_all = 0
    for i, (ln, p) in enumerate(rows):
        cid = p[0]
        if not _NUM.match(cid):
            r.items.append((i, "Error", f"line {ln}: class '{cid}' is not a number"))
            continue
        if float(cid) != int(float(cid)) or float(cid) < 0:
            r.items.append((i, "Error", f"line {ln}: class id {cid} is not a whole number ≥ 0"))
            continue
        if n_classes and int(float(cid)) >= n_classes:
            r.items.append((i, "Error", f"line {ln}: class id {int(float(cid))} (the class list has {n_classes})"))
        nums = p[1:]
        bad = [t for t in nums if not _NUM.match(t)]
        if bad:
            r.items.append((i, "Error", f"line {ln}: '{bad[0]}' is not a number"))
            continue
        v = [float(t) for t in nums]
        n = len(v)
        if n == 4 or (kpt and n > 4):
            kind = "box"
            if v[2] <= 0 or v[3] <= 0:
                r.items.append((i, "Error", f"line {ln}: box with zero / negative size ({v[2]:g} × {v[3]:g})"))
        elif n >= 6 and n % 2 == 0:
            kind = "polygon"
        elif n >= 5 and n % 2 == 1:
            r.items.append((i, "Error", f"line {ln}: {n} coordinates — a polygon needs x, y pairs"))
            continue
        else:
            r.items.append((i, "Error", f"line {ln}: {n} value(s) after the class (box: 4, polygon: 6 or more)"))
            continue
        seen.add(kind)
        geo = v[:4] if kind == "box" else v
        vals_all += 1
        if any(x > 1.0001 for x in geo):
            big += 1
        if any(x < -0.0001 for x in geo):
            r.items.append((i, "Warning", f"line {ln}: negative coordinate"))
    if vals_all and big == vals_all:
        return FormatReport(False, "labels with pixel coordinates (YOLO needs 0…1, normalised by the image size)")
    if big:
        r.problems.append(("Error", f"{big} line(s) with coordinates above 1 (pixel values, or outside the image)"))
    if len(seen) > 1:
        r.problems.append(("Warning", "boxes and polygons mixed in one file (detection and segmentation rows)"))
    return r
