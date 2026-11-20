# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
LabelMe files exactly as LabelMe 5.11.4 and 6.3.1 write them (Qt-free).

Both versions (checked against their sources bundled in AIO_QC/editors/sources) write the same layout; only the
"version" string differs:

    {"version", "flags", "shapes": [{"label", "points", "group_id", "description", "shape_type", "flags",
                                     "mask"}], "imagePath", "imageData", "imageHeight", "imageWidth"}

UTF-8, ensure_ascii=False, indent 2. imagePath is the image's path relative to the JSON's folder (forward slashes,
as LabelMe stores it on every OS). imageData is null unless asked for, then the image file's own bytes in base64
(what the auto-annotation scripts embed; LabelMe reads both).

Files are written atomically (temporary file + rename), so a stopped run never leaves half a JSON behind.
"""
from __future__ import annotations

import base64
import json
import os

VERSIONS = ("6.3.1", "5.11.4")
DEFAULT_VERSION = "6.3.1"


def rectangle(label: str, x1: float, y1: float, x2: float, y2: float, description: str = "") -> dict:
    return {"label": label, "points": [[round(float(x1), 2), round(float(y1), 2)],
                                       [round(float(x2), 2), round(float(y2), 2)]],
            "group_id": None, "description": description, "shape_type": "rectangle", "flags": {}, "mask": None}


def document(image_path: str, json_path: str, width: int, height: int, shapes: list, version: str = DEFAULT_VERSION,
             image_data: bool = False, flags: dict | None = None) -> dict:
    if version not in VERSIONS:
        raise ValueError(f"LabelMe version must be one of {', '.join(VERSIONS)} (got {version!r})")
    rel = os.path.relpath(os.path.abspath(image_path), os.path.dirname(os.path.abspath(json_path)))
    data = None
    if image_data:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
    return {"version": version, "flags": dict(flags or {}), "shapes": list(shapes),
            "imagePath": rel.replace(os.sep, "/"), "imageData": data,
            "imageHeight": int(height), "imageWidth": int(width)}


def write(json_path: str, doc: dict, origin=None, stamp: bool = True):
    """Write a LabelMe file (with its provenance: what made each shape — `origin`, or the running feature's).
    stamp=False: written as it is (a copy that already says where it came from)."""
    from AIO_QC.core import provenance as PV
    if stamp:
        if origin is None and not PV.active():
            origin = PV.make_origin("manual", "Frame Extractor")
        PV.stamp_labelme(doc, json_path, o=origin)
    tmp = f"{json_path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        os.replace(tmp, json_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def write_yolo(txt_path: str, width: int, height: int, rows):
    """rows: (class_index, x1, y1, x2, y2) in pixels -> normalised YOLO lines."""
    lines = []
    for c, x1, y1, x2, y2 in rows:
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n")
    tmp = f"{txt_path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(lines)
    os.replace(tmp, txt_path)


def labels_in(json_path: str) -> list[str]:
    """Shape labels of a LabelMe file ([] when unreadable)."""
    try:
        with open(json_path, encoding="utf-8") as f:
            return [str(s.get("label", "")) for s in (json.load(f).get("shapes") or []) if isinstance(s, dict)]
    except (OSError, ValueError, AttributeError):
        return []
