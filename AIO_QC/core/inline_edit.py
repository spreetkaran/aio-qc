# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Inline edits made on the canvas (Qt-free): geometry, delete, relabel, new box — for every dataset type and every
shape type — plus a byte-exact undo / redo history.

Safety model
    * the canvas shows the file as it was when it was loaded; every edit re-loads the file fresh, checks that it has
      not changed on disk since (mtime + size) AND that the target shape is still the one shown (label + points),
      and only then writes — anything else is refused, never guessed
    * the original of every file is stored once in the backup store (core.backups) before its first inline edit
    * undo / redo restore the exact bytes before / after each edit, and refuse when the file changed since
Coordinates: the canvas works in displayed image pixels; YOLO files are normalised — converted here.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

GEOMETRY_TYPES = ("rectangle", "polygon", "circle", "line", "linestrip", "point", "points")
NUM_POINTS = {"rectangle": 2, "circle": 2, "line": 2, "point": 1}           # fixed point counts
MIN_POINTS = {"polygon": 3, "linestrip": 2, "points": 1}


class EditRefused(Exception):
    """The edit cannot be applied safely (message is shown to the user)."""


def file_sig(path) -> tuple | None:
    try:
        st = os.stat(path)
        return st.st_mtime_ns, st.st_size
    except (OSError, TypeError):
        return None


def stype(shape: dict) -> str:
    return shape.get("shape_type") or "polygon"


def to_px(ann, pts, W, H) -> list:
    sx, sy = (W, H) if ann.normalized else (1.0, 1.0)
    return [[float(p[0]) * sx, float(p[1]) * sy] for p in pts]


def from_px(ann, pts, W, H) -> list:
    sx, sy = (1.0 / W, 1.0 / H) if ann.normalized else (1.0, 1.0)
    out = []
    for x, y in pts:
        x, y = float(x) * sx, float(y) * sy
        out.append([round(x, 6), round(y, 6)] if ann.normalized else [round(x, 2), round(y, 2)])
    return out


def same_shape(ann, idx: int, shown: dict, W, H, tol: float = 0.6) -> bool:
    """Is shape `idx` of the freshly loaded file still the one the canvas showed (label, type, points ± tol px)?"""
    if not 0 <= idx < len(ann.shapes):
        return False
    s = ann.shapes[idx]
    if str(s.get("label")) != str(shown.get("label")) or stype(s) != stype(shown):
        return False
    a, b = to_px(ann, s.get("points") or [], W, H), shown.get("points") or []
    return len(a) == len(b) and all(abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol for p, q in zip(a, b))


def check_geometry_allowed(ann, idx: int, adapter_type: str = ""):
    s = ann.shapes[idx]
    st = stype(s)
    if s.get("_error") or st not in GEOMETRY_TYPES:
        raise EditRefused(f"a {st} shape cannot be reshaped here — relabel / delete work, or use Fix in Editor")
    row = s.get("_row")
    raw = getattr(ann, "raw", None)
    if ann.normalized and row is not None and isinstance(raw, list) and st == "rectangle" and len(raw[row]) > 5:
        raise EditRefused("this box carries pose keypoints — reshape it in Fix in Editor so they move with it")


def set_points(ann, idx: int, pts_px: list, W, H):
    """Replace the points of shape `idx` (validated per shape type)."""
    check_geometry_allowed(ann, idx)
    s = ann.shapes[idx]
    st = stype(s)
    need = NUM_POINTS.get(st)
    if need is not None and len(pts_px) != need:
        raise EditRefused(f"a {st} needs exactly {need} point(s)")
    if st in MIN_POINTS and len(pts_px) < MIN_POINTS[st]:
        raise EditRefused(f"a {st} needs at least {MIN_POINTS[st]} points")
    if st == "rectangle":
        (x1, y1), (x2, y2) = pts_px
        if abs(x2 - x1) < 1 or abs(y2 - y1) < 1:
            raise EditRefused("the box would be empty")
        pts_px = [[min(x1, x2), min(y1, y2)], [max(x1, x2), max(y1, y2)]]
    if (ann.normalized and st == "polygon" and len(s.get("points") or []) == 4 and len(pts_px) != 4
            and s.get("_row") is not None):
        raise EditRefused("4-point YOLO polygons may be oriented boxes — their vertex count is kept")
    s["points"] = from_px(ann, pts_px, W, H)
    s["_geom_dirty"] = True


def delete(ann, idx: int):
    del ann.shapes[idx]


def relabel(ann, idx: int, label: str, can_use=None):
    label = str(label).strip()
    if not label:
        raise EditRefused("empty label")
    if can_use is not None and not can_use(label):
        raise EditRefused(f"'{label}' is not a class of this dataset")
    ann.shapes[idx]["label"] = label


def new_box(ann, rect_px, label: str, W, H) -> dict:
    """A new box shaped like the file's other shapes: a 4-point polygon in a YOLO file that holds only polygons
    (segment / OBB), else a rectangle; LabelMe keys copied from an existing shape (labelme 5 / 6 both)."""
    x1, y1, x2, y2 = rect_px
    polys = [s for s in ann.shapes if stype(s) == "polygon" and not s.get("_error")]
    rects = [s for s in ann.shapes if stype(s) == "rectangle"]
    as_poly = ann.normalized and polys and not rects
    pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] if as_poly else [[x1, y1], [x2, y2]]
    shape = {"label": label, "points": from_px(ann, pts, W, H), "group_id": None, "description": "",
             "shape_type": "polygon" if as_poly else "rectangle", "flags": {},
             "_row": None, "_class_id": None, "_geom_dirty": True, "_new": True}
    if not ann.normalized and ann.shapes:          # LabelMe: same keys as the file's other shapes (5.x / 6.x)
        for k, v in ann.shapes[0].items():
            if k.startswith("_") or k in shape:
                continue
            shape[k] = {} if isinstance(v, dict) else [] if isinstance(v, list) else "" if isinstance(v, str) else None
    if ann.normalized:
        for k in ("group_id", "description", "flags"):
            shape.pop(k, None)
    ann.shapes.append(shape)
    return shape


# ---------------------------------------------------------------------- undo / redo
@dataclass
class Step:
    key: str
    path: str
    before: bytes | None          # None = the file did not exist
    after: bytes | None
    after_sig: tuple | None
    what: str
    when: float = field(default_factory=time.time)
    before_sig: tuple | None = None


class EditHistory:
    """Byte snapshots per edit. Bounded by count and memory (big LabelMe JSONs with imageData)."""

    def __init__(self, max_steps: int = 200, max_bytes: int = 256 * 2 ** 20):
        self.undo_stack: list[Step] = []
        self.redo_stack: list[Step] = []
        self.max_steps, self.max_bytes = max_steps, max_bytes

    def _size(self) -> int:
        return sum(len(s.before or b"") + len(s.after or b"") for s in self.undo_stack + self.redo_stack)

    def push(self, step: Step):
        self.undo_stack.append(step)
        self.redo_stack.clear()
        while len(self.undo_stack) > self.max_steps or (len(self.undo_stack) > 1 and self._size() > self.max_bytes):
            self.undo_stack.pop(0)

    def can_undo(self) -> bool:
        return bool(self.undo_stack)

    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    @staticmethod
    def _write(path: str, data: bytes | None):
        p = Path(path)
        if data is None:
            if p.exists():
                p.unlink()
            return
        tmp = p.with_name(p.name + ".aioqc_undo")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    @staticmethod
    def _is(path: str, data: bytes | None, sig) -> bool:
        """Does the file hold exactly `data` now? (signature first — cheap — then the bytes)"""
        if sig is not None and file_sig(path) == sig:
            return True
        p = Path(path)
        if data is None:
            return not p.exists()
        try:
            return p.read_bytes() == data
        except OSError:
            return False

    def undo(self) -> Step:
        if not self.undo_stack:
            raise EditRefused("nothing to undo")
        st = self.undo_stack[-1]
        if not self._is(st.path, st.after, st.after_sig):
            self.undo_stack.pop()
            raise EditRefused(f"{Path(st.path).name} changed after “{st.what}” — not undone (it would overwrite "
                              "the newer version)")
        self._write(st.path, st.before)
        self.undo_stack.pop()
        st.before_sig = file_sig(st.path)
        self.redo_stack.append(st)
        return st

    def redo(self) -> Step:
        if not self.redo_stack:
            raise EditRefused("nothing to redo")
        st = self.redo_stack[-1]
        if not self._is(st.path, st.before, st.before_sig):
            self.redo_stack.pop()
            raise EditRefused(f"{Path(st.path).name} changed since the undo — not redone")
        self._write(st.path, st.after)
        self.redo_stack.pop()
        st.after_sig = file_sig(st.path)
        self.undo_stack.append(st)
        return st

    def forget(self, keys):
        ks = set(keys)
        self.undo_stack = [s for s in self.undo_stack if s.key not in ks]
        self.redo_stack = [s for s in self.redo_stack if s.key not in ks]
