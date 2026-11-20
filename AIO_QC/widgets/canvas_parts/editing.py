# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image QC canvas — inline editing: drawing, moving, vertices, nudging.

A mixin of AIO_QC.widgets.canvas.Canvas: the methods use the window's own attributes, set up in its __init__.
"""
from __future__ import annotations
import math
from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt
from PyQt6.QtGui import (QBrush, QColor, QCursor, QPainterPath, QPen, QPolygonF)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QGraphicsPathItem, QGraphicsPolygonItem, QGraphicsRectItem,
)
from AIO_QC.utils import distancetoline

from AIO_QC.widgets.canvas_common import (GEOMETRY_TYPES, HANDLE_PX, MIN_POINTS, PointsItem, _DrawGuide, _DrawKeys, _MAC)


class EditingMixin:

    # ================================================================ inline editing
    def set_editable(self, on: bool, note: str = ""):
        self.editable = bool(on)
        self.edit_note = note
        self._update_handles()

    @staticmethod
    def _stype(shape: dict) -> str:
        return shape.get("shape_type") or "polygon"

    def _geometry_editable(self, group) -> bool:
        if not self.editable or group is None:
            return False
        sh = group.shape_data
        return self._stype(sh) in GEOMETRY_TYPES and bool(sh.get("points")) and not sh.get("_error")

    def _shape_pts(self, group) -> list:
        return [[float(p[0]), float(p[1])] for p in group.shape_data.get("points") or []]

    # ---- handles -----------------------------------------------------------------------------------------------
    def _clear_handles(self):
        for h in self._handles:
            if h.scene() is self.scene:
                self.scene.removeItem(h)
        self._handles = []

    def _handle_specs(self, st: str, pts: list) -> list:
        """[(role, index, x, y)] — role: v = vertex, c = rectangle corner, e = rectangle edge, r = circle radius."""
        if st == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            xa, xb, ya, yb = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
            xm, ym = (xa + xb) / 2, (ya + yb) / 2
            return [("c", 0, xa, ya), ("c", 1, xb, ya), ("c", 2, xb, yb), ("c", 3, xa, yb),
                    ("e", 0, xm, ya), ("e", 1, xb, ym), ("e", 2, xm, yb), ("e", 3, xa, ym)]
        if st == "circle" and len(pts) == 2:
            return [("v", 0, *pts[0]), ("r", 1, *pts[1])]
        return [("v", i, x, y) for i, (x, y) in enumerate(pts)]

    def _update_handles(self, pts=None):
        self._clear_handles()
        g = self.selected_group
        if g is None or not self._geometry_editable(g) or self.mode != self.SELECT:
            return
        st = self._stype(g.shape_data)
        pts = pts if pts is not None else self._shape_pts(g)
        col = QColor(g.color)
        for role, i, x, y in self._handle_specs(st, pts):
            h = QGraphicsRectItem(-HANDLE_PX / 2, -HANDLE_PX / 2, HANDLE_PX, HANDLE_PX)
            h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)   # constant screen size
            h.setPos(QPointF(x, y))
            active = getattr(self, "_active_handle", None) == (role, i)
            h.setBrush(QBrush(col if active else QColor("white") if role != "e" else col.lighter(160)))
            pen = QPen(QColor("white") if active else col.darker(130), 2.0 if active else 1.5)
            pen.setCosmetic(True)
            h.setPen(pen)
            h.setZValue(self.SELECTED_Z + 10)
            h.setData(0, (role, i))
            self.scene.addItem(h)
            self._handles.append(h)

    @classmethod
    def _dir_cursor(cls, dx: float, dy: float):
        """Resize cursor closest to the direction (dx, dy) in screen orientation (y down): ↔ ↖↘ ↕ ↗↙."""
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return Qt.CursorShape.SizeAllCursor
        a = math.degrees(math.atan2(dy, dx)) % 180.0
        return cls._DIR_CURSORS[int((a + 22.5) // 45) % 4]

    def _handle_cursor(self, h) -> Qt.CursorShape:
        """The cursor tells which way a handle resizes:
        rectangle  corners ↖↘ / ↗↙, edge centres ↕ / ↔
        circle     rim handle: the radius direction · centre handle: move (✥)
        polygon    each vertex: pointing outwards from the shape's centre
        line / linestrip   end points: along the line · inner points: away from their neighbours
        point(s)   move (✥)
        The direction follows the view (rotation / flips of the image are taken into account)."""
        role, i = h.data(0)
        if role == "c":
            return Qt.CursorShape.SizeFDiagCursor if i in (0, 2) else Qt.CursorShape.SizeBDiagCursor
        if role == "e":
            return Qt.CursorShape.SizeVerCursor if i in (0, 2) else Qt.CursorShape.SizeHorCursor
        g = self.selected_group
        if g is None:
            return Qt.CursorShape.SizeAllCursor
        st = self._stype(g.shape_data)
        pts = self._shape_pts(g)
        n = len(pts)
        if not pts or i >= n or st in ("point", "points") or (st == "circle" and role == "v"):
            return Qt.CursorShape.SizeAllCursor                     # these move, not resize

        def v(k):                                                   # a point in view (screen) coordinates
            q = self.mapFromScene(QPointF(pts[k][0], pts[k][1]))
            return float(q.x()), float(q.y())
        x, y = v(i)
        if role == "r" and n == 2:                                  # circle rim: along the radius
            ox, oy = v(0)
            return self._dir_cursor(x - ox, y - oy)
        closed = st not in ("line", "linestrip")
        if not closed and (i == 0 or i == n - 1):                   # open ends: along the line
            ox, oy = v(1 if i == 0 else n - 2)
            return self._dir_cursor(x - ox, y - oy)
        if n < 3 and closed:
            return Qt.CursorShape.SizeAllCursor
        return self._corner_cursor((x, y), v((i - 1) % n), v((i + 1) % n))

    @classmethod
    def _corner_cursor(cls, p, a, b):
        """Cursor for a vertex p between neighbours a and b — from the corner's own shape, not the polygon's
        centre (a wide, flat polygon such as a plate still gets diagonal arrows on its corners).
        The direction is the corner's bisector: unit(p − a) + unit(p − b). Resize cursors are two-headed, so
        whether the bisector points in or out (convex or concave corner) does not matter. A vertex on a straight
        run (bisector ≈ 0) gets the edge's normal — perpendicular to that edge. Degenerate neighbours (same
        point) fall back to the other neighbour, then to the move cursor."""
        def unit(dx, dy):
            d = math.hypot(dx, dy)
            return (dx / d, dy / d) if d > 1e-6 else None
        u1, u2 = unit(p[0] - a[0], p[1] - a[1]), unit(p[0] - b[0], p[1] - b[1])
        if u1 is None and u2 is None:
            return Qt.CursorShape.SizeAllCursor
        if u1 is None or u2 is None:
            u = u1 or u2
            return cls._dir_cursor(u[0], u[1])
        bx, by = u1[0] + u2[0], u1[1] + u2[1]
        if math.hypot(bx, by) < 0.15:                               # nearly straight: across the edge
            return cls._dir_cursor(-u1[1], u1[0])
        return cls._dir_cursor(bx, by)

    def _handle_at(self, view_pos):
        if not self._handles:
            return None
        for item in self.items(view_pos):
            if item in self._handles:
                return item
        return None

    # ---- live geometry -----------------------------------------------------------------------------------------
    def _set_item_points(self, group, pts):
        item = group.shape_item
        q = [QPointF(x, y) for x, y in pts]
        if isinstance(item, QGraphicsRectItem) and len(q) == 2:
            item.setRect(QRectF(q[0], q[1]).normalized())
        elif isinstance(item, QGraphicsEllipseItem) and len(q) == 2:
            r = QLineF(q[0], q[1]).length()
            item.setRect(QRectF(q[0].x() - r, q[0].y() - r, 2 * r, 2 * r))
        elif isinstance(item, QGraphicsLineItem) and len(q) == 2:
            item.setLine(QLineF(q[0], q[1]))
        elif isinstance(item, QGraphicsPolygonItem):
            item.setPolygon(QPolygonF(q))
        elif isinstance(item, PointsItem) and q:
            item.set_points(q)
        elif isinstance(item, QGraphicsPathItem) and q:
            path = QPainterPath(q[0])
            for p in q[1:]:
                path.lineTo(p)
            item.setPath(path)
        for it in (group.rect_item, group.text_item):     # the tag is redrawn after the commit
            if it is not None:
                it.setVisible(False)

    def _clamp(self, x, y):
        r = self.pixmap_item.boundingRect()
        return [min(max(x, r.left()), r.right()), min(max(y, r.top()), r.bottom())]

    def _moved(self, pts, dx, dy):
        r = self.pixmap_item.boundingRect()
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        dx = min(max(dx, r.left() - min(xs)), r.right() - max(xs))       # the whole shape stays inside the image
        dy = min(max(dy, r.top() - min(ys)), r.bottom() - max(ys))
        return [[x + dx, y + dy] for x, y in pts]

    def _move_many(self, pairs, dx, dy):
        """Several shapes by the same step, the whole set kept inside the image."""
        r = self.pixmap_item.boundingRect()
        xs = [p[0] for _g, pts in pairs for p in pts]
        ys = [p[1] for _g, pts in pairs for p in pts]
        if not xs:
            return
        dx = min(max(dx, r.left() - min(xs)), r.right() - max(xs))
        dy = min(max(dy, r.top() - min(ys)), r.bottom() - max(ys))
        for grp, pts0 in pairs:
            pts = [[x + dx, y + dy] for x, y in pts0]
            self._multi_work[grp] = (pts0, pts)
            self._set_item_points(grp, pts)
            if grp is self.selected_group:
                self._update_handles(pts)
                self.emit_selection(pts)

    def _commit_many(self):
        work, self._multi_work = self._multi_work, {}
        moved = []
        for grp, (pts0, pts) in work.items():
            pts = self._norm_pts(grp, pts)
            if grp in self.shape_groups and pts != self._norm_pts(grp, pts0):
                moved.append((grp.index, pts))
        if moved:
            self.shapesMoved.emit(sorted(moved))

    # ---- gestures ----------------------------------------------------------------------------------------------
    def _begin_edit(self, event) -> bool:
        """Mouse press in Select mode: handle / outline / body of the SELECTED shape starts an edit."""
        g = self.selected_group
        if g is None:
            return False
        pos = event.pos()
        mods = event.modifiers()
        h = self._handle_at(pos)
        if h is not None:
            if not self._geometry_editable(g):
                return False
            role, i = h.data(0)
            st = self._stype(g.shape_data)
            pts = self._shape_pts(g)
            if mods & Qt.KeyboardModifier.AltModifier and role == "v":
                self._remove_vertex(g, st, pts, i)
                return True
            self._drag_edit = {"kind": role, "i": i, "group": g, "pts0": pts, "start": self.mapToScene(pos),
                               "moved": False, "press": pos}
            self.viewport().setCursor(self._handle_cursor(h))      # kept for the whole drag
            return True
        at = self._group_at(pos)
        if mods & Qt.KeyboardModifier.ControlModifier:
            # on the outline of the only selected polygon / linestrip: a new vertex; anything else: the selection.
            # Another shape under the mouse (a neighbour sharing that edge) means "select it", never a vertex.
            other = self._group_for_selecting(pos)
            if other is not None and other is not g:
                return False
            if (not self.multi and self._geometry_editable(g) and self._stype(g.shape_data) in ("polygon", "linestrip")
                    and self._insert_vertex(g, self.mapToScene(pos), quiet=True)):
                return True
            return False
        if mods & Qt.KeyboardModifier.ShiftModifier:
            return False                                   # Shift+click: the selection (mousePressEvent)
        if at is not g and at not in self.multi:
            return False
        if not self._geometry_editable(g) and not self.multi:
            if self.editable is False and self.edit_note:
                self.editRefused.emit(self.edit_note)
            return False
        self._flush_nudge()
        others = [(m, self._shape_pts(m)) for m in self.multi if self._geometry_editable(m)]
        if not self._geometry_editable(g):
            if not others:
                return False
        self._drag_edit = {"kind": "move", "i": -1, "group": g, "pts0": self._shape_pts(g),
                           "start": self.mapToScene(pos), "moved": False, "press": pos, "others": others,
                           "clicked": at}
        return True

    def _move_edit(self, view_pos):
        d = self._drag_edit
        if not d["moved"] and (view_pos - d["press"]).manhattanLength() < 4:
            return                                         # a click, not a drag: never move by accident
        d["moved"] = True
        g, pts0, kind, i = d["group"], d["pts0"], d["kind"], d["i"]
        cur = self.mapToScene(view_pos)
        st = self._stype(g.shape_data)
        if kind == "move" and d.get("others"):
            self._move_many([(g, pts0)] + d["others"] if self._geometry_editable(g) else d["others"],
                            cur.x() - d["start"].x(), cur.y() - d["start"].y())
            return
        if kind == "move":
            pts = self._moved(pts0, cur.x() - d["start"].x(), cur.y() - d["start"].y())
        else:
            pts = self._resized(st, pts0, kind, i, cur.x(), cur.y())
        self._work_pts = pts
        self._set_item_points(g, pts)
        self._update_handles(pts)
        self.emit_selection(pts)

    def _resized(self, st, pts0, kind, i, x, y):
        """pts0 with handle (kind, i) moved to (x, y) — the same rule for a mouse drag and the arrow keys."""
        x, y = self._clamp(x, y)
        pts = [list(p) for p in pts0]
        if st == "rectangle":
            (x1, y1), (x2, y2) = pts0
            xa, xb, ya, yb = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
            if kind == "c":
                xa, ya, xb, yb = {0: (x, y, xb, yb), 1: (xa, y, x, yb), 2: (xa, ya, x, y), 3: (x, ya, xb, y)}[i]
            else:
                xa, ya, xb, yb = {0: (xa, y, xb, yb), 1: (xa, ya, x, yb), 2: (xa, ya, xb, y), 3: (x, ya, xb, yb)}[i]
            return [[xa, ya], [xb, yb]]
        if st == "circle" and kind == "v":                 # centre handle moves the circle
            return self._moved(pts0, x - pts0[0][0], y - pts0[0][1])
        pts[i] = [x, y]
        return pts

    def _handle_pos(self, st, pts, kind, i):
        for role, k, x, y in self._handle_specs(st, pts):
            if role == kind and k == i:
                return x, y
        return None

    def set_active_handle(self, handle):
        """(kind, index) of the selected shape's handle that the arrow keys move — one edge, corner or vertex —
        or None: the arrow keys move the whole shape."""
        self._active_handle = handle
        self._active_for = self.selected_group.index if self.selected_group is not None else None
        self._update_handles(self._work_pts)
        self.activeHandleChanged.emit(handle is not None)

    def _end_edit(self):
        d, self._drag_edit = self._drag_edit, None
        if d is not None and d.get("others") and d["moved"]:
            self._work_pts = None
            self._commit_many()
            return
        if d is not None and not d["moved"] and self.multi and d["kind"] == "move":
            self._work_pts = None                          # a plain click inside a multi-selection: just that shape
            c = d.get("clicked")
            if c is not None:
                self.select_shape(c.index)
            return
        if d is not None and not d["moved"]:               # a click: a handle becomes the one the arrows move,
            self._work_pts = None                          # a click on the body goes back to moving the shape
            hit = None if d["kind"] == "move" else (d["kind"], d["i"])
            if hit != getattr(self, "_active_handle", None):
                self.set_active_handle(hit)
            return
        if d is None or not d["moved"] or self._work_pts is None:
            self._work_pts = None
            return
        pts, self._work_pts = self._norm_pts(d["group"], self._work_pts), None
        if pts != d["pts0"]:
            self.shapeEdited.emit(d["group"].index, pts)

    def _norm_pts(self, g, pts):
        st = self._stype(g.shape_data)
        if st == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            pts = [[min(x1, x2), min(y1, y2)], [max(x1, x2), max(y1, y2)]]
        return [[round(float(x), 3), round(float(y), 3)] for x, y in pts]

    def _insert_vertex(self, g, scene_pos, quiet: bool = False) -> bool:
        st = self._stype(g.shape_data)
        if st not in ("polygon", "linestrip"):
            if not quiet:
                self.editRefused.emit(f"Vertices can be added to polygons and linestrips, not to a {st}")
            return False
        pts = self._shape_pts(g)
        n = len(pts)
        segs = [(i, (i + 1) % n) for i in range(n if st == "polygon" else n - 1)]
        tol = self.PICK_TOLERANCE_PX * 1.5 / max(self.current_zoom, 1e-6)
        best, best_d = None, tol
        for a, b in segs:
            dd = distancetoline(scene_pos, (QPointF(*pts[a]), QPointF(*pts[b])))
            if dd <= best_d:
                best, best_d = a, dd
        if best is None:
            return False
        x, y = self._clamp(scene_pos.x(), scene_pos.y())
        pts.insert(best + 1, [x, y])
        self.shapeEdited.emit(g.index, self._norm_pts(g, pts))
        return True

    def _remove_vertex(self, g, st, pts, i):
        need = MIN_POINTS.get(st)
        if need is None or len(pts) - 1 < need:
            self.editRefused.emit(f"A {st} needs at least {need or len(pts)} point(s)" if need else
                                  f"Vertices cannot be removed from a {st}")
            return
        del pts[i]
        self.shapeEdited.emit(g.index, self._norm_pts(g, pts))

    # ---- keys --------------------------------------------------------------------------------------------------
    def _edit_key(self, event) -> bool:
        key, mods = event.key(), event.modifiers()
        plain = not (mods & ~Qt.KeyboardModifier.KeypadModifier)          # numpad keys count as plain keys
        g = self.selected_group
        if g is None or self.mode != self.SELECT:
            return False
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and plain:
            if self.editable and self.multi:
                self._flush_nudge()
                self.shapesDeleteRequested.emit(self.selected_indices())
            elif self.editable:
                self.shapeDeleteRequested.emit(g.index)
            elif self.edit_note:
                self.editRefused.emit(self.edit_note)
            return True
        digit = Qt.Key.Key_1 <= key <= Qt.Key.Key_9
        if plain and (digit or key in (Qt.Key.Key_L, Qt.Key.Key_F2)):
            if self.editable:
                n = key - Qt.Key.Key_0 if digit else 0
                self.relabelRequested.emit(g.index, int(n))
            elif self.edit_note:
                self.editRefused.emit(self.edit_note)
            return True
        arrows = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0), Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1)}
        if key in arrows and not (mods & ~Qt.KeyboardModifier.ShiftModifier & ~Qt.KeyboardModifier.KeypadModifier):
            step = 10.0 if mods & Qt.KeyboardModifier.ShiftModifier else 1.0
            dx, dy = arrows[key]
            if self.multi:                                 # the whole selection moves together
                pairs = [(m, self._multi_work[m][0] if m in self._multi_work else self._shape_pts(m))
                         for m in [g] + self.multi if self._geometry_editable(m)]
                if not pairs:
                    return False
                tx, ty = self._multi_total = (getattr(self, "_multi_total", (0.0, 0.0))[0] + dx * step,
                                              getattr(self, "_multi_total", (0.0, 0.0))[1] + dy * step)
                self._move_many(pairs, tx, ty)
                self._nudge_timer.start()
                return True
            if not self._geometry_editable(g):
                return False
            base = self._work_pts if self._work_pts is not None else self._shape_pts(g)
            if not hasattr(self, "_nudge_from") or self._nudge_from is None:
                self._nudge_from = (g, self._shape_pts(g))
            ah = getattr(self, "_active_handle", None)
            st = self._stype(g.shape_data)
            hp = self._handle_pos(st, base, *ah) if ah is not None else None
            if hp is not None:                             # just that edge / corner / vertex
                kind, i = ah
                if kind == "e":                            # an edge moves only across itself
                    dx, dy = (0, dy) if i in (0, 2) else (dx, 0)
                    if (dx, dy) == (0, 0):
                        return True
                self._work_pts = self._resized(st, base, kind, i, hp[0] + dx * step, hp[1] + dy * step)
            else:
                self._work_pts = self._moved(base, dx * step, dy * step)
            self._set_item_points(g, self._work_pts)
            self._update_handles(self._work_pts)
            self._nudge_timer.start()                     # one write after the keys stop, not one per press
            return True
        return False

    def _commit_nudge(self):
        if self._multi_work:
            self._multi_total = (0.0, 0.0)
            self._commit_many()
            return
        self._multi_total = (0.0, 0.0)
        src = getattr(self, "_nudge_from", None)
        self._nudge_from = None
        pts, self._work_pts = self._work_pts, None
        if src is None or pts is None:
            return
        g, pts0 = src
        pts = self._norm_pts(g, pts)
        if pts != pts0 and g in self.shape_groups:
            self.shapeEdited.emit(g.index, pts)

    def _flush_nudge(self):
        if self._nudge_timer.isActive():
            self._nudge_timer.stop()
            self._commit_nudge()

    # ---- drawing a new box ---------------------------------------------------------------------------------------
    def start_drawing(self, kind: str = "box") -> bool:
        """Create a new bbox (drag) or a new polygon (click its corners; a click on the first corner, Enter or a
        double-click closes it; Ctrl+Z / Backspace takes the last corner back, Ctrl+Shift+Z puts it back; Shift keeps
        an edge at 45° steps; Esc cancels). -> False when this image cannot be edited (it says why)."""
        if self.pixmap_item.pixmap().isNull():
            return False
        if not self.editable:
            if self.edit_note:
                self.editRefused.emit(self.edit_note)
            return False
        if self._drawing:
            self._stop_drawing()
        self._start_drawing(kind)
        return True

    def cancel_drawing(self):
        if self._drawing:
            self._stop_drawing()

    def _start_drawing(self, kind: str = "box"):
        if self.pixmap_item.pixmap().isNull():
            return
        if self.mode == self.DRAG:
            self.setMode(self.SELECT)
        self._draw_kind = kind
        self._poly_pts, self._poly_redo, self._poly_cursor, self._poly_snap = [], [], None, False
        self._drawing = True
        self._set_hover(None)                      # the shape under the mouse when drawing started
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self._poly_item = _DrawGuide(self)
        self._poly_item.kind = kind
        self.scene.addItem(self._poly_item)
        if self._draw_keys is None:
            self._draw_keys = _DrawKeys(self)
            self.installEventFilter(self._draw_keys)
        self.setFocus()                            # Esc / Enter / Ctrl+Z come here, wherever the focus was
        vp = self.viewport().mapFromGlobal(QCursor.pos())
        if self.viewport().rect().contains(vp):
            self._poly_mouse = vp
            self._poly_follow()
        self._update_hint()

    def _stop_drawing(self):
        self._drawing = False
        if self._draw_keys is not None:
            self.removeEventFilter(self._draw_keys)
            self._draw_keys.deleteLater()
            self._draw_keys = None
        self._draw_start = None
        self._poly_pts, self._poly_redo, self._poly_cursor, self._poly_snap = [], [], None, False
        if self._poly_item is not None and self._poly_item.scene() is self.scene:
            self.scene.removeItem(self._poly_item)
        self._poly_item = None
        if self._draw_item is not None and self._draw_item.scene() is self.scene:
            self.scene.removeItem(self._draw_item)
        self._draw_item = None
        self._update_hint()
        self._apply_mode_cursor()

    # ---- the hint over the image (bottom centre) -----------------------------------------------------------------------
    def _hint_text(self) -> str:
        k = lambda t: f"<b>{t}</b>"                                                  # noqa: E731
        if self._draw_kind == "box":
            if self._draw_start is not None:
                return f"Release to finish the box  ·  {k('Esc')} cancel"
            return f"Drag to draw a bbox  ·  {k('Esc')} cancel"
        n = len(self._poly_pts)
        if n == 0:
            return f"Click to place the first corner  ·  {k('Esc')} cancel"
        corners = f"{n} corner{'s' if n != 1 else ''}"
        if self._poly_snap:
            return f"{corners}  ·  Click to close the polygon"
        if n < 3:
            return (f"{corners}  ·  click to add  ·  {k('Shift')} 45°  ·  {k('Ctrl+Z')} undo  ·  {k('Esc')} cancel")
        return (f"{corners}  ·  click the first corner or {k('Enter')} to close  ·  {k('Ctrl+Z')} undo  ·  "
                f"{k('Esc')} cancel")

    def _update_hint(self):
        from PyQt6.QtWidgets import QLabel
        if not self._drawing:
            if self._draw_hint is not None:
                self._draw_hint.hide()
            return
        h = self._draw_hint
        if h is None:
            h = self._draw_hint = QLabel(self.viewport())
            h.setObjectName("drawHint")
            h.setTextFormat(Qt.TextFormat.RichText)
            h.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            h.setStyleSheet("#drawHint{background:rgba(20,23,28,215); color:#E3E7EC; border:1px solid "
                            "rgba(255,255,255,30); border-radius:8px; padding:5px 12px;}")
        text = self._hint_text().replace("Ctrl+", "⌘" if _MAC else "Ctrl+")
        if h.text() != text:
            h.setText(text)
            h.adjustSize()
        vw = self.viewport().width()
        if h.width() > vw - 24:
            h.setFixedWidth(max(80, vw - 24))
            h.setWordWrap(True)
            h.adjustSize()
        x = max(12, (vw - h.width()) // 2)
        y = self.viewport().height() - h.height() - 14             # bottom centre: hint cards live at the top
        if x + h.width() > vw - 240:                              # clear of the pointer read-out (bottom right)
            y -= 40
        h.move(x, max(12, y))
        h.show()
        h.raise_()

    def _begin_draw(self, view_pos):
        p = self.mapToScene(view_pos)
        self._draw_start = QPointF(*self._clamp(p.x(), p.y()))
        self._draw_item = QGraphicsRectItem(QRectF(self._draw_start, self._draw_start))
        pen = QPen(QColor("#ffd400"), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._draw_item.setPen(pen)
        self._draw_item.setZValue(self.SELECTED_Z + 20)
        self.scene.addItem(self._draw_item)
        self._update_poly()                                         # the crosshair goes while dragging
        self._update_hint()

    def _move_draw(self, view_pos):
        p = self.mapToScene(view_pos)
        q = QPointF(*self._clamp(p.x(), p.y()))
        self._draw_item.setRect(QRectF(self._draw_start, q).normalized())
        self.selectionRect.emit(self._draw_item.rect())

    def _poly_target(self, view_pos, mods=None):
        """-> (scene point for the pointer, on the first corner?). Shift keeps the edge at 0 / 45 / 90°."""
        pts = self._poly_pts
        if len(pts) >= 3:
            first = self.mapFromScene(pts[0])
            if math.hypot(first.x() - view_pos.x(), first.y() - view_pos.y()) <= self.SNAP_PX:
                return QPointF(pts[0]), True
        p = self.mapToScene(view_pos)
        if mods is None:
            from PyQt6.QtWidgets import QApplication
            mods = QApplication.keyboardModifiers()
        if pts and mods & Qt.KeyboardModifier.ShiftModifier:
            o = pts[-1]
            dx, dy = p.x() - o.x(), p.y() - o.y()
            ang = round(math.atan2(dy, dx) / (math.pi / 4)) * (math.pi / 4)
            d = math.hypot(dx, dy)
            p = QPointF(o.x() + d * math.cos(ang), o.y() + d * math.sin(ang))
        return QPointF(*self._clamp(p.x(), p.y())), False

    def _poly_follow(self, mods=None):
        """The guide follows the pointer (box: the crosshair; polygon: the rubber edge and the first-corner snap)."""
        if not self._drawing or self._poly_mouse is None:
            return
        if self._draw_kind == "box":
            p = self.mapToScene(self._poly_mouse)
            self._poly_cursor = QPointF(*self._clamp(p.x(), p.y()))
            snap = False
        else:
            self._poly_cursor, snap = self._poly_target(self._poly_mouse, mods)
        changed = snap != self._poly_snap
        self._poly_snap = snap
        self._update_poly()
        if changed:
            self._update_hint()

    def _add_poly_point(self, view_pos, mods=None):
        q, snap = self._poly_target(view_pos, mods)
        if snap:                                                    # back on the first corner: closed
            self._finish_polygon()
            return
        if self._poly_pts and (self.mapFromScene(self._poly_pts[-1]) - view_pos).manhattanLength() <= 2:
            return                                                  # the same corner twice (a double-click)
        self._poly_pts.append(q)
        self._poly_redo.clear()
        self._poly_cursor = q
        self._update_poly()
        self._update_hint()

    def undo_corner(self) -> bool:
        """Ctrl+Z while drawing: take the last corner back (with none left, the drawing goes on: Esc stops it)."""
        if not self._poly_pts:
            return False
        self._poly_redo.append(self._poly_pts.pop())
        self._poly_follow()
        self._update_poly()
        self._update_hint()
        return True

    def redo_corner(self) -> bool:
        if not self._poly_redo:
            return False
        self._poly_pts.append(self._poly_redo.pop())
        self._poly_follow()
        self._update_poly()
        self._update_hint()
        return True

    def _update_poly(self):
        g = self._poly_item
        if g is None:
            return
        g.kind = self._draw_kind
        g.pts = list(self._poly_pts)
        g.cursor = self._poly_cursor
        g.snap = self._poly_snap
        g.update()

    def _finish_polygon(self):
        pts = [[float(q.x()), float(q.y())] for q in self._poly_pts]
        if len(pts) < 3:                                            # keep drawing: say what is missing
            self.editRefused.emit(f"A polygon needs at least 3 corners — {3 - len(pts)} more to go (Esc cancels)")
            return
        self._stop_drawing()
        self._apply_mode_cursor()
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if (max(xs) - min(xs)) * self.current_zoom < 3 or (max(ys) - min(ys)) * self.current_zoom < 3:
            return
        self.polygonDrawn.emit(pts)

    def _end_draw(self, view_pos):
        rect = self._draw_item.rect() if self._draw_item is not None else QRectF()
        self._stop_drawing()
        self._refresh_hover(view_pos)
        self.emit_selection()
        if rect.width() * self.current_zoom >= 4 and rect.height() * self.current_zoom >= 4:
            self.boxDrawn.emit(rect)
