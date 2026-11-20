# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image QC canvas — mouse and keyboard helpers (the Qt event handlers stay on Canvas).

A mixin of AIO_QC.widgets.canvas.Canvas: the methods use the window's own attributes, set up in its __init__.
"""
from __future__ import annotations
from typing import Optional
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (QKeySequence)
from AIO_QC.utils import ShapeGroup



class InputMixin:

    # ---------------------------------------------------------------- events
    def outOfPixmap(self, scene_pos: QPointF) -> bool:
        if self.pixmap_item.pixmap().isNull():
            return True
        return not self.pixmap_item.sceneBoundingRect().contains(scene_pos)

    def cycle_shape(self, step: int):
        """Tab / Shift+Tab: select the next / previous visible shape (in annotation order)."""
        if not self.shape_groups:
            return
        order = sorted(self.index_to_group)
        cur = self.selected_group.index if self.selected_group else None
        if cur is None or cur not in order:
            idx = order[0] if step > 0 else order[-1]
        else:
            idx = order[(order.index(cur) + step) % len(order)]
        if self.mode == self.DRAG:
            self.setMode(self.SELECT)
        self.select_shape(idx)
        self.ensureVisible(self.index_to_group[idx].shape_item.sceneBoundingRect(), 40, 40)

    def _is_draw_key(self, e) -> bool:
        k = e.key()
        if k == Qt.Key.Key_Escape:
            return True
        if self._draw_kind != "polygon":
            return False
        return (k in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Backspace, Qt.Key.Key_Delete)
                or e.matches(QKeySequence.StandardKey.Undo) or e.matches(QKeySequence.StandardKey.Redo))

    def _modifier_select(self, event) -> bool:
        """Ctrl+click (⌘ on macOS) / Shift+click on a shape selects it — in Drag mode too (Drag mode never selects
        on a plain click; a modifier click is a selection gesture, so it switches to Select like any selection).
        Select mode keeps its own handling (_begin_edit: a vertex on the selected polygon's outline)."""
        if self.mode != self.DRAG or self._space_pan:
            return False
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if not (ctrl or shift):
            return False
        group = self._group_for_selecting(event.pos())
        if group is None:
            return False                                       # empty image: a normal pan
        self.toggle_shape(group.index) if ctrl else self.add_shape(group.index)
        self._apply_mode_cursor()
        return True

    def _refresh_hover(self, view_pos):
        """After a gesture: highlight what is under the mouse now (no move needed to bring it back)."""
        if self._drawing or self._panning():
            return
        self._set_hover(self._group_at(view_pos))

    def _set_hover(self, group: Optional[ShapeGroup]):
        if group is self.hover_group:
            return
        if self.hover_group is not None and self.hover_group in self.shape_groups:
            self.hover_group.set_hover(False)
        self.hover_group = group
        if group is not None:
            group.set_hover(True)

    def report_cursor(self, view_pos):
        """Status bar: the image pixel under view_pos (None off the image). Modes that filter the viewport's mouse
        events (ROI, sketch) call it too."""
        if view_pos is None or self.pixmap_item.pixmap().isNull():
            self.cursorMoved.emit(None)
            return
        p = self.mapToScene(view_pos)
        r = self.pixmap_item.boundingRect()
        self.cursorMoved.emit(QPointF(p) if r.contains(p) else None)

    @staticmethod
    def points_rect(pts, shape_type=None):
        try:
            xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
        except (TypeError, ValueError, IndexError):
            return None
        if not xs:
            return None
        if shape_type == "circle" and len(xs) == 2:
            r = ((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2) ** 0.5
            return QRectF(xs[0] - r, ys[0] - r, 2 * r, 2 * r)
        return QRectF(QPointF(min(xs), min(ys)), QPointF(max(xs), max(ys)))

    def emit_selection(self, pts=None):
        g = self.selected_group
        if pts is None and g is not None:
            pts = g.shape_data.get("points") or []
        if g is None and pts is None:
            self.selectionRect.emit(None)
            return
        self.selectionRect.emit(self.points_rect(pts, self._stype(g.shape_data) if g is not None else None))
