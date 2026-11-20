# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image QC canvas — mode, shown data and display adjustments.

A mixin of AIO_QC.widgets.canvas.Canvas: the methods use the window's own attributes, set up in its __init__.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Set
from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt
from PyQt6.QtGui import (QBrush, QColor, QCursor, QPainterPath, QPen, QPixmap, QPolygonF)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPathItem, QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsView,
)
from AIO_QC.utils import ShapeGroup, load_font

from AIO_QC.widgets.canvas_common import (LabelTag, PointsItem, _is_oom)


class DisplayMixin:

    # ------------------------------------------------------------------ mode
    def setMode(self, mode: str):
        if mode not in (self.DRAG, self.SELECT):
            return
        self.mode = mode
        if mode == self.DRAG:
            self.clearSelection()
            self._set_hover(None)
            if self._drawing:
                self._stop_drawing()
        self._update_handles()
        self._apply_mode_cursor()
        self.modeChanged.emit(mode)

    def _panning(self) -> bool:
        return self.mode == self.DRAG or self._space_pan

    def _over_image(self, pos) -> bool:
        """Is the viewport point `pos` on the image (not the empty canvas around it)?"""
        pix = self.pixmap_item.pixmap()
        return not pix.isNull() and self.pixmap_item.sceneBoundingRect().contains(self.mapToScene(pos))

    def _pan_cursor(self, pos=None):
        """Drag mode: the hand only over the image — the empty canvas around it keeps the normal arrow
        (a closed hand while a pan is in progress)."""
        if pos is None:
            pos = self.viewport().mapFromGlobal(QCursor.pos())
        if self._press_pos is not None:
            shape = Qt.CursorShape.ClosedHandCursor
        elif self._over_image(pos):
            shape = Qt.CursorShape.OpenHandCursor
        else:
            shape = Qt.CursorShape.ArrowCursor
        if self.viewport().cursor().shape() != shape:
            self.viewport().setCursor(shape)

    def _apply_mode_cursor(self):
        if self._panning():
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._pan_cursor()
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    # ----------------------------------------------------------------- data
    # ------------------------------------------------------------- display adjustments
    def display_is_neutral(self) -> bool:
        d = self.display
        return not d["auto"] and d["brightness"] == 0 and d["contrast"] == 0 and abs(d["gamma"] - 1.0) < 1e-3

    def set_display(self, **kw):
        """brightness / contrast (-100..100), gamma (0.3..3), auto (CLAHE). Re-renders without touching zoom."""
        self.display.update(kw)
        self._render_display()

    def _render_display(self):
        if self._original.isNull():
            return
        if self.display_is_neutral():
            if self.pixmap_item.pixmap().cacheKey() != self._original.cacheKey():
                self.pixmap_item.setPixmap(self._original)
            return
        self.pixmap_item.setPixmap(self._adjusted(self._original))

    def _adjusted(self, pixmap: QPixmap) -> QPixmap:
        """The image with brightness / contrast applied — or as it is when memory is too short for the copy."""
        try:
            return self._renderer.render(pixmap, self.display)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            self._renderer.reset()
            self.report_low_memory("brightness / contrast")
            return pixmap

    def report_low_memory(self, what: str):
        self.memoryLow.emit(what)

    def set_annotation_style(self, fill_alpha: int | None = None, outline_scale: float | None = None,
                             tag_scale: float | None = None):
        if tag_scale is not None:
            self.tag_scale = float(tag_scale)
        if fill_alpha is not None:
            self.fill_alpha = int(fill_alpha)
        if outline_scale is not None:
            self.outline_scale = float(outline_scale)
        self.drawShapes()

    def loadPixmap(self, pixmap: QPixmap, same_image: bool = False):
        """Show a new image. Opens at home (fit / 100 %), or — with Keep Previous Zoom, and always when the same
        file is re-read after an outside change (`same_image`) — on the same spot at the same magnification."""
        keep = (self.keep_zoom or same_image) and not self._original.isNull()
        if self._drawing and not same_image:
            self._stop_drawing()                # a shape half drawn belongs to the image it was started on
        if keep:
            self._kept_view = self.view_state()
        self.peek_index = None
        self.clear_overlay()
        self.clear_pred_layer()
        if not same_image:
            self.clear_suggestions()            # Model Assist suggestions belong to their image
        self.clearShapes(emit_signal=True)
        self._renderer.reset()
        self._original = pixmap
        if self.display_is_neutral():
            self.pixmap_item.setPixmap(pixmap)
        else:                                   # adjusted once, never shown unadjusted first
            self.pixmap_item.setPixmap(self._adjusted(pixmap))
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        if (self.keep_zoom or same_image) and self._kept_view is not None:
            self.apply_view_state(self._kept_view)
        else:
            self.apply_default_zoom()           # synchronous: no flash of the previous zoom
        if self._panning():
            self._pan_cursor()                  # the image under the mouse may have changed size

    # ---- Keep Previous Zoom ----------------------------------------------------------
    def view_state(self) -> Optional[dict]:
        """The current view as an intent that transfers to other images: magnification, the image point at the
        viewport centre (as a fraction of the image) and the image size it was taken on. While a glide runs, its
        target. Unchanged since this canvas set it itself (the user did not zoom or pan): the previous intent, so
        a view clamped on a small image never drifts what the next large image shows."""
        pm = self._original
        if pm.isNull():
            return self._kept_view
        moving = self.is_animating() and self._target_center is not None
        z = self._target_zoom if moving else self.current_zoom
        c = self._target_center if moving else self._view_center()
        if self._applied_view is not None and self._kept_view is not None:
            az, ac = self._applied_view
            if abs(z / az - 1) < 1e-3 and QLineF(c, ac).length() * z < 1.0:
                return self._kept_view
        home = self.home_zoom()
        return {"zoom": z, "nx": c.x() / pm.width(), "ny": c.y() / pm.height(), "w": pm.width(), "h": pm.height(),
                "fit": self.zoom_mode == self.FIT_MODE and not moving,
                "home": abs(z / home - 1.0) < 0.02 and abs(c.x() / pm.width() - 0.5) < 0.01
                and abs(c.y() / pm.height() - 0.5) < 0.01}

    def apply_view_state(self, v: dict):
        """Open the current image on a kept view. Same size: the same zoom and pixel. Another size: the same
        point (relative to the image) at the magnification that shows the same share of the image (zoom scaled
        by √(area ratio) — the width ratio for the same aspect), within the zoom limits and the image edges.
        Fit / home views stay fit / home."""
        pm = self._original
        if pm.isNull():
            return
        self.stop_zoom_animation()
        if v.get("home"):
            self.apply_default_zoom()
            self._applied_view = (self.current_zoom, self._view_center())
            return
        if v.get("fit"):
            self.fit_scale = self.compute_fit_scale()
            z, mode = self.fit_scale, self.FIT_MODE
            c = self._image_center()
        else:
            W, H = pm.width(), pm.height()
            z = v["zoom"] if (W, H) == (v["w"], v["h"]) else v["zoom"] * math.sqrt((v["w"] * v["h"]) / (W * H))
            z = max(self.MIN_ZOOM, min(z, self.MAX_ZOOM))
            mode = self.ACTUAL_MODE if abs(z - 1.0) < 1e-6 else self.CUSTOM_MODE
            c = self.clamp_center(z, QPointF(v["nx"] * W, v["ny"] * H))
        self._apply_view(z, c)
        self._finish_zoom(mode)
        self._applied_view = (self.current_zoom, self._view_center())

    def forget_view(self):
        """A new dataset starts at home again."""
        self._kept_view = self._applied_view = None

    def clearAll(self):
        if self.keep_zoom and not self._original.isNull():
            self._kept_view = self.view_state()     # a broken file in between does not lose the kept view
        self._original = QPixmap()
        self._renderer.reset()
        self.clear_overlay()
        self.clear_pred_layer()
        self.clear_suggestions()
        self.clearShapes(emit_signal=True)
        self.pixmap_item.setPixmap(QPixmap())
        if self._panning():
            self._pan_cursor()
        self.zoomFinished.emit()                  # no image: Fit / 100 % buttons un-press

    def loadShapes(self, shapes: List[dict], label_colors: Dict, visible_labels: Set):
        prev = self.selected_group.index if self.selected_group else None
        prev_multi = [g.index for g in self.multi]
        if prev is None or prev != self.peek_index:
            self.peek_index = None
        self.shapes, self.label_colors, self.visible_labels = shapes or [], label_colors, visible_labels
        self.drawShapes()
        if prev is not None and prev in self.index_to_group:
            self.select_shape(prev, emit=False)
            self._set_multi([self.index_to_group[i] for i in prev_multi if i in self.index_to_group and i != prev])
        elif prev is not None:                  # the selected shape is gone / hidden now: everyone must know
            self.peek_index = None
            self.selectionRect.emit(None)
            self.selectionCleared.emit()

    def clearShapes(self, emit_signal=False):
        self._clear_handles()
        self._drag_edit = None
        self._work_pts = None
        for group in self.shape_groups:
            for item in group.items():
                if item.scene() is self.scene:
                    self.scene.removeItem(item)
        had_selection = self.selected_group is not None
        self.selected_group = None
        self.multi = []
        self._multi_work = {}
        self.hover_group = None
        self.shape_groups.clear()
        self.item_to_group.clear()
        self.index_to_group.clear()
        if had_selection:
            self.selectionRect.emit(None)
        if emit_signal and had_selection:
            self.selectionCleared.emit()

    @staticmethod
    def _shape_priority(shape: dict) -> float:
        pts = shape.get("points") or []
        if len(pts) < 2:
            return 0.0
        try:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            return -(max(xs) - min(xs)) * (max(ys) - min(ys))
        except (TypeError, IndexError):
            return 0.0

    def _make_item(self, shape: dict, points: List[QPointF]):
        st = shape.get("shape_type") or "polygon"
        if st == "rectangle" and len(points) == 2:
            return QGraphicsRectItem(QRectF(points[0], points[1]).normalized())
        if st == "circle" and len(points) == 2:
            r = QLineF(points[0], points[1]).length()
            return QGraphicsEllipseItem(QRectF(points[0].x() - r, points[0].y() - r, 2 * r, 2 * r))
        if st == "line" and len(points) == 2:
            return QGraphicsLineItem(QLineF(points[0], points[1]))
        if st in ("point", "points") or len(points) == 1:
            return PointsItem(points, lines=False)            # dots of a constant screen size
        if st == "linestrip" or len(points) < 3:
            path = QPainterPath(points[0])
            for p in points[1:]:
                path.lineTo(p)
            return QGraphicsPathItem(path)
        return QGraphicsPolygonItem(QPolygonF(points))

    def drawShapes(self):
        self.clearShapes()
        if not self.shapes or self.pixmap_item.pixmap().isNull():
            return
        font = self._tag_font()
        pen_w = 2.0 * self.outline_scale        # screen pixels: the same thin line at every zoom (cosmetic pen)

        order = sorted(range(len(self.shapes)), key=lambda i: self._shape_priority(self.shapes[i]))
        for z, idx in enumerate(order):
            shape = self.shapes[idx]
            label = str(shape.get("label", "?"))
            if label not in self.visible_labels and idx != self.peek_index:
                continue
            color = self.label_colors.get(label)
            if color is None:
                continue
            try:
                points = [QPointF(float(p[0]), float(p[1])) for p in shape.get("points", [])]
            except (TypeError, ValueError, IndexError):
                continue
            if not points:
                continue
            item = self._make_item(shape, points)
            pen = QPen(color, pen_w)
            pen.setCosmetic(True)               # zooming in never thickens it over the pixels being checked
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            item.setPen(pen)
            if hasattr(item, "setBrush"):
                fill = QColor(color)
                fill.setAlpha(self.fill_alpha)
                item.setBrush(QBrush(fill) if self.fill_shapes else QBrush(Qt.BrushStyle.NoBrush))
            item.setZValue(self.BASE_SHAPE_Z + z)
            item.setToolTip(self._tooltip(idx, shape))
            self.scene.addItem(item)

            rect_item = text_item = None
            if self.show_tags:
                xs, ys = [p.x() for p in points], [p.y() for p in points]
                if self._stype(shape) == "circle" and len(points) == 2:
                    rr = QLineF(points[0], points[1]).length()
                    xs, ys = [points[0].x() - rr, points[0].x() + rr], [points[0].y() - rr, points[0].y() + rr]
                rect_item = LabelTag(label, color, font)
                rect_item.bbox = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                rect_item.connector = True
                from AIO_QC.core.tag_layout import outline_anchors
                rect_item.anchors = [QPointF(x, y) for x, y in
                                     outline_anchors([(p.x(), p.y()) for p in points], self._stype(shape))]
                rect_item.setPos(rect_item.anchors[0] if rect_item.anchors else rect_item.bbox.topLeft())
                rect_item.setZValue(self.LABEL_BG_Z + z)
                rect_item.area = rect_item.bbox.width() * rect_item.bbox.height()
                self.scene.addItem(rect_item)

            group = ShapeGroup(idx, shape, color, item, rect_item, text_item)
            self.shape_groups.append(group)
            self.index_to_group[idx] = group
            for sub in group.items():
                self.item_to_group[sub] = group
        self.layout_tags()

    def _tag_font(self):
        f = load_font("Inter_24pt-SemiBold.ttf", size=10)
        f.setPixelSize(max(9, round(self.TAG_PX * getattr(self, "tag_scale", 1.0))))
        f.setHintingPreference(f.HintingPreference.PreferNoHinting)
        f.setKerning(True)
        return f

    def _tags(self):
        return [g.rect_item for g in self.shape_groups if isinstance(g.rect_item, LabelTag)]

    def layout_tags(self):
        """Place every visible tag so none covers another — in SCREEN space, redone after every zoom / scroll /
        resize (tags keep their pixel size, so what collides depends on the zoom).

        Each tag is attached to a point ON its shape's outline (top-most vertex first, then left-most, right-most,
        bottom-most …; rectangles: corners; circles: the rim), its pill corner touching that point and a connector
        dot marking it — never on the empty corner of a tilted shape's bounding box. Order: the selected shape
        first, then small shapes before big ones. For each tag the first free spot next to one of its points wins;
        if none is free the pill moves out with a leader line to its point (core/tag_layout.place_anchored). Spots
        stay on the image where possible; every shape's dot is kept clear of other tags."""
        tags = [t for t in self._tags() if t.isVisible()]
        if not tags:
            return
        vp = QRectF(self.viewport().rect())
        img = self.mapFromScene(self.pixmap_item.sceneBoundingRect()).boundingRect()
        bounds = QRectF(img).intersected(vp.adjusted(-200, -200, 200, 200))
        sel = self.selected_group.rect_item if self.selected_group is not None else None
        tags.sort(key=lambda t: (t is not sel, t.area))
        from AIO_QC.core.tag_layout import place_anchored
        tr = self.viewportTransform()                 # scene -> viewport, with sub-pixel precision
        todo, items = [], []
        near = vp.adjusted(-60, -60, 60, 60)
        for t in tags:
            w, h = t.size()
            pts = [tr.map(a) for a in (t.anchors or [t.bbox.topLeft()])]
            on = [(float(q.x()), float(q.y())) for q in pts if near.contains(q)]
            if not on:                                # nothing of the outline on screen: not part of the puzzle
                if t.anchors:
                    t.setPos(t.anchors[0])
                t.set_offset(-2, -h - 3)
                continue
            todo.append((t, pts, on))
            items.append((on, w, h))
        spots = place_anchored(items, (bounds.left(), bounds.top(), bounds.right(), bounds.bottom()))
        inv, _ok = tr.inverted()
        for (t, _pts, on), (i, x, y) in zip(todo, spots):
            ax, ay = on[i]
            if t.anchors:
                t.setPos(inv.map(QPointF(ax, ay)))
            t.set_offset(x - ax, y - ay)

    def _emphasize_tag(self, group):
        """The selected shape's tag: white outline, on top of every other tag, laid out first."""
        for t in self._tags():
            on = group is not None and t is group.rect_item
            if t.emphasis != on:
                t.emphasis = on
                t.update()
        self._schedule_tag_layout()

    def _schedule_tag_layout(self):
        if not hasattr(self, "_tag_timer"):
            from PyQt6.QtCore import QTimer
            self._tag_timer = QTimer(self, singleShot=True, interval=0, timeout=self.layout_tags)
        self._tag_timer.start()

    @staticmethod
    def _tooltip(idx: int, shape: dict) -> str:
        pts = shape.get("points") or []
        try:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            size = f"{max(xs) - min(xs):.1f} × {max(ys) - min(ys):.1f}"
        except (TypeError, ValueError, IndexError):
            size = "?"
        return f"#{idx}  {shape.get('label')}\n{shape.get('shape_type', 'polygon')} · {len(pts)} pts · {size}"
