# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image QC canvas — overlays: labels, guides, highlights.

A mixin of AIO_QC.widgets.canvas.Canvas: the methods use the window's own attributes, set up in its __init__.
"""
from __future__ import annotations
from typing import Optional
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (QColor, QPen)
from PyQt6.QtWidgets import (
    QGraphicsRectItem,
)
from AIO_QC.utils import ShapeGroup, distance, distancetoline

from AIO_QC.widgets.canvas_common import (LabelTag, PointsItem)


class OverlayMixin:

    # ---------------------------------------------------------------- overlay
    def set_overlay(self, boxes: list[dict]):
        """
        Dashed boxes that are NOT annotations, e.g. model predictions:
        [{"box": (x1, y1, x2, y2), "text": "plate 0.92", "color": "#ff00ff"}]
        Cleared automatically when another image is loaded.
        """
        self.clear_overlay()
        tag_font = self._tag_font()
        for b in boxes:
            x1, y1, x2, y2 = map(float, b["box"])
            color = QColor(b.get("color", "#ff00ff"))
            rect = QGraphicsRectItem(QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized())
            pen = QPen(color, 2.0, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            rect.setPen(pen)
            rect.setZValue(self.LABEL_TEXT_Z * 2)
            self.scene.addItem(rect)
            self.overlay_items.append(rect)
            if b.get("text"):                     # same crisp, constant-size pill as the class labels
                t = LabelTag(b["text"], color, tag_font)
                t.setPos(QPointF(min(x1, x2), max(y1, y2)))
                t.set_offset(0, 2)
                t.setZValue(self.LABEL_TEXT_Z * 2 + 1)
                self.scene.addItem(t)
                self.overlay_items.append(t)

    def set_pred_layer(self, items: list[dict]):
        """
        The model-predictions layer (not annotations, never saved), kept until the next image:
        [{"box": (x1, y1, x2, y2), "poly": [(x, y), …] or None, "kpts": [(x, y, c), …] or None, "text": str,
          "color": "#rrggbb"}] — dashed outlines with a constant-size tag.
        """
        self.clear_pred_layer()
        from PyQt6.QtGui import QPolygonF
        from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsPolygonItem
        tag_font = self._tag_font()
        for b in items:
            color = QColor(b.get("color", "#ff00ff"))
            pen = QPen(color, 1.8, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            poly = b.get("poly")
            if poly is not None and len(poly) >= 3:
                item = QGraphicsPolygonItem(QPolygonF([QPointF(float(x), float(y)) for x, y in poly]))
            else:
                x1, y1, x2, y2 = map(float, b["box"])
                item = QGraphicsRectItem(QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized())
            item.setPen(pen)
            item.setZValue(self.LABEL_TEXT_Z * 2 - 2)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.scene.addItem(item)
            self.pred_items.append(item)
            for x, y, c in (b.get("kpts") or ()):
                if c >= 0.3 and (x or y):
                    r = PointsItem.RADIUS_PX                       # screen pixels, like the point shapes
                    dot = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
                    dot.setFlag(dot.GraphicsItemFlag.ItemIgnoresTransformations, True)
                    dot.setPos(float(x), float(y))
                    dot.setBrush(color)
                    dot.setPen(QPen(QColor(0, 0, 0, 160), 0))
                    dot.setZValue(self.LABEL_TEXT_Z * 2 - 1)
                    self.scene.addItem(dot)
                    self.pred_items.append(dot)
            if b.get("text"):
                x1, y1, x2, y2 = map(float, b["box"])
                t = LabelTag(b["text"], color, tag_font)
                t.setPos(QPointF(min(x1, x2), max(y1, y2)))
                t.set_offset(0, 2)
                t.setZValue(self.LABEL_TEXT_Z * 2 - 1)
                self.scene.addItem(t)
                self.pred_items.append(t)

    def set_suggestions(self, items: list[dict]):
        """
        Model Assist: suggestions that are NOT annotations until clicked —
        [{"box": (x1, y1, x2, y2), "poly": [(x, y), …] or None, "text": str, "color": "#rrggbb"}] in image pixels
        (the scene). Drawn dashed and lightly filled above the shapes; a click accepts one (suggestionClicked),
        a right-click drops it (suggestionDismissed). Cleared with the next image.
        """
        self.clear_suggestions()
        from PyQt6.QtGui import QBrush, QPolygonF
        from PyQt6.QtWidgets import QGraphicsPolygonItem
        tag_font = self._tag_font()
        for b in items:
            color = QColor(b.get("color", "#1e88e5"))
            pen = QPen(color, 2.2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            fill = QColor(color)
            fill.setAlpha(40)
            poly = b.get("poly")
            if poly is not None and len(poly) >= 3:
                item = QGraphicsPolygonItem(QPolygonF([QPointF(float(x), float(y)) for x, y in poly]))
            else:
                x1, y1, x2, y2 = map(float, b["box"])
                item = QGraphicsRectItem(QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized())
            item.setPen(pen)
            item.setBrush(QBrush(fill))
            item.setZValue(self.LABEL_TEXT_Z * 2 + 4)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            item.setCursor(Qt.CursorShape.PointingHandCursor)
            self.scene.addItem(item)
            tag = None
            if b.get("text"):
                x1, y1, x2, y2 = map(float, b["box"])
                tag = LabelTag(b["text"], color, tag_font)
                tag.setPos(QPointF(min(x1, x2), min(y1, y2)))
                tag.set_offset(0, -2)
                tag.setZValue(self.LABEL_TEXT_Z * 2 + 5)
                self.scene.addItem(tag)
            self.suggestions.append((item, tag, b))

    def clear_suggestions(self):
        for item, tag, _b in self.suggestions:
            for it in (item, tag):
                if it is not None and it.scene() is self.scene:
                    self.scene.removeItem(it)
        self.suggestions = []

    def suggestion_at(self, view_pos) -> int | None:
        """The suggestion under the mouse — the smallest when several overlap (a plate inside a car)."""
        p = self.mapToScene(view_pos)
        best, area = None, None
        for i, (item, tag, b) in enumerate(self.suggestions):
            hit = item.contains(item.mapFromScene(p)) or (tag is not None and tag.sceneBoundingRect().contains(p))
            if not hit:
                continue
            r = item.boundingRect()
            a = r.width() * r.height()
            if area is None or a < area:
                best, area = i, a
        return best

    def clear_pred_layer(self):
        for it in self.pred_items:
            if it.scene() is self.scene:
                self.scene.removeItem(it)
        self.pred_items = []

    def clear_overlay(self):
        for it in self.overlay_items:
            if it.scene() is self.scene:
                self.scene.removeItem(it)
        self.overlay_items = []

    def focus_rect(self, rect: QRectF):
        """Glide to an arbitrary scene rectangle (e.g. an unlabeled object the model found)."""
        if rect.isEmpty():
            return
        m = max(rect.width(), rect.height()) * 0.35
        rect = rect.adjusted(-m, -m, m, m)
        vp = self._viewport_size()
        z = min(vp.width() / rect.width(), vp.height() / rect.height())
        self.animate_to(z, center=rect.center(), mode=self.CUSTOM_MODE, duration=320)

    def _group_at(self, view_pos) -> Optional[ShapeGroup]:
        for item in self.items(view_pos):  # topmost first
            group = self.item_to_group.get(item)
            if group is not None:
                return group
        return self._nearest_thin_shape(view_pos)

    def _group_for_selecting(self, view_pos) -> Optional[ShapeGroup]:
        """Ctrl / Shift + click: selected shapes lie on top of everything, so a shape INSIDE a selected one (a plate
        on a selected car) would never be reached — a not-yet-selected shape under the mouse wins; only when there
        is none does the click mean the selected shape itself."""
        first = None
        for item in self.items(view_pos):
            group = self.item_to_group.get(item)
            if group is None:
                continue
            if group is not self.selected_group and group not in self.multi:
                return group
            first = first or group
        thin = self._nearest_thin_shape(view_pos)
        if thin is not None and thin is not self.selected_group and thin not in self.multi:
            return thin
        return first or thin

    def _nearest_thin_shape(self, view_pos) -> Optional[ShapeGroup]:
        """Lines, linestrips and points have almost no area: pick the closest one within the tolerance."""
        scene = self.mapToScene(view_pos)
        tol = self.PICK_TOLERANCE_PX / max(self.current_zoom, 1e-6)     # tolerance in image pixels
        best, best_d = None, tol
        for g in self.shape_groups:
            st = g.shape_data.get("shape_type") or "polygon"
            if st not in ("line", "linestrip", "point", "points"):
                continue
            pts = [QPointF(float(p[0]), float(p[1])) for p in g.shape_data.get("points", [])]
            if not pts:
                continue
            if len(pts) == 1 or st in ("point", "points"):
                d = min(distance(p - scene) for p in pts)
            else:
                d = min(distancetoline(scene, (pts[i], pts[i + 1])) for i in range(len(pts) - 1))
            if d <= best_d:
                best, best_d = g, d
        return best
