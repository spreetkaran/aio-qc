# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import time

import math
from typing import Dict, List, Optional, Set

import numpy as np

from PyQt6.QtCore import QEasingCurve, QLineF, QPointF, QRectF, Qt, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QTransform
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QGraphicsPathItem, QGraphicsPixmapItem,
    QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
)

from AIO_QC.utils import ShapeGroup, distance, distancetoline, is_close, load_font



def _auto_contrast(rgb: np.ndarray) -> np.ndarray:
    """CLAHE on the lightness channel (OpenCV); percentile stretch as a fallback."""
    try:
        import cv2
        lab = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except Exception:
        lo, hi = np.percentile(rgb, (1, 99))
        return np.clip((rgb.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)


GEOMETRY_TYPES = ("rectangle", "polygon", "circle", "line", "linestrip", "point", "points")
MIN_POINTS = {"polygon": 3, "linestrip": 2, "points": 1}
HANDLE_PX = 9


class LabelTag(QGraphicsItem):
    """A shape's class name as a crisp, constant-size pill (like CVAT / Label Studio / Roboflow):

    * drawn in screen pixels (ItemIgnoresTransformations) — the same readable size at every zoom, never blurry,
      never a giant block when zoomed in or unreadable specks when zoomed out;
    * attached to a point ON the shape's outline (a vertex / corner, chosen by Canvas.layout_tags so tags never sit
      on top of each other): the pill's corner touches that point and a small connector dot in the class colour
      marks it, so it is clear at every zoom which outline a label belongs to; when the pill had to move further
      away, a thin leader line runs from the pill to the dot;
    * rounded fill in the class colour, a hairline darker border, text in black or white by contrast, long names
      elided (full name in the tooltip); clicking it selects the shape like clicking the outline.
    """

    PAD_X, PAD_Y, RADIUS = 5.0, 2.0, 4.0
    MAX_W = 220.0

    def __init__(self, text: str, color: QColor, font):
        super().__init__()
        from PyQt6.QtGui import QFontMetricsF
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(False)
        self.font = font
        fm = QFontMetricsF(font)
        full = str(text)
        self.text = fm.elidedText(full, Qt.TextElideMode.ElideRight, self.MAX_W - 2 * self.PAD_X)
        if self.text != full:
            self.setToolTip(full)
        self.text_w = fm.horizontalAdvance(self.text)
        self.ascent = fm.ascent()
        self.w = math.ceil(self.text_w + 2 * self.PAD_X)
        self.h = math.ceil(fm.height() + 2 * self.PAD_Y)
        self.fill = QColor(color)
        self.fill.setAlpha(235)
        self.border = QColor(color).darker(165)
        lum = color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114
        self.ink = QColor(20, 20, 20) if lum > 150 else QColor(255, 255, 255)
        self.off = QPointF(0, -self.h - 1)        # top-left of the pill relative to the anchor, in screen pixels
        self.emphasis = False
        self.connector = False                    # class tags: dot at the anchor (+ leader line when detached)
        self.dot = QColor(color)
        self.anchors = []                         # scene points on the outline, best first (set by the canvas)

    def size(self):
        return self.w, self.h

    def set_offset(self, dx: float, dy: float):
        if abs(self.off.x() - dx) > 0.25 or abs(self.off.y() - dy) > 0.25:
            self.prepareGeometryChange()
            self.off = QPointF(round(dx), round(dy))   # whole pixels: sharp edges and text

    def _pill(self) -> QRectF:
        return QRectF(self.off.x(), self.off.y(), self.w, self.h)

    def _leader(self):
        """Pill edge point nearest the anchor (0, 0), and the distance — None when the pill touches the anchor."""
        r = self._pill()
        nx, ny = min(max(0.0, r.left()), r.right()), min(max(0.0, r.top()), r.bottom())
        d = math.hypot(nx, ny)
        return (QPointF(nx, ny), d) if d > 6 else (None, d)

    def boundingRect(self):
        r = self._pill().adjusted(-1, -1, 1, 1)
        if self.connector:
            r = r.united(QRectF(-4.5, -4.5, 9, 9))
        return r

    def shape(self):
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.off.x(), self.off.y(), self.w, self.h), self.RADIUS, self.RADIUS)
        return path

    def paint(self, p, _opt, _widget=None):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        if self.connector:
            end, _d = self._leader()
            if end is not None:                              # detached pill: a leader back to the outline
                halo = QPen(QColor(0, 0, 0, 120), 3.0)
                halo.setCosmetic(True)
                p.setPen(halo)
                p.drawLine(QPointF(0, 0), end)
                line = QPen(self.dot, 1.4)
                line.setCosmetic(True)
                p.setPen(line)
                p.drawLine(QPointF(0, 0), end)
            ring = QPen(QColor(255, 255, 255, 230) if self.emphasis else self.border, 1.2)
            ring.setCosmetic(True)
            p.setPen(ring)
            p.setBrush(self.dot)
            p.drawEllipse(QPointF(0, 0), 3.2, 3.2)
        r = QRectF(self.off.x() + 0.5, self.off.y() + 0.5, self.w - 1, self.h - 1)
        pen = QPen(self.border if not self.emphasis else QColor(255, 255, 255), 1.0 if not self.emphasis else 1.6)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(self.fill)
        p.drawRoundedRect(r, self.RADIUS, self.RADIUS)
        p.setFont(self.font)
        p.setPen(self.ink)
        p.drawText(QPointF(self.off.x() + self.PAD_X, self.off.y() + self.PAD_Y + self.ascent), self.text)


class Canvas(QGraphicsView):
    """
    Annotation viewer with inline editing.

    Modes
    -----
    DRAG   (H)  pan with left mouse; double-click toggles fit / 100 %
    SELECT (V)  click a shape to select it; Esc clears; hold Space to pan temporarily

    Inline editing (Select mode, when `editable`; the app commits every change to the file — with undo):
      drag a handle                 move that vertex / corner / edge (rectangles stay rectangles, circles keep
                                    their centre)
      drag inside the selected shape  move the whole shape
      arrow keys (Shift = ×10)      nudge the selected shape by one image pixel
      Ctrl+click on an outline      insert a vertex (polygons / linestrips)
      Alt+click on a vertex         remove it (never below the minimum for the shape type)
      Del / Backspace               delete the shape · 1–9 relabel to class n · L or F2 relabel (list)
      B, then drag                  draw a new box
    """
    FIT_MODE, ACTUAL_MODE, CUSTOM_MODE = "fit", "actual", "custom"
    DRAG, SELECT = "drag", "select"
    MIN_ZOOM, MAX_ZOOM = 0.01, 10.0

    zoomFinished = pyqtSignal()
    shapeSelected = pyqtSignal(int)      # index into current shapes list
    selectionCleared = pyqtSignal()
    modeChanged = pyqtSignal(str)
    shapeEdited = pyqtSignal(int, list)          # index, new points (image pixels, as displayed)
    shapeDeleteRequested = pyqtSignal(int)
    relabelRequested = pyqtSignal(int, int)       # index, class number 1-9 (0 = choose from the list)
    boxDrawn = pyqtSignal(QRectF)                 # new box (image pixels)
    editRefused = pyqtSignal(str)                 # why an edit gesture was not possible

    BASE_SHAPE_Z, LABEL_BG_Z, LABEL_TEXT_Z = 10, 100_000, 200_000
    SELECTED_Z = 1e9

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.addItem(self.pixmap_item)

        self.shapes: List[dict] = []
        self.label_colors: Dict[str, QColor] = {}
        self.visible_labels: Set[str] = set()
        self.shape_groups: List[ShapeGroup] = []
        self.item_to_group: Dict[QGraphicsItem, ShapeGroup] = {}
        self.index_to_group: Dict[int, ShapeGroup] = {}
        self.selected_group: Optional[ShapeGroup] = None
        self.hover_group: Optional[ShapeGroup] = None
        self.overlay_items: list = []           # temporary, non-annotation graphics (model predictions)

        self.show_tags = True
        self.fill_shapes = False
        self.mode = self.DRAG
        self._space_pan = False
        self._press_pos = None          # Drag mode: a press + release without movement is a click
        # Display (view-only, never written): image adjustments + annotation styling
        self.display = {"brightness": 0, "contrast": 0, "gamma": 1.0, "auto": False}
        self.fill_alpha = 55             # 0-255, used when "Fill Shapes" is on
        self.outline_scale = 1.0
        self._original = QPixmap()
        self.zoom_mode = self.FIT_MODE
        self.current_zoom = 1.0
        self.fit_scale = 1.0
        self._zoom_anim: Optional[QVariantAnimation] = None
        self._target_zoom = 1.0                 # where the running animation will end
        self._target_center: Optional[QPointF] = None
        self._tags_slow = False                 # tag layout too slow for every glide frame (many tags)
        self._padded = False                    # scene rect temporarily enlarged while gliding

        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        # Photos-style: no scrollbars (they resize the viewport and break "fit"), pan by dragging.
        # Scrollbars stay functional internally, so wheel/trackpad panning still works.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        # All zooming positions the view explicitly (centerOn), so Qt's implicit anchors are off.
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        # ---- inline editing
        self.editable = False                    # set by the app per file (locked / classification -> False)
        self.edit_note = ""                      # why editing is off (shown when a gesture is refused)
        self._handles: list = []                 # handle items of the selected shape
        self._drag_edit = None                   # active edit drag (see _begin_edit)
        self._work_pts = None                    # live points of the selected shape while editing
        self._drawing = False                    # B: next drag draws a box
        self._draw_item = None
        self._draw_start = None
        from PyQt6.QtCore import QTimer
        self._nudge_timer = QTimer(self, singleShot=True, interval=380, timeout=self._commit_nudge)
        self._apply_mode_cursor()

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
            self.pixmap_item.setPixmap(self._original)
            return
        img = self._original.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = img.width(), img.height()
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        arr = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine())[:, : w * 4].reshape(h, w, 4)
        rgb = arr[:, :, :3]
        d = self.display
        if d["auto"]:
            rgb = _auto_contrast(rgb)
        lut = np.arange(256, dtype=np.float32) / 255.0
        lut = np.power(lut, 1.0 / max(d["gamma"], 1e-3))                   # gamma
        c = d["contrast"] * 1.28                                              # -128..128
        f = (259 * (c + 255)) / (255 * (259 - c))
        lut = f * (lut * 255 - 128) + 128 + d["brightness"] * 1.28          # contrast around mid-grey, brightness
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        out = np.dstack([lut[rgb], arr[:, :, 3:4]])
        out = np.ascontiguousarray(out)
        qi = QImage(out.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        self.pixmap_item.setPixmap(QPixmap.fromImage(qi))

    def set_annotation_style(self, fill_alpha: int | None = None, outline_scale: float | None = None,
                             tag_scale: float | None = None):
        if tag_scale is not None:
            self.tag_scale = float(tag_scale)
        if fill_alpha is not None:
            self.fill_alpha = int(fill_alpha)
        if outline_scale is not None:
            self.outline_scale = float(outline_scale)
        self.drawShapes()

    def loadPixmap(self, pixmap: QPixmap):
        self.clear_overlay()
        self.clearShapes(emit_signal=True)
        self._original = pixmap
        self.pixmap_item.setPixmap(pixmap)
        if not self.display_is_neutral():
            self._render_display()
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.apply_default_zoom()               # synchronous: no flash of the previous zoom
        if self._panning():
            self._pan_cursor()                  # the image under the mouse may have changed size

    def clearAll(self):
        self._original = QPixmap()
        self.clear_overlay()
        self.clearShapes(emit_signal=True)
        self.pixmap_item.setPixmap(QPixmap())
        if self._panning():
            self._pan_cursor()
        self.zoomFinished.emit()                  # no image: Fit / 100 % buttons un-press

    def loadShapes(self, shapes: List[dict], label_colors: Dict, visible_labels: Set):
        prev = self.selected_group.index if self.selected_group else None
        self.shapes, self.label_colors, self.visible_labels = shapes or [], label_colors, visible_labels
        self.drawShapes()
        if prev is not None and prev in self.index_to_group:
            self.select_shape(prev, emit=False)

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
        self.hover_group = None
        self.shape_groups.clear()
        self.item_to_group.clear()
        self.index_to_group.clear()
        if emit_signal and had_selection:
            self.selectionCleared.emit()

    def _base_scale(self) -> float:
        r = self.pixmap_item.boundingRect()
        return max(min(r.width(), r.height()) / 800.0, 0.25)

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
        if st in ("linestrip", "point", "points") or len(points) < 3:
            path = QPainterPath(points[0])
            for p in points[1:]:
                path.lineTo(p)
            if len(points) == 1:
                rr = 3 * self._base_scale()
                path.addEllipse(points[0], rr, rr)
            return QGraphicsPathItem(path)
        return QGraphicsPolygonItem(QPolygonF(points))

    def drawShapes(self):
        self.clearShapes()
        if not self.shapes or self.pixmap_item.pixmap().isNull():
            return
        base = self._base_scale()
        font = self._tag_font()
        pen_w = max(1.0, 2.0 * base) * self.outline_scale

        order = sorted(range(len(self.shapes)), key=lambda i: self._shape_priority(self.shapes[i]))
        for z, idx in enumerate(order):
            shape = self.shapes[idx]
            label = str(shape.get("label", "?"))
            if label not in self.visible_labels:
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

    # ---- class-name tags ---------------------------------------------------------------------------------------
    TAG_PX = 12                                        # tag text size in screen pixels (View ▸ Label Size scales it)

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

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        if self.shape_groups:
            self._schedule_tag_layout()

    @staticmethod
    def _tooltip(idx: int, shape: dict) -> str:
        pts = shape.get("points") or []
        try:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            size = f"{max(xs) - min(xs):.1f} × {max(ys) - min(ys):.1f}"
        except (TypeError, ValueError, IndexError):
            size = "?"
        return f"#{idx}  {shape.get('label')}\n{shape.get('shape_type', 'polygon')} · {len(pts)} pts · {size}"

    # ------------------------------------------------------------- selection
    def select_shape(self, index: int, emit: bool = True) -> bool:
        """Selecting is a Select-mode action: from Drag mode it switches to Select first (modeChanged tells the UI)."""
        group = self.index_to_group.get(index)
        if group is None:
            return False
        if self.mode == self.DRAG:
            self.setMode(self.SELECT)
        if self.selected_group is not group:
            self._flush_nudge()
            if self.selected_group is not None:
                self.selected_group.set_selected(False)
            self.selected_group = group
            group.set_selected(True, self.SELECTED_Z)
            self._emphasize_tag(group)
        self._update_handles()
        if emit:
            self.shapeSelected.emit(group.index)
        return True

    def clearSelection(self):
        if self.selected_group is None:
            return
        self._flush_nudge()
        self.selected_group.set_selected(False)
        self.selected_group = None
        self._emphasize_tag(None)
        self._clear_handles()
        self.selectionCleared.emit()

    def focus_shape(self, index: int):
        """Select a shape and glide to it so it fills ~60 % of the view."""
        if not self.select_shape(index):
            return
        rect = self.selected_group.shape_item.sceneBoundingRect()
        if rect.isEmpty():
            self.animate_to(self.current_zoom, center=rect.center())
            return
        m = max(rect.width(), rect.height()) * 0.35
        rect = rect.adjusted(-m, -m, m, m)
        vp = self._viewport_size()
        z = min(vp.width() / rect.width(), vp.height() / rect.height())
        self.animate_to(z, center=rect.center(), mode=self.CUSTOM_MODE, duration=320)

    # ---------------------------------------------------------------- overlay
    def set_overlay(self, boxes: list[dict]):
        """
        Dashed boxes that are NOT annotations, e.g. model predictions:
        [{"box": (x1, y1, x2, y2), "text": "plate 0.92", "color": "#ff00ff"}]
        Cleared automatically when another image is loaded.
        """
        self.clear_overlay()
        base = self._base_scale()
        tag_font = self._tag_font()
        for b in boxes:
            x1, y1, x2, y2 = map(float, b["box"])
            color = QColor(b.get("color", "#ff00ff"))
            rect = QGraphicsRectItem(QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized())
            pen = QPen(color, max(1.5, 2.5 * base), Qt.PenStyle.DashLine)
            pen.setCosmetic(False)
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

    PICK_TOLERANCE_PX = 6     # screen pixels: thin shapes are selectable without pixel-perfect aim

    def _group_at(self, view_pos) -> Optional[ShapeGroup]:
        for item in self.items(view_pos):  # topmost first
            group = self.item_to_group.get(item)
            if group is not None:
                return group
        return self._nearest_thin_shape(view_pos)

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

    # ------------------------------------------------------------------ zoom
    # ------------------------------------------------------------------ zoom
    # Model: the view is fully described by (zoom, scene point at viewport centre).
    # Every zoom — double-click, wheel, spinbox, fit, focus — goes through animate_to(),
    # which sets the transform ABSOLUTELY each frame (no accumulated float error) and
    # interpolates zoom geometrically (constant perceived speed, like Windows Photos).

    def _viewport_size(self):
        # Size WITHOUT scrollbars/frame -> stable fit scale regardless of the current zoom.
        return self.maximumViewportSize()

    def compute_fit_scale(self) -> float:
        pr = self.pixmap_item.pixmap().rect()
        if pr.isNull():
            return 1.0
        vp = self._viewport_size()
        return min(vp.width() / pr.width(), vp.height() / pr.height())

    def home_zoom(self) -> float:
        """Where an image opens: fit to window, but never upscale beyond 100 %."""
        self.fit_scale = self.compute_fit_scale()
        return min(self.fit_scale, 1.0)

    def detail_zoom(self, home: float) -> float:
        """Double-click target from home: 100 %, or 2x home when 100 % would barely change anything."""
        return 1.0 if home <= 1 / 1.5 else min(home * 2.0, self.MAX_ZOOM)

    def _image_center(self) -> QPointF:
        return self.pixmap_item.sceneBoundingRect().center()

    def _view_center(self) -> QPointF:
        return self.mapToScene(self.viewport().rect().center())

    def _apply_view(self, zoom: float, center: QPointF, animating: bool = False):
        self.setTransform(QTransform.fromScale(zoom, zoom))
        self.current_zoom = zoom
        self.centerOn(center)
        if self.shape_groups:                     # tags keep their screen size: re-place them for this zoom
            if animating and self._tags_slow:
                return                            # many tags: placed once the glide lands (keeps it fluid)
            if len(self.shape_groups) <= 400:
                t0 = time.perf_counter()
                self.layout_tags()                # every frame while zooming: no tag ever slides over another
                if animating and time.perf_counter() - t0 > 0.004:
                    self._tags_slow = True        # this frame was too costly: skip until the glide lands
            else:
                self._schedule_tag_layout()

    def _set_smooth(self, zoom: float, animating: bool):
        # Smooth (bilinear) whenever it is affordable — also while gliding: nearest-neighbour shimmers while the
        # scale changes, which reads as jitter. Very large images glide with the fast filter. Crisp pixels when
        # pixel-peeping (>= 300 %) once settled.
        pm = self.pixmap_item.pixmap()
        big = pm.width() * pm.height() > 12_000_000
        smooth = (not big if animating else True) and zoom < 3.0
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation if smooth
                                               else Qt.TransformationMode.FastTransformation)

    def is_animating(self) -> bool:
        return self._zoom_anim is not None

    def stop_zoom_animation(self):
        anim, self._zoom_anim = self._zoom_anim, None
        if anim is not None:
            anim.stop()
            anim.deleteLater()
            self._unpad_scene()

    # ---- geometry helpers (scene = image pixels)
    def _image_rect(self) -> QRectF:
        return QRectF(self.pixmap_item.pixmap().rect())

    def clamp_center(self, zoom: float, c: QPointF) -> QPointF:
        """Where the view centre can really be at this zoom: an axis the image fills is kept inside the image (no
        empty band at an edge), an axis it does not fill is centred — exactly what Qt shows once settled."""
        r = self._image_rect()
        vp = self._viewport_size()
        hw, hh = vp.width() / 2 / zoom, vp.height() / 2 / zoom
        x = r.center().x() if r.width() <= 2 * hw else min(max(c.x(), r.left() + hw), r.right() - hw)
        y = r.center().y() if r.height() <= 2 * hh else min(max(c.y(), r.top() + hh), r.bottom() - hh)
        return QPointF(x, y)

    def _pad_scene(self, zoom_min: float):
        """While gliding, the view may pass through positions Qt would clamp or re-centre (that is what made the
        zoom bend towards an edge first). A scene rect larger than any position the glide visits lets every frame
        land exactly where the path says; the real rect is back before the last frame."""
        vp = self._viewport_size()
        m = max(vp.width(), vp.height()) / max(zoom_min, 1e-6)
        self.scene.setSceneRect(self._image_rect().adjusted(-m, -m, m, m))
        self._padded = True

    def _unpad_scene(self):
        if getattr(self, "_padded", False):
            self._padded = False
            self.scene.setSceneRect(self._image_rect())

    def animate_to(self, zoom: float, *, center: Optional[QPointF] = None, anchor_scene: Optional[QPointF] = None,
                   anchor_view=None, mode: Optional[str] = None, duration: int = 260,
                   easing=QEasingCurve.Type.OutCubic):
        """
        Glide to `zoom`.
          center=P                     -> P ends at the viewport centre (fit, focus shape)
          anchor_scene=P, anchor_view=V -> P stays under viewport position V (zoom at the cursor / click)

        Windows-Photos feel:
          * the END view is decided first and clamped like a settled view (no empty band at an image edge), so the
            glide never has to correct itself at the end;
          * the glide is a pure zoom about the one image point that is at the same screen spot at the start and at
            the end — every point travels in a straight line, nothing bends towards an edge;
          * the zoom grows geometrically (constant perceived speed) and every frame is set absolutely.
        """
        if self.pixmap_item.pixmap().isNull():
            return
        zoom = max(self.MIN_ZOOM, min(zoom, self.MAX_ZOOM))
        z0, c0 = self.current_zoom, self._view_center()
        self.stop_zoom_animation()
        vc = QPointF(self.viewport().rect().center())
        if anchor_scene is not None:
            r = self._image_rect()
            a = QPointF(min(max(anchor_scene.x(), r.left()), r.right()),      # a click on the letterbox band zooms
                        min(max(anchor_scene.y(), r.top()), r.bottom()))      # into the nearest image edge
            want = a - (QPointF(anchor_view) - vc) / zoom
        else:
            want = center if center is not None else c0
        c1 = self.clamp_center(zoom, want)
        self._target_zoom = zoom
        self._target_center = c1
        final_mode = mode or self.CUSTOM_MODE
        if duration <= 0 or (is_close(z0, zoom, abs_tol=1e-4) and QLineF(c0, c1).length() * zoom < 0.5):
            self._apply_view(zoom, c1)
            self._finish_zoom(final_mode)
            return
        # the fixed point p* of the move: z0 (p* − c0) = z1 (p* − c1)
        if abs(zoom - z0) > 1e-9 * max(zoom, z0):
            fixed = (c1 * zoom - c0 * z0) / (zoom - z0)

            def center_at(z, _e):
                return fixed - (fixed - c0) * (z0 / z)
        else:                                                 # same zoom: a straight pan

            def center_at(_z, e):
                return c0 + (c1 - c0) * e
        ratio = zoom / z0
        anim = QVariantAnimation(self)
        anim.setDuration(duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(easing)
        self._tags_slow = False
        self._set_smooth(zoom, True)
        self._pad_scene(min(z0, zoom))

        def step(e):
            e = float(e)
            z = z0 * ratio ** e                              # geometric interpolation
            self._apply_view(z, center_at(z, e), animating=True)

        def done():
            if self._zoom_anim is anim:
                self._zoom_anim = None
                anim.deleteLater()
                self._unpad_scene()
                self._tags_slow = False
                self._apply_view(zoom, c1)                   # land exactly on the (clamped) target
                self._finish_zoom(final_mode)

        anim.valueChanged.connect(step)
        anim.finished.connect(done)
        self._zoom_anim = anim
        anim.start()

    def _finish_zoom(self, mode: str):
        self.zoom_mode = mode
        self._target_zoom = self.current_zoom
        self._set_smooth(self.current_zoom, False)
        self.zoomFinished.emit()

    # public zoom API ------------------------------------------------------------
    def apply_default_zoom(self):
        """Instant 'home' view for a newly loaded image, centred."""
        if self.pixmap_item.pixmap().isNull():
            return
        self.stop_zoom_animation()
        home = self.home_zoom()
        self._apply_view(home, self._image_center())
        self._finish_zoom(self.FIT_MODE if home < 1.0 else self.ACTUAL_MODE)

    def fit_window(self):
        self.fit_scale = self.compute_fit_scale()
        self.animate_to(self.fit_scale, center=self._image_center(), mode=self.FIT_MODE)

    def actual_size(self):
        self.animate_to(1.0, anchor_scene=self._view_center(), anchor_view=self.viewport().rect().center(),
                        mode=self.ACTUAL_MODE)

    def zoom_to(self, zoom: float):
        """Spinbox: zoom around the viewport centre."""
        self.animate_to(zoom, anchor_scene=self._view_center(), anchor_view=self.viewport().rect().center(),
                        duration=180)

    def zoom_by(self, factor: float, view_pos):
        """Ctrl+wheel. Consecutive ticks accumulate onto the running target instead of restarting."""
        base = self._target_zoom if self.is_animating() else self.current_zoom
        self.animate_to(base * factor, anchor_scene=self.mapToScene(view_pos), anchor_view=view_pos,
                        duration=140)

    def toggle_zoom(self, view_pos):
        """
        Windows-Photos double-click: exactly two states.
          at home (fit / 100 % for small images) -> detail zoom, anchored at the click
          anywhere else                          -> back to home, centred
        Uses the animation *target*, so double-clicking mid-animation still toggles correctly.
        """
        home = self.home_zoom()
        z = self._target_zoom if self.is_animating() else self.current_zoom
        if abs(z / home - 1.0) < 0.02:
            detail = self.detail_zoom(home)
            self.animate_to(detail, anchor_scene=self.mapToScene(view_pos), anchor_view=view_pos,
                            mode=self.ACTUAL_MODE if detail == 1.0 else self.CUSTOM_MODE, duration=280)
        else:
            self.animate_to(home, center=self._image_center(),
                            mode=self.FIT_MODE if home < 1.0 else self.ACTUAL_MODE, duration=280)

    # ---------------------------------------------------------------- events
    def outOfPixmap(self, scene_pos: QPointF) -> bool:
        if self.pixmap_item.pixmap().isNull():
            return True
        return not self.pixmap_item.sceneBoundingRect().contains(scene_pos)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.shape_groups:
            self._schedule_tag_layout()
        if self.pixmap_item.pixmap().isNull() or self.is_animating():
            return
        if self.zoom_mode == self.FIT_MODE:
            home = self.home_zoom()
            self._apply_view(home, self._image_center())
            self.zoom_mode = self.FIT_MODE if home < 1.0 else self.ACTUAL_MODE
            self.zoomFinished.emit()
        else:
            self.zoomFinished.emit()              # the fit scale changed: listeners re-check "fitted / 100 %"

    def focusNextPrevChild(self, next_):
        # Keep Tab / Shift+Tab for shape cycling while an image is shown
        if self.shape_groups:
            return False
        return super().focusNextPrevChild(next_)

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

    def keyPressEvent(self, event):
        if self._edit_key(event):
            return
        if event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            back = event.key() == Qt.Key.Key_Backtab or bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.cycle_shape(-1 if back else 1)
            return
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat() and self.mode == self.SELECT:
            self._space_pan = True
            self._apply_mode_cursor()
            return
        if event.key() == Qt.Key.Key_Escape:
            if self._drawing:
                self._stop_drawing()
                return
            self.clearSelection()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat() and self._space_pan:
            self._space_pan = False
            self._apply_mode_cursor()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            if self._drawing and not self._space_pan:
                self._begin_draw(event.pos())
                event.accept()
                return
            if not self._panning() and self._begin_edit(event):
                event.accept()
                return
            if self._panning():
                self._press_pos = event.pos()
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                group = self._group_at(event.pos())
                if group is not None:
                    self.select_shape(group.index)
                else:
                    self.clearSelection()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._draw_start is not None:
            self._end_draw(event.pos())
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drag_edit is not None:
            self._end_edit()
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            press, self._press_pos = self._press_pos, None
            # Drag mode is for moving around: a click that did not pan only clears a selection (never selects)
            if press is not None and self.mode == self.DRAG and not self._space_pan \
                    and (event.pos() - press).manhattanLength() < 4:
                self.clearSelection()
            self._apply_mode_cursor()

    def _set_hover(self, group: Optional[ShapeGroup]):
        if group is self.hover_group:
            return
        if self.hover_group is not None and self.hover_group in self.shape_groups:
            self.hover_group.set_hover(False)
        self.hover_group = group
        if group is not None:
            group.set_hover(True)

    def mouseMoveEvent(self, event):
        if self._draw_start is not None:
            self._move_draw(event.pos())
            return
        if self._drag_edit is not None:
            self._move_edit(event.pos())
            return
        super().mouseMoveEvent(event)
        if self._drawing:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            return
        if not self._panning() and event.buttons() == Qt.MouseButton.NoButton:
            h = self._handle_at(event.pos())
            if h is not None:
                self.viewport().setCursor(self._handle_cursor(h))
                return
            hover = self._group_at(event.pos())
            self._set_hover(hover)
            if hover is not None and hover is self.selected_group and self._geometry_editable(hover):
                self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.viewport().setCursor(Qt.CursorShape.PointingHandCursor if hover else Qt.CursorShape.ArrowCursor)
        elif self._panning():
            self._set_hover(None)
            self._pan_cursor(event.pos())

    def leaveEvent(self, event):
        self._set_hover(None)
        super().leaveEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if delta.y():
                self.zoom_by(1.2 ** (delta.y() / 120), event.position().toPoint())
            if self._panning():
                self._pan_cursor(event.position().toPoint())
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.y())
        else:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        if self._panning():
            self._pan_cursor(event.position().toPoint())     # zoom / scroll moved the image under the mouse
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self.pixmap_item.pixmap().isNull():
            return super().mouseDoubleClickEvent(event)
        if self.mode == self.SELECT and not self._space_pan:
            group = self._group_at(event.pos())
            if group is not None:
                self.focus_shape(group.index)
                return
        # Works anywhere in the view (also on the letterbox area), like Windows Photos.
        self.toggle_zoom(event.pos())
        event.accept()

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
            h.setBrush(QBrush(QColor("white") if role != "e" else col.lighter(160)))
            pen = QPen(col.darker(130), 1.5)
            pen.setCosmetic(True)
            h.setPen(pen)
            h.setZValue(self.SELECTED_Z + 10)
            h.setData(0, (role, i))
            self.scene.addItem(h)
            self._handles.append(h)

    _DIR_CURSORS = (Qt.CursorShape.SizeHorCursor, Qt.CursorShape.SizeFDiagCursor, Qt.CursorShape.SizeVerCursor,
                    Qt.CursorShape.SizeBDiagCursor)

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
        st = self._stype(group.shape_data)
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
        elif isinstance(item, QGraphicsPathItem) and q:
            path = QPainterPath(q[0])
            for p in q[1:]:
                path.lineTo(p)
            if len(q) == 1 or st in ("point", "points"):
                rr = 3 * self._base_scale()
                for p in q:
                    path.addEllipse(p, rr, rr)
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
        if self._group_at(pos) is not g and not (mods & Qt.KeyboardModifier.ControlModifier):
            return False
        if mods & Qt.KeyboardModifier.ControlModifier:
            if self._geometry_editable(g) and self._insert_vertex(g, self.mapToScene(pos)):
                return True
            return self._group_at(pos) is g
        if not self._geometry_editable(g):
            if self.editable is False and self.edit_note:
                self.editRefused.emit(self.edit_note)
            return False
        self._drag_edit = {"kind": "move", "i": -1, "group": g, "pts0": self._shape_pts(g),
                           "start": self.mapToScene(pos), "moved": False, "press": pos}
        return True

    def _move_edit(self, view_pos):
        d = self._drag_edit
        if not d["moved"] and (view_pos - d["press"]).manhattanLength() < 4:
            return                                         # a click, not a drag: never move by accident
        d["moved"] = True
        g, pts0, kind, i = d["group"], d["pts0"], d["kind"], d["i"]
        cur = self.mapToScene(view_pos)
        st = self._stype(g.shape_data)
        if kind == "move":
            pts = self._moved(pts0, cur.x() - d["start"].x(), cur.y() - d["start"].y())
        else:
            x, y = self._clamp(cur.x(), cur.y())
            pts = [list(p) for p in pts0]
            if st == "rectangle":
                (x1, y1), (x2, y2) = pts0
                xa, xb, ya, yb = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
                if kind == "c":
                    xa, ya, xb, yb = {0: (x, y, xb, yb), 1: (xa, y, x, yb), 2: (xa, ya, x, y), 3: (x, ya, xb, y)}[i]
                else:
                    xa, ya, xb, yb = {0: (xa, y, xb, yb), 1: (xa, ya, x, yb), 2: (xa, ya, xb, y), 3: (x, ya, xb, yb)}[i]
                pts = [[xa, ya], [xb, yb]]
            elif st == "circle" and kind == "v":           # centre handle moves the circle
                pts = self._moved(pts0, cur.x() - pts0[0][0], cur.y() - pts0[0][1])
            else:
                pts[i] = [x, y]
        self._work_pts = pts
        self._set_item_points(g, pts)
        self._update_handles(pts)

    def _end_edit(self):
        d, self._drag_edit = self._drag_edit, None
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

    def _insert_vertex(self, g, scene_pos) -> bool:
        st = self._stype(g.shape_data)
        if st not in ("polygon", "linestrip"):
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
        if key == Qt.Key.Key_B and plain and self.editable:
            self._start_drawing()
            return True
        g = self.selected_group
        if g is None or self.mode != self.SELECT:
            return False
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and plain:
            if self.editable:
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
            if not self._geometry_editable(g):
                return False
            step = 10.0 if mods & Qt.KeyboardModifier.ShiftModifier else 1.0
            dx, dy = arrows[key]
            base = self._work_pts if self._work_pts is not None else self._shape_pts(g)
            if not hasattr(self, "_nudge_from") or self._nudge_from is None:
                self._nudge_from = (g, self._shape_pts(g))
            self._work_pts = self._moved(base, dx * step, dy * step)
            self._set_item_points(g, self._work_pts)
            self._update_handles(self._work_pts)
            self._nudge_timer.start()                     # one write after the keys stop, not one per press
            return True
        return False

    def _commit_nudge(self):
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
    def _start_drawing(self):
        if self.pixmap_item.pixmap().isNull():
            return
        if self.mode == self.DRAG:
            self.setMode(self.SELECT)
        self._drawing = True
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def _stop_drawing(self):
        self._drawing = False
        self._draw_start = None
        if self._draw_item is not None and self._draw_item.scene() is self.scene:
            self.scene.removeItem(self._draw_item)
        self._draw_item = None
        self._apply_mode_cursor()

    def _begin_draw(self, view_pos):
        p = self.mapToScene(view_pos)
        self._draw_start = QPointF(*self._clamp(p.x(), p.y()))
        self._draw_item = QGraphicsRectItem(QRectF(self._draw_start, self._draw_start))
        pen = QPen(QColor("#ffd400"), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._draw_item.setPen(pen)
        self._draw_item.setZValue(self.SELECTED_Z + 20)
        self.scene.addItem(self._draw_item)

    def _move_draw(self, view_pos):
        p = self.mapToScene(view_pos)
        q = QPointF(*self._clamp(p.x(), p.y()))
        self._draw_item.setRect(QRectF(self._draw_start, q).normalized())

    def _end_draw(self, view_pos):
        rect = self._draw_item.rect() if self._draw_item is not None else QRectF()
        self._stop_drawing()
        if rect.width() * self.current_zoom >= 4 and rect.height() * self.current_zoom >= 4:
            self.boxDrawn.emit(rect)
