# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations


from typing import Dict, List, Optional, Set


from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import (QColor, QKeySequence, QPainter, QPixmap)
from PyQt6.QtWidgets import (
    QGraphicsItem, QGraphicsScene, QGraphicsView,
)

from AIO_QC.core.ui_yield import interacting
from AIO_QC.utils import ShapeGroup




from AIO_QC.widgets.canvas_common import (  # noqa: F401  (re-exported)
    DisplayRenderer, GEOMETRY_TYPES, HANDLE_PX, ImageItem, LabelTag, MIN_POINTS, PointsItem, _DrawGuide, _DrawKeys, _GC, _Glide, _MAC, _REFINER, _Refiner, _auto_contrast, _gc_pause, _is_oom, _qimage_array, _refiner, display_lut)
from AIO_QC.widgets.canvas_parts.display import DisplayMixin
from AIO_QC.widgets.canvas_parts.selection import SelectionMixin
from AIO_QC.widgets.canvas_parts.overlay import OverlayMixin
from AIO_QC.widgets.canvas_parts.zoom import ZoomMixin
from AIO_QC.widgets.canvas_parts.input import InputMixin
from AIO_QC.widgets.canvas_parts.editing import EditingMixin


class Canvas(DisplayMixin, SelectionMixin, OverlayMixin, ZoomMixin, InputMixin, EditingMixin, QGraphicsView):
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
      Ctrl+click (⌘ on macOS)       add a shape to / take it out of the selection · Shift+click: add it
      Ctrl+A                        select every shape (when one is selected) — several selected shapes are moved
                                    (drag, arrows), deleted, relabelled, copied (Ctrl+C / X / V: the app) together
      Ctrl+R, then drag             draw a new box (Ctrl+N: a polygon, corner by corner)
    """
    FIT_MODE, ACTUAL_MODE, CUSTOM_MODE = "fit", "actual", "custom"
    DRAG, SELECT = "drag", "select"
    MIN_ZOOM, MAX_ZOOM = 0.01, 10.0

    zoomFinished = pyqtSignal()
    zoomChanged = pyqtSignal(float)               # every frame of a glide too (zoom box / slider follow live)
    shapeSelected = pyqtSignal(int)      # index into current shapes list
    selectionCleared = pyqtSignal()
    modeChanged = pyqtSignal(str)
    memoryLow = pyqtSignal(str)          # an image buffer could not be allocated (what was skipped)
    shapeEdited = pyqtSignal(int, list)          # index, new points (image pixels, as displayed)
    shapeDeleteRequested = pyqtSignal(int)
    relabelRequested = pyqtSignal(int, int)       # index, class number 1-9 (0 = choose from the list)
    boxDrawn = pyqtSignal(QRectF)                 # new box (image pixels)
    polygonDrawn = pyqtSignal(list)              # new polygon: [[x, y], …] (image pixels)
    imageMenuRequested = pyqtSignal(QPoint)      # right-click inside the image (viewport position)
    editRefused = pyqtSignal(str)                 # why an edit gesture was not possible
    activeHandleChanged = pyqtSignal(bool)        # the arrow keys move one handle (True) or the whole shape
    cursorMoved = pyqtSignal(object)              # image point under the mouse (QPointF) or None (status bar)
    selectionRect = pyqtSignal(object)            # bounds of the selection / box being drawn (QRectF) or None
    selectionChanged = pyqtSignal(list)           # every selected shape index (sorted) — after any change
    shapesDeleteRequested = pyqtSignal(list)      # several selected shapes: their indices
    shapesMoved = pyqtSignal(list)                # several shapes moved together: [(index, new points), …]
    suggestionClicked = pyqtSignal(int)           # Model Assist: a suggestion clicked (accept it)
    suggestionDismissed = pyqtSignal(int)         # … right-clicked (drop it)

    BASE_SHAPE_Z, LABEL_BG_Z, LABEL_TEXT_Z = 10, 100_000, 200_000
    SELECTED_Z = 1e9

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = ImageItem(self)       # Windows-Photos-quality rendering once the view rests
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.addItem(self.pixmap_item)

        self.shapes: List[dict] = []
        self.label_colors: Dict[str, QColor] = {}
        self.visible_labels: Set[str] = set()
        self.shape_groups: List[ShapeGroup] = []
        self.item_to_group: Dict[QGraphicsItem, ShapeGroup] = {}
        self.index_to_group: Dict[int, ShapeGroup] = {}
        self.selected_group: Optional[ShapeGroup] = None     # the primary selection (handles, single edits)
        self.multi: List[ShapeGroup] = []                    # the other selected shapes (Ctrl / Shift + click)
        self._multi_work: dict = {}                          # group -> live points while several are moved
        self.hover_group: Optional[ShapeGroup] = None
        self.overlay_items: list = []           # temporary, non-annotation graphics (model predictions)
        self.peek_index = None                  # a hidden-label shape shown while it is selected (see select_shape)
        self.pred_items: list = []              # the persistent predictions layer (View ▸ Model Predictions)
        self.suggestions: list = []             # Model Assist: [(item, tag, payload)] — click to accept

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
        self._renderer = DisplayRenderer()
        self.keep_zoom = False                  # View ▸ Keep Previous Zoom
        self._kept_view: Optional[dict] = None  # what the user zoomed to (carried to the next images)
        self._applied_view = None               # (zoom, centre) this canvas set itself for the current image
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
        self._active_handle = None               # (kind, i): the handle the arrow keys move (a click on it)
        self._work_pts = None                    # live points of the selected shape while editing
        self._drawing = False                    # Ctrl+R / Ctrl+N: the next drag draws a box, clicks a polygon
        self._draw_kind = "box"                   # box (drag) · polygon (click the corners)
        self._poly_pts: list = []
        self._poly_redo: list = []                # corners taken back with Ctrl+Z (Ctrl+Shift+Z / Ctrl+Y re-adds)
        self._poly_item = None                    # the drawing guide (_DrawGuide) while drawing
        self._poly_cursor = None
        self._poly_snap = False                   # the pointer is on the first corner: a click closes
        self._poly_mouse = None                   # last pointer position (view), for Shift
        self._draw_hint = None                    # the hint chip over the image
        self._draw_keys = None                    # _DrawKeys while drawing
        self._draw_item = None
        self._draw_start = None
        from PyQt6.QtCore import QTimer
        self._nudge_timer = QTimer(self, singleShot=True, interval=380, timeout=self._commit_nudge)
        self._apply_mode_cursor()

    # ---- class-name tags ---------------------------------------------------------------------------------------
    TAG_PX = 12                                        # tag text size in screen pixels (View ▸ Label Size scales it)

    def scrollContentsBy(self, dx, dy):
        interacting(0.25)                                   # panning: background loops wait
        super().scrollContentsBy(dx, dy)
        if not self.is_animating():
            self.pixmap_item.timer.start()
        if self.shape_groups:
            self._schedule_tag_layout()

    PICK_TOLERANCE_PX = 6     # screen pixels: thin shapes are selectable without pixel-perfect aim

    _gpu = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._drawing:
            self._update_hint()
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

    def keyPressEvent(self, event):
        if self._drawing:
            k = event.key()
            if k == Qt.Key.Key_Escape:
                self.cancel_drawing()
                return
            if self._draw_kind == "polygon":
                if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    self._finish_polygon()
                    return
                if (event.matches(QKeySequence.StandardKey.Undo)
                        or k in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete)):
                    self.undo_corner()
                    return
                if event.matches(QKeySequence.StandardKey.Redo):
                    self.redo_corner()
                    return
                if k == Qt.Key.Key_Shift:
                    self._poly_follow()
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
            self.escape()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self._drawing and self._draw_kind == "polygon" and event.key() == Qt.Key.Key_Shift:
            self._poly_follow(Qt.KeyboardModifier.NoModifier)
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat() and self._space_pan:
            self._space_pan = False
            self._apply_mode_cursor()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            # Hover is pre-click feedback only: a press starts a gesture (draw a box, drag a handle, pan, click),
            # so the shape under the mouse stops being highlighted until the button is released and it moves again.
            self._set_hover(None)
            if self._drawing and not self._space_pan:
                if self._draw_kind == "polygon":
                    self._add_poly_point(event.pos(), event.modifiers())
                else:
                    self._begin_draw(event.pos())
                event.accept()
                return
            if self.suggestions and not self._panning():
                i = self.suggestion_at(event.pos())
                if i is not None:
                    self.suggestionClicked.emit(i)
                    event.accept()
                    return
            if self._modifier_select(event):
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
                mods = event.modifiers()
                ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)      # ⌘ on macOS (Qt maps it)
                shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
                if (ctrl or shift) and self.mode == self.SELECT:
                    group = self._group_for_selecting(event.pos())
                    if group is not None:                 # a miss keeps the selection (no accidental clearing)
                        self.toggle_shape(group.index) if ctrl else self.add_shape(group.index)
                elif group is not None:
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

    def mouseMoveEvent(self, event):
        self.report_cursor(event.pos())
        if self._draw_start is not None:
            self._move_draw(event.pos())
            return
        if self._drag_edit is not None:
            self._move_edit(event.pos())
            return
        super().mouseMoveEvent(event)
        if self._drawing:                          # placing a new box: shapes underneath are not targets
            self._set_hover(None)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self._poly_mouse = event.pos()
            self._poly_follow(event.modifiers())
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

    def contextMenuEvent(self, event):
        """Right-click: a menu only inside the image, and never in the middle of a gesture."""
        pos = event.pos()
        if self._drawing or self._draw_start is not None or self._drag_edit is not None or not self._over_image(pos):
            event.ignore()
            return
        self._set_hover(None)
        if self.suggestions:
            i = self.suggestion_at(pos)
            if i is not None:                      # right-click on a suggestion: drop it
                self.suggestionDismissed.emit(i)
                event.accept()
                return
        self.imageMenuRequested.emit(pos)
        event.accept()

    def leaveEvent(self, event):
        self._set_hover(None)
        self.cursorMoved.emit(None)
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
        if self._drawing and self._draw_kind == "polygon":
            self._finish_polygon()                 # double-click: the last corner, and done
            event.accept()
            return
        if self.mode == self.SELECT and not self._space_pan:
            group = self._group_at(event.pos())
            if group is not None:
                self.focus_shape(group.index)
                return
        # Works anywhere in the view (also on the letterbox area), like Windows Photos.
        self.toggle_zoom(event.pos())
        event.accept()

    _DIR_CURSORS = (Qt.CursorShape.SizeHorCursor, Qt.CursorShape.SizeFDiagCursor, Qt.CursorShape.SizeVerCursor,
                    Qt.CursorShape.SizeBDiagCursor)

    # ---- new polygon -------------------------------------------------------------------------------------------
    SNAP_PX = 12                                                    # how near the first corner closes (screen px)
