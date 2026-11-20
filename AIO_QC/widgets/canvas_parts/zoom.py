# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image QC canvas — zoom and panning.

A mixin of AIO_QC.widgets.canvas.Canvas: the methods use the window's own attributes, set up in its __init__.
"""
from __future__ import annotations
import time
from typing import Optional
from PyQt6.QtCore import QEasingCurve, QLineF, QPointF, QRectF, Qt
from PyQt6.QtGui import (QPainter, QTransform)
from PyQt6.QtWidgets import (
    QGraphicsView,
)
from AIO_QC.core.ui_yield import interacting
from AIO_QC.utils import is_close

from AIO_QC.widgets.canvas_common import (PointsItem, _Glide, _gc_pause)


class ZoomMixin:

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
        self.zoomChanged.emit(zoom)
        for g in self.shape_groups:               # point dots keep their screen size: new scene bounds
            if isinstance(g.shape_item, PointsItem):
                g.shape_item.prepareGeometryChange()
        if self.shape_groups:                     # tags keep their screen size: re-place them for this zoom
            if animating and self._tags_slow:
                return                            # many tags: placed once the glide lands (keeps it fluid)
            if len(self.shape_groups) <= 400:
                t0 = time.perf_counter()
                self.layout_tags()                # every frame while zooming: no tag ever slides over another
                if animating and time.perf_counter() - t0 > 0.0015:
                    self._tags_slow = True        # this frame was too costly: skip until the glide lands
            else:
                self._schedule_tag_layout()

    def _set_smooth(self, zoom: float, animating: bool):
        # Smooth (bilinear) whenever it is affordable — also while gliding: nearest-neighbour shimmers while the
        # scale changes, which reads as jitter. Very large images glide with the fast filter. Once the view rests
        # the image item re-renders the visible part in high quality (ImageItem). Hard pixel squares only with
        # Settings ▸ Viewer ▸ Show pixels when zoomed in (from 300 %).
        pm = self.pixmap_item.pixmap()
        big = pm.width() * pm.height() > 12_000_000
        pixels = getattr(self.pixmap_item, "pixel_view", False) and zoom >= 3.0
        smooth = (not big if animating else True) and not pixels
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation if smooth
                                               else Qt.TransformationMode.FastTransformation)

    def is_animating(self) -> bool:
        return self._zoom_anim is not None

    def _rest_update_mode(self):
        _gc_pause(False)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate if self._gpu else
                                   QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)

    def use_gpu(self) -> bool:
        """Settings ▸ Viewer ▸ GPU image canvas: an OpenGL viewport (the graphics card scales and blends the image
        and shapes). Falls back to the normal canvas when OpenGL can't start."""
        try:
            from PyQt6.QtGui import QSurfaceFormat
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget
            vp = QOpenGLWidget()
            fmt = QSurfaceFormat()
            fmt.setSamples(4)
            fmt.setSwapInterval(1)                       # in step with the display: no tearing
            vp.setFormat(fmt)
            self.setViewport(vp)
            self._gpu = True
            self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
            return True
        except Exception as e:                           # no OpenGL here: keep the software canvas
            from loguru import logger
            logger.warning(f"[canvas] GPU canvas unavailable: {e}")
            return False

    def stop_zoom_animation(self):
        anim, self._zoom_anim = self._zoom_anim, None
        if anim is not None:
            anim.stop()
            anim.deleteLater()
            self._unpad_scene()
            self._rest_update_mode()

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
        anim = _Glide(self, duration, easing)
        self._tags_slow = False
        self._set_smooth(zoom, True)
        self._pad_scene(min(z0, zoom))

        interacting(duration / 1000.0 + 0.25)                # background loops wait the glide out

        def step(e):
            e = float(e)
            interacting(0.25)
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

        anim.step, anim.done = step, done
        self._zoom_anim = anim
        # one full repaint per frame while gliding (every item moves anyway; no per-item damage bookkeeping)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        _gc_pause(True)                       # no garbage-collection pause in the middle of a glide
        anim.start()

    def _finish_zoom(self, mode: str):
        self._rest_update_mode()
        self.zoom_mode = mode
        self._target_zoom = self.current_zoom
        self._set_smooth(self.current_zoom, False)
        self.pixmap_item.timer.start()            # the view rests: high-quality render of what is visible
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
        self._applied_view = None

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

    def zoom_step(self, direction: int):
        """Zoom in / out buttons: the next Windows Photos level (10-point steps to 200 %, 25 to 600 %, 50 above).
        Steps accumulate on the running target, so N quick clicks are N steps (a click during a glide continues from
        where that glide is heading, not from the frame it happens to be at)."""
        from AIO_QC.widgets.zoom_widget import photos_step
        base = self._target_zoom if self.is_animating() else self.current_zoom
        self.zoom_to(photos_step(base, 1 if direction > 0 else -1, self.MIN_ZOOM, self.MAX_ZOOM))

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
