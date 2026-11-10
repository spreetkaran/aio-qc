# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
VideoCanvas: paints frames and handles viewer interaction.

  * Zoom (Ctrl+wheel, around the cursor) and pan (middle-drag, or left-drag
    when zoomed and not editing the ROI). Double-click resets the view.
    When zoomed, pixels come from the full-resolution frame, not the
    downscaled display copy, so fine detail stays crisp; at high zoom
    nearest-neighbour shows real pixels.
  * ROI editing (ROI mode): drag on empty space to draw, drag inside to move,
    drag edges/corners to resize. Coordinates are full-resolution video pixels.
  * Mouse wheel (no modifier) steps frames.
  * Small on-screen display (OSD) for transient feedback.
"""
import math

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF, QSize, QTimer, QEvent, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QImage, QPainterPath, QFont
from PyQt6.QtWidgets import QWidget, QSizePolicy

from AIO_QC.widgets import motion as anim

ROI_COLOR = QColor(14, 180, 196)
HANDLE = 7      # px hit tolerance / handle half-size
MIN_ROI = 4     # px (video) - smaller drags are ignored


ZOOM_MIN, ZOOM_MAX = 1.0, 64.0
ZOOM_TAU = 0.075          # s, time constant of the zoom easing (critically damped feel)


class CanvasMixin:
    """Everything a video canvas does except putting the picture on screen:
    view transform, animated zoom / pan, ROI editing, overlays, mouse & wheel handling.
    Mixed into the software canvas (QPainter) and the GPU canvas (OpenGL)."""

    def _init_canvas(self, text="Load a video"):
        self._image = None      # display-size QImage
        self._bgr = None        # full-res frame (for zoomed rendering)
        self._text = text
        self._vw = self._vh = 0
        self._zoom = 1.0
        self._cx = self._cy = 0.0
        self._roi = None
        self._dets = []         # [(x1, y1, x2, y2, text, QColor, dashed)] in video pixels
        self._hud = None        # timecode overlay lines (bottom-left)
        self._stats = None      # playback statistics lines (top-right)
        self._roi_mode = False
        self._drag = None       # dict describing the current drag
        self._zoom_cache = (None, None)
        self._osd = None
        self._osd_timer = QTimer(self)
        self._osd_timer.setSingleShot(True)
        self._osd_timer.timeout.connect(self._clear_osd)
        self._frame = None      # engine Frame currently shown (GPU canvas uploads from it)
        # animated view
        self._tz = 1.0          # target zoom
        self._anchor = None     # (widget point, video point) kept fixed while zooming
        self._tc = None         # target centre (fit / 1:1) when not anchored
        self._anim_key = ("canvas-zoom", id(self))     # driven by the shared display-rate clock (utils.anim)
        self._anim_done = True
        self._osd_alpha = 0.0
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(140)
        self._settle.timeout.connect(self.update)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        font = self.font()
        font.setPointSize(28)
        self.setFont(font)

    def sizeHint(self):
        return QSize(1280, 720)

    # ------------------------------------------------------------- content
    def set_video_size(self, w, h):
        self._vw, self._vh = int(w), int(h)
        self.reset_view()

    def set_frame(self, image, bgr=None):
        self._image, self._bgr, self._text = image, bgr, None
        self._dets = []
        self.update()

    def show_frame(self, frame):
        """Show an engine Frame (keeps the object so the GPU canvas can use full-res / NV12 data)."""
        self._frame = frame
        self.set_frame(frame.image, frame.bgr)

    def set_stats(self, lines):
        if lines != self._stats:
            self._stats = lines
            self.update()

    def set_hud(self, lines):
        """Timecode overlay (list of text lines) or None."""
        if lines != self._hud:
            self._hud = lines
            self.update()

    def set_detections(self, dets):
        self._dets = list(dets or [])
        self.update()

    def set_text(self, text):
        self._image = self._bgr = None
        self._frame = None
        self._text = text
        self._roi = None
        self.update()

    def physical_size(self):
        dpr = self.devicePixelRatioF()
        return int(self.width() * dpr), int(self.height() * dpr)

    def show_osd(self, text, ms=1200):
        self._osd = text
        self._osd_timer.start(ms)
        anim.animate(("osd", id(self)), self._set_osd_alpha, self._osd_alpha, 1.0, 120, "out_cubic", widget=self)

    def _set_osd_alpha(self, a):
        self._osd_alpha = a
        self.update()

    def _clear_osd(self):
        def gone():
            self._osd = None
            self.update()
        anim.animate(("osd", id(self)), self._set_osd_alpha, self._osd_alpha, 0.0, 260, "in_out_cubic", gone,
                     widget=self)

    # ---------------------------------------------------------------- ROI
    @property
    def roi(self):
        return self._roi

    def set_roi(self, roi):
        self._roi = tuple(int(v) for v in roi) if roi else None
        self.update()

    def set_roi_mode(self, on):
        self._roi_mode = bool(on)
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        self.update()

    # --------------------------------------------------------------- view
    @property
    def zoom(self):
        return self._zoom

    def _base_scale(self):
        if not self._vw or not self._vh:
            return 1.0
        return min(self.width() / self._vw, self.height() / self._vh)

    def _scale(self):
        return self._base_scale() * self._zoom

    def v2w(self, x, y):
        s = self._scale()
        return QPointF((x - self._cx) * s + self.width() / 2, (y - self._cy) * s + self.height() / 2)

    def w2v(self, px, py):
        s = self._scale() or 1.0
        return (px - self.width() / 2) / s + self._cx, (py - self.height() / 2) / s + self._cy

    def _clamp_center(self):
        s = self._scale() or 1.0
        for attr, size, span in (("_cx", self._vw, self.width()), ("_cy", self._vh, self.height())):
            half = span / 2 / s
            if half * 2 >= size:
                setattr(self, attr, size / 2)
            else:
                setattr(self, attr, min(max(getattr(self, attr), half), size - half))

    @property
    def is_fit(self):
        return abs(self._zoom - 1.0) < 1e-3 and abs(self._tz - 1.0) < 1e-3

    def actual_zoom(self):
        """Zoom factor (relative to fit) at which 1 video pixel = 1 physical screen pixel."""
        return (1.0 / self.devicePixelRatioF()) / max(1e-9, self._base_scale())

    @property
    def is_actual(self):
        if not self._vw:
            return False
        a = max(ZOOM_MIN, min(ZOOM_MAX, self.actual_zoom()))
        return abs(self._zoom - a) / a < 2e-3 and abs(self._tz - a) / a < 2e-3

    def _anim_active(self):
        return anim.FrameClock.get().has(self._anim_key)

    def _anim_stop(self):
        anim.FrameClock.get().remove(self._anim_key)

    def reset_view(self, animate=False):
        """Fit to window."""
        if animate and self._vw and anim.enabled():
            self._tz, self._anchor, self._tc = 1.0, None, (self._vw / 2, self._vh / 2)
            self._start_anim()
            return
        self._anim_stop()
        self._zoom = self._tz = 1.0
        self._anchor = self._tc = None
        self._cx, self._cy = self._vw / 2, self._vh / 2
        self.zoom_changed.emit(self._zoom)
        self.update()

    def fit(self):
        self.reset_view(animate=True)

    def actual_pixels(self, anchor=None):
        """1 video pixel = 1 screen pixel (zoom around the anchor or the centre)."""
        if not self._vw:
            return
        self.zoom_to(max(ZOOM_MIN, min(ZOOM_MAX, self.actual_zoom())), anchor)

    def zoom_by(self, factor, anchor=None, animate=True):
        self.zoom_to(self._tz * factor, anchor, animate)

    def zoom_to(self, zoom, anchor=None, animate=True):
        if not self._vw:
            return
        anchor = anchor if anchor is not None else QPointF(self.width() / 2, self.height() / 2)
        self._tz = min(ZOOM_MAX, max(ZOOM_MIN, zoom))
        self._anchor = (QPointF(anchor), self.w2v(anchor.x(), anchor.y()))
        self._tc = None
        if not animate or not anim.enabled():
            self._zoom = self._tz
            self._apply_anchor()
            self._anim_stop()
            self.zoom_changed.emit(self._zoom)
            self.update()
            return
        self._start_anim()

    def _start_anim(self):
        if not self._anim_active():
            anim.FrameClock.get().add(self._anim_key, self._anim_tick, anim.screen_hz(self))

    def _anim_tick(self, dt):
        self._anim_step(dt)
        return self._anim_active() and not self._anim_done

    def _apply_anchor(self):
        (a, (vx, vy)) = self._anchor
        s = self._scale() or 1.0
        self._cx = vx - (a.x() - self.width() / 2) / s
        self._cy = vy - (a.y() - self.height() / 2) / s
        self._clamp_center()

    def _anim_step(self, dt):
        dt = min(0.05, dt)
        tau = ZOOM_TAU * (0.6 if anim.mode() == "reduced" else 1.0)
        k = 1.0 - math.exp(-dt / tau)
        self._anim_done = False
        lz, lt = math.log(self._zoom), math.log(self._tz)
        self._zoom = math.exp(lz + (lt - lz) * k)
        done = abs(lt - math.log(self._zoom)) < 0.002
        if self._anchor is not None:
            self._apply_anchor()
        elif self._tc is not None:
            self._cx += (self._tc[0] - self._cx) * k
            self._cy += (self._tc[1] - self._cy) * k
            self._clamp_center()
            s = self._scale() or 1.0
            done = done and abs(self._tc[0] - self._cx) * s < 0.5 and abs(self._tc[1] - self._cy) * s < 0.5
        if done:
            self._zoom = self._tz
            if self._anchor is not None:
                self._apply_anchor()
            elif self._tc is not None:
                self._cx, self._cy = self._tc
                self._clamp_center()
            self._anim_done = True
            self._settle.start()      # software canvas: re-render the crisp full-res crop once settled
        self.zoom_changed.emit(self._zoom)
        self.update()

    def event(self, ev):
        # trackpad pinch (macOS, Windows precision touchpads)
        if ev.type() == QEvent.Type.NativeGesture and ev.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self.zoom_to(self._zoom * (1.0 + ev.value()), ev.position(), animate=False)
            return True
        return super().event(ev)

    # ------------------------------------------------------------ painting
    def _zoomed_image(self):
        """Render the visible part of the full-res frame at screen resolution."""
        x0, y0 = self.w2v(0, 0)
        x1, y1 = self.w2v(self.width(), self.height())
        ix0, iy0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        ix1, iy1 = min(self._vw, int(np.ceil(x1))), min(self._vh, int(np.ceil(y1)))
        if ix1 <= ix0 or iy1 <= iy0:
            return None, None
        tl, br = self.v2w(ix0, iy0), self.v2w(ix1, iy1)
        target = QRectF(tl, br)
        dpr = self.devicePixelRatioF()
        pw, ph = max(1, int(target.width() * dpr)), max(1, int(target.height() * dpr))
        key = (id(self._bgr), ix0, iy0, ix1, iy1, pw, ph)
        if self._zoom_cache[0] == key:
            return self._zoom_cache[1], target
        crop = self._bgr[iy0:iy1, ix0:ix1]
        px_scale = self._scale() * dpr
        interp = cv2.INTER_NEAREST if px_scale >= 3 else (cv2.INTER_AREA if px_scale < 1 else cv2.INTER_LINEAR)
        img = np.ascontiguousarray(cv2.resize(crop, (pw, ph), interpolation=interp))
        qimg = QImage(img.data, pw, ph, img.strides[0], QImage.Format.Format_BGR888).copy()
        self._zoom_cache = (key, qimg)
        return qimg, target

    def _roi_rect_w(self, roi):
        x1, y1, x2, y2 = roi
        return QRectF(self.v2w(x1, y1), self.v2w(x2, y2)).normalized()

    def _has_picture(self):
        return self._vw and ((self._image is not None and not self._image.isNull()) or self._frame is not None)

    def _paint_image_software(self, p):
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        drawn = False
        # during zoom animation draw the cheap display image; the crisp crop comes when it settles
        if self._zoom > 1.01 and self._bgr is not None and not self._anim_active():
            img, target = self._zoomed_image()
            if img is not None:
                p.drawImage(target, img)
                drawn = True
        if not drawn and self._image is not None:
            p.drawImage(QRectF(self.v2w(0, 0), self.v2w(self._vw, self._vh)), self._image)

    def _paint_overlays(self, p):
        if self._has_picture():
            self._paint_dets(p)
            self._paint_roi(p)
            if self._zoom > 1.01:
                pct = self._zoom * self._base_scale() * self.devicePixelRatioF() * 100
                self._paint_badge(p, f"{pct:.0f}%", bottom=True)
            if self._hud:
                self._paint_hud(p)
            if self._stats:
                self._paint_stats(p)
        elif self._text:
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._text)
        if self._osd and self._osd_alpha > 0.01:
            p.setOpacity(self._osd_alpha)
            self._paint_badge(p, self._osd)
            p.setOpacity(1.0)

    def _paint_roi(self, p):
        roi = self._drag["roi"] if self._drag and self._drag.get("roi") else self._roi
        if not roi:
            return
        r = self._roi_rect_w(roi)
        outside = QPainterPath()
        outside.addRect(QRectF(self.rect()))
        inner = QPainterPath()
        inner.addRect(r)
        p.fillPath(outside.subtracted(inner), QColor(0, 0, 0, 110))
        p.setPen(QPen(ROI_COLOR, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        if self._roi_mode:
            p.setBrush(ROI_COLOR)
            for pt in (r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight(),
                       QPointF(r.center().x(), r.top()), QPointF(r.center().x(), r.bottom()),
                       QPointF(r.left(), r.center().y()), QPointF(r.right(), r.center().y())):
                p.drawRect(QRectF(pt.x() - 4, pt.y() - 4, 8, 8))
        x1, y1, x2, y2 = roi
        label = f" {x1},{y1} \u2013 {x2},{y2}  ({x2 - x1}\u00d7{y2 - y1}) "
        f = QFont(self.font())
        f.setPointSize(9)
        p.setFont(f)
        fm = p.fontMetrics()
        tw, th = fm.horizontalAdvance(label), fm.height()
        ty = r.top() - th - 2 if r.top() - th - 2 > 0 else r.top() + 2
        p.fillRect(QRectF(r.left(), ty, tw, th), ROI_COLOR)
        p.setPen(QColor(0, 0, 0))
        p.drawText(QRectF(r.left(), ty, tw, th), Qt.AlignmentFlag.AlignCenter, label)

    def _paint_dets(self, p):
        if not self._dets:
            return
        f = QFont(self.font())
        f.setPointSize(8)
        p.setFont(f)
        fm = p.fontMetrics()
        p.setBrush(Qt.BrushStyle.NoBrush)
        for x1, y1, x2, y2, text, color, dashed in self._dets:
            r = QRectF(self.v2w(x1, y1), self.v2w(x2, y2))
            p.setPen(QPen(color, 2, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine))
            p.drawRect(r)
            tw, th = fm.horizontalAdvance(text) + 6, fm.height()
            tr = QRectF(r.left(), r.top() - th if r.top() - th > 0 else r.top(), tw, th)
            p.fillRect(tr, color)
            p.setPen(QColor(255, 255, 255))
            p.drawText(tr, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_hud(self, p):
        from PyQt6.QtGui import QFontDatabase
        f = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        f.setPointSizeF(12)
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        w = max(fm.horizontalAdvance(l) for l in self._hud) + 20
        lh = fm.height()
        h = lh * len(self._hud) + 10
        r = QRectF(12, self.height() - h - 12, w, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 165))
        p.drawRoundedRect(r, 6, 6)
        for i, line in enumerate(self._hud):
            p.setPen(QColor(255, 255, 255) if i == 0 else QColor(200, 200, 200))
            p.drawText(QRectF(r.x() + 10, r.y() + 5 + i * lh, w - 20, lh), Qt.AlignmentFlag.AlignVCenter, line)

    def _paint_stats(self, p):
        from PyQt6.QtGui import QFontDatabase
        f = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        f.setPointSizeF(9)
        p.setFont(f)
        fm = p.fontMetrics()
        lines = self._stats
        w = max(fm.horizontalAdvance(l) for l in lines) + 18
        lh = fm.height()
        r = QRectF(self.width() - w - 12, 12, w, lh * len(lines) + 10)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 175))
        p.drawRoundedRect(r, 6, 6)
        for i, line in enumerate(lines):
            bad = line.startswith("!")
            p.setPen(QColor(255, 170, 60) if bad else QColor(230, 230, 230))
            p.drawText(QRectF(r.x() + 9, r.y() + 5 + i * lh, w - 18, lh), Qt.AlignmentFlag.AlignVCenter,
                       line[1:] if bad else line)

    def _paint_badge(self, p, text, bottom=False):
        f = QFont(self.font())
        f.setPointSize(11)
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(text) + 20, fm.height() + 10
        x = self.width() - w - 12 if bottom else 12
        y = self.height() - h - 12 if bottom else 12
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRoundedRect(QRectF(x, y, w, h), 6, 6)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRectF(x, y, w, h), Qt.AlignmentFlag.AlignCenter, text)

    # --------------------------------------------------------------- mouse
    def _hit(self, pos):
        if not self._roi:
            return None
        r = self._roi_rect_w(self._roi)
        x, y, t = pos.x(), pos.y(), HANDLE
        near_l, near_r = abs(x - r.left()) <= t, abs(x - r.right()) <= t
        near_t, near_b = abs(y - r.top()) <= t, abs(y - r.bottom()) <= t
        in_x = r.left() - t <= x <= r.right() + t
        in_y = r.top() - t <= y <= r.bottom() + t
        if not (in_x and in_y):
            return None
        edge = ("t" if near_t else "b" if near_b else "") + ("l" if near_l else "r" if near_r else "")
        if edge:
            return edge
        return "move" if r.contains(pos) else None

    _CURSORS = {
        "tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
        "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
        "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
        "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
        "move": Qt.CursorShape.SizeAllCursor,
    }

    def _clamp_v(self, x, y):
        return min(max(x, 0), self._vw), min(max(y, 0), self._vh)

    def mousePressEvent(self, ev):
        # _has_picture, not _image: the GPU canvas shows full-res frames that have no display-size copy
        # (paused exact frames) - checking _image made still frames refuse to pan
        if not self._has_picture():
            return super().mousePressEvent(ev)
        pos = ev.position()
        btn = ev.button()
        if btn == Qt.MouseButton.MiddleButton or (
                btn == Qt.MouseButton.LeftButton and not self._roi_mode and self._zoom > 1.01):
            self._drag = {"kind": "pan", "last": pos}
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if btn == Qt.MouseButton.LeftButton and self._roi_mode:
            hit = self._hit(pos)
            vx, vy = self._clamp_v(*self.w2v(pos.x(), pos.y()))
            if hit:
                self._drag = {"kind": hit, "start": (vx, vy), "orig": self._roi, "roi": self._roi}
            else:
                self._drag = {"kind": "new", "start": (vx, vy), "roi": None}
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._vw:
            vx, vy = self.w2v(pos.x(), pos.y())
            inside = 0 <= vx < self._vw and 0 <= vy < self._vh
            self.cursor_moved.emit((int(vx), int(vy)) if inside and self._has_picture() else None)
        d = self._drag
        if d is None:
            if self._roi_mode:
                hit = self._hit(pos)
                self.setCursor(self._CURSORS.get(hit, Qt.CursorShape.CrossCursor))
            elif self._zoom > 1.01:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if d["kind"] == "pan":
            s = self._scale()
            self._cx -= (pos.x() - d["last"].x()) / s
            self._cy -= (pos.y() - d["last"].y()) / s
            d["last"] = pos
            self._clamp_center()
            self.update()
            return
        vx, vy = self._clamp_v(*self.w2v(pos.x(), pos.y()))
        sx, sy = d["start"]
        if d["kind"] == "new":
            d["roi"] = (int(min(sx, vx)), int(min(sy, vy)), int(round(max(sx, vx))), int(round(max(sy, vy))))
        elif d["kind"] == "move":
            x1, y1, x2, y2 = d["orig"]
            w, h = x2 - x1, y2 - y1
            nx1 = min(max(x1 + vx - sx, 0), self._vw - w)
            ny1 = min(max(y1 + vy - sy, 0), self._vh - h)
            d["roi"] = (int(nx1), int(ny1), int(nx1) + w, int(ny1) + h)
        else:
            x1, y1, x2, y2 = d["orig"]
            k = d["kind"]
            if "l" in k: x1 = vx
            if "r" in k: x2 = vx
            if "t" in k: y1 = vy
            if "b" in k: y2 = vy
            d["roi"] = (int(min(x1, x2)), int(min(y1, y2)), int(round(max(x1, x2))), int(round(max(y1, y2))))
        self.update()

    def mouseReleaseEvent(self, ev):
        d, self._drag = self._drag, None
        if d is None:
            return super().mouseReleaseEvent(ev)
        if d["kind"] == "pan":
            self.setCursor(Qt.CursorShape.CrossCursor if self._roi_mode else Qt.CursorShape.OpenHandCursor)
            return
        roi = d.get("roi")
        if roi and (roi[2] - roi[0] >= MIN_ROI and roi[3] - roi[1] >= MIN_ROI):
            self._roi = roi
            self.roi_changed.emit(roi)
        self.update()

    def mouseDoubleClickEvent(self, ev):
        if self._zoom > 1.01:
            self.fit()
        else:
            self.actual_pixels(ev.position())

    def leaveEvent(self, ev):
        self.cursor_moved.emit(None)
        super().leaveEvent(ev)

    def wheelEvent(self, ev):
        dy = ev.angleDelta().y()
        if dy:
            mods = ev.modifiers()
            # Ctrl+wheel (Windows / Linux) and Cmd+wheel (macOS maps Cmd to ControlModifier)
            if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
                self.zoom_by(1.2 ** (dy / 120), ev.position())
            else:
                self.wheel_step.emit(1 if dy > 0 else -1)
        ev.accept()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._clamp_center()
        self.resized.emit()


class VideoCanvas(CanvasMixin, QWidget):
    """Software canvas (QPainter): used when OpenGL is unavailable."""
    resized = pyqtSignal()
    wheel_step = pyqtSignal(int)          # +1 / -1 per wheel notch
    roi_changed = pyqtSignal(object)      # (x1, y1, x2, y2) or None
    zoom_changed = pyqtSignal(float)
    cursor_moved = pyqtSignal(object)     # (x, y) video pixel under cursor, or None
    gpu = False

    def __init__(self, text="Load a video", parent=None):
        QWidget.__init__(self, parent)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self._init_canvas(text)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0))
        if self._has_picture():
            self._paint_image_software(p)
        self._paint_overlays(p)
        p.end()
