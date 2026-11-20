# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Image QC canvas: module-level helpers shared by its modules (also re-exported by AIO_QC.widgets.canvas)."""
from __future__ import annotations
import time
import math
import numpy as np
from PyQt6.QtCore import QEasingCurve, QEvent, QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF)
from PyQt6.QtWidgets import (
    QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem,
)
from AIO_QC.core.diag import ignored as _ignored
import sys as _sys


def _auto_contrast(rgb: np.ndarray) -> np.ndarray:
    """CLAHE on the lightness channel (OpenCV); percentile stretch as a fallback. Works on 3-channel RGB or BGR
    (the lightness of both is the same up to the swapped hue, which is swapped back)."""
    try:
        import cv2
        lab = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except Exception as _exc:
        _ignored(_exc)
        lo, hi = np.percentile(rgb, (1, 99))
        return np.clip((rgb.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
def display_lut(brightness: float, contrast: float, gamma: float) -> np.ndarray:
    """The 256-entry tone curve for brightness / contrast (-100..100) and gamma (0.3..3)."""
    lut = np.arange(256, dtype=np.float32) / 255.0
    lut = np.power(lut, 1.0 / max(gamma, 1e-3))                         # gamma
    c = contrast * 1.28                                                  # -128..128
    f = (259 * (c + 255)) / (255 * (259 - c))
    lut = f * (lut * 255 - 128) + 128 + brightness * 1.28              # contrast around mid-grey, brightness
    return np.clip(lut, 0, 255).astype(np.uint8)
def _is_oom(exc) -> bool:
    from AIO_QC.utils.memory import is_oom
    return is_oom(exc)
def _qimage_array(img: QImage, write: bool = False) -> np.ndarray:
    """(h, w, 4) uint8 view of a 32-bit QImage's pixels (no copy; keep `img` alive). Read-only unless `write`
    (bits() would detach a QImage that shares its pixels with a pixmap)."""
    w, h = img.width(), img.height()
    ptr = None if img.isNull() else (img.bits() if write else img.constBits())
    if ptr is None:                       # Qt could not allocate the image (out of memory): no pixels to view
        raise MemoryError(f"no memory for a {w} × {h} image")
    ptr.setsize(img.sizeInBytes())
    return np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine())[:, : w * 4].reshape(h, w, 4)
class DisplayRenderer:
    """Brightness / contrast / gamma / auto contrast for the shown image, fast on any size and any format.

    The source is the image in Qt's native 32-bit layout (RGB32, or ARGB32 when it has transparency) — grey,
    indexed, 16-bit, CMYK … images are converted once. The tone curve is one multi-threaded cv2.LUT straight into
    a QImage Qt owns, which becomes the pixmap without another conversion or copy (≈ 8× faster than per-pixel
    numpy indexing; ~0.1–0.4 s for a 108 MP image). The CLAHE result is cached per image, so moving a slider
    with Auto contrast on only re-applies the curve."""

    def __init__(self):
        self._key = None
        self._src: QImage | None = None
        self._auto = None

    def reset(self):
        self._key = self._src = self._auto = None

    def _source(self, pixmap: QPixmap) -> QImage:
        if self._key != pixmap.cacheKey() or self._src is None:
            img = pixmap.toImage()
            want = QImage.Format.Format_ARGB32 if img.hasAlphaChannel() else QImage.Format.Format_RGB32
            if img.format() != want:
                img = img.convertToFormat(want)
            self._key, self._src, self._auto = pixmap.cacheKey(), img, None
        return self._src

    def render(self, pixmap: QPixmap, d: dict) -> QPixmap:
        import cv2
        src = self._source(pixmap)
        a = _qimage_array(src)
        if d.get("auto"):
            if self._auto is None:                                  # B, G, R, A in memory (little-endian)
                bgr = _auto_contrast(np.ascontiguousarray(a[:, :, 2::-1]))[:, :, ::-1]
                self._auto = np.dstack([bgr, a[:, :, 3:4]])
            a = self._auto
        lut = display_lut(d["brightness"], d["contrast"], d["gamma"])
        lut4 = np.stack([lut, lut, lut, np.arange(256, dtype=np.uint8)], axis=-1).reshape(1, 256, 4)
        out = QImage(src.width(), src.height(), src.format())
        dst = _qimage_array(out, write=True)              # MemoryError when it could not be allocated
        if dst.flags["C_CONTIGUOUS"] and a.flags["C_CONTIGUOUS"]:
            cv2.LUT(a, lut4, dst=dst)
        else:                                                       # padded scan lines (odd widths never are)
            dst[...] = cv2.LUT(np.ascontiguousarray(a), lut4)
        return QPixmap.fromImage(out)
GEOMETRY_TYPES = ("rectangle", "polygon", "circle", "line", "linestrip", "point", "points")
MIN_POINTS = {"polygon": 3, "linestrip": 2, "points": 1}
HANDLE_PX = 9
class ImageItem(QGraphicsPixmapItem):
    """The image, drawn like Windows Photos at every zoom.

    Qt alone scales a picture with plain bilinear filtering: enlarged a lot it turns soft-blocky, shrunk a lot it
    aliases (jagged edges, shimmer). While the view moves that fast path is used; once it rests (≈ 90 ms), the
    visible part is re-rendered once at the screen's real resolution — bicubic when magnified, area-averaged when
    reduced — and drawn 1:1 on device pixels. Panning reuses it; only a new zoom or new area re-renders.
    `pixel_view` (Settings ▸ Viewer ▸ Show pixels when zoomed in) keeps hard pixel squares when magnified instead."""
    REFINE_UP, REFINE_DOWN = 1.15, 0.85          # device scale outside [DOWN, UP]: worth a high-quality render

    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas
        self.pixel_view = False
        self.hq = None                           # (device scale, scene QRectF, QImage at device resolution)
        self._arr = None                         # view of the shown pixmap's pixels (no copy; first refine)
        self._src = None                         # the QImage that owns them
        self._last_view = None                   # the view the pending render is for
        self._backoff = 0.0                      # monotonic time until which refining is paused (low memory)
        from PyQt6.QtCore import QTimer
        self.timer = QTimer(canvas)
        self.timer.setSingleShot(True)
        self.timer.setInterval(90)
        self.timer.timeout.connect(lambda: self.refine(background=True))

    def setPixmap(self, pm):
        self.hq, self._arr, self._src = None, None, None
        super().setPixmap(pm)

    def _dpr(self) -> float:
        try:
            return float(self.canvas.viewport().devicePixelRatioF())
        except RuntimeError:
            return 1.0

    def _wanted(self, s: float) -> bool:
        d = s * self._dpr()
        if d > self.REFINE_UP:
            return not self.pixel_view
        return d < self.REFINE_DOWN

    def paint(self, p, opt, w=None):
        super().paint(p, opt, w)                 # fast filter: the base (and what shows while moving)
        s = p.worldTransform().m11()
        if self.pixmap().isNull() or not self._wanted(s):
            return
        hq = self.hq
        moving = self.canvas.is_animating() or getattr(self.canvas, "_holding", False)
        if hq is not None and abs(hq[0] - s) < 1e-9 and not moving:
            tgt = p.worldTransform().mapRect(hq[1])
            p.save()
            p.resetTransform()
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            p.drawImage(QRectF(round(tgt.x()), round(tgt.y()), hq[2].width() / hq[2].devicePixelRatio(),
                               hq[2].height() / hq[2].devicePixelRatio()), hq[2])
            p.restore()
            if self._covers(hq):
                return
        if not moving:
            # Debounce on view changes only: a repaint with the same view (a notification fading or counting down
            # over the canvas, a blinking caret) must not keep postponing the high-quality render.
            key = (s, self.canvas.viewportTransform().dx(), self.canvas.viewportTransform().dy(),
                   self.canvas.viewport().size())
            if time.monotonic() < self._backoff:
                pass                                       # memory was short: keep the fast view for a while
            elif key != self._last_view or not self.timer.isActive():
                self._last_view = key
                self.timer.start()

    def _view_rect(self) -> QRectF:
        """The visible part of the image, in scene coordinates, snapped to whole viewport pixels."""
        c = self.canvas
        vr = QRectF(c.viewport().rect())
        img_v = c.mapFromScene(self.boundingRect()).boundingRect()
        r = vr.intersected(QRectF(img_v))
        if r.isEmpty():
            return QRectF()
        r = QRectF(math.floor(r.left()), math.floor(r.top()), math.ceil(r.width()) + 1, math.ceil(r.height()) + 1)
        r = r.intersected(vr)
        inv, _ok = c.viewportTransform().inverted()
        return inv.mapRect(r)

    def _covers(self, hq) -> bool:
        v = self._view_rect()
        return v.isEmpty() or hq[1].adjusted(-1e-3, -1e-3, 1e-3, 1e-3).contains(v)

    def invalidate(self):
        self.hq = None
        self.update()

    def refine(self, background: bool = False):
        """High-quality render of the visible part (background=True: computed on a worker thread — the view rests,
        the result is shown when ready). Never fatal: when memory runs out the fast view stays (it is already on
        screen), the memory this item holds is released and the app is told to free its caches."""
        if time.monotonic() < self._backoff:
            return
        try:
            self._refine(background)
        except Exception as exc:                           # MemoryError, cv2.error "Insufficient memory", …
            if not _is_oom(exc):
                raise
            self.hq, self._arr, self._src = None, None, None
            self._backoff = time.monotonic() + 15.0             # no retry storm while memory is short
            self.canvas.report_low_memory("high-quality view")

    def _refine(self, background=False):
        c = self.canvas
        pm = self.pixmap()
        if pm.isNull() or c.is_animating() or getattr(c, "_holding", False):
            return
        s = c.viewportTransform().m11()
        if not self._wanted(s):
            self.hq = None
            return
        S = self._view_rect()
        if S.isEmpty():
            return
        dpr = self._dpr()
        k = s * dpr                                             # source pixels -> device pixels
        W, H = max(1, int(round(S.width() * k))), max(1, int(round(S.height() * k))) 
        if W * H > 40_000_000:
            return
        if self._arr is None:                # the pixmap's own pixels (shared, no copy) in Qt's 32-bit layout
            img = pm.toImage()
            if img.format() not in (QImage.Format.Format_RGB32, QImage.Format.Format_ARGB32,
                                    QImage.Format.Format_ARGB32_Premultiplied):
                img = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied if img.hasAlphaChannel()
                                          else QImage.Format.Format_RGB32)
            self._src = img
            self._arr = _qimage_array(img)
        a = self._arr
        ih, iw = a.shape[:2]
        x0, y0 = max(0, int(math.floor(S.left())) - 3), max(0, int(math.floor(S.top())) - 3)
        x1, y1 = min(iw, int(math.ceil(S.right())) + 3), min(ih, int(math.ceil(S.bottom())) + 3)
        if x1 <= x0 or y1 <= y0:
            return
        src, fmt = self._src, self._src.format()
        view = (s, c.viewportTransform().dx(), c.viewportTransform().dy())
        self._job = getattr(self, "_job", 0) + 1
        job = self._job

        def work():
            # off the UI thread (OpenCV releases the GIL): zoom / pan stay fluid while the sharp view is computed.
            # `src` keeps the pixels the array views alive until this is done.
            import cv2
            _keep = src
            try:
                if k > 1:                                       # magnified: bicubic, exact sub-pixel mapping
                    sx, sy = S.width() / W, S.height() / H
                    M = np.array([[sx, 0, S.left() - x0 + 0.5 * sx - 0.5], [0, sy, S.top() - y0 + 0.5 * sy - 0.5]],
                                 np.float64)
                    out = cv2.warpAffine(a[y0:y1, x0:x1], M, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_REPLICATE)
                    rect = S
                else:                                           # reduced: area average (no aliasing)
                    sub = a[int(math.floor(S.top())):int(math.ceil(S.bottom())),
                            int(math.floor(S.left())):int(math.ceil(S.right()))]
                    if not sub.size:
                        return job, None, None, None
                    rect = QRectF(math.floor(S.left()), math.floor(S.top()), sub.shape[1], sub.shape[0])
                    out = cv2.resize(sub, (max(1, int(round(sub.shape[1] * k))), max(1, int(round(sub.shape[0] * k)))),
                                     interpolation=cv2.INTER_AREA)
                del _keep
                return job, out, rect, None
            except Exception as exc:                            # handed to the UI thread
                _ignored(exc)
                return job, None, None, exc

        def done(res):
            jid, out, rect, exc = res
            if jid != self._job or self.pixmap().isNull() or self._src is not src:
                return                                          # a newer view / image since: drop it
            if exc is not None:
                if _is_oom(exc):
                    self.hq, self._arr, self._src = None, None, None
                    self._backoff = time.monotonic() + 15.0
                    self.canvas.report_low_memory("high-quality view")
                return
            if out is None or c.is_animating() or getattr(c, "_holding", False):
                return
            now = (c.viewportTransform().m11(), c.viewportTransform().dx(), c.viewportTransform().dy())
            if any(abs(p - q) > 1e-6 for p, q in zip(now, view)):
                self.timer.start()                              # the view moved meanwhile: render it again
                return
            try:
                qi = QImage(out.shape[1], out.shape[0], fmt)    # channel order as in the source
                _qimage_array(qi, write=True)[...] = out
            except Exception as e:
                _ignored(e)
                if _is_oom(e):
                    self.hq, self._arr, self._src = None, None, None
                    self._backoff = time.monotonic() + 15.0     # no retry storm while memory is short
                    self.canvas.report_low_memory("high-quality view")
                return
            qi.setDevicePixelRatio(dpr)
            self.hq = (s, rect, qi)
            self.update()
        if background:
            _refiner().submit(work, done)
        else:
            done(work())
class _Refiner(QObject):
    """One background thread for the high-quality views; results come back on the UI thread (queued signal)."""
    result = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        import concurrent.futures as cf
        self.pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="aioqc-refine")
        self.result.connect(self._deliver)

    @staticmethod
    def _deliver(fn, res):
        try:
            fn(res)
        except RuntimeError:                                    # its canvas closed meanwhile: nothing to show
            pass

    def submit(self, work, done):
        def run():
            try:
                res = work()
            except Exception as e:                              # never lose a result silently
                _ignored(e)
                res = (None, None, None, e)
            self.result.emit(done, res)
        self.pool.submit(run)
_REFINER = None
def _refiner() -> "_Refiner":
    global _REFINER
    if _REFINER is None:
        _REFINER = _Refiner()
    return _REFINER
class PointsItem(QGraphicsPathItem):
    """Point shapes: dots with a constant on-screen radius (like the outlines and the tags), so a point looks the
    same at every zoom instead of swelling over the pixels around it."""
    RADIUS_PX = 3.5

    def __init__(self, points, lines=False):
        super().__init__()
        self.lines = lines
        self.set_points(points)

    def set_points(self, points):
        self.prepareGeometryChange()
        self.pts = [QPointF(p) for p in points]
        xs, ys = [p.x() for p in self.pts] or [0.0], [p.y() for p in self.pts] or [0.0]
        self._box = QRectF(QPointF(min(xs), min(ys)), QPointF(max(xs), max(ys)))
        path = QPainterPath()
        if self.lines and self.pts:
            path.moveTo(self.pts[0])
            for p in self.pts[1:]:
                path.lineTo(p)
        self.setPath(path)

    def _scale(self) -> float:
        views = self.scene().views() if self.scene() is not None else []
        return max(views[0].transform().m11(), 1e-6) if views else 1.0

    def _dots(self, radius_px) -> QPainterPath:
        r = radius_px / self._scale()
        path = QPainterPath(self.path())
        for p in self.pts:
            path.addEllipse(p, r, r)
        return path

    def boundingRect(self):
        """Point box + the dot radius at the current zoom (asked on every paint / hit test: no path building)."""
        m = (self.RADIUS_PX * 2 + self.pen().widthF() + 2) / self._scale()
        return self._box.adjusted(-m, -m, m, m)

    def shape(self):
        return self._dots(self.RADIUS_PX + 4)                       # a comfortable click target

    def paint(self, p, _opt, _w=None):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.lines:
            p.setPen(self.pen())
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(self.path())
        s = max(p.worldTransform().m11(), 1e-6)
        r = self.RADIUS_PX * self.pen().widthF() / 2.0 / s
        pen = QPen(QColor(0, 0, 0, 170), 1.0)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(self.pen().color())
        for pt in self.pts:
            p.drawEllipse(pt, r, r)
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
    _metrics: dict = {}                           # (font, text) -> (elided, text width, ascent, height)

    @classmethod
    def _measure(cls, font, text):
        key = (font.key(), text)
        hit = cls._metrics.get(key)
        if hit is None:
            from PyQt6.QtGui import QFontMetricsF
            fm = QFontMetricsF(font)
            el = fm.elidedText(text, Qt.TextElideMode.ElideRight, cls.MAX_W - 2 * cls.PAD_X)
            hit = (el, fm.horizontalAdvance(el), fm.ascent(), fm.height())
            if len(cls._metrics) > 20000:
                cls._metrics.clear()
            cls._metrics[key] = hit
        return hit

    def __init__(self, text: str, color: QColor, font):
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(False)
        # rendered once into a device-pixel cache: zooming / panning blits it (the tag never changes size)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.font = font
        full = str(text)
        self.text, self.text_w, self.ascent, fh = self._measure(font, full)
        if self.text != full:
            self.setToolTip(full)
        self.w = math.ceil(self.text_w + 2 * self.PAD_X)
        self.h = math.ceil(fh + 2 * self.PAD_Y)
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
_GC = {"off": False}
def _gc_pause(on: bool):
    """Python's cycle collector can stop the UI for tens of ms on a big dataset; it waits while the view glides."""
    import gc
    from AIO_QC.utils import gui_gc
    if gui_gc.pause(on):                             # the GUI-thread collector holds instead
        return
    if on and not _GC["off"] and gc.isenabled():
        gc.disable()
        _GC["off"] = True
    elif not on and _GC["off"]:
        _GC["off"] = False
        gc.enable()
class _Glide(QObject):
    """The zoom glide's clock: one step per display frame (a precise timer at the screen's refresh rate — 60, 120,
    144 Hz …), the position computed from the elapsed time, so a slow frame never slows the glide down or makes it
    jump back; Qt's animation timer ticks at a fixed ~16 ms that drifts against the display and stutters."""

    def __init__(self, canvas, duration, easing):
        super().__init__(canvas)
        from PyQt6.QtCore import QElapsedTimer, QTimer
        try:
            from AIO_QC.widgets.mini_session import refresh_hz
            hz = refresh_hz(canvas)
        except Exception as _exc:
            _ignored(_exc)
            hz = 60.0
        self.duration = max(1, int(duration))
        self.curve = QEasingCurve(easing)
        self.clock = QElapsedTimer()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(max(2, int(1000.0 / hz)))
        self.timer.timeout.connect(self._tick)
        self.step = self.done = None

    def start(self):
        self.clock.start()
        self.timer.start()
        self._tick()

    def stop(self):
        self.timer.stop()

    def _tick(self):
        p = min(1.0, self.clock.elapsed() / self.duration)
        try:
            self.step(self.curve.valueForProgress(p))
        except RuntimeError:
            self.timer.stop()
            return
        if p >= 1.0:
            self.timer.stop()
            if self.done is not None:
                self.done()
_MAC = _sys.platform == "darwin"
class _DrawGuide(QGraphicsItem):
    """What is being drawn, painted in screen pixels so it looks the same at any zoom.

    box      a crosshair through the pointer across the image (where the corner will go)
    polygon  the placed edges, the rubber edge to the pointer, a faint closing edge back to the first corner, a dot on
             every corner; the first corner grows into a ring once a click there would close the polygon (snap)."""
    INK = QColor("#ffd400")

    def __init__(self, canvas):
        super().__init__()
        self.c = canvas
        self.kind = "box"
        self.pts: list = []
        self.cursor = None
        self.snap = False
        self.setZValue(canvas.SELECTED_Z + 20)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self):
        return QRectF(-1e7, -1e7, 2e7, 2e7)

    def paint(self, p, _opt, _w=None):
        t = p.worldTransform()
        p.save()
        p.resetTransform()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.kind == "box":
            self._paint_cross(p, t)
        else:
            self._paint_poly(p, t)
        p.restore()

    def _paint_cross(self, p, t):
        if self.cursor is None or self.c._draw_start is not None:
            return
        img = t.mapRect(self.c._image_rect())
        q = t.map(self.cursor)
        for col, w in ((QColor(0, 0, 0, 110), 3), (QColor(255, 255, 255, 190), 1)):
            pen = QPen(col, w, Qt.PenStyle.DashLine if w == 1 else Qt.PenStyle.SolidLine)
            p.setPen(pen)
            p.drawLine(QPointF(img.left(), q.y()), QPointF(img.right(), q.y()))
            p.drawLine(QPointF(q.x(), img.top()), QPointF(q.x(), img.bottom()))

    def _paint_poly(self, p, t):
        if not self.pts:
            return
        P = [t.map(q) for q in self.pts]
        cur = t.map(self.cursor) if self.cursor is not None and not self.snap else None
        ring = list(P) + ([cur] if cur is not None else [])
        if len(ring) >= 3:                                              # the area it will cover
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 212, 0, 46 if self.snap else 30))
            p.drawPolygon(QPolygonF(ring))
        p.setBrush(Qt.BrushStyle.NoBrush)
        if cur is not None and len(P) >= 2:                             # the edge that closes it
            pen = QPen(QColor(255, 212, 0, 110), 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(cur, P[0])
        edges = QPolygonF(P + ([P[0]] if self.snap else []))
        for col, w in ((QColor(0, 0, 0, 120), 4), (self.INK, 2)):       # placed edges, with a dark halo
            p.setPen(QPen(col, w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.drawPolyline(edges)
        if cur is not None:                                             # the edge that follows the pointer
            p.setPen(QPen(QColor(0, 0, 0, 120), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(P[-1], cur)
            p.setPen(QPen(self.INK, 2, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(P[-1], cur)
        for i, q in enumerate(P):
            first = i == 0
            if first and self.snap:                                     # a click here closes it
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 212, 0, 90))
                p.drawEllipse(q, 15, 15)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(QColor(0, 0, 0, 150), 4))
                p.drawEllipse(q, 11, 11)
                p.setPen(QPen(QColor("#ffffff"), 2))
                p.drawEllipse(q, 11, 11)
                p.setPen(QPen(QColor(0, 0, 0, 170), 1.5))
                p.setBrush(self.INK)
                p.drawEllipse(q, 6, 6)
                continue
            r = 5 if first and len(P) >= 3 else 3.5
            p.setPen(QPen(QColor(0, 0, 0, 170), 1.5))
            p.setBrush(self.INK if first else QColor("#ffffff"))
            p.drawEllipse(q, r, r)
class _DrawKeys(QObject):
    """Installed on the canvas only while drawing: Esc / Enter / Backspace / Ctrl+Z belong to the drawing — not to the
    window's Esc (clear the selection) or Undo (the last edit of the file). The canvas then gets them as key presses."""

    def __init__(self, canvas):
        super().__init__(canvas)
        self.c = canvas

    def eventFilter(self, obj, e):
        if e.type() == QEvent.Type.ShortcutOverride and self.c._drawing and self.c._is_draw_key(e):
            e.accept()
            return True
        return False

