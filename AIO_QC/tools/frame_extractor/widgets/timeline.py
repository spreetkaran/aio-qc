# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Timeline analysis: a low-priority background pass over the video that feeds
  * hover thumbnails on the seekbar (ThumbPopup),
  * the sharpness / motion / scene-change / model strip under it (TimelineStrip).

The worker owns its own (software) decoder so playback keeps the GPU one. It
samples ~2 frames/s (bounded), coarse-to-fine when samples are far apart so the
whole strip fills in quickly, and jumps ahead to whatever the user hovers.
Motion = % difference between a sample and the very next frame (true motion,
not the change between samples).

Fast path (PyAV installed): a demux-only scan builds a keyframe index, samples
are keyframes (decoded alone, nothing in between), and motion comes from the
encoder's inter-frame packet sizes (kB/frame). The same index is handed to the
playback engine for instant keyframe previews while scrubbing.
"""
import bisect
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger
from PyQt6.QtCore import Qt, QThread, QRectF, QPointF, QPoint, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QPainterPath, QFont, QImage
from PyQt6.QtWidgets import QWidget, QSizePolicy


THUMB_W = 192


@dataclass
class Sample:
    idx: int
    sharp: float
    motion: float
    hist: np.ndarray
    thumb: QImage
    n_det: int = -1        # -1 = model not run
    min_conf: float = 1.0
    flagged: bool = False  # model uncertain / disagreement
    slot: int = -1         # grid position this sample fills (idx is the real, decoded frame)
    small: np.ndarray = None


class TimelineWorker(QThread):
    """Parallel keyframe sampling (core.fast_sampler): cost is independent of the video's length -
    about 1-2 s for typical videos, a few seconds for multi-hour ones. Coarse-to-fine, so the strip is
    covered almost immediately; hovering jumps the queue. Optional model track runs on the samples."""
    samples_ready = pyqtSignal(list)
    analysis_done = pyqtSignal()
    model_error = pyqtSignal(str)
    index_ready = pyqtSignal(object)        # kept for API compatibility (no full-file index any more)
    grid_changed = pyqtSignal(list, str)

    def __init__(self, path, backend, frame_count, fps, duration, model_cfg=None, time_of=None, index_of=None,
                 size=(0, 0), ffmpeg=None):
        super().__init__()
        from AIO_QC.tools.frame_extractor.core.fast_sampler import Analyzer, plan_grid, estimate_gop
        from AIO_QC.tools.frame_extractor.core.preview_pipe import find_ffprobe
        self.fps = fps or 30.0
        self.grid = plan_grid(frame_count, duration)
        gop, _ = estimate_gop(path, find_ffprobe(ffmpeg))      # first packets only (~0.1 s)
        if gop and len(self.grid) > 1 and (self.grid[1] - self.grid[0]) < gop:
            # samples land on keyframes: never plan more than one per keyframe interval
            self.grid = plan_grid(frame_count, duration, lo=2, hi=max(2, frame_count // gop + 1))
        self.model_cfg = model_cfg if (model_cfg and model_cfg.get("models")) else None
        t_of = time_of or (lambda f: f / self.fps)
        i_of = index_of or (lambda sec: int(round(sec * self.fps)))
        self.analyzer = Analyzer(path, self.grid, t_of, i_of, keep_full=bool(self.model_cfg),
                                 ffmpeg=ffmpeg, size=size)
        self.motion_unit = "kB/f" if self.analyzer.kind == "pyav" else "%"
        self._quit = False
        self.t_total = None

    def request(self, idx):
        self.analyzer.request(idx)

    def set_throttle(self, on):
        self.analyzer.throttle = on

    def stop(self):
        self._quit = True
        self.analyzer.stop()
        self.wait()

    def _load_detectors(self):
        if not self.model_cfg:
            return None
        try:
            from AIO_QC.tools.frame_extractor.core.detector import load_detectors
            from AIO_QC.tools.frame_extractor.widgets.models_dialog import detector_kwargs
            c = self.model_cfg
            return load_detectors(c["models"], min(c["conf"], c["unc_low"]), **detector_kwargs(c, threads=2))
        except Exception as e:
            self.model_error.emit(str(e))
            return None

    def _model_fields(self, s, f, detectors):
        from AIO_QC.tools.frame_extractor.core.detector import mining_reasons
        c = self.model_cfg
        per = [d.detect(f) for d in detectors]
        s.n_det = sum(1 for dets in per for d in dets if d.conf >= c["conf"])
        s.min_conf = min((d.conf for dets in per for d in dets), default=1.0)
        s.flagged = bool(mining_reasons(per, ["uncertain", "disagree"], c["conf"], c["unc_low"],
                                        c["unc_high"], models=detectors))

    def run(self):
        import threading as _th
        t0 = time.monotonic()
        detectors = self._load_detectors()
        model_lock = _th.Lock()
        lock = _th.Lock()
        batch, last = [], [time.monotonic()]
        self.grid_changed.emit(list(self.grid), self.motion_unit)

        def on_result(r, urgent):
            thumb = np.ascontiguousarray(r.thumb)
            q = QImage(thumb.data, thumb.shape[1], thumb.shape[0], thumb.strides[0],
                       QImage.Format.Format_BGR888).copy()
            smp = Sample(r.idx, r.sharp, -1.0 if r.motion is None else r.motion, r.hist, q)
            smp.slot, smp.small = r.slot, thumb
            if detectors and r.full is not None:
                with model_lock:
                    try:
                        self._model_fields(smp, r.full, detectors)
                    except Exception as e:
                        self.model_error.emit(str(e))
            with lock:
                batch.append(smp)
                now = time.monotonic()
                if urgent or now - last[0] > 0.1:
                    out = batch[:]
                    batch.clear()
                    last[0] = now
                else:
                    out = None
            if out:
                self.samples_ready.emit(out)

        try:
            self.analyzer.run(on_result)
        except Exception as e:
            logger.warning(f"Timeline analysis failed: {e}")
        with lock:
            out = batch[:]
            batch.clear()
        if out:
            self.samples_ready.emit(out)
        self.t_total = time.monotonic() - t0
        if not self._quit:
            logger.info(f"Timeline analysed: {len(self.grid)} samples in {self.t_total:.2f} s "
                        f"({self.analyzer.backend}, {self.analyzer.workers} workers)")
            self.analysis_done.emit()


class ThumbPopup(QWidget):
    """Floating preview above the seekbar (thumbnail + time + frame)."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._img, self._text, self._sub = None, "", ""

    def show_at(self, global_x, global_bottom, img, text, sub=""):
        self._img, self._text, self._sub = img, text, sub
        w = THUMB_W + 8 if img is not None else 150
        h = (img.height() * THUMB_W // max(1, img.width()) + 8 if img is not None else 0) + 38
        self.resize(w, h)
        self.move(int(global_x - w / 2), int(global_bottom - h - 4))
        self.update()
        if not self.isVisible():
            self.show()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        p.setPen(QColor(90, 90, 90))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        y = 4
        if self._img is not None:
            th = self._img.height() * THUMB_W // max(1, self._img.width())
            p.drawImage(QRectF(4, 4, THUMB_W, th), self._img)
            y += th + 2
        f = QFont(self.font())
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRectF(0, y, self.width(), 16), Qt.AlignmentFlag.AlignCenter, self._text)
        f.setBold(False)
        f.setPointSize(8)
        p.setFont(f)
        p.setPen(QColor(180, 180, 180))
        p.drawText(QRectF(0, y + 15, self.width(), 14), Qt.AlignmentFlag.AlignCenter, self._sub)
        p.end()


class TimelineStrip(QWidget):
    """Sharpness (bars, red below the blur threshold), motion (orange line),
    scene changes (white ticks), model results (purple = detections, amber =
    uncertain / disagreement). Aligned to the seekbar; click/drag to seek."""
    seek_requested = pyqtSignal(int)
    hovered = pyqtSignal(int, QPoint)   # frame, global point (top of strip)
    hover_left = pyqtSignal()
    marker_moved = pyqtSignal(int, int) # old frame, new frame (dragged in the marker lane)

    LANE = 13                           # marker lane height (top of the strip)
    MARKER_COLORS = {"pending": QColor("#43A047"), "saved": QColor("#1E88E5"), "suggested": QColor("#43A047")}

    CUT_THRESHOLD = 0.45

    def __init__(self, slider, parent=None):
        super().__init__(parent)
        self.slider = slider
        self.setMouseTracking(True)
        self.setFixedHeight(58)
        self.markers = {}
        self._drag = None               # (frame, current x) while dragging a marker
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip("")
        self.clear()

    def clear(self):
        self.grid = []
        self.samples = {}
        self.cuts = set()
        self.blur_threshold = 0.0
        self.current = 0
        self._sharp_norm = 1.0
        self._motion_norm = 1.0
        self.analyzing = False
        self.motion_unit = "%"
        self._cache = None
        self.update()

    def set_grid(self, grid, motion_unit=None):
        if motion_unit:
            self.motion_unit = motion_unit
        keep = {k: v for k, v in getattr(self, "samples", {}).items() if k in set(grid)}
        self.samples = keep
        self.cuts = set()
        self.grid = list(grid)
        self._cache = None
        self._gpos = {g: i for i, g in enumerate(self.grid)}
        self.analyzing = True
        self.update()

    def add_samples(self, samples):
        from AIO_QC.tools.frame_extractor.core.metrics import signature, difference
        for s in samples:
            key = s.slot if s.slot >= 0 else s.idx
            self.samples[key] = s
            i = self._gpos.get(key)
            if i is None:
                continue
            # motion unknown (ffmpeg / OpenCV samplers): change between neighbouring samples
            for j in (i - 1, i + 1):
                if 0 <= j < len(self.grid) and self.grid[j] in self.samples:
                    a, b = (self.samples[self.grid[j]], s) if j < i else (s, self.samples[self.grid[j]])
                    if b.motion < 0 and a.small is not None and b.small is not None:
                        b.motion = difference(signature(a.small), signature(b.small))
            if s.motion < 0 and i == 0:
                s.motion = 0.0
            for j in (i - 1, i + 1):   # scene change between neighbouring grid samples
                if 0 <= j < len(self.grid) and self.grid[j] in self.samples:
                    a, b = sorted((key, self.grid[j]))
                    d = cv2.compareHist(self.samples[a].hist, self.samples[b].hist, cv2.HISTCMP_BHATTACHARYYA)
                    if d > self.CUT_THRESHOLD:
                        self.cuts.add(b)
                    else:
                        self.cuts.discard(b)
        vals = [s.sharp for s in self.samples.values()]
        mot = [s.motion for s in self.samples.values() if s.motion >= 0]
        if vals:
            self._sharp_norm = math.log1p(max(1.0, float(np.percentile(vals, 97))))
        if mot:
            self._motion_norm = max(1e-3, float(np.percentile(mot, 97)))
        self.invalidate()

    def nearest_sample(self, idx, max_dist=None):
        if not self.samples:
            return None
        keys = sorted(self.samples)
        i = bisect.bisect_left(keys, idx)
        cands = [keys[j] for j in (i - 1, i) if 0 <= j < len(keys)]
        k = min(cands, key=lambda g: abs(g - idx))
        if max_dist is not None and abs(k - idx) > max_dist:
            return None
        return self.samples[k]

    def grid_spacing(self):
        return (self.grid[-1] - self.grid[0]) / max(1, len(self.grid) - 1) if len(self.grid) > 1 else 1

    def set_current(self, idx):
        if idx == self.current:
            return
        old = self.current
        self.current = idx
        # repaint only the two thin strips around the old and new playhead
        if self.grid:
            for v in (old, idx):
                x = int(self._x(v))
                self.update(x - 3, 0, 7, self.height())
        else:
            self.update()

    def set_markers(self, markers):
        """{frame: 'pending' | 'saved' | 'suggested'}"""
        self.markers = dict(markers)
        self.update()

    def marker_at(self, x, tol=6):
        best = None
        for f in self.markers:
            d = abs(self._x(f) - x)
            if d <= tol and (best is None or d < best[0]):
                best = (d, f)
        return best[1] if best else None

    def set_blur_threshold(self, v):
        self.blur_threshold = v or 0.0
        self.invalidate()

    def flagged_frames(self):
        return sorted(s.idx for s in self.samples.values() if s.flagged)

    # ------------------------------------------------------------ mapping
    def _x(self, v):
        return self.slider.x_for_value(v)

    def _value(self, x):
        return self.slider.value_at(x)

    def invalidate(self):
        """Static layer (bars, motion, cuts) must be re-rendered."""
        self._cache = None
        self.update()

    def _render_static(self):
        """Bars / motion / cuts into a cached pixmap: during playback a repaint is only a blit + the playhead."""
        from PyQt6.QtGui import QPixmap
        dpr = self.devicePixelRatioF()
        pm = QPixmap(max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr)))
        pm.setDevicePixelRatio(dpr)
        pal = self.palette()
        pm.fill(pal.base().color().darker(106))
        p = QPainter(pm)
        h = self.height()
        if not self.grid:
            p.setPen(pal.placeholderText().color())
            p.drawText(QRectF(0, 0, self.width(), h), Qt.AlignmentFlag.AlignCenter,
                       "Timeline analysis appears here after loading a video")
            p.end()
            return pm
        top, bars_h = self.LANE + 2, h - 14 - self.LANE
        base_y = top + bars_h
        thr = math.log1p(self.blur_threshold) / self._sharp_norm if self.blur_threshold else None
        keys = self.grid
        for i, g in enumerate(keys):
            x0 = self._x(g)
            x1 = self._x(keys[i + 1]) if i + 1 < len(keys) else x0 + 2
            s_ = self.samples.get(g)
            if s_ is None:
                p.fillRect(QRectF(x0, top, max(1.0, x1 - x0), bars_h), QColor(128, 128, 128, 40))
                continue
            v = min(1.0, math.log1p(s_.sharp) / self._sharp_norm)
            col = QColor(214, 69, 69, 200) if thr is not None and v < thr else QColor(47, 111, 219, 170)
            p.fillRect(QRectF(x0, base_y - v * bars_h, max(1.0, x1 - x0 - 0.5), v * bars_h), col)
            if s_.n_det >= 0:
                c = QColor(217, 138, 0) if s_.flagged else (QColor(122, 79, 214) if s_.n_det else QColor(128, 128, 128, 90))
                p.fillRect(QRectF(x0, h - 9, max(1.0, x1 - x0 - 0.5), 6), c)
        if thr is not None:
            p.setPen(QPen(QColor(214, 69, 69), 1, Qt.PenStyle.DashLine))
            y = base_y - thr * bars_h
            p.drawLine(QPointF(self._x(keys[0]), y), QPointF(self._x(keys[-1]), y))
        pts = [(self._x(g), self.samples[g].motion) for g in keys if g in self.samples and self.samples[g].motion >= 0]
        if len(pts) > 1:
            path = QPainterPath()
            for n, (x, m) in enumerate(pts):
                y = base_y - min(1.0, m / self._motion_norm) * bars_h
                path.moveTo(x, y) if n == 0 else path.lineTo(x, y)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(QPen(QColor(245, 140, 30), 1.4))
            p.drawPath(path)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(QPen(pal.text().color(), 1))
        for c in self.cuts:
            x = self._x(c)
            p.drawLine(QPointF(x, top), QPointF(x, base_y))
        p.end()
        return pm

    def paintEvent(self, ev):
        if getattr(self, "_cache", None) is None or self._cache_size != (self.width(), self.height()):
            self._cache = self._render_static()
            self._cache_size = (self.width(), self.height())
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)
        if not self.grid:
            p.end()
            return
        h = self.height()
        self._paint_markers(p, h)
        p.setPen(QPen(QColor(14, 180, 196), 2))
        x = self._x(self.current)
        p.drawLine(QPointF(x, 0), QPointF(x, h))
        if self.analyzing:
            done = sum(1 for g in self.grid if g in self.samples)
            f = QFont(self.font())
            f.setPointSize(7)
            p.setFont(f)
            p.setPen(self.palette().text().color())
            p.drawText(QRectF(4, 0, 160, 12), Qt.AlignmentFlag.AlignLeft, f"analyzing {100 * done // len(self.grid)}%")
        p.end()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._cache = None

    def changeEvent(self, ev):
        from PyQt6.QtCore import QEvent
        if ev.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self._cache = None
        super().changeEvent(ev)

    def _paint_markers(self, p, h):
        if not self.markers:
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        drag_f = self._drag[0] if self._drag else None
        for f, state in self.markers.items():
            x = self._drag[1] if f == drag_f else self._x(f)
            col = self.MARKER_COLORS.get(state, QColor("#43A047"))
            line = QColor(col)
            line.setAlpha(110)
            p.setPen(QPen(line, 1, Qt.PenStyle.DashLine if state == "suggested" else Qt.PenStyle.SolidLine))
            p.drawLine(QPointF(x, self.LANE - 2), QPointF(x, h))
            head = QPainterPath()
            head.moveTo(x - 5, 1)
            head.lineTo(x + 5, 1)
            head.lineTo(x + 5, self.LANE - 6)
            head.lineTo(x, self.LANE - 1)
            head.lineTo(x - 5, self.LANE - 6)
            head.closeSubpath()
            p.setPen(QPen(col, 1.4))
            p.setBrush(Qt.BrushStyle.NoBrush if state == "suggested" else col)
            p.drawPath(head)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    def mousePressEvent(self, ev):
        x, y = ev.position().x(), ev.position().y()
        if ev.button() == Qt.MouseButton.LeftButton and self.grid:
            f = self.marker_at(x) if y <= self.LANE + 4 else None
            if f is not None:
                self._drag = (f, x)
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                return
            self.seek_requested.emit(self._value(x))

    def mouseMoveEvent(self, ev):
        if not self.grid:
            return
        v = self._value(ev.position().x())
        if self._drag is not None:
            self._drag = (self._drag[0], ev.position().x())
            self.update()
            self.hovered.emit(v, self.mapToGlobal(QPoint(int(ev.position().x()), 0)))
            return
        if ev.position().y() <= self.LANE + 4 and self.marker_at(ev.position().x()) is not None:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.unsetCursor()
        if ev.buttons() & Qt.MouseButton.LeftButton:
            self.seek_requested.emit(v)
        self.hovered.emit(v, self.mapToGlobal(QPoint(int(ev.position().x()), 0)))

    def mouseReleaseEvent(self, ev):
        if self._drag is not None:
            old, x = self._drag
            self._drag = None
            self.unsetCursor()
            new = self._value(x)
            if new != old:
                self.marker_moved.emit(old, new)
            self.update()

    def mouseDoubleClickEvent(self, ev):
        f = self.marker_at(ev.position().x())
        if f is not None:
            self.seek_requested.emit(f)

    def leaveEvent(self, ev):
        self.hover_left.emit()

    def describe(self, idx):
        s = self.nearest_sample(idx, max_dist=self.grid_spacing())
        if s is None:
            return ""
        txt = f"sharp {s.sharp:.0f} · motion {max(0.0, s.motion):.1f}{' ' if self.motion_unit != '%' else ''}{self.motion_unit}"
        if s.n_det >= 0:
            txt += f" · {s.n_det} det" + (f" · min conf {s.min_conf:.2f}" if s.n_det else "") + (" · ⚑" if s.flagged else "")
        return txt
