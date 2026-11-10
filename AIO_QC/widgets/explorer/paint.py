# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Drawing and image loading shared by the Prediction Explorer's grid, object view and image viewer.

Objects are drawn in IMAGE coordinates through a (scale, offset) mapping, so the same routine draws on a 160 px
tile, a crop and the zoomable full-resolution viewer. Images are decoded off the GUI thread at the size they are
shown (JPEG scaled decode: a fraction of a full decode), crops decode only their region (clip rect).
"""
from __future__ import annotations

import math

from PyQt6.QtCore import QObject, QPointF, QRect, QRectF, QRunnable, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QImage, QImageIOHandler, QImageReader, QPainter, QPen, QPolygonF

from AIO_QC.core.explorer import CLASS, DUPE, EXTRA, FIT, MISSED, NA, OK, TAG_NAMES

TAG_COLORS = {OK: "#22c55e", CLASS: "#a855f7", FIT: "#f59e0b", DUPE: "#94a3b8", EXTRA: "#ef4444",
              MISSED: "#f97316", NA: "#9ca3af"}
DISAGREE_COLOR = "#06b6d4"
COLOR_MODES = [("match", "Match result"), ("class", "Class"), ("model", "Model")]


def tag_color(tag: int) -> QColor:
    return QColor(TAG_COLORS.get(tag, "#9ca3af"))


def object_color(o, mode: str, label_colors: dict, model_colors: dict) -> QColor:
    if mode == "class":
        c = label_colors.get(o.label)
        return QColor(c) if c is not None else QColor("#e5e7eb")
    if mode == "model":
        return QColor(model_colors.get(o.model, "#ff2bd6")) if o.kind == "pred" else QColor("#f8fafc")
    return tag_color(o.tag)


def object_text(o, model_names: dict | None = None, with_tag: bool = False) -> str:
    if o.kind == "pred":
        name = (model_names or {}).get(o.model, "")
        t = f"{o.label} {o.conf:.2f}"
        if name and model_names is not None:
            t = f"{name}: {t}"
    else:
        t = o.label
    if with_tag and o.tag in TAG_NAMES and o.tag != OK:
        t += f" · {TAG_NAMES[o.tag]}"
    return t


def draw_objects(p: QPainter, objs, scale: float, ox: float, oy: float, *, mode="match", label_colors=None,
                 model_colors=None, model_names=None, width=1.6, text=False, font: QFont | None = None,
                 selected: int | None = None, hover: int | None = None, fill_selected=True):
    """Objects in image pixels -> painter at (x * scale + ox, y * scale + oy). Labels solid, predictions dashed.
    selected / hover: index into objs (thicker outline, light fill)."""
    label_colors, model_colors = label_colors or {}, model_colors or {}
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    tags = []
    for n, o in enumerate(objs):
        col = object_color(o, mode, label_colors, model_colors)
        emph = n == selected or n == hover
        w = width * (2.0 if emph else 1.0)
        pts = o.poly if o.poly is not None and len(o.poly) >= 3 else None
        if pts is not None:
            shape = QPolygonF([QPointF(float(x) * scale + ox, float(y) * scale + oy) for x, y in pts])
        else:
            x1, y1, x2, y2 = o.box
            shape = QRectF(QPointF(x1 * scale + ox, y1 * scale + oy), QPointF(x2 * scale + ox, y2 * scale + oy))
        halo = QPen(QColor(0, 0, 0, 150), w + 2)
        halo.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen = QPen(col, w)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if o.kind == "pred":
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if emph and fill_selected:
            f = QColor(col)
            f.setAlpha(55)
            p.setBrush(f)
        for pn in (halo, pen):
            p.setPen(pn)
            if isinstance(shape, QRectF):
                p.drawRect(shape)
            else:
                p.drawPolygon(shape)
            p.setBrush(Qt.BrushStyle.NoBrush)
        if o.kps is not None and len(o.kps):
            p.setPen(QPen(QColor(0, 0, 0, 160), 1))
            p.setBrush(col)
            r = max(1.5, w * 1.2)
            for x, y, c in o.kps:
                if c >= 0.3 and (x or y):
                    p.drawEllipse(QPointF(float(x) * scale + ox, float(y) * scale + oy), r, r)
        if getattr(o, "disagree", False) and mode == "match" and o.kind == "pred":
            br = shape.boundingRect() if not isinstance(shape, QRectF) else shape
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(DISAGREE_COLOR))
            p.drawEllipse(QPointF(br.right(), br.top()), 3.2, 3.2)
        if text or emph:
            br = shape.boundingRect() if not isinstance(shape, QRectF) else shape
            tags.append((br, object_text(o, model_names, with_tag=emph), col))
    if tags:                                           # pills on top of every outline
        f = QFont(font) if font is not None else QFont()
        if font is None:
            f.setPointSizeF(7.5)
            f.setBold(True)
        p.setFont(f)
        fm = QFontMetricsF(f)
        for br, t, col in tags:
            tw, th = fm.horizontalAdvance(t) + 8, fm.height() + 2
            y = br.top() - th if br.top() - th >= 0 else br.top()
            r = QRectF(br.left(), y, tw, th)
            p.setPen(Qt.PenStyle.NoPen)
            bg = QColor(col)
            bg.setAlpha(225)
            p.setBrush(bg)
            p.drawRoundedRect(r, 3, 3)
            p.setPen(QColor("#111827") if bg.lightnessF() > 0.6 else QColor("white"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, t)
    p.restore()


def hit_test(objs, x: float, y: float, tol: float = 0.0) -> int | None:
    """Index of the smallest object containing image point (x, y) — or whose outline is within tol."""
    best, area = None, math.inf
    for n, o in enumerate(objs):
        x1, y1, x2, y2 = o.box
        if x1 - tol <= x <= x2 + tol and y1 - tol <= y <= y2 + tol:
            a = max(x2 - x1, 1e-6) * max(y2 - y1, 1e-6)
            if a < area:
                best, area = n, a
    return best


# =================================================================================================================
# Loading
# =================================================================================================================
class Signals(QObject):
    done = pyqtSignal(object, int, QImage, object)          # cache key, generation, image, info


def _reader(path):
    r = QImageReader(str(path))
    r.setAutoTransform(True)
    full = r.size()
    tf = r.transformation()
    rotated = tf != QImageIOHandler.Transformation.TransformationNone
    if rotated and int(tf.value) & int(QImageIOHandler.Transformation.TransformationRotate90.value):
        full = QSize(full.height(), full.width())                   # size as displayed
    return r, full, rotated


class ThumbTask(QRunnable):
    """Whole image decoded to fit `size`. info = (W, H) of the full image as displayed."""

    def __init__(self, sig, ckey, path, size: QSize, gen: int):
        super().__init__()
        self.sig, self.ckey, self.path, self.size, self.gen = sig, ckey, path, size, gen

    def run(self):
        img, info = QImage(), None
        try:
            if self.path is not None:
                r, full, rotated = _reader(self.path)
                if full.isValid() and not rotated:
                    k = min(self.size.width() / full.width(), self.size.height() / full.height(), 1.0)
                    if k < 1.0:
                        r.setScaledSize(QSize(max(1, round(full.width() * k)), max(1, round(full.height() * k))))
                img = r.read()
                if not img.isNull():
                    if not full.isValid():
                        full = img.size()
                    if img.width() > self.size.width() or img.height() > self.size.height():
                        img = img.scaled(self.size, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
                    img = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
                    info = (full.width(), full.height())
        except Exception:
            img = QImage()
        self.sig.done.emit(self.ckey, self.gen, img, info)


class CropTask(QRunnable):
    """A region around `box` (image pixels, padded to the tile's aspect) decoded at `size`.
    info = (x0, y0, scale): image point (x, y) is drawn at ((x - x0) * scale, (y - y0) * scale)."""

    def __init__(self, sig, ckey, path, box, size: QSize, gen: int, pad: float = 1.6):
        super().__init__()
        self.sig, self.ckey, self.path, self.box, self.size, self.gen, self.pad = sig, ckey, path, box, size, gen, pad

    def run(self):
        img, info = QImage(), None
        try:
            if self.path is not None:
                img, info = self._render()
        except Exception:
            img, info = QImage(), None
        self.sig.done.emit(self.ckey, self.gen, img, info)

    def _render(self):
        r, full, rotated = _reader(self.path)
        src = None
        if rotated or not full.isValid():
            src = r.read()
            if src.isNull():
                return QImage(), None
            full = src.size()
        W, H = full.width(), full.height()
        x1, y1, x2, y2 = self.box
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        tw, th = self.size.width(), self.size.height()
        cw, ch = max(bw * self.pad, 24), max(bh * self.pad, 24)
        if cw / ch < tw / th:
            cw = ch * tw / th
        else:
            ch = cw * th / tw
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rx1, ry1 = max(0, int(cx - cw / 2)), max(0, int(cy - ch / 2))
        rx2, ry2 = min(W, int(math.ceil(cx + cw / 2))), min(H, int(math.ceil(cy + ch / 2)))
        rect = QRect(rx1, ry1, max(1, rx2 - rx1), max(1, ry2 - ry1))
        k = min(tw / rect.width(), th / rect.height(), 8.0)
        size = QSize(max(1, round(rect.width() * k)), max(1, round(rect.height() * k)))
        if src is not None:
            crop = src.copy(rect).scaled(size, Qt.AspectRatioMode.IgnoreAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
        else:
            r.setClipRect(rect)
            r.setScaledSize(size)
            crop = r.read()
            if crop.isNull():
                return QImage(), None
        crop = crop.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        return crop, (rect.x(), rect.y(), size.width() / rect.width())


class FullTask(QRunnable):
    """Full-resolution image (the viewer)."""

    def __init__(self, sig, ckey, path, gen: int):
        super().__init__()
        self.sig, self.ckey, self.path, self.gen = sig, ckey, path, gen

    def run(self):
        img = QImage()
        try:
            if self.path is not None:
                r, _full, _rot = _reader(self.path)
                img = r.read()
                if not img.isNull():
                    img = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        except Exception:
            img = QImage()
        self.sig.done.emit(self.ckey, self.gen, img, (img.width(), img.height()) if not img.isNull() else None)


def image_path(session, key):
    s = session.by_key.get(key)
    if s is None:
        return None
    if s.image_path is not None and s.image_path.exists():
        return s.image_path
    try:
        return session.adapter.load(s, light=True).image_path
    except Exception:
        return None
