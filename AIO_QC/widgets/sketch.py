# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Sketch (P) and ROI capture (O) — two modes on the main image, with zoom / pan / Space-pan working as usual.

SKETCH — draw on the image to explain something (a review note, a question to an annotator, a slide):
    pen · highlighter · a shapes box (line · arrow · rectangle · ellipse · circle, scrolls sideways as in MS Paint) ·
    text · eraser (its own size) · colour picker, a colour palette (+ any colour), size ([ / ]), filled shapes, undo / redo (Ctrl+Z / Ctrl+Y), clear. Entering it unticks every label so the
    image is clean — tick the ones you want in the Labels panel; leaving it restores your label choice.
    Copy (Ctrl+C) or Save puts the image at full resolution with what is visible (drawing + the labels you ticked)
    on the clipboard / in a PNG or JPG. Drawings are kept per image (.aio_qc/sketches/) and come back next time.
    The brush cursor shows the real brush size at the current zoom.

ROI — draw boxes like a snipping tool: crosshair cursor, the rest of the image dims, live x1, y1 → x2, y2 · w × h.
    Coordinates are real image pixels (sub-pixel when zoomed in), rounded to 2 decimals, clamped to the image. Every
    image keeps its ROIs (.aio_qc/rois.json); right-click a box removes it, Esc removes the last one. Copy the
    current image's ROIs, export all images' ROIs (CSV / TXT / JSON), or save the ROI crops as images.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from PyQt6.QtCore import QEvent, QLineF, QPointF, QRect, QRectF, QSize, Qt, QTimer
from PyQt6.QtGui import (
    QColor, QCursor, QFont, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap,
    QPolygonF, QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QColorDialog, QComboBox, QFileDialog, QFrame, QGraphicsItem, QHBoxLayout,
    QInputDialog, QLabel, QMessageBox, QScrollArea, QSlider, QToolButton, QWidget,
)

from AIO_QC.core.io import atomic_write_text
from AIO_QC.utils import newIcon, style as apply_style
from AIO_QC.utils import later
from . import anim

PALETTE = ["#ff3b30", "#ff9500", "#ffcc00", "#34c759", "#00c7be", "#007aff", "#af52de", "#ff2d55", "#ffffff",
           "#000000"]
TOOLS = [  # key, label, icon, tip
    ("pen", "Pen", "sketch-pen", "Freehand pen"),
    ("hl", "Highlighter", "sketch-highlighter", "Wide see-through marker — the image shows through"),
    ("line", "Line", "sketch-line", "Straight line (Shift: 45° steps)"),
    ("arrow", "Arrow", "sketch-arrow", "Arrow (Shift: 45° steps)"),
    ("rect", "Rectangle", "sketch-rect", "Rectangle (Shift: square)"),
    ("ellipse", "Ellipse", "sketch-ellipse", "Ellipse (Shift: circle)"),
    ("circle", "Circle", "sketch-circle", "Circle — always round"),
    ("text", "Text", "sketch-text", "Click where the text goes"),
    ("erase", "Eraser", "sketch-eraser", "Erase parts of the drawing (its own size)"),
    ("pick", "Color Picker", "sketch-picker", "Colour picker: click the image to take its colour, then back to the last tool"),
]
SHAPES = ("line", "arrow", "rect", "ellipse", "circle")   # the scrollable shapes box, in this order
BOXES = ("rect", "ellipse", "circle")                    # drawn from a box (two corners); can be filled
SHAPES_VISIBLE = 2                                       # shapes shown at once; the rest scroll into view
Z_LAYER = 5e8


# ====================================================================================================== drawing
def _pts(el):
    return [QPointF(p[0], p[1]) for p in el.get("pts") or []]


def _smooth_path(pts) -> QPainterPath:
    path = QPainterPath()
    if not pts:
        return path
    path.moveTo(pts[0])
    if len(pts) < 3:
        for p in pts[1:]:
            path.lineTo(p)
        if len(pts) == 1:
            path.lineTo(pts[0] + QPointF(0.01, 0.01))
        return path
    for a, b in zip(pts[1:-1], pts[2:]):
        path.quadTo(a, (a + b) / 2)
    path.lineTo(pts[-1])
    return path


def draw_el(p: QPainter, el: dict):
    t = el["t"]
    col = QColor(el.get("c", "#ff3b30"))
    w = float(el.get("w", 6))
    pts = _pts(el)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    if t == "erase":
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        col = QColor(0, 0, 0, 255)
    if t == "hl":
        col.setAlphaF(0.38)
        w *= 3
    pen = QPen(col, w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    if t in ("pen", "hl", "erase"):
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(_smooth_path(pts))
    elif t in ("line", "arrow") and len(pts) == 2:
        a, b = pts
        p.setPen(pen)
        if t == "arrow":
            ang = math.atan2(b.y() - a.y(), b.x() - a.x())
            head = max(10.0, w * 4)
            tip = b
            base = QPointF(b.x() - head * 0.8 * math.cos(ang), b.y() - head * 0.8 * math.sin(ang))
            p.drawLine(QLineF(a, base))
            left = QPointF(b.x() - head * math.cos(ang - 0.45), b.y() - head * math.sin(ang - 0.45))
            right = QPointF(b.x() - head * math.cos(ang + 0.45), b.y() - head * math.sin(ang + 0.45))
            p.setBrush(col)
            p.setPen(QPen(col, max(1.0, w / 3), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                          Qt.PenJoinStyle.RoundJoin))
            p.drawPolygon(QPolygonF([tip, left, right]))
        else:
            p.drawLine(QLineF(a, b))
    elif t in BOXES and len(pts) == 2:
        r = QRectF(pts[0], pts[1]).normalized()
        p.setPen(pen)
        if el.get("fill"):
            f = QColor(col)
            f.setAlphaF(0.28)
            p.setBrush(f)
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
        (p.drawRect if t == "rect" else p.drawEllipse)(r)      # a circle is stored with a square box
    elif t == "text" and pts and el.get("text"):
        f = QFont()
        f.setPixelSize(int(max(10, w * 5)))
        f.setBold(True)
        path = QPainterPath()
        for i, line in enumerate(str(el["text"]).splitlines() or [""]):
            path.addText(pts[0] + QPointF(0, i * f.pixelSize() * 1.2), f, line)
        halo = QColor("#000000") if col.lightnessF() > 0.5 else QColor("#ffffff")
        halo.setAlphaF(0.8)
        p.setPen(QPen(halo, max(2.0, f.pixelSize() / 7), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        p.drawPath(path)
    p.restore()


def el_bounds(el) -> QRectF:
    pts = _pts(el)
    if not pts:
        return QRectF()
    xs, ys = [q.x() for q in pts], [q.y() for q in pts]
    w = float(el.get("w", 6)) * (3 if el["t"] == "hl" else 1)
    pad = w + (max(10.0, w * 4) if el["t"] == "arrow" else 0) + 4
    r = QRectF(min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad)
    if el["t"] == "text":
        r = r.united(QRectF(pts[0].x() - 4, pts[0].y() - w * 6, w * 5 * max(1, len(el.get("text", ""))) * 0.7,
                            w * 7 * max(1, len(str(el.get("text", "")).splitlines()))))
    return r


def _dropper_cursor(color) -> QCursor:
    """An eyedropper (the sketch-picker icon) with a white halo so it shows on any image; the tube holds the current
    colour and the hotspot is the tip, as in MS Paint."""
    d = 28
    pm = QPixmap(d, d)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(d / 256.0, d / 256.0)
    body = QPainterPath()
    for i, (x, y) in enumerate([(156, 116), (72, 200), (40, 216), (56, 184), (140, 100)]):
        body.moveTo(x, y) if i == 0 else body.lineTo(x, y)
    bulb = QPainterPath()
    bulb.moveTo(140, 100)
    bulb.lineTo(180, 60)
    bulb.arcTo(QRectF(173.4, 53.4, 45.25, 45.25), 135, -180)
    bulb.lineTo(172, 132)
    liquid = QPainterPath()
    for i, (x, y) in enumerate([(56, 184), (98, 142), (114, 158), (72, 200), (40, 216)]):
        liquid.moveTo(x, y) if i == 0 else liquid.lineTo(x, y)
    liquid.closeSubpath()
    for col, w in ((QColor(255, 255, 255, 235), 44), (QColor(20, 20, 20), 18)):   # halo, then the outline
        if w == 18:
            p.fillPath(liquid, QColor(color).toRgb())
        p.setPen(QPen(col, w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(body)
        p.drawPath(bulb)
        p.drawLine(QLineF(116, 76, 180, 140))
    p.end()
    return QCursor(pm, round(40 * d / 256.0), round(216 * d / 256.0))


class SketchLayer(QGraphicsItem):
    """The drawing at image resolution (an ARGB image, re-rendered from the vector elements on undo / load) plus the
    stroke being drawn right now (drawn live so see-through highlighter strokes never darken where they overlap)."""

    def __init__(self, w, h):
        super().__init__()
        self.setZValue(Z_LAYER)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.W, self.H = max(1, w), max(1, h)
        # very large images: the drawing layer is kept at ≤ 24 MP (~96 MB) and scaled up on screen
        self.s = min(1.0, math.sqrt(24e6 / (self.W * self.H)))
        self.img = QImage(max(1, int(self.W * self.s)), max(1, int(self.H * self.s)),
                          QImage.Format.Format_ARGB32_Premultiplied)
        self.img.fill(Qt.GlobalColor.transparent)
        self.live = None

    def boundingRect(self):
        return QRectF(0, 0, self.W, self.H)

    def _painter(self):
        p = QPainter(self.img)
        if self.s < 1.0:
            p.scale(self.s, self.s)
        return p

    def paint(self, p, _opt, _w=None):
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QRectF(0, 0, self.W, self.H), self.img)
        if self.live is not None and self.live["t"] != "erase":
            draw_el(p, self.live)

    def render_all(self, els):
        self.img.fill(Qt.GlobalColor.transparent)
        p = self._painter()
        for el in els:
            draw_el(p, el)
        p.end()
        self.update()

    def commit(self, el):
        p = self._painter()
        draw_el(p, el)
        p.end()
        self.update(el_bounds(el))


# ====================================================================================================== stores
class _Store:
    def __init__(self, root):
        self.root = Path(root)

    def rel(self, key):
        try:
            return Path(key).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return Path(key).as_posix()


class SketchStore(_Store):
    def path(self, key):
        return self.root / ".aio_qc" / "sketches" / (hashlib.sha1(self.rel(key).encode()).hexdigest()[:20] + ".json")

    def load(self, key) -> list:
        try:
            d = json.loads(self.path(key).read_text(encoding="utf-8"))
            return d.get("els") or []
        except (OSError, ValueError):
            return []

    def save(self, key, els):
        p = self.path(key)
        if not els:
            try:
                p.unlink()
            except OSError:
                pass
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(p, json.dumps({"file": self.rel(key), "els": els}, separators=(",", ":")))


class RoiStore(_Store):
    def __init__(self, root):
        super().__init__(root)
        self.path = self.root / ".aio_qc" / "rois.json"
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}
        self._t = QTimer(singleShot=True, interval=400, timeout=self.flush)

    def get(self, key) -> list:
        return [list(r) for r in self.data.get(self.rel(key), [])]

    def set(self, key, rois):
        rel = self.rel(key)
        if rois:
            self.data[rel] = [[round(float(v), 2) for v in r] for r in rois]
        else:
            self.data.pop(rel, None)
        self._t.start()

    def flush(self):
        if self.data or self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.path, json.dumps(self.data, indent=1))

    def total(self):
        return sum(len(v) for v in self.data.values()), len(self.data)


def fmt_roi(r) -> str:
    return f"{r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}, {r[3]:.2f}"


# ====================================================================================================== floating bars
class _Bar(QFrame):
    def __init__(self, canvas, title):
        super().__init__(canvas)
        self.setObjectName("modeBar")
        apply_style(self, "QFrame#modeBar{background:@surface; border:1px solid @line; border-radius:10px;}"
                          "QToolButton{border:none; border-radius:6px; padding:3px;} "
                          "QToolButton:checked{background:@accent; color:white;} "
                          "QToolButton:hover:!checked{background:@line;}")
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(8, 4, 8, 4)
        self.lay.setSpacing(3)
        t = QLabel(title)
        apply_style(t, "font-weight:600; padding-right:6px;")
        self.lay.addWidget(t)
        self.hide()

    def btn(self, icon, tip, slot=None, checkable=False, text=""):
        b = QToolButton()
        if icon:
            b.setIcon(newIcon(icon))
            b.setIconSize(QSize(20, 20))
            b.setProperty("aio_icon", icon)
        if text:
            b.setText(text)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon if icon else
                                 Qt.ToolButtonStyle.ToolButtonTextOnly)
        b.setToolTip(tip)
        b.setCheckable(checkable)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if slot is not None:
            b.clicked.connect(lambda _c=False: slot())
        self.lay.addWidget(b)
        return b

    def sep(self):
        s = QFrame()
        s.setFrameShape(QFrame.Shape.VLine)
        apply_style(s, "color:@line;")
        self.lay.addWidget(s)

    def place(self):
        c = self.parentWidget()
        self.adjustSize()
        vp = c.viewport().geometry()
        w = min(self.sizeHint().width(), vp.width() - 16)
        self.setGeometry(vp.left() + (vp.width() - w) // 2, vp.top() + 10, w, self.sizeHint().height())

    def show_bar(self):
        self.place()
        end = QRect(self.geometry())
        anim.fade(self, True, 180)
        anim.slide(self, end.translated(0, -14), end, 220)        # drops in from the top edge
        self.raise_()


class _ShapesBox(QScrollArea):
    """MS Paint's shapes gallery: a framed strip of shape buttons, SHAPES_VISIBLE wide, scrolled sideways with its
    thin scrollbar or the mouse wheel. The selected shape is kept in view."""
    CELL = 28

    def __init__(self):
        super().__init__()
        self.setObjectName("shapesBox")
        inner = QWidget()
        inner.setObjectName("shapesInner")
        self.row = QHBoxLayout(inner)
        self.row.setContentsMargins(2, 2, 2, 2)
        self.row.setSpacing(2)
        self.setWidget(inner)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.horizontalScrollBar().setSingleStep(self.CELL + 2)
        self.setFixedWidth(SHAPES_VISIBLE * (self.CELL + 2) + 2 + 4)
        self.setFixedHeight(self.CELL + 4 + 7)
        self.setToolTip("Shapes — scroll for more")
        apply_style(self, "QScrollArea#shapesBox{border:1px solid @line; border-radius:6px; background:transparent;}"
                          "QWidget#shapesInner{background:transparent;}"
                          "QScrollBar:horizontal{height:5px; background:transparent; margin:0 3px 1px 3px;}"
                          "QScrollBar::handle:horizontal{background:@line; border-radius:2px; min-width:14px;}"
                          "QScrollBar::handle:horizontal:hover{background:@accent;}"
                          "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal{width:0;}"
                          "QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal{background:none;}")

    def add(self, icon, tip) -> QToolButton:
        b = QToolButton()
        b.setIcon(newIcon(icon))
        b.setIconSize(QSize(20, 20))
        b.setFixedSize(self.CELL, self.CELL)
        b.setToolTip(tip)
        b.setCheckable(True)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setProperty("aio_icon", icon)              # re-tinted when the theme changes
        self.row.addWidget(b)
        return b

    def show_btn(self, b):
        later(0, self, lambda: self.ensureWidgetVisible(b, 0, 0))   # after the first layout

    def wheelEvent(self, e):
        d = e.angleDelta()
        step = d.x() if abs(d.x()) > abs(d.y()) else d.y()
        if step:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - (1 if step > 0 else -1) * bar.singleStep())
        e.accept()                                   # never zooms the image underneath


class _Swatch(QToolButton):
    def __init__(self, color, slot):
        super().__init__()
        self.color = color
        self.setFixedSize(20, 20)
        self.setCheckable(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip(color)
        self.clicked.connect(lambda _c=False: slot(self.color))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(3, 3, -3, -3)
        if self.isChecked():
            p.setPen(QPen(QColor(self.palette().highlight().color()), 2))
            p.drawEllipse(QRectF(self.rect()).adjusted(1, 1, -1, -1))
        p.setPen(QPen(QColor(0, 0, 0, 90), 1))
        p.setBrush(QColor(self.color))
        p.drawEllipse(r)


# ====================================================================================================== sketch
class SketchMode:
    def __init__(self, app):
        self.app = app
        self.c = app.canvas
        self.active = False
        self.tool = app.settings.value("sketch/tool", "pen")
        if self.tool not in {t[0] for t in TOOLS} or self.tool == "pick":
            self.tool = "pen"
        self._prev_tool = "pen"                     # the colour picker returns here (as in MS Paint)
        self._pick_img = None
        self.color = app.settings.value("sketch/color", PALETTE[0])
        self.size = int(app.settings.value("sketch/size", 6))
        self.esize = int(app.settings.value("sketch/esize", 24))
        self.fill = app.settings.value("sketch/fill", False, type=bool)
        self.els, self.redo_stack = [], []
        self.layer = None
        self.key = None
        self.drag = None
        self._saved_hidden = None
        self._shortcuts = []
        self._build_bar()

    # ---- UI
    def _build_bar(self):
        b = self.bar = _Bar(self.c, "✎ Sketch")
        self.group = QButtonGroup(b)
        self.group.setExclusive(True)
        self.tool_btns = {}
        self.shapes_box = None
        for key, label, icon, tip in TOOLS:
            if key in SHAPES:
                if self.shapes_box is None:
                    self.shapes_box = _ShapesBox()
                    b.lay.addWidget(self.shapes_box)
                t = self.shapes_box.add(icon, f"{label} — {tip}")
                t.clicked.connect(lambda _c=False, k=key: self.set_tool(k))
            else:
                t = b.btn(icon, f"{label} — {tip}", lambda k=key: self.set_tool(k), checkable=True)
            self.group.addButton(t)
            self.tool_btns[key] = t
        b.sep()
        self.swatches = []
        for col in PALETTE:
            s = _Swatch(col, self.set_color)
            b.lay.addWidget(s)
            self.swatches.append(s)
        self.custom = b.btn(None, "Any colour…", self._pick_color, text="＋")
        b.sep()
        self.size_lbl = QLabel()
        self.size_lbl.setMinimumWidth(48)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 120)
        self.slider.setFixedWidth(110)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.setToolTip("Size  ( [ smaller · ] bigger ) — the eraser keeps its own size")
        self.slider.valueChanged.connect(self._size_changed)
        b.lay.addWidget(self.slider)
        b.lay.addWidget(self.size_lbl)
        self.fill_btn = b.btn("fill", "Filled rectangles / ellipses / circles", self._toggle_fill, checkable=True)
        self.fill_btn.setChecked(self.fill)
        b.sep()
        self.undo_btn = b.btn("undo", "Undo (Ctrl+Z)", self.undo)
        self.redo_btn = b.btn("redo", "Redo (Ctrl+Y)", self.redo)
        b.btn("delete", "Clear the drawing on this image", self.clear)
        b.sep()
        b.btn("copy", "Copy the image as you see it — full resolution (Ctrl+C)", self.copy)
        b.btn("save", "Save the image as you see it (PNG / JPG, full resolution)", self.save_image)
        b.btn(None, "Leave sketch mode (Esc or P) — your label choice comes back", self.toggle, text="Done")
        self.set_tool(self.tool if self.tool in self.tool_btns else "pen")
        self.set_color(self.color)

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self.color), self.app, "Sketch colour",
                                  QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self.set_color(c.name(QColor.NameFormat.HexArgb) if c.alpha() < 255 else c.name())

    def set_color(self, col):
        self.color = col
        self.app.settings.setValue("sketch/color", col)
        for s in self.swatches:
            s.setChecked(s.color.lower() == col.lower())
        apply_style(self.custom, f"QToolButton{{color:{QColor(col).name()}; font-weight:700;}}")
        if self.tool == "erase":
            self.set_tool("pen")
        self._cursor()

    def set_tool(self, key):
        if key == "pick" and self.tool != "pick":
            self._prev_tool = self.tool
        self.tool = key
        if key != "pick":                           # a one-shot tool: never restored on the next start
            self.app.settings.setValue("sketch/tool", key)
        self.tool_btns[key].setChecked(True)
        self.slider.blockSignals(True)
        self.slider.setValue(self.esize if key == "erase" else self.size)
        self.slider.setEnabled(key != "pick")
        if key in SHAPES and self.shapes_box is not None:
            self.shapes_box.show_btn(self.tool_btns[key])        # a shape chosen by key / setting scrolls into view
        self.slider.blockSignals(False)
        self._size_label()
        self._cursor()

    def _size_changed(self, v):
        if self.tool == "erase":
            self.esize = v
            self.app.settings.setValue("sketch/esize", v)
        else:
            self.size = v
            self.app.settings.setValue("sketch/size", v)
        self._size_label()
        self._cursor()

    def _size_label(self):
        self.size_lbl.setText(f"{self.esize if self.tool == 'erase' else self.size} px")

    def _toggle_fill(self):
        self.fill = self.fill_btn.isChecked()
        self.app.settings.setValue("sketch/fill", self.fill)

    def _bump(self, d):
        self.slider.setValue(max(1, min(120, self.slider.value() + d)))

    # ---- cursor: the real brush size at the current zoom
    def _cursor(self):
        if not self.active:
            return
        if self.tool in ("pen", "hl", "erase"):
            zoom = abs(self.c.transform().m11()) or 1.0
            w = (self.esize if self.tool == "erase" else self.size * (3 if self.tool == "hl" else 1)) * zoom
            d = int(max(6, min(256, w)))
            pm = QPixmap(d + 4, d + 4)
            pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(QPen(QColor(0, 0, 0, 200), 1.5))
            p.drawEllipse(QRectF(2, 2, d, d))
            p.setPen(QPen(QColor(255, 255, 255, 230) if self.tool == "erase" else QColor(self.color), 1))
            p.drawEllipse(QRectF(3, 3, d - 2, d - 2))
            p.end()
            self.c.viewport().setCursor(QCursor(pm, (d + 4) // 2, (d + 4) // 2))
        elif self.tool == "text":
            self.c.viewport().setCursor(Qt.CursorShape.IBeamCursor)
        elif self.tool == "pick":
            self.c.viewport().setCursor(_dropper_cursor(self.color))
        else:
            self.c.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def _sample(self, pt) -> QColor | None:
        """The colour shown at a scene point: the drawing over the image, blended as on screen."""
        pix = self.c.pixmap_item.pixmap()
        x, y = int(pt.x()), int(pt.y())
        if not (0 <= x < pix.width() and 0 <= y < pix.height()):
            return None
        base = pix.toImage().pixelColor(x, y) if self._pick_img is None else self._pick_img.pixelColor(x, y)
        L = self.layer
        if L is not None:
            lx, ly = min(L.img.width() - 1, int(x * L.s)), min(L.img.height() - 1, int(y * L.s))
            top = L.img.pixelColor(lx, ly)
            a = top.alphaF()
            if a > 0:
                return QColor.fromRgbF(top.redF() * a + base.redF() * (1 - a),
                                       top.greenF() * a + base.greenF() * (1 - a),
                                       top.blueF() * a + base.blueF() * (1 - a))
        return QColor(base.red(), base.green(), base.blue())

    def _pick(self, pt):
        self._pick_img = self.c.pixmap_item.pixmap().toImage()       # one conversion per click, not per pixel
        try:
            col = self._sample(pt)
        finally:
            self._pick_img = None
        back = self._prev_tool if self._prev_tool in self.tool_btns and self._prev_tool != "pick" else "pen"
        if col is not None:
            self.set_color(col.name())
        self.set_tool(back)

    # ---- enter / leave
    def toggle(self):
        if self.active:
            self.leave()
        else:
            self.enter()

    def enter(self):
        app = self.app
        if app.session is None or self.c.pixmap_item.pixmap().isNull():
            app.statusBar().showMessage("Open an image first", 4000)
            return
        if getattr(app, "roi", None) is not None and app.roi.active:
            app.roi.leave()
        self.active = True
        self.c.clearSelection()
        self.c.set_editable(False, "sketch mode")
        self._saved_hidden = set(app.hidden_labels)
        app.toggle_all_labels(False)                     # a clean image; tick labels in the Labels panel as wanted
        app.a.undo_edit.setEnabled(False)                # Ctrl+Z / Ctrl+Y belong to the drawing now
        app.a.redo_edit.setEnabled(False)
        for seq, fn in (("Ctrl+Z", self.undo), ("Ctrl+Y", self.redo), ("Ctrl+Shift+Z", self.redo),
                        ("Ctrl+C", self.copy), ("[", lambda: self._bump(-2)), ("]", lambda: self._bump(2)),
                        ("Escape", self.leave)):
            sc = QShortcut(QKeySequence(seq), app)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)
        self.c.viewport().installEventFilter(self._filter)
        self.image_changed(app.current_key)
        self.bar.show_bar()
        self._cursor()
        app.statusBar().showMessage("✎ Sketch — draw on the image · labels are hidden: tick the ones you want in "
                                    "Labels · Ctrl+C copies, Esc leaves", 7000)

    def leave(self):
        if not self.active:
            return
        app = self.app
        self._save()
        self.active = False
        for sc in self._shortcuts:
            sc.setEnabled(False)
            sc.deleteLater()
        self._shortcuts = []
        self.c.viewport().removeEventFilter(self._filter)
        self._drop_layer()
        anim.fade(self.bar, False, 150)
        if self._saved_hidden is not None:                # the label choice from before the sketch
            labels = {app.unique_labels.item(i).data(Qt.ItemDataRole.UserRole)
                      for i in range(app.unique_labels.count())} | set(app.visible_labels)
            app.hidden_labels = set(self._saved_hidden)
            app.visible_labels = labels - app.hidden_labels
            app.refresh_label_list()
            app.load_canvas_shapes()
            app.update_annotations_list()
        self._saved_hidden = None
        app.inline.refresh()
        app.inline._update_actions()
        self.c._apply_mode_cursor()

    # ---- image switching
    def _drop_layer(self):
        if self.layer is not None:
            if self.layer.scene() is not None:
                self.layer.scene().removeItem(self.layer)
            self.layer = None

    def image_changed(self, key):
        """Called after an image was shown (also while in sketch mode)."""
        if not self.active:
            return
        if self.key is not None and self.key != key:
            self._save()
        self.key = key
        self._drop_layer()
        pix = self.c.pixmap_item.pixmap()
        if pix.isNull() or key is None:
            return
        self.c.set_editable(False, "sketch mode")
        self.layer = SketchLayer(pix.width(), pix.height())
        self.c.scene.addItem(self.layer)
        self.els = self._store().load(key) if self._store() else []
        self.redo_stack = []
        self.layer.render_all(self.els)
        self._buttons()

    def _store(self):
        s = self.app.session
        if s is None:
            return None
        if getattr(s, "_sketches", None) is None:
            s._sketches = SketchStore(s.root)
        return s._sketches

    def _save(self):
        if self.key is not None and self._store() is not None:
            try:
                self._store().save(self.key, self.els)
            except OSError:
                pass

    def _buttons(self):
        self.undo_btn.setEnabled(bool(self.els))
        self.redo_btn.setEnabled(bool(self.redo_stack))

    # ---- editing
    def undo(self):
        if self.els:
            self.redo_stack.append(self.els.pop())
            self.layer.render_all(self.els)
            self._buttons()
            self._save()

    def redo(self):
        if self.redo_stack:
            el = self.redo_stack.pop()
            self.els.append(el)
            self.layer.commit(el)
            self._buttons()
            self._save()

    def clear(self):
        if self.els:
            self.redo_stack = list(reversed(self.els))       # Ctrl+Y brings it all back, one step at a time
            self.els = []
            self.layer.render_all([])
            self._buttons()
            self._save()

    def _add(self, el):
        self.els.append(el)
        self.redo_stack = []
        self._buttons()
        self._save()

    # ---- export
    def render_view(self) -> QImage:
        """The image at full resolution with what is visible on it (labels you ticked + the drawing)."""
        sc = self.c.scene
        pix = self.c.pixmap_item.pixmap()
        out = QImage(pix.width(), pix.height(), QImage.Format.Format_ARGB32_Premultiplied)
        out.fill(Qt.GlobalColor.black)
        p = QPainter(out)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        sc.render(p, QRectF(0, 0, pix.width(), pix.height()), QRectF(0, 0, pix.width(), pix.height()))
        p.end()
        return out

    def copy(self):
        if not self.active:
            return
        QApplication.clipboard().setImage(self.render_view())
        self.app.statusBar().showMessage("Copied the image with your drawing (full resolution)", 4000)

    def save_image(self):
        smp = self.app.session.by_key.get(self.key) if self.app.session else None
        start = self.app.settings.value("sketch/dir", "") or str(Path(smp.key).parent if smp else "")
        name = (Path(smp.name).stem if smp else "image") + "_sketch.png"
        f, _ = QFileDialog.getSaveFileName(self.app, "Save sketch", str(Path(start) / name),
                                           "PNG (*.png);;JPEG (*.jpg)")
        if not f:
            return
        self.app.settings.setValue("sketch/dir", str(Path(f).parent))
        img = self.render_view()
        if f.lower().endswith((".jpg", ".jpeg")):
            img = img.convertToFormat(QImage.Format.Format_RGB888)
        if img.save(f, quality=92):
            self.app.statusBar().showMessage(f"Saved {f}", 5000)
        else:
            QMessageBox.warning(self.app, "Sketch", f"Could not save {f}")

    # ---- mouse
    @property
    def _filter(self):
        if not hasattr(self, "_ef"):
            self._ef = _Filter(self)
        return self._ef

    def _scene_pt(self, pos) -> QPointF:
        p = self.c.mapToScene(pos)
        pix = self.c.pixmap_item.pixmap()
        return QPointF(min(max(p.x(), -2000), pix.width() + 2000), min(max(p.y(), -2000), pix.height() + 2000))

    def press(self, e) -> bool:
        if e.button() != Qt.MouseButton.LeftButton or self.layer is None or getattr(self.c, "_space_pan", False):
            return False
        pt = self._scene_pt(e.position().toPoint())
        self._press_view = e.position().toPoint()
        if self.tool == "pick":                     # MS Paint: take the colour, back to the last tool, draw nothing
            self._pick(pt)
            return True
        if self.tool == "text":
            text, ok = QInputDialog.getMultiLineText(self.app, "Text", "Text to put on the image:")
            if ok and text.strip():
                el = {"t": "text", "c": self.color, "w": self.size, "pts": [[pt.x(), pt.y()]], "text": text.strip()}
                self.layer.commit(el)
                self._add(el)
            return True
        w = self.esize if self.tool == "erase" else self.size
        el = {"t": self.tool, "c": self.color, "w": w, "pts": [[pt.x(), pt.y()]]}
        if self.tool in BOXES:
            el["fill"] = self.fill
        if self.tool in SHAPES:
            el["pts"].append([pt.x(), pt.y()])
        self.drag = el
        self.layer.live = el
        return True

    def move(self, e) -> bool:
        if self.drag is None:
            if e.buttons() == Qt.MouseButton.NoButton:
                self._cursor_if_zoomed()
                return True                 # hovering: keep the brush cursor (no shape hover / hand cursor)
            return False                    # Space-pan / other buttons: the canvas handles them
        el = self.drag
        pt = self._scene_pt(e.position().toPoint())
        before = el_bounds(el)
        if el["t"] in ("pen", "hl", "erase"):
            last = el["pts"][-1]
            if (pt.x() - last[0]) ** 2 + (pt.y() - last[1]) ** 2 < 0.25 / max(1e-6, abs(self.c.transform().m11())) ** 2:
                return True
            el["pts"].append([round(pt.x(), 2), round(pt.y(), 2)])
            if el["t"] == "erase":                         # erasing shows at once
                seg = {"t": "erase", "w": el["w"], "pts": el["pts"][-2:]}
                self.layer.commit(seg)
                return True
        else:
            a = el["pts"][0]
            x, y = pt.x(), pt.y()
            if el["t"] == "circle" or e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                dx, dy = x - a[0], y - a[1]
                if el["t"] in BOXES:
                    s = max(abs(dx), abs(dy))
                    x, y = a[0] + math.copysign(s, dx or 1), a[1] + math.copysign(s, dy or 1)
                else:
                    ang = round(math.atan2(dy, dx) / (math.pi / 4)) * (math.pi / 4)
                    r = math.hypot(dx, dy)
                    x, y = a[0] + r * math.cos(ang), a[1] + r * math.sin(ang)
            el["pts"][1] = [round(x, 2), round(y, 2)]
        self.layer.update(before.united(el_bounds(el)))
        return True

    def release(self, e) -> bool:
        if self.drag is None or e.button() != Qt.MouseButton.LeftButton:
            QTimer.singleShot(0, self._cursor)           # after a pan the canvas resets the cursor
            return False
        el, self.drag = self.drag, None
        self.layer.live = None
        tiny = el["t"] in SHAPES and len(el["pts"]) == 2 and \
            math.hypot(el["pts"][1][0] - el["pts"][0][0], el["pts"][1][1] - el["pts"][0][1]) < 2
        pv = getattr(self, "_press_view", None)
        clicked = pv is not None and (e.position().toPoint() - pv).manhattanLength() < 4
        self._click_el = None
        if tiny:
            self.layer.update(el_bounds(el))
            return True
        if el["t"] != "erase" or len(el["pts"]) == 1:   # an eraser click erases a dot (same as on re-render)
            self.layer.commit(el)
        else:
            self.layer.update(el_bounds(el))
        self._add(el)
        if clicked:
            self._click_el = el                          # a plain click: undone again if it was half a double-click
        return True

    def double_click(self, e) -> bool:
        """Double-click zooms in / back to fit, as everywhere else. The dot the first click of the double-click drew
        is taken back first, so zooming never leaves a mark."""
        if self.drag is not None or self.layer is None:
            return True
        el = getattr(self, "_click_el", None)
        self._click_el = None
        if el is not None and self.els and self.els[-1] is el:
            self.els.pop()
            self.layer.render_all(self.els)
            self._buttons()
            self._save()
        self.c.toggle_zoom(e.position().toPoint())
        QTimer.singleShot(0, self._cursor)
        return True

    def _cursor_if_zoomed(self):
        z = round(abs(self.c.transform().m11()), 3)
        reset = self.c.viewport().cursor().shape() in (
            Qt.CursorShape.ArrowCursor, Qt.CursorShape.OpenHandCursor, Qt.CursorShape.ClosedHandCursor,
            Qt.CursorShape.PointingHandCursor, Qt.CursorShape.SizeAllCursor)          # the canvas took it back
        if reset or z != getattr(self, "_z", None):
            self._z = z
            self._cursor()


class _Filter(QWidget):
    """Viewport event filter: left-button work goes to the mode, everything else (wheel zoom, Space pan, middle
    button) to the canvas as usual."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def eventFilter(self, obj, e):
        t = e.type()
        m = self.mode
        if t == QEvent.Type.MouseButtonPress:
            return m.press(e)
        if t == QEvent.Type.MouseMove:
            return m.move(e)
        if t == QEvent.Type.MouseButtonRelease:
            return m.release(e)
        if t == QEvent.Type.MouseButtonDblClick and e.button() == Qt.MouseButton.LeftButton:
            if getattr(m.c, "_space_pan", False):
                return False
            return m.double_click(e)
        if t == QEvent.Type.Wheel:
            QTimer.singleShot(0, getattr(m, "_cursor", lambda: None))
        if t == QEvent.Type.Resize:
            QTimer.singleShot(0, m.bar.place)
        return False


# ====================================================================================================== ROI
class RoiOverlay(QGraphicsItem):
    """Snipping-tool look: everything outside the boxes dimmed, boxes with a white / dark dashed border and corner
    marks, the box being drawn with its live coordinates."""

    def __init__(self, w, h):
        super().__init__()
        self.setZValue(Z_LAYER + 1)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.W, self.H = w, h
        self.rois: list = []
        self.live = None
        self.hover = -1

    def boundingRect(self):
        return QRectF(-4000, -4000, self.W + 8000, self.H + 8000)

    def paint(self, p, _opt, _w=None):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale = 1.0 / max(1e-6, abs(p.worldTransform().m11()))
        img = QRectF(0, 0, self.W, self.H)
        boxes = [QRectF(QPointF(r[0], r[1]), QPointF(r[2], r[3])) for r in self.rois]
        if self.live is not None:
            boxes.append(self.live)
        dim = QPainterPath()
        dim.addRect(img)
        for b in boxes:
            hole = QPainterPath()
            hole.addRect(b)
            dim = dim.subtracted(hole)
        p.fillPath(dim, QColor(0, 0, 0, 120 if boxes else 70))
        for i, b in enumerate(boxes):
            live = self.live is not None and i == len(boxes) - 1
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(0, 0, 0, 200), 3 * scale))
            p.drawRect(b)
            pen = QPen(QColor("#ffffff") if not (i == self.hover and not live) else QColor("#ff3b30"), 1.4 * scale,
                       Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawRect(b)
            k = min(14 * scale, b.width() / 3, b.height() / 3)
            p.setPen(QPen(QColor("#00c7be") if not live else QColor("#ffcc00"), 3 * scale))
            for cx, cy, dx, dy in ((b.left(), b.top(), 1, 1), (b.right(), b.top(), -1, 1), (b.left(), b.bottom(), 1, -1),
                                   (b.right(), b.bottom(), -1, -1)):
                p.drawLine(QPointF(cx, cy), QPointF(cx + dx * k, cy))
                p.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * k))
            text = (f"{b.left():.2f}, {b.top():.2f} → {b.right():.2f}, {b.bottom():.2f}   ·   "
                    f"{b.width():.2f} × {b.height():.2f}") if live else f"#{i + 1}"
            f = QFont()
            f.setPixelSize(max(1, int(12 * scale)))
            f.setBold(True)
            p.setFont(f)
            fm = p.fontMetrics()
            tw, th = fm.horizontalAdvance(text) + 10 * scale, fm.height() + 4 * scale
            y = b.top() - th - 4 * scale if b.top() - th - 4 * scale > 0 else b.bottom() + 4 * scale
            box = QRectF(b.left(), y, tw, th)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 190))
            p.drawRoundedRect(box, 4 * scale, 4 * scale)
            p.setPen(QColor("#ffffff"))
            p.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)


class RoiMode:
    def __init__(self, app):
        self.app = app
        self.c = app.canvas
        self.active = False
        self.overlay = None
        self.key = None
        self.drag = None
        self._shortcuts = []
        self._build_bar()

    def _build_bar(self):
        b = self.bar = _Bar(self.c, "⬚ ROI")
        self.info = QLabel("")
        b.lay.addWidget(self.info)
        b.sep()
        b.btn("copy", "Copy this image's ROIs (name + x1, y1, x2, y2) — Ctrl+C", self.copy, text="Copy")
        b.btn("undo", "Remove the last ROI (Esc removes it too)", self.remove_last)
        b.btn("delete", "Remove all ROIs of this image", self.clear)
        b.sep()
        self.fmt = QComboBox()
        for t, k in (("CSV", "csv"), ("TXT (name x1 y1 x2 y2)", "txt"), ("JSON", "json")):
            self.fmt.addItem(t, k)
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(self.app.settings.value("roi/fmt", "csv"))))
        self.fmt.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.lay.addWidget(self.fmt)
        b.btn("export", "Export the ROIs of every image", self.export, text="Export…")
        b.btn("tool-crop-gallery", "Save the ROI crops of every image as image files", self.save_crops, text="Crops…")
        b.btn(None, "Leave ROI mode (O)", self.toggle, text="Done")

    def store(self):
        s = self.app.session
        if s is None:
            return None
        if getattr(s, "_rois", None) is None:
            s._rois = RoiStore(s.root)
        return s._rois

    def toggle(self):
        self.leave() if self.active else self.enter()

    def enter(self):
        app = self.app
        if app.session is None or self.c.pixmap_item.pixmap().isNull():
            app.statusBar().showMessage("Open an image first", 4000)
            return
        if getattr(app, "sketch", None) is not None and app.sketch.active:
            app.sketch.leave()
        self.active = True
        self.c.clearSelection()
        self.c.set_editable(False, "ROI mode")
        for seq, fn in (("Escape", self._esc), ("Ctrl+C", self.copy)):
            sc = QShortcut(QKeySequence(seq), app)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)
        self.c.viewport().installEventFilter(self._filter)
        self.image_changed(app.current_key)
        self.bar.show_bar()
        self.c.viewport().setCursor(Qt.CursorShape.CrossCursor)
        app.statusBar().showMessage("⬚ ROI — drag a box · right-click a box removes it · Ctrl+C copies · O leaves",
                                    7000)

    def leave(self):
        if not self.active:
            return
        self.active = False
        st = self.store()
        if st is not None:
            st.flush()
        for sc in self._shortcuts:
            sc.setEnabled(False)
            sc.deleteLater()
        self._shortcuts = []
        self.c.viewport().removeEventFilter(self._filter)
        self._drop()
        anim.fade(self.bar, False, 150)
        self.app.inline.refresh()
        self.c._apply_mode_cursor()

    def _drop(self):
        if self.overlay is not None and self.overlay.scene() is not None:
            self.overlay.scene().removeItem(self.overlay)
        self.overlay = None

    def image_changed(self, key):
        if not self.active:
            return
        self.key = key
        self._drop()
        pix = self.c.pixmap_item.pixmap()
        if pix.isNull() or key is None:
            return
        self.c.set_editable(False, "ROI mode")
        self.overlay = RoiOverlay(pix.width(), pix.height())
        self.overlay.rois = self.store().get(key)
        self.c.scene.addItem(self.overlay)
        self._info()

    def _info(self):
        n = len(self.overlay.rois) if self.overlay else 0
        tot, files = self.store().total() if self.store() else (0, 0)
        self.info.setText(f"{n} on this image · {tot:,} in {files:,} image(s)")
        self.bar.place()

    def _changed(self):
        self.store().set(self.key, self.overlay.rois)
        self.overlay.update()
        self._info()

    # ---- actions
    def _esc(self):
        if self.drag is not None:
            self.drag = None
            self.overlay.live = None
            self.overlay.update()
        elif self.overlay and self.overlay.rois:
            self.remove_last()
        else:
            self.leave()

    def remove_last(self):
        if self.overlay and self.overlay.rois:
            self.overlay.rois.pop()
            self._changed()

    def clear(self):
        if self.overlay and self.overlay.rois:
            self.overlay.rois = []
            self._changed()

    def _name(self, key):
        smp = self.app.session.by_key.get(key)
        return smp.name if smp else Path(key).name

    def copy(self):
        if not self.active or not self.overlay:
            return
        name = self._name(self.key)
        text = "\n".join(f"{name}\t{fmt_roi(r)}" for r in self.overlay.rois)
        QApplication.clipboard().setText(text)
        self.app.statusBar().showMessage(f"Copied {len(self.overlay.rois)} ROI(s) of {name}", 4000)

    def _all(self):
        st = self.store()
        st.flush()
        out = []
        for rel, rois in sorted(st.data.items()):
            for i, r in enumerate(rois, 1):
                out.append((rel, Path(rel).name, i, r))
        return out

    def export(self):
        rows = self._all()
        if not rows:
            QMessageBox.information(self.app, "ROI", "No ROIs yet.")
            return
        fmt = self.fmt.currentData()
        self.app.settings.setValue("roi/fmt", fmt)
        root = self.app.session.root
        f, _ = QFileDialog.getSaveFileName(self.app, "Export ROIs", str(root.parent / f"{root.name}_rois.{fmt}"),
                                           {"csv": "CSV (*.csv)", "txt": "Text (*.txt)", "json": "JSON (*.json)"}[fmt])
        if not f:
            return
        if fmt == "csv":
            import csv
            with open(f, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["image", "path", "roi", "x1", "y1", "x2", "y2", "width", "height"])
                for rel, name, i, r in rows:
                    w.writerow([name, rel, i, *(f"{v:.2f}" for v in r), f"{r[2] - r[0]:.2f}", f"{r[3] - r[1]:.2f}"])
        elif fmt == "txt":
            Path(f).write_text("\n".join(f"{name} {r[0]:.2f} {r[1]:.2f} {r[2]:.2f} {r[3]:.2f}"
                                         for _rel, name, _i, r in rows) + "\n", encoding="utf-8")
        else:
            data = {}
            for rel, name, _i, r in rows:
                data.setdefault(rel, {"image": name, "rois": []})["rois"].append([round(v, 2) for v in r])
            Path(f).write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M"), "unit": "pixels",
                                           "images": data}, indent=1), encoding="utf-8")
        self.app.statusBar().showMessage(f"Exported {len(rows):,} ROI(s) to {f}", 6000)

    def save_crops(self):
        rows = self._all()
        if not rows:
            QMessageBox.information(self.app, "ROI", "No ROIs yet.")
            return
        d = QFileDialog.getExistingDirectory(self.app, "Save the ROI crops into",
                                             self.app.settings.value("roi/crops", ""))
        if not d:
            return
        self.app.settings.setValue("roi/crops", d)
        root = self.app.session.root
        n = 0
        from AIO_QC.core.io import decode_image
        import cv2
        for rel, name, i, r in rows:
            img = decode_image(root / rel)
            if img is None:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in r)
            crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0:
                continue
            ok, buf = cv2.imencode(".png", crop)
            if ok:
                (Path(d) / f"{Path(name).stem}_roi{i}.png").write_bytes(buf.tobytes())
                n += 1
        self.app.statusBar().showMessage(f"Saved {n:,} crop(s) to {d}", 6000)

    # ---- mouse
    @property
    def _filter(self):
        if not hasattr(self, "_ef"):
            self._ef = _Filter(self)
        return self._ef

    def _pt(self, pos):
        p = self.c.mapToScene(pos)
        o = self.overlay
        return QPointF(min(max(p.x(), 0.0), o.W), min(max(p.y(), 0.0), o.H))

    def _hit(self, pt) -> int:
        for i in range(len(self.overlay.rois) - 1, -1, -1):
            r = self.overlay.rois[i]
            if r[0] <= pt.x() <= r[2] and r[1] <= pt.y() <= r[3]:
                return i
        return -1

    def press(self, e) -> bool:
        if self.overlay is None or getattr(self.c, "_space_pan", False):
            return False
        pt = self._pt(e.position().toPoint())
        if e.button() == Qt.MouseButton.RightButton:
            i = self._hit(pt)
            if i >= 0:
                del self.overlay.rois[i]
                self._changed()
            return True
        if e.button() != Qt.MouseButton.LeftButton:
            return False
        self.drag = pt
        self.overlay.live = QRectF(pt, pt)
        self.overlay.update()
        return True

    def move(self, e) -> bool:
        if self.overlay is None:
            return False
        pt = self._pt(e.position().toPoint())
        if self.drag is None:
            h = self._hit(pt)
            if h != self.overlay.hover:
                self.overlay.hover = h
                self.overlay.update()
            if e.buttons() == Qt.MouseButton.NoButton:
                self.c.viewport().setCursor(Qt.CursorShape.CrossCursor)
                return True
            return False
        self.overlay.live = QRectF(self.drag, pt).normalized()
        self.overlay.update()
        return True

    def release(self, e) -> bool:
        if self.drag is None or e.button() != Qt.MouseButton.LeftButton:
            QTimer.singleShot(0, self._cursor)
            return False
        r = self.overlay.live
        self.drag = None
        self.overlay.live = None
        if r is not None and r.width() >= 1 and r.height() >= 1:
            self.overlay.rois.append([round(r.left(), 2), round(r.top(), 2), round(r.right(), 2),
                                      round(r.bottom(), 2)])
            self._changed()
            self.app.statusBar().showMessage(f"ROI #{len(self.overlay.rois)}: {fmt_roi(self.overlay.rois[-1])}", 5000)
        else:
            self.overlay.update()
        return True

    def _cursor(self):
        if self.active:
            self.c.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def double_click(self, e) -> bool:
        """Double-click zooms in / back to fit (a click never makes a box: boxes need a drag)."""
        if self.overlay is None:
            return True
        self.drag = None
        self.overlay.live = None
        self.overlay.update()
        self.c.toggle_zoom(e.position().toPoint())
        QTimer.singleShot(0, self._cursor)
        return True

