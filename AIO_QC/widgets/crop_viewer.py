# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Crop Viewer: one annotation's crop in a window of its own (Annotations panel ▸ right-click ▸ View crop…), next to
the image canvas instead of in it.

- The crop is cut from the image already on screen (no file is read or decoded again; with brightness /
  contrast as shown, or the original pixels) — a few KB to a few MB however large the image is.
- Padding around the shape (0 … 100 % of its size), its outline, dimming outside it; fit / 1:1, wheel zoom at
  the pointer, drag to pan, double-click to fit. Smooth when enlarged, or hard pixels like the canvas's
  "Show pixels when zoomed in".
- ← / → (or the arrows in the bar) step through the image's annotations; "Follow selection" keeps it on the shape
  selected on the canvas, also across images. Copy the crop or save it as a PNG.
- Shapes without area (points, lines) get a crop around them; shapes partly outside the image are clipped; a shape
  with no usable points says so instead of failing. One window per Image QC window; its pixels are released when
  it closes.
"""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QGuiApplication, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, \
    QShortcut
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsScene,
                             QGraphicsView, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget)

from AIO_QC import utils

PADDINGS = [(0.0, "No padding"), (0.1, "10 %"), (0.25, "25 %"), (0.5, "50 %"), (1.0, "100 %")]
MIN_SIDE = 24          # px: points / lines / tiny shapes still get a visible crop
MAX_ZOOM, MIN_ZOOM = 64.0, 0.02


def shape_outline(shape: dict) -> QPainterPath | None:
    """The shape as a path in image pixels (rectangle, polygon, circle, line, linestrip, point(s)); None when it
    has no usable points."""
    pts = []
    for p in shape.get("points") or []:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        pts.append(QPointF(x, y))
    if not pts:
        return None
    st = shape.get("shape_type") or "polygon"
    path = QPainterPath()
    if st == "rectangle" and len(pts) >= 2:
        path.addRect(QRectF(pts[0], pts[1]).normalized())
    elif st == "circle" and len(pts) >= 2:
        r = math.hypot(pts[1].x() - pts[0].x(), pts[1].y() - pts[0].y())
        path.addEllipse(pts[0], r, r)
    elif st in ("point", "points") or len(pts) == 1:
        for p in pts:
            path.addEllipse(p, 1.5, 1.5)
    else:
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        if st == "polygon" and len(pts) >= 3:
            path.closeSubpath()
    return path


def crop_rect(path: QPainterPath, pad: float, img_w: int, img_h: int) -> QRectF | None:
    """The region to cut: the shape's bounds + padding (a share of its size), at least MIN_SIDE, inside the image."""
    b = path.boundingRect()
    w, h = max(b.width(), 1.0), max(b.height(), 1.0)
    m = max(w, h) * pad
    r = b.adjusted(-m, -m, m, m)
    if r.width() < MIN_SIDE:
        r.adjust(-(MIN_SIDE - r.width()) / 2, 0, (MIN_SIDE - r.width()) / 2, 0)
    if r.height() < MIN_SIDE:
        r.adjust(0, -(MIN_SIDE - r.height()) / 2, 0, (MIN_SIDE - r.height()) / 2)
    r = QRectF(math.floor(r.left()), math.floor(r.top()), math.ceil(r.width()), math.ceil(r.height()))
    img = QRectF(0, 0, img_w, img_h)
    if not b.adjusted(-0.5, -0.5, 0.5, 0.5).intersects(img):
        return None                                  # the shape is wholly outside the image
    r = r.intersected(img)
    # a tiny shape at an edge: grow the clipped crop back to MIN_SIDE towards the inside of the image
    for horiz in (True, False):
        size, full = (r.width(), img_w) if horiz else (r.height(), img_h)
        need = min(MIN_SIDE, full) - size
        if need > 0:
            lo = r.left() if horiz else r.top()
            new_lo = max(0.0, min(lo - need / 2, full - (size + need)))
            if horiz:
                r = QRectF(new_lo, r.top(), size + need, r.height())
            else:
                r = QRectF(r.left(), new_lo, r.width(), size + need)
    return r if r.width() >= 1 and r.height() >= 1 else None


class _View(QGraphicsView):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#24262A"))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.fitted = True

    def zoom(self) -> float:
        return self.transform().m11()

    def set_zoom(self, z: float):
        z = max(MIN_ZOOM, min(MAX_ZOOM, z))
        f = z / max(self.zoom(), 1e-9)
        self.scale(f, f)
        self.fitted = False
        self.owner._zoom_changed()

    def fit(self):
        r = self.scene().itemsBoundingRect()
        if r.isEmpty():
            return
        self.resetTransform()
        self.fitInView(r.adjusted(-2, -2, 2, 2), Qt.AspectRatioMode.KeepAspectRatio)
        if self.zoom() > 16:                       # a tiny crop: not more than 16× when fitting
            self.resetTransform()
            self.scale(16, 16)
        self.fitted = True
        self.owner._zoom_changed()

    def wheelEvent(self, e):
        d = e.angleDelta().y()
        if d:
            self.set_zoom(self.zoom() * (1.15 ** (d / 120)))
        e.accept()

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.fitted:
            self.fit()


class CropViewer(QWidget):
    """Separate window (not a dialog): stays open next to the main window while you work."""

    def __init__(self, app):
        super().__init__(app, Qt.WindowType.Window)
        self.app = app
        self.setObjectName("cropViewer")
        self.setWindowTitle("Crop Viewer")
        self.setWindowIcon(utils.newIcon("zoom"))
        self.index: int | None = None
        self.key: str | None = None
        st = app.settings
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        # ---- bar
        bar = QWidget()
        bar.setObjectName("cropBar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(8, 6, 8, 6)
        bl.setSpacing(6)
        self.prev_btn = self._tool("prev", "Previous annotation (←)", lambda: self.step(-1))
        self.next_btn = self._tool("next", "Next annotation (→)", lambda: self.step(1))
        self.title = QLabel("")
        f = self.title.font()
        f.setBold(True)
        self.title.setFont(f)
        self.pad = QComboBox()
        for v, t in PADDINGS:
            self.pad.addItem(t, v)
        self.pad.setToolTip("Space around the shape (share of its size)")
        self.pad.setCurrentIndex(max(0, self.pad.findData(float(st.value("crop_viewer/pad", 0.25)))))
        self.pad.currentIndexChanged.connect(lambda _i: (st.setValue("crop_viewer/pad", self.pad.currentData()),
                                                         self.refresh(refit=True)))
        self.outline = QCheckBox("Outline")
        self.outline.setChecked(st.value("crop_viewer/outline", True, type=bool))
        self.dim = QCheckBox("Dim outside")
        self.dim.setChecked(st.value("crop_viewer/dim", False, type=bool))
        self.adjusted = QCheckBox("As shown")
        self.adjusted.setToolTip("With the brightness / contrast of the canvas (off: the original pixels)")
        self.adjusted.setChecked(st.value("crop_viewer/adjusted", True, type=bool))
        self.follow = QCheckBox("Follow selection")
        self.follow.setToolTip("Show the shape selected on the canvas — also on the next images")
        self.follow.setChecked(st.value("crop_viewer/follow", True, type=bool))
        for cb, k in ((self.outline, "outline"), (self.dim, "dim"), (self.adjusted, "adjusted"),
                      (self.follow, "follow")):
            cb.toggled.connect(lambda v, k=k: (st.setValue(f"crop_viewer/{k}", bool(v)), self.refresh()))
        self.zoom_lbl = QLabel("")
        self.zoom_lbl.setMinimumWidth(48)
        self.zoom_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        fit = self._tool("fit-window", "Fit (double-click the crop)", lambda: self.view.fit())
        one = self._tool("", "Actual pixels (100 %)", lambda: self.view.set_zoom(1.0))
        one.setText("1:1")
        copy = self._tool("copy", "Copy the crop (Ctrl+C)", self.copy_crop)
        save = self._tool("save", "Save the crop as an image…", self.save_crop)
        for w in (self.prev_btn, self.next_btn, self.title):
            bl.addWidget(w)
        bl.addStretch(1)
        for w in (self.pad, self.outline, self.dim, self.adjusted, self.follow, self.zoom_lbl, fit, one, copy, save):
            bl.addWidget(w)
        lay.addWidget(bar)
        # ---- view
        self.view = _View(self)
        self.pix_item = QGraphicsPixmapItem()
        self.pix_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.dim_item = QGraphicsPathItem()
        self.dim_item.setPen(QPen(Qt.PenStyle.NoPen))
        self.dim_item.setBrush(QBrush(QColor(0, 0, 0, 140)))
        self.out_item = QGraphicsPathItem()
        self.out_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        for it in (self.pix_item, self.dim_item, self.out_item):
            self.view.scene().addItem(it)
        lay.addWidget(self.view, 1)
        self.message = QLabel("")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setWordWrap(True)
        utils.style(self.message, "color:@muted; padding:24px;")
        self.message.setVisible(False)
        lay.addWidget(self.message, 1)
        # ---- info
        self.info = QLabel("")
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        utils.style(self.info, "color:@muted; padding:6px 10px; border-top:1px solid @line;")
        lay.addWidget(self.info)
        for seq, fn in (("Left", lambda: self.step(-1)), ("Right", lambda: self.step(1)),
                        ("Ctrl+C", self.copy_crop), ("Escape", self.close), ("F", lambda: self.view.fit()),
                        ("1", lambda: self.view.set_zoom(1.0))):
            QShortcut(QKeySequence(seq), self, activated=fn)
        geo = st.value("crop_viewer/geometry")
        if geo is None or not self.restoreGeometry(geo):
            self.resize(720, 560)

    def _tool(self, icon, tip, fn):
        b = QToolButton()
        b.setAutoRaise(True)
        if icon:
            b.setIcon(utils.newIcon(icon))
        b.setToolTip(tip)
        b.setAccessibleName(tip)
        b.clicked.connect(lambda _c=False: fn())
        return b

    # ------------------------------------------------------------------ content
    def shapes(self) -> list:
        ann = getattr(self.app, "current_ann", None)
        return list(ann.shapes) if ann is not None else []

    def show_crop(self, index: int):
        self.index, self.key = int(index), self.app.current_key
        from AIO_QC.widgets.window_shelf import bring_back
        bring_back(self)                       # minimised / tucked away / behind: back in front
        self.refresh(refit=True)

    def step(self, d: int):
        n = len(self.shapes())
        if not n:
            return
        i = 0 if self.index is None else (self.index + d) % n
        self.index = i
        self.refresh(refit=True)
        if self.follow.isChecked():                 # keep the canvas in step (no zoom change there)
            self.app.canvas.select_shape(i, emit=False)
            self.app.annotations_list.blockSignals(True)
            self.app.annotations_list.setCurrentRow(next((r for r in range(self.app.annotations_list.count())
                                                          if self.app.annotations_list.item(r).data(
                                                              Qt.ItemDataRole.UserRole) == i), -1))
            self.app.annotations_list.blockSignals(False)

    def _say(self, text: str):
        self.pix_item.setPixmap(QPixmap())
        self.out_item.setPath(QPainterPath())
        self.dim_item.setPath(QPainterPath())
        self.view.setVisible(False)
        self.message.setText(text)
        self.message.setVisible(True)
        self.info.setText("")
        self.title.setText("")
        self._crop = None

    def refresh(self, refit: bool = False):
        """Re-cut the crop of `index` from the image now on screen. Cheap: one region copy of a pixmap in memory."""
        if self.isHidden():
            return
        shapes = self.shapes()
        canvas = self.app.canvas
        src = canvas.pixmap_item.pixmap() if self.adjusted.isChecked() else canvas._original
        n = len(shapes)
        self.prev_btn.setEnabled(n > 1)
        self.next_btn.setEnabled(n > 1)
        if self.index is None or not (0 <= self.index < n):
            self._say("No annotation selected.\nRight-click an annotation in the Annotations panel ▸ View crop, "
                      "or select a shape on the canvas.")
            return
        if src is None or src.isNull():
            self._say("The image could not be shown, so there is nothing to crop.")
            return
        shape = shapes[self.index]
        label = str(shape.get("label", "?"))
        st = shape.get("shape_type") or "polygon"
        self.title.setText(f"#{self.index}  {label}  ·  {st}   ({self.index + 1} / {n})")
        path = shape_outline(shape)
        if path is None:
            self._say(f"#{self.index} '{label}' has no usable points — nothing to crop.")
            self.title.setText(f"#{self.index}  {label}  ·  {st}")
            return
        r = crop_rect(path, float(self.pad.currentData() or 0), src.width(), src.height())
        if r is None:
            self._say(f"#{self.index} '{label}' lies outside the image — nothing to crop.")
            self.title.setText(f"#{self.index}  {label}  ·  {st}")
            return
        crop = src.copy(r.toRect())                     # only this region: small, whatever the image size
        self._crop = crop
        self.view.setVisible(True)
        self.message.setVisible(False)
        self.pix_item.setPixmap(crop)
        local = path.translated(-r.left(), -r.top())
        color = QColor(self.app.label_colors.get(label, QColor("#30A46C")))
        pen = QPen(color, 2)
        pen.setCosmetic(True)                            # 2 screen px at any zoom
        self.out_item.setPen(pen)
        self.out_item.setPath(local if self.outline.isChecked() else QPainterPath())
        if self.dim.isChecked() and st in ("rectangle", "polygon", "circle"):
            full = QPainterPath()
            full.addRect(QRectF(0, 0, crop.width(), crop.height()))
            self.dim_item.setPath(full.subtracted(local))
        else:
            self.dim_item.setPath(QPainterPath())
        pixel = bool(getattr(getattr(self.app, "a", None), "pixel_view", None) and self.app.a.pixel_view.isChecked())
        self.pix_item.setTransformationMode(Qt.TransformationMode.FastTransformation if pixel else
                                            Qt.TransformationMode.SmoothTransformation)
        self.view.scene().setSceneRect(QRectF(0, 0, crop.width(), crop.height()))
        b = path.boundingRect()
        desc = str(shape.get("description") or "").strip()
        name = self.app.session.by_key[self.key].name if self.app.session and self.key in self.app.session.by_key \
            else ""
        self.info.setText("  ·  ".join(x for x in (
            name, f"box {b.left():.0f}, {b.top():.0f}  {b.width():.0f} × {b.height():.0f} px",
            f"crop {crop.width()} × {crop.height()} px", f"“{desc[:120]}”" if desc else "") if x))
        self.setWindowTitle(f"Crop Viewer — #{self.index} {label}")
        if refit or self.view.fitted:
            self.view.fit()
        self._zoom_changed()

    def _zoom_changed(self):
        self.zoom_lbl.setText(f"{self.view.zoom() * 100:.0f} %")

    # ------------------------------------------------------------------ following the main window
    def on_selected(self, index: int):
        if not self.isHidden() and self.follow.isChecked() and index is not None and index >= 0:
            self.index, self.key = index, self.app.current_key
            self.refresh(refit=True)

    def on_image_changed(self):
        """Another image: stay on the same annotation number when following, else say what happened."""
        if self.isHidden():
            return
        if self.app.current_key != self.key:
            self.key = self.app.current_key
            if not self.follow.isChecked():
                self.index = None
            sel = self.app.canvas.selected_group
            if self.follow.isChecked() and sel is not None:
                self.index = sel.index
        self.refresh(refit=True)

    def on_shapes_changed(self):
        if not self.isHidden():
            self.refresh()

    # ------------------------------------------------------------------ output
    def copy_crop(self):
        crop = getattr(self, "_crop", None)
        if crop is not None and not crop.isNull():
            QGuiApplication.clipboard().setPixmap(crop)
            self.app.statusBar().showMessage("Crop copied to the clipboard", 3000)

    def save_crop(self):
        crop = getattr(self, "_crop", None)
        if crop is None or crop.isNull():
            return
        from pathlib import Path
        from AIO_QC.utils.save_dirs import ask_save
        stem = Path(self.app.session.by_key[self.key].name).stem if self.app.session and \
            self.key in self.app.session.by_key else "crop"
        label = str(self.shapes()[self.index].get("label", "shape")) if self.index is not None else "shape"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        path, _f = ask_save(self, "Save crop", f"{stem}_{self.index}_{safe}.png", "PNG (*.png);;JPEG (*.jpg)",
                            kind="images")
        if path:
            ok = crop.save(path)
            self.app.statusBar().showMessage(f"Crop saved: {path}" if ok else f"Could not save {path}", 5000)

    def closeEvent(self, e):
        self.app.settings.setValue("crop_viewer/geometry", self.saveGeometry())
        self.pix_item.setPixmap(QPixmap())               # release the pixels while closed
        self._crop = None
        super().closeEvent(e)


def open_crop(app, index: int) -> CropViewer:
    v = getattr(app, "crop_viewer", None)
    try:
        if v is not None:
            v.isHidden()
    except RuntimeError:
        v = None
    if v is None:
        v = CropViewer(app)
        app.crop_viewer = v
    v.show_crop(index)
    return v
