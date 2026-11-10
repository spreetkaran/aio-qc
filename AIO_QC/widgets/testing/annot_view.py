# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image viewers of the Model Lab.

  AnnotView    one image with ground truth (dashed) and a model's detections (model colour; errors red),
               zoom around the cursor (wheel / pinch, double-click = fit), pan (drag), optional editing:
               click to select, drag on empty space to draw a box, Delete to remove. Views can be linked so
               two of them zoom and pan together.
  LabelEditor  quick label fixes from the failure list: re-class or delete a ground-truth box, draw a missing
               one, accept a detection as ground truth, or leave the image out of scoring. Saved straight into
               the YOLO / LabelMe file (core.lab.labels_io); ← / → move through the list (auto-save).
  ABViewer     two models side by side on the same image, zoom / pan linked, one confidence slider,
               ground truth on / off, ← / → through the list, "Edit labels…" for the image shown.
  PlotViewer   one model's plotted predictions, full size with zoom / pan, model / ground truth / threshold
               controls, ← / → through the list and "Save image…" (full resolution PNG / JPG, labels in Inter).
               The viewers never touch the main player.

Saving uses a plotter(item, model, show_gt, conf) -> BGR image supplied by the Model Lab (ReportBuilder.plot_image),
so a saved plot looks exactly like the plots in the reports.
"""
import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QImage, QKeySequence, QShortcut, QFont, QPalette
from PyQt6.QtWidgets import (QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSlider,
                             QCheckBox, QListWidget, QListWidgetItem, QSplitter, QSizePolicy, QMessageBox,
                             QToolButton)

from AIO_QC.tools.frame_extractor.core.common import model_hex
from AIO_QC.widgets import motion as anim

ERR = QColor("#e34948")
SEL = QColor("#f2c94c")


def model_conf(run, mi):
    """A model's operating threshold: its scored one; the reference threshold for the reference model; else 0.25
    (never the 0.001 / 0.05 floor predictions are cached at)."""
    res = run.results[mi]
    if res is not None:
        return res.op_conf
    if run.ground == "reference" and mi == run.reference:
        return run._ref_conf()
    return 0.25


def quick_match(gt_boxes, gt_cls, preds, conf, iou_thr=0.5):
    """Greedy (by confidence) class-aware matching at IoU >= iou_thr.
    Returns (shown pred indices, pred ok flags, gt found flags)."""
    from AIO_QC.core.testing.metrics import box_iou
    preds = preds if preds is not None else np.zeros((0, 6), np.float32)
    keep = np.flatnonzero(preds[:, 4] >= conf) if len(preds) else np.zeros(0, int)
    keep = keep[np.argsort(-preds[keep, 4])] if len(keep) else keep
    gt_boxes = np.asarray(gt_boxes if gt_boxes is not None else np.zeros((0, 4)), np.float32).reshape(-1, 4)
    gt_cls = np.asarray(gt_cls if gt_cls is not None else [], int).reshape(-1)
    found = np.zeros(len(gt_boxes), bool)
    ok = np.zeros(len(keep), bool)
    if len(keep) and len(gt_boxes):
        iou = box_iou(preds[keep, :4], gt_boxes)
        for k, pi in enumerate(keep):
            cand = np.where((iou[k] >= iou_thr) & (gt_cls == int(preds[pi, 5])) & ~found)[0]
            if len(cand):
                g = cand[np.argmax(iou[k, cand])]
                found[g] = True
                ok[k] = True
    return keep, ok, found


def to_qimage(bgr):
    if bgr is None:
        return QImage()
    a = np.ascontiguousarray(bgr)
    return QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format.Format_BGR888).copy()


class AnnotView(QWidget):
    view_changed = pyqtSignal(float, float, float)      # zoom (1 = fit), centre x / y (0..1 of the image)
    box_drawn = pyqtSignal(QRectF)                      # image coordinates (edit mode)
    selection_changed = pyqtSignal(str, int)            # "gt" | "pred" | "", index

    def __init__(self, parent=None, editable=False):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(240, 160)
        self.editable = editable
        self.img = QImage()
        self.iw = self.ih = 1
        self.zoom, self.cx, self.cy = 1.0, 0.5, 0.5
        self.gt_boxes = np.zeros((0, 4), np.float32)
        self.gt_cls = np.zeros(0, int)
        self.preds = np.zeros((0, 6), np.float32)
        self.names = {}
        self.color = QColor(model_hex(0))
        self.conf = 0.25
        self.show_gt = True
        self.caption = ""
        self.sel = ("", -1)
        self._drag = None
        self._match = None
        self._zf = anim.Follower(self._set_zoom_anim, 1.0, tau=0.06, eps=1e-4, widget=self)
        self._anchor = None

    # ---- data
    def set_image(self, bgr, keep_view=False):
        self.img = to_qimage(bgr) if not isinstance(bgr, QImage) else bgr
        self.iw, self.ih = max(1, self.img.width()), max(1, self.img.height())
        if not keep_view:
            self.zoom, self.cx, self.cy = 1.0, 0.5, 0.5
            self._zf.value = self._zf.target = 1.0
        self.sel = ("", -1)
        self.update()

    def set_layers(self, gt_boxes=None, gt_cls=None, preds=None, names=None, color=None, conf=None):
        if gt_boxes is not None:
            self.gt_boxes = np.asarray(gt_boxes, np.float32).reshape(-1, 4)
            self.gt_cls = np.asarray(gt_cls if gt_cls is not None else [], int).reshape(-1)
        if preds is not None:
            self.preds = np.asarray(preds, np.float32).reshape(-1, 6)
        if names is not None:
            self.names = names
        if color is not None:
            self.color = QColor(color)
        if conf is not None:
            self.conf = conf
        self._match = None
        self.update()

    def set_conf(self, c):
        self.conf = c
        self._match = None
        self.update()

    def set_show_gt(self, on):
        self.show_gt = on
        self.update()

    def matches(self):
        if self._match is None:
            self._match = quick_match(self.gt_boxes, self.gt_cls, self.preds, self.conf)
        return self._match

    # ---- geometry
    def _fit(self):
        return min(self.width() / self.iw, self.height() / self.ih)

    def scale(self):
        return self._fit() * self.zoom

    def i2w(self, x, y):
        s = self.scale()
        return QPointF((x - self.cx * self.iw) * s + self.width() / 2, (y - self.cy * self.ih) * s + self.height() / 2)

    def w2i(self, p):
        s = self.scale()
        return QPointF((p.x() - self.width() / 2) / s + self.cx * self.iw, (p.y() - self.height() / 2) / s + self.cy * self.ih)

    def rect_w(self, b):
        return QRectF(self.i2w(b[0], b[1]), self.i2w(b[2], b[3])).normalized()

    def _clamp(self):
        if self.zoom <= 1.0:
            self.zoom, self.cx, self.cy = 1.0, 0.5, 0.5
            return
        hw = self.width() / 2 / self.scale() / self.iw
        hh = self.height() / 2 / self.scale() / self.ih
        self.cx = min(max(self.cx, hw), 1 - hw) if hw < 0.5 else 0.5
        self.cy = min(max(self.cy, hh), 1 - hh) if hh < 0.5 else 0.5

    def set_view(self, zoom, cx, cy):
        """From a linked view (no echo)."""
        self.zoom, self.cx, self.cy = zoom, cx, cy
        self._zf.value = self._zf.target = zoom
        self._clamp()
        self.update()

    def _emit_view(self):
        self.view_changed.emit(self.zoom, self.cx, self.cy)

    def zoom_at(self, factor, pos=None):
        pos = pos if pos is not None else QPointF(self.width() / 2, self.height() / 2)
        self._anchor = (QPointF(pos), self.w2i(pos))
        self._zf.set_target(max(1.0, min(40.0, self._zf.target * factor)))

    def _set_zoom_anim(self, z):
        if self._anchor is not None:
            (wp, ip) = self._anchor
            self.zoom = z
            s = self.scale()
            self.cx = (ip.x() - (wp.x() - self.width() / 2) / s) / self.iw
            self.cy = (ip.y() - (wp.y() - self.height() / 2) / s) / self.ih
        else:
            self.zoom = z
        self._clamp()
        self._emit_view()
        self.update()

    def fit(self):
        self._anchor = None
        self._zf.set_target(1.0)

    # ---- painting
    def paintEvent(self, _ev):
        p = QPainter(self)
        pal = self.palette()
        p.fillRect(self.rect(), pal.color(QPalette.ColorRole.Base).darker(115))
        if self.img.isNull():
            p.setPen(pal.color(QPalette.ColorRole.Text))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image")
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 4)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.drawImage(QRectF(self.i2w(0, 0), self.i2w(self.iw, self.ih)), self.img)
        keep, ok, found = self.matches()
        p.setFont(self._label_font())
        if self.show_gt:
            for k, (b, c) in enumerate(zip(self.gt_boxes, self.gt_cls)):
                r = self.rect_w(b)
                col = QColor(245, 245, 245) if (k < len(found) and found[k]) else ERR
                selected = self.sel == ("gt", k)
                p.setPen(QPen(QColor(0, 0, 0, 160), 3.5 if selected else 3))
                p.drawRect(r)
                pen = QPen(SEL if selected else col, 2.5 if selected else 1.5, Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.drawRect(r)
                self._chip(p, r.bottomLeft() + QPointF(0, 1), str(self.names.get(int(c), int(c))),
                           SEL if selected else (QColor(60, 60, 60, 220) if col != ERR else ERR), below=True)
        for j, pi in enumerate(keep):
            b = self.preds[pi]
            r = self.rect_w(b)
            good = ok[j] if self.show_gt else True
            col = self.color if good else ERR
            selected = self.sel == ("pred", int(pi))
            p.setPen(QPen(SEL if selected else col, 3 if selected else 2))
            p.drawRect(r)
            self._chip(p, r.topLeft(), f"{self.names.get(int(b[5]), int(b[5]))} {b[4]:.2f}", SEL if selected else col)
        if self._drag and self._drag.get("mode") == "draw":
            p.setPen(QPen(SEL, 1.5, Qt.PenStyle.DashLine))
            p.drawRect(QRectF(self._drag["start"], self._drag["pos"]).normalized())
        if self.caption:
            self._chip(p, QPointF(8, 8), self.caption, QColor(0, 0, 0, 170), big=True)
        if self.zoom > 1.01:
            self._chip(p, QPointF(8, self.height() - 26), f"{self.scale() * 100:.0f}%", QColor(0, 0, 0, 150))
        p.end()

    def _label_font(self):
        """Box labels in Inter (like the plotted images and charts), at the widget's size."""
        if getattr(self, "_lf", None) is None:
            from AIO_QC.tools.frame_extractor.utils import fonts
            f = fonts.ui_font(self.font(), 0.92, QFont.Weight.DemiBold)
            self._lf = f
        return self._lf

    def _chip(self, p, pt, text, bg, below=False, big=False):
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(text) + 8, fm.height() + 2
        y = pt.y() if below or big else pt.y() - h
        r = QRectF(pt.x(), max(0.0, y), w, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(r, 3, 3)
        p.setPen(QColor(0, 0, 0) if bg.lightness() > 150 else QColor(255, 255, 255))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
        p.setBrush(Qt.BrushStyle.NoBrush)

    # ---- mouse
    def _hit(self, pos):
        keep, _, _ = self.matches()
        best, area = ("", -1), 1e18
        if self.show_gt:
            for k, b in enumerate(self.gt_boxes):
                r = self.rect_w(b).adjusted(-4, -4, 4, 4)
                if r.contains(pos) and r.width() * r.height() < area:
                    best, area = ("gt", k), r.width() * r.height()
        for pi in keep:
            r = self.rect_w(self.preds[pi]).adjusted(-4, -4, 4, 4)
            if r.contains(pos) and r.width() * r.height() < area:
                best, area = ("pred", int(pi)), r.width() * r.height()
        return best

    def wheelEvent(self, ev):
        d = ev.angleDelta().y() or ev.pixelDelta().y()
        if d:
            self.zoom_at(1.0018 ** d, ev.position())
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        if self.zoom > 1.01:
            self.fit()
        else:
            self.zoom_at(3.0, ev.position())

    def mousePressEvent(self, ev):
        pos = ev.position()
        if ev.button() == Qt.MouseButton.MiddleButton or (ev.button() == Qt.MouseButton.LeftButton and not self.editable):
            self._drag = dict(mode="pan", last=pos)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if ev.button() == Qt.MouseButton.LeftButton and self.editable:
            hit = self._hit(pos)
            if hit[0]:
                self.sel = hit
                self.selection_changed.emit(*hit)
                self.update()
                self._drag = dict(mode="pan", last=pos) if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else None
            else:
                self.sel = ("", -1)
                self.selection_changed.emit("", -1)
                self._drag = dict(mode="draw", start=pos, pos=pos)
                self.update()

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._drag and self._drag["mode"] == "pan":
            d = pos - self._drag["last"]
            self._drag["last"] = pos
            s = self.scale()
            self.cx -= d.x() / s / self.iw
            self.cy -= d.y() / s / self.ih
            self._clamp()
            self._emit_view()
            self.update()
        elif self._drag and self._drag["mode"] == "draw":
            self._drag["pos"] = pos
            self.update()
        elif self.editable:
            self.setCursor(Qt.CursorShape.PointingHandCursor if self._hit(pos)[0] else Qt.CursorShape.CrossCursor)

    def mouseReleaseEvent(self, ev):
        d, self._drag = self._drag, None
        self.unsetCursor()
        if d and d["mode"] == "draw":
            a, b = self.w2i(d["start"]), self.w2i(ev.position())
            r = QRectF(a, b).normalized().intersected(QRectF(0, 0, self.iw, self.ih))
            if r.width() * self.scale() > 4 and r.height() * self.scale() > 4:
                self.box_drawn.emit(r)
            self.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._clamp()


def link_views(a, b):
    a.view_changed.connect(lambda z, x, y: b.set_view(z, x, y))
    b.view_changed.connect(lambda z, x, y: a.set_view(z, x, y))


# ====================================================================== label editor
class LabelEditor(QDialog):
    """Quick label fixes. items: [dataset item index]; run: evaluation.LabRun; data: pipeline.Data."""
    labels_changed = pyqtSignal(list)            # item indices whose labels / exclusion changed

    def __init__(self, parent, run, data, items, start=0, model=None, get_frame=None, slots=None):
        super().__init__(parent)
        self.setWindowTitle("Fix labels")
        self.run, self.data, self.items = run, data, list(items)
        self.get_frame = get_frame
        self.slots = slots or list(range(len(run.specs)))
        self.names = dict(run.names)
        self.pos = -1
        self.dirty = False
        self.changed = set()
        self.boxes, self.cls, self.origin = np.zeros((0, 4), np.float32), np.zeros(0, int), []
        from AIO_QC.tools.frame_extractor.utils.qt_utils import preferred_size
        preferred_size(self, 1180, 760, 0.9)

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.prev_b = QToolButton()
        self.prev_b.setText("◀")
        self.prev_b.setToolTip("Previous image (←) - saves your changes")
        self.prev_b.clicked.connect(lambda: self.go(self.pos - 1))
        self.next_b = QToolButton()
        self.next_b.setText("▶")
        self.next_b.setToolTip("Next image (→) - saves your changes")
        self.next_b.clicked.connect(lambda: self.go(self.pos + 1))
        self.where = QLabel("")
        self.model = QComboBox()
        self.model.setToolTip("Whose detections to show (to accept one as ground truth, or to compare)")
        for i, s in enumerate(run.specs):
            self.model.addItem(s.label, i)
        if model is not None:
            k = self.model.findData(model)
            self.model.setCurrentIndex(max(0, k))
        self.model.currentIndexChanged.connect(lambda _: self._show_preds())
        self.conf = QSlider(Qt.Orientation.Horizontal)
        self.conf.setRange(1, 99)
        self.conf.setFixedWidth(140)
        self.conf_lbl = QLabel("")
        self.conf.valueChanged.connect(self._conf_changed)
        for wdg in (self.prev_b, self.next_b, self.where):
            top.addWidget(wdg)
        top.addStretch(1)
        top.addWidget(QLabel("Detections of"))
        top.addWidget(self.model)
        top.addWidget(QLabel("at"))
        top.addWidget(self.conf)
        top.addWidget(self.conf_lbl)
        lay.addLayout(top)

        split = QSplitter()
        self.view = AnnotView(editable=True)
        self.view.box_drawn.connect(self._add_box)
        self.view.selection_changed.connect(self._selected)
        split.addWidget(self.view)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(6, 0, 0, 0)
        sl.addWidget(QLabel("<b>Ground truth</b>"))
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._list_row)
        sl.addWidget(self.list, 1)
        self.cls_cb = QComboBox()
        self.cls_cb.setToolTip("Class of the selected box - and of boxes you draw next")
        for c, n in sorted(self.names.items()):
            self.cls_cb.addItem(str(n), c)
        self.cls_cb.activated.connect(self._set_class)
        sl.addWidget(QLabel("Class"))
        sl.addWidget(self.cls_cb)
        row = QHBoxLayout()
        self.del_b = QPushButton("Delete box")
        self.del_b.setToolTip("Delete the selected ground-truth box (Del)")
        self.del_b.clicked.connect(self._delete)
        self.acc_b = QPushButton("Add detection as GT")
        self.acc_b.setToolTip("The selected detection is right and the label was missing (A)")
        self.acc_b.clicked.connect(self._accept_pred)
        row.addWidget(self.del_b)
        row.addWidget(self.acc_b)
        sl.addLayout(row)
        tip = QLabel("Drag on the image to draw a missing box · click a box to select it · wheel to zoom, "
                     "Shift-drag or middle-drag to pan · ← / → next image (saves)")
        tip.setWordWrap(True)
        tip.setEnabled(False)
        sl.addWidget(tip)
        self.excl = QCheckBox("Leave this image out of scoring")
        self.excl.setToolTip("For images that cannot be labelled fairly (unclear, out of scope). Reversible.")
        self.excl.toggled.connect(self._excl)
        sl.addWidget(self.excl)
        row = QHBoxLayout()
        self.revert_b = QPushButton("Revert")
        self.revert_b.setToolTip("Discard the changes to this image")
        self.revert_b.clicked.connect(lambda: self.go(self.pos, force=True))
        self.save_b = QPushButton("Save")
        self.save_b.setToolTip("Save (Ctrl+S) - also happens when you move to another image or close")
        self.save_b.clicked.connect(self.save)
        row.addWidget(self.revert_b)
        row.addWidget(self.save_b)
        sl.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        sl.addWidget(self.status)
        split.addWidget(side)
        split.setSizes([860, 280])
        split.setStretchFactor(0, 1)
        lay.addWidget(split, 1)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self.go(self.pos - 1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self.go(self.pos + 1))
        QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self, activated=self._delete)
        QShortcut(QKeySequence("Backspace"), self, activated=self._delete)
        QShortcut(QKeySequence("A"), self, activated=self._accept_pred)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save)
        QShortcut(QKeySequence("F"), self, activated=self.view.fit)
        self._conf_init()
        self.go(start)

    # ---- navigation
    def _conf_init(self):
        mi = self.model.currentData()
        r = self.run.results[mi] if mi is not None and mi < len(self.run.results) else None
        self.conf.setValue(int(round((r.op_conf if r is not None else 0.25) * 100)))

    def _conf_changed(self, v):
        self.conf_lbl.setText(f"{v / 100:.2f}")
        self.view.set_conf(v / 100)

    def go(self, pos, force=False):
        if not self.items:
            return
        pos = max(0, min(len(self.items) - 1, pos))
        if pos == self.pos and not force:
            return
        note = ""
        if self.dirty and not force:
            if self.save():
                note = "Saved ✓ the previous image's labels."
        self.dirty = False
        self.pos = pos
        it = self.run.items[self.items[pos]]
        img = self.get_frame(it) if self.get_frame else None
        self.view.set_image(img)
        if it.boxes is not None:
            self.boxes, self.cls = it.boxes.copy(), np.asarray(it.cls, int).copy()
        else:
            self.boxes, self.cls = np.zeros((0, 4), np.float32), np.zeros(0, int)
        self.origin = list(range(len(self.cls)))
        self.excl.blockSignals(True)
        self.excl.setChecked(it.meta.get("excluded") == "excluded by you")
        self.excl.blockSignals(False)
        import os
        v = it.meta.get("video")
        loc = f"{os.path.basename(v)} · frame {it.meta.get('frame')}" if v else os.path.basename(it.path)
        state = "" if it.labelled else " · not labelled yet"
        if it.labelled and not it.reviewed:
            state = " · pre-labels (unreviewed)"
        self.where.setText(f"<b>{pos + 1}</b> / {len(self.items)} · {loc}{state}")
        self.prev_b.setEnabled(pos > 0)
        self.next_b.setEnabled(pos < len(self.items) - 1)
        self._refresh()
        self._show_preds()
        self.status.setText(note)

    def _show_preds(self):
        mi = self.model.currentData()
        ii = self.items[self.pos] if 0 <= self.pos < len(self.items) else None
        if mi is None or ii is None:
            return
        pr = self.run.preds[mi][ii] if mi < len(self.run.preds) else None
        self.view.set_layers(preds=pr if pr is not None else np.zeros((0, 6), np.float32), names=self.names,
                             color=model_hex(self.slots[mi] if mi < len(self.slots) else mi))

    def _refresh(self, select=None):
        self.view.set_layers(self.boxes, self.cls, names=self.names)
        self.list.blockSignals(True)
        self.list.clear()
        for k, (b, c) in enumerate(zip(self.boxes, self.cls)):
            it = QListWidgetItem(f"{self.names.get(int(c), int(c))}   {int(b[2] - b[0])}×{int(b[3] - b[1])} px"
                                 + ("   (new)" if self.origin[k] < 0 else ""))
            self.list.addItem(it)
        self.list.blockSignals(False)
        if select is not None and 0 <= select < self.list.count():
            self.list.setCurrentRow(select)
            self.view.sel = ("gt", select)
        self.save_b.setEnabled(self.dirty)
        self.revert_b.setEnabled(self.dirty)
        self.setWindowTitle("Fix labels" + (" •" if self.dirty else ""))

    # ---- edits
    def _selected(self, kind, idx):
        if kind == "gt":
            self.list.blockSignals(True)
            self.list.setCurrentRow(idx)
            self.list.blockSignals(False)
            k = self.cls_cb.findData(int(self.cls[idx]))
            if k >= 0:
                self.cls_cb.setCurrentIndex(k)
        self.del_b.setEnabled(kind == "gt")
        self.acc_b.setEnabled(kind == "pred")

    def _list_row(self, row):
        if 0 <= row < len(self.cls):
            self.view.sel = ("gt", row)
            self.view.update()
            self._selected("gt", row)

    def _mark(self):
        self.dirty = True

    def _add_box(self, r):
        c = self.cls_cb.currentData()
        if c is None:
            return
        self.boxes = np.vstack([self.boxes, [[r.left(), r.top(), r.right(), r.bottom()]]]).astype(np.float32)
        self.cls = np.append(self.cls, int(c))
        self.origin.append(-1)
        self._mark()
        self._refresh(select=len(self.cls) - 1)

    def _set_class(self, _i):
        kind, k = self.view.sel
        c = self.cls_cb.currentData()
        if kind == "gt" and 0 <= k < len(self.cls) and c is not None and int(self.cls[k]) != int(c):
            self.cls[k] = int(c)
            self._mark()
            self._refresh(select=k)

    def _delete(self):
        kind, k = self.view.sel
        if kind != "gt" or not 0 <= k < len(self.cls):
            return
        self.boxes = np.delete(self.boxes, k, axis=0)
        self.cls = np.delete(self.cls, k)
        del self.origin[k]
        self.view.sel = ("", -1)
        self._mark()
        self._refresh()

    def _accept_pred(self):
        kind, k = self.view.sel
        if kind != "pred" or not 0 <= k < len(self.view.preds):
            return
        b = self.view.preds[k]
        self.boxes = np.vstack([self.boxes, [b[:4]]]).astype(np.float32)
        self.cls = np.append(self.cls, int(b[5]))
        self.origin.append(-1)
        self._mark()
        self._refresh(select=len(self.cls) - 1)

    def _excl(self, on):
        from AIO_QC.core.testing.labels_io import set_excluded
        ii = self.items[self.pos]
        set_excluded(self.run.items[ii], on, self.data.root, self.data.testset)
        self.changed.add(ii)
        self.status.setText("Left out of scoring." if on else "Scored again.")

    def save(self):
        if not self.dirty or not 0 <= self.pos < len(self.items):
            return True
        from AIO_QC.core.testing.labels_io import save_labels
        ii = self.items[self.pos]
        it = self.run.items[ii]
        qc = getattr(self, "qc", None)                  # the dataset open in Image QC: AIO QC's write path
        try:
            write = lambda: save_labels(it, self.boxes, self.cls, self.origin, self.names,
                                        self.data.dataset.fmt or "yolo", self.data.testset)
            if qc is not None:
                from AIO_QC.core.testing.labels_io import label_target
                target = label_target(it, self.data.dataset.fmt or "yolo", self.data.testset)[0]
                path = qc.write_labels([target], write)
            else:
                path = write()
        except Exception as e:
            QMessageBox.warning(self, "Fix labels", f"Could not save the labels:\n{e}")
            return False
        self.dirty = False
        self.origin = list(range(len(self.cls)))
        self.changed.add(ii)
        import os
        self.status.setText(f"Saved ✓ {os.path.basename(path)}")
        self._refresh()
        return True

    def done(self, r):
        if self.dirty:
            self.save()
        if self.changed:
            self.labels_changed.emit(sorted(self.changed))
        super().done(r)


# ====================================================================== A / B viewer
class ABViewer(QDialog):
    """Two models on the same image, linked zoom / pan. items: [dataset item index]."""
    edit_requested = pyqtSignal(list, int)          # items, position

    def __init__(self, parent, run, items, a, b, start=0, get_frame=None, slots=None, plotter=None, can_edit=None):
        super().__init__(parent)
        self.setWindowTitle("Compare models")
        self.plotter = plotter
        self.can_edit = can_edit
        self.run, self.items, self.get_frame = run, list(items), get_frame
        self.slots = slots or list(range(len(run.specs)))
        self.pos = -1
        from AIO_QC.tools.frame_extractor.utils.qt_utils import preferred_size
        preferred_size(self, 1320, 760, 0.92)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.prev_b = QToolButton()
        self.prev_b.setText("◀")
        self.prev_b.clicked.connect(lambda: self.go(self.pos - 1))
        self.next_b = QToolButton()
        self.next_b.setText("▶")
        self.next_b.clicked.connect(lambda: self.go(self.pos + 1))
        self.where = QLabel("")
        self.gt = QCheckBox("Ground truth")
        self.gt.setChecked(True)
        self.gt.setToolTip("Show the labels (dashed). Errors are red only while ground truth is shown.")
        self.own = QCheckBox("Each model's threshold")
        self.own.setChecked(True)
        self.own.setToolTip("Use every model's operating threshold; untick to compare both at the slider value")
        self.conf = QSlider(Qt.Orientation.Horizontal)
        self.conf.setRange(1, 99)
        self.conf.setValue(25)
        self.conf.setFixedWidth(140)
        self.conf.setEnabled(False)
        self.conf_lbl = QLabel("")
        edit = self.edit_btn = QPushButton("Fix labels…")
        edit.clicked.connect(lambda: self.edit_requested.emit(self.items, self.pos))
        save = QPushButton("Save image…")
        save.setToolTip("Save both plots side by side at full resolution (PNG or JPG)")
        save.clicked.connect(self._save)
        save.setVisible(plotter is not None)
        for wdg in (self.prev_b, self.next_b, self.where):
            top.addWidget(wdg)
        top.addStretch(1)
        for wdg in (self.gt, self.own, self.conf, self.conf_lbl, edit, save):
            top.addWidget(wdg)
        lay.addLayout(top)
        row = QHBoxLayout()
        self.combos, self.views = [], []
        panes = QSplitter()
        for side, m in (("A", a), ("B", b)):
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            cb = QComboBox()
            for i, s in enumerate(run.specs):
                if run.results[i] is not None or run.ground != "labels":
                    cb.addItem(s.label, i)
            cb.setCurrentIndex(max(0, cb.findData(m)))
            cb.currentIndexChanged.connect(lambda _=0: self._layers())
            view = AnnotView()
            v.addWidget(cb)
            v.addWidget(view, 1)
            self.combos.append(cb)
            self.views.append(view)
            panes.addWidget(w)
        link_views(*self.views)
        row.addWidget(panes)
        lay.addLayout(row, 1)
        hint = QLabel("Wheel = zoom (both sides) · drag = pan · double-click = fit / zoom · ← / → next image")
        hint.setEnabled(False)
        lay.addWidget(hint)
        self.gt.toggled.connect(lambda on: [v.set_show_gt(on) for v in self.views])
        self.own.toggled.connect(lambda on: (self.conf.setEnabled(not on), self._layers()))
        self.conf.valueChanged.connect(lambda _: self._layers())
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self.go(self.pos - 1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self.go(self.pos + 1))
        self.go(start)

    def go(self, pos):
        if not self.items:
            return
        pos = max(0, min(len(self.items) - 1, pos))
        if pos == self.pos:
            return
        self.pos = pos
        it = self.run.items[self.items[pos]]
        img = self.get_frame(it) if self.get_frame else None
        q = to_qimage(img)
        for v in self.views:
            v.set_image(q, keep_view=True)
        import os
        vid = it.meta.get("video")
        self.edit_btn.setVisible(bool(self.can_edit is None or self.can_edit(self.items[pos])))
        loc = f"{os.path.basename(vid)} · frame {it.meta.get('frame')}" if vid else os.path.basename(it.path)
        self.where.setText(f"<b>{pos + 1}</b> / {len(self.items)} · {loc}")
        self.prev_b.setEnabled(pos > 0)
        self.next_b.setEnabled(pos < len(self.items) - 1)
        self._layers()

    def _gt(self, ii):
        r = self.run
        if r.ground == "labels":
            it = r.items[ii]
            return (it.boxes, it.cls) if it.boxes is not None else (np.zeros((0, 4)), np.zeros(0, int))
        idx, gts = r.ground_truth()
        return gts[idx.index(ii)] if ii in idx else (np.zeros((0, 4)), np.zeros(0, int))

    def _layers(self):
        if not 0 <= self.pos < len(self.items):
            return
        ii = self.items[self.pos]
        gb, gc = self._gt(ii)
        self.conf_lbl.setText(f"{self.conf.value() / 100:.2f}")
        for cb, v in zip(self.combos, self.views):
            mi = cb.currentData()
            if mi is None:
                continue
            conf = model_conf(self.run, mi) if self.own.isChecked() else self.conf.value() / 100
            pr = self.run.preds[mi][ii]
            v.set_layers(gb, gc, pr if pr is not None else np.zeros((0, 6), np.float32), self.run.names,
                         model_hex(self.slots[mi] if mi < len(self.slots) else mi), conf)
            keep, ok, found = v.matches()
            fp, fn = int((~ok).sum()), int((~found).sum()) if len(found) else 0
            v.caption = f"{self.run.specs[mi].label} · conf ≥ {conf:.2f} · {int(ok.sum())} correct · {fp} false · {fn} missed"
            v.update()

    def _save(self):
        if self.plotter is None or not 0 <= self.pos < len(self.items):
            return
        ii = self.items[self.pos]
        ims = []
        for cb in self.combos:
            mi = cb.currentData()
            if mi is None:
                continue
            conf = model_conf(self.run, mi) if self.own.isChecked() else self.conf.value() / 100
            ims.append(self.plotter(ii, mi, self.gt.isChecked(), conf))
        if not ims or any(im is None for im in ims):
            save_plot_dialog(self, None, "")
            return
        gap = np.full((ims[0].shape[0], max(4, ims[0].shape[1] // 200), 3), 255, np.uint8)
        img = np.hstack([ims[0], gap, ims[1]]) if len(ims) == 2 else ims[0]
        names = "_vs_".join(self.run.specs[cb.currentData()].label for cb in self.combos if cb.currentData() is not None)
        save_plot_dialog(self, img, plot_name(self.run, ii, self.combos[0].currentData()).split("__")[0] + "__" + names)


def save_plot_dialog(parent, img, default_name):
    """Ask where to save a plotted image (PNG or JPG) and write it. Returns the path or None."""
    import os
    from PyQt6.QtWidgets import QFileDialog
    from AIO_QC.widgets.testing.lab_common import last_dir, remember_dir, safe_name
    if img is None:
        QMessageBox.warning(parent, "Save image", "The image could not be read.")
        return None
    path, flt = QFileDialog.getSaveFileName(parent, "Save plotted image",
                                            os.path.join(last_dir(), safe_name(default_name) + ".png"),
                                            "PNG (*.png);;JPEG (*.jpg)")
    if not path:
        return None
    if not os.path.splitext(path)[1]:
        path += ".jpg" if "JPEG" in flt else ".png"
    from AIO_QC.tools.frame_extractor.core.imageio import imwrite
    try:
        imwrite(path, img, quality=95)
    except Exception as e:                       # noqa: BLE001 - shown to the user
        QMessageBox.warning(parent, "Save image", f"Could not save the image:\n{e}")
        return None
    remember_dir(path)
    return path


def plot_name(run, ii, mi):
    import os
    it = run.items[ii]
    vid = it.meta.get("video")
    stem = (f"{os.path.splitext(os.path.basename(vid))[0]}_f{it.meta.get('frame', 0)}" if vid
            else os.path.splitext(os.path.basename(it.path))[0])
    return f"{stem}__{run.specs[mi].label}"


# ====================================================================== plot viewer
class PlotViewer(QDialog):
    """One model's detections on one image at a time, full size. items: [dataset item index]."""

    def __init__(self, parent, run, items, mi, start=0, get_frame=None, slots=None, plotter=None, show_gt=True):
        super().__init__(parent)
        self.setWindowTitle("Plotted predictions")
        self.run, self.items, self.get_frame, self.plotter = run, list(items), get_frame, plotter
        self.slots = slots or list(range(len(run.specs)))
        self.pos = -1
        self._img = None
        from AIO_QC.tools.frame_extractor.utils.qt_utils import preferred_size
        preferred_size(self, 1180, 780, 0.9)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.prev_b = QToolButton()
        self.prev_b.setText("◀")
        self.prev_b.clicked.connect(lambda: self.go(self.pos - 1))
        self.next_b = QToolButton()
        self.next_b.setText("▶")
        self.next_b.clicked.connect(lambda: self.go(self.pos + 1))
        self.where = QLabel("")
        self.model = QComboBox()
        for i, s_ in enumerate(run.specs):
            if run.preds[i] is not None:
                self.model.addItem(s_.label, i)
        self.model.setCurrentIndex(max(0, self.model.findData(mi)))
        self.gt = QCheckBox("Ground truth")
        self.gt.setChecked(show_gt)
        self.gt.setEnabled(bool(run.scored) or run.ground == "labels")
        self.own = QCheckBox("Model's threshold")
        self.own.setChecked(True)
        self.conf = QSlider(Qt.Orientation.Horizontal)
        self.conf.setRange(1, 99)
        self.conf.setValue(25)
        self.conf.setFixedWidth(130)
        self.conf.setEnabled(False)
        self.conf_lbl = QLabel("")
        self.conf_lbl.setMinimumWidth(30)
        save = QPushButton("Save image…")
        save.setToolTip("Save this plot at full resolution (PNG or JPG) - exactly what you see, labels in Inter")
        save.clicked.connect(self._save)
        for wdg in (self.prev_b, self.next_b, self.where):
            top.addWidget(wdg)
        top.addStretch(1)
        for wdg in (QLabel("Model"), self.model, self.gt, self.own, self.conf, self.conf_lbl, save):
            top.addWidget(wdg)
        lay.addLayout(top)
        self.view = AnnotView()
        lay.addWidget(self.view, 1)
        hint = QLabel("Wheel = zoom · drag = pan · double-click = fit / zoom · ← / → next image · Ctrl+S save")
        hint.setEnabled(False)
        lay.addWidget(hint)
        self.model.currentIndexChanged.connect(lambda _: self._layers())
        self.gt.toggled.connect(lambda on: self._layers())
        self.own.toggled.connect(lambda on: (self.conf.setEnabled(not on), self._layers()))
        self.conf.valueChanged.connect(lambda _: self._layers())
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self.go(self.pos - 1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self.go(self.pos + 1))
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self._save)
        self.view.set_show_gt(self.gt.isChecked() and self.gt.isEnabled())
        self.go(start)

    def _gt(self, ii):
        r = self.run
        if r.ground == "labels":
            it = r.items[ii]
            return (it.boxes, it.cls) if it.boxes is not None else (np.zeros((0, 4)), np.zeros(0, int))
        if not r.scored:
            return np.zeros((0, 4)), np.zeros(0, int)
        idx, gts = r.ground_truth()
        return gts[idx.index(ii)] if ii in idx else (np.zeros((0, 4)), np.zeros(0, int))

    def go(self, pos):
        if not self.items:
            return
        pos = max(0, min(len(self.items) - 1, pos))
        if pos == self.pos:
            return
        self.pos = pos
        it = self.run.items[self.items[pos]]
        self._img = self.get_frame(it) if self.get_frame else None
        self.view.set_image(to_qimage(self._img), keep_view=True)
        import os
        vid = it.meta.get("video")
        loc = f"{os.path.basename(vid)} · frame {it.meta.get('frame')}" if vid else os.path.basename(it.path)
        self.where.setText(f"<b>{pos + 1}</b> / {len(self.items)} · {loc}")
        self.prev_b.setEnabled(pos > 0)
        self.next_b.setEnabled(pos < len(self.items) - 1)
        self._layers()

    def current_conf(self):
        mi = self.model.currentData()
        res = self.run.results[mi] if mi is not None else None
        if self.own.isChecked():
            return model_conf(self.run, mi)
        return self.conf.value() / 100

    def _layers(self):
        mi = self.model.currentData()
        if mi is None or not 0 <= self.pos < len(self.items):
            return
        ii = self.items[self.pos]
        gb, gc = self._gt(ii)
        is_ref = self.run.ground == "reference" and mi == self.run.reference
        if is_ref:                                   # the reference IS the answer key: its boxes only
            gb, gc = np.zeros((0, 4)), np.zeros(0, int)
        self.gt.setEnabled(bool(self.run.scored or self.run.ground == "labels") and not is_ref)
        self.view.set_show_gt(self.gt.isChecked() and self.gt.isEnabled())
        conf = self.current_conf()
        self.conf_lbl.setText(f"{conf:.2f}")
        pr = self.run.preds[mi][ii]
        self.view.set_layers(gb, gc, pr if pr is not None else np.zeros((0, 6), np.float32), self.run.names,
                             model_hex(self.slots[mi] if mi < len(self.slots) else mi), conf)
        keep, ok, found = self.view.matches()
        if self.gt.isChecked() and self.gt.isEnabled():
            self.view.caption = (f"{self.run.specs[mi].label} · conf ≥ {conf:.2f} · {int(ok.sum())} correct · "
                                 f"{int((~ok).sum())} false · {int((~found).sum()) if len(found) else 0} missed")
        else:
            self.view.caption = f"{self.run.specs[mi].label} · conf ≥ {conf:.2f} · {len(keep)} detections"
        self.view.update()

    def plotted(self):
        """The current plot at full resolution (BGR), as saved."""
        mi = self.model.currentData()
        if mi is None or self.plotter is None or not 0 <= self.pos < len(self.items):
            return None
        return self.plotter(self.items[self.pos], mi, self.gt.isChecked() and self.gt.isEnabled(), self.current_conf())

    def _save(self):
        mi = self.model.currentData()
        if mi is not None and 0 <= self.pos < len(self.items):
            save_plot_dialog(self, self.plotted(), plot_name(self.run, self.items[self.pos], mi))
