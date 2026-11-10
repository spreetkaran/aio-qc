# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Prediction Explorer ▸ image viewer: one image at full resolution with every label and prediction, hover details,
per-object actions (accept the model's suggestion, relabel, delete, add a prediction as a label) and a side-by-side
compare of two models with synced zoom / pan. ← / → walk through the explorer's current order.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QPainter, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QSplitter,
    QTableWidget, QTableWidgetItem, QToolButton, QToolTip, QVBoxLayout, QWidget,
)

from AIO_QC.core.explorer import NA, TAG_HELP, TAG_NAMES, View, evaluate, objects, op_for, add_op, suggestion
from AIO_QC.utils import get_color_icon, newButton, style as apply_style, theme_color
from .paint import FullTask, Signals, draw_objects, hit_test, image_path, tag_color


class ImageCanvas(QWidget):
    """Zoom (wheel, at the cursor) / pan (drag) image with the objects on top. Click selects an object."""
    picked = pyqtSignal(int)                 # object index or -1
    viewChanged = pyqtSignal(float, float, float)     # zoom relative to fit, centre x, centre y (image px)

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.img = None
        self.objs = []
        self.sel = None
        self.hover = None
        self.title = ""
        self.zoom = 1.0                      # relative to fit
        self.cx = self.cy = 0.0              # image point at the widget centre
        self._drag = None
        self._moved = False

    def set_image(self, img, size=None):
        self.img = img
        self.fit()

    def set_objects(self, objs, sel=None):
        self.objs = objs
        self.sel = sel
        self.hover = None
        self.update()

    def _isize(self):
        if self.img is not None and not self.img.isNull():
            return self.img.width(), self.img.height()
        return self.viewer.image_size()

    def fit(self):
        W, H = self._isize()
        self.zoom = 1.0
        self.cx, self.cy = W / 2, H / 2
        self.update()

    def _scale(self):
        W, H = self._isize()
        if not W or not H:
            return 1.0
        return min(self.width() / W, self.height() / H) * 0.98 * self.zoom

    def _mapping(self):
        s = self._scale()
        return s, self.width() / 2 - self.cx * s, self.height() / 2 - self.cy * s

    def to_image(self, pos):
        s, ox, oy = self._mapping()
        return (pos.x() - ox) / s, (pos.y() - oy) / s

    def apply_view(self, zoom, cx, cy):
        self.zoom, self.cx, self.cy = zoom, cx, cy
        self.update()

    def focus_box(self, box):
        W, H = self._isize()
        if not W:
            return
        x1, y1, x2, y2 = box
        bw, bh = max(x2 - x1, 8), max(y2 - y1, 8)
        fit = min(self.width() / W, self.height() / H) * 0.98
        self.zoom = max(1.0, min(min(self.width() / (bw * 3), self.height() / (bh * 3)) / fit, 40))
        self.cx, self.cy = (x1 + x2) / 2, (y1 + y2) / 2
        self.update()
        self.viewChanged.emit(self.zoom, self.cx, self.cy)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0b0d10"))
        s, ox, oy = self._mapping()
        if self.img is not None and not self.img.isNull():
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)      # smooth at any zoom (no blocky pixels)
            p.drawImage(QRectF(ox, oy, self.img.width() * s, self.img.height() * s), self.img)
        elif self.img is None:
            p.setPen(QColor("#9ca3af"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "loading…")
        st = self.viewer.panel.paint_style(tile=False)
        st["text"] = self.viewer.show_text.isChecked()
        st["width"] = 2.0
        draw_objects(p, self.objs, s, ox, oy, selected=self.sel, hover=self.hover, **st)
        if self.title:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 160))
            fm = p.fontMetrics()
            r = QRectF(8, 8, fm.horizontalAdvance(self.title) + 16, fm.height() + 8)
            p.drawRoundedRect(r, 6, 6)
            p.setPen(QColor("white"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.title)

    def wheelEvent(self, e):
        before = self.to_image(e.position())
        k = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        self.zoom = float(np.clip(self.zoom * k, 0.2, 60))
        after = self.to_image(e.position())
        self.cx += before[0] - after[0]
        self.cy += before[1] - after[1]
        self.update()
        self.viewChanged.emit(self.zoom, self.cx, self.cy)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = (e.position(), self.cx, self.cy)
            self._moved = False

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            start, cx, cy = self._drag
            d = e.position() - start
            if abs(d.x()) + abs(d.y()) > 3:
                self._moved = True
            s = self._scale()
            self.cx, self.cy = cx - d.x() / s, cy - d.y() / s
            self.update()
            self.viewChanged.emit(self.zoom, self.cx, self.cy)
            return
        x, y = self.to_image(e.position())
        h = hit_test(self.objs, x, y, tol=3 / max(self._scale(), 1e-6))
        if h != self.hover:
            self.hover = h
            self.update()
            if h is not None:
                QToolTip.showText(e.globalPosition().toPoint(), self.viewer.describe(self.objs[h]), self)
            else:
                QToolTip.hideText()

    def mouseReleaseEvent(self, e):
        if self._drag is not None and not self._moved:
            x, y = self.to_image(e.position())
            h = hit_test(self.objs, x, y, tol=3 / max(self._scale(), 1e-6))
            self.sel = h
            self.update()
            self.picked.emit(-1 if h is None else h)
        self._drag = None

    def mouseDoubleClickEvent(self, _e):
        self.fit()
        self.viewChanged.emit(self.zoom, self.cx, self.cy)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.update()


COLS = ["", "Kind", "Label", "Model", "Conf", "Result", "IoU"]


class ExplorerViewer(QDialog):
    def __init__(self, panel):
        super().__init__(panel.window())
        self.panel = panel
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowTitle("Prediction Explorer · image")
        self.resize(1400, 860)
        self.i = None
        self.objs = []
        self.size = (0, 0)
        self.sig = Signals()
        self.sig.done.connect(self._loaded)
        self.gen = 0
        self._img = None
        self._compare_states = None

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.prev_btn, self.next_btn = QToolButton(), QToolButton()
        for b, arrow, k, tip in ((self.prev_btn, Qt.ArrowType.LeftArrow, -1, "Previous image (←, PgUp)"),
                                 (self.next_btn, Qt.ArrowType.RightArrow, 1, "Next image (→, PgDn)")):
            b.setArrowType(arrow)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, k=k: self.step(k))
        self.pos_lbl = QLabel()
        self.name_lbl = QLabel()
        self.name_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        apply_style(self.name_lbl, "font-weight:600;")
        self.everything = QCheckBox("Everything")
        self.everything.setToolTip("Also show predictions below the threshold and objects the filters hide (faded "
                                   "in the list)")
        self.show_text = QCheckBox("Names")
        self.show_text.setChecked(True)
        self.show_text.setToolTip("Class / confidence pills on the image")
        self.compare = QCheckBox("Compare")
        self.compare.setToolTip("Two models side by side (labels on both), zoom and pan in sync")
        self.cmp_a, self.cmp_b = QComboBox(), QComboBox()
        for c in (self.cmp_a, self.cmp_b):
            c.setVisible(False)
            c.currentIndexChanged.connect(self._compare_changed)
        fit = newButton("⤢ Fit", slot=self._fit, tip="Fit the image (double-click the image)")
        main = newButton("Open in main viewer", slot=self._open_main,
                         tip="Show this image in AIO QC's main viewer (with the predictions layer)")
        for w in (self.prev_btn, self.next_btn, self.pos_lbl, self.name_lbl):
            top.addWidget(w)
        top.addStretch(1)
        for w in (self.everything, self.show_text, self.compare, self.cmp_a, self.cmp_b, fit, main):
            top.addWidget(w)
        lay.addLayout(top)

        split = QSplitter(Qt.Orientation.Horizontal)
        canv = QWidget()
        cl = QHBoxLayout(canv)
        cl.setContentsMargins(0, 0, 0, 0)
        self.canvas = ImageCanvas(self)
        self.canvas_b = ImageCanvas(self)
        self.canvas_b.setVisible(False)
        cl.addWidget(self.canvas, 1)
        cl.addWidget(self.canvas_b, 1)
        self.canvas.viewChanged.connect(lambda z, x, y: self.canvas_b.apply_view(z, x, y))
        self.canvas_b.viewChanged.connect(lambda z, x, y: self.canvas.apply_view(z, x, y))
        self.canvas.picked.connect(lambda n: self._pick_from(self.canvas, n))
        self.canvas_b.picked.connect(lambda n: self._pick_from(self.canvas_b, n))
        split.addWidget(canv)

        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._table_selected)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._zoom_to(r))
        sl.addWidget(self.table, 1)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        apply_style(self.info, "QLabel{background:@surface; border:1px solid @line; border-radius:6px; padding:6px;}")
        sl.addWidget(self.info)
        grid = QHBoxLayout()
        self.accept_btn = newButton("✓ Accept", slot=self.accept_selected,
                                    tip="Do what the model suggests for the selected object(s) (Enter):\n"
                                        "  not labelled → add the box · wrong class → relabel · poor fit → use the "
                                        "model's box · missed label → delete it")
        self.add_btn = newButton("＋ Add as label", slot=self.add_selected,
                                 tip="Add the selected prediction(s) as labels, whatever their result (A)")
        self.relabel_btn = newButton("✎ Relabel…", slot=self.relabel_selected, tip="Change the class of the "
                                     "selected label(s) (R)")
        self.delete_btn = newButton("🗑 Delete label", slot=self.delete_selected, tip="Delete the selected label(s) "
                                    "(Delete)")
        for b in (self.accept_btn, self.add_btn, self.relabel_btn, self.delete_btn):
            grid.addWidget(b)
        sl.addLayout(grid)
        rv = QHBoxLayout()
        rv.addWidget(newButton("⚑ Flag file", slot=lambda: self._review("flag"), tip="Flag this file for review (F)"))
        rv.addWidget(newButton("✔ File OK", slot=lambda: self._review("ok"), tip="Mark this file OK (O)"))
        rv.addWidget(newButton("Clear mark", slot=lambda: self._review(None)))
        sl.addLayout(rv)
        split.addWidget(side)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setSizes([1000, 380])
        lay.addWidget(split, 1)

        for keys, fn in (((Qt.Key.Key_Left, Qt.Key.Key_PageUp), lambda: self.step(-1)),
                         ((Qt.Key.Key_Right, Qt.Key.Key_PageDown), lambda: self.step(1)),
                         ((Qt.Key.Key_Return, Qt.Key.Key_Enter), self.accept_selected),
                         ((Qt.Key.Key_A,), self.add_selected), ((Qt.Key.Key_R,), self.relabel_selected),
                         ((Qt.Key.Key_Delete,), self.delete_selected), ((Qt.Key.Key_F,), lambda: self._review("flag")),
                         ((Qt.Key.Key_O,), lambda: self._review("ok"))):
            for k in keys:
                sc = QShortcut(QKeySequence(k), self)
                sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                sc.activated.connect(fn)
        self.everything.toggled.connect(lambda _on: self.refresh())
        self.show_text.toggled.connect(lambda _on: (self.canvas.update(), self.canvas_b.update()))
        self.compare.toggled.connect(self._toggle_compare)
        self._fill_models()

    # ------------------------------------------------------------------ data
    def image_size(self):
        return self.size

    def show_image(self, i: int, focus=None):
        """focus: ("gt" | "pred", row) to select and zoom to."""
        d = self.panel.data
        self.i = int(i)
        self.size = (float(d.img_w[i] or 0), float(d.img_h[i] or 0))
        key = d.keys[i]
        self.name_lbl.setText(self.panel.display_name(key))
        self.name_lbl.setToolTip(key)
        self.gen += 1
        self._img = None
        self.canvas.set_image(None)
        self.canvas_b.set_image(None)
        QThreadPool.globalInstance().start(FullTask(self.sig, key, image_path(self.panel.session, key), self.gen))
        self.refresh(focus)
        self._update_pos()
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()

    def _loaded(self, key, gen, img, info):
        if gen != self.gen:
            return
        self._img = img
        if info and not self.size[0]:
            self.size = info
        for c in (self.canvas, self.canvas_b):
            c.img = img
            c.fit()
        if getattr(self, "_pending_focus", None) is not None:
            self._zoom_to_obj(self._pending_focus)
            self._pending_focus = None

    def _update_pos(self):
        order = self.panel.state.order if self.panel.state is not None else np.zeros(0)
        hit = np.nonzero(order == self.i)[0] if self.i is not None else []
        n = len(order)
        self.pos_lbl.setText(f"{int(hit[0]) + 1:,} / {n:,}" if len(hit) else f"– / {n:,}")

    def step(self, k):
        order = self.panel.state.order if self.panel.state is not None else np.zeros(0)
        if not len(order):
            return
        hit = np.nonzero(order == self.i)[0]
        j = (int(hit[0]) + k) if len(hit) else 0
        j = max(0, min(len(order) - 1, j))
        if len(hit) and j == int(hit[0]):
            return
        self.panel.grid.select_item(int(order[j])) if self.panel.grid.mode == "grid" else None
        self.show_image(int(order[j]))

    def refresh(self, focus=None):
        """Objects again (thresholds / layers / edits changed)."""
        if self.i is None or self.panel.state is None:
            return
        d, st = self.panel.data, self.panel.state
        if self.i >= len(d):
            return
        every = self.everything.isChecked()
        self.objs = objects(d, st, self.i, everything=every)
        if every:
            self.objs = [o for o in self.objs if o.kind == "gt" or o.model in self._visible_models()]
        sel_rows = {(o.kind, o.row) for o in self._selected()}
        if self.compare.isChecked():
            self._refresh_compare()
        else:
            self.canvas.title = ""
            self.canvas.set_objects(self.objs)
        self._fill_table(sel_rows)
        if focus is not None:
            self._pending_focus = focus
            if self._img is not None:
                self._zoom_to_obj(focus)
                self._pending_focus = None
        self._update_pos()

    def _visible_models(self):
        return {m for m in self.panel.data.models if self.panel.view.visible(m)}

    def _fill_table(self, keep=()):
        d = self.panel.data
        st = self.panel.state
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.objs))
        names = d.model_names
        for r, o in enumerate(self.objs):
            faded = (o.kind == "pred" and not st.p_show[o.row]) or (o.kind == "gt" and not st.g_show[o.row])
            vals = ["", "label" if o.kind == "gt" else "prediction", o.label, names.get(o.model, o.model),
                    "" if o.kind == "gt" else f"{o.conf:.3f}", TAG_NAMES.get(o.tag, ""),
                    f"{o.iou:.2f}" if o.kind == "pred" and o.partner >= 0 else ""]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 0:
                    it.setIcon(get_color_icon(tag_color(o.tag)))
                if faded:
                    it.setForeground(theme_color("faint"))
                self.table.setItem(r, c, it)
        self.table.clearSelection()
        for r, o in enumerate(self.objs):
            if (o.kind, o.row) in keep:
                self.table.selectRow(r)
        self.table.blockSignals(False)
        self._table_selected()

    # ------------------------------------------------------------------ selection
    def _selected(self) -> list:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.objs[r] for r in rows if r < len(self.objs)]

    def _table_selected(self):
        sel = self._selected()
        first = sel[0] if sel else None
        for c in (self.canvas, self.canvas_b):
            c.sel = next((n for n, o in enumerate(c.objs) if first is not None and o.kind == first.kind
                          and o.row == first.row), None)
            c.update()
        self._update_buttons(sel)
        self.info.setText(self.describe(first, rich=True) if first is not None else self._image_info())

    def _pick_from(self, canvas, n):
        if n < 0:
            self.table.clearSelection()
            return
        o = canvas.objs[n]
        for r, x in enumerate(self.objs):
            if x.kind == o.kind and x.row == o.row:
                self.table.selectRow(r)
                self.table.scrollToItem(self.table.item(r, 0))
                return

    def _zoom_to(self, r):
        if 0 <= r < len(self.objs):
            self.canvas.focus_box(self.objs[r].box)

    def _zoom_to_obj(self, focus):
        kind, row = focus
        for r, o in enumerate(self.objs):
            if o.kind == kind and o.row == row:
                self.table.selectRow(r)
                self.canvas.focus_box(o.box)
                return

    def _update_buttons(self, sel):
        d, st = self.panel.data, self.panel.state
        thr = self.panel.view.iou
        self.accept_btn.setEnabled(any(suggestion(d, st, o, thr) is not None for o in sel))
        self.add_btn.setEnabled(any(o.kind == "pred" for o in sel))
        self.relabel_btn.setEnabled(any(o.kind == "gt" for o in sel))
        self.delete_btn.setEnabled(any(o.kind == "gt" for o in sel))

    # ------------------------------------------------------------------ text
    def describe(self, o, rich=False) -> str:
        d, st = self.panel.data, self.panel.state
        if o is None:
            return ""
        b = "<b>{}</b>" if rich else "{}"
        lines = []
        if o.kind == "pred":
            lines.append(b.format(f"{d.model_names.get(o.model, o.model)}: {o.label} {o.conf:.3f}"))
            if o.partner >= 0:
                lines.append(f"matches the label '{d.labels[d.g_cls[o.partner]]}' at IoU {o.iou:.2f}")
        else:
            lines.append(b.format(f"label: {o.label}") + f"  (shape #{o.shape_index})")
            if o.partner >= 0:
                p = o.partner
                lines.append(f"found by {d.model_names.get(d.models[d.p_model[p]], '')} "
                             f"({d.p_conf[p]:.2f}, IoU {st.p_iou[p]:.2f})")
        lines.append(f"result: {TAG_NAMES.get(o.tag, '')} — {TAG_HELP.get(o.tag, '')}" if o.tag in TAG_HELP
                     else f"result: {TAG_NAMES.get(o.tag, '')}")
        sug = suggestion(d, st, o, self.panel.view.iou)
        if sug is not None:
            what = {"add": f"add it as a '{sug[1]}' label", "relabel": f"relabel the label to '{sug[2]}'",
                    "reshape": "replace the label's box with the model's", "delete": "delete the label"}[sug[0]]
            lines.append(("✓ Accept: " if rich else "Accept: ") + what)
        x1, y1, x2, y2 = o.box
        lines.append(f"box {x2 - x1:.0f}×{y2 - y1:.0f} px at ({x1:.0f}, {y1:.0f})")
        return ("<br>" if rich else "\n").join(lines)

    def _image_info(self) -> str:
        if self.i is None:
            return ""
        d = self.panel.data
        c = self.panel.state.counts[self.i]
        parts = [f"{int(c[t])} {TAG_NAMES[t]}" for t in range(7) if c[t] and t != NA]
        split = d.splits[d.img_split[self.i]] if len(d.splits) else ""
        return (f"<b>{self.panel.display_name(d.keys[self.i])}</b> · {int(self.size[0])}×{int(self.size[1])}"
                + (f" · split {split}" if split else "") + "<br>" + (", ".join(parts) or "nothing drawn") +
                "<br><i>Click an object (image or list) for details and actions.</i>")

    # ------------------------------------------------------------------ actions
    def accept_selected(self):
        d, st = self.panel.data, self.panel.state
        ops = {}
        for o in self._selected():
            got = op_for(d, st, o, self.panel.view.iou)
            if got is not None:
                ops.setdefault(got[0], []).append(got[1])
        if ops:
            self.panel.apply_ops(ops, "accept model suggestion")

    def add_selected(self):
        d = self.panel.data
        ops = {}
        for o in self._selected():
            if o.kind == "pred":
                k, op = add_op(d, self.i, o.label, o.box)
                ops.setdefault(k, []).append(op)
        if ops:
            self.panel.apply_ops(ops, "add prediction as label")

    def relabel_selected(self):
        sel = [o for o in self._selected() if o.kind == "gt"]
        if not sel:
            return
        labels = self.panel.label_choices()
        cur = labels.index(sel[0].label) if sel[0].label in labels else 0
        new, ok = QInputDialog.getItem(self, "Relabel", f"New class for {len(sel)} label(s):", labels, cur, True)
        if not ok or not new:
            return
        self.panel.apply_ops(self.panel.gt_ops(sel, "relabel", new), "relabel")

    def delete_selected(self):
        sel = [o for o in self._selected() if o.kind == "gt"]
        if sel:
            self.panel.apply_ops(self.panel.gt_ops(sel, "delete"), "delete label")

    def _review(self, status):
        if self.i is not None:
            self.panel.set_review([self.panel.data.keys[self.i]], status)

    def _open_main(self):
        if self.i is not None:
            self.panel.open_in_main(self.i)

    def _fit(self):
        self.canvas.fit()
        self.canvas_b.fit()

    # ------------------------------------------------------------------ compare
    def _fill_models(self):
        d = self.panel.data
        for c in (self.cmp_a, self.cmp_b):
            c.blockSignals(True)
            c.clear()
            for m in d.models:
                c.addItem(get_color_icon(QColor(d.model_colors.get(m, "#ff2bd6"))), d.model_names.get(m, m), m)
            c.blockSignals(False)
        if self.cmp_b.count() > 1:
            self.cmp_b.setCurrentIndex(1)
        self.compare.setEnabled(len(d.models) >= 1)

    def data_changed(self):
        """A new explorer run: models may differ."""
        self._fill_models()
        self._compare_states = None

    def _toggle_compare(self, on):
        self.canvas_b.setVisible(on)
        self.cmp_a.setVisible(on)
        self.cmp_b.setVisible(on)
        self._compare_states = None
        self.refresh()

    def _compare_changed(self, *_):
        self._compare_states = None
        if self.compare.isChecked():
            self.refresh()

    def view_changed(self):
        self._compare_states = None

    def _refresh_compare(self):
        d = self.panel.data
        a, b = self.cmp_a.currentData(), self.cmp_b.currentData()
        if self._compare_states is None or self._compare_states[0] != (a, b):
            sts = []
            for m in (a, b):
                v = View.from_dict(self.panel.view.to_dict())
                v.models = {x: dict(self.panel.view.models.get(x, {}), visible=(x == m)) for x in d.models}
                sts.append(evaluate(d, v, self.panel.review_map() if v.review is not None else None))
            self._compare_states = ((a, b), sts)
        sts = self._compare_states[1]
        every = self.everything.isChecked()
        for c, m, s in ((self.canvas, a, sts[0]), (self.canvas_b, b, sts[1])):
            objs = objects(d, s, self.i, everything=every)
            c.title = d.model_names.get(m, m) if m else ""
            c.set_objects([o for o in objs if o.kind == "gt" or o.model == m])
