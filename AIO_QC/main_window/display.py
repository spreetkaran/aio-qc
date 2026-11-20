# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image QC main window — Display: the image and its shapes, labels, context menus.

A mixin of AIO_QC.app.MainWindow: the methods use the window's own attributes (self.canvas, self.session …)
set up in MainWindow.__init__.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QEvent, Qt, QThreadPool
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QComboBox, QInputDialog, QLineEdit, QListWidgetItem, QMenu, QPlainTextEdit, QTextEdit,
)

from AIO_QC import utils
from AIO_QC.core import AnnotationError
from AIO_QC.widgets import (
    PreloadTask,
    load_qimage,
)
from AIO_QC.widgets.guidelines import add_selected as add_guide_example

from AIO_QC.app_common import (natsorted)


class DisplayMixin:

    # =====================================================================
    # Display
    # =====================================================================
    def show_sample(self, key: str):
        if not self.session or key not in self.session.by_key:
            return
        sample = self.session.by_key[key]
        same = key == self.current_key
        if not same:
            self._flush_display()
            self._display_for_next(key)
        self.current_key = key
        self._update_review_badge()               # first: the mark must never lag behind the image
        self.dataset_label.set_location(sample.split, Path(sample.name).stem)
        shape_to_focus, self._pending_shape = self._pending_shape, None
        extras, self._pending_extras = self._pending_extras, (None, None)
        try:
            ann = self.session.adapter.load(sample)
            error = None
        except AnnotationError as exc:
            ann, error = None, str(exc)
        pix = self.pixmap_cache.get(key)
        if pix is None:
            img = load_qimage(ann.image_path if ann else sample.image_path, ann.image_data if ann else None)
            pix = QPixmap.fromImage(img) if not img.isNull() else QPixmap()
            if not pix.isNull():
                self.pixmap_cache.put(key, pix, pix.width() * pix.height() * 4)
        if pix.isNull():
            self.canvas.clearAll()
            self.zoom_spin.setEnabled(False)
            self.zoom_spin.set_zoom(None)
            from AIO_QC.utils import memory as _mem
            msg = error or (f"Not enough free memory to open {sample.name} ({_mem.human(_mem.available())} free)"
                            if _mem.is_low() else f"Image not found for {sample.name}")
            self.statusBar().showMessage(f"⚠ {msg}", 8000)
            self.log_errors([f"{sample.name}: {msg}"], quiet=True)
        elif not same or self.canvas.pixmap_item.pixmap().cacheKey() != pix.cacheKey():
            self.canvas.loadPixmap(pix, same_image=same)       # re-read after an outside change: same view
            self.zoom_spin.setEnabled(True)
        self.statusBar().set_image(None if pix.isNull() else pix.width(), None if pix.isNull() else pix.height())
        self._status_file(sample, ann, pix)
        if ann is not None and ann.normalized and not pix.isNull():
            ann = ann.scaled(pix.width(), pix.height())
        if error:
            self.statusBar().showMessage(f"⚠ {sample.name}: {error}", 8000)
        self.current_ann = ann
        self._register_labels(ann.label_counts() if ann else Counter())
        self.update_annotations_list()
        self.load_canvas_shapes()
        self.refresh_pred_layer()
        self.inline.refresh()
        self.assist.image_shown()
        self.sketch.image_changed(key)
        self.roi.image_changed(key)
        self.history_panel.set_key(key)
        self._update_position()
        if shape_to_focus is not None or any(e is not None for e in extras):
            utils.later(30, self, lambda: self._apply_extras(shape_to_focus, *extras))
        self.preload_neighbors()
        self._update_review_badge()
        self._update_filmstrip()
        self._check_current_image()

    def _hud_pointer(self, pt):
        pm = self.canvas.pixmap_item.pixmap()
        wx, wy = len(f"{max(1, pm.width()):,}"), len(f"{max(1, pm.height()):,}")
        # padded to the image's widest values (mono digits): the read-out keeps one width while the pointer moves
        txt = "" if pt is None else f"x {int(pt.x()):>{wx},} · y {int(pt.y()):>{wy},}"
        self.hud.set_pointer(" · ".join(x for x in (txt, self._hud_sel) if x))

    def _hud_selection(self, rect):
        """Canvas read-out: the selected shape's / drawn box's / ROI's size and position."""
        if rect is None or (rect.width() <= 0 and rect.height() <= 0):
            self._hud_sel = ""
        else:
            self._hud_sel = (f"box {rect.width():.0f} × {rect.height():.0f} at {rect.left():.0f}, "
                             f"{rect.top():.0f}")
        self.hud.set_pointer(self._hud_sel)

    def _hud_file(self, sample, pix):
        sb = self.statusBar()
        details = " · ".join(x for x in (
            "" if pix.isNull() else f"{pix.width()} × {pix.height()}", sb.ftype.text.text(), sb.fsize.text.text())
            if x)
        name = Path(sample.image_path).name if sample.image_path else sample.name
        self.hud.set_file("" if pix.isNull() else name, details)

    def _status_file(self, sample, ann, pix):
        """Status bar: file type and file size of the image — shown only when it decoded."""
        sb = self.statusBar()
        try:
            self._status_file_inner(sb, sample, ann, pix)
        finally:
            self._hud_file(sample, pix)

    def _status_file_inner(self, sb, sample, ann, pix):
        if pix.isNull():
            sb.set_file(None)
            return
        path = (ann.image_path if ann is not None else None) or sample.image_path
        data = getattr(ann, "image_data", None) if ann is not None else None
        if path is not None and Path(path).is_file():
            sb.set_file(path)
        elif data:                                        # LabelMe imageData (embedded base64)
            import base64
            from PyQt6.QtCore import QBuffer, QByteArray
            from PyQt6.QtGui import QImageReader
            raw = base64.b64decode(data)
            buf = QBuffer()
            buf.setData(QByteArray(raw))
            fmt = bytes(QImageReader(buf).format()).decode() or None
            sb.set_file(None, len(raw), fmt)
        else:
            sb.set_file(None)

    def reload_current_annotation(self):
        """Refresh shapes after an edit without resetting zoom."""
        if not self.session or self.current_key not in (self.session.by_key if self.session else {}):
            return
        try:
            ann = self.session.load(self.current_key)
        except AnnotationError as exc:
            self.statusBar().showMessage(f"⚠ {exc}", 6000)
            return
        pix = self.canvas.pixmap_item.pixmap()
        if ann.normalized and not pix.isNull():
            ann = ann.scaled(pix.width(), pix.height())
        self.current_ann = ann
        self._register_labels(ann.label_counts())
        self.update_annotations_list()
        self.load_canvas_shapes()
        self.inline.refresh()

    def _update_position(self):
        i, n = self.file_dock.current_position()
        if self.file_dock.is_parked():
            self.pos_label.setText(f"– / {n:,}")
            self.pos_label.setToolTip("The open file is not in the current list (filtered out after marking)")
            return
        total = self.file_dock.model.rowCount()
        self.pos_label.setToolTip(f"{n:,} of {total:,} files in the list (filtered)" if n != total else
                                  f"{total:,} files")
        self.pos_label.setText((f"{i:,} / {n:,}" if n == total else f"{i:,} / {n:,} of {total:,}") if n else "")
        # as wide as the largest position this list can show: the items to its left stay put from file to file
        eights = str.maketrans("0123456789", "8888888888")
        widest = (f"{n:,} / {n:,}" if n == total else f"{n:,} / {n:,} of {total:,}").translate(eights)
        need = self.pos_label.fontMetrics().horizontalAdvance(widest) + 14
        if self.pos_label.minimumWidth() != need:
            self.pos_label.setMinimumWidth(need)

    def load_canvas_shapes(self):
        shapes = self.current_ann.shapes if self.current_ann else []
        self.canvas.loadShapes(shapes, self.label_colors, self.visible_labels)

    def release_memory(self, fraction: float = 1.0):
        """Image QC went to the background (another mode in front) or memory is short: drop the decoded
        neighbour images (re-read when needed — nothing is lost) and trim the thumbnail / gallery caches."""
        cur = self.pixmap_cache.get(self.current_key) if self.current_key else None
        self.pixmap_cache.clear()
        if cur is not None:
            self.pixmap_cache.put(self.current_key, cur, cur.width() * cur.height() * 4)
        utils.io.BUDGET.trim(fraction * 0.5)
        import gc
        gc.collect()

    def _on_low_memory(self, what: str):
        """An image buffer could not be allocated: free what the app can spare (cached images) and say so — once
        in a while, not for every frame. The image itself stays on screen in the fast view."""
        from AIO_QC.utils import memory as _mem
        freed = utils.io.BUDGET.total()
        self.release_memory()
        freed -= utils.io.BUDGET.total()
        logger.warning(f"[memory] low memory: {what} skipped; freed {_mem.human(freed)} of cached images "
                       f"(available now: {_mem.human(_mem.available())})")
        self._notify_low_memory(what, freed)

    def _notify_low_memory(self, what: str, freed: int | None = None):
        from AIO_QC.utils import memory as _mem
        now = time.monotonic()
        if now - self._low_mem_told < 120:
            return
        self._low_mem_told = now
        skipped = ("the high-quality view was skipped for this image (it is shown in the fast view)"
                   if what == "high-quality view" else
                   "brightness / contrast was skipped for this image (shown unadjusted)"
                   if what.startswith("brightness") else f"the {what} could not finish")
        self.toast.show_message(
            f"<b>Memory is running low</b> — {skipped}. Cached images were freed"
            + (f" ({_mem.human(freed)})" if freed else "") + f"; free memory: {_mem.human(_mem.available())}. "
            "Closing other programs or large windows helps.", "warn", key="low_memory")

    def preload_neighbors(self):
        from AIO_QC.utils import memory as _mem
        if _mem.is_low():                       # memory is short: decode only the image that is opened
            return
        keys = self.file_dock.visible_keys(around=self.current_key, before=1, after=3)
        pool = QThreadPool.globalInstance()
        for k in keys:
            if k in self.pixmap_cache or k in self._inflight or k not in self.session.by_key:
                continue
            self._inflight.add(k)
            pool.start(PreloadTask(self.preload_signals, self.session.adapter, self.session.by_key[k], self.generation))

    def _on_preloaded(self, key, gen, img):
        self._inflight.discard(key)
        if gen != self.generation or img.isNull() or key in self.pixmap_cache:
            return
        pix = QPixmap.fromImage(img)   # QPixmap only ever created on the GUI thread
        self.pixmap_cache.put(key, pix, pix.width() * pix.height() * 4)

    # ------------------------------------------------------------------ labels
    def _register_labels(self, labels):
        new = False
        for label in labels:
            if label not in self.label_colors:
                utils.update_label_colors(label, self.label_colors)
            if label not in self.visible_labels and label not in self.hidden_labels:
                self.visible_labels.add(label)
                new = True
        if new:
            self._labels_refresh.start()

    def refresh_label_list(self):
        """Rebuild the dataset labels dock with counts; keep visibility state."""
        if not self.session:
            return
        totals = self.session.label_totals()
        if self.current_ann:
            for l in self.current_ann.label_counts():
                totals.setdefault(l, 0)
        for name in self.session.adapter.class_names:
            totals.setdefault(name, 0)
        self.unique_labels.blockSignals(True)
        self.unique_labels.clear()
        for label in sorted(totals, key=lambda l: -totals[l]):      # common classes get the most distinct colours
            utils.update_label_colors(label, self.label_colors)
        from AIO_QC.utils import colors as _colors
        _colors.save(self.session.root, self.label_colors)
        from AIO_QC.core.check_guide import label_variants
        from AIO_QC.widgets.panels import COUNT_ROLE, HINT_ROLE
        same = {}                                   # the rarer spellings point at the common one
        for group in label_variants(totals):
            main = max(group, key=lambda l: totals[l])
            for l in group:
                if l != main:
                    same[l] = main
        for label in natsorted(totals):
            utils.update_label_colors(label, self.label_colors)
            if label not in self.hidden_labels:
                self.visible_labels.add(label)
            it = QListWidgetItem(label)
            it.setData(COUNT_ROLE, f"{totals[label]:,}")
            if label in same:
                it.setData(HINT_ROLE, f"same as “{same[label]}”?")
                it.setToolTip(f"'{label}' looks like '{same[label]}' (case / spaces / - or _) — QC ▸ Label "
                              "Ambiguities, or right-click ▸ Rename / merge")
            it.setData(Qt.ItemDataRole.UserRole, label)
            if label not in same:
                it.setToolTip(f"{label} — {totals[label]:,} shape(s)")
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked if label in self.hidden_labels else Qt.CheckState.Checked)
            it.setIcon(utils.get_color_icon(self.label_colors[label]))
            self.unique_labels.addItem(it)
        self.unique_labels.blockSignals(False)
        self.labels_title.set_count(self.unique_labels.count())
        self._sync_all_labels_action()

    def _sync_all_labels_action(self):
        self.a.all_labels.blockSignals(True)
        self.a.all_labels.setChecked(not self.hidden_labels)
        self.a.all_labels.blockSignals(False)
        eye = getattr(self, "_labels_eye", None)
        if eye is not None:                                   # the header's eye: show all ⇄ hide all
            eye.set_off(bool(self.hidden_labels))             # the slash glides in / out

    def _toggle_all_labels_eye(self):
        show = bool(self.hidden_labels)                       # something hidden: show all; else hide all
        self.a.all_labels.setChecked(show)
        self.toggle_all_labels(show)
        self._sync_all_labels_action()

    def toggle_label_visibility(self, item):
        label = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self.visible_labels.add(label)
            self.hidden_labels.discard(label)
        else:
            self.visible_labels.discard(label)
            self.hidden_labels.add(label)
        self._sync_all_labels_action()
        self.load_canvas_shapes()
        self.update_annotations_list()

    def toggle_all_labels(self, checked):
        self.unique_labels.blockSignals(True)
        for i in range(self.unique_labels.count()):
            it = self.unique_labels.item(i)
            label = it.data(Qt.ItemDataRole.UserRole)
            it.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            (self.visible_labels.add if checked else self.visible_labels.discard)(label)
            (self.hidden_labels.discard if checked else self.hidden_labels.add)(label)
        self.unique_labels.blockSignals(False)
        self.unique_labels.deselect()
        self.load_canvas_shapes()
        self.update_annotations_list()

    def _solo_label(self, label):
        self.hidden_labels = {self.unique_labels.item(i).data(Qt.ItemDataRole.UserRole)
                              for i in range(self.unique_labels.count())} - {label}
        self.visible_labels = {label}
        self.refresh_label_list()
        self.unique_labels.deselect()
        self.load_canvas_shapes()
        self.update_annotations_list()

    def update_annotations_list(self):
        cls = self.session is not None and self.session.is_classification
        stack = getattr(self, "_ann_stack", None)
        if stack is not None:
            stack.setCurrentIndex(1 if cls else 0)
            self.annotations_title.title.setText("Class" if cls else "Annotations")
            if cls:                                   # a classifier's image has a class, not shapes
                self.annotations_list.clear()
                self.class_panel.refresh()
                self.annotations_title.set_count(None)
                self.annotations_dock.toggleViewAction().setText("Annotations")
                return
        self.annotations_list.blockSignals(True)
        self.annotations_list.clear()
        if self.current_ann:
            for i, shape in enumerate(self.current_ann.shapes):
                label = str(shape.get("label", "?"))
                utils.update_label_colors(label, self.label_colors)
                desc = str(shape.get("description") or "").strip().replace("\n", " ")
                it = QListWidgetItem(label + (f" · {desc[:60]}" if desc else ""))
                from AIO_QC.widgets.panels import COUNT_ROLE
                it.setData(COUNT_ROLE, f"#{i}")
                it.setData(Qt.ItemDataRole.UserRole, i)
                it.setIcon(utils.shapeIcon(shape.get("shape_type"), self.label_colors[label]))
                from AIO_QC.core.provenance import describe_shape
                made = describe_shape(shape)                 # who / what made it (no person), how often changed
                it.setToolTip(f"{shape.get('shape_type') or 'polygon'} · {len(shape.get('points') or [])} points"
                              + (f"\n{made}" if made else ""))
                if label not in self.visible_labels:
                    it.setForeground(utils.theme_color("faint"))
                    it.setToolTip("Hidden (label unticked in the Labels panel)")
                self.annotations_list.addItem(it)
        self.annotations_title.set_count(self.annotations_list.count() if self.current_ann else None)
        self.annotations_dock.toggleViewAction().setText("Annotations")   # menus: the name only, never the count
        self.annotations_list.blockSignals(False)
        self._sync_annotation_selection()                    # the rebuilt list keeps the image's selection
        cv = getattr(self, "crop_viewer", None)              # an open Crop Viewer follows the image / the edits
        if cv is not None:
            try:
                cv.on_image_changed() if cv.key != self.current_key else cv.on_shapes_changed()
            except RuntimeError:
                self.crop_viewer = None

    def view_crop(self, idx: int | None = None):
        """Annotations panel ▸ right-click ▸ View crop: the shape in the Crop Viewer window."""
        if idx is None:
            sel = self.canvas.selected_group
            idx = sel.index if sel is not None else None
        if idx is None or not self.current_ann or not (0 <= idx < len(self.current_ann.shapes)):
            self.statusBar().showMessage("Select an annotation to view its crop", 4000)
            return
        from AIO_QC.widgets.crop_viewer import open_crop
        open_crop(self, idx)

    def _on_annotation_selection(self):
        """Rows picked in the Annotations panel (click, Ctrl / Shift + click, arrows, Ctrl+A) select the same shapes
        on the image; the primary one is the row clicked last (the current row)."""
        lst = self.annotations_list
        idxs = [it.data(Qt.ItemDataRole.UserRole) for it in lst.selectedItems()]
        self._ann_row_changed = True
        if not idxs:
            self.canvas.clearSelection()
            return
        if len(idxs) == 1:
            self._select_from_list(idxs[0])
            return
        cur = lst.currentItem()
        prim = cur.data(Qt.ItemDataRole.UserRole) if cur is not None and cur.isSelected() else None
        n = self.canvas.select_many(idxs, primary=prim)
        if n < len(idxs):
            self.statusBar().showMessage(f"{len(idxs) - n} of them cannot be selected on the image (hidden label "
                                         "or no geometry)", 5000)
        self._sync_annotation_selection()

    def _sync_annotation_selection(self, *_):
        """The Annotations panel shows exactly what is selected on the image (several rows for several shapes)."""
        from PyQt6.QtCore import QItemSelectionModel
        lst, c = self.annotations_list, self.canvas
        sel = set(c.selected_indices())
        prim = c.selected_group.index if c.selected_group is not None else None
        sm = lst.selectionModel()
        lst.blockSignals(True)
        try:
            cur = None
            for r in range(lst.count()):
                it = lst.item(r)
                i = it.data(Qt.ItemDataRole.UserRole)
                if it.isSelected() != (i in sel):
                    it.setSelected(i in sel)
                if i == prim:
                    cur = it
            if cur is not None:
                sm.setCurrentIndex(lst.indexFromItem(cur), QItemSelectionModel.SelectionFlag.NoUpdate)
                lst.scrollToItem(cur)
            elif not sel:
                lst.setCurrentRow(-1)
        finally:
            lst.blockSignals(False)

    def _shape_clip(self, kind: str):
        """Ctrl+C / X / V / A in the Image QC window. A text field with the focus keeps its own meaning (Qt already
        gives it these keys first — this is the safety net); Ctrl+A in the file list selects its files; everywhere
        else they act on the shapes of the shown image."""
        from PyQt6.QtGui import QKeyEvent
        fw = QApplication.focusWidget()
        texty = isinstance(fw, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)) or (
            isinstance(fw, QComboBox) and fw.isEditable())
        if texty and fw.isEnabled():
            key = {"copy": Qt.Key.Key_C, "cut": Qt.Key.Key_X, "paste": Qt.Key.Key_V, "select_all": Qt.Key.Key_A}[kind]
            for t in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                QApplication.sendEvent(fw, QKeyEvent(t, key, Qt.KeyboardModifier.ControlModifier))
            return
        if kind == "select_all" and isinstance(fw, QAbstractItemView) and fw is not self.annotations_list and \
                fw.selectionMode() in (QAbstractItemView.SelectionMode.ExtendedSelection,
                                       QAbstractItemView.SelectionMode.MultiSelection):
            fw.selectAll()
            return
        getattr(self.inline, kind)()

    def _relabel_key(self, n: int):
        """L / F2 (n = 0: choose) and 1 … 9 (that class) for the selected shape(s), from anywhere in the window. A
        text field with the focus types the key instead."""
        from PyQt6.QtGui import QKeyEvent
        fw = QApplication.focusWidget()
        texty = isinstance(fw, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)) or (
            isinstance(fw, QComboBox) and fw.isEditable())
        if texty and fw.isEnabled():
            key = Qt.Key.Key_L if n == 0 else Qt.Key(Qt.Key.Key_0 + n)
            text = "l" if n == 0 else str(n)
            for t in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                QApplication.sendEvent(fw, QKeyEvent(t, key, Qt.KeyboardModifier.NoModifier, text))
            return
        if self.session is not None and self.session.is_classification:
            self.change_class() if n == 0 else self.class_key(int(n))    # classification: the image's class
            return
        c = self.canvas
        g = c.selected_group
        if g is None or self.current_ann is None:
            self.statusBar().showMessage("Select a shape first (click it on the image or in Annotations), then L — "
                                         "or 1 … 9 for that class", 4000)
            return
        if not c.editable:
            self.statusBar().showMessage(f"✋ {c.edit_note or 'this file cannot be edited'}", 5000)
            return
        c._flush_nudge()
        self.inline.on_relabel(g.index, int(n))

    def _select_from_list(self, idx):
        """The list and the canvas always agree: a shape that cannot be selected leaves no row selected, and a
        hidden-label shape is shown while selected (deselecting hides it again)."""
        if not self.canvas.select_shape(idx, emit=False):
            self.onSelectionCleared()
            self.statusBar().showMessage(f"#{idx} cannot be shown (no geometry)", 4000)
            return
        if self.canvas.is_peeked(idx) and self.current_ann and 0 <= idx < len(self.current_ann.shapes):
            label = self.current_ann.shapes[idx].get("label")
            self.statusBar().showMessage(f"#{idx} '{label}' is hidden in the Labels panel — shown while selected; "
                                         "click elsewhere to hide it again", 6000)

    def _on_annotation_clicked(self, item):
        """Clicking the row that is already selected toggles the selection off (works in both modes)."""
        if self._ann_row_changed:              # this click just selected the row
            self._ann_row_changed = False
            return
        if QApplication.keyboardModifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            return                             # Ctrl / Shift + click: the selection handler has it
        idx = item.data(Qt.ItemDataRole.UserRole)
        sel = self.canvas.selected_group
        if sel is not None and sel.index == idx:
            self.canvas.clearSelection()
        else:
            self._select_from_list(idx)

    def onShapeSelected(self, index: int):
        cv = getattr(self, "crop_viewer", None)
        if cv is not None:
            try:
                cv.on_selected(index)
            except RuntimeError:
                self.crop_viewer = None
        if index is not None and index >= 0:
            self.unique_labels._deselect_now()        # a shape is the selection now, not a Labels dock row
        self._sync_annotation_selection()
        if self.current_ann and 0 <= index < len(self.current_ann.shapes):
            s = self.current_ann.shapes[index]
            self.statusBar().showMessage(f"#{index} {s.get('label')} · {s.get('shape_type', 'polygon')} · "
                                         f"{len(s.get('points', []))} pts", 4000)

    def onSelectionCleared(self):
        self.annotations_list.blockSignals(True)
        self.annotations_list.clearSelection()
        self.annotations_list.setCurrentRow(-1)
        self.annotations_list.blockSignals(False)

    # ------------------------------------------------------------ context menus
    def _annotations_menu(self, pos):
        it = self.annotations_list.itemAt(pos)
        if not it or not self.current_ann:
            return
        idx = it.data(Qt.ItemDataRole.UserRole)
        label = str(self.current_ann.shapes[idx].get("label"))
        m = QMenu(self)
        I = utils.newIcon
        m.addAction(I("crop-view"), "View crop…  (Shift+C)", lambda: self.view_crop(idx))
        m.addAction(I("zoom"), "Zoom to shape", lambda: self.canvas.focus_shape(idx))
        m.addAction(I("edit"), f"Rename '{label}'…", lambda: self.open_label_map(focus=label, scope=0))
        sel = self.canvas.selected_indices()
        many = len(sel) > 1 and idx in sel
        if self.canvas.editable:
            m.addSeparator()
            if many:
                m.addAction(I("tool-class"), f"Relabel these {len(sel)} shapes…  (L)",
                            lambda: self.inline.on_relabel(idx, 0))
                m.addAction(I("delete"), f"Delete these {len(sel)} shapes  (Del)",
                            lambda: self.inline.on_delete_many(list(sel)))
            else:
                m.addAction(I("tool-class"), "Relabel this shape…  (L)", lambda: self.inline.on_relabel(idx, 0))
                m.addAction(I("delete"), "Delete this shape  (Del)", lambda: self.inline.on_delete(idx))
        m.addSeparator()
        for act in (self.a.copy_shapes, self.a.cut_shapes, self.a.paste_shapes, self.a.select_all_shapes):
            m.addAction(act)
        m.addSeparator()
        m.addAction(I("review-ok"), f"Add as good example of '{label}'  (G)",
                    lambda: (self.canvas.select_shape(idx), add_guide_example(self, "good")))
        m.addAction(I("review-clear"), f"Add as bad example of '{label}'  (Shift+G)",
                    lambda: (self.canvas.select_shape(idx), add_guide_example(self, "bad")))
        m.addAction(I("tool-guidelines"), f"Guideline for '{label}'…", lambda: self.open_guidelines(label))
        m.exec(self.annotations_list.mapToGlobal(pos))

    def _labels_list_menu(self, pos):
        it = self.unique_labels.itemAt(pos)
        if not it or not self.session:
            return
        label = it.data(Qt.ItemDataRole.UserRole)
        m = QMenu(self)
        I = utils.newIcon
        m.addAction(I("edit"), f"Rename / merge '{label}'…", lambda: self.open_label_map(focus=label))
        m.addAction(I("list"), "Show files with this label",
                    lambda: self.apply_file_filter(self.session.keys_with_label(label), f"Files with '{label}'"))
        m.addAction(I("zoom"), "Show only this label", lambda: self._solo_label(label))
        m.addSeparator()
        m.addAction(I("tool-label-map"), "Label mapping…", self.open_label_map)
        m.exec(self.unique_labels.mapToGlobal(pos))

    def _rename_selected_label(self):
        if not self.session:
            return
        label = None
        if self.canvas.selected_group is not None:
            label = str(self.canvas.selected_group.shape_data.get("label"))
        elif self.unique_labels.currentItem() is not None:
            label = self.unique_labels.currentItem().data(Qt.ItemDataRole.UserRole)
        if label is None:
            labels = natsorted(self.session.label_totals())
            if not labels:
                return
            label, ok = QInputDialog.getItem(self, "Rename label", "Label:", labels, 0, False)
            if not ok:
                return
        self.open_label_map(focus=label)

    def _on_annotations_changed(self, keys):
        self.history_panel.touched(keys)
        self._class_scan = None                    # class ids may have changed
        for k in keys:
            self.session.index_errors.pop(k, None)
        self.file_dock.model.set_flags({k: ("empty" if not self.session.index.get(k) else None) for k in keys})
        if self.current_key in keys:
            self.reload_current_annotation()
        self._refresh_open_checks(keys)

    def _refresh_open_checks(self, keys):
        """A fix made on the image (or anywhere): every open check that listed those files looks at them again, so a
        fixed finding leaves its list (F8 then goes on to the next one)."""
        ks = set(keys)
        for dlg in list(self.open_dialogs.values()):
            try:
                if not getattr(dlg, "RESCAN_AFTER_EDIT", False) or getattr(dlg, "ctx", None) is None:
                    continue
                if dlg.worker is not None and dlg.worker.isRunning():
                    continue
                hit = {r.key for r in dlg.rows if r is not None and r.key in ks}
                if hit:
                    dlg.after_edit(hit)
                    if hasattr(dlg, "summarize"):
                        dlg.status.setText(dlg.summarize())
            except RuntimeError:                       # closed meanwhile
                continue
            except Exception as exc:
                logger.debug(f"refresh after edit: {exc}")

    def _on_samples_removed(self, keys):
        for k in keys:
            self.pixmap_cache.pop(k)
        was_current = self.current_key in set(keys)
        if was_current:
            self.current_key = None
        self.file_dock.remove_keys(keys)
        if was_current and self.file_dock.current_key() is None:
            self.canvas.clearAll()
            self.current_ann = None
            self.update_annotations_list()
        self.dataset_label.set_info(f"{self.session.type.title} · {len(self.session.samples):,} files")
        self._update_position()
        self.refresh_status_buttons()
