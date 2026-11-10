# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
🔭 Prediction Explorer — every image of the dataset with your labels and every model's predictions drawn on it.

    sidebar     layers (labels + each model with its own confidence), colour by match / class / model, match IoU,
                match results (correct · wrong class · poor fit · duplicate · not labelled · missed · not checked ·
                models disagree), classes, split / folder / review / training filters — every count live
    grid        images, objects drawn live over cached thumbnails; sort by errors, missed, unlabelled, wrong
                class, box fit, uncertainty, model disagreement, confidence, name, random
    objects     every object as a crop (labels, predictions or only the problems), sorted by class / confidence
    map         the objects Visual Similarity embedded, on a 2-D map coloured by match result or class
    thresholds  per model (and class): confidence histogram with a draggable threshold, precision / recall curve
    viewer      double-click: full resolution, hover details, accept / relabel / delete / add, compare 2 models
    views       save / restore named views (filters + sort + thresholds) per dataset
Edits go through the session (backed up, undoable); the labels of changed files are re-matched on the fly.
"""
from __future__ import annotations

import os
import threading
import time
from collections import Counter

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QMenu, QMessageBox, QSlider, QSpinBox, QSplitter, QStackedWidget,
    QToolButton, QVBoxLayout, QWidget,
)

from AIO_QC.core import explorer as E
from AIO_QC.core.explorer import TAG_NAMES, View
from AIO_QC.utils import get_color_icon, newButton, style as apply_style
from ..inference_service import FuncWorker
from .charts import ConfHistogram, PRCurve, ScatterMap
from .grid import ExplorerGrid, chips_for
from .paint import object_text, tag_color
from .sidebar import Sidebar

PATCH_SOURCES = [("errors", "Problems only"), ("gt", "Labels"), ("pred", "Predictions")]
PATCH_SORTS = [("class", "Class, least confident first"), ("conf", "Confidence"), ("tag", "Match result"),
               ("image", "Image")]
SETTINGS_VIEW = "explorer/last_view"


class ExplorerPanel(QWidget):
    def __init__(self, dialog):
        super().__init__(dialog)
        self.dlg = dialog
        self.app = dialog.app
        self.session = dialog.session
        self.data: E.ExplorerData | None = None
        self.state: E.State | None = None
        self.view = View()
        self.lock = threading.RLock()
        self.data_version = 0
        self.patch_source, self.patch_sort = "errors", "class"
        self.patch_override = None                    # explicit object codes (map selection)
        self.undo_stack: list = []
        self.viewer = None
        self._eval_gen = 0
        self._worker = None
        self._pending = False
        self._obj_cache: dict = {}
        self._dirty_keys: set = set()
        self._map = None                              # (label rows, xy) of the computed embedding map
        self._unsaved = False

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self.split)
        self.sidebar = Sidebar()
        self.sidebar.changed.connect(self.schedule)
        self.split.addWidget(self.sidebar)

        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(4)
        cv.addLayout(self._build_topbar())
        self.banner = QLabel()
        self.banner.setTextFormat(Qt.TextFormat.RichText)
        self.banner.setVisible(False)
        self.banner.linkActivated.connect(lambda _l: self.clear_keys())
        apply_style(self.banner, "QLabel{background:@banner_bg; color:@banner_fg; border:1px solid @banner_line; "
                                 "border-radius:4px; padding:4px 8px;}")
        cv.addWidget(self.banner)
        self.pages = QStackedWidget()
        self.grid = ExplorerGrid(self)
        self.grid.opened.connect(self.open_item)
        self.grid.selectionChangedSoon.connect(self._selection_changed)
        self.grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self._grid_menu)
        self.grid.set_tile_width(self.size.value())
        self.pages.addWidget(self._wrap_grid())
        self.pages.addWidget(self._build_map())
        cv.addWidget(self.pages, 1)
        self.status = QLabel("Run the check (▶ Scan) — or reopen: the last run is restored automatically.")
        apply_style(self.status, "color:@muted;")
        cv.addWidget(self.status)
        self.split.addWidget(center)
        self.insight = self._build_insight()
        self.insight.setVisible(False)
        self.split.addWidget(self.insight)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([300, 1000, 320])

        self._timer = QTimer(self, singleShot=True, interval=90, timeout=self.refresh)
        self._edit_timer = QTimer(self, singleShot=True, interval=250, timeout=self._apply_dirty)
        self.session.annotationsChanged.connect(self.on_annotations_changed)
        self.session.reviewChanged.connect(lambda _k: self._on_review_changed())
        self._enable(False)

    # ================================================================== layout
    def _build_topbar(self):
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.mode_group = QButtonGroup(self)
        self.mode_btns = {}
        for k, text, tip in (("grid", "▦ Images", "Every image with its labels and predictions"),
                             ("objects", "▣ Objects", "Every object as a crop — find mislabels fast"),
                             ("map", "◎ Map", "Objects on a 2-D map by how they look (Visual Similarity "
                                             "embeddings), coloured by match result")):
            b = QToolButton()
            b.setText(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            self.mode_group.addButton(b)
            self.mode_btns[k] = b
            bar.addWidget(b)
            b.clicked.connect(lambda _c, k=k: self.set_mode(k))
        self.mode_btns["grid"].setChecked(True)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter file names…")
        self.search.setClearButtonEnabled(True)
        self.search.setMaximumWidth(220)
        self.search.setMinimumWidth(130)
        self.search.textChanged.connect(lambda _t: self.schedule())
        bar.addWidget(self.search)
        self.sort = QComboBox()
        for k, t in E.SORTS:
            self.sort.addItem(t, k)
        self.sort.setToolTip("Order of the images")
        self.sort.currentIndexChanged.connect(lambda _i: self.schedule())
        self.desc = QToolButton()
        self.desc.setCheckable(True)
        self.desc.setChecked(True)
        self.desc.setText("↓")
        self.desc.setToolTip("Descending / ascending")
        self.desc.toggled.connect(lambda on: (self.desc.setText("↓" if on else "↑"), self.schedule()))
        self.shuffle = QToolButton()
        self.shuffle.setText("🎲")
        self.shuffle.setToolTip("New random order")
        self.shuffle.clicked.connect(self._reshuffle)
        self.psource = QComboBox()
        for k, t in PATCH_SOURCES:
            self.psource.addItem(t, k)
        self.psort = QComboBox()
        for k, t in PATCH_SORTS:
            self.psort.addItem(t, k)
        self.psource.currentIndexChanged.connect(self._patch_settings)
        self.psort.currentIndexChanged.connect(self._patch_settings)
        for w in (self.sort, self.desc, self.shuffle, self.psource, self.psort):
            bar.addWidget(w)
        self.size = QSlider(Qt.Orientation.Horizontal)
        self.size.setRange(110, 560)
        self.size.setValue(int(self.app.settings.value("explorer/tile", 230)))
        self.size.setFixedWidth(110)
        self.size.setToolTip("Tile size (Ctrl + wheel over the grid)")
        self.size.valueChanged.connect(self._tile_size)
        bar.addWidget(self.size)
        bar.addStretch(1)
        self.views = QComboBox()
        self.views.setMinimumWidth(140)
        self.views.setToolTip("Saved views of this dataset (filters, sort, thresholds)")
        self.views.activated.connect(self._apply_saved_view)
        bar.addWidget(self.views)
        vb = QToolButton()
        vb.setText("☰")
        vb.setToolTip("Save / delete views")
        vm = QMenu(vb)
        vm.addAction("Save current view as…", self._save_view)
        vm.addAction("Delete the selected view", self._delete_view)
        vm.addAction("Reset to defaults", self.reset_view)
        vb.setMenu(vm)
        vb.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        bar.addWidget(vb)
        self.accept_btn = newButton("✓ Accept…", slot=self.accept_dialog,
                                    tip="Accept the model's suggestions on the selected images / objects (or on "
                                        "every image of this view): add unlabelled boxes, fix classes, refit boxes")
        self.relabel_btn = newButton("✎ Relabel…", slot=self.relabel_selection,
                                     tip="Objects view: change the class of the selected labels")
        self.delete_btn = newButton("🗑", slot=self.delete_selection, tip="Objects view: delete the selected labels")
        self.undo_btn = newButton("↶", slot=self.undo, tip="Undo the last explorer edit")
        self.undo_btn.setEnabled(False)
        self.thr_btn = QToolButton()
        self.thr_btn.setText("📈 Thresholds")
        self.thr_btn.setCheckable(True)
        self.thr_btn.setToolTip("Confidence histogram (drag the threshold) and precision / recall per model")
        self.thr_btn.toggled.connect(self._toggle_insight)
        more = QToolButton()
        more.setText("⋯")
        more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        mm = QMenu(more)
        mm.addAction("⚑ Flag the selected / shown files", lambda: self.review_selection("flag"))
        mm.addAction("✔ Mark the selected / shown files OK", lambda: self.review_selection("ok"))
        mm.addSeparator()
        mm.addAction("Show in the file list", self.show_in_file_list)
        mm.addAction("Open in Review Grid", self.open_review_grid)
        mm.addAction("Export the list (CSV)…", self.export_csv)
        mm.addSeparator()
        self.layer_action = mm.addAction("Predictions layer in the main viewer")
        self.layer_action.setCheckable(True)
        self.layer_action.setChecked(bool(getattr(self.app, "pred_layer_on", lambda: False)()))
        self.layer_action.toggled.connect(lambda on: getattr(self.app, "set_pred_layer", lambda _on: None)(on))
        more.setMenu(mm)
        for w in (self.accept_btn, self.relabel_btn, self.delete_btn, self.undo_btn, self.thr_btn, more):
            bar.addWidget(w)
        self._mode_widgets()
        return bar

    def _wrap_grid(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.empty = QLabel()
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)
        apply_style(self.empty, "color:@muted; font-size:14px; padding:40px;")
        self.empty.setText("No predictions yet.\n\nPress ▶ Scan above: the models run once (cached), then every "
                           "image is shown here with your labels (solid) and the predictions (dashed).")
        lay.addWidget(self.empty)
        lay.addWidget(self.grid, 1)
        self.grid.setVisible(False)
        return w

    def _build_map(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.map_src = QComboBox()
        self.map_src.setToolTip("Object embeddings cached by Visual Similarity (one per vision model)")
        self.map_method = QComboBox()
        for k, t in (("pca", "PCA (instant)"), ("tsne", "t-SNE (scikit-learn)"), ("umap", "UMAP (umap-learn)")):
            self.map_method.addItem(t, k)
        self.map_n = QSpinBox()
        self.map_n.setRange(500, 200_000)
        self.map_n.setSingleStep(5000)
        self.map_n.setValue(20_000)
        self.map_n.setToolTip("At most this many objects (a random sample)")
        self.map_color = QComboBox()
        self.map_color.addItem("Match result", "match")
        self.map_color.addItem("Class", "class")
        self.map_color.currentIndexChanged.connect(lambda _i: self._recolor_map())
        go = newButton("Compute", slot=self.compute_map)
        sim = newButton("Visual Similarity…", slot=lambda: self.app.open_tool("similarity"),
                        tip="Embed the objects with a vision model first (DINOv3 / SigLIP2 / CLIP)")
        for x in (QLabel("Embeddings"), self.map_src, self.map_method, QLabel("max"), self.map_n, go,
                  QLabel("colour"), self.map_color):
            row.addWidget(x)
        row.addStretch(1)
        row.addWidget(sim)
        lay.addLayout(row)
        self.map = ScatterMap()
        self.map.selected.connect(self._map_selected)
        self.map.activated.connect(self._map_open)
        lay.addWidget(self.map, 1)
        mrow = QHBoxLayout()
        self.map_info = QLabel("Drag a rectangle to select objects · double-click a point to open it")
        apply_style(self.map_info, "color:@muted;")
        self.map_show_img = newButton("Show their images", slot=lambda: self._map_to("grid"))
        self.map_show_obj = newButton("Show the objects", slot=lambda: self._map_to("objects"))
        for b in (self.map_show_img, self.map_show_obj):
            b.setEnabled(False)
        mrow.addWidget(self.map_info, 1)
        mrow.addWidget(self.map_show_img)
        mrow.addWidget(self.map_show_obj)
        lay.addLayout(mrow)
        return w

    def _build_insight(self):
        w = QWidget()
        w.setMinimumWidth(280)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 0, 0, 0)
        t = QLabel("📈 Thresholds")
        apply_style(t, "font-weight:700;")
        lay.addWidget(t)
        f = QFormLayout()
        self.ins_model = QComboBox()
        self.ins_class = QComboBox()
        self.ins_model.currentIndexChanged.connect(lambda _i: self._refresh_insight())
        self.ins_class.currentIndexChanged.connect(lambda _i: self._refresh_insight())
        f.addRow("Model", self.ins_model)
        f.addRow("Class", self.ins_class)
        lay.addLayout(f)
        self.hist = ConfHistogram()
        self.hist.thresholdChanged.connect(self._hist_threshold)
        lay.addWidget(self.hist)
        self.pr = PRCurve()
        lay.addWidget(self.pr)
        self.ins_text = QLabel()
        self.ins_text.setWordWrap(True)
        self.ins_text.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.ins_text)
        self.best_btn = newButton("Use the best-F1 threshold", slot=self._use_best)
        lay.addWidget(self.best_btn)
        lay.addStretch(1)
        return w

    def _mode_widgets(self):
        mode = self.mode()
        for w in (self.sort, self.desc, self.shuffle):
            w.setVisible(mode == "grid")
        for w in (self.psource, self.psort):
            w.setVisible(mode == "objects")
        self.relabel_btn.setVisible(mode == "objects")
        self.delete_btn.setVisible(mode == "objects")
        self.size.setVisible(mode != "map")

    def _enable(self, on: bool):
        for w in (self.sidebar, self.accept_btn, self.thr_btn, self.views, self.sort, self.search):
            w.setEnabled(on)

    # ================================================================== data
    def set_data(self, d: E.ExplorerData | None, note: str = ""):
        with self.lock:
            self.data = d
            self.state = None
            self.data_version += 1
            self._obj_cache.clear()
            self._map = None
        self.undo_stack.clear()
        self.undo_btn.setEnabled(False)
        self.patch_override = None
        self.map.set_points(np.zeros((0, 2)), [])
        if d is None:
            self.grid.set_items("grid", [])
            self._enable(False)
            return
        v = self._restore_view(d)
        self.view = v
        self.sidebar.set_data(d, v)
        self.sort.blockSignals(True)
        self.sort.setCurrentIndex(max(0, self.sort.findData(v.sort)))
        self.sort.blockSignals(False)
        self.desc.blockSignals(True)
        self.desc.setChecked(bool(v.desc))
        self.desc.setText("↓" if v.desc else "↑")
        self.desc.blockSignals(False)
        self.search.blockSignals(True)
        self.search.setText(v.search or "")
        self.search.blockSignals(False)
        self._fill_saved_views()
        self._fill_insight_combos()
        self._fill_map_sources()
        self._enable(True)
        self.empty.setVisible(False)
        self.grid.setVisible(True)
        self.grid.invalidate_images()
        self._note = note
        if self.viewer is not None:
            self.viewer.data_changed()
        self.refresh()

    def _restore_view(self, d) -> View:
        import json
        v = View()
        try:
            raw = self.app.settings.value(SETTINGS_VIEW, "")
            if raw:
                v = View.from_dict(json.loads(raw))
        except (ValueError, TypeError):
            v = View()
        v.keys, v.keys_title = None, ""
        meta_conf = (d.meta or {}).get("conf", 0.25)
        v.models = {m: {"visible": (v.models.get(m) or {}).get("visible", True),
                        "conf": float((v.models.get(m) or {}).get("conf", meta_conf))} for m in d.models}
        if v.classes is not None:
            v.classes = [c for c in v.classes if c in d.labels] or None
        if v.splits is not None:
            v.splits = [s for s in v.splits if s in d.splits] or None
        if v.folders is not None:
            v.folders = [f for f in v.folders if f in d.folders] or None
        return v

    def _remember_view(self):
        import json
        v = View.from_dict(self.view.to_dict())
        v.keys, v.keys_title = None, ""
        try:
            self.app.settings.setValue(SETTINGS_VIEW, json.dumps(v.to_dict()))
        except (TypeError, ValueError):
            pass

    # ================================================================== evaluation
    def schedule(self, *_):
        if self.data is not None:
            self._timer.start()

    def _read_view(self) -> View:
        v = View.from_dict(self.view.to_dict())
        self.sidebar.read_into(v)
        v.sort = self.sort.currentData()
        v.desc = self.desc.isChecked()
        v.search = self.search.text().strip()
        return v

    def review_map(self) -> dict:
        rv = self.session.review
        out = {}
        for k in self.data.keys:
            s = rv.status(k)
            if s:
                out[k] = s
        return out

    def refresh(self):
        if self.data is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._pending = True                       # one more pass when this one finishes
            return
        self.view = self._read_view()
        v = View.from_dict(self.view.to_dict())
        review = self.review_map() if v.review else None
        d = self.data
        self._eval_gen += 1
        gen = self._eval_gen
        t0 = time.monotonic()

        def run():
            with self.lock:
                return E.evaluate(d, v, review)

        def done(st):
            self._worker = None
            if gen != self._eval_gen or d is not self.data:
                return
            self._took = time.monotonic() - t0
            self._apply_state(st)
            if self._pending:
                self._pending = False
                self.refresh()

        def failed(msg):
            self._worker = None
            self.status.setText(f"❌ {msg}")

        n = len(d.p_img) + len(d.g_img)
        if n < 150_000:                                   # small: synchronous, no thread hop
            try:
                done(run())
            except Exception as exc:                      # never leave the panel half-updated
                failed(f"{type(exc).__name__}: {exc}")
            return
        self.status.setText("updating…")
        w = FuncWorker(run, self)
        w.done.connect(done)
        w.failed.connect(failed)
        self._worker = w
        w.start()

    def _apply_state(self, st: E.State):
        self.state = st
        self._obj_cache.clear()
        mode = self.mode()
        if mode == "objects":
            self.grid.set_items("objects", self._patch_codes(), keep_scroll=True)
        elif mode == "grid":
            self.grid.set_items("grid", st.order, keep_scroll=True)
        self.grid.viewport().update()
        self.sidebar.update_counts(st, self.data)
        self._update_status()
        self._remember_view()
        if self.insight.isVisible():
            self._refresh_insight()
        if self.mode() == "map":
            self._recolor_map()
        if self.viewer is not None and self.viewer.isVisible():
            self.viewer.view_changed()
            self.viewer.refresh()
        refresh_layer = getattr(self.app, "refresh_pred_layer", None)
        if refresh_layer is not None:
            refresh_layer()

    def _update_status(self):
        st, d = self.state, self.data
        if st is None:
            return
        n_err = sum(st.tag_counts.get(TAG_NAMES[t], 0) for t in E.ERROR_TAGS)
        parts = [f"<b>{st.n_images:,}</b> of {len(d):,} images"]
        if self.mode() == "objects":
            parts.append(f"<b>{self.grid.model_.rowCount():,}</b> objects")
        parts.append(f"{n_err:,} problems drawn")
        sel = len(self.grid.selected_items())
        if sel:
            parts.append(f"{sel:,} selected")
        took = getattr(self, "_took", 0.0)
        note = getattr(self, "_note", "")
        self.status.setText(" · ".join(parts) + (f" · {note}" if note else "") +
                            (f" · {took * 1000:.0f} ms" if took > 0.05 else ""))
        self.status.setTextFormat(Qt.TextFormat.RichText)
        v = self.view
        if v.keys is not None:
            self.banner.setText(f"Showing {len(v.keys):,} file(s) from <b>{v.keys_title or 'a selection'}</b> — "
                                f"<a href='clear'>show all</a>")
            self.banner.setVisible(True)
        else:
            self.banner.setVisible(False)

    def _selection_changed(self):
        self._update_status()

    # ================================================================== modes
    def mode(self) -> str:
        return next((k for k, b in self.mode_btns.items() if b.isChecked()), "grid")

    def set_mode(self, mode: str):
        self.mode_btns[mode].setChecked(True)
        self._mode_widgets()
        self.pages.setCurrentIndex(1 if mode == "map" else 0)
        if mode != "objects":
            self.patch_override = None
        if self.state is not None:
            if mode == "objects":
                self.grid.set_items("objects", self._patch_codes())
            elif mode == "grid":
                self.grid.set_items("grid", self.state.order)
            elif mode == "map":
                self._fill_map_sources()
                self._recolor_map()
        self._update_status()

    def _patch_settings(self, *_):
        self.patch_source = self.psource.currentData()
        self.patch_sort = self.psort.currentData()
        self.patch_override = None
        if self.state is not None and self.mode() == "objects":
            self.grid.set_items("objects", self._patch_codes())
            self._update_status()

    def _patch_codes(self):
        if self.patch_override is not None:
            return self.patch_override
        return E.patch_rows(self.data, self.state, self.patch_source, self.state.order, self.patch_sort)

    def _tile_size(self, w):
        self.grid.set_tile_width(w)
        self.app.settings.setValue("explorer/tile", int(w))

    def zoom_tiles(self, direction: int):
        self.size.setValue(self.size.value() + direction * 30)

    def _reshuffle(self):
        self.view.seed = E.new_seed()
        i = self.sort.findData("random")
        if self.sort.currentIndex() != i:
            self.sort.setCurrentIndex(i)
        else:
            self.refresh()

    # ================================================================== drawing hooks (grid / viewer)
    def display_name(self, key) -> str:
        s = self.session.by_key.get(key)
        return s.display if s is not None else os.path.basename(key)

    def tile_objects(self, i: int) -> list:
        objs = self._obj_cache.get(i)
        if objs is None:
            if len(self._obj_cache) > 4000:
                self._obj_cache.clear()
            objs = E.objects(self.data, self.state, i) if self.state is not None else []
            self._obj_cache[i] = objs
        return objs

    def tile_chips(self, i: int) -> list:
        if self.state is None:
            return []
        chips = chips_for(self.state.counts[i])
        if not chips and self.state.counts[i].sum():
            chips = [("✓ agrees", tag_color(E.OK).name())]
        return chips

    def paint_style(self, tile=True, small=False) -> dict:
        d = self.data
        return dict(mode=self.view.color, label_colors=self.app.label_colors,
                    model_colors=d.model_colors if d else {}, model_names=None if tile else (d.model_names if d else {}),
                    width=1.3 if small else 1.7, text=bool(tile and self.view.tile_names and not small))

    def _obj_of(self, kind, row):
        d = self.data
        i = int(d.g_img[row] if kind == "gt" else d.p_img[row])
        for o in self.tile_objects(i):
            if o.kind == kind and o.row == row:
                return o, i
        for o in E.objects(d, self.state, i, everything=True):
            if o.kind == kind and o.row == row:
                return o, i
        return None, i

    def object_caption(self, kind, row):
        o, i = self._obj_of(kind, row)
        key = self.data.keys[i]
        if o is None:
            return key, self.display_name(key), []
        chips = [(TAG_NAMES.get(o.tag, ""), tag_color(o.tag).name())]
        if o.kind == "pred":
            chips.append((f"{o.label} {o.conf:.2f}", self.data.model_colors.get(o.model, "#64748b")))
        else:
            chips.append((o.label, "#64748b"))
        return key, self.display_name(key), chips

    def patch_objects(self, code):
        kind, row = E.decode_id(code)
        o, i = self._obj_of(kind, row)
        if o is None:
            return [], None
        out = [x for x in self.tile_objects(i) if not (x.kind == o.kind and x.row == o.row)]
        if o.partner >= 0:                                    # its match, even when a filter hides it
            pk = "gt" if o.kind == "pred" else "pred"
            if not any(x.kind == pk and x.row == o.partner for x in out):
                out += [x for x in E.objects(self.data, self.state, i, everything=True)
                        if x.kind == pk and x.row == o.partner]
        return out + [o], len(out)

    def object_box(self, code):
        kind, row = E.decode_id(code)
        d = self.data
        if kind == "gt":
            return d.keys[int(d.g_img[row])], tuple(map(float, d.g_box[row]))
        return d.keys[int(d.p_img[row])], tuple(map(float, d.p_box[row]))

    def tooltip(self, mode, item) -> str:
        d, st = self.data, self.state
        if mode == "grid":
            c = st.counts[item]
            parts = [f"{int(c[t])} {TAG_NAMES[t]}" for t in range(7) if c[t]]
            sp = d.splits[d.img_split[item]] if len(d.splits) > 1 else ""
            return (f"{d.keys[item]}\n{int(d.img_w[item])}×{int(d.img_h[item])}" + (f" · {sp}" if sp else "") +
                    "\n" + (", ".join(parts) or "nothing drawn") + "\n\ndouble-click: open · right-click: actions")
        kind, row = E.decode_id(item)
        o, i = self._obj_of(kind, row)
        if o is None:
            return d.keys[i]
        return f"{object_text(o, d.model_names, with_tag=True)}\n{d.keys[i]}\n\ndouble-click: open"

    # ================================================================== open
    def open_item(self, item: int):
        if self.mode() == "objects":
            kind, row = E.decode_id(item)
            i = int(self.data.g_img[row] if kind == "gt" else self.data.p_img[row])
            self.open_image(i, (kind, row))
        else:
            self.open_image(item)

    def open_image(self, i: int, focus=None):
        if self.viewer is None:
            from .viewer import ExplorerViewer
            self.viewer = ExplorerViewer(self)
        self.viewer.show_image(int(i), focus)

    def open_in_main(self, i: int):
        self.app.open_sample(self.data.keys[i])

    # ================================================================== keys filter (analysis drill-down)
    def show_keys(self, keys, title: str):
        if self.data is None:
            return
        self.view.keys = list(dict.fromkeys(keys))
        self.view.keys_title = title
        self.set_mode("grid")
        self.refresh()

    def clear_keys(self):
        self.view.keys, self.view.keys_title = None, ""
        self.refresh()

    # ================================================================== selection helpers
    def selected_images(self) -> list[int]:
        items = self.grid.selected_items()
        if self.mode() == "objects":
            d = self.data
            out = []
            for c in items:
                k, r = E.decode_id(c)
                out.append(int(d.g_img[r] if k == "gt" else d.p_img[r]))
            return list(dict.fromkeys(out))
        return items

    def target_images(self) -> list[int]:
        """Selected images, else every image of the view."""
        sel = self.selected_images()
        return sel if sel else [int(i) for i in (self.state.order if self.state is not None else [])]

    def selected_objects(self) -> list:
        if self.mode() != "objects":
            return []
        out = []
        for c in self.grid.selected_items():
            k, r = E.decode_id(c)
            o, _i = self._obj_of(k, r)
            if o is not None:
                out.append(o)
        return out

    # ================================================================== edits
    def label_choices(self) -> list[str]:
        labels = list(self.session.true_classes or []) if hasattr(self.session, "true_classes") else []
        seen = set(labels)
        for l in list(self.app.label_colors) + (self.data.labels if self.data else []):
            if l not in seen:
                labels.append(l)
                seen.add(l)
        return labels

    def gt_ops(self, objs, kind, new=None) -> dict:
        d = self.data
        ops = {}
        for o in objs:
            if o.kind != "gt":
                continue
            sug = ("relabel", o.row, new) if kind == "relabel" else ("delete", o.row)
            got = E.fix_op(d, sug)
            if got is not None:
                ops.setdefault(got[0], []).append(got[1])
        return ops

    def apply_ops(self, ops: dict, what: str) -> dict | None:
        if not ops:
            return None
        n = sum(len(v) for v in ops.values())
        st = self.session.apply_fixes(self, dict(ops), what=f"explorer: {what}")
        if st["files"]:
            self.undo_stack.append(st["backup"])
            self.undo_btn.setEnabled(True)
        msg = f"{what}: {st['done']:,} of {n:,} change(s) in {st['files']:,} file(s)"
        if st["missing"]:
            msg += f" · {st['missing']:,} skipped (the label changed since)"
        if st["skipped"]:
            msg += f" · {st['skipped']:,} already labelled"
        if st["refused"]:
            msg += f" · {st['refused']:,} not possible (class not in the dataset / not a box)"
        self.app.statusBar().showMessage(msg + " — ↶ undoes it", 8000)
        if st["errors"]:
            self.app.log_errors(st["errors"])
        self._apply_dirty()                         # re-match now (the signal may arrive later)
        return st

    def undo(self):
        if not self.undo_stack:
            return
        b = self.undo_stack.pop()
        keys, skipped = self.session.undo_backup(b)
        self.undo_btn.setEnabled(bool(self.undo_stack))
        self._dirty_keys.update(keys)
        self._apply_dirty()
        self.app.statusBar().showMessage(f"Undone: {len(keys):,} file(s) restored" +
                                         (f" · {len(skipped):,} changed since, left alone" if skipped else ""), 6000)

    def on_annotations_changed(self, keys):
        if self.data is None:
            return
        self._dirty_keys.update(k for k in keys if self.data.index_of(k) is not None)
        if self._dirty_keys:
            self._edit_timer.start()

    def _apply_dirty(self):
        keys, self._dirty_keys = self._dirty_keys, set()
        if not keys or self.data is None:
            return
        agnostic = bool((self.data.meta or {}).get("agnostic"))
        updates = {}
        for k in keys:
            s = self.session.by_key.get(k)
            if s is None:
                continue
            try:
                ann = self.session.adapter.load(s, light=True)
            except Exception:
                continue
            shapes = ann.shapes
            if agnostic:
                shapes = [dict(sh, label="object") for sh in shapes]
            updates[k] = (shapes, ann.normalized, None, None)
        if not updates:
            return
        with self.lock:
            E.replace_gt(self.data, updates)
            self.data_version += 1
        self._unsaved = True
        self.refresh()

    def accept_dialog(self):
        if self.state is None:
            return
        d, st = self.data, self.state
        if self.mode() == "objects" and self.grid.selected_items():
            ops = {}
            for o in self.selected_objects():
                got = E.op_for(d, st, o, self.view.iou)
                if got is not None:
                    ops.setdefault(got[0], []).append(got[1])
            if not ops:
                QMessageBox.information(self, "Accept", "The selected objects have nothing to accept (correct, "
                                        "duplicates or not checked).")
                return
            n = sum(len(v) for v in ops.values())
            if n > 1 and not self.dlg.confirm(f"Accept the model's suggestion for {n:,} object(s) in "
                                              f"{len(ops):,} file(s)? Backed up (↶ undoes it)."):
                return
            self.apply_ops(ops, "accept")
            return
        sel = self.selected_images()
        imgs = sel if sel else [int(i) for i in st.order]
        err = st.counts[:, [E.EXTRA, E.CLASS, E.FIT]].sum(1)
        imgs = [i for i in imgs if err[i]]
        dlg = QDialog(self)
        dlg.setWindowTitle("Accept the model's suggestions")
        lay = QVBoxLayout(dlg)
        scope = f"the {len(sel):,} selected image(s)" if sel else f"the {st.n_images:,} image(s) of this view"
        intro = QLabel(f"On {scope}, accept what the visible models suggest. Only drawn predictions count (filters "
                       "and thresholds apply); per label only the most confident one. Deleting labels is never done "
                       "in bulk. Everything is backed up — ↶ undoes it.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QFormLayout()
        conf = QDoubleSpinBox()
        conf.setRange(0.05, 1.0)
        conf.setSingleStep(0.05)
        conf.setValue(float(self.app.settings.value("explorer/bulk_conf", 0.6)))
        form.addRow("Model confidence at least", conf)
        lay.addLayout(form)
        boxes = {}
        for k, text in (("add", "Add unlabelled objects (not labelled)"), ("relabel", "Fix wrong classes"),
                        ("reshape", "Refit loose boxes (poor fit)")):
            c = QCheckBox(text)
            c.setChecked(self.app.settings.value(f"explorer/bulk_{k}", k == "add", type=bool))
            boxes[k] = c
            lay.addWidget(c)
        count = QLabel()
        apply_style(count, "font-weight:600;")
        lay.addWidget(count)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        result = {}

        def update():
            kinds = {k for k, c in boxes.items() if c.isChecked()}
            ops = E.bulk_ops(d, st, imgs, kinds, conf.value(), self.view.iou) if kinds else {}
            result["ops"] = ops
            per = Counter(op[0] for v in ops.values() for op in v)
            for k, c in boxes.items():
                base = c.text().split("  (")[0]
                c.setText(f"{base}  ({per.get(k, 0):,})")
            n = sum(per.values())
            count.setText(f"→ {n:,} change(s) in {len(ops):,} file(s)")
            ok = bb.button(QDialogButtonBox.StandardButton.Ok)
            ok.setEnabled(bool(n))
            ok.setText(f"Accept {n:,}")
        conf.valueChanged.connect(lambda _v: update())
        for c in boxes.values():
            c.toggled.connect(lambda _on: update())
        update()
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.app.settings.setValue("explorer/bulk_conf", conf.value())
        for k, c in boxes.items():
            self.app.settings.setValue(f"explorer/bulk_{k}", c.isChecked())
        self.apply_ops(result.get("ops") or {}, "accept suggestions")

    def relabel_selection(self):
        objs = [o for o in self.selected_objects() if o.kind == "gt"]
        if not objs:
            QMessageBox.information(self, "Relabel", "Select label objects first (Objects view: Labels or Problems).")
            return
        labels = self.label_choices()
        cur = labels.index(objs[0].label) if objs[0].label in labels else 0
        new, ok = QInputDialog.getItem(self, "Relabel", f"New class for {len(objs):,} label(s):", labels, cur, True)
        if ok and new:
            self.apply_ops(self.gt_ops(objs, "relabel", new), "relabel")

    def delete_selection(self):
        objs = [o for o in self.selected_objects() if o.kind == "gt"]
        if not objs:
            return
        if self.dlg.confirm(f"Delete {len(objs):,} label(s)? Backed up (↶ undoes it)."):
            self.apply_ops(self.gt_ops(objs, "delete"), "delete labels")

    # ================================================================== review / lists
    def set_review(self, keys, status):
        from AIO_QC.core.state import REVIEW_FLAG, REVIEW_OK
        st = {"flag": REVIEW_FLAG, "ok": REVIEW_OK}.get(status)
        if st == REVIEW_FLAG:
            self.session.set_review(keys, REVIEW_FLAG, "Prediction Explorer")
        else:
            self.session.set_review(keys, st)

    def review_selection(self, status):
        imgs = self.target_images()
        if not imgs:
            return
        if not self.selected_images() and len(imgs) > 1 and not self.dlg.confirm(
                f"Nothing selected — apply to all {len(imgs):,} image(s) of this view?"):
            return
        self.set_review([self.data.keys[i] for i in imgs], status)

    def _on_review_changed(self):
        self.grid.viewport().update()
        if self.view.review:
            self.schedule()

    def _keys(self):
        return [self.data.keys[i] for i in self.target_images()]

    def show_in_file_list(self):
        if self.data is not None:
            self.app.apply_file_filter(self._keys(), "🔭 Prediction Explorer")

    def open_review_grid(self):
        if self.data is not None:
            self.app.open_review_grid(self._keys(), "🔭 Prediction Explorer")

    def export_csv(self):
        if self.state is None:
            return
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Export the list", "explorer.csv", "CSV (*.csv)")
        if not path:
            return
        import csv
        d, st = self.data, self.state
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["path", "split"] + [TAG_NAMES[t] for t in range(7)])
            for i in self.target_images():
                w.writerow([d.keys[i], d.splits[d.img_split[i]] if len(d.splits) else ""] +
                           [int(st.counts[i, t]) for t in range(7)])
        self.app.statusBar().showMessage(f"Exported {path}", 4000)

    # ================================================================== grid menu
    def _grid_menu(self, pos):
        idx = self.grid.indexAt(pos)
        if idx.isValid() and not self.grid.selectionModel().isSelected(idx):
            self.grid.setCurrentIndex(idx)
        if not self.grid.selected_items():
            return
        m = QMenu(self)
        item = idx.data(Qt.ItemDataRole.UserRole) if idx.isValid() else None
        if item is not None:
            m.addAction("Open", lambda: self.open_item(int(item)))
            img = int(item) if self.mode() == "grid" else self.selected_images()[0]
            m.addAction("Open in the main viewer", lambda: self.open_in_main(img))
            m.addSeparator()
        m.addAction("✓ Accept the suggestions…", self.accept_dialog)
        if self.mode() == "objects":
            m.addAction("✎ Relabel…", self.relabel_selection)
            m.addAction("🗑 Delete labels", self.delete_selection)
        m.addSeparator()
        m.addAction("⚑ Flag", lambda: self.review_selection("flag"))
        m.addAction("✔ Mark OK", lambda: self.review_selection("ok"))
        m.addAction("Clear review mark", lambda: self.review_selection(None))
        m.exec(self.grid.viewport().mapToGlobal(pos))

    # ================================================================== saved views
    def _fill_saved_views(self, select=None):
        views = E.load_views(self.session.root)
        self.views.blockSignals(True)
        self.views.clear()
        self.views.addItem("Views…", None)
        for name in sorted(views, key=str.lower):
            self.views.addItem(name, name)
        if select is not None:
            self.views.setCurrentIndex(max(0, self.views.findData(select)))
        self.views.blockSignals(False)

    def _save_view(self):
        if self.data is None:
            return
        name, ok = QInputDialog.getText(self, "Save view", "Name:", text=self.views.currentData() or "")
        if not ok or not name.strip():
            return
        views = E.load_views(self.session.root)
        views[name.strip()] = self._read_view().to_dict()
        E.save_views(self.session.root, views)
        self._fill_saved_views(name.strip())

    def _delete_view(self):
        name = self.views.currentData()
        if not name:
            return
        views = E.load_views(self.session.root)
        views.pop(name, None)
        E.save_views(self.session.root, views)
        self._fill_saved_views()

    def _apply_saved_view(self, _i=None):
        name = self.views.currentData()
        raw = E.load_views(self.session.root).get(name) if name else None
        if raw is None or self.data is None:
            return
        self.apply_view(View.from_dict(raw))

    def apply_view(self, v: View):
        d = self.data
        v.models = {m: {"visible": v.visible(m), "conf": v.conf_of(m)} for m in d.models}
        self.view = v
        self.sidebar.load(v)
        for w, val in ((self.sort, v.sort),):
            w.blockSignals(True)
            w.setCurrentIndex(max(0, w.findData(val)))
            w.blockSignals(False)
        self.desc.blockSignals(True)
        self.desc.setChecked(bool(v.desc))
        self.desc.setText("↓" if v.desc else "↑")
        self.desc.blockSignals(False)
        self.search.blockSignals(True)
        self.search.setText(v.search or "")
        self.search.blockSignals(False)
        self.refresh()

    def reset_view(self):
        if self.data is None:
            return
        v = View()
        conf = (self.data.meta or {}).get("conf", 0.25)
        v.models = {m: {"visible": True, "conf": conf} for m in self.data.models}
        self.apply_view(v)

    # ================================================================== thresholds panel
    def _toggle_insight(self, on):
        self.insight.setVisible(on)
        if on:
            self._refresh_insight()

    def _fill_insight_combos(self):
        d = self.data
        self.ins_model.blockSignals(True)
        self.ins_model.clear()
        for m in d.models:
            self.ins_model.addItem(get_color_icon(QColor(d.model_colors.get(m, "#ff2bd6"))), d.model_names.get(m, m), m)
        self.ins_model.blockSignals(False)
        self.ins_class.blockSignals(True)
        self.ins_class.clear()
        self.ins_class.addItem("All classes", None)
        for l in sorted(d.labels, key=str.lower):
            self.ins_class.addItem(l, l)
        self.ins_class.blockSignals(False)

    def _refresh_insight(self):
        d = self.data
        if d is None or not d.models:
            return
        mid = self.ins_model.currentData()
        if mid is None:
            return
        mi = d.models.index(mid)
        cl = self.ins_class.currentData()
        ci = d.label_id(cl) if cl is not None else None
        with self.lock:
            c = E.curves(d, mi, ci, self.view.iou)
        thr = self.view.conf_of(mid)
        self._curve = c
        self.hist.set_data(c, thr)
        self.pr.set_data(c, thr)
        p, r, n = E.at_threshold(c, thr)
        fmt = lambda x: "–" if x != x else f"{x:.3f}"
        self.ins_text.setText(
            f"<b>{d.model_names.get(mid, mid)}</b>{' · ' + cl if cl else ''} at IoU {self.view.iou:.2f}<br>"
            f"threshold {thr:.2f}: precision <b>{fmt(p)}</b> · recall <b>{fmt(r)}</b> · {n:,} predictions<br>"
            f"AP {fmt(c['ap'])} · {c['npos']:,} labels it could find<br>"
            f"best F1 {fmt(c['best_f1'])} at conf <b>{fmt(c['best_conf'])}</b>")
        self.best_btn.setEnabled(c["best_conf"] == c["best_conf"])

    def _hist_threshold(self, v: float):
        mid = self.ins_model.currentData()
        row = self.sidebar.model_row(mid)
        if row is not None:
            row.set_conf(v, emit=True)

    def _use_best(self):
        c = getattr(self, "_curve", None)
        if c is not None and c["best_conf"] == c["best_conf"]:
            self._hist_threshold(round(c["best_conf"], 2))

    # ================================================================== embedding map
    def _fill_map_sources(self):
        import json
        from pathlib import Path
        folder = Path(self.session.root) / ".aio_qc" / "embeddings"
        names = {}
        try:
            for m in json.loads(self.app.settings.value("embed/models", "[]") or "[]"):
                try:
                    from AIO_QC.core.embeddings import model_id
                    names[model_id(m["path"])] = m.get("name") or Path(m["path"]).name
                except (OSError, KeyError, TypeError):
                    pass
        except (ValueError, TypeError):
            pass
        cur = self.map_src.currentData()
        self.map_src.blockSignals(True)
        self.map_src.clear()
        for f in sorted(folder.glob("*_shape.npz")) if folder.is_dir() else []:
            mid = f.name[: -len("_shape.npz")]
            self.map_src.addItem(names.get(mid, mid), str(f))
        if not self.map_src.count():
            self.map_src.addItem("none yet — run Visual Similarity", None)
        self.map_src.setCurrentIndex(max(0, self.map_src.findData(cur)))
        self.map_src.blockSignals(False)

    def compute_map(self):
        path = self.map_src.currentData()
        if not path or self.data is None:
            QMessageBox.information(self, "Map", "No object embeddings yet: run Visual Similarity (Models ▸ Visual "
                                    "Similarity) once with a vision model — its vectors are reused here.")
            return
        d, root = self.data, self.session.root
        method, n = self.map_method.currentData(), self.map_n.value()
        self.map_info.setText("computing the map…")

        def run():
            with np.load(path, allow_pickle=False) as z:
                ids, vecs = z["ids"], z["vecs"]
            with self.lock:
                pos, rows = E.embedded_rows(d, root, ids, n)
            if not len(pos):
                return None
            xy, used = E.project(vecs[pos].astype(np.float32), method)
            return rows, xy, used

        def done(res):
            if res is None:
                self.map_info.setText("None of the embedded objects matches a current label (re-run Visual "
                                      "Similarity after big edits).")
                return
            rows, xy, used = res
            self._map = (rows, xy)
            self._recolor_map(full=True)
            self.map_info.setText(f"{len(rows):,} objects · {used.upper()} · drag a rectangle to select")

        w = FuncWorker(run, self)
        w.done.connect(done)
        w.failed.connect(lambda m: self.map_info.setText(f"❌ {m}"))
        self._map_worker = w
        w.start()

    def _map_colors(self):
        rows, _xy = self._map
        d, st = self.data, self.state
        if self.map_color.currentData() == "class":
            n = max(1, len(d.labels))
            pal = {c: QColor.fromHsvF((c * 0.61803) % 1.0, 0.65, 0.95).name() for c in range(n)}
            return [pal[int(d.g_cls[r])] for r in rows]
        tags = st.g_tag[rows] if st is not None else np.full(len(rows), E.NA)
        return [tag_color(int(t)).name() for t in tags]

    def _recolor_map(self, full=False):
        if self._map is None:
            return
        rows, xy = self._map
        d = self.data
        if full:
            tip = lambda n: (f"{d.labels[d.g_cls[rows[n]]]} · {TAG_NAMES.get(int(self.state.g_tag[rows[n]]), '')}\n"
                             f"{self.display_name(d.keys[d.g_img[rows[n]]])}")
            self.map.set_points(xy, self._map_colors(), tip)
        else:
            self.map.recolor(self._map_colors())

    def _map_selected(self, idx):
        n = len(idx)
        self._map_sel = idx
        self.map_show_img.setEnabled(bool(n))
        self.map_show_obj.setEnabled(bool(n))
        if n and self._map is not None:
            rows = self._map[0][idx]
            tags = Counter(TAG_NAMES.get(int(t), "") for t in self.state.g_tag[rows])
            self.map_info.setText(f"{n:,} selected: " + ", ".join(f"{c:,} {t}" for t, c in tags.most_common()))

    def _map_open(self, n):
        rows = self._map[0]
        r = int(rows[n])
        self.open_image(int(self.data.g_img[r]), ("gt", r))

    def _map_to(self, mode):
        idx = getattr(self, "_map_sel", None)
        if idx is None or not len(idx) or self._map is None:
            return
        rows = self._map[0][idx]
        if mode == "grid":
            keys = [self.data.keys[int(self.data.g_img[r])] for r in rows]
            self.show_keys(keys, f"{len(rows):,} objects selected on the map")
        else:
            self.mode_btns["objects"].setChecked(True)
            self._mode_widgets()
            self.pages.setCurrentIndex(0)
            self.patch_override = np.asarray(rows, np.int64)
            self.grid.set_items("objects", self.patch_override)
            self._update_status()

    # ================================================================== persistence
    def save_now(self, background=False):
        """Write the (edited) explorer data back to .aio_qc/explorer.npz (atomic). background: a plain thread that
        finishes even while the window closes (large runs take seconds to compress)."""
        if self.data is None or not self._unsaved:
            return
        d, path, lock = self.data, E.data_path(self.session.root), self.lock
        self._unsaved = False

        def run():
            try:
                with lock:
                    E.save(d, path)
            except OSError:
                pass
        if background:
            t = threading.Thread(target=run, name="explorer-save", daemon=False)
            t.start()
            self._save_thread = t
        else:
            run()

    def shutdown(self):
        self._timer.stop()
        self._edit_timer.stop()
        for sig, slot in ((self.session.annotationsChanged, self.on_annotations_changed),):
            try:
                sig.disconnect(slot)
            except TypeError:
                pass
        if self._worker is not None:
            self._worker.wait(5000)
        w = getattr(self, "_map_worker", None)
        if w is not None:
            w.wait(5000)
        self.save_now(background=True)
        if self.viewer is not None:
            self.viewer.close()


def layer_objects(app, key) -> tuple:
    """Objects for the main viewer's predictions layer: from an open Model Check (its current view), else from the
    last saved run with default thresholds. [] when there is nothing."""
    dlg = getattr(app, "open_dialogs", {}).get("model_check")
    panel = getattr(dlg, "explorer", None)
    if panel is not None and panel.data is not None and panel.state is not None:
        i = panel.data.index_of(key)
        if i is None:
            return [], panel.data
        return [o for o in E.objects(panel.data, panel.state, i, everything=False) if o.kind == "pred"], panel.data
    cache = getattr(app, "_explorer_layer", None)
    root = getattr(getattr(app, "session", None), "root", None)
    if root is None:
        return [], None
    path = E.data_path(root)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return [], None
    if cache is None or cache[0] != (path, mtime):
        d = E.load(path)
        if d is None:
            return [], None
        v = View(models={m: {"visible": True, "conf": (d.meta or {}).get("conf", 0.25)} for m in d.models})
        cache = ((path, mtime), d, E.evaluate(d, v))
        app._explorer_layer = cache
    _k, d, st = cache
    i = d.index_of(key)
    if i is None:
        return [], d
    return [o for o in E.objects(d, st, i) if o.kind == "pred"], d
