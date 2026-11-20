# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Testing (Model Lab) — step 4, the results.

A mixin of AIO_QC.widgets.testing.model_lab.ModelLab: the methods use the window's own attributes, set up in its __init__.
"""
import os
import numpy as np
from PyQt6.QtCore import Qt, QSize, QTimer, QThread
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QHeaderView, QAbstractItemView, QComboBox, QCheckBox, QFileDialog, QMessageBox, QTabWidget, QSlider, QSplitter, QMenu,
    QStackedWidget, QToolButton,
)
from AIO_QC.widgets import motion as anim
from AIO_QC.widgets.testing.lab_common import (
    Job,
    ChartGrid,
    theme_mode,
    human_time,
    open_path,
    IssueBar,
    RowsModel,
    StatTile,
    Gallery,
    fit_pixmap,
    last_dir,
    remember_dir,
)
from AIO_QC.core.diag import ignored as _ignored

from AIO_QC.widgets.testing.model_lab_parts.cls_results import ClsResultsMixin
from AIO_QC.widgets.testing.model_lab_common import (T_FAILURES, T_VIDEOS, _hint, _item, _make_table, _num_item, _release_thread_reader, _swatch_icon, slot_hex)


class ResultsMixin(ClsResultsMixin):

    # ------------------------------------------------------------------ 4 results
    def _page_results(self):
        outer = QWidget()
        ol_ = QVBoxLayout(outer)
        ol_.setContentsMargins(0, 0, 0, 0)
        self.res_stack = QStackedWidget()
        ol_.addWidget(self.res_stack)
        # empty state
        empty = QWidget()
        el = QVBoxLayout(empty)
        el.addStretch(1)
        t = QLabel("<div style='font-size:15pt; font-weight:600'>No results yet</div>"
                   "<div>Choose test data and models, then run them - results, charts and failures appear here.</div>")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setWordWrap(True)
        el.addWidget(t)
        row = QHBoxLayout()
        row.addStretch(1)
        go = QPushButton("Go to Run  →")
        go.clicked.connect(lambda: self.nav.setCurrentRow(2))
        row.addWidget(go)
        row.addStretch(1)
        el.addLayout(row)
        el.addStretch(2)
        self.res_stack.addWidget(empty)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 10, 12, 8)
        self.issue_bars[3] = IssueBar()
        lay.addWidget(self.issue_bars[3])
        top = QHBoxLayout()
        self.res_title = QLabel("<span style='font-size:15pt; font-weight:600'>Results</span>")
        self.res_title.setWordWrap(True)
        top.addWidget(self.res_title, 1)
        self.res_busy = QLabel("")
        self.res_busy.setEnabled(False)
        top.addWidget(self.res_busy)
        self.res_best = QCheckBox("Best F1 per model")
        self.res_best.setChecked(True)
        self.res_best.setToolTip("Operating threshold: each model at its best-F1 confidence, or one fixed value for "
                                 "all (move the slider). Errors, confusion matrices and galleries follow it.")
        self.res_slider = QSlider(Qt.Orientation.Horizontal)
        self.res_slider.setRange(1, 99)
        self.res_slider.setValue(25)
        self.res_slider.setFixedWidth(160)
        self.res_slider.setEnabled(False)
        self.res_thr = QLabel("")
        self.res_thr.setMinimumWidth(34)
        self.res_best.toggled.connect(self._threshold_changed)
        self.res_slider.valueChanged.connect(lambda v: (self.res_thr.setText(f"{v / 100:.2f}"),
                                                        self._thr_timer.start()))
        self._thr_timer = QTimer(self)
        self._thr_timer.setSingleShot(True)
        self._thr_timer.setInterval(220)
        self._thr_timer.timeout.connect(self._threshold_changed)
        top.addWidget(QLabel("Threshold:"))
        top.addWidget(self.res_best)
        top.addWidget(self.res_slider)
        top.addWidget(self.res_thr)
        lay.addLayout(top)
        self.tabs = QTabWidget()
        anim.animate_tabs(self.tabs)
        self.tabs.currentChanged.connect(self._render_tab)
        lay.addWidget(self.tabs, 1)

        # overview
        ov = QWidget()
        ol = QVBoxLayout(ov)
        self.verdict = QLabel("")
        self.verdict.setWordWrap(True)
        self.verdict.setTextFormat(Qt.TextFormat.RichText)
        self.verdict.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        ol.addWidget(self.verdict)
        self.tiles = QHBoxLayout()
        self.tiles.setSpacing(8)
        ol.addLayout(self.tiles)
        self.board = _make_table(["#", "", "Model", "mAP50-95", "mAP50", "Precision", "Recall", "F1", "Conf",
                                  "ms / img", "img / s", "False det.", "Missed"], stretch_col=2)
        self.board.setSortingEnabled(True)
        self.board.setToolTip("P / R / F1 at the best-F1 confidence (YOLO val method). False detections and missed "
                              "objects at the operating threshold. Click a header to sort; double-click a model for "
                              "its failures.")
        self.board.cellDoubleClicked.connect(self._board_dbl)
        ol.addWidget(self.board, 1)
        self.tabs.addTab(ov, "Overview")

        chw = QWidget()
        chl = QVBoxLayout(chw)
        row = QHBoxLayout()
        row.addWidget(_hint("Hover for values · click a bar to see those images · Save on each chart, or all at once."), 1)
        row.addWidget(self._save_all_button(lambda: self.charts))
        chl.addLayout(row)
        self.charts = ChartGrid(empty_text="Charts appear after a run with ground truth (labels or a reference model).")
        self.charts.link_clicked.connect(self._on_link)
        chl.addWidget(self.charts, 1)
        self.tabs.addTab(chw, "Charts")

        # confusion
        cw = QWidget()
        cl = QVBoxLayout(cw)
        row = QHBoxLayout()
        self.cm_model = QComboBox()
        self.cm_model.currentIndexChanged.connect(lambda _: self._show_confusion())
        row.addWidget(QLabel("Model"))
        row.addWidget(self.cm_model)
        row.addWidget(_hint("Click a cell to see those images."))
        row.addStretch(1)
        row.addWidget(self._save_all_button(lambda: self.cm_grid))
        cl.addLayout(row)
        self.cm_grid = ChartGrid(empty_text="")
        self.cm_grid.link_clicked.connect(self._on_link)
        cl.addWidget(self.cm_grid, 1)
        self.tabs.addTab(cw, "Confusion matrix")

        # per class
        pw = QWidget()
        pl = QVBoxLayout(pw)
        row = QHBoxLayout()
        self.pc_model = QComboBox()
        self.pc_model.currentIndexChanged.connect(lambda _: self._show_classes())
        row.addWidget(QLabel("Model"))
        row.addWidget(self.pc_model)
        row.addWidget(_hint("Double-click a class to see its failures."))
        row.addStretch(1)
        pl.addLayout(row)
        self.pc_table = _make_table(["Class", "Objects", "AP50", "AP50-95", "Precision", "Recall", "F1", "Correct",
                                     "False det.", "Missed"], stretch_col=0)
        self.pc_table.setSortingEnabled(True)
        self.pc_table.cellDoubleClicked.connect(self._class_dbl)
        pl.addWidget(self.pc_table, 1)
        self.pc_note = _hint("")
        pl.addWidget(self.pc_note)
        self.tabs.addTab(pw, "Per class")

        # failures
        fw = QWidget()
        fl = QVBoxLayout(fw)
        row = QHBoxLayout()
        self.f_model = QComboBox()
        self.f_model.currentIndexChanged.connect(lambda _: self._show_failures())
        self.f_kind = QComboBox()
        for label, k in (("All errors", "all"), ("False detections", "fp"), ("Missed objects", "fn"),
                         ("Wrong class", "cls"), ("Poor box", "loc"), ("Duplicates", "dup"),
                         ("Detections on background", "bkg"), ("Missed (found below threshold)", "missed_low_conf")):
            self.f_kind.addItem(label, k)
        self.f_kind.currentIndexChanged.connect(lambda _: self._show_failures())
        self.f_class = QComboBox()
        self.f_class.currentIndexChanged.connect(lambda _: self._show_failures())
        self.f_zoom = QCheckBox("Zoom to errors")
        self.f_zoom.setChecked(True)
        self.f_zoom.toggled.connect(lambda _: self._show_failures())
        for lab, wdg in (("Model", self.f_model), ("Show", self.f_kind), ("Class", self.f_class)):
            row.addWidget(QLabel(lab))
            row.addWidget(wdg)
        row.addWidget(self.f_zoom)
        row.addStretch(1)
        self.f_review = QPushButton("Send to review")
        self.f_review.setToolTip("Flag these images in Image QC's review queue and show them in its file list "
                                 "(available when the test data is the dataset open in Image QC)")
        self.f_review.clicked.connect(self._send_failures_to_review)
        row.addWidget(self.f_review)
        fl.addLayout(row)
        chip = QHBoxLayout()
        self.f_chip = QLabel("")
        self.f_chip.setTextFormat(Qt.TextFormat.RichText)
        self.f_chip_clear = QToolButton()
        self.f_chip_clear.setText("✕ Clear filter")
        self.f_chip_clear.setAutoRaise(True)
        self.f_chip_clear.clicked.connect(self._clear_cm_filter)
        chip.addWidget(self.f_chip)
        chip.addWidget(self.f_chip_clear)
        chip.addStretch(1)
        self.f_chip.setVisible(False)
        self.f_chip_clear.setVisible(False)
        fl.addLayout(chip)
        self.f_hint = _hint("Dashed = ground truth · model colour = correct · red = error · red dashed = missed. "
                            "Double-click to fix labels or view full size; right-click to compare models or save the "
                            "image.")
        fl.addWidget(self.f_hint)
        self.f_gallery = Gallery(QSize(300, 190), 2, "No images with this kind of error.")
        self.f_list = self.f_gallery.list
        self.f_list.itemDoubleClicked.connect(self._failure_open)
        self.f_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.f_list.customContextMenuRequested.connect(self._failure_menu)
        fl.addWidget(self.f_gallery, 1)
        self.f_more = QPushButton("Show more")
        self.f_more.clicked.connect(lambda: self._show_failures(more=True))
        fl.addWidget(self.f_more, 0, Qt.AlignmentFlag.AlignCenter)
        self.tabs.addTab(fw, "Failures")
        self.tabs.addTab(self._plots_tab(), "Plots")

        # compare
        mw = QWidget()
        ml = QVBoxLayout(mw)
        row = QHBoxLayout()
        self.cmp_a = QComboBox()
        self.cmp_b = QComboBox()
        self.cmp_a.currentIndexChanged.connect(lambda _: self._show_compare())
        self.cmp_b.currentIndexChanged.connect(lambda _: self._show_compare())
        row.addWidget(QLabel("Compare"))
        row.addWidget(self.cmp_a)
        row.addWidget(QLabel("with"))
        row.addWidget(self.cmp_b)
        sbs = QPushButton("Side by side…")
        sbs.setToolTip("Both models on the same images, zoom and pan linked")
        sbs.clicked.connect(lambda: self._open_ab(None))
        row.addWidget(sbs)
        self.cmp_review = QPushButton("Send to review")
        self.cmp_review.setToolTip("Flag the images where the two models differ in Image QC's review queue")
        self.cmp_review.clicked.connect(self._send_compare_to_review)
        row.addWidget(self.cmp_review)
        row.addStretch(1)
        row.addWidget(self._save_all_button(lambda: self.cmp_grid))
        ml.addLayout(row)
        self.cmp_tiles = QLabel("")
        self.cmp_tiles.setTextFormat(Qt.TextFormat.RichText)
        self.cmp_tiles.setWordWrap(True)
        ml.addWidget(self.cmp_tiles)
        split = QSplitter(Qt.Orientation.Vertical)
        self.cmp_grid = ChartGrid(empty_text="Pick two scored models.")
        self.cmp_grid.link_clicked.connect(self._on_link)
        split.addWidget(self.cmp_grid)
        lists = QWidget()
        ll = QHBoxLayout(lists)
        ll.setContentsMargins(0, 0, 0, 0)
        self.cmp_lists = []
        self.cmp_galleries = []
        for k in range(2):
            col = QVBoxLayout()
            lab = QLabel("")
            lab.setWordWrap(True)
            lab.setTextFormat(Qt.TextFormat.RichText)
            gal = Gallery(QSize(300, 112), 1, "No images.")
            lst = gal.list
            lst.itemDoubleClicked.connect(lambda it, lst=lst: self._compare_open(lst, it))
            col.addWidget(lab)
            col.addWidget(gal, 1)
            ll.addLayout(col, 1)
            self.cmp_lists.append((lab, lst))
            self.cmp_galleries.append(gal)
        split.addWidget(lists)
        split.setSizes([500, 300])
        ml.addWidget(split, 1)
        self.tabs.addTab(mw, "Compare A / B")

        # videos
        vw = QWidget()
        vl = QVBoxLayout(vw)
        row = QHBoxLayout()
        self.v_pick = QComboBox()
        self.v_pick.currentIndexChanged.connect(lambda _: self._show_video())
        row.addWidget(QLabel("Video"))
        row.addWidget(self.v_pick, 1)
        row.addWidget(self._save_all_button(lambda: self.v_grid))
        vl.addLayout(row)
        vsplit = QSplitter(Qt.Orientation.Vertical)
        self.v_grid = ChartGrid(empty_text="Add videos on page 1 and run to see behaviour over time.")
        self.v_grid.link_clicked.connect(self._on_link)
        vsplit.addWidget(self.v_grid)
        from PyQt6.QtWidgets import QTableView
        self.v_events_model = RowsModel(["Time", "Frame", "What", "Model", "Class"])
        self.v_events = QTableView()
        self.v_events.setModel(self.v_events_model)
        self.v_events.setSortingEnabled(True)
        self.v_events.verticalHeader().setVisible(False)
        self.v_events.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.v_events.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.v_events.horizontalHeader().setStretchLastSection(True)
        self.v_events.setToolTip("Every flicker, blip and disagreement. Double-click to see that moment plotted.")
        self.v_events.doubleClicked.connect(self._event_open)
        vsplit.addWidget(self.v_events)
        vsplit.setSizes([560, 220])
        vl.addWidget(vsplit, 1)
        self.tabs.addTab(vw, "Videos")
        self.res_stack.addWidget(w)
        return outer

    def _plots_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.pl_model = QComboBox()
        self.pl_show = QComboBox()
        for label, k in (("Predictions", "pred"), ("Predictions + ground truth", "gt"), ("Errors highlighted", "err")):
            self.pl_show.addItem(label, k)
        self.pl_show.setCurrentIndex(1)
        self.pl_class = QComboBox()
        self.pl_sort = QComboBox()
        for label, k in (("Test order", "order"), ("Most detections", "most"), ("Fewest detections", "fewest")):
            self.pl_sort.addItem(label, k)
        self.pl_size = QComboBox()
        for label, _ in self.PLOT_SIZES:
            self.pl_size.addItem(label)
        self.pl_size.setCurrentIndex(1)
        for lab, wdg in (("Model", self.pl_model), ("Show", self.pl_show), ("Class", self.pl_class),
                         ("Sort", self.pl_sort), ("Size", self.pl_size)):
            row.addWidget(QLabel(lab))
            row.addWidget(wdg)
            wdg.currentIndexChanged.connect(lambda _=0: self._show_plots())
        row.addStretch(1)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.pl_hint = _hint("Each test image with the model's detections drawn (labels in Inter), at the operating "
                             "threshold above. Double-click to view full size; right-click or use the buttons to save.")
        row.addWidget(self.pl_hint, 1)
        self.pl_save_sel = QPushButton("Save selected…")
        self.pl_save_sel.clicked.connect(lambda: self._save_plots(selected=True))
        self.pl_save_all = QPushButton("Save all…")
        self.pl_save_all.setToolTip("Save every plotted image of this model (with the filters above) into a folder")
        self.pl_save_all.clicked.connect(lambda: self._save_plots(selected=False))
        row.addWidget(self.pl_save_sel)
        row.addWidget(self.pl_save_all)
        lay.addLayout(row)
        self.pl_gallery = Gallery(QSize(300, 190), 2, "Run the models to see their plotted predictions.")
        self.pl_list = self.pl_gallery.list
        self.pl_list.itemDoubleClicked.connect(self._plot_open)
        self.pl_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.pl_list.customContextMenuRequested.connect(self._plot_menu)
        lay.addWidget(self.pl_gallery, 1)
        self.pl_more = QPushButton("Show more")
        self.pl_more.clicked.connect(lambda: self._fill_plots(None))
        self.pl_more.setVisible(False)
        lay.addWidget(self.pl_more, 0, Qt.AlignmentFlag.AlignCenter)
        self._pl_rows, self._pl_shown, self._pl_epoch = [], 0, 0
        return w

    def _plot_rows(self, r, mi, cls, sort, conf):
        """[(ii, detections at the operating threshold conf)] - on the view thread (no Qt here)."""
        if self._is_cls(r):
            return self._cls_plot_rows(r, mi, cls, sort, conf)
        if r is None or mi is None or r.preds[mi] is None:
            return []
        order = list(r.eval_idx) if r.scored else list(range(len(r.items)))
        rows = []
        for ii in order:
            p = r.preds[mi][ii]
            if p is None:
                continue
            p = p[p[:, 4] >= conf] if len(p) else p
            if cls is not None and cls >= 0:
                it = r.items[ii]
                in_gt = it.cls is not None and cls in set(np.asarray(it.cls, int).tolist())
                if not in_gt and not (len(p) and (p[:, 5].astype(int) == cls).any()):
                    continue
            rows.append((ii, int(len(p))))
        if sort == "most":
            rows.sort(key=lambda x: -x[1])
        elif sort == "fewest":
            rows.sort(key=lambda x: x[1])
        return rows

    def _plot_builder(self, r=None):
        rb = self._ink_builder(r if r is not None else self.run)
        return rb

    def _plotter(self):
        """plotter(item, model, show_gt, conf) -> full-resolution plotted BGR image (what is saved)."""
        rb = self._plot_builder()
        return lambda ii, mi, show_gt, conf=None: rb.plot_image(ii, mi, show_gt, True, conf)

    def _show_plots(self):
        r = self.run
        icon = self.PLOT_SIZES[max(0, self.pl_size.currentIndex())][1]
        if self.pl_list.iconSize() != icon:
            self.pl_gallery.set_icon_size(icon, 2)
        self._pl_epoch += 1
        self._stop(getattr(self, "_pl_job", None))
        self.pl_gallery.clear()
        self.pl_more.setVisible(False)
        has_gt = r is not None and (bool(r.scored) or r.ground == "labels")
        self.pl_show.setEnabled(has_gt)
        if r is None or not r.items:
            self.pl_gallery.show_empty(True)
            return
        mi, cls, sort = self.pl_model.currentData(), self.pl_class.currentData(), self.pl_sort.currentData()
        conf = self._plot_builder(r).model_conf(mi) if mi is not None else 0.25      # UI thread
        self._bg("plots", lambda: self._plot_rows(r, mi, cls, sort, conf), lambda rows: self._fill_plots(rows))

    def _plot_mode(self):
        has_gt = self.run is not None and (bool(self.run.scored) or self.run.ground == "labels")
        k = self.pl_show.currentData() if has_gt else "pred"
        return k != "pred", k == "err"

    def _fill_plots(self, rows):
        if rows is not None:
            self.pl_gallery.clear(("No images of this class or answered with it." if self._is_cls() else
                                   "No images with detections of this class.") if self.pl_class.currentData() not in
                                  (None, -1) else "This model has no predictions on the test images.")
            self._pl_rows, self._pl_shown = rows, 0
        rows = self._pl_rows[self._pl_shown:self._pl_shown + 60]
        self._pl_shown += len(rows)
        self.pl_more.setVisible(self._pl_shown < len(self._pl_rows))
        self.pl_save_all.setText(f"Save all ({len(self._pl_rows):,})…" if self._pl_rows else "Save all…")
        if not rows:
            if not self._pl_shown:
                self.pl_gallery.show_empty(True)
            return
        r, mi = self.run, self.pl_model.currentData()
        show_gt, errors = self._plot_mode()
        tw = self.pl_list.iconSize().width()
        rb = self._plot_builder()
        epoch = self._pl_epoch
        jobs = []
        for ii, n in rows:
            what = n if isinstance(n, str) else f"{n} detection{'s' if n != 1 else ''}"
            it = self.pl_gallery.add(self._loc_text(r.items[ii]) + f"\n{what}",
                                     (mi, ii), r.items[ii].path)
            jobs.append((it, ii))

        def work(job):
            import cv2
            for it, ii in jobs:
                if job.cancelled() or self._pl_epoch != epoch:
                    break
                try:
                    img = rb.plot_image(ii, mi, show_gt, errors, None, tw)     # labels sized for the shown width
                except Exception as _exc:
                    _ignored(_exc)
                    img = None
                if img is not None and img.shape[1] > tw * 2:
                    s = tw * 2 / img.shape[1]
                    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
                job.progress.emit((it, img))

        def prog(p):
            it, img = p
            if self._pl_epoch == epoch and img is not None:
                it.setIcon(QIcon(fit_pixmap(img, self.pl_list.iconSize())))
        j = Job(work, self)
        j.progress.connect(prog)
        j.finished.connect(j.deleteLater)
        self._pl_job = j
        j.start(QThread.Priority.LowPriority)

    def _plot_items(self):
        return [ii for ii, _ in self._pl_rows]

    def _plot_open(self, it):
        d = it.data(Qt.ItemDataRole.UserRole)
        if d:
            items = self._plot_items()
            self._open_plot(items, items.index(d[1]) if d[1] in items else 0, d[0], self._plot_mode()[0])

    def _plot_menu(self, pos):
        it = self.pl_list.itemAt(pos)
        d = it.data(Qt.ItemDataRole.UserRole) if it else None
        if not d:
            return
        mi, ii = d
        items = self._plot_items()
        m = QMenu(self)
        m.addAction("View full size…").triggered.connect(
            lambda: self._open_plot(items, items.index(ii) if ii in items else 0, mi, self._plot_mode()[0]))
        m.addAction("Save image…").triggered.connect(lambda: self._save_plot(ii, mi, self._plot_mode()[0]))
        if len(self.pl_list.selectedItems()) > 1:
            m.addAction(f"Save {len(self.pl_list.selectedItems())} selected…").triggered.connect(
                lambda: self._save_plots(selected=True))
        if self._can_edit(ii):
            m.addAction("Change class…" if self._is_cls() else "Fix labels…").triggered.connect(
                lambda: self._open_editor(items, items.index(ii) if ii in items else 0, mi))
        item = self.run.items[ii]
        m.addSeparator()
        m.addAction("Show file").triggered.connect(lambda: open_path(os.path.dirname(item.path)))
        m.exec(self.pl_list.viewport().mapToGlobal(pos))

    def _show_moment(self, video, frame):
        """A video moment (timeline click / events table): the nearest sampled frame, plotted, in the image viewer -
        the Model Lab never drives the main player."""
        r = self.run
        if r is None or not r.items:
            self.status.showMessage("Plotted frames need a run of the models first.", 6000)
            return
        cand = [(abs(int(it.meta.get("frame", -10 ** 9)) - int(frame)), ii) for ii, it in enumerate(r.items)
                if it.meta.get("video") and os.path.normcase(os.path.abspath(it.meta["video"])) ==
                os.path.normcase(os.path.abspath(video))]
        if not cand:
            self.status.showMessage("That video's frames are not in the result views.", 6000)
            return
        ii = min(cand)[1]
        items = sorted((ii2 for _, ii2 in cand), key=lambda k: r.items[k].meta.get("frame", 0))
        mi = r.ranking()[0] if r.ranking() else next((i for i in range(len(r.specs)) if r.preds[i] is not None), 0)
        self._open_plot(items, items.index(ii), mi, True)

    def _open_plot(self, items, start, mi, show_gt=True):
        if self.run is None or not items:
            return
        if self._is_cls():
            self._cls_open(items, start, [mi])
            return
        from AIO_QC.widgets.testing.annot_view import PlotViewer
        from AIO_QC.core.testing.pipeline import frame_getter
        try:
            PlotViewer(self, self.run, items, mi, start, frame_getter, self._run_slots(self.run), self._plotter(),
                       show_gt).exec()
        finally:
            _release_thread_reader()

    def _save_plot(self, ii, mi, show_gt=True):
        from AIO_QC.widgets.testing.annot_view import save_plot_dialog, plot_name
        save_plot_dialog(self, self._plotter()(ii, mi, show_gt), plot_name(self.run, ii, mi))

    def _save_plots(self, selected=False):
        """Selected / all plotted images of the current model into a folder (background, cancellable)."""
        r, mi = self.run, self.pl_model.currentData()
        if r is None or mi is None:
            return
        if selected:
            items = [it.data(Qt.ItemDataRole.UserRole)[1] for it in self.pl_list.selectedItems()
                     if it.data(Qt.ItemDataRole.UserRole)]
        else:
            items = self._plot_items()
        if not items:
            QMessageBox.information(self, "Save plots", "Select images first (Ctrl / Shift + click)." if selected
                                    else "There are no plotted images to save.")
            return
        d = QFileDialog.getExistingDirectory(self, f"Save {len(items):,} plotted images into", last_dir())
        if not d:
            return
        remember_dir(d)
        fmt = self.st.value("plot_fmt", "jpg", str)
        box = QMessageBox(self)
        box.setWindowTitle("Save plots")
        box.setText(f"Save {len(items):,} images of {r.specs[mi].label} into\n{d}")
        jpg = box.addButton("JPG (smaller)", QMessageBox.ButtonRole.AcceptRole)
        png = box.addButton("PNG (lossless)", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(png if fmt == "png" else jpg)
        box.exec()
        if box.clickedButton() not in (jpg, png):
            return
        fmt = "png" if box.clickedButton() is png else "jpg"
        self.st.setValue("plot_fmt", fmt)
        show_gt, errors = self._plot_mode()
        rb = self._plot_builder()
        from AIO_QC.widgets.testing.annot_view import plot_name
        from AIO_QC.widgets.testing.lab_common import safe_name
        from AIO_QC.tools.frame_extractor.core.imageio import imwrite

        def work(job):
            ok, used = 0, set()
            for k, ii in enumerate(items):
                if job.cancelled():
                    break
                img = rb.plot_image(ii, mi, show_gt, errors)
                if img is not None:
                    name = safe_name(plot_name(r, ii, mi), 140)
                    base, n = name, 2
                    while name in used:
                        name, n = f"{base}_{n}", n + 1
                    used.add(name)
                    imwrite(os.path.join(d, f"{name}.{fmt}"), img, quality=95)
                    ok += 1
                job.progress.emit((k + 1, len(items)))
            return ok

        def prog(p):
            self.status.showMessage(f"Saving plots… {p[0]:,} / {p[1]:,}")
            self.pl_save_all.setEnabled(False)
            self.pl_save_sel.setEnabled(False)

        def done(n):
            self.pl_save_all.setEnabled(True)
            self.pl_save_sel.setEnabled(True)
            self.status.showMessage(f"Saved {n:,} plotted images to {d}", 10000)
            self._log(f"Saved {n:,} plotted images ({r.specs[mi].label}) to {d}")

        def failed(msg):
            self.pl_save_all.setEnabled(True)
            self.pl_save_sel.setEnabled(True)
            QMessageBox.warning(self, "Save plots", f"Could not save the plots:\n{msg}")
        j = Job(work, self)
        j.progress.connect(prog)
        j.done.connect(done)
        j.failed.connect(failed)
        j.finished.connect(j.deleteLater)
        self._pl_save_job = j
        j.start()

    def _save_all_button(self, grid_of):
        b = QToolButton()
        b.setText("Save all charts…")
        b.setToolTip("Save every chart on this tab into a folder (PNG at 2x, or SVG)")
        b.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        m = QMenu(b)
        m.addAction("As PNG…").triggered.connect(lambda: self._save_all_charts(grid_of(), "png"))
        m.addAction("As SVG…").triggered.connect(lambda: self._save_all_charts(grid_of(), "svg"))
        b.setMenu(m)
        return b

    def _save_all_charts(self, grid, kind):
        if not grid.cards():
            QMessageBox.information(self, "Save charts", "There are no charts on this tab yet.")
            return
        d = QFileDialog.getExistingDirectory(self, "Save all charts into", last_dir())
        if not d:
            return
        remember_dir(d)
        try:
            paths = grid.save_all(d, kind)
        except OSError as e:
            QMessageBox.warning(self, "Save charts", f"Could not save the charts:\n{e}")
            return
        self.status.showMessage(f"Saved {len(paths)} charts to {d}", 8000)

    # ---- results: built lazily (only the visible tab) and off the UI thread
    def _ink_builder(self, run=None, videos=None):
        from AIO_QC.core.testing import charts as C
        from AIO_QC.core.testing.report import ReportBuilder
        from AIO_QC.core.testing.pipeline import frame_getter
        if self._is_cls(run) or (run is None and videos and videos[0][1].get("task") == "classify"):
            from AIO_QC.core.testing.cls_report import ClsReportBuilder as ReportBuilder
        rb = ReportBuilder(run, "", slots=self._run_slots(run, videos), frame_getter=frame_getter,
                           video=videos or [], version=self.version)
        rb.ink = C.Ink("qt", theme_mode())
        return rb

    def _run_slots(self, run=None, videos=None):
        """Colour slot per model of a run (by model key, so a reordered table keeps colours)."""
        by_key = {s.key: self.slots[i] for i, s in enumerate(self.specs) if i < len(self.slots)}
        if run is not None:
            return [by_key.get(s.key, i) for i, s in enumerate(run.specs)]
        if videos:
            keys = videos[0][1].get("specs") or []
            return [by_key.get(k, i) for i, k in enumerate(keys)]
        return None

    def _pair_combos(self, ca, cb):
        """Fill two model pickers for a comparison. Reference runs (no labels) offer every model, A = the reference
        (a model is compared with the answer key); otherwise the scored models, A = runner-up, B = best."""
        r = self.run
        ref_mode = r is not None and r.ground == "reference" and r.reference is not None
        for combo in (ca, cb):
            self._model_combo(combo, scored_only=not ref_mode)
        if ref_mode:
            best = r.ranking()[0] if r.ranking() else None
            for combo, want in ((ca, r.reference), (cb, best)):
                if want is not None and combo.findData(want) >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(combo.findData(want))
                    combo.blockSignals(False)
        elif cb.count() > 1 and ca.currentIndex() == cb.currentIndex():
            ca.blockSignals(True)
            ca.setCurrentIndex(1)
            ca.blockSignals(False)

    def _model_combo(self, combo, scored_only=True):
        cur = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        r = self.run
        slots = self._run_slots(r) or []
        if r is not None:
            for i in r.ranking() if scored_only else range(len(r.specs)):
                combo.addItem(_swatch_icon(slot_hex(slots[i] if i < len(slots) else i)), r.specs[i].label, i)
        k = combo.findData(cur)
        combo.setCurrentIndex(k if k >= 0 else 0)
        combo.blockSignals(False)

    def _busy(self, key, on):
        (self._busy_keys.add if on else self._busy_keys.discard)(key)
        self.res_busy.setText("updating…" if self._busy_keys else "")

    def _bg(self, key, fn, done):
        """fn() on the view thread; done(result) back on the UI thread (latest request per key wins)."""
        self._busy(key, True)

        def ok(res):
            self._busy(key, False)
            done(res)

        def bad(msg):
            self._busy(key, False)
            self._log(f"Could not build {key}: {msg}")
        self.views.submit(key, fn, ok, bad)

    def _no_score_reason(self):
        """Why accuracy views are empty (None when the run is scored)."""
        r = self.run
        if r is not None and r.scored:
            return None
        if r is None and not self.videos_out:
            return "Run the models to see results."
        n = len(r.specs) if r is not None else len(self.specs)
        if n < 2:
            return ("Accuracy needs ground truth: labelled images (page 1), or two or more models with one chosen as "
                    "the reference (Run page). Plots and Videos work without it.")
        return ("Accuracy needs ground truth: there are no labelled images, and no reference model is chosen. Pick a "
                "reference model on the Run page (the others are scored against it) and run again - cached "
                "predictions make that instant. Plots and Videos work without it.")

    def _show_results(self):
        r = self.run
        has = (r is not None and bool(r.items)) or bool(self.videos_out)
        self.res_stack.setCurrentIndex(1 if has else 0)
        self._cls_mode()
        cls_run = self._is_cls()
        for combo in (self.cm_model, self.pc_model, self.f_model):
            self._model_combo(combo)
        self._model_combo(self.pl_model, scored_only=False)
        self._pair_combos(self.cmp_a, self.cmp_b)
        for combo in (self.cm_model, self.pc_model, self.f_model, self.cmp_a, self.cmp_b, self.pl_model):
            combo.setEnabled(combo.count() > 0)
        self.f_class.blockSignals(True)
        self.f_class.clear()
        self.f_class.addItem("All classes", -1)
        if r is not None:
            for c, n in sorted(r.names.items()):
                self.f_class.addItem(str(n), n if cls_run else c)
        self.f_class.blockSignals(False)
        cur = self.pl_class.currentData()
        self.pl_class.blockSignals(True)
        self.pl_class.clear()
        self.pl_class.addItem("All classes", -1)
        if r is not None:
            for c, n in sorted(r.names.items()):
                self.pl_class.addItem(str(n), n if cls_run else c)
        self.pl_class.setCurrentIndex(max(0, self.pl_class.findData(cur)))
        self.pl_class.blockSignals(False)
        self.v_pick.blockSignals(True)
        self.v_pick.clear()
        for p, _ in self.videos_out:
            self.v_pick.addItem(os.path.basename(p), p)
        self.v_pick.blockSignals(False)
        title = "Results"
        if r is not None and r.scored:
            what = "sampled video frames" if r.source == "videos" else "scored images"
            ref = (f" · vs reference {r.specs[r.reference].label}"
                   if r.ground == "reference" and r.reference is not None else "")
            title += f" · {len(r.eval_idx):,} {what}{ref}"
        elif r is not None and r.source == "videos":
            title += f" · {len(r.items):,} sampled video frames"
        self.res_title.setText(f"<span style='font-size:15pt; font-weight:600'>{title}</span>")
        if r is None or not r.scored:
            if self.videos_out and self.tabs.currentIndex() != T_VIDEOS and not (r is not None and r.items):
                self.tabs.setCurrentIndex(T_VIDEOS)
        self._results_changed()

    def _results_changed(self):
        """Something the views depend on changed: rebuild the visible tab now, the others when opened."""
        self._thumb_cache = {}
        self._res_dirty = set(range(self.tabs.count()))
        self._render_tab(self.tabs.currentIndex())
        self._revalidate()

    def _render_tab(self, i):
        if i not in getattr(self, "_res_dirty", set()):
            return
        self._res_dirty.discard(i)
        (self._show_overview, self._show_charts, self._show_confusion, self._show_classes, self._show_failures,
         self._show_plots, self._show_compare, self._show_video)[i]()

    def _threshold_changed(self, *_):
        best = self.res_best.isChecked()
        self.res_slider.setEnabled(not best)
        self.res_thr.setText("" if best else f"{self.res_slider.value() / 100:.2f}")
        r = self.run
        if r is None:
            return
        thr = None if best else self.res_slider.value() / 100
        self._bg("threshold", lambda: r.set_threshold(thr), lambda _res: self._results_changed())

    def _show_overview(self):
        from html import escape
        r = self.run
        for i in reversed(range(self.tiles.count())):
            wdg = self.tiles.itemAt(i).widget()
            if wdg:
                self.tiles.removeWidget(wdg)
                wdg.setParent(None)
        self.board.setSortingEnabled(False)
        self.board.setRowCount(0)
        self.board.setVisible(r is not None and bool(r.scored))
        if r is None or not r.scored:
            msg = self._no_score_reason() or ""
            if self.videos_out:
                msg += " The Videos tab shows each model's behaviour over time."
            self.verdict.setText(msg)
            return
        lab = {"best": ("Best", "#199e70"), "fast": ("Speed", "#3987e5"), "tradeoff": ("Trade-off", "#9085e9"),
               "weak": ("Weakness", "#c98500"), "warn": ("Check", "#d03b3b")}
        lines = [f"<tr><td style='padding:2px 8px 2px 0'><b style='color:{lab[k][1]}'>{lab[k][0]}</b></td>"
                 f"<td style='padding:2px 0'>{escape(t)}</td></tr>" for k, t in r.verdict()]
        self.verdict.setText("<table>" + "".join(lines) + "</table>" if lines else "")
        rows = r.summary_table()
        slots = self._run_slots(r)
        for label, val, sub in r.kpis():
            self.tiles.addWidget(StatTile(label, val, sub))
        if self._is_cls(r):
            self._cls_overview_rows(r)
            rows = []
        for x in rows:
            k = self.board.rowCount()
            self.board.insertRow(k)
            self.board.setItem(k, 0, _num_item(x["rank"], "{:.0f}"))
            sw = _item("")
            sw.setIcon(_swatch_icon(slot_hex(slots[x["idx"]])))
            self.board.setItem(k, 1, sw)
            name = _item(x["model"], tip=r.specs[x["idx"]].path)
            name.setData(Qt.ItemDataRole.UserRole, x["idx"])
            self.board.setItem(k, 2, name)
            for c, (key, fmt) in enumerate((("map5095", "{:.3f}"), ("map50", "{:.3f}"), ("precision", "{:.3f}"),
                                            ("recall", "{:.3f}"), ("f1", "{:.3f}"), ("op_conf", "{:.2f}"),
                                            ("speed_ms", "{:.1f}"), ("fps", "{:.0f}"), ("fp", "{:,.0f}"),
                                            ("fn", "{:,.0f}")), 3):
                self.board.setItem(k, c, _num_item(x[key], fmt))
        self.board.setSortingEnabled(True)

    def _board_dbl(self, row, _col):
        it = self.board.item(row, 2)
        if it is not None:
            self._goto_failures(mi=it.data(Qt.ItemDataRole.UserRole))

    def _show_charts(self):
        r = self.run
        if r is None or not r.scored:
            self.charts.set_empty_text(self._no_score_reason() or "")
            self.charts.set_figures([])
            return
        from AIO_QC.core.testing.report import extract_figures
        rb = self._ink_builder(r)

        def build():
            return extract_figures("".join(x for x in (rb._pr_curves(), rb._f1_curves(), rb._per_class(), rb._errors(),
                                                       rb._sizes(), rb._speed_scatter(), rb._class_distribution()) if x))
        self._bg("charts", build, self.charts.set_figures)

    def _show_confusion(self):
        r = self.run
        mi = self.cm_model.currentData()
        self.cm_model.setEnabled(self.cm_model.count() > 0)
        if r is None or mi is None or r.results[mi] is None:
            self.cm_grid.set_empty_text(self._no_score_reason() or "Pick a model.")
            self.cm_grid.set_figures([])
            return
        from AIO_QC.core.testing.report import extract_figures
        rb = self._ink_builder(r)
        self._bg("confusion", lambda: extract_figures(rb._confusion(mi, r.results[mi])), self.cm_grid.set_figures)

    def _show_classes(self):
        r = self.run
        self.pc_table.setSortingEnabled(False)
        self.pc_table.setRowCount(0)
        mi = self.pc_model.currentData()
        self.pc_model.setEnabled(self.pc_model.count() > 0)
        self.pc_table.setVisible(not (r is None or mi is None or r.results[mi] is None))
        if r is None or mi is None or r.results[mi] is None:
            self.pc_note.setText(self._no_score_reason() or "")
            return
        res = r.results[mi]
        if self._is_cls(r):
            self._cls_classes(r, res)
            self.pc_table.setSortingEnabled(True)
            return
        for x in sorted(res.class_rows(), key=lambda x: x["ap50"]):
            k = self.pc_table.rowCount()
            self.pc_table.insertRow(k)
            name = _item(x["name"])
            name.setData(Qt.ItemDataRole.UserRole, x["cls"])
            self.pc_table.setItem(k, 0, name)
            for c, (key, fmt) in enumerate((("gt", "{:,.0f}"), ("ap50", "{:.3f}"), ("ap", "{:.3f}"), ("p", "{:.3f}"),
                                            ("r", "{:.3f}"), ("f1", "{:.3f}"), ("tp", "{:,.0f}"), ("fp", "{:,.0f}"),
                                            ("fn", "{:,.0f}")), 1):
                self.pc_table.setItem(k, c, _num_item(x[key], fmt))
        self.pc_table.setSortingEnabled(True)
        note = (f"Sorted weakest first. P / R / F1 at the best confidence {res.best_conf:.2f}; correct / false / missed "
                f"at the operating threshold {res.op_conf:.2f}.")
        if res.unscored_classes:
            note += " Predicted but not in the ground truth (not scored): " + ", ".join(
                str(r.names.get(c, c)) for c in res.unscored_classes)
        self.pc_note.setText(note)

    def _class_dbl(self, row, _col):
        it = self.pc_table.item(row, 0)
        if it is not None:
            self._goto_failures(mi=self.pc_model.currentData(), cls=it.data(Qt.ItemDataRole.UserRole))

    # ---- click-through from charts
    def _on_link(self, link):
        """cm:<model>:<true>:<pred> · class:<model|-1>:<cls> · err:<model>:<type> · vframe:<frame>"""
        parts = link.split(":")
        if self._is_cls() and parts[0] in ("cm", "class") and len(parts) >= 3:
            try:
                if parts[0] == "cm" and len(parts) == 4:
                    self._goto_failures(mi=int(parts[1]), cm=(parts[2], parts[3]))
                elif parts[0] == "class":
                    mi = int(parts[1])
                    self._goto_failures(mi=None if mi < 0 else mi, cls=":".join(parts[2:]))
            except ValueError:
                pass
            return
        try:
            if parts[0] == "cm":
                self._goto_failures(mi=int(parts[1]), cm=(int(parts[2]), int(parts[3])))
            elif parts[0] == "class":
                mi = int(parts[1])
                self._goto_failures(mi=None if mi < 0 else mi, cls=int(parts[2]))
            elif parts[0] == "err":
                t = parts[2]
                self._goto_failures(mi=int(parts[1]), kind={"missed": "fn"}.get(t, t))
            elif parts[0] == "vframe":
                p = self.v_pick.currentData()
                if p:
                    self._show_moment(p, int(parts[1]))
        except (ValueError, IndexError):
            pass

    def _goto_failures(self, mi=None, cls=None, kind=None, cm=None):
        for combo, val in ((self.f_model, mi), (self.f_class, cls if cls is not None else -1),
                           (self.f_kind, kind or "all")):
            if val is None:
                continue
            k = combo.findData(val)
            if k >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(k)
                combo.blockSignals(False)
        self._f_cm = cm
        names = self.run.names if self.run else {}
        if cm is not None and isinstance(cm[0], str):
            from html import escape
            self.f_chip.setText(f"Showing images of <b>{escape(cm[0])}</b> the model called <b>{escape(cm[1])}</b>")
        elif cm is not None:
            t = "background" if cm[0] < 0 else names.get(cm[0], cm[0])
            p = "nothing (missed)" if cm[1] < 0 else names.get(cm[1], cm[1])
            self.f_chip.setText(f"Showing images where <b>{t}</b> was detected as <b>{p}</b>")
        self.f_chip.setVisible(cm is not None)
        self.f_chip_clear.setVisible(cm is not None)
        self._res_dirty.add(T_FAILURES)
        if self.tabs.currentIndex() == T_FAILURES:
            self._render_tab(T_FAILURES)
        else:
            self.tabs.setCurrentIndex(T_FAILURES)

    def _clear_cm_filter(self):
        self._f_cm = None
        self.f_chip.setVisible(False)
        self.f_chip_clear.setVisible(False)
        self._show_failures()

    # ---- failures gallery
    def _failure_rows(self, r, mi, kind, cls, cm):
        """Computed on the view thread."""
        from AIO_QC.core.testing import metrics as M
        if self._is_cls(r):
            return self._cls_failure_rows(r, mi, kind, cls, cm)
        if r is None or mi is None or r.results[mi] is None:
            return []
        res = r.results[mi]
        pr = r.preds[mi]
        nc = (max(r.names) + 1) if r.names else 1
        gt_lookup = None
        if cm is not None or (cls is not None and cls >= 0):
            idx, gts = r.ground_truth()
            gt_lookup = dict(zip(idx, gts))
        rows = []
        for ii, tp, fp, fn, pt, keep, gtt in r.image_errors(mi):
            if kind == "fp":
                score = fp
            elif kind == "fn":
                score = fn
            elif kind in ("cls", "loc", "dup", "bkg"):
                score = sum(1 for t in pt if t == kind)
            elif kind == "missed_low_conf":
                score = sum(1 for t in gtt if t == "missed_low_conf")
            else:
                score = fp + fn * 1.5
            if cm is not None:
                gb, gc = gt_lookup.get(ii, (np.zeros((0, 4)), np.zeros(0, int)))
                p = pr[ii] if pr[ii] is not None else np.zeros((0, 6), np.float32)
                mat = np.zeros((nc + 1, nc + 1), np.int32)
                M.confusion_update(mat, nc, np.asarray(gb, np.float32).reshape(-1, 4), gc, p, res.op_conf)
                ti = nc if cm[0] < 0 else cm[0]
                pj = nc if cm[1] < 0 else cm[1]
                n = int(mat[ti, pj]) if ti <= nc and pj <= nc else 0
                if n <= 0:
                    continue
                score = n
            elif score <= 0:
                continue
            if cls is not None and cls >= 0:
                gb, gc = gt_lookup.get(ii, (np.zeros((0, 4)), np.zeros(0, int)))
                gc = np.asarray(gc, int)
                p = pr[ii]
                bad_pred = [int(p[k, 5]) for k, t in zip(keep, pt) if t != "tp"] if p is not None else []
                bad_gt = [int(gc[k]) for k, t in enumerate(gtt) if t != "tp" and k < len(gc)]
                if cls not in bad_pred + bad_gt:
                    continue
            rows.append((score, ii, tp, fp, fn))
        rows.sort(key=lambda x: -x[0])
        return rows

    @staticmethod
    def _stop(job):
        try:
            if job is not None and job.isRunning():
                job.cancel()
        except RuntimeError:          # already deleted
            pass

    def _show_failures(self, more=False):
        r = self.run
        if not more:
            self._f_epoch += 1               # thumbnails still loading for the old list are dropped
            self._stop(getattr(self, "_f_job", None))
            mi, kind, cls, cm = self.f_model.currentData(), self.f_kind.currentData(), self.f_class.currentData(), \
                self._f_cm
            self._bg("failures", lambda: self._failure_rows(r, mi, kind, cls, cm),
                     lambda rows: self._fill_failures(rows))
            return
        self._fill_failures(None)

    def _fill_failures(self, rows):
        if rows is not None:
            self.f_gallery.clear()
            self._f_rows, self._f_shown = rows, 0
        rows = self._f_rows[self._f_shown:self._f_shown + 48]
        self._f_shown += len(rows)
        self.f_more.setVisible(self._f_shown < len(self._f_rows))
        if not rows:
            if not self._f_shown:
                self.f_gallery.empty.setText(self._no_score_reason() or (
                    "No images like this." if self._is_cls() else "No images with this kind of error."))
                self.f_gallery.show_empty(True)
            return
        mi = self.f_model.currentData()
        zoom = self.f_zoom.isChecked()
        rb = self._ink_builder(self.run)
        r = self.run
        epoch = self._f_epoch
        placeholders = []
        for score, ii, *rest in rows:
            if self._is_cls(r):
                what = self._cls_row_text(*rest)
            else:
                tp, fp, fn = rest
                what = f"{fp} false · {fn} missed · {tp} correct"
            it = self.f_gallery.add(self._loc_text(r.items[ii]) + f"\n{what}", (mi, ii), r.items[ii].path)
            placeholders.append((it, ii))
        items = {ii: it for it, ii in placeholders}

        def work(job):
            import cv2
            for _, ii in placeholders:
                if job.cancelled() or self._f_epoch != epoch:
                    break
                key = (mi, ii, zoom)
                img = self._thumb_cache.get(key)
                if img is None:
                    try:
                        img = rb.shot_image(ii, mi, focus=zoom, tw=300)
                    except Exception as _exc:
                        _ignored(_exc)
                        img = None
                    if img is not None:
                        s = min(1.0, 600 / img.shape[1])
                        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                                         interpolation=cv2.INTER_AREA) if s < 1 else img
                        self._thumb_cache[key] = img
                job.progress.emit((ii, img))

        def prog(p):
            ii, img = p
            if self._f_epoch == epoch and img is not None and ii in items:
                items[ii].setIcon(QIcon(fit_pixmap(img, self.f_list.iconSize())))
        j = Job(work, self)
        j.progress.connect(prog)
        j.finished.connect(j.deleteLater)
        self._f_job = j
        j.start(QThread.Priority.LowPriority)

    @staticmethod
    def _loc_text(it):
        v = it.meta.get("video")
        if v:
            return f"{os.path.basename(v)} · {human_time(it.meta.get('time', 0))}"
        return os.path.basename(it.path)

    # ---- Testing -> QC review
    def _review_target(self):
        """The bridge when the run's images are the dataset open in Image QC, else None (+ a note)."""
        if self.qc is None or self.run is None or not self.qc.is_this(self.data):
            QMessageBox.information(self, "Send to review", "Sending to review works when the test data is the "
                                    "dataset open in Image QC (Test data ▸ This dataset).")
            return None
        return self.qc

    def _send_failures_to_review(self):
        qc = self._review_target()
        if qc is None:
            return
        items = self._failure_items()
        if not items:
            self.status.showMessage("No failures shown.", 4000)
            return
        r = self.run
        mi = self.f_model.currentData()
        what = f"{self.f_kind.currentText()} · {r.specs[mi].label if mi is not None and mi < len(r.specs) else ''}"
        cls = self.f_class.currentText()
        if cls and not cls.lower().startswith("all"):
            what += f" · {cls}"
        n = qc.send_to_review([(r.items[ii].label_path, r.items[ii].path) for ii in items], f"Testing failures: {what}")
        self.status.showMessage(f"{n:,} image(s) flagged for review in Image QC (file list filtered to them)", 8000)

    def _send_compare_to_review(self):
        qc = self._review_target()
        if qc is None:
            return
        items = []
        for _lab, lst in self.cmp_lists:
            items += list(lst.property("items") or [])
        if not items:
            self.status.showMessage("No differing images.", 4000)
            return
        r = self.run
        a, b = self.cmp_a.currentText(), self.cmp_b.currentText()
        n = qc.send_to_review([(r.items[ii].label_path, r.items[ii].path) for ii in sorted(set(items))],
                              f"Testing: {a} vs {b} disagree")
        self.status.showMessage(f"{n:,} image(s) flagged for review in Image QC", 8000)

    def _failure_items(self):
        return [ii for _, ii, *_ in self._f_rows]

    def _failure_open(self, it):
        d = it.data(Qt.ItemDataRole.UserRole)
        if not d:
            return
        mi, ii = d
        items = self._failure_items()
        if self._is_cls():
            self._cls_open(items, items.index(ii) if ii in items else 0, [mi])
        elif self._can_edit(ii):
            self._open_editor(items, items.index(ii) if ii in items else 0, mi)
        else:
            self._open_plot(items, items.index(ii) if ii in items else 0, mi, True)

    def _can_edit(self, ii):
        """Labels can be fixed on test images (not on sampled video frames, which have no label files)."""
        r = self.run
        if self._is_cls(r):
            return self._cls_can_edit(ii)
        return (self.data is not None and r is not None and r.ground == "labels"
                and r.items[ii].meta.get("source") != "video")

    def _failure_menu(self, pos):
        it = self.f_list.itemAt(pos)
        d = it.data(Qt.ItemDataRole.UserRole) if it else None
        if not d:
            return
        mi, ii = d
        item = self.run.items[ii]
        items = self._failure_items()
        k = items.index(ii) if ii in items else 0
        m = QMenu(self)
        if self._can_edit(ii):
            m.addAction("Change class…" if self._is_cls() else "Fix labels…").triggered.connect(
                lambda: self._open_editor(items, k, mi))
        if len(self.run.scored) > 1:
            m.addAction("Compare with another model…").triggered.connect(lambda: self._open_ab(items, k, mi))
        m.addAction("View plotted image…").triggered.connect(lambda: self._open_plot(items, k, mi, True))
        m.addAction("Save plotted image…").triggered.connect(lambda: self._save_plot(ii, mi, True))
        m.addSeparator()
        if not item.meta.get("source") == "video":
            m.addAction("Show file").triggered.connect(lambda: open_path(os.path.dirname(item.path)))
        m.exec(self.f_list.viewport().mapToGlobal(pos))

    # ---- viewers / editor
    def _open_editor(self, items, start=0, mi=None):
        if self.run is None or not items or self.data is None:
            return
        if self._is_cls():
            self._cls_open(items, start, [mi if mi is not None else (self.run.ranking() or [0])[0]])
            return
        from AIO_QC.widgets.testing.annot_view import LabelEditor
        from AIO_QC.core.testing.pipeline import frame_getter
        d = LabelEditor(self, self.run, self.data, items, start, mi, frame_getter, self._run_slots(self.run))
        d.qc = self.qc if self.qc is not None and self.qc.is_this(self.data) else None
        d.labels_changed.connect(self._labels_changed)
        try:
            d.exec()
        finally:
            _release_thread_reader()

    def _open_ab(self, items=None, start=0, a=None):
        r = self.run
        if r is None or len(r.specs) < 2:
            return
        b_ = self.cmp_b.currentData()
        a_ = a if a is not None else self.cmp_a.currentData()
        if a_ is None:                               # unscored / vs-reference runs: any two models side by side
            a_ = r.reference if r.reference is not None else 0
        if b_ is None:
            b_ = next((i for i in range(len(r.specs)) if i != a_), a_)
        if a_ == b_:
            b_ = next((i for i in r.ranking() if i != a_), next((i for i in range(len(r.specs)) if i != a_), b_))
        if not items:
            c = r.comparison(a_, b_) if r.results[a_] is not None and r.results[b_] is not None else None
            items = ([ii for ii, _ in c["b_better"]] + [ii for ii, _ in c["a_better"]]) if c else []
            if not items and r.ground == "reference" and r.results[b_ if b_ != r.reference else a_] is not None:
                m = b_ if b_ != r.reference else a_  # the frames where the model disagrees with the reference most
                items = ([x[0] for x in r.mistakes(m)[:500]] if self._is_cls(r) else
                         [x[0] for x in sorted(r.image_errors(m), key=lambda x: -(x[2] + x[3]))[:500]])
            items = items or list(r.eval_idx) or list(range(len(r.items)))
        if self._is_cls(r):
            self._cls_open(items, start, [a_, b_])
            return
        from AIO_QC.widgets.testing.annot_view import ABViewer
        from AIO_QC.core.testing.pipeline import frame_getter
        v = ABViewer(self, r, items, a_, b_, start, frame_getter, self._run_slots(r),
                     plotter=self._plotter(), can_edit=self._can_edit)
        v.edit_requested.connect(lambda its, pos: self._open_editor(its, pos, a_))
        try:
            v.exec()
        finally:
            _release_thread_reader()

    def _labels_changed(self, indices):
        """Labels / exclusions were edited: re-score in the background, refresh what is shown."""
        r = self.run
        if r is None:
            return
        self._log(f"Labels changed on {len(indices)} image(s) - re-scoring")

        thr = None if self.res_best.isChecked() else self.res_slider.value() / 100      # read on the UI thread

        def rescore():
            r.evaluate()
            if thr is not None:
                r.set_threshold(thr)
            return True
        self._bg("rescore", rescore, lambda _res: (self._show_results(), self._show_summary(), self._autosave()))

    # ---- compare
    def _show_compare(self):
        r = self.run
        a, b = self.cmp_a.currentData(), self.cmp_b.currentData()
        self._cmp_epoch += 1
        self._stop(getattr(self, "_cmp_job", None))
        for (lab, _), gal in zip(self.cmp_lists, self.cmp_galleries):
            lab.setText("")
            gal.clear()
            gal.show_empty(False)
        for cb in (self.cmp_a, self.cmp_b):
            cb.setEnabled(cb.count() > 0)
        if r is None or a is None or b is None or a == b:
            reason = self._no_score_reason()
            if reason is None and r is not None and len(r.scored) < 2:
                reason = ("Comparing needs two scored models" + (
                    f" - the reference ({r.specs[r.reference].label}) is the answer key, so add a third model or "
                    "compare each one with it in the Overview." if r.ground == "reference" else "."))
            self.cmp_grid.set_empty_text(reason or "Pick two different models.")
            self.cmp_grid.set_figures([])
            self.cmp_tiles.setText("")
            return
        from AIO_QC.core.testing.report import extract_figures
        from AIO_QC.core.testing import charts as C
        rbld = self._ink_builder(r)
        if r.ground == "reference" and r.reference in (a, b):
            (self._cls_compare_reference if self._is_cls(r) else self._show_compare_reference)(
                r, a if a == r.reference else b, b if a == r.reference else a, rbld)
            return
        if self._is_cls(r):
            self._bg("compare", lambda: self._cls_compare_build(r, a, b, rbld),
                     lambda res: self._fill_compare(res, a, b, rbld))
            return

        def build():
            c = r.comparison(a, b)
            if not c:
                return None
            ra, rb_ = r.results[a], r.results[b]
            pc = sorted(c["per_class"], key=lambda x: x["delta"])
            svg = C.diverging_bars([x["name"] for x in pc], [x["delta"] for x in pc], 560, rbld.ink,
                                   "AP50 difference per class", labels=(f"{ra.name} better", f"{rb_.name} better"))
            figs = [("Per-class difference", f"AP50 of {rb_.name} minus {ra.name}.", svg)]
            figs += extract_figures(rbld._pr_curves_subset([a, b]))
            figs += extract_figures(rbld._confusion(a, ra) + rbld._confusion(b, rb_))
            return c, figs
        self._bg("compare", build, lambda res: self._fill_compare(res, a, b, rbld))

    def _fill_compare(self, res, a, b, rbld):
        r = self.run
        if res is None or r is None:
            self.cmp_grid.set_figures([])
            return
        c, figs = res
        from html import escape
        ra, rb_ = r.results[a], r.results[b]

        def d(v, good_up=True):
            col = "#199e70" if (v > 0) == good_up and abs(v) > 1e-9 else ("#d03b3b" if abs(v) > 1e-9 else "inherit")
            return f"<b style='color:{col}'>{v:+.3f}</b>"
        sp = ""
        if ra.speed_ms and rb_.speed_ms:
            pct = (rb_.speed_ms - ra.speed_ms) / ra.speed_ms * 100
            sp = (f" · speed {rb_.speed_ms:.1f} vs {ra.speed_ms:.1f} ms "
                  f"(<b style='color:{'#199e70' if pct < 0 else '#d03b3b'}'>{pct:+.0f}%</b>)")
        if self._is_cls(r):
            self.cmp_tiles.setText(self._cls_compare_tiles(c, ra, rb_))
        else:
            self.cmp_tiles.setText(f"<b>{escape(rb_.name)}</b> vs <b>{escape(ra.name)}</b>: mAP50-95 "
                                   f"{d(c['map_delta'])} · mAP50 {d(c['map50_delta'])}{sp} · better on "
                                   f"<b>{len(c['b_better']):,}</b> images, worse on <b>{len(c['a_better']):,}</b>")
        self.cmp_grid.set_figures(figs)
        epoch = self._cmp_epoch
        jobs = []
        cls_run = self._is_cls(r)
        for (lab, lst), (title, rows) in zip(self.cmp_lists, (
                (f"Only {rb_.name} right" if cls_run else f"Where {rb_.name} does better", c["b_better"]),
                (f"Only {ra.name} right" if cls_run else f"Where {ra.name} does better", c["a_better"]))):
            lab.setText(f"<b>{escape(title)}</b> ({len(rows):,}) - left {escape(ra.name)}, right {escape(rb_.name)}. "
                        "Double-click for the side-by-side viewer.")
            lst.setProperty("items", [ii for ii, _ in rows])
            gal = self.cmp_galleries[[x[1] for x in self.cmp_lists].index(lst)]
            gal.empty.setText("No images where this model does better.")
            gal.show_empty(not rows)
            for ii, gain in rows[:24]:
                it = gal.add(self._loc_text(r.items[ii]) + (f" · other said it at {gain:.0%}" if cls_run else
                                                             f" · +{gain}"), ii, r.items[ii].path)
                jobs.append((it, ii))

        def work(job):
            for it, ii in jobs:
                if job.cancelled() or self._cmp_epoch != epoch:
                    break
                try:
                    img = rbld.shot_image(ii, a, focus=False, compare=(a, b), tw=320)
                except Exception as _exc:
                    _ignored(_exc)
                    img = None
                job.progress.emit((it, img))

        def prog(p):
            it, img = p
            if self._cmp_epoch == epoch and img is not None:
                it.setIcon(QIcon(fit_pixmap(img, it.listWidget().iconSize() if it.listWidget() else QSize(300, 112))))
        j = Job(work, self)
        j.progress.connect(prog)
        j.finished.connect(j.deleteLater)
        self._cmp_job = j
        j.start(QThread.Priority.LowPriority)

    def _show_compare_reference(self, r, ref, m, rbld):
        """Compare a model with the reference model (videos-only / unlabelled runs): agreement numbers, per-class
        agreement, its confusion against the reference, and the frames where it misses what the reference finds or
        finds what the reference does not."""
        from html import escape
        from AIO_QC.core.testing.report import extract_figures
        res = r.results[m]
        if res is None:
            self.cmp_grid.set_empty_text("Pick a model to compare with the reference.")
            self.cmp_grid.set_figures([])
            return

        def build():
            figs = extract_figures(rbld._confusion(m, res, f"{res.name} vs reference {r.specs[ref].label}"))
            figs += extract_figures(rbld._per_class())
            figs += extract_figures(rbld._errors())
            errs = r.image_errors(m)
            misses = sorted(((ii, fn) for ii, tp, fp, fn, *_ in errs if fn), key=lambda x: -x[1])
            extras = sorted(((ii, fp) for ii, tp, fp, fn, *_ in errs if fp), key=lambda x: -x[1])
            return figs, misses, extras
        epoch = self._cmp_epoch

        def fill(out):
            if epoch != self._cmp_epoch:
                return
            figs, misses, extras = out
            p, rc, f1 = res.op_prf()
            rn, mn = escape(r.specs[ref].label), escape(res.name)
            self.cmp_tiles.setText(
                f"<b>{mn}</b> vs reference <b>{rn}</b> at conf {res.op_conf:.2f}: finds <b>{rc * 100:.0f}%</b> of what "
                f"{rn} finds · <b>{p * 100:.0f}%</b> of its detections are confirmed by {rn} · mAP50 "
                f"<b>{res.map50:.3f}</b> · differs on <b>{len({ii for ii, _ in misses} | {ii for ii, _ in extras}):,}</b> "
                f"of {len(r.eval_idx):,} frames")
            self.cmp_grid.set_figures(figs)
            jobs = []
            for (lab, lst), gal, (title, rows) in zip(self.cmp_lists, self.cmp_galleries, (
                    (f"{res.name} misses what {r.specs[ref].label} finds", misses),
                    (f"{res.name} finds what {r.specs[ref].label} does not", extras))):
                lab.setText(f"<b>{escape(title)}</b> ({len(rows):,}) - left {rn}, right {mn}. "
                            "Double-click for the side-by-side viewer.")
                lst.setProperty("items", [ii for ii, _ in rows])
                gal.empty.setText("None.")
                gal.show_empty(not rows)
                for ii, n in rows[:24]:
                    jobs.append((gal.add(self._loc_text(r.items[ii]) + f" · {n}", ii, r.items[ii].path), ii))

            def work(job):
                for it, ii in jobs:
                    if job.cancelled() or self._cmp_epoch != epoch:
                        break
                    try:
                        img = rbld.shot_image(ii, m, focus=False, compare=(ref, m), tw=320)
                    except Exception as _exc:
                        _ignored(_exc)
                        img = None
                    job.progress.emit((it, img))

            def prog(pp):
                it, img = pp
                if self._cmp_epoch == epoch and img is not None:
                    it.setIcon(QIcon(fit_pixmap(img, it.listWidget().iconSize() if it.listWidget() else QSize(300, 112))))
            j = Job(work, self)
            j.progress.connect(prog)
            j.finished.connect(j.deleteLater)
            self._cmp_job = j
            j.start(QThread.Priority.LowPriority)
        self._bg("compare", build, fill)

    def _compare_open(self, lst, it):
        ii = it.data(Qt.ItemDataRole.UserRole)
        if ii is None:
            return
        items = lst.property("items") or [ii]
        self._open_ab(items, items.index(ii) if ii in items else 0, self.cmp_a.currentData())

    # ---- videos
    def _show_video(self):
        p = self.v_pick.currentData()
        v = next((a for path, a in self.videos_out if path == p), None)
        self.v_events_model.set_rows([])
        if v is None:
            self.v_grid.set_figures([])
            return
        from AIO_QC.core.testing.report import extract_figures
        rb = self._ink_builder(None, [(p, v)])
        rb.get_frame = None                         # no gallery here: the events table opens the plot viewer

        def build():
            figs = extract_figures(rb._video_section(p, v))
            names = ({"disagree": "Models disagree", "flicker": "Drop-out (unsure for a moment)",
                      "blip": "Brief flip (another class for a moment)"} if v.get("task") == "classify" else
                     {"disagree": "Models disagree", "flicker": "Flicker (missed for a moment)",
                      "blip": "Blip (one-off detection)"})
            evs = sorted(v["events"], key=lambda e: (e["type"] != "disagree", e.get("score", 1), e["time"]))[:20000]
            rows = [(human_time(e["time"]), e["frame"], names.get(e["type"], e["type"]), e["model"], e.get("cls") or "")
                    for e in evs]
            keys = [(e["time"], e["frame"], e["type"], e["model"], e.get("cls") or "") for e in evs]
            return figs, rows, keys
        self._bg("video", build, lambda res: self._fill_video(v, *res))

    def _fill_video(self, v, figs, rows, keys):
        self.v_grid.set_figures(figs)
        self.v_grid.add_widget(self._video_table(v))
        self.v_events_model.set_rows(rows, keys)
        step = v.get("step_s")
        if step:
            self.v_grid.set_empty_text("")
            self.status.showMessage(f"Video analysed every {step:.2f} s · {len(v['times']):,} samples · "
                                    f"{len(v['events']):,} events", 8000)

    def _video_table(self, v):
        cls = v.get("task") == "classify"
        t = _make_table(["", "Model", "Confidence", "Answers", "Frames answered", "Class runs", "Drop-outs",
                         "Brief flips", "Stability"] if cls else
                        ["", "Model", "Threshold", "Detections", "Frames with detections", "Tracked objects", "Flicker",
                         "Blips", "Stability"], stretch_col=1)
        slots = self._run_slots(None, self.videos_out) or list(range(len(v["models"])))
        for i, m in enumerate(v["models"]):
            t.insertRow(i)
            sw = _item("")
            sw.setIcon(_swatch_icon(slot_hex(slots[i] if i < len(slots) else i)))
            t.setItem(i, 0, sw)
            t.setItem(i, 1, _item(m["label"]))
            for c, (val, fmt) in enumerate(((m["op_conf"], "{:.2f}"), (m["detections"], "{:,.0f}"),
                                            (m["frames_with_dets"], "{:,.0f}"), (m["tracks"], "{:,.0f}"),
                                            (m["flicker"], "{:,.0f}"), (m["blip"], "{:,.0f}"),
                                            (m["stability"] * 100, "{:.1f}%")), 2):
                t.setItem(i, c, _num_item(val, fmt))
        t.setFixedHeight(32 + 30 * max(1, len(v["models"])))
        t.setToolTip("Class runs: stretches with one answer. Brief flip: an answer that lasts at most two samples. "
                     "Drop-out: unsure for a moment. Stability: share of frames in neither." if cls else
                     "Flicker: an object disappears for a frame or two and comes back. Blip: an object seen for at "
                     "most two samples. Stability: share of detections involved in neither.")
        return t

    def _event_open(self, index):
        p = self.v_pick.currentData()
        if p and index.isValid():
            row = self.v_events_model.keys[index.row()]
            self._show_moment(p, int(row[1]))
