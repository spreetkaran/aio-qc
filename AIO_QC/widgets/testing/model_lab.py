# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Model Lab - test and compare YOLO weights on your own footage, visually.

  1  Test data   build a test set: unique frames from any number of videos and image folders (duplicates removed
                 inside each video and across all of them) - or use a labelled YOLO / LabelMe folder or data.yaml.
                 Pre-label with a model, fix the labels in any tool, mark them reviewed.
                 Optional: whole videos for label-free behaviour (stability, agreement between models).
  2  Models      any number of weights (.pt / .onnx / .engine), each in its own colour everywhere
  3  Run         every model on every image (batched, cached, resumable - re-runs are instant)
  4  Results     leaderboard, plain-language verdict, PR / F1 curves, per-class AP, error types, recall by size,
                 speed vs accuracy, confusion matrices, failure gallery, A-vs-B comparison, video timelines -
                 with a live operating-threshold control
  5  Report      single-file HTML (summary, detailed, comparison, video) + metrics.json / CSV
"""
import os
import time

import numpy as np
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer, QThread, QPointF, QRectF
from PyQt6.QtGui import QColor, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QComboBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QRadioButton, QButtonGroup, QGroupBox, QLineEdit, QFileDialog, QMessageBox, QApplication,
    QPlainTextEdit, QTabWidget, QSlider, QScrollArea, QFrame, QSplitter, QDialog, QDialogButtonBox, QMenu,
    QStackedWidget, QToolButton,
)

from AIO_QC.widgets import motion as anim
from AIO_QC.widgets.testing.lab_common import (
    Job,
    ChartGrid,
    ProgressRow,
    theme_mode,
    human_time,
    open_path,
    bgr_pixmap,
    IssueBar,
    RowsModel,
    SerialWorker,
    StatTile,
    Gallery,
    fit_pixmap,
    last_dir,
    remember_dir,
)

from AIO_QC.tools.frame_extractor.core.common import VIDEO_FILTER, WEIGHTS_FILTER, parse_range
from AIO_QC.tools.frame_extractor.core.imageio import imread
from AIO_QC.utils.fx_settings import testing_settings
from AIO_QC.utils.qt_utils import later   # owner-bound timer: never fires after `self` is gone
def _release_thread_reader():
    from AIO_QC.core.testing.pipeline import release_reader
    release_reader()


if _release_thread_reader not in Job.thread_cleanups:
    Job.thread_cleanups.append(_release_thread_reader)

# Results tabs
T_OVERVIEW, T_CHARTS, T_CONFUSION, T_CLASSES, T_FAILURES, T_PLOTS, T_COMPARE, T_VIDEOS = range(8)
STEPS = [("Test data", "Videos, image folders, labels"), ("Models", "Weights to test"),
         ("Run", "Predict, cached & resumable"), ("Results", "Charts, plots, failures"),
         ("Report", "HTML report & data"), ("Review videos", "Play videos with every model's detections")]


def _shared_env():
    """The one environment for .pt / .engine models (Image QC ▸ Models & Environment)."""
    from AIO_QC.utils.qt_utils import app_settings
    return app_settings().value("inference/python", "") or ""


def _set_shared_env(path):
    if path:
        from AIO_QC.utils.qt_utils import app_settings
        app_settings().setValue("inference/python", path)


def _unique_tip(k):
    from AIO_QC.tools.frame_extractor.core.dedup import PRESET_HELP
    return "Keep every sampled frame." if k == "off" else PRESET_HELP.get(k, "")


def slot_hex(slot, mode=None):
    from AIO_QC.tools.frame_extractor.core.common import model_hex
    return model_hex(slot, mode or theme_mode())


def _swatch_icon(color, size=12):
    from AIO_QC.utils import painted_icon
    c = QColor(color)
    return painted_icon(lambda p: p.fillRect(QRectF(0, 0, size, size), c), size)


def _item(text, sort=None, align_right=False, tip=None):
    it = QTableWidgetItem(str(text))
    if sort is not None:
        it.setData(Qt.ItemDataRole.UserRole, sort)
    if align_right:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return it


class NumItem(QTableWidgetItem):
    """Sorts by its numeric value (UserRole), not its text."""

    def __lt__(self, other):
        a, b = self.data(Qt.ItemDataRole.UserRole), other.data(Qt.ItemDataRole.UserRole)
        try:
            return float(a) < float(b)
        except (TypeError, ValueError):
            return super().__lt__(other)


def _num_item(v, fmt="{:.3f}", tip=None):
    it = NumItem("–" if v is None else fmt.format(v))
    it.setData(Qt.ItemDataRole.UserRole, -1e18 if v is None else float(v))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if tip:
        it.setToolTip(tip)
    return it


def _make_table(headers, stretch_col=None):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setAlternatingRowColors(True)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    if stretch_col is not None:
        hh.setSectionResizeMode(stretch_col, QHeaderView.ResizeMode.Stretch)
    return t


def _status_icon(level, size=14):
    """Step status for the navigation: ✓ ok · ! warning · ✕ problem · ○ not done yet."""
    from AIO_QC.utils import painted_icon
    col = {"ok": "#199e70", "ready": "#199e70", "warn": "#c98500", "error": "#d03b3b"}.get(level, "#8f8d86")
    return painted_icon(lambda p: _draw_status(p, level, size, col), size)


def _draw_status(p, level, size, col):
    from PyQt6.QtGui import QPen
    from PyQt6.QtCore import QRectF
    r = QRectF(1, 1, size - 2, size - 2)
    if level == "todo":
        p.setPen(QPen(QColor(col), 1.5))
        p.drawEllipse(r)
    else:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(col))
        p.drawEllipse(r)
        p.setPen(QPen(QColor("white"), 1.8))
        c = size / 2
        if level in ("ok", "ready"):
            p.drawPolyline([QPointF(c - 3, c), QPointF(c - 0.8, c + 2.4), QPointF(c + 3.4, c - 2.6)])
        elif level == "warn":
            p.drawLine(QPointF(c, c - 3.2), QPointF(c, c + 0.8))
            p.drawPoint(QPointF(c, c + 3))
        else:
            p.drawLine(QPointF(c - 2.6, c - 2.6), QPointF(c + 2.6, c + 2.6))
            p.drawLine(QPointF(c + 2.6, c - 2.6), QPointF(c - 2.6, c + 2.6))


def _strip_tags(t):
    import re
    return re.sub(r"<[^>]+>", "", t)


def _hint(text):
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setEnabled(False)           # muted in both themes
    return lab


def _scroll(widget):
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setWidget(widget)
    return sa


class ImagePreview(QDialog):
    """Large view of one result image, with actions."""

    def __init__(self, parent, img, title, actions=()):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        scr = parent.screen().availableGeometry() if parent is not None and parent.screen() else None
        mw = int(scr.width() * 0.8) if scr else 1400
        mh = int(scr.height() * 0.75) if scr else 800
        lab = QLabel()
        lab.setPixmap(bgr_pixmap(img, mw, mh))
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lab, 1)
        bb = QDialogButtonBox()
        for text, fn in actions:
            b = bb.addButton(text, QDialogButtonBox.ButtonRole.ActionRole)
            b.clicked.connect(fn)
        bb.addButton(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)
        lay.addWidget(bb)


# ====================================================================== window
class ModelLab(QMainWindow):
    help_requested = pyqtSignal(str)               # guide page
    HOME = 5                                       # page index of the start page
    REVIEW = 6                                     # page index of "6 · Review videos" (after the start page)

    def __init__(self, parent=None, model_cfg=None, ffmpeg_getter=None, version="", qc=None):
        super().__init__(parent)
        self.setWindowTitle("Testing")
        self.qc = qc                    # QCBridge to the dataset open in Image QC (None: standalone / no window)
        from AIO_QC.core.fx_storage import migrate_files
        migrate_files()
        self.setObjectName("ModelLab")
        self.st = testing_settings()
        self.model_cfg = dict(model_cfg or {})
        self.ffmpeg_getter = ffmpeg_getter or (lambda: None)
        self.version = version
        self.job = None
        self.data = None                # pipeline.Data
        self.testset = None             # testset.TestSet when the data is a test set
        self.run = None                 # evaluation.LabRun
        self.videos_out = []            # [(path, analysis)]
        self.infos = {}
        self.cache = None
        self.specs = []                 # ModelSpec in table order (colour slot = position when added)
        self.slots = []
        self._thumb_cache = {}
        self._report_path = None
        self.prog_rows = []
        self._f_epoch = self._cmp_epoch = 0
        self._f_rows, self._f_shown = [], 0
        self._f_cm = None                # confusion-cell filter of the failure gallery (true, predicted)
        self._res_dirty = set()
        self._busy_keys = set()
        self.issue_bars = {}
        self.views = SerialWorker(self)  # builds result views off the UI thread
        self._project_path, self._project_name = None, "Untitled"
        self._dirty = self._loading = False
        self._summary = {}
        self._run_specs_keys = []
        self._pending_reference = None
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(1500)
        self._autosave_timer.timeout.connect(self._autosave)
        self._valid_timer = QTimer(self)
        self._valid_timer.setSingleShot(True)
        self._valid_timer.setInterval(120)
        self._valid_timer.timeout.connect(self._apply_validation)
        from AIO_QC.tools.frame_extractor.utils import fonts
        fonts.load()                     # Inter for chart and plot labels only (the window keeps the app font)
        self._build()
        self._build_menu()
        self._watch_changes()
        self.setAcceptDrops(True)
        self._drop_hint = QLabel("Drop videos, image folders, a data.yaml, weights or a .fxlab project", self)
        self._drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_hint.setWordWrap(True)
        self._drop_hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._drop_hint.setStyleSheet("background: rgba(57,135,229,0.10); border: 2px dashed #3987e5; "
                                      "border-radius: 10px; font-size: 14pt; color: #3987e5;")
        self._drop_hint.hide()
        self._restore()
        from AIO_QC.tools.frame_extractor.utils.qt_utils import preferred_size
        preferred_size(self, 1120, 740)
        anim.fade_on_show(self, 200)

    # ------------------------------------------------------------------ layout
    def _build(self):
        central = QWidget()
        h = QVBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        # the steps as a stepper across the top (fits the Models tab's tool windows): ⌂ Projects | 1 … 5
        top = QWidget()
        top.setObjectName("testingStepper")
        sv = QHBoxLayout(top)
        sv.setContentsMargins(8, 6, 8, 4)
        sv.setSpacing(6)
        home_b = QPushButton("⌂  Projects")
        home_b.setFlat(True)
        home_b.setToolTip("Start page: new / open / recent projects")
        home_b.clicked.connect(lambda: self._show_home())
        sv.addWidget(home_b)
        self.nav = QListWidget()
        self.nav.setFlow(QListWidget.Flow.LeftToRight)
        self.nav.setWrapping(False)
        self.nav.setIconSize(QSize(16, 16))
        self.nav.setSpacing(2)
        self.nav.setFixedHeight(40)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.nav.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for i, (t, sub) in enumerate(STEPS):
            it = QListWidgetItem(f"{i + 1}  {t}" + ("   ›" if i < len(STEPS) - 1 else ""))
            it.setToolTip(sub)
            it.setSizeHint(QSize(150, 32))
            self.nav.addItem(it)
        self.nav.currentRowChanged.connect(self._go)
        self.nav.setFrameShape(QFrame.Shape.NoFrame)
        sv.addWidget(self.nav, 1)
        h.addWidget(top)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        h.addWidget(line)
        self.pages = anim.AnimatedStack()
        h.addWidget(self.pages, 1)
        self.setCentralWidget(central)
        self.pages.addWidget(self._page_data())
        # every page scrolls when the window is small: the window never needs more than the screen
        self.pages.addWidget(_scroll(self._page_models()))
        self.pages.addWidget(_scroll(self._page_run()))
        self.pages.addWidget(self._page_results())
        self.pages.addWidget(_scroll(self._page_report()))
        self.pages.addWidget(self._page_home())              # HOME
        self.pages.addWidget(_scroll(self._page_review()))   # REVIEW
        self.status = self.statusBar()
        # long model / file names must not widen the window: combos elide, labels wrap
        for cb in self.findChildren(QComboBox):
            cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(10)
        self.run_info.setWordWrap(True)
        self.nav.setCurrentRow(0)

    def _go(self, row):
        if row < 0:
            return
        self.pages.setCurrentIndex(self.REVIEW if row == 5 else row)
        if row == 5:
            self._refresh_review_page()
        if row == 1:
            self._refresh_mapping_hint()
        if row == 2:
            self._refresh_run_page()
        if row == 4:
            self._refresh_report_page()

    def _page(self, title, subtitle, step=None):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)
        t = QLabel(f"<span style='font-size:15pt; font-weight:600'>{title}</span>")
        lay.addWidget(t)
        if subtitle:
            lay.addWidget(_hint(subtitle))
        if step is not None:
            self.issue_bars[step] = IssueBar()
            lay.addWidget(self.issue_bars[step])
        return w, lay

    # ------------------------------------------------------------------ 1 data
    def _page_data(self):
        w, lay = self._page("Test data", step=0, subtitle="The images every model is measured on. Build a reusable test set from "
                            "videos and image folders (unique frames only), or use a labelled dataset you already "
                            "have. Labels are optional: without them, models are compared with each other.")
        kind = QHBoxLayout()
        self.kind_qc = QRadioButton("This dataset (Image QC)")
        self.kind_qc.setToolTip("The dataset open in Image QC: its files, labels and classes. Label fixes made here "
                                "go through AIO QC (backed up, in the file's history, visible in Image QC at once).")
        self.kind_ts = QRadioButton("Test set from videos")
        self.kind_ds = QRadioButton("Labelled dataset (YOLO / LabelMe folder or data.yaml)")
        self.kind_none = QRadioButton("Videos only")
        self.kind_ts.setToolTip("A folder AIO QC manages: frames, labels, cache and reports. Opens in any "
                                "YOLO tool (data.yaml). Re-use it to measure next month's model on the same frames.")
        self.kind_ds.setToolTip("An existing dataset: YOLO txt labels (images/ + labels/), LabelMe json next to the "
                                "images, or a YOLO data.yaml (val split).")
        self.kind_none.setToolTip("Skip accuracy on images: analyse whole videos only (stability, agreement).")
        g = QButtonGroup(self)
        for b in (self.kind_qc, self.kind_ts, self.kind_ds, self.kind_none):
            g.addButton(b)
            kind.addWidget(b)
        kind.addStretch(1)
        lay.addLayout(kind)
        self.kind_ts.setChecked(True)
        for b in (self.kind_qc, self.kind_ts, self.kind_ds, self.kind_none):
            b.toggled.connect(self._kind_changed)

        # --- the dataset open in Image QC
        self.qc_box = QGroupBox("Dataset open in Image QC")
        ql = QHBoxLayout(self.qc_box)
        self.qc_lbl = QLabel("")
        self.qc_lbl.setWordWrap(True)
        ql.addWidget(self.qc_lbl, 1)
        b = QPushButton("Use it")
        b.setToolTip("Load the open dataset (again) as the test data")
        b.clicked.connect(lambda: self._load_qc())
        ql.addWidget(b)
        lay.addWidget(self.qc_box)

        # --- test set
        self.ts_box = QGroupBox("Test set")
        tl = QVBoxLayout(self.ts_box)
        row = QHBoxLayout()
        self.ts_path = QLineEdit()
        self.ts_path.setReadOnly(True)
        self.ts_path.setPlaceholderText("Create a new test set or open one")
        b_new = QPushButton("New…")
        b_new.clicked.connect(self._new_testset)
        b_open = QPushButton("Open…")
        b_open.clicked.connect(self._open_testset)
        row.addWidget(self.ts_path, 1)
        row.addWidget(b_new)
        row.addWidget(b_open)
        tl.addLayout(row)
        self.src_table = _make_table(["Type", "Source", "Range"], stretch_col=1)
        self.src_table.setMinimumHeight(110)
        self.src_table.setToolTip("Sources to add to the test set. Double-click Range to limit a video "
                                  "(e.g. 0:30-5:00).")
        self.src_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        tl.addWidget(self.src_table)
        row = QHBoxLayout()
        for text, fn in (("Add videos…", self._add_src_videos), ("Add image folder…", self._add_src_folder),
                         ("Remove", self._remove_src)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        tl.addLayout(row)
        sam = QHBoxLayout()
        self.sampling = QComboBox()
        for label, key in (("A frame every", "interval"), ("Every Nth frame", "every_n"), ("N frames per video", "count"),
                           ("All frames", "all")):
            self.sampling.addItem(label, key)
        self.sampling.setToolTip("How frames are taken from each video before duplicates are removed.")
        self.samp_val = QDoubleSpinBox()
        self.samp_val.setRange(0.04, 100000)
        self.samp_val.setDecimals(2)
        self.samp_val.setValue(1.0)
        self.sampling.currentIndexChanged.connect(self._sampling_changed)
        self.max_per = QSpinBox()
        self.max_per.setRange(0, 1000000)
        self.max_per.setSpecialValueText("no limit")
        self.max_per.setToolTip("At most this many frames from one video (0 = no limit) - keeps one long video "
                                "from dominating the test set.")
        self.unique = QComboBox()
        for k in ("off", "exact", "conservative", "balanced", "aggressive"):
            self.unique.addItem(k.capitalize(), k)
            self.unique.setItemData(self.unique.count() - 1, _unique_tip(k), Qt.ItemDataRole.ToolTipRole)
        self.unique.setCurrentIndex(3)
        self.unique.setToolTip("Unique frames: duplicates are removed inside each video AND across all videos and "
                               "folders in the set (a scene repeating an hour later, or in another video, is kept "
                               "once). Exposure changes and sensor noise are ignored.")
        self.min_sharp = QDoubleSpinBox()
        self.min_sharp.setRange(0, 100000)
        self.min_sharp.setDecimals(0)
        self.min_sharp.setSpecialValueText("off")
        self.min_sharp.setToolTip("Skip blurry frames below this sharpness (Laplacian variance; the main window's "
                                  "status bar shows the current frame's value). 0 = keep all.")
        self.copy_folders = QCheckBox("Copy folder images")
        self.copy_folders.setToolTip("Copy images from folders into the test set (else they are referenced in place, "
                                     "and their labels are read from where they are).")
        for lab, wdg in (("Sampling", self.sampling), (None, self.samp_val), ("Max per video", self.max_per),
                         ("Unique frames", self.unique), ("Min sharpness", self.min_sharp)):
            if lab:
                sam.addWidget(QLabel(lab))
            sam.addWidget(wdg)
        sam.addWidget(self.copy_folders)
        sam.addStretch(1)
        tl.addLayout(sam)
        row = QHBoxLayout()
        self.build_btn = QPushButton("Add frames to the test set")
        self.build_btn.setToolTip("Sample, drop duplicates and blurry frames, save. Interrupting keeps what was added.")
        self.build_btn.clicked.connect(self._build_testset)
        self.build_bar = anim.SmoothProgressBar()
        self.build_bar.setRange(0, 1000)
        self.build_bar.setVisible(False)
        self.build_lbl = QLabel("")
        row.addWidget(self.build_btn)
        row.addWidget(self.build_bar, 1)
        row.addWidget(self.build_lbl, 2)
        tl.addLayout(row)
        lay.addWidget(self.ts_box)

        # --- dataset
        self.ds_box = QGroupBox("Labelled dataset")
        dl = QHBoxLayout(self.ds_box)
        self.ds_path = QLineEdit()
        self.ds_path.setPlaceholderText("Folder with images + YOLO / LabelMe labels, or a data.yaml")
        self.ds_path.editingFinished.connect(lambda: self._load_data(self.ds_path.text().strip()))
        self.ds_fmt = QComboBox()
        for label, k in (("Auto-detect", "auto"), ("YOLO txt", "yolo"), ("LabelMe json", "labelme")):
            self.ds_fmt.addItem(label, k)
        b1 = QPushButton("Folder…")
        b1.clicked.connect(self._browse_ds_folder)
        b2 = QPushButton("data.yaml…")
        b2.clicked.connect(self._browse_ds_yaml)
        dl.addWidget(self.ds_path, 1)
        dl.addWidget(self.ds_fmt)
        dl.addWidget(b1)
        dl.addWidget(b2)
        lay.addWidget(self.ds_box)

        # --- summary + labels
        self.sum_box = QGroupBox("What is in it")
        sl = QVBoxLayout(self.sum_box)
        self.sum_lbl = QLabel("No test data yet.")
        self.sum_lbl.setWordWrap(True)
        self.sum_lbl.setTextFormat(Qt.TextFormat.RichText)
        sl.addWidget(self.sum_lbl)
        self.preview = QListWidget()
        self.preview.setViewMode(QListWidget.ViewMode.IconMode)
        self.preview.setFlow(QListWidget.Flow.LeftToRight)
        self.preview.setWrapping(False)
        self.preview.setIconSize(QSize(144, 81))
        self.preview.setFixedHeight(112)
        self.preview.setMovement(QListWidget.Movement.Static)
        self.preview.setToolTip("A sample of the test images (evenly spread). Double-click to view.")
        self.preview.itemDoubleClicked.connect(self._preview_open)
        sl.addWidget(self.preview)
        row = QHBoxLayout()
        self.prelabel_btn = QPushButton("Pre-label with a model…")
        self.prelabel_btn.setToolTip("Write labels from a model for images that have none - then fix them in any "
                                     "labelling tool. They count as ground truth only after you have edited them "
                                     "(or marked them reviewed).")
        self.prelabel_btn.clicked.connect(self._prelabel)
        self.reviewed_btn = QPushButton("Mark pre-labels reviewed")
        self.reviewed_btn.setToolTip("Accept every pre-label as ground truth (only after checking them!).")
        self.reviewed_btn.clicked.connect(self._mark_reviewed)
        self.folder_btn = QPushButton("Open folder")
        self.folder_btn.setToolTip("Open the images folder - label it with LabelImg, X-AnyLabeling, CVAT, Label Studio…")
        self.folder_btn.clicked.connect(lambda: self.data and open_path(self.testset.images_dir if self.testset
                                                                        else self.data.root))
        self.reload_btn = QPushButton("Reload labels")
        self.reload_btn.clicked.connect(lambda: self._reload_data())
        for b in (self.prelabel_btn, self.reviewed_btn, self.folder_btn, self.reload_btn):
            row.addWidget(b)
        row.addStretch(1)
        sl.addLayout(row)
        lay.addWidget(self.sum_box)

        # --- videos
        vb = QGroupBox("Videos for behaviour analysis (optional - no labels needed)")
        vl = QVBoxLayout(vb)
        vl.addWidget(_hint("Each model runs on the whole video (a frame every N seconds): detections over time, "
                           "flicker and one-frame blips, agreement between models, and the moments they disagree."))
        self.vid_list = QListWidget()
        self.vid_list.setMaximumHeight(90)
        vl.addWidget(self.vid_list)
        row = QHBoxLayout()
        b = QPushButton("Add videos…")
        b.clicked.connect(self._add_analysis_videos)
        row.addWidget(b)
        b = QPushButton("Same as test set videos")
        b.setToolTip("Analyse the videos the test set was built from.")
        b.clicked.connect(self._videos_from_testset)
        row.addWidget(b)
        b = QPushButton("Remove")
        b.clicked.connect(lambda: ([self.vid_list.takeItem(self.vid_list.row(i)) for i in self.vid_list.selectedItems()],
                                   self._changed()))
        row.addWidget(b)
        row.addStretch(1)
        row.addWidget(QLabel("A frame every"))
        self.vid_every = QDoubleSpinBox()
        self.vid_every.setRange(0.03, 60)
        self.vid_every.setSingleStep(0.1)
        self.vid_every.setValue(0.2)
        self.vid_every.setSuffix(" s")
        self.vid_every.setToolTip("Sampling step. Flicker is measured between consecutive samples: 0.1-0.5 s works "
                                  "well; every frame is the most exact but slowest.")
        row.addWidget(self.vid_every)
        vl.addLayout(row)
        lay.addWidget(vb)
        lay.addStretch(1)
        nxt = QPushButton("Next: models  →")
        nxt.clicked.connect(lambda: self.nav.setCurrentRow(1))
        lay.addWidget(nxt, 0, Qt.AlignmentFlag.AlignRight)
        self._kind_changed()
        return _scroll(w)

    def _kind_changed(self, *_):
        self.qc_box.setVisible(self.kind_qc.isChecked())
        if self.kind_qc.isChecked():
            why = self.qc.unavailable() if self.qc is not None else "Testing was opened without Image QC."
            s = self.qc.session if self.qc is not None else None
            self.qc_lbl.setText(why or f"<b>{s.root.name}</b> · {s.type.title} · {len(s.samples):,} files")
            if not why and not self._loading and not (self.qc.is_this(self.data)) and self.job is None:
                self._load_qc()
        self.ts_box.setVisible(self.kind_ts.isChecked())
        self.ds_box.setVisible(self.kind_ds.isChecked())
        self.sum_box.setVisible(not self.kind_none.isChecked())
        self._changed()

    def _sampling_changed(self, *_):
        k = self.sampling.currentData()
        self.samp_val.setVisible(k != "all")
        if k == "interval":
            self.samp_val.setDecimals(2)
            self.samp_val.setSuffix(" s")
            self.samp_val.setValue(float(self.st.value("interval_s", 1.0)))
        elif k == "every_n":
            self.samp_val.setDecimals(0)
            self.samp_val.setSuffix(" frames")
            self.samp_val.setValue(30)
        elif k == "count":
            self.samp_val.setDecimals(0)
            self.samp_val.setSuffix(" frames")
            self.samp_val.setValue(100)

    def _ts_settings(self, save=True):
        k = self.sampling.currentData()
        s = dict(sampling=k, max_per_video=self.max_per.value(), unique=self.unique.currentData(),
                 min_sharpness=self.min_sharp.value(), copy_folders=self.copy_folders.isChecked())
        if k == "interval":
            s["interval_s"] = self.samp_val.value()
            if save:
                self.st.setValue("interval_s", s["interval_s"])
        elif k == "every_n":
            s["every_n"] = int(self.samp_val.value())
        elif k == "count":
            s["count"] = int(self.samp_val.value())
        return s

    def _new_testset(self):
        from AIO_QC.utils.save_dirs import ask_dir
        d = ask_dir(self, "Folder for the new test set (an empty folder)", kind="datasets",
                    fallback=self.st.value("last_dir", ""))
        if not d:
            return
        from AIO_QC.core.testing.testset import TestSet
        if TestSet.is_testset(d):
            self._load_data(d)
            return
        if os.listdir(d) and QMessageBox.question(
                self, "Testing", "This folder is not empty. Create the test set inside it anyway "
                "(images/, labels/, testset.json are added)?") != QMessageBox.StandardButton.Yes:
            return
        TestSet.create(d, settings=self._ts_settings())
        self.st.setValue("last_dir", os.path.dirname(d))
        self._load_data(d)

    def _open_testset(self):
        d = QFileDialog.getExistingDirectory(self, "Open a test set folder", self.st.value("last_dir", ""))
        if d:
            from AIO_QC.core.testing.testset import TestSet
            if not TestSet.is_testset(d):
                QMessageBox.information(self, "Testing", "That folder is not a test set (no testset.json). To use "
                                        "an existing labelled dataset, pick 'Labelled dataset'.")
                return
            self.st.setValue("last_dir", os.path.dirname(d))
            self._load_data(d)

    def _add_src_rows(self, rows):
        for typ, path in rows:
            r = self.src_table.rowCount()
            self.src_table.insertRow(r)
            t = _item("Video" if typ == "video" else "Folder")
            t.setData(Qt.ItemDataRole.UserRole, typ)
            self.src_table.setItem(r, 0, t)
            p = _item(path, tip=path)
            self.src_table.setItem(r, 1, p)
            rng = QTableWidgetItem("whole video" if typ == "video" else "–")
            if typ != "video":
                rng.setFlags(rng.flags() & ~Qt.ItemFlag.ItemIsEditable)
            rng.setToolTip("Optional time range, e.g. 1:00-4:30 or 60-270 (seconds)")
            self.src_table.setItem(r, 2, rng)
        self._changed()

    def _add_src_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Videos for the test set", self.st.value("video_dir", ""),
                                                VIDEO_FILTER)
        if paths:
            self.st.setValue("video_dir", os.path.dirname(paths[0]))
        self._add_src_rows([("video", p) for p in paths])

    def _add_src_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Image folder (labels are picked up when present)",
                                             self.st.value("last_dir", ""))
        if d:
            self._add_src_rows([("folder", d)])

    def _remove_src(self):
        for r in sorted({i.row() for i in self.src_table.selectedIndexes()}, reverse=True):
            self.src_table.removeRow(r)
        self._changed()

    def _build_testset(self):
        if self.job is not None:
            if self.build_btn.text() == "Stop":          # our own build: this click stops it
                self.job.cancel()
            else:
                self.status.showMessage("Busy — wait for the current task to finish", 4000)
            return
        if self.testset is None:
            self._new_testset()
            if self.testset is None:
                return
        sources = []
        for r in range(self.src_table.rowCount()):
            typ = self.src_table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            path = self.src_table.item(r, 1).text()
            src = dict(type=typ, path=path)
            if typ == "video":
                try:
                    a, b = parse_range(self.src_table.item(r, 2).text())
                except ValueError:
                    QMessageBox.warning(self, "Testing", f"Could not read the range for {os.path.basename(path)}.")
                    return
                if a is not None:
                    src["start"] = a
                if b is not None:
                    src["end"] = b
            sources.append(src)
        if not sources:
            QMessageBox.information(self, "Testing", "Add videos or image folders first.")
            return
        ts, settings = self.testset, self._ts_settings()

        def work(job):
            return ts.add_sources(sources, settings, progress=lambda d, t, txt="": job.progress.emit((d, t, txt)),
                                  cancel=job.cancelled)

        def done(stats):
            msg = (f"+{stats['added']:,} images · {stats['duplicates']:,} duplicates and {stats['blurry']:,} blurry "
                   f"skipped · {stats['seconds']:.1f} s" + (" · stopped" if stats.get("cancelled") else ""))
            if stats.get("failed"):
                msg += f" · {stats['failed']} source(s) failed"
            self.build_lbl.setText(msg)
            self.src_table.setRowCount(0)
            self._load_data(ts.root)
        self._start(work, done, bar=self.build_bar, button=self.build_btn, busy="Stop")

    def _browse_ds_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Dataset folder", self.st.value("last_dir", ""))
        if d:
            self.ds_path.setText(d)
            self._load_data(d)

    def _browse_ds_yaml(self):
        p, _ = QFileDialog.getOpenFileName(self, "YOLO data.yaml", self.st.value("last_dir", ""),
                                           "YAML (*.yaml *.yml)")
        if p:
            self.ds_path.setText(p)
            self._load_data(p)

    def _reload_data(self):
        if self.data is not None:
            self._load_data(self.testset.root if self.testset else self.ds_path.text().strip())

    def _load_qc(self, then=None):
        """The dataset open in Image QC as test data (read on the UI thread, loaded on a worker)."""
        if self.qc is None or self.job is not None:
            return
        snap = self.qc.snapshot()
        if snap is None:
            self.sum_lbl.setText(self.qc.unavailable() or "No dataset open in Image QC.")
            return
        from AIO_QC.core.testing import pipeline as P
        self.sum_lbl.setText("Reading labels…")

        def work(job):
            return P.open_qc(snap["root"], snap["fmt"], snap["pairs"], snap["names"], snap["label"],
                             progress=lambda d, t, txt="": job.progress.emit((d, t, txt)), cancel=job.cancelled)

        def done(data):
            self.data, self.testset = data, None
            self.cache = P.cache_for(data)
            self.run = None
            self.kind_qc.blockSignals(True)
            self.kind_qc.setChecked(True)
            self.kind_qc.blockSignals(False)
            self.qc_box.setVisible(True)
            self.ts_box.setVisible(False)
            self.ds_box.setVisible(False)
            self._show_summary()
            self._load_preview()
            self._changed()
            if then is not None:
                then()
        self._start(work, done, bar=self.build_bar, failed=lambda m: self.sum_lbl.setText(f"Could not read it: {m}"))

    def _load_data(self, path, then=None):
        if not path or self.job is not None:
            return
        from AIO_QC.core.testing import pipeline as P
        fmt = self.ds_fmt.currentData() if self.kind_ds.isChecked() else "auto"
        self.sum_lbl.setText("Reading labels…")

        def work(job):
            return P.open_data(path, fmt, progress=lambda d, t, txt="": job.progress.emit((d, t, txt)),
                               cancel=job.cancelled)

        def done(data):
            self.data, self.testset = data, data.testset
            self.cache = P.cache_for(data)
            self.run = None
            if data.testset:
                self.kind_ts.setChecked(True)
                self.ts_path.setText(data.testset.root)
                self.st.setValue("testset", data.testset.root)
            else:
                self.kind_ds.setChecked(True)
                self.ds_path.setText(path)
                self.st.setValue("dataset", path)
            self._show_summary()
            self._load_preview()
            self._changed()
            if then is not None:
                then()
        self._start(work, done, bar=self.build_bar, failed=lambda m: self.sum_lbl.setText(f"Could not read it: {m}"))

    def _show_summary(self):
        d = self.data
        if d is None:
            self.sum_lbl.setText("No test data yet.")
            return
        ds = d.dataset
        n = len(ds.items)
        lab = sum(1 for it in ds.items if it.labelled)
        unrev = sum(1 for it in ds.items if it.labelled and not it.reviewed)
        objs = sum(len(it.cls) for it in ds.items if it.labelled and it.reviewed)
        vids = len({it.meta.get("video") for it in ds.items if it.meta.get("video")})
        counts = {}
        for it in ds.items:
            if it.labelled and it.reviewed:
                for c in np.asarray(it.cls, int):
                    counts[int(c)] = counts.get(int(c), 0) + 1
        chips = " ".join(f"<span style='white-space:nowrap'>{ds.names.get(c, c)} <b>{counts[c]:,}</b></span> ·"
                         for c in sorted(counts, key=lambda c: -counts[c])[:24]).rstrip(" ·")
        if lab - unrev == 0:
            state = ("<b>No reviewed labels</b> - models will be compared with each other (pick a reference model "
                     "in Run), or pre-label, fix and mark reviewed to measure true accuracy.")
        else:
            state = f"<b>{lab - unrev:,}</b> labelled images, <b>{objs:,}</b> objects"
        if unrev:
            state += f" · <b>{unrev:,}</b> with unreviewed pre-labels (not scored)"
        warn = ("<br><span style='color:#c98500'>" + "<br>".join(ds.warnings[:4]) + "</span>") if ds.warnings else ""
        self.sum_lbl.setText(f"<b>{n:,}</b> images" + (f" from <b>{vids}</b> video(s)" if vids else "") +
                             f" · {len(ds.names)} classes<br>{state}" + (f"<br>{chips}" if chips else "") + warn)
        is_ts = self.testset is not None
        self.prelabel_btn.setEnabled(is_ts)
        self.reviewed_btn.setEnabled(is_ts and unrev > 0)
        self.status.showMessage(f"{d.label}: {n:,} images", 5000)

    def _load_preview(self):
        self.preview.clear()
        d = self.data
        if d is None or not d.dataset.items:
            return
        items = d.dataset.items
        pick = sorted(set(np.linspace(0, len(items) - 1, min(40, len(items))).astype(int).tolist()))
        paths = [items[i].path for i in pick]

        def work(job):
            import cv2
            out = []
            for p in paths:
                if job.cancelled():
                    break
                img = imread(p, cv2.IMREAD_REDUCED_COLOR_4)
                if img is None:
                    img = imread(p)
                out.append((p, img))
            return out

        def done(res):
            for p, img in res:
                it = QListWidgetItem(QIcon(bgr_pixmap(img, 288, 162)), "")
                it.setToolTip(p)
                it.setData(Qt.ItemDataRole.UserRole, p)
                self.preview.addItem(it)
        j = Job(work, self)
        j.done.connect(done)
        j.finished.connect(j.deleteLater)
        self._preview_job = j
        j.start()

    def _preview_open(self, it):
        p = it.data(Qt.ItemDataRole.UserRole)
        img = imread(p) if p else None
        if img is not None:
            ImagePreview(self, img, os.path.basename(p), [("Show file", lambda: open_path(os.path.dirname(p)))]).exec()

    def _add_analysis_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Videos to analyse", self.st.value("video_dir", ""), VIDEO_FILTER)
        if paths:
            self.st.setValue("video_dir", os.path.dirname(paths[0]))
        have = {self.vid_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.vid_list.count())}
        for p in paths:
            if p not in have:
                self._add_analysis_video(p)

    def _add_analysis_video(self, p, start=None, end=None):
        txt = os.path.basename(p) + (f"   [{human_time(start or 0)} – {human_time(end) if end else 'end'}]"
                                     if start or end else "")
        it = QListWidgetItem(txt)
        it.setData(Qt.ItemDataRole.UserRole, p)
        it.setData(Qt.ItemDataRole.UserRole + 1, (start, end))
        it.setToolTip(p)
        self.vid_list.addItem(it)
        self._changed()

    def _videos_from_testset(self):
        if not self.testset:
            QMessageBox.information(self, "Testing", "Open or build a test set first.")
            return
        have = {self.vid_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.vid_list.count())}
        for s in self.testset.manifest["sources"]:
            if s.get("type") == "video" and s["path"] not in have and os.path.isfile(s["path"]):
                self._add_analysis_video(s["path"], s.get("start"), s.get("end"))
                have.add(s["path"])

    def analysis_videos(self):
        out = []
        for i in range(self.vid_list.count()):
            it = self.vid_list.item(i)
            a, b = it.data(Qt.ItemDataRole.UserRole + 1) or (None, None)
            out.append(dict(path=it.data(Qt.ItemDataRole.UserRole), start=a, end=b))
        return out

    def add_video(self, path, start=None, end=None):
        """From the main window: analyse this video (optionally only in -> out)."""
        self._add_analysis_video(path, start, end)

    def _prelabel(self):
        if self.testset is None:
            return
        self._sync_specs()
        if not self.specs:
            QMessageBox.information(self, "Testing", "Add a model on the Models page first.")
            self.nav.setCurrentRow(1)
            return
        m = QMenu(self)
        for i, s in enumerate(self.specs):
            m.addAction(_swatch_icon(slot_hex(self.slots[i])), s.label).setData(i)
        act = m.exec(self.prelabel_btn.mapToGlobal(self.prelabel_btn.rect().bottomLeft()))
        if act is None:
            return
        spec = self.specs[act.data()]
        fmt = "labelme" if QMessageBox.question(
            self, "Pre-label", "Write LabelMe json (next to the images) instead of YOLO txt (labels/)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes else "yolo"
        ts, cache = self.testset, self.cache
        conf = float(self.st.value("prelabel_conf", 0.25))

        def work(job):
            from AIO_QC.core.testing import pipeline as P
            return P.prelabel(ts, spec, cache, conf=conf, fmt=fmt,
                              progress=lambda mi, d, t, ms, txt: job.progress.emit((d, t, txt)), cancel=job.cancelled)

        def done(n):
            QMessageBox.information(self, "Pre-label", f"Labels written for {n:,} images (confidence ≥ {conf:.2f}).\n\n"
                                    "Fix them in your labelling tool; edited files count as ground truth automatically. "
                                    "'Mark pre-labels reviewed' accepts the rest.")
            self._reload_data()
        self._start(work, done, bar=self.build_bar)

    def _mark_reviewed(self):
        if self.testset and QMessageBox.question(
                self, "Testing", "Accept every unreviewed pre-label as ground truth?\n\nOnly do this after checking "
                "them - otherwise a model is scored against its own predictions.") == QMessageBox.StandardButton.Yes:
            self.testset.mark_reviewed()
            self._reload_data()

    # ------------------------------------------------------------------ 2 models
    def _page_models(self):
        w, lay = self._page("Models", step=1, subtitle="Add the weights to test - two versions of the same model, different sizes, "
                            "or completely different models. Each keeps its colour in every chart, image and report.")
        self.m_table = _make_table(["", "Name", "Weights", "Runs on", "Image size", "FP16"], stretch_col=2)
        self.m_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                     QAbstractItemView.EditTrigger.EditKeyPressed)
        self.m_table.setToolTip("Double-click a name or image size to change it (0 = the size the model was "
                                "trained at).")
        self.m_table.itemChanged.connect(self._model_edited)
        lay.addWidget(self.m_table, 1)
        row = QHBoxLayout()
        for text, fn, tip in (("Add weights…", self._add_weights, ".pt / .onnx / .engine - several at once (also added "
                                                                  "to AIO QC's models)"),
                              ("Add AIO QC's models ▾", self._add_app_models, "The models of the open dataset "
                               "(Models & Environment) and every model added anywhere in AIO QC"),
                              ("Remove", self._remove_model, None), ("↑", lambda: self._move_model(-1), "Move up"),
                              ("↓", lambda: self._move_model(1), "Move down")):
            b = QPushButton(text)
            if tip:
                b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        box = QGroupBox("How .pt / .engine models run")
        f = QFormLayout(box)
        env_row = QHBoxLayout()
        self.env_edit = QLineEdit(_shared_env() or self.st.value("env_python", "") or self.model_cfg.get("env_python", ""))
        self.env_edit.editingFinished.connect(lambda: (_set_shared_env(self.env_edit.text().strip()),
                                                       self._sync_specs(), self._refresh_mapping_hint()))
        self.env_edit.setPlaceholderText("Python with ultralytics (the environment of Models & Environment)")
        self.env_edit.setToolTip(".onnx runs inside the app. .pt / .engine run in this Python environment "
                                 "(conda / venv with ultralytics + torch), on its GPU when it has one.")
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_env)
        env_row.addWidget(self.env_edit, 1)
        env_row.addWidget(b)
        f.addRow("Environment", env_row)
        self.device_edit = QComboBox()
        self.device_edit.setEditable(True)
        for d in ("auto", "cpu", "0", "1"):
            self.device_edit.addItem(d)
        self.device_edit.setCurrentText(str(self.st.value("device", "") or self.model_cfg.get("device", "auto")))
        self.device_edit.currentTextChanged.connect(lambda t: self.st.setValue("device", t.strip()))
        self.device_edit.setToolTip("auto = GPU when available. 0, 1 = CUDA device index.")
        f.addRow("Device", self.device_edit)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(int(self.st.value("batch", 8)))
        self.batch_spin.setToolTip("Images per forward pass. Larger is faster on a GPU (until memory runs out).")
        f.addRow("Batch size", self.batch_spin)
        lay.addWidget(box)
        self.map_lbl = QLabel("")
        self.map_lbl.setWordWrap(True)
        self.map_lbl.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.map_lbl)
        row = QHBoxLayout()
        self.check_btn = QPushButton("Check models")
        self.check_btn.setToolTip("Load every model once: backend, classes, and how its classes match the test data.")
        self.check_btn.clicked.connect(self._check_models)
        row.addWidget(self.check_btn)
        row.addStretch(1)
        nxt = QPushButton("Next: run  →")
        nxt.clicked.connect(lambda: self.nav.setCurrentRow(2))
        row.addWidget(nxt)
        lay.addLayout(row)
        return w

    def _browse_env(self):
        p, _ = QFileDialog.getOpenFileName(self, "Python executable of the environment", "",
                                           "Python (python python.exe python3);;All files (*)")
        if p:
            self.env_edit.setText(p)
            _set_shared_env(p)
            self._sync_specs()
            self._refresh_mapping_hint()

    def _add_weights(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Model weights", self.st.value("weights_dir", ""),
                                                WEIGHTS_FILTER)
        if paths:
            self.st.setValue("weights_dir", os.path.dirname(paths[0]))
            from AIO_QC.core import model_registry as R
            root = self.data.root if self.data is not None and self.data.qc else None
            rows = []
            for p in paths:
                e = R.register(p, root)                 # one registry: Image QC, the Frame Extractor, Testing
                rows.append(dict(path=p, label=e.get("name", ""), class_map=e.get("class_map") or {}))
            self._add_models(rows)

    def _registry_models(self):
        from AIO_QC.core import model_registry as R
        root = self.data.root if self.data is not None and self.data.qc else \
            (str(self.qc.session.root) if self.qc is not None and self.qc.session is not None else None)
        have = {os.path.normcase(os.path.abspath(self.m_table.item(r, 2).text())) for r in range(self.m_table.rowCount())}
        return [m for m in R.all_models(root) if os.path.isfile(m.get("path", "")) and
                os.path.normcase(os.path.abspath(m["path"])) not in have]

    def _add_app_models(self):
        ms = self._registry_models()
        if not ms:
            QMessageBox.information(self, "Testing", "Every model AIO QC knows is already in the list (add models in "
                                    "Image QC ▸ Models & Environment, or with Add weights…).")
            return
        m = QMenu(self)
        allb = m.addAction(f"All {len(ms)} models")
        m.addSeparator()
        for e in ms:
            a = m.addAction(f"{e.get('name') or os.path.basename(e['path'])}   ({'this dataset' if e.get('source') == 'dataset' else 'AIO QC'})")
            a.setData(e["path"])
        act = m.exec(self.m_table.mapToGlobal(self.m_table.rect().bottomLeft()))
        if act is None:
            return
        pick = ms if act is allb else [e for e in ms if e["path"] == act.data()]
        from AIO_QC.core import model_registry as R
        self._add_models([dict(path=e["path"], label=e.get("name", ""), imgsz=R.effective_imgsz(e),
                               class_map=e.get("class_map") or {}) for e in pick])

    def _add_models(self, dicts):
        have = {self.m_table.item(r, 2).text() for r in range(self.m_table.rowCount())}
        used = {self.m_table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in range(self.m_table.rowCount())}
        self.m_table.blockSignals(True)
        for d in dicts:
            p = os.path.abspath(d["path"])
            if p in have:
                continue
            slot = d.get("slot")
            if slot is None or slot in used:
                slot = next(s for s in range(64) if s not in used)
            used.add(slot)
            r = self.m_table.rowCount()
            self.m_table.insertRow(r)
            sw = _item("")
            sw.setIcon(_swatch_icon(slot_hex(slot), 14))
            sw.setData(Qt.ItemDataRole.UserRole, slot)
            self.m_table.setItem(r, 0, sw)
            name = QTableWidgetItem(d.get("label") or os.path.splitext(os.path.basename(p))[0])
            name.setData(Qt.ItemDataRole.UserRole, dict(d.get("class_map") or {}))   # AIO QC's class map
            if d.get("class_map"):
                name.setToolTip("Classes mapped to this dataset's labels (Image QC ▸ Models & Environment): " +
                                ", ".join(f"{k}→{v or 'ignored'}" for k, v in list(d["class_map"].items())[:12]))
            self.m_table.setItem(r, 1, name)
            self.m_table.setItem(r, 2, _item(p, tip=p))
            ext = os.path.splitext(p)[1].lower()
            self.m_table.setItem(r, 3, _item("in app (ONNX Runtime)" if ext == ".onnx" else "environment"))
            sz = QTableWidgetItem(str(d.get("imgsz", 0)))
            sz.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.m_table.setItem(r, 4, sz)
            half = QTableWidgetItem("")
            half.setFlags((half.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable)
            half.setCheckState(Qt.CheckState.Checked if d.get("half") else Qt.CheckState.Unchecked)
            half.setToolTip("Half precision on the GPU (.pt / .engine): faster, tiny accuracy change")
            self.m_table.setItem(r, 5, half)
            have.add(p)
        self.m_table.blockSignals(False)
        self._sync_specs()
        self._save_models()
        self._refresh_mapping_hint()

    def _remove_model(self):
        for r in sorted({i.row() for i in self.m_table.selectedIndexes()}, reverse=True):
            self.m_table.removeRow(r)
        self._sync_specs()
        self._save_models()

    def _move_model(self, d):
        rows = sorted({i.row() for i in self.m_table.selectedIndexes()})
        if len(rows) != 1:
            return
        r = rows[0]
        t = r + d
        if not 0 <= t < self.m_table.rowCount():
            return
        self.m_table.blockSignals(True)
        for c in range(self.m_table.columnCount()):
            a, b = self.m_table.takeItem(r, c), self.m_table.takeItem(t, c)
            self.m_table.setItem(r, c, b)
            self.m_table.setItem(t, c, a)
        self.m_table.blockSignals(False)
        self.m_table.selectRow(t)
        self._sync_specs()
        self._save_models()

    def _model_edited(self, _item):
        self._sync_specs()
        self._save_models()

    def _sync_specs(self):
        """Table -> ModelSpec list (+ colour slots); labels made unique."""
        from AIO_QC.core.testing import pipeline as P
        cfg = dict(self.model_cfg)
        cfg.update(env_python=self.env_edit.text().strip() if hasattr(self, "env_edit") else cfg.get("env_python", ""),
                   device=self.device_edit.currentText().strip() if hasattr(self, "device_edit") else "auto")
        specs, slots = [], []
        for r in range(self.m_table.rowCount()):
            try:
                imgsz = int(float(self.m_table.item(r, 4).text() or 0))
            except ValueError:
                imgsz = 0
            half = self.m_table.item(r, 5).checkState() == Qt.CheckState.Checked
            label = self.m_table.item(r, 1).text().strip()
            cmap = self.m_table.item(r, 1).data(Qt.ItemDataRole.UserRole) or None
            specs.append(P.spec_from_cfg(self.m_table.item(r, 2).text(), cfg, label=label, imgsz=imgsz, half=half,
                                         class_map=dict(cmap) if cmap else None))
            slots.append(int(self.m_table.item(r, 0).data(Qt.ItemDataRole.UserRole)))
        P.unique_labels(specs)
        self.specs, self.slots = specs, slots
        return specs

    def _save_models(self):
        """The model list is part of the project (autosaved)."""
        self._changed()

    def _refresh_mapping_hint(self):
        n = self.m_table.rowCount()
        pt = sum(1 for r in range(n) if not self.m_table.item(r, 2).text().lower().endswith(".onnx"))
        env = self.env_edit.text().strip()
        msg = []
        if pt and not env:
            msg.append("<span style='color:#d03b3b'>.pt / .engine models need an environment with ultralytics "
                       "(set it above or in Settings → Models).</span>")
        if self.data is not None and self.data.dataset.names:
            msg.append(f"Test data classes: {', '.join(list(self.data.dataset.names.values())[:20])}"
                       f"{'…' if len(self.data.dataset.names) > 20 else ''}. Model classes are matched by name "
                       "(order does not matter); others are ignored - 'Check models' shows the match.")
        self.map_lbl.setText("<br>".join(msg))

    def _check_models(self):
        specs = self._sync_specs()
        if not specs:
            return
        names = dict(self.data.dataset.names) if self.data is not None else {}

        def work(job):
            from AIO_QC.core.testing import metrics as M
            out = []
            for i, s in enumerate(specs):
                if job.cancelled():
                    break
                job.progress.emit((i, len(specs), f"Loading {s.label}…"))
                t0 = time.perf_counter()
                try:
                    det = s.load()
                    mn = dict(getattr(det, "names", None) or {})
                    _, matched, un = M.build_class_map(mn, names) if names else ({}, [], [])
                    out.append((s, getattr(det, "provider", ""), time.perf_counter() - t0, mn, matched, un, None))
                except Exception as e:
                    out.append((s, "", 0, {}, [], [], f"{type(e).__name__}: {e}"))
            return out

        def done(res):
            from html import escape
            lines = []
            for s, backend, secs, mn, matched, un, err in res:
                if err:
                    lines.append(f"<b>{escape(s.label)}</b>: <span style='color:#d03b3b'>{escape(err)}</span>")
                    continue
                txt = f"<b>{escape(s.label)}</b>: {escape(str(backend))}, {len(mn)} classes, loaded in {secs:.1f} s"
                if names:
                    txt += f" · {len(matched)} match the test data"
                    if un:
                        txt += f" · ignored: {escape(', '.join(map(str, un[:10])))}{'…' if len(un) > 10 else ''}"
                    if not matched:
                        txt += " · <span style='color:#d03b3b'>no class names in common - check the dataset classes</span>"
                lines.append(txt)
            self.map_lbl.setText("<br>".join(lines))
        self._start(work, done, button=self.check_btn, busy="Stop")

    # ------------------------------------------------------------------ 3 run
    def _page_run(self):
        w, lay = self._page("Run", step=2, subtitle="Every model on every test image (and analysed video). Predictions are cached: "
                            "stop any time, continue later; a model already run on these images costs nothing.")
        box = QGroupBox("Scoring")
        f = QFormLayout(box)
        self.ground = QComboBox()
        self.ground.addItem("Labels when present, else a reference model", "auto")
        self.ground.addItem("Labels (true accuracy)", "labels")
        self.ground.addItem("A reference model's detections", "reference")
        self.ground.setToolTip("Without labels, one model's detections can serve as the answer key: the others are "
                               "scored on how closely they reproduce it (useful to compare a new model with the one "
                               "in production - not true accuracy).")
        f.addRow("Ground truth", self.ground)
        self.reference = QComboBox()
        self.reference.setToolTip("Reference model (when there are no labels) and for video agreement.")
        f.addRow("Reference model", self.reference)
        self.unreviewed = QCheckBox("Also score unreviewed pre-labels")
        self.unreviewed.setToolTip("Off (recommended): images whose labels came from a model and were never edited "
                                   "are left out - they would favour that model.")
        f.addRow("", self.unreviewed)
        thr = QHBoxLayout()
        self.thr_best = QRadioButton("Best F1 per model")
        self.thr_best.setToolTip("Each model at the confidence where its precision/recall balance (F1) is best - "
                                 "the fair way to compare models.")
        self.thr_fixed = QRadioButton("Fixed")
        g = QButtonGroup(self)
        g.addButton(self.thr_best)
        g.addButton(self.thr_fixed)
        self.thr_best.setChecked(True)
        self.thr_val = QDoubleSpinBox()
        self.thr_val.setRange(0.01, 0.99)
        self.thr_val.setSingleStep(0.05)
        self.thr_val.setValue(0.25)
        thr.addWidget(self.thr_best)
        thr.addWidget(self.thr_fixed)
        thr.addWidget(self.thr_val)
        thr.addStretch(1)
        f.addRow("Confidence threshold", thr)
        lay.addWidget(box)
        row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._run_all)
        self.run_info = QLabel("")
        row.addWidget(self.run_btn)
        row.addWidget(self.run_info, 1)
        lay.addLayout(row)
        self.rows_box = QWidget()
        self.rows_lay = QVBoxLayout(self.rows_box)
        self.rows_lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.rows_box)
        self.run_bar = anim.SmoothProgressBar()
        self.run_bar.setRange(0, 1000)
        self.run_bar.setVisible(False)
        lay.addWidget(self.run_bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("Log")
        lay.addWidget(self.log, 1)
        return w

    def _refresh_run_page(self):
        specs = self._sync_specs()
        cur = self.reference.currentIndex()
        self.reference.clear()
        self.reference.addItem("(none)", -1)
        for i, s in enumerate(specs):
            self.reference.addItem(_swatch_icon(slot_hex(self.slots[i])), s.label, i)
        want = self._pending_reference
        if want is not None and self.reference.findData(want) >= 0:
            self.reference.setCurrentIndex(self.reference.findData(want))
            self._pending_reference = None
        elif 0 <= cur < self.reference.count():
            self.reference.setCurrentIndex(cur)
        elif specs:
            self.reference.setCurrentIndex(1)
        for i in reversed(range(self.rows_lay.count())):
            wdg = self.rows_lay.itemAt(i).widget()
            if wdg:
                self.rows_lay.removeWidget(wdg)
                wdg.setParent(None)
        self.prog_rows = []
        for i, s in enumerate(specs):
            pr = ProgressRow(s.label, slot_hex(self.slots[i]))
            self.rows_lay.addWidget(pr)
            self.prog_rows.append(pr)
        vids = self.analysis_videos()
        n_img = len(self.data.dataset.items) if self.data is not None and not self.kind_none.isChecked() else 0
        bits = []
        if n_img:
            bits.append(f"{n_img:,} images")
        if vids:
            bits.append(f"{len(vids)} video(s)")
        self.run_info.setText(f"{len(specs)} model(s) × " + (" + ".join(bits) if bits else "nothing yet - add test data "
                                                                                          "or videos on page 1"))
        self._revalidate()

    def _log(self, text):
        self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")

    def _run_all(self, quiet=False):
        if self.job is not None:
            if self.run_btn.text().startswith("■"):     # our own run: this click stops it
                self.job.cancel()
                self._log("Stopping… (finished predictions are kept)")
            else:                                      # e.g. the project still loading: never cancel that
                self.status.showMessage("Busy — wait for the current task to finish", 4000)
            return
        specs = self._sync_specs()
        if not specs:
            return
        if not self.prog_rows:
            self._refresh_run_page()
        for s in specs:
            if not os.path.isfile(s.path):
                QMessageBox.warning(self, "Testing", f"Weights not found:\n{s.path}")
                return
        self.st.setValue("batch", self.batch_spin.value())
        self.st.setValue("env_python", self.env_edit.text().strip())
        from AIO_QC.core.testing import pipeline as P
        data = None if self.kind_none.isChecked() else self.data
        if data is not None and not data.dataset.items:
            data = None
        vids = self.analysis_videos()
        cache = self.cache if data is not None else P.cache_for()
        self.cache = cache
        ground = self.ground.currentData()
        ref = self.reference.currentData()
        ref = None if ref is None or ref < 0 else ref
        op_conf = None if self.thr_best.isChecked() else self.thr_val.value()
        only_rev = not self.unreviewed.isChecked()
        batch = self.batch_spin.value()
        every = self.vid_every.value()
        rows = self.prog_rows
        n_models = len(specs)

        last = [0.0]

        def work(job):
            run = infos = None
            if data is not None:
                def prog(mi, done, total, ms, text):
                    now = time.monotonic()
                    if now - last[0] > 0.066 or done >= total or "loading" in text:     # <= 15 UI updates/s
                        last[0] = now
                        job.progress.emit(("img", mi, done, total, ms, text))
                run, infos = P.evaluate(data, specs, cache, only_reviewed=only_rev, ground=ground, reference=ref,
                                        op_conf=op_conf, batch=batch, progress=prog, cancel=job.cancelled,
                                        log=job.log.emit)
            out = []
            if vids and not job.cancelled():
                out = P.analyse_videos(vids, specs, cache, every_s=every, op_conf=op_conf or 0.25, reference=ref,
                                       batch=batch, cancel=job.cancelled, log=job.log.emit,
                                       progress=lambda vi, d, t, txt: job.progress.emit(("vid", vi, d, t, None, txt)))
            if run is None and out and not job.cancelled():
                # videos only: score the sampled frames (vs the reference model) so every result tab has content
                run = P.video_run(out, specs, cache, ground, ref, op_conf)
                if run is not None:
                    how = (f"vs reference {specs[run.reference].label}" if run.ground == "reference"
                           else "no reference model - plots and behaviour only")
                    job.log.emit(f"Videos: {len(run.items):,} sampled frames in the result views ({how})")
            return run, infos, out, job.cancelled()

        def progress(p):
            kind, i, done, total, ms, text = p
            if kind == "img" and i < len(rows):
                extra = f" · {ms:.1f} ms/img" if ms else ""
                rows[i].set(done, total, f"{done:,}/{total:,}{extra}" if "loading" not in text else text)
                self.run_bar.setValue(int(1000 * (i + (done / total if total else 0)) / max(1, n_models)))
            elif kind == "vid":
                self.run_bar.setValue(int(1000 * done / total) if total else 0)
                self.run_bar.setFormat(text)
                self.status.showMessage(f"Video {i + 1}/{len(vids)}: {text}")

        def done(res):
            run, infos, vout, cancelled = res
            self.run, self.infos, self.videos_out = run, infos or {}, vout
            for i, s in enumerate(specs):
                info = (infos or {}).get(s.key) or {}
                if i < len(rows):
                    if info.get("error"):
                        rows[i].set(0, 1, "failed")
                        rows[i].text.setToolTip(info["error"])
                    elif info:
                        ms = info.get("speed_ms")
                        rows[i].set(1, 1, f"done · {info.get('ran', 0):,} new" + (f" · {ms:.1f} ms/img" if ms else ""))
            secs = time.monotonic() - self._t_run
            self._log(("Stopped" if cancelled else "Finished") + f" in {secs:.1f} s")
            if run is not None and not run.scored and run.ground == "none":
                self._log("No ground truth: add labels or choose a reference model to get accuracy numbers.")
            from AIO_QC.core.testing import project as PJ
            self._run_specs_keys = [s.key for s in specs]
            self._summary = PJ.summarize(run, vout)
            self._show_results()
            self._changed()
            if self._project_path:
                PJ.add_recent(self._project_path, dict(name=self._project_name, summary=self._summary))
            if not quiet and ((run is not None and run.scored) or vout):
                self.nav.setCurrentRow(3)
        for r in rows:
            r.set(0, 1, "waiting")
        self._t_run = time.monotonic()
        self._log(("Loading cached results: " if quiet else "Run: ") + f"{n_models} model(s)"
                  + (f" on {len(data.dataset.items):,} images" if data else "") + (f" + {len(vids)} video(s)" if vids else ""))
        self._start(work, done, bar=self.run_bar, button=self.run_btn, busy="■  Stop", progress=progress)

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
        fl.addWidget(_hint("Dashed = ground truth · model colour = correct · red = error · red dashed = missed. "
                           "Double-click to fix labels or view full size; right-click to compare models or save the image."))
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

    # ---- plots: every image with a model's predictions drawn, saveable
    PLOT_SIZES = (("Small", QSize(220, 140)), ("Medium", QSize(300, 190)), ("Large", QSize(440, 270)))

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
        row.addWidget(_hint("Each test image with the model's detections drawn (labels in Inter), at the operating "
                            "threshold above. Double-click to view full size; right-click or use the buttons to save."), 1)
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
            self.pl_gallery.clear("No images with detections of this class." if self.pl_class.currentData() not in
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
            it = self.pl_gallery.add(self._loc_text(r.items[ii]) + f"\n{n} detection{'s' if n != 1 else ''}",
                                     (mi, ii), r.items[ii].path)
            jobs.append((it, ii))

        def work(job):
            import cv2
            for it, ii in jobs:
                if job.cancelled() or self._pl_epoch != epoch:
                    break
                try:
                    img = rb.plot_image(ii, mi, show_gt, errors, None, tw)     # labels sized for the shown width
                except Exception:
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
            m.addAction("Fix labels…").triggered.connect(
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
                self.f_class.addItem(str(n), c)
        self.f_class.blockSignals(False)
        cur = self.pl_class.currentData()
        self.pl_class.blockSignals(True)
        self.pl_class.clear()
        self.pl_class.addItem("All classes", -1)
        if r is not None:
            for c, n in sorted(r.names.items()):
                self.pl_class.addItem(str(n), c)
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
        if cm is not None:
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
                self.f_gallery.empty.setText(self._no_score_reason() or "No images with this kind of error.")
                self.f_gallery.show_empty(True)
            return
        mi = self.f_model.currentData()
        zoom = self.f_zoom.isChecked()
        rb = self._ink_builder(self.run)
        r = self.run
        epoch = self._f_epoch
        placeholders = []
        for score, ii, tp, fp, fn in rows:
            it = self.f_gallery.add(self._loc_text(r.items[ii]) + f"\n{fp} false · {fn} missed · {tp} correct",
                                    (mi, ii), r.items[ii].path)
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
                    except Exception:
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
        if self._can_edit(ii):
            self._open_editor(items, items.index(ii) if ii in items else 0, mi)
        else:
            self._open_plot(items, items.index(ii) if ii in items else 0, mi, True)

    def _can_edit(self, ii):
        """Labels can be fixed on test images (not on sampled video frames, which have no label files)."""
        r = self.run
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
            m.addAction("Fix labels…").triggered.connect(lambda: self._open_editor(items, k, mi))
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
                items = [x[0] for x in sorted(r.image_errors(m), key=lambda x: -(x[2] + x[3]))[:500]]
            items = items or list(r.eval_idx) or list(range(len(r.items)))
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
            self._show_compare_reference(r, a if a == r.reference else b, b if a == r.reference else a, rbld)
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
        self.cmp_tiles.setText(f"<b>{escape(rb_.name)}</b> vs <b>{escape(ra.name)}</b>: mAP50-95 "
                               f"{d(c['map_delta'])} · mAP50 {d(c['map50_delta'])}{sp} · better on "
                               f"<b>{len(c['b_better']):,}</b> images, worse on <b>{len(c['a_better']):,}</b>")
        self.cmp_grid.set_figures(figs)
        epoch = self._cmp_epoch
        jobs = []
        for (lab, lst), (title, rows) in zip(self.cmp_lists, ((f"Where {rb_.name} does better", c["b_better"]),
                                                              (f"Where {ra.name} does better", c["a_better"]))):
            lab.setText(f"<b>{escape(title)}</b> ({len(rows):,}) - left {escape(ra.name)}, right {escape(rb_.name)}. "
                        "Double-click for the side-by-side viewer.")
            lst.setProperty("items", [ii for ii, _ in rows])
            gal = self.cmp_galleries[[x[1] for x in self.cmp_lists].index(lst)]
            gal.empty.setText("No images where this model does better.")
            gal.show_empty(not rows)
            for ii, gain in rows[:24]:
                it = gal.add(self._loc_text(r.items[ii]) + f" · +{gain}", ii, r.items[ii].path)
                jobs.append((it, ii))

        def work(job):
            for it, ii in jobs:
                if job.cancelled() or self._cmp_epoch != epoch:
                    break
                try:
                    img = rbld.shot_image(ii, a, focus=False, compare=(a, b), tw=320)
                except Exception:
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
                    except Exception:
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
            names = {"disagree": "Models disagree", "flicker": "Flicker (missed for a moment)",
                     "blip": "Blip (one-off detection)"}
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
        t = _make_table(["", "Model", "Threshold", "Detections", "Frames with detections", "Tracked objects", "Flicker",
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
        t.setToolTip("Flicker: an object disappears for a frame or two and comes back. Blip: an object seen for at "
                     "most two samples. Stability: share of detections involved in neither.")
        return t

    def _event_open(self, index):
        p = self.v_pick.currentData()
        if p and index.isValid():
            row = self.v_events_model.keys[index.row()]
            self._show_moment(p, int(row[1]))

    # ------------------------------------------------------------------ 5 report
    def _page_report(self):
        w, lay = self._page("Report", step=4, subtitle="A single HTML file - charts, tables and images inside, opens in any browser, "
                            "light and dark, prints well. Data files (metrics.json, per-class and per-image CSV) "
                            "go next to it.")
        box = QGroupBox("Type")
        bl = QVBoxLayout(box)
        self.rep_btns = {}
        g = QButtonGroup(self)
        for key, title, desc in (
                ("summary", "Summary", "One page for a decision: verdict, leaderboard, precision-recall, per-class "
                                       "accuracy, error types, speed."),
                ("detailed", "Detailed", "Everything: all curves, recall by object size, class balance, and per "
                                         "model a confusion matrix, per-class table and its hardest images."),
                ("comparison", "Comparison (A vs B)", "Two models head to head: differences per class, curves, "
                                                      "confusion matrices, and the images where each one wins."),
                ("video", "Video behaviour", "Label-free: detections over time, flicker and blips, agreement "
                                             "between models, moments worth a look.")):
            rb = QRadioButton(title)
            rb.setToolTip(desc)
            g.addButton(rb)
            self.rep_btns[key] = rb
            bl.addWidget(rb)
            bl.addWidget(_hint("      " + desc))
            if key == "comparison":
                row = QHBoxLayout()
                row.addSpacing(24)
                self.rep_a = QComboBox()
                self.rep_b = QComboBox()
                row.addWidget(self.rep_a)
                row.addWidget(QLabel("vs"))
                row.addWidget(self.rep_b)
                row.addStretch(1)
                bl.addLayout(row)
        self.rep_btns["summary"].setChecked(True)
        lay.addWidget(box)
        f = QFormLayout()
        self.rep_title = QLineEdit()
        self.rep_title.setPlaceholderText("Report title (default: Model evaluation · <test data>)")
        f.addRow("Title", self.rep_title)
        row = QHBoxLayout()
        self.rep_dir = QLineEdit()
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: (lambda d: d and self.rep_dir.setText(d))(
            QFileDialog.getExistingDirectory(self, "Report folder", self.rep_dir.text())))
        row.addWidget(self.rep_dir, 1)
        row.addWidget(b)
        f.addRow("Folder", row)
        self.rep_data = QCheckBox("Also write metrics.json, per_class.csv, per_image.csv")
        self.rep_data.setChecked(True)
        f.addRow("", self.rep_data)
        lay.addLayout(f)
        row = QHBoxLayout()
        self.rep_go = QPushButton("Create report")
        self.rep_go.setMinimumHeight(32)
        self.rep_go.clicked.connect(self._make_report)
        self.rep_open = QPushButton("Open report")
        self.rep_open.clicked.connect(lambda: self._report_path and open_path(self._report_path))
        self.rep_open.setEnabled(False)
        self.rep_folder = QPushButton("Show folder")
        self.rep_folder.clicked.connect(lambda: self._report_path and open_path(os.path.dirname(self._report_path)))
        self.rep_folder.setEnabled(False)
        row.addWidget(self.rep_go)
        row.addWidget(self.rep_open)
        row.addWidget(self.rep_folder)
        row.addStretch(1)
        lay.addLayout(row)
        self.rep_status = QLabel("")
        self.rep_status.setWordWrap(True)
        lay.addWidget(self.rep_status)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        lay.addWidget(sep)
        lay.addWidget(QLabel("<b>Annotated video</b>"))
        lay.addWidget(_hint("Draw up to four models on a video (one picture or side by side) - to show a result, or "
                            "to watch models behave. Uses the cached predictions."))
        row = QHBoxLayout()
        self.ann_btn = QPushButton("Export annotated video…")
        self.ann_btn.clicked.connect(self._annotated)
        row.addWidget(self.ann_btn)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)
        return w

    def _refresh_report_page(self):
        r = self.run
        self._pair_combos(self.rep_a, self.rep_b)
        has_img = r is not None and bool(r.scored)
        can_cmp = has_img and (len(r.scored) > 1 or (r.ground == "reference" and r.reference is not None
                                                     and len(r.specs) > 1))
        for k in ("summary", "detailed", "comparison"):
            self.rep_btns[k].setEnabled(bool(has_img if k != "comparison" else can_cmp))
        for combo in (self.rep_a, self.rep_b):
            combo.setEnabled(bool(can_cmp))
        self.rep_btns["video"].setEnabled(bool(self.videos_out))
        if not any(b.isChecked() and b.isEnabled() for b in self.rep_btns.values()):
            for b in self.rep_btns.values():
                if b.isEnabled():
                    b.setChecked(True)
                    break
        if not self.rep_dir.text():
            if self.testset is not None:
                self.rep_dir.setText(self.testset.reports_dir)
            else:
                self.rep_dir.setText(self.st.value("report_dir", os.path.join(os.path.expanduser("~"), "Documents")))
        ok = has_img or bool(self.videos_out)
        self.rep_go.setEnabled(ok)
        self.rep_status.setText("" if ok else "Run the models first (page 3).")

    def _make_report(self):
        kind = next((k for k, b in self.rep_btns.items() if b.isChecked()), "summary")
        folder = self.rep_dir.text().strip() or os.getcwd()
        os.makedirs(folder, exist_ok=True)
        self.st.setValue("report_dir", folder)
        src = self.data.label if self.data is not None else (os.path.basename(self.videos_out[0][0]) if self.videos_out else "")
        title = self.rep_title.text().strip() or (
            ("Model comparison" if kind == "comparison" else "Video behaviour" if kind == "video" else "Model evaluation")
            + (f" · {src}" if src else ""))
        safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in title).strip().replace(" ", "_")[:60]
        path = os.path.join(folder, f"{safe}_{kind}_{time.strftime('%Y%m%d_%H%M')}.html")
        pair = (self.rep_a.currentData(), self.rep_b.currentData()) if kind == "comparison" else None
        if pair is not None and (None in pair or pair[0] == pair[1]):
            QMessageBox.information(self, "Report", "Pick two different models to compare.")
            return
        run, vids, slots, data_files = self.run, self.videos_out, self._run_slots(self.run, self.videos_out), \
            self.rep_data.isChecked()
        if kind == "video" and run is None:
            slots = self._run_slots(None, vids)
        version = self.version

        def work(job):
            from AIO_QC.core.testing import pipeline as P
            job.progress.emit((0, 1, "Drawing charts and images…"))
            return P.make_report(path, kind, run, vids if kind in ("video", "detailed") else [], title, slots=slots,
                                 pair=pair, exports=data_files, version=version)

        def done(paths):
            self._report_path = paths[0]
            self.rep_open.setEnabled(True)
            self.rep_folder.setEnabled(True)
            kb = os.path.getsize(paths[0]) / 1024
            self.rep_status.setText(f"Saved {os.path.basename(paths[0])} ({kb:,.0f} KB)"
                                    + (f" + {len(paths) - 1} data files" if len(paths) > 1 else ""))
            open_path(paths[0])
        self._start(work, done, button=self.rep_go, busy="Working…",
                    failed=lambda m: self.rep_status.setText(f"Failed: {m}"))

    def _annotated(self):
        specs = self._sync_specs()
        if not specs:
            QMessageBox.information(self, "Testing", "Add models first.")
            return
        vids = [v["path"] for v in self.analysis_videos()]
        if self.testset is not None:
            vids += [s["path"] for s in self.testset.manifest["sources"] if s.get("type") == "video"]
        vids = [v for i, v in enumerate(vids) if v not in vids[:i] and os.path.isfile(v)]
        video = None
        if vids:
            m = QMenu(self)
            for v in vids:
                m.addAction(os.path.basename(v)).setData(v)
            m.addSeparator()
            m.addAction("Other video…").setData("")
            act = m.exec(self.ann_btn.mapToGlobal(self.ann_btn.rect().bottomLeft()))
            if act is None:
                return
            video = act.data()
        if not video:
            video, _ = QFileDialog.getOpenFileName(self, "Video", self.st.value("video_dir", ""), VIDEO_FILTER)
        if not video:
            return
        self.st.setValue("video_dir", os.path.dirname(video))
        ff = self.ffmpeg_getter()
        if not ff:
            QMessageBox.warning(self, "Testing", "ffmpeg was not found (Settings → Playback → ffmpeg location).")
            return
        from AIO_QC.tools.frame_extractor.core.sources import open_source
        from AIO_QC.tools.frame_extractor.widgets.export_dialogs import AnnotatedExportDialog
        src = open_source(video, "auto", hw_accel=False)
        dur = src.duration()
        src.release()
        order = sorted(range(len(specs)), key=lambda i: self.slots[i])
        dlg = AnnotatedExportDialog(self, video, [specs[i] for i in order], ff, 0.0, dur, dur, cache=self.cache,
                                    conf=self.thr_val.value() if self.thr_fixed.isChecked() else 0.25,
                                    slots=[self.slots[i] for i in order])
        dlg.exec()

    # ------------------------------------------------------------------ jobs
    def _start(self, fn, done, bar=None, button=None, busy=None, failed=None, progress=None):
        if self.job is not None:
            QMessageBox.information(self, "Testing", "Something is still running - wait for it or stop it first.")
            return
        old = button.text() if button is not None else None
        job = Job(fn, self)
        self.job = job
        from AIO_QC.widgets import tasks
        what = (old or "").strip(" ▶■") or self.status.currentMessage() or "working"
        task = tasks.start(f"Testing · {what}", cancel=job.cancel)
        if bar is not None:
            bar.setVisible(True)
            bar.setValue(0)
            bar.setFormat("")
        if button is not None and busy:
            button.setText(busy)

        def prog(p):
            if isinstance(p, tuple) and len(p) >= 3 and isinstance(p[0], (int, float)) and \
                    isinstance(p[1], (int, float)):
                task.update(p[0] / p[1] if p[1] else None, p[2] if isinstance(p[2], str) else None)
            elif isinstance(p, tuple) and len(p) >= 4 and isinstance(p[2], (int, float)) and \
                    isinstance(p[3], (int, float)):
                task.update(p[2] / p[3] if p[3] else None)
            if progress is not None:
                progress(p)
                return
            if bar is not None and isinstance(p, tuple) and len(p) >= 3:
                d, t, text = p[0], p[1], p[2]
                bar.setValue(int(1000 * d / t) if t else 0)
                if text:
                    self.status.showMessage(str(text))
        state = {"clean": False}

        def cleanup():
            # runs once, BEFORE the result handler: a handler may start the next job (e.g. reload after build)
            if state["clean"]:
                return
            state["clean"] = True
            task.finish()
            if self.job is job:
                self.job = None
            if bar is not None:
                bar.setVisible(False)
            if button is not None and old is not None:
                button.setText(old)
            self.status.clearMessage()
            self._revalidate()

        def on_done(r):
            cleanup()
            done(r)

        def on_failed(m):
            cleanup()
            if failed is not None:
                failed(m)
            else:
                self._log(f"Error: {m}")
                QMessageBox.warning(self, "Testing", m)
        job.progress.connect(prog)
        job.log.connect(self._log)
        job.done.connect(on_done)
        job.failed.connect(on_failed)
        job.finished.connect(cleanup)
        job.start()

    # ------------------------------------------------------------------ persistence
    # ------------------------------------------------------------------ projects & start page
    # ------------------------------------------------------------------ 6 review videos
    def _page_review(self):
        w, lay = self._page(
            "Review videos", step=None,
            subtitle="Play videos in the Frame Extractor with every model's detections drawn on them - one after another. "
                     "Testing only sends the playlist: the player analyses ahead of the playhead, keeps playback smooth "
                     "and shows the boxes of the exact frame on screen. Detections are saved in the prediction cache.")
        self.rv_table = _make_table(["Video", "In", "Out", "Length"], stretch_col=0)
        self.rv_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                      QAbstractItemView.EditTrigger.EditKeyPressed)
        self.rv_table.setToolTip("Double-click In / Out to play only part of a video (1:30, 90, 0:01:30.5).")
        self.rv_table.itemChanged.connect(self._review_edited)
        lay.addWidget(self.rv_table, 2)
        row = QHBoxLayout()
        for text, fn, tip in (("Add videos…", self._review_add_videos, "Several at once"),
                              ("Add folder…", self._review_add_folder, "Every video in a folder (and its sub-folders)"),
                              ("Use step 1's videos", self._review_from_step1, "The videos for behaviour analysis"),
                              ("Remove", self._review_remove, None),
                              ("↑", lambda: self._review_move(-1), "Play earlier"),
                              ("↓", lambda: self._review_move(1), "Play later")):
            b = QPushButton(text)
            if tip:
                b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        mbox = QGroupBox("Models (from step 2)")
        mv = QVBoxLayout(mbox)
        self.rv_models = QLabel("")
        self.rv_models.setWordWrap(True)
        self.rv_models.setTextFormat(Qt.TextFormat.RichText)
        self.rv_models.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        mv.addWidget(self.rv_models)
        mrow = QHBoxLayout()
        b = QPushButton("Edit models…")
        b.clicked.connect(lambda: self.nav.setCurrentRow(1))
        mrow.addWidget(b)
        b = QPushButton("Refresh details")
        b.clicked.connect(lambda: self._refresh_review_page(force=True))
        mrow.addWidget(b)
        mrow.addStretch(1)
        mv.addLayout(mrow)
        lay.addWidget(mbox)
        obox = QGroupBox("Analysis")
        f = QFormLayout(obox)
        self.rv_mode = QComboBox()
        for key, text in (("balanced", "Balanced - smooth real-time playback (recommended)"),
                          ("best", "Best quality - every frame; playback waits when needed"),
                          ("fixed", "Fixed step - every N-th frame, interpolated in between")):
            self.rv_mode.addItem(text, key)
        self.rv_mode.setToolTip("Balanced: when the models keep up, every frame is analysed live; when they can't, "
                                "the frame step adapts, frames in between are interpolated from the analysed frames on "
                                "both sides, and spare time fills them in afterwards.")
        self.rv_stride = QSpinBox()
        self.rv_stride.setRange(2, 30)
        self.rv_stride.setSuffix(" (every N-th frame)")
        self.rv_lead = QDoubleSpinBox()
        self.rv_lead.setRange(0.5, 30)
        self.rv_lead.setSingleStep(0.5)
        self.rv_lead.setSuffix(" s")
        self.rv_lead.setToolTip("How far ahead of the playhead detections are kept ready")
        self.rv_conf = QDoubleSpinBox()
        self.rv_conf.setRange(0.01, 0.99)
        self.rv_conf.setSingleStep(0.05)
        self.rv_conf.setToolTip("Boxes below this are hidden at first - a slider in the player changes it live")
        self.rv_loop = QCheckBox("Start again after the last video")
        from AIO_QC.widgets.settings_pages import review_defaults
        dflt = review_defaults()
        self.rv_mode.setCurrentIndex(max(0, self.rv_mode.findData(dflt["mode"])))
        self.rv_stride.setValue(dflt["stride"])
        self.rv_lead.setValue(dflt["lead_s"])
        self.rv_conf.setValue(dflt["conf"])
        self.rv_mode.currentIndexChanged.connect(lambda _i: (self.rv_stride.setEnabled(
            self.rv_mode.currentData() == "fixed"), self._changed()))
        self.rv_stride.setEnabled(self.rv_mode.currentData() == "fixed")
        for x in (self.rv_stride, self.rv_lead, self.rv_conf):
            x.valueChanged.connect(lambda _v: self._changed())
        self.rv_loop.toggled.connect(lambda _on: self._changed())
        f.addRow("Mode", self.rv_mode)
        f.addRow("Step", self.rv_stride)
        f.addRow("Look-ahead", self.rv_lead)
        f.addRow("Show boxes from confidence", self.rv_conf)
        f.addRow("", self.rv_loop)
        lay.addWidget(obox)
        row = QHBoxLayout()
        self.rv_hint = QLabel("")
        self.rv_hint.setWordWrap(True)
        row.addWidget(self.rv_hint, 1)
        self.rv_go = QPushButton("▶  Review in player")
        self.rv_go.setObjectName("review_in_player")
        self.rv_go.setToolTip("Switches to the Frame Extractor and plays the videos with the detections")
        self.rv_go.clicked.connect(self._review_go)
        row.addWidget(self.rv_go)
        lay.addLayout(row)
        self._rv_info = {}                          # model path -> describe() result
        return w

    def _review_videos(self):
        from AIO_QC.tools.frame_extractor.core.sources import parse_time
        out = []
        for r in range(self.rv_table.rowCount()):
            path = self.rv_table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            vals = []
            for c in (1, 2):
                t = (self.rv_table.item(r, c).text() or "").strip()
                try:
                    vals.append(parse_time(t) if t else None)
                except ValueError:
                    vals.append(None)
            out.append(dict(path=path, start=vals[0], end=vals[1]))
        return out

    def _review_add(self, path, start=None, end=None):
        from AIO_QC.tools.frame_extractor.core.sources import format_time
        have = {self.rv_table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in range(self.rv_table.rowCount())}
        if path in have:
            return
        self.rv_table.blockSignals(True)
        r = self.rv_table.rowCount()
        self.rv_table.insertRow(r)
        it = QTableWidgetItem(os.path.basename(path))
        it.setData(Qt.ItemDataRole.UserRole, path)
        it.setToolTip(path)
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.rv_table.setItem(r, 0, it)
        self.rv_table.setItem(r, 1, QTableWidgetItem(format_time(start) if start else ""))
        self.rv_table.setItem(r, 2, QTableWidgetItem(format_time(end) if end else ""))
        ln = QTableWidgetItem("")
        ln.setFlags(ln.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.rv_table.setItem(r, 3, ln)
        self.rv_table.blockSignals(False)
        self._review_probe(path)
        self._changed()

    def _review_probe(self, path):
        """Length and frame rate, off the UI thread."""
        def fn():
            from AIO_QC.tools.frame_extractor.core.sources import open_source, format_time
            src = open_source(path, "auto", hw_accel=False)
            try:
                return f"{format_time(src.duration())[:-4]} · {src.fps:.2f} fps · {src.frame_count:,} frames"
            finally:
                src.release()

        def ok(text):
            for r in range(self.rv_table.rowCount()):
                if self.rv_table.item(r, 0).data(Qt.ItemDataRole.UserRole) == path:
                    self.rv_table.blockSignals(True)
                    self.rv_table.item(r, 3).setText(text)
                    self.rv_table.blockSignals(False)

        def bad(msg):
            ok(f"can't open: {msg}")
        self.views.submit(f"rv-probe:{path}", fn, ok, bad)

    def _review_add_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Videos to review", self.st.value("video_dir", ""), VIDEO_FILTER)
        if paths:
            self.st.setValue("video_dir", os.path.dirname(paths[0]))
        for p in paths:
            self._review_add(p)
        self._refresh_review_page()

    def _review_add_folder(self):
        from AIO_QC.tools.frame_extractor.core.common import VIDEO_EXTS
        d = QFileDialog.getExistingDirectory(self, "Folder with videos", self.st.value("video_dir", ""))
        if not d:
            return
        found = []
        for root, _dirs, files in os.walk(d):
            found += [os.path.join(root, f) for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
        for p in sorted(found):
            self._review_add(p)
        self._refresh_review_page()

    def _review_from_step1(self):
        for v in self.analysis_videos():
            if os.path.isfile(v["path"]):
                self._review_add(v["path"], v.get("start"), v.get("end"))
        self._refresh_review_page()

    def _review_remove(self):
        rows = sorted({i.row() for i in self.rv_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.rv_table.removeRow(r)
        self._changed()
        self._refresh_review_page()

    def _review_move(self, d):
        r = self.rv_table.currentRow()
        j = r + d
        if r < 0 or not 0 <= j < self.rv_table.rowCount():
            return
        self.rv_table.blockSignals(True)
        for c in range(self.rv_table.columnCount()):
            a, b = self.rv_table.takeItem(r, c), self.rv_table.takeItem(j, c)
            self.rv_table.setItem(r, c, b)
            self.rv_table.setItem(j, c, a)
        self.rv_table.blockSignals(False)
        self.rv_table.setCurrentCell(j, 0)
        self._changed()

    def _review_edited(self, item):
        self._changed()
        self._refresh_review_page()

    def _review_state(self):
        return dict(videos=self._review_videos(), mode=self.rv_mode.currentData(), stride=self.rv_stride.value(),
                    lead_s=self.rv_lead.value(), conf=self.rv_conf.value(), loop=self.rv_loop.isChecked())

    def _apply_review_state(self, st):
        self.rv_table.setRowCount(0)
        for v in st.get("videos", []):
            if v.get("path"):
                self._review_add(v["path"], v.get("start"), v.get("end"))
        if st.get("mode"):
            self.rv_mode.setCurrentIndex(max(0, self.rv_mode.findData(st["mode"])))
        for x, k in ((self.rv_stride, "stride"), (self.rv_lead, "lead_s"), (self.rv_conf, "conf")):
            if k in st:
                x.setValue(st[k])
        self.rv_loop.setChecked(bool(st.get("loop")))

    def _refresh_review_page(self, force=False):
        """Model details (read off the UI thread, once per file) and what still blocks the review."""
        self._sync_specs()
        if force:
            self._rv_info = {}
        todo = [s for s in self.specs if s.path not in self._rv_info]
        for s in todo:
            self._rv_info[s.path] = None
            from AIO_QC.core.testing.review import describe

            def ok(info, path=s.path):
                self._rv_info[path] = info
                self._show_review_models()

            def bad(msg, path=s.path, label=s.label):
                self._rv_info[path] = dict(label=label, error=msg)
                self._show_review_models()
            self.views.submit(f"rv-model:{s.path}", lambda s=s: describe(s), ok, bad)
        self._show_review_models()

    def _show_review_models(self):
        from AIO_QC.core.testing.review import describe_text
        if not self.specs:
            self.rv_models.setText("<i>No models yet - add weights in step 2.</i>")
        else:
            cards = []
            for i, s in enumerate(self.specs):
                col = slot_hex(self.slots[i]) if i < len(self.slots) else "#888"
                info = self._rv_info.get(s.path)
                body = describe_text(info) if info else f"<b>{s.label}</b> <i>reading…</i>"
                cards.append(f"<table><tr><td style='background:{col}' width='6'></td>"
                             f"<td style='padding-left:6px'>{body}</td></tr></table>")
            self.rv_models.setText("".join(cards))
        vids = self._review_videos()
        problems = []
        if not vids:
            problems.append("add videos")
        if not self.specs:
            problems.append("add models (step 2)")
        bad = [os.path.basename(v["path"]) for v in vids if not os.path.isfile(v["path"])]
        if bad:
            problems.append("missing: " + ", ".join(bad[:3]))
        err = [i["label"] for i in self._rv_info.values() if i and i.get("error")]
        shell = getattr(QApplication.instance(), "shell", None)
        if shell is None:
            problems.append("the Frame Extractor is available when AIO QC is started normally")
        self.rv_go.setEnabled(not problems)
        self.rv_hint.setText(("<span style='color:#d03b3b'>" + " · ".join(problems) + "</span>") if problems else
                             f"{len(vids)} video(s) · {len(self.specs)} model(s)"
                             + (f" · <span style='color:#c77700'>{len(err)} can't be read: "
                                f"{', '.join(err)}</span>" if err else ""))
        it = self.nav.item(5)
        if it is not None:
            it.setIcon(_status_icon("ready" if not problems else "todo"))

    def _review_go(self):
        from AIO_QC.core.testing.review import ReviewItem, ReviewPlaylist
        self._sync_specs()
        pl = ReviewPlaylist(
            items=[ReviewItem(v["path"], v["start"], v["end"]) for v in self._review_videos()],
            specs=list(self.specs), colors=[slot_hex(sl) for sl in self.slots], mode=self.rv_mode.currentData(),
            stride=self.rv_stride.value(), lead_s=self.rv_lead.value(), conf=self.rv_conf.value(),
            loop=self.rv_loop.isChecked(), cache_path=getattr(self.cache, "path", "") or "",
            source=f"Testing · {self._project_name}")
        shell = getattr(QApplication.instance(), "shell", None)
        if shell is None:
            QMessageBox.information(self, "Review in player", "The Frame Extractor is available when AIO QC is started "
                                                             "normally.")
            return
        shell.review_in_player(pl)

    def _build_menu(self):
        mb = self.menuBar()
        m = mb.addMenu("Project")
        self.act_new = m.addAction("New project", self._new_project)
        self.act_new.setShortcut(QKeySequence("Ctrl+N"))
        self.act_open = m.addAction("Open project…", self._open_project_dialog)
        self.act_open.setShortcut(QKeySequence("Ctrl+O"))
        self.recent_menu = m.addMenu("Open recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        m.addSeparator()
        self.act_save = m.addAction("Save project", self._save_project)
        self.act_save.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save_as = m.addAction("Save project as…", lambda: self._save_project(ask=True))
        self.act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        m.addSeparator()
        m.addAction("Start page", self._show_home)
        m.addSeparator()
        m.addAction("Close", self.close).setShortcut(QKeySequence("Ctrl+W"))
        h = mb.addMenu("Help")
        h.addAction("Testing guide", lambda: self.help_requested.emit("09-model-lab.html"))
        h.addAction("Metrics explained", lambda: self.help_requested.emit("13-metrics.html"))

    def _page_home(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(28, 22, 28, 18)
        lay.setSpacing(12)
        t = QLabel("<span style='font-size:20pt; font-weight:600'>Testing</span>")
        lay.addWidget(t)
        lay.addWidget(_hint("Measure and compare detection models on your own footage. A project remembers the test "
                            "data, models and settings - reopen it next month to test a new model on exactly the "
                            "same frames. Tip: drop videos, folders, data.yaml or weights anywhere on this window."))
        row = QHBoxLayout()
        for text, fn, tip in (("New project", self._new_project, "Start from scratch (Ctrl+N)"),
                              ("Open project…", self._open_project_dialog, "Open a .fxlab file (Ctrl+O)"),
                              ("Continue last session", self._continue_last, "Everything as you left it")):
            b = QPushButton(text)
            b.setMinimumHeight(34)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
            if text.startswith("Continue"):
                self.continue_btn = b
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(QLabel("<b>Recent projects</b>"))
        self.recent_list = QListWidget()
        self.recent_list.setSpacing(2)
        self.recent_list.itemActivated.connect(lambda it: self._open_project(it.data(Qt.ItemDataRole.UserRole)))
        self.recent_list.itemDoubleClicked.connect(lambda it: self._open_project(it.data(Qt.ItemDataRole.UserRole)))
        self.recent_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.recent_list.customContextMenuRequested.connect(self._recent_menu_at)
        lay.addWidget(self.recent_list, 1)
        steps = QLabel("<b>How it works</b> &nbsp; 1 Test data → 2 Models → 3 Run → 4 Results → 5 Report")
        steps.setEnabled(False)
        lay.addWidget(steps)
        return w

    def _fill_recent(self):
        from AIO_QC.core.testing import project as PJ
        self.recent_list.clear()
        rows = PJ.recent()
        for r in rows:
            s = r.get("summary") or {}
            when = time.strftime("%d %b %Y %H:%M", time.localtime(r.get("saved") or 0))
            res = (f"best {s['best']} · mAP50-95 {s['map5095']:.3f} · {s.get('images', 0):,} images"
                   if s.get("best") else "no results yet")
            it = QListWidgetItem(f"{r.get('name')}\n{res} · saved {when}\n{r['path']}")
            it.setData(Qt.ItemDataRole.UserRole, r["path"])
            it.setToolTip(r["path"])
            self.recent_list.addItem(it)
        if not rows:
            it = QListWidgetItem("No projects yet - start a new one, or just begin on the Test data page "
                                 "(your work is kept automatically).")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent_list.addItem(it)
        self.continue_btn.setEnabled(os.path.isfile(PJ.AUTOSAVE))

    def _fill_recent_menu(self):
        from AIO_QC.core.testing import project as PJ
        self.recent_menu.clear()
        for r in PJ.recent():
            self.recent_menu.addAction(r.get("name") or os.path.basename(r["path"]),
                                       lambda p=r["path"]: self._open_project(p))
        if self.recent_menu.isEmpty():
            self.recent_menu.addAction("(none)").setEnabled(False)

    def _recent_menu_at(self, pos):
        it = self.recent_list.itemAt(pos)
        p = it.data(Qt.ItemDataRole.UserRole) if it else None
        if not p:
            return
        from AIO_QC.core.testing import project as PJ
        m = QMenu(self)
        m.addAction("Open", lambda: self._open_project(p))
        m.addAction("Show in folder", lambda: open_path(os.path.dirname(p)))
        m.addAction("Remove from list", lambda: (PJ.remove_recent(p), self._fill_recent()))
        m.exec(self.recent_list.viewport().mapToGlobal(pos))

    def _show_home(self):
        self._fill_recent()
        self.nav.blockSignals(True)
        self.nav.clearSelection()
        self.nav.setCurrentRow(-1)
        self.nav.blockSignals(False)
        self.pages.setCurrentIndex(self.HOME)

    def _collect_state(self):
        kind = "qc" if self.kind_qc.isChecked() else "testset" if self.kind_ts.isChecked() else \
            "dataset" if self.kind_ds.isChecked() else "none"
        path = (self.testset.root if self.testset is not None else self.ts_path.text()) if kind == "testset" \
            else self.ds_path.text().strip() if kind == "dataset" \
            else (self.data.root if self.data is not None and self.data.qc else "") if kind == "qc" else ""
        models = []
        for r in range(self.m_table.rowCount()):
            models.append(dict(path=self.m_table.item(r, 2).text(), label=self.m_table.item(r, 1).text(),
                               imgsz=self.m_table.item(r, 4).text(),
                               half=self.m_table.item(r, 5).checkState() == Qt.CheckState.Checked,
                               slot=self.m_table.item(r, 0).data(Qt.ItemDataRole.UserRole),
                               class_map=self.m_table.item(r, 1).data(Qt.ItemDataRole.UserRole) or {}))
        return dict(name=self._project_name, data=dict(kind=kind, path=path, fmt=self.ds_fmt.currentData()),
                    videos=self.analysis_videos(), every_s=self.vid_every.value(), models=models,
                    env=self.env_edit.text().strip(), device=self.device_edit.currentText().strip(),
                    batch=self.batch_spin.value(), sampling=self._ts_settings(save=False),
                    run=dict(ground=self.ground.currentData(), reference=self.reference.currentData(),
                             unreviewed=self.unreviewed.isChecked(), best=self.thr_best.isChecked(),
                             thr=self.thr_val.value()),
                    report=dict(dir=self.rep_dir.text(), title=self.rep_title.text()),
                    review=self._review_state(), summary=self._summary or {})

    def _apply_state(self, st):
        self._loading = True
        try:
            self._project_name = st.get("name") or "Untitled"
            self._summary = st.get("summary") or {}
            self.m_table.setRowCount(0)
            self._add_models([m for m in st.get("models", []) if m.get("path")])
            self.env_edit.setText(_shared_env() or st.get("env", "") or self.model_cfg.get("env_python", ""))
            if st.get("device"):
                self.device_edit.setCurrentText(st["device"])
            self.batch_spin.setValue(int(st.get("batch", 8)))
            self.vid_list.clear()
            for v in st.get("videos", []):
                if v.get("path"):
                    self._add_analysis_video(v["path"], v.get("start"), v.get("end"))
            self.vid_every.setValue(float(st.get("every_s", 0.2)))
            run = st.get("run") or {}
            k = self.ground.findData(run.get("ground", "auto"))
            self.ground.setCurrentIndex(max(0, k))
            self.unreviewed.setChecked(bool(run.get("unreviewed")))
            (self.thr_best if run.get("best", True) else self.thr_fixed).setChecked(True)
            self.thr_val.setValue(float(run.get("thr", 0.25)))
            self._pending_reference = run.get("reference")
            self._apply_review_state(st.get("review") or {})
            rep = st.get("report") or {}
            self.rep_dir.setText(rep.get("dir", ""))
            self.rep_title.setText(rep.get("title", ""))
            self.run, self.videos_out, self.infos = None, [], {}
            self.data = self.testset = None
            self.ts_path.setText("")
            self.ds_path.setText("")
            self.src_table.setRowCount(0)
            d = st.get("data") or {}
            kind = d.get("kind", "testset")
            qc_ok = kind == "qc" and self.qc is not None and not self.qc.unavailable() and \
                os.path.normcase(os.path.abspath(str(self.qc.session.root))) == \
                os.path.normcase(os.path.abspath(d.get("path") or ""))
            if kind == "qc" and not qc_ok:          # made on a dataset that is not open now: read it as a dataset
                kind = "dataset"
                d = dict(d, fmt=d.get("fmt") or "auto")
                self.ds_path.setText(d.get("path") or "")
            {"qc": self.kind_qc, "testset": self.kind_ts, "dataset": self.kind_ds,
             "none": self.kind_none}.get(kind, self.kind_ts).setChecked(True)
            if d.get("fmt"):
                self.ds_fmt.setCurrentIndex(max(0, self.ds_fmt.findData(d["fmt"])))
            self._show_summary()
            self._show_results()
        finally:
            self._loading = False
        self._dirty = False
        self._update_title()
        path = d.get("path")
        if kind == "qc":
            self._load_qc(then=self._load_cached_results)
        elif kind != "none" and path and os.path.exists(path):
            self._load_data(path, then=self._load_cached_results)
        else:
            self._load_cached_results()
        self._revalidate()

    def _new_project(self):
        if not self._confirm_discard():
            return
        self._project_path = None
        self._apply_state(dict(name="Untitled", models=[dict(path=p) for p in self.model_cfg.get("models", [])
                                                         if os.path.isfile(p)]))
        self.nav.setCurrentRow(0)

    def _open_project_dialog(self):
        from AIO_QC.core.testing import project as PJ
        p, _ = QFileDialog.getOpenFileName(self, "Open Testing project", self.st.value("project_dir", ""),
                                           f"Testing project (*{PJ.EXT});;All files (*)")
        if p:
            self._open_project(p)

    def _open_project(self, path):
        from AIO_QC.core.testing import project as PJ
        if not path or not self._confirm_discard():
            return
        try:
            st = PJ.load(path)
        except Exception as e:
            QMessageBox.warning(self, "Testing", f"Could not open the project:\n{e}")
            return
        self._project_path = path
        self.st.setValue("project_dir", os.path.dirname(path))
        PJ.add_recent(path, st)
        self._apply_state(st)
        self.nav.setCurrentRow(3 if self._summary.get("best") else 0)

    def _continue_last(self):
        from AIO_QC.core.testing import project as PJ
        if not self._dirty and (self.data is not None or self.specs):     # the last session is already open
            self.nav.setCurrentRow(3 if (self.run is not None or self.videos_out) else 0)
            return
        if os.path.isfile(PJ.AUTOSAVE):
            try:
                st = PJ.load(PJ.AUTOSAVE)
            except Exception:
                return
            self._project_path = st.get("project_path") if st.get("project_path") and \
                os.path.isfile(st.get("project_path")) else None
            self._apply_state(st)
            self.nav.setCurrentRow(3 if self._summary.get("best") else 0)

    def _save_project(self, ask=False):
        from AIO_QC.core.testing import project as PJ
        path = self._project_path
        if ask or not path:
            base = self.testset.root if self.testset is not None else self.st.value("project_dir", "")
            name = (self._project_name if self._project_name != "Untitled" else "model_lab") + PJ.EXT
            from AIO_QC.utils.save_dirs import ask_save
            path, _ = ask_save(self, "Save Testing project", name, f"Testing project (*{PJ.EXT})", kind="exports",
                               fallback=base or None)
            if not path:
                return False
            if not path.endswith(PJ.EXT):
                path += PJ.EXT
            self._project_name = os.path.splitext(os.path.basename(path))[0]
        st = self._collect_state()
        PJ.save(path, st)
        PJ.add_recent(path, st)
        self._project_path = path
        self.st.setValue("project_dir", os.path.dirname(path))
        self._dirty = False
        self._update_title()
        self._autosave()
        self.status.showMessage(f"Saved {path}", 4000)
        return True

    def _confirm_discard(self):
        if not self._dirty or self._project_path is None:
            return True                     # unnamed sessions live on in the autosave
        r = QMessageBox.question(self, "Testing", f"Save changes to “{self._project_name}”?",
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                 QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if r == QMessageBox.StandardButton.Save:
            return self._save_project()
        return r == QMessageBox.StandardButton.Discard

    def _autosave(self):
        if self._loading:
            return
        from AIO_QC.core.testing import project as PJ
        try:
            st = self._collect_state()
            st["project_path"] = self._project_path
            PJ.save(PJ.AUTOSAVE, st)
            if self.data is not None and self.data.qc:             # part of the dataset's AIO QC session
                PJ.save(PJ.dataset_project(self.data.root), st)
        except Exception as e:
            self._log(f"Autosave failed: {e}")

    def _changed(self, *_):
        if self._loading:
            return
        if not self._dirty:
            self._dirty = True
            self._update_title()
        self._autosave_timer.start()
        self._revalidate()

    def _update_title(self):
        name = self._project_name or "Untitled"
        self.setWindowTitle(f"Testing — {name}{' •' if self._dirty else ''}")

    def _watch_changes(self):
        """Every setting marks the project changed (title •, autosave, step checks)."""
        for w in self.findChildren(QComboBox):
            w.currentIndexChanged.connect(self._changed)
        for w in self.findChildren(QDoubleSpinBox) + self.findChildren(QSpinBox):
            w.valueChanged.connect(self._changed)
        for w in self.findChildren(QCheckBox) + self.findChildren(QRadioButton):
            w.toggled.connect(self._changed)
        for w in (self.env_edit, self.rep_title, self.rep_dir, self.ds_path):
            w.textChanged.connect(self._changed)

    def _load_cached_results(self):
        """Opening a project: if every prediction is already cached, show the results right away (seconds)."""
        specs = self._sync_specs()
        data = None if self.kind_none.isChecked() else self.data
        vids = self.analysis_videos()
        if not specs or (data is None and not vids):
            return
        from AIO_QC.core.testing import pipeline as P
        cache = self.cache if data is not None and self.cache is not None else P.cache_for()

        def ready():
            if data is not None and data.dataset.items:
                keys = set(data.keys)
                for s in specs:
                    if not keys <= cache.keys_for(s.key):
                        return False
            for v in vids:
                for s in specs:
                    if not cache.keys_for(P.video_spec(s).key):
                        return False
            return True
        self._bg("cached?", ready, lambda ok: ok and self.job is None and self._run_all(quiet=True))

    # ------------------------------------------------------------------ step checks
    def _revalidate(self):
        self._valid_timer.start()

    def _validate(self):
        """{step: [(level, text, fix label, fix fn)]} - cheap checks only (files, settings)."""
        out = {i: [] for i in range(5)}
        d = out[0]
        vids = self.analysis_videos()
        missing_v = [v["path"] for v in vids if not os.path.isfile(v["path"])]
        if self.kind_none.isChecked():
            if not vids:
                d.append(("error", "Add at least one video to analyse (or choose test images).", "Add videos…",
                          self._add_analysis_videos))
        elif self.data is None:
            if self.kind_ts.isChecked():
                d.append(("error", "No test set yet: create one, then add videos or image folders to it.",
                          "New test set…", self._new_testset))
            else:
                d.append(("error", "Choose a labelled folder or a data.yaml.", "Folder…", self._browse_ds_folder))
        else:
            ds = self.data.dataset
            if not ds.items:
                d.append(("error", "The test set has no images yet: add videos or folders above and press "
                                   "<b>Add frames to the test set</b>.", "Add videos…", self._add_src_videos))
            else:
                reviewed = sum(1 for it in ds.items if it.labelled and it.reviewed)
                unrev = sum(1 for it in ds.items if it.labelled and not it.reviewed)
                if not reviewed:
                    d.append(("warn", "No reviewed labels: models will be compared with a reference model instead of "
                                      "true accuracy.", "Pre-label…" if self.testset is not None else None,
                              self._prelabel if self.testset is not None else None))
                if unrev:
                    d.append(("warn", f"{unrev:,} image(s) have pre-labels nobody reviewed - they are not scored.",
                              "Review…", lambda: self._review_prelabels()))
        if missing_v:
            d.append(("error", f"{len(missing_v)} video(s) not found: {', '.join(os.path.basename(p) for p in missing_v[:3])}",
                      None, None))
        m = out[1]
        specs = self.specs
        if not specs:
            m.append(("error", "Add the weights you want to test.", "Add weights…", self._add_weights))
        miss = [s for s in specs if not os.path.isfile(s.path)]
        if miss:
            m.append(("error", f"Weights not found: {', '.join(os.path.basename(s.path) for s in miss[:3])}", None, None))
        if any(not s.path.lower().endswith(".onnx") for s in specs) and not self.env_edit.text().strip():
            m.append(("error", ".pt / .engine models need a Python environment with ultralytics.", "Choose…",
                      self._browse_env))
        r = out[2]
        if any(lv == "error" for lv, *_ in out[0] + out[1]):
            r.append(("error", "Fix the problems on the Test data and Models pages first.", None, None))
        elif self.data is not None and self.data.dataset.items and not any(
                it.labelled and it.reviewed for it in self.data.dataset.items) and len(specs) < 2:
            r.append(("warn", "No labels and a single model: there is nothing to score against (add labels or a "
                              "second model).", None, None))
        elif self.ground.currentData() == "reference" or (
                self.data is not None and not any(it.labelled and it.reviewed for it in self.data.dataset.items)):
            r.append(("info", "Scores will be relative to the reference model - not true accuracy.", None, None))
        if self.run is not None and self._run_specs_keys and \
                self._run_specs_keys != [s.key for s in specs]:
            r.append(("warn", "Models changed since the last run.", "Run again", self._run_all))
        res = out[3]
        if self.run is None and not self.videos_out:
            res.append(("info", "No results yet.", "Go to Run", lambda: self.nav.setCurrentRow(2)))
        rep = out[4]
        if self.run is None and not self.videos_out:
            rep.append(("info", "Run the models first - the report is built from the results.", "Go to Run",
                        lambda: self.nav.setCurrentRow(2)))
        return out

    def _apply_validation(self):
        v = self._validate()
        for step, issues in v.items():
            bar = self.issue_bars.get(step)
            if bar is not None:
                bar.set_issues(issues)
            level = "error" if any(i[0] == "error" for i in issues) else "warn" if any(i[0] == "warn" for i in issues) \
                else "ok"
            if step == 3:
                level = "ok" if (self.run is not None and self.run.scored) or self.videos_out else "todo"
            if step == 4:
                level = "ok" if self._report_path else ("todo" if self.run is None and not self.videos_out else "ready")
            it = self.nav.item(step)
            if it is not None:
                it.setIcon(_status_icon(level))
                tip = "\n".join(_strip_tags(i[1]) for i in issues) or {"ok": "Ready", "todo": "Not done yet",
                                                                        "ready": "Ready"}.get(level, "")
                it.setToolTip(tip)
        blocked = [_strip_tags(i[1]) for s in (0, 1, 2) for i in v[s] if i[0] == "error"]
        can = not blocked and self.job is None or (self.job is not None and self.run_btn.text().startswith("■"))
        self.run_btn.setEnabled(bool(can))
        self.run_btn.setToolTip("Start (cached predictions are reused)" if not blocked else "Not ready:\n" + "\n".join(blocked))

    def _review_prelabels(self):
        """Open the label editor on the images with unreviewed pre-labels (needs a scored or loaded run)."""
        if self.run is None:
            QMessageBox.information(self, "Testing", "Run the models once first - the editor shows their detections "
                                    "next to the pre-labels. Or open the images folder in your labelling tool.")
            return
        items = [i for i, it in enumerate(self.run.items) if it.labelled and not it.reviewed]
        if items:
            self._open_editor(items, 0, self.run.ranking()[0] if self.run.ranking() else 0)

    # ------------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            self._drop_hint.setGeometry(self.centralWidget().geometry())
            self._drop_hint.raise_()
            anim.fade_in(self._drop_hint, 120)

    def dragLeaveEvent(self, ev):
        self._drop_hint.hide()

    def dropEvent(self, ev):
        self._drop_hint.hide()
        paths = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        if paths:
            ev.acceptProposedAction()
            later(0, self, lambda: self.handle_paths(paths))
    def handle_paths(self, paths):
        """Route dropped files: projects, weights, data.yaml, test sets, labelled folders, image folders, videos."""
        from AIO_QC.tools.frame_extractor.core.common import is_video, WEIGHT_EXTS
        from AIO_QC.core.testing import project as PJ
        from AIO_QC.core.testing.testset import TestSet
        projects = [p for p in paths if p.lower().endswith(PJ.EXT)]
        if projects:
            self._open_project(projects[0])
            return
        weights = [p for p in paths if p.lower().endswith(WEIGHT_EXTS)]
        yamls = [p for p in paths if p.lower().endswith((".yaml", ".yml"))]
        videos = [p for p in paths if is_video(p)]
        folders = [p for p in paths if os.path.isdir(p)]
        if weights:
            self._add_models([dict(path=p) for p in weights])
            self.status.showMessage(f"Added {len(weights)} model(s)", 4000)
        if yamls:
            self.kind_ds.setChecked(True)
            self.ds_path.setText(yamls[0])
            self._load_data(yamls[0])
            self.nav.setCurrentRow(0)
        for f in folders:
            if TestSet.is_testset(f):
                self.kind_ts.setChecked(True)
                self._load_data(f)
                self.nav.setCurrentRow(0)
                continue
            m = QMenu(self)
            a_ds = m.addAction(f"Use “{os.path.basename(f)}” as the labelled dataset")
            a_src = m.addAction("Add its images to the test set")
            act = m.exec(self.cursor().pos())
            if act is a_ds:
                self.kind_ds.setChecked(True)
                self.ds_path.setText(f)
                self._load_data(f)
            elif act is a_src:
                self.kind_ts.setChecked(True)
                self._add_src_rows([("folder", f)])
            self.nav.setCurrentRow(0)
        if videos:
            m = QMenu(self)
            a_ts = m.addAction(f"Add {len(videos)} video(s) to the test set (unique frames)")
            a_an = m.addAction("Analyse the whole video(s) - behaviour, no labels needed")
            a_both = m.addAction("Both")
            act = m.exec(self.cursor().pos())
            if act in (a_ts, a_both):
                self.kind_ts.setChecked(True)
                self._add_src_rows([("video", p) for p in videos])
            if act in (a_an, a_both):
                have = {v["path"] for v in self.analysis_videos()}
                for p in videos:
                    if p not in have:
                        self._add_analysis_video(p)
            if act is not None:
                self.nav.setCurrentRow(0)
        self._changed()

    def _restore(self):
        """Reopen the last session in the background and show the start page. With a dataset open in Image QC:
        that dataset's own Testing session (or a new one on it, with its models)."""
        from AIO_QC.core.testing import project as PJ
        if self.qc is not None and not self.qc.unavailable():
            st = self._dataset_state()
            later(0, self, lambda: self._apply_state(st))
            later(0, self, lambda: self.nav.setCurrentRow(3 if (st.get("summary") or {}).get("best") else 0))
            return
        st = None
        if os.path.isfile(PJ.AUTOSAVE):
            try:
                st = PJ.load(PJ.AUTOSAVE)
            except Exception:
                st = None
        if st is None:                                 # first run: carry over the earlier (pre-project) settings
            import json
            try:
                rows = json.loads(self.st.value("models", "[]") or "[]")
            except ValueError:
                rows = []
            if not rows:
                rows = [dict(path=p) for p in self.model_cfg.get("models", []) if os.path.isfile(p)]
            st = dict(name="Untitled", models=[r for r in rows if os.path.isfile(r.get("path", ""))])
            ts, ds = self.st.value("testset", ""), self.st.value("dataset", "")
            if ts and os.path.isdir(ts):
                st["data"] = dict(kind="testset", path=ts)
            elif ds and os.path.exists(ds):
                st["data"] = dict(kind="dataset", path=ds)
        else:
            pp = st.get("project_path")
            self._project_path = pp if pp and os.path.isfile(pp) else None
        later(0, self, lambda: self._apply_state(st))
        self._show_home()

    def _dataset_state(self):
        """The Testing project stored with the open dataset, else a fresh one on it with the dataset's models."""
        from AIO_QC.core.testing import project as PJ
        from AIO_QC.core import model_registry as R
        root = str(self.qc.session.root)
        p = PJ.dataset_project(root)
        if os.path.isfile(p):
            try:
                st = PJ.load(p)
                st["data"] = dict(st.get("data") or {}, kind="qc", path=root)
                return st
            except Exception:
                pass
        models = [dict(path=m["path"], label=m.get("name", ""), imgsz=R.effective_imgsz(m),
                       class_map=m.get("class_map") or {})
                  for m in R.dataset_models(root) if m.get("enabled", True) and os.path.isfile(m.get("path", ""))]
        return dict(name=os.path.basename(root), data=dict(kind="qc", path=root), models=models)

    def use_qc_dataset(self):
        """Image QC opened another dataset (or Testing was opened from it): switch to it."""
        if self.qc is None or self.qc.unavailable():
            return
        if self.qc.is_this(self.data):
            return
        if self.job is not None:
            return
        self._apply_state(self._dataset_state())

    def set_model_cfg(self, cfg):
        self.model_cfg = dict(cfg or {})
        if not self.env_edit.text().strip():
            self.env_edit.setText(self.model_cfg.get("env_python", ""))

    def closeEvent(self, ev):
        if not self._confirm_discard():
            ev.ignore()
            return
        self._autosave()                 # (the view thread stays: the window can be reopened)
        for j in (self.job, getattr(self, "_f_job", None), getattr(self, "_cmp_job", None),
                  getattr(self, "_preview_job", None)):
            try:
                if j is not None and j.isRunning():
                    j.cancel()
                    j.wait(4000)
            except RuntimeError:
                pass
        super().closeEvent(ev)
