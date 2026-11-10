# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Batch queue: extract frames from many videos in parallel worker processes.

Each row is a video (optional start/end/ROI, editable in the table). Long videos
are split into chunks when there are fewer videos than workers; chunk results
are merged back into the row. Workers are spawn-based processes (decoding and
inference are CPU/GPU bound; threads would serialise on the GIL). Progress
arrives through a Manager queue polled by a QTimer.
"""
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from loguru import logger
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QTabWidget, QSizePolicy, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QPushButton, QLabel, QGroupBox, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox,
)

from AIO_QC.widgets.motion import SmoothProgressBar, animate_tabs
from AIO_QC.tools.frame_extractor.core.extract import ExtractSettings, Job, job_process
from AIO_QC.tools.frame_extractor.core.batch import plan_tasks, worker_threads, merge_stats
from AIO_QC.tools.frame_extractor.core.detector import MINING_CRITERIA
from AIO_QC.tools.frame_extractor.core.sources import parse_time, HAVE_PYAV
from AIO_QC.tools.frame_extractor.core.hashdb import default_db_path
from AIO_QC.tools.frame_extractor.core import tracker as roi_tracker
from AIO_QC.tools.frame_extractor.utils import newIcon

from AIO_QC.tools.frame_extractor.core.common import VIDEO_EXTS, VIDEO_FILTER
from AIO_QC.utils.fx_settings import fx_settings
C_VIDEO, C_START, C_END, C_ROI, C_STATUS, C_PROGRESS, C_FPS, C_ETA, C_SAVED = range(9)
RED, GREEN = QColor(214, 69, 69), QColor(31, 157, 107)


class ElidedLabel(QLabel):
    """Shows as much of the text as fits ("…" in the middle) and never forces the window wider."""

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._full = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(60)
        self.setText(text)

    def setText(self, text):
        self._full = text or ""
        self.setToolTip(self._full)
        self._elide()

    def _elide(self):
        fm = self.fontMetrics()
        QLabel.setText(self, fm.elidedText(self._full, Qt.TextElideMode.ElideMiddle, max(20, self.width())))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._elide()


def _fmt_eta(sec):
    if sec is None or sec != sec or sec == float("inf"):
        return "–"
    sec = int(sec)
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


class BatchQueueDialog(QDialog):
    def __init__(self, parent, output_dir_getter, sharpness_getter=None, model_cfg_getter=None, open_models=None,
                 open_gallery=None, output_cfg_getter=None, output_cfg_setter=None, open_output=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Extract Queue")
        self.setWindowIcon(newIcon("batch"))
        self.setAcceptDrops(True)
        self._fit_to_screen()
        self.output_dir_getter = output_dir_getter
        self.sharpness_getter = sharpness_getter
        self.model_cfg_getter = model_cfg_getter or (lambda: {"models": []})
        self.open_models = open_models
        self.open_gallery = open_gallery
        from AIO_QC.tools.frame_extractor.widgets.output_dialog import DEFAULTS as _OUT_DEFAULTS
        self.output_cfg_getter = output_cfg_getter or (lambda: dict(_OUT_DEFAULTS))
        self.output_cfg_setter = output_cfg_setter
        self.open_output = open_output
        self.store = fx_settings()

        self.executor = self.manager = self.queue = self.cancel_event = None
        self.futures = {}     # task id -> future
        self.tasks = {}       # task id -> row
        self.tstate = {}      # task id -> dict(done,total,finished,stats,error,t0)
        self.run_t0 = None
        self._next_id = 0

        lay = QVBoxLayout(self)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Video", "Start", "End", "ROI (x1,y1,x2,y2)",
                                              "Status", "Progress", "FPS", "ETA", "Saved"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(C_VIDEO, QHeaderView.ResizeMode.Stretch)
        for c in (C_START, C_END, C_ROI, C_FPS, C_ETA, C_SAVED):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(C_STATUS, 200)
        self.table.setColumnWidth(C_PROGRESS, 110)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)   # long statuses elide (full text in tooltip)
        self.table.setWordWrap(False)
        self.table.setToolTip("Start/End accept HH:MM:SS.mmm, MM:SS or seconds (blank = whole video).\n"
                              "Double-click a cell to edit. Drop video files or folders here to add them.")
        lay.addWidget(self.table, 1)

        row = QHBoxLayout()
        for text, slot in (("Add videos…", self.browse_videos), ("Remove selected", self.remove_selected),
                           ("Clear finished", self.clear_finished)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
            if text == "Remove selected":
                self.remove_btn = b
        self.out_label = ElidedLabel()
        self.out_label.setStyleSheet("color: gray;")
        self.out_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.out_label, 1)
        lay.addLayout(row)

        # ---- frame selection
        box = QGroupBox("Frame selection")
        g = QGridLayout(box)
        self.mode_cb = QComboBox()
        self.mode_cb.setToolTip('<b>How frames are picked</b><br><b>Every Nth frame</b>: fast, evenly spaced.<br><b>Sharpest of every N</b>: decodes every frame and keeps the least blurry one per window - best for annotation, slower.')
        self.mode_cb.addItems(["Every Nth frame", "Sharpest frame of every N frames"])
        self.step_sb = QSpinBox()
        self.step_sb.setToolTip('N: the spacing (Every Nth) or the window size (Sharpest of every N), in frames. At 30 fps, N = 30 ≈ one frame per second.')
        self.step_sb.setRange(1, 100000)
        self.step_sb.setPrefix("N = ")
        g.addWidget(QLabel("Sampling:"), 0, 0)
        g.addWidget(self.mode_cb, 0, 1)
        g.addWidget(self.step_sb, 0, 2)
        self.sharp_sb = QDoubleSpinBox()
        self.sharp_sb.setRange(0, 100000)
        self.sharp_sb.setDecimals(1)
        self.sharp_sb.setSpecialValueText("off")
        self.sharp_sb.setToolTip("Minimum sharpness (variance of Laplacian at ~640px). 0 = off.")
        self.sharp_from_btn = QPushButton("From current frame")
        self.sharp_from_btn.setToolTip('Use the sharpness of the frame shown in the player as the minimum: pause on a frame that is just acceptably sharp, then click.')
        self.sharp_from_btn.clicked.connect(self._sharp_from_current)
        g.addWidget(QLabel("Min sharpness:"), 1, 0)
        g.addWidget(self.sharp_sb, 1, 1)
        g.addWidget(self.sharp_from_btn, 1, 2)
        self.neg_sb = QSpinBox()
        self.neg_sb.setRange(0, 100000)
        self.neg_sb.setSpecialValueText("don't keep")
        self.neg_sb.setPrefix("1 in ")
        self.neg_sb.setToolTip("Keep every K-th rejected blurry frame in 'negatives_blurry/'.")
        g.addWidget(QLabel("Blurry negatives:"), 2, 0)
        g.addWidget(self.neg_sb, 2, 1)
        from AIO_QC.tools.frame_extractor.core.dedup import PRESETS as DD_PRESETS, PRESET_HELP as DD_HELP
        self._dd_presets = DD_PRESETS
        self.dedup_chk = QCheckBox("Skip duplicates")
        self.dedup_chk.setToolTip(
            "<b>Skip frames that repeat what was already saved.</b><br>"
            "Each candidate is compared with the last few saved frames (also catches A → B → A, e.g. a camera "
            "panning back). Brightness / exposure changes are ignored; a change anywhere in the picture - even a "
            "small plate in one corner of a 4K frame - keeps the frame.<br><i>Counted as 'dup' in the status.</i>")
        self.dedup_cb = QComboBox()
        for key, label in (("exact", "Exact repeats only"), ("conservative", "Conservative"),
                           ("balanced", "Balanced (recommended)"), ("aggressive", "Aggressive"), ("custom", "Custom…")):
            self.dedup_cb.addItem(label, key)
            if key in DD_HELP:
                self.dedup_cb.setItemData(self.dedup_cb.count() - 1, DD_HELP[key], Qt.ItemDataRole.ToolTipRole)
        self.dedup_cb.setToolTip("<b>How similar counts as a duplicate</b><br>" + "<br>".join(
            f"<b>{k.title()}</b>: {v}" for k, v in DD_HELP.items()) + "<br><b>Custom</b>: set the two limits yourself.")
        self.dedup_sb = QDoubleSpinBox()
        self.dedup_sb.setRange(0.1, 50)
        self.dedup_sb.setSingleStep(0.5)
        self.dedup_sb.setPrefix("whole < ")
        self.dedup_sb.setSuffix(" %")
        self.dedup_sb.setToolTip("Custom: largest average change over the whole (exposure-normalised) picture that "
                                 "still counts as 'the same'. Sensor noise ≈ 1 %, a new car ≈ 4 %.")
        self.dedup_local_sb = QDoubleSpinBox()
        self.dedup_local_sb.setRange(0.5, 100)
        self.dedup_local_sb.setSingleStep(1.0)
        self.dedup_local_sb.setPrefix("local < ")
        self.dedup_local_sb.setSuffix(" %")
        self.dedup_local_sb.setToolTip("Custom: largest change allowed in any one of 16 × 9 picture cells. This is what "
                                       "protects small objects: a new plate in a corner of a 4K frame changes its "
                                       "cell by ~47 % while the whole picture changes only ~1.5 %.")
        self.dedup_win_sb = QSpinBox()
        self.dedup_win_sb.setRange(1, 50)
        self.dedup_win_sb.setPrefix("vs last ")
        self.dedup_win_sb.setToolTip("Compare with this many most recently saved frames (1 = only the previous one).")
        self.dedup_chk.toggled.connect(self._dedup_ui)
        self.dedup_cb.currentIndexChanged.connect(self._dedup_ui)
        drow = QHBoxLayout()
        for w_ in (self.dedup_cb, self.dedup_sb, self.dedup_local_sb, self.dedup_win_sb):
            drow.addWidget(w_)
        g.addWidget(self.dedup_chk, 0, 3)
        g.addLayout(drow, 0, 4)
        self.scene_chk = QCheckBox("Detect scene cuts")
        self.scene_chk.setToolTip('<b>Detect scene cuts</b>: sampling restarts at every cut, so every shot contributes frames (useful for edited footage or multi-camera files). Also resets duplicate checks at each cut.')
        self.scene_sb = QDoubleSpinBox()
        self.scene_sb.setToolTip('Cut sensitivity: colour-histogram distance between consecutive frames. Lower = more cuts detected (0.3 sensitive, 0.4 default, 0.6 only hard cuts).')
        self.scene_sb.setRange(0.05, 1.0)
        self.scene_sb.setSingleStep(0.05)
        self.scene_chk.toggled.connect(self.scene_sb.setEnabled)
        g.addWidget(self.scene_chk, 1, 3)
        g.addWidget(self.scene_sb, 1, 4)
        self.backend_cb = QComboBox()
        self.backend_cb.setToolTip("<b>Decoder</b><br><b>auto</b>: OpenCV with exact timestamps from the file index; PyAV for VFR files it can't index.<br><b>opencv</b>: fastest.<br><b>pyav</b>: exact timestamps for any container (needs pip install av).")
        self.backend_cb.addItems(["auto", "opencv", "pyav"])
        g.addWidget(QLabel("Decoder:"), 2, 3)
        g.addWidget(self.backend_cb, 2, 4)
        orow = QHBoxLayout()
        self.out_summary = ElidedLabel()
        self.out_summary.setStyleSheet("font-family: Consolas, monospace;")
        ob = QPushButton("Format && naming…")
        ob.setToolTip('Image format, quality and the file-naming template (shared with Save frame and markers).')
        ob.setIcon(newIcon("save"))
        ob.clicked.connect(lambda: self.open_output and self.open_output())
        orow.addWidget(QLabel("Output:"))
        orow.addWidget(self.out_summary, 1)
        orow.addWidget(ob)
        self._out_row = orow
        self.track_chk = QCheckBox("ROI follows the object (tracker)")
        kinds = roi_tracker.available()
        self.track_chk.setToolTip("The row's ROI is taken at its Start time and tracked through the video\n"
                                  f"(available trackers: {', '.join(kinds) or 'none'}). Tracked videos are not split.")
        self.track_chk.setEnabled(bool(kinds))
        g.addWidget(self.track_chk, 3, 0, 1, 3)
        g.setColumnStretch(5, 1)

        # ---- models
        mbox = QGroupBox("Model-assisted mining")
        m = QGridLayout(mbox)
        self.models_label = QLabel()
        mb = QPushButton("Models…")
        mb.setToolTip('Choose YOLO models (.pt / .onnx) and the Python environment that runs them.')
        mb.clicked.connect(self._open_models)
        m.addWidget(self.models_label, 0, 0, 1, 3)
        m.addWidget(mb, 0, 3)
        self.mine_chk = {}
        for i, (key, tip) in enumerate(MINING_CRITERIA.items()):
            c = QCheckBox({"uncertain": "Uncertain", "empty": "No detections", "disagree": "Models disagree",
                           "classes": "Classes:", "temporal": "Flicker / blip"}[key])
            c.setToolTip(tip)
            self.mine_chk[key] = c
            m.addWidget(c, 1 + i // 2, (i % 2) * 2)
        self.classes_edit = QLineEdit()
        self.classes_edit.setToolTip('Mine frames containing these classes (names or ids, comma separated) - e.g. rare classes you need more examples of: bus, truck, 7')
        self.classes_edit.setPlaceholderText("rare classes, e.g. bus,truck")
        m.addWidget(self.classes_edit, 2, 3)
        tr = QHBoxLayout()
        self.tstride_sb = QSpinBox()
        self.tstride_sb.setRange(1, 60)
        self.tstride_sb.setPrefix("every ")
        self.tstride_sb.setSuffix(" fr")
        self.tstride_sb.setToolTip("Run the primary model on every Nth frame for the temporal check (1 = every frame).")
        self.tgap_sb = QSpinBox()
        self.tgap_sb.setRange(1, 10)
        self.tgap_sb.setPrefix("gap ≤ ")
        self.tgap_sb.setToolTip("An object that vanishes (flicker) or exists (blip) for at most this many checks is flagged.")
        tr.addWidget(self.tstride_sb)
        tr.addWidget(self.tgap_sb)
        m.addLayout(tr, 3, 1, 1, 3)
        self.mine_chk["temporal"].setToolTip(MINING_CRITERIA["temporal"] + ".\nThese hard examples are saved even "
                                             "if blurry or similar to neighbours. Runs the model on many frames.")
        self.preannot_cb = QComboBox()
        self.preannot_cb.addItems(["none", "labelme", "yolo"])
        self.preannot_cb.setToolTip("Write primary-model detections next to each saved frame:\n"
                                    "LabelMe JSON (mining reasons stored as image flags) or YOLO txt + classes.txt")
        m.addWidget(QLabel("Pre-annotations:"), 4, 0)
        m.addWidget(self.preannot_cb, 4, 1)
        hint = QLabel("A frame is kept if it matches any checked criterion. No criteria = keep all "
                      "(useful for pre-annotation only).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 8pt;")
        m.addWidget(hint, 5, 0, 1, 4)

        # ---- performance
        pbox = QGroupBox("Performance")
        pg = QGridLayout(pbox)
        self.workers_sb = QSpinBox()
        self.workers_sb.setRange(0, max(1, os.cpu_count() or 1))
        self.workers_sb.setSpecialValueText("Auto")
        self.workers_sb.setToolTip(
            "<b>Parallel workers</b> - videos (or chunks of one long video) processed at the same time, one process "
            "each.<br><b>Auto</b> (recommended): planned when you press Start from the queued videos' resolution, "
            "the free RAM and how busy the CPU is right now, within the share set below - so the PC stays usable."
            "<br>A fixed number overrides that; each worker then gets an equal share of the allowed cores.")
        self.share_cb = QComboBox()
        for key, label in (("light", "Light - keep most of the PC free (25 %)"),
                           ("balanced", "Balanced (50 %)"), ("max", "Maximum speed (all cores)")):
            self.share_cb.addItem(label, key)
        self.share_cb.setToolTip(
            "<b>How much of this PC a batch may use.</b> Workers run at below-normal priority either way, so "
            "anything you do yourself takes precedence; this also caps how many cores the workers' decoders may "
            "use in total (each worker's OpenCV would otherwise use every core).<br>Light: browsing / office work "
            "alongside. Balanced: default. Maximum: nothing else running.")
        self.split_chk = QCheckBox("Split long videos across workers")
        self.split_chk.setToolTip("When there are fewer videos than workers, long videos are cut into\n"
                                  "chunks aligned to the sampling grid (same frames as one pass).")
        self.chunk_sb = QSpinBox()
        self.chunk_sb.setToolTip("Long videos are split only into chunks at least this long, so short videos aren't over-split (each chunk has a small start-up cost).")
        self.chunk_sb.setRange(5, 36000)
        self.chunk_sb.setSuffix(" s min chunk")
        pg.addWidget(QLabel("Parallel workers:"), 0, 0)
        pg.addWidget(self.workers_sb, 0, 1)
        pg.addWidget(QLabel("Use of this PC:"), 3, 0)
        pg.addWidget(self.share_cb, 3, 1)
        pg.addWidget(self.split_chk, 1, 0, 1, 2)
        pg.addWidget(self.chunk_sb, 2, 1)
        self.resume_chk = QCheckBox("Resume interrupted jobs")
        self.resume_chk.setToolTip("Continue crashed/cancelled jobs from their last checkpoint (.fx_resume/);\n"
                                   "jobs already finished with the same settings are skipped instantly.")
        pg.addWidget(self.resume_chk, 4, 0, 1, 2)
        pr = QHBoxLayout()
        for text, slot in (("Save preset…", self.save_preset), ("Load preset…", self.load_preset)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            pr.addWidget(b)
        pg.addLayout(pr, 3, 0, 1, 2)

        dbox = QGroupBox("Dataset")
        dg = QGridLayout(dbox)
        self.xdedup_chk = QCheckBox("Skip frames already in the dataset")
        self.xdedup_chk.setToolTip("<b>Skip frames already in this dataset</b> (any video, any earlier run).<br>"
                                   "A perceptual hash plus a small thumbnail of every saved frame is kept in the output "
                                   "root. A hash match is double-checked on the thumbnail with the same local test as "
                                   "'Skip duplicates', so a different car at the same gate is never mistaken for an old "
                                   "frame. Frames rejected in Review stay in the database, so they aren't re-extracted.<br>"
                                   "<i>Counted as 'in dataset' in the status.</i>")
        self.xdedup_sb = QSpinBox()
        self.xdedup_sb.setRange(0, 20)
        self.xdedup_sb.setPrefix("≤ ")
        self.xdedup_sb.setSuffix(" bits")
        self.xdedup_sb.setToolTip("How different two hashes may be and still be checked as a possible repeat "
                                  "(0 = identical hash only, 6 = near-identical, up to 20 = loose). Loose values are "
                                  "safe: every match is verified on the thumbnail before a frame is skipped.")
        self.xdedup_chk.toggled.connect(self.xdedup_sb.setEnabled)
        self.db_label = QLabel()
        self.db_label.setStyleSheet("color: gray; font-size: 8pt;")
        self.db_label.setWordWrap(True)
        clear_db = QPushButton("Clear hashes…")
        clear_db.setToolTip("Forget every frame recorded in this output folder's dataset database. Afterwards nothing counts as 'already in the dataset'. Extracted files are not touched.")
        clear_db.clicked.connect(self.clear_hashes)
        self.review_btn = QPushButton("Review results…")
        self.review_btn.setIcon(newIcon("gallery"))
        self.review_btn.setToolTip("Open the review gallery for the selected row (or double-click a row)")
        self.review_btn.clicked.connect(self.review_selected)
        dg.addWidget(self.xdedup_chk, 0, 0, 1, 2)
        dg.addWidget(self.xdedup_sb, 1, 0)
        dg.addWidget(clear_db, 1, 1)
        dg.addWidget(self.db_label, 2, 0, 1, 2)
        dg.addWidget(self.review_btn, 3, 0, 1, 2)
        self.qc_btn = QPushButton("Open in Image QC")
        self.qc_btn.setIcon(newIcon("open"))
        self.qc_btn.setToolTip("Open the selected row's output (frames + any pre-labels) as the dataset in AIO QC's "
                               "Image QC workspace")
        self.qc_btn.clicked.connect(self.open_selected_in_qc)
        dg.addWidget(self.qc_btn, 4, 0, 1, 2)

        # settings as tabs: the window is only as wide as the widest single group
        self.tabs = QTabWidget()
        animate_tabs(self.tabs)
        for grp, icon, title in ((box, "marker", "Frames"), (mbox, "detect", "Models"),
                                 (pbox, "run", "Performance"), (dbox, "gallery", "Dataset")):
            grp.setTitle("")
            grp.setFlat(True)
            self.tabs.addTab(grp, newIcon(icon), title)
        self.tabs.setCurrentIndex(int(self.store.value("batch/tab", 0)))
        self.tabs.currentChanged.connect(lambda i: self.store.setValue("batch/tab", i))
        self.tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        lay.addWidget(self.tabs)
        lay.addLayout(self._out_row)

        row = QHBoxLayout()
        self.summary = QLabel("Queue is empty.")
        row.addWidget(self.summary, 1)
        self.start_btn = QPushButton("Start")
        self.start_btn.setToolTip("Start every queued video that isn't Done. Finished jobs are skipped instantly when Resume is on.")
        self.start_btn.setIcon(newIcon("play"))
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip('Stop the running batch. Each job saves a checkpoint - Start again continues where it stopped.')
        self.cancel_btn.setIcon(newIcon("delete"))
        self.cancel_btn.setEnabled(False)
        close_btn = QPushButton("Close")
        close_btn.setToolTip('Close this window (a running batch is cancelled after confirmation).')
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn.clicked.connect(self.cancel)
        close_btn.clicked.connect(self.close)
        for b in (self.start_btn, self.cancel_btn, close_btn):
            row.addWidget(b)
        lay.addLayout(row)

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(150)
        self.poll_timer.timeout.connect(self.poll)

        s = ExtractSettings.from_dict(self._stored_settings())
        self.apply_settings(s)
        self.workers_sb.setValue(int(self.store.value("batch/workers2", 0)))           # 0 = Auto
        self.share_cb.setCurrentIndex(max(0, self.share_cb.findData(self.store.value("batch/share", "balanced"))))
        self.split_chk.setChecked(self.store.value("batch/split", True, type=bool))
        self.chunk_sb.setValue(int(self.store.value("batch/min_chunk", 60)))
        self.table.cellDoubleClicked.connect(lambda r, c: c in (C_VIDEO, C_STATUS, C_SAVED) and self.review_row(r))
        self.refresh_db_label()
        self.refresh_output_summary()
        self.refresh_output_label()
        self.refresh_models_label()

    def _dedup_ui(self, *_):
        on = self.dedup_chk.isChecked()
        key = self.dedup_cb.currentData()
        if key in self._dd_presets:
            g_, l_ = self._dd_presets[key]
            self.dedup_sb.setValue(g_)
            self.dedup_local_sb.setValue(l_)
        custom = key == "custom"
        self.dedup_cb.setEnabled(on)
        for w_ in (self.dedup_sb, self.dedup_local_sb):
            w_.setVisible(custom)
            w_.setEnabled(on)
        self.dedup_win_sb.setEnabled(on and key != "exact")

    def _fit_to_screen(self):
        from AIO_QC.tools.frame_extractor.utils.qt_utils import preferred_size
        preferred_size(self, 960, 700, frac=0.9)
        self.setMinimumWidth(min(640, self.width()))

    # ------------------------------------------------------------ settings
    def _stored_settings(self):
        d = {}
        for k, v in ExtractSettings().to_dict().items():
            if k in ("models", "det_threads", "hw_accel", "det_conf", "det_iou", "det_imgsz",
                     "det_device", "unc_low", "unc_high", "hash_db", "env_python", "det_half", "det_sahi",
                     "sahi_slice", "sahi_overlap", "ext", "quality", "name_template", "on_exists"):
                continue
            raw = self.store.value(f"batch/{k}", None)
            if raw is None:
                continue
            if isinstance(v, bool):
                d[k] = raw in (True, "true", "1", 1)
            elif isinstance(v, list):
                d[k] = [x for x in (raw if isinstance(raw, list) else [raw]) if x]
            else:
                d[k] = type(v)(raw)
        return d

    def apply_settings(self, s, output=False):
        """output=True (loading a preset) also applies its format / quality / naming app-wide."""
        self.mode_cb.setCurrentIndex(1 if s.mode == "sharpest" else 0)
        self.step_sb.setValue(s.step)
        self.sharp_sb.setValue(s.min_sharpness)
        self.neg_sb.setValue(s.keep_blurry_every)
        self.dedup_chk.setChecked(s.dedup)
        self.dedup_sb.setValue(s.dedup_threshold)
        self.dedup_local_sb.setValue(s.dedup_local)
        self.dedup_win_sb.setValue(s.dedup_window)
        key = "exact" if s.dedup_mode == "exact" else next(
            (k for k, (g_, l_) in self._dd_presets.items() if abs(g_ - s.dedup_threshold) < 1e-6
             and abs(l_ - s.dedup_local) < 1e-6), "custom")
        self.dedup_cb.setCurrentIndex(max(0, self.dedup_cb.findData(key)))
        self._dedup_ui()
        self.scene_chk.setChecked(s.scene_detect)
        self.scene_sb.setValue(s.scene_threshold)
        self.scene_sb.setEnabled(s.scene_detect)
        if output and self.output_cfg_setter and (s.ext, s.quality, s.name_template, s.on_exists) != tuple(
                self.output_cfg_getter()[k] for k in ("ext", "quality", "template", "on_exists")):
            self.output_cfg_setter(dict(ext=s.ext, quality=s.quality, template=s.name_template,
                                        on_exists=s.on_exists), True)
        self.backend_cb.setCurrentText(s.backend)
        for k, c in self.mine_chk.items():
            c.setChecked(k in s.mine)
        self.classes_edit.setText(",".join(s.mine_classes))
        self.preannot_cb.setCurrentText(s.preannot)
        self.tstride_sb.setValue(s.temporal_stride)
        self.tgap_sb.setValue(s.temporal_gap)
        self.track_chk.setChecked(s.roi_track and self.track_chk.isEnabled())
        self.xdedup_chk.setChecked(s.cross_dedup)
        self.xdedup_sb.setValue(s.cross_dedup_dist)
        self.xdedup_sb.setEnabled(s.cross_dedup)
        self.resume_chk.setChecked(s.resume)

    def current_settings(self):
        cfg = self.model_cfg_getter()
        return ExtractSettings(
            mode="sharpest" if self.mode_cb.currentIndex() == 1 else "step",
            step=self.step_sb.value(),
            min_sharpness=self.sharp_sb.value(),
            keep_blurry_every=self.neg_sb.value() if self.sharp_sb.value() > 0 else 0,
            dedup=self.dedup_chk.isChecked(),
            dedup_mode="exact" if self.dedup_cb.currentData() == "exact" else "near",
            dedup_threshold=self.dedup_sb.value(),
            dedup_local=self.dedup_local_sb.value(),
            dedup_window=self.dedup_win_sb.value(),
            scene_detect=self.scene_chk.isChecked(),
            scene_threshold=self.scene_sb.value(),
            ext=self.output_cfg_getter()["ext"], quality=self.output_cfg_getter()["quality"],
            name_template=self.output_cfg_getter()["template"], on_exists=self.output_cfg_getter()["on_exists"],
            backend=self.backend_cb.currentText(),
            models=list(cfg.get("models", [])),
            det_conf=cfg.get("conf", 0.25), det_iou=cfg.get("iou", 0.45), det_imgsz=cfg.get("imgsz", 0),
            det_device=cfg.get("device", "auto"),
            env_python=cfg.get("env_python", ""), det_half=cfg.get("half", False), det_sahi=cfg.get("sahi", False),
            sahi_slice=cfg.get("sahi_slice", 640), sahi_overlap=cfg.get("sahi_overlap", 0.2),
            unc_low=cfg.get("unc_low", 0.25), unc_high=cfg.get("unc_high", 0.6),
            mine=[k for k, c in self.mine_chk.items() if c.isChecked()],
            mine_classes=[c.strip() for c in self.classes_edit.text().split(",") if c.strip()],
            preannot=self.preannot_cb.currentText(),
            temporal_stride=self.tstride_sb.value(),
            temporal_gap=self.tgap_sb.value(),
            cross_dedup=self.xdedup_chk.isChecked(),
            cross_dedup_dist=self.xdedup_sb.value(),
            hash_db=default_db_path(self.output_dir_getter()),
            roi_track=self.track_chk.isChecked(),
            resume=self.resume_chk.isChecked(),
        )

    def _save_settings(self, s):
        for k, v in s.to_dict().items():
            if k not in ("models", "det_threads", "hw_accel", "hash_db", "env_python", "det_half", "det_sahi",
                         "sahi_slice", "sahi_overlap", "ext", "quality", "name_template", "on_exists"):
                self.store.setValue(f"batch/{k}", v)
        self.store.setValue("batch/workers2", self.workers_sb.value())
        self.store.setValue("batch/share", self.share_cb.currentData())
        self.store.setValue("batch/split", self.split_chk.isChecked())
        self.store.setValue("batch/min_chunk", self.chunk_sb.value())

    def save_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", "preset.json", "Preset (*.json)")
        if path:
            self.current_settings().save_preset(path)

    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load preset", "", "Preset (*.json)")
        if not path:
            return
        try:
            s = ExtractSettings.load_preset(path)
        except Exception as e:
            QMessageBox.warning(self, "Preset", f"Could not load preset:\n{e}")
            return
        self.apply_settings(s, output=True)
        if s.models and not self.model_cfg_getter().get("models"):
            QMessageBox.information(self, "Preset", "The preset lists models:\n" + "\n".join(s.models) +
                                    "\n\nAdd them in Settings → Models to use them.")

    def refresh_output_summary(self):
        from AIO_QC.tools.frame_extractor.widgets.output_dialog import summary
        self.out_summary.setText(summary(self.output_cfg_getter()))

    def _sharp_from_current(self):
        v = self.sharpness_getter() if self.sharpness_getter else None
        if v is None:
            QMessageBox.information(self, "Sharpness", "Load a video and pause on a frame first.")
            return
        self.sharp_sb.setValue(round(v, 1))

    def _open_models(self):
        if self.open_models:
            self.open_models()
        self.refresh_models_label()

    def refresh_models_label(self):
        models = self.model_cfg_getter().get("models", [])
        if models:
            names = ", ".join(os.path.basename(m) for m in models)
            self.models_label.setText(f"<b>{len(models)} model(s):</b> {names}")
        else:
            self.models_label.setText("<i>No models configured.</i>")
        for c in self.mine_chk.values():
            c.setEnabled(bool(models))
        self.mine_chk["disagree"].setEnabled(len(models) >= 2)
        self.preannot_cb.setEnabled(bool(models))

    def refresh_db_label(self):
        path = default_db_path(self.output_dir_getter())
        n = 0
        if os.path.exists(path):
            try:
                import sqlite3
                con = sqlite3.connect(path, timeout=5)
                n = con.execute("SELECT COUNT(*) FROM hashes").fetchone()[0]
                con.close()
            except Exception:
                n = "?"
        self.db_label.setText(f"{n} frame hash(es) in {os.path.basename(path)} (output root)")

    def clear_hashes(self):
        path = default_db_path(self.output_dir_getter())
        if not os.path.exists(path) or self.is_running():
            return
        if QMessageBox.question(self, "Clear hashes", "Forget all extracted-frame hashes for this output folder?\n"
                                "Frames will no longer be recognised as already extracted.") \
                != QMessageBox.StandardButton.Yes:
            return
        from AIO_QC.tools.frame_extractor.core.hashdb import HashDB
        db = HashDB(path)
        db.clear()
        db.close()
        self.refresh_db_label()

    def _row_output(self, r):
        path = self.table.item(r, C_VIDEO).data(Qt.ItemDataRole.UserRole)
        name = os.path.splitext(os.path.basename(path))[0]
        sub = "markers" if self.table.item(r, C_VIDEO).data(Qt.ItemDataRole.UserRole + 1) else "batch_extract"
        return os.path.join(self.output_dir_getter(), name, sub)

    def review_row(self, r):
        out = self._row_output(r)
        if not os.path.isdir(out):
            QMessageBox.information(self, "Review", f"Nothing extracted yet:\n{out}")
            return
        if self.open_gallery:
            self.open_gallery(out)

    def open_selected_in_qc(self):
        from PyQt6.QtWidgets import QApplication
        shell = getattr(QApplication.instance(), "shell", None)
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if shell is None or not rows:
            QMessageBox.information(self, "Open in Image QC", "Select a row whose extraction has finished.")
            return
        out = self._row_output(rows[0])
        if not os.path.isdir(out):
            QMessageBox.information(self, "Open in Image QC", f"Nothing extracted yet:\n{out}")
            return
        shell.open_in_qc(out)

    def review_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if rows:
            self.review_row(rows[0])
        elif self.open_gallery:
            self.open_gallery(None)

    def refresh_output_label(self):
        self.out_label.setText(f"Output: {self.output_dir_getter()}/<video>/batch_extract")
        if hasattr(self, "db_label"):
            self.refresh_db_label()

    # --------------------------------------------------------------- jobs
    def add_marker_job(self, video_path, frames):
        """A row that saves exactly these frames ([frame, roi|None], from timeline markers)."""
        r = self.add_job(video_path)
        self.table.item(r, C_VIDEO).setData(Qt.ItemDataRole.UserRole + 1, [list(f) for f in frames])
        for c, text in ((C_START, f"{len(frames)} markers"), (C_END, ""), (C_ROI, "per marker")):
            it = self.table.item(r, c)
            it.setText(text)
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.item(r, C_VIDEO).setToolTip(self.table.item(r, C_VIDEO).toolTip() +
                                               "\nMarker job: saves exactly the marked frames")
        return r

    def add_job(self, video_path, start="", end="", roi=None):
        r = self.table.rowCount()
        self.table.insertRow(r)
        it = QTableWidgetItem(os.path.basename(video_path))
        it.setToolTip(video_path)
        it.setData(Qt.ItemDataRole.UserRole, video_path)
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(r, C_VIDEO, it)
        self.table.setItem(r, C_START, QTableWidgetItem(start))
        self.table.setItem(r, C_END, QTableWidgetItem(end))
        self.table.setItem(r, C_ROI, QTableWidgetItem(",".join(map(str, roi)) if roi else ""))
        for c, text in ((C_STATUS, "Queued"), (C_FPS, ""), (C_ETA, ""), (C_SAVED, "")):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, c, item)
        bar = SmoothProgressBar()
        bar.setRange(0, 1000)
        bar.setTextVisible(False)
        self.table.setCellWidget(r, C_PROGRESS, bar)
        self._update_summary()
        return r

    def browse_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add videos", "", VIDEO_FILTER)
        for f in files:
            self.add_job(f)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for url in ev.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                for name in sorted(os.listdir(path)):
                    if name.lower().endswith(VIDEO_EXTS):
                        self.add_job(os.path.join(path, name))
            elif path.lower().endswith(VIDEO_EXTS):
                self.add_job(path)

    def _row_status(self, r):
        return self.table.item(r, C_STATUS).text()

    def remove_selected(self):
        if self.is_running():
            return
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)
        self._update_summary()

    def clear_finished(self):
        if self.is_running():
            return
        for r in reversed(range(self.table.rowCount())):
            if self._row_status(r).startswith(("Done", "Error", "Cancelled")):
                self.table.removeRow(r)
        self._update_summary()

    def _set_row_editable(self, r, editable):
        if self.table.item(r, C_VIDEO).data(Qt.ItemDataRole.UserRole + 1):
            editable = False          # marker rows are defined by their markers
        for c in (C_START, C_END, C_ROI):
            it = self.table.item(r, c)
            f = it.flags()
            it.setFlags(f | Qt.ItemFlag.ItemIsEditable if editable else f & ~Qt.ItemFlag.ItemIsEditable)

    def _parse_row(self, r):
        path = self.table.item(r, C_VIDEO).data(Qt.ItemDataRole.UserRole)
        frames = self.table.item(r, C_VIDEO).data(Qt.ItemDataRole.UserRole + 1)
        if frames:
            name = os.path.splitext(os.path.basename(path))[0]
            return Job(path, os.path.join(self.output_dir_getter(), name, "markers"), frames=frames)
        start = self.table.item(r, C_START).text().strip()
        end = self.table.item(r, C_END).text().strip()
        for t in (start, end):
            if t:
                parse_time(t)
        if start and end and parse_time(end) <= parse_time(start):
            raise ValueError("End must be after Start")
        roi_txt = self.table.item(r, C_ROI).text().strip()
        roi = None
        if roi_txt:
            vals = [int(float(v)) for v in roi_txt.replace(" ", "").split(",")]
            if len(vals) != 4 or vals[0] >= vals[2] or vals[1] >= vals[3] or min(vals) < 0:
                raise ValueError("ROI must be x1,y1,x2,y2 with x1<x2, y1<y2")
            roi = tuple(vals)
        name = os.path.splitext(os.path.basename(path))[0]
        return Job(path, os.path.join(self.output_dir_getter(), name, "batch_extract"), start, end, roi)

    # ---------------------------------------------------------------- run
    def is_running(self):
        return self.executor is not None

    def start(self):
        if self.is_running():
            return
        rows = [r for r in range(self.table.rowCount()) if not self._row_status(r).startswith("Done")]
        if not rows:
            QMessageBox.information(self, "Batch", "Nothing to run. Add videos first.")
            return
        jobs, errors = [], []
        for r in rows:
            try:
                jobs.append(self._parse_row(r))
            except ValueError as e:
                errors.append(f"Row {r + 1} ({self.table.item(r, C_VIDEO).text()}): {e}")
        if errors:
            QMessageBox.warning(self, "Invalid jobs", "\n".join(errors))
            return
        s = self.current_settings()
        if s.backend == "pyav" and not HAVE_PYAV:
            QMessageBox.warning(self, "PyAV missing", "PyAV is not installed. Run: pip install av")
            return
        if s.roi_track and any(j.roi is None and not j.frames for j in jobs):
            QMessageBox.warning(self, "ROI tracking", "ROI tracking is on, but some rows have no ROI.")
            return
        if s.mine and not s.models:
            QMessageBox.warning(self, "Mining", "Mining criteria are checked but no models are configured.")
            return
        from AIO_QC.core import procs
        share = self.share_cb.currentData()
        sizes = []
        for job in jobs:
            try:
                import cv2 as _cv
                c = _cv.VideoCapture(job.video_path)
                sizes.append((int(c.get(_cv.CAP_PROP_FRAME_WIDTH)) or 1920, int(c.get(_cv.CAP_PROP_FRAME_HEIGHT)) or 1080))
                c.release()
            except Exception:
                sizes.append((1920, 1080))
        auto_workers, threads = procs.plan_batch_workers(sizes, share)
        workers = self.workers_sb.value() or auto_workers
        if self.workers_sb.value():
            threads = max(1, procs.cpu_budget(share) // workers)
        logger.info(f"Batch: {workers} worker(s) x {threads} decode thread(s) "
                    f"({'auto' if not self.workers_sb.value() else 'fixed'}, share {share}, {procs.cpu_count()} cores)")
        self._save_settings(s)
        s = replace(s, det_threads=worker_threads(workers))
        plan = plan_tasks(jobs, workers, self.split_chk.isChecked(), float(self.chunk_sb.value()),
                          no_split=s.roi_track or "{counter" in s.name_template)

        ctx = mp.get_context("spawn")
        self.manager = procs.start_manager(ctx)                  # dies with the app, low priority
        self.queue = self.manager.Queue()
        self.cancel_event = self.manager.Event()
        self.executor = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=procs.worker_init,
                                            initargs=(threads, True))
        self.summary.setText(f"{workers} worker(s) × {threads} thread(s)")
        self.futures, self.tasks, self.tstate = {}, {}, {}
        self.run_t0 = time.monotonic()
        for r in rows:
            self._set_status(r, "Waiting for a worker…")
            self.table.cellWidget(r, C_PROGRESS).setValue(0)
            for c in (C_FPS, C_ETA, C_SAVED):
                self.table.item(r, c).setText("")
            self._set_row_editable(r, False)
        for job_index, job in plan:
            tid = self._next_id
            self._next_id += 1
            self.tasks[tid] = rows[job_index]
            self.tstate[tid] = dict(done=0, total=0, saved=0, finished=False, stats=None, error=None, t0=None)
            self.futures[tid] = self.executor.submit(job_process, tid, job, s, self.queue, self.cancel_event)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.remove_btn.setEnabled(False)
        from AIO_QC.widgets import tasks
        self._task = tasks.start(f"Batch extraction · {len(jobs)} video(s)", cancel=self.cancel)
        self.poll_timer.start()
        logger.info(f"Batch started: {len(jobs)} video(s) -> {len(plan)} task(s), {workers} worker(s), {s}")

    def _set_status(self, r, text, color=None):
        it = self.table.item(r, C_STATUS)
        it.setText(text)
        it.setToolTip(text)
        it.setForeground(color or self.palette().text().color())

    def poll(self):
        touched = set()
        try:
            while not self.queue.empty():
                touched.add(self._handle(self.queue.get_nowait()))
        except Exception:
            pass
        for tid, fut in self.futures.items():
            st = self.tstate[tid]
            if fut.done() and not st["finished"]:
                exc = None if fut.cancelled() else fut.exception()
                st["finished"] = True
                st["error"] = str(exc) if exc else None
                st["cancelled"] = fut.cancelled()
                touched.add(self.tasks[tid])
        for r in touched - {None}:
            self._refresh_row(r)
        self._update_summary()
        if getattr(self, "_task", None) is not None and self.tstate:
            done = sum(st["done"] for st in self.tstate.values())
            total = sum(st["total"] for st in self.tstate.values())
            saved = sum(st.get("saved", 0) or 0 for st in self.tstate.values())
            self._task.update(done / total if total else None, f"{saved:,} saved")
        if all(s["finished"] for s in self.tstate.values()):
            self._finish_run()

    def _handle(self, msg):
        kind, tid = msg[0], msg[1]
        st = self.tstate.get(tid)
        if st is None:
            return None
        if kind == "started":
            st["t0"] = time.monotonic()
        elif kind == "progress":
            st.update(done=msg[2], total=msg[3], saved=msg[4])
        elif kind == "done":
            st.update(finished=True, stats=msg[2])
            st["done"] = st["total"] = max(st["total"], msg[2].get("frames", 0), 1)
        elif kind == "error":
            st.update(finished=True, error=msg[2])
        return self.tasks[tid]

    def _refresh_row(self, r):
        parts = [self.tstate[t] for t, row in self.tasks.items() if row == r]
        done = sum(p["done"] for p in parts)
        total = sum(p["total"] for p in parts)
        started = [p["t0"] for p in parts if p["t0"]]
        bar = self.table.cellWidget(r, C_PROGRESS)
        if all(p["finished"] for p in parts):
            errs = [p["error"] for p in parts if p["error"]]
            stats = None
            for p in parts:
                if p["stats"]:
                    stats = p["stats"] if stats is None else merge_stats(stats, p["stats"])
            if errs:
                self._set_status(r, f"Error: {errs[0]}", RED)
                logger.error(f"Batch job failed: {self.table.item(r, C_VIDEO).text()}: {errs[0]}")
            elif stats is None or stats.get("cancelled") or any(p.get("cancelled") for p in parts):
                self._set_status(r, "Cancelled")
            else:
                bar.setValue(1000)
                bits = [f"saved {stats['saved']}"]
                for k, label in (("duplicates", "dup"), ("cross_duplicates", "in dataset"), ("mined_out", "not mined")):
                    if stats.get(k):
                        bits.append(f"{label} {stats[k]}")
                if stats.get("blurry"):
                    bits.append(f"blurry {stats['blurry']} (neg {stats['negatives']})")
                mined = [f"{k[6:]} {v}" for k, v in stats.items() if k.startswith("mined_") and k != "mined_out"]
                if mined:
                    bits.append("mined: " + ", ".join(mined))
                if stats.get("scenes", 1) > 1:
                    bits.append(f"scenes {stats['scenes']}")
                if len(parts) > 1:
                    bits.append(f"{len(parts)} chunks")
                if any((p["stats"] or {}).get("resumed") for p in parts):
                    bits.append("already complete")
                elif any((p["stats"] or {}).get("resumed_from") for p in parts):
                    bits.append("resumed")
                if stats.get("track_lost_at"):
                    bits.append(f"tracker lost @{stats['track_lost_at']}")
                self._set_status(r, "Done: " + ", ".join(bits), GREEN)
                logger.info(f"Batch job done: {self.table.item(r, C_VIDEO).text()} {stats}")
            if stats:
                self.table.item(r, C_SAVED).setText(str(stats["saved"]))
            self.table.item(r, C_ETA).setText("")
            return
        if started:
            n_run = sum(1 for p in parts if p["t0"] and not p["finished"])
            self._set_status(r, f"Running ({n_run}/{len(parts)} chunks)" if len(parts) > 1 else "Running")
        if total:
            bar.setValue(int(1000 * done / total))
            el = time.monotonic() - min(started) if started else 0
            fps = done / el if el > 0 else 0
            self.table.item(r, C_FPS).setText(f"{fps:.0f}")
            unknown = any(not p["total"] for p in parts)
            self.table.item(r, C_ETA).setText("…" if unknown else _fmt_eta((total - done) / fps if fps else None))
        self.table.item(r, C_SAVED).setText(str(sum(p["saved"] for p in parts)))

    def _update_summary(self):
        n = self.table.rowCount()
        if not self.is_running():
            self.summary.setText(f"{n} video(s) in queue." if n else "Queue is empty. Add or drop videos.")
            return
        states = list(self.tstate.values())
        finished = sum(s["finished"] for s in states)
        done = sum(s["done"] for s in states)
        elapsed = time.monotonic() - self.run_t0
        fps = done / elapsed if elapsed > 0 else 0
        unknown = any(not s["total"] for s in states if not s["finished"])
        remaining = sum(s["total"] - s["done"] for s in states if not s["finished"])
        eta = _fmt_eta(remaining / fps) if fps and not unknown else "estimating…"
        self.summary.setText(f"{finished}/{len(states)} tasks finished · {fps:.0f} frames/s · ETA {eta}")

    def _finish_run(self):
        if getattr(self, "_task", None) is not None:
            self._task.finish()
            self._task = None
        self.poll_timer.stop()
        self.refresh_db_label()
        self._shutdown()
        for r in range(self.table.rowCount()):
            self._set_row_editable(r, True)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.remove_btn.setEnabled(True)
        elapsed = time.monotonic() - self.run_t0
        self._update_summary()
        self.summary.setText(self.summary.text() + f"  ·  finished in {_fmt_eta(elapsed)}")
        logger.info("Batch run finished")

    def _shutdown(self, force=False):
        """force: stop running workers now (closing during a batch) instead of letting them finish."""
        if getattr(self, "_task", None) is not None:
            self._task.finish()
            self._task = None
        if self.executor:
            if force:
                from AIO_QC.core import procs
                procs.terminate_pool(self.executor, grace=2.0)
            else:
                self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
        if self.manager:
            try:
                self.manager.shutdown()
            except Exception:
                pass
            self.manager = None

    def cancel(self):
        if not self.is_running():
            return
        self.cancel_event.set()
        for tid, fut in self.futures.items():
            if fut.cancel():
                self.tstate[tid].update(finished=True, cancelled=True)
        self.cancel_btn.setEnabled(False)
        self.summary.setText("Cancelling…")

    def closeEvent(self, ev):
        if self.is_running():
            if QMessageBox.question(self, "Batch running", "Cancel the running batch and close?") \
                    != QMessageBox.StandardButton.Yes:
                ev.ignore()
                return
            self.cancel()
            deadline = time.monotonic() + 5
            while not all(s["finished"] for s in self.tstate.values()) and time.monotonic() < deadline:
                self.poll()
                time.sleep(0.1)
            self.poll_timer.stop()
            self._shutdown(force=True)     # don't let Python's exit wait for half-finished jobs
            self.start_btn.setEnabled(True)
        ev.accept()
