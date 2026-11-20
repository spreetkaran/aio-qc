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

from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QLabel,
)

from AIO_QC.widgets import motion as anim
from AIO_QC.widgets.testing.lab_common import (
    SerialWorker,
)

from AIO_QC.utils.fx_settings import testing_settings
from AIO_QC.utils.qt_utils import later   # owner-bound timer: never fires after `self` is gone


# Results tabs


# ====================================================================== window
from AIO_QC.widgets.testing.model_lab_common import (  # noqa: F401  (re-exported)
    ImagePreview, NumItem, STEPS, T_CHARTS, T_CLASSES, T_COMPARE, T_CONFUSION, T_FAILURES, T_OVERVIEW, T_PLOTS, T_VIDEOS, _draw_status, _hint, _item, _make_table, _num_item, _release_thread_reader, _scroll, _set_shared_env, _shared_env, _status_icon, _strip_tags, _swatch_icon, _unique_tip, slot_hex)
from AIO_QC.widgets.testing.model_lab_parts.layout import LayoutMixin
from AIO_QC.widgets.testing.model_lab_parts.data import DataMixin
from AIO_QC.widgets.testing.model_lab_parts.models import ModelsMixin
from AIO_QC.widgets.testing.model_lab_parts.run import RunMixin
from AIO_QC.widgets.testing.model_lab_parts.results import ResultsMixin
from AIO_QC.widgets.testing.model_lab_parts.report import ReportMixin
from AIO_QC.widgets.testing.model_lab_parts.review import ReviewMixin


class ModelLab(LayoutMixin, DataMixin, ModelsMixin, RunMixin, ResultsMixin, ReportMixin, ReviewMixin, QMainWindow):
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
        from AIO_QC.utils.qt_utils import fit_size
        fit_size(self, "large")
        anim.fade_on_show(self, 200)

    # ---- plots: every image with a model's predictions drawn, saveable
    PLOT_SIZES = (("Small", QSize(220, 140)), ("Medium", QSize(300, 190)), ("Large", QSize(440, 270)))

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
