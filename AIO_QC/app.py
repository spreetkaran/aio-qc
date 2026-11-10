#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
AIO QC App — annotation quality control for LabelMe and YOLO datasets.

Dependencies: pip install PyQt6 numpy opencv-python loguru
Optional:     pip install natsort pillow pyyaml orjson
"""
from __future__ import annotations

import functools
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QSize, Qt, QThreadPool, QTimer, QUrl
from PyQt6.QtGui import QAction, QActionGroup, QColor, QKeySequence, QPixmap, QTextDocument
from PyQt6.QtWidgets import (
    QApplication, QDialog, QDockWidget, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
    QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from AIO_QC import __appname__, __version__, utils
from AIO_QC.core import AnnotationError, DatasetType, detect_dataset_types, open_dataset
from AIO_QC.core.class_map import load_classes_file
from AIO_QC.core.state import REVIEW_FLAG, REVIEW_FIXED, REVIEW_OK
from AIO_QC.widgets import (
    AmbiguityDialog, Canvas, ClassRemovalDialog, DimAreaDialog, DuplicateDialog, EmptyDialog, FileDock, Filmstrip,
    ImageDuplicatesDialog, IndexWorker, LabelMapDialog, OutlierDialog, PreloadSignals, PreloadTask,
    Session, SmartFilterDialog, StatsDialog, StatusButton, ToolBar, ValidationDialog, load_qimage,
)
from AIO_QC.widgets.filmstrip import WINDOW as FILMSTRIP_WINDOW
from AIO_QC.widgets.crop_gallery import CropGalleryDialog
from AIO_QC.widgets.sequence_dialog import SequenceDialog
from AIO_QC.widgets.similarity_dialog import SimilarityDialog
from AIO_QC.widgets.review_grid import ReviewGridDialog
from AIO_QC.widgets.views_dialog import ViewsDialog
from AIO_QC.widgets.handoff_dialog import ExportBatchDialog, ImportBatchDialog
from AIO_QC.widgets.split_health_dialog import SplitHealthDialog
from AIO_QC.widgets import anim
from AIO_QC.widgets.report_dialog import ReportDialog
from AIO_QC.widgets.guidelines import CanvasGuide, GuidelinesDialog, add_selected as add_guide_example
from AIO_QC.widgets.keymap import KeyMapOverlay, PanelCycler
from AIO_QC.widgets.model_check_dialog import ModelCheckDialog
from AIO_QC.widgets.sampling_dialog import SamplingDialog
from AIO_QC.widgets.coverage_dialog import CoverageDialog
from AIO_QC.widgets.zoom_widget import ZoomWidget
from AIO_QC.widgets.issues_dialog import IssuesDialog
from AIO_QC.widgets.rules_dialog import RulesDialog
from AIO_QC.widgets.image_quality_dialog import ImageQualityDialog, IntegritySweep, image_of, integrity_findings
from AIO_QC.widgets.export_dialog import ExportDialog
from AIO_QC.widgets.flagged_dialog import FlaggedFilesDialog
from AIO_QC.core.image_quality import TITLES as IQ_TITLES
from AIO_QC.widgets.lifecycle import set_exit_reason, stop_everything, stop_threads
from AIO_QC.widgets.editor_session_dialog import EditorSessionDialog
from AIO_QC.widgets.dataset_watcher import DatasetWatcher
from AIO_QC.widgets.toast import ToastManager
from AIO_QC.widgets.ocr_dialog import OcrCheckDialog
from AIO_QC.widgets.text_labels_dialog import TextLabelsDialog
from AIO_QC.widgets.train_size_dialog import TrainSizeDialog
from AIO_QC.core.ocr_service import OcrService
from AIO_QC.widgets.inference_service import FuncWorker, InferenceService
from AIO_QC.widgets.model_setup_dialog import ModelSetupDialog
from AIO_QC.core.env_manager import remove_env

try:
    from natsort import natsorted
except ImportError:  # pragma: no cover
    def natsorted(it, key=None):
        return sorted(it, key=key)


@dataclass(frozen=True)
class QCTool:
    key: str
    title: str
    icon: str           # icon file name in AIO_QC/icons
    shortcut: str
    group: str          # a key of GROUPS
    dialog: type | None
    tip: str
    action: str = ""    # tools that are plain actions (e.g. Label Mapping) reuse that QAction


# Tabs = groups, in workflow order: integrity → names → shapes → model-assisted → sign-off.
GROUPS = [   # (key, tab title, question) — tab icon: icons/group-<key>.png
    ("dataset", "Dataset", "Are the files and images sound?"),
    ("labels", "Labels", "Are the label NAMES right?"),
    ("shapes", "Shapes", "Are the annotations themselves right?"),
    ("models", "Models", "Where do models and labels disagree?"),
    ("review", "Review", "Is it done, and is it good enough?"),
]

# Single source of truth for tabs, menus and shortcuts.
QC_TOOLS = [
    # ---- DATASET
    QCTool("stats", "Dataset Stats", "tool-stats", "Ctrl+Shift+S", "dataset", StatsDialog,
           "Label balance per split, image profile (resolution, aspect, lighting …)"),
    QCTool("validation", "Validation", "tool-validation", "Ctrl+Shift+V", "dataset", ValidationDialog,
           "File & geometry integrity: pairing, schema, class ids, points, image size"),
    QCTool("empty", "Empty Annotations", "tool-empty", "Ctrl+Shift+E", "dataset", EmptyDialog,
           "Annotations without shapes / images without a label file"),
    QCTool("image_dups", "Duplicate Images", "tool-image-dups", "Ctrl+Shift+I", "dataset", ImageDuplicatesDialog,
           "Exact / near-duplicate images and split leakage"),
    # ---- LABELS (names)
    QCTool("image_quality", "Image Quality", "tool-quality", "Ctrl+Shift+Q", "dataset", ImageQualityDialog,
           "Corrupt / truncated images, EXIF rotation and size disagreements; blur and exposure (information)"),
    QCTool("split_health", "Split Health", "tool-splits", "Ctrl+Alt+S", "dataset", SplitHealthDialog,
           "Every class in every split with enough examples, val / test like train, no near-identical images across "
           "splits (leakage) — with suggested file moves (undoable)"),
    QCTool("ambiguity", "Label Ambiguities", "tool-ambiguity", "Ctrl+Shift+A", "labels", AmbiguityDialog,
           "Find label-name variants, misspellings and unknown labels"),
    QCTool("label_map", "Label Mapping", "tool-label-map", "Ctrl+M", "labels", None,
           "The one place to rename / merge labels (view-only or written)", action="label_map"),
    QCTool("guidelines", "Guidelines", "tool-guidelines", "Ctrl+Alt+G", "labels", GuidelinesDialog,
           "Write the labelling rule per class / QC rule with ✓ good and ✗ bad example crops; shown next to the "
           "selected shape and in the tools, exported for annotators"),
    QCTool("text_labels", "Text Labels", "tool-text-labels", "Ctrl+Alt+T", "labels", TextLabelsDialog,
           "Check the text stored in your annotations without OCR: formats and known texts learned from the dataset, "
           "your validators (optional), what other frames of the same object say — with the correction applied in "
           "one click"),
    QCTool("remove_class", "Remove / Re-order Classes", "tool-remove-class", "Ctrl+Shift+R", "labels", ClassRemovalDialog,
           "Delete classes from every file; YOLO: renumber ids and rewrite the class map"),
    # ---- SHAPES (annotations)
    QCTool("crop_gallery", "Crop Gallery", "tool-crop-gallery", "Ctrl+Shift+G", "shapes", CropGalleryDialog,
           "Every instance of a label as a grid of crops — spot wrong labels and junk boxes fast"),
    QCTool("duplicates", "Duplicate Shapes", "tool-dup-shapes", "Ctrl+D", "shapes", DuplicateDialog,
           "Same object annotated twice (exact or IoU)"),
    QCTool("outliers", "Outliers", "tool-outliers", "Ctrl+Shift+O", "shapes", OutlierDialog,
           "Shapes whose size is statistically unusual for their class"),
    QCTool("sequence", "Sequence Check", "tool-sequence", "Ctrl+Shift+J", "shapes", SequenceDialog,
           "Consecutive frames compared with their neighbours: objects missing for a frame, label flips, size / "
           "position spikes, attribute changes, one-frame objects — missing boxes can be added"),
    QCTool("dim_area", "Dim / Area", "tool-dim-area", "Ctrl+Shift+D", "shapes", DimAreaDialog,
           "Shapes inside width / height / area ranges you define"),
    QCTool("train_size", "Training Size", "tool-train-size", "Ctrl+Alt+Z", "shapes", TrainSizeDialog,
           "Every object measured at the training size (longer side = imgsz) and after scale augmentation: objects "
           "too small to learn, and per class the imgsz that keeps 95 % of them big enough"),
    # ---- MODELS
    QCTool("rules", "Relationship Rules", "tool-rules", "Ctrl+Shift+K", "shapes", RulesDialog,
           "How annotations relate: inside / contains, counts, requires / excludes, overlaps, size, shape type, "
           "attributes, groups, one class per image — or let AIO QC suggest rules from the data"),
    QCTool("model_setup", "Models & Environment", "tool-model-setup", "Ctrl+Alt+N", "models", None,
           "Python env, models, class maps, device, imgsz, ONNX / TensorRT", action="model_setup"),
    QCTool("ocr_check", "OCR Check", "tool-ocr", "Ctrl+Shift+T", "models", OcrCheckDialog,
           "Read the text in text-class boxes (PaddleOCR / RapidOCR / EasyOCR / Tesseract from your environment) and "
           "check it: validators, readability, box fit, stored text, identity leakage"),
    QCTool("similarity", "Visual Similarity", "tool-similar", "Ctrl+Shift+Y", "models", SimilarityDialog,
           "Find mislabels by appearance with DINOv3 / SigLIP2 / CLIP (folder, zip or download): likely "
           "mislabels, atypical objects, look-alikes with different labels"),
    QCTool("model_check", "Model Check", "tool-model-check", "Ctrl+Shift+M", "models", ModelCheckDialog,
           "Findings + analysis: report card, confusion, error types, box quality, slices, label-error estimate"),
    QCTool("testing", "Testing", "tool-testing", "Ctrl+Alt+L", "models", None,
           "Measure and compare detection models (formerly the Frame Extractor's Model Lab): this dataset, a test "
           "set from videos, or videos only — leaderboard, charts, confusion, per class, failures, plots, A/B, "
           "video behaviour, reports; label fixes go through AIO QC", action="testing"),
    # ---- REVIEW
    QCTool("inbox", "Issues Inbox", "tool-inbox", "Ctrl+I", "review", IssuesDialog,
           "Every tool's findings in one place, a risk score per file, and one review queue — worst first"),
    QCTool("export", "Export & Split", "tool-export", "Ctrl+Shift+X", "review", ExportDialog,
           "Build train / val / test splits (duplicates and video frames kept together, rare cases kept in train), "
           "see how they look, and export a clean dataset (YOLO, LabelMe, COCO, class folders)"),
    QCTool("sampling", "QC Sampling", "tool-sampling", "Ctrl+Shift+P", "review", SamplingDialog,
           "Review a random / stratified sample and estimate the dataset error rate"),
    QCTool("fix_editor", "Fix in Editor", "tool-editor", "Ctrl+E", "review", EditorSessionDialog,
           "Open flagged files in labelme / labelImg; every save is checked, written back and marked Fixed"),
    QCTool("coverage", "Coverage Map", "tool-coverage", "Ctrl+Shift+C", "review", CoverageDialog,
           "Every file as a square coloured by review state — see what is QC'ed, click to open"),
    QCTool("smart_filter", "Smart Filter", "tool-smart-filter", "Ctrl+Shift+F", "review", SmartFilterDialog,
           "Build a file subset to review by labels and object counts"),
    QCTool("review_grid", "Review Grid", "tool-review-grid", "Ctrl+Shift+W", "review", ReviewGridDialog,
           "Whole images with their annotations, a page at a time: X the bad ones, Space marks the rest OK"),
    QCTool("views", "Saved Views", "tool-saved-filter", "Ctrl+Shift+B", "review", ViewsDialog,
           "Named file queries (labels, counts, review state, findings, names …) with live counts and a "
           "plain-language explanation; show in the file list or review in the grid"),
    QCTool("report", "QC Report", "tool-report", "Ctrl+Alt+R", "review", ReportDialog,
           "One click: a self-contained HTML / PDF of class balance, review coverage, findings, the QC sample "
           "estimate, the riskiest files (thumbnails) and what changed since the last report"),
    QCTool("handoff_export", "Export Hand-off Batch", "tool-handoff", "Ctrl+Alt+E", "review", ExportBatchDialog,
           "Send flagged / risky files to an annotator: their own folder with notes, findings and hint images"),
    QCTool("handoff_import", "Import Hand-off Batch", "import", "Ctrl+Alt+I", "review", ImportBatchDialog,
           "Merge a returned batch: each file compared three ways (as sent · as returned · now), conflicts shown, "
           "what changed listed, every write backed up"),
]
MAX_RECENT = 10

PIXMAP_CACHE_BYTES = 768 * 1024 * 1024


def _find_ffmpeg(user_path=""):
    """ffmpeg for Testing's annotated-video export (the Frame Extractor's lookup: its setting, PATH, bundled)."""
    try:
        from AIO_QC.tools.frame_extractor.core.preview_pipe import find_ffmpeg
        return find_ffmpeg(user_path)
    except Exception:
        return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._update_title()
        self.setWindowIcon(utils.appIcon())
        screen = QApplication.primaryScreen().availableGeometry()
        w = int(screen.width() * 0.75)
        self.resize(w, int(w * 0.55))
        self.settings = utils.app_settings()
        from AIO_QC.widgets.status_fields import PaintStatusBar
        self.setStatusBar(PaintStatusBar(self))           # pointer · selection · image size, then messages
        from AIO_QC.widgets import motion
        anim.ENABLED = motion.enabled()                    # one switch for every animation (ui/animations)
        from AIO_QC.core import backups as _backups
        _backups.EXACT_IMAGE_DATA = self.settings.value("storage/exact_image_data", False, type=bool)
        self.inference = InferenceService(self.settings, self)       # shared model server (Model Setup / Check)
        self._cleanup_temp_envs()

        self.session: Session | None = None
        self.user_classes: list[str] | None = None
        self.user_classes_path: Path | None = None
        self.current_key: str | None = None
        self.current_ann = None
        self._pending_shape: int | None = None
        self._pending_extras: tuple = (None, None)
        self._index_started = 0.0
        self.label_colors: dict[str, QColor] = {}
        self.visible_labels: set[str] = set()
        self.hidden_labels: set[str] = set()      # survives label-list rebuilds
        self.error_log: list[str] = []
        self.open_dialogs: dict[str, QDialog] = {}
        self.editor_dialog = None                  # active external-editor session window (Fix in Editor)
        self.watcher = self.tracker = None         # live updates: background scanner + change tracker
        self.sweep = None                          # background image-integrity check
        self.broken: dict = {}                     # key -> [(code, severity, message)] (integrity errors)
        self._issues_save_timer = QTimer(self, singleShot=True, interval=2000,
                                         timeout=lambda: self.session.issues.save() if self.session else None)
        self.ocr = OcrService(log=lambda line: logger.debug(f"[ocr] {line}"))   # OCR engine process (warm)
        from AIO_QC.core.embed_service import EmbedService
        self.embedder = EmbedService(log=lambda line: logger.debug(f"[embed] {line}"))   # Visual Similarity model
        self.embed_index = None                    # vectors of the last Visual Similarity run (Crop Gallery sorts)
        self._external_log: list = []
        self.pixmap_cache = utils.LRUCache(capacity=24, max_bytes=PIXMAP_CACHE_BYTES)
        self.generation = 0                       # invalidates stale background results
        self._inflight: set[str] = set()
        self.index_worker: IndexWorker | None = None
        self.preload_signals = PreloadSignals()
        self.preload_signals.loaded.connect(self._on_preloaded)
        self._labels_refresh = QTimer(self, singleShot=True, interval=300, timeout=self.refresh_label_list)
        self._focus_state = None                  # window layout saved while Focus Mode hides the panels
        self._palette = None

        self._build_canvas()
        self._build_docks()
        self._build_actions()
        self._build_toolbar()
        self._build_qc_panel()
        self._build_menus()
        self._build_statusbar()
        for act in (self.a.fit, self.a.actual):     # pressed look = "the view is like this now"
            act.setCheckable(True)
            act.triggered.connect(lambda _c=False: QTimer.singleShot(0, self._sync_zoom_buttons))
        from AIO_QC.widgets.inline_editor import InlineEditor
        self.inline = InlineEditor(self)            # edits made directly on the image
        self.canvas_guide = CanvasGuide(self)       # the selected class's guideline, over the image
        from AIO_QC.widgets.sketch import RoiMode, SketchMode
        self.sketch = SketchMode(self)              # P: draw on the image to explain something
        self.roi = RoiMode(self)                    # O: snipping-tool style ROI boxes with real coordinates
        self.panel_cycler = PanelCycler(self)       # F6
        self._keymap = None
        self._restore_settings()
        self._orient_filmstrip()
        self._watch_theme()
        self.toast = ToastManager(self)             # colour-coded pop-up notifications
        QApplication.instance().applicationStateChanged.connect(self._on_app_state)
        if not self.settings.value("get_started/shown", False, type=bool) or \
                self.settings.value("get_started/at_start", False, type=bool):
            self.settings.setValue("get_started/shown", True)
            self.settings.setValue("guide_shown", True)
            QTimer.singleShot(450, self.show_get_started)    # first start: the guided overview
        self.set_dataset_actions_enabled(False)
        self.refresh_status_buttons()

    # =====================================================================
    # UI construction
    # =====================================================================
    def _build_canvas(self):
        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)
        self.canvas.zoomFinished.connect(self.updateZoomSpinbox)
        self.canvas.zoomChanged.connect(self._zoom_live)           # every click / wheel tick / glide frame
        self.canvas.shapeSelected.connect(self.onShapeSelected)
        self.canvas.selectionCleared.connect(self.onSelectionCleared)
        self.canvas.modeChanged.connect(self._on_mode_changed)

    def _dock(self, title, widget, name, area=Qt.DockWidgetArea.RightDockWidgetArea):
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self):
        # Dataset labels (visibility + counts)
        self.unique_labels = QListWidget()
        self.unique_labels.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.unique_labels.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.unique_labels.itemChanged.connect(self.toggle_label_visibility)
        self.unique_labels.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.unique_labels.customContextMenuRequested.connect(self._labels_list_menu)
        self.unique_labels_dock = self._dock("Labels", self.unique_labels, "labels_list_dock")

        # Shapes of the current file
        self.annotations_list = QListWidget()
        self.annotations_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.annotations_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.annotations_list.currentRowChanged.connect(self._on_annotation_row)
        self.annotations_list.itemClicked.connect(self._on_annotation_clicked)
        self._ann_row_changed = False
        self.annotations_list.itemDoubleClicked.connect(
            lambda it: self.canvas.focus_shape(it.data(Qt.ItemDataRole.UserRole)))
        self.annotations_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.annotations_list.customContextMenuRequested.connect(self._annotations_menu)
        # objectName stays "labels_dock" so saved window layouts keep the dock position
        self.annotations_dock = self._dock("Annotations", self.annotations_list, "labels_dock")

        # Files
        self.file_dock = FileDock()
        self.file_dock.view.customContextMenuRequested.connect(self._file_list_menu)
        self.file_dock.currentKeyChanged.connect(self.show_sample)
        self.file_dock.parked.connect(lambda _k: self._update_position())
        self.file_list_dock = self._dock("File List", self.file_dock, "file_dock")

        # History of the current file (audit trail + saved versions)
        from AIO_QC.widgets.history_panel import HistoryPanel
        self.history_panel = HistoryPanel(self)
        self.history_dock = self._dock("History", self.history_panel, "history_dock")
        self.tabifyDockWidget(self.annotations_dock, self.history_dock)
        self.annotations_dock.raise_()
        self.history_dock.hide()

        # Thumbnails
        self.filmstrip = Filmstrip()
        self.filmstrip.keyActivated.connect(lambda k: self.file_dock.select_key(k))
        self.filmstrip_dock = self._dock("Thumbnails", self.filmstrip, "filmstrip_dock",
                                         Qt.DockWidgetArea.BottomDockWidgetArea)
        self.filmstrip_dock.visibilityChanged.connect(lambda v: v and self._update_filmstrip())
        self.filmstrip_dock.dockLocationChanged.connect(self._orient_filmstrip)
        self.filmstrip_dock.topLevelChanged.connect(lambda _floating: self._orient_filmstrip())
        self.setAcceptDrops(True)

    def _build_actions(self):
        A = functools.partial(utils.newAction, self)
        self.a = utils.struct(
            open_dir=A("Open &Directory…", self.open_dir, "Ctrl+O", icon="open",
                       tip="Open a LabelMe / YOLO / YOLO-split dataset root"),
            reload=A("&Reload Dataset", self.reload_dataset, "F5", tip="Re-scan the dataset from disk"),
            change_class=A("Change &Class…", self.change_class, "C", icon="tool-class", enabled=False,
                           tip="Classification datasets: give the selected image(s) another class "
                               "(class folders: the image is moved; LabelMe flags: the flags are rewritten)"),
            watch=A("Watch Dataset for Outside Changes", self._toggle_watch, checkable=True,
                    checked=self.settings.value("watch/enabled", True, type=bool),
                    tip="Pick up files changed, added or removed by other programs (re-read only, never written)"),
            load_classes=A("Load &Classes…", self.load_classes_file, "Ctrl+L", icon="list",
                           tip="Load the reference class list (.txt one per line, or .yaml names)"),
            open_trash=A("Open Trash Folder", self.open_trash, tip="Show files moved by QC deletes"),
            clear_cache=A("Clear Dataset Cache", self.clear_cache,
                          tip="Delete .aio_qc/cache.sqlite contents (labels, image hashes, model predictions)"),
            undo_edit=A("Undo", lambda: self.inline.undo(), "Ctrl+Z", icon="undo", enabled=False,
                        tip="Undo the last edit made on the image (exact bytes; refused if the file changed since)"),
            redo_edit=A("Redo", lambda: self.inline.redo(), ["Ctrl+Shift+Z", "Ctrl+Y"], enabled=False,
                        tip="Redo the last undone edit"),
            history=A("File History", self._show_history, "Ctrl+H", icon="history",
                      tip="Every change to the open file — who, when, which tool — and its saved versions (revert)"),
            guide_good=A("Add as Good Example", lambda: add_guide_example(self, "good"), "G", icon="tool-guidelines",
                         tip="The selected shape becomes a ✓ good example in its class's guideline"),
            guide_bad=A("Add as Bad Example", lambda: add_guide_example(self, "bad"), "Shift+G",
                        icon="tool-guidelines", tip="The selected shape becomes a ✗ bad example in its class's guideline"),
            guide_cards=A("Guideline Cards", lambda on: self.canvas_guide.set_enabled(on), checkable=True,
                          checked=self.settings.value("ui/guide_cards", True, type=bool),
                          tip="Show a class's guideline next to the selected shape"),
            new_box=A("New Box", self._start_new_box, "B",
                      tip="Draw a new box on the image (then pick its label)"),
            compact_toolbar=A("Compact Toolbar (icons only)", self._set_toolbar_compact, checkable=True,
                              checked=self.settings.value("ui/compact_toolbar", True, type=bool),
                              tip="Icons without text in the toolbar — more room for the image (names in tooltips)"),
            animations=A("Animations", self._set_animations, checkable=True,
                         checked=anim.ENABLED,
                         tip="Smooth motion for panels, drawers, dialogs and the tour (off = everything changes at once)"),
            focus=A("Focus Mode", self.toggle_focus_mode, ["F11", "Ctrl+Alt+F"], checkable=True,
                    tip="Only the image: hide every panel and the toolbar; the same key restores your layout"),
            palette=A("Command Palette…", self.open_palette, "Ctrl+K",
                      tip="Run any command by typing a few letters (tools, review marks, view options …)"),
            pin_qc=A("Pin QC Panel", lambda on: self.tabs.set_pinned(on), checkable=True,
                     tip="Keep the QC tools open under the tabs (unpinned: they drop down over the image)"),
            storage=A("Storage…", self.open_storage,
                      tip="What AIO QC keeps in .aio_qc (cache, backups, sessions) — export, delete, retention"),
            exit=A("E&xit", self.close, "Ctrl+Q", icon="quit"),
            # modes
            drag=A("Drag", lambda _c=False: self.canvas.setMode(Canvas.DRAG), "H", icon="hand",
                   tip="Drag mode: pan the image", checkable=True, checked=True),
            select=A("Select", lambda _c=False: self.canvas.setMode(Canvas.SELECT), "V", icon="cursor",
                     tip="Select mode: click shapes to select (hold Space to pan)", checkable=True),
            # navigation / view
            next=A("Next Image", lambda: self.file_dock.step(1), ["D", "PgDown"], icon="next", tip="Next image"),
            prev=A("Prev Image", lambda: self.file_dock.step(-1), ["A", "PgUp"], icon="prev", tip="Previous image"),
            fit=A("Fit Window", self.canvas.fit_window, "Ctrl+0", icon="fit-window", tip="Fit image to window"),
            actual=A("Actual Size", self.canvas.actual_size, "Ctrl+1", icon="zoom", tip="100 %"),
            zoom_in=A("Zoom In", lambda: self.canvas.zoom_step(1), ["Ctrl++", "Ctrl+="],
                      icon="fluent-zoom-in", tip="Zoom in around the centre"),
            zoom_out=A("Zoom Out", lambda: self.canvas.zoom_step(-1), "Ctrl+-",
                       icon="fluent-zoom-out", tip="Zoom out around the centre"),
            tags=A("Label Tags", self._toggle_tags, "T", icon="tag", tip="Show label text on shapes", checkable=True, checked=True),
            fill=A("Fill Shapes", self._toggle_fill, "F", icon="fill", tip="Semi-transparent fill", checkable=True),
            pixel_view=A("Show Pixels When Zoomed In", self._set_pixel_view, checkable=True,
                         checked=self.settings.value("view/pixel_view", False, type=bool),
                         tip="From 300 %: hard pixel squares (nearest neighbour) instead of the smooth, Windows "
                             "Photos-like high-quality enlargement"),
            pred_layer=A("Model Predictions", self.set_pred_layer, "Shift+P", icon="tool-model-check",
                         tip="Draw the models' predictions (dashed) on every image — from Model Check's last run, "
                             "at its explorer's thresholds", checkable=True,
                         checked=self.settings.value("view/pred_layer", False, type=bool)),
            all_labels=A("Show All Labels", self.toggle_all_labels, "Ctrl+Shift+H", icon="eye",
                         tip="Toggle all labels", checkable=True, checked=True),
            search=A("Search Files", self._focus_search, "Ctrl+F"),
            clear_selection=A("Clear Selection", lambda: self.canvas.clearSelection(), "Esc",
                              tip="Deselect the shape (any mode)"),
            next_shape=A("Next Shape", lambda: self.canvas.cycle_shape(1), tip="Select next shape (Tab in canvas)"),
            prev_shape=A("Previous Shape", lambda: self.canvas.cycle_shape(-1),
                         tip="Select previous shape (Shift+Tab in canvas)"),
            # tools
            rename=A("Rename / Merge Label…", self._rename_selected_label, "Ctrl+R", icon="edit",
                     tip="Opens Label Mapping focused on the selected label (the single place for renames)"),
            model_setup=A("Models & Environment…", self.open_model_setup, "Ctrl+Alt+N", icon="tool-model-setup",
                          tip="Python environment, models, class mapping, device and image size for Model Check"),
            label_map=A("Label Mapping…", self.open_label_map, "Ctrl+M", icon="tool-label-map",
                        tip="Map labels to other names for viewing and checking (files unchanged)"),
            # review
            mark_ok=A("Mark OK", lambda: self.mark_review(REVIEW_OK), "S", icon="review-ok", tip="Mark current file as reviewed OK"),
            mark_flag=A("Flag", lambda: self.mark_review(REVIEW_FLAG), "X", icon="review-flag", tip="Flag current file for rework"),
            flag_note=A("Flag with note…", self.flag_with_note, "Shift+X", icon="review-note", tip="Flag and write a note"),
            mark_clear=A("Clear mark", lambda: self.mark_review(None), "U", icon="review-clear", tip="Remove the review mark"),
            next_unreviewed=A("Next unreviewed", self.goto_next_unreviewed, "N", icon="review-next",
                              tip="Jump to the next file without a review mark"),
            auto_advance=A("Auto-advance after marking",
                           lambda on: self.settings.setValue("review/auto_advance", bool(on)),
                           tip="Go to the next file after ✓ / ⚑", checkable=True,
                           checked=self.settings.value("review/auto_advance", True, type=bool)),
            export_review=A("Export Review Report…", self.export_review, icon="export", tip="CSV of every file's review state"),
            save_flagged=A("Save Flagged Files…", self.save_flagged, icon="review-flag",
                           tip="Copy the flagged files to a folder, or save their names to a text file"),
            reset=A("Reset View", self.reset_all, "R", icon="undo", tip="All labels visible, filters cleared, fit"),
            logs=A("Logs", self.show_log, "Ctrl+Shift+L", icon="logs", tip="Errors collected during this session",
                   enabled=False),
            sketch=A("Sketch on Image", lambda: self.sketch.toggle(), "P", icon="tool-sketch", checkable=False,
                     tip="Draw on the image (pen, highlighter, shapes, arrows, text, eraser) to explain something; "
                         "labels are unticked — tick the ones you want; copy / save the result"),
            roi=A("ROI Capture", lambda: self.roi.toggle(), "O", icon="tool-roi",
                  tip="Draw boxes like a snipping tool and keep their real image coordinates (2 decimals) per image; "
                      "copy, export CSV / TXT / JSON, save crops"),
            data_prep=A("Data Prep…", self.open_data_prep, "Ctrl+Alt+P", icon="tool-dataprep",
                        tip="Raw data → dataset: unzip, combine & flatten, orphans, integrity, exact duplicates, resize "
                            "(no dataset needed; every run can be undone)"),
            testing=A("Testing…", lambda: self.open_testing(), "Ctrl+Alt+L", icon="tool-testing",
                      tip="Measure and compare detection models on this dataset, a test set or videos"),
            next_mode=A("Next Mode", lambda: self._shell_call("cycle"), "Ctrl+Alt+M",
                        tip="Frame Extractor → Image QC → Data Studio"),
            to_video=A("Switch to Frame Extractor", lambda: self._shell_call("switch", "video"),
                       icon="mode-video", tip="The video mode of AIO QC: play, mark and extract frames for annotation"),
            source_video=A("Show Source Video in Frame Extractor", lambda: self.show_source_video(),
                           icon="mode-video", tip="The video this image was extracted from, at its frame, in the "
                                                  "Frame Extractor"),
            to_studio=A("Switch to Data Studio", lambda: self._shell_call("switch", "data"),
                        icon="mode-data", tip="The table-data mode of AIO QC: CSV / Excel / Parquet analysis, "
                                              "cleaning, features, charts, PySpark"),
            ann_table=A("Annotations → Data Studio", lambda: self._shell_call("annotations_to_studio"),
                        icon="mode-data", tip="One row per annotation shape of this dataset, opened as a table in "
                                              "Data Studio (sizes, classes per split, review state, attributes …)"),
            get_started=A("Get Started", self.show_get_started, icon="get-started",
                          tip="The guided overview: what the app does, the window, the workflow, team features, keys"),
            guide=A("User Guide", self.show_guide, "F1", icon="info",
                    tip="What the app is for and how to use it effectively"),
            shortcuts=A("Keyboard Shortcuts", self.show_shortcuts, "Ctrl+F1"),
            keymap=A("Key Map", self.toggle_keymap, "?", icon="keyboard",
                     tip="Every shortcut over the window — type to filter, click to run (?)"),
            next_panel=A("Next Panel", lambda: self.panel_cycler.cycle(1), "F6",
                         tip="Move the keyboard focus to the next panel"),
            prev_panel=A("Previous Panel", lambda: self.panel_cycler.cycle(-1), "Shift+F6"),
            tool_map=A("Which tool for what?", self.show_tool_map, "Shift+F1",
                       tip="One responsibility per tool — where each kind of problem is handled"),
        )
        group = QActionGroup(self)
        group.setExclusive(True)
        group.addAction(self.a.drag)
        group.addAction(self.a.select)
        self.qc_actions: dict[str, QAction] = {}
        for t in QC_TOOLS:
            if t.action:                      # e.g. Label Mapping: same QAction everywhere, one shortcut
                act = getattr(self.a, t.action)
                act.setText(f"{t.title}…")
                utils.set_icon(act, t.icon)
            else:
                act = A(t.title, functools.partial(self.open_tool, t.key), t.shortcut, icon=t.icon, tip=t.tip)
            self.qc_actions[t.key] = act

    def _build_toolbar(self):
        self.toolbar = ToolBar("Quick Actions")
        self.toolbar.setObjectName("main_toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)   # labelme style
        self.toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        self.open_dir_btn = StatusButton("Open Dir", icon="open", slot=self.open_dir)
        self.open_dir_btn.set_base_tooltip("Open dataset root: LabelMe dir, YOLO root or YOLO split root (Ctrl+O)")
        self.load_classes_btn = StatusButton("Load Classes", icon="list", slot=self.load_classes_file)
        self.load_classes_btn.set_base_tooltip("Class list (Ctrl+L): classes.txt (one per line) or .yaml.\n"
                                               "YOLO: dataset.yaml is used automatically; classes.txt only if it is absent.")
        self.dataset_label = QLabel("  No dataset  ")
        utils.style(self.dataset_label, "color:@muted;")

        self.zoom_spin = ZoomWidget(Canvas.MIN_ZOOM, Canvas.MAX_ZOOM)      # "Zoom" + level to one decimal
        self.zoom_spin.setEnabled(False)
        self.zoom_spin.zoomRequested.connect(self.onZoomSpinboxChanged)
        from AIO_QC.widgets.zoom_widget import ZoomSlider
        self.zoom_slider = ZoomSlider(Canvas.MIN_ZOOM, Canvas.MAX_ZOOM)    # Paint-style: − ━━●━━ +
        self.zoom_slider.zoomRequested.connect(self._zoom_from_slider)

        self.toolbar.addWidget(self.open_dir_btn)
        self.toolbar.addWidget(self.load_classes_btn)
        self.toolbar.addWidget(self.dataset_label)
        self.toolbar.addSeparator()
        utils.addActions(self.toolbar, (self.a.prev, self.a.next, None, self.a.drag, self.a.select, None,
                                        self.zoom_spin, self.a.fit, self.a.actual, self.a.zoom_out,
                                        self.zoom_slider, self.a.zoom_in,
                                        None, self.a.reset, self.a.logs))
        self.toolbar.visibilityChanged.connect(lambda v: self.a_toolbar.setChecked(v))
        self.canvas.pixmap_item.pixel_view = self.a.pixel_view.isChecked()
        from AIO_QC.widgets.zoom_widget import HoldToZoom
        self._hold_zoom = []
        for act, d in ((self.a.zoom_in, 1), (self.a.zoom_out, -1)):      # click = one step · hold = smooth zoom
            for btn in (self.toolbar.widgetForAction(a) for a in self.toolbar.actions()):
                if getattr(btn, "defaultAction", lambda: None)() is act:
                    self._hold_zoom.append(HoldToZoom(self.canvas, btn, d))

    # ------------------------------------------------------------------ Display tab
    DISPLAY_DEFAULTS = {"brightness": 0, "contrast": 0, "gamma": 100, "auto": False, "fill": 55, "outline": 100, "tag": 100}

    def _build_display_tab(self) -> QWidget:
        """
        View-only adjustments that exist nowhere else (the old View tab only repeated toolbar actions):
        image brightness / contrast / gamma / auto-contrast, annotation fill opacity and outline width.
        Files are never changed; values are remembered between sessions.
        """
        from PyQt6.QtWidgets import QCheckBox, QSlider
        w = QWidget()
        lay = QHBoxLayout(w)
        st = self.settings
        self.disp = {}

        def slider(key, lo, hi, tip, fmt):
            box = QVBoxLayout()
            lab = QLabel()
            utils.style(lab, "color: @muted;")
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setFixedWidth(130)
            sl.setToolTip(tip)
            sl.setValue(int(st.value(f"display/{key}", self.DISPLAY_DEFAULTS[key])))
            upd = lambda v, l=lab: l.setText(fmt(v))
            upd(sl.value())
            sl.valueChanged.connect(upd)
            sl.valueChanged.connect(lambda _v: self._display_timer.start())
            box.addWidget(lab)
            box.addWidget(sl)
            lay.addLayout(box)
            self.disp[key] = sl

        title = QLabel("<b>Image</b><br><span style='color:gray'>view only</span>")
        lay.addWidget(title)
        slider("brightness", -100, 100, "Brighten / darken the displayed image", lambda v: f"Brightness {v:+d}")
        slider("contrast", -100, 100, "More / less contrast in the displayed image", lambda v: f"Contrast {v:+d}")
        slider("gamma", 30, 300, "Gamma > 1 lifts shadows (low-light frames)", lambda v: f"Gamma {v / 100:.2f}")
        self.disp_auto = QCheckBox("Auto contrast")
        self.disp_auto.setToolTip("Local contrast enhancement (CLAHE) — great for dark / hazy images")
        self.disp_auto.setChecked(st.value("display/auto", False, type=bool))
        self.disp_auto.toggled.connect(lambda _v: self._display_timer.start())
        lay.addWidget(self.disp_auto)
        lay.addWidget(utils.custom_v_separator())
        lay.addWidget(QLabel("<b>Annotations</b>"))
        slider("fill", 0, 255, "Opacity of the shape fill (when Fill Shapes is on)", lambda v: f"Fill opacity {v * 100 // 255}%")
        slider("outline", 50, 300, "Outline thickness", lambda v: f"Outline ×{v / 100:.1f}")
        slider("tag", 75, 200, "Size of the class-name labels on the shapes (they keep this size at every zoom)",
               lambda v: f"Label size {v}%")
        lay.addWidget(self._tool_button(self.a.fill))
        lay.addWidget(self._tool_button(self.a.tags))
        lay.addWidget(utils.custom_v_separator())
        reset = utils.newButton("Reset", icon="undo", slot=self._reset_display, tip="Back to the original image and default styling")
        lay.addWidget(reset)
        lay.addStretch(1)
        self._display_timer = QTimer(self, singleShot=True, interval=60, timeout=self._apply_display)
        QTimer.singleShot(0, self._apply_display)
        return w

    def _apply_display(self):
        v = {k: sl.value() for k, sl in self.disp.items()}
        auto = self.disp_auto.isChecked()
        for k, val in v.items():
            self.settings.setValue(f"display/{k}", val)
        self.settings.setValue("display/auto", auto)
        self.canvas.display.update(brightness=v["brightness"], contrast=v["contrast"], gamma=v["gamma"] / 100, auto=auto)
        self.canvas._render_display()
        if (self.canvas.fill_alpha, self.canvas.outline_scale, getattr(self.canvas, "tag_scale", 1.0)) != \
                (v["fill"], v["outline"] / 100, v["tag"] / 100):
            self.canvas.set_annotation_style(fill_alpha=v["fill"], outline_scale=v["outline"] / 100,
                                             tag_scale=v["tag"] / 100)
        active = not self.canvas.display_is_neutral()
        self.tabs.setTabText(len(GROUPS), "Display •" if active else "Display")
        self.tabs.setTabToolTip(len(GROUPS), "Image adjustments are active (view only)" if active else
                                "View-only image adjustments and annotation styling")

    def _reset_display(self):
        for k, sl in self.disp.items():
            sl.blockSignals(True)
            sl.setValue(self.DISPLAY_DEFAULTS[k])
            sl.blockSignals(False)
            sl.valueChanged.emit(sl.value())
        self.disp_auto.setChecked(False)
        self._apply_display()

    def _tool_button(self, action: QAction) -> QToolButton:
        b = QToolButton()
        b.setDefaultAction(action)
        if action.icon().isNull():
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            b.setMinimumHeight(30)
        else:                                   # same look as the main toolbar (labelme style)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            b.setIconSize(QSize(24, 24))
        return b

    def _build_qc_panel(self):
        """One tab per GROUP (workflow order), then View. Tabs, QC menu and Review menu share QC_TOOLS."""
        from AIO_QC.widgets.ribbon import Ribbon
        self.tabs = Ribbon(self.settings)              # tab row; tools drop down over the image unless pinned
        for key, title, question in GROUPS:
            tab = QWidget()
            lay = QHBoxLayout(tab)
            lay.setContentsMargins(6, 2, 6, 2)
            for t in QC_TOOLS:
                if t.group == key:
                    lay.addWidget(self._tool_button(self.qc_actions[t.key]))
            if key == "review":
                for act in (None, self.a.mark_ok, self.a.mark_flag, self.a.flag_note, self.a.mark_clear, None,
                            self.a.next_unreviewed, self.a.auto_advance, None, self.a.save_flagged,
                            self.a.export_review):
                    lay.addWidget(utils.custom_v_separator() if act is None else self._tool_button(act))
                self.review_progress = QLabel("")
                utils.style(self.review_progress, "color:@muted; padding-left:10px;")
                lay.addWidget(self.review_progress)
            lay.addStretch(1)
            hint = QLabel(question)
            utils.style(hint, "color:@faint; font-style:italic;")
            lay.addWidget(hint)
            i = self.tabs.addTab(tab, utils.newIcon(f"group-{key}"), title)
            self.tabs.setTabToolTip(i, question)
        self.tabs.addTab(self._build_display_tab(), utils.newIcon("brightness-contrast"), "Display")
        self.tabs.setIconSize(QSize(18, 18))
        self.qc_dock = self._dock("QC Panel", self.tabs, "qc_dock", Qt.DockWidgetArea.TopDockWidgetArea)
        self.qc_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.qc_dock.setTitleBarWidget(QWidget())        # no title bar: it only repeated "QC Panel"
        self.a.pin_qc.setChecked(self.tabs.pinned)
        self.tabs.pinnedChanged.connect(self.a.pin_qc.setChecked)
        # side docks own the corners: File List / Annotations / Labels run the full height of the window
        for corner, area in ((Qt.Corner.TopRightCorner, Qt.DockWidgetArea.RightDockWidgetArea),
                             (Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea),
                             (Qt.Corner.TopLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea),
                             (Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)):
            self.setCorner(corner, area)

    def _build_menus(self):
        mb = self.menuBar()
        self.menus = utils.struct(file=mb.addMenu("&File"), edit=mb.addMenu("&Edit"), view=mb.addMenu("&View"), go=mb.addMenu("&Go"),
                                     review=mb.addMenu("&Review"), qc=mb.addMenu("&QC"), tools=mb.addMenu("&Tools"),
                                     help=mb.addMenu("&Help"))
        self.recent_menu = QMenu("Open &Recent", self)
        self.recent_menu.aboutToShow.connect(self._build_recent_menu)
        utils.addActions(self.menus.file, (self.a.open_dir,))
        self.menus.file.addMenu(self.recent_menu)
        utils.addActions(self.menus.file, (self.a.reload, self.a.load_classes, None, self.a.open_trash, None,
                                           self.a.next_mode, self.a.to_video, self.a.to_studio, self.a.source_video,
                                           None, self.a.exit))
        # preferences (compact toolbar, watch, guideline cards, animations, auto-advance, storage …) live in
        # File ▸ Settings… (widgets.settings_pages.image_sections); their actions stay for shortcuts / handlers
        for act in (self.a.compact_toolbar, self.a.watch, self.a.guide_cards, self.a.animations,
                    self.a.auto_advance, self.a.clear_cache):
            self.addAction(act)
        from AIO_QC.widgets.mode_switch import ModeSwitch
        self.mode_switch = ModeSwitch("image", self)
        self.mode_switch.switched.connect(lambda m: self._shell_call("switch", m))
        mb.setCornerWidget(self.mode_switch, Qt.Corner.TopRightCorner)
        self.mode_switch.setVisible(getattr(QApplication.instance(), "shell", None) is not None)
        from AIO_QC.widgets.settings_dialog import install_settings
        install_settings(self)                              # gear in the corner + File ▸ Settings… (Ctrl+,)
        utils.addActions(self.menus.edit, (self.a.undo_edit, self.a.redo_edit, None, self.a.new_box, None,
                                           self.a.guide_good, self.a.guide_bad, None, self.a.history))
        A = functools.partial(utils.newAction, self)
        self.a_toolbar = A("Toolbar", lambda c: self.toolbar.setVisible(c), checkable=True, checked=True)
        dock_toggles = [self._panel_action(d, t) for d, t in
                        ((self.file_list_dock, "File List"), (self.annotations_dock, "Annotations"),
                         (self.history_dock, "History"), (self.unique_labels_dock, "Labels"),
                         (self.qc_dock, "QC Panel"), (self.filmstrip_dock, "Thumbnails"))]
        utils.addActions(self.menus.view, (
            {"menu": "Interaction Mode", "actions": (self.a.drag, self.a.select)},
            None, self.a.zoom_in, self.a.zoom_out, self.a.fit, self.a.actual, None,
            self.a.tags, self.a.fill, self.a.all_labels, self.a.pred_layer, None,
            {"menu": "Panels", "actions": (*dock_toggles, None, self.a_toolbar)},
            self.a.pin_qc, self.a.focus,
        ))
        utils.addActions(self.menus.go, (self.a.next, self.a.prev, self.a.next_unreviewed, None,
                                         self.a.next_shape, self.a.prev_shape, self.a.clear_selection, None,
                                         self.a.search))
        # QC menu mirrors the first four tabs; the Review menu mirrors the Review tab
        utils.addActions(self.menus.qc, [
            {"menu": title, "actions": [self.qc_actions[t.key] for t in QC_TOOLS if t.group == key]
             + ([None, self.a.rename] if key == "labels" else [])}
            for key, title, _q in GROUPS if key != "review"])
        utils.addActions(self.menus.review, [self.qc_actions[t.key] for t in QC_TOOLS if t.group == "review"] + [
            None, self.a.change_class, self.a.mark_ok, self.a.mark_flag, self.a.flag_note, self.a.mark_clear, None,
            self.a.next_unreviewed, None, self.a.save_flagged, self.a.export_review])
        utils.addActions(self.menus.tools, (self.a.sketch, self.a.roi, None, self.a.data_prep, self.a.ann_table, None,
                                            self.a.palette, None, self.a.storage, None, self.a.reset, self.a.logs))
        about = QAction(utils.appIcon(), "About Image QC", self)
        about.setMenuRole(QAction.MenuRole.AboutRole)
        from AIO_QC.widgets.about import show_about
        about.triggered.connect(lambda _c=False: show_about(
            self, "Image QC", "Quality control for annotated image datasets (LabelMe / YOLO)", utils.appIcon()))
        self.a.about = about
        utils.addActions(self.menus.help, (self.a.get_started, None, self.a.guide, self.a.tool_map, self.a.shortcuts,
                                           self.a.keymap, None, about))
        utils.addActions(self.menus.go, (None, self.a.next_panel, self.a.prev_panel))
        for i in range(min(9, self.tabs.count())):          # Alt+1… : a QC tab, keyboard focus on its first tool
            sc = QAction(self)
            sc.setShortcut(QKeySequence(f"Alt+{i + 1}"))
            sc.triggered.connect(lambda _c=False, i=i: self.tabs.open_tab_keyboard(i))
            self.addAction(sc)

    def _build_statusbar(self):
        sb = self.statusBar()
        self.canvas.cursorMoved.connect(sb.set_pointer)
        self.canvas.selectionRect.connect(sb.set_selection)
        self.pos_label = QLabel("")
        self.index_label = QLabel("")
        self.mode_label = QLabel("✋ Drag")
        self.review_badge = QLabel("")
        self.mapping_btn = QToolButton()
        self.mapping_btn.setAutoRaise(True)
        self.mapping_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        utils.set_icon(self.mapping_btn, "tool-label-map")
        self.mapping_btn.setToolTip("Label mapping is active — click to edit")
        self.mapping_btn.clicked.connect(lambda _c=False: self.open_label_map())
        self.mapping_btn.setVisible(False)
        self.statusBar().addPermanentWidget(self.mapping_btn)
        for w in (self.review_badge, self.index_label, self.pos_label, self.mode_label):
            utils.style(w, "color:@muted; padding: 0 6px;")
            self.statusBar().addPermanentWidget(w)
        self._review_save = QTimer(self, singleShot=True, interval=800, timeout=self._save_review)

    def _restore_settings(self):
        geo = self.settings.value("geometry")
        state = self.settings.value("windowState")
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None:
            self.restoreState(state)
        self._set_toolbar_compact(self.a.compact_toolbar.isChecked())
        if not self.settings.value("ui/ribbon_v1", False, type=bool):
            self.settings.setValue("ui/ribbon_v1", True)     # the panel is slim now: show it once after the update
            self.qc_dock.show()
            utils.later(1200, self, lambda: self.statusBar().showMessage(
                "The QC Panel is now a slim tab row — click a tab for its tools, 📌 to keep them open, "
                "Ctrl+K to run any command, F11 for focus mode", 12000))

    def request_quit(self, reason: str):
        """Quit without prompts (Ctrl+C, SIGTERM): an open fixing session is kept for later, everything is saved."""
        self._quit_reason = reason
        dlg = self.editor_dialog
        if dlg is not None and dlg.es is not None and dlg.es.active:
            dlg.keep_for_later(f"kept for later ({reason})")
            dlg._shutdown("app quitting")
            dlg.hide()
        self.close()

    def closeEvent(self, event):
        if getattr(self, "_quit_reason", None) is None and not self._end_editor_session("quit"):
            event.ignore()
            logger.debug("close cancelled: an editor session is still active")
            return
        set_exit_reason(getattr(self, "_quit_reason", None) or "the user closed the window")
        logger.info(f"closing AIO QC ({getattr(self, '_quit_reason', None) or 'window closed'}) — saving state")
        self._stop_watcher()
        self._stop_sweep()
        self.ocr.shutdown()
        self.embedder.shutdown()
        self._save_review(compact=True)
        self._stop_index()
        if self.session is not None:
            self.session.close()
        self.inference.shutdown()          # stops the model server; deletes a temporary env if asked to
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self._focus_state if self._focus_state is not None else self.saveState())
        self._stop_index()
        for d in list(self.open_dialogs.values()):
            d.close()
        if getattr(self, "testing", None) is not None:
            self.testing._dirty = False                  # its session is autosaved (with the dataset)
            self.testing.close()
        stop_everything(self, "app closing")             # nothing left running when Qt tears the window down
        logger.info("state saved, background work stopped")
        super().closeEvent(event)

    def set_dataset_actions_enabled(self, on: bool):
        for act in (self.a.reload, self.a.next, self.a.prev, self.a.rename, self.a.open_trash, self.a.label_map,
                    self.a.clear_cache, self.a.storage, self.a.ann_table,
                    self.a.mark_ok, self.a.mark_flag, self.a.flag_note, self.a.mark_clear, self.a.next_unreviewed,
                    self.a.export_review, self.a.save_flagged):
            act.setEnabled(on)
        for t in QC_TOOLS:
            act = self.qc_actions[t.key]
            if t.dialog is None:
                act.setEnabled(on)
                continue
            supported = t.dialog.SUPPORTED is None or (self.session and self.session.type in t.dialog.SUPPORTED)
            act.setEnabled(on and bool(supported))
            act.setToolTip(f"{t.tip} ({t.shortcut})" if supported or not on
                           else f"{t.tip} — not available for {self.session.type.title}")

    # =====================================================================
    # Dataset lifecycle
    # =====================================================================
    def open_dir(self):
        start = self.settings.value("last_dir", "")
        path = QFileDialog.getExistingDirectory(self, "Select dataset root (LabelMe dir, YOLO root, or YOLO split root)", start)
        if path:
            self.settings.setValue("last_dir", path)
            self.open_dataset(Path(path))

    def open_dataset(self, root: Path, dtype: DatasetType | None = None):
        if dtype is None:
            types = detect_dataset_types(root)
            if not types:
                self.open_dir_btn.flash("error", f"No dataset found in {root}")
                QMessageBox.warning(self, "Open", "No LabelMe JSONs, root images/+labels/, or "
                                                  "<split>/images+labels/ found in:\n" + str(root))
                return
            dtype = types[0]
            if len(types) > 1:
                names = [t.title for t in types]
                choice, ok = QInputDialog.getItem(self, "Dataset type",
                                                  "Several layouts were detected. Open as:", names, 0, False)
                if not ok:
                    return
                dtype = types[names.index(choice)]
        if not self._end_editor_session("open another dataset"):
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            adapter = open_dataset(root, dtype)
            session = Session(root, adapter, self.user_classes)
        except Exception as exc:
            logger.exception("open failed")
            self.open_dir_btn.flash("error", f"Last open failed: {exc}")
            QMessageBox.critical(self, "Open", f"Could not open dataset:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._close_session()
        self.session = session
        session.annotationsChanged.connect(self._on_annotations_changed)
        session.annotationsChanged.connect(self._note_own_writes)        # our writes are not "outside" changes
        session.samplesRemoved.connect(self._on_samples_removed)
        session.samplesAdded.connect(self._on_samples_added)
        session.ownPaths.connect(lambda paths: self.tracker.note_own(paths) if self.tracker is not None else None)
        session.issuesChanged.connect(self._issues_save_timer.start)
        session.indexChanged.connect(self._labels_refresh.start)
        session.reviewChanged.connect(self._on_review_changed)
        self.file_dock.model.review = session.review
        self.filmstrip.set_session(session)
        self._add_recent(root, dtype)

        # YOLO class names: reuse a loaded classes file if the dataset has none, then CHECK them against the labels
        self._class_scan = None
        if session.is_yolo and not adapter.class_names and self.user_classes:
            adapter.set_class_names(self.user_classes, self.user_classes_path)
            self.statusBar().showMessage("No dataset.yaml/classes.txt in dataset → using the loaded classes file", 6000)
        if session.is_yolo:
            QTimer.singleShot(0, self._check_yolo_classes_on_open)
        QTimer.singleShot(50, self._offer_unfinished_editor_sessions)

        # Label colours: the dataset's saved ones first, then the class list in order (most distinct first)
        from AIO_QC.utils import colors as _colors
        _colors.load(root, self.label_colors)
        for name in adapter.class_names:
            utils.update_label_colors(name, self.label_colors)

        self.refresh_status_buttons()
        self.dataset_label.setText(f"  📂 {root.name} · {dtype.title} · {len(session.samples):,} files  ")
        self._update_title()
        self._external_log = []
        self._start_watcher()
        self._start_sweep()
        self.a.change_class.setEnabled(session.is_classification)
        split_note = getattr(adapter, "split_note", "")
        self.dataset_label.setToolTip(str(root) + (f"\n{split_note}" if split_note else ""))
        if split_note:
            self.statusBar().showMessage(split_note, 10000)
            logger.info(f"[splits] {split_note}")
        if dtype in (DatasetType.YOLO, DatasetType.YOLO_SPLIT) and adapter.class_map.problems:
            self.log_errors([f"[class map] {p}" for p in adapter.class_map.problems])
        self.file_dock.set_samples(session.samples, session.splits, session.folders)
        self.set_dataset_actions_enabled(True)
        if getattr(self, "testing", None) is not None and self.testing.isVisible():
            utils.later(0, self.testing, self.testing.use_qc_dataset)     # Testing follows the open dataset
        self._update_mapping_indicator()
        self._update_review_progress()
        if session.label_map:
            self.statusBar().showMessage(f"🔀 Saved label mapping loaded ({len(session.label_map)} label(s)) — "
                                         "Tools ▸ Label Mapping to change", 8000)
        self._start_index()
        if session.samples:
            self.file_dock.select_row(0)
        else:
            self.statusBar().showMessage("Dataset contains no samples", 5000)

    def reload_dataset(self):
        if self.session:
            key = self.current_key
            self.open_dataset(self.session.root, self.session.type)
            if key:
                self.file_dock.select_key(key)

    # ------------------------------------------------------------------ live updates (outside changes)
    def _start_watcher(self):
        self._stop_watcher()
        s = self.session
        if s is None or not self.a.watch.isChecked():
            return
        from AIO_QC.core.io import IMAGE_EXTENSIONS
        exts = set(IMAGE_EXTENSIONS) | ({".txt"} if s.is_yolo else {".json"})
        self.tracker = None
        w = DatasetWatcher(s.root, exts, self)
        w.snapshotReady.connect(lambda snap, took, w=w: self._on_snapshot(snap, took, w))
        self.watcher = w
        w.start()

    def _stop_watcher(self):
        w = getattr(self, "watcher", None)
        if w is not None:
            try:
                w.snapshotReady.disconnect()
            except TypeError:
                pass
            w.stop()
        self.watcher, self.tracker = None, None

    def _toggle_watch(self, on: bool):
        self.settings.setValue("watch/enabled", bool(on))
        if on:
            self._start_watcher()
            self.statusBar().showMessage("Watching the dataset for changes made outside AIO QC", 4000)
        else:
            self._stop_watcher()

    def _note_own_writes(self, keys):
        if getattr(self, "tracker", None) is not None and self.session is not None:
            self.tracker.note_own(self.session.paths_of(keys))

    def _on_app_state(self, state):
        if state == Qt.ApplicationState.ApplicationActive and getattr(self, "watcher", None) is not None:
            self.watcher.poke()                       # back from another program: check right away

    def _on_snapshot(self, snap, _took, watcher=None):
        from AIO_QC.core.watch import ChangeTracker
        if self.session is None or watcher is not self.watcher:      # a stopped watcher's late snapshot
            return
        if self.tracker is None:
            self.tracker = ChangeTracker(snap)        # first scan = baseline
            return
        ch = self.tracker.step(snap)
        if ch:
            try:
                self._apply_external_changes(ch)
            except Exception as exc:                   # a live update must never take the app down
                logger.exception(exc)
                self.log_errors([f"[live update] {exc}"], quiet=True)

    def _apply_external_changes(self, ch):
        from AIO_QC.core.watch import is_meta
        s = self.session
        kp = s.key_by_path                                      # persistent path -> key map (no O(n) rebuild)
        ann_to_key = {p: kp[p] for p in ch.added + ch.modified + ch.removed
                      if p in kp and str(s.by_key[kp[p]].ann_path) == p}
        img_to_key = {p: kp[p] for p in ch.added + ch.modified + ch.removed
                      if p in kp and str(s.by_key[kp[p]].image_path) == p}
        modified, removed, new, image_changed = set(), set(), [], set()
        structural = [p for p in ch.added + ch.modified + ch.removed if is_meta(p)]
        for p in ch.modified:
            if p in ann_to_key:
                modified.add(ann_to_key[p])
            elif p in img_to_key:
                modified.add(img_to_key[p])
                image_changed.add(img_to_key[p])
            elif self.current_ann is not None and str(self.current_ann.image_path or "") == p and self.current_key:
                image_changed.add(self.current_key)       # LabelMe: the current file's image was replaced
        for p in ch.removed:
            if p in ann_to_key:
                (modified if s.is_yolo else removed).add(ann_to_key[p])     # YOLO: label gone -> background image
            elif p in img_to_key:
                removed.add(img_to_key[p])
        for p in ch.added:
            if p in ann_to_key:                           # YOLO: a label file created for an existing image
                modified.add(ann_to_key[p])
                continue
            if is_meta(p):
                continue
            smp, struct = s.base_adapter.sample_for_path(p)
            if struct:
                structural.append(p)
            elif smp is not None:
                new.append(smp)
        locked = {k for k in modified | removed if k in s.locked}
        modified -= removed
        if modified:                                       # audit trail: changes nobody made in AIO QC
            s.log(sorted(modified), "external", "changed on disk by another program", tool="Outside AIO QC")
        if removed:
            s.log(sorted(removed), "external", "removed on disk by another program", tool="Outside AIO QC")
        if removed:
            s.forget_samples(sorted(removed))
        added_keys = s.add_samples(new) if new else []
        if modified:
            s.reindex_keys(sorted(modified))
        if self.current_key in image_changed:
            self.pixmap_cache.clear()
            self.show_sample(self.current_key)
        elif self.current_key in modified:
            self.reload_current_annotation()
        n = len(modified) + len(removed) + len(added_keys)
        if n:
            for d in list(self.open_dialogs.values()):
                if hasattr(d, "mark_stale"):
                    d.mark_stale(n)
        now = time.strftime("%H:%M:%S")
        for kind, keys in (("changed", modified), ("added", added_keys), ("removed", removed)):
            for k in sorted(keys):
                self._external_log.append((now, kind, k))
        for p in structural:
            self._external_log.append((now, "layout", p))
        self._external_log = self._external_log[-2000:]
        if n:
            logger.info(f"[live update] outside changes: {len(modified)} changed · {len(added_keys)} added · "
                        f"{len(removed)} removed")
        parts = [f"{v:,} {t}" for v, t in ((len(modified), "changed"), (len(added_keys), "added"),
                                            (len(removed), "removed")) if v]
        if structural:
            names = ", ".join(sorted({Path(p).name for p in structural})[:3])
            self.toast.show_message(
                "<b>The dataset layout changed outside AIO QC</b><br>"
                f"{names}{' …' if len(structural) > 3 else ''} — class names or split folders may differ now. "
                "Reload to apply." + (f"<br>Also: {' · '.join(parts)} (already updated)." if parts else ""),
                kind="warn", sticky=True,
                actions=[("Reload dataset", self.reload_dataset)], details=self._external_log_text)
        elif parts:
            # one notification while it is on screen: counts accumulate, its timer restarts
            if self.toast.find("external") is None:
                self._ext_counts = Counter()
                self._ext_locked = 0
            self._ext_counts.update({"changed": len(modified), "added": len(added_keys), "removed": len(removed)})
            self._ext_locked += len(locked)
            total = [f"{v:,} {k}" for k, v in self._ext_counts.items() if v]
            extra = (f"<br>{self._ext_locked} of them are open in the editor session (it will show a conflict)."
                     if self._ext_locked else "")
            self.toast.show_message(f"<b>Updated from disk</b><br>{' · '.join(total)} outside AIO QC.{extra}",
                                    kind="warn" if self._ext_locked else "info", key="external",
                                    details=self._external_log_text)
            self.statusBar().showMessage(f"Dataset updated from disk: {' · '.join(parts)}", 6000)

    def _on_samples_added(self, keys):
        s = self.session
        self.file_dock.insert_samples(s.last_inserted)        # discovery-order rows, like a reload
        self.dataset_label.setText(f"  📂 {s.root.name} · {s.type.title} · {len(s.samples):,} files  ")
        self._update_position()

    def _external_log_text(self) -> str:
        """Changes made outside AIO QC this session, newest first (the notification's Details)."""
        by_key = self.session.by_key if self.session else {}
        icon = {"changed": "✎", "added": "＋", "removed": "−", "layout": "⚠"}
        lines = [f"{t}  {icon.get(k, '·')} {k:8s} {by_key[x].display if x in by_key else x}"
                 for t, k, x in reversed(self._external_log)]
        return "\n".join(lines) or "No outside changes in this session."

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "toast"):
            self.toast.reposition()

    def _update_title(self):
        """One pattern in every mode: '<mode> — <what is open> — AIO QC App'."""
        s = getattr(self, "session", None)
        self.setWindowTitle(f"Image QC — {s.root.name} — {__appname__}" if s is not None
                            else f"Image QC — {__appname__}")

    def _close_session(self):
        for mode in (getattr(self, "sketch", None), getattr(self, "roi", None)):
            if mode is not None and mode.active:
                mode.leave()                         # saves the drawing / ROIs of the dataset that is closing
        self._stop_watcher()
        self._stop_sweep()
        if self.session is not None:
            try:
                self.session.issues.save()
            except OSError as exc:
                logger.warning(f"could not save the issues inbox: {exc}")
        self._save_review(compact=True)
        self._stop_index()
        hk = getattr(self, "_housekeeper", None)
        if hk is not None:
            try:
                if hk.isRunning():
                    hk.wait(30000)                  # it uses the cache that is closed next
            except RuntimeError:                    # already deleted
                pass
            self._housekeeper = None
        if self.session is not None:
            self.session.close()
        for d in list(self.open_dialogs.values()):
            d.close()
        self.open_dialogs.clear()
        self.generation += 1
        self._inflight.clear()
        self.pixmap_cache.clear()
        self.session = None
        self._update_title()
        self.current_key = self.current_ann = None
        self.inline.reset()
        self.embed_index = None
        self.label_colors.clear()
        self.visible_labels.clear()
        self.hidden_labels.clear()
        self.unique_labels.clear()
        self.annotations_list.clear()
        self.canvas.clearAll()
        self.statusBar().set_image(None, None)
        self.statusBar().set_file(None)
        self.statusBar().set_pointer(None)
        self.zoom_spin.setEnabled(False)
        self.zoom_spin.set_zoom(None)
        self.zoom_slider.set_zoom(None)
        self.file_dock.model.review = None
        self.filmstrip.set_session(None)

    def load_classes_file(self) -> bool:
        """
        LabelMe: reference list for ambiguity / unknown-label checks.
        YOLO:    becomes the class map (id -> name) when the dataset has no dataset.yaml.
        """
        if self.session and self.session.is_yolo and self.session.base_adapter.has_yaml:
            src = self.session.base_adapter.class_map.source.name
            if QMessageBox.question(
                    self, "Load classes",
                    f"This YOLO dataset defines its classes in {src}, which takes precedence.\n"
                    "A loaded classes.txt will NOT change its class names.\n\nLoad one anyway (kept for LabelMe datasets)?"
            ) != QMessageBox.StandardButton.Yes:
                return False
        start = str(self.session.root) if self.session else self.settings.value("last_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, "Select classes list", start,
                                              "Class lists (classes.txt *.txt *.names *.yaml *.yml);;All files (*)")
        if not path:
            return False
        try:
            classes = load_classes_file(path)
        except Exception as exc:
            self.load_classes_btn.flash("error", f"Could not read {Path(path).name}: {exc}")
            QMessageBox.warning(self, "Load classes", str(exc))
            return False
        if not classes:
            self.load_classes_btn.flash("error", f"{Path(path).name} contains no class names")
            QMessageBox.warning(self, "Load classes", "No class names found in that file.")
            return False
        if len(set(classes)) != len(classes):
            dups = sorted({c for c in classes if classes.count(c) > 1})
            QMessageBox.warning(self, "Load classes", f"Duplicate class names: {', '.join(dups)}\n"
                                                      "Loaded anyway — fix the file to avoid wrong id mapping.")
        s = self.session
        if s and s.is_yolo and not s.base_adapter.has_yaml:
            res = self._verify_yolo_classes(classes, Path(path))
            if res == "retry":
                return self.load_classes_file()
            if res == "cancel":
                return False
        self.user_classes, self.user_classes_path = classes, Path(path)
        if s:
            s.user_classes, s.user_classes_path = classes, Path(path)
            if s.is_yolo and not s.base_adapter.has_yaml:
                self._apply_yolo_classes(classes, Path(path))
        self.refresh_status_buttons()
        self.statusBar().showMessage(f"Loaded {len(classes)} classes from {Path(path).name}", 5000)
        return True

    # ------------------------------------------------------------------ YOLO class verification
    def _scan_class_ids(self):
        """Class ids used by the label files — scanned once per session (threaded), refreshed after edits."""
        from PyQt6.QtCore import QEventLoop
        from PyQt6.QtWidgets import QProgressDialog
        from AIO_QC.core.class_check import scan_class_ids
        from AIO_QC.widgets.inference_service import FuncWorker
        s = self.session
        cached = getattr(self, "_class_scan", None)
        if cached is not None and cached[0] is s:
            return cached[1]
        files = [x.ann_path for x in s.samples if x.ann_path is not None]
        dlg = QProgressDialog(f"Checking the class ids used in {len(files):,} label files …", None, 0, 0, self)
        dlg.setWindowTitle("Class check")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(400)
        loop, out = QEventLoop(), {}
        w = FuncWorker(lambda: scan_class_ids(files), self)
        w.done.connect(lambda r: (out.update(scan=r), loop.quit()))
        w.failed.connect(lambda m: (out.update(error=m), loop.quit()))
        w.start()
        loop.exec()
        w.wait()
        dlg.close()
        if "error" in out:
            logger.warning(f"class id scan failed: {out['error']}")
            return None
        self._class_scan = (s, out["scan"])
        return out["scan"]

    def _verify_yolo_classes(self, names: list[str], source: Path | None) -> str:
        """
        Are these class names right for THIS dataset?  -> "ok" | "accept" (mismatch, used anyway) | "retry" | "cancel"
          mismatch (a label uses an id without a name): say so immediately and ask for the correct classes.txt;
          match: ask the user to confirm it is the final list (a list in the wrong ORDER also "matches"),
                 remembered per dataset until the file changes.
        """
        from AIO_QC.core.class_check import verify_classes
        from AIO_QC.core.state import classes_confirmed, confirm_classes
        scan = self._scan_class_ids()
        if scan is None:
            return "ok"
        src_name = source.name if source else "the class list"
        v = verify_classes(names, scan, src_name)
        if v.kind == "no_labels":
            return "ok"
        box = QMessageBox(self)
        box.setWindowTitle("Class check")
        if v.kind == "mismatch":
            self.load_classes_btn.flash("error", f"{src_name} does not match the labels")
            box.setIcon(QMessageBox.Icon.Warning)
            box.setText(f"<b>{v.title}</b>")
            box.setInformativeText(v.details + "\n\nLoad the correct classes.txt for this dataset?")
            retry = box.addButton("Load the correct classes.txt…", QMessageBox.ButtonRole.AcceptRole)
            anyway = box.addButton("Use it anyway", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(retry)
            box.exec()
            if box.clickedButton() is retry:
                return "retry"
            if box.clickedButton() is anyway:
                self.log_errors([f"[class check] {v.details}"], quiet=True)
                return "accept"
            return "cancel"
        if source is not None and classes_confirmed(self.session.root, source, names):
            return "ok"
        preview = "\n".join(f"{i:>3}  {n}" for i, n in enumerate(names[:20])) + \
                  (f"\n  …  ({len(names) - 20} more)" if len(names) > 20 else "")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(f"<b>Is {src_name} the final class list for this dataset?</b>")
        box.setInformativeText(f"{v.details}\n\nThe ids match, but a list in the wrong order would match too — "
                               "please confirm the names below are correct (line N = class id N).")
        box.setDetailedText(preview)
        yes = box.addButton("Yes, it's the final list", QMessageBox.ButtonRole.AcceptRole)
        other = box.addButton("Load a different file…", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(yes)
        box.exec()
        if box.clickedButton() is yes:
            if source is not None:
                confirm_classes(self.session.root, source, names)
            return "ok"
        return "retry" if box.clickedButton() is other else "cancel"

    def _check_yolo_classes_on_open(self):
        s = self.session
        if s is None or not s.is_yolo:
            return
        ad = s.base_adapter
        if ad.has_yaml:                        # the YAML is authoritative: only warn about a mismatch
            from AIO_QC.core.class_check import verify_classes
            scan = self._scan_class_ids()
            src_name = ad.class_map.source.name if ad.class_map.source else "dataset.yaml"
            v = verify_classes(ad.class_names, scan, src_name) if scan else None
            if v is not None and v.kind == "mismatch":
                self.load_classes_btn.flash("warning", v.details)
                self.log_errors([f"[class check] {v.details}"], quiet=True)
                self.statusBar().showMessage(f"⚠ {v.title}: {v.details.splitlines()[0]}", 12000)
            return
        if not ad.class_names:
            self._ask_yolo_classes()
            return
        res = self._verify_yolo_classes(list(ad.class_names), ad.class_map.source)
        if res == "retry":
            self.load_classes_file()

    def _apply_yolo_classes(self, classes, path):
        """Switch a YOLO dataset's id -> name map; everything label-derived is rebuilt."""
        s = self.session
        old = list(s.base_adapter.class_names)
        s.base_adapter.set_class_names(classes, path)
        s.set_label_map(s.label_map)           # rebuild the mapped view on top of the new names
        if old == list(classes):
            return
        for name in s.adapter.class_names:
            utils.update_label_colors(name, self.label_colors)
        self.visible_labels.clear()            # labels like "class_3" are gone
        self.hidden_labels.clear()
        s.index.clear()
        s.raw_index.clear()
        s.index_errors.clear()
        s.index_complete = False
        for d in list(self.open_dialogs.values()):
            d.close()                          # their results used the old names
        self.reload_current_annotation()
        self._start_index()
        self.log_errors([f"[class map] {p}" for p in s.base_adapter.class_map.problems], quiet=True)

    def _ask_yolo_classes(self):
        s = self.session
        if not s or not s.is_yolo or s.base_adapter.class_names:
            return
        self.load_classes_btn.flash("warning", "No dataset.yaml / classes.txt — class names unknown")
        reply = QMessageBox.question(
            self, "Class names missing",
            "This YOLO dataset has no dataset.yaml / data.yaml and no classes.txt.\n\n"
            "Load a classes.txt now? (line N = class id N)\n"
            "Without it labels show as class_0, class_1, … and the ambiguity check is unavailable.")
        if reply == QMessageBox.StandardButton.Yes:
            self.load_classes_file()

    def refresh_status_buttons(self):
        """Single place that decides the colour of Open Dir / Load Classes."""
        s = self.session
        if s is None:
            self.open_dir_btn.set_status("default")
        elif not s.samples:
            self.open_dir_btn.set_status("warning", f"{s.root}\nNo samples found")
        else:
            self.open_dir_btn.set_status("success", f"{s.root}\n{s.type.title} · {len(s.samples):,} files")

        if s is None:
            if self.user_classes:
                self.load_classes_btn.set_status("success", f"{len(self.user_classes)} classes from "
                                                            f"{self.user_classes_path.name}")
            else:
                self.load_classes_btn.set_status("default")
        elif s.is_yolo:
            cm = s.base_adapter.class_map
            if not cm.names:
                self.load_classes_btn.set_status("warning", "No dataset.yaml / classes.txt — load classes.txt")
            elif cm.problems:
                self.load_classes_btn.set_status("warning", f"{len(cm.names)} classes from {s.classes_source}\n"
                                                            + "\n".join(cm.problems))
            else:
                self.load_classes_btn.set_status("success", f"{len(cm.names)} classes from {s.classes_source}")
        elif s.user_classes:
            self.load_classes_btn.set_status("success", f"{len(s.user_classes)} classes from "
                                                        f"{s.user_classes_path.name if s.user_classes_path else 'file'}")
        else:
            self.load_classes_btn.set_status("default", "Optional for LabelMe: enables unknown-label checks")

    def _watch_theme(self):
        """Dark ↔ light while running (Qt ≥ 6.5 colour scheme, or a palette change): swap icon sets."""
        hints = QApplication.styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(lambda _s: self._on_theme_changed())
        self._dark = utils.is_dark()

    def changeEvent(self, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.PaletteChange and getattr(self, "_dark", None) is not None \
                and utils.is_dark() != self._dark:
            self._on_theme_changed()
        super().changeEvent(event)

    def _on_theme_changed(self):
        self._dark = utils.is_dark()
        utils.refresh_icons()
        for i, (key, _t, _q) in enumerate(GROUPS):
            self.tabs.setTabIcon(i, utils.newIcon(f"group-{key}"))
        self.tabs.setTabIcon(len(GROUPS), utils.newIcon("brightness-contrast"))
        self.statusBar().showMessage(f"{'Dark' if self._dark else 'Light'} icon set applied", 3000)

    def open_model_setup(self):
        if not self.validate_inputs():
            return
        dlg = self.open_dialogs.get("model_setup")
        if dlg is None:
            dlg = ModelSetupDialog(self)
            dlg.tool_key = "model_setup"
            dlg.setWindowIcon(utils.newIcon("tool-model-setup"))
            self.open_dialogs["model_setup"] = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _cleanup_temp_envs(self):
        """Temporary envs that could not be deleted last time (locked files, crash) are removed now."""
        pending = list(self.settings.value("inference/pending_delete", []) or [])
        tmp = self.inference.temp_env
        if tmp and self.inference.delete_temp_on_exit and Path(tmp).exists():
            pending.append(tmp)          # the app did not exit cleanly last time
        left = []
        for p in dict.fromkeys(pending):
            if Path(p).exists() and not remove_env(Path(p)):
                left.append(p)
        self.settings.setValue("inference/pending_delete", left)
        if tmp and not Path(tmp).exists():
            if self.inference.python.startswith(tmp):
                self.inference.set_python("")
            self.inference.set_temp_env("", self.inference.delete_temp_on_exit)

    def clear_cache(self):
        s = self.session
        if s is None or s.cache is None:
            return
        st = s.cache.stats()
        if QMessageBox.question(self, "Clear cache",
                                f"Delete cached data for this dataset?\n\n{st['labels']:,} label entries · "
                                f"{st['images']:,} image entries · {st['preds']:,} model predictions\n\n"
                                "Everything is rebuilt automatically when needed.") != QMessageBox.StandardButton.Yes:
            return
        s.cache.clear()
        self.statusBar().showMessage("Dataset cache cleared", 4000)

    # ------------------------------------------------------------------ space for the image
    def _set_toolbar_compact(self, on: bool):
        """Compact: one tidy row — the zoom box goes single-line and every icon button becomes a square of exactly
        its height (Open / Load Classes match it too). Full: icons with text under them, as before."""
        from PyQt6.QtWidgets import QPushButton, QToolButton
        on = bool(on)
        if self.isVisible():
            anim.crossfade(self.toolbar, 160)
        self.settings.setValue("ui/compact_toolbar", on)
        self.zoom_spin.set_compact(on)
        h = max(26, self.zoom_spin.sizeHint().height())
        icon = QSize(18, 18) if on else QSize(24, 24)
        style = Qt.ToolButtonStyle.ToolButtonIconOnly if on else Qt.ToolButtonStyle.ToolButtonTextUnderIcon
        self.toolbar.setToolButtonStyle(style)
        self.toolbar.setIconSize(icon)
        for b in self.toolbar.findChildren(QToolButton):
            act = b.defaultAction()
            if act is None or b.parent() is not self.toolbar:
                continue
            b.setToolButtonStyle(style)
            b.setIconSize(icon)
            if on:
                b.setFixedSize(h, h)
            else:
                b.setMinimumSize(0, 0)
                b.setMaximumSize(16777215, 16777215)
            name = act.text().replace("&", "").replace("…", "")
            tip = act.toolTip()
            b.setToolTip(tip if (not on or name.lower() in tip.lower()) else f"<b>{name}</b> — {tip}")
        for sb in (self.open_dir_btn, self.load_classes_btn):
            for pb in sb.findChildren(QPushButton):
                if on:
                    pb.setFixedHeight(h)
                else:
                    pb.setMinimumHeight(0)
                    pb.setMaximumHeight(16777215)
        utils.style(self.toolbar, "QToolBar { spacing: 3px; padding: 2px 4px; }"
                                  "QToolBar::separator { background: @line; width: 1px; margin: %dpx 7px; }"
                                  % (5 if on else 8))

    def toggle_keymap(self):
        if self._keymap is None:
            self._keymap = KeyMapOverlay(self)
        self._keymap.toggle()

    def open_guidelines(self, scope=None):
        self.open_tool("guidelines")
        dlg = self.open_dialogs.get("guidelines")
        if dlg is not None and scope:
            dlg.select(scope)

    def _panel_action(self, dock, title):
        """View ▸ Panels entry that slides the panel open / closed (instead of the dock's instant toggle)."""
        act = QAction(title, self)
        act.setCheckable(True)
        act.setChecked(not dock.isHidden())
        act.triggered.connect(lambda on, d=dock: anim.toggle_dock(self, d, on))
        dock.visibilityChanged.connect(lambda _v, a=act, d=dock: a.setChecked(not d.isHidden()))
        return act

    def _set_animations(self, on: bool):
        from AIO_QC.widgets import motion
        motion.set_mode("full" if on else "off")           # also sets anim.ENABLED

    # ------------------------------------------------------------------ settings
    def settings_sections(self):
        from AIO_QC.widgets.settings_pages import image_sections
        return image_sections(self)

    def open_settings(self, key=None):
        from AIO_QC.widgets.settings_dialog import open_settings
        return open_settings(self, self.settings_sections(), "Settings — Image QC", key)

    def toggle_focus_mode(self, on=None):
        """Only the image: every dock and the toolbar hidden; the exact layout comes back on the second press.
        (The canvas re-fits by itself when it was fitted to the window.)"""
        if not isinstance(on, bool):
            on = self._focus_state is None
        anim.crossfade(self)
        if on and self._focus_state is None:
            self._focus_state = self.saveState()
            for d in self.findChildren(QDockWidget):
                d.hide()
            self.toolbar.hide()
            self.statusBar().showMessage("Focus mode — F11 (or Ctrl+Alt+F) brings the panels back", 5000)
        elif not on and self._focus_state is not None:
            state, self._focus_state = self._focus_state, None
            self.restoreState(state)
            self.toolbar.show()
        self.a.focus.blockSignals(True)
        self.a.focus.setChecked(self._focus_state is not None)
        self.a.focus.blockSignals(False)

    def _show_history(self):
        anim.toggle_dock(self, self.history_dock, True)
        self.history_dock.raise_()
        self.history_panel.set_key(self.current_key)

    def _start_new_box(self):
        c = self.canvas
        if not c.editable:
            self.statusBar().showMessage(f"✋ {c.edit_note or 'nothing to edit'}", 6000)
            return
        c._start_drawing()
        self.statusBar().showMessage("Drag a box on the image — Esc cancels", 5000)

    def open_palette(self):
        from AIO_QC.widgets.command_palette import CommandPalette
        if self._palette is None:
            self._palette = CommandPalette(self, self.settings)
        self._palette.popup()

    def open_storage(self):
        if not self.session:
            return
        from AIO_QC.widgets.storage_dialog import StorageDialog
        dlg = self.open_dialogs.get("storage")
        if dlg is None or dlg.session is not self.session:
            dlg = StorageDialog(self)
            self.open_dialogs["storage"] = dlg
        else:
            dlg.refresh()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def backup_ops_in_use(self) -> set:
        """Backup operations an open undo still needs (never pruned)."""
        keep = set()
        b = getattr(self.open_dialogs.get("dim_area"), "_last_backup", None)
        if getattr(b, "op", None) is not None:
            keep.add(b.op)
        es = getattr(self.editor_dialog, "es", None) if getattr(self, "editor_dialog", None) else None
        if es is not None and es.backup_op is not None:
            keep.add(es.backup_op)
        return keep

    def _housekeeping(self):
        """Keep .aio_qc small (background, after indexing): retention, unused results, removed files, temp files."""
        from AIO_QC.core import storage as S
        from AIO_QC.widgets.storage_dialog import load_cfg
        s = self.session
        cfg = load_cfg(self.settings)
        if s is None or not cfg["auto"]:
            return
        root, cache, keep = s.root, s.cache, self.backup_ops_in_use()
        alive = [p for x in s.samples for p in (x.ann_path, x.image_path)]
        w = FuncWorker(lambda: S.housekeeping(root, cache, cfg, alive_paths=alive, keep_ops=keep), self)

        def done(r):
            for n in r["notes"]:
                logger.info(f"storage: {n}")
            if r["freed"] >= 5 * 2 ** 20 and self.session is s:
                self.toast.show_message(f"<b>Storage tidied</b><br>Freed {S.human(r['freed'])} in .aio_qc — "
                                        + "; ".join(r["notes"][:2]) + ".", kind="success", key="storage",
                                        actions=[("Storage…", self.open_storage)],
                                        details="\n".join(r["notes"]) or None)

        w.done.connect(done)
        w.failed.connect(lambda m: logger.warning(f"storage housekeeping failed: {m}"))
        w.finished.connect(w.deleteLater)
        self._housekeeper = w
        w.start()

    def open_trash(self):
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        if not self.session:
            return
        trash = self.session.root / ".aio_qc_trash"
        if not trash.exists():
            QMessageBox.information(self, "Trash", "Nothing has been moved to the trash yet.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(trash)))

    # ------------------------------------------------------------ label index
    def _start_index(self):
        self._stop_index()
        s = self.session
        self.index_worker = IndexWorker(s.adapter, s.samples, s.cache, s.label_namespace())
        self._index_started = time.monotonic()
        gen = self.generation
        self.index_worker.batch.connect(lambda b, g=gen: self._on_index_batch(b, g))
        self.index_worker.progress.connect(
            lambda i, n: self.index_label.setText(f"Indexing {i * 100 // max(n, 1)} %" if i < n else ""))
        w = self.index_worker
        w.completed.connect(lambda cancelled, g=gen, w=w: self._on_index_done(cancelled, g, w.hits))
        self.index_worker.finished.connect(self.index_worker.deleteLater)
        self.index_worker.start()

    def _stop_index(self):
        w = self.index_worker
        self.index_worker = None
        if w is not None:
            try:
                if w.isRunning():
                    w.stop()
                    w.wait(3000)
            except RuntimeError:  # already deleted
                pass

    def _on_index_batch(self, batch, gen):
        if gen != self.generation or not self.session:
            return
        flags = {}
        for key, counts, error in batch:
            self.session.update_index(key, counts, error)
            flags[key] = "error" if error else ("empty" if counts is not None and not counts else None)
        self.file_dock.model.set_flags(flags)
        self._labels_refresh.start()

    def _on_index_done(self, cancelled, gen, hits: int = 0):
        if gen != self.generation or not self.session:
            return
        self.session.index_complete = not cancelled
        self.index_label.setText("")
        if not cancelled and not getattr(self.session, "_housekept", False):
            self.session._housekept = True
            QTimer.singleShot(1500, self._housekeeping)
        n_err = len(self.session.index_errors)
        if n_err:
            self.log_errors([f"[index] {Path(k).name}: {e}" for k, e in self.session.index_errors.items()])
        self.refresh_label_list()
        self.statusBar().showMessage(
            f"Indexed {len(self.session.samples):,} files in {time.monotonic() - self._index_started:.1f}s"
            + (f" ({hits:,} from cache)" if hits else "")
            + (f" · {n_err} unreadable (red in file list)" if n_err else ""),
            6000)

    # =====================================================================
    # Display
    # =====================================================================
    def show_sample(self, key: str):
        if not self.session or key not in self.session.by_key:
            return
        sample = self.session.by_key[key]
        same = key == self.current_key
        self.current_key = key
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
            msg = error or f"Image not found for {sample.name}"
            self.statusBar().showMessage(f"⚠ {msg}", 8000)
            self.log_errors([f"{sample.name}: {msg}"], quiet=True)
        elif not same or self.canvas.pixmap_item.pixmap().cacheKey() != pix.cacheKey():
            self.canvas.loadPixmap(pix)
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

    def _status_file(self, sample, ann, pix):
        """Status bar: file type and file size of the image — shown only when it decoded."""
        sb = self.statusBar()
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
        self.pos_label.setToolTip("")
        self.pos_label.setText(f"{i:,} / {n:,}" if n else "")

    def load_canvas_shapes(self):
        shapes = self.current_ann.shapes if self.current_ann else []
        self.canvas.loadShapes(shapes, self.label_colors, self.visible_labels)

    def preload_neighbors(self):
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
        for label in natsorted(totals):
            utils.update_label_colors(label, self.label_colors)
            if label not in self.hidden_labels:
                self.visible_labels.add(label)
            it = QListWidgetItem(f"{label}   ({totals[label]:,})")
            it.setData(Qt.ItemDataRole.UserRole, label)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked if label in self.hidden_labels else Qt.CheckState.Checked)
            it.setIcon(utils.get_color_icon(self.label_colors[label]))
            self.unique_labels.addItem(it)
        self.unique_labels.blockSignals(False)
        self._sync_all_labels_action()

    def _sync_all_labels_action(self):
        self.a.all_labels.blockSignals(True)
        self.a.all_labels.setChecked(not self.hidden_labels)
        self.a.all_labels.blockSignals(False)

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
        self.load_canvas_shapes()
        self.update_annotations_list()

    def _solo_label(self, label):
        self.hidden_labels = {self.unique_labels.item(i).data(Qt.ItemDataRole.UserRole)
                              for i in range(self.unique_labels.count())} - {label}
        self.visible_labels = {label}
        self.refresh_label_list()
        self.load_canvas_shapes()
        self.update_annotations_list()

    def update_annotations_list(self):
        self.annotations_list.blockSignals(True)
        self.annotations_list.clear()
        if self.current_ann:
            for i, shape in enumerate(self.current_ann.shapes):
                label = str(shape.get("label", "?"))
                utils.update_label_colors(label, self.label_colors)
                it = QListWidgetItem(f"#{i}  {label}")
                it.setData(Qt.ItemDataRole.UserRole, i)
                it.setIcon(utils.shapeIcon(shape.get("shape_type"), self.label_colors[label]))
                it.setToolTip(f"{shape.get('shape_type') or 'polygon'} · {len(shape.get('points') or [])} points")
                if label not in self.visible_labels:
                    it.setForeground(utils.theme_color("faint"))
                    it.setToolTip("Hidden (label unticked in the Labels panel)")
                self.annotations_list.addItem(it)
        self.annotations_dock.setWindowTitle(
            f"Annotations ({self.annotations_list.count()})" if self.current_ann else "Annotations")
        self.annotations_dock.toggleViewAction().setText("Annotations")   # menus: the name only, never the count
        self.annotations_list.blockSignals(False)

    def _on_annotation_row(self, row):
        it = self.annotations_list.item(row)
        if it is not None:
            self._ann_row_changed = True
            self._select_from_list(it.data(Qt.ItemDataRole.UserRole))

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
        idx = item.data(Qt.ItemDataRole.UserRole)
        sel = self.canvas.selected_group
        if sel is not None and sel.index == idx:
            self.canvas.clearSelection()
        else:
            self._select_from_list(idx)

    def onShapeSelected(self, index: int):
        self.annotations_list.blockSignals(True)
        for r in range(self.annotations_list.count()):
            it = self.annotations_list.item(r)
            if it.data(Qt.ItemDataRole.UserRole) == index:
                self.annotations_list.setCurrentItem(it)
                self.annotations_list.scrollToItem(it)
                break
        self.annotations_list.blockSignals(False)
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
        m.addAction("🔎 Zoom to shape", lambda: self.canvas.focus_shape(idx))
        m.addAction(f"✏️ Rename '{label}'…", lambda: self.open_label_map(focus=label, scope=0))
        if self.canvas.editable:
            m.addSeparator()
            m.addAction("🏷 Relabel this shape…  (L)", lambda: self.inline.on_relabel(idx, 0))
            m.addAction("🗑 Delete this shape  (Del)", lambda: self.inline.on_delete(idx))
        m.addSeparator()
        m.addAction(utils.newIcon("tool-guidelines"), f"✓ Add as good example of '{label}'  (G)",
                    lambda: (self.canvas.select_shape(idx), add_guide_example(self, "good")))
        m.addAction(utils.newIcon("tool-guidelines"), f"✗ Add as bad example of '{label}'  (Shift+G)",
                    lambda: (self.canvas.select_shape(idx), add_guide_example(self, "bad")))
        m.addAction(f"📘 Guideline for '{label}'…", lambda: self.open_guidelines(label))
        m.exec(self.annotations_list.mapToGlobal(pos))

    def _labels_list_menu(self, pos):
        it = self.unique_labels.itemAt(pos)
        if not it or not self.session:
            return
        label = it.data(Qt.ItemDataRole.UserRole)
        m = QMenu(self)
        m.addAction(f"✏️ Rename / merge '{label}'…", lambda: self.open_label_map(focus=label))
        m.addAction("📄 Show files with this label",
                    lambda: self.apply_file_filter(self.session.keys_with_label(label), f"Files with '{label}'"))
        m.addAction("👁 Show only this label", lambda: self._solo_label(label))
        m.addSeparator()
        m.addAction("🔀 Label mapping…", self.open_label_map)
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
        self.dataset_label.setText(f"  📂 {self.session.root.name} · {self.session.type.title} · "
                                   f"{len(self.session.samples):,} files  ")
        self._update_position()
        self.refresh_status_buttons()

    # =====================================================================
    # QC tools
    # =====================================================================
    def open_tool(self, key: str):
        if key == "testing":                     # works without a dataset too (test sets, videos only)
            self.open_testing()
            return
        if not self.validate_inputs():
            return
        tool = next(t for t in QC_TOOLS if t.key == key)
        if key == "fix_editor":
            self.open_editor_session()
            return
        if tool.dialog is None:
            getattr(self.a, tool.action).trigger()
            return
        if tool.dialog.SUPPORTED is not None and self.session.type not in tool.dialog.SUPPORTED:
            QMessageBox.information(self, tool.title, f"{tool.title} is not available for {self.session.type.title}.")
            return
        if key == "remove_class" and not self.session.index_complete:
            QMessageBox.information(self, tool.title, "The label index is still being built (see status bar). "
                                    "Class removal needs complete counts — try again in a moment.")
            return
        if key == "ambiguity" and self.session.is_yolo and not self.session.true_classes:
            self.load_classes_btn.flash("warning", "Load classes.txt first")
            reply = QMessageBox.question(self, tool.title,
                                         "No dataset.yaml or classes.txt found for this YOLO dataset.\n"
                                         "Load classes.txt first to check label ambiguity?")
            if reply != QMessageBox.StandardButton.Yes or not self.load_classes_file():
                return
        dlg = self.open_dialogs.get(key)
        if dlg is None:
            dlg = tool.dialog(self)
            dlg.tool_key = key
            dlg.setWindowIcon(utils.newIcon(tool.icon))
            self.open_dialogs[key] = dlg
        fresh = not dlg.isVisible()
        dlg.show()
        if fresh:
            anim.window_fade_in(dlg)
        dlg.raise_()
        dlg.activateWindow()

    # ------------------------------------------------------------------ external editor sessions
    def _end_editor_session(self, why: str) -> bool:
        """An editor session must end (or be kept for later) before the dataset goes away."""
        dlg = self.editor_dialog
        if dlg is None or dlg.es is None or not dlg.es.active:
            return True
        if dlg.mini is not None and dlg.mini.isVisible():
            dlg.from_mini()
        else:
            dlg.show()
            dlg.raise_()
        QMessageBox.information(self, "Fix in Editor", f"Finish the editor session before you {why} "
                                                       "(or choose “Keep for later” when closing its window).")
        return False

    def open_editor_session(self, keys: list[str] | None = None):
        if not self.validate_inputs():
            return
        if self.editor_dialog is not None and self.editor_dialog.es is not None and self.editor_dialog.es.active:
            if self.editor_dialog.mini is not None and self.editor_dialog.mini.isVisible():
                self.editor_dialog.from_mini()
            else:
                self.editor_dialog.show()
                self.editor_dialog.raise_()
            return
        old = self.open_dialogs.pop("fix_editor", None)
        if old is not None:
            old.close()
        dlg = EditorSessionDialog(self, keys=keys)
        dlg.tool_key = "fix_editor"
        self.open_dialogs["fix_editor"] = dlg
        dlg.show()

    def _offer_unfinished_editor_sessions(self):
        from AIO_QC.core.edit_session import EditSession
        s = self.session
        if s is None:
            return
        for work in EditSession.unfinished(s.root):
            try:
                es = EditSession.load(work, s.root)
            except Exception as exc:
                logger.warning(f"unreadable editor session {work}: {exc}")
                continue
            c = es.counts()
            box = QMessageBox(self)
            box.setWindowTitle("Unfinished editor session")
            box.setIcon(QMessageBox.Icon.Question)
            box.setText(f"An editor session from {es.created} ({es.source_title}, {len(es.entries)} files) was not "
                        f"finished.\n{c.get('fixed', 0)} fixed · {c.get('waiting', 0)} waiting.\n\n"
                        "Resume it (edits saved in the editor since then are synced now)?")
            resume = box.addButton("Resume", QMessageBox.ButtonRole.AcceptRole)
            finish = box.addButton("Sync and finish", QMessageBox.ButtonRole.ActionRole)
            box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() in (resume, finish):
                dlg = EditorSessionDialog(self, resume=work)
                dlg.tool_key = "fix_editor"
                self.open_dialogs["fix_editor"] = dlg
                dlg.show()
                if box.clickedButton() is finish:
                    dlg._finish()
            return                                  # one at a time

    def _file_list_menu(self, pos):
        """Right-click on the file list: open / review marks / fix in editor for the selected files."""
        view = self.file_dock.view
        idx = view.indexAt(pos)
        if idx.isValid() and not view.selectionModel().isSelected(idx):
            view.setCurrentIndex(idx)
        keys = self.file_dock.selected_keys()
        if not keys or self.session is None:
            return
        n = f"{len(keys)} file(s)" if len(keys) > 1 else "file"
        m = QMenu(self)
        m.addAction(utils.newIcon("eye"), "Open", lambda: self.open_sample(keys[0]))
        m.addSeparator()
        m.addAction(utils.newIcon("review-flag"), f"Flag {n}", lambda: self.session.set_review(keys, REVIEW_FLAG))
        m.addAction(utils.newIcon("review-ok"), f"Mark {n} OK", lambda: self.session.set_review(keys, REVIEW_OK))
        m.addAction(utils.newIcon("review-clear"), "Clear review mark", lambda: self.session.set_review(keys, None))
        m.addSeparator()
        m.addAction(utils.newIcon("tool-editor"), f"Fix {n} in editor…", lambda: self.open_editor_session(keys))
        if len(keys) == 1:
            m.addSeparator()
            m.addAction(utils.newIcon("mode-video"), "Show source video in Frame Extractor",
                        lambda: self.show_source_video(keys[0]))
        m.exec(view.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------ image integrity (background + viewer)
    def _start_sweep(self):
        self._stop_sweep()
        s = self.session
        if s is None or not s.samples:
            return
        self.broken = {}
        sw = IntegritySweep(s.samples, s.adapter, s.cache, self)
        sw.batch.connect(lambda found, checked, sw=sw: self._on_sweep_batch(found, checked, sw))
        sw.finished.connect(lambda sw=sw: self._on_sweep_done(sw))
        self.sweep = sw
        sw.start()

    def _stop_sweep(self):
        sw = self.sweep
        self.sweep = None
        if sw is not None:
            try:
                sw.batch.disconnect()
            except TypeError:
                pass
            sw.stop()

    def _on_sweep_batch(self, found, checked, sw):
        from AIO_QC.core.issues import Issue
        if sw is not self.sweep or self.session is None:          # a previous dataset's sweep
            return
        issues = {k: [Issue("integrity", sev, f"{IQ_TITLES.get(c, c)}: {m}") for c, sev, m in f] for k, f in found.items()}
        self.session.issues.publish("integrity", "Image integrity (automatic)", set(checked), issues)
        self.session.issuesChanged.emit()
        new = {k: [x for x in f if x[1] == "error"] for k, f in found.items()}
        new = {k: v for k, v in new.items() if v}
        if new:
            self.broken.update(new)
            self.file_dock.model.set_flags({k: "corrupt" for k in new})
            self.notify_broken_images(sorted(self.broken), source="the background image check")
            if self.current_key in new:
                self._check_current_image()

    def _on_sweep_done(self, sw):
        if sw is self.sweep:
            logger.info(f"[image check] {len(self.session.samples) if self.session else 0:,} images checked, "
                        f"{len(self.broken):,} broken")

    def notify_broken_images(self, keys, source: str = "Image Quality"):
        """One amber notification (updated in place) with the broken images; 'Show' filters the file list."""
        keys = sorted(set(keys))
        for k in keys:
            self.broken.setdefault(k, [("corrupt", "error", "see Image Quality")])
        self.file_dock.model.set_flags({k: "corrupt" for k in keys})
        allk = sorted(self.broken)
        self.toast.show_message(
            f"<b>⚠ {len(allk):,} broken image(s)</b><br>Corrupt, truncated or inconsistent files found by {source}. "
            "Training tools may crash or silently skip them.",
            kind="warn", key="broken-images",
            actions=[("Show", lambda: self.file_dock.set_key_filter(sorted(self.broken), "🩺 Broken images")),
                     ("Image Quality…", lambda: self.open_tool("image_quality"))],
            details=self._broken_text)

    def _broken_text(self) -> str:
        by_key = self.session.by_key if self.session else {}
        lines = []
        for k in sorted(self.broken):
            why = "; ".join(str(f[-1]) for f in (self.broken[k] or [])[:3])
            lines.append(f"{by_key[k].display if k in by_key else k}  —  {why}")
        return "\n".join(lines) or "No broken images."

    def _check_current_image(self):
        """Red banner over the viewer when the open image is broken (cached check, ~2 ms otherwise)."""
        banner = getattr(self, "_banner", None)
        if banner is None:
            banner = self._banner = QLabel(self.canvas)
            banner.setWordWrap(True)
            banner.setStyleSheet("QLabel{background: rgba(198,40,40,215); color: white; border-radius: 8px;"
                                 "padding: 8px 12px; font-weight: 600;}")
            banner.hide()
        chip = getattr(self, "_class_chip", None)
        if chip is None:
            chip = self._class_chip = QLabel(self.canvas)
            chip.setStyleSheet("QLabel{background: rgba(21,101,192,220); color: white; border-radius: 12px;"
                               "padding: 4px 14px; font-weight: 700; font-size: 13pt;}")
            chip.hide()
        s = self.session
        if s is None or self.current_key not in (s.by_key if s else {}):
            banner.hide()
            chip.hide()
            return
        if s.is_classification and self.current_ann is not None:
            tags = self.current_ann.tags
            chip.setText("  ·  ".join(tags) if tags else "no class")
            chip.adjustSize()
            chip.move(14, 14)
            chip.show()
            chip.raise_()
        else:
            chip.hide()
        found = self.broken.get(self.current_key)
        if found is None:
            try:
                path, w, h = image_of(s.adapter, s.by_key[self.current_key])
                found = [f for f in integrity_findings(s.cache, path, w, h) if f[1] == "error"] if path else []
            except Exception:
                found = []
            if found:
                self.broken[self.current_key] = found
                self.file_dock.model.set_flags({self.current_key: "corrupt"})
        if found:
            banner.setText("⚠ " + "\n⚠ ".join(m for _c, _s, m in found[:3]))
            banner.setFixedWidth(max(240, self.canvas.width() - 28))
            banner.adjustSize()
            banner.move(14, 14 + (chip.height() + 8 if chip.isVisible() else 0))
            banner.show()
            banner.raise_()
        else:
            banner.hide()

    # ------------------------------------------------------------------ classification
    def change_class(self):
        s = self.session
        if s is None or not s.is_classification:
            return
        keys = self.file_dock.selected_keys() or ([self.current_key] if self.current_key else [])
        if not keys:
            return
        classes = sorted(set(s.base_adapter.class_names) | set(s.label_totals()), key=str.lower)
        cur = s.class_of(keys[0])
        name, ok = QInputDialog.getItem(self, "Change class", f"New class for {len(keys)} image(s):", classes,
                                        max(0, classes.index(cur)) if cur in classes else 0, True)
        name = (name or "").strip()
        if not ok or not name or not s.base_adapter.can_use_label(name):
            return
        if s.type.value == "cls_folders" and name not in classes and QMessageBox.question(
                self, "Change class", f"Create the new class folder “{name}”?") != QMessageBox.StandardButton.Yes:
            return
        was_current = self.current_key in keys
        done, errors = s.set_class(keys, name)
        if errors:
            self.log_errors([f"[change class] {e}" for e in errors])
        if done and was_current:
            self.file_dock.select_key(done[0] if len(done) == 1 or self.current_key not in s.by_key else self.current_key)
        self.statusBar().showMessage(f"{len(done)} image(s) → {name}" + (f" · {len(errors)} failed" if errors else ""),
                                     5000)

    def forget_dialog(self, dlg):
        stop_threads(dlg, f"{getattr(dlg, 'TITLE', type(dlg).__name__)} closing")      # before Qt destroys it
        key = getattr(dlg, "tool_key", None)
        if key and self.open_dialogs.get(key) is dlg:
            del self.open_dialogs[key]

    def validate_inputs(self) -> bool:
        if self.session is None:
            QMessageBox.warning(self, "No dataset", "Open a dataset directory first (Ctrl+O).")
            self.flash_button(self.open_dir_btn, "warning", "Open a dataset first")
            return False
        return True

    def flash_button(self, status_btn, state: str = "warning", detail: str = ""):
        status_btn.flash(state, detail)

    def open_sample(self, key: str, shape_index: int | None = None, overlay: list | None = None,
                    focus_box=None):
        """
        Called by QC dialogs: jump to a file (and shape), even if it is currently filtered out.
        overlay: temporary dashed boxes (e.g. model predictions); focus_box: glide to a pixel box.
        """
        if not self.session or key not in self.session.by_key:
            self.statusBar().showMessage(f"Not a viewable sample: {key}", 6000)
            return
        if key == self.current_key:
            self.reload_current_annotation()
            self._apply_extras(shape_index, overlay, focus_box)
            return
        self._pending_extras = (overlay, focus_box)
        self._pending_shape = shape_index
        if not self.file_dock.select_key(key):
            self.file_dock.clear_all_filters()
            self.file_dock.select_key(key)

    def _apply_extras(self, shape_index, overlay, focus_box):
        if overlay:
            self.canvas.set_overlay(overlay)
        if shape_index is not None:
            self.canvas.focus_shape(shape_index)
        elif focus_box is not None:
            from PyQt6.QtCore import QPointF, QRectF
            x1, y1, x2, y2 = focus_box
            self.canvas.focus_rect(QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized())

    def open_review_grid(self, keys, title: str):
        self.open_tool("review_grid")
        dlg = self.open_dialogs.get("review_grid")
        if dlg is not None:
            dlg.load_custom(keys, title)

    def open_crop_instances(self, pairs, title: str):
        """Crop gallery on an explicit list of (file key, shape index) — used by Model Check drill-downs."""
        self.open_tool("crop_gallery")
        dlg = self.open_dialogs.get("crop_gallery")
        if dlg is not None:
            dlg.show_instances(pairs, title)

    def apply_file_filter(self, keys, title: str):
        keys = [k for k in keys if self.session and k in self.session.by_key]
        if not keys:
            self.statusBar().showMessage("No matching files", 4000)
            return
        self.file_dock.set_key_filter(keys, title)
        self.file_list_dock.show()
        self.file_list_dock.raise_()
        self._update_position()

    # =====================================================================
    # Logs / misc
    # =====================================================================
    def log_errors(self, errors, quiet: bool = False):
        new = [e for e in errors if e not in self.error_log[-500:]]
        if not new:
            return
        self.error_log.extend(new)
        for e in new:
            logger.warning(e)
        self.a.logs.setEnabled(True)
        self.a.logs.setText(f"Logs ({len(self.error_log)})")
        if not quiet:
            self.statusBar().showMessage(f"{len(new)} new issue(s) — see Logs", 5000)

    def show_log(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Logs")
        lay = QVBoxLayout(dialog)
        text = QTextEdit(readOnly=True)
        text.setPlainText("\n".join(self.error_log))
        lay.addWidget(text)
        row = QHBoxLayout()

        def clear():
            self.error_log.clear()
            text.clear()
            self.a.logs.setText("Logs")
            self.a.logs.setEnabled(False)

        clear_btn = utils.newButton("Clear", slot=clear)
        close = utils.newButton("Close", slot=dialog.accept)
        row.addStretch(1)
        row.addWidget(clear_btn)
        row.addWidget(close)
        lay.addLayout(row)
        dialog.resize(760, 420)
        dialog.exec()

    GESTURES = [
        ("Zoom at the cursor", "Ctrl+wheel"), ("Pan", "drag (Drag mode) · hold Space (Select mode)"),
        ("Zoom in ↔ back to fit", "double-click"), ("Zoom to a shape", "double-click a shape (Select mode)"),
        ("Cycle shapes", "Tab, Shift+Tab"), ("Clear selection", "Esc"),
        ("Crop gallery: relabel / delete", "R, Del"), ("Label-error audit: error / model wrong / undo", "Y, N, U"),
        ("Open a dataset", "drop a folder on the window"),
    ]

    def show_shortcuts(self):
        """Generated from the real QActions, so the list can never drift from the app."""
        seen, rows = set(), []
        for act in list(vars(self.a).values()) + list(self.qc_actions.values()):
            if not isinstance(act, QAction) or id(act) in seen or not act.shortcuts():
                continue
            seen.add(id(act))
            keys = ", ".join(k.toString(QKeySequence.SequenceFormat.NativeText) for k in act.shortcuts())
            rows.append((act.text().replace("&", "").rstrip("…"), keys))
        rows.sort(key=lambda r: r[0].lstrip("🔀🤖📊🛡️📭🖼️📝✂️🧩🔍🚨📐⚙🎯🧠✓⚑ ").lower())
        tr = lambda a, b: f"<tr><td style='padding:2px 12px 2px 0'>{a}</td><td>{utils.fmtShortcut(b)}</td></tr>"
        html = ("<h3>Keyboard</h3><table>" + "".join(tr(a, b) for a, b in rows) + "</table>"
                "<h3>Mouse &amp; gestures</h3><table>" + "".join(
                    f"<tr><td style='padding:2px 12px 2px 0'>{a}</td><td><b>{b}</b></td></tr>" for a, b in self.GESTURES)
                + "</table>")
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        lay = QVBoxLayout(dlg)
        text = QTextEdit(readOnly=True)
        text.setHtml(html)
        lay.addWidget(text)
        dlg.resize(620, 720)
        dlg.exec()

    TOOL_NOT_HERE = {       # "what this tool deliberately does NOT do → which tool does"
        "stats": "Nothing is edited here.",
        "validation": "Label names → Labels tab · empty files → Empty · duplicates → Duplicate Images / Shapes.",
        "ambiguity": "Renaming / merging → Label Mapping.",
        "remove_class": "Renaming → Label Mapping.",
        "crop_gallery": "Whole-label renames → Label Mapping.",
        "outliers": "Ranges you define → Dim / Area.",
        "dim_area": "Statistical outliers → Outliers.",
        "train_size": "Sizes in the original image (any ranges you define) → Dim / Area.",
        "model_setup": "Running the comparison → Model Check.",
        "model_check": "Setup → Models & Environment.",
        "sampling": "Per-file marks → the review keys S / X.",
        "smart_filter": "Shape-level ranges → Dim / Area.",
        "coverage": "Setting marks → the review keys S / X.",
        "fix_editor": "Drawing inside AIO QC — geometry is edited in labelme / labelImg.",
        "inbox": "Finding issues itself → it collects what the other tools found.",
        "export": "Changing annotations → it writes a separate, clean copy.",
        "rules": "Per-class size statistics → Dim / Area and Outliers.",
        "image_quality": "Annotation problems → Validation.",
        "ocr_check": "Class / geometry comparison with a detector → Model Check.",
        "text_labels": "Reading the text from the image itself → OCR Check.",
        "sequence": "Single images without neighbours → Validation / Model Check.",
        "similarity": "Detector-based comparison (boxes, missed objects) → Model Check.",
        "review_grid": "Editing shapes → on the image (select a shape) or Fix in Editor.",
        "views": "Shape sizes / positions → Dim / Area (its results can be pinned as a fixed-list view).",
        "split_health": "Building new splits from scratch → Export & Split · duplicates inside one split → Duplicate "
                        "Images.",
        "guidelines": "Checking labels against the rules → the Shapes / Labels tools (the rule shows next to them).",
        "report": "Finding new problems → it summarises what the tools already found.",
        "handoff_export": "Editing here → on the image, or Fix in Editor for your own editor.",
        "handoff_import": "Files changed by another program in the dataset itself → picked up live (History shows them).",
    }

    def show_source_video(self, key=None):
        """QC -> Frame Extractor: the video an image was extracted from, at its frame. Only on the user's click."""
        s = self.session
        key = key or self.current_key
        smp = s.by_key.get(key) if s is not None and key else None
        if smp is None:
            return
        img = smp.image_path
        if img is None and hasattr(s.adapter, "resolve_image"):
            try:
                img = s.adapter.resolve_image(smp, None)
            except Exception:
                img = None
        if img is None:
            QMessageBox.information(self, "Source video", "This file has no image.")
            return
        from AIO_QC.tools.frame_extractor.core.frame_sources import lookup
        from AIO_QC.utils.fx_settings import fx_settings
        recent = fx_settings().value("recent/files", []) or []
        r = lookup(str(img), recent if isinstance(recent, list) else [recent])
        if r is None:
            QMessageBox.information(self, "Source video",
                                    f"No source video is known for\n{img}\n\nFrames saved by the Frame Extractor "
                                    "record their video; older frames named <video>_frame_<n> are matched against "
                                    "the Frame Extractor's recent videos.")
            return
        self._shell_call("show_in_frame_extractor", r["video"], r["frame"], r.get("time"))

    def open_testing(self, video=None, start=None, end=None):
        """Testing (Models tab) - one window per app, on the dataset open here when there is one. Never opens or
        drives the Frame Extractor's player."""
        from AIO_QC.widgets.testing.model_lab import ModelLab
        from AIO_QC.widgets.testing.qc_bridge import QCBridge
        lab = getattr(self, "testing", None)
        if lab is None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                ff = self.settings.value("fx/playback/ffmpeg", "") or ""
                lab = ModelLab(self, {}, lambda: _find_ffmpeg(ff), __version__, qc=QCBridge(self))
                lab.setWindowFlag(Qt.WindowType.Window, True)
                lab.setWindowIcon(utils.newIcon("tool-testing"))
                lab.help_requested.connect(lambda page: self.show_help_page(page or "09-model-lab.html"))
            finally:
                QApplication.restoreOverrideCursor()
            self.testing = lab
        else:
            lab.use_qc_dataset()
        if video:
            lab.add_video(video, start, end)
        fresh = not lab.isVisible()
        lab.show()
        if fresh:
            anim.window_fade_in(lab)
        lab.raise_()
        lab.activateWindow()
        return lab

    def _shell_call(self, name, *args):
        """The app shell (both modes) — absent when the window runs on its own (tests, embedding)."""
        shell = getattr(QApplication.instance(), "shell", None)
        if shell is None:
            QMessageBox.information(self, "AIO QC", "The other modes are available when AIO QC is started normally.")
            return
        getattr(shell, name)(*args)

    def open_data_prep(self):
        from AIO_QC.widgets.data_prep import DataPrepDialog
        dlg = self.open_dialogs.get("data_prep")
        if dlg is None:
            dlg = DataPrepDialog(self)
            dlg.tool_key = "data_prep"
            self.open_dialogs["data_prep"] = dlg
        fresh = not dlg.isVisible()
        dlg.show()
        if fresh:
            anim.window_fade_in(dlg)
        dlg.raise_()
        dlg.activateWindow()

    def show_get_started(self):
        from AIO_QC.widgets.get_started import GetStarted
        dlg = self.open_dialogs.get("get_started")
        if dlg is None:
            dlg = GetStarted(self)
            dlg.tool_key = "get_started"
            dlg.finished.connect(lambda _r: self.open_dialogs.pop("get_started", None))
            self.open_dialogs["get_started"] = dlg
        fresh = not dlg.isVisible()
        dlg.show()
        if fresh:
            anim.window_fade_in(dlg, 220)
        dlg.raise_()
        dlg.activateWindow()

    def show_guide(self):
        """F1: the app's one Help Center, at the Image QC guide."""
        from AIO_QC.widgets.help_center_bridge import open_help
        open_help(self, "image")

    def show_help_page(self, page=None):
        """Help for one feature. Frame Extractor and Testing chapters live in the shared Help Center."""
        from AIO_QC.widgets.help_center_bridge import open_help
        open_help(self, page)

    def show_tool_map(self):
        html = []
        for key, title, question in GROUPS:
            img = f"<img src='aio-icon:group-{key}' width=22 height=22> "
            html.append(f"<tr><td colspan=3 style='padding:10px 4px 2px 4px'>{img}<b style='font-size:13pt'>{title}</b>"
                        f" &nbsp;<i style='color:{utils.hexc('muted')}'>{question}</i></td></tr>")
            for t in QC_TOOLS:
                if t.group == key:
                    img = f"<img src='aio-icon:{t.icon}' width=20 height=20> "
                    html.append(f"<tr><td style='padding:3px 10px'>{img}<b>{t.title}</b></td>"
                                f"<td style='padding:3px 10px'>{t.tip}</td>"
                                f"<td style='padding:3px 10px;color:{utils.hexc('muted')}'>{self.TOOL_NOT_HERE.get(t.key, '—')}</td></tr>")
        dlg = QDialog(self)
        dlg.setWindowTitle("Which tool for what?")
        lay = QVBoxLayout(dlg)
        text = QTextEdit(readOnly=True)
        # icons rendered through the SVG engine (theme ink, crisp at this screen's scale) as document images
        dpr = self.devicePixelRatioF()
        doc = text.document()
        for name in [f"group-{k}" for k, _t, _q in GROUPS] + [t.icon for t in QC_TOOLS]:
            pm = utils.newIcon(name).pixmap(QSize(22, 22), dpr)
            doc.addResource(QTextDocument.ResourceType.ImageResource, QUrl(f"aio-icon:{name}"), pm.toImage())
        text.setHtml("<table cellspacing=0 border=0><tr><th align=left>Tool</th><th align=left>Answers</th>"
                     f"<th align=left>Not here → goes to</th></tr>{''.join(html)}</table>")
        lay.addWidget(text)
        dlg.resize(1000, 620)
        dlg.exec()

    def reset_all(self):
        """Soft reset: all labels visible, filters cleared, fit to window."""
        self.hidden_labels.clear()
        self.refresh_label_list()
        self.file_dock.clear_all_filters()
        self.canvas.clearSelection()
        self.load_canvas_shapes()
        self.update_annotations_list()
        self.canvas.apply_default_zoom()

    def _focus_search(self):
        self.file_list_dock.show()
        self.file_dock.search.setFocus()
        self.file_dock.search.selectAll()

    def _toggle_tags(self, on):
        self.canvas.show_tags = on
        self.load_canvas_shapes()

    def _toggle_fill(self, on):
        self.canvas.fill_shapes = on
        self.load_canvas_shapes()

    def _set_pixel_view(self, on):
        self.settings.setValue("view/pixel_view", bool(on))
        self.canvas.pixmap_item.pixel_view = bool(on)
        self.canvas._set_smooth(self.canvas.current_zoom, False)
        self.canvas.pixmap_item.invalidate()

    # ---- model predictions layer (Prediction Explorer data)
    def pred_layer_on(self) -> bool:
        return bool(self.a.pred_layer.isChecked()) if hasattr(self, "a") else False

    def set_pred_layer(self, on):
        on = bool(on)
        self.settings.setValue("view/pred_layer", on)
        if self.a.pred_layer.isChecked() != on:
            self.a.pred_layer.blockSignals(True)
            self.a.pred_layer.setChecked(on)
            self.a.pred_layer.blockSignals(False)
        dlg = self.open_dialogs.get("model_check")
        panel = getattr(dlg, "explorer", None)
        if panel is not None and panel.layer_action.isChecked() != on:
            panel.layer_action.blockSignals(True)
            panel.layer_action.setChecked(on)
            panel.layer_action.blockSignals(False)
        self.refresh_pred_layer()

    def refresh_pred_layer(self):
        if not self.pred_layer_on() or not self.session or not self.current_key:
            self.canvas.clear_pred_layer()
            return
        try:
            from AIO_QC.widgets.explorer import layer_objects
            from AIO_QC.widgets.explorer.paint import object_color, object_text
            objs, d = layer_objects(self, self.current_key)
        except Exception as exc:                          # the viewer must never break for the layer
            logger.warning(f"predictions layer: {exc}")
            objs, d = [], None
        if not objs:
            self.canvas.clear_pred_layer()
            return
        pix = self.canvas.pixmap_item.pixmap()
        i = d.index_of(self.current_key)
        W = float(d.img_w[i] or 0) if i is not None else 0.0
        k = (pix.width() / W) if W and not pix.isNull() else 1.0         # predictions are in image pixels
        dlg = self.open_dialogs.get("model_check")
        panel = getattr(dlg, "explorer", None)
        mode = panel.view.color if panel is not None and panel.data is d else "match"
        items = []
        for o in objs:
            col = object_color(o, mode, self.label_colors, d.model_colors)
            items.append({"box": tuple(v * k for v in o.box),
                          "poly": None if o.poly is None else [(x * k, y * k) for x, y in o.poly],
                          "kpts": None if o.kps is None else [(x * k, y * k, c) for x, y, c in o.kps],
                          "text": object_text(o, d.model_names, with_tag=True), "color": col.name()})
        self.canvas.set_pred_layer(items)

    def _on_mode_changed(self, mode):
        (self.a.select if mode == Canvas.SELECT else self.a.drag).setChecked(True)
        self.mode_label.setText("🖱 Select" if mode == Canvas.SELECT else "✋ Drag")

    # =====================================================================
    # Zoom
    # =====================================================================
    def updateZoomSpinbox(self):
        self._sync_zoom_buttons()
        if self.canvas.pixmap_item.pixmap().isNull():
            self.zoom_slider.set_zoom(None)
            return
        self.zoom_spin.set_zoom(self.canvas.current_zoom)
        self.zoom_slider.set_zoom(self.canvas.current_zoom)

    def _zoom_live(self, zoom: float):
        """Zoom box and slider follow the canvas while it glides (like Windows Photos: N clicks, N updates)."""
        if not self.canvas.pixmap_item.pixmap().isNull():
            self.zoom_spin.set_zoom(zoom)
            self.zoom_slider.set_zoom(zoom)

    def _zoom_from_slider(self, zoom: float):
        """Live while dragging: no glide (the slider is the animation), around the view centre."""
        c = self.canvas
        if zoom > 0 and not c.pixmap_item.pixmap().isNull():
            c.animate_to(zoom, anchor_scene=c._view_center(), anchor_view=c.viewport().rect().center(), duration=0)
            self.zoom_spin.set_zoom(c.current_zoom)

    def _sync_zoom_buttons(self):
        """Fit Window / Actual Size look pressed whenever the view IS fitted / at 100 % — however it got there
        (button, shortcut, double-click, wheel, zoom box, a new image, a window resize)."""
        if not hasattr(self, "a"):
            return
        c = self.canvas
        on_fit = on_actual = False
        if not c.pixmap_item.pixmap().isNull():
            z = c._target_zoom if c.is_animating() else c.current_zoom
            fit = c.compute_fit_scale()
            on_fit = fit > 0 and abs(z / fit - 1.0) < 0.004
            on_actual = abs(z - 1.0) < 0.004
        for act, on in ((self.a.fit, on_fit), (self.a.actual, on_actual)):
            if act.isChecked() != on:
                act.blockSignals(True)
                act.setChecked(on)
                act.blockSignals(False)

    def onZoomSpinboxChanged(self, zoom: float):
        """ZoomWidget: typed value, preset, wheel or ↑ / ↓ (1.0 = 100 %)."""
        if zoom > 0 and not self.canvas.pixmap_item.pixmap().isNull():
            self.canvas.zoom_to(zoom)

    # =====================================================================
    # 🔀 Label mapping (view-time)
    # =====================================================================
    def open_label_map(self, prefill: dict | None = None, focus: str | None = None, scope: int = 2):
        """The single place to rename / merge / map labels (view-only, or written to files)."""
        if self.validate_inputs():
            LabelMapDialog(self, prefill=prefill, focus=focus, scope=scope).exec()

    def set_label_map(self, mapping: dict[str, str]):
        s = self.session
        if s is None or dict(mapping) == dict(s.label_map):
            return
        for d in list(self.open_dialogs.values()):
            d.close()                               # their results used the previous names
        s.set_label_map(mapping)
        self.visible_labels.clear()
        self.hidden_labels.clear()
        for name in s.adapter.class_names:
            utils.update_label_colors(name, self.label_colors)
        self.reload_current_annotation()
        self.refresh_label_list()
        self._update_mapping_indicator()
        self.refresh_status_buttons()
        self.statusBar().showMessage(f"🔀 Label mapping: {len(s.label_map)} label(s) mapped" if s.label_map
                                     else "Label mapping disabled", 6000)

    def _update_mapping_indicator(self):
        n = len(self.session.label_map) if self.session else 0
        self.mapping_btn.setText(f"Mapping ({n})")
        self.mapping_btn.setVisible(n > 0)
        self.a.label_map.setText(f"Label Mapping… ({n} active)" if n else "Label Mapping…")

    # =====================================================================
    # ✓ Review markers
    # =====================================================================
    def mark_review(self, status, note=None):
        """Mark the open file, then (auto-advance) open the NEXT one. At the end of the list it stays on the last file
        — it never goes back. With a review filter the marked file may leave the list: it stays open (parked) and the
        file after it is next (core of FileDock's parking; works with live inserts / removals too)."""
        key = self.current_key
        if not self.session or not key or key not in self.session.by_key:
            return
        self.session.set_review([key], status, note)          # the dock parks the file if it left the list
        parked = self.file_dock.current_key() != key
        nxt = self.file_dock.key_at_offset(1)                 # strictly forward (from the gap when parked)
        advance = status is not None and self.a.auto_advance.isChecked()
        if (advance or parked) and nxt is not None:
            self.file_dock.select_key(nxt)
        elif advance or parked:
            mark = {REVIEW_OK: "✓ OK", REVIEW_FLAG: "⚑ flagged", REVIEW_FIXED: "🔧 fixed"}.get(status, "cleared")
            self.statusBar().showMessage(f"Marked {mark} · that was the last file in the list", 5000)
            self._update_position()

    def flag_with_note(self):
        key = self.current_key
        if not self.session or not key:
            return
        entry = self.session.review.get(key) or {}
        note, ok = QInputDialog.getText(self, "Flag with note", "What needs fixing?", text=entry.get("note", ""))
        if ok:
            self.mark_review(REVIEW_FLAG, note.strip())

    def goto_next_unreviewed(self):
        if not self.session:
            return
        status = self.session.review.status
        k = self.file_dock.next_matching(lambda key: status(key) is None)     # stops at the first hit
        if k is not None:
            self.file_dock.select_key(k)
            return
        self.statusBar().showMessage("🎉 Every file in the list has been reviewed", 6000)

    def _on_review_changed(self, keys):
        self.history_panel.touched(keys)
        self.file_dock.review_changed(keys)
        self.filmstrip.refresh_labels()
        self._update_review_badge()
        self._update_review_progress()
        self._review_save.start()

    def _save_review(self, compact: bool = False):
        """Periodic: append the new marks to the journal. compact=True (closing): one full rewrite."""
        if self.session is not None:
            try:
                self.session.review.save(compact=compact)
                if compact:
                    self.session.issues.save()
            except OSError as exc:
                self.log_errors([f"[review] could not save: {exc}"])

    def _update_review_badge(self):
        e = self.session.review.get(self.current_key) if self.session and self.current_key else None
        if not e:
            self.review_badge.setText("○ unreviewed")
            utils.style(self.review_badge, "color:@muted; padding: 0 6px;")
            self.review_badge.setToolTip("S = OK · X = flag · Shift+X = flag with note")
        elif e["status"] == REVIEW_FIXED:
            self.review_badge.setText("🔧 Fixed" + (f": {e['note'][:40]}" if e.get("note") else ""))
            utils.style(self.review_badge, "color:@info; font-weight:bold; padding: 0 6px;")
            self.review_badge.setToolTip((e.get("note", "") or "Corrected in an editor") +
                                         "\nPress S after checking it to mark it OK")
        elif e["status"] == REVIEW_OK:
            self.review_badge.setText("✓ OK")
            utils.style(self.review_badge, "color:@ok; font-weight:bold; padding: 0 6px;")
            self.review_badge.setToolTip(f"Reviewed {e.get('ts', '')}")
        else:
            self.review_badge.setText("⚑ Flagged" + (f": {e['note'][:40]}" if e.get("note") else ""))
            utils.style(self.review_badge, "color:@warn; font-weight:bold; padding: 0 6px;")
            self.review_badge.setToolTip(e.get("note", "") or f"Flagged {e.get('ts', '')}")

    def _update_review_progress(self):
        if not self.session:
            self.review_progress.setText("")
            return
        c = self.session.review_totals()                       # running counts: O(1) per keypress
        n = len(self.session.samples)
        done = c.get(REVIEW_OK, 0) + c.get(REVIEW_FLAG, 0) + c.get(REVIEW_FIXED, 0)
        pct = done * 100 / max(n, 1)
        c = {k: c.get(k, 0) for k in (REVIEW_OK, REVIEW_FLAG, REVIEW_FIXED)}
        self.review_progress.setText(f"Reviewed {done:,}/{n:,} ({pct:.1f} %) · ✓ {c[REVIEW_OK]:,} · "
                                     f"⚑ {c[REVIEW_FLAG]:,} · 🔧 {c[REVIEW_FIXED]:,}")

    def save_flagged(self):
        if not self.session:
            return
        self._review_save.stop()
        self._save_review()                                # marks are on disk before anything reads them
        dlg = FlaggedFilesDialog(self)
        dlg.exec()

    def export_review(self):
        import csv
        if not self.session:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export review report",
                                              str(self.session.root / "review_report.csv"), "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "split", "status", "note", "reviewed_at"])
            for s in self.session.samples:
                e = self.session.review.get(s.key) or {}
                w.writerow([s.ann_path or s.image_path, s.split, e.get("status", "unreviewed"),
                            e.get("note", ""), e.get("ts", "")])
        self.statusBar().showMessage(f"Review report written: {path}", 6000)

    # =====================================================================
    # Filmstrip, recent datasets, drag & drop
    # =====================================================================
    def _orient_filmstrip(self, area=None):
        """Left/right dock -> vertical column; top/bottom -> horizontal strip; floating -> by window shape."""
        dock = self.filmstrip_dock
        if dock.isFloating():
            vertical = dock.height() > dock.width()
        else:
            area = area if area is not None else self.dockWidgetArea(dock)
            vertical = area in (Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.RightDockWidgetArea)
        self.filmstrip.set_vertical(vertical)

    def _update_filmstrip(self):
        if self.session and self.current_key and self.filmstrip_dock.isVisible():
            keys = self.file_dock.visible_keys(around=self.current_key, before=FILMSTRIP_WINDOW,
                                               after=FILMSTRIP_WINDOW)
            self.filmstrip.show_around(keys, self.current_key)

    def _recent(self) -> list[tuple[str, str]]:
        raw = self.settings.value("recent_datasets", []) or []
        if isinstance(raw, str):
            raw = [raw]
        out = []
        for item in raw:
            path, _, dtype = str(item).partition("|")
            if path:
                out.append((path, dtype))
        return out

    def _add_recent(self, root: Path, dtype: DatasetType):
        entry = f"{root}|{dtype.value}"
        items = [f"{p}|{t}" for p, t in self._recent() if p != str(root)]
        self.settings.setValue("recent_datasets", [entry] + items[:MAX_RECENT - 1])

    def _build_recent_menu(self):
        self.recent_menu.clear()
        recent = self._recent()
        if not recent:
            act = self.recent_menu.addAction("(none yet)")
            act.setEnabled(False)
            return
        for i, (path, dtype) in enumerate(recent):
            exists = Path(path).is_dir()
            try:
                dt = DatasetType(dtype)
                label = dt.title
            except ValueError:
                dt, label = None, "?"
            act = self.recent_menu.addAction(f"&{(i + 1) % 10}  {Path(path).name}  —  {label}")
            act.setToolTip(path)
            act.setEnabled(exists)
            act.triggered.connect(lambda _c=False, p=path, t=dt: self.open_dataset(Path(p), t))
        self.recent_menu.addSeparator()
        self.recent_menu.addAction("Clear list", lambda: self.settings.setValue("recent_datasets", []))

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if urls and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        p = Path(urls[0].toLocalFile())
        if p.is_file():
            p = p.parent
        if p.is_dir():
            self.settings.setValue("last_dir", str(p))
            self.open_dataset(p)
