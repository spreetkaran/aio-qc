# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image QC main window — UI construction: docks, actions, toolbar, Display tab, QC panel, menus.

A mixin of AIO_QC.app.MainWindow: the methods use the window's own attributes (self.canvas, self.session …)
set up in MainWindow.__init__.
"""
from __future__ import annotations

import functools

from PyQt6.QtCore import QEvent, QObject, QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QPushButton, QSizePolicy
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence

from AIO_QC.app_common import LABEL_MAP_KEY
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QDockWidget, QFrame,
    QHBoxLayout, QLabel, QListWidget, QMenu, QToolButton, QVBoxLayout, QWidget,
)

from AIO_QC import utils
from AIO_QC.core.state import REVIEW_FLAG, REVIEW_OK
from AIO_QC.widgets import (
    Canvas, FileDock, Filmstrip,
    StatusButton, ToolBar,
)
from AIO_QC.widgets import anim
from AIO_QC.widgets import tasks as bg_tasks
from AIO_QC.widgets.guidelines import add_selected as add_guide_example
from AIO_QC.widgets.zoom_widget import ZoomWidget

from AIO_QC.app_common import (GROUPS, QC_TOOLS)



# what to use instead when a tool does not apply to a classification dataset (its disabled button's tooltip)
CLS_ALTERNATIVES = {
    "ambiguity": "Classification Check finds class names that differ only in case, spacing or a typo",
    "text_labels": "it reads text inside boxes — a classifier has none",
    "duplicates": "Image Duplicates and Classification Check (same image, other class) find duplicate images",
    "outliers": "Class Gallery ▸ Least typical first shows the images that do not look like their class",
    "sequence": "it follows boxes from frame to frame — a classifier has none",
    "dim_area": "it measures boxes — Image Quality checks image sizes",
    "train_size": "it measures objects at training size — a classifier sees the whole image",
    "ocr_check": "it reads text inside boxes — a classifier has none",
    "validation": "class folders have no annotation files to validate",
    "pairs": "class folders have no annotation files to pair",
    "fix_editor": "change classes in place: Class panel, 1 … 9, Class Gallery",
}

class UIMixin:

    # =====================================================================
    # UI construction
    # =====================================================================
    def _build_canvas(self):
        self.canvas = Canvas(self)
        if self.settings.value("viewer/gpu", False, type=bool):
            self.canvas.use_gpu()
        from AIO_QC.widgets.screenshot import Screenshots
        self.shots = Screenshots(self)              # E: the canvas as you see it, saved as a picture
        from AIO_QC.widgets.hint_banner import HintBanner
        self.hint_banner = HintBanner()             # guidance floating over the top of the image ("Start here: …")
        central = QWidget()
        central.setObjectName("canvasArea")
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.canvas, 1)
        self.setCentralWidget(central)
        self.canvas.zoomFinished.connect(self.updateZoomSpinbox)
        self.canvas.zoomChanged.connect(self._zoom_live)           # every click / wheel tick / glide frame
        self.apply_canvas_background()
        self.canvas.memoryLow.connect(self._on_low_memory)
        from AIO_QC.widgets.canvas_hud import CanvasHud
        self.hud = CanvasHud(self.canvas, self.settings.value("viewer/hud", True, type=bool))
        self.hud.top_offset = self.hint_banner.occupied
        _hud_place = self.hud.place
        self.hud.place = lambda: (_hud_place(), self._place_class_chip())     # the class chip moves with them
        self.hint_banner.attach(self.canvas, self.hud.place)   # over the image: the canvas never resizes for it
        utils.later(0, self, lambda: self.statusBar().set_fields_visible(not self.hud.enabled))
        self._hud_sel = ""
        self.canvas.cursorMoved.connect(self._hud_pointer)
        self.canvas.selectionRect.connect(self._hud_selection)
        self.canvas.shapeSelected.connect(self.onShapeSelected)
        self.canvas.imageMenuRequested.connect(lambda pos: self.shots.canvas_menu(pos))
        self.canvas.selectionCleared.connect(self.onSelectionCleared)
        self.canvas.selectionChanged.connect(self._sync_annotation_selection)    # several shapes: several rows
        self.canvas.modeChanged.connect(self._on_mode_changed)

    def _dock(self, title, widget, name, area=Qt.DockWidgetArea.RightDockWidgetArea):
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self):
        # Dataset labels (visibility + counts)
        from AIO_QC.widgets.labels_list import LabelsList
        self.unique_labels = LabelsList()
        self.unique_labels.setTextElideMode(Qt.TextElideMode.ElideNone)
        # long names scroll sideways; the counts stay pinned to the panel's right edge (LabelRowDelegate)
        self.unique_labels.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.unique_labels.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.unique_labels.horizontalScrollBar().valueChanged.connect(      # the pinned counts: repaint, not shift
            lambda _v: self.unique_labels.viewport().update())
        self.unique_labels.itemChanged.connect(self.toggle_label_visibility)
        self.unique_labels.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.unique_labels.customContextMenuRequested.connect(self._labels_list_menu)
        self.unique_labels_dock = self._dock("Labels", self.unique_labels, "labels_list_dock")
        from AIO_QC.widgets import panels
        self.unique_labels.setItemDelegate(panels.LabelRowDelegate(self.unique_labels, lambda: self.label_colors))
        self.labels_title = panels.titled(self.unique_labels_dock, "Labels")
        self._labels_eye = self.labels_title.add_widget(panels.EyeToggle("Hide all labels", "Show all labels"))
        self._labels_eye.clicked.connect(self._toggle_all_labels_eye)

        # Shapes of the current file
        self.unique_labels.setIconSize(QSize(14, 14))         # 3.0: small swatches, the name is what matters
        from AIO_QC.widgets.labels_list import PinnedRowList
        self.annotations_list = PinnedRowList()             # long text scrolls sideways, #n stays pinned right
        self.annotations_list.setIconSize(QSize(16, 16))
        # several rows (Ctrl / Shift + click, Ctrl+A) = several shapes on the image, both ways
        self.annotations_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.annotations_list.itemSelectionChanged.connect(self._on_annotation_selection)
        from AIO_QC.widgets.labels_list import GuardSelectAll
        GuardSelectAll(self.annotations_list, lambda: self.canvas.selected_group is not None)
        self.annotations_list.itemClicked.connect(self._on_annotation_clicked)
        self._ann_row_changed = False
        self.annotations_list.itemDoubleClicked.connect(
            lambda it: self.canvas.focus_shape(it.data(Qt.ItemDataRole.UserRole)))
        self.annotations_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.annotations_list.customContextMenuRequested.connect(self._annotations_menu)
        # classification datasets: the same panel shows the image's class(es), changed with one click
        from PyQt6.QtWidgets import QStackedWidget
        from AIO_QC.widgets.class_panel import ClassPanel
        self.class_panel = ClassPanel(self)
        self._ann_stack = QStackedWidget()
        self._ann_stack.addWidget(self.annotations_list)
        self._ann_stack.addWidget(self.class_panel)
        # objectName stays "labels_dock" so saved window layouts keep the dock position
        self.annotations_dock = self._dock("Annotations", self._ann_stack, "labels_dock")
        self.annotations_list.setItemDelegate(panels.AnnotationRowDelegate(self.annotations_list))
        self.annotations_title = panels.titled(self.annotations_dock, "Annotations")

        # Files
        self.file_dock = FileDock()
        self.file_dock.view.customContextMenuRequested.connect(self._file_list_menu)
        self.file_dock.currentKeyChanged.connect(self.show_sample)
        self.file_dock.view.selectionModel().selectionChanged.connect(       # several images: the Class panel
            lambda *_a: self.class_panel.refresh() if self.session is not None and self.session.is_classification
            else None)
        self.file_dock.parked.connect(lambda _k: self._update_position())
        self.file_dock.countChanged.connect(lambda _s, _t: self._update_position() if self.session else None)
        self.file_dock.countChanged.connect(lambda sh, tot: self.files_title.set_count(
            (f"{sh:,} / {tot:,}" if sh != tot else tot) if tot else None))
        self.file_list_dock = self._dock("File List", self.file_dock, "file_dock")
        self.files_title = panels.titled(self.file_list_dock, "Files")
        self.file_dock.view.setItemDelegate(panels.FileRowDelegate(
            self.file_dock, counts=lambda: self.session.index if self.session else None,
            classes=lambda: self.session is not None and self.session.is_classification))

        # History of the current file (audit trail + saved versions)
        from AIO_QC.widgets.history_panel import HistoryPanel
        self.history_panel = HistoryPanel(self)
        self.history_dock = self._dock("History", self.history_panel, "history_dock")
        self.tabifyDockWidget(self.annotations_dock, self.history_dock)
        self.annotations_dock.raise_()
        self.history_dock.hide()

        # Thumbnails
        self.filmstrip = Filmstrip()
        self.filmstrip.class_colors = lambda: self.label_colors
        self.filmstrip.keyActivated.connect(lambda k: self.file_dock.select_key(k))
        self.filmstrip_dock = self._dock("Thumbnails", self.filmstrip, "filmstrip_dock",
                                         Qt.DockWidgetArea.BottomDockWidgetArea)
        self.filmstrip_dock.visibilityChanged.connect(lambda v: v and self._update_filmstrip())
        self.filmstrip_dock.setTitleBarWidget(QWidget())   # 3.0 mock-up: the strip alone (View ▸ Thumbnails)
        panels.titled(self.history_dock, "History")
        self.filmstrip_dock.dockLocationChanged.connect(self._orient_filmstrip)
        self.filmstrip_dock.topLevelChanged.connect(lambda _floating: self._orient_filmstrip())
        self.setAcceptDrops(True)

    def _build_actions(self):
        A = functools.partial(utils.newAction, self)
        self.a = utils.struct(
            open_dir=A("Open &Directory…", self.open_dir, "Ctrl+O", icon="open",
                       tip="Open a LabelMe / YOLO / YOLO-split dataset root"),
            reload=A("&Reload Dataset", self.reload_dataset, "F5", tip="Re-scan the dataset from disk"),
            view_crop=A("View &Crop of Shape", lambda: self.view_crop(), "Shift+C", icon="crop-view",
                        tip="The selected annotation's crop in the Crop Viewer window (also: right-click it in the "
                            "Annotations panel)"),
            bg_tasks=A("Background &Tasks…", lambda: bg_tasks.show_viewer(self),
                       icon="tasks", tip="Background tasks: every job of this session, running and finished — "
                                         "progress, timings, Cancel"),
            check_guide=A("Recommended &Checks…", lambda: self.check_guide.show_dialog(), icon="next",
                          tip="The QC checks that matter most for this dataset, in order — and which are done"),
            change_class=A("Change &Class…", self.change_class, "C", icon="tool-class", enabled=False,
                           tip="Classification datasets: give the selected image(s) another class "
                               "(class folders: the image is moved; LabelMe flags: the flags are rewritten)"),
            send_issues=A("Send to &Issues", lambda: self.send_to_issues(), "I", icon="review-flag", enabled=False,
                          tip="Class folders: move the selected image(s) to the issues folder, kept by class "
                              "(issues/<class>/) — out of the dataset until you put them back; Ctrl+Z undoes"),
            send_issues_note=A("Send to Issues with Note…", lambda: self.send_to_issues(note=True), "Shift+I",
                               enabled=False, tip="Send to Issues, with a note in the issues log"),
            open_issues=A("Open Issues Folder", lambda: self.open_issues_folder(), enabled=False,
                          tip="Show the issues folder of this dataset in the file browser"),
            review_issues=A("Review Issues…", lambda: self.review_issues(), enabled=False,
                            tip="Open the issues folder as a class-folder dataset: look at the set-aside images in "
                                "the Class Gallery, fix their classes, then put them back"),
            return_issues=A("Put Issues Back…", lambda: self.return_issues(), enabled=False,
                            tip="Move every image in the issues folder back into the dataset — into the class "
                                "folder it is in now"),
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
            copy_shapes=A("Copy Shapes", lambda: self._shape_clip("copy"), "Ctrl+C",
                          enabled=False, tip="Copy the selected shapes (Ctrl / Shift + click selects several) — paste "
                                             "them into this or another image"),
            cut_shapes=A("Cut Shapes", lambda: self._shape_clip("cut"), "Ctrl+X",
                         enabled=False, tip="Copy the selected shapes and remove them from this image"),
            paste_shapes=A("Paste Shapes", lambda: self._shape_clip("paste"), "Ctrl+V",
                           tip="Paste the copied shapes into the shown image at exactly the same place (on top of "
                               "their source when it is this image); each pasted shape is one Undo step"),
            relabel_shape=A("Change Label…", lambda: self._relabel_key(0), ["L", "F2"], enabled=True,
                            tip="Change the label of the selected shape(s) — 1 … 9 set class 1 … 9 directly. Works "
                                "wherever the focus is in the window (not while typing in a text box)"),
            select_all_shapes=A("Select All Shapes", lambda: self._shape_clip("select_all"),
                                "Ctrl+A", enabled=False,
                                tip="Every shape of the image — once one is selected (as in LabelMe)"),
            history=A("File History", self._show_history, "Ctrl+H", icon="history",
                      tip="Every change to the open file — who, when, which tool — and its saved versions (revert)"),
            guide_good=A("Add as Good Example", lambda: add_guide_example(self, "good"), "G", icon="tool-guidelines",
                         tip="The selected shape becomes a ✓ good example in its class's guideline"),
            guide_bad=A("Add as Bad Example", lambda: add_guide_example(self, "bad"), "Shift+G",
                        icon="tool-guidelines", tip="The selected shape becomes a ✗ bad example in its class's guideline"),
            guide_cards=A("Guideline Cards", lambda on: self.canvas_guide.set_enabled(on), checkable=True,
                          checked=self.settings.value("ui/guide_cards", True, type=bool),
                          tip="Show a class's guideline next to the selected shape"),
            new_box=A("Create new bbox", self._start_new_box, "Ctrl+R", icon="shape-rectangle",
                      tip="Drag a new bounding box on the image, then pick its label (Esc cancels)"),
            new_polygon=A("Create new polygon", self._start_new_polygon, "Ctrl+N", icon="shape-polygon",
                          tip="Click the corners of a new polygon — click the first corner, press Enter or double-click "
                              "to close it; Ctrl+Z takes a corner back, Shift keeps 45° steps, Esc cancels — then pick "
                              "its label"),
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
            fit=A("Fit Window", self.canvas.fit_window, "Ctrl+F", icon="fit-window", tip="Fit image to window"),
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
            keep_zoom=A("Keep Previous Zoom", self._set_keep_zoom, checkable=True,
                        checked=self.settings.value("view/keep_zoom", False, type=bool),
                        tip="Open the next images on the same spot at the same magnification (scaled for images of "
                            "another size) — fit stays fit"),
            keep_display=A("Keep Previous Brightness/Contrast", self._set_keep_display, checkable=True,
                           checked=self.settings.value("view/keep_display", False, type=bool),
                           tip="Carry the Display tab's brightness, contrast, gamma and auto contrast to the next "
                               "images (off: an image opens unadjusted, or as you adjusted it earlier)"),
            pred_layer=A("Model Predictions", self.set_pred_layer, "Shift+P", icon="tool-model-check",
                         tip="Draw the models' predictions (dashed) on every image — from Model Check's last run, "
                             "at its explorer's thresholds", checkable=True,
                         checked=self.settings.value("view/pred_layer", False, type=bool)),
            all_labels=A("Show All Labels", self.toggle_all_labels, "Ctrl+Shift+H", icon="eye",
                         tip="Toggle all labels", checkable=True, checked=True),
            search=A("Search Files", self._focus_search, "Ctrl+P",
                     tip="Jump to the file list's search box"),
            clear_selection=A("Clear Selection", lambda: self.canvas.escape(), "Esc",
                              tip="Deselect the shape (any mode)"),
            next_shape=A("Next Shape", lambda: self.canvas.cycle_shape(1), tip="Select next shape (Tab in canvas)"),
            next_finding=A("Next Finding", lambda: self.step_finding(1), "F8", icon="review-next",
                           tip="Next finding of the check used last — opened on the image, ready to fix"),
            prev_finding=A("Previous Finding", lambda: self.step_finding(-1), "Shift+F8",
                           tip="Previous finding of the check used last"),
            prev_shape=A("Previous Shape", lambda: self.canvas.cycle_shape(-1),
                         tip="Select previous shape (Shift+Tab in canvas)"),
            # tools
            rename=A("Rename / Merge Label…", self._rename_selected_label, "Ctrl+E", icon="edit",
                     tip="Opens Label Mapping focused on the selected label (the single place for renames)"),
            model_setup=A("Models && Environment…", self.open_model_setup, "Ctrl+Alt+N", icon="tool-model-setup",
                          tip="Python environment, models, class mapping, device and image size for Model Check"),
            label_map=A("Label Mapping…", self.open_label_map, LABEL_MAP_KEY, icon="tool-label-map",
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
            screenshot=A("Save Screenshot", lambda: self.shots.save(), "E", icon="screenshot",
                         tip="Save the image as you see it — zoom, labels, adjustments, sketch — into the "
                             "screenshots folder (E; right-click the image for more)"),
            screenshot_as=A("Save Screenshot As…", lambda: self.shots.save_as(), "Shift+E", icon="save",
                            tip="Save the image as you see it under a name and folder you choose"),
            screenshot_copy=A("Copy Screenshot", lambda: self.shots.copy(), icon="copy",
                              tip="Copy the image as you see it to the clipboard"),
            screenshot_folder=A("Open Screenshots Folder", lambda: self.shots.open_folder(), icon="open",
                                tip="Where E saves screenshots (Settings ▸ Screenshots)"),
            logs=A("Logs", self.show_log, "Ctrl+Shift+L", icon="logs", tip="Errors collected during this session",
                   enabled=False),
            sketch=A("Sketch on Image", lambda: self.sketch.toggle(), "P", icon="tool-sketch", checkable=False,
                     tip="Draw on the image (pen, highlighter, shapes, arrows, text, eraser) to explain something; "
                         "labels are unticked — tick the ones you want; copy / save the result"),
            roi=A("ROI Capture", lambda: self.roi.toggle(), "O", icon="tool-roi",
                  tip="Draw boxes like a snipping tool and keep their real image coordinates (2 decimals) per image; "
                      "copy, export CSV / TXT / JSON, save crops"),
            data_prep=A("Preparation…", self.open_data_prep, "Ctrl+Alt+P", icon="tool-prep",
                        tip="Raw data → dataset: unzip, combine & flatten, orphans, integrity, exact duplicates, resize "
                            "(no dataset needed; every run can be undone)"),
            autoannotate=A("Auto-Annotate Videos…", self.open_autoannotate, "Ctrl+Shift+U", icon="tool-autoannotate",
                           tip="Videos → unique frames → LabelMe files with your model, in steps you can run one by "
                               "one (frames only, annotations only, …)"),
            testing=A("Testing…", lambda: self.open_testing(), "Ctrl+Alt+L", icon="tool-testing",
                      tip="Measure and compare detection models on this dataset, a test set or videos"),
            convert_yolo=A("Convert to YOLO…", lambda: self.open_tool("convert_yolo"), "Ctrl+Alt+Y",
                           icon="tool-convert", tip="Convert & Split, set to YOLO (Output step)"),
            train=A("Train Model…", lambda: self.open_training(), "Ctrl+Alt+B", icon="tool-train",
                    tip="Train an Ultralytics YOLO model on a YOLO dataset in your Python environment — live progress, "
                        "stop / pause / resume, per-class results"),
            assist_suggest=A("Suggest Labels", lambda: self.assist.suggest(), "Ctrl+Alt+J", icon="tool-assist",
                             tip="Model Assist: the dataset's models suggest labels on this image — click one to "
                                 "accept it, right-click to drop it (Models & Environment)"),
            assist_auto=A("Suggest on Every Image", self._set_assist_auto, checkable=True,
                          checked=self.settings.value("assist/auto", False, type=bool),
                          tip="Model Assist on every image shown (predictions are cached: instant on a revisit)"),
            assist_accept_all=A("Accept All Suggestions", lambda: self.assist.accept_all(), "Ctrl+Alt+Return",
                                enabled=False, tip="Every suggestion at or above the confidence — each its own "
                                                   "undo step"),
            assist_clear=A("Clear Suggestions", lambda: self.assist.clear(), enabled=False),
            assist_sam=A("Outline Boxes with SAM", self._set_assist_sam, "Ctrl+Alt+X", checkable=True,
                         checked=self.settings.value("assist/sam", False, type=bool),
                         tip="A box you draw — or a box suggestion you accept — becomes the object's outline (SAM / "
                             "SAM2 in the models' environment; ultralytics downloads the weights the first time)"),
            assist_settings=A("Model Assist Settings…", self._assist_settings,
                              tip="The confidence suggestions need, and the SAM weights"),
            improve=A("Improve Model…", lambda: self.open_improve(), "Ctrl+Alt+U", icon="tool-improve",
                      tip="Check a trained model on this dataset, label where it is weak, fine-tune — round by round"),
            augment=A("Image Augmentation…", lambda: self.open_augment(), "Ctrl+Alt+A", icon="tool-augment",
                      tip="Build an augmentation recipe with a live preview — train with it in memory, or write an "
                          "augmented copy of the dataset"),
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
            health_check=A("Health Check…", self.open_health_check, "Ctrl+Alt+H", icon="status-ok",
                           tip="Run the common checks in one go and see what needs fixing, in plain words"),
            tour=A("Quick Tour", lambda: self.start_tour(force=True), icon="get-started",
                   tip="A short guided tour of the window: where everything is"),
            sample=A("Open a Sample Dataset", self.open_sample_dataset, icon="folder-line",
                     tip="A small street dataset with a few problems planted on purpose — to learn without your own "
                         "data"),
            guide=A("User Guide", self.show_guide, "F1", icon="info",
                    tip="How everything works — every mode, every tool, and which tool for what"),
            keymap=A("Keyboard Shortcuts", self.toggle_keymap, ["?", "Ctrl+F1"], icon="keyboard",
                     tip="Every key and mouse gesture, over the window — type to filter, click to run"),
            next_panel=A("Next Panel", lambda: self.panel_cycler.cycle(1), "F6",
                         tip="Move the keyboard focus to the next panel"),
            prev_panel=A("Previous Panel", lambda: self.panel_cycler.cycle(-1), "Shift+F6"),
        )
        group = QActionGroup(self)
        group.setExclusive(True)
        group.addAction(self.a.drag)
        group.addAction(self.a.select)
        self.qc_actions: dict[str, QAction] = {}
        for t in QC_TOOLS:
            if t.action:                      # e.g. Label Mapping: same QAction everywhere, one shortcut
                act = getattr(self.a, t.action)
                act.setText(f"{t.title}…".replace("&", "&&"))       # "&" is a mnemonic in Qt labels
                utils.set_icon(act, t.icon)
            else:
                act = A(t.title.replace("&", "&&"), functools.partial(self.open_tool, t.key), t.shortcut, icon=t.icon, tip=t.tip)
            self.qc_actions[t.key] = act
        self.addAction(self.a.convert_yolo)           # Ctrl+Alt+Y: Convert & Split on its Convert tab (no own entry)

    def _build_toolbar(self):
        self.toolbar = ToolBar("Quick Actions")
        self.toolbar.setObjectName("main_toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)   # labelme style
        self.toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        self.open_dir_btn = StatusButton("Open", icon="folder-line", slot=self.open_dir, hint="Ctrl+O",
                                         dot_on_success=False)
        self.open_dir_btn.set_base_tooltip("Open dataset root: LabelMe dir, YOLO root or YOLO split root (Ctrl+O)")
        self.load_classes_btn = StatusButton("Classes", icon="list", slot=self.load_classes_file)
        self.load_classes_btn.set_status("default")
        self.load_classes_btn.set_base_tooltip("Class list (Ctrl+L): classes.txt (one per line) or .yaml.\n"
                                               "YOLO: dataset.yaml is used automatically; classes.txt only if it is absent.")
        from AIO_QC.widgets.breadcrumb import Breadcrumb
        self.dataset_label = Breadcrumb()                 # dataset › split › file  [LabelMe · 30 files]
        self.dataset_label.rootClicked.connect(lambda: self.session and QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.session.root))))

        self.zoom_spin = ZoomWidget(Canvas.MIN_ZOOM, Canvas.MAX_ZOOM)      # "Zoom" + level to one decimal
        self.zoom_spin.setEnabled(False)
        self.zoom_spin.zoomRequested.connect(self.onZoomSpinboxChanged)
        from AIO_QC.widgets.zoom_widget import ZoomSlider
        self.zoom_slider = ZoomSlider(Canvas.MIN_ZOOM, Canvas.MAX_ZOOM)    # Paint-style: − ━━●━━ +
        self.zoom_slider.zoomRequested.connect(self._zoom_from_slider)

        self.health_btn = StatusButton("Health check", icon="status-ok", slot=lambda: self.open_health_check(),
                                       dot_on_success=False)
        self.health_btn.set_base_tooltip("Run the common checks in one go and see what needs fixing (Ctrl+Alt+H)")
        self.toolbar.addWidget(self.open_dir_btn)
        self.toolbar.addWidget(self.load_classes_btn)
        self.toolbar.addWidget(self.health_btn)
        self.toolbar.addWidget(self.dataset_label)
        spacer = QWidget()                                 # 3.0: dataset on the left, view tools on the right
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(spacer)
        # ---- the right-hand groups, as in the approved mock-up
        from AIO_QC.widgets.chrome import Group, Segmented, TasksButton, action_button
        self.zoom_spin.caption.setVisible(False)          # the − 37 % ▾ + group speaks for itself
        self.zoom_spin.set_compact(True)
        self.zoom_spin.field.setStyleSheet("QFrame#zoomField{border:0; background:transparent;}")
        self._chrome_btns = {
            "prev": action_button(self.a.prev, "chevron-left"), "next": action_button(self.a.next, "chevron-right"),
            "drag": action_button(self.a.drag, "hand-line", size=17),
            "select": action_button(self.a.select, "cursor-line", size=17),
            "zoom_out": action_button(self.a.zoom_out, text="−"), "zoom_in": action_button(self.a.zoom_in, text="+"),
            "fit": action_button(self.a.fit, "fit-corners"), "actual": action_button(self.a.actual, text="1:1"),
            "shot": action_button(self.a.screenshot, "camera")}
        for b in self._chrome_btns.values():
            b.setProperty("chrome", True)
        self.tool_segment = Segmented([self._chrome_btns["drag"], self._chrome_btns["select"]])
        self.zoom_group = Group([self._chrome_btns["zoom_out"], self.zoom_spin, self._chrome_btns["zoom_in"]])
        self.tasks_btn = TasksButton(self.a.bg_tasks)
        self.tasks_btn.setProperty("chrome", True)
        for w in (self._chrome_btns["prev"], self._chrome_btns["next"], None, self.tool_segment, None,
                  self.zoom_group, self._chrome_btns["fit"], self._chrome_btns["actual"], None,
                  self._chrome_btns["shot"], self.tasks_btn):
            if w is None:
                self.toolbar.addSeparator()
            else:
                self.toolbar.addWidget(w)
        self._slider_action = self.toolbar.addWidget(self.zoom_slider)          # Settings ▸ Viewer (off by default)
        self.set_zoom_slider_visible(self.settings.value("viewer/zoom_slider", False, type=bool))
        self.toolbar.visibilityChanged.connect(lambda v: self.a_toolbar.setChecked(v))
        self.canvas.pixmap_item.pixel_view = self.a.pixel_view.isChecked()
        self.canvas.keep_zoom = self.a.keep_zoom.isChecked()
        from AIO_QC.widgets.zoom_widget import HoldToZoom
        self._hold_zoom = []
        for key, d in (("zoom_in", 1), ("zoom_out", -1)):      # click = one step · hold = smooth zoom
            self._hold_zoom.append(HoldToZoom(self.canvas, self._chrome_btns[key], d))

    def _build_display_tab(self) -> QWidget:
        """
        View-only adjustments that exist nowhere else (the old View tab only repeated toolbar actions):
        image brightness / contrast / gamma / auto-contrast, annotation fill opacity and outline width.
        Files are never changed; values are remembered between sessions.
        """
        from PyQt6.QtWidgets import QCheckBox, QScrollArea, QSlider
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(12, 4, 12, 4)
        lay.setSpacing(14)
        st = self.settings
        self.disp = {}
        self._disp_fmt = {}

        def slider(key, lo, hi, tip, fmt):
            box = QVBoxLayout()
            box.setSpacing(4)
            lab = QLabel()
            utils.style(lab, "color: @muted;")
            # as wide as its widest value: never cut ("Gamma 1.0…"), never moving the sliders beside it
            lab.setMinimumWidth(max(lab.fontMetrics().horizontalAdvance(fmt(v)) for v in (lo, hi, (lo + hi) // 2)) + 6)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setFixedWidth(120)
            sl.setToolTip(tip)
            keep = key not in self.IMAGE_KEYS or self.a.keep_display.isChecked()
            sl.setValue(int(st.value(f"display/{key}", self.DISPLAY_DEFAULTS[key]) if keep
                            else self.DISPLAY_DEFAULTS[key]))
            upd = lambda v, l=lab: l.setText(fmt(v))
            self._disp_fmt[key] = upd
            upd(sl.value())
            sl.valueChanged.connect(upd)
            sl.valueChanged.connect(lambda _v, k=key: self._display_changed(k))
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
        self.disp_auto.setChecked(st.value("display/auto", False, type=bool) and self.a.keep_display.isChecked())
        self.disp_auto.toggled.connect(lambda _v: self._display_changed("auto"))
        checks = QVBoxLayout()
        checks.setSpacing(4)
        checks.addWidget(self.disp_auto)
        lay.addLayout(checks)
        self.disp_keep = QCheckBox("Keep for next images")
        self.disp_keep.setToolTip(self.a.keep_display.toolTip())
        self.disp_keep.setChecked(self.a.keep_display.isChecked())
        self.disp_keep.toggled.connect(lambda on: self.a.keep_display.isChecked() != on and self.a.keep_display.trigger())
        self.a.keep_display.toggled.connect(lambda on: self.disp_keep.isChecked() != on and self.disp_keep.setChecked(on))
        checks.addWidget(self.disp_keep)
        lay.addWidget(utils.custom_v_separator())
        lay.addWidget(QLabel("<b>Annotations</b>"))
        slider("fill", 0, 255, "Opacity of the shape fill (when Fill Shapes is on)", lambda v: f"Fill opacity {v * 100 // 255}%")
        slider("outline", 50, 300, "Outline thickness", lambda v: f"Outline ×{v / 100:.1f}")
        slider("tag", 75, 200, "Size of the class-name labels on the shapes (they keep this size at every zoom)",
               lambda v: f"Label size {v}%")
        from AIO_QC.widgets.ribbon_page import RibbonButton
        toggles = QVBoxLayout()
        toggles.setSpacing(2)
        for act in (self.a.fill, self.a.tags):
            b = RibbonButton(act)
            b.set_large(False)
            toggles.addWidget(b)
        lay.addLayout(toggles)
        lay.addWidget(utils.custom_v_separator())
        reset = utils.newButton("Reset", icon="undo", slot=self._reset_display, tip="Back to the original image and default styling")
        lay.addWidget(reset)
        lay.addStretch(1)
        self._display_timer = QTimer(self, singleShot=True, interval=60, timeout=self._apply_display)
        QTimer.singleShot(0, self._apply_display)
        # narrower windows: the controls scroll sideways instead of being squeezed and cut
        area = QScrollArea()
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidgetResizable(True)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        area.setWidget(w)
        area.setMinimumHeight(w.sizeHint().height() + 14)
        return area

    def _display_changed(self, key: str):
        """A slider / Auto contrast moved by the user: re-render soon, and remember the image adjustment for this
        file (labelme-style: coming back to it shows it adjusted the same way)."""
        if key in self.IMAGE_KEYS and self.current_key is not None:
            self._display_owner = self.current_key
        self._display_timer.start()

    def _image_display_values(self) -> dict:
        v = {k: self.disp[k].value() for k in self.IMAGE_KEYS if k in self.disp}
        v["auto"] = self.disp_auto.isChecked()
        return v

    def _flush_display(self):
        """Before another image opens: a pending slider change still belongs to the image it was made on."""
        if self._display_timer.isActive():
            self._display_timer.stop()
            self._apply_display()

    def _display_for_next(self, key: str):
        """labelme's rule for the image about to open: its own adjustment from earlier in this session, else the
        previous image's (Keep Previous Brightness/Contrast), else none. Set before the pixmap is shown, so it is
        rendered once, already adjusted."""
        if not hasattr(self, "disp"):
            return
        if key in self._display_by_key:
            want = self._display_by_key[key]
        elif self.a.keep_display.isChecked():           # carried over, and from now on this image's own (labelme)
            cur = self._image_display_values()
            if cur != {k: self.DISPLAY_DEFAULTS[k] for k in self.IMAGE_KEYS}:
                self._display_by_key[key] = cur
            return
        else:
            want = {k: self.DISPLAY_DEFAULTS[k] for k in self.IMAGE_KEYS}
        if want == self._image_display_values():
            return
        for k in self.IMAGE_KEYS:
            if k == "auto":
                self.disp_auto.blockSignals(True)
                self.disp_auto.setChecked(bool(want[k]))
                self.disp_auto.blockSignals(False)
            else:
                sl = self.disp[k]
                sl.blockSignals(True)
                sl.setValue(int(want[k]))
                sl.blockSignals(False)
                self._disp_fmt[k](sl.value())                # its value label
        self._apply_display(render=False)                    # canvas.loadPixmap renders it

    def _set_keep_zoom(self, on):
        self.settings.setValue("view/keep_zoom", bool(on))
        self.canvas.keep_zoom = bool(on)

    def _set_keep_display(self, on):
        self.settings.setValue("view/keep_display", bool(on))

    def _apply_display(self, render: bool = True):
        v = {k: sl.value() for k, sl in self.disp.items()}
        auto = self.disp_auto.isChecked()
        for k, val in v.items():
            self.settings.setValue(f"display/{k}", val)
        self.settings.setValue("display/auto", auto)
        owner = getattr(self, "_display_owner", None)
        if owner is not None:                                  # the user adjusted this image
            self._display_owner = None
            self._display_by_key[owner] = self._image_display_values()
        self.canvas.display.update(brightness=v["brightness"], contrast=v["contrast"], gamma=v["gamma"] / 100, auto=auto)
        if render:
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
        if self.current_key is not None:
            self._display_owner = self.current_key
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
        from AIO_QC.widgets.ribbon_page import RibbonPage
        q = self.qc_actions
        # shorter names in the drawer only (menus, palette and tooltips keep the full names)
        labels = {"Flagged Screenshots": "Screenshots", "Remove / Re-order Classes": "Remove / Reorder",
                  "Models & Environment": "Models & Setup", "Auto-advance after marking": "Auto-advance",
                  "Export Review Report…": "Review report", "Save Flagged Files…": "Save flagged",
                  "Export Hand-off Batch": "Hand-off", "Flag with note…": "Flag + note"}
        # Classification datasets: the Shapes tab is "Images" (classifier versions of its checks); Relationship Rules
        # for image classes (never together, requires …) sit on the Labels tab — this button shows only for them
        self.rules_cls_act = QAction(utils.newIcon("tool-rules"), "Relationship Rules", self)
        self.rules_cls_act.setObjectName("rules_cls")
        self.rules_cls_act.setToolTip("Rules for image classes — never together, requires, how many per image "
                                      "(Ctrl+Shift+K)")
        self.rules_cls_act.triggered.connect(lambda: self.open_tool("rules"))
        self.rules_cls_act.setProperty("aio_type_hidden", True)
        self._ribbon_pages = {}
        for key, title, question in GROUPS:
            tools = [q[t.key] for t in QC_TOOLS if t.group == key]
            if key == "labels":
                tools.append(self.rules_cls_act)
            extra = None
            if key == "review":
                order = ("inbox", "review_grid", "smart_filter", "views", "coverage", "sampling", "fix_editor",
                         "screenshots", "report", "export")
                groups = [[q[k] for k in order[:6]],
                          [q[k] for k in order[6:]] + [(q["handoff_export"], [q["handoff_export"], q["handoff_import"]])],
                          ("small", [self.a.mark_ok, self.a.mark_flag, self.a.flag_note, self.a.mark_clear,
                                     self.a.next_unreviewed, self.a.auto_advance]),
                          ("small", [self.a.save_flagged, self.a.export_review])]
                self.review_progress = QLabel("")
                utils.style(self.review_progress, "color:@muted; padding-left:6px;")
                extra = self.review_progress
            else:
                groups = [tools]
            tab = RibbonPage(groups, question, labels, extra=extra)
            self._ribbon_pages[key] = tab
            i = self.tabs.addTab(tab, utils.newIcon(f"group-{key}"), title)
            self.tabs.setTabToolTip(i, question)
        self.tabs.addTab(self._build_display_tab(), utils.newIcon("brightness-contrast"), "Display")
        self.tabs.setIconSize(QSize(18, 18))
        for i in range(min(9, self.tabs.count())):       # 3.0: the key beside each tab (Alt+N opens it)
            self.tabs.set_tab_hint(i, f"Alt {i + 1}")
        search = QPushButton("")
        search.setObjectName("paletteField")
        search.setCursor(Qt.CursorShape.IBeamCursor)
        search.setToolTip("Command palette: run any tool or action by typing a few letters (Ctrl+K)")
        search.setFixedSize(300, 32)                # 300 px when there is room; it gives way before the tabs do
        utils.style(search, "QPushButton#paletteField{text-align:left; background:@surface; border:1px solid @line;"
                            " border-radius:6px; padding:0;}"
                            "QPushButton#paletteField:hover{border-color:@accent;}")
        from AIO_QC.widgets.chrome import KeyCap
        sl = QHBoxLayout(search)
        sl.setContentsMargins(10, 0, 6, 0)
        sl.setSpacing(8)
        mag = utils.IconLabel("search", 15)
        mag.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        ph = QLabel("Search tools & actions…")
        ph.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        utils.style(ph, "color:@muted; background:transparent;")
        ph.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        cap = KeyCap("Ctrl K")
        sl.addWidget(mag)
        sl.addWidget(ph, 1)
        sl.addWidget(cap)

        class _Narrow(QObject):                    # a narrow window: the field becomes just the magnifier
            def eventFilter(self, obj, ev):
                if ev.type() == QEvent.Type.Resize:
                    w = obj.width()
                    ph.setVisible(w >= 210)
                    cap.setVisible(w >= 120)
                return False
        self._search_narrow = _Narrow(search)
        search.installEventFilter(self._search_narrow)
        search.clicked.connect(lambda _c=False: self.open_palette())
        wrap = QWidget()
        wl = QHBoxLayout(wrap)
        wl.setContentsMargins(0, 4, 12, 4)
        wl.addWidget(search)
        self.tabs.corner.addWidget(wrap)
        self.search_box = search
        bar, row = self.tabs.bar, self.tabs.row

        class _Share(QObject):                     # the tabs keep their full width; the search field takes the rest
            def eventFilter(self, obj, ev):
                if ev.type() == QEvent.Type.Resize:
                    room = obj.width() - bar.sizeHint().width() - 40
                    search.setFixedWidth(max(44, min(300, room)))
                return False
        self._search_share = _Share(row)
        row.installEventFilter(self._search_share)
        self.palette_field = search
        self.tabs.setObjectName("qcRibbon")
        self.tabs.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        utils.style(self.tabs, "#qcRibbon{background:@base; border-bottom:1px solid @line;}")
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
        utils.addActions(self.menus.file, (self.a.reload, self.a.load_classes, None,
                                           {"menu": "Screenshot", "icon": "screenshot", "actions": self.shots.menu_actions()},
                                           None, self.a.open_trash, None,
                                           self.a.next_mode, self.a.to_video, self.a.to_studio, self.a.source_video,
                                           None, self.a.exit))
        # preferences (compact toolbar, watch, guideline cards, animations, auto-advance, storage …) live in
        # File ▸ Settings… (widgets.settings_pages.image_sections); their actions stay for shortcuts / handlers
        for act in (self.a.compact_toolbar, self.a.watch, self.a.guide_cards, self.a.animations,
                    self.a.auto_advance, self.a.clear_cache):
            self.addAction(act)
        from AIO_QC.widgets.window_frame import install_brand
        install_brand(self)                                # 3.0: the app's mark + name, the same in every mode
        from AIO_QC.widgets.mode_switch import ModeSwitch
        self.mode_switch = ModeSwitch("image", self)
        self.mode_switch.switched.connect(lambda m: self._shell_call("switch", m))
        mb.setCornerWidget(self.mode_switch, Qt.Corner.TopRightCorner)
        self.mode_switch.setVisible(getattr(QApplication.instance(), "shell", None) is not None)
        from AIO_QC.widgets.settings_dialog import install_settings
        install_settings(self)                              # gear in the corner + File ▸ Settings… (Ctrl+,)
        for n in range(1, 10):                              # 1 … 9: that class for the selected shape(s)
            act = QAction(f"Set Class {n}", self)
            act.setShortcut(QKeySequence(str(n)))
            act.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            act.triggered.connect(lambda _c=False, n=n: self._relabel_key(n))
            self.addAction(act)
        for act in (self.a.copy_shapes, self.a.cut_shapes, self.a.paste_shapes, self.a.select_all_shapes,
                    self.a.relabel_shape):
            # The whole window (image, Annotations / Files panels, toolbar), not only a focused image: shapes are
            # often picked in the Annotations panel, and Ctrl+V comes after clicking the next file. Text fields
            # keep their own copy / paste (Qt gives them these keys first; _shape_clip hands them back otherwise).
            act.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            self.addAction(act)
        utils.addActions(self.menus.edit, (self.a.undo_edit, self.a.redo_edit, None, self.a.copy_shapes,
                                           self.a.cut_shapes, self.a.paste_shapes, self.a.select_all_shapes,
                                           self.a.relabel_shape, None,
                                           self.a.new_box, self.a.new_polygon, None,
                                           {"menu": "Model Assist", "actions": (
                                               self.a.assist_suggest, self.a.assist_auto, self.a.assist_accept_all,
                                               self.a.assist_clear, None, self.a.assist_sam, None,
                                               self.a.assist_settings)}, None,
                                           self.a.guide_good, self.a.guide_bad, None, self.a.history))
        A = functools.partial(utils.newAction, self)
        self.a_toolbar = A("Toolbar", lambda c: self.toolbar.setVisible(c), checkable=True, checked=True)
        dock_toggles = [self._panel_action(d, t) for d, t in
                        ((self.file_list_dock, "File List"), (self.annotations_dock, "Annotations"),
                         (self.history_dock, "History"), (self.unique_labels_dock, "Labels"),
                         (self.qc_dock, "QC Panel"), (self.filmstrip_dock, "Thumbnails"))]
        utils.addActions(self.menus.view, (
            {"menu": "Interaction Mode", "actions": (self.a.drag, self.a.select)},
            None, self.a.zoom_in, self.a.zoom_out, self.a.fit, self.a.actual, self.a.keep_zoom,
            self.a.keep_display, None,
            self.a.tags, self.a.fill, self.a.all_labels, self.a.pred_layer, None,
            {"menu": "Panels", "actions": (*dock_toggles, None, self.a_toolbar)},
            self.a.pin_qc, self.a.focus,
        ))
        utils.addActions(self.menus.go, (self.a.next, self.a.prev, self.a.next_unreviewed, None,
                                         self.a.next_finding, self.a.prev_finding, None,
                                         self.a.next_shape, self.a.prev_shape, self.a.clear_selection, None,
                                         self.a.search))
        # QC menu mirrors the first four tabs; the Review menu mirrors the Review tab
        utils.addActions(self.menus.qc, [
            {"menu": title, "actions": [self.qc_actions[t.key] for t in QC_TOOLS if t.group == key]
             + ([None, self.a.rename] if key == "labels" else [])}
            for key, title, _q in GROUPS if key != "review"])
        utils.addActions(self.menus.review, [self.a.health_check, self.a.check_guide, None] +
                         [self.qc_actions[t.key] for t in QC_TOOLS if t.group == "review"] + [
            None, self.a.change_class, self.a.mark_ok, self.a.mark_flag, self.a.flag_note, self.a.mark_clear, None,
            {"menu": "Issues Folder", "actions": [self.a.send_issues, self.a.send_issues_note, None,
                                                  self.a.review_issues, self.a.return_issues, self.a.open_issues]},
            None, self.a.save_flagged, self.a.export_review])       # moving to the next one: Go menu (one place each)
        utils.addActions(self.menus.tools, (self.a.sketch, self.a.roi, self.a.view_crop, None, self.a.autoannotate, self.a.ann_table, None,
                                            self.a.palette, None, self.a.storage, None, self.a.reset, self.a.logs,
                                            self.a.bg_tasks))
        about = QAction(utils.appIcon(), "About Image QC", self)
        about.setMenuRole(QAction.MenuRole.AboutRole)
        from AIO_QC.widgets.about import show_about
        about.triggered.connect(lambda _c=False: show_about(
            self, "Image QC", "Quality control for annotated image datasets (LabelMe / YOLO)", utils.appIcon()))
        self.a.about = about
        # Help: one of each — the tour (where things are), the guide (how they work, incl. which tool for what),
        # the shortcuts (the live key map: every key and gesture, filter, click to run)
        utils.addActions(self.menus.help, (self.a.tour, self.a.guide, self.a.keymap, None, self.a.sample, None,
                                           about))
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
        from AIO_QC.widgets.review_badge import IconText
        self.mode_label = IconText("hand-line", "Drag")      # the toolbar's Drag / Select icons
        fm = self.mode_label.label.fontMetrics()
        self.mode_label.setFixedWidth(max(fm.horizontalAdvance("Select"), fm.horizontalAdvance("Drag")) + 14 + 5 + 8 + 14)
        from AIO_QC.widgets.review_badge import ReviewBadge
        self.review_badge = ReviewBadge()                # icon + Pending / OK / Flagged / Fixed, fixed width
        self.mapping_btn = QToolButton()
        self.mapping_btn.setAutoRaise(True)
        self.mapping_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        utils.set_icon(self.mapping_btn, "tool-label-map")
        self.mapping_btn.setToolTip("Label mapping is active — click to edit")
        self.mapping_btn.clicked.connect(lambda _c=False: self.open_label_map())
        self.mapping_btn.setVisible(False)
        self.statusBar().addPermanentWidget(self.mapping_btn)
        self.flagged_btn = QToolButton()                 # ⚑ N flagged — click: show only them (again: all files)
        self.flagged_btn.setAutoRaise(True)
        self.flagged_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        utils.set_icon(self.flagged_btn, "review-flag")
        self.flagged_btn.clicked.connect(lambda _c=False: self.toggle_flagged_filter())
        self.flagged_btn.setVisible(False)
        self.statusBar().addPermanentWidget(self.flagged_btn)
        from AIO_QC.widgets.check_guide import CheckGuide
        self.check_guide = CheckGuide(self)              # ▶ Next check: … (recommended checks for this dataset)
        self.file_dock.review_box.currentIndexChanged.connect(lambda _i: self._update_flagged_count())
        self.statusBar().addPermanentWidget(self.review_badge)
        for w in (self.index_label, self.pos_label, self.mode_label):
            utils.style(w, "color:@muted; padding: 0 6px;")
            self.statusBar().addPermanentWidget(w)
        self.pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._review_save = QTimer(self, singleShot=True, interval=800, timeout=self._save_review)

    def _restore_settings(self):
        geo = self.settings.value("geometry")
        state = self.settings.value("windowState")
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None and self.settings.value("ui/layout_v3", False, type=bool):
            self.restoreState(state)
        else:                                             # 3.0: start from the approved layout once
            self.settings.setValue("ui/layout_v3", True)
            utils.later(400, self, self._default_dock_sizes)     # once the window has its size
        self._set_toolbar_compact(self.a.compact_toolbar.isChecked())
        if not self.settings.value("ui/ribbon_v1", False, type=bool):
            upgraded = bool(self.settings.value("recent_datasets"))    # someone who knew the old QC Panel
            self.settings.setValue("ui/ribbon_v1", True)     # the panel is slim now: show it once after the update
            self.qc_dock.show()
            if upgraded:
                utils.later(1200, self, lambda: self.statusBar().showMessage(
                    "The QC Panel is now a slim tab row: click a tab for its tools (the pin keeps them open) · "
                    "Ctrl+K runs any command · F11 focus mode", 12000))

    def _default_dock_sizes(self):
        """The mock-up's proportions: a 320 px side panel — Labels and Annotations as tall as a few rows, Files the
        rest — and a thumbnail strip under the image."""
        right = [d for d in (self.unique_labels_dock, self.annotations_dock, self.file_list_dock) if d.isVisible()]
        if right:
            self.resizeDocks(right, [320] * len(right), Qt.Orientation.Horizontal)
            h = max(400, self.height() - 200)
            sizes = {self.unique_labels_dock: int(h * 0.22), self.annotations_dock: int(h * 0.18),
                     self.file_list_dock: int(h * 0.60)}
            self.resizeDocks(right, [sizes[d] for d in right], Qt.Orientation.Vertical)

    def request_quit(self, reason: str):
        """Quit without prompts (Ctrl+C, SIGTERM): an open fixing session is kept for later, everything is saved."""
        self._quit_reason = reason
        dlg = self.editor_dialog
        if dlg is not None and dlg.es is not None and dlg.es.active:
            dlg.keep_for_later(f"kept for later ({reason})")
            dlg._shutdown("app quitting")
            dlg.hide()
        self.close()

    def set_dataset_actions_enabled(self, on: bool):
        if hasattr(self, "health_btn"):
            self.health_btn.setEnabled(on)
        cls = bool(on and self.session is not None and self.session.is_classification)
        if hasattr(self, "tabs") and hasattr(self.tabs, "set_tab_shown"):
            # classifiers: the Shapes tab becomes Images — its checks open their classifier versions (whole images
            # and their classes); its box Relationship Rules give way to the Labels tab's rules for image classes
            i = [g[0] for g in GROUPS].index("shapes")
            self.tabs.set_tab_shown(i, True)
            self.tabs.setTabText(i, "Images" if cls else "Shapes")
            self.tabs.setTabToolTip(i, "Are the images right for their class? (duplicates, outliers, sequences, "
                                       "sizes)" if cls else "Are the annotations themselves right?")
            rules = getattr(self, "qc_actions", {}).get("rules")
            page = getattr(self, "_ribbon_pages", {}).get("shapes")
            q = getattr(page, "question", None)
            if q is not None and hasattr(q, "setText"):
                q.setText("Are the images right for their class?" if cls else GROUPS[i][2])
            if rules is not None and page is not None and bool(rules.property("aio_type_hidden")) != cls:
                rules.setProperty("aio_type_hidden", cls)
                page.refilter()
        act = getattr(self, "rules_cls_act", None)
        if act is not None and bool(act.property("aio_type_hidden")) == cls:    # Rules on the Labels tab
            act.setProperty("aio_type_hidden", not cls)
            page = getattr(self, "_ribbon_pages", {}).get("labels")
            if page is not None:
                page.refilter()
        if act is not None:
            act.setEnabled(cls)
        for act in (self.a.reload, self.a.next, self.a.prev, self.a.rename, self.a.open_trash, self.a.label_map,
                    self.a.clear_cache, self.a.storage, self.a.ann_table, self.a.health_check,
                    self.a.mark_ok, self.a.mark_flag, self.a.flag_note, self.a.mark_clear, self.a.next_unreviewed,
                    self.a.export_review, self.a.save_flagged):
            act.setEnabled(on)
        for t in QC_TOOLS:
            act = self.qc_actions[t.key]
            if t.key in ("train", "augment", "data_prep"):   # a dataset file / a test picture / folders: no dataset
                act.setEnabled(True)
                continue
            if t.dialog is None:
                act.setEnabled(on)
                continue
            from AIO_QC.app_common import tool_for, tool_supported
            supported = tool_supported(t, self.session)          # classification: the classifier version
            act.setEnabled(on and bool(supported))
            key = t.shortcut or ("Ctrl+Shift+G" if t.key == "class_gallery" else "")
            alt = CLS_ALTERNATIVES.get(t.key) if self.session is not None and self.session.is_classification \
                else None
            tv = tool_for(t, self.session)
            tip = (f"Classifier version: {tv.dialog.HELP}" if tv is not t and getattr(tv.dialog, "HELP", "") else t.tip)
            act.setToolTip((f"{tip} ({key})" if key else tip) if supported or not on
                           else f"{t.tip} — not available for {self.session.type.title}"
                           + (f".\nFor a classifier: {alt}" if alt else ""))
