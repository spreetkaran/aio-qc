# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Created on Sun Sep 28 12:39:36 2025

@author: singh
"""

import os, functools, time
import numpy as np
from loguru import logger
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QMessageBox, QApplication, QFileDialog, QComboBox,
    QStatusBar, QToolButton, QMenu,
)
from PyQt6.QtCore import Qt, QSize, QTimer, QEvent, QThread, QPointF, QKeyCombination
from PyQt6.QtGui import QActionGroup, QColor, QKeySequence, QPixmap
from types import SimpleNamespace
from AIO_QC.tools.frame_extractor import __appname__, __version__
from AIO_QC.tools.frame_extractor import utils
from AIO_QC.tools.frame_extractor.core import session
from AIO_QC.tools.frame_extractor.core.metrics import sharpness
from AIO_QC.tools.frame_extractor.core.sources import format_time, HAVE_PYAV
from AIO_QC.tools.frame_extractor.widgets import (
    VideoLoadDialog,
    VideoEngine,
    VideoCanvas,
    SeekSlider,
    BookmarksDock,
    BatchQueueDialog,
)
from AIO_QC.tools.frame_extractor.core.common import VIDEO_EXTS, model_hex
from AIO_QC.tools.frame_extractor.widgets.timeline import TimelineWorker, TimelineStrip, ThumbPopup
from AIO_QC.tools.frame_extractor.widgets.detect_worker import DetectWorker
from AIO_QC.tools.frame_extractor.widgets.models_dialog import (
    ModelsDialog,
    load_model_cfg,
    save_model_cfg,
    shutdown_models,
)
from AIO_QC.tools.frame_extractor.widgets.markers_dock import MarkersDock, MarkerWorker
from AIO_QC.tools.frame_extractor.widgets.timecode_display import TimecodeDisplay
from AIO_QC.tools.frame_extractor.widgets.output_dialog import (
    OutputQuickBar,
    OutputSettingsDialog,
    load_output_cfg,
    save_output_cfg,
    summary as output_summary,
)
from AIO_QC.tools.frame_extractor.widgets.track_worker import TrackWorker
from AIO_QC.tools.frame_extractor.widgets.video_engine import Frame
from AIO_QC.tools.frame_extractor.widgets.command_palette import CommandPalette
from AIO_QC.tools.frame_extractor.widgets.shortcuts_dialog import ShortcutsDialog
from AIO_QC.tools.frame_extractor.widgets.gallery import ReviewGallery
from AIO_QC.utils.fx_settings import fx_settings
from AIO_QC.utils.qt_utils import later   # owner-bound timer: never fires after `self` is gone


MOUSE_HELP = [
    ("Video", [("Mouse wheel", "Previous / next frame"),
               ("Ctrl/Cmd + wheel, pinch", "Smooth zoom around the cursor"),
               ("Ctrl/Cmd + = / + / numpad +  ·  - / numpad -", "Zoom in / out (animated)"),
               ("Ctrl/Cmd + 0 · Ctrl/Cmd + 1", "Fit to window (button lights up while fitted) · actual pixels 100%"),
               ("Double-click", "Toggle fit ↔ 100% at the cursor"),
               ("Middle-drag, or drag when zoomed", "Pan"), ("Double-click", "Fit to window"),
               ("ROI mode: drag", "Draw a new ROI; drag inside to move, edges/corners to resize"),
               ("Right-click", "Context menu (save, ROI, marks, zoom)"), ("Drop a file", "Load it (several: batch queue)")]),
    ("Seekbar & timeline strip", [("Click / drag seekbar", "Jump / scrub (keyframe previews while dragging with PyAV)"),
               ("Hover", "Thumbnail preview"), ("Blue bars", "Sharpness (red = below the blur threshold)"),
               ("Orange line", "Motion (% change, or encoder kB/frame in keyframe mode)"),
               ("White ticks", "Scene changes"), ("Bottom row", "Model: purple = detections, amber = uncertain/disagree"),
               ("Right-click strip", "Blur threshold, model track, re-analyze")]),
    ("Extraction markers (top lane of the strip)", [("M", "Drop / remove a marker at the playhead (also while playing)"),
               ("Drag a marker", "Move it"), ("Double-click / right-click a marker", "Jump / snap, extract, remove"),
               ("Green / blue / hollow", "Pending / saved / suggested"),
               ("Ctrl+E · Ctrl+Shift+E", "Extract pending · auto-pick sharpest per window → snap → extract")]),
    ("Position display (under the seekbar)", [("Click", "Type a position: 1:23.5 · 00:01:23:12 · 1234f · 42% · "
                                                         "+5s · -30f · in · out · end (↑/↓ step while typing)"),
               ("Scroll on a unit", "Nudge hours / minutes / seconds / frames (Shift ×10)"),
               ("Drag on a unit", "Scrub by that unit"), ("Format chip / Ctrl+Shift+T", "Time · Timecode (SMPTE) · Frames · Seconds · Remaining"),
               ("Right-click", "Formats, copy time / timecode / frame, timecode overlay on the video")]),
    ("Command line", [("aio-qc extract -h", "Batch extraction without the GUI"),
                      ("… extract v.mp4 -o out --frames @marks.txt --model best.pt --env <conda env>",
                       "Save picked frames with pre-annotations from a .pt model"),
                      ("aio-qc probe VIDEO", "Decoder / VFR / rotation info"),
                      ("… testset OUT --add a.mp4 b.mp4 frames/ --every 1s", "Test set of unique frames from many sources"),
                      ("… eval DATA -m old.pt -m new.onnx --env yolo --report comparison",
                       "Score weights: mAP, P/R, confusion matrix, HTML report"),
                      ("… clip VIDEO 1:00 1:30 -o out.mp4 [--exact]", "Cut a clip (lossless, or frame-exact)")]),
    ("Search & clips", [("Ctrl+F", "Search frames by text (CLIP) or by example; click a result to jump"),
                          ("Ctrl+Shift+X", "Export the in → out range as a clip")]),
]


class MainWindow(QMainWindow):
    SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]
    CACHE_MB = 768        # RAM budget for decoded full-res frames

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{__appname__} — AIO QC")
        from AIO_QC.core.fx_storage import migrate_files
        migrate_files()                        # once: ~/.frame_extractor of the standalone app -> AIO QC's data
        self.setAcceptDrops(True)
        self.settings = fx_settings()
        screen_size = QApplication.primaryScreen().availableGeometry()
        height = int(screen_size.height() * 0.7)
        width = int(height * 16 / 9)
        self.imsize = SimpleNamespace(width=max(width, 640), height=max(height, 360))
        self.iconsize = SimpleNamespace(width=28, height=28)

        self.video_path = None
        self.output_dir = self.settings.value("output_dir", os.getcwd())
        if not os.path.isdir(self.output_dir):
            self.output_dir = os.getcwd()
        self.engine = None
        self.frame_rate = 30.0
        self.frame_count = 0
        self.current_frame = 0
        self.current = None
        self.requested_frame = 0
        self.time_step = 1 / self.frame_rate
        self.out_cfg = load_output_cfg()      # format, quality, naming template, collision policy
        self.backend = self.settings.value("decoder/backend", "auto")
        # In-process hardware decoding (OpenCV D3D11/VAAPI) is off by default: some driver / file combinations
        # crash inside the decoder and take the whole app down. The real-time playback pipeline still decodes
        # on the GPU - in a separate ffmpeg process, where a driver crash can't hurt the app.
        self.hw_decode = self.settings.value("decoder/hw_inproc", False, type=bool)
        self.video_paused = False
        self.seekbar_moving = False
        self.resume_after_seek = False
        self.status_priority = False
        self.bookmarks = {}
        self.in_frame = self.out_frame = None
        self.roi = None
        self.current_sharpness = None
        self.detections_cache = {}
        self._sharp_t = 0.0
        self.batch_dialog = None
        self.model_cfg = load_model_cfg()
        self.timeline = None
        self.detector = None
        self._hover = None       # (frame, QPoint) currently previewed
        self.timeline_models = self.settings.value("timeline/models", False, type=bool)
        self.tracker = None
        self._track_last = None
        self.markers = {}
        self.marker_worker = None
        self._open_token = 0
        self._ui_ms = 0.0
        self._ui_t = 0.0
        self._stats_hist = []
        self._prev_crash = self._read_previous_crash()        # before the canvas: may disable the GPU renderer
        self._sentinel_timer = QTimer(self)
        self._sentinel_timer.setSingleShot(True)
        self._sentinel_timer.timeout.connect(self._sentinel_clear)
        self.playback_mode = self.settings.value("playback/realtime", "auto")
        self.rt_info = None
        self._bg = []
        self.mpv = None               # core.mpv_lib.MpvPlayer (libmpv playback engine) or None
        self._mpv_playing = False
        self.review = None              # widgets.review_dock.ReviewDock once Testing sent a review playlist
        self._mpv_t_play = 0.0
        self._mpv_hw_known = True
        self._save_after_exact = None
        self._extract_after_snap = set()
        self.ended = False
        self.gallery = None
        self._goto_after_open = None    # (path, frame, seconds) to show once a requested video has opened
        # one theme for the whole app: AIO QC's (the palette follows the system); this window only repaints on changes

        self.setup_ui()
        utils.qt_utils.preferred_size(self, self.imsize.width + 300, self.imsize.height + 170, frac=0.95)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.on_tick)

        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.setInterval(120)
        self.resize_timer.timeout.connect(self.apply_display_size)
        self.canvas.resized.connect(self.resize_timer.start)
        self.update_renderer_status()
        if self._prev_crash:
            later(400, self, lambda: self._report_previous_crash(self._prev_crash))
        QTimer.singleShot(900, self._maybe_get_started)

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(500)
        self._stats_timer.timeout.connect(self._update_stats)

        self.default_status_timer = QTimer(self)
        self.default_status_timer.setSingleShot(True)
        self.default_status_timer.timeout.connect(self.reset_status_priority)

        self.set_video_actions_enabled(False)
        self.refresh_marks(save=False)
        self.actions_.tc_overlay.setChecked(self.tc.overlay)
        self._init_shortcuts()
        from AIO_QC.widgets.settings_dialog import install_settings
        install_settings(self)                              # gear in the corner + File ▸ Settings… (Ctrl+,)
        self._default_state = self.saveState()
        geom, state = self.settings.value("ui/geometry"), self.settings.value("ui/state")
        if geom is not None:
            self.restoreGeometry(geom)
        if state is not None:
            self.restoreState(state)
        self.activateWindow()
        self.raise_()
        self.setFocus()

    # ================================================================== settings
    @staticmethod
    def _menu_leaves(menu):
        out = []
        for act in menu.actions():
            if act.menu():
                out += MainWindow._menu_leaves(act.menu())
            elif not act.isSeparator():
                out.append(act)
        return out

    # ================================================================== video review (from Testing)
    def start_review(self, playlist):
        """Play a ReviewPlaylist (videos + models) with detections - Testing only hands it over."""
        from AIO_QC.tools.frame_extractor.widgets.review_dock import ReviewDock
        from AIO_QC.tools.frame_extractor.widgets.get_started import GetStarted
        for d in self.findChildren(GetStarted):
            d.close()                                       # the first-start tour must not cover the review
        if self.review is None:
            self.review = ReviewDock(self)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.review)
            self.tabifyDockWidget(self.mdock, self.review)  # a full-height tab next to Marks / Extract / Search
            self.review.toggleViewAction().setObjectName("toggle_review")
            self.menus.view.insertAction(self.actions_.zoom_in, self.review.toggleViewAction())
        self.review.start(playlist)
        self.review.raise_()
        return self.review

    def settings_sections(self):
        from AIO_QC.widgets.settings_pages import fx_sections
        return fx_sections(self)

    def open_settings(self, key=None):
        from AIO_QC.widgets.settings_dialog import open_settings
        return open_settings(self, self.settings_sections(), "Settings — Frame Extractor", key)

    # ================================================================== UI
    def setup_ui(self):
        self.toolbar = utils.ToolBar("Toolbar")
        self.toolbar.setIconSize(QSize(self.iconsize.width, self.iconsize.height))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.toolbar.setMovable(False)
        self.toolbar.installEventFilter(self)
        self.toolbar.visibilityChanged.connect(self.sync_toggle_toolbar)
        self.toolbar.setObjectName("MainToolbar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        self.menubar = self.menuBar()
        self.menubar.installEventFilter(self)
        self.menus = SimpleNamespace(
            file=self.menubar.addMenu("File"),
            edit=self.menubar.addMenu("Playback"),
            marks=self.menubar.addMenu("Marks"),
            view=self.menubar.addMenu("View"),
            settings=self.menubar.addMenu("Settings"),
            tools=self.menubar.addMenu("Tools"),
            help=self.menubar.addMenu("Help"),
        )

        A = functools.partial(utils.newAction, self)
        a = self.actions_ = SimpleNamespace()
        a.load = A("Load\nVideo", lambda: self.load_video(), "Ctrl+O", "open", "Load a video (or drop a file on the window)")
        a.outdir = A("Output\nDir", self.change_output_dir, None, "output_dir", "Change output directory")
        a.reopen_last = A("Reopen last video", self.reopen_last, "Ctrl+Alt+O", "open",
                          "Open the most recent video that still exists")
        self.recent_menu = QMenu("Open Recent", self)
        self.recent_menu.setIcon(utils.newIcon("open"))
        self.recent_menu.aboutToShow.connect(self._build_recent_menu)
        a.save = A("Save\nFrame", lambda: self.save_frame(), "Ctrl+S", "save",
                   "Save the current frame (ROI crop if an ROI is set)")
        a.save_full = A("Save\nFull", lambda: self.save_frame(full=True), "Ctrl+Shift+S", "save_full",
                        "Save the full current frame, ignoring the ROI")
        a.reset = A("Reset\nApp", self.reset_state, "Ctrl+R", "reset", "Reset application for new video")
        a.quit = A("Quit", self.close_window, "Ctrl+Q", "quit", "Quit application")
        a.rewind = A("Rewind", self.rewind_video, "Left", "prev", "Rewind by the selected step (←)")
        a.forward = A("Forward", self.forward_video, "Right", "next", "Forward by the selected step (→)")
        a.prev_frame = A("Previous frame", lambda: self.step_frames(-1), ",", "frame_prev", "Previous frame (,)")
        a.next_frame = A("Next frame", lambda: self.step_frames(1), ".", "frame_next", "Next frame (.)")
        a.go_start = A("Go to start", lambda: self.goto_frame(0), "Home", tip="First frame")
        a.go_end = A("Go to end", lambda: self.goto_frame(self.frame_count - 1), "End", tip="Last frame")
        a.slower = A("Slower", lambda: self.nudge_speed(-1), "[", tip="Decrease playback speed")
        a.faster = A("Faster", lambda: self.nudge_speed(1), "]", tip="Increase playback speed")
        a.mark_in = A("Mark\nIn", self.set_in, "I", "mark_in", "Set in point at current frame (I)")
        a.mark_out = A("Mark\nOut", self.set_out, "O", "mark_out", "Set out point at current frame (O)")
        a.go_in = A("Go to in point", lambda: self.in_frame is not None and self.goto_frame(self.in_frame), "Shift+I")
        a.go_out = A("Go to out point", lambda: self.out_frame is not None and self.goto_frame(self.out_frame), "Shift+O")
        a.clear_in_out = A("Clear in/out", self.clear_in_out, "X", tip="Clear in and out points")
        a.bookmark = A("Book-\nmark", self.toggle_bookmark, "B", "bookmark", "Toggle bookmark on current frame (B)")
        a.prev_bm = A("Previous bookmark", lambda: self.jump_bookmark(-1), "Ctrl+Left")
        a.next_bm = A("Next bookmark", lambda: self.jump_bookmark(1), "Ctrl+Right")
        a.roi = A("ROI", self.toggle_roi_mode, "R", "roi",
                  "ROI edit mode (R): drag to draw, drag inside to move, drag edges to resize", checkable=True)
        a.clear_roi = A("Clear\nROI", self.clear_roi, "Shift+R", "roi_clear", "Clear the ROI (Shift+R)")
        a.zoom_in = A("Zoom in", self.zoom_in, "Ctrl+=", "zoom_in", "Zoom in (Ctrl/Cmd + =, +, numpad +, Ctrl + wheel)")
        a.zoom_out = A("Zoom out", self.zoom_out, "Ctrl+-", "zoom_out", "Zoom out (Ctrl/Cmd + -, numpad -)")
        a.zoom_fit = A("Fit to window", self.zoom_fit, "Ctrl+0", "zoom_fit",
                       "Fit the whole frame in the window (Ctrl/Cmd + 0) - highlighted while fitted", checkable=True,
                       checked=True)
        a.zoom_actual = A("Actual pixels (100%)", self.zoom_actual, "Ctrl+1", "eye",
                          "1 video pixel = 1 screen pixel (Ctrl/Cmd + 1, or double-click) - highlighted while at 100%",
                          checkable=True)
        C, K = Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.KeypadModifier
        for act, extra in ((a.zoom_in, [QKeySequence("Ctrl++"), QKeySequence(QKeyCombination(C | K, Qt.Key.Key_Plus))]),
                           (a.zoom_out, [QKeySequence(QKeyCombination(C | K, Qt.Key.Key_Minus))]),
                           (a.zoom_fit, [QKeySequence(QKeyCombination(C | K, Qt.Key.Key_0))]),
                           (a.zoom_actual, [QKeySequence(QKeyCombination(C | K, Qt.Key.Key_1))])):
            act.setShortcuts([act.shortcut()] + extra)
        a.detect = A("Detect", self.toggle_detections, "D", "detect",
                     "Show model detections on the paused frame (D). Configure in Settings → Models", checkable=True)
        a.models = A("Models…", lambda: self.open_models(), None, "models", "Configure YOLO models (.onnx / .pt)")
        a.timeline_models = A("Run models in timeline analysis", self.set_timeline_models,
                              tip="Adds a model track (detections / uncertainty) to the timeline strip. Uses CPU.",
                              checkable=True, checked=self.timeline_models)
        a.marker = A("Mark\nFrame", self.toggle_marker, "M", "marker_add",
                     "Drop / remove an extraction marker at the playhead (M) - works while playing")
        a.next_marker = A("Next marker", lambda: self.jump_marker(1), "Alt+Right", "marker")
        a.prev_marker = A("Previous marker", lambda: self.jump_marker(-1), "Alt+Left")
        a.extract_markers = A("Extract pending markers", lambda: self.extract_markers(), "Ctrl+E", "extract",
                              "Save every pending marker (Ctrl+E)")
        a.auto_pick = A("Auto-pick frames (sharpest per window)", self.auto_pick, "Ctrl+Shift+E", "suggest",
                        "Suggest the sharpest frame in every window, snap to the exact sharpest, extract")
        a.next_flag = A("Next flagged sample", lambda: self.jump_flag(1), "F", "flag")
        a.prev_flag = A("Previous flagged sample", lambda: self.jump_flag(-1), "Shift+F")
        a.batch = A("Batch\nExtract", lambda: self.open_batch_queue(), "Ctrl+B", "batch",
                    "Batch extract queue: many videos, parallel workers, blur/duplicate/scene filters")
        a.guide = A("User's Guide", lambda: self.open_help(), "F1", "help",
                    "The complete guide: every feature, step by step, searchable")
        a.shortcut_ref = A("Keyboard shortcuts", lambda: self.open_help("shortcuts"), "Ctrl+/", "command",
                           "Every shortcut as it is set now, plus mouse and timeline gestures (printable)")
        a.help = A("Customize keyboard shortcuts…", self.show_shortcuts, None, "command",
                   "Change any shortcut; conflicts are highlighted")
        a.get_started = A("Get Started", lambda: self.show_get_started(), None, "icon",
                          "A one-minute tour of the main features")
        a.palette = A("Command palette…", self.open_palette, "Ctrl+K", "command", "Search and run any command")
        a.track = A("Track\nROI", self.toggle_tracking, "T", "track",
                    "ROI follows the object while playing/stepping (T). Draw an ROI first.", checkable=True)
        a.gallery = A("Review\nFrames", lambda: self.open_gallery(), "Ctrl+G", "gallery",
                      "Review extracted frames: preview with labels, reject/restore (Ctrl+G)")
        a.reset_layout = A("Reset window layout", self.reset_layout)
        a.stats = A("Playback statistics", self.toggle_stats, "Ctrl+Shift+I", "info",
                    "Overlay: shown fps, drops by cause, decode / UI / upload time per frame", checkable=True)
        a.output_settings = A("Output format && naming…", self.open_output_settings, "Ctrl+Shift+N", "save",
                              "Image format, quality and file-naming template")
        a.goto_time = A("Go to time / frame…", lambda: self.tc.start_edit(), "Ctrl+J", "timecode",
                        "Type a time, timecode, frame, percentage or +/- offset (Ctrl+J)")
        a.copy_time = A("Copy current time", self.copy_time, "Ctrl+Shift+C", "copy",
                        "Copy the position in the current display format")
        a.cycle_tc = A("Cycle time format", lambda: self.tc.cycle_format(), "Ctrl+Shift+T",
                       tip="Time · Timecode · Frames · Seconds · Remaining")
        a.tc_overlay = A("Timecode overlay on video", lambda on: self.tc.set_overlay(on), "Ctrl+Shift+O",
                         tip="Burn-in style time / timecode / frame number on the video", checkable=True)
        a.open_in_qc = A("Open output in Image QC", self.open_output_in_qc, None, "open",
                         "This video's extracted frames (+ any pre-labels) as the dataset in AIO QC's Image QC "
                         "workspace")
        a.export_clip = A("Export clip (in → out)…", self.export_clip, "Ctrl+Shift+X", "save",
                          "Cut the in → out range: lossless and instant, or frame-exact")
        a.export_annotated = A("Export annotated video…", self.export_annotated, None, "detect",
                               "The video with 1-4 models' detections drawn (overlay or side by side)")
        a.about = A("About", self.show_about)
        a.logs = A("Open log folder", self.open_log_folder, tip="Logs and crash reports")
        self.toggleToolbar = A(text="Toolbar", func=self.toggle_toolbar, tip="Show/hide the toolbar",
                               checkable=True, checked=True)

        self.play_pause = utils.newAction(self, "Play\nPause")
        self.play_icons = {"no_video": utils.newIcon("play_pause"),
                           "play": utils.newIcon("play"), "pause": utils.newIcon("pause")}
        self.set_play_pause_state("no_video")
        self.play_pause.setShortcut("Space")
        self.play_pause.triggered.connect(self.toggle_play_pause)

        # Decoder settings
        dec_menu = QMenu("Decoder", self)
        group = QActionGroup(self)
        self.decoder_actions = {}
        labels = {"auto": "Auto (PyAV for VFR/rotated video, else OpenCV)",
                  "opencv": "OpenCV (fastest, HW decode, assumes constant FPS)",
                  "pyav": "PyAV (exact timestamps)" + ("" if HAVE_PYAV else " – not installed")}
        for key, text in labels.items():
            act = utils.newAction(self, text, checkable=True, checked=(key == self.backend),
                                  enabled=(key != "pyav" or HAVE_PYAV))
            act.triggered.connect(functools.partial(self.set_backend, key))
            act.setObjectName(f"decoder_{key}")
            group.addAction(act)
            dec_menu.addAction(act)
            self.decoder_actions[key] = act
        dec_menu.addSeparator()
        hw = utils.newAction(self, "Hardware decoding in-app (faster exact frames; rare driver crashes)",
                             checkable=True, checked=self.hw_decode)
        hw.toggled.connect(self.set_hw_decode)
        hw.setObjectName("hw_decode")
        dec_menu.addAction(hw)
        self.hw_action = hw

        # Combos
        self.time_cb = QComboBox()
        self.time_cb.addItems(['1 frame', '0.1 second', '0.25 second', '0.5 second', '1 second',
                               '2 seconds', '5 seconds', '10 seconds'])
        self.time_cb.currentTextChanged.connect(self.on_combobox_changed)
        self.time_cb.setToolTip("Step used by Rewind/Forward (← / →)")
        self.speed_cb = QComboBox()
        self.speed_cb.addItems([f"{s:g}x" for s in self.SPEEDS])
        self.speed_cb.setCurrentIndex(self.SPEEDS.index(1.0))
        self.speed_cb.currentIndexChanged.connect(self.on_speed_changed)
        self.speed_cb.setToolTip("Playback speed ([ / ])")
        self.outbar = OutputQuickBar(self.out_cfg)
        self.outbar.changed.connect(self.set_output_cfg)
        self.outbar.open_dialog.connect(self.open_output_settings)
        for cb in (self.time_cb, self.speed_cb):
            cb.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Menus
        utils.addActions(self.menus.file, (a.load,))
        self.menus.file.addMenu(self.recent_menu)
        utils.addActions(self.menus.file, (a.reopen_last, None, a.outdir, None, a.save, a.save_full, None,
                                           a.batch, a.open_in_qc, None, a.export_clip, a.export_annotated, None, a.reset, a.quit))
        utils.addActions(self.menus.edit, (self.play_pause, None, a.rewind, a.forward, a.prev_frame,
                                           a.next_frame, a.go_start, a.go_end, None, a.goto_time, a.copy_time,
                                           a.cycle_tc, None, a.slower, a.faster))
        utils.addActions(self.menus.marks, (a.mark_in, a.mark_out, a.go_in, a.go_out, a.clear_in_out, None,
                                            a.bookmark, a.prev_bm, a.next_bm, None,
                                            a.marker, a.next_marker, a.prev_marker, a.extract_markers, a.auto_pick,
                                            None, a.next_flag, a.prev_flag,
                                            None, a.roi, a.clear_roi))
        self.menus.settings.addMenu(dec_menu)
        utils.addActions(self.menus.settings, (None, a.models, a.timeline_models))
        pb = QMenu("Playback", self)
        pb.setIcon(utils.newIcon("play"))
        pg = QActionGroup(self)
        self.playback_actions = {}
        for key, label in (("auto", "Real-time pipeline: auto (videos ≥ 2560×1440)"),
                           ("on", "Real-time pipeline: always"), ("off", "Real-time pipeline: off (exact decoder)")):
            act = utils.newAction(self, label, checkable=True, checked=(key == self.playback_mode))
            act.setObjectName(f"playback_{key}")
            act.triggered.connect(functools.partial(self.set_playback_mode, key))
            pg.addAction(act)
            pb.addAction(act)
            self.playback_actions[key] = act
        pb.addSeparator()
        eg = QActionGroup(self)
        self.engine_actions = {}
        cur_engine = self.settings.value("playback/engine", "auto")
        for key, label, tip in (
                ("auto", "Engine: libmpv when available (recommended)",
                 "libmpv (mpv's engine) when its library is found and the GPU renderer is active, else built-in"),
                ("mpv", "Engine: libmpv", "Always libmpv; a message tells you if it can't be used"),
                ("builtin", "Engine: built-in", "AIO QC's own decoder / ffmpeg pipeline")):
            act = utils.newAction(self, label, checkable=True, checked=(key == cur_engine), tip=tip)
            act.setObjectName(f"engine_{key}")
            act.triggered.connect(functools.partial(self.set_playback_engine, key))
            eg.addAction(act)
            pb.addAction(act)
            self.engine_actions[key] = act
        mpv_loc = pb.addAction("libmpv…")
        mpv_loc.setObjectName("playback_libmpv")
        mpv_loc.setToolTip("Is libmpv in use, why not, and how to get it on this computer (Windows, macOS, Linux)")
        mpv_loc.triggered.connect(self.open_libmpv_dialog)
        pb.addSeparator()
        bench = pb.addAction("Benchmark decoders for this video")
        bench.setObjectName("playback_benchmark")
        bench.triggered.connect(lambda: self.configure_realtime(force=True))
        ffa = pb.addAction("ffmpeg location…")
        ffa.setObjectName("playback_ffmpeg")
        ffa.triggered.connect(self.choose_ffmpeg)
        self.menus.settings.addMenu(pb)
        # Preferences live in File ▸ Settings… (widgets.settings_pages.fx_sections). This menu stays as the home of
        # their actions - keyboard shortcuts, the command palette and shortcut customisation find them here - but it
        # is no longer on the menu bar.
        utils.addActions(self.menus.settings, (None, a.output_settings, a.help))
        self.menubar.removeAction(self.menus.settings.menuAction())
        for act in self._menu_leaves(self.menus.settings):
            self.addAction(act)                     # shortcuts of actions in a detached menu still work
        utils.addActions(self.menus.edit, (None, bench))
        utils.addActions(self.menus.help, (a.get_started, a.guide, a.shortcut_ref, None, a.logs, a.about))
        utils.addActions(self.menus.tools, (a.palette, None, a.batch, a.gallery, None, a.detect, a.track,
                                            None, a.models))

        # Toolbar: file/annotation tools. Transport controls live under the video.
        utils.addActions(self.toolbar, (
            a.load, a.outdir, None, a.save, a.save_full, None,
            a.roi, a.clear_roi, a.track, None, a.mark_in, a.mark_out, a.bookmark, a.marker, None, a.detect, None,
            a.batch, a.gallery, None, a.reset, None,
            self.outbar,
        ))

        lb = self.toolbar.widgetForAction(a.load)
        if lb is not None:      # split button: click = open dialog, arrow = recent videos
            lb.setMenu(self.recent_menu)
            lb.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

        # Central: canvas + seekbar + transport row
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(2)
        self.canvas = self._make_canvas()
        self._connect_canvas(self.canvas)
        self.canvas_layout = layout
        layout.addWidget(self.canvas, 1)

        self.seekbar = SeekSlider(Qt.Orientation.Horizontal)
        self.seekbar.time_text = lambda f: format_time(self.engine.time_of(f)) if self.engine else ""
        self.seekbar.sliderPressed.connect(self.on_seek_pressed)
        self.seekbar.sliderReleased.connect(self.on_seek_released)
        self.seekbar.sliderMoved.connect(self.on_seek_moved)
        self.seekbar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.seekbar.hovered.connect(self.on_timeline_hover)
        self.seekbar.hover_left.connect(self.hide_preview)
        layout.addWidget(self.seekbar)
        self.strip = TimelineStrip(self.seekbar)
        self.strip.seek_requested.connect(lambda f: self.goto_frame(f))
        self.strip.hovered.connect(self.on_timeline_hover)
        self.strip.hover_left.connect(self.hide_preview)
        self.strip.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.strip.customContextMenuRequested.connect(self.strip_menu)
        self.strip.marker_moved.connect(self.move_marker)
        layout.addWidget(self.strip)
        self.preview = ThumbPopup()

        transport = QHBoxLayout()
        transport.setSpacing(2)

        def btn(action, size=26):
            b = QToolButton()
            b.setDefaultAction(action)
            b.setAutoRaise(True)
            b.setIconSize(QSize(size, size))
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            return b
        for act, size in ((a.rewind, 22), (a.prev_frame, 22), (self.play_pause, 30),
                          (a.next_frame, 22), (a.forward, 22)):
            transport.addWidget(btn(act, size))
        self.tc = TimecodeDisplay()
        self.tc.seek_requested.connect(lambda f, scrub: self.goto_frame(f, fast=scrub))
        self.tc.overlay_toggled.connect(self.on_tc_overlay)
        self.tc.format_changed.connect(lambda f: self.flash_status(f"Time format: {f}", 1500))
        transport.addSpacing(8)
        transport.addWidget(self.tc)
        transport.addStretch(1)
        transport.addWidget(QLabel("Step: "))
        transport.addWidget(self.time_cb)
        transport.addSpacing(10)
        transport.addWidget(QLabel("Speed: "))
        transport.addWidget(self.speed_cb)
        transport.addSpacing(6)
        for act in (a.zoom_out, a.zoom_fit, a.zoom_actual, a.zoom_in):
            b = btn(act, 20)
            if act in (a.zoom_fit, a.zoom_actual):
                b.setStyleSheet("QToolButton:checked { background: palette(highlight); border-radius: 4px; }")
            transport.addWidget(b)
        layout.addLayout(transport)
        self.setCentralWidget(central)

        # Marks dock
        self.dock = BookmarksDock(self, (a.mark_in, a.mark_out, a.clear_in_out, a.bookmark))
        self.dock.time_text = lambda f: format_time(self.engine.time_of(f)) if self.engine else str(f)
        self.dock.jump_requested.connect(self.goto_frame)
        self.dock.note_changed.connect(self.on_note_changed)
        self.dock.remove_requested.connect(self.remove_bookmarks)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.dock.setMinimumWidth(260)
        self.mdock = MarkersDock(self, a.marker)
        self.mdock.time_text = lambda f: format_time(self.engine.time_of(f)) if self.engine else str(f)
        self.mdock.jump_requested.connect(self.goto_frame)
        self.mdock.remove_requested.connect(self.remove_markers)
        self.mdock.snap_requested.connect(lambda frames, r: self.snap_markers(frames, r))
        self.mdock.suggest_requested.connect(self.suggest_markers)
        self.mdock.auto_pick_requested.connect(self.auto_pick)
        self.mdock.extract_requested.connect(lambda: self.extract_markers())
        self.mdock.batch_requested.connect(self.send_markers_to_batch)
        self.mdock.review_requested.connect(self.review_markers)
        self.mdock.options_changed.connect(self.on_marker_options)
        self.mdock.set_options({"crop": True})
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.mdock)
        self.tabifyDockWidget(self.dock, self.mdock)
        self.mdock.raise_()
        self.mdock.toggleViewAction().setObjectName("toggle_markers")
        from AIO_QC.tools.frame_extractor.widgets.search_dock import SearchDock
        self.sdock = SearchDock(self, self._search_context)
        self.sdock.seek_requested.connect(lambda v, t: self.open_at(v, seconds=t))
        self.sdock.marker_requested.connect(self._marker_at)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.sdock)
        self.tabifyDockWidget(self.mdock, self.sdock)
        self.mdock.raise_()
        sa = self.sdock.toggleViewAction()
        sa.setObjectName("toggle_search")
        sa.setText("Search frames")
        sa.setIcon(utils.newIcon("search"))
        sa.setShortcut("Ctrl+F")
        sa.setToolTip("Find moments by describing them (CLIP) or by example (Ctrl+F)")
        sa.triggered.connect(lambda on: on and (self.sdock.raise_(), self.sdock.query.setFocus()))
        self.actions_.search = sa
        utils.addActions(self.menus.view, (self.toggleToolbar, self.dock.toggleViewAction(),
                                           self.mdock.toggleViewAction(), sa, None,
                                           a.zoom_in, a.zoom_out, a.zoom_fit, a.zoom_actual, None, a.stats))
        am = QMenu("Animations", self)
        ag = QActionGroup(self)
        from AIO_QC.widgets import motion as _anim
        for key, label, tip in (("full", "Full", "Smooth motion at your display's refresh rate"),
                                ("reduced", "Reduced", "Shorter, gentler motion"),
                                ("off", "Off", "No animations - everything changes instantly")):
            act = utils.newAction(self, label, checkable=True, checked=(key == _anim.mode()), tip=tip)
            act.setObjectName(f"animations_{key}")
            act.triggered.connect(functools.partial(_anim.set_mode, key))
            ag.addAction(act)
            am.addAction(act)
        self.menus.settings.addMenu(am)
        rm = QMenu("Renderer", self)
        rg = QActionGroup(self)
        self.renderer_actions = {}
        cur = self.settings.value("ui/renderer", "auto")
        for key, label in (("auto", "Auto (GPU when available)"), ("gpu", "GPU (OpenGL)"), ("software", "Software")):
            act = utils.newAction(self, label, checkable=True, checked=(key == cur))
            act.setObjectName(f"renderer_{key}")
            act.triggered.connect(functools.partial(self.set_renderer, key))
            rg.addAction(act)
            rm.addAction(act)
            self.renderer_actions[key] = act
        self.menus.settings.addMenu(rm)
        for act in self._menu_leaves(am) + self._menu_leaves(rm):
            self.addAction(act)
        self.menus.view.addAction(a.reset_layout)

        # Status bar with live readouts
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.pixel_label = QLabel()
        self.roi_label = QLabel()
        self.sharp_label = QLabel()
        self.sharp_label.setToolTip("Sharpness of the current frame (inside the ROI if set).\n"
                                    "Use it to choose the batch blur threshold.")
        self.decoder_label = QLabel()
        self.play_label = QLabel()
        self.render_label = QLabel()
        for w in (self.pixel_label, self.roi_label, self.sharp_label, self.decoder_label, self.play_label,
                  self.render_label):
            w.setStyleSheet("padding: 0 6px;")
            self.status.addPermanentWidget(w)

        self.update_labels()
        self.update_log()
        self.update_status()

    def set_video_actions_enabled(self, on):
        a = self.actions_
        for act in (a.save, a.save_full, a.rewind, a.forward, a.prev_frame, a.next_frame, a.go_start,
                    a.go_end, a.mark_in, a.mark_out, a.go_in, a.go_out, a.clear_in_out, a.bookmark,
                    a.prev_bm, a.next_bm, a.roi, a.clear_roi, a.zoom_in, a.zoom_out, a.zoom_fit, a.zoom_actual, a.detect,
                    a.next_flag, a.prev_flag, a.track, a.marker, a.next_marker, a.prev_marker,
                    a.extract_markers, a.auto_pick, a.goto_time, a.copy_time, a.export_clip, a.export_annotated):
            act.setEnabled(on)

    # ============================================================== status
    def update_status(self):
        if getattr(self, 'status_priority', False):
            return
        if self.video_paused:
            self.status.showMessage("Paused")
        else:
            self.status.showMessage(f"Output Dir: {self.output_dir}")

    def flash_status(self, msg, ms=3000):
        self.status_priority = True
        self.status.showMessage(msg)
        self.default_status_timer.start(ms)

    def reset_status_priority(self):
        self.status_priority = False
        self.update_status()

    def update_log(self):
        logger.info(f"Output Dir: '{self.output_dir}'")

    def osd(self, text, ms=1200):
        self.canvas.show_osd(text, ms)

    # =============================================================== state
    @property
    def is_playing(self):
        return bool(self.engine and (self.engine.is_playing or self._mpv_playing))

    def stop_engine(self):
        self.timer.stop()
        if self._mpv_playing:                 # a new video replaces the one libmpv is playing
            self._mpv_playing = False
            if self.mpv is not None:
                self.mpv.pause()
            if getattr(self.canvas, "gpu", False):
                self.canvas.set_playing(False)
                self.canvas.set_mpv_active(False)
        self.stop_marker_worker()
        self.stop_tracking(save=False)
        self.actions_.track.setChecked(False)
        self.stop_timeline()
        if self.engine:
            self.engine.stop()
            self.engine = None

    def stop_timeline(self):
        if getattr(self, "_ttask", None) is not None:
            self._ttask.finish()
            self._ttask = None
        if self.timeline:
            self.timeline.stop()
            self.timeline = None
        self.strip.clear()
        self.hide_preview()

    def start_timeline(self):
        self.stop_timeline()
        if not self.engine:
            return
        e = self.engine
        cfg = dict(self.model_cfg) if (self.timeline_models and self.model_cfg["models"]) else None
        from AIO_QC.tools.frame_extractor.core.preview_pipe import find_ffmpeg
        self.timeline = TimelineWorker(self.video_path, e.backend, e.frame_count, e.fps, e.duration(), cfg,
                                       time_of=e.time_of, index_of=e.index_of, size=(e.width, e.height),
                                       ffmpeg=find_ffmpeg(self.settings.value("playback/ffmpeg", "")))
        self.strip.set_grid(self.timeline.grid)
        self.strip.set_blur_threshold(float(self.settings.value("batch/min_sharpness", 0) or 0))
        self.timeline.samples_ready.connect(self.on_samples)
        self.timeline.index_ready.connect(self.on_keyframe_index)
        self.timeline.grid_changed.connect(self.on_grid_changed)
        self.timeline.analysis_done.connect(self.on_analysis_done)
        from AIO_QC.widgets import tasks
        self._ttask = tasks.start(f"Timeline analysis · {os.path.basename(self.video_path or '')}",
                                  cancel=self.stop_timeline)
        self.timeline.model_error.connect(lambda m: self.flash_status(f"Timeline model error: {m}", 6000))
        self.timeline.set_throttle(self.is_playing)
        self.timeline.start(QThread.Priority.LowPriority)

    def reset_state(self):
        self.stop_engine()
        self.setWindowTitle(f"{__appname__} — AIO QC")
        self.video_path = None
        self.current = None
        self.current_frame = self.requested_frame = 0
        self.frame_count = 0
        self.frame_rate = 30.0
        self.bookmarks, self.in_frame, self.out_frame, self.roi = {}, None, None, None
        self.markers = {}
        self.refresh_markers(save=False)
        self.seekbar.setMaximum(0)
        self.seekbar.setValue(0)
        self.canvas.set_text("Load or drop a video")
        self.canvas.set_roi(None)
        self.actions_.roi.setChecked(False)
        self.canvas.set_roi_mode(False)
        self.set_play_pause_state("no_video")
        self.set_video_actions_enabled(False)
        self.refresh_marks()
        self.update_labels()
        for w in (self.pixel_label, self.roi_label, self.sharp_label, self.decoder_label):
            w.clear()
        self.update_status()

    def set_play_pause_state(self, state):
        if state == "no_video":
            icon, text, tip, self.video_paused = "no_video", "Play\nPause", "Play/Pause (Space)", False
        elif state == "play":
            icon, text, tip, self.video_paused = "play", "Play\nVideo", "Play (Space)", True
        else:
            icon, text, tip, self.video_paused = "pause", "Pause\nVideo", "Pause (Space)", False
        if hasattr(self, "tc"):
            self.tc.set_state(playing=(state == "pause"))
        self.play_pause.setIcon(self.play_icons[icon])
        self.play_pause.setText("Play / Pause" if state == "no_video" else utils.menu_text(text))
        self.play_pause.setIconText(text)
        self.play_pause.setToolTip(tip)
        self.play_pause.setStatusTip(tip)

    # ======================================================== load / files
    def load_video(self, file_path=None):
        if self.is_playing:
            self.pause()
        if not file_path:
            file_path, _ = VideoLoadDialog.getOpenFileName(self, directory=self.settings.value("last_dir", ""))
        if not file_path:
            return
        self.stop_engine()
        self._open_token += 1
        token = self._open_token
        name = os.path.basename(file_path)
        logger.info(f"Opening {file_path}")
        self._sentinel_set("load", file_path)
        self.canvas.set_text(f"Opening {name}…")
        self.flash_status(f"Opening {name}…", 60000)
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        backend, hw = self.backend, self.hw_decode
        from AIO_QC.tools.frame_extractor.core.sources import open_source

        def work():      # probing / indexing can take a while (VFR timestamp index, slow disks): not on the UI thread
            return open_source(file_path, backend=backend, hw_accel=hw)

        def opened(src):
            QApplication.restoreOverrideCursor()
            if token != self._open_token:
                src.release()
                return
            try:
                self._finish_open(file_path, src)
            except Exception:
                self._sentinel_clear()
                raise

        def failed(msg):
            QApplication.restoreOverrideCursor()
            self._sentinel_clear()
            if token != self._open_token:
                return
            self.canvas.set_text("Load or drop a video")
            logger.error(f"Failed to open video: {file_path} ({msg})")
            QMessageBox.warning(self, __appname__, f"Failed to open video:\n{file_path}\n\n{msg}")
        self._run_bg(work, opened, failed)

    def _run_bg(self, fn, done, failed=None, priority=None):
        """fn() off the UI thread; done(result) / failed(message) back on it."""
        from AIO_QC.widgets.testing.lab_common import Job
        w = Job(lambda job: fn(), self)
        w.done.connect(done)
        w.failed.connect(failed or (lambda m: logger.warning(f"background task: {m}")))
        w.finished.connect(lambda: self._bg.remove(w) if w in self._bg else None)
        self._bg.append(w)
        w.start(priority) if priority is not None else w.start()

    def _finish_open(self, file_path, src):
        engine = VideoEngine(file_path, self.canvas.physical_size(), hw_accel=self.hw_decode,
                             cache_mb=self.CACHE_MB, backend=self.backend, src=src)
        engine.display_images = not getattr(self.canvas, "gpu", False)
        self.settings.setValue("last_dir", os.path.dirname(file_path))
        self.add_recent(file_path)
        self._first_frame_pending = True

        self.engine = engine
        engine.still_ready.connect(self.on_still_ready)
        engine.length_changed.connect(self.on_length_changed)
        engine.set_speed(self.SPEEDS[self.speed_cb.currentIndex()])
        engine.start()

        self.video_path = file_path
        self.frame_rate = engine.fps
        self.frame_count = engine.frame_count
        self.current = None
        self.current_frame = self.requested_frame = 0
        self.canvas.set_video_size(engine.width, engine.height)
        self.seekbar.setMaximum(max(0, self.frame_count - 1))
        self.seekbar.setValue(0)
        self.on_combobox_changed(self.time_cb.currentText())
        self.set_play_pause_state("play")
        self.set_video_actions_enabled(True)

        sess = session.load(file_path)
        self.bookmarks, self.in_frame, self.out_frame = sess["bookmarks"], sess["in"], sess["out"]
        self.markers = sess["markers"]
        self.mdock.thumbs = {}
        if sess["marker_opts"]:
            self.mdock.set_options(sess["marker_opts"])
        self.refresh_markers(save=False)
        self.roi = sess["roi"]
        self.canvas.set_roi(self.roi)
        self.refresh_marks(save=False)
        self.update_roi_label()
        self.detections_cache = {}
        target = self._goto_after_open
        self._goto_after_open = None
        if target and target[0] == file_path:
            f = target[1] if target[1] is not None else engine.index_of(target[2] or 0.0)
            self.goto_frame(max(0, min(self.frame_count - 1, int(f))))
        else:
            self.goto_frame(0)
        if self.review is not None and self.review.active:
            self.review.on_video_loaded(file_path)
        # analysis starts once the first frame is up - never competes with opening the file
        later(600, self, lambda e=engine: e is self.engine and self.start_timeline())
        name = os.path.basename(file_path)
        self.setWindowTitle(f"{__appname__} — {name} — AIO QC")
        dec = f"{engine.backend}{' +GPU' if engine.hw_accel else ''}"
        extra = (" · VFR" if engine.is_vfr else "") + (f" · rotated {engine.rotation}°" if engine.rotation else "")
        self.decoder_label.setText(f"{engine.width}×{engine.height} · {self.frame_rate:.2f} fps · {dec}{extra}")
        self.tc.set_video(engine.fps, engine.frame_count, engine.time_of, engine.index_of, engine.duration(),
                          engine.is_vfr, name)
        self.tc.set_state(playing=False, speed=self.SPEEDS[self.speed_cb.currentIndex()], ended=False)
        self.update_labels()
        self.configure_realtime()
        info = f"Loaded {name}"
        if sess["bookmarks"] or sess["in"] is not None or sess["roi"]:
            info += " (restored bookmarks / in-out / ROI from last session)"
        logger.info(f"{info} | {self.decoder_label.text()}")
        self.flash_status(info, 5000)
        self.osd(name, 1500)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [u.toLocalFile() for u in ev.mimeData().urls()]
        videos = [p for p in paths if p.lower().endswith(VIDEO_EXTS)]
        if len(videos) == 1:
            self.load_video(videos[0])
        elif len(videos) > 1:   # several files: straight into the batch queue
            self.open_batch_queue(add_current=False)
            for p in videos:
                self.batch_dialog.add_job(p)

    def change_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir)
        if directory:
            self.set_output_dir(directory)

    def set_output_dir(self, directory):
        if directory and directory != self.output_dir:
            os.makedirs(directory, exist_ok=True)
            self.output_dir = directory
            self.settings.setValue("output_dir", directory)
            logger.info(f"Output Dir changed to: '{self.output_dir}'")
            if self.batch_dialog:
                self.batch_dialog.refresh_output_label()
            self.update_status()

    def set_output_cfg(self, cfg, from_dialog=False):
        self.out_cfg = dict(cfg)
        save_output_cfg(self.out_cfg)
        if from_dialog:
            self.outbar.set_cfg(self.out_cfg)
        self.stop_marker_worker()                  # restarted with the new output settings on next use
        if self.batch_dialog:
            self.batch_dialog.refresh_output_summary()
        self.flash_status("Output: " + output_summary(self.out_cfg), 2500)

    def open_output_settings(self):
        fr = self.current if self.current is not None and not self.current.preview else None
        ctx = dict(video_path=self.video_path, frame=fr.idx if fr else 2503,
                   sec=self.engine.time_of(fr.idx) if (fr and self.engine) else None,
                   fps=self.frame_rate, image=fr.bgr if fr else None, roi=self.roi, output_dir=self.output_dir)
        if ctx["sec"] is None:
            ctx.pop("sec")
        d = OutputSettingsDialog(self, self.out_cfg, ctx)
        if d.exec():
            self.set_output_cfg(d.result_cfg(), from_dialog=True)

    def set_backend(self, key):
        self.backend = key
        self.settings.setValue("decoder/backend", key)
        self.flash_status(f"Decoder: {key} (applies to the next loaded video)")

    def set_hw_decode(self, on):
        self.hw_decode = on
        self.settings.setValue("decoder/hw_inproc", on)

    def on_combobox_changed(self, val):
        self.time_step = 1 / self.frame_rate if val == "1 frame" else float(val.split(' ')[0])

    # ============================================================ playback
    def refresh_interval(self):
        scr = self.canvas.screen() if self.canvas is not None else None
        hz = scr.refreshRate() if scr is not None else 60.0
        return 1.0 / (hz if hz and hz > 20 else 60.0)

    def timer_interval(self):
        # software canvas: tick at least as fast as the display (16 ms for 60 Hz, not 17 - a slower tick puts
        # two 60 fps frames into one tick and skips one); frames are matched to the nearest tick
        return max(4, int(self.refresh_interval() * 1000))

    def play(self, start=None):
        if not self.engine:
            return
        if start is None:
            start = self.current_frame + 1 if self.current is not None else 0
            if self.ended:
                start = 0          # after the video ended, Play restarts it
        if self.frame_count and start >= self.frame_count:
            start = 0
        self.ended = False
        self.tc.set_state(ended=False)
        if self._mpv_ready():
            self._play_mpv(start)
        else:
            m = self.mpv
            if m is not None and m.failed and getattr(self, "_mpv_failed_shown", None) != m.path:
                self._mpv_failed_shown = m.path      # say it once per video, not on every Play
                self.flash_status(f"libmpv: {m.failed} - built-in engine plays this video", 9000)
            self.engine.set_display_rate(1.0 / self.refresh_interval())
            self.engine.play(start)
        if self.timeline:
            self.timeline.set_throttle(True)
        self.canvas.set_detections([])
        self.set_play_pause_state("pause")
        if not self.settings.value("safe/played_ok", False, type=bool) or self._sentinel_timer.isActive():
            self._sentinel_set("play", secs=6)
        if getattr(self.canvas, "gpu", False):
            self.canvas.set_playing(True)          # vsync-paced: every swap presents the next due frame
        elif not self._mpv_playing:
            self.timer.start(self.timer_interval())
        self.update_status()

    def pause(self):
        if not self.engine:
            return
        if self._mpv_playing:
            # libmpv: stop on the frame it shows (its picture stays up until the exact frame replaces it)
            self._mpv_playing = False
            m = self.mpv
            if m is not None:
                m.pause()
                t = m.time_pos
                if t is not None:
                    idx = self._mpv_idx(t)
                    self.current, self.current_frame = Frame(idx, None, None), idx
                    self.strip.set_current(idx)
        self.engine.pause()
        self.timer.stop()
        if getattr(self.canvas, "gpu", False):
            self.canvas.set_playing(False)
        if self.timeline:
            self.timeline.set_throttle(False)
        self.requested_frame = self.current_frame
        self.set_play_pause_state("play")
        self.update_labels()
        if self.current is not None and self.current.bgr is None and not self.current.preview:
            # the paused pipeline stays alive (instant resume); get the exact full-res frame without a seek:
            # NV12 frames convert instantly, display-size ones are decoded in the background
            exact = self.engine.exact_from_nv12(self.current) if self.current.nv12 is not None else None
            if exact is not None:
                self.display(exact)
            else:
                f = self.engine.fetch_exact(self.current_frame)
                if f is not None:
                    self.display(f)
        elif self.current is not None:
            self.update_sharpness(force=True)
            if self.actions_.detect.isChecked():
                self.request_detections(self.current)
        self.update_status()

    def toggle_play_pause(self):
        if not self.engine:
            return
        self.pause() if self.is_playing else self.play()

    def on_tick(self):
        """Software canvas timer (display refresh rate)."""
        self._present(0.0, self.refresh_interval())

    def on_present(self, present_in):
        """GPU canvas: called after each buffer swap with the time until the next vsync."""
        if self.sender() is not None and self.sender() is not self.canvas:
            return
        self._present(present_in, present_in)

    def _present(self, present_in, period):
        if self._mpv_playing:
            self._mpv_tick()
            return
        if not self.engine or not self.engine.is_playing:
            return
        frame = self.engine.pop_due_frame(present_in, period)
        if frame is not None:
            self.display(frame)
        elif self.engine.at_end():
            QTimer.singleShot(0, self.on_video_end)

    def on_video_end(self):
        """Playback reached the real end (or the decoder stopped): back to 'Play'."""
        self.pause()
        self.ended = True
        self.tc.set_state(ended=True)
        if self.review is not None and self.review.on_video_end():
            return                                          # video review: the next video plays
        self.osd("End of video · Space replays")

    def on_length_changed(self, n):
        if self.sender() is not self.engine:
            return
        self.frame_count = n
        self.seekbar.setMaximum(max(0, n - 1))
        self.tc.set_length(n, self.engine.duration())
        self.update_labels()
        logger.info(f"Video is {n} frames long (container metadata was wrong)")

    def on_speed_changed(self, index):
        self.tc.set_state(speed=self.SPEEDS[index])
        if self.mpv is not None:
            self.mpv.set_double("speed", self.SPEEDS[index])
        if self.engine:
            self.engine.set_speed(self.SPEEDS[index])
            if self.timer.isActive():
                self.timer.setInterval(self.timer_interval())

    def nudge_speed(self, delta):
        i = max(0, min(len(self.SPEEDS) - 1, self.speed_cb.currentIndex() + delta))
        self.speed_cb.setCurrentIndex(i)
        self.osd(f"Speed {self.SPEEDS[i]:g}x")

    def display(self, frame):
        t_ui = time.perf_counter()
        try:
            self._display(frame)
        finally:
            self._ui_ms = 0.9 * self._ui_ms + 0.1 * (time.perf_counter() - t_ui) * 1000

    def _display(self, frame):
        if getattr(self, "_first_frame_pending", False):
            self._first_frame_pending = False
            self._sentinel_set("gpu" if getattr(self.canvas, "gpu", False) else "load", secs=8)
        self.current = frame
        self.current_frame = frame.idx
        self.canvas.show_frame(frame)
        if self.review is not None and self.review.active:
            self.review.on_frame(frame.idx, self.is_playing)     # video review: this frame's detections
        if self.mpv is not None and not getattr(self, "_mpv_verified", False) and not self.is_playing:
            later(0, self, self._mpv_verify)
        self.strip.set_current(frame.idx)
        if not self.seekbar_moving:
            self.seekbar.set_playhead(frame.idx)       # every frame: cheap partial repaint, smooth glide
        # while playing, text widgets (timecode, seekbar, labels) refresh at most 20x/s - smooth to the eye and
        # it keeps the UI thread free to hit every display refresh; paused: always immediately
        now = time.monotonic()
        if not self.is_playing or now - self._ui_t >= 0.05 or frame.preview:
            self._ui_t = now
            self.update_labels()
        if frame.preview:          # keyframe preview while scrubbing: skip the heavy stuff
            return
        if self._save_after_exact is not None and frame.bgr is not None and frame.idx == self.requested_frame:
            full, self._save_after_exact = self._save_after_exact, None
            later(0, self, lambda: self.save_frame(full))
        self.update_sharpness()
        if self.actions_.detect.isChecked() and not self.is_playing:
            self.request_detections(frame)
        if self.tracker is not None:
            self.track_frame(frame)

    def on_still_ready(self, frame):
        if self.sender() is not self.engine or self.is_playing:
            return
        if (frame.bgr is not None and not frame.preview and self.current is not None and self.current.bgr is None
                and frame.idx != self.current_frame and frame.idx != self.requested_frame):
            return          # a late exact frame for a position the user already left
        self.display(frame)

    def goto_frame(self, idx, pause=True, fast=False):
        if not self.engine:
            return
        if pause and self.is_playing:
            self.pause()
        if self.ended:
            self.ended = False
            self.tc.set_state(ended=False)
        idx = self.engine.clamp(idx)
        self.requested_frame = idx
        frame = self.engine.request_frame(idx, fast=fast)
        if frame is not None:
            self.display(frame)

    def step_frames(self, n):
        if not self.engine:
            return
        base = self.current_frame if self.is_playing else self.requested_frame
        self.goto_frame(base + n)
        self.update_status()

    def rewind_video(self):
        self.step_frames(-max(1, round(self.time_step * self.frame_rate)))

    def forward_video(self):
        self.step_frames(max(1, round(self.time_step * self.frame_rate)))

    def apply_display_size(self):
        self._sync_zoom_buttons()
        if not self.engine:
            return
        self.engine.set_display_size(*self.canvas.physical_size())
        if self.current is not None and not self.is_playing:
            self.canvas.show_frame(self.engine.make_display(self.current))

    # ============================================================= seekbar
    def on_seek_pressed(self):
        if not self.engine:
            return
        self.seekbar_moving = True
        self.resume_after_seek = self.is_playing
        if self.is_playing:
            self.pause()
        self.goto_frame(self.seekbar.value(), pause=False, fast=True)

    def on_seek_moved(self, value):
        if self.seekbar_moving:
            self.goto_frame(value, pause=False, fast=True)   # keyframe preview; exact frame on release

    def on_seek_released(self):
        self.seekbar_moving = False
        if not self.engine:
            return
        value = self.seekbar.value()
        if self.resume_after_seek:
            self.requested_frame = value
            self.play(start=value)
        else:
            self.goto_frame(value, pause=False)
        self.update_status()

    def update_labels(self):
        if self.engine:
            pre = self.current is not None and self.current.preview
            self.tc.set_position(self.current_frame, pre)
            self.update_tc_context()
            if self.tc.overlay:
                self.canvas.set_hud(self.tc.overlay_lines())
        elif hasattr(self, "tc"):
            self.tc.clear()
            self.canvas.set_hud(None)
        if not self.seekbar_moving:
            if self.is_playing:
                self.seekbar.sync_value(self.current_frame)     # playhead itself moves every frame
            else:
                self.seekbar.setValue(self.current_frame)

    # ======================================================== live readouts
    def update_sharpness(self, force=False):
        if self.current is None:
            return
        now = time.monotonic()
        if not force and self.is_playing and now - self._sharp_t < 0.3:
            return   # throttle during playback
        self._sharp_t = now
        img, sc = self.frame_image()
        if img is None:
            return
        if self.roi:
            x1, y1, x2, y2 = (int(v * sc) for v in self.roi)
            img = img[y1:y2, x1:x2]
        if img.size:
            self.current_sharpness = sharpness(img)
            self.sharp_label.setText(f"Sharpness{' (ROI)' if self.roi else ''}: {self.current_sharpness:.0f}")

    def on_cursor_moved(self, pt):
        if pt is None or self.current is None:
            self.pixel_label.clear()
            return
        x, y = pt
        img, sc = self.frame_image()
        if img is None:
            return
        b, g, r = img[min(img.shape[0] - 1, int(y * sc)), min(img.shape[1] - 1, int(x * sc))][:3]
        self.pixel_label.setText(f"x {x}, y {y}  RGB({r},{g},{b})")

    # ================================================================= ROI
    def toggle_roi_mode(self, checked):
        self.canvas.set_roi_mode(checked)
        self.osd("ROI edit: drag to draw / move / resize" if checked else "ROI edit off")

    def on_roi_changed(self, roi):
        self.roi = roi
        if self.tracker is not None and self.current is not None:
            img, sc = self.frame_image()
            if img is not None:
                self.tracker.init(self.current.idx, img, roi, sc)
                self._track_last = self.current.idx
        self.update_roi_label()
        self.update_sharpness(force=True)
        self.save_session()

    def clear_roi(self):
        if self.roi is None:
            return
        if self.tracker is not None:
            self.actions_.track.setChecked(False)
            self.stop_tracking()
        self.roi = None
        self.canvas.set_roi(None)
        self.update_roi_label()
        self.update_sharpness(force=True)
        self.save_session()
        self.osd("ROI cleared")

    def update_roi_label(self):
        if self.roi:
            x1, y1, x2, y2 = self.roi
            self.roi_label.setText(f"ROI {x1},{y1}–{x2},{y2} ({x2 - x1}×{y2 - y1})")
        else:
            self.roi_label.setText("ROI: none")

    def canvas_menu(self, pos):
        if not self.engine:
            return
        a = self.actions_
        m = QMenu(self)
        m.addAction(a.save)
        m.addAction(a.save_full)
        m.addSeparator()
        m.addAction(a.roi)
        if self.roi:
            m.addAction(a.clear_roi)
            copy = m.addAction("Copy ROI as x1,y1,x2,y2")
            copy.triggered.connect(lambda: QApplication.clipboard().setText(",".join(map(str, self.roi))))
        m.addSeparator()
        m.addAction(a.bookmark)
        m.addAction(a.mark_in)
        m.addAction(a.mark_out)
        m.addSeparator()
        m.addAction(a.zoom_fit)
        m.exec(self.canvas.mapToGlobal(pos))

    # =============================================================== marks
    def set_in(self):
        if not self.engine:
            return
        self.in_frame = self.current_frame
        if self.out_frame is not None and self.out_frame < self.in_frame:
            self.out_frame = None
        self.refresh_marks()
        self.osd(f"In: {format_time(self.engine.time_of(self.in_frame))}")

    def set_out(self):
        if not self.engine:
            return
        self.out_frame = self.current_frame
        if self.in_frame is not None and self.in_frame > self.out_frame:
            self.in_frame = None
        self.refresh_marks()
        self.osd(f"Out: {format_time(self.engine.time_of(self.out_frame))}")

    def clear_in_out(self):
        self.in_frame = self.out_frame = None
        self.refresh_marks()
        self.osd("In/out cleared")

    def toggle_bookmark(self):
        if not self.engine:
            return
        f = self.current_frame
        if f in self.bookmarks:
            del self.bookmarks[f]
            self.osd("Bookmark removed")
        else:
            self.bookmarks[f] = ""
            self.osd(f"Bookmark #{f}")
        self.refresh_marks()

    def jump_bookmark(self, direction):
        if not self.bookmarks:
            return
        marks = sorted(self.bookmarks)
        base = self.requested_frame if not self.is_playing else self.current_frame
        cands = [m for m in marks if (m > base if direction > 0 else m < base)]
        if cands:
            self.goto_frame(cands[0] if direction > 0 else cands[-1])

    def on_note_changed(self, frame, note):
        if frame in self.bookmarks:
            self.bookmarks[frame] = note
            self.save_session()

    def remove_bookmarks(self, frames):
        for f in frames:
            self.bookmarks.pop(f, None)
        self.refresh_marks()

    def update_tc_context(self):
        m = self.markers.get(self.current_frame)
        state = None if m is None else ("saved" if m.get("saved") else m.get("src", "manual"))
        self.tc.set_context(self.in_frame, self.out_frame, state, self.current_frame in self.bookmarks)

    def on_tc_overlay(self, on):
        self.actions_.tc_overlay.setChecked(on)
        self.canvas.set_hud(self.tc.overlay_lines() if on and self.engine else None)

    def copy_time(self):
        if not self.engine:
            return
        text = self.tc.text_for(self.tc.fmt)[0]
        QApplication.clipboard().setText(text)
        self.osd(f"Copied {text}")

    def refresh_marks(self, save=True):
        self.seekbar.set_in_out(self.in_frame, self.out_frame)
        if hasattr(self, "tc"):
            self.update_tc_context()
        self.seekbar.set_bookmarks(list(self.bookmarks))
        self.dock.set_data(self.bookmarks, self.in_frame, self.out_frame)
        if save:
            self.save_session()

    def save_session(self):
        if not self.video_path:
            return
        try:
            session.save(self.video_path, self.bookmarks, self.in_frame, self.out_frame, self.roi,
                         self.markers, self.mdock.options())
        except OSError as e:
            logger.warning(f"Could not save session: {e}")

    # ======================================================= libmpv playback engine
    def set_playback_engine(self, key):
        self.settings.setValue("playback/engine", key)
        self.engine_actions[key].setChecked(True)
        if self.is_playing:
            self.pause()
        self._configure_mpv()

    def mpv_state(self):
        """How playback uses libmpv right now (for the libmpv dialog)."""
        m = self.mpv
        st = {"gpu": bool(getattr(self.canvas, "gpu", False) and not getattr(self.canvas, "_dead", False)),
              "engine": self.settings.value("playback/engine", "auto")}
        if m is None:
            st["state"] = "" if not self.engine else "Not in use for this video."
            return st
        bits = ["In use for this video" if not m.failed else f"Could not play this video: {m.failed}"]
        if getattr(self, "_mpv_verified", False):
            bits.append("picture verified")
        if getattr(self, "_mpv_basic", False):
            bits.append("basic renderer (the full one drew a wrong picture on this GPU)")
        if getattr(self, "_mpv_clock_sync", False):
            bits.append("clock-synced (the display's refresh could not be followed)")
        st["state"] = " · ".join(bits) + "."
        return st

    def open_libmpv_dialog(self):
        from AIO_QC.tools.frame_extractor.widgets.libmpv_dialog import LibmpvDialog

        def changed():
            self.stop_mpv()
            self._mpv_verified = self._mpv_basic = False
            self._configure_mpv()
        LibmpvDialog(self, self.settings, self.mpv_state, changed, self.libmpv_job).exec()

    # ---- getting libmpv automatically (core.mpv_fetch; Windows / macOS download, Linux package manager)
    def libmpv_job(self, start=True):
        """The running libmpv download / install, started now when start=True (None: nothing to do here)."""
        job = getattr(self, "_mpv_job", None)
        if job is not None and job.running:
            return job
        if not start:
            return None
        from AIO_QC.tools.frame_extractor.core import mpv_fetch, mpv_lib
        from AIO_QC.tools.frame_extractor.widgets.libmpv_dialog import MpvFetchJob
        from AIO_QC.widgets import tasks
        plan = mpv_fetch.plan()
        if not plan["how"] or not mpv_lib.user_lib_dir():
            return None
        job = MpvFetchJob(mpv_lib.user_lib_dir(), self)
        task = tasks.start("Getting libmpv · smoother playback", cancel=job.cancel)
        job.progress.connect(lambda f, t: task.update(f, t))
        job.finished.connect(task.finish)
        job.succeeded.connect(self._on_libmpv_fetched)
        job.failed.connect(self._on_libmpv_fetch_failed)
        self._mpv_job = job
        logger.info(f"libmpv: getting {plan['what']} from {plan['source']}")
        job.start()
        return job

    def _maybe_auto_fetch(self):
        """libmpv is missing: get it in the background, once per session (Settings ▸ Playback ▸ libmpv… turns
        this off). Windows / macOS download silently; Linux asks once (the package manager needs a password).
        A failure waits 3 days before the next automatic attempt. True when something was started / asked."""
        if getattr(self, "_mpv_auto_tried", False) or os.environ.get("AIOQC_NO_LIBMPV_DOWNLOAD"):
            return False
        self._mpv_auto_tried = True
        if not self.settings.value("playback/libmpv_auto", True, type=bool):
            return False
        try:
            failed_at = float(self.settings.value("playback/libmpv_fetch_failed", 0) or 0)
        except (TypeError, ValueError):
            failed_at = 0.0
        if time.time() - failed_at < 3 * 86400:
            return False
        from AIO_QC.tools.frame_extractor.core import mpv_fetch
        plan = mpv_fetch.plan()
        if plan["how"] == "download":
            self.flash_status("Getting libmpv in the background for smoother playback (Settings ▸ Playback ▸ "
                              "libmpv…) - the built-in engine plays meanwhile", 8000)
            later(1500, self, lambda: self.libmpv_job(start=True))
            return True
        if plan["how"] == "package" and not self.settings.value("playback/libmpv_asked", False, type=bool):
            self.settings.setValue("playback/libmpv_asked", True)
            later(1200, self, self._ask_libmpv_package)
            return True
        return False

    def _ask_libmpv_package(self):
        from AIO_QC.tools.frame_extractor.core import mpv_fetch
        plan = mpv_fetch.plan()
        r = QMessageBox.question(
            self, "libmpv - smoother playback",
            "libmpv makes playback smoother (GPU decoding, display-synced timing). Install it now with your "
            f"system's package manager?\n\n{plan.get('command', '')}\n\nYour system asks for your password. "
            "The built-in engine keeps working either way; Settings ▸ Playback ▸ libmpv… can do this later.")
        if r == QMessageBox.StandardButton.Yes:
            self.libmpv_job(start=True)

    def _on_libmpv_fetched(self, path):
        from AIO_QC.tools.frame_extractor.core import mpv_lib
        self.settings.remove("playback/libmpv_fetch_failed")
        logger.info(f"libmpv installed{': ' + path if path else ' (system package)'}")
        mpv_lib.reset()
        self._mpv_verified = self._mpv_basic = False
        if self.engine:
            self._configure_mpv()
        self.flash_status("libmpv is ready - smoother playback is on" if mpv_lib.load_library(
            self.settings.value("playback/libmpv", "")) is not None else
            f"libmpv was installed but can't be used: {mpv_lib.last_error()}", 8000)

    def _on_libmpv_fetch_failed(self, why):
        if why == "cancelled":
            self.flash_status("libmpv download cancelled", 4000)
            return
        self.settings.setValue("playback/libmpv_fetch_failed", time.time())
        logger.warning(f"libmpv could not be installed automatically: {why}")
        self.flash_status(f"Could not get libmpv automatically ({why[:120]}) - Settings ▸ Playback ▸ libmpv…", 10000)

    def choose_libmpv(self):
        filt = "libmpv (libmpv-2.dll mpv-2.dll mpv-1.dll);;All files (*)" if os.name == "nt" else "libmpv (libmpv*);;All files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "libmpv library", "", filt)
        if not path:
            return
        from AIO_QC.tools.frame_extractor.core import mpv_lib
        self.stop_mpv()
        mpv_lib.reset()
        if mpv_lib.load_library(path) is None:
            QMessageBox.warning(self, "libmpv", f"Can't use this file:\n{mpv_lib.last_error()}")
            return
        self.settings.setValue("playback/libmpv", path)
        self._configure_mpv()

    def _configure_mpv(self):
        """libmpv plays when the GPU canvas is active and the library is found (Settings > Playback > Engine).
        The built-in engine stays configured underneath as the fallback."""
        key = self.settings.value("playback/engine", "auto")
        gpu = getattr(self.canvas, "gpu", False) and not getattr(self.canvas, "_dead", False)
        if key == "builtin" or not gpu or not self.engine:
            if key != "builtin" and self.engine and not gpu and key == "mpv":
                self.flash_status("libmpv needs the GPU renderer (Settings ▸ Display ▸ Video renderer)", 6000)
            self.stop_mpv()
            return
        from AIO_QC.tools.frame_extractor.core import mpv_lib
        lib = mpv_lib.load_library(self.settings.value("playback/libmpv", ""))
        if lib is None:
            if not self._maybe_auto_fetch() and key == "mpv":
                self.flash_status(f"libmpv unavailable ({mpv_lib.last_error()}) - Settings ▸ Playback ▸ "
                                  "libmpv…", 9000)
            self.stop_mpv()
            return
        try:
            if self.mpv is not None and getattr(self.canvas, "_mpv", None) is not self.mpv:
                self.stop_mpv()                           # the canvas (or its GL context) changed: start afresh
            if self.mpv is None:
                self.mpv = mpv_lib.MpvPlayer(lib, hwdec=self.settings.value("playback/hwdec", "auto-safe"),
                                             basic=getattr(self, "_mpv_basic", False))
                if getattr(self, "_mpv_clock_sync", False):
                    self.mpv.set("video-sync", "audio")
                self.canvas.attach_mpv(self.mpv)
            if not self.canvas.prepare_mpv():             # OpenGL not initialised yet: try again shortly
                self._mpv_tries = getattr(self, "_mpv_tries", 0) + 1
                if self._mpv_tries <= 50:
                    later(100, self, self._configure_mpv)
                else:
                    raise RuntimeError("the GPU canvas never initialised OpenGL")
                return
            self._mpv_tries = 0
            e = self.engine
            self.mpv.load(e.path, aspect=f"{e.width}:{e.height}")
            if not getattr(self, "_mpv_verified", False):
                self._mpv_verify_polls = 0
                later(200, self, self._mpv_verify)
            self.mpv_label = "libmpv"
            self.rt_status("libmpv · hardware decoding on the display GPU · display-synced")
        except Exception as ex:
            logger.warning(f"libmpv disabled: {ex}")
            self.flash_status(f"libmpv disabled: {ex}", 8000)
            self.stop_mpv()

    def stop_mpv(self):
        m = getattr(self, "mpv", None)
        self._mpv_playing = False
        if m is None:
            return
        self.mpv = None
        try:
            if hasattr(self.canvas, "release_mpv"):
                self.canvas.release_mpv()
        finally:
            m.terminate()

    def on_mpv_failed(self, why):
        logger.warning(f"libmpv rendering failed, using the built-in engine: {why}")
        self.flash_status("libmpv rendering failed - built-in playback engine in use", 8000)
        was = self._mpv_playing
        start = self.current_frame
        self._mpv_playing = False
        self.stop_mpv()
        self.settings.setValue("playback/engine", "builtin")
        if hasattr(self, "engine_actions"):
            self.engine_actions["builtin"].setChecked(True)
        if was:
            self.play(start=start)

    def _mpv_ready(self):
        m = self.mpv
        return bool(m is not None and m.loaded and not m.failed and getattr(self.canvas, "gpu", False)
                    and not getattr(self.canvas, "_dead", False) and m.path == (self.engine.path if self.engine else None))

    def _mpv_t(self, idx):
        """mpv time of frame idx (a hair early so hr-seek lands on it, never on its predecessor)."""
        e = self.engine
        t = e.time_of(idx)
        gap = (t - e.time_of(idx - 1)) if idx > 0 else 0.004
        return t + (self.mpv.t0 or 0.0) - min(0.0005, max(1e-5, gap / 4))

    def _mpv_idx(self, t):
        return self.engine.clamp(self.engine.index_of(t - (self.mpv.t0 or 0.0) + 1e-4))

    def _play_mpv(self, start):
        m, e = self.mpv, self.engine
        e.pause()
        cur = m.time_pos
        wait = None
        target = self._mpv_t(start)
        if cur is None or self._mpv_idx(cur) != start:
            wait = m.restarts
            m.seek(target)
        m.set_display_fps(1.0 / self.refresh_interval())
        if getattr(self, "_mpv_clock_sync", False):
            m.set("video-sync", "audio")      # (reset-on-next-file=all drops runtime changes with each video)
        m.set_double("speed", self.SPEEDS[self.speed_cb.currentIndex()])
        m.play()
        self._mpv_playing = True
        self._mpv_t_play = time.monotonic()
        self._mpv_hw_known = False
        self._mpv_rate_ref = None                # (wall, time-pos) once playback runs: speed check
        self.canvas.set_mpv_active(True, wait)
        self.canvas.set_playing(True)

    def _mpv_verify(self):
        """Once per session: compare mpv's picture with the exact decoder's picture of the same frame (paused).
        A mismatch retries with mpv's basic renderer (gpu-dumb-mode); a second one turns libmpv off."""
        m = self.mpv
        if m is None or getattr(self, "_mpv_verified", False) or m.failed or self.is_playing:
            return
        cur = self.current
        t = m.time_pos if m.loaded else None
        same = (cur is not None and cur.bgr is not None and not getattr(cur, "preview", False) and t is not None
                and m.restarts > 0 and self._mpv_idx(t) == cur.idx)
        if not same:
            self._mpv_verify_polls = getattr(self, "_mpv_verify_polls", 0) + 1
            if self._mpv_verify_polls <= 25:
                later(200, self, self._mpv_verify)      # mpv still opening / seeking: look again shortly
            return
        ok = self.canvas.verify_mpv(cur.bgr)
        if ok is None:
            return                                      # can't tell on this frame: checked again on the next pause
        if ok:
            self._mpv_verified = True
            mode = "basic renderer" if getattr(self, "_mpv_basic", False) else "full renderer"
            logger.info(f"libmpv picture verified ({mode})")
            return
        if not getattr(self, "_mpv_basic", False):
            logger.warning("libmpv drew a wrong / empty picture on this GPU - retrying with its basic renderer")
            self._mpv_basic = True
            self.stop_mpv()
            self._configure_mpv()
            return
        self.on_mpv_failed("its picture did not match the video on this GPU / driver")

    def _mpv_check_rate(self, m, t):
        """display-resample times video by the canvas' swaps. When they do not come at the display rate (vsync off,
        a slow or software GL, a monitor whose rate Qt misreports) video would play too slowly or too fast: then
        mpv is switched to its own clock for the rest of the session (still smooth, not vsync-locked)."""
        if t is None or getattr(self, "_mpv_clock_sync", False) or not self.canvas.mpv_showing:
            return
        now = time.monotonic()
        ref = getattr(self, "_mpv_rate_ref", None)
        if ref is None:
            if now - self._mpv_t_play > 0.4:
                self._mpv_rate_ref = (now, t)
            return
        if now - ref[0] < 1.2:
            return
        want = self.SPEEDS[self.speed_cb.currentIndex()]
        rate = (t - ref[1]) / (now - ref[0]) / max(want, 1e-6)
        self._mpv_rate_ref = (now, t)
        if 0.93 <= rate <= 1.07:
            return
        est = m.get_double("estimated-display-fps")
        logger.warning(f"libmpv played at {rate:.2f}x (display swaps at {est or 0:.1f}/s, expected "
                       f"{1.0 / self.refresh_interval():.1f}/s) - switching it to its own clock")
        self._mpv_clock_sync = True
        m.set("video-sync", "audio")
        self.rt_status("libmpv · clock-synced (the display's refresh could not be followed)")

    def _mpv_tick(self):
        m = self.mpv
        if m is None or m.failed:
            if self._mpv_playing:
                self.on_mpv_failed(m.failed if m else "player gone")
            return
        t = m.time_pos
        if t is not None and self.canvas.mpv_showing:
            idx = self._mpv_idx(t)
            if idx != self.current_frame or self.current is None:
                self.current = Frame(idx, None, None)
                self.current_frame = idx
                if self.review is not None and self.review.active:
                    self.review.on_frame(idx, True)
                self.strip.set_current(idx)
                if not self.seekbar_moving:
                    self.seekbar.set_playhead(idx)
                now = time.monotonic()
                if now - self._ui_t >= 0.05:
                    self._ui_t = now
                    self.update_labels()
        self._mpv_check_rate(m, t)
        if not self._mpv_hw_known and time.monotonic() - self._mpv_t_play > 1.0:
            self._mpv_hw_known = True
            hw = m.get_str("hwdec-current") or "no"
            self.rt_status("libmpv · " + ("software decoding" if hw in ("no", "") else f"hardware decoding ({hw})")
                           + " · display-synced")
        if m.get_flag("eof-reached"):
            self.on_video_end()                     # pauses: no further ticks


    # ================================================================ playback stats
    def toggle_stats(self, on):
        if on:
            if self.engine:
                self.engine.reset_stats()
            self._stats_hist = []
            self._stats_timer.start()
            self._update_stats()
        else:
            self._stats_timer.stop()
            self.canvas.set_stats(None)

    def _update_stats(self):
        e = self.engine
        if not e:
            self.canvas.set_stats(["Playback statistics", "no video loaded"])
            return
        if self.mpv is not None and (self._mpv_playing or self.canvas.mpv_showing):
            st = self.mpv.stats()
            hz = 1.0 / self.refresh_interval()
            f = lambda v, fmt="{:.2f}": "?" if v is None else fmt.format(v)
            lines = [f"video   {e.width}x{e.height} · {e.fps:.3f} fps · {st['codec'] or ''}",
                     f"engine  libmpv · video-sync {st['sync']} · display {f(st['display_fps'])} Hz ({hz:.0f})",
                     f"decode  hwdec: {st['hwdec']}" + ("  (software)" if st['hwdec'] in ('no', '') else ""),
                     f"shown   {f(st['fps'])} fps (target {e.fps * e.speed:.2f})",
                     f"drops   output {f(st['drops'], '{}')} · decoder {f(st['decoder_drops'], '{}')} "
                     f"· late {f(st['delayed'], '{}')}"]
            if st["drops"] or st["decoder_drops"]:
                lines.append("!frames dropped: the decoder can't keep up (try another hwdec in Settings)")
            self.canvas.set_stats(lines)
            return
        st = e.stats()
        now = time.monotonic()
        self._stats_hist.append((now, st["presented"], st["drop_present"], st["drop_decode"], st["drop_resync"]))
        self._stats_hist = [h for h in self._stats_hist if now - h[0] <= 2.0]
        h0, h1 = self._stats_hist[0], self._stats_hist[-1]
        dt = max(1e-3, h1[0] - h0[0])
        shown_fps = (h1[1] - h0[1]) / dt
        hz = 1.0 / self.refresh_interval()
        playing = e.is_playing
        rate = e.fps * (e.speed if playing else 1.0)
        dp, dd, dr = (h1[2] - h0[2]), (h1[3] - h0[3]), (h1[4] - h0[4])
        gpu = getattr(self.canvas, "gpu", False)
        par = int((e.realtime or {}).get("parallel", 1))
        mode = ("pipeline · " + ("original NV12" if st["out"] == "nv12" else "screen-size")
                + (f" · {par} parallel decoders" if par > 1 else "") if st["pipeline"]
                else f"exact decoder ({e.backend})")
        lines = [
            f"video   {e.width}x{e.height} · {e.fps:.3f} fps{' VFR' if e.is_vfr else ''} · speed {e.speed:g}x",
            f"display {hz:.0f} Hz · {'GPU' if gpu else 'software'} canvas"
            + (f" · display sync {(e.sync_factor - 1) * 100:+.2f}%" if abs(e.sync_factor - 1) > 1e-9 else ""),
            f"decode  {mode}",
            f"shown   {shown_fps:5.1f} fps (target {rate:.1f})" if playing else "shown   paused",
            f"queue   {st['queue']} frames · decode {st['decode_ms']:.1f} ms/frame",
            f"ui      {self._ui_ms:.1f} ms/frame" + (f" · upload {getattr(self.canvas, 'upload_ms', 0.0):.1f} ms" if gpu else ""),
            f"drops   display {st['drop_present']} · decoder {st['drop_decode']} · resync {st['drop_resync']}"
            f" · source {st.get('drop_source', 0)}",
        ]
        if playing and dt > 0.9:
            if dp:
                lines.append(f"!{dp / dt:.1f}/s missed refreshes: UI thread too busy")
            if dd or dr:
                lines.append(f"!{(dd + dr) / dt:.1f}/s decoder behind: try Settings ▸ Playback ▸ Real-time pipeline")
            if rate > hz + 0.5:
                lines.append(f"!video {rate:.0f} fps > display {hz:.0f} Hz: frames must be skipped")
        self.canvas.set_stats(lines)


    # ================================================================ recent files
    RECENT_MAX = 12

    def recent_files(self):
        v = self.settings.value("recent/files", [])
        return [p for p in (v if isinstance(v, list) else [v]) if p]

    def add_recent(self, path):
        p = os.path.abspath(path)
        items = [p] + [x for x in self.recent_files() if os.path.normcase(x) != os.path.normcase(p)]
        self.settings.setValue("recent/files", items[:self.RECENT_MAX])

    def remove_recent(self, paths):
        drop = {os.path.normcase(p) for p in paths}
        self.settings.setValue("recent/files", [x for x in self.recent_files() if os.path.normcase(x) not in drop])

    def _build_recent_menu(self):
        """Rebuilt every time it opens, so 'not found' reflects drives plugged in / out."""
        m = self.recent_menu
        m.clear()
        items = self.recent_files()
        if not items:
            a = m.addAction("No recent videos")
            a.setEnabled(False)
            return
        missing = []
        for i, p in enumerate(items):
            name, folder = os.path.basename(p), os.path.dirname(p)
            exists = os.path.exists(p)
            if not exists:
                missing.append(p)
            label = (f"&{i + 1}  " if i < 9 else "    ") + name.replace("&", "&&")
            short = folder if len(folder) <= 60 else "…" + folder[-58:]
            a = m.addAction(f"{label}    —  {short.replace('&', '&&')}" + ("" if exists else "   (not found)"))
            a.setToolTip(p)
            a.setStatusTip(p)
            a.setEnabled(exists)
            if self.video_path and os.path.normcase(os.path.abspath(self.video_path)) == os.path.normcase(p):
                a.setIcon(utils.newIcon("play"))
            a.triggered.connect(lambda _=False, p=p: self.open_recent(p))
        m.setToolTipsVisible(True)
        m.addSeparator()
        rm = m.addAction(f"Remove missing ({len(missing)})")
        rm.setEnabled(bool(missing))
        rm.triggered.connect(lambda: self.remove_recent(missing))
        m.addAction(utils.newIcon("delete"), "Clear list").triggered.connect(
            lambda: self.settings.setValue("recent/files", []))

    def open_recent(self, path):
        if not os.path.exists(path):
            if QMessageBox.question(self, "Open Recent", f"This video can't be found:\n{path}\n\n"
                                    "Remove it from the recent list?") == QMessageBox.StandardButton.Yes:
                self.remove_recent([path])
            return
        self.load_video(path)

    def reopen_last(self):
        for p in self.recent_files():
            if os.path.exists(p):
                self.load_video(p)
                return
        self.osd("No recent video found")


    # ================================================ crash sentinel / safe mode
    def _sentinel_set(self, stage, path=None, secs=None):
        """Written to disk *before* a risky step; cleared once it is stable. If the app dies in between,
        the next start knows exactly what it was doing and turns on safe settings."""
        import json
        info = dict(stage=stage, path=path or self.video_path, gpu=bool(getattr(self.canvas, "gpu", False)),
                    hw=bool(self.hw_decode), t=time.time(), version=__version__)
        self.settings.setValue("crash/sentinel", json.dumps(info))
        self.settings.sync()
        if secs:
            self._sentinel_timer.start(int(secs * 1000))

    def _sentinel_clear(self):
        self._sentinel_timer.stop()
        self.settings.remove("crash/sentinel")
        self.settings.sync()

    def _read_previous_crash(self):
        """Called before the canvas exists: apply safe settings if the last run died mid-way."""
        import json
        raw = self.settings.value("crash/sentinel")
        if not raw:
            return None
        self.settings.remove("crash/sentinel")
        try:
            info = json.loads(raw)
        except ValueError:
            return None
        what = {"load": "opening", "play": "starting playback of", "gpu": "drawing the first frames of"}.get(
            info.get("stage"), "working on")
        if info.get("gpu"):
            self.settings.setValue("ui/gpu_failed", f"the previous session stopped while {what} a video")
        if info.get("hw"):
            self.settings.setValue("decoder/hw_inproc", False)
        self.settings.setValue("safe/last_crash", raw)
        self.settings.sync()
        logger.warning(f"Previous session ended unexpectedly while {what} {info.get('path')} - safe settings applied")
        return info, what

    def _report_previous_crash(self, prev):
        from AIO_QC.tools.frame_extractor.core import crash
        info, what = prev
        log = crash.latest_crash_log()
        changed = []
        if info.get("gpu"):
            changed.append("• the software renderer is used instead of the GPU renderer (Settings ▸ Display ▸ Video renderer)")
        if info.get("hw"):
            changed.append("• in-app hardware decoding is off (Settings ▸ Decoding)")
        if not changed:
            changed.append("• background analysis is limited while the file loads")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Frame Extractor stopped unexpectedly")
        box.setText(f"The last session stopped while {what}:\n{info.get('path')}\n\n"
                    "Safe settings are now active:\n" + "\n".join(changed) +
                    "\n\nIf it happens again, please send the crash report:\n" +
                    (log or crash.LOG_FILE))
        open_btn = box.addButton("Open log folder", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is open_btn:
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(crash.LOG_DIR))

    def open_log_folder(self):
        from AIO_QC.tools.frame_extractor.core import crash
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        os.makedirs(crash.LOG_DIR, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(crash.LOG_DIR))


    # ============================================================ canvas / renderer
    def _make_canvas(self):
        mode = self.settings.value("ui/renderer", "auto")
        gpu_failed = self.settings.value("ui/gpu_failed", "")
        if mode == "gpu" or (mode == "auto" and not gpu_failed):
            try:
                from AIO_QC.tools.frame_extractor.widgets.gl_canvas import GLVideoCanvas
                c = GLVideoCanvas("Load or drop a video")
                c.failed.connect(self.on_gpu_failed)
                return c
            except Exception as e:
                logger.warning(f"GPU canvas unavailable: {e}")
                self.settings.setValue("ui/gpu_failed", str(e))
        return VideoCanvas("Load or drop a video")

    def _connect_canvas(self, c):
        c.wheel_step.connect(self.step_frames)
        c.roi_changed.connect(self.on_roi_changed)
        c.cursor_moved.connect(self.on_cursor_moved)
        c.customContextMenuRequested.connect(self.canvas_menu)
        c.zoom_changed.connect(self.on_zoom_changed)
        if hasattr(c, "mpv_failed"):
            c.mpv_failed.connect(self.on_mpv_failed)
        if hasattr(self, "resize_timer"):
            c.resized.connect(self.resize_timer.start)
        if getattr(c, "gpu", False):
            c.present_hook = self.on_present

    def _swap_canvas(self, new):
        """Replace the canvas widget keeping the view state (renderer switch / GPU failure)."""
        self.stop_mpv()              # its render context belongs to the old canvas' GL context
        old = self.canvas
        playing = self.is_playing
        if playing:
            self.pause()
        new._vw, new._vh = old._vw, old._vh
        new._zoom, new._tz, new._cx, new._cy = old._zoom, old._tz, old._cx, old._cy
        new._roi, new._hud = old._roi, old._hud
        new.set_roi_mode(self.actions_.roi.isChecked())
        self._connect_canvas(new)
        self.canvas_layout.replaceWidget(old, new)
        old.hide()
        old.deleteLater()
        self.canvas = new
        if self.engine:
            self.engine.display_images = not getattr(new, "gpu", False)
        if self.current is not None:
            if self.current.image is None and self.current.bgr is not None and self.engine:
                self.engine.make_display(self.current)     # software canvas needs the display copy
            new.show_frame(self.current)
        if self.actions_.stats.isChecked():
            self._update_stats()
        elif not self.engine:
            new.set_text("Load or drop a video")
        self._apply_colorspace()
        self.update_renderer_status()
        if self.engine:
            self.configure_realtime()

    def on_gpu_failed(self, why):
        if self.sender() is not self.canvas:
            return
        self.settings.setValue("ui/gpu_failed", why)
        self.flash_status(f"GPU renderer unavailable ({why}) - using the software renderer", 8000)
        self._swap_canvas(VideoCanvas("Load or drop a video"))
        self.renderer_actions["software" if self.settings.value("ui/renderer", "auto") == "software"
                              else "auto"].setChecked(True)

    def set_renderer(self, mode):
        self.settings.setValue("ui/renderer", mode)
        if mode == "gpu":
            self.settings.remove("ui/gpu_failed")        # explicit choice: try again
        want_gpu = mode == "gpu" or (mode == "auto" and not self.settings.value("ui/gpu_failed", ""))
        if want_gpu != getattr(self.canvas, "gpu", False):
            self._swap_canvas(self._make_canvas())
        self.update_renderer_status()

    def update_renderer_status(self):
        gpu = getattr(self.canvas, "gpu", False)
        self.render_label.setText("GPU" if gpu else "CPU")
        why = "" if gpu else self._cpu_reason()
        self.render_label.setToolTip("Renderer: " + ("GPU (OpenGL) - full-resolution frames, vsync-paced playback"
                                                      if gpu else "software (QPainter)") +
                                     (f"\nWhy: {why}" if why else "") + "  ·  Settings ▸ Display ▸ Video renderer")
        if why and not getattr(self, "_cpu_told", False):
            self._cpu_told = True
            logger.warning(f"Frame Extractor: software renderer - {why}")
            later(1500, self, lambda: self.flash_status(f"CPU renderer: {why} — Settings ▸ Display ▸ Video renderer ▸ GPU to try the GPU "
                                                        "again", 10000))

    def _cpu_reason(self):
        mode = self.settings.value("ui/renderer", "auto")
        if mode == "software":
            return "chosen in Settings ▸ Display ▸ Video renderer"
        failed = self.settings.value("ui/gpu_failed", "")
        return f"the GPU renderer was turned off ({failed})" if failed else ""

    def _apply_colorspace(self):
        if getattr(self.canvas, "gpu", False) and getattr(self, "_colorspace", None):
            self.canvas.set_colorspace(*self._colorspace)

    # ================================================================ zoom
    def _zoom_anchor(self):
        from PyQt6.QtGui import QCursor
        p = self.canvas.mapFromGlobal(QCursor.pos())
        return QPointF(p) if self.canvas.rect().contains(p) else None

    def zoom_in(self):
        self.canvas.zoom_by(1.25, self._zoom_anchor())

    def zoom_out(self):
        self.canvas.zoom_by(0.8, self._zoom_anchor())

    def zoom_fit(self, *_):
        self.canvas.fit()
        self.actions_.zoom_fit.setChecked(True)        # stays lit while fitted

    def zoom_actual(self, *_):
        self.canvas.actual_pixels(self._zoom_anchor())
        self.actions_.zoom_actual.setChecked(True)      # stays lit while at 100%

    def on_zoom_changed(self, z):
        if self.sender() is not self.canvas:
            return
        self._sync_zoom_buttons()

    def _sync_zoom_buttons(self):
        """Fit / 100% buttons stay lit while the view is exactly there (also after a window resize)."""
        fit = self.canvas.is_fit
        if self.actions_.zoom_fit.isChecked() != fit:
            self.actions_.zoom_fit.setChecked(fit)
        actual = self.canvas.is_actual
        if self.actions_.zoom_actual.isChecked() != actual:
            self.actions_.zoom_actual.setChecked(actual)


    # ======================================================= real-time playback
    def frame_image(self, frame=None):
        """(image, scale) for analysis: full-res bgr (scale 1) or the display-size frame of the
        real-time pipeline (scale = display / full). Coordinates in full-res * scale."""
        f = frame or self.current
        if f is None:
            return None, 1.0
        if f.bgr is not None:
            return f.bgr, 1.0
        if getattr(f, "nv12", None) is not None and self.engine:
            from AIO_QC.tools.frame_extractor.widgets.video_engine import nv12_small_bgr
            img = nv12_small_bgr(*f.nv12, max_w=1280)
            return img, img.shape[1] / max(1, self.engine.width)
        if f.small is not None and self.engine:
            return f.small, f.small.shape[1] / max(1, self.engine.width)
        return None, 1.0

    def set_playback_mode(self, mode):
        self.playback_mode = mode
        self.settings.setValue("playback/realtime", mode)
        self.playback_actions[mode].setChecked(True)
        if self.engine:
            self.configure_realtime()

    def choose_ffmpeg(self):
        path, _ = QFileDialog.getOpenFileName(self, "ffmpeg executable", "", "ffmpeg (ffmpeg ffmpeg.exe);;All files (*)")
        if path:
            self.settings.setValue("playback/ffmpeg", path)
            self.flash_status(f"ffmpeg: {path}")
            if self.engine:
                self.configure_realtime(force=True)

    def configure_realtime(self, force=False):
        """Decide (and, if needed, benchmark in the background) how this video is played."""
        from AIO_QC.tools.frame_extractor.core import preview_pipe as PP
        e = self.engine
        if not e:
            return
        e.set_realtime(None)
        self.rt_info = None
        self._configure_mpv()
        ff = PP.find_ffmpeg(self.settings.value("playback/ffmpeg", ""))
        mode = self.playback_mode
        big = e.width * e.height >= 2560 * 1440
        gpu = getattr(self.canvas, "gpu", False)
        from AIO_QC.tools.frame_extractor.core.sources import HAVE_PYAV
        e.set_preview_sampler("pyav" if HAVE_PYAV else ("ffmpeg" if ff else None), ff)
        fp = PP.find_ffprobe(ff) if ff else None
        if fp:
            from AIO_QC.tools.frame_extractor.widgets.gl_shaders import colorspace_from_probe

            def got_color(r, e=e):
                if e is self.engine:
                    self._colorspace = colorspace_from_probe(*r)
                    e.colorspace = self._colorspace
                    self._apply_colorspace()
            self._run_bg(lambda: PP.probe_color(e.path, fp), got_color, priority=QThread.Priority.LowPriority)
        # keyframes help resync to real time without decode preroll (only for files small enough to scan fast)
        small_file = os.path.getsize(e.path) < 300_000_000 if os.path.exists(e.path) else False
        if ff and not e._keyframes and small_file:
            if fp:
                self._run_bg(lambda: PP.keyframes_ffprobe(e.path, fp),
                             lambda r, e=e: r and e is self.engine and e.set_keyframes(*r),
                             priority=QThread.Priority.LowPriority)
        # GPU canvas: the pipeline's NV12 frames are the cheapest to show at any size (no CPU colour conversion,
        # half the upload bytes) and pausing converts them to the exact frame instantly -> use it for all videos
        if mode == "off" or not ff or (mode == "auto" and not (big or gpu)):
            why = ("off" if mode == "off" else "ffmpeg not found — Settings ▸ Playback" if not ff
                   else "exact decoder (below 2560×1440)")
            self.rt_status(why)
            return
        w, h = PP.display_size(e.width, e.height, *self.canvas.physical_size())
        key = (f"playback/chain4/{ff}|{os.path.splitext(e.path)[1].lower()}|{e.width}x{e.height}|{e.fps:.2f}"
               f"|{'gpu' if gpu else 'sw'}")
        cached = self.settings.value(key)
        if cached and not force:
            chain, fps, out, par = (cached.split("|") + ["bgr", "1"])[:4]
            self._enable_rt(ff, chain, float(fps), out, int(par))
            return
        self.rt_status("benchmarking decoders…")
        need = e.fps * max(1.0, self.SPEEDS[self.speed_cb.currentIndex()])

        def work():
            """Ordered search for the first way to reach the video's rate (+5 %), best picture first:
               1 original NV12 (GPU canvas), one decoder      2 original NV12, parallel decoders
               3 screen-size, one decoder                     4 screen-size, parallel decoders
            Parallel decoder count is sized to the shortfall (procs.plan_parallel_decoders), not fixed."""
            from AIO_QC.core import procs
            target = need * 1.05
            chain, fps, results = PP.choose_chain(ff, e.path, w, h, need)
            if not chain:
                return None, 0.0, results, "bgr", 1
            _, ver = PP.candidate_chains(ff)
            fw, fh = e.width // 2 * 2, e.height // 2 * 2
            kf = list(e._keyframes or [])              # not known yet while a video is still being indexed
            can_par = bool(kf and e.timestamps) and procs.cpu_count() >= 2
            seg = max(1, int(min(e.fps * 2, max(np.diff(kf)) if len(kf) > 1 else e.fps)))
            tried = []

            def par(ow, oh, pix, single):
                n = procs.plan_parallel_decoders(need, single, ow * oh * (3 if pix == "bgr24" else 1.5), seg)
                if n <= 1 or not can_par:
                    return None
                pfps, nw = PP.benchmark_parallel(ff, e.path, chain, ow, oh, e._keyframes, e.time_of,
                                                 e.frame_count, n, ver, pix)
                tried.append(f"{pix} {ow}x{oh} x{nw}: {pfps and round(pfps)} fps")
                return (pfps, nw) if pfps else None

            options = []
            if gpu:
                nat = PP.benchmark(ff, e.path, chain, fw, fh, version=ver, pix_fmt="nv12")
                tried.append(f"nv12 {fw}x{fh} x1: {nat and round(nat)} fps")
                if nat:
                    options.append((nat, "nv12", 1))
                if nat and nat >= target:
                    return chain, nat, results, "nv12", 1
                r = par(fw, fh, "nv12", nat or fps / 2)
                if r:
                    options.append((r[0], "nv12", r[1]))
                    if r[0] >= target:
                        logger.info("Playback plan: " + " · ".join(tried))
                        return chain, r[0], results, "nv12", r[1]
            options.append((fps, "bgr", 1))
            if fps < target:
                r = par(w, h, "bgr24", fps)
                if r:
                    options.append((r[0], "bgr", r[1]))
            logger.info("Playback plan: " + " · ".join(tried) + f" · bgr {w}x{h} x1: {round(fps)} fps")
            # nothing reaches the rate: the fastest (prefer original quality on a tie within 3 %)
            best = max(options, key=lambda o: o[0] * (1.03 if o[1] == "nv12" else 1.0))
            return chain, best[0], results, best[1], best[2]

        def done(res, e=e):
            if e is not self.engine or res is None:
                return
            chain, fps, results, out, parallel = res
            logger.info("Playback benchmark: " + ", ".join(f"{k} {v:.0f} fps" if v else f"{k} n/a" for k, v in results.items()))
            if not chain:
                self.rt_status("pipeline unavailable — exact decoder")
                return
            self.settings.setValue(key, f"{chain}|{fps:.1f}|{out}|{parallel}")
            self._enable_rt(ff, chain, fps, out, parallel)
        self._run_bg(work, done, priority=QThread.Priority.LowPriority)

    def _enable_rt(self, ff, chain, fps, out="bgr", parallel=1):
        mpv_status = self.mpv is not None
        from AIO_QC.tools.frame_extractor.core.preview_pipe import candidate_chains, CHAIN_LABELS
        _, ver = candidate_chains(ff)
        if out == "nv12" and not getattr(self.canvas, "gpu", False):
            out = "bgr"
        self.engine.set_realtime(dict(ffmpeg=ff, chain=chain, version=ver, out=out, parallel=parallel))
        self.rt_info = (chain, fps)
        ok = fps >= self.engine.fps
        q = ("original quality" if out == "nv12" else "screen resolution") + (
            f" · {parallel} parallel decoders" if parallel > 1 else "")
        if mpv_status:
            return      # libmpv plays; the pipeline is only the fallback - keep the libmpv status text
        self.rt_status(f"real-time: {CHAIN_LABELS.get(chain, chain)} · {q} · {fps:.0f} fps capable"
                       + ("" if ok else " (below the video's frame rate — frames will be skipped to stay in time)"))

    def rt_status(self, text):
        self.play_label.setText("▶ " + text.split(" (")[0])
        self.play_label.setToolTip("Playback: " + text + "\nPausing, stepping, saving and zooming always use the "
                                   "exact full-resolution decoder.")


    # ============================================================ markers
    def marker_out_dir(self):
        name = os.path.splitext(os.path.basename(self.video_path))[0] if self.video_path else "video"
        return os.path.join(self.output_dir, name, "markers")

    def refresh_markers(self, save=True):
        states = {f: ("saved" if m.get("saved") else m.get("src") == "suggested" and "suggested" or "pending")
                  for f, m in self.markers.items()}
        self.strip.set_markers(states)
        if hasattr(self, "tc"):
            self.update_tc_context()
        busy = self.marker_worker.pending() if self.marker_worker else 0
        self.mdock.set_markers(self.markers, lambda f: self.markers.get(f, {}).get("sharp"),
                               self.marker_out_dir() if self.video_path else "", busy)
        if save:
            self.save_session()

    def _marker_frame(self):
        if self.current is None or self.current.preview:
            return None
        return self.current

    def toggle_marker(self):
        fr = self._marker_frame()
        if fr is None:
            return
        if fr.idx in self.markers:
            self.remove_markers([fr.idx])
            self.osd(f"Marker #{fr.idx} removed")
            return
        self.add_marker(fr.idx, fr.bgr)
        pend = sum(1 for m in self.markers.values() if not m.get("saved"))
        self.osd(f"Marker #{fr.idx}" + (" · saving" if self.mdock.auto.isChecked() else f" · {pend} pending"))

    def add_marker(self, idx, bgr=None, src="manual", refresh=True):
        roi = list(self.roi) if (self.roi and self.mdock.crop.isChecked()) else None
        sharp = None
        if bgr is None and self.current is not None and self.current.idx == idx:
            img0, sc = self.frame_image()
            if img0 is not None and sc != 1.0:
                r = [int(v * sc) for v in roi] if roi else None
                img = img0[r[1]:r[3], r[0]:r[2]] if r else img0
                sharp = round(sharpness(img), 1) if img.size else None
        if bgr is not None:
            img = bgr[roi[1]:roi[3], roi[0]:roi[2]] if roi else bgr
            sharp = round(sharpness(img), 1) if img.size else None
        else:
            s = self.strip.nearest_sample(idx, max_dist=1)
            sharp = round(s.sharp, 1) if s else None
        self.markers[idx] = dict(roi=roi, saved=None, src=src, sharp=sharp)
        if self.mdock.auto.isChecked() and src == "manual":
            self.ensure_marker_worker().save(idx, roi)
        if refresh:
            self.refresh_markers()

    def remove_markers(self, frames):
        for f in frames:
            self.markers.pop(f, None)
            self.mdock.thumbs.pop(f, None)
        self.refresh_markers()

    def move_marker(self, old, new):
        if old not in self.markers or new in self.markers or not self.engine:
            return
        m = self.markers.pop(old)
        m.update(saved=None, sharp=None, src="manual")
        self.markers[self.engine.clamp(new)] = m
        self.mdock.thumbs.pop(old, None)
        self.refresh_markers()
        self.goto_frame(new)

    def jump_marker(self, direction):
        marks = sorted(self.markers)
        base = self.current_frame if self.is_playing else self.requested_frame
        cands = [m for m in marks if (m > base if direction > 0 else m < base)]
        if cands:
            self.goto_frame(cands[0] if direction > 0 else cands[-1])

    def ensure_marker_worker(self):
        if self.marker_worker is None and self.video_path:
            o = self.mdock.options()
            from AIO_QC.tools.frame_extractor.core.hashdb import default_db_path
            self.marker_worker = MarkerWorker(self.video_path, self.marker_out_dir(),
                                              self.engine.backend if self.engine else "auto", self.out_cfg,
                                              dict(self.model_cfg) if o["preannot"] != "none" else None,
                                              o["preannot"], default_db_path(self.output_dir))
            self.marker_worker.saved.connect(self.on_marker_saved)
            self.marker_worker.snapped.connect(self.on_marker_snapped)
            self.marker_worker.failed.connect(self.on_marker_failed)
            self.marker_worker.idle.connect(lambda: (self.refresh_markers(save=False), self._marker_task_tick()))
            self.marker_worker.saved.connect(lambda *_: self._marker_task_tick())
            self.marker_worker.start(QThread.Priority.LowPriority)
        return self.marker_worker

    def stop_marker_worker(self):
        if self.marker_worker is not None:
            self.marker_worker.stop()
            self.marker_worker = None
        if getattr(self, "_mtask", None) is not None:
            self._mtask.finish()
            self._mtask, self._mtask_total = None, 0

    def on_marker_options(self):
        o = self.mdock.options()
        if o["preannot"] != "none" and not self.model_cfg["models"]:
            self.flash_status("Pre-annotations need models: Settings → Models…")
        self.stop_marker_worker()        # restarted with the new options on next use
        self.save_session()

    def extract_markers(self, frames=None):
        if not self.video_path:
            return
        todo = [f for f in (frames or sorted(self.markers)) if f in self.markers and not self.markers[f].get("saved")]
        if not todo:
            self.osd("No pending markers")
            return
        w = self.ensure_marker_worker()
        for f in todo:
            w.save(f, self.markers[f].get("roi"))
        self._marker_task(len(todo))
        self.osd(f"Extracting {len(todo)} marker(s)…")
        self.refresh_markers(save=False)

    def _marker_task(self, n):
        """Marker saving in the shared background-task indicator (finished when the worker goes idle)."""
        from AIO_QC.widgets import tasks
        if getattr(self, "_mtask", None) is None:
            self._mtask = tasks.start(f"Saving markers · {os.path.basename(self.video_path or '')}",
                                      cancel=lambda: self.stop_marker_worker())
        self._mtask_total = getattr(self, "_mtask_total", 0) + n

    def _marker_task_tick(self):
        t = getattr(self, "_mtask", None)
        if t is None:
            return
        busy = self.marker_worker.pending() if self.marker_worker else 0
        total = max(1, getattr(self, "_mtask_total", 1))
        if not busy:
            t.finish()
            self._mtask, self._mtask_total = None, 0
        else:
            t.update(1 - busy / total, f"{busy} left")

    def on_marker_saved(self, idx, path, thumb):
        if self.sender() is not self.marker_worker:
            return
        if idx in self.markers:
            self.markers[idx]["saved"] = path
            self.mdock.thumbs[idx] = QPixmap.fromImage(thumb)
        self.refresh_markers()

    def on_marker_failed(self, idx, msg):
        self.flash_status(f"Marker {idx}: {msg}" if idx >= 0 else msg, 6000)
        logger.warning(f"marker {idx}: {msg}")

    def snap_markers(self, frames, radius, then_extract=False):
        frames = [f for f in (frames or [f for f, m in self.markers.items() if not m.get("saved")]) if f in self.markers]
        if not frames:
            self.osd("No markers to snap")
            return
        w = self.ensure_marker_worker()
        if then_extract:
            self._extract_after_snap |= set(frames)
        for f in frames:
            w.snap(f, radius, self.markers[f].get("roi"))
        self.osd(f"Snapping {len(frames)} marker(s) to the sharpest frame (±{radius})…")

    def on_marker_snapped(self, old, new, sharp):
        if self.sender() is not self.marker_worker or old not in self.markers:
            return
        m = self.markers.pop(old)
        target = new if new not in self.markers else old
        m.update(sharp=round(sharp, 1))
        if target != old:
            m.update(saved=None)
            self.mdock.thumbs.pop(old, None)
        self.markers[target] = m
        if old in self._extract_after_snap:
            self._extract_after_snap.discard(old)
            if not m.get("saved"):
                self.marker_worker.save(target, m.get("roi"))
        self.refresh_markers()

    def suggest_markers(self, mode, every_s, quiet=False):
        from AIO_QC.tools.frame_extractor.core.markers import suggest
        if not self.engine or not self.strip.samples:
            self.osd("Timeline analysis hasn't produced samples yet")
            return []
        samples = {s.idx: (s.sharp, max(0.0, s.motion), s.flagged) for s in self.strip.samples.values()}
        if mode == "flagged" and not any(v[2] for v in samples.values()):
            QMessageBox.information(self, "Suggest", "No model-flagged samples. Turn on Settings → "
                                    "'Run models in timeline analysis' (needs models).")
            return []
        lo = self.in_frame or 0
        hi = self.out_frame if self.out_frame is not None else self.frame_count - 1
        frames = suggest(samples, mode, lo, hi, self.frame_rate, every_s,
                         float(self.settings.value("batch/min_sharpness", 0) or 0), self.strip.cuts,
                         self.engine.timestamps)
        near = max(1, int(self.frame_rate / 4))
        added = [f for f in frames if not any(abs(f - m) <= near for m in self.markers)]
        for f in added:
            self.add_marker(f, None, src="suggested", refresh=False)
        self.refresh_markers()
        done = 100 * len(samples) // max(1, len(self.strip.grid))
        if not quiet:
            self.osd(f"{len(added)} suggestion(s)" + (f" · timeline {done}% analysed" if done < 100 else ""))
        return added

    def auto_pick(self):
        """Suggest the sharpest frame per window, snap each to the exact sharpest frame, then extract."""
        added = self.suggest_markers("sharpest", self.mdock.every.value(), quiet=True)
        if added:
            self.snap_markers(added, self.mdock.radius.value(), then_extract=True)
            self.osd(f"Auto-pick: {len(added)} frame(s) → snap → extract")

    def send_markers_to_batch(self):
        pending = sorted(f for f, m in self.markers.items() if not m.get("saved")) or sorted(self.markers)
        if not pending:
            self.osd("No markers")
            return
        self.open_batch_queue(add_current=False)
        self.batch_dialog.add_marker_job(self.video_path, [[f, self.markers[f].get("roi")] for f in pending])

    def review_markers(self):
        d = self.marker_out_dir()
        if os.path.isdir(d):
            self.open_gallery(d)
        else:
            self.osd("Nothing extracted yet")


    # ============================================================ timeline
    def on_samples(self, samples):
        if self.sender() is not self.timeline:
            return   # late batch from a stopped worker
        self.strip.add_samples(samples)
        if getattr(self, "_ttask", None) is not None and self.strip.grid:
            self._ttask.update(len(self.strip.samples) / max(1, len(self.strip.grid)))
        if self._hover and self.preview.isVisible():
            self.on_timeline_hover(*self._hover)

    def on_analysis_done(self):
        if self.sender() is self.timeline:
            self.strip.analyzing = False
            self.strip.update()
            if getattr(self, "_ttask", None) is not None:
                self._ttask.finish()
                self._ttask = None

    def on_timeline_hover(self, frame, gpt):
        if not self.engine:
            return
        self._hover = (frame, gpt)
        sample = self.strip.nearest_sample(frame, max_dist=self.strip.grid_spacing() * 0.75)
        if sample is None and self.timeline:
            self.timeline.request(frame)   # analyse this spot next
        sub = f"frame {frame}"
        desc = self.strip.describe(frame)
        if desc:
            sub += " · " + desc
        top = self.seekbar.mapToGlobal(self.seekbar.rect().topLeft()).y()
        self.preview.show_at(gpt.x(), top, sample.thumb if sample else None,
                             format_time(self.engine.time_of(frame)), sub)

    def hide_preview(self):
        self._hover = None
        if hasattr(self, "preview"):
            self.preview.hide()

    def strip_menu(self, pos):
        m = QMenu(self)
        f = self.strip.marker_at(pos.x(), tol=8)
        if f is not None:
            mk = self.markers.get(f, {})
            m.addSection(f"Marker #{f}" + (" (saved)" if mk.get("saved") else ""))
            m.addAction("Go to marker").triggered.connect(lambda: self.goto_frame(f))
            m.addAction(utils.newIcon("snap"), "Snap to sharpest").triggered.connect(
                lambda: self.snap_markers([f], self.mdock.radius.value()))
            ex = m.addAction(utils.newIcon("extract"), "Extract now")
            ex.setEnabled(not mk.get("saved"))
            ex.triggered.connect(lambda: self.extract_markers([f]))
            m.addAction(utils.newIcon("delete"), "Remove marker").triggered.connect(lambda: self.remove_markers([f]))
            m.exec(self.strip.mapToGlobal(pos))
            return
        thr = m.addAction("Set blur threshold from current frame")
        thr.setEnabled(self.current_sharpness is not None)
        thr.triggered.connect(lambda: self._set_blur_threshold(round(self.current_sharpness, 1)))
        clr = m.addAction("Clear blur threshold")
        clr.triggered.connect(lambda: self._set_blur_threshold(0))
        m.addSeparator()
        m.addAction(self.actions_.timeline_models)
        re = m.addAction("Re-analyze timeline")
        re.triggered.connect(self.start_timeline)
        m.exec(self.strip.mapToGlobal(pos))

    def _set_blur_threshold(self, v):
        self.settings.setValue("batch/min_sharpness", v)
        self.strip.set_blur_threshold(v)
        if self.batch_dialog:
            self.batch_dialog.sharp_sb.setValue(v)
        self.osd(f"Blur threshold {v:g}" if v else "Blur threshold off")

    def jump_flag(self, direction):
        flagged = self.strip.flagged_frames() or sorted(self.strip.cuts)
        if not flagged:
            self.osd("No flagged samples yet")
            return
        base = self.current_frame if self.is_playing else self.requested_frame
        cands = [f for f in flagged if (f > base if direction > 0 else f < base)]
        if cands:
            self.goto_frame(cands[0] if direction > 0 else cands[-1])

    def set_timeline_models(self, on):
        self.timeline_models = on
        self.settings.setValue("timeline/models", on)
        if on and not self.model_cfg["models"]:
            self.flash_status("No models configured: Settings → Models…")
        if self.engine:
            self.start_timeline()

    # ============================================================== models
    def open_models(self, tab=0):
        d = ModelsDialog(self, self.model_cfg,
                         frame_getter=lambda: self.current.bgr if self.current is not None and not self.current.preview else None,
                         show_detections=lambda per_model, cfg: self.show_detections(per_model, cfg))
        d.tabs.setCurrentIndex(tab)
        if d.exec():
            self.model_cfg = d.cfg_out()
            self.stop_marker_worker()
            save_model_cfg(self.model_cfg)
            self.stop_detector()
            if self.batch_dialog:
                self.batch_dialog.refresh_models_label()
            if self.engine and self.timeline_models:
                self.start_timeline()
            if self.actions_.detect.isChecked() and self.current is not None:
                self.request_detections(self.current)

    def stop_detector(self):
        if self.detector:
            self.detector.stop()
            self.detector = None
        self.detections_cache = {}

    def toggle_detections(self, on):
        if on and not self.model_cfg["models"]:
            self.actions_.detect.setChecked(False)
            self.open_models()
            if not self.model_cfg["models"]:
                return
            self.actions_.detect.setChecked(True)   # setChecked doesn't re-trigger; fall through
        if on and self.current is not None:
            self.request_detections(self.current)
        elif not on:
            self.canvas.set_detections([])
            self.pixel_label.clear()

    def request_detections(self, frame):
        if not self.model_cfg["models"] or (self.review is not None and self.review.active):
            return
        cached = self.detections_cache.get(frame.idx)
        if cached is not None:
            self.show_detections(cached)
            return
        if self.detector is None:
            self.detector = DetectWorker(dict(self.model_cfg))
            self.detector.result.connect(self.on_detections)
            self.detector.error.connect(self.on_detect_error)
            self.detector.start()
        self.detector.request(frame.idx, frame.bgr)

    def on_detections(self, idx, per_model):
        if self.sender() is not self.detector:
            return
        if len(self.detections_cache) > 500:
            self.detections_cache.clear()
        self.detections_cache[idx] = per_model
        if idx == self.current_frame and self.actions_.detect.isChecked() and not self.is_playing:
            self.show_detections(per_model)

    def show_detections(self, per_model, cfg=None):
        c = cfg or self.model_cfg
        boxes = []
        multi = len(per_model) > 1
        for m, dets in enumerate(per_model):
            col = QColor(model_hex(m))
            for d in dets:
                uncertain = c["unc_low"] <= d.conf < c["unc_high"]
                if d.conf < c["conf"] and not uncertain:
                    continue
                label = f"{d.model + ': ' if multi else ''}{d.name} {d.conf:.2f}"
                boxes.append((d.x1, d.y1, d.x2, d.y2, label, col, uncertain))
        self.canvas.set_detections(boxes)
        n_unc = sum(1 for b in boxes if b[6])
        self.pixel_label.setText(f"{len(boxes)} detection(s)" + (f", {n_unc} uncertain (dashed)" if n_unc else ""))

    def on_detect_error(self, msg):
        self.actions_.detect.setChecked(False)
        self.stop_detector()
        QMessageBox.warning(self, "Models", f"Could not run the models:\n{msg}")

    # =============================================================== batch
    def open_batch_queue(self, add_current=True):
        if self.is_playing:
            self.pause()
        if self.batch_dialog is None:
            self.batch_dialog = BatchQueueDialog(
                self, lambda: self.output_dir,
                sharpness_getter=lambda: self.current_sharpness if self.current is not None else None,
                model_cfg_getter=lambda: self.model_cfg, open_models=self.open_models,
                open_gallery=self.open_gallery, output_cfg_getter=lambda: self.out_cfg,
                output_cfg_setter=self.set_output_cfg, open_output=self.open_output_settings)
            self.batch_dialog.sharp_sb.valueChanged.connect(self.strip.set_blur_threshold)
        self.batch_dialog.refresh_output_summary()
        self.batch_dialog.refresh_output_label()
        if add_current and self.engine and not self.batch_dialog.is_running():
            tf = lambda f: format_time(self.engine.time_of(f)) if f is not None else ""
            self.batch_dialog.add_job(self.video_path, tf(self.in_frame), tf(self.out_frame), self.roi)
        self.batch_dialog.show()
        self.batch_dialog.raise_()
        self.batch_dialog.activateWindow()

    # ================================================================ save
    def save_frame(self, full=False):
        if self.current is None:
            return
        if self.current.preview:
            self.flash_status("Showing a keyframe preview - release the seekbar to load the exact frame first")
            return
        if self.current.bgr is None:          # real-time playback frame: pause, save the exact frame
            self._save_after_exact = full
            self.pause()
            return
        from AIO_QC.tools.frame_extractor.core.naming import Namer
        from AIO_QC.tools.frame_extractor.core.imageio import imwrite
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        save_dir = os.path.join(self.output_dir, video_name)
        img, kind, roi = self.current.bgr, "frame", None
        if self.roi and not full:
            x1, y1, x2, y2 = self.roi
            img, kind, roi = img[y1:y2, x1:x2], "roi", self.roi
        c = self.out_cfg
        tpl = c["template"]
        if kind == "roi" and "{roi" not in tpl and "{kind" not in tpl:
            tpl += "_roi"                       # never overwrite the full frame with its crop
        namer = Namer(tpl, c["on_exists"], self.video_path, self.frame_rate, self.engine.time_of)
        save_path, rel = namer.path(save_dir, self.current.idx, c["ext"], img, kind, "", roi)
        if save_path is None:
            self.flash_status(f"{rel}{c['ext']} exists — skipped (Output ▸ If the file exists)")
            self.osd("Exists — skipped")
            return
        try:
            imwrite(save_path, img, c["ext"], c["quality"])
            from AIO_QC.tools.frame_extractor.core import frame_sources
            frame_sources.record(save_path, self.video_path, self.current.idx, self.engine.time_of(self.current.idx))
        except Exception as e:
            logger.error(f"Save failed: {e}")
            self.flash_status(f"Save failed: {e}")
            return
        logger.info(f"Frame saved to {save_path}")
        self.flash_status(f"Frame saved to {save_path}")
        self.osd(f"Saved {os.path.basename(save_path)}")

    # ================================================================ misc
    def toggle_toolbar(self, checked):
        self.toolbar.setVisible(checked)
        self.update_status()

    def sync_toggle_toolbar(self, checked):
        self.toggleToolbar.setChecked(checked)

    # ====================================================== commands & keys
    def _registry(self):
        """(id, QAction, category) for every named command, from the menus."""
        out, seen = [], set()
        for title, menu in (("File", self.menus.file), ("Playback", self.menus.edit), ("Marks", self.menus.marks),
                            ("View", self.menus.view), ("Tools", self.menus.tools),
                            ("Settings", self.menus.settings), ("Help", self.menus.help)):
            stack = [(title, menu)]
            while stack:
                cat, m = stack.pop()
                for act in m.actions():
                    if act.menu():
                        stack.append((f"{cat} › {act.menu().title()}", act.menu()))
                    elif act.objectName() and act.objectName() not in seen and not act.isSeparator():
                        seen.add(act.objectName())
                        out.append((act.objectName(), act, cat))
        return out

    def _init_shortcuts(self):
        for k, act in vars(self.actions_).items():
            act.setObjectName(k)
        self.play_pause.setObjectName("play_pause")
        self.toggleToolbar.setObjectName("toggle_toolbar")
        self.dock.toggleViewAction().setObjectName("toggle_marks")
        self._default_shortcuts = {aid: act.shortcut().toString() for aid, act, _ in self._registry()}
        self.settings.beginGroup("shortcuts")
        custom = {k: self.settings.value(k, "") for k in self.settings.childKeys()}
        self.settings.endGroup()
        self._apply_shortcuts(custom)

    def _apply_shortcuts(self, mapping):
        for aid, act, _ in self._registry():
            if aid in mapping:
                act.setShortcut(QKeySequence(mapping[aid] or ""))

    def show_shortcuts(self):
        d = ShortcutsDialog(self, self._registry(), self._default_shortcuts)
        d.reference_requested.connect(lambda: self.open_help("shortcuts"))
        if d.exec():
            new = d.result_map()
            self._apply_shortcuts(new)
            self._refresh_help()
            self.settings.beginGroup("shortcuts")
            self.settings.remove("")
            for aid, seq in new.items():
                if seq != self._default_shortcuts.get(aid, ""):
                    self.settings.setValue(aid, seq)
            self.settings.endGroup()
            self.flash_status("Shortcuts saved")

    def open_palette(self):
        CommandPalette(self, self._registry()).exec()

    def refresh_theme(self):
        """AIO QC's palette changed (light <-> dark): repaint what draws its own colours."""
        self.strip.update()
        self._refresh_help()

    def _refresh_help(self):
        """The Help Center is shared by every mode and may have been closed and deleted: never keep a wrapper."""
        from AIO_QC.widgets.help_center_bridge import live_help
        w = live_help()
        if w is not None:
            w.refresh()

    def changeEvent(self, ev):
        from PyQt6.QtCore import QEvent
        if ev.type() == QEvent.Type.PaletteChange:
            self.refresh_theme()
        super().changeEvent(ev)

    # ============================================================ help
    def open_help(self, page=None):
        """F1: AIO QC's one Help Center (this mode's chapters, Testing, the other modes, all shortcuts)."""
        from AIO_QC.widgets.help_center_bridge import open_help
        open_help(self, page or "video")

    def _shortcut_entries(self):
        """[(category, command, [key sequences])] as configured right now (for the shortcut reference)."""
        out = []
        for _aid, act, cat in self._registry():
            name = act.text().replace("\n", " ").replace("&&", "\0").replace("&", "").replace("\0", "&").rstrip("…")
            out.append((cat, name, [k.toString() for k in act.shortcuts()]))
        return out

    def show_get_started(self):
        from AIO_QC.tools.frame_extractor.widgets.get_started import GetStarted
        d = GetStarted(self)
        d.action.connect(self._get_started_action)
        d.exec()

    def _get_started_action(self, key):
        {"open": lambda: self.load_video(), "batch": lambda: self.open_batch_queue(), "qc": lambda: self.open_output_in_qc(),
         "guide": lambda: self.open_help()}.get(key, lambda: None)()

    def _maybe_get_started(self):
        from AIO_QC.tools.frame_extractor.widgets.get_started import should_show_at_startup
        reviewing = self.review is not None and self.review.active
        if should_show_at_startup() and not self._prev_crash and self.isVisible() and not reviewing:
            self.show_get_started()

    # ============================================================ search, export
    def _ffmpeg(self):
        from AIO_QC.tools.frame_extractor.core.preview_pipe import find_ffmpeg
        return find_ffmpeg(self.settings.value("playback/ffmpeg", ""))

    def open_output_in_qc(self, folder=None):
        """Frame Extractor -> Image QC: this video's output folder (markers, batch extraction, saved frames)."""
        if folder is None and self.video_path:
            cand = os.path.join(self.output_dir, os.path.splitext(os.path.basename(self.video_path))[0])
            folder = cand if os.path.isdir(cand) else None
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Folder with extracted frames", self.output_dir)
            if not folder:
                return
        shell = getattr(QApplication.instance(), "shell", None)
        if shell is None:
            QMessageBox.information(self, __appname__, "Image QC is available when AIO QC is started normally.")
            return
        shell.open_in_qc(folder)

    def _testing(self):
        """Image QC's Testing window if it exists (its models and colours are reused for the annotated export)."""
        shell = getattr(QApplication.instance(), "shell", None)
        img = getattr(shell, "image", None) if shell is not None else None
        return getattr(img, "testing", None) if img is not None else None

    def open_at(self, path, frame=None, seconds=None):
        """Show a video at a frame (or time) - loads it first when it is not the current one."""
        if not path:
            return
        if self.video_path and self.engine and os.path.normcase(os.path.abspath(path)) == \
                os.path.normcase(os.path.abspath(self.video_path)):
            f = frame if frame is not None else self.engine.index_of(seconds or 0.0)
            self.goto_frame(max(0, min(self.frame_count - 1, int(f))))
        elif os.path.isfile(path):
            self._goto_after_open = (path, frame, seconds)
            self.load_video(path)
        else:
            self.flash_status(f"Not found: {path}", 4000)
            return
        self.raise_()
        self.activateWindow()

    def _search_context(self):
        fr = self.current if self.current is not None and not self.current.preview else None
        return dict(video=self.video_path, frame=fr.bgr if fr is not None else None,
                    time=self.engine.time_of(fr.idx) if (fr is not None and self.engine) else None,
                    roi=list(self.roi) if self.roi else None, env_python=self.model_cfg.get("env_python", ""),
                    ffmpeg=self._ffmpeg())

    def _marker_at(self, video, seconds):
        if not (self.video_path and self.engine and os.path.abspath(video) == os.path.abspath(self.video_path)):
            self.open_at(video, seconds=seconds)
            self.flash_status("Opened the video - press M to drop the marker", 5000)
            return
        f = max(0, min(self.frame_count - 1, int(self.engine.index_of(seconds))))
        if f not in self.markers:
            self.add_marker(f)
        self.goto_frame(f)
        self.osd(f"Marker #{f}")

    def export_clip(self):
        if not self.engine:
            return
        ff = self._ffmpeg()
        if not ff:
            QMessageBox.warning(self, __appname__, "ffmpeg was not found (Settings → Playback → ffmpeg location).")
            return
        from AIO_QC.tools.frame_extractor.widgets.export_dialogs import ClipExportDialog
        dur = self.engine.duration()
        a = self.engine.time_of(self.in_frame) if self.in_frame is not None else self.engine.time_of(self.current_frame)
        b = self.engine.time_of(self.out_frame) if self.out_frame is not None else dur
        if self.in_frame is None and self.out_frame is None:
            self.flash_status("Tip: set in / out points (I / O) to choose the range", 4000)
        if self.is_playing:
            self.pause()
        ClipExportDialog(self, self.video_path, a, b, dur, ff).exec()

    def export_annotated(self):
        if not self.engine:
            return
        ff = self._ffmpeg()
        if not ff:
            QMessageBox.warning(self, __appname__, "ffmpeg was not found (Settings → Playback → ffmpeg location).")
            return
        from AIO_QC.core.testing import pipeline as P
        specs, slots = [], None
        lab = self._testing()
        if lab is not None and lab.specs:
            specs, slots = list(lab.specs), list(lab.slots)
        else:
            specs = P.unique_labels([P.spec_from_cfg(m, self.model_cfg) for m in self.model_cfg.get("models", [])
                                     if os.path.isfile(m)])
        if not specs:
            QMessageBox.information(self, __appname__, "Add a model first (Settings → Models, or Testing).")
            return
        from AIO_QC.tools.frame_extractor.widgets.export_dialogs import AnnotatedExportDialog
        dur = self.engine.duration()
        a = self.engine.time_of(self.in_frame) if self.in_frame is not None else 0.0
        b = self.engine.time_of(self.out_frame) if self.out_frame is not None else dur
        if self.is_playing:
            self.pause()
        AnnotatedExportDialog(self, self.video_path, specs, ff, a, b, dur, conf=float(self.model_cfg.get("conf", 0.25)),
                              slots=slots).exec()

    def reset_layout(self):
        self.restoreState(self._default_state)
        self.toolbar.show()
        self.dock.show()

    # ============================================================ gallery
    def open_gallery(self, folder=None):
        if folder is None:
            cand = []
            if self.video_path:
                name = os.path.splitext(os.path.basename(self.video_path))[0]
                cand = [os.path.join(self.output_dir, name, "batch_extract"), os.path.join(self.output_dir, name)]
            folder = next((c for c in cand if os.path.isdir(c)), None)
            if folder is None:
                folder = QFileDialog.getExistingDirectory(self, "Folder with extracted frames", self.output_dir)
                if not folder:
                    return
        if self.gallery is None:
            self.gallery = ReviewGallery(None, folder)
            self.gallery.setWindowIcon(utils.newIcon("gallery"))
        else:
            self.gallery.set_folder(folder)
        self.gallery.show()
        self.gallery.raise_()
        self.gallery.activateWindow()

    # =========================================================== tracking
    def toggle_tracking(self, on):
        if not on:
            self.stop_tracking()
            self.osd("ROI tracking off")
            return
        if not self.roi or self.current is None or self.current.preview:
            self.actions_.track.setChecked(False)
            self.osd("Draw an ROI on the object first (R)")
            return
        self.tracker = TrackWorker()
        self.tracker.result.connect(self.on_track_result)
        self.tracker.error.connect(self.on_track_error)
        self.tracker.start()
        img, sc = self.frame_image()
        self.tracker.init(self.current.idx, img, self.roi, sc)
        self._track_last = self.current.idx
        self.osd("ROI tracking on: play or step forward")

    def track_frame(self, frame):
        last = self._track_last
        img, sc = self.frame_image(frame)
        if img is None:
            return
        if last is not None and 0 < frame.idx - last <= 60:
            self.tracker.update(frame.idx, img, sc)
        elif frame.idx != last and self.roi:
            self.tracker.init(frame.idx, img, self.roi, sc)   # jumped: restart from here
        self._track_last = frame.idx

    def on_track_result(self, idx, roi):
        if self.sender() is not self.tracker:
            return
        if roi is None:
            self.actions_.track.setChecked(False)
            self.stop_tracking()
            self.osd("Tracking lost - redraw the ROI and press T")
            return
        self.roi = roi
        self.canvas.set_roi(roi)
        self.update_roi_label()

    def on_track_error(self, msg):
        self.actions_.track.setChecked(False)
        self.stop_tracking(save=False)
        QMessageBox.warning(self, "ROI tracking", f"Tracker unavailable:\n{msg}\n\n"
                            "Install opencv-contrib-python for the CSRT/KCF trackers.")

    def stop_tracking(self, save=True):
        if self.tracker is not None:
            self.tracker.stop()
            self.tracker = None
            if save:
                self.save_session()
        self._track_last = None

    # ======================================================= keyframe index
    def on_keyframe_index(self, index):
        if self.sender() is self.timeline and self.engine:
            self.engine.set_keyframe_index(index)
            if self.engine.fast_scrub:
                logger.info(f"Fast keyframe scrubbing on ({len(index.keyframes)} keyframes)")

    def on_grid_changed(self, grid, unit):
        if self.sender() is self.timeline:
            self.strip.set_grid(grid, unit)


    def show_about(self):
        from AIO_QC.tools.frame_extractor.core import mpv_lib
        try:
            mpv = "available" if mpv_lib.load_library(self.settings.value("playback/libmpv", "")) is not None \
                else "not found (Settings ▸ Playback)"
        except Exception:
            mpv = "not found (Settings ▸ Playback)"
        QMessageBox.about(self, f"About {__appname__}",
                          f"<h3>{__appname__} {__version__}</h3>"
                          "<p>Video frame extraction for image annotation - AIO QC's video mode. Model testing is in "
                          "Image QC ▸ Models ▸ Testing.</p>"
                          f"<p>PyAV: {'available' if HAVE_PYAV else 'not installed (pip install av)'}<br>"
                          f"ffmpeg: {self._ffmpeg() or 'not found'}<br>libmpv: {mpv}</p>"
                          "<p>Help ▸ User's Guide (F1) · Keyboard shortcuts (Ctrl+/)</p>")

    def close_window(self):
        self.close()

    def closeEvent(self, ev):
        if self.batch_dialog and self.batch_dialog.is_running():
            self.batch_dialog.close()
            if self.batch_dialog.is_running():
                ev.ignore()
                return
        logger.info("Exiting...")
        if self.review is not None:
            self.review.shutdown()
        job = getattr(self, "_mpv_job", None)
        if job is not None and job.running:
            job.cancel()                      # its thread is a daemon: never holds the app open
        self._sentinel_clear()
        self.settings.setValue("ui/geometry", self.saveGeometry())
        self.settings.setValue("ui/state", self.saveState())
        if self.gallery:
            self.gallery.close()
        self.sdock.shutdown()
        self.stop_mpv()
        self.stop_engine()
        self.stop_timeline()
        self.stop_marker_worker()
        self.stop_detector()
        self.preview.close()
        # end every child process first (benchmarks, samplers, pipes): background threads waiting on them then
        # finish at once instead of holding up the exit
        from AIO_QC.core import procs
        procs.kill_all()
        deadline = time.monotonic() + 3.0
        for w in list(self._bg):
            w.wait(max(1, int((deadline - time.monotonic()) * 1000)))
        shutdown_models(self.model_cfg)     # env servers; temporary environment if "delete on exit"
        procs.kill_all()
        ev.accept()

    def eventFilter(self, source, ev):
        if ev.type() == QEvent.Type.Leave and (source == self.menubar or source == self.toolbar):
            self.update_status()
        return super().eventFilter(source, ev)
