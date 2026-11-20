# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Frame Extractor window — settings, video review and building the window.

A mixin of AIO_QC.tools.frame_extractor.app.MainWindow: the methods use the window's own attributes, set up in its __init__.
"""
import functools
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QComboBox,
    QStatusBar, QToolButton, QMenu, QStackedWidget,
)
from PyQt6.QtCore import Qt, QSize, QKeyCombination
from PyQt6.QtGui import QActionGroup, QKeySequence
from types import SimpleNamespace
from AIO_QC.tools.frame_extractor import utils
from AIO_QC.tools.frame_extractor.core.sources import format_time, HAVE_PYAV
from AIO_QC.tools.frame_extractor.widgets import (
    SeekSlider,
    BookmarksDock,
)
from AIO_QC.tools.frame_extractor.widgets.timeline import TimelineStrip, ThumbPopup
from AIO_QC.tools.frame_extractor.widgets.markers_dock import MarkersDock
from AIO_QC.tools.frame_extractor.widgets.timecode_display import TimecodeDisplay
from AIO_QC.tools.frame_extractor.widgets.output_dialog import (
    OutputQuickBar,
)

from AIO_QC.tools.frame_extractor.app_common import (_ElidedLabel)


class UIMixin:

    # ================================================================== video review (from Testing)
    def start_review(self, playlist):
        """Play a ReviewPlaylist (videos + models) with detections - Testing only hands it over."""
        from AIO_QC.tools.frame_extractor.widgets.review_dock import ReviewDock
        tour = getattr(self, "_coach", None)
        if tour is not None:
            tour.finish()                                   # the first-run tour must not cover the review
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
        a.autoannotate = A("Auto-\nAnnotate", lambda: self.open_autoannotate(), "Ctrl+Shift+U", "autoannotate",
                           "Auto-Annotate: a folder of videos → unique frames → LabelMe files with your model. Every "
                           "step is optional — frames only, or annotations on frames you already have")
        a.guide = A("User Guide", lambda: self.open_help(), "F1", "help",
                    "The complete guide: every feature, step by step, searchable")
        a.help = A("Customize keyboard shortcuts…", self.show_shortcuts, None, "command",
                   "Change any shortcut; conflicts are highlighted")
        a.palette = A("Command palette…", self.open_palette, "Ctrl+K", "command", "Search and run any command")
        a.keymap = A("Keyboard Shortcuts", self.toggle_keymap, "?", "command",
                     "Every key and mouse gesture, over the window — type to filter, click to run")
        a.keymap.setShortcuts([QKeySequence("?"), QKeySequence("Ctrl+/")])   # Ctrl+/ as before
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
        a.about = A("About Frame Extractor", self.show_about)
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
                                           a.open_in_qc, None, a.export_clip, a.export_annotated, None, a.reset, a.quit))
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
        a.tour = A("Quick Tour", lambda: self.start_tour(force=True), None, "icon",
                   "A short guided tour of the window: where everything is")
        a.sample = A("Open a Sample Video", self.open_sample_video, None, "open",
                     "A 12-second street video made on this computer — to learn without your own footage")
        # Help: one of each — the tour, the guide (its Shortcuts page is the printable list), the live shortcuts
        utils.addActions(self.menus.help, (a.tour, a.guide, a.keymap, None, a.sample, None, a.logs, a.about))
        utils.addActions(self.menus.tools, (a.palette, None, a.batch, a.autoannotate, a.gallery, None, a.detect, a.track,
                                            None, a.models))

        # Toolbar: file/annotation tools. Transport controls live under the video.
        utils.addActions(self.toolbar, (
            a.load, a.outdir, None, a.save, a.save_full, None,
            a.roi, a.clear_roi, a.track, None, a.mark_in, a.mark_out, a.bookmark, a.marker, None, a.detect, None,
            a.batch, a.autoannotate, a.gallery, None, a.reset, None,
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

        # Transport: play controls + time readout on the left, step / speed / zoom on the right. On a narrow
        # window (a 1280 px laptop with the side panel open) the right group moves under the left one instead of
        # forcing the window wider than the screen.
        left_box, right_box = QWidget(), QWidget()
        transport = QHBoxLayout(left_box)
        transport.setContentsMargins(0, 0, 0, 0)
        transport.setSpacing(2)
        tools_row = QHBoxLayout(right_box)
        tools_row.setContentsMargins(0, 0, 0, 0)
        tools_row.setSpacing(2)

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
        tools_row.addWidget(QLabel("Step: "))
        tools_row.addWidget(self.time_cb)
        tools_row.addSpacing(10)
        tools_row.addWidget(QLabel("Speed: "))
        tools_row.addWidget(self.speed_cb)
        tools_row.addSpacing(6)
        for act in (a.zoom_out, a.zoom_fit, a.zoom_actual, a.zoom_in):
            b = btn(act, 20)
            if act in (a.zoom_fit, a.zoom_actual):
                b.setStyleSheet("QToolButton:checked { background: palette(highlight); border-radius: 4px; }")
            tools_row.addWidget(b)
        from AIO_QC.utils.qt_utils import SplitRow
        self.transport_row = QWidget()
        self.transport_row.setObjectName("transportRow")
        SplitRow(left_box, right_box, self.transport_row)
        sp = self.transport_row.sizePolicy()
        sp.setHeightForWidth(True)
        self.transport_row.setSizePolicy(sp)
        layout.addWidget(self.transport_row)
        # the start page is a page of its own beside the player (never laid over it: a GPU canvas paints on top of
        # anything laid over it, so parts of the page showed through around the seek bar)
        self.central_stack = QStackedWidget()
        self.central_stack.addWidget(central)
        self.setCentralWidget(self.central_stack)

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
        # the left slot: the output-folder line ("Output Dir …" / "Paused" / short notes) — background jobs take its
        # place while they run (tasks.install calls task_slot), so nothing to the right ever moves
        self.msg_label = _ElidedLabel(self)
        self.msg_label.setObjectName("statusMessage")
        self.msg_label.setStyleSheet("padding: 0 6px;")
        from PyQt6.QtWidgets import QSizePolicy
        self.msg_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.msg_label.setMinimumWidth(40)
        self.status.addWidget(self.msg_label, 1)
        self._msg_full = ""
        self._task_ind = None
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
