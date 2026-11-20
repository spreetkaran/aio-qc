# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Created on Sun Sep 28 12:39:36 2025

@author: singh
"""

import os, time
from loguru import logger
from PyQt6.QtWidgets import (
    QMainWindow, QApplication,
)
from PyQt6.QtCore import Qt, QTimer, QEvent
from types import SimpleNamespace
from AIO_QC.tools.frame_extractor import __appname__
from AIO_QC.tools.frame_extractor import utils
from AIO_QC.tools.frame_extractor.core.common import VIDEO_EXTS
from AIO_QC.tools.frame_extractor.widgets.models_dialog import (
    load_model_cfg,
    shutdown_models,
)
from AIO_QC.tools.frame_extractor.widgets.output_dialog import (
    load_output_cfg,
)
from AIO_QC.utils.fx_settings import fx_settings
from AIO_QC.utils.qt_utils import later   # owner-bound timer: never fires after `self` is gone


# left out of the menus by View ▸ Simple Menus (Ctrl+K still finds them, their shortcuts still work)


from AIO_QC.tools.frame_extractor.app_common import (  # noqa: F401  (re-exported)
    FX_ADVANCED, MOUSE_HELP, _ElidedLabel)
from AIO_QC.tools.frame_extractor.main_window.ui import UIMixin
from AIO_QC.tools.frame_extractor.main_window.files import FilesMixin
from AIO_QC.tools.frame_extractor.main_window.playback import PlaybackMixin
from AIO_QC.tools.frame_extractor.main_window.engine import EngineMixin
from AIO_QC.tools.frame_extractor.main_window.tools import ToolsMixin


class MainWindow(UIMixin, FilesMixin, PlaybackMixin, EngineMixin, ToolsMixin, QMainWindow):
    SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]
    CACHE_MB = 768        # RAM budget for decoded full-res frames

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{__appname__} — AIO QC App")
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
        self._sync_start_page()                         # nothing open yet: what to do first
        from AIO_QC.widgets import ui_level
        lvl = ui_level.install(self, advanced=FX_ADVANCED, reason=self._disabled_reason)
        self.menus.view.addSeparator()
        self.menus.view.addAction(lvl.action)
        lvl._apply(ui_level.simple())

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

    def event(self, e):
        # A button's or menu item's hint (status tip) goes in the left slot like any short note. Left to Qt, it would
        # hide the whole slot (a running job's place too) while the mouse is over the button.
        from PyQt6.QtCore import QEvent
        if e.type() == QEvent.Type.StatusTip and hasattr(self, "msg_label"):
            tip = e.tip()
            if tip:
                if not self.default_status_timer.isActive() or getattr(self, "_tip_shown", False):
                    self._tip_shown = True
                    self.status_priority = True
                    self._set_msg(tip)
                    self._sync_left_slot()
            elif getattr(self, "_tip_shown", False):
                self.reset_status_priority()
            return True
        return super().event(e)

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


    # ================================================================ recent files
    RECENT_MAX = 12

    def changeEvent(self, ev):
        from PyQt6.QtCore import QEvent
        if ev.type() == QEvent.Type.PaletteChange:
            self.refresh_theme()
        super().changeEvent(ev)

    keymap_canvas_keys = False                     # the key map's "On the image" group is Image QC's
    keymap_extra = {"Keyboard": [("Play / pause", "Space"), ("Step one frame", "← →"), ("Mark a frame", "M"),
                                 ("Save the frame", "Ctrl+S"), ("Find a command", "Ctrl+K"),
                                 ("These shortcuts", "? · Ctrl+/")]}
    keymap_extra.update({g: [(what, how) for how, what in rows] for g, rows in MOUSE_HELP})   # mouse & timeline

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
        unloading = getattr(self, "_unloading", False)     # only this mode closes: other modes' processes stay
        if not unloading:
            procs.kill_all()
        deadline = time.monotonic() + 3.0
        for w in list(self._bg):
            w.wait(max(1, int((deadline - time.monotonic()) * 1000)))
        from AIO_QC.widgets.lifecycle import stop_threads
        stop_threads(self, "Frame Extractor closing", 1000)   # stragglers (dialogs' jobs too) detached, not destroyed
        if not unloading:
            shutdown_models(self.model_cfg)     # env servers; temporary environment if "delete on exit"
            procs.kill_all()
        ev.accept()

    def eventFilter(self, source, ev):
        if ev.type() == QEvent.Type.Leave and (source == self.menubar or source == self.toolbar):
            self.update_status()
        return super().eventFilter(source, ev)
