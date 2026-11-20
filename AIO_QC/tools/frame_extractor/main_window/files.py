# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Frame Extractor window — status, state, loading videos and files.

A mixin of AIO_QC.tools.frame_extractor.app.MainWindow: the methods use the window's own attributes, set up in its __init__.
"""
import os
from loguru import logger
from PyQt6.QtWidgets import (
    QMessageBox, QApplication, QFileDialog,
)
from PyQt6.QtCore import Qt, QThread
from AIO_QC.tools.frame_extractor import __appname__
from AIO_QC.tools.frame_extractor import utils
from AIO_QC.tools.frame_extractor.core import session
from AIO_QC.tools.frame_extractor.widgets import (
    VideoLoadDialog,
    VideoEngine,
)
from AIO_QC.tools.frame_extractor.widgets.timeline import TimelineWorker
from AIO_QC.tools.frame_extractor.widgets.output_dialog import (
    OutputSettingsDialog,
    save_output_cfg,
    summary as output_summary,
)
from AIO_QC.utils.qt_utils import later



class FilesMixin:

    # ============================================================== status
    def task_slot(self, indicator):
        """tasks.install: background jobs go in the left slot, in place of the output-folder line."""
        self._task_ind = indicator
        self.status.insertWidget(0, indicator, 1)
        indicator.active_changed.connect(lambda _on: self._sync_left_slot())
        self._sync_left_slot()

    def _sync_left_slot(self):
        ind = self._task_ind
        jobs = ind is not None and ind.active()
        flashing = getattr(self, "status_priority", False)
        self.msg_label.setVisible(flashing or not jobs)          # a short note shows over the jobs for a moment
        if ind is not None:
            ind.setVisible(jobs and not flashing)

    def _set_msg(self, text):
        self._msg_full = text
        self.msg_label.setToolTip(text if len(text) > 60 else "")
        self._elide_msg()

    def _elide_msg(self):
        self.msg_label.set_full(self._msg_full)

    def status_text(self) -> str:
        """What the left slot says (the output-folder line or a short note)."""
        return self._msg_full

    def update_status(self):
        if getattr(self, 'status_priority', False):
            return
        if self.video_paused:
            self._set_msg("Paused")
        else:
            self._set_msg(f"Output Dir: {self.output_dir}")
        self._sync_left_slot()

    def flash_status(self, msg, ms=3000):
        self.status_priority = True
        self._set_msg(msg)
        self._sync_left_slot()
        self.default_status_timer.start(ms)

    def reset_status_priority(self):
        self.status_priority = False
        self._tip_shown = False
        self.update_status()
        self._sync_left_slot()

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
        self.setWindowTitle(f"{__appname__} — AIO QC App")
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
        self._sync_start_page()
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
        if getattr(self, "start_page", None) is not None:
            self.central_stack.setCurrentIndex(0)
        from AIO_QC.widgets import coach
        if not coach.done("video"):                  # the first video ever: a short tour of the window
            later(900, self, lambda: self.video_path and self.start_tour(force=False))
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
        self.setWindowTitle(f"{__appname__} — {name} — AIO QC App")
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
