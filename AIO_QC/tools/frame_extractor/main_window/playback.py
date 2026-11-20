# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Frame Extractor window — playback, seek bar, live readouts, ROI, marks.

A mixin of AIO_QC.tools.frame_extractor.app.MainWindow: the methods use the window's own attributes, set up in its __init__.
"""
import time
from loguru import logger
from PyQt6.QtWidgets import (
    QApplication, QMenu,
)
from PyQt6.QtCore import QTimer
from AIO_QC.tools.frame_extractor.core import session
from AIO_QC.tools.frame_extractor.core.metrics import sharpness
from AIO_QC.tools.frame_extractor.core.sources import format_time
from AIO_QC.tools.frame_extractor.widgets.video_engine import Frame
from AIO_QC.utils.qt_utils import later



class PlaybackMixin:

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
