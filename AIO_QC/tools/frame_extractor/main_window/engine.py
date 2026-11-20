# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Frame Extractor window — the libmpv engine, playback stats, recent files, crash sentinel, canvas, zoom, real-time playback.

A mixin of AIO_QC.tools.frame_extractor.app.MainWindow: the methods use the window's own attributes, set up in its __init__.
"""
import os, time
import numpy as np
from loguru import logger
from PyQt6.QtWidgets import (
    QMessageBox, QFileDialog,
)
from PyQt6.QtCore import QThread, QPointF
from AIO_QC.tools.frame_extractor import __version__
from AIO_QC.tools.frame_extractor import utils
from AIO_QC.tools.frame_extractor.widgets import (
    VideoCanvas,
)
from AIO_QC.tools.frame_extractor.widgets.video_engine import Frame
from AIO_QC.utils.qt_utils import later



class EngineMixin:

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
