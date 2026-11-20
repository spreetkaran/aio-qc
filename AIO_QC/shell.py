# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
One app, three modes. The shell owns the main windows — Image QC (annotated image datasets), the Frame Extractor
(videos -> frames for annotation) and Data Studio (tables) — and switches between them: the other window takes the
same place and size, the first one hides (keeping its state). Each window is a complete app of its own: menus,
panels, recent files, settings. A window is only built the first time its mode is used, so starting in one mode
never loads another mode's machinery (the Frame Extractor, PyAV, ONNX Runtime and pandas load on first use).

    Ctrl+Alt+M (next mode) or the switch in the menu-bar corner · File ▸ Switch to … · aio-qc --mode image|video|data
The app starts in Image QC (aio-qc --mode video|data, or a video / table file, starts elsewhere). Closing any mode window quits the app (every window gets to save first).
"""
from __future__ import annotations

import sys

from loguru import logger
from PyQt6.QtCore import QEvent, QObject, QTimer, Qt
from PyQt6.QtWidgets import QApplication

from AIO_QC.utils import app_settings
from AIO_QC.core.diag import ignored as _ignored

MODES = ("video", "image", "data")          # switch / Ctrl+Alt+M order; the app starts in Image QC
NAMES = {"video": "Frame Extractor", "image": "Image QC", "data": "Data Studio"}


class AppShell(QObject):
    def __init__(self, app: QApplication):
        super().__init__(app)
        self.app = app
        import atexit
        import weakref
        ref = weakref.ref(self)
        atexit.register(lambda: ref() is not None and ref()._teardown())   # before PyQt's own exit clean-up
        from AIO_QC.widgets import ui_style
        ui_style.install(app)                    # centred pressed buttons, one menu arrow (whole app)
        from AIO_QC.utils import gui_gc
        gui_gc.install(app)                      # cycles collected on the GUI thread only (never a worker's)
        self.st = app_settings()
        from AIO_QC.core import provenance
        if not self.st.value("app/provenance", True, type=bool):
            provenance.set_enabled(False)            # Settings ▸ General ▸ Record provenance
        self.image = None
        self.studio = None
        self.video = None
        self.mode: str | None = None
        self._closing = False
        from AIO_QC.widgets.window_frame import FrameHost
        self.host = FrameHost(self)              # 3.0.1: one window; the modes are its pages
        if sys.platform == "darwin":             # Cmd+M: minimise the window in front, as in every Mac app
            from PyQt6.QtGui import QKeySequence, QShortcut
            from AIO_QC.widgets import window_shelf
            self._min_shortcut = QShortcut(QKeySequence("Ctrl+M"), self.host)
            self._min_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            self._min_shortcut.activated.connect(window_shelf.minimise_active)

    # ------------------------------------------------------------------ windows
    def start_mode(self, requested: str | None) -> str:
        """Image QC unless another mode is asked for (--mode, or a video / table file on the command line)."""
        return requested if requested in MODES else "image"

    MODE_ICONS = {"image": "app", "video": "mode-video", "data": "studio-app"}

    def _teardown(self):
        """Interpreter exit: delete the window (and every widget in it) while Python is still fully alive. Left to
        PyQt's exit clean-up, thousands of widget wrappers are destroyed while it walks its own table of them, and
        Python code that runs during those deletions can make that walk read freed memory (a crash on exit that
        comes and goes with the number of objects)."""
        try:
            from PyQt6 import sip
            from PyQt6.QtCore import QCoreApplication, QEvent
            host = getattr(self, "host", None)
            if host is not None and not sip.isdeleted(host):
                host.hide()
                sip.delete(host)
            for w in (self.image, self.video, self.studio):
                if w is not None and not sip.isdeleted(w):
                    sip.delete(w)
            app = QCoreApplication.instance()
            if app is not None:
                app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        except Exception as _exc:
            _ignored(_exc)

    def window(self, mode: str):
        w = self._window(mode)
        if w not in self.host.pages:
            # each mode's own app icon, kept on the page: the window (and taskbar / dock) shows it whenever that
            # mode is in front — never left on the previous mode's icon
            from AIO_QC.utils import appIcon
            from AIO_QC.utils.qt_utils import raster_icon
            # multi-resolution PNG sets (icons/fx_*.png, studio_*.png): crisp in the taskbar, Alt+Tab and the dock
            w._mode_icon = appIcon() if mode == "image" else raster_icon({"video": "fx", "data": "studio"}[mode])
            if mode != "image":
                w.setWindowIcon(w._mode_icon)
            self.host.add(w)
        return w

    def _window(self, mode: str):
        if mode == "image":
            if self.image is None:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    from AIO_QC.app import MainWindow
                    self.image = MainWindow()
                finally:
                    QApplication.restoreOverrideCursor()
                self.image.installEventFilter(self)
                from AIO_QC.widgets import tasks
                tasks.install(self.image)
                if hasattr(self.image, "_sync_status_bar"):
                    self.image._sync_status_bar()          # a job on the welcome page brings the status bar back
            return self.image
        if mode == "video":
            if self.video is None:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    from AIO_QC.tools.frame_extractor.app import MainWindow as FrameExtractorWindow
                    self.video = FrameExtractorWindow()
                finally:
                    QApplication.restoreOverrideCursor()
                self._add_switch(self.video, "video")
                self.video.installEventFilter(self)
                from AIO_QC.widgets import tasks
                tasks.install(self.video)
            return self.video
        if self.studio is None:
            from AIO_QC.widgets.studio.window import StudioWindow
            self.studio = StudioWindow(self)
            self.studio.installEventFilter(self)
            from AIO_QC.widgets import tasks
            tasks.install(self.studio)
        return self.studio

    def _add_switch(self, win, mode):
        """The mode switch in the window's menu-bar corner, and File ▸ Switch to … (the Frame Extractor is a vendored
        app: the shell adds these instead of the window building them itself)."""
        from PyQt6.QtGui import QAction, QKeySequence
        from AIO_QC.widgets.mode_switch import ModeSwitch
        win.mode_switch = ModeSwitch(mode, win)
        win.mode_switch.switched.connect(self.switch)
        win.menuBar().setCornerWidget(win.mode_switch, Qt.Corner.TopRightCorner)
        file_menu = win.menuBar().actions()[0].menu() if win.menuBar().actions() else None
        if file_menu is not None:
            first = file_menu.actions()[0] if file_menu.actions() else None
            nxt = QAction("Next mode", win)
            nxt.setShortcut(QKeySequence("Ctrl+Alt+M"))
            nxt.setToolTip("Frame Extractor → Image QC → Data Studio (Ctrl+Alt+M)")
            nxt.triggered.connect(self.cycle)
            acts = [nxt] + [self._switch_action(win, m) for m in MODES if m != mode]
            for a in acts:
                file_menu.insertAction(first, a)
            file_menu.insertSeparator(first)
        from AIO_QC.widgets.settings_dialog import install_settings
        install_settings(win)                                   # keep the gear next to the new mode switch

    def _switch_action(self, win, mode):
        from PyQt6.QtGui import QAction
        from AIO_QC.utils import newIcon
        a = QAction(newIcon(f"mode-{mode}"), f"Switch to {NAMES[mode]}", win)
        a.triggered.connect(lambda _c=False, m=mode: self.switch(m))
        return a

    def open_testing(self, video=None, start=None, end=None):
        """Testing lives in Image QC's Models tab: its window opens on top of whichever mode is in front (the Image
        QC window is built if needed, without switching to it)."""
        img = self.window("image")
        return img.open_testing(video, start, end)

    def show_training(self, run=None):
        """Train Model (it lives in Image QC) on `run`, from whichever mode the chip was clicked in."""
        if self.mode != "image":
            self.switch("image")
        win = self.window("image")
        if win is not None and hasattr(win, "open_training"):
            win.open_training(run=run)

    def review_in_player(self, playlist):
        """Testing → Frame Extractor: hand over a video-review playlist. Testing does nothing else; the player owns
        it from here (analysis, playback, the next video, back to the video open before)."""
        from PyQt6.QtWidgets import QMessageBox
        problems = playlist.validate()
        if problems:
            QMessageBox.warning(self.window(self.mode) if self.mode else None, "Review in player",
                                "The review can't start:\n\n" + "\n".join(problems[:10]))
            return None
        self.switch("video")
        if self.video is None or self.mode != "video":
            return None                                     # the Frame Extractor can't run here (told by switch)
        return self.video.start_review(playlist)

    # ------------------------------------------------------------------ cross-mode actions
    def open_in_qc(self, folder) -> bool:
        """Frame Extractor -> Image QC: an extraction output folder (frames + any pre-labels) as the dataset."""
        from pathlib import Path
        from PyQt6.QtWidgets import QMessageBox
        from AIO_QC.core import fx_to_qc as X
        from AIO_QC.core.dataset import DatasetType
        parent = self.window(self.mode) if self.mode else None
        if not folder or not Path(folder).is_dir():
            return False
        info = X.inspect(folder)
        if info["kind"] == "empty":
            QMessageBox.information(parent, "Open in Image QC", f"No frames in\n{folder}")
            return False
        if info["kind"] == "cls_folders":                  # a classifier sorted the frames into class folders
            self.switch("image") if self.mode != "image" else None
            img = self.window("image")
            img.open_dataset(Path(info.get("root") or folder), DatasetType.CLS_FOLDERS)
            return getattr(img, "session", None) is not None
        if info["kind"] not in ("labelme", "cls_labelme") or info["missing"]:
            what = {"yolo": f"{info['images']:,} frames with YOLO pre-labels: LabelMe files are added next to them "
                            "(boxes converted with classes.txt; the .txt files stay).",
                    "images": f"{info['images']:,} frames without labels: empty LabelMe files are added next to "
                              "them, ready to label."}.get(info["kind"],
                                                          f"{info['missing']:,} of {info['images']:,} frames have "
                                                          "no LabelMe file yet: empty ones are added so every "
                                                          "frame is in the dataset.")
            r = QMessageBox.question(parent, "Open in Image QC", f"{folder}\n\n{what}\n\nAdd them and open the "
                                     "folder in Image QC?", QMessageBox.StandardButton.Yes |
                                     QMessageBox.StandardButton.Cancel)
            if r != QMessageBox.StandardButton.Yes:
                return False
            n = X.prepare_labelme(folder)
            logger.info(f"[open in QC] {n} LabelMe file(s) added in {folder}")
        self.switch("image") if self.mode != "image" else None
        img = self.window("image")
        img.open_dataset(Path(folder), DatasetType.CLS_LABELME if info["kind"] == "cls_labelme"
                         else DatasetType.LABELME)                 # a classifier's flags: a classification dataset
        return getattr(img, "session", None) is not None

    def show_in_frame_extractor(self, video, frame=None, seconds=None):
        """Image QC -> Frame Extractor: the video an image came from, at its frame (explicit user action only;
        never from Testing)."""
        self.switch("video") if self.mode != "video" else None
        v = self.window("video")
        v.open_at(video, frame, seconds)
        return v

    def cycle(self):
        """Ctrl+Alt+M: the next mode."""
        cur = self.mode if self.mode in MODES else "image"
        self.switch(MODES[(MODES.index(cur) + 1) % len(MODES)])

    def ready(self, mode: str) -> str | None:
        return self.data_ready() if mode == "data" else self.video_ready() if mode == "video" else None

    @staticmethod
    def video_ready() -> str | None:
        """The Frame Extractor needs OpenCV (always installed with AIO QC); PyAV / ffmpeg / ONNX Runtime are optional
        and their absence only switches individual features off."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            return "The Frame Extractor needs OpenCV:  pip install opencv-contrib-python-headless"
        return None

    def show(self, mode: str):
        if mode not in MODES:
            mode = "image"
        if self.ready(mode):
            logger.warning(self.ready(mode))
            mode = "image"
        w = self.window(mode)
        self.mode = mode
        self.st.setValue("shell/mode", mode)
        self.host.set_current(w, animate=False)
        if not self.host.isVisible():
            self.host.show_restored()
        self.host.raise_()
        self.host.activateWindow()
        sw = getattr(w, "mode_switch", None)
        if sw is not None:
            sw.set_current(mode, animate=False)
        return w

    @staticmethod
    def data_ready() -> str | None:
        """Data Studio needs pandas (>= 2.0). -> a message when it is missing, else None."""
        try:
            import pandas as pd
        except ImportError:
            return "Data Studio needs pandas:  pip install \"aio-qc[data]\"  (or: pip install pandas pyarrow openpyxl)"
        if int(pd.__version__.split(".")[0]) < 2:
            return f"Data Studio needs pandas 2.0 or newer (found {pd.__version__}):  pip install -U pandas"
        return None

    def switch(self, mode: str):
        if mode == self.mode or mode not in MODES:
            return
        if self.ready(mode):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self.window(self.mode) if self.mode else None, NAMES[mode], self.ready(mode))
            return
        old_mode = self.mode
        old = self.window(self.mode) if self.mode else None
        first = not self._built(mode)
        new = self._load_mode(mode, f"Opening {NAMES[mode]}…") if first else self.window(mode)
        self.host.set_current(new, animate=not first)  # the same window: the page cross-fades, nothing re-opens
        if first:
            self.host.hide_loading()                    # the ring fades away over the new page
        if not self.host.isVisible():
            self.host.show_restored()
        self.host.raise_()
        self.host.activateWindow()
        new.setFocus()
        if old is not None and old is not new:
            rel = getattr(old, "release_memory", None)      # in the background: keep only what it shows
            if rel is not None:
                try:
                    rel()
                except Exception as _exc:
                    _ignored(_exc)
        self.mode = mode
        self.st.setValue("shell/mode", mode)
        sw = getattr(new, "mode_switch", None)
        if sw is not None:
            sw.set_current(mode, from_key=old_mode)       # its pill glides over from the mode before
        if mode == "data" and hasattr(new, "_show_welcome") and not new.ws.tables:
            new._show_welcome()
        if old is not None and old is self.video and self.video.is_playing:
            self.video.pause()                            # a hidden Frame Extractor never keeps playing
        if mode == "video" and first:
            self._resume_video(new)                       # unloaded earlier: back on the same video and frame
        self._arm_unload()
        logger.info(f"switched to {NAMES[mode]}")

    # ------------------------------------------------------------------ bridge
    def image_session(self):
        return getattr(self.image, "session", None) if self.image is not None else None

    def annotations_to_studio(self):
        """The open image dataset as a table (one row per shape) in Data Studio."""
        session = self.image_session()
        if session is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self.window(self.mode or "data"), "Annotations → Data Studio",
                                    "Open an image dataset in Image QC first.")
            return
        self.switch("data")
        studio = self.window("data")
        from AIO_QC.core.tabular import annotations as AN
        from AIO_QC.widgets.studio.window import Job
        samples, adapter, review = list(session.samples), session.adapter, session.review
        name = f"{session.root.name}_annotations"
        studio.status_lbl.setText(f"Reading the annotations of {len(samples):,} file(s)…")
        job = Job(lambda: AN.build(samples, adapter, review), studio)
        job.done.connect(lambda df: studio.add_frame(df, name, f"annotations of {session.root}"))
        job.failed.connect(lambda m, _e: studio.status_lbl.setText(f"⚠ {m}"))
        job.finished.connect(lambda j=job: studio.jobs.remove(j) if j in studio.jobs else None)
        studio.jobs.append(job)
        job.start()

    # ------------------------------------------------------------------ quitting
    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Close and obj in (self.image, self.studio, self.video):
            if getattr(self, "_unloading", False):
                return False
            if not self._closing:
                ev.ignore()
                QTimer.singleShot(0, self.quit_all)
                return True
            QTimer.singleShot(0, self._hide_host_if_empty)
        return False

    def _hide_host_if_empty(self):
        try:
            if not any(p.isVisible() for p in self.host.pages):
                self.host.hide()
        except RuntimeError:
            pass

    def quit_all(self):
        """Closing either window quits the app: Data Studio may ask to save, Image QC may keep an editor session."""
        if self._closing:
            return
        self._closing = True
        if self.studio is not None and not self.studio.maybe_save():
            self._closing = False
            return
        if self.video is not None and not self.video.close():          # e.g. a batch extraction is still running
            self._closing = False
            if self.mode != "video":
                self.switch("video")
            return
        if self.image is not None and not self.image.close():          # e.g. an editor session is still open
            self._closing = False
            if self.mode != "image":
                self.switch("image")
            return
        if self.studio is not None:
            self.studio.shutdown()
            self.studio.close()
        self.host.save_geometry()
        self.host.close()
        self.app.quit()

    def request_quit(self, reason: str):
        """Ctrl+C / SIGTERM: no questions; everything saves its state."""
        self._closing = True
        if self.video is not None:
            if self.video.batch_dialog is not None and self.video.batch_dialog.is_running():
                self.video.batch_dialog.cancel()
            self.video.close()
        if self.image is not None:
            self.image.request_quit(reason)
        if self.studio is not None:
            self.studio.shutdown()
            self.studio.close()
        self.host.save_geometry()
        self.host.close()
        self.app.quit()

    def log_errors(self, lines):
        """Unhandled errors (AIO QC's one excepthook): the error log of Image QC, also when another mode is in front
        (that window flashes a short note in its status bar)."""
        cur = self.window(self.mode) if self.mode in ("video", "data") and self._built(self.mode) else None
        if cur is not None:
            try:
                note = f"⚠ {lines[-1]} — details in the log"
                if hasattr(cur, "flash_status"):     # Frame Extractor: in its left slot, so a running job's place
                    cur.flash_status(note, 10000)    # and everything to its right stay as they are
                else:
                    cur.statusBar().showMessage(note, 10000)
            except Exception as _exc:
                _ignored(_exc)
        if self.image is not None:
            self.image.log_errors(lines)
        else:
            for line in lines:
                logger.error(line)

    # ------------------------------------------------------------------ preparing the other modes
    MODULES = {"video": "AIO_QC.tools.frame_extractor.app", "data": "AIO_QC.widgets.studio.window",
               "image": "AIO_QC.app"}

    @classmethod
    def _import_plan(cls, mode) -> list[str]:
        """The mode's module and what it imports at its top level, in order — imported one per event-loop turn,
        so a loading ring keeps turning (and nothing here runs off the UI thread)."""
        import ast
        import importlib.util
        import sys
        name = cls.MODULES[mode]
        if name in sys.modules:
            return []
        plan = []
        try:
            spec = importlib.util.find_spec(name)
            tree = ast.parse(open(spec.origin, encoding="utf-8").read())
            pkg = name.rsplit(".", 1)[0]
            for node in tree.body:
                if isinstance(node, ast.Import):
                    plan += [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module is not None or \
                        isinstance(node, ast.ImportFrom) and node.level:
                    if node.level:
                        base = pkg.rsplit(".", node.level - 1)[0] if node.level > 1 else pkg
                        mod = f"{base}.{node.module}" if node.module else base
                    else:
                        mod = node.module
                    plan.append(mod)
        except Exception as _exc:
            _ignored(_exc)
            plan = []
        return [m for m in dict.fromkeys(plan + [name]) if m not in sys.modules]

    def _load_mode(self, mode, text=None):
        """Build a mode's window with the big loading ring showing (imports step by step, then the window)."""
        if self._built(mode):
            return self.window(mode)
        import importlib
        from PyQt6.QtCore import QEventLoop
        host = self.host
        if text and host.isVisible():
            host.show_loading(text)
        quiet = QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        self.app.processEvents(quiet)
        plan = self._import_plan(mode)
        if plan:
            # The code is read on a helper thread (only module imports — no window or Qt object is made there)
            # while this thread keeps the loading ring turning at the display's rate; Python's import lock makes
            # the window's own imports below wait for anything still in progress.
            import threading
            done = threading.Event()

            def run():
                for m in plan:
                    try:
                        importlib.import_module(m)
                    except Exception as e:               # the real import below reports it properly
                        logger.debug(f"[load {mode}] {m}: {e}")
                        break
                done.set()
            threading.Thread(target=run, name=f"aioqc-load-{mode}", daemon=True).start()
            import time
            t_end = time.monotonic() + 60
            while not done.is_set() and time.monotonic() < t_end:
                self.app.processEvents(quiet, 5)
                done.wait(0.004)
        try:
            return self.window(mode)
        finally:
            self.app.processEvents(quiet)

    # ------------------------------------------------------------------ unloading an idle Frame Extractor
    def _unload_minutes(self) -> int:
        try:
            return max(0, int(self.st.value("shell/fx_unload_min", 15)))
        except (TypeError, ValueError):
            return 15

    def _arm_unload(self):
        """Settings ▸ General ▸ Unload the Frame Extractor when unused: N minutes after you leave it (0 = never)."""
        t = getattr(self, "_unload_timer", None)
        if t is None:
            t = self._unload_timer = QTimer(self, singleShot=True, timeout=self._unload_video)
        t.stop()
        n = self._unload_minutes()
        if n > 0 and self.video is not None and self.mode != "video":
            t.start(n * 60_000)

    def _video_busy(self) -> str | None:
        v = self.video
        if v is None:
            return None
        if getattr(v, "batch_dialog", None) is not None and v.batch_dialog.is_running():
            return "a batch extraction is running"
        if getattr(v, "review", None) is not None and getattr(v.review, "active", False):
            return "a video review is open"
        job = getattr(v, "_mpv_job", None)
        if job is not None and getattr(job, "running", False):
            return "libmpv is being downloaded"
        from AIO_QC.widgets import tasks
        c = tasks.center()
        if c is not None and c.tasks:
            return "a background task is running"          # (conservative: could be one of its jobs)
        from PyQt6.QtWidgets import QWidget
        if any(w.isWindow() and w.isVisible() for w in v.findChildren(QWidget)):
            return "one of its windows is open"
        return None

    def _unload_video(self):
        """Close the Frame Extractor (its state saved, as when the app quits) to give its memory back; the next
        switch opens it again — on the same video and frame."""
        v = self.video
        if v is None or self.mode == "video":
            return
        busy = self._video_busy()
        if busy:
            logger.info(f"[unload] Frame Extractor kept: {busy}")
            self._arm_unload()                            # try again later
            return
        self._fx_resume = (v.video_path, int(getattr(v, "current_frame", 0) or 0)) if v.video_path else None
        v._unloading = True                               # its own closeEvent: no app-wide process kill
        self._unloading = True
        try:
            ok = v.close()
        finally:
            self._unloading = False
        if not ok:
            v._unloading = False
            self._arm_unload()
            return
        self.host.remove(v)
        v.deleteLater()
        self.video = None
        import gc
        gc.collect()
        logger.info("[unload] Frame Extractor unloaded after being unused "
                    f"{self._unload_minutes()} min (opens again on the next switch)")

    def _resume_video(self, win):
        r = getattr(self, "_fx_resume", None)
        self._fx_resume = None
        if r and r[0]:
            try:
                win.open_at(r[0], r[1])
            except Exception as e:
                logger.debug(f"[unload] could not reopen {r[0]}: {e}")

    def _built(self, mode):
        return {"image": self.image, "video": self.video, "data": self.studio}.get(mode) is not None
