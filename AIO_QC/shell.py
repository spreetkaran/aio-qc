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

from loguru import logger
from PyQt6.QtCore import QEvent, QObject, QTimer, Qt
from PyQt6.QtWidgets import QApplication

from AIO_QC.utils import app_settings

MODES = ("video", "image", "data")          # switch / Ctrl+Alt+M order; the app starts in Image QC
NAMES = {"video": "Frame Extractor", "image": "Image QC", "data": "Data Studio"}


class AppShell(QObject):
    def __init__(self, app: QApplication):
        super().__init__(app)
        self.app = app
        from AIO_QC.widgets import ui_style
        ui_style.install(app)                    # centred pressed buttons, one menu arrow (whole app)
        self.st = app_settings()
        self.image = None
        self.studio = None
        self.video = None
        self.mode: str | None = None
        self._closing = False

    # ------------------------------------------------------------------ windows
    def start_mode(self, requested: str | None) -> str:
        """Image QC unless another mode is asked for (--mode, or a video / table file on the command line)."""
        return requested if requested in MODES else "image"

    def window(self, mode: str):
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
            return self.image
        if mode == "video":
            if self.video is None:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    from AIO_QC.tools.frame_extractor.app import MainWindow as FrameExtractorWindow
                    self.video = FrameExtractorWindow()
                finally:
                    QApplication.restoreOverrideCursor()
                from AIO_QC.utils import newIcon
                self.video.setWindowIcon(newIcon("mode-video"))       # the Frame Extractor keeps its own app icon
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
        if info["kind"] != "labelme" or info["missing"]:
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
        img.open_dataset(Path(folder), DatasetType.LABELME)
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
        w.show()
        w.raise_()
        w.activateWindow()
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
        old = self.window(self.mode) if self.mode else None
        new = self.window(mode)
        if old is not None and old.isVisible():
            if old.isMaximized() or old.isFullScreen():
                new.showMaximized()
            else:
                new.setGeometry(old.geometry())
                new.showNormal()
        else:
            new.show()
        if new is not self.video:                 # never on the Frame Extractor: opacity breaks its OpenGL canvas
            try:
                from AIO_QC.widgets import anim
                anim.window_fade_in(new, 140)
            except Exception:
                pass
        new.raise_()
        new.activateWindow()
        if old is not None:
            old.hide()
            rel = getattr(old, "release_memory", None)      # in the background: keep only what it shows
            if rel is not None:
                try:
                    rel()
                except Exception:
                    pass
        self.mode = mode
        self.st.setValue("shell/mode", mode)
        sw = getattr(new, "mode_switch", None)
        if sw is not None:
            sw.set_current(mode)
        if mode == "data" and hasattr(new, "_show_welcome") and not new.ws.tables:
            new._show_welcome()
        if old is not None and old is self.video and self.video.is_playing:
            self.video.pause()                            # a hidden Frame Extractor never keeps playing
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
        if ev.type() == QEvent.Type.Close and obj in (self.image, self.studio, self.video) and not self._closing:
            ev.ignore()
            QTimer.singleShot(0, self.quit_all)
            return True
        return False

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
        self.app.quit()

    def log_errors(self, lines):
        """Unhandled errors (AIO QC's one excepthook): the error log of Image QC, also when another mode is in front
        (that window flashes a short note in its status bar)."""
        cur = self.window(self.mode) if self.mode in ("video", "data") and self._built(self.mode) else None
        if cur is not None:
            try:
                cur.statusBar().showMessage(f"⚠ {lines[-1]} — details in the log", 10000)
            except Exception:
                pass
        if self.image is not None:
            self.image.log_errors(lines)
        else:
            for line in lines:
                logger.error(line)

    def _built(self, mode):
        return {"image": self.image, "video": self.video, "data": self.studio}.get(mode) is not None
