# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
One app, two modes. The shell owns the two main windows — Image QC (annotated image datasets) and Data Studio
(tables) — and switches between them: the other window takes the same place and size, the first one hides (keeping
its state). Each window is a complete app of its own: menus, panels, recent files, settings. A window is only built
the first time its mode is used, so starting in Data Studio never loads the image machinery.

    Ctrl+Alt+M or the switch in the menu-bar corner · File ▸ Switch to … · aio-qc --mode data|image
The app starts in the mode used last. Closing either window quits the app (both get to save first).
"""
from __future__ import annotations

from loguru import logger
from PyQt6.QtCore import QEvent, QObject, QTimer, Qt
from PyQt6.QtWidgets import QApplication

from AIO_QC.utils import app_settings

MODES = ("image", "data")


class AppShell(QObject):
    def __init__(self, app: QApplication):
        super().__init__(app)
        self.app = app
        self.st = app_settings()
        self.image = None
        self.studio = None
        self.mode: str | None = None
        self._closing = False

    # ------------------------------------------------------------------ windows
    def start_mode(self, requested: str | None) -> str:
        m = requested or self.st.value("shell/mode", "image")
        return m if m in MODES else "image"

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
            return self.image
        if self.studio is None:
            from AIO_QC.widgets.studio.window import StudioWindow
            self.studio = StudioWindow(self)
            self.studio.installEventFilter(self)
        return self.studio

    def show(self, mode: str):
        if mode == "data" and self.data_ready():
            logger.warning(self.data_ready())
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
        if mode == "data" and self.data_ready():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self.window(self.mode) if self.mode else None, "Data Studio", self.data_ready())
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
        try:
            from AIO_QC.widgets import anim
            anim.window_fade_in(new, 140)
        except Exception:
            pass
        new.raise_()
        new.activateWindow()
        if old is not None:
            old.hide()
        self.mode = mode
        self.st.setValue("shell/mode", mode)
        sw = getattr(new, "mode_switch", None)
        if sw is not None:
            sw.set_current(mode)
        if mode == "data" and hasattr(new, "_show_welcome") and not new.ws.tables:
            new._show_welcome()
        logger.info(f"switched to {'Data Studio' if mode == 'data' else 'Image QC'}")

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
        if ev.type() == QEvent.Type.Close and obj in (self.image, self.studio) and not self._closing:
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
        if self.image is not None:
            self.image.request_quit(reason)
        if self.studio is not None:
            self.studio.shutdown()
            self.studio.close()
        self.app.quit()

    def log_errors(self, lines):
        if self.image is not None:
            self.image.log_errors(lines)
        else:
            for line in lines:
                logger.error(line)
