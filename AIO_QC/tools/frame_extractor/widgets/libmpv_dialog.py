# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Settings ▸ Playback ▸ libmpv… - is libmpv in use, why not, and get it for this computer (automatically).

MpvFetchJob runs core.mpv_fetch in a plain worker thread (Qt-free: it only fills a queue) and reports progress,
success and failure as signals from the UI thread (a timer drains the queue). The Frame Extractor starts one by
itself when libmpv is missing (Windows / macOS: a background download; Linux: asks once, then the package manager
via pkexec), and the dialog's "Get libmpv" button starts or shows the same job.

Shows the library in use (or the exact reason none could be loaded), how playback uses it (verified picture,
basic renderer, clock sync), every place searched, and per-OS instructions with one-click actions: choose a file,
install a DLL into the per-user libmpv folder (Windows), copy the install command (Linux / macOS), open the
download page. All library work is in core.mpv_lib (Qt-free); this is only the window.
"""
from __future__ import annotations

import os
import queue
import shutil
import sys
import threading

from PyQt6.QtCore import QObject, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QGuiApplication
from PyQt6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                             QPlainTextEdit, QProgressBar, QPushButton, QToolButton, QVBoxLayout)

from AIO_QC.tools.frame_extractor.core import mpv_fetch, mpv_lib
from AIO_QC.utils.qt_utils import fit_to_screen

WIN_URL = "https://github.com/zhongfly/mpv-winbuild/releases"
MAC_URL = "https://formulae.brew.sh/formula/mpv"
LINUX_URL = "https://mpv.io/installation/"


def linux_command(os_release: str | None = None) -> str:
    """The install command for this distribution (from /etc/os-release)."""
    text = os_release
    if text is None:
        try:
            with open("/etc/os-release", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            text = ""
    info = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip().strip('"')
    ids = (info.get("ID", "") + " " + info.get("ID_LIKE", "")).lower().split()
    if any(i in ids for i in ("ubuntu", "debian", "linuxmint", "pop")):
        return "sudo apt install libmpv2 || sudo apt install libmpv1"
    if any(i in ids for i in ("fedora", "rhel", "centos", "rocky", "almalinux")):
        return "sudo dnf install mpv-libs        # from RPM Fusion (rpmfusion.org)"
    if any(i in ids for i in ("arch", "manjaro", "endeavouros")):
        return "sudo pacman -S mpv"
    if any(i in ids for i in ("opensuse", "suse", "opensuse-leap", "opensuse-tumbleweed")):
        return "sudo zypper install libmpv2"
    if "alpine" in ids:
        return "sudo apk add mpv-libs"
    return "Install your distribution's libmpv package (libmpv2 / libmpv1 / mpv-libs)"


def instructions(platform: str = sys.platform) -> tuple[str, str, str]:
    """(html, command to copy or '', download / info URL) for this OS."""
    if platform == "win32":
        return ("<ol><li>Open the download page and get <b>mpv-dev-lgpl-x86_64-…7z</b> (ARM PCs: <b>…aarch64…</b>). "
                "The <i>lgpl</i> build is LGPL-2.1+; the others are GPL.</li>"
                "<li>Extract only <b>libmpv-2.dll</b> from it (7-Zip).</li>"
                "<li>Click <b>Install DLL…</b> and pick it: it is copied to your libmpv folder and kept across "
                "updates.</li></ol>", "", WIN_URL)
    if platform == "darwin":
        return ("<ol><li>Install Homebrew (brew.sh) if you don't have it.</li>"
                "<li>Run the command below in Terminal, then click <b>Check again</b>.</li></ol>"
                "<p>App builds can also carry their own copy (see packaging/LIBMPV.md).</p>",
                "brew install mpv", MAC_URL)
    return ("<p>Use your distribution's package - it matches the libraries already on your system "
            "(a libmpv copied from elsewhere often fails with <i>undefined symbol</i>). Run the command below, "
            "then click <b>Check again</b>.</p>", linux_command(), LINUX_URL)


class MpvFetchJob(QObject):
    """One automatic libmpv download / install. Signals are emitted on the UI thread."""
    progress = pyqtSignal(object, str)          # fraction 0..1 or None (busy), text
    succeeded = pyqtSignal(str)                 # libmpv path ("" = system package installed)
    failed = pyqtSignal(str)                    # reason ("cancelled" when stopped)
    finished = pyqtSignal()

    def __init__(self, dest, parent=None, fetch_fn=None):
        super().__init__(parent)
        self.dest = dest
        self.fetch_fn = fetch_fn or mpv_fetch.fetch
        self.running, self.last = False, (None, "")
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._drain)

    def start(self):
        if self.running:
            return
        self.running = True
        self._stop.clear()
        threading.Thread(target=self._work, name="libmpv-fetch", daemon=True).start()
        self._timer.start()

    def cancel(self):
        self._stop.set()

    def _work(self):                            # worker thread: no Qt here, only the queue
        q = self._q
        try:
            path = self.fetch_fn(self.dest, lambda f, t: q.put(("progress", f, t)), self._stop.is_set)
            if path:
                lib, msg = mpv_lib.probe(path)
                if lib is None:
                    raise RuntimeError(f"downloaded, but it can't be used: {msg}")
            q.put(("ok", path))
        except mpv_fetch.Cancelled:
            q.put(("fail", "cancelled"))
        except Exception as e:
            q.put(("fail", f"{e}"))

    def _drain(self):
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return
            if item[0] == "progress":
                self.last = (item[1], item[2])
                self.progress.emit(item[1], item[2])
                continue
            self.running = False
            self._timer.stop()
            (self.succeeded if item[0] == "ok" else self.failed).emit(item[1])
            self.finished.emit()
            return


class LibmpvDialog(QDialog):
    def __init__(self, parent, settings, status_fn=None, on_changed=None, job_fn=None):
        """status_fn() -> dict with 'gpu', 'engine', 'state' (how playback uses libmpv now);
        on_changed() re-configures playback after a new library was chosen;
        job_fn() -> the app's MpvFetchJob (started, or already running) for "Get libmpv"."""
        super().__init__(parent)
        self.setWindowTitle("libmpv - playback engine")
        self.settings, self.status_fn, self.on_changed, self.job_fn = settings, status_fn, on_changed, job_fn
        self.job = None
        v = QVBoxLayout(self)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.status)
        # ---- automatic
        plan = mpv_fetch.plan()
        row = QHBoxLayout()
        self.get_btn = QPushButton("Install with your package manager…" if plan["how"] == "package"
                                   else "Get libmpv automatically")
        self.get_btn.setObjectName("libmpv_get")
        self.get_btn.setToolTip(f"{plan.get('what', '')} - from {plan.get('source', '')}" if plan["how"] else
                                plan.get("why", ""))
        self.get_btn.setEnabled(bool(plan["how"] and job_fn is not None))
        self.get_btn.clicked.connect(self.get_libmpv)
        row.addWidget(self.get_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("libmpv_cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(lambda _c=False: self.job is not None and self.job.cancel())
        row.addWidget(self.cancel_btn)
        row.addStretch(1)
        v.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        v.addWidget(self.bar)
        self.fetch_msg = QLabel(f"<small>{plan.get('what', '')}"
                                + (f" · {plan['source']}" if plan.get("source") else "") + "</small>"
                                if plan["how"] else f"<small>{plan.get('why', '')}</small>")
        self.fetch_msg.setWordWrap(True)
        v.addWidget(self.fetch_msg)
        self.auto = QCheckBox("Get libmpv automatically when it is missing" + (" (asks first)" if plan["how"] ==
                                                                               "package" else ""))
        self.auto.setObjectName("libmpv_auto")
        self.auto.setChecked(bool(settings.value("playback/libmpv_auto", True, type=bool)))
        self.auto.toggled.connect(lambda on: self.settings.setValue("playback/libmpv_auto", bool(on)))
        v.addWidget(self.auto)
        # ---- by hand
        how, cmd, url = instructions()
        self.url = url
        guide = QLabel("<b>Or by hand</b>" + how)
        guide.setWordWrap(True)
        guide.setOpenExternalLinks(True)
        v.addWidget(guide)
        self.cmd = cmd
        if cmd:
            row = QHBoxLayout()
            self.cmd_label = QLabel(f"<code>{cmd}</code>")
            self.cmd_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row.addWidget(self.cmd_label, 1)
            copy = QPushButton("Copy command")
            copy.setObjectName("libmpv_copy")
            copy.clicked.connect(self.copy_command)
            row.addWidget(copy)
            v.addLayout(row)
        row = QHBoxLayout()
        for name, text, slot, show in (
                ("libmpv_page", "Open download page", self.open_page, True),
                ("libmpv_install", "Install DLL…", self.install_dll, sys.platform == "win32"),
                ("libmpv_choose", "Choose file…", self.choose_file, True),
                ("libmpv_folder", "Open libmpv folder", self.open_folder, True),
                ("libmpv_check", "Check again", self.check_again, True)):
            b = QPushButton(text)
            b.setObjectName(name)
            b.clicked.connect(slot)
            b.setVisible(bool(show))
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.more = QToolButton()
        self.more.setText("Places searched ▸")
        self.more.setCheckable(True)
        self.more.toggled.connect(self._toggle_details)
        v.addWidget(self.more)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setVisible(False)
        self.details.setMinimumHeight(140)
        v.addWidget(self.details)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.resize(720, 480)
        self.refresh()
        if job_fn is not None:
            running = job_fn(start=False)
            if running is not None and running.running:
                self._follow(running)

    # ------------------------------------------------------------------ state
    def refresh(self):
        user = self.settings.value("playback/libmpv", "") or ""
        lib = mpv_lib.load_library(user)
        st = self.status_fn() if self.status_fn else {}
        if lib is not None:
            api = getattr(lib, "api", None)
            head = (f"<p style='color:#2e7d32'><b>✓ libmpv found</b> - {lib.path}"
                    + (f" (client API {api[0]}.{api[1]})" if api else "") + "</p>")
        else:
            head = (f"<p style='color:#c62828'><b>✗ libmpv is not available</b> - {mpv_lib.last_error()}</p>"
                    "<p>Playback uses the built-in engine (it works; libmpv is smoother on long / high-fps "
                    "videos).</p>")
        notes = []
        if st:
            if not st.get("gpu", True):
                notes.append("libmpv needs the GPU renderer (Settings ▸ Display ▸ Video renderer ▸ GPU); the CPU renderer is active.")
            if st.get("engine") == "builtin":
                notes.append("Settings ▸ Playback ▸ Engine is set to built-in.")
            if st.get("state"):
                notes.append(st["state"])
        self.status.setText(head + "".join(f"<p>{n}</p>" for n in notes))
        rows = mpv_lib.diagnose(user)
        lines = [f"{'✓ ' + s if s else '·  not there':<14} {p}" for p, s in rows]
        lines.append("then the system's library search path (" + ", ".join(mpv_lib._NAMES) + ")")
        self.details.setPlainText("\n".join(lines))

    def _toggle_details(self, on):
        self.details.setVisible(bool(on))
        self.more.setText("Places searched ▾" if on else "Places searched ▸")

    def _apply(self, path):
        mpv_lib.reset()
        if mpv_lib.load_library(path) is None:
            QMessageBox.warning(self, "libmpv", f"This file can't be used:\n{mpv_lib.last_error()}")
            mpv_lib.reset()
            self.refresh()
            return False
        self.settings.setValue("playback/libmpv", path)
        if self.on_changed:
            self.on_changed()
        self.refresh()
        return True

    # ------------------------------------------------------------------ automatic
    def get_libmpv(self):
        if self.job_fn is None:
            return
        self._follow(self.job_fn(start=True))

    def _follow(self, job):
        if job is None:
            return
        self.job = job
        self.get_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        job.progress.connect(self._on_progress)
        job.succeeded.connect(self._on_ok)
        job.failed.connect(self._on_fail)
        if job.last[1]:
            self._on_progress(*job.last)

    def _on_progress(self, frac, text):
        if frac is None:
            self.bar.setRange(0, 0)
        else:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(frac * 1000))
        self.fetch_msg.setText(f"<small>{text}</small>")

    def _done(self):
        self.get_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.bar.setVisible(False)

    def _on_ok(self, path):
        self._done()
        self.fetch_msg.setText("<small>✓ libmpv is installed" + (f": {path}" if path else "") + "</small>")
        self.refresh()

    def _on_fail(self, why):
        self._done()
        self.fetch_msg.setText(f"<small>✗ {why}</small>")

    # ------------------------------------------------------------------ actions
    def copy_command(self):
        QGuiApplication.clipboard().setText(self.cmd.split("  #")[0].strip())

    def open_page(self):
        QDesktopServices.openUrl(QUrl(self.url))

    def open_folder(self):
        d = mpv_lib.user_lib_dir()
        if d:
            os.makedirs(d, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def choose_file(self):
        filt = ("libmpv (libmpv-2.dll mpv-2.dll mpv-1.dll);;All files (*)" if sys.platform == "win32"
                else "libmpv (libmpv*);;All files (*)")
        path, _ = QFileDialog.getOpenFileName(self, "libmpv library", "", filt)
        if path:
            self._apply(path)

    def install_dll(self):
        """Windows: copy a self-contained libmpv-2.dll into the per-user libmpv folder (kept across updates)."""
        path, _ = QFileDialog.getOpenFileName(self, "libmpv-2.dll from an mpv-dev build", "",
                                              "libmpv (libmpv-2.dll mpv-2.dll);;DLL (*.dll)")
        if not path:
            return
        lib, msg = mpv_lib.probe(path)
        if lib is None:
            QMessageBox.warning(self, "libmpv", f"This file can't be used:\n{msg}")
            return
        d = mpv_lib.user_lib_dir()
        try:
            os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, "libmpv-2.dll")
            if os.path.normcase(os.path.abspath(path)) != os.path.normcase(os.path.abspath(dst)):
                shutil.copy2(path, dst + ".new")
                try:
                    os.replace(dst + ".new", dst)
                except OSError:                       # the old copy is loaded: it is swapped in at the next start
                    QMessageBox.information(self, "libmpv", "The new DLL is used from the next start "
                                                            "(the current one is in use).")
                    return
        except OSError as e:
            QMessageBox.warning(self, "libmpv", f"Could not copy it to {d}:\n{e}")
            return
        self._apply(dst)

    def check_again(self):
        mpv_lib.reset()
        if self.on_changed:
            self.on_changed()
        self.refresh()

    def showEvent(self, ev):
        super().showEvent(ev)
        fit_to_screen(self)
