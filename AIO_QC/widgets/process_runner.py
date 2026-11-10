# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Run a list of commands one after another (QProcess), streaming their output, with friendly failure messages.
Used for editor environment installs; the editors themselves are launched with `clean_process_env()`.
"""
from __future__ import annotations

import re

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, pyqtSignal

# Our app runs PyQt6; these would point a PyQt5 editor at the wrong Qt plugins / Python packages.
_LEAKY = ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
          "QML2_IMPORT_PATH", "QT_API")

HINTS = [
    (re.compile(r"Temporary failure in name resolution|Failed to establish a new connection|ConnectionError|"
                r"NewConnectionError|Network is unreachable|getaddrinfo failed", re.I),
     "No internet connection (or a proxy blocks pip). Connect and try again — or choose an existing Python "
     "environment that already has the editor's packages."),
    (re.compile(r"No matching distribution found|Could not find a version that satisfies", re.I),
     "A package has no build for this Python version. Choose a different Python (3.10 – 3.12 work best)."),
    (re.compile(r"No space left on device|Errno 28", re.I), "The disk is full."),
    (re.compile(r"Permission denied|Access is denied", re.I),
     "Permission denied — another program may be using the environment folder."),
    (re.compile(r"No module named venv|ensurepip is not available", re.I),
     "This Python cannot create virtual environments (install the python3-venv package, or choose another Python)."),
]


def clean_process_env(extra: dict | None = None) -> QProcessEnvironment:
    env = QProcessEnvironment.systemEnvironment()
    for k in _LEAKY:
        env.remove(k)
    env.insert("PYTHONUNBUFFERED", "1")
    env.insert("PYTHONIOENCODING", "utf-8")
    for k, v in (extra or {}).items():
        env.insert(k, v)
    return env


class StepRunner(QObject):
    output = pyqtSignal(str)
    stepStarted = pyqtSignal(int, int, str)       # index, total, title
    finished = pyqtSignal(bool, str)              # ok, message

    def __init__(self, steps: list[tuple[str, list[str]]], parent=None):
        super().__init__(parent)
        self.steps, self.i = list(steps), -1
        self.proc: QProcess | None = None
        self._tail: list[str] = []
        self._cancelled = False

    def start(self):
        self._next()

    def cancel(self):
        self._cancelled = True
        if self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning:
            self.proc.kill()

    def _next(self):
        self.i += 1
        if self.i >= len(self.steps):
            self.finished.emit(True, "Done")
            return
        title, argv = self.steps[self.i]
        self.stepStarted.emit(self.i, len(self.steps), title)
        self.output.emit(f"\n▶ {title}\n$ {' '.join(argv)}\n")
        self._tail = []
        p = QProcess(self)
        p.setProcessEnvironment(clean_process_env())
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        p.readyReadStandardOutput.connect(self._read)
        p.finished.connect(self._done)
        p.errorOccurred.connect(self._error)
        self.proc = p
        p.start(argv[0], argv[1:])

    def _read(self):
        text = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._tail = (self._tail + text.splitlines())[-60:]
        self.output.emit(text)

    def _error(self, err):
        if err == QProcess.ProcessError.FailedToStart:
            self.finished.emit(False, f"Could not start: {self.steps[self.i][1][0]}")

    def _done(self, code, _status):
        if self._cancelled:
            self.finished.emit(False, "Cancelled")
            return
        if code != 0:
            tail = "\n".join(self._tail)
            hint = next((h for rx, h in HINTS if rx.search(tail)), "See the log above for details.")
            self.finished.emit(False, f"“{self.steps[self.i][0]}” failed (exit code {code}). {hint}")
            return
        self._next()
