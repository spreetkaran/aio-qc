# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
App-wide inference service: one server process in the chosen environment, shared by the Model Setup
window and Model Check. Blocking methods are meant to be called from worker threads (FuncWorker /
scan worker); they are serialised with a lock.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from AIO_QC.core.env_manager import missing_packages, remove_env
from AIO_QC.core.inference import InferenceClient, InferenceError
from AIO_QC.widgets.lifecycle import RetainedThread

DEFAULTS = {"mode": "standard", "device": "cpu", "half": False, "batch": 8, "slice": 640, "overlap": 0.2,
            "nms_iou": 0.6}


class FuncWorker(RetainedThread, QThread):
    """Run fn() off the GUI thread; result / error come back as signals."""
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.setObjectName(f"FuncWorker({getattr(fn, '__qualname__', getattr(fn, '__name__', 'fn'))})")

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class InferenceService(QObject):
    log = pyqtSignal(str)
    changed = pyqtSignal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.python: str = settings.value("inference/python", "") or ""
        self.temp_env: str = settings.value("inference/temp_env", "") or ""
        self.delete_temp_on_exit = settings.value("inference/delete_temp_on_exit", True, type=bool)
        self.opts = {k: settings.value(f"inference/{k}", v, type=type(v)) for k, v in DEFAULTS.items()}
        self.client: InferenceClient | None = None
        self.info: dict | None = None
        self.loaded_key: str | None = None
        self.loaded: dict[str, dict] = {}          # model id -> info from the server (names, task, ...)
        self._lock = threading.RLock()
        self._workers: list = []

    # ------------------------------------------------------------------ config
    def set_python(self, python: str):
        python = str(python or "")
        if python == self.python:
            return
        self.stop()
        self.python, self.info = python, None
        self.settings.setValue("inference/python", python)
        self.changed.emit()

    def set_option(self, key, value):
        self.opts[key] = value
        self.settings.setValue(f"inference/{key}", value)

    def set_temp_env(self, path: str, delete_on_exit: bool):
        self.temp_env, self.delete_temp_on_exit = str(path or ""), bool(delete_on_exit)
        self.settings.setValue("inference/temp_env", self.temp_env)
        self.settings.setValue("inference/delete_temp_on_exit", self.delete_temp_on_exit)

    @property
    def mode(self) -> str:
        return self.opts["mode"]

    def missing(self) -> list[str]:
        return missing_packages(self.info, self.mode)

    @property
    def ready(self) -> bool:
        return bool(self.python) and self.info is not None and not self.missing()

    # --------------------------------------------------------- blocking calls
    def ensure_started(self) -> dict:
        with self._lock:
            if not self.python:
                raise InferenceError("No Python environment selected (Model Setup ▸ Environment)")
            if self.client is not None and self.client.alive and self.client.python == self.python and self.info:
                return self.info
            self._stop_locked()
            self.log.emit(f"▶ starting inference process: {self.python}")
            self.client = InferenceClient(self.python, log=self.log.emit)
            self.client.start()
            self.info = self.client.probe()
            self.loaded_key, self.loaded = None, {}
            self.changed.emit()
            return self.info

    def ensure_loaded(self, specs: list[dict]) -> dict[str, dict]:
        """Load / reuse models on the server. Returns {id: info}."""
        with self._lock:
            self.ensure_started()
            miss = self.missing()
            if miss:
                raise InferenceError(f"The environment lacks: {', '.join(miss)}")
            key = json.dumps(specs, sort_keys=True)
            if key == self.loaded_key and self.client.alive:
                return self.loaded
            self.log.emit(f"▶ loading {len(specs)} model(s)…")
            infos = self.client.load(specs)
            self.loaded = {i["id"]: i for i in infos}
            self.loaded_key = key if all(i.get("ok") for i in infos) else None
            self.changed.emit()
            return self.loaded

    def predict(self, images, ids, conf=0.05, batch=None, classes=None, cancel=None) -> dict:
        with self._lock:
            if self.client is None or not self.client.alive:
                raise InferenceError("inference process is not running")
            return self.client.predict(images, ids, conf, batch or self.opts["batch"], classes, cancel)

    # -------------------------------------------------------------- lifecycle
    def run_async(self, fn, on_done=None, on_fail=None):
        w = FuncWorker(fn)
        if on_done:
            w.done.connect(on_done)
        if on_fail:
            w.failed.connect(on_fail)
        w.finished.connect(lambda: self._workers.remove(w) if w in self._workers else None)
        self._workers.append(w)
        w.start()
        return w

    def _stop_locked(self):
        if self.client is not None:
            self.client.close()
            self.client = None
        self.loaded_key, self.loaded = None, {}

    def stop(self):
        with self._lock:
            self._stop_locked()

    def shutdown(self):
        """App exit: stop the server; delete the temporary environment if asked to ('flush after use')."""
        self.stop()
        for w in list(self._workers):
            w.wait(3000)
        if self.temp_env and self.delete_temp_on_exit:
            path = Path(self.temp_env)
            ok = remove_env(path) if path.exists() else True
            if ok:
                if self.python.startswith(self.temp_env):
                    self.settings.setValue("inference/python", "")
                self.settings.setValue("inference/temp_env", "")
            else:  # locked files (Windows) -> retry at next start
                pending = set(self.settings.value("inference/pending_delete", []) or [])
                pending.add(str(path))
                self.settings.setValue("inference/pending_delete", sorted(pending))
