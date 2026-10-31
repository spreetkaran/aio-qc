# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Client for infer_server.py, which runs in the user's chosen Python environment (subprocess + JSON lines).

Why a subprocess: the app never imports torch/ultralytics itself, so any environment works (conda, venv,
system, a temporary one), binary clashes are impossible, a CUDA crash can't take the GUI down, and models
stay loaded between scans.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

SERVER = Path(__file__).with_name("infer_server.py")
RESP_PREFIX = "@@AIOQC@@ "


class InferenceError(RuntimeError):
    pass


class InferenceClient:
    def __init__(self, python: str, log=None, server: Path | None = None):
        """server: the standalone script to run in the env (default: infer_server.py; OCR uses ocr_server.py)."""
        self.python = str(python)
        self.server = Path(server) if server else SERVER
        self.log = log or (lambda line: None)
        self.proc: subprocess.Popen | None = None
        self._responses: dict = {}
        self._cond = threading.Condition()
        self._lock = threading.Lock()          # one request on the wire at a time
        self._rid = itertools.count(1)
        self._dead: str | None = None

    # ------------------------------------------------------------------ process
    def start(self):
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("PYTHONHOME", None)            # never leak the GUI's interpreter settings into the env
        env.pop("PYTHONPATH", None)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen([self.python, "-u", str(self.server)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                     errors="replace", bufsize=1, env=env, creationflags=flags)
        self._dead = None
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self):
        for line in self.proc.stdout:
            if line.startswith(RESP_PREFIX):
                try:
                    msg = json.loads(line[len(RESP_PREFIX):])
                except ValueError:
                    continue
                with self._cond:
                    self._responses[msg.get("rid")] = msg
                    self._cond.notify_all()
            elif line.strip():
                self.log(line.rstrip())        # stray C-level prints
        with self._cond:
            self._dead = f"inference process exited (code {self.proc.poll()})"
            self._cond.notify_all()

    def _read_stderr(self):
        for line in self.proc.stderr:
            if line.strip():
                self.log(line.rstrip())

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self, timeout: float = 5.0):
        if self.proc is None:
            return
        try:
            if self.alive:
                self.request("shutdown", timeout=timeout)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except Exception:
            self.proc.kill()
        self.proc = None

    # ------------------------------------------------------------------ calls
    def request(self, cmd: str, timeout: float | None = None, cancel: threading.Event | None = None, **payload):
        if not self.alive:
            raise InferenceError(self._dead or "inference process is not running")
        with self._lock:
            rid = next(self._rid)
            self.proc.stdin.write(json.dumps({"rid": rid, "cmd": cmd, **payload}) + "\n")
            self.proc.stdin.flush()
            with self._cond:
                waited = 0.0
                while rid not in self._responses:
                    if self._dead:
                        raise InferenceError(self._dead)
                    if cancel is not None and cancel.is_set():
                        raise InferenceError("cancelled")
                    if timeout is not None and waited >= timeout:
                        raise InferenceError(f"'{cmd}' timed out after {timeout:.0f}s")
                    self._cond.wait(0.25)
                    waited += 0.25
                msg = self._responses.pop(rid)
        if not msg.get("ok"):
            raise InferenceError(msg.get("error", "unknown error"))
        return msg

    def probe(self, timeout=180) -> dict:
        return self.request("probe", timeout=timeout)["info"]

    def load(self, specs: list[dict], timeout=900) -> list[dict]:
        return self.request("load", timeout=timeout, models=specs)["models"]

    def export(self, spec: dict) -> dict:
        return self.request("export", timeout=None, spec=spec)

    def predict(self, images, ids, conf=0.05, batch=8, classes=None, cancel=None) -> dict:
        return self.request("predict", timeout=None, cancel=cancel, images=[str(p) for p in images], ids=list(ids),
                            conf=conf, batch=batch, classes=classes or {})["results"]
