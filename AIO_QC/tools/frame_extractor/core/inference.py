# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Client for infer_server.py, which runs in the user's chosen Python environment (subprocess + JSON lines).
Adapted from AIO QC.

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
RESP_PREFIX = "@@FXINFER@@ "


class InferenceError(RuntimeError):
    pass


class InferenceClient:
    def __init__(self, python: str, log=None):
        self.python = str(python)
        self.log = log or (lambda line: None)
        self.proc: subprocess.Popen | None = None
        self._responses: dict = {}
        self._cond = threading.Condition()
        self._lock = threading.Lock()          # one request on the wire at a time
        self._rid = itertools.count(1)
        self._dead: str | None = None
        self._shm = None                       # reusable shared-memory block for frame batches
        self._shm_ok = True

    # ------------------------------------------------------------------ process
    def start(self):
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("PYTHONHOME", None)            # never leak the GUI's interpreter settings into the env
        env.pop("PYTHONPATH", None)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        from AIO_QC.core import procs
        self.proc = procs.popen([self.python, "-u", str(SERVER)], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                errors="replace", bufsize=1, env=env)
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
        self._free_shm()
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

    def load(self, specs: list[dict], timeout=900, replace=False) -> list[dict]:
        return self.request("load", timeout=timeout, models=specs, replace=replace)["models"]

    def export(self, spec: dict) -> dict:
        return self.request("export", timeout=None, spec=spec)

    def predict_arrays(self, paths, ids, conf=0.05, classes=None, cancel=None) -> dict:
        return self.request("predict_arrays", timeout=None, cancel=cancel, arrays=[str(p) for p in paths],
                            ids=list(ids), conf=conf, classes=classes or {})["results"]

    def _free_shm(self):
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None

    def _frames_to_shm(self, frames):
        """Copy frames into the reusable shared-memory block -> [[offset, h, w], ...]."""
        import numpy as np
        from multiprocessing import shared_memory
        need = sum(f.nbytes for f in frames)
        if self._shm is None or self._shm.size < need:
            self._free_shm()
            self._shm = shared_memory.SharedMemory(create=True, size=max(need, 64 * 2 ** 20))
        desc, off = [], 0
        for f in frames:
            self._shm.buf[off:off + f.nbytes] = f.reshape(-1).view(np.uint8)
            desc.append([off, f.shape[0], f.shape[1]])
            off += f.nbytes
        return desc

    def embed_load(self, spec, timeout=1800) -> dict:
        return self.request("embed_load", timeout=timeout, spec=spec)["info"]

    def embed_frames(self, frames, cancel=None) -> list:
        import numpy as np
        frames = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames]
        desc = self._frames_to_shm(frames)
        return self.request("embed_shm", timeout=None, cancel=cancel, shm=self._shm.name, frames=desc)["vectors"]

    def embed_text(self, texts) -> list:
        return self.request("embed_text", timeout=300, texts=list(texts))["vectors"]

    def predict_frames(self, frames, ids, conf=0.05, batch=8, classes=None, cancel=None) -> list:
        """BGR uint8 frames (any sizes) -> list (per frame) of {"by_model": {id: [[x1,y1,x2,y2,conf,cls], ...]},
        "error"}. Frames travel through shared memory (no disk); falls back to .npy files if that fails."""
        import numpy as np
        frames = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames]
        if self._shm_ok:
            try:
                desc = self._frames_to_shm(frames)
                res = self.request("predict_shm", timeout=None, cancel=cancel, shm=self._shm.name, frames=desc,
                                   ids=list(ids), conf=conf, batch=batch, classes=classes or {})["results"]
                return [res[str(i)] for i in range(len(frames))]
            except InferenceError as e:
                if "unknown command" not in str(e):
                    raise
                self._shm_ok = False              # an older server: use files
            except (OSError, ValueError, ImportError):
                self._shm_ok = False
        import tempfile
        d = tempfile.mkdtemp(prefix="fx_frames_")
        try:
            paths = []
            for i, f in enumerate(frames):
                p = os.path.join(d, f"{i}.npy")
                np.save(p, f)
                paths.append(p)
            res = self.predict_arrays(paths, ids, conf, classes, cancel)
            return [res[p] for p in paths]
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def predict(self, images, ids, conf=0.05, batch=8, classes=None, cancel=None) -> dict:
        return self.request("predict", timeout=None, cancel=cancel, images=[str(p) for p in images], ids=list(ids),
                            conf=conf, batch=batch, classes=classes or {})["results"]
