# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""Keeps one OCR server process (in the chosen environment) alive and its engine loaded between scans."""
from __future__ import annotations

import threading
from pathlib import Path

from .inference import InferenceClient

OCR_SERVER = Path(__file__).resolve().with_name("ocr_server.py")


class OcrService:
    def __init__(self, log=None):
        self.log = log or (lambda s: None)
        self.client: InferenceClient | None = None
        self.python: str | None = None
        self.loaded: tuple | None = None          # (engine, lang, gpu, mode)
        self.load_info: dict = {}
        self._lock = threading.RLock()

    def _ensure_client(self, python: str):
        if self.client is not None and self.python == python and self.client.proc and self.client.proc.poll() is None:
            return
        self.shutdown()
        self.client = InferenceClient(python, log=self.log, server=OCR_SERVER)
        self.client.start()
        self.python, self.loaded = python, None

    def probe(self, python: str) -> dict:
        with self._lock:
            self._ensure_client(python)
            return self.client.request("probe", timeout=120)["info"]

    def ensure_loaded(self, python: str, engine: str, langs: list[str], gpu: bool, mode: str,
                      strategy: str = "best", fallback_conf: float = 0.6) -> dict:
        """langs: the ENGINE's language codes (core.ocr_langs.resolve)."""
        with self._lock:
            self._ensure_client(python)
            key = (engine, tuple(langs), bool(gpu), mode, strategy, round(float(fallback_conf), 3))
            if self.loaded != key:
                self.load_info = self.client.request("load", timeout=None, engine=engine, langs=list(langs),
                                                     gpu=bool(gpu), mode=mode, strategy=strategy,
                                                     fallback_conf=float(fallback_conf))
                self.loaded = key
            return self.load_info

    def read(self, items, pad: float, min_h: float, cancel=None) -> list:
        return self.client.request("read", timeout=None, cancel=cancel, items=items, pad=pad, min_h=min_h)["results"]

    def shutdown(self):
        with self._lock:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass
            self.client, self.loaded = None, None
