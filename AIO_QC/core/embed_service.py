# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Keeps one embedding server (core/embed_server.py, in the chosen environment) alive and its model loaded."""
from __future__ import annotations

import threading
from pathlib import Path

from .inference import InferenceClient

EMBED_SERVER = Path(__file__).resolve().with_name("embed_server.py")
NEEDS = {"transformers": ["torch", "transformers"], "timm": ["torch", "timm"], "open_clip": ["torch", "open_clip"],
         "DINOv3": ["torch"], "download": ["huggingface_hub"]}
PIP_NAMES = {"open_clip": "open_clip_torch", "huggingface_hub": "huggingface_hub", "transformers": "transformers",
             "timm": "timm", "torch": "torch"}


class EmbedService:
    def __init__(self, log=None):
        self.log = log or (lambda s: None)
        self.client: InferenceClient | None = None
        self.python: str | None = None
        self.loaded: tuple | None = None
        self.load_info: dict = {}
        self.info: dict = {}
        self._lock = threading.RLock()

    def _ensure(self, python: str):
        if self.client is not None and self.python == python and self.client.alive:
            return
        self.shutdown()
        self.client = InferenceClient(python, log=self.log, server=EMBED_SERVER)
        self.client.start()
        self.python, self.loaded = python, None

    def probe(self, python: str) -> dict:
        with self._lock:
            self._ensure(python)
            self.info = self.client.request("probe", timeout=180)["info"]
            return self.info

    def missing(self, kind: str) -> list[str]:
        """Packages the environment lacks for a model kind ('transformers · dinov3_vit' -> transformers)."""
        key = next((k for k in NEEDS if kind.startswith(k)), "transformers")
        return [PIP_NAMES.get(p, p) for p in NEEDS[key] if not self.info.get(p)]

    def ensure_loaded(self, python: str, path: str, device: str, half: bool) -> dict:
        with self._lock:
            self._ensure(python)
            key = (str(path), device, bool(half))
            if self.loaded != key:
                self.load_info = self.client.request("load", timeout=None, path=str(path), device=device,
                                                     half=bool(half))
                self.loaded = key
            return self.load_info

    def embed(self, items, out, batch, pad, cancel=None) -> dict:
        return self.client.request("embed", timeout=None, cancel=cancel, items=items, out=str(out), batch=int(batch),
                                   pad=float(pad))

    def knn(self, inp, k, out, cancel=None) -> dict:
        return self.client.request("knn", timeout=None, cancel=cancel, inp=str(inp), k=int(k), out=str(out))

    def download(self, python: str, repo: str, dest: str, token: str | None, cancel=None) -> str:
        with self._lock:
            self._ensure(python)
            return self.client.request("download", timeout=None, cancel=cancel, repo=repo, dest=dest,
                                       token=token or None)["path"]

    def shutdown(self):
        with self._lock:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass
            self.client, self.loaded = None, None
