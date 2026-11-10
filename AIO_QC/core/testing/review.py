# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Video review: the playlist Testing hands to the Frame Extractor, and the model details Testing shows - Qt-free.

Testing never opens or drives the player. It builds a ReviewPlaylist (videos with optional in / out times, the model
specs, colours, analysis mode) and gives it to the app shell, which passes it to the Frame Extractor; from then on
the player owns it (core.testing.stream analyses, widgets.review_dock plays).
"""
from __future__ import annotations

import ast
import json
import os
from dataclasses import asdict, dataclass, field

from AIO_QC.core.testing.stream import MODES


@dataclass
class ReviewItem:
    path: str
    start: float | None = None          # seconds (None: from the beginning)
    end: float | None = None            # seconds (None: to the end)

    @property
    def name(self):
        return os.path.basename(self.path)


@dataclass
class ReviewPlaylist:
    items: list = field(default_factory=list)          # [ReviewItem]
    specs: list = field(default_factory=list)          # [ModelSpec]
    colors: list = field(default_factory=list)         # one "#rrggbb" per spec (Testing's model colours)
    mode: str = "balanced"                             # balanced | best | fixed
    stride: int = 2                                    # fixed mode: every N-th frame
    lead_s: float = 3.0                                # look-ahead
    conf: float = 0.25                                 # display threshold at start (changeable in the player)
    loop: bool = False
    cache_path: str = ""                               # Testing's prediction cache (shared)
    source: str = "Testing"

    def validate(self):
        """Problems that stop a review: [] when it can start."""
        p = []
        if not self.items:
            p.append("no videos")
        if not self.specs:
            p.append("no models")
        if self.mode not in MODES:
            p.append(f"unknown mode {self.mode!r}")
        for it in self.items:
            if not os.path.isfile(it.path):
                p.append(f"missing video: {it.path}")
            if it.start is not None and it.end is not None and it.end <= it.start:
                p.append(f"{it.name}: out point before in point")
        for s in self.specs:
            if not os.path.isfile(s.path):
                p.append(f"missing weights: {s.path}")
        return p

    def to_dict(self):
        return dict(items=[asdict(i) for i in self.items], specs=[s.to_dict() for s in self.specs],
                    colors=list(self.colors), mode=self.mode, stride=int(self.stride), lead_s=float(self.lead_s),
                    conf=float(self.conf), loop=bool(self.loop), cache_path=self.cache_path, source=self.source)

    @classmethod
    def from_dict(cls, d):
        from AIO_QC.core.testing.runner import ModelSpec
        return cls(items=[ReviewItem(**i) for i in d.get("items", [])],
                   specs=[ModelSpec.from_dict(s) for s in d.get("specs", [])], colors=list(d.get("colors", [])),
                   mode=d.get("mode", "balanced"), stride=int(d.get("stride", 2)), lead_s=float(d.get("lead_s", 3.0)),
                   conf=float(d.get("conf", 0.25)), loop=bool(d.get("loop", False)),
                   cache_path=d.get("cache_path", ""), source=d.get("source", "Testing"))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ------------------------------------------------------------------ model details
def _names(raw):
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        return dict(enumerate(map(str, raw)))
    try:
        return _names(ast.literal_eval(str(raw)))
    except (ValueError, SyntaxError):
        return {}


def in_registry(path):
    """Weights the app already knows (added through Models & Environment / Testing / the Frame Extractor)."""
    try:
        from AIO_QC.core import model_registry as R
        return R.find(path) is not None
    except Exception:
        return False


def describe(spec, load=True) -> dict:
    """What a weights file is: classes, input size, task, batch, backend, provenance, warnings.
    ONNX is read with onnxruntime on the CPU (metadata only - no inference); .pt / .engine through the shared model
    environment (load=False: file facts only)."""
    path = spec.path
    ext = os.path.splitext(path)[1].lower()
    info = dict(path=path, label=spec.label, file=os.path.basename(path), format=ext.lstrip(".").upper(),
                size_mb=round(os.path.getsize(path) / 1e6, 1) if os.path.isfile(path) else None,
                imgsz=spec.imgsz or None, names={}, task="", dynamic_batch=None, input="", meta={}, warnings=[],
                error=None, key=None)
    if not os.path.isfile(path):
        info["error"] = "file not found"
        return info
    try:
        info["key"] = spec.key
    except Exception:
        pass
    if ext == ".onnx":
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            meta = dict(sess.get_modelmeta().custom_metadata_map)
            info["names"] = _names(meta.get("names"))
            info["task"] = meta.get("task", "")
            info["input"] = f"{inp.name} {list(inp.shape)} {inp.type.replace('tensor(', '').rstrip(')')}"
            info["dynamic_batch"] = not isinstance(inp.shape[0], int)
            shp = inp.shape
            if not info["imgsz"]:
                if isinstance(shp[2], int) and isinstance(shp[3], int):
                    info["imgsz"] = shp[2] if shp[2] == shp[3] else f"{shp[2]}×{shp[3]}"
                elif meta.get("imgsz"):
                    v = ast.literal_eval(meta["imgsz"])
                    info["imgsz"] = v[0] if isinstance(v, (list, tuple)) and v[0] == v[-1] else v
            info["meta"] = {k: meta[k] for k in ("author", "version", "date", "stride", "batch", "license",
                                                 "description") if k in meta}
            del sess
        except ImportError:
            info["warnings"].append("onnxruntime is not installed: pip install \"aio-qc[models]\"")
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
    else:
        if ext == ".pt":
            info["warnings"].append("A .pt file can run code when it is loaded - use weights from people you trust"
                                    + ("" if in_registry(path) else " (this one is new to the app)"))
        if ext == ".engine":
            info["warnings"].append("A TensorRT .engine runs only on the GPU type and TensorRT version it was built "
                                    "for")
        if load:
            try:
                d = spec.load()
                info["names"] = _names(getattr(d, "names", None))
                if not info["imgsz"] and getattr(d, "imgsz", None):
                    v = d.imgsz
                    info["imgsz"] = v[0] if isinstance(v, (list, tuple)) else v
                info["backend"] = getattr(d, "provider", "")
            except Exception as e:
                info["error"] = f"{type(e).__name__}: {e}"
    if not info["names"] and not info["error"]:
        info["warnings"].append("no class names in the file - classes show as numbers (add a .names / data.yaml "
                                "next to it)")
    info["nc"] = len(info["names"])
    return info


def describe_text(info) -> str:
    """One compact HTML card for the model list."""
    if info.get("error"):
        return f"<b>{info['label']}</b> - <span style='color:#d03b3b'>{info['error']}</span>"
    names = list(info["names"].values())
    cls = ", ".join(names[:12]) + (f" … (+{len(names) - 12})" if len(names) > 12 else "")
    bits = [f"{info['format']}", f"{info['size_mb']} MB" if info.get("size_mb") is not None else "",
            f"input {info['imgsz']}" if info.get("imgsz") else "", info.get("task") or "",
            "batch: dynamic" if info.get("dynamic_batch") else ("batch: 1" if info.get("dynamic_batch") is False else ""),
            info.get("backend") or ""]
    meta = info.get("meta") or {}
    prov = " · ".join(f"{k} {meta[k]}" for k in ("version", "date") if k in meta)
    html = (f"<b>{info['label']}</b> <span style='color:gray'>{info['file']}</span><br>"
            f"{' · '.join(b for b in bits if b)}<br>"
            f"<b>{info['nc']}</b> classes: {cls or '-'}")
    if prov:
        html += f"<br><span style='color:gray'>{prov}</span>"
    for w in info.get("warnings", []):
        html += f"<br><span style='color:#c77700'>⚠ {w}</span>"
    return html
