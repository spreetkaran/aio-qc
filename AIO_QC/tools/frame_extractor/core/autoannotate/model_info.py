# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
What a model file says about itself — its classes, its image size, its task — without running it (Qt-free).

  .onnx  ultralytics export metadata ("names", "imgsz", "task"), else the input shape; classes from a sidecar file
         (<model>.names / .txt / .yaml / data.yaml / classes.txt) when the export has none.
  .pt    read straight from the checkpoint's pickle WITHOUT torch: every class it refers to is replaced by an inert
         stand-in, so nothing in the file is executed, and the plain data (train_args, the model's `names`) is read.
         Ultralytics checkpoints store the training image size in train_args["imgsz"].

    info(path) -> {"names": {id: name}, "imgsz": int | None, "task": str, "kind": "onnx" | "pt" | ...,
                   "error": str}
"""
from __future__ import annotations

import ast
import io
import os
import pickle
import zipfile
from AIO_QC.core.diag import ignored as _ignored


class _Inert:
    """Stand-in for any class a checkpoint refers to: keeps the pickled state, runs nothing."""

    def __new__(cls, *a, **k):
        return object.__new__(cls)

    def __init__(self, *a, **k):
        self.__dict__["_args"] = a

    def __setstate__(self, state):
        self.__dict__["_state"] = state

    def __call__(self, *a, **k):
        return _Inert()

    def __setitem__(self, k, v):          # SETITEM(S) on a stand-in dict subclass
        self.__dict__.setdefault("_items", {})[k] = v

    def append(self, v):                  # APPEND(S) on a stand-in list subclass
        self.__dict__.setdefault("_list", []).append(v)

    def extend(self, vs):
        self.__dict__.setdefault("_list", []).extend(vs)


_SAFE = {("builtins", n) for n in ("dict", "list", "set", "frozenset", "tuple", "int", "float", "complex", "str",
                                   "bytes", "bytearray", "bool", "slice")} | {("collections", "OrderedDict")}


class _Reader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _SAFE:
            import builtins
            import collections
            return getattr(collections if module == "collections" else builtins, name)
        return _Inert

    def persistent_load(self, pid):       # tensors' storages: not needed
        return None


def _state(obj):
    if isinstance(obj, _Inert):
        st = obj.__dict__.get("_state")
        if isinstance(st, tuple) and st and isinstance(st[0], dict):     # (dict, slotstate)
            st = st[0]
        return st if isinstance(st, dict) else {}
    return obj if isinstance(obj, dict) else {}


def _names(v):
    if isinstance(v, dict):
        try:
            return {int(k): str(n) for k, n in v.items()}
        except (TypeError, ValueError):
            return {}
    if isinstance(v, (list, tuple)):
        return {i: str(n) for i, n in enumerate(v)}
    if isinstance(v, str):
        try:
            return _names(ast.literal_eval(v))
        except (ValueError, SyntaxError):
            return {}
    return {}


def _imgsz(v):
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return None
    if isinstance(v, (list, tuple)):
        v = max((int(x) for x in v if isinstance(x, (int, float))), default=0)
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v if 16 <= v <= 16384 else None


def _pt_checkpoint(path):
    with open(path, "rb") as f:
        head = f.read(4)
    if head[:2] == b"PK":
        with zipfile.ZipFile(path) as z:
            pkl = next((n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl"), None)
            if pkl is None:
                raise ValueError("not a PyTorch checkpoint (no data.pkl)")
            return _Reader(io.BytesIO(z.read(pkl))).load()
    with open(path, "rb") as f:                # legacy torch.save: magic, protocol, sys info, then the object
        r = _Reader(f)
        first = r.load()
        if isinstance(first, int):
            r.load()
            r.load()
            return r.load()
        return first


def _pt_info(path):
    ck = _pt_checkpoint(path)
    ck = ck if isinstance(ck, dict) else _state(ck)
    targs = ck.get("train_args") if isinstance(ck.get("train_args"), dict) else {}
    names, task, imgsz = {}, str(targs.get("task") or ""), _imgsz(targs.get("imgsz"))
    for key in ("model", "ema"):
        st = _state(ck.get(key))
        if not st:
            continue
        names = names or _names(st.get("names"))
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        imgsz = imgsz or _imgsz(args.get("imgsz"))
        task = task or str(args.get("task") or st.get("task") or "")
        yaml = st.get("yaml") if isinstance(st.get("yaml"), dict) else {}
        if not names and yaml.get("names") is not None:
            names = _names(yaml.get("names"))
    return {"names": names, "imgsz": imgsz, "task": task or "detect"}


def _onnx_info(path):
    meta, shape = {}, None
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        meta = {p.key: p.value for p in m.metadata_props}
        dims = m.graph.input[0].type.tensor_type.shape.dim
        shape = [d.dim_value if d.HasField("dim_value") else None for d in dims]
    except ImportError:
        import onnxruntime as ort
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        meta = dict(s.get_modelmeta().custom_metadata_map)
        shape = [d if isinstance(d, int) else None for d in s.get_inputs()[0].shape]
    imgsz = _imgsz(meta.get("imgsz"))
    if imgsz is None and shape and len(shape) == 4 and shape[2] and shape[3]:
        imgsz = max(int(shape[2]), int(shape[3]))
    return {"names": _names(meta.get("names")), "imgsz": imgsz, "task": meta.get("task") or "detect",
            "dynamic_size": not (shape and len(shape) == 4 and shape[2] and shape[3])}


def info(path: str) -> dict:
    out = {"names": {}, "imgsz": None, "task": "", "kind": os.path.splitext(path)[1].lower().lstrip("."),
           "error": ""}
    if not path or not os.path.isfile(path):
        out["error"] = "file not found"
        return out
    try:
        if out["kind"] == "onnx":
            out.update(_onnx_info(path))
        elif out["kind"] == "pt":
            out.update(_pt_info(path))
        else:
            out["error"] = "classes and image size are read when the model loads"
    except Exception as exc:                   # a damaged or unusual file: the model loader has the last word
        _ignored(exc)
        out["error"] = f"could not read the file's details ({type(exc).__name__}: {exc})"
    if not out["names"]:
        try:
            from AIO_QC.tools.frame_extractor.core.detector import _sidecar_names
            out["names"] = _sidecar_names(path) or {}
        except Exception as _exc:
            _ignored(_exc)
    return out
