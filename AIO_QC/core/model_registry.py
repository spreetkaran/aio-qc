# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
One model registry for the whole app (Qt-free).

Where models live:
  * per dataset - <dataset>/.aio_qc/models.json (Image QC ▸ Models & Environment): each entry has id, name, path,
    enabled, imgsz, names (the model's classes) and class_map {model class id: dataset label, "" = ignore};
  * app-wide    - <AIO QC data>/model_registry.json: every weights file added anywhere (the Frame Extractor's
    detections, Testing without an Image QC dataset), with the same fields; class_map_by_name {model class name:
    dataset label} carries a mapping to another dataset with the same label names.
One environment for .pt / .engine models: the app setting inference/python (Models & Environment); one set of
inference options (device, FP16, SAHI) from the same place.

    all_models(root)                -> dataset models first, then app-wide ones not in the dataset (dedup by path)
    register(path, root=None, ...)  -> add weights (to the dataset when one is open, and always to the app list)
    spec_for(entry, opts, env)      -> core.testing.runner.ModelSpec (Testing / Frame Extractor) with the class map
    shared_cache_path(root)         -> <dataset>/.aio_qc/predictions.db: ONE prediction cache per dataset (Testing
                                       writes it; Model Check reads it before running a model)
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from AIO_QC.core.appdirs import data_home

APP_FILE = "model_registry.json"


def _app_file() -> Path:
    return data_home() / APP_FILE


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(p: Path, d: dict):
    from AIO_QC.core.io import atomic_write_text
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(p, json.dumps(d, indent=1, ensure_ascii=False))


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p)) if p else ""


def new_entry(path: str, name: str | None = None, **kw) -> dict:
    e = {"id": uuid.uuid4().hex[:8], "name": name or Path(path).stem, "path": os.path.abspath(path), "enabled": True,
         "imgsz": 0, "class_map": {}, "names": {}, "task": "", "train_imgsz": None}
    e.update(kw)
    return e


# ------------------------------------------------------------------ dataset / app lists
def dataset_models(root) -> list[dict]:
    if not root:
        return []
    from AIO_QC.core.state import load_models_config
    return list(load_models_config(Path(root)).get("models", []) or [])


def save_dataset_models(root, models: list[dict]):
    from AIO_QC.core.state import load_models_config, save_models_config
    cfg = load_models_config(Path(root)) or {}
    cfg["version"] = cfg.get("version", 1)
    cfg["models"] = models
    save_models_config(Path(root), cfg)


def app_models() -> list[dict]:
    return list(_read(_app_file()).get("models", []) or [])


def save_app_models(models: list[dict]):
    _write(_app_file(), {"version": 1, "models": models, "updated": time.time()})


def all_models(root=None) -> list[dict]:
    """Dataset models (with their class maps) first, then the app-wide ones not already in the dataset."""
    out, seen = [], set()
    for m in dataset_models(root):
        out.append(dict(m, source="dataset"))
        seen.add(_norm(m.get("path", "")))
    for m in app_models():
        if _norm(m.get("path", "")) not in seen:
            out.append(dict(m, source="app"))
            seen.add(_norm(m.get("path", "")))
    return out


def find(path: str, root=None) -> dict | None:
    n = _norm(path)
    return next((m for m in all_models(root) if _norm(m.get("path", "")) == n), None)


def register(path: str, root=None, name: str | None = None, **kw) -> dict:
    """Add weights: to the dataset's models.json when a dataset is given, and (always) to the app-wide list.
    Returns the entry (an existing one when the file is known)."""
    path = os.path.abspath(path)
    n = _norm(path)
    apps = app_models()
    entry = next((m for m in apps if _norm(m.get("path", "")) == n), None)
    if entry is None:
        entry = new_entry(path, name, **kw)
        apps.append({k: v for k, v in entry.items() if k != "source"})
        save_app_models(apps)
    if root:
        ds = dataset_models(root)
        hit = next((m for m in ds if _norm(m.get("path", "")) == n), None)
        if hit is None:
            hit = {k: v for k, v in entry.items() if k != "source"}
            hit["class_map"] = class_map_for_dataset(entry, None)
            ds.append(hit)
            save_dataset_models(root, ds)
        return dict(hit, source="dataset")
    return dict(entry, source="app")


def remember_names(path: str, names: dict, root=None):
    """The model's classes, learnt when it was loaded (Testing / Frame Extractor): stored on every entry of it."""
    names = {str(k): str(v) for k, v in (names or {}).items()}
    if not names:
        return
    n = _norm(path)
    apps = app_models()
    changed = False
    for m in apps:
        if _norm(m.get("path", "")) == n and m.get("names") != names:
            m["names"] = names
            changed = True
    if changed:
        save_app_models(apps)
    if root:
        ds = dataset_models(root)
        dch = False
        for m in ds:
            if _norm(m.get("path", "")) == n and not m.get("names"):
                m["names"] = names
                dch = True
        if dch:
            save_dataset_models(root, ds)


# ------------------------------------------------------------------ class mapping (by name)
def class_map_for_dataset(entry: dict, labels) -> dict:
    """{model class id: dataset label} — the entry's own map, else its by-name map, else ids whose name is a
    dataset label. labels=None: just carry the by-name map over (ids resolved once names are known)."""
    if entry.get("class_map"):
        return dict(entry["class_map"])
    by_name = entry.get("class_map_by_name") or {}
    names = entry.get("names") or {}
    out = {}
    for cid, nm in names.items():
        if nm in by_name:
            out[str(cid)] = by_name[nm]
        elif labels is not None and nm in labels:
            out[str(cid)] = nm
    return out


def names_after_map(model_names: dict, class_map: dict) -> dict:
    """The model's {id: name} with the class map applied ({id: dataset label}); "" = ignore that class."""
    if not class_map:
        return dict(model_names or {})
    return {int(k): (class_map.get(str(k), v) if str(k) in class_map else v) for k, v in (model_names or {}).items()}


# ------------------------------------------------------------------ runtime
def effective_imgsz(entry: dict) -> int:
    return int(entry.get("imgsz") or entry.get("train_imgsz") or 0)


def spec_for(entry: dict, opts: dict | None = None, env_python: str = "", **over):
    """Registry entry -> ModelSpec for Testing / the Frame Extractor. The class map renames the model's classes
    to the dataset's labels before they are matched by name (it does not change the cached predictions)."""
    from AIO_QC.core.testing.runner import ModelSpec
    opts = opts or {}
    dev = str(opts.get("device") or "auto")
    sahi = {"slice": int(opts.get("slice", 640)), "overlap": float(opts.get("overlap", 0.2))} \
        if opts.get("mode") == "sahi" else None
    kw = dict(path=os.path.abspath(entry["path"]), label=entry.get("name") or "", imgsz=effective_imgsz(entry),
              half=bool(opts.get("half")) and dev not in ("cpu", "mps"), sahi=sahi, env_python=env_python or "",
              device=dev, class_map=dict(entry.get("class_map") or {}))
    kw.update(over)
    return ModelSpec(**kw)


def shared_cache_path(root) -> Path:
    from AIO_QC.core.state import state_dir
    return state_dir(Path(root)) / "predictions.db"


def qc_image_key(path: str, root) -> str:
    """Prediction-cache key of a dataset image: its path in the dataset + size + mtime (an edited image is new)."""
    try:
        rel = os.path.relpath(path, root).replace(os.sep, "/")
    except ValueError:
        rel = os.path.abspath(path)
    try:
        st = os.stat(path)
        return f"{rel}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        return rel


def shared_lookup(root, path: str, imgsz: int, sahi=None):
    """Testing's predictions for these weights in the dataset's shared cache, usable by Model Check: same file, same
    SAHI setting, run at the same image size and stored down to a confidence of at most 0.05. -> (PredCache, model
    key) or None. One way only: Model Check's own predictions (conf >= 0.05) never enter this cache - mAP needs
    Testing's low floor."""
    p = shared_cache_path(root)
    if not p.is_file():
        return None
    import sqlite3
    try:
        con = sqlite3.connect(str(p), timeout=5)
        rows = con.execute("SELECT key, path, spec, info FROM models").fetchall()
        con.close()
    except sqlite3.Error:
        return None
    want = _norm(path)
    best = None
    for key, mpath, spec, info in rows:
        if _norm(mpath or "") != want:
            continue
        try:
            spec, info = json.loads(spec or "{}"), json.loads(info or "{}")
        except ValueError:
            continue
        if float(spec.get("conf", 1)) > 0.05 or bool(spec.get("sahi")) != bool(sahi):
            continue
        if sahi and (spec["sahi"].get("slice"), spec["sahi"].get("overlap")) != (sahi.get("slice"), sahi.get("overlap")):
            continue
        used = info.get("imgsz_used")
        used = max(used) if isinstance(used, list) and used else used
        if int(used or spec.get("imgsz") or 0) != int(imgsz or 0):
            continue
        best = key
    if best is None:
        return None
    from AIO_QC.core.testing.runner import PredCache
    return PredCache(str(p)), best
