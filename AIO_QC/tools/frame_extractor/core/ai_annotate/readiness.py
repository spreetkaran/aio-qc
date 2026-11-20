# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Is everything ready for AI annotate? (Qt-free) — the checks the app shows before anything else, each with what to do.

    check(cfg, env_python, connection=True) -> [Item]

Item.status: ok | warn (works, but …) | fail (blocks a run) | skip (not needed for this project's mode)
Item.fix:    what the app's button does: install_app / install_env / choose_env / enter_key / test_connection
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field

from AIO_QC.tools.frame_extractor.core.ai_annotate import config as C
from AIO_QC.tools.frame_extractor.core.ai_annotate import keys
from AIO_QC.tools.frame_extractor.core.ai_annotate import providers as PV
from AIO_QC.core.diag import ignored as _ignored

ENV_PACKAGES = {      # what each local back-end needs in the Python environment
    "grounding_dino": ["torch", "transformers", "PIL"],
    "owlv2": ["torch", "transformers", "PIL"],
    "florence2": ["torch", "transformers", "timm", "einops", "PIL"],
    "yolo_world": ["torch", "ultralytics"],
    "yolo": ["torch", "ultralytics"],
    "local_vlm": ["torch", "transformers", "accelerate", "PIL"],
    "sam2": ["torch", "ultralytics"],
}
PIP_NAMES = {"PIL": "pillow", "cv2": "opencv-python"}


@dataclass
class Item:
    key: str
    title: str
    status: str
    detail: str = ""
    fix: str = ""
    fix_args: list = field(default_factory=list)


def env_needs(cfg) -> list[str]:
    need = []
    mode = cfg["mode"]
    if mode in ("detect", "detect_vlm"):
        need += ENV_PACKAGES.get(cfg["detector"].get("type"), [])
    if cfg["connection"].get("provider") == "local" and mode != "detect":
        need += ENV_PACKAGES["local_vlm"]
    if cfg["segment"].get("type", "none") != "none":
        need += ENV_PACKAGES["sam2"]
    return list(dict.fromkeys(need))


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def check(cfg: dict, env_python: str = "", connection: bool = True, env_info: dict | None = None) -> list[Item]:
    cfg = C.normalize(cfg)
    items = []
    # 1 · the app itself
    if importlib.util.find_spec("yaml") is None:
        items.append(Item("app", "AIO QC can read project files", "fail",
                          "PyYAML is missing in the app's Python — needed to save and load AI annotate projects.",
                          "" if frozen() else "install_app", ["PyYAML"]))
    else:
        items.append(Item("app", "AIO QC can read project files", "ok", "PyYAML is installed."))
    # 2 · the Python environment (local models)
    need = env_needs(cfg)
    if not C.needs_env(cfg):
        items.append(Item("env", "Python environment", "skip",
                          "Not needed: this project only uses an online AI service."
                          + (f" (Chosen: {env_python})" if env_python else "")))
    elif not env_python or not os.path.exists(env_python):
        items.append(Item("env", "Python environment", "fail",
                          "This project runs a model on your computer — choose the Python environment with PyTorch "
                          "(the same one your .pt models use).", "choose_env"))
    else:
        info = env_info
        if info is None:
            try:
                from AIO_QC.tools.frame_extractor.core.ai_annotate.runner import probe_env
                info = probe_env(env_python)
            except Exception as exc:
                _ignored(exc)
                info = {"error": str(exc)}
        if info.get("error"):
            items.append(Item("env", "Python environment", "fail", f"It did not start: {info['error']}",
                              "choose_env"))
        else:
            pk = info.get("packages") or {}
            missing = [m for m in need if not pk.get(m)]
            have = ", ".join(f"{m} {pk[m]}" for m in need if pk.get(m))
            items.append(Item("env", "Python environment", "fail" if missing else "ok",
                              (f"Missing: {', '.join(missing)}. " if missing else "") +
                              f"Python {info.get('python', '?')}" + (f" · {have}" if have else ""),
                              "install_env" if missing else "", [PIP_NAMES.get(m, m) for m in missing]))
            gpu = info.get("devices") or []
            if info.get("cuda") or any("mps" in d for d in gpu):
                items.append(Item("gpu", "Graphics card", "ok", " · ".join(gpu)))
            else:
                items.append(Item("gpu", "Graphics card", "warn",
                                  "No GPU found in this environment (PyTorch without CUDA, or no NVIDIA card) — local "
                                  "models will run on the CPU, slowly."))
    # 3 · the AI service
    if not C.needs_api(cfg) or cfg["connection"].get("provider") == "local":
        items.append(Item("key", "API key", "skip", "Not needed: no online AI service in this project."))
        items.append(Item("service", "AI service", "skip", "Not needed."))
    else:
        con = cfg["connection"]
        name = con.get("api_key_env") or ""
        kind = PV.kind(cfg)
        if kind == "bedrock":
            ok = importlib.util.find_spec("boto3") is not None
            items.append(Item("key", "AWS access", "ok" if ok else "fail",
                              "boto3 is installed; your AWS credentials (aws configure) are used." if ok else
                              "AWS Bedrock needs the boto3 package in the app's Python.",
                              "" if ok or frozen() else "install_app", ["boto3"]))
        elif not name:
            items.append(Item("key", "API key", "ok", "This service needs no key."))
        elif keys.get(name):
            items.append(Item("key", "API key", "ok", f"Found in {name} ({keys.mask(keys.get(name))})."))
        else:
            items.append(Item("key", "API key", "fail", f"The environment variable {name} is not set. Enter the key "
                                                       f"once — it is saved as {name} for every later run.",
                              "enter_key", [name]))
        if not connection:
            items.append(Item("service", "AI service", "warn", "Not checked yet.", "test_connection"))
        elif any(i.key == "key" and i.status == "fail" for i in items):
            items.append(Item("service", "AI service", "warn", "Checked once the key is there.", "test_connection"))
        else:
            t = PV.test_connection(cfg)
            if not t["ok"]:
                items.append(Item("service", "AI service", "fail", t["message"], "test_connection"))
            else:
                msg = f"{t['message']} ({t['ms']} ms)."
                if t["model_listed"] is False:
                    items.append(Item("service", "AI service", "warn",
                                      msg + f" “{cfg['vlm']['model']}” is not in its list — check the name.",
                                      "test_connection"))
                else:
                    items.append(Item("service", "AI service", "ok", msg + (f" “{cfg['vlm']['model']}” is offered."
                                                                           if t["model_listed"] else "")))
    return items


def ready(items) -> bool:
    return not any(i.status == "fail" for i in items)
