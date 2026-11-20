# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Models that run their own code. Some Hugging Face models (Florence-2, Moondream, several chat VLMs) ship Python code
that transformers runs when the model loads (`trust_remote_code`). That code can do anything your user account can, so
it is never allowed by default: each such model must be allowed once, by name, on this computer.

The list lives in the app's data folder — never in a project's config file, so a config someone shares cannot allow a
model on your computer.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

FILE = "trusted_model_code.json"
NEEDS = ("florence", "moondream")          # model names known to need their own code to load


def _file() -> Path:
    from AIO_QC.core.appdirs import data_home
    return data_home() / FILE


def _norm(model: str) -> str:
    m = str(model or "").strip()
    return os.path.normcase(os.path.abspath(m)) if os.path.isdir(m) else m.lower()


def _load() -> list:
    try:
        d = json.loads(_file().read_text(encoding="utf-8"))
        return [x for x in d.get("models", []) if isinstance(x, str)]
    except (OSError, ValueError):
        return []


def needs_own_code(kind: str, model: str) -> bool:
    """Whether this detector type / model is known to need its own code (the user can still allow any model)."""
    low = f"{kind} {model}".lower()
    return any(n in low for n in NEEDS)


def trusted(model: str) -> bool:
    return bool(model) and _norm(model) in _load()


def set_trusted(model: str, on: bool) -> None:
    cur = [m for m in _load() if m != _norm(model)]
    if on and model:
        cur.append(_norm(model))
    p = _file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"models": cur}, indent=1), encoding="utf-8")


def refusal(model: str) -> str:
    return (f"“{model}” runs code from its download when it loads, and it has not been allowed on this computer. "
            "Allow it in AI annotate ▸ Who annotates (“Allow this model to run its own code”) — only for a model "
            "from a source you trust.")
