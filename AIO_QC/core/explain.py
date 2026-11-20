# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Plain errors (Qt-free): an exception or error text -> what happened, why, and the one thing that fixes it.

    e = explain(exc_or_text)
    e.what     one sentence, no jargon ("The file could not be saved")
    e.why      the likely cause, in the user's terms ("another program has it open — Windows locks open files")
    e.fix      the button text ("Open Models & Environment") — "" when there is nothing to click
    e.action   what the button does, a key the window resolves ("model_setup", "install:pyyaml", "open_log", …)
    e.detail   the original text (shown under "Details", kept for the log)

The first rule that matches wins; rules are ordered from the most specific. Messages that are already written for
people (the app's own ValueErrors) keep their wording as `what` and only gain the fix.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Explanation:
    what: str
    why: str = ""
    fix: str = ""
    action: str = ""
    detail: str = ""


# (pattern, what, why, fix button, action) — "{0}" in a text is the pattern's first group
RULES = [
    # --- models & environment (the app's own messages first: they keep their words)
    (r"Add (at least one|a) model", None, "Model tools need at least one model, set up once per dataset.",
     "Open Models & Environment", "model_setup"),
    (r"Choose (a|the) Python environment|No Python environment selected|choose the environment",
     None, "Models run in a Python environment of yours that has PyTorch and Ultralytics.",
     "Open Models & Environment", "model_setup"),
    (r"Load the models once|No classes mapped", None,
     "The models' classes have to be matched to this dataset's labels (done automatically by name when they load).",
     "Open Models & Environment", "model_setup"),
    (r"None of the enabled model files exist|model file .* (not found|missing)", None,
     "A model file was moved or deleted since it was added.", "Open Models & Environment", "model_setup"),
    (r"The environment lacks: (.+)", "The Python environment is missing {0}.",
     "Model tools need these packages in the environment you chose.", "Open Models & Environment", "model_setup"),
    (r"No module named '([\w.\-]+)'", "The Python environment is missing the package “{0}”.",
     "A part of the work needs it and it is not installed there.", "Install it", "install:{0}"),
    (r"inference process is not running|Inference unavailable|server (exited|died|stopped)|BrokenPipe",
     "The model process stopped.", "It may have run out of memory, or the environment is broken — the log has its "
     "last lines.", "Show the log", "open_log"),
    (r"runs code from its download|trust_remote_code", None,
     "Some models ship code that runs when they load; it can do anything your account can, so each one must be "
     "allowed once, by name.", "", ""),
    # --- memory
    (r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory", "The graphics card ran out of memory.",
     "The batch or the image size is too large for this GPU.", "Open Models & Environment", "model_setup"),
    (r"MemoryError|Unable to allocate|cannot allocate memory|bad_alloc", "The computer ran out of memory.",
     "The task holds more in memory than is free — close other programs, or work on fewer files at a time.", "", ""),
    # --- files
    (r"No space left on device|disk (is )?full|ENOSPC|There is not enough space on the disk",
     "The disk is full.", "Nothing more can be written there until space is freed.", "", ""),
    (r"being used by another process|WinError 32|sharing violation", "The file is open in another program.",
     "Windows does not let a file be changed while another program has it open — close it there, then try again.",
     "Try again", "retry"),
    (r"Permission denied|Access is denied|WinError 5|PermissionError|Operation not permitted",
     "AIO QC is not allowed to change that file or folder.",
     "It is read-only, or in a protected folder (Program Files, another user's folder, a synced folder that is "
     "busy).", "", ""),
    (r"No such file or directory|FileNotFoundError|cannot find the (file|path)|WinError [23]\b|does not exist",
     "A file or folder was not found.", "It was moved, renamed or deleted — or the path was typed wrong.", "", ""),
    (r"already exists|FileExistsError", "Something with that name is already there.",
     "Nothing is overwritten — choose another name or an empty folder.", "", ""),
    (r"JSONDecodeError|Expecting (value|property name)|Unterminated string|Invalid control character",
     "A JSON file is damaged.", "It is not valid JSON (often a half-written file after a crash, or an edit by hand).",
     "", ""),
    (r"cannot identify image file|image file is truncated|Premature end of JPEG|not a valid image|decod",
     "An image could not be read.", "It is damaged or not really an image — Image–Label Pairs and Preparation ▸ "
     "Integrity find all of them.", "", ""),
    # --- network
    (r"ConnectionError|getaddrinfo|Name or service not known|urlopen error|Max retries exceeded|timed out|"
     r"HTTP Error 40\d|SSL", "A download or connection failed.",
     "No internet, a proxy or firewall in the way, or the server is down.", "Try again", "retry"),
    # --- datasets
    (r"AnnotationError|not a LabelMe|invalid annotation|class id .* (outside|exceeds)",
     "An annotation file could not be used.", "It does not follow the dataset's format (Validation lists them all).",
     "", ""),
]

_COMPILED = [(re.compile(rx, re.I), *rest) for rx, *rest in RULES]


def _text(err) -> str:
    if isinstance(err, BaseException):
        name = type(err).__name__
        msg = str(err)
        return msg if name in ("ValueError", "RuntimeError", "Exception") and msg else f"{name}: {msg}"
    return str(err or "")


def explain(err) -> Explanation:
    text = _text(err).strip()
    for rx, what, why, fix, action in _COMPILED:
        m = rx.search(text)
        if not m:
            continue
        g = (m.group(1) if m.groups() and m.group(1) else "").strip()
        if action.startswith("install:"):
            from .training import missing_module
            g = missing_module(text) or g
        own = what is None
        return Explanation(text if own else what.format(g), why, fix, action.format(g) if action else "",
                           "" if own else text)
    first = text.splitlines()[0] if text else "Something went wrong."
    return Explanation(first[:300], "", "Show the log", "open_log", text if text != first else "")
