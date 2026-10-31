# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Where AIO QC keeps its own files (Qt-free) — the same rules for a pip install, a standalone .exe and a portable folder.

    is_frozen()   running from a PyInstaller build (standalone exe / portable folder)
    app_dir()     the folder of the executable (frozen) or of the project (source / pip)
    is_portable() a portable build: a file named `portable` (any content) or a folder `data/` sits next to the
                  executable, or AIO_QC_PORTABLE=1 is set
    data_home()   portable → <app dir>/data   ·   otherwise the per-user folder:
                  Windows %LOCALAPPDATA%\\AIO_QC · macOS ~/Library/Application Support/AIO_QC · Linux ~/.local/share/aio_qc
    log_dir()     where aio_qc_tool.log goes: the current folder when run from source (as before), data_home()/logs
                  for frozen builds (an installed .exe may sit in a read-only folder)
    settings_file()  portable builds keep their settings in data/settings.ini (else: the OS store, as before)
    app_python()  the Python that runs AIO QC — None for frozen builds (the .exe cannot create environments;
                  another Python is found instead)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent


def is_portable() -> bool:
    if os.environ.get("AIO_QC_PORTABLE", "").strip() in ("1", "true", "yes"):
        return True
    if not is_frozen():
        return False
    d = app_dir()
    return (d / "portable").exists() or (d / "data").is_dir()


def _user_home() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "AIO_QC"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "AIO_QC"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "aio_qc"


def data_home() -> Path:
    return app_dir() / "data" if is_portable() else _user_home()


def log_dir() -> Path:
    if not is_frozen():
        return Path.cwd()
    d = data_home() / "logs"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Path.cwd()
    return d


def settings_file() -> Path | None:
    """An INI file for portable builds, None = the OS settings store (registry / plist / ~/.config)."""
    if not is_portable():
        return None
    p = data_home() / "settings.ini"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def app_python() -> str | None:
    return None if is_frozen() else sys.executable
