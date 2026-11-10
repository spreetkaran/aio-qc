# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Where saves go — one rule for every Save / Export dialog in every mode (Settings ▸ General ▸ Saving).

    path, _f = ask_save(parent, "Export CSV", "issues.csv", "CSV (*.csv)", kind="exports")
    folder = ask_dir(parent, "Export folder", kind="datasets")

The folder a dialog starts in, first match wins:
    1. the folder last used for this kind of file (Settings: "Start where I saved last", on by default)
    2. the folder set for this kind (Screenshots · Exports & lists · Reports · Images · Dataset outputs)
    3. the default folder for saved files (one setting for all kinds)
    4. the folder given by the feature (e.g. next to the dataset), else the Documents folder
Screenshots saved with E have no dialog: they go to folder_for("screenshots") (default Pictures/AIO QC Screenshots).
"""
from __future__ import annotations

from pathlib import Path

KINDS = {   # key -> (title, what goes there)
    "screenshots": ("Screenshots", "E / Shift+E and batch screenshots"),
    "exports": ("Exports & lists", "CSV / JSON / HTML exports — result tables, issues, rules, views, label maps, "
                                   "guidelines, flagged lists, settings, data tables and scripts"),
    "reports": ("Reports", "QC report, review report, model report cards, OCR reads, data profiles"),
    "images": ("Images", "sketches, ROI crops, plotted images and charts"),
    "datasets": ("Dataset outputs", "Export & Split, copies of flagged files, hand-off batches, exported originals"),
}
KEY = {k: f"saves/{k}" for k in KINDS}
KEY["screenshots"] = "screenshot/dir"          # kept from 2.3.0 betas
DEFAULT_KEY = "saves/default"
REMEMBER_KEY = "saves/remember_last"


def _settings():
    from AIO_QC.utils import app_settings
    return app_settings()


def _dir(v) -> Path | None:
    if not v:
        return None
    p = Path(str(v)).expanduser()
    return p if p.is_dir() else None


def documents() -> Path:
    from PyQt6.QtCore import QStandardPaths
    d = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    return Path(d) if d else Path.home()


def default_screenshots() -> Path:
    from PyQt6.QtCore import QStandardPaths
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation) or str(Path.home())
    return Path(base) / "AIO QC Screenshots"


def remember_on(st=None) -> bool:
    st = _settings() if st is None else st
    try:
        return st.value(REMEMBER_KEY, True, type=bool)
    except (TypeError, ValueError):
        return True


def folder_for(kind: str, st=None) -> Path | None:
    """The folder configured for this kind (its own, else the default for saves); None = not configured.
    Screenshots: never None (their default is Pictures/AIO QC Screenshots, created when first used)."""
    st = _settings() if st is None else st
    own = st.value(KEY[kind], "") or ""
    if own:
        return Path(str(own)).expanduser()
    base = st.value(DEFAULT_KEY, "") or ""
    if kind == "screenshots":
        return (Path(str(base)).expanduser() / "AIO QC Screenshots") if base else default_screenshots()
    return Path(str(base)).expanduser() if base else None


def start_dir(kind: str, fallback=None, st=None) -> Path:
    st = _settings() if st is None else st
    if remember_on(st):
        last = _dir(st.value(f"saves/last/{kind}", ""))
        if last is not None:
            return last
    conf = folder_for(kind, st)
    if conf is not None and (conf.is_dir() or kind == "screenshots"):
        return conf
    fb = _dir(fallback)
    return fb if fb is not None else documents()


def remember(kind: str, path, st=None):
    """After a successful save: this kind of file starts here next time (when that setting is on)."""
    if not path:
        return
    st = _settings() if st is None else st
    p = Path(str(path))
    st.setValue(f"saves/last/{kind}", str(p if p.is_dir() else p.parent))


def start_path(kind: str, name: str = "", fallback=None) -> str:
    d = start_dir(kind, fallback)
    return str(d / Path(name).name) if name else str(d)


def ask_save(parent, caption: str, name: str = "", filt: str = "", kind: str = "exports", fallback=None,
             keep_dir: bool = False):
    """QFileDialog.getSaveFileName that starts in the right folder and remembers where you saved.
    name: a file name (or a path: with keep_dir=True its folder is used as is — e.g. a path the user typed)."""
    from PyQt6.QtWidgets import QFileDialog
    p = Path(str(name)) if name else None
    if keep_dir and p is not None and p.parent != Path(".") and p.parent.is_dir():
        start = str(p)
    else:
        if p is not None and p.parent != Path(".") and fallback is None:
            fallback = p.parent
        start = start_path(kind, p.name if p is not None else "", fallback)
    path, chosen = QFileDialog.getSaveFileName(parent, caption, start, filt)
    if path:
        remember(kind, path)
    return path, chosen


def ask_dir(parent, caption: str, kind: str = "datasets", current: str = "", fallback=None) -> str:
    """QFileDialog.getExistingDirectory that starts in the right folder (or `current`, a folder the user typed)."""
    from PyQt6.QtWidgets import QFileDialog
    start = current if current and Path(current).is_dir() else str(start_dir(kind, fallback))
    d = QFileDialog.getExistingDirectory(parent, caption, start)
    if d:
        remember(kind, Path(d) / "_")             # the folder itself
    return d


def settings_section():
    """Settings ▸ General ▸ Saving (shared by every mode)."""
    from AIO_QC.widgets.settings_dialog import Section, Setting, from_store
    st = _settings()
    rows = [
        from_store("app.save_default", st, DEFAULT_KEY, "folder", "", "Default folder for saved files",
                   description="Where Save / Export dialogs start when nothing else is set (empty: next to the "
                               "dataset, else Documents). Screenshots go into “AIO QC Screenshots” inside it.",
                   aliases=("save", "export", "location", "folder", "default", "output")),
        from_store("app.save_remember", st, REMEMBER_KEY, "bool", True, "Start where I saved last",
                   description="Each kind of file opens the folder you last saved that kind into (screenshots "
                               "saved with E always use the screenshots folder).",
                   aliases=("remember", "last folder", "recent")),
    ]
    for k, (title, what) in KINDS.items():
        rows.append(from_store(f"app.save_{k}", st, KEY[k], "folder", "", f"{title} folder",
                               description=f"{what}. Empty: the default folder for saved files"
                                           + (" (else Pictures/AIO QC Screenshots)." if k == "screenshots" else "."),
                               aliases=("save", "location", "folder", title.lower())))
    rows.append(Setting("app.save_forget", "Forget the last folders", "button", button="Forget",
                        set=lambda _v=None: [st.remove(f"saves/last/{k}") for k in KINDS],
                        description="Every dialog starts in the configured folders again.", exportable=False))
    return Section("Saving", "save", rows, "Where screenshots, exports, reports and other saved files go — in every "
                                           "mode.")
