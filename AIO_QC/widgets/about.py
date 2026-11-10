# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Help ▸ About — one dialog for every mode, laid out like LabelMe's: the app icon on the left; on the right the
name (bold), what it is for, the version and the author; OK at the bottom right."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QVBoxLayout

AUTHOR = "Preetkaran Singh"          # fallback only: the installed package's metadata is the source (author())
DIST = "aio-qc"                      # pyproject.toml [project] name


def author() -> str:
    """The author as recorded in the installed package's metadata (pyproject.toml → authors), so editing this file
    alone does not change what About shows; running from a source checkout (not installed) uses AUTHOR."""
    try:
        from importlib.metadata import PackageNotFoundError, metadata
        try:
            md = metadata(DIST)
        except PackageNotFoundError:
            return AUTHOR
        from AIO_QC import __version__
        if (md.get("Version") or "") != __version__:
            return AUTHOR                              # metadata of another build lying around (e.g. a stale
                                                       # egg-info next to a source checkout): not this code's
        name = (md.get("Author") or "").strip()
        if not name:                                   # "Name <mail>" style lives in Author-email
            name = (md.get("Author-email") or "").split("<")[0].strip().strip('"')
        return name or AUTHOR
    except Exception:
        return AUTHOR


class AboutDialog(QDialog):
    def __init__(self, parent, name: str, purpose: str, icon):
        super().__init__(parent)
        from AIO_QC import __version__
        from AIO_QC.utils import IconLabel
        self.setWindowTitle(f"About {name}")
        self.setWindowIcon(icon)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 14)
        root.setSpacing(12)
        row = QHBoxLayout()
        row.setSpacing(22)
        self.icon = IconLabel(icon, 96)                       # painted at the screen's resolution: sharp anywhere
        row.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(12)
        self.title = QLabel(name)
        f = self.title.font()
        f.setPointSizeF(f.pointSizeF() * 1.35)
        f.setBold(True)
        self.title.setFont(f)
        self.purpose = QLabel(purpose)
        self.purpose.setWordWrap(True)
        self.purpose.setMinimumWidth(360)
        self.version = QLabel(f"Version: {__version__}")
        self.author = QLabel(f"Author: {author()}")
        self.source = QLabel("Source : <a href='https://github.com/spreetkaran/aio-qc'>GitHub</a>")
        self.source.setOpenExternalLinks(True)
        self.source.setToolTip("https://github.com/spreetkaran/aio-qc")
        for w in (self.title, self.purpose, self.version, self.author):
            w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            col.addWidget(w)
        col.addWidget(self.source)
        col.addStretch(1)
        row.addLayout(col, 1)
        root.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        bb.accepted.connect(self.accept)
        root.addWidget(bb)
        self.setSizeGripEnabled(False)
        self.adjustSize()
        self.setFixedSize(self.sizeHint())


def show_about(parent, name: str, purpose: str, icon):
    dlg = AboutDialog(parent, name, purpose, icon)
    dlg.exec()
    return dlg
