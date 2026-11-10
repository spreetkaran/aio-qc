# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Inter, the typeface of the Model Lab's charts and plotted images (box labels), and of the HTML reports (bundled in AIO QC/fonts, SIL OFL 1.1).

  load()          register the bundled weights with Qt once; returns the family name ("Inter"), or "" when the
                  files are missing (the system font is used then - nothing breaks)
  ui_font(...)    a QFont in Inter at the app's normal size (tabular figures, so numbers line up in tables)
  (the HTML reports embed a small web subset themselves: core.lab.report.font_css - Qt-free)
"""
import os

from AIO_QC.core.testing.textdraw import FONT_DIR   # AIO QC's one font folder
FAMILY = "Inter"
_loaded = None


def load():
    global _loaded
    if _loaded is not None:
        return _loaded
    _loaded = ""
    try:
        from PyQt6.QtGui import QFontDatabase
        fams = set()
        for w in ("Regular", "Medium", "SemiBold", "Bold"):
            p = os.path.join(FONT_DIR, f"Inter-{w}.otf")
            if os.path.isfile(p):
                fid = QFontDatabase.addApplicationFont(p)
                if fid >= 0:
                    fams.update(QFontDatabase.applicationFontFamilies(fid))
        if FAMILY in fams or any(f.startswith(FAMILY) for f in fams):
            _loaded = FAMILY
    except Exception:
        _loaded = ""
    return _loaded


def ui_font(base=None, scale=1.0, weight=None):
    """Inter (if available) at the size of `base` (a QFont, default: the application font) x scale."""
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import QApplication
    f = QFont(base) if base is not None else QFont(QApplication.font())
    fam = load()
    if fam:
        f.setFamilies([fam] + [x for x in f.families() if x != fam])
    if scale != 1.0 and f.pointSizeF() > 0:
        f.setPointSizeF(f.pointSizeF() * scale)
    if weight is not None:
        f.setWeight(weight)
    try:
        f.setFeature(QFont.Tag("tnum"), 1)            # tabular figures (Qt >= 6.7)
    except Exception:
        pass
    return f


