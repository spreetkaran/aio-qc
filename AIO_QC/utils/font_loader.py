# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
from functools import lru_cache
from pathlib import Path

from loguru import logger
from PyQt6.QtGui import QFont, QFontDatabase

FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"


@lru_cache(maxsize=None)
def _family(font_name: str) -> str:
    """Register a bundled font ONCE and return its family (the old code re-registered per shape)."""
    path = FONTS_DIR / font_name
    if path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(path))
        families = QFontDatabase.applicationFontFamilies(font_id) if font_id != -1 else []
        if families:
            return families[0]
        logger.error(f"Font could not be loaded: {font_name}")
    else:
        logger.error(f"Font not found: {path}")
    return QFont().defaultFamily()


def load_font(font_name: str, size: int = 10, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont(_family(font_name))
    font.setPointSize(size)
    font.setWeight(weight)
    return font
