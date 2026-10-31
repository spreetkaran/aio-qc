# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Continuous red → yellow → green scale (hue interpolation in HSL, so the middle is a clean yellow
instead of the muddy brown you get from RGB blending).
"""
import math

from PyQt6.QtGui import QColor


def gradient_color(t: float, lightness: float = 0.62, saturation: float = 0.70) -> QColor:
    """t = 0 → red, 0.5 → yellow, 1 → green."""
    t = 0.0 if t != t else max(0.0, min(1.0, t))         # NaN-safe clamp
    return QColor.fromHslF(t * 120 / 360, saturation, lightness)


def gradient_hex(t: float, lightness: float = 0.62, saturation: float = 0.70) -> str:
    return gradient_color(t, lightness, saturation).name()


def balance_score(share: float, n_classes: int, floor_ratio: float = 1 / 8) -> float:
    """
    Score of a class's share relative to an even split (1/N), on a log scale:
      ≥ even share → 1 (green) · ½ of even → ~0.67 · ¼ → ~0.33 (yellow/orange) · ≤ 1/8 → 0 (red)
    """
    if n_classes <= 0 or share <= 0:
        return 0.0
    ratio = share * n_classes                    # 1.0 == exactly even
    lo = math.log2(floor_ratio)
    return max(0.0, min(1.0, (math.log2(ratio) - lo) / -lo))


def imbalance_score(ratio: float, good: float = 3.0, bad: float = 50.0) -> float:
    """largest/smallest class ratio → 1 (≤ good) … 0 (≥ bad), log scale."""
    if ratio <= good:
        return 1.0
    if ratio >= bad:
        return 0.0
    return 1.0 - (math.log(ratio) - math.log(good)) / (math.log(bad) - math.log(good))


CSS_GRADIENT = ("qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {r}, stop:0.5 {y}, stop:1 {g})"
                .format(r=gradient_hex(0), y=gradient_hex(0.5), g=gradient_hex(1)))


# -----------------------------------------------------------------------------
# Theme-aware tones: every tinted surface / coloured text in the app goes through these, so nothing turns
# into white-on-white (or black-on-black) when the theme changes.
# -----------------------------------------------------------------------------
def _dark() -> bool:
    from .qt_utils import is_dark
    return is_dark()


def tone(score: float, role: str) -> str:
    """score 0 (red) … 1 (green). role: 'bg' soft surface · 'fg' readable text on the window · 'border' · 'bar'."""
    d = _dark()
    if role == "bg":
        return gradient_hex(score, 0.20 if d else 0.94, 0.40 if d else 0.75)
    if role == "fg":
        return gradient_hex(score, 0.72 if d else 0.30, 0.70 if d else 0.85)
    if role == "border":
        return gradient_hex(score, 0.45 if d else 0.55, 0.60)
    return gradient_hex(score, 0.55 if d else 0.60, 0.65)          # 'bar'


def heat(score: float, intensity: float) -> str:
    """Cell background for heatmaps (confusion matrix): stronger = more intense, readable text either way."""
    i = max(0.0, min(1.0, intensity))
    d = _dark()
    return gradient_hex(score, 0.16 + 0.22 * i if d else 0.96 - 0.30 * i, 0.55 if d else 0.70)


def emphasis_bg(widget=None) -> QColor:
    """Background for TOTAL / ALL rows: a step away from the table base colour in the theme's direction."""
    from PyQt6.QtGui import QPalette
    from PyQt6.QtWidgets import QApplication
    pal = widget.palette() if widget is not None else QApplication.palette()
    base = pal.color(QPalette.ColorRole.Base)
    return base.lighter(125) if _dark() else base.darker(106)
