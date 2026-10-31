# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Semantic, theme-aware colours. Nothing in the UI hard-codes a text colour: it asks for a ROLE, and the role is
derived from the live palette (or a light / dark pair), so text is readable in both themes.

    theme_color("muted")        -> QColor     hexc("warn") -> "#b26a00" (light) / "#ffb74d" (dark)
    style(widget, "color:@muted; border:1px solid @line;")   # re-applied automatically on theme change
    restyle_all()                                            # called when the theme changes

Roles: text · muted (secondary text) · faint (hints) · line (borders / separators) · surface (subtle panel bg)
       ok · warn · error · info · accent · banner_bg / banner_fg / banner_line · placeholder (empty thumbnails)
"""
from PyQt6.QtGui import QColor, QGuiApplication, QPalette
from PyQt6.QtWidgets import QApplication

STYLE_PROP = "aio_style"

_PAIRS = {  # role: (light theme, dark theme)
    "ok": ("#2e7d32", "#66bb6a"),
    "warn": ("#b26a00", "#ffb74d"),
    "error": ("#c62828", "#ef5350"),
    "info": ("#1565c0", "#64b5f6"),
    "accent": ("#00838f", "#4dd0e1"),
    "banner_bg": ("#fff3cd", "#3d3214"),
    "banner_fg": ("#5c4a00", "#ffe08a"),
    "banner_line": ("#e0c36a", "#7a6220"),
}


def _palette() -> QPalette:
    app = QGuiApplication.instance()
    return app.palette() if app is not None else QPalette()


def is_dark() -> bool:
    """What the widgets actually paint (window colour), not what the OS claims: on Windows the default style can
    report a dark colour scheme while still drawing light widgets."""
    return _palette().color(QPalette.ColorRole.Window).lightness() < 128


def _blend(a: QColor, b: QColor, t: float) -> QColor:
    """t = 0 -> a, 1 -> b."""
    return QColor(int(a.red() + (b.red() - a.red()) * t), int(a.green() + (b.green() - a.green()) * t),
                  int(a.blue() + (b.blue() - a.blue()) * t))


def theme_color(role: str) -> QColor:
    pal = _palette()
    text = pal.color(QPalette.ColorRole.WindowText)
    window = pal.color(QPalette.ColorRole.Window)
    if role == "text":
        return QColor(text)
    if role == "muted":                       # secondary text: ~4.5:1 contrast on both themes
        return _blend(text, window, 0.38)
    if role == "faint":                       # hints, placeholders
        return _blend(text, window, 0.55)
    if role == "line":                        # borders, separators
        return _blend(text, window, 0.78)
    if role == "surface":                     # subtle panel background
        return _blend(window, text, 0.05)
    if role == "placeholder":                 # empty thumbnail tiles
        return _blend(window, text, 0.10)
    if role in _PAIRS:
        return QColor(_PAIRS[role][1 if is_dark() else 0])
    return QColor(text)


def hexc(role: str) -> str:
    return theme_color(role).name()


def _render(template: str) -> str:
    out = template
    for role in sorted(("text", "muted", "faint", "line", "surface", "placeholder", *_PAIRS), key=len, reverse=True):
        out = out.replace("@" + role, hexc(role))
    return out


def style(widget, template: str):
    """setStyleSheet with @role tokens, remembered for theme changes."""
    widget.setProperty(STYLE_PROP, template)
    widget.setStyleSheet(_render(template))
    return widget


def restyle_all():
    app = QApplication.instance()
    if app is None:
        return
    for w in app.allWidgets():
        t = w.property(STYLE_PROP)
        if t:
            w.setStyleSheet(_render(t))
