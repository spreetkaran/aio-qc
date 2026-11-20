# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Text templates for sketches — Sketch on Image's Text tool and the auto-sketch label at each arrow share one list.

A template is plain text with optional fields, filled in when it is used:
    {label}   the shape's label (auto-sketch: the shape the arrow points at; Sketch: the shape under the click)
    {n}       the shape's number in its file (#n, as in the Annotations panel)
    {file}    the file name
    {date}    today's date (YYYY-MM-DD)
Unknown fields are left as typed. The list is kept in the app settings (sketch/text_templates); the built-in one
comes back with "Reset".

Looks — how a text is drawn, also a list to pick from (LOOKS): a style (outlined, on a chip, with a shadow, plain)
and its colours. A colour can be fixed, the sketch colour, or "auto": chosen for the background the text lands on —
the candidate with the highest WCAG contrast against that part of the image; on a busy background (light and dark
mixed) it keeps to black or white, which the outline keeps readable — or pick a chip, readable on anything. Colours are worked out once, when the text is
placed, and stored with it: a drawing always looks the same afterwards.
"""
from __future__ import annotations

import json
import re
import time

DEFAULT_TEMPLATES = [
    "{label}",
    "Wrong label — should be ?",
    "Loose box",
    "Missing annotation",
    "Marked twice",
    "Overlapping boxes",
    "Check this",
    "{label} #{n}",
]
FIELDS = ("label", "n", "file", "date")
SETTING = "sketch/text_templates"
TEXT_SIZES = [("S", 0.7), ("M", 1.0), ("L", 1.4), ("XL", 2.0)]     # auto-sketch: × the arrow-based size
_FIELD = re.compile(r"\{(\w+)\}")


def expand(template: str, label: str | None = None, n: int | None = None, file: str | None = None) -> str:
    vals = {"label": label, "n": None if n is None else str(n), "file": file, "date": time.strftime("%Y-%m-%d")}

    def sub(m):
        v = vals.get(m.group(1))
        return m.group(0) if m.group(1) not in vals else (v if v is not None else "")
    return _FIELD.sub(sub, str(template or "")).strip()


def load_templates(settings) -> list[str]:
    try:
        raw = settings.value(SETTING, "", type=str)
        vals = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        vals = None
    if not isinstance(vals, list) or not vals:
        return list(DEFAULT_TEMPLATES)
    return [str(v) for v in vals if str(v).strip()] or list(DEFAULT_TEMPLATES)


def save_templates(settings, items: list[str]):
    items = [str(v).strip() for v in items if str(v).strip()]
    seen, out = set(), []
    for v in items:                                   # no duplicates, order kept
        if v not in seen:
            seen.add(v)
            out.append(v)
    settings.setValue(SETTING, json.dumps(out if out != DEFAULT_TEMPLATES else []))


# ====================================================================================================== looks
STYLES = ("outline", "chip", "shadow", "plain")
LOOKS = [  # key, name, style, text colour, chip colour ("auto" | "sketch" | "#rrggbb")
    ("auto_outline", "Outlined · colour for the background", "outline", "auto", None),
    ("auto_chip", "Chip · light or dark for the background", "chip", "auto", "auto"),
    ("sketch_outline", "Outlined · sketch colour", "outline", "sketch", None),
    ("alert_chip", "Alert · white on red", "chip", "#ffffff", "#e5332a"),
    ("note_chip", "Note · black on yellow", "chip", "#111111", "#ffd60a"),
    ("info_chip", "Info · white on blue", "chip", "#ffffff", "#0a6cff"),
    ("shadow", "Shadow · colour for the background", "shadow", "auto", None),
    ("plain", "Plain · sketch colour", "plain", "sketch", None),
]
LOOK_KEYS = [k for k, *_r in LOOKS]
DEFAULT_LOOK = "auto_outline"
AUTO_TEXT = ["#ffffff", "#111111", "#ffd60a", "#ff3b30", "#00c7be", "#0a6cff"]   # candidates for "auto" text
CHIP_DARK, CHIP_LIGHT = "#1c1c1e", "#f5f5f7"


def look(key: str) -> dict:
    for k, name, style, fg, bg in LOOKS:
        if k == key:
            return {"key": k, "name": name, "style": style, "fg": fg, "bg": bg}
    return look(DEFAULT_LOOK)


def _rgb(c) -> tuple[float, float, float]:
    c = str(c).lstrip("#")
    if len(c) == 8:                                  # #aarrggbb (Qt's HexArgb)
        c = c[2:]
    if len(c) != 6:
        return (1.0, 0.23, 0.19)
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def luminance(rgb) -> float:
    """WCAG relative luminance of an (r, g, b) in 0..1."""
    def ch(v):
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(c1, c2) -> float:
    """WCAG contrast ratio (1 … 21) of two colours (hex or rgb 0..1)."""
    a = luminance(_rgb(c1) if isinstance(c1, str) else c1)
    b = luminance(_rgb(c2) if isinstance(c2, str) else c2)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def best_contrast(bg_rgb, candidates=AUTO_TEXT) -> str:
    return max(candidates, key=lambda c: contrast(c, bg_rgb))


def halo_for(fg: str) -> str:
    """The outline around a text: black for light text, white for dark text."""
    return "#000000" if luminance(_rgb(fg)) > 0.35 else "#ffffff"


def resolve_look(key: str, bg_mean, bg_spread: float = 0.0, sketch_color: str = "#ff3b30") -> dict:
    """Concrete colours for a look on a background (its mean colour, rgb 0..1, and how busy it is, 0..1):
    {"ts": style, "c": text colour, "bg": chip colour or None}."""
    lk = look(key)
    style = lk["style"]
    if style == "chip":
        bg = lk["bg"]
        if bg == "auto":                              # a chip that stands out from the image
            bg = CHIP_DARK if luminance(bg_mean) > 0.3 else CHIP_LIGHT
        elif bg == "sketch":
            bg = sketch_color
        fg = lk["fg"]
        if fg == "auto":
            fg = best_contrast(_rgb(bg), ["#ffffff", "#111111"])
        elif fg == "sketch":
            fg = sketch_color
        return {"ts": "chip", "c": fg, "bg": bg}
    fg = lk["fg"]
    if fg == "auto":
        fg = best_contrast(bg_mean)
        if style == "outline" and bg_spread > 0.28:   # a busy background: the outline carries the contrast
            fg = best_contrast(bg_mean, ["#ffffff", "#111111"])
    elif fg == "sketch":
        fg = sketch_color
    return {"ts": style, "c": fg, "bg": None}
