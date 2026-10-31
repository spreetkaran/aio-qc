# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Qt helpers — the labelme `qt.py` utilities ported to PyQt6 and extended for AIO QC.
Every button, action, menu and toolbar in the app is built through these.

    newIcon(name)                                   icons/<name>.svg (duotone, theme ink), PNG fallback
    shapeIcon(shape_type, color)                    shape-type glyph in a label colour
    refresh_icons()                                 re-skin every tagged action / button (theme change)
    newButton(text, icon=None, slot=None, tip=None)
    newAction(parent, text, slot=None, shortcut=None, icon=None, tip=None, checkable=False, enabled=True,
              checked=False)
    addActions(widget, [QAction | QMenu | QWidget | None | {"menu": title, "actions": [...], "icon": name}])
    labelValidator()                                no leading whitespace in typed labels
    struct(**kw)                                    attribute bag (action / menu registries)
    distance(p), distancetoline(point, line)        geometry for picking thin shapes
    fmtShortcut(text)                               "Ctrl+Shift+S" -> "<b>Ctrl</b>+<b>Shift</b>+<b>S</b>"
"""
from functools import lru_cache
from math import sqrt
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QByteArray, QRect, QRectF, QRegularExpression, QSize, Qt
from PyQt6.QtGui import (QAction, QGuiApplication, QIcon, QIconEngine, QKeySequence, QPainter, QPalette, QPixmap,
                         QRegularExpressionValidator)
from PyQt6.QtWidgets import QAbstractButton, QApplication, QMenu, QPushButton, QToolBar, QWidget

try:                                   # ships with PyQt6; without it icons fall back to PNG files if present
    from PyQt6.QtSvg import QSvgRenderer
except ImportError:  # pragma: no cover
    QSvgRenderer = None

ICONS_DIR = Path(__file__).resolve().parent.parent / "icons"
ICON_PROP = "aio_icon"            # actions / buttons remember their icon name for theme changes

# Leading emoji in button / action text -> icon (the emoji is removed from the text)
EMOJI_ICONS = {
    "🗑": "delete", "▶": "run", "⏹": "stop", "⤓": "export", "📤": "export", "📥": "open", "👁": "eye",
    "✏️": "edit", "✏": "edit", "🩹": "edit", "⚑": "review-flag", "✓": "review-ok", "✔": "review-ok",
    "☑": "review-ok", "✖": "review-clear", "↺": "undo", "↻": "undo", "➕": "new", "➖": "delete",
    "🔀": "tool-label-map", "🎯": "tool-sampling", "🎲": "tool-sampling", "⚙": "tool-model-setup",
    "🏷": "group-labels", "🔍": "zoom", "🧩": "tool-crop-gallery", "🤖": "tool-model-check",
    "⚡": "export", "💾": "export", "⟳": "undo", "🎚": "tool-smart-filter",
}


def is_dark() -> bool:
    """Palette-first (what widgets really paint) — see utils.theme.is_dark."""
    from .theme import is_dark as _is_dark
    return _is_dark()


def ink_color() -> str:
    """The theme's text colour: what `currentColor` in the SVGs becomes."""
    app = QGuiApplication.instance()
    if app is None:
        return "#333333"
    return app.palette().color(QPalette.ColorRole.WindowText).name()


def icon_path(name: str, dark: bool | None = None) -> Path | None:
    for ext in (".svg", ".png"):
        p = ICONS_DIR / f"{name}{ext}"
        if p.is_file():
            return p
    return None


@lru_cache(maxsize=None)
def _svg_text(name: str) -> str | None:
    p = ICONS_DIR / f"{name}.svg"
    return p.read_text(encoding="utf-8") if p.is_file() else None


_RENDERERS: dict = {}


def _renderer(name: str, ink: str):
    key = (name, ink)
    r = _RENDERERS.get(key)
    if r is None and QSvgRenderer is not None:
        text = _svg_text(name)
        if text is None:
            return None
        r = QSvgRenderer(QByteArray(text.replace("currentColor", ink).encode("utf-8")))
        _RENDERERS[key] = r
    return r


class SvgIconEngine(QIconEngine):
    """
    Renders a duotone SVG at exactly the requested size (any DPI). `currentColor` becomes the palette text
    colour at paint time (so theme switches need no new files), or a fixed colour when one is given
    (e.g. shape-type icons in their label colour). Disabled icons are drawn faded.
    """

    def __init__(self, name: str, color: str | None = None):
        super().__init__()
        self.name, self.color = name, color

    def paint(self, painter, rect, mode, state):
        r = _renderer(self.name, self.color or ink_color())
        if r is None:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if mode == QIcon.Mode.Disabled:
            painter.setOpacity(0.35)
        r.render(painter, QRectF(rect))
        painter.restore()

    def scaledPixmap(self, size, mode, state, scale):
        w, h = max(1, round(size.width() * scale)), max(1, round(size.height() * scale))
        pm = QPixmap(w, h)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        self.paint(p, QRect(0, 0, w, h), mode, state)
        p.end()
        pm.setDevicePixelRatio(scale)
        return pm

    def pixmap(self, size, mode, state):
        return self.scaledPixmap(size, mode, state, 1.0)

    def actualSize(self, size, mode, state):
        return size

    def clone(self):
        return SvgIconEngine(self.name, self.color)

    def key(self):
        return "aio-svg"


@lru_cache(maxsize=None)
def _icon(name: str, dark: bool) -> QIcon:
    if _svg_text(name) is not None and QSvgRenderer is not None:
        return QIcon(SvgIconEngine(name))
    p = ICONS_DIR / f"{name}.png"                 # raster fallback (e.g. the app icon)
    return QIcon(str(p)) if p.is_file() else QIcon()


SHAPE_ICONS = {"rectangle": "shape-rectangle", "polygon": "shape-polygon", "line": "shape-line",
               "linestrip": "shape-linestrip", "circle": "shape-circle", "point": "shape-circle",
               "points": "shape-circle"}


@lru_cache(maxsize=None)
def _shape_icon(name: str, color: str) -> QIcon:
    return QIcon(SvgIconEngine(name, color)) if QSvgRenderer is not None and _svg_text(name) else QIcon()


def tintedIcon(name: str, color: str) -> QIcon:
    """Any icon drawn in one fixed colour (e.g. white on a coloured notification)."""
    return _shape_icon(name, color)


def shapeIcon(shape_type: str | None, color) -> QIcon:
    """Shape-type glyph (rectangle / polygon / line / linestrip / circle / point) drawn in the label colour."""
    name = SHAPE_ICONS.get(shape_type or "polygon", "shape-polygon")
    return _shape_icon(name, color.name() if hasattr(color, "name") else str(color))


def newIcon(icon: str) -> QIcon:
    """Cached per theme. Unknown names give an empty icon instead of an error."""
    return _icon(icon, is_dark()) if icon else QIcon()


def _emoji_icon(text, icon):
    """'🗑 Delete' -> ('Delete', 'delete') when no explicit icon was given."""
    t = str(text)
    if icon is None:
        for emo in sorted(EMOJI_ICONS, key=len, reverse=True):
            if t.startswith(emo):
                return t[len(emo):].lstrip("\ufe0f ").strip(), EMOJI_ICONS[emo]
    return t, icon


def refresh_icons():
    """Theme changed: drop cached icons and re-apply them to every tagged action and button; restyle text."""
    from .theme import restyle_all
    restyle_all()
    _icon.cache_clear()
    _RENDERERS.clear()
    app = QApplication.instance()
    if app is None:
        return
    for w in app.allWidgets():
        if isinstance(w, QAbstractButton) and w.property(ICON_PROP):
            w.setIcon(newIcon(w.property(ICON_PROP)))
    for top in app.topLevelWidgets():
        for a in top.findChildren(QAction):
            if a.property(ICON_PROP):
                a.setIcon(newIcon(a.property(ICON_PROP)))


def set_icon(obj, name: str):
    """Assign an icon to an existing action / button / tool button and tag it for theme refreshes."""
    obj.setIcon(newIcon(name))
    obj.setProperty(ICON_PROP, name)


@lru_cache(maxsize=None)
def app_settings():
    """The app's QSettings: the OS store, or data/settings.ini next to a portable build (core.appdirs)."""
    from PyQt6.QtCore import QSettings
    from AIO_QC.core.appdirs import settings_file
    f = settings_file()
    return QSettings(str(f), QSettings.Format.IniFormat) if f is not None else QSettings("AIO_QC", "AIO_QC")


def appIcon() -> QIcon:
    """Multi-resolution application icon (title bar, taskbar, Alt+Tab each pick the best size)."""
    icon = QIcon()
    for s in (16, 32, 64, 128, 256):
        p = ICONS_DIR / f"app_{s}.png"
        if p.is_file():
            icon.addFile(str(p), QSize(s, s))
    return icon


def _no_args(slot):
    """clicked(bool) / triggered(bool) -> slot(): plain slots never receive the checked flag."""
    return lambda _checked=False, f=slot: f()


def newButton(text, icon=None, slot=None, tip=None):
    text, icon = _emoji_icon(text, icon)
    b = QPushButton(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
        b.setProperty(ICON_PROP, icon)
    if slot is not None:
        b.clicked.connect(_no_args(slot))
    if tip:
        b.setToolTip(tip)
    return b


def newAction(parent, text, slot=None, shortcut=None, icon=None, tip=None,
              checkable=False, enabled=True, checked=False):
    """Create a new action and assign callbacks, shortcuts, etc."""
    text, icon = _emoji_icon(text, icon)
    a = QAction(str(text), parent)
    if icon is not None:
        a.setIconText(str(text).replace(" ", "\n").replace("…", ""))  # text under the icon (labelme style)
        a.setIcon(newIcon(icon))
        a.setProperty(ICON_PROP, icon)
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts([QKeySequence(s) for s in shortcut])
        else:
            a.setShortcut(QKeySequence(shortcut))
    if tip is not None:
        sc = ", ".join(s.toString(QKeySequence.SequenceFormat.NativeText) for s in a.shortcuts())
        a.setToolTip(f"{tip} ({sc})" if sc else str(tip))
        a.setStatusTip(str(tip))
    if slot is not None:
        # checkable actions pass their state to the slot; plain actions call it with no arguments
        a.triggered.connect(slot if checkable else _no_args(slot))
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    a.setChecked(checked)
    return a


def addActions(widget, actions):
    """Menus and toolbars. None → separator · QMenu → submenu · dict → new submenu · QAction · QWidget (toolbars)."""
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        elif isinstance(action, dict) and "menu" in action:
            submenu = widget.addMenu(action["menu"])
            if action.get("icon"):
                submenu.setIcon(newIcon(action["icon"]))
            addActions(submenu, action.get("actions", []))
        elif isinstance(action, QAction):
            widget.addAction(action)
        elif isinstance(action, QWidget) and isinstance(widget, QToolBar):
            widget.addWidget(action)


def labelValidator():
    """Labels may not start with a space or tab (prevents 'plate' vs ' plate' variants)."""
    return QRegularExpressionValidator(QRegularExpression(r"^[^ \t].*"), None)


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def distancetoline(point, line):
    p1, p2 = line
    p1 = np.array([p1.x(), p1.y()])
    p2 = np.array([p2.x(), p2.y()])
    p3 = np.array([point.x(), point.y()])
    if np.dot((p3 - p1), (p2 - p1)) < 0:
        return np.linalg.norm(p3 - p1)
    if np.dot((p3 - p2), (p1 - p2)) < 0:
        return np.linalg.norm(p3 - p2)
    if np.linalg.norm(p2 - p1) == 0:
        return np.linalg.norm(p3 - p1)
    d = p2 - p1
    return abs(d[0] * (p1 - p3)[1] - d[1] * (p1 - p3)[0]) / np.linalg.norm(d)   # 2-D cross product


def fmtShortcut(text):
    """'Ctrl+Shift+S' -> '<b>Ctrl</b>+<b>Shift</b>+<b>S</b>'; single keys ('D') and lists ('A, PgUp') work too."""
    parts = []
    for seq in str(text).split(","):
        seq = seq.strip()
        keys = [k for k in seq.split("+") if k] or ([seq] if seq else [])
        if seq.endswith("++"):
            keys.append("+")
        parts.append("+".join(f"<b>{k}</b>" for k in keys))
    return ", ".join(parts)
