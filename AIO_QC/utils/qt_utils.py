# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
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
from PyQt6.QtWidgets import QAbstractButton, QApplication, QLayout, QMenu, QPushButton, QToolBar, QWidget

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


_PICTO = None


def no_emoji(text: str) -> str:
    """Text without colour emoji (🔍 📝 ✅ ❌ 🖼️ …) — the 3.0 look uses icons, never emoji. Typographic symbols stay
    (→ ✓ ✗ ⚑ · …), and ⚠️ becomes the plain ⚠."""
    global _PICTO
    if not text:
        return text
    if _PICTO is None:
        import re
        _PICTO = re.compile("[\U0001F000-\U0001FAFF\u2705\u274C\u274E\u2753-\u2757\u2B50\u2B55"
                            "\u23E9-\u23FA\u231A\u231B\u2614\u2615\u26A1\u26BD\u26C4\u26D4\u26EA"
                            "\u26F5\u26FA\u26FD\u2728\u2795-\u2797\u27B0\u27BF]\uFE0F?[ \u00A0]?")
    out = _PICTO.sub("", str(text)).replace("\uFE0F", "")
    return out.replace("  ", " ").strip() if out != text else out


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


class PaintedIconEngine(QIconEngine):
    """An icon drawn by code at exactly the requested pixel size (any DPI — FHD at 125 %, 4K, 8K): `fn(painter)`
    draws in a base × base coordinate system and is scaled to the target, never a bitmap stretched afterwards."""

    def __init__(self, fn, base: float = 16.0, key: str = "aio-painted"):
        super().__init__()
        self.fn, self.base, self._key = fn, float(base), key

    def paint(self, painter, rect, mode, state):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(rect)
        k = min(r.width(), r.height()) / self.base
        painter.translate(r.x() + (r.width() - self.base * k) / 2, r.y() + (r.height() - self.base * k) / 2)
        painter.scale(k, k)
        if mode == QIcon.Mode.Disabled:
            painter.setOpacity(0.35)
        self.fn(painter)
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
        return PaintedIconEngine(self.fn, self.base, self._key)

    def key(self):
        return self._key


def painted_icon(fn, base: float = 16.0) -> QIcon:
    return QIcon(PaintedIconEngine(fn, base))


def screen_dpr(widget=None) -> float:
    """The device pixel ratio to render for: the widget's screen, else the sharpest connected screen."""
    if widget is not None:
        try:
            return float(widget.devicePixelRatioF())
        except RuntimeError:
            pass
    screens = QGuiApplication.screens()
    return max([sc.devicePixelRatio() for sc in screens] or [1.0])


class IconLabel(QWidget):
    """A fixed-size icon that paints itself through QIcon at the real screen resolution (QLabel.setPixmap with
    a 1× pixmap is stretched — blurry — on scaled / HiDPI screens)."""

    def __init__(self, icon, size: int, parent=None):
        super().__init__(parent)
        self.icon = icon if isinstance(icon, QIcon) else newIcon(icon)
        self._name = None if isinstance(icon, QIcon) else icon
        self.isz = size
        self.setFixedSize(size, size)

    def set_icon(self, icon):
        self.icon = icon if isinstance(icon, QIcon) else newIcon(icon)
        self._name = None if isinstance(icon, QIcon) else icon
        self.update()

    def paintEvent(self, _e):
        if self._name:                                   # theme ink may have changed
            self.icon = newIcon(self._name)
        p = QPainter(self)
        s = min(self.isz, self.width(), self.height())
        self.icon.paint(p, QRect((self.width() - s) // 2, (self.height() - s) // 2 if self.height() < 2 * s else 0, s, s))


@lru_cache(maxsize=None)
def _icon(name: str, dark: bool) -> QIcon:
    if _svg_text(name) is not None and QSvgRenderer is not None:
        return QIcon(SvgIconEngine(name))
    return raster_icon(name)


@lru_cache(maxsize=None)
def raster_icon(name: str) -> QIcon:
    """A raster icon from every size shipped (<name>_16.png … _512.png, plus <name>.png): Qt picks the closest size
    for the screen's scaling, so it is never one small bitmap stretched (blurry) or a huge one squeezed."""
    icon = QIcon()
    for f in sorted(ICONS_DIR.glob(f"{name}_*.png")):
        tail = f.stem[len(name) + 1:]
        if tail.isdigit():
            icon.addFile(str(f), QSize(int(tail), int(tail)))
    p = ICONS_DIR / f"{name}.png"
    if p.is_file():
        icon.addFile(str(p))
    return icon


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


_LATER: dict = {}          # id(timer) -> (timer, fire): alive until the timer fired or its owner went away


def set_shown(w, on: bool):
    """setVisible for a child widget that may not have its parent yet (built before its layout is attached):
    setVisible(True) on a parentless widget opens it as a window of its own for a moment — on Windows a flash that
    takes the focus. A new widget is visible by default anyway, so it simply shows with its parent."""
    if on and w.parentWidget() is None:
        return
    w.setVisible(bool(on))


def later(ms: int, owner, fn):
    """Run fn after ms milliseconds — unless `owner` (a QObject) is destroyed first. Unlike
    QTimer.singleShot(ms, lambda: ...), a lambda that uses a closed window can never fire.

    The timer's Python wrapper and its slot are kept in _LATER until it fires: a wrapper nobody references can be
    collected by Python's cycle collector while Qt still owns the timer, and its slot function was then cleared
    under Qt's feet (a crash when the timer fired — seen once style sheets made timing different)."""
    from PyQt6 import sip
    from PyQt6.QtCore import QTimer
    for k, (tt, _f) in list(_LATER.items()):            # timers whose owner was deleted before they fired
        try:
            if sip.isdeleted(tt):
                del _LATER[k]
        except (TypeError, RuntimeError):
            _LATER.pop(k, None)
    t = QTimer(owner)
    t.setSingleShot(True)
    key = id(t)

    def fire():
        _LATER.pop(key, None)
        try:
            alive = not sip.isdeleted(owner)
        except TypeError:
            alive = True
        try:
            if alive:
                fn()
        finally:
            try:
                if not sip.isdeleted(t):
                    t.deleteLater()
            except (TypeError, RuntimeError):
                pass
    _LATER[key] = (t, fire)
    t.timeout.connect(fire)
    t.start(max(0, int(ms)))
    return t


def app_settings():
    """The app's QSettings: the OS store, or data/settings.ini next to a portable build (core.appdirs)."""
    from PyQt6.QtCore import QSettings
    from AIO_QC.core.appdirs import settings_file
    f = settings_file()
    return QSettings(str(f), QSettings.Format.IniFormat) if f is not None else QSettings("AIO_QC", "AIO_QC")


def appIcon() -> QIcon:
    """Multi-resolution application icon (title bar, taskbar, Alt+Tab each pick the best size)."""
    return raster_icon("app")


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


# ------------------------------------------------------------------ windows vs. the screen
def available_geometry(widget=None):
    """Usable area (without taskbar / dock) of the screen the widget is on (its parent's, else the primary)."""
    scr = None
    for w in (widget, widget.parentWidget() if widget is not None else None):
        if w is not None:
            try:
                scr = w.screen()
            except RuntimeError:
                scr = None
            if scr is not None:
                break
    scr = scr or QGuiApplication.primaryScreen()
    return scr.availableGeometry() if scr is not None else None


def preferred_size(win, w, h, frac=0.85):
    """Resize a window to (w, h), but never larger than frac of the available screen area."""
    g = available_geometry(win)
    if g is not None:
        w, h = min(w, int(g.width() * frac)), min(h, int(g.height() * frac))
    win.resize(w, h)


# Window classes, sized by the screen they open on (its usable area): share of width / height, and a cap so a window
# on a very large screen stays a comfortable size instead of filling it.
WINDOW_SIZES = {
    "large": (0.92, 0.90, 2600, 1600),     # workspaces: Model Check, Testing, galleries, Preparation, Convert & Split
    "medium": (0.72, 0.80, 1760, 1180),    # checks and lists: every scan window, Issues Inbox, Settings
    "small": (0.52, 0.66, 1180, 860),      # readers and helpers: Help, Storage, Background Tasks
}


TITLE_ROOM = 44                            # the app's own title bar (feature windows), added after sizing


def fit_size(win, kind: str = "medium"):
    """Size a feature window for the screen it opens on (its parent's): `kind` sets the share of the usable area and
    the cap (WINDOW_SIZES); never smaller than its content needs, never larger than the screen."""
    g = available_geometry(win)
    if g is None or g.width() < 200 or g.height() < 200:
        return
    fw, fh, cw, ch = WINDOW_SIZES.get(kind, WINDOW_SIZES["medium"])
    wrap_long_labels(win)
    m = win.minimumSizeHint()
    w = max(min(int(g.width() * fw), cw), m.width())
    h = max(min(int(g.height() * fh), ch), m.height())
    # within the limits fit_before_show keeps (96 % × 94 %), with room for the title bar the frame adds on top
    win.resize(min(w, int(g.width() * 0.96)), min(h, int(g.height() * 0.94) - TITLE_ROOM))


def wrap_long_labels(win, limit: int = 560):
    """Labels whose one line is wider than `limit` px wrap instead — a long hint never sets a window's width."""
    from PyQt6.QtWidgets import QLabel
    for lab in win.findChildren(QLabel):
        try:
            if not lab.wordWrap() and lab.pixmap().isNull() and lab.sizeHint().width() > limit:
                lab.setWordWrap(True)
        except RuntimeError:
            pass


class FlowLayout(QLayout):
    """Widgets in a row that wraps onto the next line when the window is narrow (toolbars of buttons), instead of
    forcing the window as wide as all of them. The minimum width is the widest single widget."""

    def __init__(self, parent=None, spacing: int = 6, align_right: bool = False):
        super().__init__(parent)
        self._items = []
        self._sp = spacing
        self._right = align_right
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def insertWidget(self, index: int, w):
        self.addChildWidget(w)
        from PyQt6.QtWidgets import QWidgetItem
        self._items.insert(max(0, min(index, len(self._items))), QWidgetItem(w))
        self.invalidate()

    def addStretch(self, *_a):                     # compatibility with QHBoxLayout callers: rows wrap instead
        pass

    def addSpacing(self, *_a):
        pass

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._place(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._place(rect, False)

    def sizeHint(self):
        w = h = 0
        for it in self._visible():
            s = it.sizeHint()
            w += s.width() + self._sp
            h = max(h, s.height())
        m = self.contentsMargins()
        return QSize(max(0, w - self._sp) + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self):
        s = QSize()
        for it in self._visible():
            s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _visible(self):
        return [it for it in self._items if not (it.widget() is not None and it.widget().isHidden())]

    def _place(self, rect, test):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        rows, row, row_w = [], [], 0
        for it in self._visible():
            s = it.sizeHint()
            need = s.width() + (self._sp if row else 0)
            if row and row_w + need > r.width():
                rows.append(row)
                row, row_w = [], 0
                need = s.width()
            row.append((it, s))
            row_w += need
        if row:
            rows.append(row)
        y = r.y()
        for row in rows:
            h = max(s.height() for _it, s in row)
            used = sum(s.width() for _it, s in row) + self._sp * (len(row) - 1)
            x = r.x() + (max(0, r.width() - used) if self._right else 0)
            for it, s in row:
                if not test:
                    it.setGeometry(QRect(x, y + (h - s.height()) // 2, s.width(), s.height()))
                x += s.width() + self._sp
            y += h + self._sp
        return (y - self._sp - r.y() if rows else 0) + m.top() + m.bottom()


class SplitRow(QLayout):
    """Two groups on one line — the left one takes the spare room, the right one sits at the right edge — that
    become two lines (the right group under the left, still right-aligned) when the window is too narrow for both.
    The minimum width is the wider of the two groups, not their sum."""

    def __init__(self, left: QWidget, right: QWidget, parent=None, spacing: int = 8):
        super().__init__(parent)
        from PyQt6.QtWidgets import QWidgetItem
        self._sp = spacing
        self.addChildWidget(left)
        self.addChildWidget(right)
        self._items = [QWidgetItem(left), QWidgetItem(right)]
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):                          # fixed two items
        pass

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation.Horizontal

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._place(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._place(rect, False)

    def sizeHint(self):
        a, b = (it.sizeHint() for it in self._items)
        m = self.contentsMargins()
        return QSize(a.width() + self._sp + b.width() + m.left() + m.right(),
                     max(a.height(), b.height()) + m.top() + m.bottom())

    def minimumSize(self):
        a, b = (it.minimumSize() for it in self._items)
        m = self.contentsMargins()
        return QSize(max(a.width(), b.width()) + m.left() + m.right(),
                     max(a.height(), b.height()) + m.top() + m.bottom())

    def _place(self, rect, test):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        L, R = self._items
        lh, rh = L.sizeHint(), R.sizeHint()
        lmin = L.minimumSize()
        if not R.widget().isHidden() and lmin.width() + self._sp + rh.width() > r.width():     # two lines
            if not test:
                L.setGeometry(QRect(r.x(), r.y(), r.width(), lh.height()))
                rw = min(rh.width(), r.width())
                R.setGeometry(QRect(r.right() + 1 - rw, r.y() + lh.height() + 4, rw, rh.height()))
            return lh.height() + 4 + rh.height() + m.top() + m.bottom()
        h = max(lh.height(), rh.height())
        if not test:
            rw = rh.width() if not R.widget().isHidden() else 0
            lw = r.width() - rw - (self._sp if rw else 0)
            L.setGeometry(QRect(r.x(), r.y() + (h - lh.height()) // 2, lw, lh.height()))
            if rw:
                R.setGeometry(QRect(r.right() + 1 - rw, r.y() + (h - rh.height()) // 2, rw, rh.height()))
        return h + m.top() + m.bottom()


def scrolled(widget, horizontal: bool = False):
    """A page in a frameless scroll area: it keeps its natural size on a big screen and scrolls on a small one,
    instead of forcing the whole window taller (or wider) than the screen."""
    from PyQt6.QtWidgets import QFrame, QScrollArea
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if horizontal
                                    else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setWidget(widget)
    return sa


def flow_row(items, spacing: int = 10, align_right: bool = False) -> QWidget:
    """A wrapping row: `items` are widgets or (label text, widget) pairs — a pair stays together on one line."""
    from PyQt6.QtWidgets import QHBoxLayout, QLabel
    host = QWidget()
    flow = FlowLayout(host, spacing=spacing, align_right=align_right)
    for it in items:
        if isinstance(it, tuple):
            text, w = it
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            if text:
                lab = QLabel(text)
                if w.toolTip():
                    lab.setToolTip(w.toolTip())
                h.addWidget(lab)
            h.addWidget(w)
            flow.addWidget(box)
        else:
            flow.addWidget(it)
    return host


def fit_to_screen(win):
    """Keep a top-level window inside its screen: shrink it (and its minimum size) to the available area and move
    it so no part - title bar included - is off-screen. Maximized / full-screen windows are left alone."""
    if not win.isWindow() or win.isMaximized() or win.isFullScreen():
        return
    g = available_geometry(win)
    if g is None or g.width() < 200 or g.height() < 200:
        return
    fg, cg = win.frameGeometry(), win.geometry()
    dw, dh = max(0, fg.width() - cg.width()), max(0, fg.height() - cg.height())
    maxw, maxh = g.width() - dw, g.height() - dh
    if win.minimumWidth() > maxw:
        win.setMinimumWidth(maxw)
    if win.minimumHeight() > maxh:
        win.setMinimumHeight(maxh)
    w, h = min(win.width(), maxw), min(win.height(), maxh)
    if (w, h) != (win.width(), win.height()):
        win.resize(w, h)
    fg = win.frameGeometry()
    x = min(max(fg.x(), g.left()), g.right() + 1 - fg.width())
    y = min(max(fg.y(), g.top()), g.bottom() + 1 - fg.height())
    if (x, y) != (fg.x(), fg.y()):
        win.move(x, y)
