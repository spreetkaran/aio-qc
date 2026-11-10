# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
import os
from functools import lru_cache

from PyQt6.QtGui import QIcon, QAction, QIconEngine, QPainter, QPixmap, QPalette, QGuiApplication
from PyQt6.QtWidgets import QToolBar, QComboBox, QLabel
from PyQt6.QtCore import QByteArray, QRect, QRectF, Qt

try:  # ships with PyQt6
    from PyQt6.QtSvg import QSvgRenderer
except ImportError:  # pragma: no cover
    QSvgRenderer = None

here = os.path.dirname(os.path.abspath(__file__))
ICONS_DIR = os.path.normpath(os.path.join(here, "..", "icons"))


def ink_color():
    """The theme's text colour: what `currentColor` in the icon SVGs becomes."""
    app = QGuiApplication.instance()
    return app.palette().color(QPalette.ColorRole.WindowText).name() if app else "#333333"


@lru_cache(maxsize=None)
def _svg_text(name):
    p = os.path.join(ICONS_DIR, name + ".svg")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return f.read()


_RENDERERS = {}


def _renderer(name, ink):
    r = _RENDERERS.get((name, ink))
    if r is None and QSvgRenderer is not None:
        text = _svg_text(name)
        if text is None:
            return None
        r = QSvgRenderer(QByteArray(text.replace("currentColor", ink).encode("utf-8")))
        _RENDERERS[(name, ink)] = r
    return r


class SvgIconEngine(QIconEngine):
    """Duotone SVG rendered at exactly the requested size (any DPI). `currentColor` becomes the palette's
    text colour at paint time, so icons follow theme switches without extra files (AIO QC recipe).
    Disabled icons are drawn faded."""

    def __init__(self, name, color=None):
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
        return "fx-svg"


def newIcon(icon, color=None):
    """icons/<name>.svg (duotone, theme ink) or a PNG (e.g. the app icon). Unknown -> empty icon."""
    if not icon:
        return QIcon()
    if QSvgRenderer is not None and _svg_text(icon) is not None:
        return QIcon(SvgIconEngine(icon, color))
    if icon == "icon":                     # the Frame Extractor's app icon: a vector (sharp at any size / DPI)
        from AIO_QC.utils.qt_utils import newIcon as aio_icon
        return aio_icon("mode-video")
    p = os.path.join(ICONS_DIR, icon + ".png")
    return QIcon(p) if os.path.isfile(p) else QIcon()


def menu_text(text):
    """A toolbar label written on two lines ("Load\\nVideo", "Book-\\nmark") as one menu line."""
    return str(text).replace("-\n", "").replace("\n", " ")


def newAction(
    parent,
    text,
    func=None,
    shortcut=None,
    icon=None,
    tip=None,
    checkable=False,
    enabled=True,
    checked=False,
):
    text = str(text)
    action = QAction(menu_text(text), parent)                # menus: one line ("Load Video", "Bookmark")
    if "\n" in text:
        action.setIconText(text)                              # toolbar: two lines under the icon
    if icon is not None:
        action.setIcon(newIcon(icon))
    if shortcut is not None:
        action.setShortcut(shortcut)
    if tip is not None:
        action.setToolTip(str(tip))
        action.setStatusTip(str(tip))
    if func is not None:
        action.triggered.connect(func)
    if checkable:
        action.setCheckable(True)
    action.setEnabled(enabled)
    action.setChecked(checked)
    return action

def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif all([isinstance(widget, QToolBar), any([isinstance(action, QComboBox), isinstance(action, QLabel)])]):
            widget.addWidget(action)
        elif isinstance(action, QAction):
            widget.addAction(action)
            

# windows vs. the screen: shared with the whole app
from AIO_QC.utils.qt_utils import available_geometry, fit_to_screen, preferred_size  # noqa: F401,E402
