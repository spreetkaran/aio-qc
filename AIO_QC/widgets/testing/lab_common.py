# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Small Qt building blocks shared by the Model Lab, the export dialogs and the frame search dock:

  Job          background thread with progress, log, cancel and a result (never blocks the UI)
  ChartView    an SVG chart from core.lab.charts drawn natively (QtSvg), sharp at any size, with the same
               hover tooltips as the HTML report and right-click "Copy image / Save as PNG / SVG"
  ChartGrid    responsive grid of titled chart cards (2 columns when there is room)
  bgr_pixmap   numpy BGR image -> QPixmap
  theme_mode   'light' / 'dark' from the current palette (charts use the matching validated colours)
"""
import os
import re
import threading
import time

import numpy as np
from PyQt6.QtCore import (Qt, QThread, QObject, pyqtSignal, QRectF, QSize, QByteArray, QPointF, QTimer,
                          QAbstractTableModel, QModelIndex)
from PyQt6.QtGui import QPainter, QPixmap, QImage, QGuiApplication, QPalette
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout, QToolTip, QMenu, QFileDialog, QApplication, QFrame,
    QSizePolicy, QScrollArea, QPushButton,
)

try:
    from PyQt6.QtSvg import QSvgRenderer
except Exception:          # pragma: no cover - QtSvg ships with PyQt6; keep the app usable without it
    QSvgRenderer = None


# ------------------------------------------------------------------ helpers
def theme_mode():
    pal = QApplication.instance().palette() if QApplication.instance() else None
    if pal is None:
        return "light"
    return "dark" if pal.color(QPalette.ColorRole.Window).lightness() < 128 else "light"


def bgr_pixmap(img, max_w=None, max_h=None):
    if img is None:
        return QPixmap()
    import cv2
    h, w = img.shape[:2]
    s = 1.0
    if max_w and w > max_w:
        s = max_w / w
    if max_h and h * s > max_h:
        s = max_h / h
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    rgb = np.ascontiguousarray(img[:, :, ::-1]) if img.ndim == 3 else np.ascontiguousarray(img)
    fmt = QImage.Format.Format_RGB888 if rgb.ndim == 3 else QImage.Format.Format_Grayscale8
    qimg = QImage(rgb.data, w, h, rgb.strides[0], fmt).copy()
    return QPixmap.fromImage(qimg)


from AIO_QC.tools.frame_extractor.core.common import short_time as human_time  # noqa: E402,F401  (one formatter)
from AIO_QC.utils.fx_settings import testing_settings


def safe_name(text, limit=80):
    """A file name from a title (letters, digits, - and _)."""
    t = re.sub(r"[^\w\-]+", "_", str(text), flags=re.UNICODE).strip("_")
    return (t or "image")[:limit]


def last_dir():
    d = testing_settings().value("save_dir", "", str)
    return d if d and os.path.isdir(d) else os.path.expanduser("~")


def remember_dir(path):
    testing_settings().setValue("save_dir", os.path.dirname(os.path.abspath(path)) if os.path.splitext(path)[1] else path)


def open_path(path):
    """Open a file / folder with the system's default application."""
    from PyQt6.QtGui import QDesktopServices
    from PyQt6.QtCore import QUrl
    QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(path)))


# ------------------------------------------------------------------ background job
class _Chan:
    """A Job signal: emit() from the worker thread, connect() on the UI side. Emissions travel through a queued
    slot of the Job (UI thread) and reach the connected slots only while the Job's owner widget still exists — a
    result arriving after its window closed is dropped instead of touching deleted widgets."""
    __slots__ = ("_raw", "_pub")

    def __init__(self, raw, pub):
        self._raw, self._pub = raw, pub

    def emit(self, *a):
        self._raw.emit(*a)

    def connect(self, *a, **k):
        return self._pub.connect(*a, **k)

    def disconnect(self, *a):
        return self._pub.disconnect(*a)


class Job(QThread):
    """Runs fn(job) off the UI thread. fn reports with job.progress.emit(...) / job.log.emit(text) and checks
    job.cancelled (a callable, pass it straight to core functions as cancel=). done / failed / progress / log are
    delivered on the UI thread, and only while the owner (the parent passed in) is alive."""
    _progress_raw, _progress_pub = pyqtSignal(object), pyqtSignal(object)
    _log_raw, _log_pub = pyqtSignal(str), pyqtSignal(str)
    _done_raw, _done_pub = pyqtSignal(object), pyqtSignal(object)
    _failed_raw, _failed_pub = pyqtSignal(str), pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self._cancel = threading.Event()
        self.t0 = 0.0
        self._owner = parent
        for name in ("progress", "log", "done", "failed"):
            raw, pub = getattr(self, f"_{name}_raw"), getattr(self, f"_{name}_pub")
            raw.connect(getattr(self, f"_deliver_{name}"))            # bound method: queued to the UI thread
            setattr(self, name, _Chan(raw, pub))

    def _alive(self) -> bool:
        from PyQt6 import sip
        return self._owner is None or not sip.isdeleted(self._owner)

    def _deliver_progress(self, v):
        if self._alive():
            self._progress_pub.emit(v)

    def _deliver_log(self, v):
        if self._alive():
            self._log_pub.emit(v)

    def _deliver_done(self, v):
        if self._alive():
            self._done_pub.emit(v)

    def _deliver_failed(self, v):
        if self._alive():
            self._failed_pub.emit(v)

    def cancel(self):
        self._cancel.set()

    def cancelled(self):
        return self._cancel.is_set()

    thread_cleanups = []          # called at the end of every job, on its thread (e.g. close per-thread decoders)

    def run(self):
        self.t0 = time.monotonic()
        try:
            self.done.emit(self.fn(self))
        except Exception as exc:          # noqa: BLE001 - reported to the user
            import traceback
            try:
                from loguru import logger
                logger.error(traceback.format_exc())
            except Exception:
                pass
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            for fn in Job.thread_cleanups:
                try:
                    fn()
                except Exception:
                    pass


class FuncWorker(Job):
    """Job for fn(note): note(text) reports progress text (the `note` signal)."""

    def __init__(self, fn, parent=None):
        super().__init__(lambda job: fn(job.log.emit), parent)

    @property
    def note(self):
        return self.log


# ------------------------------------------------------------------ charts
_EL_RE = re.compile(r'<(rect|circle|path|text)\s([^>]*?)/?>')
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _attr(attrs, name):
    m = re.search(rf'\b{name}="([^"]*)"', attrs)
    return float(m.group(1)) if m else None


def svg_hotspots(svg):
    """[(QRectF in SVG units, tip, link)] for elements with a data-tip and/or data-link (smallest first, the
    hit-test order: a cell wins over the background it sits on)."""
    from html import unescape
    out = []
    for tag, attrs in _EL_RE.findall(svg):
        if "data-tip" not in attrs and "data-link" not in attrs:
            continue
        tip = re.search(r'data-tip="([^"]*)"', attrs)
        link = re.search(r'data-link="([^"]*)"', attrs)
        r = None
        if tag == "rect":
            x, y, w, h = (_attr(attrs, k) for k in ("x", "y", "width", "height"))
            if None not in (x, y, w, h):
                r = QRectF(x, y, max(w, 2), max(h, 2))
        elif tag == "circle":
            cx, cy, rad = (_attr(attrs, k) for k in ("cx", "cy", "r"))
            if None not in (cx, cy, rad):
                r = QRectF(cx - rad, cy - rad, rad * 2, rad * 2)
        elif tag == "path":
            m = re.search(r'\bd="([^"]*)"', attrs)
            if m:
                nums = [float(v) for v in _NUM.findall(re.sub(r"A[^A-Za-z]*", " ", m.group(1)))]
                xs, ys = nums[0::2], nums[1::2]
                if xs and ys:
                    r = QRectF(min(xs), min(ys), max(2.0, max(xs) - min(xs)), max(2.0, max(ys) - min(ys)))
        elif tag == "text":
            x, y = _attr(attrs, "x"), _attr(attrs, "y")
            if x is not None and y is not None:
                r = QRectF(x - 40, y - 12, 80, 16)
        if r is not None:
            out.append((r, unescape(tip.group(1)) if tip else "", unescape(link.group(1)) if link else ""))
    out.sort(key=lambda t: t[0].width() * t[0].height())
    return out


def svg_tips(svg):
    return [(r, t) for r, t, _ in svg_hotspots(svg) if t]


class ChartView(QWidget):
    """An SVG chart drawn natively: rendered ONCE into a cached pixmap for its size (repaints are a blit, so a
    page of charts scrolls and animates smoothly), crisp re-render after a resize settles, a short fade-in,
    hover tooltips (data-tip) and clickable elements (data-link -> link_clicked)."""
    link_clicked = pyqtSignal(str)

    def __init__(self, svg="", parent=None, max_scale=1.6):
        super().__init__(parent)
        from AIO_QC.tools.frame_extractor.utils import fonts
        fonts.load()                                   # chart labels are drawn in Inter
        self.title = "chart"
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.max_scale = max_scale
        self._renderer = None
        self._vw, self._vh = 560.0, 320.0
        self._spots = []
        self._svg = ""
        self._pm = None              # cached rendering: QPixmap
        self._pm_key = None
        self._alpha = 1.0
        self._crisp = QTimer(self)
        self._crisp.setSingleShot(True)
        self._crisp.setInterval(120)
        self._crisp.timeout.connect(self._rerender)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.set_svg(svg)

    def set_svg(self, svg, fade=True):
        self._svg = svg or ""
        self._renderer = None
        self._spots = []
        self._pm = self._pm_key = None
        if self._svg and QSvgRenderer is not None:
            self._renderer = QSvgRenderer(QByteArray(self._svg.encode("utf-8")))
            m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', self._svg)
            if m:
                self._vw, self._vh = float(m.group(1)), float(m.group(2))
            self._spots = svg_hotspots(self._svg)
        if fade and self._svg:
            from AIO_QC.widgets import motion as anim
            self._alpha = 0.0
            anim.animate(("chart-in", id(self)), self._set_alpha, 0.0, 1.0, 240, "out_cubic", widget=self,
                         retarget=False)
        self.update()

    def _set_alpha(self, a):
        self._alpha = a
        self.update()

    def svg(self):
        return self._svg

    def aspect(self):
        return self._vh / self._vw if self._vw else 0.5

    def height_for(self, w):
        return int(self._vh * min(self.max_scale, max(0.3, w / self._vw)))

    def sizeHint(self):
        # the width hint is deliberately small: a grid of charts splits its columns evenly instead of by viewBox
        return QSize(min(int(self._vw), 320), int(self._vh))

    def minimumSizeHint(self):
        return QSize(160, 40)

    def _target(self):
        s = min(self.max_scale, max(0.3, min(self.width() / self._vw, self.height() / self._vh)))
        w, h = self._vw * s, self._vh * s
        return QRectF((self.width() - w) / 2, 0, w, h), s

    def _render_to_pixmap(self, rect):
        dpr = self.devicePixelRatioF()
        pm = QPixmap(max(1, int(rect.width() * dpr)), max(1, int(rect.height() * dpr)))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self._renderer.render(p, QRectF(0, 0, rect.width(), rect.height()))
        p.end()
        return pm

    def _rerender(self):
        self._pm_key = None
        self.update()

    def paintEvent(self, _ev):
        if self._renderer is None or self._alpha <= 0.0:
            return
        rect, _ = self._target()
        key = (int(rect.width()), int(rect.height()), self.devicePixelRatioF())
        if self._pm is None:
            self._pm, self._pm_key = self._render_to_pixmap(rect), key
        elif key != self._pm_key and not self._crisp.isActive():
            if self._pm_key is None:                       # explicit re-render
                self._pm, self._pm_key = self._render_to_pixmap(rect), key
            else:                                          # resizing: stretch the old one, crisp when settled
                self._crisp.start()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self._alpha < 1.0:
            p.setOpacity(self._alpha)
        p.drawPixmap(rect, self._pm, QRectF(0, 0, self._pm.width(), self._pm.height()))
        p.end()

    def _hit(self, pos):
        rect, s = self._target()
        pt = QPointF((pos.x() - rect.x()) / s, (pos.y() - rect.y()) / s)
        for r, tip, link in self._spots:
            if r.adjusted(-3, -3, 3, 3).contains(pt):
                return tip, link
        return None, None

    def mouseMoveEvent(self, ev):
        tip, link = self._hit(ev.position())
        self.setCursor(Qt.CursorShape.PointingHandCursor if link else Qt.CursorShape.ArrowCursor)
        if tip:
            QToolTip.showText(ev.globalPosition().toPoint(), tip + ("\nClick to see these images" if link else ""), self)
        else:
            QToolTip.hideText()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            _, link = self._hit(ev.position())
            if link:
                self.link_clicked.emit(link)
                return
        super().mouseReleaseEvent(ev)

    def render_pixmap(self, scale=2.0, background=True):
        """The chart at scale x its natural size, on the window background (a saved PNG is readable anywhere)."""
        pm = QPixmap(int(self._vw * scale), int(self._vh * scale))
        if background:
            pm.fill(self.palette().color(QPalette.ColorRole.Base))
        else:
            pm.fill(Qt.GlobalColor.transparent)
        if self._renderer is not None:
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._renderer.render(p, QRectF(0, 0, pm.width(), pm.height()))
            p.end()
        return pm

    def save_menu(self, parent=None):
        m = QMenu(parent or self)
        m.addAction("Save as PNG…").triggered.connect(lambda: self.save_dialog("png"))
        m.addAction("Save as SVG…").triggered.connect(lambda: self.save_dialog("svg"))
        m.addSeparator()
        m.addAction("Copy image").triggered.connect(lambda: QGuiApplication.clipboard().setPixmap(self.render_pixmap()))
        return m

    def _menu(self, pos):
        if self._svg:
            self.save_menu().exec(self.mapToGlobal(pos))

    def save_dialog(self, kind):
        path, _ = QFileDialog.getSaveFileName(self, "Save chart", os.path.join(last_dir(), f"{safe_name(self.title)}.{kind}"),
                                              f"{kind.upper()} (*.{kind})")
        if path:
            self.save(path)
            remember_dir(path)

    def save(self, path):
        """PNG (2x, on the window background) or SVG (fonts fall back to the system font where Inter is missing)."""
        if path.lower().endswith(".svg"):
            from AIO_QC.core.testing.charts import FONT
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._svg.replace("font-family:Inter;", f"font-family:{FONT};"))
        else:
            self.render_pixmap(2.0).save(path)
        return path


class ChartCard(QFrame):
    """Title, description, the chart, and a Save button (PNG / SVG / copy)."""

    def __init__(self, title, desc, svg, parent=None):
        super().__init__(parent)
        self.setObjectName("chartCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.title = title
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        t = QLabel(f"<b>{_esc(title)}</b>")
        t.setWordWrap(True)
        head.addWidget(t, 1)
        from PyQt6.QtWidgets import QToolButton
        self.save_btn = QToolButton()
        self.save_btn.setText("Save")
        self.save_btn.setAutoRaise(True)
        self.save_btn.setToolTip("Save this chart as PNG or SVG, or copy it")
        self.save_btn.clicked.connect(lambda: self.view.save_menu(self).exec(
            self.save_btn.mapToGlobal(self.save_btn.rect().bottomLeft())))
        head.addWidget(self.save_btn, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(head)
        if desc:
            d = QLabel(desc)
            d.setWordWrap(True)
            d.setEnabled(False)            # muted text, readable in both themes
            f = d.font()
            f.setPointSizeF(max(7.5, f.pointSizeF() * 0.9))
            d.setFont(f)
            lay.addWidget(d)
        self.view = ChartView(svg)
        self.view.title = title
        lay.addWidget(self.view)
        self.link_clicked = self.view.link_clicked


def _esc(s):
    from html import escape
    return escape(str(s))


class ChartGrid(QScrollArea):
    """Titled chart cards in a responsive grid: 2 equal columns when there is room, 1 when narrow; a lone chart, a
    wide chart (timelines) and extra widgets (tables) take the full row. Chart heights are set here from the
    column width (fixed aspect) and never exceed the visible height, and the vertical scroll bar is always present
    - so the layout can never oscillate between "fits" and "needs a scroll bar"."""
    link_clicked = pyqtSignal(str)

    def __init__(self, parent=None, empty_text="Nothing to show yet."):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._host = QWidget()
        self._outer = QVBoxLayout(self._host)
        self._outer.setContentsMargins(4, 4, 4, 4)
        self._outer.setSpacing(0)
        self._gridw = QWidget()
        self._grid = QGridLayout(self._gridw)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(10)
        self._outer.addWidget(self._gridw)
        self._empty = QLabel(empty_text)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self._empty.setEnabled(False)
        self._empty.setMinimumHeight(120)
        self._outer.addWidget(self._empty)
        self._outer.addStretch(1)                  # spare height goes below the charts, never between them
        self.setWidget(self._host)
        self._cards = []
        self._extra = []
        self._cols = 0
        self._size = None
        self.empty_text = empty_text

    def set_empty_text(self, text):
        self._empty.setText(text)
        self._empty.setVisible(bool(text) and not self._cards and not self._extra)

    def cards(self):
        return list(self._cards)

    def set_figures(self, figs, wide=()):
        """figs: [(title, description, svg)]; wide: titles that take a full row."""
        for c in self._cards + self._extra:
            self._grid.removeWidget(c)
            c.setParent(None)
            c.deleteLater()
        self._cards, self._extra = [], []
        wide = set(wide)
        for t, d, svg in figs:
            card = ChartCard(t, d, svg)
            card.wide = t in wide or ' width="1080"' in svg
            card.link_clicked.connect(self.link_clicked.emit)
            self._cards.append(card)
        self._empty.setVisible(not self._cards and bool(self._empty.text()))
        self._relayout()

    def add_widget(self, w):
        """A non-chart widget (table, gallery) spanning the full width after the charts."""
        self._extra.append(w)
        self._empty.setVisible(False)
        self._relayout()

    def _relayout(self):
        self._cols, self._size = 0, None
        self._layout()

    def _layout(self):
        vp = self.viewport()
        vw, vh = vp.width(), vp.height()
        cols = 2 if vw >= 1000 and len(self._cards) > 1 else 1
        if cols != self._cols:
            self._cols = cols
            for c in self._cards + self._extra:
                self._grid.removeWidget(c)
            for k in range(self._grid.rowCount()):
                self._grid.setRowStretch(k, 0)
            for k in range(2):
                self._grid.setColumnStretch(k, 1 if k < cols else 0)
                self._grid.setColumnMinimumWidth(k, 0)
            r = c_ = 0
            for card in self._cards:
                if getattr(card, "wide", False) or cols == 1:
                    if c_:
                        r, c_ = r + 1, 0
                    self._grid.addWidget(card, r, 0, 1, cols)
                    r += 1
                    continue
                self._grid.addWidget(card, r, c_)
                c_ += 1
                if c_ >= cols:
                    r, c_ = r + 1, 0
            if c_:
                r += 1
            for w in self._extra:
                self._grid.addWidget(w, r, 0, 1, cols)
                r += 1
        if (vw, vh) != self._size and vw > 0:
            self._size = (vw, vh)
            inner = vw - 8 - 22                              # grid margins, card margins + frame
            col = (inner - 10 * (cols - 1)) / cols
            max_h = max(160, vh - 80)                         # a chart is never taller than the view
            for card in self._cards:
                w = inner if getattr(card, "wide", False) or cols == 1 else col
                h = card.view.height_for(max(160, w))
                card.view.setFixedHeight(int(max(40, min(h, max_h))))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._layout()

    def save_all(self, folder, kind="png"):
        """Every chart into folder (title.png / .svg); returns the paths."""
        out, used = [], set()
        for card in self._cards:
            base = safe_name(card.title)
            name, k = base, 2
            while name in used:
                name, k = f"{base}_{k}", k + 1
            used.add(name)
            out.append(card.view.save(os.path.join(folder, f"{name}.{kind}")))
        return out


class StatTile(QFrame):
    """A headline number: label, value, sub-line (native widgets in the app font, so it scales with the UI)."""

    def __init__(self, label, value, sub="", parent=None):
        super().__init__(parent)
        self.setObjectName("statTile")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(0)
        a = QLabel(label)
        a.setEnabled(False)
        v = QLabel(str(value))
        f = v.font()
        f.setPointSizeF(f.pointSizeF() * 1.7)
        from PyQt6.QtGui import QFont
        f.setWeight(QFont.Weight.DemiBold)
        v.setFont(f)
        lay.addWidget(a)
        lay.addWidget(v)
        if sub:
            c = QLabel(sub)
            c.setEnabled(False)
            c.setWordWrap(True)
            lay.addWidget(c)
        self.setMinimumWidth(120)


class Gallery(QWidget):
    """Thumbnails in a wrapping grid with a fixed cell size (thumbnails that arrive later never change the
    layout), an empty-state message instead of a fake tile, and the usual list signals via .list."""

    def __init__(self, icon=QSize(300, 190), text_lines=2, empty_text="Nothing to show.", parent=None):
        super().__init__(parent)
        from PyQt6.QtWidgets import QListWidget, QStackedLayout
        self.stack = QStackedLayout(self)
        self.stack.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        lw = self.list
        lw.setViewMode(QListWidget.ViewMode.IconMode)
        lw.setResizeMode(QListWidget.ResizeMode.Adjust)
        lw.setMovement(QListWidget.Movement.Static)
        lw.setWrapping(True)
        lw.setWordWrap(True)
        lw.setUniformItemSizes(True)
        lw.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        lw.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        lw.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.empty = QLabel(empty_text)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)
        self.empty.setEnabled(False)
        self.stack.addWidget(lw)
        self.stack.addWidget(self.empty)
        self.set_icon_size(icon, text_lines)
        self._blank = None

    def set_icon_size(self, icon, text_lines=2):
        self.list.setIconSize(icon)
        lh = self.list.fontMetrics().lineSpacing()
        self.list.setGridSize(QSize(icon.width() + 16, icon.height() + lh * text_lines + 18))
        self._blank = None

    def blank_icon(self):
        """A neutral placeholder of the icon size (shown until the thumbnail arrives)."""
        if self._blank is None:
            from PyQt6.QtGui import QIcon
            sz = self.list.iconSize()
            pm = QPixmap(sz)
            pm.fill(self.palette().color(QPalette.ColorRole.AlternateBase))
            self._blank = QIcon(pm)
        return self._blank

    def clear(self, empty_text=None):
        self.list.clear()
        if empty_text is not None:
            self.empty.setText(empty_text)

    def show_empty(self, on):
        self.stack.setCurrentIndex(1 if on else 0)

    def add(self, text, data=None, tip=None):
        from PyQt6.QtWidgets import QListWidgetItem
        it = QListWidgetItem(self.blank_icon(), text)
        it.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        if data is not None:
            it.setData(Qt.ItemDataRole.UserRole, data)
        if tip:
            it.setToolTip(tip)
        self.list.addItem(it)
        self.show_empty(False)
        return it


def fit_pixmap(img, size):
    """BGR image -> QPixmap that fits size (logical px) at the screen's pixel ratio (sharp on HiDPI)."""
    dpr = 2.0
    try:
        dpr = max(1.0, QGuiApplication.primaryScreen().devicePixelRatio())
    except Exception:
        pass
    pm = bgr_pixmap(img, int(size.width() * dpr), int(size.height() * dpr))
    pm.setDevicePixelRatio(dpr)
    return pm


class ProgressRow(QWidget):
    """Label · progress bar · detail text (one per model in the Run page)."""

    def __init__(self, label, color=None, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.sw = QLabel()
        self.sw.setFixedSize(12, 12)
        if color:
            self.sw.setStyleSheet(f"background:{color}; border-radius:3px;")
        self.name = QLabel(label)
        self.name.setMinimumWidth(180)
        from AIO_QC.widgets.motion import SmoothProgressBar
        self.bar = SmoothProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        self.text = QLabel("waiting")
        self.text.setMinimumWidth(220)
        lay.addWidget(self.sw)
        lay.addWidget(self.name)
        lay.addWidget(self.bar, 1)
        lay.addWidget(self.text)

    def set(self, done, total, text=None):
        self.bar.setValue(int(1000 * done / total) if total else 0)
        if text is not None:
            self.text.setText(text)


# ------------------------------------------------------------------ background view building
class SerialWorker(QObject):
    """One background thread that runs view computations in order, latest-wins per key: asking for the same
    view again drops the older, not yet started request, and a stale result is never applied. Running them one
    after another (not in parallel) keeps the UI thread responsive and avoids races on shared results."""
    _result = pyqtSignal(object, int, object)
    _error = pyqtSignal(object, int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        import queue
        self._q = queue.Queue()
        self._latest = {}
        self._cbs = {}
        self._result.connect(self._deliver)
        self._error.connect(self._deliver_error)
        self._alive = True
        self._t = threading.Thread(target=self._loop, name="lab-views", daemon=True)
        self._t.start()

    def submit(self, key, fn, done=None, failed=None):
        tok = self._latest.get(key, 0) + 1
        self._latest[key] = tok
        self._cbs[(key, tok)] = (done, failed)
        if os.environ.get("FX_SYNC_VIEWS"):            # tests: run inline, deterministic
            try:
                res = fn()
            except Exception as exc:                   # noqa: BLE001
                self._deliver_error(key, tok, f"{type(exc).__name__}: {exc}")
                return tok
            self._deliver(key, tok, res)
            return tok
        self._q.put((key, tok, fn))
        return tok

    def cancel(self, key):
        self._latest[key] = self._latest.get(key, 0) + 1

    def busy(self):
        return not self._q.empty()

    def _loop(self):
        import queue
        from PyQt6 import sip
        while self._alive:
            try:
                key, tok, fn = self._q.get(timeout=2.0)
            except queue.Empty:
                if sip.isdeleted(self):                # the owner window is gone: end the thread
                    break
                continue
            if key is None or sip.isdeleted(self):
                break
            if self._latest.get(key) != tok:
                self._cbs.pop((key, tok), None)
                continue
            try:
                res = fn()
            except Exception as exc:                   # noqa: BLE001
                import traceback
                try:
                    from loguru import logger
                    logger.debug(traceback.format_exc())
                except Exception:
                    pass
                try:
                    self._error.emit(key, tok, f"{type(exc).__name__}: {exc}")
                except RuntimeError:                   # deleted meanwhile
                    break
                continue
            try:
                self._result.emit(key, tok, res)
            except RuntimeError:
                break

    def _deliver(self, key, tok, res):
        done, _ = self._cbs.pop((key, tok), (None, None))
        if done is not None and self._latest.get(key) == tok:
            done(res)

    def _deliver_error(self, key, tok, msg):
        _, failed = self._cbs.pop((key, tok), (None, None))
        if failed is not None and self._latest.get(key) == tok:
            failed(msg)

    def stop(self):
        self._alive = False
        self._q.put((None, 0, None))


class RowsModel(QAbstractTableModel):
    """Read-only, sortable table model for large lists (thousands of rows render instantly, unlike items).
    rows: [tuple of display values]; sort_keys: same shape with sortable values (optional)."""

    def __init__(self, headers, parent=None):
        super().__init__(parent)
        self.headers, self.rows, self.keys = list(headers), [], []

    def set_rows(self, rows, keys=None):
        self.beginResetModel()
        self.rows = list(rows)
        self.keys = list(keys) if keys is not None else self.rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self.rows[index.row()][index.column()])
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(self.keys[index.row()][index.column()], (int, float)):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        self.layoutAboutToBeChanged.emit()
        idx = sorted(range(len(self.rows)), key=lambda i: (self.keys[i][column] is None, self.keys[i][column]),
                     reverse=order == Qt.SortOrder.DescendingOrder)
        self.rows = [self.rows[i] for i in idx]
        self.keys = [self.keys[i] for i in idx]
        self.layoutChanged.emit()


class IssueBar(QFrame):
    """Problems of one step, each with a one-click fix: shown at the top of the page, hidden when all is well."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("issueBar")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(10, 6, 10, 6)
        self._lay.setSpacing(4)
        self.hide()

    def set_issues(self, issues):
        """issues: [(level 'error'|'warn'|'info', text, fix label or None, fix callable or None)]."""
        while self._lay.count():
            it = self._lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        if not issues:
            self.hide()
            return
        worst = "error" if any(i[0] == "error" for i in issues) else "warn" if any(i[0] == "warn" for i in issues) else "info"
        col = {"error": "#d03b3b", "warn": "#c98500", "info": "#3987e5"}[worst]
        self.setStyleSheet(f"#issueBar {{ border-left: 3px solid {col}; background: palette(alternate-base); "
                           f"border-radius: 4px; }}")
        icon = {"error": "✗", "warn": "⚠", "info": "ℹ"}
        for level, text, fix, fn in issues:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            lab = QLabel(f"<span style='color:{ {'error': '#d03b3b', 'warn': '#c98500', 'info': '#3987e5'}[level]}'>"
                         f"{icon[level]}</span>&nbsp; {text}")
            lab.setWordWrap(True)
            lab.setTextFormat(Qt.TextFormat.RichText)
            h.addWidget(lab, 1)
            if fix and fn:
                b = QPushButton(fix)
                b.clicked.connect(lambda _=False, f=fn: f())
                h.addWidget(b)
            self._lay.addWidget(row)
        self.show()
