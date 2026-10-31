# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Picture-in-picture for a Fix-in-Editor session, INSIDE the main window: a small overlay (draggable by its header,
resizable, kept inside the window) with the editor state, the counts and one row per file with its status. The full
session window shrinks into it and grows back.

FrameAnimator drives those transitions at the screen's refresh rate: a precise timer ticking every 1000 / Hz ms and
progress computed from the ELAPSED time (QElapsedTimer), so motion stays smooth even if a frame is late — instead of
Qt's fixed ~60 Hz animation tick.
"""
from __future__ import annotations

from PyQt6.QtCore import QElapsedTimer, QEasingCurve, QPoint, QRect, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QPainter
from PyQt6.QtWidgets import (QGraphicsOpacityEffect, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QToolButton,
                             QVBoxLayout, QWidget)

from AIO_QC.utils import newIcon, theme_color


def refresh_hz(widget=None) -> float:
    scr = (widget.screen() if widget is not None and widget.screen() is not None
           else QGuiApplication.primaryScreen())
    hz = scr.refreshRate() if scr is not None else 60.0
    return hz if 24.0 <= hz <= 500.0 else 60.0


_LIVE: set = set()          # running animators: PyQt keeps only a WEAK reference to a bound-method slot, so an
                            # animator nobody holds would be garbage-collected and silently stop mid-way


def _settings():
    from AIO_QC.utils import app_settings
    return app_settings()


class FrameAnimator:
    """step(t) with t = eased progress 0..1, once per display frame; done() at the end. Keeps itself alive while
    running (no need to store it)."""

    def __init__(self, owner, duration_ms: int, step, done=None, easing=QEasingCurve.Type.OutCubic):
        self.duration = max(1, duration_ms)
        self.step, self.done = step, done
        self.curve = QEasingCurve(easing)
        self.clock = QElapsedTimer()
        self.timer = QTimer(owner)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(max(2, int(round(1000.0 / refresh_hz(owner if isinstance(owner, QWidget) else None)))))
        self.timer.timeout.connect(self._tick)

    def start(self):
        _LIVE.add(self)
        self.timer.destroyed.connect(lambda *_: _LIVE.discard(self))
        self.clock.start()
        self.step(0.0)
        self.timer.start()

    def stop(self):
        self.timer.stop()
        _LIVE.discard(self)

    def running(self) -> bool:
        return self.timer.isActive()

    def _tick(self):
        p = min(1.0, self.clock.elapsed() / self.duration)
        self.step(self.curve.valueForProgress(p))
        if p >= 1.0:
            self.timer.stop()
            _LIVE.discard(self)
            if self.done is not None:
                self.done()


def lerp_rect(a: QRect, b: QRect, t: float) -> QRect:
    return QRect(round(a.x() + (b.x() - a.x()) * t), round(a.y() + (b.y() - a.y()) * t),
                 round(a.width() + (b.width() - a.width()) * t), round(a.height() + (b.height() - a.height()) * t))


STATUS = {  # status -> (icon, theme role)
    "waiting": ("review-next", "muted"), "fixed": ("review-fixed", "info"), "unchanged": ("review-ok", "muted"),
    "rejected": ("review-flag", "error"), "conflict": ("review-flag", "warn"), "error": ("review-clear", "error"),
}


class _Grip(QWidget):
    """Bottom-right resize handle for an in-window overlay (QSizeGrip only resizes top-level windows)."""

    def __init__(self, target):
        super().__init__(target)
        self.target = target
        self.setFixedSize(16, 16)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self._start = None

    def mousePressEvent(self, e):
        self._start = (e.globalPosition().toPoint(), self.target.size())
        e.accept()

    def mouseMoveEvent(self, e):
        if self._start is not None:
            p0, s0 = self._start
            d = e.globalPosition().toPoint() - p0
            self.target.resize(max(self.target.minimumWidth(), s0.width() + d.x()),
                               max(self.target.minimumHeight(), s0.height() + d.y()))
            self.target.clamp()
            e.accept()

    def mouseReleaseEvent(self, e):
        self._start = None
        self.target.remember()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setPen(theme_color("muted"))
        for i in (4, 8, 12):
            p.drawLine(15, 15 - i, 15 - i, 15)
        p.end()


class MiniSessionWindow(QWidget):
    """In-app picture-in-picture: an overlay INSIDE the main window (not a system window), dragged by its header or any
    empty part, resized from the corner, kept inside the window, remembered relative to the bottom-right corner."""
    restoreRequested = pyqtSignal()
    openRequested = pyqtSignal(str)          # key
    MARGIN = 8

    def __init__(self, host: QWidget):
        super().__init__(host)
        self.host = host
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setMinimumSize(260, 180)
        self.resize(420, 300)
        self._drag: QPoint | None = None
        self._rows: dict[str, QListWidgetItem] = {}
        self._flash: dict[str, float] = {}
        self._flash_anim = None
        self.fx = QGraphicsOpacityEffect(self)          # the only effect: fade in / out
        self.fx.setOpacity(1.0)
        self.setGraphicsEffect(self.fx)

        self.title = QLabel("")
        self.title.setTextFormat(Qt.TextFormat.RichText)
        self.counts = QLabel("")
        self.counts.setTextFormat(Qt.TextFormat.RichText)
        self.counts.setWordWrap(True)
        for lab in (self.title, self.counts):           # labels must not swallow the press that starts a drag
            lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        restore = QToolButton()
        restore.setIcon(newIcon("restore-window"))
        restore.setIconSize(QSize(18, 18))
        restore.setToolTip("Restore the full session window (or double-click the panel)")
        restore.setAutoRaise(True)
        restore.setCursor(Qt.CursorShape.PointingHandCursor)
        restore.clicked.connect(self.restoreRequested.emit)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        ic = QLabel()
        ic.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        ic.setPixmap(newIcon("tool-editor").pixmap(18, 18))
        top.addWidget(ic)
        top.addWidget(self.title, 1)
        top.addWidget(restore)
        self.list = QListWidget()
        self.list.setFrameShape(QListWidget.Shape.NoFrame)
        self.list.setUniformItemSizes(True)
        self.list.setCursor(Qt.CursorShape.ArrowCursor)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setStyleSheet("QListWidget{background: transparent;} QListWidget::item{padding: 1px 2px;}")
        self.list.itemDoubleClicked.connect(lambda it: self.openRequested.emit(it.data(Qt.ItemDataRole.UserRole)))
        self.grip = _Grip(self)
        foot = QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("drag to move · double-click a file to open it")
        hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        hint.setStyleSheet(f"color: {theme_color('muted').name()}; font-size: 8pt;")
        foot.addWidget(hint, 1)
        foot.addWidget(self.grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 12, 10)       # room for the shadow
        lay.setSpacing(6)
        lay.addLayout(top)
        lay.addWidget(self.counts)
        lay.addWidget(self.list, 1)
        lay.addLayout(foot)
        host.installEventFilter(self)                 # keep inside when the main window is resized
        self.hide()

    # ------------------------------------------------------------------ content
    def set_header(self, title_html: str, counts_html: str):
        self.title.setText(title_html)
        self.counts.setText(counts_html)

    def set_entries(self, entries):
        self.list.clear()
        self._rows = {}
        for e in entries:
            it = QListWidgetItem()
            it.setData(Qt.ItemDataRole.UserRole, e.key)
            self.list.addItem(it)
            self._rows[e.key] = it
            self._paint(e, it)

    def set_entry(self, e, flash: bool = True):
        it = self._rows.get(e.key)
        if it is None:
            return
        self._paint(e, it)
        if flash and e.status not in ("waiting",):
            self.list.scrollToItem(it, QListWidget.ScrollHint.PositionAtCenter)
            self._flash[e.key] = 1.0
            self._run_flash()

    def _paint(self, e, it):
        icon, role = STATUS.get(e.status, ("review-next", "muted"))
        it.setIcon(newIcon(icon))
        it.setText(f"{e.name}  —  {e.status}")
        it.setForeground(theme_color(role) if e.status != "waiting" else self.palette().text().color())
        it.setToolTip(e.message or e.status)

    def _run_flash(self):
        """Changed rows light up and fade out (1.2 s), animated at the display rate."""
        if self._flash_anim is not None and self._flash_anim.running():
            self._flash_anim.stop()
        start = dict(self._flash)
        hl = theme_color("info")

        def step(t):
            for k, v0 in start.items():
                it = self._rows.get(k)
                if it is None:
                    continue
                a = max(0.0, v0 * (1.0 - t))
                c = QColor(hl)
                c.setAlphaF(0.35 * a)
                it.setBackground(c)
                self._flash[k] = a

        def done():
            for k in start:
                it = self._rows.get(k)
                if it is not None:
                    it.setBackground(QColor(0, 0, 0, 0))
            self._flash = {k: v for k, v in self._flash.items() if v > 0.01}

        self._flash_anim = FrameAnimator(self, 1200, step, done, QEasingCurve.Type.InOutQuad)
        self._flash_anim.start()

    # ------------------------------------------------------------------ placement inside the main window
    def bounds(self) -> QRect:
        return self.host.rect().adjusted(self.MARGIN, self.MARGIN, -self.MARGIN, -self.MARGIN)

    def clamp(self):
        b = self.bounds()
        w, h = min(self.width(), b.width()), min(self.height(), b.height())
        if (w, h) != (self.width(), self.height()):
            self.resize(w, h)
        x = min(max(self.x(), b.left()), b.right() - w + 1)
        y = min(max(self.y(), b.top()), b.bottom() - h + 1)
        if (x, y) != (self.x(), self.y()):
            self.move(x, y)

    DEFAULT = (6, 64, 420, 300)        # distance from the right edge, from the top (below the menu bar), w, h

    def default_rect(self) -> QRect:
        """Top right by default; remembered as distances from the window's top-right corner (follows resizes)."""
        d = _settings().value("editor/mini_place_tr")
        right, top, w, h = self.DEFAULT
        if isinstance(d, (list, tuple)) and len(d) == 4:
            try:
                right, top, w, h = (int(v) for v in d)
            except (TypeError, ValueError):
                pass
        b = self.bounds()
        w, h = min(max(w, self.minimumWidth()), b.width()), min(max(h, self.minimumHeight()), b.height())
        x = max(b.left(), b.right() - right - w + 1)
        y = min(max(b.top(), top), b.bottom() - h + 1)
        return QRect(x, y, w, h)

    def remember(self):
        b = self.bounds()
        g = self.geometry()
        _settings().setValue("editor/mini_place_tr", [b.right() - g.right(), g.top(), g.width(),
                                                                       g.height()])

    def eventFilter(self, obj, ev):
        if obj is self.host and ev.type() == ev.Type.Resize and self.isVisible():
            self.setGeometry(self.default_rect())
            self.clamp()
        return False

    # ------------------------------------------------------------------ painting + dragging
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = QRectF(self.rect()).adjusted(6, 4, -6, -8)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(6, 0, -1):                      # soft shadow
            p.setBrush(QColor(0, 0, 0, int(16 * (1 - i / 7))))
            p.drawRoundedRect(body.adjusted(-i, -i + 3, i, i + 3), 12 + i, 12 + i)
        bg = QColor(self.palette().window().color())
        bg.setAlpha(245)
        p.setBrush(bg)
        p.setPen(theme_color("line"))
        p.drawRoundedRect(body, 12, 12)
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.raise_()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(self.mapToParent(e.position().toPoint()) - self._drag)
            self.clamp()
            e.accept()

    def mouseReleaseEvent(self, e):
        if self._drag is not None:
            self._drag = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.remember()

    def mouseDoubleClickEvent(self, e):
        self.restoreRequested.emit()


class Ghost(QWidget):
    """In-app snapshot used for the shrink / grow transition (no per-frame relayout of the real window)."""

    def __init__(self, host: QWidget, pixmap):
        super().__init__(host)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.pix, self.opacity = pixmap, 1.0

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setOpacity(max(0.0, min(1.0, self.opacity)))
        p.drawPixmap(self.rect(), self.pix)
        p.end()
