# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Notifications: cards in the bottom-right corner of the main window, in the app's own look (surface, border, soft
shadow), light and dark.

    window.toast.show_message("<b>Updated from disk</b><br>3 changed", kind="info", details=fn_returning_text)

Text: "<b>Title</b> …<br>rest" shows as a title line (semibold) over a muted second line; any other text is one line in
the normal ink. kind: info · success · warn · error — a tinted badge with the kind's icon, and the countdown line along
the bottom edge, both in the theme's colour for that kind.

Actions: [(label, fn)] — the first one is the main action (teal), the others are outlined; a click runs it and closes
the notification. details=…: a "Details" button that opens a panel inside the notification (no extra window), with
"Copy". The text is fetched when it opens (a callable), so it is always current.

Width is fixed (TEXT_W, narrower in a small window): the message never re-wraps while anything animates. Details open
by HEIGHT only — one smooth movement, anchored at the bottom so the card grows upwards; the panel is laid out once at
its full size and revealed by a clipping box whose height is the only value that changes per frame.

Motion: fade in while rising, fade out while drifting right, the others glide into place. The countdown is a
continuous animation (only the line's strip is repainted); it runs only while AIO QC is the active window, the mouse is
not over the notification and the details are closed. key=…: a notification with the same key is updated in place (a
short pulse, the line refills) instead of stacking. Max 4 stacked; sticky=True stays until dismissed.

Painting: the fade is the ONLY graphics effect — Qt does not support an effect inside another one.
"""
from __future__ import annotations

import re

from PyQt6.QtCore import (QEasingCurve, QParallelAnimationGroup, QPoint, QPropertyAnimation, QRect, QRectF, QSize, Qt,
                          QVariantAnimation, pyqtSignal)
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QTextOption
from PyQt6.QtWidgets import (QApplication, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from AIO_QC.utils import tintedIcon
from AIO_QC.core.diag import ignored as _ignored

KINDS = {  # kind -> (theme colour key, icon)
    "info": ("info", "info"),
    "success": ("ok", "status-ok"),
    "warn": ("warn", "status-warning"),
    "error": ("error", "status-error"),
}
MAX_VISIBLE = 4
MARGIN = 16
GAP = 2
SLIDE = 14                       # px the notification rises while appearing
TEXT_W = 400                     # the card's width (fixed; narrower only in a small window)
DETAILS_H = 220                  # tallest details panel (it scrolls beyond)
EXPAND_MS = 260
RADIUS = 10
PAD = 14
SHADOW = (8, 6, 8, 10)           # room around the card for its shadow: left, top, right, bottom
BAR_H = 3

_TITLE = re.compile(r"^\s*(<b>.*?</b>[^<]*(?:<(?!br)[^>]*>[^<]*)*?)\s*<br\s*/?>\s*(.*)$", re.S | re.I)


def _tokens():
    from AIO_QC.widgets import design
    return design.tokens()


def split_message(html: str):
    """'<b>Title</b> · extra<br>rest' -> (title html, body html); anything else -> ('', html)."""
    m = _TITLE.match(html or "")
    if m and m.group(2).strip():
        return m.group(1).strip(), m.group(2).strip()
    s = (html or "").strip()
    if s.lower().startswith("<b>") and s.lower().endswith("</b>") and s.lower().count("<b>") == 1:
        return s, ""                                        # a title alone
    return "", s


class _Badge(QWidget):
    """The kind's icon on a tinted rounded square."""

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setFixedSize(32, 32)

    def paintEvent(self, _e):
        t = _tokens()
        key, icon = KINDS.get(self.kind, KINDS["info"])
        col = QColor(t[key])
        soft = QColor(col)
        soft.setAlpha(38 if QColor(t["surface"]).lightness() < 80 else 28)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(soft)
        p.drawRoundedRect(QRectF(self.rect()), 8, 8)
        tintedIcon(icon, col.name()).paint(p, QRect(7, 7, 18, 18))


class _Reveal(QWidget):
    """Clips the details panel: the panel keeps its full size and position; only this box's height animates."""

    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.panel = panel
        panel.setParent(self)
        panel.move(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(0)
        self.setVisible(False)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.panel.resize(self.width(), self.panel.height())


class _Card(QFrame):
    """Layout container for the content (the Toast paints body, border, shadow and countdown itself)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


class Toast(QWidget):
    closed = pyqtSignal(object)

    def __init__(self, parent, html: str, kind: str, actions, sticky: bool, timeout_ms: int, details=None,
                 on_resize=None, width: int = TEXT_W):
        super().__init__(parent)
        self.kind = kind if kind in KINDS else "info"
        self.details, self.on_resize = details, on_resize
        self._expanded = False
        self._expand_t = 0.0                   # 0 closed … 1 open (animated)
        self._expand_anim = None
        self._details_h = 0
        self._base_h = 0
        self.sticky, self.timeout_ms = sticky, timeout_ms
        self.key = None
        self.progress = None if sticky else 1.0
        self._pulse = 0.0                      # 0..1 brightening when the content is updated in place
        self._closing = False
        self._hover = False
        self._placed = False
        self.card_w = int(width)
        self.card = _Card(self)
        t = _tokens()

        self.badge = _Badge(self.kind)
        self.title = QLabel()
        self.text = QLabel()                   # the message (kept as .text for callers and tests)
        for lab in (self.title, self.text):
            lab.setWordWrap(True)
            lab.setTextFormat(Qt.TextFormat.RichText)
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._style_text(t)
        close = QToolButton()                                     # a drawn icon: never a missing font glyph
        close.setIconSize(QSize(12, 12))
        close.setAutoRaise(True)
        close.setFixedSize(24, 24)
        close.setToolTip("Close")
        close.setAccessibleName("Close notification")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.clicked.connect(lambda _c=False: self.dismiss())
        self.close_btn = close
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(3)
        col.addWidget(self.title)
        col.addWidget(self.text)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(12)
        top.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignTop)
        top.addLayout(col, 1)
        top.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        lay = QVBoxLayout(self.card)
        lay.setContentsMargins(PAD, PAD, PAD - 4, PAD + BAR_H)
        lay.setSpacing(0)
        lay.addLayout(top)

        self.panel = self.reveal = None
        self.details_btn = self.copy_btn = None
        if details is not None:
            self.panel = QTextBrowser()
            self.panel.setOpenExternalLinks(False)
            self.panel.setFrameShape(QFrame.Shape.NoFrame)
            self.panel.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)
            self.panel.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            self.panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.reveal = _Reveal(self.panel)
            lay.addWidget(self.reveal)
        self.buttons = []
        if actions or details is not None:
            row = QHBoxLayout()
            row.setContentsMargins(0, 12, 4, 0)
            row.setSpacing(8)
            row.addStretch(1)
            if details is not None:
                self.copy_btn = self._button("Copy", lambda: self._copy_details())
                self.copy_btn.setToolTip("Copy the details")
                self.copy_btn.setVisible(False)
                row.addWidget(self.copy_btn)
                self.details_btn = self._button("Details", self.toggle_details)
                row.addWidget(self.details_btn)
            for i, (label, fn) in enumerate(actions or ()):
                b = self._button(label, lambda f=fn: (f(), self.dismiss()), primary=(i == 0))
                row.addWidget(b)
            lay.addLayout(row)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(*SHADOW)
        outer.addWidget(self.card)
        self._restyle(t)
        self.set_message(html)

        # ---- fade (the only graphics effect) + motion
        self.fx = QGraphicsOpacityEffect(self)
        self.fx.setOpacity(0.0)
        self.setGraphicsEffect(self.fx)
        self._motion = None                            # running appear / leave group
        self._glide = QPropertyAnimation(self, b"pos", self)
        self._glide.setDuration(220)
        self._glide.setEasingCurve(QEasingCurve.Type.OutCubic)

        # ---- countdown: continuous, display-rate, pausable
        self.count = QVariantAnimation(self)
        self.count.setStartValue(1.0)
        self.count.setEndValue(0.0)
        self.count.setDuration(max(1, timeout_ms))
        self.count.setEasingCurve(QEasingCurve.Type.Linear)
        self.count.valueChanged.connect(self._on_progress)
        self.count.finished.connect(self.dismiss)
        self._refill = None
        app = QApplication.instance()
        app.applicationStateChanged.connect(self._sync_countdown)

    # ------------------------------------------------------------------ content
    def _button(self, label, slot, primary=False):
        b = QPushButton(label)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setAutoDefault(False)
        if primary:
            b.setProperty("variant", "primary")
        b.clicked.connect(lambda _c=False: slot())
        self.buttons.append(b)
        return b

    def _style_text(self, t):
        self.title.setStyleSheet(f"QLabel{{color:{t['text']}; background:transparent; font-weight:600;}}")
        self.text.setStyleSheet(f"QLabel{{color:{t['muted']}; background:transparent;}}")

    def _restyle(self, t):
        self._style_text(t)
        self.close_btn.setIcon(tintedIcon("close-x", t["faint"]))
        self.close_btn.setStyleSheet(f"QToolButton{{border:none; border-radius:6px; background:transparent;}}"
                                     f"QToolButton:hover{{background:{t['hover']};}}")
        if self.panel is not None:
            self.panel.setStyleSheet(
                f"QTextBrowser{{color:{t['text']}; background:{t['surface2']}; border:1px solid {t['border_soft']};"
                f" border-radius:7px; padding:6px 4px;}}")

    def _text_width(self) -> int:
        return max(120, self.card_w - PAD - (PAD - 4) - 32 - 24 - 2 * 12)

    def set_message(self, html: str):
        self._html = html
        title, body = split_message(html)
        if title and not body:
            title, body = "", f"<span style='font-weight:600'>{title[3:-4]}</span>"
            self.text.setStyleSheet(f"QLabel{{color:{_tokens()['text']}; background:transparent;}}")
        elif not title:
            self.text.setStyleSheet(f"QLabel{{color:{_tokens()['text']}; background:transparent;}}")
        else:
            self._style_text(_tokens())
        self.title.setText(title)
        self.title.setVisible(bool(title))
        self.text.setText(body)
        self.text.setVisible(bool(body))
        tw = self._text_width()
        for lab in (self.title, self.text):
            lab.setFixedWidth(tw)                       # fixed: the text never re-wraps while the card animates
        self._fit()

    def _fit(self):
        """Fixed width; height from the content (closed: base height + the revealed part of the details)."""
        lay = self.card.layout()
        lay.invalidate()
        lay.activate()
        self.layout().invalidate()
        self.layout().activate()
        rv = self.reveal.height() if self.reveal is not None else 0
        h = self.layout().heightForWidth(self.card_w + SHADOW[0] + SHADOW[2]) \
            if self.layout().hasHeightForWidth() else self.sizeHint().height()
        self._base_h = h - rv
        self.setFixedSize(self.card_w + SHADOW[0] + SHADOW[2], h)

    # ------------------------------------------------------------------ painting
    def _body(self) -> QRectF:
        return QRectF(self.card.geometry()).adjusted(0.5, 0.5, -0.5, -0.5)

    def _bar_rect(self) -> QRect:
        b = self._body()
        return QRect(int(b.left()), int(b.bottom()) - BAR_H - 1, int(b.width()) + 1, BAR_H + 2)

    def paintEvent(self, _e):
        t = _tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = self._body()
        dark = QColor(t["surface"]).lightness() < 80
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(8, 0, -1):                       # soft shadow: widening, fading rings below the body
            p.setBrush(QColor(0, 0, 0, int((24 if dark else 11) * (1 - i / 9))))
            p.drawRoundedRect(body.adjusted(-i, -i + 3, i, i + 3), RADIUS + i, RADIUS + i)
        path = QPainterPath()
        path.addRoundedRect(body, RADIUS, RADIUS)
        fill = QColor(t["surface"])
        kind = QColor(t[KINDS[self.kind][0]])
        if self._pulse > 0:                             # updated in place: a brief tint of the kind's colour
            a = self._pulse * 0.10
            fill = QColor(int(fill.red() * (1 - a) + kind.red() * a), int(fill.green() * (1 - a) + kind.green() * a),
                          int(fill.blue() * (1 - a) + kind.blue() * a))
        p.setPen(QPen(QColor(t["border"]), 1))
        p.setBrush(fill)
        p.drawPath(path)
        if self.progress is not None:                   # time left: a line along the bottom edge, clipped to the body
            p.save()
            p.setClipPath(path)
            p.setPen(Qt.PenStyle.NoPen)
            track = QRectF(body.left(), body.bottom() - BAR_H, body.width(), BAR_H + 1)
            p.setBrush(QColor(t["track"]))
            p.drawRect(track)
            w = track.width() * max(0.0, min(1.0, self.progress))
            if w > 0.5:
                p.setBrush(kind)
                p.drawRect(QRectF(track.left(), track.top(), w, track.height()))
            p.restore()
        p.end()

    def _on_progress(self, v):
        self.progress = float(v)
        self.update(self._bar_rect())                   # repaint only the thin strip, every frame

    def changeEvent(self, e):
        from PyQt6.QtCore import QEvent
        if e.type() == QEvent.Type.PaletteChange and hasattr(self, "_html") and not self._closing:
            self._restyle(_tokens())                    # light <-> dark while shown
            rv = self.reveal.height() if self.reveal is not None else 0
            self.set_message(self._html)
            if rv:
                self.setFixedHeight(self._base_h + rv)
        super().changeEvent(e)

    # ------------------------------------------------------------------ countdown control
    def showEvent(self, e):
        super().showEvent(e)
        self._sync_countdown()          # posted while its window was in the background: count down once seen

    def hideEvent(self, e):
        super().hideEvent(e)
        self._sync_countdown()

    def _should_run(self) -> bool:
        win = self.window()
        return (not self.sticky and not self._closing and not self._hover and not self._expanded and self.isVisible()
                and QApplication.applicationState() == Qt.ApplicationState.ApplicationActive
                and not (win is not None and win.isMinimized()))

    def _sync_countdown(self, *_):
        if self.sticky or self._closing or (self._refill is not None
                                            and self._refill.state() == QVariantAnimation.State.Running):
            return
        st = self.count.state()
        if self._should_run():
            if st == QVariantAnimation.State.Paused:
                self.count.resume()
            elif st == QVariantAnimation.State.Stopped:
                self.count.start()
        elif st == QVariantAnimation.State.Running:
            self.count.pause()                          # freezes exactly where it is

    def enterEvent(self, e):
        self._hover = True
        self._sync_countdown()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        self._sync_countdown()
        super().leaveEvent(e)

    # ------------------------------------------------------------------ details (height only, in place)
    def _details_html(self) -> str:
        try:
            v = self.details() if callable(self.details) else self.details
        except Exception as exc:                                  # never break the notification for it
            _ignored(exc)
            v = f"Details unavailable: {exc}"
        v = "" if v is None else str(v)
        if "<" in v and ">" in v:
            return v
        from html import escape
        return ("<pre style='white-space:pre-wrap; margin:0; font-family:monospace; font-size:9pt;'>"
                f"{escape(v) or 'Nothing to show.'}</pre>")

    def _copy_details(self):
        if self.panel is not None:
            QApplication.clipboard().setText(self.panel.toPlainText())
            if self.copy_btn is not None:
                self.copy_btn.setText("Copied")
                from AIO_QC.utils.qt_utils import later
                later(1200, self, lambda: self.copy_btn.setText("Copy"))

    def toggle_details(self):
        if self.panel is None or self._closing:
            return
        self._expanded = not self._expanded
        if self._expanded:
            self.panel.setHtml(self._details_html())
            pw = self.card_w - PAD - (PAD - 4) - 4
            self.panel.document().setTextWidth(pw - 22)       # padding + scroll bar
            need = int(self.panel.document().size().height()) + 16
            self._details_h = max(48, min(DETAILS_H, need))
            self.panel.setFixedSize(pw, self._details_h)       # laid out ONCE at full size; revealed by clipping
            self.reveal.setFixedWidth(pw)
            self.details_btn.setText("Hide details")
            self.copy_btn.setVisible(True)
        else:
            self.details_btn.setText("Details")
        self._sync_countdown()
        self._animate_expand(1.0 if self._expanded else 0.0)

    def _animate_expand(self, to: float):
        if self._expand_anim is not None:
            self._expand_anim.stop()
        a = QVariantAnimation(self)
        a.setStartValue(float(self._expand_t))
        a.setEndValue(float(to))
        from AIO_QC.widgets import anim as _anim
        a.setDuration(EXPAND_MS if getattr(_anim, "ENABLED", True) else 1)
        a.setEasingCurve(QEasingCurve.Type.OutCubic if to > self._expand_t else QEasingCurve.Type.InOutCubic)
        a.valueChanged.connect(self._set_expand)
        a.finished.connect(lambda: self._expand_done(to))
        self._expand_anim = a
        a.start()

    def _set_expand(self, t):
        """One frame: only the revealed height changes (plus the 10 px gap above the panel, revealed with it)."""
        t = float(t)
        self._expand_t = t
        full = self._details_h + 10
        rv = int(round(full * t))
        self.reveal.setFixedHeight(rv)
        self.reveal.setVisible(rv > 0)
        self.panel.move(0, 10)
        self.setFixedHeight(self._base_h + rv)
        if self.on_resize is not None:
            self.on_resize(self)                        # stays bottom-right anchored, every frame

    def _expand_done(self, to):
        self._expand_anim = None
        if to == 0.0 and self.panel is not None:
            self.copy_btn.setVisible(False)
            self._fit()
            if self.on_resize is not None:
                self.on_resize(self)
        self._sync_countdown()

    # ------------------------------------------------------------------ life cycle
    def appear(self, target: QPoint):
        """Fade in while rising SLIDE px into place."""
        self._placed = True
        self.move(target + QPoint(0, SLIDE))
        self.show()
        self.raise_()
        fade = QPropertyAnimation(self.fx, b"opacity", self)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setDuration(240)
        fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        rise = QPropertyAnimation(self, b"pos", self)
        rise.setStartValue(self.pos())
        rise.setEndValue(target)
        rise.setDuration(240)
        rise.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._rise = rise
        self._run_motion(fade, rise)
        self._sync_countdown()

    def _appearing(self) -> bool:
        rise = getattr(self, "_rise", None)
        return (rise is not None and self._motion is not None and not self._closing
                and self._motion.state() == QParallelAnimationGroup.State.Running)

    def glide_to(self, target: QPoint):
        """Another notification left / arrived: move smoothly to the new slot."""
        if not self._placed or self._closing:
            return
        if self._appearing():
            # still rising into place: aim the rise at the new slot. A second animation of `pos` would make Qt stop
            # the first one — and with it the fade, leaving the notification invisible (several posted at once).
            self._rise.setEndValue(target)
            return
        if self.pos() == target and self._glide.state() != QPropertyAnimation.State.Running:
            return
        self._glide.stop()
        self._glide.setStartValue(self.pos())
        self._glide.setEndValue(target)
        self._glide.start()

    def dismiss(self):
        """Fade out while drifting right, then go."""
        if self._closing:
            return
        self._closing = True
        self.count.stop()
        self._glide.stop()
        fade = QPropertyAnimation(self.fx, b"opacity", self)
        fade.setStartValue(self.fx.opacity())
        fade.setEndValue(0.0)
        fade.setDuration(200)
        fade.setEasingCurve(QEasingCurve.Type.InCubic)
        drift = QPropertyAnimation(self, b"pos", self)
        drift.setStartValue(self.pos())
        drift.setEndValue(self.pos() + QPoint(24, 0))
        drift.setDuration(200)
        drift.setEasingCurve(QEasingCurve.Type.InCubic)
        group = self._run_motion(fade, drift)
        group.finished.connect(self._gone)

    def _run_motion(self, *anims) -> QParallelAnimationGroup:
        if self._motion is not None:
            self._motion.stop()
        g = QParallelAnimationGroup(self)
        for a in anims:
            g.addAnimation(a)
        g.start()
        self._motion = g
        return g

    def _gone(self):
        try:
            QApplication.instance().applicationStateChanged.disconnect(self._sync_countdown)
        except (TypeError, RuntimeError):
            pass
        self.hide()
        self.closed.emit(self)
        self.deleteLater()

    def update_message(self, html: str, timeout_ms: int | None = None):
        """Same notification, new content: a short pulse, the line refills smoothly, full time again."""
        rv = self.reveal.height() if self.reveal is not None else 0
        self.set_message(html)
        if rv:
            self.setFixedHeight(self._base_h + rv)
        self.timeout_ms = timeout_ms or self.timeout_ms
        pulse = QVariantAnimation(self)
        pulse.setStartValue(1.0)
        pulse.setEndValue(0.0)
        pulse.setDuration(420)
        pulse.setEasingCurve(QEasingCurve.Type.OutCubic)
        pulse.valueChanged.connect(lambda v: (setattr(self, "_pulse", float(v)), self.update()))
        pulse.start(QVariantAnimation.DeletionPolicy.DeleteWhenStopped)
        if self.sticky:
            return
        self.count.stop()
        self.count.setDuration(max(1, self.timeout_ms))
        refill = QVariantAnimation(self)                # from the current level back to full, quickly
        refill.setStartValue(self.progress if self.progress is not None else 0.0)
        refill.setEndValue(1.0)
        refill.setDuration(280)
        refill.setEasingCurve(QEasingCurve.Type.OutCubic)
        refill.valueChanged.connect(self._on_progress)

        def done():
            self._refill = None
            self._sync_countdown()
        refill.finished.connect(done)
        self._refill = refill
        refill.start(QVariantAnimation.DeletionPolicy.DeleteWhenStopped)


class ToastManager:
    """Stacks notifications in the bottom-right corner of `window` (above the status bar); newest at the bottom."""

    def __init__(self, window):
        self.window = window
        self.toasts: list[Toast] = []

    def find(self, key: str):
        return next((t for t in self.toasts if t.key == key and not t._closing), None)

    def _card_width(self) -> int:
        return max(240, min(TEXT_W, self.window.width() - 2 * MARGIN - SHADOW[0] - SHADOW[2]))

    def show_message(self, html: str, kind: str = "info", actions=(), sticky: bool = False, timeout_ms: int | None = None,
                     icon: str | None = None, key: str | None = None, details=None):
        timeout_ms = timeout_ms or (12000 if kind in ("warn", "error") else 8000)
        if key is not None:
            t = self.find(key)
            if t is not None:
                if details is not None:
                    t.details = details
                    if t._expanded:
                        t.panel.setHtml(t._details_html())
                t.update_message(html, timeout_ms)
                self.reposition()
                return t
        live = [t for t in self.toasts if not t._closing]
        while len(live) >= MAX_VISIBLE:
            live.pop(0).dismiss()
        t = Toast(self.window, html, kind, actions, sticky, timeout_ms, details=details, on_resize=self._resized,
                  width=self._card_width())
        t.key = key
        t.closed.connect(self._closed)
        self.toasts.append(t)
        targets = self._targets()
        for other in self.toasts:
            if other is not t and other in targets:      # one dismissed above is fading out where it is
                other.glide_to(targets[other])
        t.appear(targets[t])
        return t

    def _closed(self, t):
        if t in self.toasts:
            self.toasts.remove(t)
        self.reposition()

    def _targets(self) -> dict:
        w = self.window
        sb = w.statusBar().height() if w.statusBar() and not w.statusBar().isHidden() else 0    # window may be hidden
        y = w.height() - sb - MARGIN + SHADOW[3]
        out = {}
        for t in reversed([t for t in self.toasts if not t._closing]):     # newest at the bottom
            y -= t.height()
            out[t] = QPoint(max(0, w.width() - t.width() - MARGIN + SHADOW[2]), max(0, y))
            y -= GAP
        return out

    def _resized(self, toast):
        """A notification grew / shrank (details): it and the ones above follow at once — no lagging glide."""
        for t, pos in self._targets().items():
            if t is toast or t.pos().y() < toast.pos().y():
                t._glide.stop()
                if t._appearing():
                    t._rise.setEndValue(pos)
                elif t._placed and not t._closing:
                    t.move(pos)
            else:
                t.glide_to(pos)

    def reposition(self):
        for t, pos in self._targets().items():
            t.glide_to(pos)
