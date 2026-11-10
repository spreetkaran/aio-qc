# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Notifications: small, colour-coded, translucent pop-ups in the bottom-right corner of the main window.

    window.toast.show_message("<b>Updated from disk</b> — 3 changed", kind="info", details=fn_returning_text)

details=…: a "Details" button that grows the notification in place (taller, and wider when the text needs it) into a
scrollable panel — no extra window. One smooth animation for size and position (the panel is revealed as it grows; no
second opacity effect — Qt cannot nest them); while open, the
countdown waits. The text is fetched when it opens (a callable) so it is always current.

kind: info (blue) · success (green) · warn (amber) · error (red). Fixed saturated colours with white text, so they read
the same in light and dark themes.

Motion: they fade in while sliding up, fade out while drifting right, and the others glide into place. The countdown
bar is a continuous animation at the display's frame rate (sub-pixel width, only the bar's strip is repainted); it
only runs while AIO QC is the active window and the mouse is not over the notification, so a change noticed while
you work elsewhere waits for you. key=…: a notification with the same key is updated in place (a short pulse, the bar
refills smoothly) instead of stacking. Max 4 stacked; sticky=True stays until dismissed.

Painting: the fade is the ONLY graphics effect — Qt does not support an effect inside another one (that produced
"A paint device can only be painted by one painter at a time" floods and blank notifications).
"""
from __future__ import annotations

from PyQt6.QtCore import (QEasingCurve, QParallelAnimationGroup, QPoint, QPropertyAnimation, QRect, QRectF, Qt,
                          QVariantAnimation, pyqtSignal)
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (QApplication, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QPushButton,
                             QTextBrowser, QVBoxLayout, QWidget)

from AIO_QC.utils import tintedIcon

KINDS = {  # kind -> (rgb, icon)
    "info": ((21, 101, 192), "undo"),
    "success": ((46, 125, 50), "review-ok"),
    "warn": ((178, 106, 0), "review-flag"),
    "error": ((198, 40, 40), "review-clear"),
}
ALPHA = 225                      # ~88 % opaque: the window shows through slightly
MAX_VISIBLE = 4
MARGIN = 16
GAP = 2
SLIDE = 14                       # px the notification rises while appearing
TEXT_W = 380                     # message width
DETAILS_W = 600                  # widest a notification grows for its details
DETAILS_H = 300                  # tallest details panel (it scrolls beyond)
EXPAND_MS = 340
BTN_CSS = ("QPushButton{color:white; background: rgba(255,255,255,0.14); border:1px solid rgba(255,255,255,0.55); "
           "border-radius:6px; padding:3px 12px;} QPushButton:hover{background: rgba(255,255,255,0.28);}")


class _Card(QFrame):
    """Layout container for the content (the Toast paints body, shadow and bar itself)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


class Toast(QWidget):
    closed = pyqtSignal(object)

    def __init__(self, parent, html: str, kind: str, actions, sticky: bool, timeout_ms: int, details=None,
                 on_resize=None):
        super().__init__(parent)
        self.details, self.on_resize = details, on_resize
        self._expanded = False
        self._expand_t = 0.0                   # 0 collapsed … 1 open (animated)
        self._expand_anim = None
        self._details_size = (0, 0)
        self.rgb, icon = KINDS.get(kind, KINDS["info"])
        self.sticky, self.timeout_ms = sticky, timeout_ms
        self.key = None
        self.progress = None if sticky else 1.0
        self._pulse = 0.0                      # 0..1 brightening when the content is updated in place
        self._closing = False
        self._hover = False
        self._placed = False
        self.card = _Card(self)

        from AIO_QC.utils import IconLabel
        ic = IconLabel(tintedIcon(icon, "#ffffff"), 22)          # sharp at any screen scaling
        self.text = QLabel(html)
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.TextFormat.RichText)
        self.text.setMaximumWidth(TEXT_W)
        self.text.setStyleSheet("color: white; background: transparent;")
        close = QPushButton("✕")
        close.setFlat(True)
        close.setFixedSize(22, 22)
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.setStyleSheet("QPushButton{color: rgba(255,255,255,0.85); border:none; background:transparent;}"
                            "QPushButton:hover{color:white;}")
        close.clicked.connect(lambda _c=False: self.dismiss())
        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(ic, 0, Qt.AlignmentFlag.AlignTop)
        top.addWidget(self.text, 1)
        top.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        lay = QVBoxLayout(self.card)
        lay.setContentsMargins(14, 10, 10, 12)
        lay.addLayout(top)
        self.panel = None
        if details is not None:
            self.panel = QTextBrowser()
            self.panel.setOpenExternalLinks(False)
            self.panel.setFrameShape(QFrame.Shape.NoFrame)
            self.panel.setStyleSheet("QTextBrowser{color:white; background: rgba(0,0,0,0.22); border-radius:6px; "
                                     "padding:6px;} QScrollBar:vertical{width:8px; background:transparent;}"
                                     "QScrollBar::handle:vertical{background: rgba(255,255,255,0.45); "
                                     "border-radius:4px; min-height:20px;}"
                                     "QScrollBar::add-line, QScrollBar::sub-line{height:0;}")
            self.panel.setFixedHeight(0)
            self.panel.setVisible(False)
            lay.addWidget(self.panel)
        if actions or details is not None:
            row = QHBoxLayout()
            row.addStretch(1)
            if details is not None:
                self.details_btn = QPushButton("Details  ▾")
                self.details_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                self.details_btn.setStyleSheet(BTN_CSS)
                self.details_btn.clicked.connect(lambda _c=False: self.toggle_details())
                row.addWidget(self.details_btn)
            for label, fn in actions or ():
                b = QPushButton(label)
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.setStyleSheet(BTN_CSS)
                b.clicked.connect(lambda _c=False, f=fn: (f(), self.dismiss()))
                row.addWidget(b)
            lay.addLayout(row)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 10)          # room for the shadow
        outer.addWidget(self.card)
        self.adjustSize()

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

    # ------------------------------------------------------------------ painting
    def _body(self) -> QRectF:
        return QRectF(self.card.geometry()).adjusted(0.5, 0.5, -0.5, -0.5)

    def _bar_rect(self) -> QRect:
        b = self._body()
        return QRect(int(b.left()) + 6, int(b.bottom()) - 8, int(b.width()) - 10, 9)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = self._body()
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(7, 0, -1):                       # soft shadow: widening, fading rings below the body
            p.setBrush(QColor(0, 0, 0, int(14 * (1 - i / 8))))
            p.drawRoundedRect(body.adjusted(-i, -i + 3, i, i + 3), 10 + i, 10 + i)
        c = QColor(*self.rgb, ALPHA)
        if self._pulse > 0:                             # updated in place: briefly brighter
            c = c.lighter(100 + int(28 * self._pulse))
        p.setPen(QColor(255, 255, 255, 60))
        p.setBrush(c)
        p.drawRoundedRect(body, 10, 10)
        if self.progress is not None:                   # time left: track + sub-pixel bar
            track = QRectF(body.left() + 10, body.bottom() - 5.5, body.width() - 20, 3.0)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 45))
            p.drawRoundedRect(track, 1.5, 1.5)
            w = track.width() * max(0.0, min(1.0, self.progress))
            if w > 0.5:
                p.setBrush(QColor(255, 255, 255, 190))
                p.drawRoundedRect(QRectF(track.left(), track.top(), w, track.height()), 1.5, 1.5)
        p.end()

    def _on_progress(self, v):
        self.progress = float(v)
        self.update(self._bar_rect())                   # repaint only the thin strip, every frame

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

    # ------------------------------------------------------------------ details (expand in place)
    def _details_html(self) -> str:
        try:
            v = self.details() if callable(self.details) else self.details
        except Exception as exc:                                  # never break the notification for it
            v = f"Details unavailable: {exc}"
        v = "" if v is None else str(v)
        if "<" in v and ">" in v:
            return v
        from html import escape
        return ("<pre style='white-space:pre-wrap; margin:0; font-family:monospace; font-size:9pt;'>"
                f"{escape(v) or 'Nothing to show.'}</pre>")

    def toggle_details(self):
        if self.panel is None or self._closing:
            return
        self._expanded = not self._expanded
        if self._expanded:
            if self._expand_t == 0.0:                   # the collapsed message width: grow from it, return to it
                self._base_w = max(80, self.text.width())
            self.panel.setHtml(self._details_html())
            fm = self.panel.fontMetrics()
            doc = self.panel.document()
            longest = max((fm.horizontalAdvance(l) for l in doc.toPlainText().splitlines()[:400]), default=0)
            w = int(min(DETAILS_W, max(TEXT_W, longest + 40)))
            doc.setTextWidth(w - 24)
            h = int(min(DETAILS_H, doc.size().height() + 16))
            self._details_size = (w, max(48, h))
            self.panel.setVisible(True)
            self.details_btn.setText("Hide details  ▴")
        else:
            self.details_btn.setText("Details  ▾")
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
        """One frame: panel height + message width, then the whole notification re-anchors."""
        t = float(t)
        self._expand_t = t
        w, h = self._details_size
        base = getattr(self, "_base_w", TEXT_W)
        cur = int(round(base + (max(w, base) - base) * t))
        self.panel.setFixedHeight(int(round(h * t)))
        self.panel.setFixedWidth(cur)
        self.text.setMaximumWidth(max(TEXT_W, cur) if t > 0 else TEXT_W)
        if t > 0:
            self.text.setMinimumWidth(max(0, cur - 64))  # beside icon + ✕: as wide as the panel, no jump
        else:
            self.text.setMinimumWidth(0)
        self._fit()
        if self.on_resize is not None:
            self.on_resize(self)                                 # stays bottom-right anchored, every frame

    def _fit(self):
        """Resize to the content — also narrower again (adjustSize keeps the width of a word-wrapped layout)."""
        for lay in (self.card.layout(), self.layout()):
            if lay is not None:
                lay.invalidate()
                lay.activate()
        self.resize(self.sizeHint())

    def _expand_done(self, to):
        self._expand_anim = None
        if to == 0.0 and self.panel is not None:
            self.panel.setVisible(False)
            self.panel.setMinimumWidth(0)
            self.panel.setMaximumWidth(16777215)
            self.text.setMinimumWidth(0)
            self.text.setMaximumWidth(TEXT_W)
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
        fade.setDuration(260)
        fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        rise = QPropertyAnimation(self, b"pos", self)
        rise.setStartValue(self.pos())
        rise.setEndValue(target)
        rise.setDuration(260)
        rise.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._run_motion(fade, rise)
        self._sync_countdown()

    def glide_to(self, target: QPoint):
        """Another notification left / arrived: move smoothly to the new slot."""
        if not self._placed or self._closing:
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
        fade.setDuration(220)
        fade.setEasingCurve(QEasingCurve.Type.InCubic)
        drift = QPropertyAnimation(self, b"pos", self)
        drift.setStartValue(self.pos())
        drift.setEndValue(self.pos() + QPoint(24, 0))
        drift.setDuration(220)
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
        """Same notification, new content: a short pulse, the bar refills smoothly, full time again."""
        self.text.setText(html)
        self.adjustSize()
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
        t = Toast(self.window, html, kind, actions, sticky, timeout_ms, details=details, on_resize=self._resized)
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
        y = w.height() - sb - MARGIN
        out = {}
        for t in reversed([t for t in self.toasts if not t._closing]):     # newest at the bottom
            t.adjustSize()
            y -= t.height()
            out[t] = QPoint(max(MARGIN, w.width() - t.width() - MARGIN), max(MARGIN, y))
            y -= GAP
        return out

    def _resized(self, toast):
        """A notification grew / shrank (details): it and the ones above follow at once — no lagging glide."""
        for t, pos in self._targets().items():
            if t is toast or t.pos().y() < toast.pos().y():
                t._glide.stop()
                if t._placed and not t._closing:
                    t.move(pos)
            else:
                t.glide_to(pos)

    def reposition(self):
        for t, pos in self._targets().items():
            t.glide_to(pos)
