# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
A slim, dismissible card floating at the top of the image canvas for guidance ("Start here: Label Ambiguities — …").
3.0.1: it floats over the image (the canvas never resizes when it comes or goes), its text is cut with "…" to the
width it has (full text in the tooltip), and it never makes the window wider. Corner notifications stay for results.

    banner.show_hint("Start here: Label Ambiguities", "'Car' and 'car' look like one class", [("Run check", fn), …])
    banner.clear()
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QTimer, Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolButton, QWidget

from AIO_QC import utils


class _HintText(QLabel):
    """Title (accent, bold) — detail; the detail is elided to the width the label has."""

    def __init__(self):
        super().__init__("")
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self._title = self._detail = ""
        self._full = ""

    def set_parts(self, title, detail):
        self._title, self._detail = title, detail
        self._render()

    def text(self):
        return self._full

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()

    def _render(self):
        from html import escape
        from PyQt6.QtGui import QFont, QFontMetrics
        from AIO_QC.widgets.design import tokens
        t = tokens()
        bf = QFont(self.font())
        bf.setWeight(QFont.Weight.DemiBold)
        fm, bfm = QFontMetrics(self.font()), QFontMetrics(bf)
        avail = max(0, self.width() - 4)
        title = bfm.elidedText(self._title, Qt.TextElideMode.ElideRight, avail)
        left = avail - bfm.horizontalAdvance(title) - fm.horizontalAdvance(" — ")
        detail = fm.elidedText(self._detail, Qt.TextElideMode.ElideRight, left) if self._detail and left > 24 else ""
        self._full = (f"<span style='color:{t['accent_text']}; font-weight:600'>{escape(self._title)}</span>"
                      + (f"<span style='color:{t['text']}'> — {escape(self._detail)}</span>" if self._detail else ""))
        QLabel.setText(self, f"<span style='color:{t['accent_text']}; font-weight:600'>{escape(title)}</span>"
                       + (f"<span style='color:{t['text']}'> — {escape(detail)}</span>" if detail else ""))


class HintBanner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("hintBanner")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 6, 8, 6)
        lay.setSpacing(10)
        self.icon = utils.IconLabel("lightbulb", 18)
        self.text = _HintText()
        self.buttons = QHBoxLayout()
        self.buttons.setSpacing(6)
        close = QToolButton()
        close.setAutoRaise(True)
        close.setIcon(utils.newIcon("close-x"))
        close.setProperty("aio_icon", "close-x")
        close.setToolTip("Hide this hint")
        close.setAccessibleName("Hide this hint")
        close.clicked.connect(lambda _c=False: self.clear())
        lay.addWidget(self.icon)
        lay.addWidget(self.text, 1)
        lay.addLayout(self.buttons)
        lay.addWidget(close)
        self.key = None
        self._timer = QTimer(self, singleShot=True, timeout=self.clear)
        self._restyle()
        self.setVisible(False)

    def _restyle(self):
        from AIO_QC.widgets import design
        t = design.tokens()
        css = (f"#hintBanner {{ background: {t['accent_soft']}; border: 1px solid {t['accent_mid']}; "
               f"border-radius: 8px; }}"
               f"#hintBanner QLabel {{ color: {t['text']}; background: transparent; }}")
        if css != self.styleSheet():                 # setStyleSheet changes the palette: never loop on it
            self.setStyleSheet(css)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() == e.Type.PaletteChange:
            self._restyle()

    def show_hint(self, title: str, detail: str = "", actions=(), key: str | None = None, icon: str = "lightbulb",
                  timeout_ms: int = 0):
        self.key = key
        self.icon.set_icon(icon)
        self.text.set_parts(title, detail)
        self.text.setToolTip(f"{title} — {detail}" if detail else title)
        while self.buttons.count():
            w = self.buttons.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for i, (label, fn) in enumerate(actions or ()):
            b = QPushButton(label)
            if i == 0:
                b.setProperty("variant", "primary")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c=False, f=fn: (self.clear(), f()))
            self.buttons.addWidget(b)
        self._restyle()
        fresh = not self.isVisible() or self._hiding
        self._hiding = False
        self.setVisible(True)
        if fresh:
            self._k = 0.0
            self._animate(1.0, 300, "out")                 # glides down into place while fading in
        else:
            self._pulse()                                  # new text in place: a soft fade-through
        self.place()
        self.raise_()
        if timeout_ms:
            self._timer.start(timeout_ms)
        else:
            self._timer.stop()

    def clear(self, key: str | None = None):
        if key is not None and key != self.key:
            return
        self._timer.stop()
        self.key = None
        if self.isVisible() and not self._hiding:
            self._hiding = True
            self._animate(0.0, 220, "in", done=self._hidden)   # glides up and fades out
        elif not self._hiding:
            self.setVisible(False)
        self._notify()

    # ---- motion (one eased step per display frame; instant with Settings ▸ Animations ▸ Off)
    _k = 1.0
    _hiding = False
    _anim = None

    def _effect(self):
        from PyQt6.QtWidgets import QGraphicsOpacityEffect
        eff = self.graphicsEffect()
        if eff is None:
            eff = QGraphicsOpacityEffect(self)
            self.setGraphicsEffect(eff)
        return eff

    def _animate(self, to, ms, ease, done=None):
        from PyQt6.QtCore import QEasingCurve
        from AIO_QC.widgets import motion
        from AIO_QC.widgets.mini_session import FrameAnimator
        if self._anim is not None:
            self._anim.stop()
        ms = motion.scaled(ms)
        start = self._k
        if ms <= 0 or abs(to - start) < 1e-3:
            self._set_k(to)
            self._settle(done)
            return

        def step(t):
            self._set_k(start + (to - start) * t)

        curve = QEasingCurve.Type.OutCubic if ease == "out" else QEasingCurve.Type.InCubic
        self._anim = FrameAnimator(self, ms, step, lambda: self._settle(done), curve)
        self._anim.start()

    def _set_k(self, k):
        self._k = max(0.0, min(1.0, k))
        self._effect().setOpacity(self._k)
        self.place()

    def _settle(self, done=None):
        self._anim = None
        if self._k >= 1.0:
            self.setGraphicsEffect(None)                  # no effect at rest: crisp text, no extra rendering
        if done is not None:
            done()

    def _hidden(self):
        if self._hiding:
            self._hiding = False
            self.setVisible(False)
            self.setGraphicsEffect(None)
            self._k = 1.0
            self._notify()

    def _pulse(self):
        from AIO_QC.widgets import motion
        from AIO_QC.widgets.mini_session import FrameAnimator
        ms = motion.scaled(260)
        if ms <= 0:
            return
        eff = self._effect()

        def step(t):
            eff.setOpacity(0.35 + 0.65 * t)
        self._anim = FrameAnimator(self, ms, step, lambda: self._settle())
        self._anim.start()

    # ---- floating over the canvas
    MARGIN = 12

    def attach(self, view, on_change=None):
        """Float over a QGraphicsView / scroll area (a child of the view, not of its viewport, so it never scrolls
        with the image); on_change() runs when it shows, hides or changes height (the read-outs move below it)."""
        self.setParent(view)
        self._view = view
        self._on_change = on_change
        view.installEventFilter(self)
        self.setVisible(False)

    def eventFilter(self, obj, e):
        if obj is getattr(self, "_view", None) and e.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.place()
        return False

    def place(self):
        v = getattr(self, "_view", None)
        if v is None:
            return
        g = v.viewport().geometry()
        m = self.MARGIN
        w = max(120, g.width() - 2 * m)
        h = self.sizeHint().height()
        y = g.top() + m - round((1.0 - self._k) * (h + m) * 0.5)      # slides a little while it fades
        self.setGeometry(g.left() + m, y, w, h)
        self._notify()

    def occupied(self) -> int:
        """Height taken at the top of the canvas (0 when hidden), margin included."""
        if not self.isVisible() or getattr(self, "_view", None) is None:
            return 0
        return round((self.height() + self.MARGIN) * self._k)      # the read-outs glide along

    def _notify(self):
        fn = getattr(self, "_on_change", None)
        if fn is not None:
            try:
                fn()
            except RuntimeError:
                pass
