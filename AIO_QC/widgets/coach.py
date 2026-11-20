# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
A guided first run over the real window: the window dims, one control at a time is lit with a short note next to
it — Next / Back / Skip, Esc to stop, ← → to step.

    CoachTour(window, [Step(lambda: window.open_btn, "Your dataset", "Open a folder here …"), …], key="image").start()

A step whose control is hidden or missing is shown in the middle without a spotlight. Once finished (or skipped) a
tour is remembered (settings "tour/<key>"); maybe_start() runs it only the first time, Help ▸ Quick Tour always.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PyQt6.QtCore import QEvent, QPoint, QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from AIO_QC import utils
from AIO_QC.core.diag import ignored as _ignored

PAD = 6
BUBBLE_W = 340


@dataclass
class Step:
    target: Callable[[], QWidget | None] | None
    title: str
    text: str


def done(key: str) -> bool:
    try:
        return utils.app_settings().value(f"tour/{key}", False, type=bool)
    except Exception as _exc:
        _ignored(_exc)
        return False


def mark_done(key: str, on: bool = True):
    utils.app_settings().setValue(f"tour/{key}", bool(on))


class _Bubble(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("coachBubble")
        utils.style(self, "#coachBubble{background:@base; border:1px solid @line; border-radius:12px;}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(6)
        self.count = QLabel("")
        utils.style(self.count, "QLabel{color:@accent; font-weight:600;}")
        self.title = QLabel("")
        f = QFont(self.title.font())
        f.setPixelSize(16)
        f.setWeight(QFont.Weight.DemiBold)
        self.title.setFont(f)
        self.title.setWordWrap(True)
        self.text = QLabel("")
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.TextFormat.RichText)
        utils.style(self.text, "QLabel{color:@muted;}")
        lay.addWidget(self.count)
        lay.addWidget(self.title)
        lay.addWidget(self.text)
        lay.addSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.skip = QPushButton("Skip tour")
        self.skip.setFlat(True)
        self.skip.setCursor(Qt.CursorShape.PointingHandCursor)
        utils.style(self.skip, "QPushButton{color:@muted; border:none; background:transparent; padding:4px 0;}"
                               "QPushButton:hover{color:@text;}")
        self.back = QPushButton("Back")
        self.next = QPushButton("Next")
        self.next.setProperty("variant", "primary")
        for b in (self.back, self.next):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setAutoDefault(False)
        row.addWidget(self.skip)
        row.addStretch(1)
        row.addWidget(self.back)
        row.addWidget(self.next)
        lay.addLayout(row)
        self.setFixedWidth(BUBBLE_W)


class CoachTour(QWidget):
    def __init__(self, window: QWidget, steps: list[Step], key: str):
        super().__init__(window)
        self.win, self.steps, self.key = window, steps, key
        self.i = 0
        self.hole = QRect()
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.bubble = _Bubble(self)
        self.bubble.next.clicked.connect(self.next)
        self.bubble.back.clicked.connect(self.back)
        self.bubble.skip.clicked.connect(self.finish)
        window.installEventFilter(self)

    # ------------------------------------------------------------------ flow
    def start(self):
        self.setGeometry(self.win.rect())
        self.show()
        self.raise_()
        self.setFocus()
        self._show_step()
        return self

    def next(self):
        if self.i >= len(self.steps) - 1:
            self.finish()
            return
        self.i += 1
        self._show_step()

    def back(self):
        if self.i > 0:
            self.i -= 1
            self._show_step()

    def finish(self):
        mark_done(self.key)
        self.win.removeEventFilter(self)
        self.hide()
        self.deleteLater()
        tours = getattr(self.win, "_coach", None)
        if tours is self:
            self.win._coach = None

    # ------------------------------------------------------------------ layout
    def _target(self) -> QWidget | None:
        s = self.steps[self.i]
        try:
            w = s.target() if s.target else None
        except Exception as _exc:
            _ignored(_exc)
            w = None
        if w is None or not w.isVisible() or w.width() < 4 or w.height() < 4:
            return None
        return w

    def _show_step(self):
        s = self.steps[self.i]
        b = self.bubble
        b.count.setText(f"{self.i + 1} of {len(self.steps)}")
        b.title.setText(s.title)
        b.text.setText(s.text)
        b.back.setVisible(self.i > 0)
        b.next.setText("Done" if self.i == len(self.steps) - 1 else "Next")
        b.adjustSize()
        self._place()
        b.next.setFocus()

    def _place(self):
        self.setGeometry(self.win.rect())
        w = self._target()
        r = self.rect()
        b = self.bubble
        bw, bh = b.width(), b.sizeHint().height()
        b.resize(bw, bh)
        if w is None:
            self.hole = QRect()
            b.move((r.width() - bw) // 2, (r.height() - bh) // 2)
        else:
            tl = w.mapTo(self.win, QPoint(0, 0))
            self.hole = QRect(tl, w.size()).adjusted(-PAD, -PAD, PAD, PAD).intersected(r.adjusted(2, 2, -2, -2))
            h = self.hole
            gap = 14
            spots = [QPoint(h.left(), h.bottom() + gap),                      # below
                     QPoint(h.left(), h.top() - gap - bh),                    # above
                     QPoint(h.right() + gap, h.top()),                        # right
                     QPoint(h.left() - gap - bw, h.top())]                    # left
            best = None
            for p in spots:
                cand = QRect(p, b.size())
                if r.adjusted(8, 8, -8, -8).contains(cand):
                    best = cand
                    break
            if best is None:                                                  # a big target: inside its corner
                best = QRect(QPoint(h.right() - bw - 16, h.bottom() - bh - 16), b.size())
            x = min(max(best.x(), 8), r.width() - bw - 8)
            y = min(max(best.y(), 8), r.height() - bh - 8)
            b.move(x, y)
        b.raise_()
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        dim = QPainterPath()
        dim.addRect(QRectF(self.rect()))
        if not self.hole.isNull():
            hole = QPainterPath()
            hole.addRoundedRect(QRectF(self.hole), 10, 10)
            dim = dim.subtracted(hole)
        p.fillPath(dim, QColor(10, 12, 16, 120))
        if not self.hole.isNull():
            p.setPen(QPen(QColor(utils.hexc("accent")), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(self.hole).adjusted(1, 1, -1, -1), 10, 10)
        p.end()

    # ------------------------------------------------------------------ events
    def eventFilter(self, obj, e):
        if obj is self.win and e.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.LayoutRequest):
            self._place()
        return False

    def mousePressEvent(self, e):
        e.accept()                                    # the window under the dimming stays untouched

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_Escape:
            self.finish()
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.next()
        elif k == Qt.Key.Key_Left:
            self.back()
        else:
            super().keyPressEvent(e)


def run(window, steps, key: str, force: bool = True):
    """Start the tour (force) or only if it never ran (first use). -> the tour or None."""
    if not force and done(key):
        return None
    old = getattr(window, "_coach", None)
    if old is not None:
        try:
            old.finish()
        except RuntimeError:
            pass
    window._coach = CoachTour(window, steps, key).start()
    return window._coach
