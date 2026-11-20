# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The 3.0 window chrome, as in the approved mock-up: segmented groups (mode switch, Drag / Select), buttons with an
icon, a name, a key hint and a status dot (Open / Classes), key caps ("Ctrl K"), a bordered zoom group, and the
tasks button with its activity dot. Colours come from the design tokens (light / dark).
"""
from __future__ import annotations

from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolButton, QWidget

from AIO_QC import utils
from AIO_QC.core.diag import ignored as _ignored


def T(key: str) -> str:
    from AIO_QC.widgets.design import tokens
    return tokens()[key]


class Dot(QWidget):
    """A small filled circle (status)."""

    def __init__(self, size: int = 8, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.color = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_color(self, color: str | None):
        self.color = color
        self.setVisible(bool(color))
        self.update()

    def paintEvent(self, _e):
        if not self.color:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self.color))
        p.drawEllipse(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1))


class KeyCap(QLabel):
    """“Ctrl K”: a key in the mono font, framed."""

    def __init__(self, text: str, framed: bool = True, parent=None):
        super().__init__(text, parent)
        from AIO_QC.widgets.design import mono
        self.setFont(mono(11))
        self.framed = framed
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._restyle()

    def _restyle(self):
        css = (f"QLabel{{color:{T('muted')}; background:{T('surface')}; border:1px solid {T('control')}; "
               f"border-radius:4px; padding:0 5px;}}" if self.framed else
               f"QLabel{{color:{T('faint')}; background:transparent;}}")
        if css != self.styleSheet():
            self.setStyleSheet(css)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() == e.Type.PaletteChange:
            self._restyle()


class RichButton(QPushButton):
    """[icon] Name [key hint] [● status]  — the toolbar's Open / Classes buttons."""

    def __init__(self, text: str, icon: str = "", hint: str = "", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(7)
        self.icon_lbl = utils.IconLabel(icon, 16) if icon else None
        self.text_lbl = QLabel(text)
        self.text_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        f = self.text_lbl.font()
        f.setWeight(f.Weight.Medium)
        self.text_lbl.setFont(f)
        self.hint = KeyCap(hint, framed=False) if hint else None
        self.dot = Dot(8)
        self.dot.setVisible(False)
        for w in (self.icon_lbl, self.text_lbl, self.hint, self.dot):
            if w is not None:
                if w is not self.text_lbl and w is not self.hint and w is not self.dot:
                    w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                lay.addWidget(w)
        self.setMinimumHeight(32)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def setText(self, text):                        # the name lives in the label (QPushButton's own text stays empty)
        if hasattr(self, "text_lbl"):
            self.text_lbl.setText(text)
            self.updateGeometry()
        else:
            super().setText(text)

    def text(self):
        return self.text_lbl.text() if hasattr(self, "text_lbl") else super().text()

    def sizeHint(self):
        return self.layout().sizeHint().expandedTo(QSize(40, 32))

    def minimumSizeHint(self):
        return self.sizeHint()


class Segmented(QWidget):
    """A grey track holding buttons; the checked one sits on a white pill (mode switch, Drag / Select). 3.0.1: the
    pill glides between the buttons (eased, ~220 ms; instant with Settings ▸ Animations ▸ Off)."""

    DURATION = 220

    def __init__(self, buttons, parent=None, radius: int = 8):
        super().__init__(parent)
        from PyQt6.QtCore import QEasingCurve, QVariantAnimation
        self.setObjectName("segmented")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.buttons = list(buttons)
        for b in self.buttons:
            lay.addWidget(b)
            b.toggled.connect(lambda on, b=b: self._on_toggled(b, on))
            b.installEventFilter(self)
        self.radius = radius
        self._pill = None                      # QRectF in this widget's coordinates (None: no checked button)
        self._anim = QVariantAnimation(self)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._set_pill)
        self._restyle()

    # ---- the pill
    def _checked(self):
        return next((b for b in self.buttons if b.isChecked()), None)

    @staticmethod
    def _rect(b):
        return QRectF(b.geometry())

    def _set_pill(self, r):
        self._pill = QRectF(r) if r is not None else None
        self.update()

    def _on_toggled(self, b, on):
        if on:
            self.glide_to(b)

    def glide_to(self, b, from_button=None):
        """Move the pill to b (from from_button's place, or from where it is now)."""
        start = self._rect(from_button) if from_button is not None else self._pill
        end = self._rect(b)
        try:
            from AIO_QC.widgets import motion
            ms = motion.scaled(self.DURATION)
        except Exception as _exc:
            _ignored(_exc)
            ms = self.DURATION
        self._anim.stop()
        if start is None or ms <= 0 or not self.isVisible() or start == end:
            self._set_pill(end)
            return
        self._anim.setDuration(ms)
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.start()

    def snap(self):
        self._anim.stop()
        b = self._checked()
        self._set_pill(self._rect(b) if b is not None else None)

    def eventFilter(self, obj, e):
        if e.type() in (e.Type.Resize, e.Type.Move) and self._anim.state() != self._anim.State.Running:
            self.snap()
        return False

    def showEvent(self, e):
        super().showEvent(e)
        self.snap()

    def paintEvent(self, _e):
        from PyQt6.QtGui import QPen
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.radius
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(T("hover")))
        p.drawRoundedRect(QRectF(self.rect()), r, r)
        if self._pill is not None:
            pill = QRectF(self._pill)
            sh = QColor(0, 0, 0, 22)                           # a hairline shadow under the pill
            p.setBrush(sh)
            p.drawRoundedRect(pill.translated(0, 0.8), r - 2, r - 2)
            p.setBrush(QColor(T("surface")))
            p.setPen(QPen(QColor(T("border")), 0.8))
            p.drawRoundedRect(pill.adjusted(0.4, 0.4, -0.4, -0.4), r - 2, r - 2)

    def _restyle(self):
        r = self.radius
        css = (f"#segmented QToolButton{{border:0; border-radius:{r - 2}px; background:transparent; "
               f"color:{T('muted')}; padding:3px 10px;}}"
               f"#segmented QToolButton:hover{{color:{T('text')}; background:transparent;}}"
               f"#segmented QToolButton:pressed{{background:transparent;}}"
               f"#segmented QToolButton:checked{{background:transparent; color:{T('text')}; font-weight:600;}}")
        if css != self.styleSheet():
            self.setStyleSheet(css)
        for b in self.buttons:                 # the bold checked text never changes a button's width
            if b.toolButtonStyle() != Qt.ToolButtonStyle.ToolButtonIconOnly and b.text():
                from PyQt6.QtGui import QFont, QFontMetrics
                f = QFont(b.font())
                f.setWeight(QFont.Weight.DemiBold)
                b.setMinimumWidth(QFontMetrics(f).horizontalAdvance(b.text()) + 26)
        self.update()

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() == e.Type.PaletteChange:
            self._restyle()


class Group(QWidget):
    """A bordered, rounded group of controls (the zoom − 37 % ▾ + box)."""

    def __init__(self, widgets, parent=None):
        super().__init__(parent)
        self.setObjectName("ctlGroup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.setSpacing(0)
        for i, w in enumerate(widgets):
            if i:
                sep = QWidget()
                sep.setObjectName("ctlSep")
                sep.setFixedWidth(1)
                sep.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
                lay.addWidget(sep)
            lay.addWidget(w)
        self._restyle()

    def _restyle(self):
        css = (f"#ctlGroup{{background:{T('surface')}; border:1px solid {T('control')}; border-radius:6px;}}"
               f"#ctlGroup #ctlSep{{background:{T('border')};}}"
               f"#ctlGroup QToolButton{{border:0; border-radius:0; background:transparent; padding:3px 7px;}}"
               f"#ctlGroup QToolButton:hover{{background:{T('hover')};}}")
        if css != self.styleSheet():
            self.setStyleSheet(css)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() == e.Type.PaletteChange:
            self._restyle()


def action_button(action, icon: str | None = None, text: str | None = None, size: int = 18) -> QToolButton:
    """A tool button driven by an action (enabled / checked / tooltip follow it), optionally with its own icon."""
    b = QToolButton()
    b.setAutoRaise(True)
    b.setIconSize(QSize(size, size))
    if text is not None:
        # a text button ("1:1", "−", "+"): not the action's own text, so it is not the default action — it
        # triggers the action and follows its enabled state and tooltip
        b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        b.setText(text)
        from AIO_QC.widgets.design import mono
        if text in ("−", "+"):
            f = b.font()
            f.setPixelSize(17)
            b.setFont(f)
        else:
            b.setFont(mono(13, None))
        def click(_c=False, b=b, a=action):
            a.trigger()
            b.setChecked(a.isChecked())         # the action decides (e.g. 1:1 stays lit when already at 100 %)
        b.clicked.connect(click)

        def sync(b=b, a=action):
            b.setEnabled(a.isEnabled())
            b.setCheckable(a.isCheckable())
            b.setChecked(a.isChecked())
            sc = a.shortcut().toString()
            b.setToolTip(a.toolTip() + (f" ({sc})" if sc and sc not in a.toolTip() else ""))
        action.changed.connect(sync)
        sync()
        return b
    b.setDefaultAction(action)
    b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
    if icon:
        utils.set_icon(b, icon)
        action.changed.connect(lambda b=b, i=icon: (b.setIcon(utils.newIcon(i))))
    return b


class TasksButton(QToolButton):
    """Background Tasks with a blue dot while something runs."""

    def __init__(self, action, parent=None):
        super().__init__(parent)
        self.setDefaultAction(action)
        self.setAutoRaise(True)
        self.setIconSize(QSize(18, 18))
        utils.set_icon(self, "tasks-line")
        action.changed.connect(lambda: self.setIcon(utils.newIcon("tasks-line")))
        self.dot = Dot(9, self)
        self.dot.setVisible(False)
        from AIO_QC.widgets import tasks
        c = tasks.center()
        if c is not None:
            c.changed.connect(self._refresh)
        self._refresh()

    def _refresh(self):
        from AIO_QC.widgets import tasks
        c = tasks.center()
        n = len(c.tasks) if c is not None else 0
        self.dot.set_color(T("info") if n else None)
        self.dot.setVisible(bool(n))
        self.setToolTip("Background tasks" + (f" — {n} running" if n else ""))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.dot.move(self.width() - self.dot.width() - 3, 3)
