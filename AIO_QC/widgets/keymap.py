# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Keyboard-only work:

KeyMapOverlay   "?" (or Help ▸ Key Map) — every shortcut of the app over the window, grouped like the menus, the
                group for where the keyboard focus is (image, file list, …) first and highlighted. Type to filter,
                click an entry (or arrow + Enter) to run it, Esc / ? closes. Generated from the real actions, so it
                is always right.
PanelCycler     F6 / Shift+F6 — move the keyboard focus image → files → annotations → labels → history → QC tabs;
                the panel that receives it flashes briefly.
Ribbon keys     Alt+1 … Alt+6 open a QC tab with the focus on its first tool; arrows move, Enter runs, Esc closes
                (widgets/ribbon.Ribbon.open_tab_keyboard).
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QPoint, QRect, Qt
from PyQt6.QtGui import QAction, QColor, QKeySequence, QPainter
from PyQt6.QtWidgets import (
    QAbstractButton, QFrame, QGraphicsOpacityEffect, QGridLayout, QLabel, QLineEdit, QScrollArea, QVBoxLayout, QWidget,
)

from AIO_QC.utils import hexc, style as apply_style
from . import anim
from .mini_session import FrameAnimator

CANVAS_KEYS = [
    ("Select a shape / cycle shapes", "click · Tab · Shift+Tab"), ("Move / reshape the selected shape", "drag handles"),
    ("Nudge the selected shape", "Arrow keys (Shift = 10 px)"), ("Delete the selected shape", "Del"),
    ("Relabel the selected shape", "L · 1–9"), ("Add / remove a vertex", "Ctrl+click outline · Alt+click vertex"),
    ("Zoom at the cursor", "Ctrl+wheel"), ("Pan", "drag (Drag mode) · hold Space"),
    ("Zoom to fit ↔ in", "double-click"), ("Clear the selection", "Esc"),
]
KEYBOARD_KEYS = [
    ("This key map", "?"), ("Sketch on the image / ROI capture", "P · O"),
    ("Sketch: brush size · undo · copy · leave", "[ ] · Ctrl+Z / Ctrl+Y · Ctrl+C · Esc"), ("Next / previous panel", "F6 · Shift+F6"),
    ("Open a QC tab with the keyboard", "Alt+1 … Alt+6 (arrows, Enter, Esc)"), ("Command palette", "Ctrl+K"),
]


def _keys(act: QAction) -> str:
    return " · ".join(k.toString(QKeySequence.SequenceFormat.NativeText) for k in act.shortcuts())


def _walk(menu, prefix=""):
    for a in menu.actions():
        if a.menu() is not None:
            yield from _walk(a.menu(), prefix + a.menu().title().replace("&", "") + " ▸ ")
        elif not a.isSeparator() and a.shortcuts():
            yield prefix, a


class _Item(QLabel):
    def __init__(self, text, keys, act, overlay):
        super().__init__()
        self.act, self.overlay = act, overlay
        self.blob = f"{text} {keys}".lower()
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setText(f"<span>{text}</span>")
        self.keys = QLabel(f"<b>{keys}</b>")
        self.keys.setTextFormat(Qt.TextFormat.RichText)
        apply_style(self.keys, "color:@accent;")
        self.keys.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if act is not None:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip("Click to run")

    def mousePressEvent(self, e):
        if self.act is not None and self.act.isEnabled():
            self.overlay.close_map(then=self.act.trigger)


class KeyMapOverlay(QWidget):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.t = 0.0
        self._anim = None
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.panel = QFrame(self)
        self.panel.setObjectName("keymap")
        apply_style(self.panel, "QFrame#keymap{background:@surface; border:1px solid @line; border-radius:12px;}")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Type to filter — e.g. “flag”, “zoom”, “Ctrl+Shift” · click an entry to run it")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        self.search.installEventFilter(self)
        title = QLabel("⌨  Key map")
        apply_style(title, "font-size:14pt; font-weight:600;")
        self.ctx = QLabel("")
        apply_style(self.ctx, "color:@muted;")
        self.body = QWidget()
        self.grid = QGridLayout(self.body)
        self.grid.setHorizontalSpacing(18)
        self.grid.setVerticalSpacing(2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.body)
        lay = QVBoxLayout(self.panel)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.addWidget(title)
        lay.addWidget(self.ctx)
        lay.addWidget(self.search)
        lay.addWidget(scroll, 1)
        foot = QLabel("Esc or ? closes · Help ▸ Keyboard Shortcuts has the printable list")
        apply_style(foot, "color:@faint;")
        lay.addWidget(foot)
        self.items: list = []
        self.heads: list = []
        self.hide()

    # ------------------------------------------------------------------ content
    def _context(self) -> str:
        fw = self.app.focusWidget()
        a = self.app
        chain = []
        w = fw
        while w is not None and len(chain) < 64:
            chain.append(w)
            w = w.parentWidget()
        if a.canvas in chain:
            return "On the image"
        if getattr(a, "file_dock", None) in chain:
            return "Go"
        if getattr(a, "tabs", None) in chain:
            return "QC"
        return ""

    def _build(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.items, self.heads = [], []
        groups: dict[str, list] = {}
        seen = set()
        for top in self.app.menuBar().actions():
            m = top.menu()
            if m is None:
                continue
            name = m.title().replace("&", "")
            for prefix, a in _walk(m):
                if id(a) in seen:
                    continue
                seen.add(id(a))
                groups.setdefault(name, []).append((prefix + a.text().replace("&", "").rstrip("…"), _keys(a), a))
        groups["On the image"] = [(t, k, None) for t, k in CANVAS_KEYS]
        groups["Keyboard"] = [(t, k, None) for t, k in KEYBOARD_KEYS]
        ctx = self._context()
        order = sorted(groups, key=lambda g: (g != ctx, g not in ("On the image", "Keyboard"), 0))
        self.ctx.setText(f"Focus: {ctx} — its keys are shown first" if ctx else "Every shortcut, grouped like the menus")
        cols = 2
        col_rows = [0] * cols
        for gi, g in enumerate(order):
            c = col_rows.index(min(col_rows))
            r = col_rows[c]
            head = QLabel(g)
            col = hexc("accent") if g == ctx else ""
            head.setStyleSheet(f"font-weight:600; margin-top:8px; color:{col};")
            self.grid.addWidget(head, r, c * 2, 1, 2)
            self.heads.append((head, []))
            r += 1
            for text, keys, act in groups[g]:
                it = _Item(text, keys, act, self)
                self.grid.addWidget(it, r, c * 2)
                self.grid.addWidget(it.keys, r, c * 2 + 1)
                self.items.append(it)
                self.heads[-1][1].append(it)
                r += 1
            col_rows[c] = r
        self.grid.setRowStretch(max(col_rows) + 1, 1)

    def _filter(self, text):
        t = text.lower().strip()
        for head, members in self.heads:
            any_on = False
            for it in members:
                on = not t or all(w in it.blob for w in t.split())
                it.setVisible(on)
                it.keys.setVisible(on)
                any_on |= on
            head.setVisible(any_on)

    # ------------------------------------------------------------------ show / hide
    def toggle(self):
        if self.isVisible() and self.t > 0.5:
            self.close_map()
        else:
            self.open_map()

    def _layout(self):
        r = self.app.rect()
        top = self.app.menuBar().height()
        self.setGeometry(0, top, r.width(), r.height() - top)
        w = min(980, int(self.width() * 0.88))
        h = min(720, int(self.height() * 0.86))
        self.final = QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def open_map(self):
        self._build()
        self.search.clear()
        self._layout()
        self.show()
        self.raise_()
        self.app.installEventFilter(self)
        self.search.setFocus()
        self._go(1.0)

    def close_map(self, then=None):
        self.app.removeEventFilter(self)

        def done():
            self.hide()
            if then is not None:
                then()
        self._go(0.0, done)

    def _go(self, target, done=None):
        if self._anim is not None:
            self._anim.stop()
        start = self.t
        if not anim.ENABLED:
            self._apply(target)
            if done:
                done()
            return

        def step(p):
            self._apply(start + (target - start) * p)
        self._anim = FrameAnimator(self, 200 if target else 150, step, done)
        self._anim.start()

    def _apply(self, t):
        self.t = t
        f = self.final
        dy = int((1.0 - t) * 26)
        self.panel.setGeometry(f.translated(0, dy))
        eff = self.panel.graphicsEffect()
        if t >= 0.999:
            if eff is not None:
                self.panel.setGraphicsEffect(None)       # settled: render directly (crisp, no offscreen pass)
        else:
            if eff is None:
                eff = QGraphicsOpacityEffect(self.panel)
                self.panel.setGraphicsEffect(eff)
            eff.setOpacity(t)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        c = QColor(0, 0, 0)
        c.setAlphaF(0.38 * self.t)
        p.fillRect(self.rect(), c)

    def mousePressEvent(self, e):
        if not self.panel.geometry().contains(e.position().toPoint()):
            self.close_map()

    def eventFilter(self, obj, e):
        if e.type() == QEvent.Type.KeyPress:
            if e.key() == Qt.Key.Key_Escape or (e.text() == "?" and not self.search.text()):
                self.close_map()
                return True
            if obj is self.search and e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                vis = [i for i in self.items if i.isVisible() and i.act is not None]
                if len(vis) >= 1:
                    self.close_map(then=vis[0].act.trigger)
                return True
        if obj is self.app and e.type() == QEvent.Type.Resize and self.isVisible():
            self._layout()
            self._apply(self.t)
        return False


# ---------------------------------------------------------------------------------------------------- F6
class PanelCycler:
    def __init__(self, app):
        self.app = app

    def targets(self):
        a = self.app
        out = [("Image", a.canvas)]
        for name, dock, w in (("File List", "file_list_dock", lambda: a.file_dock.view),
                              ("Annotations", "annotations_dock", lambda: a.annotations_list),
                              ("Labels", "unique_labels_dock", lambda: a.unique_labels),
                              ("History", "history_dock", lambda: a.history_panel.events),
                              ("Thumbnails", "filmstrip_dock", lambda: a.filmstrip)):
            d = getattr(a, dock, None)
            if d is not None and d.isVisible():
                out.append((name, w()))
        if a.tabs.isVisible():
            out.append(("QC tabs", a.tabs.bar))
        return [(n, w) for n, w in out if w is not None and w.isVisible()]

    def cycle(self, step=1):
        ts = self.targets()
        if not ts:
            return
        fw = self.app.focusWidget()
        cur = -1
        for i, (_n, w) in enumerate(ts):
            if fw is not None and (fw is w or w.isAncestorOf(fw)):
                cur = i
        name, w = ts[(cur + step) % len(ts)]
        w.setFocus(Qt.FocusReason.ShortcutFocusReason)
        flash(self.app, w)
        self.app.statusBar().showMessage(f"⌨ Focus: {name} — F6 next panel · ? key map", 2500)


def flash(main, target: QWidget, ms: int = 520):
    """A rounded accent frame around `target` that fades out."""
    if not target.isVisible():
        return
    tl = target.mapTo(main, QPoint(0, 0))
    ring = _Ring(main)
    ring.setGeometry(QRect(tl, target.size()).adjusted(-2, -2, 2, 2))
    ring.show()
    ring.raise_()
    if not anim.ENABLED:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(ms, ring.deleteLater)
        return

    def step(t):
        try:
            ring.a = 1.0 - t
            ring.update()
        except RuntimeError:
            pass
    FrameAnimator(ring, ms, step, ring.deleteLater).start()


class _Ring(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.a = 1.0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(hexc("accent"))
        c.setAlphaF(max(0.0, self.a))
        from PyQt6.QtGui import QPen
        p.setPen(QPen(c, 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 6, 6)


def focus_buttons(page: QWidget) -> list:
    return [b for b in page.findChildren(QAbstractButton) if b.isVisibleTo(page) and b.isEnabled()]
