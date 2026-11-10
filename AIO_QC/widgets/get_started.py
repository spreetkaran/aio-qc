# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Get Started — the guided overview shown on the first start (and any time from Help ▸ Get Started).

A short tour in pages (← / → or the dots, Enter = next, Esc = close): what the app is for, opening a dataset, the
window, finding problems, reviewing and fixing, team work (hand-off batches, guidelines, history, reports), safety
and the keyboard. "Show me" dims the window and spotlights the real part of the interface being described, with a
caption; click or press any key to come back to the tour.

Motion (all on FrameAnimator — one step per display refresh, eased; Settings ▸ General ▸ Animations off = instant):
    window fades in · pages slide and cross-fade · the hero icon rises in · the active progress dot stretches ·
    the spotlight glides from one part of the window to the next.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget,
)

from AIO_QC.utils import hexc, newButton, newIcon, style as apply_style
from . import anim
from .mini_session import FrameAnimator, lerp_rect


@dataclass
class Page:
    icon: str
    title: str
    lead: str
    points: list = field(default_factory=list)         # (icon, bold, text)
    show: list = field(default_factory=list)           # [(target name, caption)] for "Show me"


PAGES = [
    Page("get-started", "Welcome to AIO QC",
         "Find and fix the problems in an annotated image dataset <b>before</b> you train on it — broken files, "
         "wrong label names, sloppy or duplicate shapes, split leakage and, with your own models, labels that are "
         "probably wrong.",
         [("group-dataset", "Ranked findings", "every tool lists problems worst-first and jumps to each one"),
          ("review-ok", "You decide", "nothing changes your files until you ask — and every change can be undone"),
          ("tool-inbox", "One queue", "all findings meet in the Issues Inbox with a risk score per file")]),
    Page("open", "Open a dataset",
         "Toolbar <b>Open Dir</b> (Ctrl+O), or drop a folder on the window. LabelMe, YOLO (images / labels, "
         "split folders) and classification layouts are detected for you.",
         [("list", "Class names", "YOLO reads dataset.yaml / classes.txt; for LabelMe, Load Classes (Ctrl+L) "
                                  "lets the app spot unknown labels"),
          ("logs", "Fast on huge datasets", "a small cache in .aio_qc makes the second open of a million files "
                                            "near-instant"),
          ("refresh", "Live", "files changed by other programs are picked up while you work"),
          ("tool-dataprep", "Raw data first?", "Tools ▸ Data Prep (Ctrl+Alt+P) unzips, combines & flattens, sorts "
                                               "orphans, quarantines broken files, removes exact duplicates and "
                                               "resizes — every run can be undone")],
         [("toolbar", "Open Dir, navigation, zoom and the review buttons live here.")]),
    Page("fit-window", "The window",
         "The image in the middle, panels around it. The <b>QC tabs</b> above the image slide down when you rest on "
         "or click a tab, and slide away when you move off — pin them to keep them open.",
         [("list", "File List", "every file with its review mark; filter, search, sort"),
          ("edit", "Annotations", "the shapes of the image — right-click for relabel, delete, guideline examples"),
          ("history", "History", "who changed this file, when and with which tool (Ctrl+H)")],
         [("ribbon", "QC tabs — hover or click to open, double-click (or the pin) to keep them open."),
          ("files", "File List — review marks, filters and search."),
          ("canvas", "The image — select a shape to edit it right here."),
          ("annotations", "The shapes of the current image.")]),
    Page("group-shapes", "Find problems, tab by tab",
         "Work left to right — each tab answers one question:",
         [("group-dataset", "Dataset", "are files and images sound? (validation, duplicates, quality, split health)"),
          ("group-labels", "Labels", "are the names right? (ambiguities, mapping, guidelines)"),
          ("group-shapes", "Shapes", "are the annotations right? (crop gallery, outliers, dim / area, rules)"),
          ("group-models", "Models", "what do your models disagree with? (model check, OCR, visual similarity)"),
          ("group-review", "Review", "sign off: inbox, sampling, review grid, report, hand-off batches")],
         [("ribbon", "Five tabs, in workflow order. Alt+1 … Alt+5 open them from the keyboard.")]),
    Page("review-flag", "Review and fix",
         "Mark files as you go: <b>S</b> OK · <b>X</b> flag (with a note) · <b>N</b> next unreviewed. Fix small "
         "things directly on the image:",
         [("cursor", "Select a shape", "drag its handles or body; arrows nudge; Del deletes"),
          ("edit", "Relabel", "L, or 1–9 for the first nine classes"),
          ("undo", "Undo", "Ctrl+Z / Ctrl+Shift+Z — exact, byte for byte"),
          ("tool-editor", "Bigger fixes", "Fix in Editor opens flagged files in labelme / labelImg and checks "
                                          "every save")],
         [("canvas", "Click a shape (Select mode) to edit it here."),
          ("status", "The review count of the dataset.")]),
    Page("tool-handoff", "Work as a team",
         "Hand work out, keep everyone on the same rules, and show where things stand:",
         [("tool-handoff", "Hand-off batches", "send flagged files to an annotator with notes and hint images; "
                                               "merge the returned folder with conflict checks"),
          ("tool-guidelines", "Guidelines", "the rule for each class with ✓ / ✗ example crops, shown next to the "
                                            "selected shape and in the tools"),
          ("tool-report", "QC report", "one click: an HTML / PDF of balance, coverage, findings and changes"),
          ("history", "History", "every change per file, with revert to any saved version"),
          ("tool-sketch", "Sketch & ROI", "P draws on the image to explain a point (copy / save it); O captures "
                                          "boxes with their real coordinates")]),
    Page("undo", "Safe by design",
         "Every write keeps the original first. Undo from the tool, from <b>Tools ▸ Storage…</b>, or "
         "from the file's History.",
         [("delete", "Trash, not delete", "removed files go to .aio_qc_trash and can be restored"),
          ("logs", "Small footprint", "originals are stored compressed and deduplicated; Storage shows and trims it"),
          ("review-fixed", "Fixed ≠ OK", "a file corrected anywhere becomes 🔧 Fixed until someone looks again")]),
    Page("keyboard", "Keyboard first",
         "Everything has a key. The four to remember:",
         [("keyboard", "?", "the key map — every shortcut, type to filter, click to run"),
          ("zoom", "Ctrl+K", "command palette — any action or tool by name"),
          ("list", "F6", "jump between panels (image, files, annotations, …)"),
          ("group-review", "Alt+1 … 6", "open a QC tab with the keyboard; arrows and Enter pick a tool")]),
    Page("review-ok", "You're ready",
         "Open a dataset to begin. The full User Guide (F1) explains every tool; Get Started is always in the "
         "Help menu.", []),
]


# ---------------------------------------------------------------------------------------------------- widgets
class Dots(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, n):
        super().__init__()
        self.n, self.pos_f = n, 0.0
        self.setFixedHeight(18)
        self.setMinimumWidth(n * 18 + 20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _xs(self):
        w = self.n * 14 + 16
        x0 = (self.width() - w) / 2
        return x0

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        acc, mut = QColor(hexc("accent")), QColor(hexc("muted"))
        mut.setAlphaF(0.45)
        x = self._xs()
        y = self.height() / 2
        for i in range(self.n):
            k = max(0.0, 1.0 - abs(self.pos_f - i))          # 1 at the active dot, eased between neighbours
            w = 6 + 16 * k
            c = QColor(mut)
            c.setRedF(mut.redF() + (acc.redF() - mut.redF()) * k)
            c.setGreenF(mut.greenF() + (acc.greenF() - mut.greenF()) * k)
            c.setBlueF(mut.blueF() + (acc.blueF() - mut.blueF()) * k)
            c.setAlphaF(0.45 + 0.55 * k)
            p.setBrush(c)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(x, y - 3, w, 6), 3, 3)
            x += w + 8

    def mousePressEvent(self, e):
        x = self._xs()
        for i in range(self.n):
            k = max(0.0, 1.0 - abs(self.pos_f - i))
            w = 6 + 16 * k
            if x - 4 <= e.position().x() <= x + w + 4:
                self.clicked.emit(i)
                return
            x += w + 8


class _Transition(QWidget):
    """Paints the old page sliding out and the new one sliding in (both pre-rendered — no layout per frame)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.old = self.new = None
        self.t, self.dir = 0.0, 1
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        d = 70 * self.dir
        if self.old is not None:
            p.setOpacity(max(0.0, 1.0 - self.t * 1.4))
            p.drawPixmap(QPointF(-d * self.t, 0), self.old)
        if self.new is not None:
            p.setOpacity(min(1.0, self.t * 1.25))
            p.drawPixmap(QPointF(d * (1.0 - self.t), 0), self.new)


def _page_widget(pg: Page, on_show) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(36, 18, 36, 6)
    lay.setSpacing(10)
    head = QHBoxLayout()
    from AIO_QC.utils import IconLabel
    icon = IconLabel(pg.icon, 64)                   # painted at the screen's resolution: sharp at any scaling
    icon.setFixedSize(72, 72)
    title = QLabel(pg.title)
    f = QFont(title.font())
    f.setPointSizeF(f.pointSizeF() * 1.9)
    f.setWeight(QFont.Weight.DemiBold)
    title.setFont(f)
    head.addWidget(icon)
    head.addSpacing(10)
    head.addWidget(title, 1)
    lay.addLayout(head)
    lead = QLabel(pg.lead)
    lead.setWordWrap(True)
    lead.setTextFormat(Qt.TextFormat.RichText)
    lf = QFont(lead.font())
    lf.setPointSizeF(lf.pointSizeF() * 1.12)
    lead.setFont(lf)
    lay.addWidget(lead)
    lay.addSpacing(4)
    for ic, bold, text in pg.points:
        row = QHBoxLayout()
        i = IconLabel(ic, 26)
        i.setFixedWidth(34)
        t = QLabel(f"<b>{bold}</b> — {text}")
        t.setWordWrap(True)
        t.setTextFormat(Qt.TextFormat.RichText)
        row.addWidget(i)
        row.addWidget(t, 1)
        lay.addLayout(row)
    lay.addStretch(1)
    if pg.show:
        b = newButton("👁 Show me", slot=lambda: on_show(pg.show),
                      tip="Highlight these parts of the window (click or any key to come back)")
        r = QHBoxLayout()
        r.addWidget(b)
        r.addStretch(1)
        lay.addLayout(r)
    w.hero = icon
    return w


class GetStarted(QDialog):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle("Get Started — AIO QC App")
        self.setWindowIcon(newIcon("get-started"))
        self.setModal(False)
        self.resize(820, 560)
        self.i = 0
        self._anim = self._dots_anim = None
        self.stack = QStackedWidget()
        self.pages = [_page_widget(p, self._show_me) for p in PAGES]
        for w in self.pages:
            self.stack.addWidget(w)
        self.trans = _Transition(self)
        self.dots = Dots(len(PAGES))
        self.dots.clicked.connect(self.go)
        self.back = newButton("← Back", slot=lambda: self.go(self.i - 1))
        self.next = newButton("Next →", slot=self._next)
        self.next.setDefault(True)
        self.open_btn = newButton("Open a dataset…", icon="open", slot=self._open)
        self.guide_btn = newButton("User Guide", icon="info", slot=self._guide)
        self.again = QCheckBox("Show at startup")
        self.again.setChecked(app.settings.value("get_started/at_start", False, type=bool))
        self.again.toggled.connect(lambda on: app.settings.setValue("get_started/at_start", on))
        foot = QHBoxLayout()
        foot.setContentsMargins(24, 6, 24, 16)
        foot.addWidget(self.again)
        foot.addStretch(1)
        foot.addWidget(self.dots)
        foot.addStretch(1)
        foot.addWidget(self.guide_btn)
        foot.addWidget(self.open_btn)
        foot.addWidget(self.back)
        foot.addWidget(self.next)
        band = QWidget()
        band.setFixedHeight(6)
        apply_style(band, "background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 @accent, stop:1 @info);")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(band)
        lay.addWidget(self.stack, 1)
        lay.addLayout(foot)
        for k, fn in (("Right", self._next), ("Left", lambda: self.go(self.i - 1)), ("PgDown", self._next),
                      ("PgUp", lambda: self.go(self.i - 1)), ("Home", lambda: self.go(0))):
            QShortcut(QKeySequence(k), self, activated=fn)
        self._buttons()
        self.spot = None

    # ------------------------------------------------------------------ navigation
    def _buttons(self):
        last = self.i == len(PAGES) - 1
        self.back.setEnabled(self.i > 0)
        self.next.setText("Finish" if last else ("Take the tour →" if self.i == 0 else "Next →"))
        self.open_btn.setVisible(self.i in (0, 1, len(PAGES) - 1))
        self.guide_btn.setVisible(last)

    def _next(self):
        if self.i >= len(PAGES) - 1:
            self.close()
        else:
            self.go(self.i + 1)

    def go(self, j: int):
        j = max(0, min(len(PAGES) - 1, j))
        if j == self.i:
            return
        old_i, self.i = self.i, j
        self._buttons()
        self._animate_dots(old_i, j)
        if not anim.ENABLED or not self.isVisible():
            self.stack.setCurrentIndex(j)
            return
        if self._anim is not None:
            self._anim.stop()
        self.trans.old = self.stack.grab()
        self.stack.setCurrentIndex(j)
        page = self.pages[j]
        self.trans.new = self.stack.grab()
        self.trans.dir = 1 if j > old_i else -1
        self.trans.setGeometry(self.stack.geometry())
        self.trans.t = 0.0
        self.trans.show()
        self.trans.raise_()
        self.stack.setVisible(True)
        page.setVisible(False)                               # only the painted transition is seen meanwhile
        hero, hero_pos = page.hero, page.hero.pos()

        def step(t):
            self.trans.t = t
            self.trans.update()

        def done():
            self.trans.hide()
            self.trans.old = self.trans.new = None
            page.setVisible(True)
            self._rise(hero, hero_pos)
        self._anim = FrameAnimator(self, 300, step, done)
        self._anim.start()

    def _rise(self, w, pos):
        """The hero icon settles in with a small lift."""
        start = QRect(pos + QPoint(0, 10), w.size())
        anim.slide(w, start, QRect(pos, w.size()), 260)

    def _animate_dots(self, a, b):
        if not anim.ENABLED:
            self.dots.pos_f = float(b)
            self.dots.update()
            return
        if self._dots_anim is not None:
            self._dots_anim.stop()
        start = self.dots.pos_f

        def step(t):
            self.dots.pos_f = start + (b - start) * t
            self.dots.update()
        self._dots_anim = FrameAnimator(self, 280, step)
        self._dots_anim.start()

    # ------------------------------------------------------------------ actions
    def _open(self):
        self.close()
        self.app.a.open_dir.trigger()

    def _guide(self):
        self.close()
        self.app.show_guide()

    def _show_me(self, items):
        targets = []
        for name, caption in items:
            w = target_widget(self.app, name)
            if w is not None and w.isVisible():
                targets.append((w, caption))
        if not targets:
            self.app.statusBar().showMessage("That part of the window is hidden right now (View ▸ Panels).", 5000)
            return
        self.hide()
        self.spot = Spotlight(self.app, targets, on_done=self._back_from_spot)

    def _back_from_spot(self):
        self.spot = None
        self.show()
        self.raise_()
        self.activateWindow()
        anim.window_fade_in(self)

    def showEvent(self, e):
        super().showEvent(e)
        self.dots.pos_f = float(self.i)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not self.open_btn.hasFocus() \
                and not self.guide_btn.hasFocus() and not self.back.hasFocus():
            self._next()
            return
        super().keyPressEvent(e)


def target_widget(app, name):
    return {"ribbon": getattr(app, "tabs", None), "files": getattr(app, "file_list_dock", None),
            "canvas": getattr(app, "canvas", None), "annotations": getattr(app, "annotations_dock", None),
            "toolbar": getattr(app, "toolbar", None), "history": getattr(app, "history_dock", None),
            "status": app.statusBar()}.get(name)


class Spotlight(QWidget):
    """Dims the window except a rounded hole around each target in turn, with a caption. The hole glides from one
    target to the next; click / any key = next, after the last one (or Esc) back to the tour."""

    def __init__(self, app, targets, on_done=None):
        super().__init__(app)
        self.app, self.targets, self.on_done = app, targets, on_done
        self.k = 0
        self.a = 0.0
        self.hole = self._rect(0)
        self._anim = None
        self.setGeometry(app.rect())
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.show()
        self.raise_()
        self.setFocus()
        self._fade(1.0)

    def _rect(self, k) -> QRect:
        w = self.targets[k][0]
        tl = w.mapTo(self.app, QPoint(0, 0))
        return QRect(tl, w.size()).adjusted(-6, -6, 6, 6)

    def _fade(self, target, done=None):
        start = self.a

        def step(t):
            self.a = start + (target - start) * t
            self.update()
        if not anim.ENABLED:
            step(1.0)
            if done:
                done()
            return
        FrameAnimator(self, 240, step, done).start()

    def advance(self):
        if self.k + 1 >= len(self.targets):
            self.finish()
            return
        a, self.k = self.hole, self.k + 1
        b = self._rect(self.k)
        if not anim.ENABLED:
            self.hole = b
            self.update()
            return

        def step(t):
            self.hole = lerp_rect(a, b, t)
            self.update()
        if self._anim is not None:
            self._anim.stop()
        self._anim = FrameAnimator(self, 380, step)
        self._anim.start()

    def finish(self):
        def done():
            self.hide()
            self.deleteLater()
            if self.on_done:
                self.on_done()
        self._fade(0.0, done)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRect(QRectF(self.rect()))
        hole = QPainterPath()
        hole.addRoundedRect(QRectF(self.hole), 10, 10)
        c = QColor(0, 0, 0)
        c.setAlphaF(0.55 * self.a)
        p.fillPath(path.subtracted(hole), c)
        ring = QColor(hexc("accent"))
        ring.setAlphaF(self.a)
        p.setPen(QPen(ring, 3))
        p.drawRoundedRect(QRectF(self.hole), 10, 10)
        # caption bubble below (or above) the hole
        text = self.targets[self.k][1] + (f"   ({self.k + 1}/{len(self.targets)} · click to continue)"
                                          if len(self.targets) > 1 else "   (click to return)")
        f = QFont(self.font())
        f.setPointSizeF(f.pointSizeF() * 1.1)
        p.setFont(f)
        fm = p.fontMetrics()
        w = min(self.width() - 40, fm.horizontalAdvance(text) + 28)
        h = fm.height() * (2 if fm.horizontalAdvance(text) + 28 > w else 1) + 18
        x = max(20, min(self.width() - w - 20, self.hole.center().x() - w // 2))
        y = self.hole.bottom() + 14 if self.hole.bottom() + 14 + h < self.height() - 10 else self.hole.top() - h - 14
        y = max(10, min(self.height() - h - 10, y))
        box = QRectF(x, y, w, h)
        bg = QColor(hexc("surface"))
        bg.setAlphaF(0.97 * self.a)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(box, 10, 10)
        fg = QColor(hexc("text"))
        fg.setAlphaF(self.a)
        p.setPen(fg)
        p.drawText(box.adjusted(14, 8, -14, -8), int(Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap), text)

    def mousePressEvent(self, _e):
        self.advance()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.finish()
        else:
            self.advance()
