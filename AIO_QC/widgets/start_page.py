# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The page a mode shows while nothing is open: what this mode is for, the one obvious first step, a sample to learn
with, and what was open recently (in any mode).

    page = StartPage(title, subtitle, actions=[Action("Open dataset…", fn, primary=True, hint="Ctrl+O")],
                     links=[Action("Try a sample dataset", fn)], cards=[ModeCard…], recents=fn -> [Recent],
                     drop_hint="…or drop a dataset folder here")

Image QC's page is also the app's welcome: three cards for the three jobs (check annotations · frames from videos ·
clean a table) and the recent items of all three modes, each opened in its own mode. The page scrolls on a small
window; the cards go under each other on a narrow one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
                             QToolButton, QVBoxLayout, QWidget)

from AIO_QC import utils
from AIO_QC.core.diag import ignored as _ignored

KIND_ICON = {"image": "mode-image", "video": "mode-video", "data": "mode-data"}
KIND_NAME = {"image": "Dataset", "video": "Video", "data": "Table"}


@dataclass
class Action:
    text: str
    fn: Callable
    primary: bool = False
    icon: str | None = None
    hint: str = ""                   # a key shown next to the text ("Ctrl+O")
    tip: str = ""


@dataclass
class Recent:
    kind: str                        # image · video · data
    path: str
    open: Callable

    @property
    def exists(self) -> bool:
        return os.path.exists(self.path)


@dataclass
class Card:
    kind: str
    title: str
    text: str
    actions: list
    current: bool = False


def _font(px: int, weight=QFont.Weight.Normal) -> QFont:
    f = QFont()
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


def _button(a: Action) -> QPushButton:
    b = QPushButton(a.text + (f"   {a.hint}" if a.hint else ""))
    if a.icon:
        b.setIcon(utils.newIcon(a.icon))
    if a.primary:
        b.setProperty("variant", "primary")
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setMinimumHeight(34)
    if a.tip:
        b.setToolTip(a.tip)
    b.clicked.connect(lambda _c=False, f=a.fn: f())
    return b


def _link(a: Action) -> QPushButton:
    b = QPushButton(a.text)
    b.setFlat(True)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    utils.style(b, "QPushButton{color:@accent; border:none; background:transparent; padding:4px 2px; "
                   "font-weight:600;} QPushButton:hover{text-decoration:underline;}")
    if a.tip:
        b.setToolTip(a.tip)
    b.clicked.connect(lambda _c=False, f=a.fn: f())
    return b


class _IconTile(QLabel):
    def __init__(self, icon: str, size=40):
        super().__init__()
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setPixmap(utils.newIcon(icon).pixmap(QSize(size // 2 + 2, size // 2 + 2)))
        utils.style(self, f"QLabel{{background:@accent_soft; border-radius:{size // 4}px;}}")


class _CardWidget(QFrame):
    def __init__(self, c: Card):
        super().__init__()
        self.setObjectName("startCard")
        utils.style(self, "#startCard{background:@base; border:1px solid @line; border-radius:12px;}"
                    + ("#startCard{border:1.5px solid @accent;}" if c.current else ""))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 16)
        lay.setSpacing(8)
        lay.addWidget(_IconTile(KIND_ICON.get(c.kind, "app")))
        lay.addSpacing(4)
        t = QLabel(c.title)
        t.setFont(_font(16, QFont.Weight.DemiBold))
        lay.addWidget(t)
        d = QLabel(c.text)
        d.setWordWrap(True)
        utils.style(d, "QLabel{color:@muted;}")
        lay.addWidget(d, 1)
        lay.addSpacing(6)
        for a in c.actions:
            lay.addWidget(_button(a) if a.primary or not c.current else _link(a))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)


class _RecentRow(QToolButton):
    """One recent item: its mode's icon, the name, where it is, and its kind."""

    def __init__(self, r: Recent):
        super().__init__()
        self.r = r
        self.setObjectName("recentRow")
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor if r.exists else Qt.CursorShape.ArrowCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(48)
        utils.style(self, "#recentRow{border:none; border-radius:8px; background:transparent; text-align:left;}"
                          "#recentRow:hover{background:@pill;}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 12, 4)
        lay.setSpacing(12)
        ic = QLabel()
        ic.setPixmap(utils.newIcon(KIND_ICON.get(r.kind, "open")).pixmap(QSize(20, 20)))
        ic.setFixedWidth(22)
        col = QVBoxLayout()
        col.setSpacing(0)
        name = QLabel(os.path.basename(r.path.rstrip("/\\")) or r.path)
        name.setFont(_font(13, QFont.Weight.DemiBold))
        where = QLabel(os.path.dirname(r.path.rstrip("/\\")) if r.exists else "Not found — moved or deleted")
        utils.style(where, "QLabel{color:@faint;}" if r.exists else "QLabel{color:@warn;}")
        where.setMinimumWidth(40)
        where.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._where, self._where_text = where, where.text()
        col.addWidget(name)
        col.addWidget(where)
        kind = QLabel(KIND_NAME.get(r.kind, ""))
        utils.style(kind, "QLabel{color:@muted; background:@pill; border-radius:9px; padding:1px 8px;}")
        for w in (ic, name, where, kind):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lay.addWidget(ic)
        lay.addLayout(col, 1)
        lay.addWidget(kind)
        self.setToolTip(r.path)
        self.setEnabled(r.exists)
        self.clicked.connect(lambda _c=False: r.open())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        fm = self._where.fontMetrics()                       # long folders: the middle gives way
        self._where.setText(fm.elidedText(self._where_text, Qt.TextElideMode.ElideMiddle,
                                          max(40, self._where.width())))


class StartPage(QWidget):
    """The empty state of a mode (see the module doc)."""
    changed = pyqtSignal()

    def __init__(self, title: str, subtitle: str, actions=(), links=(), cards=(), recents=None,
                 drop_hint: str = "", footer=(), parent=None):
        super().__init__(parent)
        self.setObjectName("startPage")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        utils.style(self, "#startPage{background:@surface;}")
        self._recents_fn = recents
        self._cards = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        utils.style(scroll, "QScrollArea{background:transparent;} QScrollArea > QWidget > QWidget{background:transparent;}")
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        row = QHBoxLayout(body)
        row.setContentsMargins(24, 0, 24, 0)
        row.addStretch(1)
        col_w = QWidget()
        col_w.setMaximumWidth(900)
        col_w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        row.addWidget(col_w, 100)
        row.addStretch(1)
        col = QVBoxLayout(col_w)
        col.setContentsMargins(0, 36, 0, 32)
        col.setSpacing(0)
        col.addStretch(1)
        self.title = QLabel(title)
        self.title.setFont(_font(26, QFont.Weight.DemiBold))
        self.title.setWordWrap(True)
        col.addWidget(self.title)
        col.addSpacing(6)
        self.subtitle = QLabel(subtitle)
        self.subtitle.setWordWrap(True)
        self.subtitle.setFont(_font(15))
        utils.style(self.subtitle, "QLabel{color:@muted;}")
        col.addWidget(self.subtitle)
        col.addSpacing(22)
        if cards:
            self.grid = QGridLayout()
            self.grid.setHorizontalSpacing(14)
            self.grid.setVerticalSpacing(14)
            self._cards = [_CardWidget(c) for c in cards]
            self._cols = 0
            col.addLayout(self.grid)
            self._place_cards(3)
        if actions or links:
            bar = QHBoxLayout()
            bar.setSpacing(10)
            for a in actions:
                bar.addWidget(_button(a))
            for a in links:
                bar.addWidget(_link(a))
            bar.addStretch(1)
            col.addLayout(bar)
        if drop_hint:
            col.addSpacing(10)
            hint = QLabel(drop_hint)
            utils.style(hint, "QLabel{color:@faint;}")
            col.addWidget(hint)
        self.recent_title = QLabel("Recent")
        self.recent_title.setFont(_font(13, QFont.Weight.DemiBold))
        utils.style(self.recent_title, "QLabel{color:@muted;}")
        self.recent_box = QVBoxLayout()
        self.recent_box.setSpacing(2)
        col.addSpacing(28)
        col.addWidget(self.recent_title)
        col.addSpacing(6)
        col.addLayout(self.recent_box)
        if footer:
            col.addSpacing(20)
            fb = QHBoxLayout()
            fb.setSpacing(18)
            for a in footer:
                fb.addWidget(a if isinstance(a, QWidget) else _link(a))
            fb.addStretch(1)
            col.addLayout(fb)
        col.addStretch(2)
        self.refresh()

    # ------------------------------------------------------------------
    def _place_cards(self, cols: int):
        if cols == self._cols:
            return
        self._cols = cols
        for c in self._cards:
            self.grid.removeWidget(c)
        for i, c in enumerate(self._cards):
            self.grid.addWidget(c, i // cols, i % cols)
        for k in range(3):
            self.grid.setColumnStretch(k, 1 if k < cols else 0)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._cards:
            self._place_cards(3 if self.width() >= 820 else 1)

    def refresh(self):
        while self.recent_box.count():
            it = self.recent_box.takeAt(0)
            row = it.widget()
            if row is not None:
                row.setParent(None)                          # gone now (not when the event loop gets to it)
                row.deleteLater()
        items = []
        if self._recents_fn is not None:
            try:
                items = list(self._recents_fn())[:6]
            except Exception as _exc:
                _ignored(_exc)
                items = []
        for r in items:
            self.recent_box.addWidget(_RecentRow(r))
        self.recent_title.setVisible(bool(items))

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh()


# ---------------------------------------------------------------------------------------------- recent items
def all_recents(shell=None, first: str | None = None) -> list[Recent]:
    """The recent items of the three modes, newest of each first, interleaved (the asking mode's first)."""
    from AIO_QC.utils import app_settings
    from AIO_QC.utils.fx_settings import fx_settings
    st, fx = app_settings(), fx_settings()

    def as_list(v):
        return [v] if isinstance(v, str) else list(v or [])

    def open_in(kind, path):
        sh = shell or getattr(__import__("PyQt6.QtWidgets", fromlist=["QApplication"]).QApplication.instance(),
                              "shell", None)
        return lambda: open_recent(sh, kind, path)

    image = [str(x).partition("|")[0] for x in as_list(st.value("recent_datasets", []))]
    video = [str(x) for x in as_list(fx.value("recent/files", []))]
    data = [str(x) for x in as_list(st.value("studio/recent", []))]
    lists = {"image": image, "video": video, "data": data}
    order = [first] + [k for k in ("image", "video", "data") if k != first] if first else ["image", "video", "data"]
    out, i = [], 0
    while any(i < len(lists[k]) for k in order):
        for k in order:
            if i < len(lists[k]) and lists[k][i]:
                out.append(Recent(k, lists[k][i], open_in(k, lists[k][i])))
        i += 1
    return out


def open_recent(shell, kind: str, path: str):
    from pathlib import Path
    if shell is None:
        return
    if kind == "image":
        w = shell.window("image")
        if shell.mode != "image":
            shell.switch("image")
        w.open_dataset(Path(path))
    elif kind == "video":
        shell.switch("video")
        shell.window("video").load_video(path)
    else:
        shell.switch("data")
        shell.window("data")._open_path(path)
