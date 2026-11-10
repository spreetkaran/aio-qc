# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Profile tab: the table's health at a glance and one card per column — with one-click fixes."""
from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from AIO_QC.core.tabular.profile import describe_value
from AIO_QC.utils import style as apply_style, theme_color

from .chart_view import SERIES

KIND_SYM = {"number": "#", "integer": "#", "text": "Aa", "boolean": "✓", "date": "◷", "category": "◆", "empty": "∅"}


class Spark(QWidget):
    """A tiny histogram or top-values bar list."""

    def __init__(self, col):
        super().__init__()
        self.col = col
        self.setFixedHeight(54)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, _):
        p = QPainter(self)
        c = self.col
        color = QColor(SERIES[0])
        if c.hist:
            counts = c.hist[0]
            m = max(counts) or 1
            bw = self.width() / max(1, len(counts))
            for i, n in enumerate(counts):
                h = n / m * (self.height() - 2)
                p.fillRect(QRectF(i * bw, self.height() - h, max(1.0, bw - 1), h), color)
        elif c.top:
            top = c.top[:3]
            m = max(n for _v, n in top) or 1
            rh = self.height() / 3
            p.setPen(theme_color("text"))
            for i, (v, n) in enumerate(top):
                w = n / m * self.width() * 0.55
                col = QColor(color)
                col.setAlpha(110)
                p.fillRect(QRectF(0, i * rh + 2, w, rh - 4), col)
                p.drawText(QRectF(4, i * rh, self.width() - 8, rh), Qt.AlignmentFlag.AlignVCenter,
                           f"{str(v)[:30]}  ·  {n:,}")


class ColumnCard(QFrame):
    fix = pyqtSignal(dict)
    show = pyqtSignal(str)

    def __init__(self, c, n_rows):
        super().__init__()
        self.col = c
        self.setObjectName("colCard")
        apply_style(self, "QFrame#colCard{border:1px solid @line; border-radius:8px; background:@surface;}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(3)
        head = QHBoxLayout()
        name = QLabel(f"<b>{KIND_SYM.get(c.kind, '')}&nbsp; {_e(c.name)}</b>")
        name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        head.addWidget(name, 1)
        t = QLabel(c.dtype)
        apply_style(t, "color:@muted; font-size:9pt;")
        head.addWidget(t)
        lay.addLayout(head)
        lay.addWidget(Spark(c))
        facts = [f"missing {c.missing:,} ({c.missing_pct:.1%})", f"distinct {c.distinct:,}"]
        st = c.stats
        if c.kind in ("number", "integer") and st:
            facts += [f"min {describe_value(st['min'])} · median {describe_value(st['median'])} · "
                      f"max {describe_value(st['max'])}",
                      f"mean {describe_value(st['mean'])} ± {describe_value(st['std'])}"]
        elif c.kind == "date" and st:
            facts += [f"{st['min']} → {st['max']}"]
        elif st and "min_len" in st:
            facts += [f"length {st['min_len']}–{st['max_len']} (avg {st['mean_len']:.1f})"]
        f = QLabel(" · ".join(facts[:2]) + ("<br>" + "<br>".join(facts[2:]) if facts[2:] else ""))
        f.setWordWrap(True)
        apply_style(f, "color:@muted;")
        lay.addWidget(f)
        for _k, text in c.warnings:
            w = QLabel(f"⚠ {_e(text)}")
            w.setWordWrap(True)
            apply_style(w, "color:@warn;")
            lay.addWidget(w)
        if c.fixes:
            row = QHBoxLayout()
            for label, step in c.fixes[:3]:
                b = QPushButton(label)
                b.setToolTip("Adds this step to the recipe (undo with Ctrl+Z)")
                b.clicked.connect(lambda _c=False, s=step: self.fix.emit(s))
                row.addWidget(b)
            row.addStretch(1)
            lay.addLayout(row)
        lay.addStretch(1)


class ProfileView(QWidget):
    fix = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setWordWrap(True)
        top.addWidget(self.summary, 1)
        self.find = QLineEdit()
        self.find.setPlaceholderText("Find a column…")
        self.find.setMaximumWidth(220)
        self.find.textChanged.connect(self._filter)
        self.only = QPushButton("Only columns with warnings")
        self.only.setCheckable(True)
        self.only.toggled.connect(lambda _on: self._filter(self.find.text()))
        top.addWidget(self.only)
        top.addWidget(self.find)
        lay.addLayout(top)
        self.corr = QLabel("")
        self.corr.setWordWrap(True)
        apply_style(self.corr, "color:@muted;")
        lay.addWidget(self.corr)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        lay.addWidget(self.scroll, 1)
        self.cards = []

    def set_profile(self, prof, note=""):
        host = QWidget()
        grid = QGridLayout(host)
        grid.setSpacing(10)
        cols = max(1, self.width() // 340)
        self.cards = []
        for i, c in enumerate(prof["columns"]):
            card = ColumnCard(c, prof["rows"])
            card.fix.connect(self.fix.emit)
            grid.addWidget(card, i // cols, i % cols)
            self.cards.append(card)
        grid.setRowStretch(len(prof["columns"]) // cols + 1, 1)
        self.scroll.setWidget(host)
        score = prof["score"]
        tone = "@ok" if score >= 85 else ("@warn" if score >= 60 else "@error")
        warn = sum(1 for c in prof["columns"] if c.warnings)
        self.summary.setText(
            f"<span style='font-size:15pt; font-weight:600'>{prof['rows']:,} rows × {prof['cols']} columns</span>"
            f" &nbsp; quality <b>{score}/100</b> · {prof['missing_cells']:,} missing cells · "
            f"{prof['duplicate_rows']:,} duplicate rows · {warn} column(s) with warnings"
            + (" · <i>statistics from a 200,000-row sample</i>" if prof.get("sampled") else "") + (note or ""))
        apply_style(self.summary, f"QLabel{{border-left:4px solid {tone}; padding-left:8px;}}")
        self.corr.setText(("Strong correlations: " + " · ".join(f"{a} ↔ {b} ({r:+.2f})" for a, b, r in
                                                                prof["correlations"][:8]))
                          if prof["correlations"] else "")
        self._filter(self.find.text())

    def _filter(self, t):
        t = (t or "").lower()
        only = self.only.isChecked()
        for c in self.cards:
            c.setVisible((not t or t in c.col.name.lower()) and (not only or bool(c.col.warnings)))


def _e(s):
    import html
    return html.escape(str(s))
