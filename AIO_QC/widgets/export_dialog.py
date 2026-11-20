# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The split preview's pictures (Convert & Split ▸ Split): SplitBar — the whole dataset as one stacked bar; ClassBars —
one bar per class (the share of its images in train / val / test, the target dashed).
"""
from __future__ import annotations

from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from AIO_QC.utils import theme_color

SPLIT_COLORS = {"train": QColor("#1E88E5"), "val": QColor("#FB8C00"), "test": QColor("#8E24AA")}
ORDER = ("train", "val", "test")


class SplitBar(QWidget):
    """The whole dataset: one stacked bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.counts: dict = {}
        self.setMinimumHeight(64)

    def set_counts(self, counts):
        self.counts = dict(counts)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        total = sum(self.counts.values())
        r = QRectF(self.rect()).adjusted(2, 4, -2, -26)
        if not total:
            p.setPen(theme_color("muted"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Press Preview to see the split")
            return
        x = r.left()
        f = p.font()
        f.setBold(True)
        for sp in ORDER:
            n = self.counts.get(sp, 0)
            if not n:
                continue
            w = r.width() * n / total
            seg = QRectF(x, r.top(), w, r.height())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(SPLIT_COLORS[sp])
            p.drawRoundedRect(seg, 6, 6)
            p.setPen(QColor("white"))
            p.setFont(f)
            txt = f"{sp.capitalize()}  {n:,}  ({100 * n / total:.1f} %)"
            if p.fontMetrics().horizontalAdvance(txt) > w - 8:
                txt = f"{100 * n / total:.0f} %"
            p.drawText(seg, Qt.AlignmentFlag.AlignCenter, txt)
            x += w
        p.setPen(theme_color("muted"))
        p.setFont(self.font())
        p.drawText(QRectF(r.left(), r.bottom() + 4, r.width(), 20), Qt.AlignmentFlag.AlignLeft,
                   f"{total:,} images")


class ClassBars(QWidget):
    """One row per class: the share of its images in train / val / test; the target split marked with a dashed line."""
    ROW = 24

    def __init__(self, parent=None):
        super().__init__(parent)
        self.classes: dict = {}
        self.target: dict = {}

    def set_data(self, classes: dict, target: dict):
        self.classes, self.target = classes, target
        self.setMinimumHeight(self.ROW * max(1, len(classes)) + 8)
        self.update()

    def sizeHint(self):
        return QSize(600, self.ROW * max(1, len(self.classes)) + 8)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fm = p.fontMetrics()
        names = list(self.classes)
        name_w = min(220, max((fm.horizontalAdvance(n) for n in names), default=60) + 12)
        count_w = 170
        bar_l, bar_r = name_w, self.width() - count_w - 8
        for i, (name, v) in enumerate(self.classes.items()):
            y = 4 + i * self.ROW
            tot = sum(v.values()) or 1
            p.setPen(self.palette().text().color())
            p.drawText(QRectF(0, y, name_w - 8, self.ROW - 4), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       fm.elidedText(name, Qt.TextElideMode.ElideMiddle, name_w - 8))
            x = bar_l
            for sp in ORDER:
                n = v.get(sp, 0)
                if not n:
                    continue
                w = (bar_r - bar_l) * n / tot
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(SPLIT_COLORS[sp])
                p.drawRect(QRectF(x, y + 3, w, self.ROW - 10))
                x += w
            # target boundaries (dashed)
            p.setPen(QPen(self.palette().text().color(), 1.2, Qt.PenStyle.DashLine))
            acc = 0.0
            for sp in ORDER[:-1]:
                acc += self.target.get(sp, 0.0)
                if 0 < acc < 1:
                    xx = bar_l + (bar_r - bar_l) * acc
                    p.drawLine(int(xx), y, int(xx), y + self.ROW - 4)
            p.setPen(theme_color("muted"))
            txt = " / ".join(f"{v.get(sp, 0):,}" for sp in ORDER if sp in self.target or v.get(sp))
            p.drawText(QRectF(bar_r + 8, y, count_w, self.ROW - 4), Qt.AlignmentFlag.AlignVCenter, txt)
