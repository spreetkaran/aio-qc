# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Toolbar breadcrumb: dataset › split › file, then a pill with the dataset type and size ("LabelMe · 30 files").
Clicking the dataset name opens its folder.

3.0.1: the path has a fixed area (it changes only with the window width), long names are cut in the middle with "…",
so the pill never moves while you go through the images.
"""
from __future__ import annotations

from html import escape

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from AIO_QC import utils


class Breadcrumb(QWidget):
    rootClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 6, 0)
        lay.setSpacing(10)
        self.icon = utils.IconLabel("open", 16)
        self.icon.setVisible(False)                       # 3.0 mock-up: the path alone
        self.path = QLabel("No dataset")
        self.path.setTextFormat(Qt.TextFormat.RichText)
        self.path.linkActivated.connect(lambda _l: self.rootClicked.emit())
        self.path.setFixedWidth(320)
        self.pill = QLabel("")
        self.pill.setObjectName("crumbPill")
        self.pill.setVisible(False)
        utils.style(self.pill, "QLabel{color:@muted; background:@pill; border:0; border-radius:9px;"
                               " padding:2px 8px;}")
        lay.addWidget(self.icon)
        lay.addWidget(self.path)
        lay.addWidget(self.pill)
        self._root = self._split = self._file = ""
        self._render()

    # label-like API kept for older callers ("ds · LabelMe · 30 files")
    def setText(self, text: str):
        parts = [p.strip() for p in str(text).split(" · ")]
        if len(parts) >= 3:
            self.set_dataset(parts[0], " · ".join(parts[1:]))
        else:
            self._root, self._split, self._file = str(text), "", ""
            self.pill.setVisible(False)
            self._render()

    def text(self) -> str:
        return " · ".join(x for x in (self._root, self.pill.text() if self.pill.isVisibleTo(self) else "") if x)

    def set_dataset(self, root: str, info: str):
        if root == self._root:                    # same dataset, new size (files added / removed outside AIO QC):
            self.set_info(info)                   # keep the split and file shown and the sized area
            return
        self._root, self._split, self._file = root, "", ""
        self._typical = None
        self.pill.setText(info)
        self.pill.setVisible(bool(info))
        self._render()

    def set_info(self, info: str):
        """Only the pill ("LabelMe · 31 files"); the path stays as it is."""
        if info != self.pill.text():
            self.pill.setText(info)
        self.pill.setVisible(bool(info))

    def reset(self):
        """Forget the dataset (the next set_dataset starts fresh even for the same folder name)."""
        self._root, self._split, self._file = "", "", ""
        self._typical = None

    def set_location(self, split: str = "", file: str = ""):
        if (split, file) != (self._split, self._file):
            self._split, self._file = split or "", file or ""
            self._render()

    _max_w = 420
    _typical = None              # (median file-name length, median split length) of the open dataset

    def set_area_width(self, w: int):
        """The most the path may take (follows the window width)."""
        self._max_w = int(max(140, w))
        self._apply_width()

    def set_typical(self, file_chars: int, split_chars: int = 0):
        """Size the area for the dataset's MEDIAN file name (a few long names are cut in the middle instead of
        pushing the pill far to the right for everyone else)."""
        self._typical = (max(4, int(file_chars)), max(0, int(split_chars)))
        self._apply_width()

    def _apply_width(self):
        from PyQt6.QtGui import QFont, QFontMetrics
        w = self._max_w
        if self._typical is not None and self._root:
            f = QFont(self.path.font())
            bold, med = QFont(f), QFont(f)
            bold.setWeight(QFont.Weight.DemiBold)
            med.setWeight(QFont.Weight.Medium)
            fm, bfm, mfm = QFontMetrics(f), QFontMetrics(bold), QFontMetrics(med)
            file_chars, split_chars = self._typical
            sep = fm.horizontalAdvance(" › ") + 2
            need = bfm.horizontalAdvance(self._root) + sep + mfm.horizontalAdvance("n" * file_chars) + 14
            if split_chars:
                need += fm.horizontalAdvance("n" * split_chars) + sep
            w = min(w, max(160, need))
        if w != self.path.width():
            self.path.setFixedWidth(w)
        self._render()

    def _render(self):
        from PyQt6.QtGui import QFont, QFontMetrics
        from AIO_QC.utils import hexc
        muted, text = hexc("muted"), hexc("text")
        if not self._root:
            self.path.setText(f"<span style='color:{muted}'>No dataset</span>")
            self.path.setToolTip("")
            return
        f = QFont(self.path.font())
        bold, med = QFont(f), QFont(f)
        bold.setWeight(QFont.Weight.DemiBold)
        med.setWeight(QFont.Weight.Medium)
        fm, bfm, mfm = QFontMetrics(f), QFontMetrics(bold), QFontMetrics(med)
        sep_w = fm.horizontalAdvance(" › ") + 2
        avail = self.path.width() - 6
        E = Qt.TextElideMode
        root, split, file = self._root, self._split, self._file
        # the file name gets what the dataset and split leave (they keep at most ~40 % / ~22 % of the area)
        root_w = min(bfm.horizontalAdvance(root) + 4, int(avail * (0.40 if file else 0.75)))
        root = bfm.elidedText(root, E.ElideMiddle, root_w)
        used = bfm.horizontalAdvance(root)
        if split:
            split = fm.elidedText(split, E.ElideMiddle, min(fm.horizontalAdvance(split) + 4, int(avail * 0.22)))
            used += sep_w + fm.horizontalAdvance(split)
        if file:
            file = mfm.elidedText(file, E.ElideMiddle, max(24, avail - used - sep_w))
        sep = f"<span style='color:{muted}'>&nbsp;›&nbsp;</span>"
        bits = [f"<a href='root' style='color:{text}; text-decoration:none; font-weight:600'>{escape(root)}</a>"]
        if split:
            bits.append(f"<span style='color:{muted}'>{escape(split)}</span>")
        if file:
            bits.append(f"<span style='color:{text}; font-weight:500'>{escape(file)}</span>")
        self.path.setText(sep.join(bits))
        self.path.setToolTip(" › ".join(x for x in (self._root, self._split, self._file) if x))

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() == e.Type.PaletteChange:
            self._render()
