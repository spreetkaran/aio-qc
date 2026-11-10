# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""A virtual table over a pandas DataFrame: only the visible cells are formatted, so millions of rows scroll fine."""
from __future__ import annotations

import math

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QFont

from AIO_QC.utils import theme_color


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if v.is_integer() and abs(v) < 1e15:
            return f"{int(v)}"
        return f"{v:.6g}"
    s = str(v)
    if s in ("<NA>", "NaT", "nan"):
        return ""
    return s if len(s) <= 300 else s[:300] + "…"


class FrameModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = None
        self._kinds = []
        self._italic = QFont()
        self._italic.setItalic(True)

    def set_frame(self, df, kinds=None):
        self.beginResetModel()
        self.df = df
        self._kinds = kinds or []
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if self.df is None or parent.isValid() else len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if self.df is None or parent.isValid() else self.df.shape[1]

    def _raw(self, r, c):
        try:
            return self.df.iat[r, c]
        except Exception:
            return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if self.df is None or not index.isValid():
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            v = self._raw(index.row(), index.column())
            t = fmt(v)
            if role == Qt.ItemDataRole.ToolTipRole:
                return str(v)[:2000] if len(str(v)) > 40 else None
            return t if t != "" else "·"
        if role == Qt.ItemDataRole.ForegroundRole:
            if fmt(self._raw(index.row(), index.column())) == "":
                return theme_color("faint")                       # missing values: a faint dot
        if role == Qt.ItemDataRole.TextAlignmentRole:
            k = self._kinds[index.column()] if index.column() < len(self._kinds) else ""
            if k in ("number", "integer"):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if self.df is None:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                k = self._kinds[section] if section < len(self._kinds) else ""
                sym = {"number": "#", "integer": "#", "text": "Aa", "boolean": "✓", "date": "◷", "category": "◆",
                       "empty": "∅"}.get(k, "")
                return f"{sym}  {self.df.columns[section]}" if sym else str(self.df.columns[section])
            if role == Qt.ItemDataRole.ToolTipRole:
                return f"{self.df.columns[section]} · {self.df.dtypes.iloc[section]}"
        elif role == Qt.ItemDataRole.DisplayRole:
            return f"{section + 1:,}"
        return None
