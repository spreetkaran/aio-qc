# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Labels dock list: rows are checkboxes (label visibility), so a row highlight never lingers.

The highlight clears after a checkbox is toggled, on a click in the empty part of the list, and on Esc. The row
under the mouse still becomes the current row while it is clicked (keyboard navigation, Space to toggle)."""
from __future__ import annotations

from PyQt6.QtCore import QModelIndex, Qt
from PyQt6.QtWidgets import QAbstractItemView, QListWidget


class LabelsList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.itemChanged.connect(lambda _it: self.deselect())

    def deselect(self):
        """Drop the row highlight and the current row (deferred: the click that toggled a box selects afterwards)."""
        from AIO_QC.utils import later
        later(0, self, self._deselect_now)

    def _deselect_now(self):
        self.clearSelection()
        self.selectionModel().setCurrentIndex(QModelIndex(), self.selectionModel().SelectionFlag.NoUpdate) \
            if self.selectionModel() else None

    def mousePressEvent(self, e):
        if self.itemAt(e.position().toPoint()) is None:      # empty space: nothing stays selected
            self._deselect_now()
            return
        super().mousePressEvent(e)

    def event(self, e):
        # Esc is also a window shortcut (deselect the shape): claim it while a row is highlighted here
        if e.type() == e.Type.ShortcutOverride and e.key() == Qt.Key.Key_Escape and \
                (self.selectedItems() or self.currentItem() is not None):
            e.accept()
            return True
        return super().event(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape and (self.selectedItems() or self.currentItem() is not None):
            self._deselect_now()
            e.accept()
            return
        super().keyPressEvent(e)
