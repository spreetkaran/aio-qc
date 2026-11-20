# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Labels dock list: rows are checkboxes (label visibility), so a row highlight never lingers.

The highlight clears after a checkbox is toggled, on a click in the empty part of the list, and on Esc. The row
under the mouse still becomes the current row while it is clicked (keyboard navigation, Space to toggle)."""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QModelIndex, QObject, Qt
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QAbstractItemView, QListWidget


class PinnedRowList(QListWidget):
    """A list whose rows keep something pinned to the right edge (counts, #n) while long text scrolls sideways."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

    def visualRect(self, index):
        """A row is the panel's full width: hover / selection / data changes repaint the whole row, never just the
        part where the text is (that left stale or cut pixels)."""
        r = super().visualRect(index)
        if r.isValid():
            vp = self.viewport().rect()
            r.setLeft(min(r.left(), vp.left()))
            r.setRight(max(r.right(), vp.right()))
        return r

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        if dx:
            self.viewport().update()          # the pinned parts and the fades follow the view, not the content


class LabelsList(PinnedRowList):
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


class GuardSelectAll(QObject):
    """Ctrl+A in a list only while `allowed()` (the Annotations panel: once a shape is selected, as in LabelMe) —
    otherwise the key is swallowed instead of selecting every row."""

    def __init__(self, view, allowed):
        super().__init__(view)
        self.allowed = allowed
        view.installEventFilter(self)

    def eventFilter(self, obj, e):
        if e.type() == QEvent.Type.KeyPress and e.matches(QKeySequence.StandardKey.SelectAll) and not self.allowed():
            return True
        return False
