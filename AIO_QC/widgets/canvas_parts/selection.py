# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Image QC canvas — selecting shapes (one, several, all).

A mixin of AIO_QC.widgets.canvas.Canvas: the methods use the window's own attributes, set up in its __init__.
"""
from __future__ import annotations



class SelectionMixin:

    # ------------------------------------------------------------- selection
    def select_shape(self, index: int, emit: bool = True, keep_multi: bool = False) -> bool:
        """Selecting is a Select-mode action: from Drag mode it switches to Select first (modeChanged tells the UI)."""
        if self.peek_index is not None and index != self.peek_index:
            self.peek_index = None                       # another shape: the peeked hidden one goes away again
            self.drawShapes()
        group = self.index_to_group.get(index)
        if group is None:
            group = self._peek(index)
            if group is None:
                return False
        if self.mode == self.DRAG:
            self.setMode(self.SELECT)
        if self.multi and not keep_multi:
            self._flush_nudge()
            self._set_multi([], tell=False)
        if self.selected_group is not group:
            self._flush_nudge()
            if getattr(self, "_active_for", None) != group.index:   # the same shape re-selected after its edit
                self._active_handle = None                          # was saved keeps its active handle
            self._active_for = group.index
            if self.selected_group is not None:
                self.selected_group.set_selected(False)
            self.selected_group = group
            group.set_selected(True, self.SELECTED_Z)
            self._emphasize_tag(group)
        self._update_handles()
        self.emit_selection()
        if emit:
            self.shapeSelected.emit(group.index)
        self.selectionChanged.emit(self.selected_indices())
        return True

    # ---- several shapes -----------------------------------------------------------------------------------------
    def selected_indices(self) -> list:
        """Every selected shape index, sorted (the primary one included)."""
        out = {g.index for g in self.multi}
        if self.selected_group is not None:
            out.add(self.selected_group.index)
        return sorted(out)

    def _set_multi(self, groups, tell: bool = True):
        groups = [g for g in dict.fromkeys(groups) if g is not self.selected_group and g in self.shape_groups]
        for g in self.multi:
            if g not in groups and g is not self.selected_group and g in self.shape_groups:
                g.set_selected(False)
        for g in groups:
            g.set_selected(True, self.SELECTED_Z - 1)
        self.multi = groups
        self._multi_work = {}
        self._multi_total = (0.0, 0.0)
        if tell:
            self.selectionChanged.emit(self.selected_indices())

    def toggle_shape(self, index: int) -> bool:
        """Ctrl+click: in / out of the selection. The shape clicked last is the primary one (handles, single keys)."""
        group = self.index_to_group.get(index)
        if group is None:
            return False
        self._flush_nudge()
        if group is self.selected_group:
            if not self.multi:
                self.clearSelection()
                return True
            rest = list(self.multi)
            nxt = rest.pop()                                  # the one added before it takes over
            self.select_shape(nxt.index, keep_multi=True)
            group.set_selected(False)
            self._set_multi(rest)
            return True
        if group in self.multi:
            group.set_selected(False)
            self._set_multi([g for g in self.multi if g is not group])
            return True
        return self.add_shape(index)

    def add_shape(self, index: int) -> bool:
        """Shift+click: add to the selection (an already selected shape becomes the primary one)."""
        group = self.index_to_group.get(index)
        if group is None:
            return False
        self._flush_nudge()
        old = self.selected_group
        if old is None or old is group:
            return self.select_shape(index, keep_multi=True)
        rest = [g for g in self.multi if g is not group] + [old]
        self.select_shape(index, keep_multi=True)
        self._set_multi(rest)
        return True

    def select_many(self, indices, primary=None) -> int:
        """Select exactly these shapes (those that are drawn). -> how many."""
        groups = [self.index_to_group[i] for i in dict.fromkeys(indices) if i in self.index_to_group]
        if not groups:
            self.clearSelection()
            return 0
        first = self.index_to_group.get(primary) if primary is not None else None
        first = first if first in groups else groups[-1]
        self.select_shape(first.index, keep_multi=False)
        self._set_multi([g for g in groups if g is not first])
        return len(groups)

    def select_all(self) -> int:
        """Ctrl+A — every drawn shape; only once a shape is selected (as in LabelMe), so the key stays free else.
        Works in both modes (selecting switches Drag to Select, as every selection does)."""
        if self.selected_group is None or self._drawing:
            return 0
        if self.mode == self.DRAG:
            self.setMode(self.SELECT)
        self._flush_nudge()
        self._set_multi([g for g in self.shape_groups if g is not self.selected_group])
        return len(self.multi) + 1

    def _peek(self, index: int):
        """A shape whose label is unticked in the Labels panel was asked for (Annotations dock, a QC finding):
        draw it while it is selected instead of failing silently — deselecting hides it again."""
        if not (0 <= index < len(self.shapes)) or self.pixmap_item.pixmap().isNull():
            return None
        if str(self.shapes[index].get("label", "?")) in self.visible_labels:
            return None                                  # visible but not drawable (no points / colour)
        self.peek_index = index
        self.drawShapes()
        group = self.index_to_group.get(index)
        if group is None:
            self.peek_index = None
        return group

    def is_peeked(self, index: int) -> bool:
        return self.peek_index is not None and index == self.peek_index

    def escape(self):
        """Esc: first it stops a drawing, lets go of the active handle (the arrows move the whole shape again), then of
        the shape."""
        if self._drawing:
            self.cancel_drawing()
            return
        if self._active_handle is not None and self.selected_group is not None:
            self._flush_nudge()
            self.set_active_handle(None)
            return
        self.clearSelection()

    def clearSelection(self):
        if self.selected_group is None:
            if self.peek_index is not None:
                self.peek_index = None
                self.drawShapes()
            return
        self._flush_nudge()
        self._set_multi([], tell=False)
        if self._active_handle is not None:
            self._active_handle = None
            self.activeHandleChanged.emit(False)
        self.selected_group.set_selected(False)
        self.selected_group = None
        self._emphasize_tag(None)
        self._clear_handles()
        if self.peek_index is not None:                  # the hidden shape was only shown while selected
            self.peek_index = None
            self.drawShapes()
        self.selectionRect.emit(None)
        self.selectionCleared.emit()
        self.selectionChanged.emit([])

    def focus_shape(self, index: int):
        """Select a shape and glide to it so it fills ~60 % of the view."""
        if not self.select_shape(index):
            return
        rect = self.selected_group.shape_item.sceneBoundingRect()
        if rect.isEmpty():
            self.animate_to(self.current_zoom, center=rect.center())
            return
        m = max(rect.width(), rect.height()) * 0.35
        rect = rect.adjusted(-m, -m, m, m)
        vp = self._viewport_size()
        z = min(vp.width() / rect.width(), vp.height() / rect.height())
        self.animate_to(z, center=rect.center(), mode=self.CUSTOM_MODE, duration=320)
