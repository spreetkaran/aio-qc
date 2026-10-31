# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
File list dock: model/view (instant for 100k files, unlike QListWidget),
text search, split filter and a "QC filter" banner used by every QC tool.
"""
from __future__ import annotations

from contextlib import contextmanager

from PyQt6.QtCore import QAbstractListModel, QModelIndex, QSize, QSortFilterProxyModel, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from AIO_QC.utils import newIcon, style as apply_style, theme_color
from PyQt6.QtWidgets import (
    QSizePolicy,
    QAbstractItemView, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QListView, QToolButton,
    QVBoxLayout, QWidget,
)

KEY_ROLE = Qt.ItemDataRole.UserRole + 1
REVIEW_FILTERS = [("All files", None), ("Unreviewed", "none"), ("✓ OK", "ok"), ("⚑ Flagged", "flag"),
                  ("🔧 Fixed", "fixed")]


def _badge(color: str, glyph: str) -> QIcon:
    pix = QPixmap(14, 14)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(0, 0, 14, 14)
    p.setPen(QColor("white"))
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
    p.end()
    return QIcon(pix)


class SampleListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.samples = []
        self.flags: dict[str, str] = {}     # key -> "error" | "empty"
        self.review = None                  # ReviewStore (set by the main window)
        self._icons = None
        self.hint = None                    # uniform row size = the longest name (horizontal scrolling)
        self._rows = None                   # key -> row, rebuilt lazily

    def set_samples(self, samples):
        self.beginResetModel()
        self.samples = list(samples)
        self._rows = None
        self.endResetModel()

    # ---- key -> row (rebuilt lazily after inserts / removals: one O(n) pass instead of one per lookup)
    def row_of(self, key: str) -> int:
        if getattr(self, "_rows", None) is None:
            self._rows = {s.key: i for i, s in enumerate(self.samples)}
        return self._rows.get(key, -1)

    def insert_samples(self, rows_and_samples):
        """[(final row, sample)] in ascending row order (Session.last_inserted)."""
        pairs = list(rows_and_samples)
        if len(pairs) > 2000:                      # many at once: one reset is cheaper than thousands of inserts
            for row, s in pairs:
                self.samples.insert(row, s)
            self.set_samples(self.samples)
            return
        for row, s in pairs:
            self.beginInsertRows(QModelIndex(), row, row)
            self.samples.insert(row, s)
            self.endInsertRows()
        self._rows = None

    def remove_keys(self, keys):
        rows = sorted((r for r in (self.row_of(k) for k in set(keys)) if r >= 0), reverse=True)
        i = 0
        while i < len(rows):                        # contiguous ranges, bottom-up
            hi = lo = rows[i]
            while i + 1 < len(rows) and rows[i + 1] == lo - 1:
                i += 1
                lo = rows[i]
            self.beginRemoveRows(QModelIndex(), lo, hi)
            del self.samples[lo:hi + 1]
            self.endRemoveRows()
            i += 1
        self._rows = None

    def rows_changed(self, keys, roles):
        """Signal only the rows that changed (a dynamic proxy re-filters just those)."""
        for k in keys:
            r = self.row_of(k)
            if r >= 0:
                idx = self.index(r)
                self.dataChanged.emit(idx, idx, roles)

    def set_flags(self, updates: dict[str, str | None]):
        for k, v in updates.items():
            if v:
                self.flags[k] = v
            else:
                self.flags.pop(k, None)
        self.rows_changed(updates.keys(), [Qt.ItemDataRole.ForegroundRole, Qt.ItemDataRole.ToolTipRole])

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.samples)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        s = self.samples[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return s.display
        if role == KEY_ROLE:
            return s.key
        if role == Qt.ItemDataRole.ToolTipRole:
            flag = self.flags.get(s.key)
            extra = {"error": "\n⚠ annotation could not be parsed", "empty": "\n(no shapes)",
                     "corrupt": "\n⚠ broken image (corrupt / truncated / inconsistent) — see Image Quality"}.get(flag, "")
            rv = self.review.get(s.key) if self.review else None
            if rv:
                extra += "\n" + {"ok": "✓ Reviewed OK", "flag": "⚑ Flagged", "fixed": "🔧 Fixed (check, then S = OK)"}.get(
                    rv["status"], rv["status"]) + f" · {rv.get('ts', '')}"
                if rv.get("note"):
                    extra += f"\n📝 {rv['note']}"
            where = " · ".join(x for x in (s.split and f"split: {s.split}", s.folder and f"folder: {s.folder}") if x)
            return f"{s.ann_path or s.image_path}" + (f"\n{where}" if where else "") + extra
        if role == Qt.ItemDataRole.SizeHintRole:
            return self.hint
        if role == Qt.ItemDataRole.DecorationRole and self.review is not None:
            if self._icons is None:            # same ⚑ / ✓ icons as every other view; blank keeps names aligned
                self._icons = {"ok": newIcon("review-ok"), "flag": newIcon("review-flag"),
                               "fixed": newIcon("review-fixed"),
                               None: _badge("#00000000", "")}
            return self._icons.get(self.review.status(s.key), self._icons[None])
        if role == Qt.ItemDataRole.ForegroundRole:
            flag = self.flags.get(s.key)
            if flag in ("error", "corrupt"):
                return theme_color("error")
            if flag == "empty":
                return theme_color("faint")
        return None


class SampleFilterProxy(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = ""
        self.split = ""
        self.folder = ""
        self.allowed: set[str] | None = None
        self.review_filter: str | None = None     # None | "none" | "ok" | "flag"
        self.rank: dict[str, int] | None = None   # key -> position (Issues Inbox: worst first); None = dataset order

    def lessThan(self, left, right):
        if self.rank is None:
            return left.row() < right.row()
        a = self.rank.get(left.data(KEY_ROLE), 1 << 30)
        b = self.rank.get(right.data(KEY_ROLE), 1 << 30)
        return a < b if a != b else left.row() < right.row()

    def refresh(self):
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        return self.accepts(self.sourceModel().samples[row])

    def filtering(self) -> bool:
        return bool(self.text or self.split or self.folder or self.allowed is not None
                    or self.review_filter is not None)

    def accepts(self, s) -> bool:
        if self.split and s.split != self.split:
            return False
        if self.folder and s.folder != self.folder:
            return False
        if self.allowed is not None and s.key not in self.allowed:
            return False
        if self.review_filter is not None:
            rv = self.sourceModel().review
            st = rv.status(s.key) if rv else None
            if (st or "none") != self.review_filter:
                return False
        return not self.text or self.text in s.display.lower()


class FileDock(QWidget):
    """
    "Parked" file: when the open file leaves the visible list (e.g. it is marked while the Unreviewed filter is on), Qt
    would silently move the selection to a NEIGHBOUR and load it — at the end of the list that is the previous file.
    Instead the open file stays open and the list remembers the gap it left: Next continues with the file that came
    after it, Previous with the one before; the file is re-selected if it becomes visible again.
    """
    currentKeyChanged = pyqtSignal(str)
    filterCleared = pyqtSignal()
    parked = pyqtSignal(str)            # the open file left the visible list (it stays open)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._silent = False
        self._parked_key: str | None = None
        self._parked_row: int | None = None
        self.model = SampleListModel(self)
        self.proxy = SampleFilterProxy(self)
        self.proxy.setSourceModel(self.model)
        # keep the parked gap in place while rows come and go (live updates, filters, sorting)
        self.proxy.rowsInserted.connect(self._parked_rows_inserted)
        self.proxy.rowsRemoved.connect(self._parked_rows_removed)
        self.proxy.layoutChanged.connect(self._parked_relayout)
        self.proxy.modelReset.connect(self._parked_relayout)

        self.search = QLineEdit(placeholderText="Search files…  (Ctrl+F)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._on_text)
        self._search_timer = QTimer(self, singleShot=True, interval=250, timeout=self._apply_text)
        self.split_box = QComboBox()
        self.split_box.addItem("All splits", "")
        self.split_box.currentIndexChanged.connect(self._on_split)
        self.split_box.setVisible(False)
        self.split_box.setToolTip("Split (train / val / test)")
        self.folder_box = QComboBox()
        self.folder_box.addItem("All folders", "")
        self.folder_box.setToolTip("Source sub-folder")
        self.folder_box.currentIndexChanged.connect(self._on_folder)
        self.folder_box.setVisible(False)
        self.review_box = QComboBox()
        for text, val in REVIEW_FILTERS:
            self.review_box.addItem(text, val)
        self.review_box.setToolTip("Filter by review status")
        self.review_box.currentIndexChanged.connect(self._on_review_filter)

        self.banner = QFrame()
        apply_style(self.banner, "QFrame{background:@banner_bg;border:1px solid @banner_line;border-radius:4px;}"
                                 "QLabel{color:@banner_fg;border:none;}")
        bl = QHBoxLayout(self.banner)
        bl.setContentsMargins(6, 2, 2, 2)
        self.banner_label = QLabel()
        self.banner_label.setWordWrap(True)
        clear = QToolButton(text="✕ Clear")
        clear.setToolTip("Show all files again")
        clear.clicked.connect(self.clear_key_filter)
        bl.addWidget(self.banner_label, 1)
        bl.addWidget(clear)
        self.banner.setVisible(False)

        self.view = QListView()
        self.view.setModel(self.proxy)
        self.view.setUniformItemSizes(True)
        # long names scroll sideways instead of widening the dock or being cut
        self.view.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        # Ctrl / Shift-click selects several files (right-click: flag / mark / fix them); navigation follows the
        # current file as before
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.view.selectionModel().currentChanged.connect(self._on_current)
        self.count_label = QLabel("0 files")
        apply_style(self.count_label, "color:@muted;")

        # search gets its own full-width row; filters share the second row and never force the dock wider
        for box in (self.split_box, self.folder_box, self.review_box):
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(6)
            box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            box.currentTextChanged.connect(lambda t, b=box: b.setToolTip(t))
        filters = QHBoxLayout()
        filters.setSpacing(4)
        filters.addWidget(self.split_box, 1)
        filters.addWidget(self.folder_box, 1)
        filters.addWidget(self.review_box, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.search)
        lay.addLayout(filters)
        lay.addWidget(self.banner)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.count_label)

    # ------------------------------------------------------------------ data
    def set_samples(self, samples, splits: list[str], folders: list[str] | None = None):
        self.proxy.allowed = None
        self.banner.setVisible(False)
        self.model.set_samples(samples)
        self.model.flags.clear()
        self.split_box.blockSignals(True)
        self.split_box.clear()
        self.split_box.addItem("All splits", "")
        for sp in splits:
            self.split_box.addItem(sp, sp)
        self.split_box.blockSignals(False)
        self.split_box.setVisible(bool(splits))
        self.proxy.split = ""
        self.folder_box.blockSignals(True)
        self.folder_box.clear()
        self.folder_box.addItem("All folders", "")
        for f in folders or []:
            self.folder_box.addItem(f, f)
        self.folder_box.blockSignals(False)
        self.folder_box.setVisible(bool(folders))
        self._fit_popup(self.folder_box)
        self._fit_popup(self.split_box)
        self.proxy.folder = ""
        self._update_hint()
        self.proxy.refresh()
        self._update_count()

    @staticmethod
    def _fit_popup(box: QComboBox):
        """The drop-down list is as wide as its longest entry (the box itself stays compact)."""
        fm = box.fontMetrics()
        w = max((fm.horizontalAdvance(box.itemText(i)) for i in range(box.count())), default=0)
        box.view().setMinimumWidth(w + 40)
        box.setToolTip(box.currentText())

    def _grow_hint(self, samples):
        """Widen the uniform row if a new name is longer (O(new), not O(all))."""
        fm = self.view.fontMetrics()
        icon = self.view.iconSize().width() or 16
        need = max((fm.horizontalAdvance(s.display) for s in samples), default=0) + icon + 20
        if self.model.hint is None or need > self.model.hint.width():
            self.model.hint = QSize(need, max(fm.height() + 6, icon + 4))
            self.view.doItemsLayout()

    def _update_hint(self):
        """Uniform row size from the longest name — computed once, so 100k+ files stay fast."""
        fm = self.view.fontMetrics()
        longest = max((fm.horizontalAdvance(s.display) for s in self.model.samples), default=0)
        icon = self.view.iconSize().width() or 16
        self.model.hint = QSize(longest + icon + 20, max(fm.height() + 6, icon + 4))

    def insert_samples(self, rows_and_samples):
        """New files (live updates) at their discovery-order rows — exactly where a reload would put them.
        The proxy maps the inserted rows itself (no full re-filter); the current file stays selected."""
        pairs = list(rows_and_samples)
        if not pairs:
            return
        current = self.current_key()
        self.model.insert_samples(pairs)
        self._grow_hint([s for _r, s in pairs])
        self._update_count()
        if current and self.current_key() != current:
            self.select_key(current, emit=False)

    def remove_keys(self, keys):
        current = self.current_key()
        row = self.view.currentIndex().row()
        ks = set(keys)
        if self._parked_key in ks:                  # the parked file was deleted: keep its gap for Next / Previous
            self._parked_key = None
        self.model.remove_keys(ks)                  # contiguous row removals, no reset
        if self.proxy.allowed is not None:
            self.proxy.allowed -= ks
        self._update_count()
        if current in ks and self.proxy.rowCount() > 0:
            self.select_row(min(max(row, 0), self.proxy.rowCount() - 1))
        elif current and current not in ks and self.current_key() != current:
            self.select_key(current, emit=False)

    # --------------------------------------------------------------- ordering
    def set_rank(self, ordered_keys: list[str] | None):
        """Show the list in this order (e.g. worst risk first); None restores the dataset order."""
        cur = self.current_key()
        if ordered_keys is None:
            self.proxy.rank = None
            self.proxy.sort(-1)
            self.proxy.invalidate()
        else:
            self.proxy.rank = {k: i for i, k in enumerate(ordered_keys)}
            self.proxy.sort(0)
            self.proxy.invalidate()
        if cur:
            self.select_key(cur, emit=False)

    # --------------------------------------------------------------- filters
    def set_key_filter(self, keys, title: str):
        self.proxy.allowed = set(keys)
        self.proxy.refresh()
        self.banner_label.setText(f"🔎 {title} — {self.proxy.rowCount():,} file(s)")
        self.banner.setVisible(True)
        self._update_count()
        if self.proxy.rowCount():
            cur = self.current_key()
            if not cur or cur not in self.proxy.allowed:
                self.select_row(0)

    def clear_key_filter(self):
        if self.proxy.allowed is None:
            return
        if self.proxy.rank is not None:              # a ranked queue ends with its filter
            self.set_rank(None)
        cur = self.current_key()
        self.proxy.allowed = None
        self.proxy.refresh()
        self.banner.setVisible(False)
        self._update_count()
        if cur:
            self.select_key(cur, emit=False)
        self.filterCleared.emit()

    def clear_all_filters(self):
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self.proxy.text = ""
        self.review_box.blockSignals(True)
        self.review_box.setCurrentIndex(0)
        self.review_box.blockSignals(False)
        self.proxy.review_filter = None
        self.split_box.blockSignals(True)
        self.split_box.setCurrentIndex(0)
        self.split_box.blockSignals(False)
        self.proxy.split = ""
        self.folder_box.blockSignals(True)
        self.folder_box.setCurrentIndex(0)
        self.folder_box.blockSignals(False)
        self.proxy.folder = ""
        self.clear_key_filter()
        self.proxy.refresh()
        self._update_count()

    def _on_text(self, text):
        self.proxy.text = text.strip().lower()
        if len(self.model.samples) > 20000:        # big lists: filter once typing pauses
            self._search_timer.start()
        else:
            self._apply_text()

    def _apply_text(self):
        self.proxy.refresh()
        self._update_count()

    def _on_folder(self, _):
        self.proxy.folder = self.folder_box.currentData() or ""
        self.proxy.refresh()
        self._update_count()

    def _on_split(self, _):
        self.proxy.split = self.split_box.currentData() or ""
        self.proxy.refresh()
        self._update_count()

    def _on_review_filter(self, _):
        self.proxy.review_filter = self.review_box.currentData()
        self.proxy.refresh()
        self._update_count()

    def review_changed(self, keys):
        """Only the changed rows: badges repaint, and with a review filter the dynamic proxy re-checks just them.
        If that removes the open file from the list, it is parked instead of jumping to a neighbour."""
        cur, row = self.current_key(), self.view.currentIndex().row()
        with self.hold():
            self.model.rows_changed(keys, [Qt.ItemDataRole.DecorationRole, Qt.ItemDataRole.ToolTipRole])
        if self.proxy.review_filter is not None:
            self._update_count()
        if cur and self.proxy_row_of(cur) < 0:
            self._park(cur, row)

    # --------------------------------------------------------------- parked file
    @contextmanager
    def hold(self):
        """Selection moves made by Qt inside this block do not open another file."""
        prev = self._silent
        self._silent = True
        try:
            yield
        finally:
            self._silent = prev

    def proxy_row_of(self, key: str) -> int:
        row = self.model.row_of(key)
        if row < 0:
            return -1
        p = self.proxy.mapFromSource(self.model.index(row, 0))
        return p.row() if p.isValid() else -1

    def is_parked(self) -> bool:
        return self._parked_row is not None and not self.view.currentIndex().isValid()

    def _park(self, key: str, row: int):
        self._parked_key = key
        self._parked_row = max(0, min(row, self.proxy.rowCount()))
        with self.hold():
            self.view.selectionModel().clear()             # no neighbour highlighted for a file that is not open
        self.parked.emit(key)

    def _unpark(self):
        self._parked_key = self._parked_row = None

    def _parked_rows_inserted(self, _parent, first, last):
        """Rows inserted at or before the gap move it only if they come BEFORE the parked file in dataset order
        (a row inserted exactly at the gap may belong on either side of it)."""
        if self._parked_row is not None and first <= self._parked_row:
            anchor = self.model.row_of(self._parked_key) if self._parked_key else -1
            before = 0
            for r in range(first, last + 1):
                src = self.proxy.mapToSource(self.proxy.index(r, 0)).row()
                before += anchor < 0 or src < anchor
            self._parked_row += before
        QTimer.singleShot(0, self._try_unpark)

    def _parked_rows_removed(self, _parent, first, last):
        if self._parked_row is not None:
            if last < self._parked_row:
                self._parked_row -= last - first + 1
            elif first < self._parked_row:
                self._parked_row = first

    def _parked_relayout(self, *_):
        if self._parked_row is not None:
            self._parked_row = min(self._parked_row, self.proxy.rowCount())
        QTimer.singleShot(0, self._try_unpark)

    def _try_unpark(self):
        """The parked file is visible again (filter changed): select it again, without reloading it."""
        if self._parked_key is not None and self.proxy_row_of(self._parked_key) >= 0:
            key = self._parked_key
            self._unpark()
            self.select_key(key, emit=False)

    def _anchor(self) -> tuple[int, bool]:
        """(row, parked): the current row, or the gap the parked file left."""
        idx = self.view.currentIndex()
        if idx.isValid():
            return idx.row(), False
        if self._parked_row is not None:
            return self._parked_row, True
        return -1, False

    def key_at_offset(self, delta: int) -> str | None:
        n = self.proxy.rowCount()
        row, parked = self._anchor()
        if parked:                                  # the gap: the next file now sits AT the gap row
            row = row + delta - 1 if delta > 0 else row + delta
        else:
            row += delta
        return self.proxy.index(row, 0).data(KEY_ROLE) if 0 <= row < n else None

    def _update_count(self):
        total, shown = self.model.rowCount(), self.proxy.rowCount()
        self.count_label.setText(f"{shown:,} / {total:,} files" if shown != total else f"{total:,} files")

    # ------------------------------------------------------------ navigation
    def _on_current(self, current, _previous):
        if current.isValid():
            self._unpark()
            if not self._silent:
                self.currentKeyChanged.emit(current.data(KEY_ROLE))

    def current_key(self) -> str | None:
        idx = self.view.currentIndex()
        return idx.data(KEY_ROLE) if idx.isValid() else None

    def current_position(self) -> tuple[int, int]:
        """(1-based row, rows); row 0 = the open file is not in the list (parked)."""
        return self.view.currentIndex().row() + 1, self.proxy.rowCount()

    def select_row(self, row: int, emit: bool = True):
        if not 0 <= row < self.proxy.rowCount():
            return
        idx = self.proxy.index(row, 0)
        self._silent = not emit
        try:
            self.view.setCurrentIndex(idx)
            self.view.scrollTo(idx)
        finally:
            self._silent = False

    def select_key(self, key: str, emit: bool = True) -> bool:
        row = self.model.row_of(key)               # O(1) (was a scan of every file)
        if row < 0:
            return False
        pidx = self.proxy.mapFromSource(self.model.index(row, 0))
        if not pidx.isValid():
            return False
        self.select_row(pidx.row(), emit)
        return True

    def _all_visible_keys(self, n: int) -> list[str]:
        """Every listed key in list order, computed from the samples (no Qt call per row: ~50× faster, which
        matters at a million files). Falls back to asking the proxy if the two ever disagree."""
        p, samples = self.proxy, self.model.samples
        if not p.filtering():
            keys = [s.key for s in samples]
        else:
            acc = p.accepts
            keys = [s.key for s in samples if acc(s)]
        if p.rank is not None:
            big = 1 << 30
            keys.sort(key=lambda k: p.rank.get(k, big))           # stable: ties keep dataset order
        if len(keys) != n:
            return [p.index(r, 0).data(KEY_ROLE) for r in range(n)]
        return keys

    def next_matching(self, pred) -> str | None:
        """First visible file after the current one (wrapping) for which pred(key) is true — stops at the first hit."""
        n = self.proxy.rowCount()
        if n == 0:
            return None
        start = self.view.currentIndex().row()
        for step in range(1, n + 1):
            k = self.proxy.index((start + step) % n, 0).data(KEY_ROLE)
            if pred(k):
                return k
        return None

    def step(self, delta: int):
        """Next / previous; stops at both ends (never wraps, never bounces back)."""
        n = self.proxy.rowCount()
        if n == 0:
            return
        row, parked = self._anchor()
        if row < 0:
            self.select_row(0)
            return
        target = (row + delta - 1 if delta > 0 else row + delta) if parked else row + delta
        if 0 <= target < n:
            self.select_row(target)

    def selected_keys(self) -> list[str]:
        rows = sorted(self.view.selectionModel().selectedRows(), key=lambda i: i.row())
        keys = [self.model.samples[self.proxy.mapToSource(i).row()].key for i in rows]
        if not keys:
            cur = self.view.currentIndex()
            if cur.isValid():
                keys = [self.model.samples[self.proxy.mapToSource(cur).row()].key]
        return keys

    def visible_keys(self, around: str | None = None, before: int = 0, after: int = 0) -> list[str]:
        n = self.proxy.rowCount()
        if around is None:
            return self._all_visible_keys(n)
        row = self.view.currentIndex().row()
        rows = range(max(0, row - before), min(n, row + after + 1))
        return [self.proxy.index(r, 0).data(KEY_ROLE) for r in rows]
