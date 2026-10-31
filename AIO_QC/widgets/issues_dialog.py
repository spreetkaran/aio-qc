# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
📥 Issues Inbox — every tool's findings in one place, one risk score per file, one review queue (worst first).
Model/view table (scales to hundreds of thousands of files); refreshes live as tools publish and marks change.
"""
from __future__ import annotations

import csv
from collections import Counter

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QRectF, QSortFilterProxyModel, Qt, QTimer
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMenu, QSplitter, QStyledItemDelegate, QTableView, QToolButton, QVBoxLayout,
)

from AIO_QC.core.state import REVIEW_FIXED, REVIEW_FLAG, REVIEW_OK
from AIO_QC.utils import hexc, newButton, newIcon, style as apply_style, theme_color

COLS = ["Risk", "File", "Errors", "Warnings", "Info", "Tools", "Top issue", "Review"]
SEV_ICON = {"error": "review-clear", "warn": "review-flag", "info": "review-next"}
SEV_ROLE = {"error": "error", "warn": "warn", "info": "muted"}


class _Model(QAbstractTableModel):
    def __init__(self, dlg):
        super().__init__(dlg)
        self.dlg = dlg
        self.rows: list[tuple] = []        # (key, risk, err, warn, info, tools, top, display)

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return len(COLS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return COLS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        key, risk, e, w, i, tools, top, disp = self.rows[index.row()]
        c = index.column()
        if role == Qt.ItemDataRole.UserRole:
            return key
        if role == Qt.ItemDataRole.UserRole + 1:                 # sort value
            return (risk, disp, e, w, i, tools, top, self.dlg.review_text(key))[c]
        if role == Qt.ItemDataRole.DisplayRole:
            return (f"{risk:.0f}", disp, e or "", w or "", i or "", tools, top, self.dlg.review_text(key))[c]
        if role == Qt.ItemDataRole.DecorationRole and c == 7:
            st = self.dlg.session.review.status(key)
            return newIcon({"ok": "review-ok", "flag": "review-flag", "fixed": "review-fixed"}[st]) if st else None
        if role == Qt.ItemDataRole.ForegroundRole and c in (2, 3):
            return theme_color("error" if c == 2 else "warn")
        if role == Qt.ItemDataRole.ToolTipRole and c in (1, 6):
            return "\n".join(f"[{x.severity}] {self.dlg.session.issues.titles.get(x.tool, x.tool)}: {x.text}"
                             for x in self.dlg.session.issues.file_issues(key)[:20])
        return None


class _SortProxy(QSortFilterProxyModel):
    def lessThan(self, a, b):
        return a.data(Qt.ItemDataRole.UserRole + 1) < b.data(Qt.ItemDataRole.UserRole + 1)


class _RiskBar(QStyledItemDelegate):
    def paint(self, p: QPainter, opt, index):
        risk = float(index.data(Qt.ItemDataRole.UserRole + 1) or 0)
        r = QRectF(opt.rect).adjusted(4, 6, -4, -6)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        track = theme_color("line")
        p.setBrush(track)
        p.drawRoundedRect(r, 3, 3)
        col = theme_color("error" if risk >= 60 else "warn" if risk >= 25 else "info")
        p.setBrush(col)
        p.drawRoundedRect(QRectF(r.left(), r.top(), max(3.0, r.width() * risk / 100), r.height()), 3, 3)
        p.setPen(opt.palette.text().color())
        p.drawText(opt.rect.adjusted(0, 0, -6, 0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"{risk:.0f}")
        p.restore()


class IssuesDialog(QDialog):
    TITLE, ICON = "Issues Inbox", "📥"
    SUPPORTED = None

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(newIcon("tool-inbox"))
        self.resize(int(app.width() * 0.7), int(app.height() * 0.75))
        self.tool_filter: set[str] = set()

        self.head = QLabel("")
        self.head.setTextFormat(Qt.TextFormat.RichText)
        f = self.head.font()
        f.setPointSizeF(f.pointSizeF() + 1)
        self.head.setFont(f)
        self.tools_row = QHBoxLayout()
        self.tools_row.setSpacing(6)
        flt = QHBoxLayout()
        self.sev = QComboBox()
        self.sev.addItem("All findings", "info")
        self.sev.addItem("Warnings and errors", "warn")
        self.sev.addItem("Errors only", "error")
        self.rev = QComboBox()
        for t, d in (("Any review state", None), ("Unreviewed", "none"), ("Flagged", REVIEW_FLAG),
                     ("Fixed", REVIEW_FIXED), ("Marked OK", REVIEW_OK)):
            self.rev.addItem(t, d)
        self.rev.setCurrentIndex(1)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search files or issues…")
        self.search.setClearButtonEnabled(True)
        for w in (self.sev, self.rev):
            w.currentIndexChanged.connect(lambda _i: self.refresh())
        self.search.textChanged.connect(lambda _t: self._debounce.start())
        flt.addWidget(QLabel("Show:"))
        flt.addWidget(self.sev)
        flt.addWidget(self.rev)
        flt.addWidget(self.search, 1)

        self.model = _Model(self)
        self.proxy = _SortProxy(self)
        self.proxy.setSourceModel(self.model)
        self.view = QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        self.view.sortByColumn(0, Qt.SortOrder.DescendingOrder)
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(26)
        self.view.setItemDelegateForColumn(0, _RiskBar(self.view))
        hh = self.view.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.view.setColumnWidth(0, 110)
        self.view.selectionModel().currentRowChanged.connect(lambda cur, _p: self._show_details(cur))
        self.view.doubleClicked.connect(lambda idx: self._open(idx))
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._menu)
        self.details = QListWidget()
        self.details.itemDoubleClicked.connect(self._open_issue)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.view)
        split.addWidget(self.details)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)

        foot = QHBoxLayout()
        foot.addWidget(newButton("Review worst first", icon="review-next", slot=self._queue,
                                 tip="Show these files in the file list, worst first, and open the first unreviewed one"))
        foot.addWidget(newButton("Flag", icon="review-flag", slot=lambda: self._mark(REVIEW_FLAG)))
        foot.addWidget(newButton("Mark OK", icon="review-ok", slot=lambda: self._mark(REVIEW_OK)))
        foot.addWidget(newButton("Fix in editor", icon="tool-editor", slot=self._fix))
        foot.addStretch(1)
        self.clear_btn = QToolButton()
        self.clear_btn.setText("Clear findings ▾")
        self.clear_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.clear_btn.setMenu(QMenu(self.clear_btn))
        self.clear_btn.menu().aboutToShow.connect(self._fill_clear)
        foot.addWidget(self.clear_btn)
        foot.addWidget(newButton("Export CSV", icon="save", slot=self._export))
        foot.addWidget(newButton("Close", slot=self.close))

        lay = QVBoxLayout(self)
        lay.addWidget(self.head)
        lay.addLayout(self.tools_row)
        lay.addLayout(flt)
        lay.addWidget(split, 1)
        lay.addLayout(foot)

        self._debounce = QTimer(self, singleShot=True, interval=250, timeout=self.refresh)
        self.session.issuesChanged.connect(self._debounce.start)
        self.session.reviewChanged.connect(lambda _k: self._debounce.start())
        self.session.samplesRemoved.connect(lambda _k: self._debounce.start())
        self.refresh()

    # ------------------------------------------------------------------ data
    def review_text(self, key):
        return {"ok": "OK", "flag": "flagged", "fixed": "fixed"}.get(self.session.review.status(key), "")

    def _tool_chips(self, summ):
        while self.tools_row.count():
            w = self.tools_row.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        store = self.session.issues
        lbl = QLabel("Tools:")
        apply_style(lbl, "color:@muted;")
        self.tools_row.addWidget(lbl)
        for tool, n in summ["tools"].most_common():
            b = QToolButton()
            b.setCheckable(True)
            b.setChecked(not self.tool_filter or tool in self.tool_filter)
            b.setText(f"{store.titles.get(tool, tool)}  {n:,}")
            b.setToolTip(f"Last run {store.when.get(tool, '?')} — click to show / hide its findings")
            b.toggled.connect(lambda on, t=tool: self._toggle_tool(t, on))
            self.tools_row.addWidget(b)
        self.tools_row.addStretch(1)

    def _toggle_tool(self, tool, on):
        all_tools = set(self.session.issues.by_tool)
        cur = self.tool_filter or set(all_tools)
        cur = (cur | {tool}) if on else (cur - {tool})
        self.tool_filter = set() if cur == all_tools else cur
        self.refresh(rebuild_chips=False)

    def refresh(self, rebuild_chips: bool = True):
        s = self.session
        store = s.issues
        alive = set(s.by_key)
        minsev = self.sev.currentData()
        order = {"error": 0, "warn": 1, "info": 2}
        rev = self.rev.currentData()
        q = self.search.text().strip().lower()
        rows = []
        for key, _risk in store.ranked(alive):
            iss = [x for x in store.file_issues(key) if (not self.tool_filter or x.tool in self.tool_filter)
                   and order.get(x.severity, 2) <= order[minsev]]
            if not iss:
                continue
            st = s.review.status(key)
            if rev == "none" and st is not None or rev not in (None, "none") and st != rev:
                continue
            disp = s.by_key[key].display
            top = f"{store.titles.get(iss[0].tool, iss[0].tool)}: {iss[0].text}"
            if q and q not in disp.lower() and not any(q in x.text.lower() for x in iss):
                continue
            c = Counter(x.severity for x in iss)
            tools = ", ".join(sorted({store.titles.get(x.tool, x.tool) for x in iss}))
            rows.append((key, store.risk(key), c["error"], c["warn"], c["info"], tools, top, disp))
        self.model.set_rows(rows)
        summ = store.summary(alive)
        sev = summ["severity"]
        self.head.setText(
            f"<b>{summ['files']:,}</b> file(s) with findings · "
            f"<span style='color:{hexc('error')}'>{sev['error']:,} errors</span> · "
            f"<span style='color:{hexc('warn')}'>{sev['warn']:,} warnings</span> · "
            f"<span style='color:{hexc('muted')}'>{sev['info']:,} info</span>"
            f"<span style='color:{hexc('muted')}'> — showing {len(rows):,}. Run QC tools to fill the inbox; "
            "each rescan replaces that tool's findings.</span>")
        if rebuild_chips:
            self._tool_chips(summ)

    # ------------------------------------------------------------------ selection / actions
    def _key_at(self, idx):
        return self.proxy.data(idx, Qt.ItemDataRole.UserRole) if idx.isValid() else None

    def _selected_keys(self):
        rows = {i.row() for i in self.view.selectionModel().selectedRows()}
        if not rows and self.view.currentIndex().isValid():
            rows = {self.view.currentIndex().row()}
        return [self.proxy.index(r, 0).data(Qt.ItemDataRole.UserRole) for r in sorted(rows)]

    def _show_details(self, idx):
        self.details.clear()
        key = self._key_at(idx)
        if not key:
            return
        store = self.session.issues
        for x in store.file_issues(key):
            it = QListWidgetItem(newIcon(SEV_ICON.get(x.severity, "review-next")),
                                 f"{store.titles.get(x.tool, x.tool)} — {x.text}")
            it.setForeground(theme_color(SEV_ROLE.get(x.severity, "muted")))
            it.setData(Qt.ItemDataRole.UserRole, (key, x.shape))
            self.details.addItem(it)

    def _open(self, idx):
        key = self._key_at(idx)
        if key:
            self.app.open_sample(key)

    def _open_issue(self, item):
        key, shape = item.data(Qt.ItemDataRole.UserRole)
        self.app.open_sample(key, shape_index=shape)

    def _queue(self):
        keys = [self.proxy.index(r, 0).data(Qt.ItemDataRole.UserRole) for r in range(self.proxy.rowCount())]
        if not keys:
            return
        self.app.file_dock.set_key_filter(keys, f"📥 Issues Inbox — {len(keys):,} file(s), worst first")
        self.app.file_dock.set_rank(keys)
        status = self.session.review.status
        first = next((k for k in keys if status(k) is None), keys[0])
        self.app.file_dock.select_key(first)

    def _mark(self, status):
        keys = self._selected_keys()
        if keys:
            self.session.set_review(keys, status, "from the Issues Inbox" if status == REVIEW_FLAG else None)

    def _fix(self):
        keys = self._selected_keys()
        if keys:
            self.app.open_editor_session(keys)

    def _menu(self, pos):
        if not self.view.indexAt(pos).isValid():
            return
        m = QMenu(self)
        m.addAction(newIcon("eye"), "Open", lambda: self._open(self.view.currentIndex()))
        m.addAction(newIcon("review-flag"), "Flag", lambda: self._mark(REVIEW_FLAG))
        m.addAction(newIcon("review-ok"), "Mark OK", lambda: self._mark(REVIEW_OK))
        m.addAction(newIcon("tool-editor"), "Fix in editor…", self._fix)
        m.exec(self.view.viewport().mapToGlobal(pos))

    def _fill_clear(self):
        m = self.clear_btn.menu()
        m.clear()
        store = self.session.issues
        for t in sorted(store.by_tool, key=lambda t: store.titles.get(t, t)):
            m.addAction(f"{store.titles.get(t, t)} (run {store.when.get(t, '?')})",
                        lambda t=t: (store.clear_tool(t), self.session.issuesChanged.emit()))

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export issues", "issues.csv", "CSV (*.csv)")
        if not path:
            return
        store = self.session.issues
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "risk", "tool", "severity", "issue", "shape", "review"])
            for key, risk, *_ in self.model.rows:
                for x in store.file_issues(key):
                    w.writerow([self.session.by_key[key].display, f"{risk:.0f}", store.titles.get(x.tool, x.tool),
                                x.severity, x.text, "" if x.shape is None else x.shape, self.review_text(key)])
        self.app.statusBar().showMessage(f"Exported to {path}", 5000)

    def closeEvent(self, e):
        self.app.forget_dialog(self)
        super().closeEvent(e)
