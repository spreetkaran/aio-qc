# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Generic QC tool dialog.

Subclasses implement:
    build_params(form)            add parameter widgets (main thread)
    make_context() -> dict        snapshot params for the worker (main thread)
    scan_sample(ctx, sample)      -> list[ResultRow]    (WORKER THREAD - no widgets!)
    finalize(ctx)                 -> list[ResultRow]    (worker thread, dataset-level)
and add action buttons with add_action().
"""
from __future__ import annotations

import csv
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from loguru import logger
from PyQt6.QtCore import QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QColor, QFont, QPainter, QPalette
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar, QPushButton, QStyle, QStyledItemDelegate,
    QStyleOptionViewItem, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

from AIO_QC.utils import theme_color, style as apply_style
from AIO_QC.core.state import REVIEW_FLAG, REVIEW_OK
from AIO_QC.utils import newIcon, emphasis_bg, get_color_icon, newButton
from AIO_QC.widgets.lifecycle import RetainedThread

ROW_ID = Qt.ItemDataRole.UserRole + 10
BAR_ROLE = Qt.ItemDataRole.UserRole + 20     # (fraction 0..1, "#rrggbb") -> drawn as a data bar
TONE_ROLES = {"error": "error", "warning": "warn", "muted": "muted", "high": "ok", "good": "accent",
              "low": "warn", "rare": "error", "bold": None}


def TONE(name):
    """Row tone -> theme colour, resolved when the row is drawn (readable in both themes)."""
    role = TONE_ROLES.get(name)
    return theme_color(role) if role else None


@dataclass(slots=True)             # slots: ~2-3x less memory per row (scans of millions of files)
class ResultRow:
    key: str
    cells: tuple
    shape_index: int | None = None
    payload: Any = None
    tone: str | None = None
    bg: str | None = None                 # row background (hex)
    fg: str | None = None                 # first-column text colour (hex, drawn bold)
    bars: dict | None = None              # {cell index: (fraction, hex)}
    dot: str | None = None                # coloured status dot in front of the first cell (hex)
    emphasis: bool = False                # TOTAL-style row: bold + theme-aware shading


class BarDelegate(QStyledItemDelegate):
    """
    Number in the theme's text colour (right-aligned) with a thin 3 px bar underneath — readable in light and
    dark themes, no coloured backgrounds behind text.
    """

    def paint(self, painter, option, index):
        bar = index.data(BAR_ROLE)
        if not bar:
            return super().paint(painter, option, index)
        frac, color = bar
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text, opt.text = opt.text, ""
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)   # bg / selection
        r = QRectF(option.rect.adjusted(8, 0, -8, 0))
        track = QRectF(r.x(), r.bottom() - 6, r.width(), 3)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        t = QColor(opt.palette.color(QPalette.ColorRole.Text))
        t.setAlpha(40)
        painter.setBrush(t)
        painter.drawRoundedRect(track, 1.5, 1.5)
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(QRectF(track.x(), track.y(), max(2.0, track.width() * max(0.0, min(float(frac), 1.0))),
                                       track.height()), 1.5, 1.5)
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        painter.setPen(opt.palette.color(QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.Text))
        painter.setFont(opt.font)
        painter.drawText(r.adjusted(0, 0, 0, -5), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), text)
        painter.restore()


def readable_table(table, weight=QFont.Weight.Normal, delta_pt: float = 0.0, row_height: int = 26):
    """Product look: normal-weight body text at the system size, semi-bold header, comfortable rows."""
    f = QFont(table.font())
    f.setPointSizeF(max(f.pointSizeF(), 9.0) + delta_pt)
    f.setWeight(weight)
    table.setFont(f)
    hf = QFont(f)
    hf.setWeight(QFont.Weight.DemiBold)
    table.horizontalHeader().setFont(hf)
    table.verticalHeader().setDefaultSectionSize(row_height)
    table.setItemDelegate(BarDelegate(table))


def dspin(lo, hi, value, decimals=2, step=None, suffix=""):
    from PyQt6.QtWidgets import QDoubleSpinBox
    s = QDoubleSpinBox()
    s.setDecimals(decimals)
    s.setRange(lo, hi)
    if step:
        s.setSingleStep(step)
    s.setSuffix(suffix)
    s.setValue(value)
    s.setKeyboardTracking(False)
    return s


def ispin(lo, hi, value):
    from PyQt6.QtWidgets import QSpinBox
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    s.setKeyboardTracking(False)
    return s


class ScanWorker(RetainedThread, QThread):
    rows_ready = pyqtSignal(list)
    progress = pyqtSignal(int, int)
    completed = pyqtSignal(list, list, bool)   # final rows, errors, cancelled

    def __init__(self, samples, scan_fn, finalize_fn, ctx):
        super().__init__()
        self.samples, self.scan_fn, self.finalize_fn, self.ctx = list(samples), scan_fn, finalize_fn, ctx
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        errors, batch, last, total = [], [], time.monotonic(), len(self.samples)
        for i, s in enumerate(self.samples, 1):
            if self._stop:
                break
            try:
                rows = self.scan_fn(self.ctx, s)
                if rows:
                    batch.extend(rows)
            except Exception as exc:
                errors.append(f"{s.name}: {type(exc).__name__}: {exc}")
            now = time.monotonic()
            if now - last > 0.15 or i == total:
                if batch:
                    self.rows_ready.emit(batch)
                    batch = []
                self.progress.emit(i, total)
                last = now
        if batch:
            self.rows_ready.emit(batch)
        final = []
        if not self._stop:
            try:
                final = self.finalize_fn(self.ctx) or []
            except Exception as exc:
                logger.exception("finalize failed")
                errors.append(f"finalize: {type(exc).__name__}: {exc}")
        self.completed.emit(final, errors, self._stop)


class BaseScanDialog(QDialog):
    TITLE = "QC Scan"
    ICON = "🔍"
    HELP = ""
    COLUMNS: list[str] = []
    FILE_COLUMN = True        # show "File" column
    REVIEW_ACTIONS = True     # "⚑ Flag files" action + right-click review menu
    CHECKABLE = True          # rows have checkboxes
    FILTER_COLUMN: int | None = None   # index into COLUMNS for the quick filter combo
    AUTO_START = False
    RESCAN_AFTER_EDIT = True
    SUPPORTED = None          # None = all dataset types, else set of DatasetType
    PUBLISH_ISSUES = None     # findings go to the Issues Inbox (None = same as REVIEW_ACTIONS)

    def __init__(self, app, parent=None):
        super().__init__(parent or app)
        self.app = app
        self.session = app.session
        self.rows: list[ResultRow] = []
        self.errors: list[str] = []
        self.ctx: dict = {}
        self.worker: ScanWorker | None = None
        self._scan_started = 0.0

        self.setWindowTitle(self.TITLE)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(int(app.width() * 0.62), int(app.height() * 0.75))

        root = QVBoxLayout(self)
        if self.HELP:
            help_lbl = QLabel(self.HELP)
            help_lbl.setWordWrap(True)
            apply_style(help_lbl, "color:@muted;")
            root.addWidget(help_lbl)

        from .guidelines import GuideCard
        self.guide_card = GuideCard(self, compact=True)      # this tool's rule from Labels ▸ Guidelines, if any
        self.guide_card.editRequested.connect(lambda sc: self.app.open_guidelines(sc))
        root.addWidget(self.guide_card)

        self.actions_layout = QHBoxLayout()   # subclasses add buttons from build_params()

        # parameters -----------------------------------------------------------
        self.params_box = QGroupBox("Parameters")
        self.params_form = QFormLayout(self.params_box)
        self.build_params(self.params_form)
        self.params_box.setVisible(self.params_form.rowCount() > 0)
        root.addWidget(self.params_box)

        # run controls ---------------------------------------------------------
        run = QHBoxLayout()
        self.start_btn = newButton("▶ Scan", slot=self.start_scan)
        self.start_btn.setDefault(True)
        self.stop_btn = newButton("⏹ Stop", slot=self.stop_scan)
        self.stop_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v / %m")
        self.status = QLabel("Ready")
        apply_style(self.status, "color:@muted;")
        run.addWidget(self.start_btn)
        run.addWidget(self.stop_btn)
        run.addWidget(self.progress, 1)
        run.addWidget(self.status)
        root.addLayout(run)

        # results toolbar ------------------------------------------------------
        bar = QHBoxLayout()
        self.select_all = QCheckBox("Select all")
        self.select_all.setVisible(self.CHECKABLE)
        self.select_all.toggled.connect(self._toggle_all)
        bar.addWidget(self.select_all)
        self.filter_box = QComboBox()
        self.filter_box.setVisible(self.FILTER_COLUMN is not None)
        self.filter_box.currentIndexChanged.connect(self._apply_filter)
        if self.FILTER_COLUMN is not None:
            bar.addWidget(QLabel(f"{self.COLUMNS[self.FILTER_COLUMN]}:"))
        bar.addWidget(self.filter_box)
        bar.addStretch(1)
        bar.addLayout(self.actions_layout)
        self.show_files_btn = newButton("👁 Show in file list", slot=self.show_in_file_list, tip="Restrict the main file list to the files in these results")
        export_btn = newButton("⤓ CSV", slot=self.export_csv, tip="Export visible results to CSV")
        bar.addWidget(self.show_files_btn)
        bar.addWidget(export_btn)
        root.addLayout(bar)

        # table ----------------------------------------------------------------
        self.table = QTableWidget(0, 0)
        self._set_headers(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        readable_table(self.table)
        self.table.cellDoubleClicked.connect(lambda r, _c: self.on_row_activated(self._row_at(r)))
        self.table_host = QVBoxLayout()         # subclasses may wrap the table (e.g. in tabs)
        self.table_host.addWidget(self.table)
        root.addLayout(self.table_host, 1)

        foot = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        foot.addWidget(self.summary, 1)
        hint = QLabel("Double-click a row to open it")
        apply_style(hint, "color:@faint;")
        foot.addWidget(hint)
        close = newButton("Close", slot=self.close)
        foot.addWidget(close)
        root.addLayout(foot)

        self.session.samplesRemoved.connect(self._on_samples_removed)
        self.stale = QLabel("")
        self.stale.setWordWrap(True)
        self.stale.setVisible(False)
        self.stale.setTextFormat(Qt.TextFormat.RichText)
        self.stale.linkActivated.connect(lambda _l: self.start_scan())
        apply_style(self.stale, "QLabel{background:@banner_bg; color:@banner_fg; border:1px solid @banner_line; "
                                "border-radius:4px; padding:4px 8px;}")
        self.table_host.insertWidget(0, self.stale)
        # review marks from every tool: action, right-click menu, live badges
        if self.REVIEW_ACTIONS:
            self.add_action("⚑ Flag files", self.flag_files,
                            "Flag the files of the ticked (or selected) rows for review — nothing ticked: every file "
                            "found (in the current filter). The note says which tool and why; shown in the file list, "
                            "filmstrip, coverage map and every tool")
        if self.REVIEW_ACTIONS:
            self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.table.customContextMenuRequested.connect(self._review_menu)
        self.session.reviewChanged.connect(self._on_review_changed)
        self.post_init()
        if self.AUTO_START:
            QTimer.singleShot(0, self.start_scan)

    # ============================================================ subclass API
    def build_params(self, form: QFormLayout):
        pass

    def make_context(self) -> dict:
        return {}

    def scan_sample(self, ctx: dict, sample) -> list[ResultRow]:
        return []

    def finalize(self, ctx: dict) -> list[ResultRow]:
        return []

    def rescan_sample(self, ctx: dict, sample) -> list[ResultRow]:
        return self.scan_sample(ctx, sample)

    def columns_for(self, ctx: dict) -> list[str] | None:
        return None

    def post_init(self):
        """Main thread, after every widget (incl. the table) exists."""

    def on_scan_complete(self, ctx: dict):
        """Main thread, after all rows are in."""

    def summarize(self) -> str:
        return f"{len(self.rows):,} result(s)"

    def on_row_activated(self, row: ResultRow | None):
        if row is not None and row.key:
            self.app.open_sample(row.key, row.shape_index)

    def add_action(self, text: str, slot, tip: str = "") -> QPushButton:
        btn = newButton(text, slot=lambda _checked=False: slot(), tip=tip)
        self.actions_layout.addWidget(btn)
        return btn

    # =============================================================== scanning
    def mark_stale(self, n_files: int):
        """Files changed outside AIO QC since this scan: say so (a rescan can be expensive, so it is not automatic)."""
        if not self.rows and self.worker is None:
            return
        self._stale_n = getattr(self, "_stale_n", 0) + n_files
        self.stale.setText(f"⚠ {self._stale_n:,} file(s) changed outside AIO QC since this scan — results may be "
                           "outdated. <a href='rescan'>Rescan</a>")
        self.stale.setVisible(True)

    def start_scan(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self._stale_n = 0
        if hasattr(self, "stale"):
            self.stale.setVisible(False)
        if hasattr(self, "_marked"):
            self._marked.clear()
        self.table.setRowCount(0)
        self.rows.clear()
        self.errors.clear()
        self.select_all.blockSignals(True)
        self.select_all.setChecked(False)
        self.select_all.blockSignals(False)
        try:
            self.ctx = self.make_context()
        except ValueError as exc:
            QMessageBox.warning(self, self.TITLE, str(exc))
            return
        samples = self.samples_to_scan()
        self._scan_scope = {s.key for s in samples}
        self.progress.setRange(0, max(1, len(samples)))
        self.progress.setValue(0)
        self.status.setText("Scanning…")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.params_box.setEnabled(False)
        self._scan_started = time.monotonic()
        from AIO_QC.widgets import tasks
        if getattr(self, "_task", None) is not None:
            self._task.finish()
        self._task = tasks.start(f"{self.TITLE} · {len(samples):,} files", cancel=self.stop_scan)
        self.worker = ScanWorker(samples, self.scan_sample, self.finalize, self.ctx)
        self.worker.rows_ready.connect(self._append_rows)
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_completed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    # ------------------------------------------------------------------ marks
    # Visual views (cards, galleries, charts) share the table's tick-boxes as their "marked" state, so every view
    # of a tool always agrees. Call init_marks() in post_init() to enable.
    def init_marks(self):
        self._marked: set[int] = set()
        self._mark_listeners: list = []
        self.table.itemChanged.connect(self._marks_item_changed)

    def on_marks_changed(self, fn):
        self._mark_listeners.append(fn)

    def _marks_item_changed(self, item):
        if item.column() != 0:
            return
        row = self._row_at(item.row())
        if row is None:
            return
        (self._marked.add if item.checkState() == Qt.CheckState.Checked else self._marked.discard)(id(row))
        for fn in self._mark_listeners:
            fn()

    def is_marked(self, row) -> bool:
        return id(row) in getattr(self, "_marked", ())

    def set_marked(self, row, on: bool):
        item = getattr(self, "_chk_items", {}).get(id(row))       # O(1): no walk over the table
        try:
            if item is not None:
                item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
                return
        except RuntimeError:                                     # the row was removed (Qt deleted the item)
            self._chk_items.pop(id(row), None)
        for r in range(self.table.rowCount()):                   # fallback (should not be needed)
            if self._row_at(r) is row:
                self.table.item(r, 0).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
                return

    def prune_marks(self):
        """Python may reuse the id of a removed row: keep only live rows."""
        if hasattr(self, "_marked"):
            self._marked &= {id(r) for r in self.rows if r is not None}

    # ------------------------------------------------------------------ review marks (any tool)
    def _flag_note(self, rows) -> dict:
        """key -> note: which tool flagged the file and why (first result column, e.g. the issue)."""
        why: dict[str, set] = {}
        for r in rows:
            why.setdefault(r.key, set()).add(str(r.cells[0]) if r.cells else "")
        return {k: f"{self.TITLE}: " + ", ".join(sorted(x for x in v if x))[:200] for k, v in why.items()}

    def flag_files(self, rows=None, note: str | None = None):
        rows = rows if rows is not None else self.target_rows()
        if not rows:                                       # nothing ticked: every case found (in the filter)
            rows = [r for r in self.visible_rows() if r.key in self.session.by_key]
            if not rows:
                QMessageBox.information(self, self.TITLE, "Nothing found to flag.")
                return
            n = len({r.key for r in rows})
            if QMessageBox.question(self, self.TITLE, f"No rows are ticked. Flag all {n:,} file(s) found"
                                    + (" (the current filter)" if self.FILTER_COLUMN is not None and
                                       self.filter_box.currentData() is not None else "") + "?") \
                    != QMessageBox.StandardButton.Yes:
                return
        notes = self._flag_note(rows)
        for k, n in notes.items():
            self.session.set_review([k], REVIEW_FLAG, note if note is not None else n)
        self.app.statusBar().showMessage(f"⚑ Flagged {len(notes)} file(s) from {self.TITLE}", 4000)

    def _selected_rows(self):
        seen, out = set(), []
        for idx in self.table.selectionModel().selectedRows():
            row = self._row_at(idx.row())
            if row is not None and id(row) not in seen:
                seen.add(id(row))
                out.append(row)
        return out

    def _review_menu(self, pos):
        from PyQt6.QtWidgets import QInputDialog, QMenu
        idx = self.table.indexAt(pos)
        if idx.isValid() and not self.table.selectionModel().isRowSelected(idx.row(), idx.parent()):
            self.table.selectRow(idx.row())
        rows = self._selected_rows()
        if not rows:
            return
        keys = list(dict.fromkeys(r.key for r in rows))
        m = QMenu(self)
        n = f"{len(keys)} file(s)"
        m.addAction(newIcon("review-flag"), f"Flag {n}", lambda: self.flag_files(rows))

        def with_note():
            text, ok = QInputDialog.getText(self, "Flag with note", f"Note for {n}:",
                                            text=next(iter(self._flag_note(rows).values()), ""))
            if ok:
                self.flag_files(rows, note=text)
        m.addAction(newIcon("review-note"), f"Flag {n} with note…", with_note)
        m.addAction(newIcon("review-ok"), f"Mark {n} OK", lambda: self.session.set_review(keys, REVIEW_OK))
        m.addAction(newIcon("review-clear"), "Clear review mark", lambda: self.session.set_review(keys, None))
        m.addSeparator()
        m.addAction(newIcon("eye"), "Open", lambda: self.on_row_activated(rows[0]))
        m.exec(self.table.viewport().mapToGlobal(pos))

    def _review_icon(self, key):
        st = self.session.review.status(key)
        return newIcon({REVIEW_FLAG: "review-flag", REVIEW_OK: "review-ok", "fixed": "review-fixed"}.get(st, "")) \
            if st else QIcon()

    def _on_review_changed(self, keys):
        """Badges on the File column follow review marks set anywhere (keyboard, other tools, galleries)."""
        ks = set(keys)
        if not ks:
            return
        items = getattr(self, "_file_items", {})
        for k in ks:                                             # O(changed files), not O(table rows)
            alive = []
            for it in items.get(k, ()):
                try:
                    it.setIcon(self._review_icon(k))
                    alive.append(it)
                except RuntimeError:                             # row removed meanwhile
                    pass
            if k in items:
                items[k] = alive
        for fn in getattr(self, "_mark_listeners", []):
            fn()                                   # visual views repaint their flag badges

    def showEvent(self, e):
        super().showEvent(e)
        guides = getattr(self.session, "guides", None)
        if guides is not None:
            guides.reload()
            self.guide_card.setVisible(self.guide_card.set_scope(guides, f"tool:{self.tool_key_name()}"))

    def samples_to_scan(self):
        return list(self.session.samples)

    def stop_scan(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.stop_btn.setEnabled(False)

    def _on_progress(self, i, total):
        if getattr(self, "_task", None) is not None:
            self._task.update(i / total if total else None)
        self.progress.setMaximum(total)
        self.progress.setValue(i)
        self.status.setText(f"{len(self.rows):,} found")

    def _on_completed(self, final_rows, errors, cancelled):
        if getattr(self, "_task", None) is not None:
            self._task.finish()
            self._task = None
        cols = self.columns_for(self.ctx)
        if cols is not None:
            self._set_headers(cols)
        self._append_rows(final_rows)
        self.errors = errors
        self.worker = None
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Rescan")
        self.stop_btn.setEnabled(False)
        self.params_box.setEnabled(True)
        took = time.monotonic() - self._scan_started
        state = "Stopped" if cancelled else "Done"
        self.status.setText(f"{state} in {took:.1f}s · {len(self.rows):,} found"
                            + (f" · {len(errors)} error(s)" if errors else ""))
        self._rebuild_filter()
        if self.table.rowCount() < 5000:   # resizeColumnsToContents is O(rows)
            self.table.resizeColumnsToContents()
        if self.FILE_COLUMN:
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.on_scan_complete(self.ctx)
        self.summary.setText(self.summarize())
        if not cancelled:
            self.publish_issues(getattr(self, "_scan_scope", None), self.rows)
            guide = getattr(self.app, "check_guide", None)
            if guide is not None:                       # recommended checks: done → suggest the next one
                guide.record(self.tool_key_name(), len(self.rows))
        if errors:
            self.app.log_errors([f"[{self.TITLE}] {e}" for e in errors])

    # ------------------------------------------------------------------ Issues Inbox
    def issue_of(self, row):
        """ResultRow -> Issue (None = not an issue). Severity from the row's tone; text from its first columns."""
        from AIO_QC.core.issues import Issue
        sev = {"error": "error", "warning": "warn", "muted": "info"}.get(row.tone, "warn")
        return Issue(self.tool_key_name(), sev, self.issue_text(row), row.shape_index)

    def issue_text(self, row) -> str:
        cells = [str(c) for c in row.cells if c not in (None, "")][:2]
        return " · ".join(cells) or self.TITLE

    def tool_key_name(self) -> str:
        return getattr(self, "tool_key", None) or type(self).__name__

    def publish_issues(self, scope, rows):
        publish = self.REVIEW_ACTIONS if self.PUBLISH_ISSUES is None else self.PUBLISH_ISSUES
        store = getattr(self.session, "issues", None)
        if not publish or store is None:
            return
        found = {}
        for row in rows:
            if row is None:
                continue
            it = self.issue_of(row)
            if it is not None:
                found.setdefault(row.key, []).append(it)
        store.publish(self.tool_key_name(), self.TITLE, scope, found)
        self.session.issuesChanged.emit()

    def closeEvent(self, event):
        self.stop_scan()
        if self.worker is not None and not self.worker.wait(3000):
            logger.warning(f"[{self.TITLE}] the scan did not stop within 3 s")
        try:
            self.session.samplesRemoved.disconnect(self._on_samples_removed)
        except TypeError:
            pass
        self.app.forget_dialog(self)
        super().closeEvent(event)

    # ================================================================== table
    def _set_headers(self, cols):
        self.columns = list(cols)
        headers = ["✓", "File"] + self.columns
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setColumnHidden(0, not self.CHECKABLE)
        self.table.setColumnHidden(1, not self.FILE_COLUMN)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(True)

    @staticmethod
    def _cell(value) -> QTableWidgetItem:
        item = QTableWidgetItem()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            item.setData(Qt.ItemDataRole.DisplayRole, value)   # numeric sort
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        else:
            item.setText("" if value is None else str(value))
        return item

    def _append_rows(self, rows: list[ResultRow]):
        try:
            self._append_rows_impl(rows)
        finally:                                          # never leave the table frozen / unsortable
            self.table.setUpdatesEnabled(True)
            self.table.setSortingEnabled(True)

    def _append_rows_impl(self, rows: list[ResultRow]):
        if not rows:
            return
        self.table.setSortingEnabled(False)
        self.table.setUpdatesEnabled(False)               # one repaint for the batch, not one per cell
        start = self.table.rowCount()
        if start == 0:                                    # a fresh table: forget items of the previous one
            self._file_items, self._chk_items = {}, {}
        self.table.setRowCount(start + len(rows))
        base_id = len(self.rows)
        self.rows.extend(rows)
        for n, row in enumerate(rows):
            r = start + n
            chk = QTableWidgetItem()
            if self.CHECKABLE:
                chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                chk.setCheckState(Qt.CheckState.Unchecked)
            chk.setData(ROW_ID, base_id + n)
            self.table.setItem(r, 0, chk)
            self._chk_items[id(row)] = chk
            sample = self.session.by_key.get(row.key)
            f = QTableWidgetItem(sample.display if sample else (row.key.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]))
            f.setToolTip(row.key)
            if self.session.review.status(row.key):
                f.setIcon(self._review_icon(row.key))
            self.table.setItem(r, 1, f)
            self._file_items.setdefault(row.key, []).append(f)
            color = TONE(row.tone) if row.tone else None
            bg = QColor(row.bg) if row.bg else None
            if bg is not None:
                chk.setBackground(bg)
                f.setBackground(bg)
            for c, value in enumerate(row.cells):
                item = self._cell(value)
                if color is not None:
                    item.setForeground(color)
                if bg is not None:
                    item.setBackground(bg)
                if c == 0 and row.fg:
                    item.setForeground(QColor(row.fg))
                if c == 0 and row.dot:
                    item.setIcon(get_color_icon(QColor(row.dot)))
                if row.emphasis:
                    bf = QFont(self.table.font())
                    bf.setBold(True)
                    item.setFont(bf)
                    item.setBackground(emphasis_bg(self.table))
                if row.bars and c in row.bars:
                    item.setData(BAR_ROLE, row.bars[c])
                self.table.setItem(r, 2 + c, item)
        self.table.setUpdatesEnabled(True)
        self.table.setSortingEnabled(True)
        self.status.setText(f"{len(self.rows):,} found")

    def _row_at(self, table_row: int) -> ResultRow | None:
        item = self.table.item(table_row, 0)
        if item is None:
            return None
        rid = item.data(ROW_ID)
        return self.rows[rid] if rid is not None and rid < len(self.rows) and self.rows[rid] is not None else None

    def visible_rows(self) -> list[ResultRow]:
        out = []
        for r in range(self.table.rowCount()):
            if not self.table.isRowHidden(r):
                row = self._row_at(r)
                if row is not None:
                    out.append(row)
        return out

    def target_rows(self) -> list[ResultRow]:
        """Checked rows; if none are checked, the highlighted (selected) rows."""
        checked = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if not self.table.isRowHidden(r) and it is not None and it.checkState() == Qt.CheckState.Checked:
                row = self._row_at(r)
                if row is not None:
                    checked.append(row)
        if checked:
            return checked
        rows = sorted({i.row() for i in self.table.selectedItems()})
        return [x for x in (self._row_at(r) for r in rows) if x is not None]

    def _toggle_all(self, on: bool):
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        self.table.setUpdatesEnabled(False)
        self.table.setSortingEnabled(False)
        for r in range(self.table.rowCount()):
            if not self.table.isRowHidden(r):
                self.table.item(r, 0).setCheckState(state)
        self.table.setSortingEnabled(True)
        self.table.setUpdatesEnabled(True)

    def _rebuild_filter(self):
        if self.FILTER_COLUMN is None:
            return
        current = self.filter_box.currentData()
        values = sorted({str(r.cells[self.FILTER_COLUMN]) for r in self.rows if r is not None})
        self.filter_box.blockSignals(True)
        self.filter_box.clear()
        self.filter_box.addItem(f"All ({len([r for r in self.rows if r])})", None)
        counts = defaultdict(int)
        for r in self.rows:
            if r is not None:
                counts[str(r.cells[self.FILTER_COLUMN])] += 1
        for v in values:
            self.filter_box.addItem(f"{v} ({counts[v]})", v)
        idx = self.filter_box.findData(current)
        self.filter_box.setCurrentIndex(max(idx, 0))
        self.filter_box.blockSignals(False)
        self._apply_filter()

    def _apply_filter(self):
        want = self.filter_box.currentData() if self.FILTER_COLUMN is not None else None
        for r in range(self.table.rowCount()):
            row = self._row_at(r)
            hide = row is None or (want is not None and str(row.cells[self.FILTER_COLUMN]) != want)
            self.table.setRowHidden(r, hide)

    # ====================================================== after modification
    def remove_rows(self, predicate):
        self.table.setSortingEnabled(False)
        for r in range(self.table.rowCount() - 1, -1, -1):
            row = self._row_at(r)
            if row is None or predicate(row):
                it = self.table.item(r, 0)
                if it is not None and it.data(ROW_ID) is not None and it.data(ROW_ID) < len(self.rows):
                    self.rows[it.data(ROW_ID)] = None
                self.table.removeRow(r)
        self.table.setSortingEnabled(True)

    def after_edit(self, keys):
        """Drop rows of edited files and rescan just those files (shape indices shift)."""
        ks = set(keys)
        self.remove_rows(lambda row: row.key in ks)
        if self.RESCAN_AFTER_EDIT:
            new = []
            for k in ks:
                s = self.session.by_key.get(k)
                if s is not None:
                    try:
                        new.extend(self.rescan_sample(self.ctx, s))
                    except Exception as exc:
                        self.errors.append(f"{s.name}: {exc}")
            self._append_rows(new)
            self.publish_issues(ks, [r for r in self.rows if r is not None and r.key in ks])
        self._compact()
        self._rebuild_filter()
        self.summary.setText(self.summarize())

    def _compact(self):
        """Drop tombstones so row ids stay dense."""
        if all(r is not None for r in self.rows):
            return
        remap, live = {}, []
        for i, r in enumerate(self.rows):
            if r is not None:
                remap[i] = len(live)
                live.append(r)
        self.rows = live
        self.table.setSortingEnabled(False)
        for t in range(self.table.rowCount()):
            it = self.table.item(t, 0)
            it.setData(ROW_ID, remap.get(it.data(ROW_ID)))
        self.table.setSortingEnabled(True)

    def _on_samples_removed(self, keys):
        ks = set(keys)
        self.remove_rows(lambda row: row.key in ks)
        self._compact()
        self.summary.setText(self.summarize())

    def report(self, title: str, lines: list[str], errors: list[str]):
        msg = "\n".join(lines)
        if errors:
            msg += f"\n\n{len(errors)} problem(s):\n" + "\n".join(errors[:15])
            if len(errors) > 15:
                msg += "\n…see Logs for the rest"
            self.app.log_errors(errors)
        (QMessageBox.warning if errors else QMessageBox.information)(self, title, msg)

    def confirm(self, text: str) -> bool:
        return QMessageBox.question(self, self.TITLE, text) == QMessageBox.StandardButton.Yes

    # ================================================================ generic
    def show_in_file_list(self):
        keys = [r.key for r in self.visible_rows() if r.key in self.session.by_key]
        if not keys:
            QMessageBox.information(self, self.TITLE, "No files to show.")
            return
        self.app.apply_file_filter(dict.fromkeys(keys), self.TITLE)

    def export_csv(self):
        from AIO_QC.utils.save_dirs import ask_save
        path, _ = ask_save(self, "Export CSV", f"{self.TITLE.lower().replace(' ', '_')}.csv", "CSV (*.csv)",
                           kind="exports", fallback=self.session.root.parent)
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow((["path"] if self.FILE_COLUMN else []) + self.columns)
            for row in self.visible_rows():
                w.writerow(([row.key] if self.FILE_COLUMN else []) + list(row.cells))
        self.app.statusBar().showMessage(f"Exported {path}", 4000)

    # helpers for subclasses ------------------------------------------------
    def delete_shape_rows(self, rows: list[ResultRow], what: str = "shape(s)"):
        rows = [r for r in rows if r.shape_index is not None]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Check (or select) rows that refer to a shape first.")
            return
        by_key = defaultdict(set)
        for r in rows:
            by_key[r.key].add(r.shape_index)
        n = sum(len(v) for v in by_key.values())
        if not self.confirm(f"Delete {n} {what} from {len(by_key)} file(s)?\n\nFiles are rewritten in place."):
            return
        changed, errors, removed = self.session.delete_shapes(self, by_key)
        self.after_edit(by_key.keys())
        self.report("Done", [f"Deleted {removed} shape(s) in {len(changed)} file(s)."], errors)
