# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Review ▸ QC Presets — named sets of checks (each with its own settings and a limit) run on the open dataset with one
click, one after the other in the background, then a pass / fail report: HTML for people, CSV for spreadsheets
(core.qc_presets). Use one per kind of delivery ("LPR delivery", "Gate cameras"), share it as a file.

Settings are the check windows' own parameters, taken when a check is added (or later from its open window with
“Take settings from its window”) and put back in that window before every run — the preset always runs the same way,
whatever the window was left at.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from AIO_QC.core import qc_presets as QP
from AIO_QC.core.state import REVIEW_FLAG, state_dir
from AIO_QC.utils import newButton, newIcon, style as apply_style
from AIO_QC.core.diag import ignored as _ignored
from .base_scan import BaseScanDialog, readable_table

INPUTS = (QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QLineEdit)
NOT_CHECKS = {"smart_filter"}                 # a scan window, but a filter you build — not a check with a verdict


# ================================================================================================ settings of a window
def _label_of(dlg, w) -> str:
    if isinstance(w, QCheckBox):
        return w.text()
    form = getattr(dlg, "params_form", None)
    x = w
    while x is not None and form is not None:
        lab = form.labelForField(x)
        if lab is not None and hasattr(lab, "text"):
            return lab.text()
        x = x.parentWidget()
        if x is dlg:
            break
    return w.objectName()


def _inputs(dlg) -> list[tuple[str, QWidget]]:
    """(stable key, widget) of every setting of a check window: its Parameters box + PRESET_EXTRA widgets."""
    ws = []
    box = getattr(dlg, "params_box", None)
    if box is not None:
        ws += [w for w in box.findChildren(QWidget) if isinstance(w, INPUTS)]
    for name in getattr(dlg, "PRESET_EXTRA", ()):
        w = getattr(dlg, name, None)
        if isinstance(w, INPUTS) and w not in ws:
            ws.append(w)
    out, seen = [], {}
    for w in ws:
        if isinstance(w, QLineEdit) and w.parent() is not None and isinstance(w.parent(), (QSpinBox, QComboBox,
                                                                                           QDoubleSpinBox)):
            continue                                       # the editor inside a spin / combo box
        base = f"{type(w).__name__}|{_label_of(dlg, w)}"
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append((f"{base}|{n}", w))
    return out


def _value(w):
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
        return w.value()
    if isinstance(w, QComboBox):
        d = w.currentData()
        try:
            json.dumps(d)
        except (TypeError, ValueError):
            d = None
        return {"text": w.currentText(), "data": d}
    return w.text()


def capture(dlg) -> dict:
    return {k: _value(w) for k, w in _inputs(dlg)}


def apply_settings(dlg, params: dict) -> int:
    """Put a preset's settings back into the window. -> how many could not be applied (window changed since)."""
    missed = 0
    widgets = dict(_inputs(dlg))
    for k, v in (params or {}).items():
        w = widgets.get(k)
        if w is None:
            missed += 1
            continue
        try:
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QSpinBox):
                w.setValue(int(v))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(v))
            elif isinstance(w, QComboBox):
                i = w.findData(v.get("data")) if v.get("data") is not None else -1
                if i < 0:
                    i = w.findText(v.get("text", ""))
                if i >= 0:
                    w.setCurrentIndex(i)
                elif w.isEditable():
                    w.setCurrentText(v.get("text", ""))
                else:
                    missed += 1
            elif isinstance(w, QLineEdit):
                w.setText(str(v))
        except (TypeError, ValueError, AttributeError) as exc:
            _ignored(exc)
            missed += 1
    return missed


def checkable_tools():
    """QC tools that can be part of a preset: scan windows whose findings are per file."""
    from AIO_QC.app_common import QC_TOOLS
    out = []
    for t in QC_TOOLS:
        d = t.dialog
        if d is None or not isinstance(d, type) or not issubclass(d, BaseScanDialog) or t.key in NOT_CHECKS:
            continue
        publish = d.REVIEW_ACTIONS if d.PUBLISH_ISSUES is None else d.PUBLISH_ISSUES
        if publish:
            out.append(t)
    return out


def _tool(key):
    return next((t for t in checkable_tools() if t.key == key), None)


# ================================================================================================ runner
class PresetRun(QObject):
    """Runs the checks of a preset one after the other, each in its own (hidden) window."""
    progress = pyqtSignal(int, int, str)          # check number, checks, text
    finished = pyqtSignal(list)                   # [result dicts]

    def __init__(self, app, preset: dict, parent=None):
        super().__init__(parent)
        self.app = app
        self.preset = preset
        self.results: list = []
        self.i = -1
        self.dlg = None
        self.stopped = False

    def start(self):
        self._next()

    def stop(self):
        self.stopped = True
        if self.dlg is not None:
            self.dlg.stop_scan()

    def _next(self):
        self.i += 1
        checks = self.preset.get("checks") or []
        if self.stopped or self.i >= len(checks):
            self.finished.emit(self.results)
            return
        c = checks[self.i]
        t = _tool(c["tool"])
        title = c.get("title") or c["tool"]
        self.progress.emit(self.i, len(checks), title)
        session = self.app.session
        if t is None:
            self.results.append(QP.evaluate(c, [], "this check does not exist in this version of AIO QC"))
            QTimer.singleShot(0, self._next)
            return
        from AIO_QC.app_common import tool_for, tool_supported
        if not tool_supported(t, session):
            self.results.append(QP.evaluate(c, [], f"not available for {session.type.title}"))
            QTimer.singleShot(0, self._next)
            return
        t = tool_for(t, session)                    # classification: the classifier version
        try:
            dlg = t.dialog(self.app)
        except Exception as exc:
            _ignored(exc)
            self.results.append(QP.evaluate(c, [], f"{type(exc).__name__}: {exc}"))
            QTimer.singleShot(0, self._next)
            return
        dlg.tool_key = t.key
        dlg.quiet = True
        dlg.setWindowIcon(newIcon(t.icon))
        apply_settings(dlg, c.get("params") or {})
        self.dlg = dlg
        dlg.scanFinished.connect(lambda ok, c=c, d=dlg: self._done(c, d, ok))
        if not dlg.AUTO_START:
            QTimer.singleShot(0, dlg.start_scan)

    def _done(self, c, dlg, ok):
        rows = []
        err = ""
        if not ok:
            err = dlg.scan_error or ("stopped" if self.stopped else "did not finish")
        else:
            for r in dlg.rows:
                if r is None:
                    continue
                try:
                    it = dlg.issue_of(r)
                except Exception as exc:
                    _ignored(exc)
                    it = None
                if it is not None:
                    rows.append((r.key, it.severity, it.text))
        self.results.append(QP.evaluate(c, rows, err))
        self.dlg = None
        try:
            dlg.scanFinished.disconnect()
        except TypeError:
            pass
        QTimer.singleShot(0, lambda: (dlg.close(), dlg.deleteLater()))
        QTimer.singleShot(0, self._next)


# ================================================================================================ window
class QCPresetsDialog(QDialog):
    TITLE = "QC Presets"
    SUPPORTED = None

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(newIcon("tool-presets"))
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.presets = QP.load()
        self.run = None
        self.results: list = []
        self.report = None
        self._filling = False

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("<b>Presets</b>"))
        self.listw = QListWidget()
        self.listw.currentRowChanged.connect(self._select)
        ll.addWidget(self.listw, 1)
        r1 = QHBoxLayout()
        r1.addWidget(newButton("New", icon="new", slot=self.new_preset))
        r1.addWidget(newButton("Copy", icon="copy", slot=self.copy_preset))
        r1.addWidget(newButton("Rename", slot=self.rename_preset))
        r1.addWidget(newButton("Delete", icon="delete", slot=self.delete_preset))
        ll.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(newButton("Import…", icon="import", slot=self.import_preset,
                               tip="A preset file someone shared (Export…)"))
        r2.addWidget(newButton("Export…", icon="export", slot=self.export_preset,
                               tip="Save this preset as a file to share — same checks, same settings, same limits"))
        ll.addLayout(r2)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)
        self.name = QLabel("")
        f = self.name.font()
        f.setPointSizeF(f.pointSizeF() * 1.25)
        f.setBold(True)
        self.name.setFont(f)
        rl.addWidget(self.name)
        self.note = QLineEdit()
        self.note.setPlaceholderText("What this preset is for (shown in the report)")
        self.note.editingFinished.connect(self._note_changed)
        rl.addWidget(self.note)
        self.checks = QTableWidget(0, 4)
        self.checks.setHorizontalHeaderLabels(["Check", "Passes with at most … files", "Counting", "Settings"])
        self.checks.verticalHeader().setVisible(False)
        self.checks.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.checks.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.checks.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        readable_table(self.checks)
        rl.addWidget(self.checks, 1)
        r3 = QHBoxLayout()
        self.tool_box = QComboBox()
        for t in checkable_tools():
            self.tool_box.addItem(newIcon(t.icon), t.title, t.key)
        r3.addWidget(self.tool_box, 1)
        r3.addWidget(newButton("Add check", icon="new", slot=self.add_check,
                               tip="With the check window's current settings"))
        r3.addWidget(newButton("Remove", icon="delete", slot=self.remove_check))
        r3.addWidget(newButton("Take settings from its window", slot=self.take_settings,
                               tip="The selected check's window is open and set the way you want: use those settings"))
        r3.addWidget(newButton("Open with these settings", slot=self.open_check,
                               tip="Open the selected check's window with the preset's settings (to look or adjust)"))
        rl.addLayout(r3)

        r4 = QHBoxLayout()
        self.run_btn = newButton("Run preset", icon="run", slot=self.start)
        self.run_btn.setDefault(True)
        self.stop_btn = newButton("Stop", icon="stop", slot=self.stop)
        self.stop_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.status = QLabel("Choose a preset, then Run — every check runs on the whole dataset.")
        self.status.setWordWrap(True)
        apply_style(self.status, "color:@muted;")
        r4.addWidget(self.run_btn)
        r4.addWidget(self.stop_btn)
        r4.addWidget(self.progress, 1)
        rl.addLayout(r4)
        rl.addWidget(self.status)
        self.verdict = QLabel("")
        rl.addWidget(self.verdict)
        self.out = QTableWidget(0, 4)
        self.out.setHorizontalHeaderLabels(["Check", "Result", "Files", "Allowed"])
        self.out.verticalHeader().setVisible(False)
        self.out.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.out.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        readable_table(self.out)
        rl.addWidget(self.out, 1)
        r5 = QHBoxLayout()
        self.open_btn = newButton("Open report", icon="tool-report", slot=self.open_report)
        self.csv_btn = newButton("Open CSV", slot=self.open_csv)
        self.flag_btn = newButton("Flag the failing files", icon="review-flag", slot=self.flag_failing,
                                  tip="Every file found by a check that failed — with which check in the note")
        self.batch_btn = newButton("Hand-off batch…", icon="tool-handoff",
                                   slot=lambda: self.app.open_tool("handoff_export"),
                                   tip="Send the flagged files to an annotator (Export Hand-off Batch)")
        for b in (self.open_btn, self.csv_btn, self.flag_btn, self.batch_btn):
            b.setEnabled(False)
            r5.addWidget(b)
        r5.addStretch(1)
        r5.addWidget(newButton("Close", slot=self.close))
        rl.addLayout(r5)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([240, 760])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.addWidget(split)
        from AIO_QC.utils.qt_utils import fit_size
        fit_size(self, "large")
        self._fill_list(self.app.settings.value("qc_presets/last", ""))

    # ------------------------------------------------------------------ presets
    def current(self) -> dict | None:
        i = self.listw.currentRow()
        return self.presets[i] if 0 <= i < len(self.presets) else None

    def _save(self):
        try:
            QP.save(self.presets)
        except OSError as exc:
            from .errors import show_error
            show_error(self, self.TITLE, exc, app=self.app)

    def _fill_list(self, select: str = ""):
        self.listw.blockSignals(True)
        self.listw.clear()
        for p in self.presets:
            last = p.get("last") or {}
            it = QListWidgetItem(p["name"] + (f"  ·  {last['verdict']}" if last.get("verdict") else ""))
            it.setToolTip(p.get("note", "") + (f"\nLast run {last.get('when', '')} on {last.get('dataset', '')}"
                                               if last else ""))
            self.listw.addItem(it)
        self.listw.blockSignals(False)
        i = next((n for n, p in enumerate(self.presets) if p["name"] == select), 0 if self.presets else -1)
        self.listw.setCurrentRow(i)
        self._select(i)

    def _select(self, _i):
        p = self.current()
        self.name.setText(p["name"] if p else "No preset yet — New makes one")
        self.note.setText(p.get("note", "") if p else "")
        self.note.setEnabled(p is not None)
        self.run_btn.setEnabled(bool(p and p.get("checks")))
        if p:
            self.app.settings.setValue("qc_presets/last", p["name"])
        self._fill_checks()

    def _fill_checks(self):
        p = self.current()
        self._filling = True
        self.checks.setRowCount(0)
        for n, c in enumerate((p or {}).get("checks") or []):
            self.checks.insertRow(n)
            t = _tool(c["tool"])
            it = QTableWidgetItem(newIcon(t.icon) if t else newIcon("status-warning"), c.get("title") or c["tool"])
            self.checks.setItem(n, 0, it)
            sp = QSpinBox()
            sp.setRange(0, 10_000_000)
            sp.setValue(int(c.get("max_files", 0)))
            sp.valueChanged.connect(lambda v, c=c: self._set(c, "max_files", int(v)))
            self.checks.setCellWidget(n, 1, sp)
            cb = QComboBox()
            cb.addItem("every finding", "any")
            cb.addItem("errors only", "errors")
            cb.setCurrentIndex(max(0, cb.findData(c.get("count", "any"))))
            cb.currentIndexChanged.connect(lambda _i, c=c, cb=cb: self._set(c, "count", cb.currentData()))
            self.checks.setCellWidget(n, 2, cb)
            self.checks.setItem(n, 3, QTableWidgetItem(f"{len(c.get('params') or {})} setting(s)"))
        self.checks.resizeColumnsToContents()
        self.checks.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._filling = False
        self.run_btn.setEnabled(bool(p and p.get("checks")) and self.run is None)

    def _set(self, c, key, value):
        if not self._filling:
            c[key] = value
            self._save()

    def _note_changed(self):
        p = self.current()
        if p is not None and p.get("note", "") != self.note.text():
            p["note"] = self.note.text()
            self._save()

    def new_preset(self):
        name, ok = QInputDialog.getText(self, self.TITLE, "Name of the new preset:", text="Delivery check")
        if not ok or not name.strip():
            return
        p = {"name": QP.unique_name(name.strip(), self.presets), "note": "", "checks": [],
             "created": time.strftime("%Y-%m-%d %H:%M")}
        self.presets.append(p)
        self._save()
        self._fill_list(p["name"])

    def copy_preset(self):
        p = self.current()
        if p is None:
            return
        q = json.loads(json.dumps(p))
        q["name"] = QP.unique_name(p["name"] + " (copy)", self.presets)
        q.pop("last", None)
        self.presets.append(q)
        self._save()
        self._fill_list(q["name"])

    def rename_preset(self):
        p = self.current()
        if p is None:
            return
        name, ok = QInputDialog.getText(self, self.TITLE, "New name:", text=p["name"])
        if ok and name.strip() and name.strip() != p["name"]:
            p["name"] = QP.unique_name(name.strip(), [x for x in self.presets if x is not p])
            self._save()
            self._fill_list(p["name"])

    def delete_preset(self):
        p = self.current()
        if p is None:
            return
        if QMessageBox.question(self, self.TITLE, f"Delete the preset “{p['name']}”?") != QMessageBox.StandardButton.Yes:
            return
        self.presets.remove(p)
        self._save()
        self._fill_list()

    def import_preset(self):
        f, _ = QFileDialog.getOpenFileName(self, "Import a QC preset", "", "AIO QC preset (*.json)")
        if not f:
            return
        try:
            p = QP.import_preset(f)
        except (OSError, ValueError) as exc:
            from .errors import show_error
            show_error(self, self.TITLE, exc, app=self.app)
            return
        p["name"] = QP.unique_name(p["name"], self.presets)
        self.presets.append(p)
        self._save()
        self._fill_list(p["name"])

    def export_preset(self):
        p = self.current()
        if p is None:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Export the preset", p["name"] + ".json", "AIO QC preset (*.json)")
        if f:
            q = dict(p)
            q.pop("last", None)
            QP.export_preset(q, f)
            self.status.setText(f"Saved to {f} — Import it on another computer.")

    # ------------------------------------------------------------------ checks
    def _make_window(self, t):
        """The tool's own window if it is open, else a new hidden one (closed again by the caller)."""
        dlg = self.app.open_dialogs.get(t.key)
        if dlg is not None:
            return dlg, False
        dlg = t.dialog(self.app)
        dlg.tool_key = t.key
        dlg.quiet = True
        return dlg, True

    def add_check(self):
        p = self.current()
        key = self.tool_box.currentData()
        t = _tool(key)
        if p is None or t is None:
            if p is None:
                self.status.setText("Make a preset first (New).")
            return
        from AIO_QC.app_common import tool_for, tool_supported
        if not tool_supported(t, self.app.session):
            self.status.setText(f"{t.title} is not available for this kind of dataset.")
            return
        t = tool_for(t, self.app.session)
        dlg, made = self._make_window(t)
        try:
            params = capture(dlg)
        finally:
            if made:
                dlg.stop_scan()
                dlg.close()
                dlg.deleteLater()
        p.setdefault("checks", []).append(QP.new_check(t.key, t.title, params))
        self._save()
        self._fill_checks()
        self.checks.selectRow(self.checks.rowCount() - 1)

    def _selected_check(self):
        p = self.current()
        r = self.checks.currentRow()
        if p is None or not (0 <= r < len(p.get("checks") or [])):
            return None
        return p["checks"][r]

    def remove_check(self):
        p, c = self.current(), self._selected_check()
        if c is not None:
            p["checks"].remove(c)
            self._save()
            self._fill_checks()

    def take_settings(self):
        c = self._selected_check()
        if c is None:
            return
        dlg = self.app.open_dialogs.get(c["tool"])
        if dlg is None:
            self.status.setText("Open the check's window first (Open with these settings), set it up, then come back.")
            return
        c["params"] = capture(dlg)
        self._save()
        self._fill_checks()
        self.status.setText(f"Settings of {c.get('title')} taken from its window.")

    def open_check(self):
        c = self._selected_check()
        if c is None:
            return
        self.app.open_tool(c["tool"])
        dlg = self.app.open_dialogs.get(c["tool"])
        if dlg is not None:
            missed = apply_settings(dlg, c.get("params") or {})
            if missed:
                self.status.setText(f"{missed} setting(s) did not fit this version of the window — check them, then "
                                    "Take settings from its window.")

    # ------------------------------------------------------------------ run
    def start(self):
        p = self.current()
        if p is None or not p.get("checks") or self.run is not None:
            return
        self.results = []
        self.out.setRowCount(0)
        self.verdict.setText("")
        for b in (self.open_btn, self.csv_btn, self.flag_btn):
            b.setEnabled(False)
        self.progress.setRange(0, len(p["checks"]))
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        from AIO_QC.widgets import tasks
        self._task = tasks.start(f"QC preset · {p['name']}", cancel=self.stop)
        self.run = PresetRun(self.app, p, self)
        self.run.progress.connect(self._on_progress)
        self.run.finished.connect(lambda res, p=p: self._finished(p, res))
        self.run.start()

    def stop(self):
        if self.run is not None:
            self.run.stop()
            self.status.setText("Stopping after the check that is running …")

    def _on_progress(self, i, n, title):
        self.progress.setValue(i)
        self.status.setText(f"Check {i + 1} of {n}: {title} …")
        if getattr(self, "_task", None) is not None:
            self._task.update(i / max(1, n), title)

    def _finished(self, p, results):
        self.run = None
        if getattr(self, "_task", None) is not None:
            self._task.finish()
            self._task = None
        self.results = results
        self.progress.setValue(self.progress.maximum())
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        for r in results:
            n = self.out.rowCount()
            self.out.insertRow(n)
            word = ("⚠ " + r["error"]) if r["error"] else ("✓ pass" if r["passed"] else "✗ fail")
            for k, t in enumerate([r["title"], word, f"{r['files']:,}", f"≤ {r['limit']:,}"]):
                self.out.setItem(n, k, QTableWidgetItem(t))
        self.out.resizeColumnsToContents()
        self.out.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        v = QP.verdict(results)
        color = {"PASS": "@ok", "FAIL": "@error"}.get(v, "@warn")
        self.verdict.setText(f"<span style='font-size:18pt;font-weight:700'>{v}</span>  "
                             f"{sum(r['passed'] for r in results)} of {len(results)} check(s) passed")
        apply_style(self.verdict, f"color:{color};")
        s = self.app.session
        root = Path(s.root)

        def rel(k):
            try:
                return str(Path(s.by_key[k].ann_path or s.by_key[k].image_path).relative_to(root))
            except (KeyError, ValueError, TypeError):
                return str(k)
        try:
            self.report = QP.write_report(state_dir(root) / "qc_runs", p, results, str(root), rel)
        except OSError as exc:
            self.report = None
            from .errors import message
            self.status.setText(f"⚠ The report could not be written — {message(exc)}")
        else:
            self.status.setText(f"Report: {self.report[0]}")
        p["last"] = {"when": time.strftime("%Y-%m-%d %H:%M"), "verdict": v, "dataset": str(root)}
        self._save()
        cur = p["name"]
        self._fill_list(cur)
        self.open_btn.setEnabled(self.report is not None)
        self.csv_btn.setEnabled(self.report is not None)
        failing = any(not r["passed"] and r["rows"] for r in results)
        self.flag_btn.setEnabled(failing)
        self.batch_btn.setEnabled(True)

    def open_report(self):
        if self.report:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.report[0])))

    def open_csv(self):
        if self.report:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.report[1])))

    def flag_failing(self) -> int:
        s = self.app.session
        by = {}
        for r in self.results:
            if r["passed"]:
                continue
            for k, _sev, _t in r["rows"]:
                by.setdefault(k, []).append(r["title"])
        n = 0
        for titles, keys in _group(by).items():
            s.set_review(keys, REVIEW_FLAG, "QC preset: " + ", ".join(titles))
            n += len(keys)
        self.status.setText(f"⚑ {n:,} file(s) flagged — Hand-off batch… sends them to an annotator.")
        return n

    def closeEvent(self, e):
        if self.run is not None:
            self.run.stop()
        if hasattr(self.app, "forget_dialog"):
            self.app.forget_dialog(self)
        super().closeEvent(e)


def _group(by: dict) -> dict:
    """{key: [check titles]} -> {(titles…): [keys]} so files with the same reasons are flagged in one go."""
    out = {}
    for k, titles in by.items():
        out.setdefault(tuple(dict.fromkeys(titles)), []).append(k)
    return out
