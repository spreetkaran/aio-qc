# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Health check (Image QC ▸ Health Check, Ctrl+Shift+H, the toolbar button): the common checks in one go, one summary.

Runs, one after the other in the background, the checks that need no model and no settings — files and images that
cannot be read, images and label files without their partner, label names spelt more than one way, files without
shapes, duplicate images, the same object marked twice — and shows each one's result in plain words. Every check
keeps its full results: "Open" shows them (the same window the QC tabs open, no second scan), and "Review the worst
files first" opens the Issues Inbox, where all of them are ranked per file.
"""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget)

from AIO_QC import utils
from AIO_QC.core.diag import ignored as _ignored

# (tool key, what it answers in plain words)
CHECKS = [
    ("validation", "Can every file be read?"),
    ("pairs", "Does every image have its label file (and the other way round)?"),
    ("image_quality", "Are the images whole and readable?"),
    ("ambiguity", "Is every label name spelt one way?"),
    ("empty", "Which files have no annotations?"),
    ("image_dups", "Are there duplicate images?"),
    ("duplicates", "Is any object marked twice?"),
]
INFO_ONLY = {"empty"}            # findings to know about, not necessarily mistakes


class _Row(QFrame):
    def __init__(self, title: str, question: str, icon: str):
        super().__init__()
        self.setObjectName("hcRow")
        utils.style(self, "#hcRow{border-bottom:1px solid @line;}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 8, 4, 8)
        lay.setSpacing(12)
        ic = QLabel()
        ic.setPixmap(utils.newIcon(icon).pixmap(QSize(20, 20)))
        col = QVBoxLayout()
        col.setSpacing(1)
        self.title = QLabel(f"<b>{title}</b>")
        self.question = QLabel(question)
        utils.style(self.question, "QLabel{color:@muted;}")
        col.addWidget(self.title)
        col.addWidget(self.question)
        self.state = QLabel("Waiting")
        utils.style(self.state, "QLabel{color:@faint;}")
        self.state.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.open = QPushButton("Open")
        self.open.setEnabled(False)
        self.open.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(ic)
        lay.addLayout(col, 1)
        lay.addWidget(self.state)
        lay.addWidget(self.open)

    def set_state(self, text: str, role: str = "muted"):
        utils.style(self.state, f"QLabel{{color:@{role}; font-weight:{600 if role in ('error', 'warn', 'ok') else 400};}}")
        self.state.setText(text)


class HealthCheckDialog(QDialog):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle("Health Check")
        self.setWindowIcon(utils.newIcon("status-ok"))
        self.results: dict[str, int] = {}
        self._queue: list[str] = []
        self._running = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 16)
        lay.setSpacing(10)
        head = QLabel("Health check")
        f = head.font()
        f.setPixelSize(20)
        f.setWeight(f.Weight.DemiBold)
        head.setFont(f)
        self.sub = QLabel("The common checks, one after the other. Nothing is changed — you decide what to fix.")
        self.sub.setWordWrap(True)
        utils.style(self.sub, "QLabel{color:@muted;}")
        lay.addWidget(head)
        lay.addWidget(self.sub)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        lay.addWidget(self.bar)
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        from AIO_QC.app import QC_TOOLS
        tools = {t.key: t for t in QC_TOOLS}
        self.rows: dict[str, _Row] = {}
        session = app.session
        for key, question in CHECKS:
            t = tools.get(key)
            if t is None or t.dialog is None:
                continue
            from AIO_QC.app_common import tool_supported
            if session is not None and not tool_supported(t, session):
                continue
            if key == "ambiguity" and session is not None and session.is_yolo and not session.true_classes:
                continue                      # needs the class list first (it asks for it when opened)
            r = _Row(t.title, question, t.icon)
            r.open.clicked.connect(lambda _c=False, k=key: self._open(k))
            self.rows[key] = r
            bl.addWidget(r)
        lay.addWidget(box)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.summary)
        btns = QHBoxLayout()
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.inbox_btn = QPushButton("Review the worst files first")
        self.inbox_btn.setProperty("variant", "primary")
        self.inbox_btn.setToolTip("The Issues Inbox: every finding of every check, one row per file, worst first")
        self.inbox_btn.setEnabled(False)
        self.inbox_btn.clicked.connect(self._inbox)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        for b in (self.stop_btn, self.inbox_btn, close):
            b.setAutoDefault(False)
            b.setDefault(False)
        btns.addWidget(self.stop_btn)
        btns.addStretch(1)
        btns.addWidget(self.inbox_btn)
        btns.addWidget(close)
        lay.addLayout(btns)
        from AIO_QC.utils.qt_utils import fit_size
        fit_size(self, "medium")

    # ------------------------------------------------------------------ run
    def start(self):
        if self.app.session is None:
            return
        self._queue = list(self.rows)
        self.results.clear()
        self.bar.setRange(0, max(1, len(self._queue)))
        self.bar.setValue(0)
        self.summary.setText("")
        self.inbox_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        for r in self.rows.values():
            r.set_state("Waiting", "faint")
            r.open.setEnabled(False)
        QTimer.singleShot(0, self._next)

    def stop(self):
        self._queue.clear()
        d = self._running
        if d is not None:
            d.stop_scan()
        self.stop_btn.setEnabled(False)

    def _dialog(self, key):
        """The tool's own window (made hidden if it is not open yet), so "Open" shows these very results."""
        from AIO_QC.app import QC_TOOLS
        t = next(t for t in QC_TOOLS if t.key == key)
        dlg = self.app.open_dialogs.get(key)
        if dlg is None:
            dlg = t.dialog(self.app)
            dlg.tool_key = key
            dlg.setWindowIcon(utils.newIcon(t.icon))
            self.app.open_dialogs[key] = dlg
        return dlg

    def _next(self):
        if not self._queue:
            self._finish()
            return
        key = self._queue.pop(0)
        row = self.rows[key]
        try:
            dlg = self._dialog(key)
        except Exception as exc:
            _ignored(exc)
            row.set_state(f"Could not run: {exc}", "error")
            self.bar.setValue(self.bar.value() + 1)
            QTimer.singleShot(0, self._next)
            return
        self._running = dlg
        row.set_state("Checking…", "accent")
        dlg.start_scan()
        w = dlg.worker
        if w is None:                                    # it did not start (it said why)
            row.set_state("Skipped", "faint")
            self.bar.setValue(self.bar.value() + 1)
            QTimer.singleShot(0, self._next)
            return
        w.progress.connect(lambda i, n, r=row: r.set_state(f"Checking… {i * 100 // max(n, 1)} %", "accent"))
        w.completed.connect(lambda _rows, _err, cancelled, k=key, d=dlg: QTimer.singleShot(
            0, lambda: self._done(k, d, cancelled)))

    def _done(self, key, dlg, cancelled):
        self._running = None
        row = self.rows[key]
        files = len({r.key for r in dlg.rows if r is not None and dlg.issue_of(r) is not None}) \
            if not cancelled else 0
        self.results[key] = files
        if cancelled:
            row.set_state("Stopped", "faint")
        elif files == 0:
            row.set_state("No problems", "ok")
        else:
            row.set_state(f"{files:,} file{'s' if files != 1 else ''}", "muted" if key in INFO_ONLY else "warn")
        row.open.setEnabled(True)
        self.bar.setValue(self.bar.value() + 1)
        QTimer.singleShot(0, self._next)

    def _finish(self):
        self.stop_btn.setEnabled(False)
        problems = {k: n for k, n in self.results.items() if n and k not in INFO_ONLY}
        if not problems:
            self.summary.setText(f"<span style='color:{utils.hexc('ok')}'><b>All clear.</b></span> "
                                 "None of these checks found a problem. The other checks in the QC tabs go deeper "
                                 "(models, rules, text labels).")
        else:
            n = sum(problems.values())
            self.summary.setText(
                f"<b>{len(problems)} check{'s' if len(problems) != 1 else ''} found something</b> "
                f"({n:,} findings by file). Start with <b>Review the worst files first</b>, or open one check to "
                f"see and fix its findings.")
        self.inbox_btn.setEnabled(True)
        guide = getattr(self.app, "check_guide", None)
        if guide is not None:
            for k, n in self.results.items():
                try:
                    guide.record(k, n)
                except Exception as _exc:
                    _ignored(_exc)

    # ------------------------------------------------------------------ open
    def _open(self, key):
        self.app.open_tool(key)

    def _inbox(self):
        self.app.open_tool("inbox")

    def closeEvent(self, e):
        self.stop()
        super().closeEvent(e)


def open_health_check(app):
    if not app.validate_inputs():
        return None
    dlg = app.open_dialogs.get("health_check")
    if dlg is None:
        dlg = HealthCheckDialog(app)
        dlg.tool_key = "health_check"
        dlg.finished.connect(lambda _r: app.open_dialogs.pop("health_check", None))
        app.open_dialogs["health_check"] = dlg
        dlg.show()
        dlg.start()
    else:
        dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg
