# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Recommended checks: when a dataset opens, find the QC checks that matter most for it and guide the user through them
— “Start here: Validation — 3 files could not be read”, then the next one after each check finishes.

Settings ▸ Interface ▸ "Recommend checks when a dataset opens" (on by default) turns the hint, the status-bar
button and the after-check suggestions on or off; Review ▸ Recommended Checks… always works.
Done checks (and what they found) are kept per dataset in .aio_qc/guide.json.
"""
from __future__ import annotations

import json
from html import escape
import time
from collections import Counter
from pathlib import Path

from PyQt6.QtCore import QObject, QSize, Qt
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
                             QLabel, QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout)

from AIO_QC import utils
from AIO_QC.core import check_guide as CG

SETTING = "guide/enabled"
SAMPLE = 300                     # files read in the background for the relationship / text / frame signals
NO_COUNT = {"stats", "sampling", "fix_editor"}      # their "rows" are not issues


def enabled(st) -> bool:
    try:
        return st.value(SETTING, True, type=bool)
    except (TypeError, ValueError):
        return True


def sample_signals(adapter, samples) -> dict:
    """Read up to SAMPLE files spread over the dataset (background thread): relations, texts, shape types, names."""
    from AIO_QC.core import rules as RL
    n = len(samples)
    pick = samples if n <= SAMPLE else [samples[i * n // SAMPLE] for i in range(SAMPLE)]
    st, multi, texts, poly, shapes = RL.RuleStats(), 0, 0, 0, 0
    for s in pick:
        try:
            ann = adapter.load(s, light=True)
        except Exception:
            continue
        labs = {str(x.get("label", "")) for x in ann.shapes}
        multi += len(labs) >= 2
        for x in ann.shapes:
            shapes += 1
            texts += bool(str(x.get("description") or "").strip())
            poly += x.get("shape_type") == "polygon"
        try:
            st.add(ann.shapes, ann.tags, ann.width or None, ann.height or None, normalized=bool(ann.normalized))
        except Exception:
            pass
    parts = []
    for (child, parent), n_in in st.inside.items():
        tot = st.inst[child]
        if child != parent and tot >= 10 and n_in / tot >= 0.9:
            parts.append((child, parent, n_in / tot))
    parts.sort(key=lambda p: (-p[2], p[0]))
    read = max(st.images, 1)
    return dict(multi_class=multi / read, texts=texts, polygons=poly / max(shapes, 1), parts=parts,
                frame_like=CG.frame_share(Path(s.name).name for s in pick))


class _NextButton(QToolButton):
    """Status-bar button whose text shortens (down to the icon, or away) instead of widening the window."""

    def __init__(self):
        super().__init__()
        self.full = ""
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def set_full_text(self, t: str):
        self.full = t
        self.updateGeometry()
        self._elide()

    def _extra(self) -> int:
        return self.iconSize().width() + 18

    def sizeHint(self):
        h = super().sizeHint()
        return QSize(min(260, self._extra() + self.fontMetrics().horizontalAdvance(self.full)), h.height())

    def minimumSizeHint(self):           # may shrink away entirely: never makes the window wider
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def _elide(self):
        room = self.width() - self._extra()
        self.setText(self.fontMetrics().elidedText(self.full, Qt.TextElideMode.ElideRight, room) if room > 24 else "")


class CheckGuide(QObject):
    """One per Image QC window (app.check_guide)."""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self._sample: dict | None = None
        self._worker = None
        self._gen = None
        self.done: dict = {}
        self.recs: list[CG.Rec] = []
        self.btn = _NextButton()
        self.btn.setAutoRaise(True)
        self.btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        utils.set_icon(self.btn, "next")
        self.btn.setObjectName("nextCheckButton")
        self.btn.clicked.connect(lambda _c=False: self.run_next())
        self.btn.setVisible(False)
        app.statusBar().addPermanentWidget(self.btn)

    # ------------------------------------------------------------------ state
    @property
    def on(self) -> bool:
        return enabled(self.app.settings)

    def _path(self) -> Path | None:
        s = self.app.session
        if s is None:
            return None
        from AIO_QC.core.state import state_dir
        return state_dir(s.root) / "guide.json"

    def _load(self):
        p = self._path()
        try:
            self.done = dict(json.loads(p.read_text(encoding="utf-8")).get("done", {})) if p and p.is_file() else {}
        except (OSError, ValueError, AttributeError):
            self.done = {}

    def _save(self):
        p = self._path()
        if p is None:
            return
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"done": self.done, "saved": time.strftime("%Y-%m-%d %H:%M")}, indent=1),
                         encoding="utf-8")
        except OSError:
            pass

    def signals(self) -> CG.Signals:
        app, s = self.app, self.app.session
        from AIO_QC.core import rules as RL
        from AIO_QC.core.model_registry import dataset_models
        from AIO_QC.core.state import state_dir
        from AIO_QC.core.state import REVIEW_FLAG
        rv = s.review_totals()
        files = len(s.samples)
        try:
            n_rules = sum(1 for r in RL.load_rules(state_dir(s.root) / "rules.json") if r.get("enabled", True))
        except Exception:
            n_rules = 0
        try:
            models = len(dataset_models(s.root))
        except Exception:
            models = 0
        sig = CG.Signals(
            files=files, unreadable=len(s.index_errors), broken_images=len(getattr(app, "broken", {}) or {}),
            empty=sum(1 for c in s.index.values() if c is not None and not c),
            labels=Counter(s.label_totals(raw=True)), class_list=list(s.true_classes or []), splits=list(s.splits),
            models=models, rules=n_rules, flagged=rv.get(REVIEW_FLAG, 0),
            reviewed=sum(v for k, v in rv.items() if k is not None) / max(files, 1),
            classification=bool(s.is_classification))
        for k, v in (self._sample or {}).items():
            setattr(sig, k, v)
        return sig

    def refresh(self):
        """Recompute the list (cheap) and the status-bar button."""
        if self.app.session is None:
            self.recs = []
        else:
            self.recs = [r for r in CG.recommend(self.signals(), self.done) if self._tool(r.tool) is not None]
        self._update_button()

    def _tool(self, key):
        from AIO_QC.app import QC_TOOLS
        return next((t for t in QC_TOOLS if t.key == key), None)

    def title(self, key) -> str:
        t = self._tool(key)
        return t.title if t else key

    def next(self) -> CG.Rec | None:
        return CG.next_check(self.recs)

    def _update_button(self):
        nxt = self.next() if self.on and self.app.session is not None else None
        self.btn.setVisible(nxt is not None)
        if nxt is not None:
            left = sum(1 for r in self.recs if not r.done)
            self.btn.set_full_text(f"Next check: {self.title(nxt.tool)}")
            self.btn.setToolTip(f"Recommended next: {self.title(nxt.tool)} — {nxt.why}\n"
                                f"{left} recommended check(s) left · click to run · Review ▸ Recommended Checks… "
                                "for the whole list")

    # ------------------------------------------------------------------ lifecycle
    def dataset_ready(self):
        """Indexing finished: read a small sample in the background, then suggest where to start."""
        s = self.app.session
        if s is None:
            return
        self._load()
        self._sample = None
        self.refresh()
        gen = self._gen = self.app.generation
        from AIO_QC.widgets.inference_service import FuncWorker
        adapter, samples = s.adapter, list(s.samples)
        w = FuncWorker(lambda: sample_signals(adapter, samples), self)
        w.done.connect(lambda r: self._sampled(r, gen))
        w.failed.connect(lambda m: self._sampled({}, gen))
        self._worker = w
        w.start()

    def _sampled(self, r, gen):
        self._worker = None
        if gen != self.app.generation or self.app.session is None:
            return
        self._sample = r or {}
        self.refresh()
        if self.on:
            self._toast_start()

    def reset(self):
        """Dataset closed."""
        self._gen, self._sample, self.done, self.recs = None, None, {}, []
        self.btn.setVisible(False)
        toast = getattr(self.app, "toast", None)
        t = toast.find("check_guide") if toast is not None else None
        if t is not None:
            t.dismiss()

    def setting_changed(self, _v=None):
        self._update_button()
        if not self.on:
            t = self.app.toast.find("check_guide")
            if t is not None:
                t.dismiss()

    def record(self, tool: str, found: int | None = None):
        """A check finished (or, for tools without a scan, was opened): mark it done and point at the next one."""
        if self.app.session is None or self._tool(tool) is None:
            return
        first = tool not in self.done
        self.done[tool] = None if tool in NO_COUNT else found
        self._save()
        was_rec = any(r.tool == tool for r in self.recs)
        self.refresh()
        if self.on and first and was_rec:
            self._toast_next(tool, found)

    def forget(self):
        self.done = {}
        self._save()
        self.refresh()

    # ------------------------------------------------------------------ UI
    def run(self, tool):
        self.app.open_tool(tool)

    def run_next(self):
        nxt = self.next()
        if nxt is not None:
            self.run(nxt.tool)
        else:
            self.show_dialog()

    def _toast_start(self):
        nxt = self.next()
        if nxt is None:
            return
        todo = [r for r in self.recs if not r.done]
        then = ", ".join(self.title(r.tool) for r in todo[1:3])
        html = (f"<b>Start here:</b> {self.title(nxt.tool)} — {escape(nxt.why)}"
                + (f"<br><span style='color:gray'>then {escape(then)} …</span>" if then else ""))
        self.app.toast.show_message(html, "info", key="check_guide", timeout_ms=15000, icon="next",
                                    actions=[("Run", lambda: self.run(nxt.tool)),
                                             ("All recommended", self.show_dialog)])

    def _toast_next(self, tool, found):
        nxt = self.next()
        what = "" if tool in NO_COUNT or found is None else f" · {found:,} found"
        if nxt is None:
            html = f"✓ {self.title(tool)} done{what}. <b>All recommended checks are done.</b>"
            acts = [("Review flagged", lambda: self.app.toggle_flagged_filter())] if self.signals().flagged else []
        else:
            html = (f"✓ {self.title(tool)} done{what}. <b>Next:</b> {self.title(nxt.tool)} — "
                    f"{escape(nxt.why)}")
            acts = [("Run", lambda: self.run(nxt.tool)), ("All recommended", self.show_dialog)]
        self.app.toast.show_message(html, "info", key="check_guide", timeout_ms=12000, icon="next", actions=acts)

    def show_dialog(self):
        if not self.app.validate_inputs():
            return
        self.refresh()
        GuideDialog(self, self.app).exec()


class GuideDialog(QDialog):
    def __init__(self, guide: CheckGuide, parent):
        super().__init__(parent)
        self.g = guide
        self.setWindowTitle("Recommended Checks")
        self.resize(820, 460)
        lay = QVBoxLayout(self)
        intro = QLabel("The checks that matter most for this dataset, most important first — worked out from what "
                       "the dataset holds (unreadable files, label names, empty files, splits, file names, how "
                       "classes relate). Done checks move to the end with what they found.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "Check", "Why", "Status"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setWordWrap(True)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.doubleClicked.connect(lambda _i: self._run())
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.auto = QCheckBox("Recommend checks when a dataset opens")
        self.auto.setToolTip("Start-here hint, the Next check button in the status bar and a suggestion after each "
                             "check (also in Settings ▸ Interface)")
        self.auto.setChecked(guide.on)
        self.auto.toggled.connect(self._toggle)
        row.addWidget(self.auto)
        row.addStretch(1)
        forget = QPushButton("Start over")
        forget.setToolTip("Mark every check as not done")
        forget.clicked.connect(lambda: (self.g.forget(), self._fill()))
        row.addWidget(forget)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.run_btn = bb.addButton("Run", QDialogButtonBox.ButtonRole.AcceptRole)
        self.run_btn.clicked.connect(self._run)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._fill()

    def _toggle(self, v):
        self.g.app.settings.setValue(SETTING, bool(v))
        self.g.setting_changed(v)

    def _fill(self):
        recs = self.g.recs
        self.table.setRowCount(len(recs))
        n = 0
        for i, r in enumerate(recs):
            if not r.done:
                n += 1
            status = ("next" if r is self.g.next() else "") if not r.done else (
                "✓ done" + (f" · {r.found:,} found" if r.found is not None else ""))
            cells = [str(n) if not r.done else "", self.g.title(r.tool), r.why, status]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c == 1:
                    it.setData(Qt.ItemDataRole.UserRole, r.tool)
                if r.done:
                    it.setForeground(self.palette().placeholderText())
                self.table.setItem(i, c, it)
        self.table.resizeRowsToContents()
        nxt = next((i for i, r in enumerate(recs) if not r.done), 0)
        if recs:
            self.table.selectRow(nxt)

    def selected_tool(self):
        r = self.table.currentRow()
        it = self.table.item(r, 1) if r >= 0 else None
        return it.data(Qt.ItemDataRole.UserRole) if it is not None else None

    def _run(self):
        tool = self.selected_tool()
        if tool:
            self.accept()
            self.g.run(tool)
