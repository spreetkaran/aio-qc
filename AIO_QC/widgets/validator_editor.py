# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Validator library and editor for OCR Check.

Create a validator without writing code (pattern template, or learn from examples) or write validate(text) in the
code editor; test it live as you type and against the dataset's OCR reads before saving.
"""
from __future__ import annotations

from collections import Counter

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPlainTextEdit, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from AIO_QC.core import validators as VL
from AIO_QC.utils import hexc, newButton, newIcon, style as apply_style

ORIGIN_TITLE = {"built-in": "Built-in", "user": "Mine (all datasets)", "dataset": "This dataset"}


class ValidatorEditor(QDialog):
    def __init__(self, parent, root, reads_provider=None, class_names=(), normalize_opts=None):
        """reads_provider(class_name) -> list[str] of normalised OCR reads (for "Test on this dataset")."""
        super().__init__(parent)
        self.root, self.reads_provider = root, reads_provider
        self.normalize_opts = normalize_opts
        self.setWindowTitle("Validators")
        self.setWindowIcon(newIcon("tool-validation"))
        self.resize(1020, 700)
        self.current = None                     # Validator being viewed (None = new draft)
        self.saved_name: str | None = None

        self.listw = QListWidget()
        self.listw.currentItemChanged.connect(lambda cur, _p: self._select(cur))
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("<b>Library</b>"))
        ll.addWidget(self.listw, 1)
        ll.addWidget(newButton("New validator", icon="new", slot=self._new))

        # ---- create / edit
        self.name = QLineEdit()
        self.name.setPlaceholderText("Name, e.g. “Container ID” or “Meter reading”")
        self.desc = QLineEdit()
        self.desc.setPlaceholderText("What valid text looks like (shown as a tooltip)")
        self.tabs = QTabWidget()
        self.tabs.addTab(self._pattern_tab(), "Pattern")
        self.tabs.addTab(self._examples_tab(), "From examples")
        self.tabs.addTab(self._code_tab(), "Code")
        # ---- test
        self.test_in = QLineEdit()
        self.test_in.setPlaceholderText("Type text to test — e.g. KA01AB1234")
        self.test_in.textChanged.connect(lambda _t: self._live_test())
        self.test_out = QLabel("")
        self.test_out.setTextFormat(Qt.TextFormat.RichText)
        self.ds_class = QComboBox()
        for c in class_names:
            self.ds_class.addItem(c)
        self.ds_btn = newButton("Test on this dataset", icon="tool-stats", slot=self._dataset_test,
                                tip="Run the draft over this class's OCR reads (run an OCR scan first)")
        self.ds_out = QPlainTextEdit()
        self.ds_out.setReadOnly(True)
        self.ds_out.setMaximumHeight(150)
        self.ds_out.setFont(QFont("Monospace", 9))
        self.ds_class.setEnabled(bool(class_names) and reads_provider is not None)
        self.ds_btn.setEnabled(bool(class_names) and reads_provider is not None)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.addRow("Name:", self.name)
        form.addRow("Description:", self.desc)
        rl.addLayout(form)
        rl.addWidget(self.tabs, 1)
        box = QHBoxLayout()
        box.addWidget(QLabel("<b>Try it:</b>"))
        box.addWidget(self.test_in, 1)
        rl.addLayout(box)
        rl.addWidget(self.test_out)
        ds = QHBoxLayout()
        ds.addWidget(QLabel("Class:"))
        ds.addWidget(self.ds_class)
        ds.addWidget(self.ds_btn)
        ds.addStretch(1)
        rl.addLayout(ds)
        rl.addWidget(self.ds_out)
        foot = QHBoxLayout()
        self.where = QComboBox()
        self.where.addItem("Save for all my datasets", "user")
        self.where.addItem("Save for this dataset only", "dataset")
        self.save_btn = newButton("Save", icon="save", slot=self._save)
        self.copy_btn = newButton("Copy to edit", icon="copy", slot=self._copy)
        self.delete_btn = newButton("Delete", icon="delete", slot=self._delete)
        foot.addWidget(self.where)
        foot.addWidget(self.save_btn)
        foot.addWidget(self.copy_btn)
        foot.addWidget(self.delete_btn)
        foot.addStretch(1)
        foot.addWidget(newButton("Close", slot=self.accept))
        rl.addLayout(foot)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 3)
        lay = QVBoxLayout(self)
        lay.addWidget(split)
        self._debounce = QTimer(self, singleShot=True, interval=250, timeout=self._live_test)
        self._refresh_list()
        self._new()

    # ------------------------------------------------------------------ tabs
    def _pattern_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        self.template = QLineEdit()
        self.template.setPlaceholderText("e.g.  AA 00 A{1,2} 0000  |  00 BH 0000 A{1,2}")
        self.template.textChanged.connect(lambda _t: self._from_template())
        helpl = QLabel(VL.TEMPLATE_HELP)
        helpl.setWordWrap(True)
        apply_style(helpl, "color:@muted;")
        self.regex_out = QLabel("")
        self.regex_out.setWordWrap(True)
        self.regex_out.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        l.addWidget(QLabel("Template:"))
        l.addWidget(self.template)
        l.addWidget(helpl)
        l.addWidget(self.regex_out)
        l.addStretch(1)
        return w

    def _examples_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        self.examples = QPlainTextEdit()
        self.examples.setPlaceholderText("Paste correct examples, one per line:\nKA01AB1234\nMH12DE1433\n22BH1234AA")
        l.addWidget(QLabel("Correct examples (one per line) — AIO QC works out the pattern:"))
        l.addWidget(self.examples, 1)
        row = QHBoxLayout()
        row.addWidget(newButton("Learn the pattern", icon="tool-smart-filter", slot=self._learn))
        if self.reads_provider is not None:
            row.addWidget(newButton("Use confident reads from the dataset", icon="tool-stats", slot=self._examples_from_data,
                                    tip="Fill in the most common high-confidence reads of the class chosen below"))
        row.addStretch(1)
        l.addLayout(row)
        self.learn_out = QLabel("")
        self.learn_out.setWordWrap(True)
        l.addWidget(self.learn_out)
        return w

    def _code_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        self.code = QPlainTextEdit()
        self.code.setFont(QFont("Monospace", 10))
        self.code.setTabStopDistance(self.code.fontMetrics().horizontalAdvance(" ") * 4)
        self.code.textChanged.connect(lambda: self._debounce.start())
        self.code_err = QLabel("")
        self.code_err.setWordWrap(True)
        l.addWidget(self.code, 1)
        l.addWidget(self.code_err)
        return w

    # ------------------------------------------------------------------ library
    def _refresh_list(self, select: str | None = None):
        self.vs = VL.discover(self.root)
        self.listw.blockSignals(True)
        self.listw.clear()
        for origin in ("dataset", "user", "built-in"):
            items = [v for v in self.vs.values() if v.origin == origin]
            if not items:
                continue
            head = QListWidgetItem(ORIGIN_TITLE[origin])
            head.setFlags(Qt.ItemFlag.NoItemFlags)
            f = head.font()
            f.setBold(True)
            head.setFont(f)
            self.listw.addItem(head)
            for v in sorted(items, key=lambda v: v.title.lower()):
                it = QListWidgetItem(("⚠ " if v.error else "   ") + v.title)
                it.setData(Qt.ItemDataRole.UserRole, v.name)
                it.setToolTip(v.error or v.description)
                self.listw.addItem(it)
                if v.name == select:
                    self.listw.setCurrentItem(it)
        self.listw.blockSignals(False)

    def _select(self, item):
        name = item.data(Qt.ItemDataRole.UserRole) if item else None
        v = self.vs.get(name) if name else None
        if v is None:
            return
        self.current = v
        self.name.setText(v.title)
        self.desc.setText(v.description)
        self.code.setPlainText(v.path.read_text(encoding="utf-8"))
        self.tabs.setCurrentIndex(2)
        ro = v.origin == "built-in"
        for w in (self.name, self.desc):
            w.setReadOnly(ro)
        self.code.setReadOnly(ro)
        self.save_btn.setEnabled(not ro)
        self.delete_btn.setEnabled(not ro)
        self.copy_btn.setEnabled(True)
        self._live_test()

    def _new(self):
        self.current = None
        self.listw.clearSelection()
        for w in (self.name, self.desc):
            w.setReadOnly(False)
            w.clear()
        self.code.setReadOnly(False)
        self.code.setPlainText(VL.BLANK_TEMPLATE)
        self.template.clear()
        self.save_btn.setEnabled(True)
        self.delete_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.tabs.setCurrentIndex(0)

    def _copy(self):
        if self.current is None:
            return
        code = self.current.path.read_text(encoding="utf-8")
        title = f"{self.current.title} (copy)"
        code = code.replace(f"NAME = {self.current.title!r}", f"NAME = {title!r}").replace(
            f'NAME = "{self.current.title}"', f'NAME = "{title}"')
        self.current = None
        self.name.setText(title)
        for w in (self.name, self.desc):
            w.setReadOnly(False)
        self.code.setReadOnly(False)
        self.code.setPlainText(code)
        self.save_btn.setEnabled(True)
        self.delete_btn.setEnabled(False)

    # ------------------------------------------------------------------ builders
    def _from_template(self):
        t = self.template.text().strip()
        if not t:
            self.regex_out.setText("")
            return
        try:
            code = VL.template_code(self.name.text().strip() or "Pattern", t, self.desc.text().strip())
        except (ValueError, Exception) as exc:
            self.regex_out.setText(f"<span style='color:{hexc('error')}'>{exc}</span>")
            return
        self.regex_out.setText(f"<span style='color:{hexc('muted')}'>regex: {VL.template_to_regex(t)}</span>")
        self.code.setPlainText(code)

    def _learn(self):
        lines = self.examples.toPlainText().splitlines()
        try:
            tpl, notes = VL.learn_template(lines, self.normalize_opts)
        except ValueError as exc:
            self.learn_out.setText(str(exc))
            return
        self.tabs.setCurrentIndex(0)
        self.template.setText(tpl)
        n = len([x for x in lines if x.strip()])
        self.learn_out.setText(f"Learned from {n} example(s): <b>{tpl}</b>" + (f"<br>{'; '.join(notes)}" if notes else "")
                               + "<br>Check it on the Pattern tab, adjust if needed, then Save.")

    def _examples_from_data(self):
        reads = self.reads_provider(self.ds_class.currentText()) if self.reads_provider else []
        common = [t for t, _n in Counter(t for t in reads if t).most_common(40)]
        if not common:
            self.learn_out.setText("No reads yet for this class — run the OCR scan first.")
            return
        self.examples.setPlainText("\n".join(common))
        self.learn_out.setText("These are the most common confident reads — remove any wrong ones, then Learn.")

    # ------------------------------------------------------------------ testing
    def _draft(self) -> VL.Validator:
        return VL.load_code(self.code.toPlainText(), VL.safe_name(self.name.text() or "draft"))

    def _live_test(self):
        v = self._draft()
        if v.fn is None:
            self.code_err.setText(f"<span style='color:{hexc('error')}'>✗ {v.error}</span>")
            self.test_out.setText("")
            return
        self.code_err.setText(f"<span style='color:{hexc('ok')}'>✓ compiles</span>")
        raw = self.test_in.text()
        if not raw:
            self.test_out.setText("")
            return
        text = VL.normalize(raw, self.normalize_opts)
        ok, why = v(text)
        col, sym = ("ok", "✓ valid") if ok else (("error", "✗ invalid") if ok is False else ("warn", "⚠ error"))
        self.test_out.setText(f"<span style='color:{hexc(col)}'><b>{sym}</b></span> — checked “{text}”"
                              + (f" · {why}" if why else ""))

    def _dataset_test(self):
        v = self._draft()
        if v.fn is None:
            self.ds_out.setPlainText(f"Fix the code first: {v.error}")
            return
        reads = [t for t in (self.reads_provider(self.ds_class.currentText()) or []) if t]
        if not reads:
            self.ds_out.setPlainText("No reads for this class yet — run the OCR scan first.")
            return
        res = [(t, v(t)) for t in reads]
        passed = sum(1 for _t, (ok, _w) in res if ok)
        errors = sum(1 for _t, (ok, _w) in res if ok is None)
        fails = Counter(t for t, (ok, _w) in res if ok is False)
        why = {t: w for t, (ok, w) in res if ok is False}
        lines = [f"{passed:,} / {len(res):,} reads pass ({passed / len(res) * 100:.1f} %)"
                 + (f" · {errors:,} validator errors" if errors else ""), "", "Most common failing reads:"]
        lines += [f"  {n:>5} × {t}" + (f"   ({why[t]})" if why.get(t) else "") for t, n in fails.most_common(25)]
        self.ds_out.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------ save / delete
    def _save(self):
        title = self.name.text().strip()
        if not title:
            QMessageBox.information(self, "Validators", "Give the validator a name.")
            return
        code = self.code.toPlainText()
        if VL.safe_name(title) in self.vs and self.vs[VL.safe_name(title)].origin == "built-in":
            title = f"{title} custom"
        desc = self.desc.text().strip()
        lines = [l for l in code.splitlines() if not l.startswith(("NAME =", "DESCRIPTION ="))]
        code = f"NAME = {title!r}\nDESCRIPTION = {desc!r}\n" + "\n".join(lines).lstrip("\n") + "\n"
        v = VL.load_code(code, VL.safe_name(title))
        if v.fn is None:
            QMessageBox.warning(self, "Validators", f"The code does not work yet:\n{v.error}")
            return
        path = VL.save(title, code, self.where.currentData(), self.root)
        self.saved_name = path.stem
        self._refresh_list(select=path.stem)

    def _delete(self):
        v = self.current
        if v is None or v.origin == "built-in":
            return
        if QMessageBox.question(self, "Validators", f"Delete “{v.title}”?\n{v.path}") == QMessageBox.StandardButton.Yes:
            try:
                v.path.unlink()
            except OSError as exc:
                QMessageBox.warning(self, "Validators", str(exc))
            self._refresh_list()
            self._new()
