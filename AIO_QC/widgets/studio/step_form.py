# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""The form for one recipe step, built from the step's parameter list (core/tabular/steps.py)."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from AIO_QC.core.tabular import expr as X
from AIO_QC.core.tabular import steps as S
from AIO_QC.utils import style as apply_style


class ColumnsList(QWidget):
    """Tick-list of columns (or options) with a filter box and All / None."""

    def __init__(self, items, chosen=()):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.find = QLineEdit()
        self.find.setPlaceholderText("Filter…")
        self.find.textChanged.connect(self._filter)
        top.addWidget(self.find, 1)
        for text, on in (("All", True), ("None", False)):
            b = QPushButton(text)
            b.setFixedWidth(52)
            b.clicked.connect(lambda _c=False, v=on: self._all(v))
            top.addWidget(b)
        lay.addLayout(top)
        self.list = QListWidget()
        self.list.setMinimumHeight(140)
        chosen = set(chosen or ())
        for it in items:
            w = QListWidgetItem(str(it))
            w.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            w.setCheckState(Qt.CheckState.Checked if it in chosen else Qt.CheckState.Unchecked)
            self.list.addItem(w)
        lay.addWidget(self.list)

    def _filter(self, t):
        t = t.lower()
        for i in range(self.list.count()):
            self.list.item(i).setHidden(bool(t) and t not in self.list.item(i).text().lower())

    def _all(self, on):
        for i in range(self.list.count()):
            if not self.list.item(i).isHidden():
                self.list.item(i).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def value(self):
        return [self.list.item(i).text() for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.CheckState.Checked]


class AggsEditor(QWidget):
    """Rows of (column, summary)."""

    def __init__(self, columns, aggs=()):
        super().__init__()
        self.columns = columns
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.rows = []
        add = QPushButton("+ Add a summary")
        add.clicked.connect(lambda: self.add())
        self.lay.addWidget(add)
        for c, f in (aggs or [(columns[0] if columns else "", "rows")]):
            self.add(c, f)

    def add(self, col=None, func="sum"):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        fn = QComboBox()
        fn.addItems(S.AGG_FUNCS)
        fn.setCurrentText(func)
        cb = QComboBox()
        cb.addItems(self.columns)
        if col in self.columns:
            cb.setCurrentText(col)
        rm = QPushButton("✕")
        rm.setFixedWidth(28)
        h.addWidget(fn)
        h.addWidget(QLabel("of"))
        h.addWidget(cb, 1)
        h.addWidget(rm)
        fn.currentTextChanged.connect(lambda t, c=cb: c.setEnabled(t != "rows"))
        cb.setEnabled(func != "rows")
        row = (w, cb, fn)
        rm.clicked.connect(lambda: self._remove(row))
        self.rows.append(row)
        self.lay.insertWidget(self.lay.count() - 1, w)

    def _remove(self, row):
        if row in self.rows and len(self.rows) > 1:
            self.rows.remove(row)
            row[0].deleteLater()

    def value(self):
        return [[cb.currentText(), fn.currentText()] for _w, cb, fn in self.rows]


class StepDialog(QDialog):
    """Create or edit a step. df = the data the step will receive (columns, types, formula preview)."""

    def __init__(self, parent, op_key, df, tables=(), params=None):
        super().__init__(parent)
        self.op = S.OPS[op_key]
        self.df = df
        self.columns = [str(c) for c in df.columns]
        self.setWindowTitle(self.op.title)
        self.setMinimumWidth(520)
        p = dict(params or {})
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)
        self.w = {}
        for prm in self.op.params:
            v = p.get(prm.key, prm.default)
            k = prm.kind
            if k == "column":
                w = QComboBox()
                w.addItems(self.columns)
                if v in self.columns:
                    w.setCurrentText(v)
            elif k == "columns":
                items = prm.options or self.columns
                w = ColumnsList(items, v if isinstance(v, list) else [])
            elif k == "choice":
                w = QComboBox()
                w.addItems([str(o) for o in prm.options])
                if v is not None:
                    w.setCurrentText(str(v))
            elif k == "number":
                w = QDoubleSpinBox()
                w.setRange(-1e12, 1e12)
                w.setDecimals(4)
                w.setValue(float(v if v is not None else 0))
            elif k == "int":
                w = QSpinBox()
                w.setRange(-2_000_000_000, 2_000_000_000)
                w.setValue(int(v if v is not None else 0))
            elif k == "bool":
                w = QCheckBox(prm.label)
                w.setChecked(bool(v))
            elif k == "table":
                w = QComboBox()
                w.addItems([t for t in tables])
                if v in tables:
                    w.setCurrentText(v)
            elif k == "aggs":
                w = AggsEditor(self.columns, v or None)
            elif k == "formula":
                w = QLineEdit("" if v is None else str(v))
                w.setPlaceholderText(prm.tip)
                w.setFont(_mono())
            else:
                w = QLineEdit("" if v is None else (", ".join(map(str, v)) if isinstance(v, list) else str(v)))
            if prm.tip:
                w.setToolTip(prm.tip)
            self.w[prm.key] = w
            form.addRow("" if k == "bool" else prm.label + ":", w)
        if self.op.key == "derive":
            self.preview = QLabel("")
            self.preview.setWordWrap(True)
            apply_style(self.preview, "color:@muted;")
            form.addRow("", self.preview)
            self.w["formula"].textChanged.connect(self._check_formula)
            hint = QLabel("Columns: " + ", ".join(f"`{c}`" if not c.isidentifier() else c for c in self.columns[:40])
                          + (" …" if len(self.columns) > 40 else "") + "<br>Functions: "
                          + ", ".join(sorted(k for k in X.FUNCS if k != "col")))
            hint.setWordWrap(True)
            hint.setTextFormat(Qt.TextFormat.RichText)
            apply_style(hint, "color:@faint; font-size:9pt;")
            form.addRow("", hint)
            self._check_formula()
        if self.op.key == "filter":
            self.w["cmp"].currentTextChanged.connect(self._filter_fields)
            self._filter_fields()
        self.err = QLabel("")
        apply_style(self.err, "color:@error;")
        self.err.setWordWrap(True)
        lay.addWidget(self.err)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _filter_fields(self):
        c = self.w["cmp"].currentText()
        self.w["value"].setEnabled(c not in ("is missing", "is not missing"))
        self.w["value2"].setEnabled(c == "between")
        self.w["ignore_case"].setEnabled(c in ("contains", "starts with", "ends with", "matches pattern"))
        if c == "is one of":
            self.w["value"].setPlaceholderText("a, b, c")

    def _check_formula(self):
        t = self.w["formula"].text()
        if not t.strip():
            self.preview.setText("")
            return
        try:
            head = self.df.head(5)
            v = X.evaluate(t, head)
            vals = list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else [v] * len(head)
            self.preview.setText("First rows → " + ", ".join(str(x) for x in vals[:5]))
            apply_style(self.preview, "color:@muted;")
        except Exception as exc:
            self.preview.setText(f"⚠ {exc}")
            apply_style(self.preview, "color:@error;")

    def _typed(self, col, text):
        """A filter value typed like its column: numbers for numeric columns, else text."""
        import pandas as pd
        s = str(text).strip()
        if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col].dtype) and \
                not pd.api.types.is_bool_dtype(self.df[col].dtype):
            try:
                f = float(s.replace(",", ""))
                return int(f) if f.is_integer() else f
            except ValueError:
                raise ValueError(f"“{s}” is not a number ({col} holds numbers)")
        if col in self.df.columns and pd.api.types.is_bool_dtype(self.df[col].dtype):
            return s.lower() in ("true", "1", "yes")
        return text

    def params(self) -> dict:
        p = {}
        for prm in self.op.params:
            w = self.w[prm.key]
            if isinstance(w, (ColumnsList, AggsEditor)):
                p[prm.key] = w.value()
            elif isinstance(w, QCheckBox):
                p[prm.key] = w.isChecked()
            elif isinstance(w, (QDoubleSpinBox, QSpinBox)):
                p[prm.key] = w.value()
            elif isinstance(w, QComboBox):
                p[prm.key] = w.currentText()
            else:
                p[prm.key] = w.text()
        if self.op.key == "filter" and p["cmp"] not in ("is missing", "is not missing", "contains", "starts with",
                                                        "ends with", "matches pattern"):
            if p["cmp"] == "is one of":
                p["value"] = [self._typed(p["col"], x.strip()) for x in str(p["value"]).split(",") if x.strip()]
            else:
                p["value"] = self._typed(p["col"], p["value"])
                if p["cmp"] == "between":
                    p["value2"] = self._typed(p["col"], p["value2"])
        if self.op.key == "derive":
            p["text_cols"] = X.text_columns_of(self.df)
        if self.op.key in ("replace",):
            col = p["col"]
            for k in ("old", "new"):
                try:
                    p[k] = self._typed(col, p[k])
                except ValueError:
                    pass
        return p

    def _ok(self):
        try:
            p = self.params()
        except ValueError as exc:
            self.err.setText(str(exc))
            return
        e = self.op.check(p, self.columns)
        if e:
            self.err.setText(e)
            return
        self.result_params = p
        self.accept()


def _mono():
    from PyQt6.QtGui import QFontDatabase
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
