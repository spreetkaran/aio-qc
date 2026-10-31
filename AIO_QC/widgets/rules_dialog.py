# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
🔗 Relationship Rules — how annotations relate to each other and to the image (containment, counts, co-occurrence,
overlaps, size, shape type, attributes, groups, classification). Configure once per dataset, or let AIO QC suggest
rules that already hold for nearly all of the data.
"""
from __future__ import annotations

import json
import random

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QMenu, QProgressBar, QSpinBox,
    QTableWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from AIO_QC.core import rules as RL
from AIO_QC.core.state import state_dir
from AIO_QC.utils import hexc, newButton, newIcon, style as apply_style
from .base_scan import BaseScanDialog, ResultRow
from .card_gallery import Card, CardView, Panel, Shape, box_points
from .inference_service import FuncWorker

GROUP_TITLES = {"objects": "Between objects", "image": "Per image", "object": "Per object",
                "classification": "Classification"}
SEVERITIES = [("error", "Error"), ("warn", "Warning"), ("info", "Info")]
TONE = {"error": "error", "warn": "warning", "info": "muted"}
LABEL_PARAMS = {"child", "parent", "label", "a", "b"}
LIST_PARAMS = {"labels": "classes", "types": "shape types"}
SHAPE_TYPES = ["rectangle", "polygon", "circle", "line", "linestrip", "point", "mask"]


class RuleForm(QDialog):
    """Form generated from the rule type's parameters."""

    def __init__(self, parent, rule: dict, labels: list[str]):
        super().__init__(parent)
        self.rule = dict(rule)
        title = RL.TYPES[rule["type"]][0]
        self.setWindowTitle(f"Rule — {title}")
        form = QFormLayout(self)
        self.widgets = {}
        defaults = RL.TYPES[rule["type"]][2]
        for k, dv in defaults.items():
            v = rule.get(k, dv)
            if k in LABEL_PARAMS:
                w = QComboBox()
                w.setEditable(True)
                if k == "label" or (rule["type"] in ("count", "size", "border") and k == "label"):
                    w.addItem("*")
                w.addItems(labels)
                w.setCurrentText(str(v) if v else (labels[0] if labels else ""))
            elif k in LIST_PARAMS:
                w = QListWidget()
                w.setMaximumHeight(150)
                for opt in (SHAPE_TYPES if k == "types" else labels):
                    it = QListWidgetItem(opt)
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    it.setCheckState(Qt.CheckState.Checked if opt in (v or []) else Qt.CheckState.Unchecked)
                    w.addItem(it)
            elif k == "value":
                w = QComboBox()
                w.addItems(["any", "true", "false"])
                w.setCurrentText(str(v))
            elif k == "flag":
                w = QComboBox()
                w.setEditable(True)
                w.setCurrentText(str(v))
            elif isinstance(dv, float):
                w = QDoubleSpinBox()
                w.setDecimals(3)
                w.setRange(0, 1e6)
                w.setSingleStep(0.05 if dv <= 1 else 1)
                w.setValue(float(v))
            else:
                w = QSpinBox()
                w.setRange(0, 1_000_000)
                w.setValue(int(v))
                if k in ("max", "max_w", "max_h"):
                    w.setSpecialValueText("no limit")
            self.widgets[k] = w
            form.addRow(self._title(k) + ":", w)
        self.sev = QComboBox()
        for v, t in SEVERITIES:
            self.sev.addItem(t, v)
        self.sev.setCurrentIndex(max(0, self.sev.findData(rule.get("severity", "error"))))
        form.addRow("Severity:", self.sev)
        self.preview = QLabel("")
        self.preview.setWordWrap(True)
        apply_style(self.preview, "color:@muted;")
        form.addRow(self.preview)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        for w in self.widgets.values():
            sig = getattr(w, "currentTextChanged", None) or getattr(w, "valueChanged", None) or getattr(w, "itemChanged", None)
            if sig is not None:
                sig.connect(lambda *_: self._update_preview())
        self._update_preview()

    @staticmethod
    def _title(k):
        return {"child": "Child class", "parent": "Parent class", "label": "Class", "a": "Class A", "b": "Class B",
                "min_frac": "Inside at least (fraction)", "max_iou": "Max overlap (IoU)", "min": "Minimum",
                "max": "Maximum", "labels": "Classes", "types": "Allowed shape types", "min_w": "Min width px",
                "min_h": "Min height px", "max_w": "Max width px", "max_h": "Max height px",
                "min_area_pct": "Min area % of image", "max_area_pct": "Max area % of image",
                "margin_px": "Border margin px", "flag": "Attribute (shape flag)", "value": "Required value"}.get(k, k)

    def value(self) -> dict:
        r = dict(self.rule)
        for k, w in self.widgets.items():
            if isinstance(w, QComboBox):
                r[k] = w.currentText().strip()
            elif isinstance(w, QListWidget):
                r[k] = [w.item(i).text() for i in range(w.count()) if w.item(i).checkState() == Qt.CheckState.Checked]
            else:
                r[k] = w.value()
        r["severity"] = self.sev.currentData()
        return r

    def _update_preview(self):
        self.preview.setText("Rule: " + RL.describe(self.value()))


class RulesDialog(BaseScanDialog):
    TITLE, ICON = "Relationship Rules", "🔗"
    HELP = ("Checks how annotations relate: a line must be inside its plate, a plate holds 1–2 lines, at most one truck "
            "per image, text boxes are rectangles, classes that never appear together … Add rules by hand or let "
            "AIO QC suggest the ones that already hold for nearly all of your data.")
    COLUMNS = ["Rule", "Problem", "Shape"]
    FILTER_COLUMN = 0
    RESCAN_AFTER_EDIT = True

    # ================================================================== rules table
    def build_params(self, form):
        self.path = state_dir(self.session.root) / "rules.json"
        self.rules = RL.load_rules(self.path)
        self.labels = sorted(self.session.label_totals())
        self.rtable = QTableWidget(0, 3)
        self.rtable.setHorizontalHeaderLabels(["Rule", "Severity", "Held by"])
        self.rtable.verticalHeader().setVisible(False)
        self.rtable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.rtable.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.rtable.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.rtable.setMaximumHeight(190)
        self.rtable.cellDoubleClicked.connect(lambda r, _c: self._edit(r))
        self.rtable.itemChanged.connect(self._toggled)
        form.addRow(self.rtable)
        row = QHBoxLayout()
        add = QToolButton()
        add.setText("Add rule ▾")
        add.setIcon(newIcon("new"))
        add.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        m = QMenu(add)
        cls = self.session.is_classification
        for group in ("objects", "image", "object", "classification"):
            if cls and group in ("objects", "object"):
                continue
            m.addSection(GROUP_TITLES[group])
            for t, (title, g, _d, _px) in RL.TYPES.items():
                if g == group:
                    m.addAction(title, lambda t=t: self._add(t))
        add.setMenu(m)
        row.addWidget(add)
        row.addWidget(newButton("Suggest from data…", icon="tool-smart-filter", slot=self._suggest,
                                tip="Find rules that already hold for nearly all of this dataset"))
        row.addWidget(newButton("Edit", icon="edit", slot=lambda: self._edit(self.rtable.currentRow())))
        row.addWidget(newButton("Delete", icon="delete", slot=self._delete))
        row.addStretch(1)
        row.addWidget(newButton("Import…", icon="open", slot=self._import))
        row.addWidget(newButton("Export…", icon="save", slot=self._export))
        w = QWidget()
        w.setLayout(row)
        form.addRow(w)
        self._fill_rules()

    def _fill_rules(self):
        self.rtable.blockSignals(True)
        self.rtable.setRowCount(len(self.rules))
        for i, r in enumerate(self.rules):
            it = QTableWidgetItem(RL.describe(r))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if r.get("enabled", True) else Qt.CheckState.Unchecked)
            it.setToolTip(RL.TYPES[r["type"]][0])
            self.rtable.setItem(i, 0, it)
            sv = QTableWidgetItem(dict(SEVERITIES).get(r.get("severity", "error"), "Error"))
            self.rtable.setItem(i, 1, sv)
            self.rtable.setItem(i, 2, QTableWidgetItem(r.get("held", "")))
        self.rtable.blockSignals(False)

    def _save(self):
        RL.save_rules(self.path, self.rules)
        self._fill_rules()

    def _toggled(self, item):
        if item.column() == 0 and 0 <= item.row() < len(self.rules):
            self.rules[item.row()]["enabled"] = item.checkState() == Qt.CheckState.Checked
            RL.save_rules(self.path, self.rules)

    def _add(self, rtype):
        f = RuleForm(self, RL.new_rule(rtype), self.labels)
        if f.exec():
            self.rules.append(f.value())
            self._save()

    def _edit(self, row):
        if not 0 <= row < len(self.rules):
            return
        f = RuleForm(self, self.rules[row], self.labels)
        if f.exec():
            self.rules[row] = f.value()
            self._save()

    def _delete(self):
        rows = sorted({i.row() for i in self.rtable.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.rules[r]
        self._save()

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import rules", "", "Rules (*.json)")
        if path:
            new = RL.load_rules(path)
            for r in new:
                r["id"] = RL.new_rule(r["type"])["id"]
            self.rules += new
            self._save()

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export rules", "rules.json", "Rules (*.json)")
        if path:
            RL.save_rules(path, self.rules)

    # ================================================================== suggestions
    def _suggest(self):
        samples = list(self.session.samples)
        if len(samples) > 20000:
            samples = random.Random(0).sample(samples, 20000)
        adapter = self.session.adapter
        dlg = QDialog(self)
        dlg.setWindowTitle("Suggested rules")
        dlg.resize(760, 520)
        lay = QVBoxLayout(dlg)
        info = QLabel(f"Reading {len(samples):,} files …")
        bar = QProgressBar()
        bar.setRange(0, 0)
        lst = QListWidget()
        lay.addWidget(info)
        lay.addWidget(bar)
        lay.addWidget(lst, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Add ticked rules")
        bb.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        existing = {json.dumps({k: v for k, v in r.items() if k not in ("id", "enabled", "severity", "held")},
                               sort_keys=True) for r in self.rules}

        def mine():
            st = RL.RuleStats()
            for s in samples:
                try:
                    ann = adapter.load(s, light=True)
                except Exception:
                    continue
                st.add(ann.shapes, ann.tags)
            return RL.suggest(st, min_support=max(5, min(20, len(samples) // 50)))

        def done(sugs):
            bar.setVisible(False)
            n = 0
            for sg in sugs:
                sig = json.dumps({k: v for k, v in sg.rule.items() if k not in ("id", "enabled", "severity", "held")},
                                 sort_keys=True)
                if sig in existing:
                    continue
                it = QListWidgetItem(f"{RL.describe(sg.rule)}\n      {sg.evidence}")
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if sg.strength >= 0.99 else Qt.CheckState.Unchecked)
                it.setData(Qt.ItemDataRole.UserRole, sg)
                lst.addItem(it)
                n += 1
            info.setText(f"{n} rule(s) hold for at least 97 % of {len(samples):,} files. Tick the ones that describe how "
                         "your data SHOULD be — the files that break them are then reported." if n else
                         "No new rules found (they may already be in your list).")
            bb.button(QDialogButtonBox.StandardButton.Ok).setEnabled(n > 0)

        w = FuncWorker(mine, dlg)
        w.done.connect(done)
        w.failed.connect(lambda m: info.setText(f"Failed: {m}"))
        w.start()
        if dlg.exec():
            for i in range(lst.count()):
                it = lst.item(i)
                if it.checkState() == Qt.CheckState.Checked:
                    sg = it.data(Qt.ItemDataRole.UserRole)
                    r = dict(sg.rule)
                    r["held"] = f"{sg.strength * 100:.1f} %"
                    self.rules.append(r)
            self._save()
        w.wait()

    # ================================================================== scan
    def post_init(self):
        self.init_marks()
        self.cards = CardView(self, n_panels=1)
        self.tabs = QTabWidget()
        self.table_host.removeWidget(self.table)
        self.tabs.addTab(self.cards, "Cards")
        self.tabs.addTab(self.table, "Table")
        self.table_host.addWidget(self.tabs)
        self.filter_box.currentIndexChanged.connect(lambda _i: self._refresh_cards())

    def make_context(self):
        active = [r for r in self.rules if r.get("enabled", True)]
        if not active:
            raise ValueError("Add (or tick) at least one rule — or use “Suggest from data…”.")
        return {"adapter": self.session.adapter, "rules": active,
                "px": any(RL.TYPES[r["type"]][3] for r in active)}

    def scan_sample(self, ctx, s):
        ann = ctx["adapter"].load(s, light=True)
        W = H = None
        if ctx["px"] or ann.normalized:
            size = (ann.width, ann.height) if ann.width and ann.height else ctx["adapter"].image_size(s, ann)
            if size:
                W, H = size
        vs = RL.check_image(ctx["rules"], ann.shapes, ann.tags, W, H, normalized=ann.normalized)
        rows = []
        for v in vs:
            title = RL.TYPES[v.rule["type"]][0]
            rows.append(ResultRow(s.key, (title, v.text, v.shapes[0] if v.shapes else ""),
                                  shape_index=v.shapes[0] if v.shapes else None,
                                  payload=(v, [ann.shapes[i] for i in v.shapes if i < len(ann.shapes)], ann.normalized),
                                  tone=TONE.get(v.rule.get("severity", "error"))))
        return rows

    def issue_text(self, row):
        return f"{row.cells[0]}: {row.cells[1]}"

    def _refresh_cards(self):
        if not hasattr(self, "cards"):
            return
        self.prune_marks()
        cards = []
        for row in self.visible_rows()[:5000]:
            v, shapes, norm = row.payload
            s = self.session.by_key.get(row.key)
            if shapes:
                pts = [p for sh in shapes for p in (sh.get("points") or [])]
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                focus = (min(xs), min(ys), max(xs), max(ys))
                drawn = []
                for j, sh in enumerate(shapes):
                    col = self.app.label_colors.get(sh.get("label"))
                    sp = sh.get("points") or []
                    if len(sp) >= 2:
                        bx = (min(p[0] for p in sp), min(p[1] for p in sp), max(p[0] for p in sp), max(p[1] for p in sp))
                        drawn.append(Shape(box_points(*bx), (col.name() if col else hexc("info")), dashed=j > 0))
                panel = Panel(row.key, focus, drawn, normalized=norm)
            else:
                panel = Panel(row.key, (0, 0, 1, 1), [], normalized=True)
            sev = v.rule.get("severity", "error")
            cards.append(Card(row, [panel], [(RL.TYPES[v.rule["type"]][0], {"warn": "warn", "info": "muted"}.get(sev, "error"))],
                              [v.text, RL.describe(v.rule), s.display if s else row.key]))
        self.cards.set_cards(cards)
        self.tabs.setTabText(0, f"Cards ({len(cards):,})")

    def summarize(self):
        self._refresh_cards()
        from collections import Counter
        c = Counter(r.cells[0] for r in self.rows if r)
        return " · ".join(f"{n:,} {k}" for k, n in c.most_common()) or "✅ Every file follows every rule"
