# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
🔗 Relationship Rules — how annotations relate to each other and to the image (parent / part, counts, co-occurrence,
overlaps, size, shape type, attributes, groups, classification). Configure once per dataset, or let AIO QC suggest
rules that already hold for nearly all of the data.

Results: one row per broken rule in a file; the filter lists every rule by its name (or description), so one rule's
files are one click away; a rule's guideline (Labels ▸ Guidelines ▸ Rule · …) shows above them. Files that cannot
break any rule (none of its classes present, from the label index) are not even opened.
"""
from __future__ import annotations

import random

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QLineEdit, QMessageBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
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
PERCENT_PARAMS = {"min_frac": "Inside at least", "max_rel": "Not larger than", "min_rel": "Not smaller than"}
GEOMETRY = {"part", "touch", "tilt", "shape_type", "overlap"}     # need the image size (fixes are stored 0..1)
PROBLEM, FIX = "#ffd400", "#00e676"      # card colours: the shape at fault (never a class colour), the fix
LIST_PARAMS = {"labels": "classes", "types": "shape types"}
SHAPE_TYPES = ["rectangle", "polygon", "circle", "line", "linestrip", "point", "mask"]


class RuleForm(QDialog):
    """Form generated from the rule type's parameters — with choices detected from the dataset: under each field
    the value the data suggests (and why), one click to use it; class fields list the likely classes."""

    def __init__(self, parent, rule: dict, labels: list[str], profile=None):
        super().__init__(parent)
        self.rule = dict(rule)
        self.profile = profile
        self.hints: dict[str, QLabel] = {}
        self._recs: dict = {}
        title = RL.TYPES[rule["type"]][0]
        self.setWindowTitle(f"Rule — {title}")
        form = QFormLayout(self)
        self.widgets = {}
        self.data_note = QLabel("")
        self.data_note.setWordWrap(True)
        apply_style(self.data_note, "color:@muted;")
        self.use_all = newButton("Use all suggestions", icon="tool-smart-filter", slot=self.use_all_suggestions,
                                 tip="Fill every field the dataset suggests a value for")
        top = QHBoxLayout()
        top.addWidget(self.data_note, 1)
        top.addWidget(self.use_all)
        form.addRow(top)
        self.name = QLineEdit(str(rule.get("name", "")))
        self.name.setPlaceholderText("optional short name, e.g. “Wheels on their car”")
        self.name.textChanged.connect(lambda *_: self._update_preview())
        form.addRow("Name:", self.name)
        defaults = RL.TYPES[rule["type"]][2]
        for k, dv in defaults.items():
            v = rule.get(k, dv)
            if isinstance(dv, bool):
                w = QCheckBox(self._title(k))
                w.setChecked(bool(v))
                self.widgets[k] = w
                self._add_row(form, "", w, k)
                continue
            if k in PERCENT_PARAMS:
                w = QDoubleSpinBox()
                w.setDecimals(0)
                w.setRange(0, 1000)
                w.setSingleStep(5)
                w.setSuffix(" %")
                w.setValue(float(v) * 100)
                if k in ("max_rel", "min_rel") or (k == "min_frac" and rule["type"] == "touch"):
                    w.setSpecialValueText("off" if k != "min_frac" else "any overlap")
                self.widgets[k] = w
                self._add_row(form, self._title(k) + ":", w, k)
                continue
            if k in LABEL_PARAMS:
                w = QComboBox()
                w.setEditable(True)
                if k in RL.WILDCARD.get(rule["type"], ()):
                    w.addItem("*")                                 # any class
                    w.setItemData(0, "* = any class", Qt.ItemDataRole.ToolTipRole)
                    w.setToolTip("A class, or * for any class")
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
                for fl in (sorted({f for c in profile.flags.values() for f in c}) if profile is not None else []):
                    w.addItem(fl)                             # attributes that exist in the data
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
                if rule["type"] == "part" and k == "min":
                    w.setSpecialValueText("—")
            if rule["type"] == "part" and k in ("apart_iou", "max_angle"):
                w.setSpecialValueText("off")
            self.widgets[k] = w
            label = {"min": "Parts per parent: at least", "max": "Parts per parent: at most"}.get(k) \
                if rule["type"] == "part" else None
            self._add_row(form, (label or self._title(k)) + ":", w, k)
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
            sig = (getattr(w, "currentTextChanged", None) or getattr(w, "valueChanged", None)
                   or getattr(w, "itemChanged", None) or getattr(w, "toggled", None))
            if sig is not None:
                sig.connect(lambda *_: self._update_preview())
        for k, w in self.widgets.items():                     # a new class choice -> new suggestions
            if k in LABEL_PARAMS:
                w.currentTextChanged.connect(lambda *_: self._refresh_hints())
        self._update_preview()
        self.set_profile(profile)

    # ---- choices from the dataset
    def _add_row(self, form, label, w, k):
        form.addRow(label, w)
        hl = QLabel("")
        hl.setWordWrap(True)
        hl.setTextFormat(Qt.TextFormat.RichText)
        hl.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
        apply_style(hl, "color:@muted; font-size:9pt;")
        hl.linkActivated.connect(lambda href, k=k: self._use(k, href))
        hl.setVisible(False)
        form.addRow("", hl)
        self.hints[k] = hl

    def set_profile(self, profile):
        """The dataset profile (RL.RuleStats) — None while it is still being read."""
        self.profile = profile
        if profile is None:
            self.data_note.setText("Reading the dataset to suggest choices …")
        else:
            self.data_note.setText(f"Suggestions from {profile.images:,} files of this dataset — limits are set so "
                                   "~99.5 % of the data passes; the rest is what the rule finds.")
        self._refresh_hints()

    @staticmethod
    def _fmt(k, v) -> str:
        if isinstance(v, bool):
            return "on" if v else "off"
        if isinstance(v, list):
            return ", ".join(map(str, v)) or "none"
        if k in PERCENT_PARAMS:
            return f"{float(v) * 100:.0f} %"
        return f"{v:g}" if isinstance(v, float) else str(v)

    def _refresh_hints(self):
        import html
        if not self.hints:
            return
        params = self.value() if hasattr(self, "sev") else dict(self.rule)
        rtype = self.rule["type"]
        self._recs = RL.recommend(self.profile, rtype, params) if self.profile is not None else {}
        note = self._recs.pop("_note", None)                   # the classes chosen do not go together in the data
        if self.profile is not None:
            self.data_note.setText(note.text if note else
                                   f"Suggestions from {self.profile.images:,} files of this dataset — limits are set "
                                   "so ~99.5 % of the data passes; the rest is what the rule finds.")
            apply_style(self.data_note, "color:@warn; font-weight:600;" if note else "color:@muted;")
        for k, hl in self.hints.items():
            text = ""
            if self.profile is not None and k in LABEL_PARAMS:
                ranked = [(l, why) for l, why in RL.rank(self.profile, rtype, k, params)
                          if l != params.get(k)]
                if ranked:
                    text = "Likely: " + " · ".join(f"<a href='{html.escape(l)}'>{html.escape(l)}</a> "
                                                   f"<span>({html.escape(why)})</span>" for l, why in ranked[:4])
            elif k in self._recs:
                hnt = self._recs[k]
                cur = params.get(k)
                same = (cur == hnt.value) or (isinstance(cur, float) and isinstance(hnt.value, (int, float))
                                              and abs(cur - float(hnt.value)) < 1e-9)
                text = (f"Data: {html.escape(hnt.text)}" +
                        ("" if same else f" — <a href='use'>use {html.escape(self._fmt(k, hnt.value))}</a>"))
            hl.setText(text)
            hl.setVisible(bool(text))
        self.use_all.setEnabled(any(k not in LABEL_PARAMS for k in self._recs))

    def _use(self, k, href):
        if k in LABEL_PARAMS:
            self._set(k, href)
        elif k in self._recs:
            self._set(k, self._recs[k].value)
        self._refresh_hints()

    def use_all_suggestions(self):
        for k, hnt in list(self._recs.items()):
            if k in self.widgets and k not in LABEL_PARAMS:
                self._set(k, hnt.value)
        self._refresh_hints()

    def _set(self, k, v):
        w = self.widgets.get(k)
        if w is None:
            return
        if isinstance(w, QCheckBox):
            w.setChecked(bool(v))
        elif k in PERCENT_PARAMS:
            w.setValue(float(v) * 100)
        elif isinstance(w, QListWidget):
            want = set(v or [])
            for i in range(w.count()):
                w.item(i).setCheckState(Qt.CheckState.Checked if w.item(i).text() in want else Qt.CheckState.Unchecked)
        elif isinstance(w, QComboBox):
            if w.findText(str(v)) < 0:
                w.addItem(str(v))
            w.setCurrentText(str(v))
        else:
            w.setValue(type(w.value())(v))

    @staticmethod
    def _title(k):
        return {"child": "Part class", "parent": "Parent class", "label": "Class", "a": "Class A", "b": "Class B",
                "need_parent": "Every part must lie inside a parent",
                "min_frac": "Inside at least (share of the part's area)",
                "max_rel": "Not larger than (of the parent's long / short side)",
                "min_rel": "Not smaller than (of the parent's long / short side)",
                "max_angle": "Turned like the parent: max difference (°, 0 = off)",
                "per": "At most (× the second class)", "max_deg": "Max tilt (°)",
                "apart_iou": "Parts of one parent apart: max overlap (IoU, 0 = off)",
                "max_iou": "Max overlap (IoU)", "min": "Minimum",
                "max": "Maximum", "labels": "Classes", "types": "Allowed shape types", "min_w": "Min width px",
                "min_h": "Min height px", "max_w": "Max width px", "max_h": "Max height px",
                "min_area_pct": "Min area % of image", "max_area_pct": "Max area % of image",
                "margin_px": "Border margin px", "flag": "Attribute (shape flag)", "value": "Required value"}.get(k, k)

    def value(self) -> dict:
        r = dict(self.rule)
        name = self.name.text().strip()
        if name:
            r["name"] = name
        else:
            r.pop("name", None)
        for k, w in self.widgets.items():
            if isinstance(w, QCheckBox):
                r[k] = w.isChecked()
            elif k in PERCENT_PARAMS:
                r[k] = round(w.value() / 100, 4)
            elif isinstance(w, QComboBox):
                r[k] = w.currentText().strip()
            elif isinstance(w, QListWidget):
                r[k] = [w.item(i).text() for i in range(w.count()) if w.item(i).checkState() == Qt.CheckState.Checked]
            else:
                r[k] = w.value()
        r["severity"] = self.sev.currentData()
        return r

    def _update_preview(self):
        v = self.value()
        v.pop("name", None)
        self.preview.setText("Checks: " + RL.describe(v))


class RulesDialog(BaseScanDialog):
    TITLE, ICON = "Relationship Rules", "🔗"
    HELP = ("Checks how annotations relate, for any classes (* = any class) and any shape type. Parent / part: a "
            "part lies inside its parent, is not larger than it, each parent has min–max parts (wheels on a car, "
            "heads on persons, nuclei in cells, any object inside a region). Also counts per image, classes that "
            "must touch or never appear together, overlaps, size, shape type, tilt, attributes … Add rules by hand "
            "or let AIO QC suggest the ones that already hold for nearly all of your data. Filter results by rule.")
    COLUMNS = ["Rule", "Problem", "Proposed fix", "Shape"]
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
        row.addWidget(newButton("Guideline…", icon="tool-guidelines", slot=self._guideline,
                                tip="Write this rule's guideline with ✓ / ✗ examples (shown above its results)"))
        row.addStretch(1)
        row.addWidget(newButton("Import…", icon="open", slot=self._import))
        row.addWidget(newButton("Export…", icon="save", slot=self._export))
        w = QWidget()
        w.setLayout(row)
        form.addRow(w)
        self._fill_rules()
        self.profile = None                              # what the dataset does (RL.RuleStats), read once
        self._form = None
        self._profile_worker = None
        self._read_profile()

    # ================================================================== dataset profile
    PROFILE_FILES = 5000

    def _profile_samples(self):
        samples = list(self.session.samples)
        if len(samples) > self.PROFILE_FILES:
            samples = random.Random(0).sample(samples, self.PROFILE_FILES)
        return samples

    @staticmethod
    def build_profile(adapter, samples) -> "RL.RuleStats":
        """One pass over (a sample of) the dataset: everything forms and suggestions need."""
        st = RL.RuleStats()
        for s in samples:
            try:
                ann = adapter.load(s, light=True)
            except Exception:
                continue
            W, H = ann.width, ann.height
            if not (W and H):
                try:
                    size = adapter.image_size(s, ann)
                except Exception:
                    size = None
                W, H = size if size else (None, None)
            st.add(ann.shapes, ann.tags, W, H, normalized=bool(ann.normalized))
        return st

    def _read_profile(self):
        adapter, samples = self.session.adapter, self._profile_samples()
        w = FuncWorker(lambda: self.build_profile(adapter, samples), self)
        w.done.connect(self._profile_ready)
        w.failed.connect(lambda m: self.app.log_errors([f"[rules] reading the dataset: {m}"], quiet=True))
        self._profile_worker = w
        w.start()

    def _profile_ready(self, st):
        self.profile = st
        if self._form is not None:
            self._form.set_profile(st)

    def _open_form(self, rule) -> "RuleForm":
        f = RuleForm(self, rule, self.labels, self.profile)
        self._form = f
        return f

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

    def _duplicate_of(self, rule, skip=None):
        sig = RL.signature(rule)
        return next((r for r in self.rules if r is not skip and RL.signature(r) == sig), None)

    def _accept_rule(self, rule, skip=None) -> bool:
        dup = self._duplicate_of(rule, skip)
        if dup is not None:
            QMessageBox.information(self, self.TITLE, f"This rule is already in the list:\n\n{RL.describe(dup)}")
            return False
        return True

    def _add(self, rtype):
        f = self._open_form(RL.new_rule(rtype))
        ok = f.exec()
        self._form = None
        if ok and self._accept_rule(f.value()):
            self.rules.append(f.value())
            self._save()

    def _edit(self, row):
        if not 0 <= row < len(self.rules):
            return
        f = self._open_form(self.rules[row])
        ok = f.exec()
        self._form = None
        if ok and self._accept_rule(f.value(), skip=self.rules[row]):
            self.rules[row] = f.value()
            self._save()

    def _guideline(self):
        row = self.rtable.currentRow()
        if 0 <= row < len(self.rules):
            self.app.open_guidelines(f"rule:{self.rules[row]['id']}")

    def _delete(self):
        rows = sorted({i.row() for i in self.rtable.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.rules[r]
        self._save()

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import rules", "", "Rules (*.json)")
        if path:
            new = RL.load_rules(path)
            have = {RL.signature(r) for r in self.rules}
            added = 0
            for r in new:
                if RL.signature(r) in have:
                    continue                                  # already there: never twice
                have.add(RL.signature(r))
                r["id"] = RL.new_rule(r["type"])["id"]
                self.rules.append(r)
                added += 1
            self._save()
            QMessageBox.information(self, self.TITLE, f"Imported {added} rule(s)" +
                                    (f"; {len(new) - added} were already in the list." if added < len(new) else "."))

    def _export(self):
        from AIO_QC.utils.save_dirs import ask_save
        path, _ = ask_save(self, "Export rules", f"{self.session.root.name}_rules.json", "Rules (*.json)",
                           kind="exports", fallback=self.session.root.parent)
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
        existing = {RL.signature(r) for r in self.rules}

        profile = self.profile if len(samples) <= self.PROFILE_FILES else None    # read already: reuse it

        def mine():
            st = profile if profile is not None else self.build_profile(adapter, samples)
            if profile is None and len(samples) <= self.PROFILE_FILES:
                self.profile = st
            return RL.suggest(st, min_support=max(5, min(20, len(samples) // 50)))

        def done(sugs):
            bar.setVisible(False)
            n = 0
            for sg in sugs:
                if RL.signature(sg.rule) in existing:
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
        self.filter_box.currentIndexChanged.connect(lambda _i: self._rule_guide())
        self.fix_btn = self.add_action("✔ Apply proposed fix", self.apply_fixes,
                                       "Apply each ticked (or selected) row's proposed fix — e.g. fit a part inside its "
                                       "parent, delete a stray part or a duplicate, rectangle ↔ polygon. One backup: "
                                       "Undo puts the files back. The files are marked 🔧 Fixed.")
        self.undo_btn = self.add_action("↶ Undo", self._undo, "Put back the files changed by the last Apply")
        self.undo_btn.setEnabled(False)
        self.add_action("🗑 Delete the shape", lambda: self.delete_shape_rows(self.target_rows()),
                        "Delete the shape each ticked (or selected) row is about")
        self._last_backup = None

    def make_context(self):
        active = [r for r in self.rules if r.get("enabled", True)]
        if not active:
            raise ValueError("Add (or tick) at least one rule — or use “Suggest from data…”.")
        self._rule_by_name = {RL.describe(r): r for r in active}
        return {"adapter": self.session.adapter, "rules": active,
                "px": any(RL.TYPES[r["type"]][3] or r["type"] in GEOMETRY for r in active)}

    def samples_to_scan(self):
        """Only files that can break a rule: when every active rule needs one of its classes in the file, files
        without any of them (label index) are skipped unopened."""
        s = self.session
        samples = list(s.samples)
        active = [r for r in self.rules if r.get("enabled", True)]
        triggers = [RL.trigger_labels(r) for r in active]
        if not active or any(tr is None for tr in triggers) or not s.index_complete:
            return samples
        need = set().union(*triggers)
        idx = s.index
        return [x for x in samples if x.key not in idx or any(idx[x.key].get(l) for l in need)]

    def scan_sample(self, ctx, s):
        ann = ctx["adapter"].load(s, light=True)
        W = H = None
        if ctx["px"] or ann.normalized:
            size = (ann.width, ann.height) if ann.width and ann.height else ctx["adapter"].image_size(s, ann)
            if size:
                W, H = size
        vs = RL.check_image(ctx["rules"], ann.shapes, ann.tags, W, H, normalized=ann.normalized)
        FW, FH = (1.0, 1.0) if ann.normalized else (float(W or 0), float(H or 0))   # = Session._frame_wh
        rows = []
        for v in vs:
            title = RL.describe(v.rule)                     # unique per rule: the filter picks one rule
            op = self._fix_op(v, ann.shapes, FW, FH)
            rows.append(ResultRow(s.key, (title, v.text, v.fix["title"] if op else "", v.shapes[0] if v.shapes else ""),
                                  shape_index=v.shapes[0] if v.shapes else None,
                                  payload=(v, [ann.shapes[i] for i in v.shapes if i < len(ann.shapes)], ann.normalized,
                                           op, (W, H)),
                                  tone=TONE.get(v.rule.get("severity", "error"))))
        return rows

    @staticmethod
    def _fix_op(v, shapes, FW, FH):
        """The violation's proposed fix as a Session.apply_fixes op (relative coordinates), or None."""
        f = v.fix
        if not f or f.get("shape") is None or not FW or not FH or f["shape"] >= len(shapes):
            return None
        pts = shapes[f["shape"]].get("points") or []
        if len(pts) < 1:
            return None
        xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
        if (shapes[f["shape"]].get("shape_type") == "circle") and len(pts) == 2:
            r = ((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2) ** 0.5
            xs, ys = [xs[0] - r, xs[0] + r], [ys[0] - r, ys[0] + r]
        box = (min(xs) / FW, min(ys) / FH, max(xs) / FW, max(ys) / FH)
        if f["op"] == "delete":
            return ("delete", f["shape"], box)
        return ("set", f["shape"], box, {"points": [[float(x) / FW, float(y) / FH] for x, y in f["points"]],
                                         "type": f.get("type")})

    def _rule_guide(self):
        """The guideline of the rule picked in the filter (else this tool's general one)."""
        guides = getattr(self.session, "guides", None)
        if guides is None:
            return
        r = getattr(self, "_rule_by_name", {}).get(self.filter_box.currentData())
        scope = f"rule:{r['id']}" if r is not None and guides.has(f"rule:{r['id']}") else f"tool:{self.tool_key_name()}"
        from .guidelines import scope_title
        self.guide_card.setVisible(self.guide_card.set_scope(guides, scope, titles={scope: scope_title(scope)
                                   if r is None else f"Rule · {RL.describe(r)}"}))

    def _flag_note(self, rows) -> dict:
        """⚑ Flag files: the note says which rule and what exactly — 'Rules: Wheels on their car — wheel #1: larger
        than car #0 (107 % of it)' — so the review queue, batch screenshots and hand-offs carry the reason."""
        why: dict[str, list] = {}
        for r in rows:
            v = r.payload[0] if r.payload else None
            text = f"{RL.describe(v.rule)} — {v.text}" if v is not None else str(r.cells[1] if len(r.cells) > 1 else "")
            lst = why.setdefault(r.key, [])
            if text not in lst:
                lst.append(text)
        return {k: (f"{self.TITLE}: " + "; ".join(v[:3]) + (f" (+{len(v) - 3} more)" if len(v) > 3 else ""))[:400]
                for k, v in why.items()}

    def issue_text(self, row):
        return f"{row.cells[0]}: {row.cells[1]}"

    @staticmethod
    def _outline(sh) -> list:
        """A shape's real outline for a card (rectangles as 4 corners, circles as a polygon)."""
        sp = [(float(p[0]), float(p[1])) for p in (sh.get("points") or [])]
        st = sh.get("shape_type") or "polygon"
        if st == "rectangle" and len(sp) >= 2:
            (x1, y1), (x2, y2) = sp[0], sp[1]
            return box_points(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if st == "circle" and len(sp) == 2:
            import math
            r = math.hypot(sp[1][0] - sp[0][0], sp[1][1] - sp[0][1])
            return [(sp[0][0] + r * math.cos(a * math.pi / 12), sp[0][1] + r * math.sin(a * math.pi / 12))
                    for a in range(24)]
        if len(sp) == 1:                                    # a point: a small diamond
            x, y = sp[0]
            return [(x - 3, y), (x, y - 3), (x + 3, y), (x, y + 3)]
        return sp

    @staticmethod
    def _headline(text: str) -> str:
        """'wheel #1: larger than car #0 (106 % × 107 % of it) and only 87 % inside car #0'
        -> 'Larger than car (107 %) · +1' (fits a card line)."""
        import re
        body = text.split(": ", 1)[1] if ": " in text else text
        parts = body.split(" and ")
        head = re.sub(r"\((\d+) % × (\d+) % of it\)", lambda m: f"({max(int(m[1]), int(m[2]))} %)", parts[0])
        head = re.sub(r" #\d+", "", head)                  # numbers are in the tooltip and the table
        head = head[:1].upper() + head[1:]
        return head + (f" · +{len(parts) - 1}" if len(parts) > 1 else "")

    def _refresh_cards(self):
        """One card per problem: the shape at fault (red), the shapes it relates to (dashed, class colour) and the
        proposed fix (green) — what is wrong and what the fix would do, at a glance."""
        if not hasattr(self, "cards"):
            return
        self.prune_marks()
        cards = []
        for row in self.visible_rows()[:5000]:
            v, shapes, norm, op, _wh = row.payload
            s = self.session.by_key.get(row.key)
            drawn, pts = [], []
            meas = RL.measures(v.info)
            for j, sh in enumerate(shapes):
                out = self._outline(sh)
                if len(out) < 2:
                    continue
                pts += out
                col = self.app.label_colors.get(sh.get("label"))
                idx = v.shapes[j] if j < len(v.shapes) else ""
                tag = f"{sh.get('label')} #{idx}" + (f" · {meas}" if j == 0 and meas else "")
                drawn.append(Shape(out, PROBLEM if j == 0 else (col.name() if col else hexc("info")), dashed=j > 0,
                                   width=3.0 if j == 0 else 2.0, text=tag))
            if v.fix and v.fix.get("op") == "set" and v.fix.get("points"):
                fx = self._outline({"shape_type": v.fix.get("type") or (shapes[0].get("shape_type") if shapes else ""),
                                    "points": v.fix["points"]})
                if len(fx) >= 2:
                    drawn.append(Shape(fx, FIX, dashed=True, width=2.0))
            if pts:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                panel = Panel(row.key, (min(xs), min(ys), max(xs), max(ys)), drawn, normalized=norm)
            else:
                panel = Panel(row.key, (0, 0, 1, 1), [], normalized=True)
            sev = v.rule.get("severity", "error")
            names = " → ".join(dict.fromkeys(str(sh.get("label")) for sh in shapes[:2]))
            parts = meas.split(" · ") if meas else []          # the tag on the crop carries every value
            head = f"{names} · {parts[0]}" + (f" +{len(parts) - 1}" if len(parts) > 1 else "") if names and parts \
                else self._headline(v.text)
            lines = [head]                                    # what · how much
            lines.append(f"✔ Fix: {v.fix['title']}" if op else (s.display if s else row.key))
            lines += [v.text, RL.describe(v.rule), s.display if s else row.key]      # the rest: tooltip
            cards.append(Card(row, [panel], [(RL.TYPES[v.rule["type"]][0],
                                              {"warn": "warn", "info": "muted"}.get(sev, "error"))], lines))
        self.cards.set_cards(cards)
        self.tabs.setTabText(0, f"Cards ({len(cards):,})")

    # ================================================================== open · fix · undo
    def on_row_activated(self, row):
        """Open the file framed on every shape involved (the part AND its parent), the shape at fault selected,
        the proposed fix drawn as a green dashed box."""
        if row is None or not row.key:
            return
        v, shapes, norm, op, (W, H) = row.payload
        sx, sy = (float(W or 1), float(H or 1)) if norm else (1.0, 1.0)
        pts = [(x * sx, y * sy) for sh in shapes for x, y in self._outline(sh)]
        focus = None
        if pts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            pad = 0.15 * max(max(xs) - min(xs), max(ys) - min(ys), 8)
            focus = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
        overlay = None
        if v.fix and v.fix.get("op") == "set" and v.fix.get("points"):
            fx = [(float(x) * sx, float(y) * sy) for x, y in v.fix["points"]]
            overlay = [{"box": (min(p[0] for p in fx), min(p[1] for p in fx), max(p[0] for p in fx),
                                max(p[1] for p in fx)), "text": f"fix: {v.fix['title']}", "color": FIX}]
        self.app.open_sample(row.key, row.shape_index, overlay=overlay, focus_box=focus)

    def apply_fixes(self):
        rows = [r for r in self.target_rows() if r.payload and r.payload[3]]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick (or select) rows that have a proposed fix first — the "
                                                      "“Proposed fix” column says what each one would do.")
            return
        from collections import Counter
        kinds = Counter(r.payload[0].fix["title"].split(" #")[0] for r in rows)
        files = {r.key for r in rows}
        if not self.confirm(f"Apply {len(rows):,} fix(es) in {len(files):,} file(s)?\n\n" +
                            "\n".join(f"• {n:,} × {k}" for k, n in kinds.most_common()) +
                            "\n\nOne backup is kept: Undo puts every file back."):
            return
        fixes, seen = {}, set()
        for r in rows:
            op = r.payload[3]
            if (r.key, op[1]) in seen:                     # one change per shape (two rules may propose one)
                continue
            seen.add((r.key, op[1]))
            fixes.setdefault(r.key, []).append(op)
        st = self.session.apply_fixes(self, fixes, what="relationship rules")
        if st["files"]:
            self._last_backup = st["backup"]
            self.undo_btn.setEnabled(True)
            from AIO_QC.core.state import REVIEW_FIXED
            self.session.set_review(st["changed"], REVIEW_FIXED, "Relationship Rules: fixed — check it")
        self.after_edit(files)
        lines = [f"Applied {st['done']:,} fix(es) in {st['files']:,} file(s); they are marked 🔧 Fixed."]
        if st["missing"]:
            lines.append(f"{st['missing']:,} skipped: the shape changed since the scan.")
        if st["refused"]:
            lines.append(f"{st['refused']:,} skipped: that shape type cannot be changed this way.")
        self.report("Done", lines, st["errors"])

    def _undo(self):
        b = self._last_backup
        if b is None or not self.confirm(f"Put back the {len(b.after):,} file(s) changed by the last Apply?"):
            return
        keys, skipped = self.session.undo_backup(b)
        self._last_backup = None
        self.undo_btn.setEnabled(False)
        self.after_edit(keys)
        self.report("Undone", [f"Restored {len(keys):,} file(s)."] +
                    ([f"{len(skipped):,} changed since and left alone."] if skipped else []), [])

    def summarize(self):
        self._refresh_cards()
        from collections import Counter
        live = [r for r in self.rows if r]
        c = Counter(RL.TYPES[r.payload[0].rule["type"]][0] for r in live)
        fixable = sum(1 for r in live if r.payload[3])
        files = len({r.key for r in live})
        if not live:
            return "✅ Every file follows every rule"
        return (f"{len(live):,} problem(s) in {files:,} file(s) · " + " · ".join(f"{n:,} {k}" for k, n in c.most_common())
                + (f" · {fixable:,} with a proposed fix (✔ Apply)" if fixable else ""))
