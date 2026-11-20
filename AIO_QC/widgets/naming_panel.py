# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
The naming panel (core.naming) — the same everywhere names can clash: a preset or your own pattern, live preview
and verification as you type, and nothing can go ahead while a name still clashes.

    panel = NamingPanel(parent, context="convert_yolo")      # its settings are remembered per context
    panel.set_items(items, flat=True, existing=…, root_len=…)
    panel.planChanged.connect(…)                             # after every (debounced) verification
    P = panel.plan()                                         # None until verified; P.ok = ready

Presets: the built-in ones plus your own (Save as preset…), kept app-wide. Big sets are verified in the background;
a newer edit always wins over an older verification still running.
"""
from __future__ import annotations

import json

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from AIO_QC.core import naming as N
from AIO_QC.utils import newButton, style as apply_style

PREVIEW = 200                     # rows shown (problems first)
THREAD_ABOVE = 20000              # verify in the background above this many files


def _settings():
    from AIO_QC import utils
    return utils.app_settings()


def user_presets() -> list[dict]:
    try:
        v = json.loads(_settings().value("naming/presets", "[]") or "[]")
        return [p for p in v if isinstance(p, dict) and p.get("name") and isinstance(p.get("scheme"), dict)]
    except (TypeError, ValueError):
        return []


def save_user_presets(presets: list[dict]) -> None:
    _settings().setValue("naming/presets", json.dumps(presets))


class NamingPanel(QWidget):
    planChanged = pyqtSignal(object)          # NamePlan (verified)

    def __init__(self, parent=None, context: str = "default", allow_scope: bool = True, preview: bool = True):
        """preview=False: just the settings (the files are not known yet — the tool verifies when it plans)."""
        super().__init__(parent)
        self.context = context
        self.items, self.flat, self.existing, self.root_len, self.touch = [], True, {}, 0, None
        self._plan = None
        self._gen = 0
        self._worker = None
        self._loading = False
        f = QFormLayout()
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.preset = QComboBox()
        self.preset.currentIndexChanged.connect(self._preset_chosen)
        row = QHBoxLayout()
        row.addWidget(self.preset, 1)
        self.save_btn = newButton("Save as preset…", slot=self._save_preset,
                                  tip="Keep these settings under a name of yours (for every dataset)")
        self.del_btn = newButton("", icon="delete", slot=self._delete_preset, tip="Delete this preset of yours")
        row.addWidget(self.save_btn)
        row.addWidget(self.del_btn)
        f.addRow("Preset", row)
        self.pattern = QLineEdit()
        self.pattern.setPlaceholderText("{folder}{sep}{stem}")
        self.pattern.textEdited.connect(self._pattern_edited)
        f.addRow("Pattern", self.pattern)
        tok = QLabel(" ".join(f"<a href='{t}'>{{{t}}}</a>" for t in N.TOKENS))
        tok.setToolTip("Click a token to add it. {n:04d} numbers with 4 digits; {folder2} = the two folders above it; "
                       "{path} = every folder below the dataset; {hash8} = from the file's content (the same image "
                       "always gets the same name); {date} = the file's date")
        tok.linkActivated.connect(self._insert_token)
        apply_style(tok, "color:@muted;")
        f.addRow("", tok)
        opts = QHBoxLayout()
        self.sep = QLineEdit("__")
        self.sep.setMaximumWidth(60)
        self.sep.setToolTip("What goes between the folder and the name")
        self.case = QComboBox()
        self.case.addItem("Keep letter case", "keep")
        self.case.addItem("lower case", "lower")
        self.spaces = QCheckBox("Spaces → _")
        self.spaces.setChecked(True)
        self.max_len = QSpinBox()
        self.max_len.setRange(8, 240)
        self.max_len.setValue(150)
        self.max_len.setPrefix("max ")
        self.max_len.setSuffix(" chars")
        self.start = QSpinBox()
        self.start.setRange(0, 10 ** 9)
        self.start.setValue(1)
        self.start.setPrefix("{n} from ")
        for w in (QLabel("Separator"), self.sep, self.case, self.spaces, self.max_len, self.start):
            opts.addWidget(w)
        opts.addStretch(1)
        f.addRow("", opts)
        self.scope = QComboBox()
        self.scope.addItem("Only files whose names clash", True)
        self.scope.addItem("Every file", False)
        self.scope.setVisible(allow_scope)
        if allow_scope:
            f.addRow("Rename", self.scope)
        for w in (self.sep,):
            w.textChanged.connect(self._changed)
        for w in (self.case, self.scope):
            w.currentIndexChanged.connect(self._changed)
        self.spaces.toggled.connect(self._changed)
        self.max_len.valueChanged.connect(self._changed)
        self.start.valueChanged.connect(self._changed)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.fix_btn = newButton("Number the rest (_1, _2 …)", slot=self.fix_remaining,
                                 tip="Give every name that still clashes a number, fix what Windows refuses, shorten "
                                     "what is too long — then verify again")
        self.fix_btn.setVisible(False)
        self.only_problems = QCheckBox("Show only problems")
        self.only_problems.toggled.connect(lambda _on: self._fill_table())
        srow = QHBoxLayout()
        srow.addWidget(self.summary, 1)
        srow.addWidget(self.only_problems)
        srow.addWidget(self.fix_btn)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Now", "Becomes", "Problem"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        for i in range(3):
            hh.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(f)
        lay.addLayout(srow)
        lay.addWidget(self.table, 1)
        for wdg in (self.summary, self.only_problems, self.table):
            wdg.setVisible(preview)
        self.preview = preview
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._verify)
        self._fill_presets()
        self._load_last()

    # ------------------------------------------------------------------ presets
    def _fill_presets(self, select: str | None = None):
        self._loading = True
        self.preset.clear()
        for k, (title, _pat, example) in N.PRESETS.items():
            self.preset.addItem(f"{title}" + (f"  —  {example}" if example else ""), {"builtin": k})
        for p in user_presets():
            self.preset.addItem(f"★ {p['name']}", {"user": p["name"]})
        if select:
            i = next((n for n in range(self.preset.count()) if (self.preset.itemData(n) or {}).get("user") == select),
                     -1)
            if i >= 0:
                self.preset.setCurrentIndex(i)
        self._loading = False
        self.del_btn.setEnabled(bool((self.preset.currentData() or {}).get("user")))

    def scheme(self) -> N.Scheme:
        pat = self.pattern.text().strip()
        preset = next((k for k, v in N.PRESETS.items() if k != "custom" and v[1] == pat), "custom")
        d = self.preset.currentData() or {}
        if d.get("builtin") in N.PRESETS and d["builtin"] != "custom" and N.PRESETS[d["builtin"]][1] == pat:
            preset = d["builtin"]                      # (suffix and a custom "{stem}" share a pattern)
        return N.Scheme(preset=preset, pattern=pat, sep=self.sep.text(), case=self.case.currentData() or "keep",
                        spaces=self.spaces.isChecked(), max_len=self.max_len.value(), start=self.start.value(),
                        only_clashes=bool(self.scope.currentData()) if self.scope.isVisible() else True)

    def set_scheme(self, sc: N.Scheme, keep_preset: bool = False):
        self._loading = True
        k = sc.preset if sc.preset in N.PRESETS else "custom"
        if not keep_preset:
            i = next((n for n in range(self.preset.count())
                      if (self.preset.itemData(n) or {}).get("builtin") == k), 0)
            self.preset.setCurrentIndex(i)
        self.pattern.setText(sc.template() if k != "custom" else sc.pattern)
        self.sep.setText(sc.sep)
        self.case.setCurrentIndex(max(0, self.case.findData(sc.case)))
        self.spaces.setChecked(bool(sc.spaces))
        self.max_len.setValue(int(sc.max_len or 150))
        self.start.setValue(int(sc.start))
        self.scope.setCurrentIndex(0 if sc.only_clashes else 1)
        self._loading = False
        self._changed()

    def _preset_chosen(self, _i):
        if self._loading:
            return
        d = self.preset.currentData() or {}
        self.del_btn.setEnabled(bool(d.get("user")))
        if d.get("user"):
            p = next((x for x in user_presets() if x["name"] == d["user"]), None)
            if p:
                self.set_scheme(N.Scheme.from_dict(p["scheme"]), keep_preset=True)
                return
        k = d.get("builtin", "folder")
        if k != "custom":
            self.pattern.setText(N.PRESETS[k][1])
        self.start.setVisible("{n" in (self.pattern.text() or ""))
        self._changed()

    def _pattern_edited(self, _t):
        d = self.preset.currentData() or {}
        if d.get("builtin") != "custom":
            self._loading = True
            self.preset.setCurrentIndex(next(n for n in range(self.preset.count())
                                             if (self.preset.itemData(n) or {}).get("builtin") == "custom"))
            self._loading = False
        self.start.setVisible("{n" in self.pattern.text())
        self._changed()

    def _insert_token(self, tok):
        self.pattern.insert("{" + tok + "}")
        self._pattern_edited(self.pattern.text())

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Naming preset", "Name of this preset:")
        if not ok or not name.strip():
            return
        ps = [p for p in user_presets() if p["name"] != name.strip()]
        ps.append({"name": name.strip(), "scheme": self.scheme().to_dict()})
        save_user_presets(ps)
        self._fill_presets(select=name.strip())

    def _delete_preset(self):
        d = self.preset.currentData() or {}
        if d.get("user"):
            save_user_presets([p for p in user_presets() if p["name"] != d["user"]])
            self._fill_presets()
            self._preset_chosen(0)

    def _load_last(self):
        raw = _settings().value(f"naming/last/{self.context}", "")
        try:
            sc = N.Scheme.from_dict(json.loads(raw)) if raw else N.Scheme()
        except (TypeError, ValueError):
            sc = N.Scheme()
        self.set_scheme(sc)

    def remember(self):
        _settings().setValue(f"naming/last/{self.context}", json.dumps(self.scheme().to_dict()))

    # ------------------------------------------------------------------ verification
    def set_items(self, items, flat: bool = True, existing=None, root_len: int = 0, touch=None):
        """touch: the keys renamed under "Only files whose names clash" (default: the ones that clash)."""
        self.items, self.flat, self.existing, self.root_len = list(items), flat, existing or {}, root_len
        self.touch = set(touch) if touch is not None else None
        self._changed()

    def _changed(self, *_):
        if self._loading:
            return
        self._plan = None
        bad = N.bad_tokens(self.pattern.text())
        if bad:
            self.summary.setText(f"⚠ Unknown in the pattern: {', '.join(bad)} — the tokens are listed below it")
            apply_style(self.summary, "color:@error;")
            self.fix_btn.setVisible(False)
            self.planChanged.emit(None)
            return
        self.summary.setText("Checking the names …")
        apply_style(self.summary, "color:@muted;")
        self._timer.start()

    def _verify(self):
        self._gen += 1
        gen, sc = self._gen, self.scheme()
        items, flat, ex, rl, touch = self.items, self.flat, self.existing, self.root_len, self.touch

        def job():
            return N.plan(items, sc, flat, ex, rl, touch=touch)
        if len(items) <= THREAD_ABOVE:
            try:
                self._verified(gen, job())
            except (ValueError, KeyError, IndexError) as exc:
                self._failed(gen, exc)
            return
        from .inference_service import FuncWorker
        w = FuncWorker(job, self)
        w.done.connect(lambda P, g=gen: self._verified(g, P))
        w.failed.connect(lambda m, g=gen: self._failed(g, m))
        self._worker = w
        w.start()

    def _failed(self, gen, err):
        if gen != self._gen:
            return
        self.summary.setText(f"⚠ The pattern does not work: {err}")
        apply_style(self.summary, "color:@error;")
        self.planChanged.emit(None)

    def _verified(self, gen, P):
        if gen != self._gen:
            return                                          # a newer edit is being checked
        self._plan = P
        self._show()
        self.remember()
        self.planChanged.emit(P)

    def fix_remaining(self):
        if self._plan is None:
            return
        self._plan = N.resolve(self._plan, self.existing, self.root_len)
        self._show()
        self.planChanged.emit(self._plan)

    def _show(self):
        P = self._plan
        self.summary.setText(("✓ " if P.ok else "⚠ ") + P.summary() +
                             ("" if P.ok else " — change the pattern, or Number the rest"))
        apply_style(self.summary, "color:@ok;" if P.ok else "color:@warn;")
        self.fix_btn.setVisible(not P.ok and self.preview)
        self._fill_table()

    def _fill_table(self):
        P = self._plan
        self.table.setRowCount(0)
        if P is None:
            return
        rows = [it for it in P.items if it.key in P.problems]
        if not self.only_problems.isChecked():
            rows += [it for it in P.items if it.key not in P.problems and P.new_rel(it.key) != it.rel]
        rows = rows[:PREVIEW]
        self.table.setRowCount(len(rows))
        for r, it in enumerate(rows):
            for c, t in enumerate([it.rel, P.new_rel(it.key), "; ".join(P.problems.get(it.key, []))]):
                cell = QTableWidgetItem(t)
                cell.setToolTip(t)
                if c == 2 and t:
                    from AIO_QC.utils import theme_color
                    cell.setForeground(theme_color("warn"))
                self.table.setItem(r, c, cell)

    def plan(self):
        return self._plan

    def is_ready(self) -> bool:
        return self._plan is not None and self._plan.ok

    def wait(self, qtbot=None, timeout_ms: int = 30000):
        """(tests / callers that need the result now) run any pending verification at once."""
        if self._timer.isActive():
            self._timer.stop()
            self._verify()
        if self._worker is not None:
            self._worker.wait(timeout_ms)
        return self._plan


__all__ = ["NamingPanel", "user_presets", "Qt"]
