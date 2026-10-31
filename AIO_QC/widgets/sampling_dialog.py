# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
🎯 QC sampling: review a random (stratified) sample instead of every file, and get a statistically
sound estimate of the dataset's error rate with a pass / fail verdict.

Workflow: create sample → it opens as a file-list filter → review with S (OK) / X (flag) →
the estimate, confidence interval and verdict update live.
"""
from __future__ import annotations

import random
import time
from collections import Counter, defaultdict

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from AIO_QC.core.sampling import SamplePlan, draw_sample, estimate, sample_size
from AIO_QC.core.state import list_samples, save_sample
from AIO_QC.utils import style as apply_style
from AIO_QC.utils import newButton, gradient_hex, tone
from .base_scan import BAR_ROLE, dspin, readable_table

STRATEGIES = [
    ("Simple random", "random"),
    ("Stratified by split", "split"),
    ("Stratified by rarest label in file (rare classes guaranteed)", "rare_label"),
    ("Stratified by split × rarest label", "split_label"),
]
CONFIDENCES = [("90 %", 0.90), ("95 %", 0.95), ("99 %", 0.99)]


class SamplingDialog(QDialog):
    TITLE, ICON = "QC Sampling", "🎯"
    SUPPORTED = None

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.plan: SamplePlan | None = None
        self.setWindowTitle(self.TITLE)
        self.resize(int(app.width() * 0.55), int(app.height() * 0.8))

        # ---------------------------------------------------------------- create
        create = QGroupBox("1 · Create a sample")
        f = QFormLayout(create)
        self.scope = QComboBox()
        self.scope.addItems(["All files", "Files in current list"])
        self.strategy = QComboBox()
        for text, val in STRATEGIES:
            self.strategy.addItem(text, val)
        self.allocation = QComboBox()
        self.allocation.addItems(["Proportional to stratum size", "Equal per stratum"])
        self.conf = QComboBox()
        for text, v in CONFIDENCES:
            self.conf.addItem(text, v)
        self.conf.setCurrentIndex(1)
        self.margin = dspin(0.5, 20, 3.0, 1, 0.5, suffix=" %")
        self.expected = dspin(0.5, 50, 5.0, 1, 0.5, suffix=" %")
        self.expected.setToolTip("Your guess of the error rate. 50 % is the most conservative (largest sample).")
        self.n = QSpinBox()
        self.n.setRange(1, 1_000_000)
        self.min_per = QSpinBox()
        self.min_per.setRange(0, 1000)
        self.min_per.setValue(3)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(random.randint(1, 999_999))
        self.name = QLineEdit(time.strftime("sample_%Y%m%d_%H%M"))
        self.recommend = QLabel()
        apply_style(self.recommend, "color:@muted;")
        for w in (self.conf, self.scope):
            w.currentIndexChanged.connect(self._recompute)
        for w in (self.margin, self.expected):
            w.valueChanged.connect(self._recompute)
        calc = QWidget()
        h = QHBoxLayout(calc)
        h.setContentsMargins(0, 0, 0, 0)
        for text, w in (("confidence", self.conf), ("margin ±", self.margin), ("expected error", self.expected)):
            h.addWidget(QLabel(text))
            h.addWidget(w)
        h.addStretch(1)
        nrow = QWidget()
        h2 = QHBoxLayout(nrow)
        h2.setContentsMargins(0, 0, 0, 0)
        h2.addWidget(self.n)
        h2.addWidget(self.recommend, 1)
        f.addRow("Population:", self.scope)
        f.addRow("Strategy:", self.strategy)
        f.addRow("Allocation:", self.allocation)
        f.addRow("Sample size from:", calc)
        f.addRow("Files to review:", nrow)
        extra = QWidget()
        h3 = QHBoxLayout(extra)
        h3.setContentsMargins(0, 0, 0, 0)
        for text, w in (("min per stratum", self.min_per), ("seed", self.seed), ("name", self.name)):
            h3.addWidget(QLabel(text))
            h3.addWidget(w, 1 if w is self.name else 0)
        f.addRow(extra)
        mk = newButton("🎲 Create sample && start reviewing", slot=lambda: self.create_sample())
        f.addRow(mk)

        # ------------------------------------------------------------------ open
        pick = QGroupBox("2 · Review")
        p = QHBoxLayout(pick)
        self.existing = QComboBox()
        self.existing.currentIndexChanged.connect(self._load_selected)
        show = newButton("👁 Show in file list", slot=lambda: self.show_in_list())
        nxt = newButton("⏭ Next unreviewed", slot=lambda: self.next_unreviewed(), tip="Also: N in the main window")
        p.addWidget(QLabel("Sample:"))
        p.addWidget(self.existing, 1)
        p.addWidget(show)
        p.addWidget(nxt)

        # --------------------------------------------------------------- results
        res = QGroupBox("3 · Result (updates live as you mark files with S / X)")
        r = QVBoxLayout(res)
        top = QHBoxLayout()
        self.verdict = QLabel("—")
        vf = self.verdict.font()
        vf.setPointSizeF(max(vf.pointSizeF(), 9.0) + 8)
        vf.setBold(True)
        self.verdict.setFont(vf)
        self.verdict.setMinimumWidth(230)
        self.verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        df = self.detail.font()
        df.setPointSizeF(max(df.pointSizeF(), 9.0) + 1.5)
        df.setWeight(QFont.Weight.DemiBold)
        self.detail.setFont(df)
        top.addWidget(self.verdict)
        top.addWidget(self.detail, 1)
        r.addLayout(top)
        acc = QHBoxLayout()
        self.max_err = dspin(0.1, 50, 5.0, 1, 0.5, suffix=" %")
        self.max_err.valueChanged.connect(self.update_results)
        self.res_conf = QComboBox()
        for text, v in CONFIDENCES:
            self.res_conf.addItem(text, v)
        self.res_conf.setCurrentIndex(1)
        self.res_conf.currentIndexChanged.connect(self.update_results)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m reviewed")
        for text, w in (("Accept if error rate ≤", self.max_err), ("at confidence", self.res_conf)):
            acc.addWidget(QLabel(text))
            acc.addWidget(w)
        acc.addWidget(self.progress, 1)
        r.addLayout(acc)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Stratum", "Population", "Sampled", "Reviewed", "Flagged",
                                              "Error rate", "CI"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        readable_table(self.table)
        self.table.cellDoubleClicked.connect(self._stratum_activated)
        r.addWidget(self.table, 1)

        lay = QVBoxLayout(self)
        lay.addWidget(create)
        lay.addWidget(pick)
        lay.addWidget(res, 1)
        self.session.reviewChanged.connect(self.update_results)
        self._recompute()
        self._refresh_existing()

    # ------------------------------------------------------------------- create
    def _population(self) -> list:
        if self.scope.currentIndex() == 1:
            keys = set(self.app.file_dock.visible_keys())
            return [s for s in self.session.samples if s.key in keys]
        return list(self.session.samples)

    def _recompute(self, *_):
        N = len(self._population())
        n = sample_size(N, self.conf.currentData(), self.margin.value() / 100, self.expected.value() / 100)
        self.n.setValue(n)
        self.recommend.setText(f"recommended for {N:,} files (you can change it)")

    def _strata(self, samples) -> dict[str, list[str]]:
        mode = self.strategy.currentData()
        if mode == "random":
            return {"all": [s.key for s in samples]}
        idx = self.session.index
        totals = self.session.label_totals()
        out = defaultdict(list)
        for s in samples:
            parts = []
            if mode in ("split", "split_label"):
                parts.append(s.split or "(no split)")
            if mode in ("rare_label", "split_label"):
                labels = idx.get(s.key) or Counter()
                parts.append(min(labels, key=lambda l: (totals[l], l)) if labels else "(no shapes)")
            out[" · ".join(parts)].append(s.key)
        return dict(out)

    def create_sample(self):
        mode = self.strategy.currentData()
        if mode in ("rare_label", "split_label") and not self.session.index_complete:
            QMessageBox.information(self, self.TITLE, "The label index is still being built — try again in a moment.")
            return
        samples = self._population()
        if not samples:
            QMessageBox.information(self, self.TITLE, "The population is empty.")
            return
        strata = self._strata(samples)
        alloc = "equal" if self.allocation.currentIndex() == 1 else "proportional"
        picked, pop = draw_sample(strata, self.n.value(), alloc, self.seed.value(), self.min_per.value())
        keys = [k for ks in picked.values() for k in ks]
        name = self.name.text().strip() or time.strftime("sample_%Y%m%d_%H%M")
        plan = SamplePlan(name, mode, self.seed.value(), keys, picked, pop,
                          {"allocation": alloc, "confidence": self.conf.currentData(),
                           "margin": self.margin.value(), "expected": self.expected.value(),
                           "population_scope": self.scope.currentText(), "created": time.strftime("%Y-%m-%d %H:%M")})
        save_sample(self.session.root, plan.to_json())
        self.plan = plan
        self._refresh_existing(select=name)
        self.show_in_list()
        self.name.setText(time.strftime("sample_%Y%m%d_%H%M%S"))
        self.seed.setValue(random.randint(1, 999_999))
        self.app.statusBar().showMessage(f"🎯 Sample '{name}': {len(keys):,} files in {len(picked)} strata — "
                                         "review with S (OK) / X (flag)", 8000)

    # --------------------------------------------------------------- existing
    def _refresh_existing(self, select: str | None = None):
        self.existing.blockSignals(True)
        self.existing.clear()
        for d in list_samples(self.session.root):
            self.existing.addItem(f"{d['name']}  ({len(d['keys']):,} files · {d.get('strategy', '?')})", d)
        self.existing.blockSignals(False)
        if select is not None:
            for i in range(self.existing.count()):
                if self.existing.itemData(i)["name"] == select:
                    self.existing.setCurrentIndex(i)
                    break
        self._load_selected()

    def _load_selected(self, *_):
        d = self.existing.currentData()
        self.plan = SamplePlan.from_json(d) if d else None
        self.update_results()

    def _live_keys(self, keys):
        return [k for k in keys if k in self.session.by_key]

    def show_in_list(self, keys=None, title=None):
        if self.plan is None:
            return
        keys = self._live_keys(keys if keys is not None else self.plan.keys)
        self.app.apply_file_filter(keys, title or f"🎯 QC sample '{self.plan.name}'")

    def next_unreviewed(self):
        if self.plan is None:
            return
        for k in self._live_keys(self.plan.keys):
            if self.session.review.status(k) is None:
                self.app.open_sample(k)
                return
        QMessageBox.information(self, self.TITLE, "Every file in this sample has been reviewed. 🎉")

    # ---------------------------------------------------------------- results
    def update_results(self, *_):
        if self.plan is None:
            self.verdict.setText("—")
            self.detail.setText("Create or pick a sample.")
            self.table.setRowCount(0)
            return
        conf = self.res_conf.currentData()
        max_err = self.max_err.value() / 100
        est, per = estimate(self.plan, self.session.review.status, conf, max_err)
        self.progress.setRange(0, max(est.target, 1))
        self.progress.setValue(est.reviewed)
        text, score = {
            "pass": ("✅ PASS", 1.0), "fail": ("❌ FAIL", 0.0),
            "undecided": ("⏳ Keep reviewing", 0.5), "not started": ("— not started", None),
        }[est.verdict]
        self.verdict.setText(text)
        if score is None:
            apply_style(self.verdict, "border:2px solid @line; border-radius:10px; padding:8px;")
        else:
            self.verdict.setStyleSheet(f"border:2px solid {tone(score, 'border')}; border-radius:10px; padding:8px;"
                                       f"background:{tone(score, 'bg')}; color:{tone(score, 'fg')};")
        pct = lambda v: f"{v * 100:.1f} %"
        if est.reviewed:
            why = {"pass": f"the upper bound {pct(est.high)} is within the {pct(max_err)} limit",
                   "fail": f"even the lower bound {pct(est.low)} exceeds the {pct(max_err)} limit",
                   "undecided": f"the interval straddles the {pct(max_err)} limit — review more files"}[est.verdict]
            self.detail.setText(f"Estimated error rate {pct(est.rate)}  ({int(conf * 100)} % CI {pct(est.low)} – "
                                f"{pct(est.high)})\n{est.flagged:,} flagged of {est.reviewed:,} reviewed · {why}")
        else:
            self.detail.setText(f"{est.target:,} files to review. Mark each with S (OK) or X (flag).")
        rows = sorted(per.items(), key=lambda kv: -kv[1][6])
        self.table.setRowCount(len(rows))
        for r, (h, (p, lo, hi, rev, flg, n, Nh)) in enumerate(rows):
            vals = (h, Nh, n, rev, flg, round(p * 100, 2) if rev else "", f"{pct(lo)} – {pct(hi)}" if rev else "")
            for c, v in enumerate(vals):
                it = QTableWidgetItem()
                if isinstance(v, (int, float)):
                    it.setData(Qt.ItemDataRole.DisplayRole, v)
                else:
                    it.setText(str(v))
                if c == 3:
                    it.setData(BAR_ROLE, (rev / max(n, 1), "#9cc3f5"))
                if c == 5 and rev:
                    sc = 1.0 - min(p / max(2 * max_err, 1e-6), 1.0)        # 0 → green, ≥ 2× limit → red
                    it.setData(BAR_ROLE, (min(p / max(2 * max_err, 1e-6), 1.0), gradient_hex(sc, 0.68)))
                self.table.setItem(r, c, it)

    def _stratum_activated(self, row, _col):
        if self.plan is None:
            return
        h = self.table.item(row, 0).text()
        keys = self.plan.strata.get(h, [])
        if keys:
            self.show_in_list(keys, f"🎯 '{self.plan.name}' · {h}")

    def closeEvent(self, event):
        try:
            self.session.reviewChanged.disconnect(self.update_results)
        except TypeError:
            pass
        self.app.forget_dialog(self)
        super().closeEvent(event)
