# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
🤖 Model check: run one or several YOLO models (in the environment chosen in 🤖 Models & Environment) and
rank likely label errors.

Optimisations
- models live in a persistent server process (loaded + warmed up once)
- batched inference (batch size from the setup), classes not mapped to any label are filtered out on the GPU
- raw detections (conf ≥ 0.05) cached per model version / imgsz / mode / image version in .aio_qc/cache.sqlite:
  changing thresholds, mappings or merge IoU re-uses them instantly
- coverage-aware matching: a label no model knows is never reported, and a class mismatch needs a model
  that knows both classes
"""
from __future__ import annotations

import os

from AIO_QC.core.dataset import DETECTION_TYPES

import random
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
    QMessageBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from AIO_QC.core.model_check import (
    MISMATCH, MISSED, MISSING, POOR_FIT, coverage, gt_boxes, match, merge_predictions, model_signature,
)
from AIO_QC.core.evaluation import (
    BACKGROUND, ERROR_TYPES, IOU_GRID, Record, View, box_quality, build, confusion, error_breakdown, make_seen_fn,
    report, slices,
)
from AIO_QC.core.sampling import ERROR_KINDS, Audit, create_audit, estimate_audit, finding_label
from AIO_QC.core.state import list_audits, save_audit
from AIO_QC.core.state import REVIEW_FLAG, load_models_config
from AIO_QC.utils import theme_color, hexc, style as apply_style
from AIO_QC.utils import emphasis_bg, get_color_icon, gradient_hex, heat, newButton, tone
from .base_scan import BAR_ROLE, BaseScanDialog, ResultRow, dspin, ispin, readable_table
from .card_gallery import Card, CardView, Panel, Shape, box_points
from .model_setup_dialog import model_color, server_specs

BULK_KINDS = (MISSING, MISMATCH, POOR_FIT)        # "missed by model" deletes labels: one at a time only

KIND_TONE = {MISMATCH: "error", MISSING: "error", POOR_FIT: "warning", MISSED: "warning"}
REPORT_COLS = [("label", "Label"), ("gt", "GT"), ("pred", "Pred"), ("tp", "TP"), ("fp", "FP"), ("fn", "FN"),
               ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("ap50", "AP50"),
               ("ap5095", "AP50-95"), ("ratio", "Pred / GT"), ("best_conf", "Best conf")]


def findings_rows(ctx, key, gt_rec, pred_rec, W, H, thr) -> list:
    """Per-instance findings for one image from the stored record (pure; runs in worker threads). Findings the user
    rejected earlier ("label is right") are hidden unless ctx["show_rejected"]."""
    from AIO_QC.core.fix_ledger import finding_key
    cov = ctx["cov"]
    rejected, show_rej, root = ctx.get("rejected") or set(), ctx.get("show_rejected", False), ctx.get("root")
    gt = [(g[0], g[1], g[2]) for g in gt_rec]
    preds = [(p[0], p[1], p[2], p[3]) for p in pred_rec]
    display = {g[0]: g[4] for g in gt_rec}
    gt_box = {g[0]: tuple(float(v) for v in g[2]) for g in gt_rec}
    rows = []
    for f in match(gt, preds, thr["iou"], thr["conf"], thr["missing"], thr["mismatch"], agnostic=ctx["agnostic"],
                   covered=set(cov), mismatch_ok=lambda gl, p: p[3] in cov.get(gl, set())):
        gl = display.get(f.gt_index, f.gt_label)
        f.gt_box = gt_box.get(f.gt_index)
        rk = finding_key(root, key, f, W, H) if root is not None else None
        is_rej = rk in rejected
        if is_rej and not show_rej:
            continue
        rows.append(ResultRow(key, (f.kind, gl, f.pred_label, ctx["names"].get(f.model, ""), round(f.conf, 3),
                                    round(f.iou, 3), "" if f.gt_index is None else f.gt_index, round(f.score, 3)),
                              shape_index=f.gt_index, payload=(f, W, H),
                              tone="muted" if is_rej else KIND_TONE.get(f.kind), dot="#9e9e9e" if is_rej else None))
    return rows


class ModelCheckDialog(BaseScanDialog):
    """
    ONE run, two views that share the live threshold panel:
      Findings — per-instance, actionable list (relabel / add box / delete / flag)
      Analysis — dataset-level GT vs predictions: report card, confusion matrix, insights; views by split and by
                 seen / unseen / out-of-fold (models' "Trained on" setting)
    """
    SUPPORTED = DETECTION_TYPES          # needs shapes (not for classification datasets)
    TITLE, ICON = "Model Check", "🤖"
    HELP = ("Compares one or more YOLO models with your labels. Findings: class mismatch, missing annotation, poor "
            "box fit, missed by model — ranked by likelihood of a real error. Analysis: per-class precision / recall / "
            "AP, confusion matrix and insights. Every threshold above re-computes instantly from the cached "
            "predictions — no re-inference. Labels no model covers are skipped.")
    COLUMNS = ["Issue", "Label", "Model says", "Model", "Conf", "IoU", "Shape", "Score"]
    FILTER_COLUMN = 0
    RESCAN_AFTER_EDIT = False

    def build_params(self, form):
        self.setup_info = QLabel()
        self.setup_info.setWordWrap(True)
        self.setup_info.setTextFormat(Qt.TextFormat.RichText)
        cfg = newButton("⚙ Models & environment…", slot=lambda: self.app.open_model_setup())
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.setup_info, 1)
        h.addWidget(cfg)
        self.agnostic = QCheckBox("Class-agnostic (only missing / extra boxes)")
        self.scope = QComboBox()
        self.scope.addItems(["All files", "Files in current list (e.g. a QC sample)"])
        form.addRow(row)
        form.addRow("Scope:", self.scope)
        form.addRow(self.agnostic)
        self.names_info = QLabel("")
        self.names_info.setWordWrap(True)
        apply_style(self.names_info, "color:@muted;")
        form.addRow(self.names_info)
        self.add_action("✓ Accept (Enter)", self.accept_rows,
                        "Apply what the finding suggests to the ticked (or selected) rows:\n"
                        "  class mismatch → relabel to the model's class\n"
                        "  missing annotation → add the model's box\n"
                        "  poor box fit → replace the box with the model's box\n"
                        "  missed by model → delete the labelled shape (the label is the error)\n"
                        "Backed up, undoable; the next finding opens automatically.")
        self.add_action("✓≥ Accept by confidence…", self.accept_by_confidence,
                        "Accept every finding the model is sure about in one go: choose the kinds (missing box, "
                        "class mismatch, poor fit) and the minimum confidence, see how many — then accept. Only the "
                        "rows the current filter shows; rejected findings are never included. Backed up, undoable.")
        self.add_action("✕ Reject (X)", self.reject_rows,
                        "The label is right, the model is wrong: hide this finding now and on every later run "
                        "(“Show rejected” brings them back)")
        self.undo_btn = self.add_action("↶ Undo", self.undo_accept, "Put back the files changed by the last Accept")
        self.undo_btn.setEnabled(False)
        self.add_action("🗑 Delete label shapes", lambda: self.delete_shape_rows(
            [r for r in self.target_rows() if r.shape_index is not None], "labelled shape(s)"))
        self.show_rejected = QCheckBox("Show rejected")
        self.show_rejected.setToolTip("Also list findings you rejected before (grey); “Forget” un-rejects the "
                                      "ticked ones")
        self.show_rejected.toggled.connect(self._toggle_rejected)
        self.actions_layout.addWidget(self.show_rejected)
        self.add_action("↺ Forget", self.unreject_rows, "Un-reject the ticked (or selected) grey rows")
        self.auto_next = QCheckBox("Open next")
        self.auto_next.setChecked(True)
        self.auto_next.setToolTip("After Accept / Reject, select the next finding and show it in the viewer")
        self.actions_layout.addWidget(self.auto_next)
        from AIO_QC.core.fix_ledger import RejectLedger
        self.ledger = RejectLedger(self.session.root)
        self._last_backup = None
        self.app.inference.changed.connect(self._refresh_setup_info)
        self._refresh_setup_info()

    # ---------------------------------------------------------- live panel + tabs
    def post_init(self):
        for keys, slot in ((("Return", "Enter"), self.accept_rows), (("X",), self.reject_rows)):
            for k in keys:
                sc = QShortcut(QKeySequence(k), self.table)
                sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                sc.activated.connect(slot)
        live = QWidget()
        lv = QHBoxLayout(live)
        lv.setContentsMargins(0, 0, 0, 0)
        self.conf = dspin(0.05, 0.95, 0.25, 2, 0.05)
        self.iou = QComboBox()
        for v in IOU_GRID:
            self.iou.addItem(f"{v:.2f}", float(v))
        self.iou.setCurrentIndex(int(np.argmin(np.abs(IOU_GRID - 0.5))))
        self.missing_conf = dspin(0.1, 0.99, 0.5, 2, 0.05)
        self.mismatch_conf = dspin(0.1, 0.99, 0.4, 2, 0.05)
        title = QLabel("⚡ Live thresholds")
        title.setStyleSheet("font-weight:700;")
        lv.addWidget(title)
        for text, w, tip in (("conf ≥", self.conf, "Predictions below this confidence are ignored"),
                             ("match IoU ≥", self.iou, "Same-class overlap needed to count as a match"),
                             ("missing ≥", self.missing_conf, "Findings: confidence to call an unlabeled detection"),
                             ("mismatch ≥", self.mismatch_conf, "Findings: confidence to call a class mismatch")):
            lab = QLabel(text)
            lab.setToolTip(tip)
            w.setToolTip(tip)
            lv.addWidget(lab)
            lv.addWidget(w)
        self.live_note = QLabel("")
        apply_style(self.live_note, "color:@muted;")
        lv.addWidget(self.live_note, 1)
        self.table_host.insertWidget(0, live)
        self.conf.valueChanged.connect(self._thresholds_changed)
        self.iou.currentIndexChanged.connect(self._thresholds_changed)
        self.missing_conf.valueChanged.connect(self._thresholds_changed)
        self.mismatch_conf.valueChanged.connect(self._thresholds_changed)
        self._refind = QTimer(self, singleShot=True, interval=350, timeout=self._recompute_findings)
        self._refind_worker = None

        self.init_marks()
        self.cards = CardView(self, n_panels=1)
        cards = QWidget()
        cl = QVBoxLayout(cards)
        cl.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Solid = your label · dashed = the model · click an image to open it · tick to select for the "
                      "actions above · order and filter follow the table")
        apply_style(hint, "color:@muted;")
        cl.addWidget(hint)
        cl.addWidget(self.cards, 1)
        self.tabs = QTabWidget()
        self.table_host.removeWidget(self.table)
        self.tabs.addTab(cards, "Cards")
        self.tabs.addTab(self.table, "Findings table")
        self.tabs.addTab(self._build_analysis(), "Analysis")
        self.table_host.addWidget(self.tabs)
        self.filter_box.currentIndexChanged.connect(lambda _i: self._refresh_cards())
        self.table.horizontalHeader().sortIndicatorChanged.connect(lambda *_: self._refresh_cards())

    def _build_analysis(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        top = QHBoxLayout()
        self.v_split = QComboBox()
        self.v_split.addItem("All splits", None)
        for sp in self.session.splits:
            self.v_split.addItem(sp, sp)
        self.v_group = QComboBox()
        for text, key in (("All images", "all"), ("Seen by the model (it trained on them)", "seen"),
                          ("Unseen (held-out)", "unseen"), ("Out-of-fold (only models that never saw the image)", "oof")):
            self.v_group.addItem(text, key)
        self.v_norm = QCheckBox("Row %")
        self.v_norm.setToolTip("Confusion matrix as % of each GT row")
        export = newButton("⤓ Report CSV", slot=lambda: self._export_report())
        for wd in (QLabel("View:"), self.v_split, self.v_group, self.v_norm):
            top.addWidget(wd)
        top.addStretch(1)
        top.addWidget(export)
        for wd in (self.v_split, self.v_group):
            wd.currentIndexChanged.connect(lambda _i: self._refresh_analysis())
        self.v_norm.toggled.connect(lambda _v: self._refresh_confusion())
        self.inner = QTabWidget()
        self.inner.addTab(self._build_overview(), "Report & confusion")
        self.inner.addTab(self._build_errors(), "Error types")
        self.inner.addTab(self._build_boxes(), "Box quality")
        self.inner.addTab(self._build_slices(), "Slices")
        self.inner.addTab(self._build_audit(), "Label-error estimate")
        self.inner.currentChanged.connect(lambda _i: self._refresh_current_inner())
        lay.addLayout(top)
        lay.addWidget(self.inner, 1)
        return w

    @staticmethod
    def _table(cols, tip=""):
        t = QTableWidget(0, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        readable_table(t, row_height=26)
        if tip:
            t.setToolTip(tip)
        return t

    @staticmethod
    def _note():
        lbl = QLabel("")
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        apply_style(lbl, "background:palette(base); border:1px solid @line; border-radius:6px; padding:8px;")
        return lbl

    def _build_overview(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.report_tbl = self._table([c[1] for c in REPORT_COLS],
                                      "Double-click a class: its missed labels (FN) in the crop gallery")
        self.report_tbl.cellDoubleClicked.connect(self._report_activated)
        self.conf_tbl = QTableWidget(0, 0)
        readable_table(self.conf_tbl, row_height=26)
        self.conf_tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.conf_tbl.setToolTip("Rows: your label · Columns: model's label · 'background' = no counterpart.\n"
                                 "Click: show the files · Double-click: open the labelled instances as crops")
        self.conf_tbl.cellClicked.connect(lambda r, c: self._cell_activated(r, c, crops=False))
        self.conf_tbl.cellDoubleClicked.connect(lambda r, c: self._cell_activated(r, c, crops=True))
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("<b>Report card</b>"))
        ll.addWidget(self.report_tbl, 1)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("<b>Confusion matrix</b> (rows = your labels, columns = model)"))
        rl.addWidget(self.conf_tbl, 1)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self.insights = self._note()
        self.insights.setText("Run a scan to see the analysis.")
        lay.addWidget(split, 1)
        lay.addWidget(self.insights)
        return w

    def _build_errors(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        cols = ["Label", "TP"] + [t[1] for t in ERROR_TYPES] + ["Errors", "Main issue"]
        self.err_tbl = self._table(cols, "Click a count: those files · double-click: the labelled instances as crops")
        self.err_tbl.cellClicked.connect(lambda r, c: self._err_activated(r, c, crops=False))
        self.err_tbl.cellDoubleClicked.connect(lambda r, c: self._err_activated(r, c, crops=True))
        legend = " · ".join(f"<b>{t[1]}</b>: {t[2]} → <i>{t[3]}</i>" for t in ERROR_TYPES)
        self.err_note = self._note()
        lay.addWidget(self.err_tbl, 1)
        lay.addWidget(self.err_note)
        help_ = QLabel(legend)
        help_.setWordWrap(True)
        apply_style(help_, "color:@muted;")
        lay.addWidget(help_)
        return w

    def _build_boxes(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        cols = ["Label", "Matched", "Mean IoU", "Median IoU", "IoU < 0.75", "Left %", "Top %", "Right %", "Bottom %",
                "Label / model area", "IoU distribution 0.5→1"]
        self.box_tbl = self._table(cols, "Double-click a class: its loosest-fitting labels first (crop gallery)")
        self.box_tbl.cellDoubleClicked.connect(self._box_activated)
        self.box_note = self._note()
        help_ = QLabel("Edge offsets are medians over matched pairs, as % of the label's size. <b>Positive</b> = the "
                       "label extends further than the model on that side (label looser); <b>negative</b> = the label "
                       "is tighter. Consistent offsets across a class point to an annotation habit or guideline gap.")
        help_.setWordWrap(True)
        apply_style(help_, "color:@muted;")
        lay.addWidget(self.box_tbl, 1)
        lay.addWidget(self.box_note)
        lay.addWidget(help_)
        return w

    def _build_slices(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        top = QHBoxLayout()
        self.slice_cls = QComboBox()
        self.slice_cls.addItem("All classes", None)
        self.slice_cls.currentIndexChanged.connect(lambda _i: self._refresh_slices())
        top.addWidget(QLabel("Class:"))
        top.addWidget(self.slice_cls)
        top.addStretch(1)
        self.slice_tbl = self._table(["Dimension", "Value", "Labels", "Recall", "Δ vs overall", "Predictions",
                                      "Precision"], "Double-click: the missed labels of that slice as crops")
        self.slice_tbl.cellDoubleClicked.connect(self._slice_activated)
        self.slice_note = self._note()
        lay.addLayout(top)
        lay.addWidget(self.slice_tbl, 1)
        lay.addWidget(self.slice_note)
        return w

    def _build_audit(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        intro = QLabel("Estimate how many labels are actually wrong <b>without judging every finding</b>: a small random "
                       "sample per class × finding type is reviewed (<b>Y</b> = real label error, <b>N</b> = the model "
                       "is wrong, <b>U</b> = undo). The finder's precision per stratum × its number of findings gives "
                       "the estimate with a confidence interval. 20–30 per stratum is a good start.")
        intro.setWordWrap(True)
        ctl = QHBoxLayout()
        self.audit_kinds = {}
        for kind in ERROR_KINDS:
            cb = QCheckBox(kind)
            cb.setChecked(True)
            self.audit_kinds[kind] = cb
            ctl.addWidget(cb)
        self.audit_n = ispin(5, 500, 20)
        self.audit_seed = ispin(0, 2_000_000_000, random.randint(1, 999_999))
        mk = newButton("🎲 Create audit sample", slot=lambda: self._create_audit())
        for text, wd in (("per stratum", self.audit_n), ("seed", self.audit_seed)):
            ctl.addWidget(QLabel(text))
            ctl.addWidget(wd)
        ctl.addWidget(mk)
        ctl.addStretch(1)
        pick = QHBoxLayout()
        self.audit_box = QComboBox()
        self.audit_box.currentIndexChanged.connect(lambda _i: self._load_audit())
        self.audit_follow = QCheckBox("Follow in viewer")
        self.audit_follow.setChecked(True)
        self.audit_follow.setToolTip("Selecting a row opens the finding in the main window")
        pick.addWidget(QLabel("Audit:"))
        pick.addWidget(self.audit_box, 1)
        pick.addWidget(self.audit_follow)
        for text, verdict in (("✔ Real error (Y)", "error"), ("✖ Model wrong (N)", "ok"), ("↺ Undo (U)", None)):
            b = newButton(text, slot=lambda v=verdict: self._set_verdict(v))
            pick.addWidget(b)
        self.audit_tbl = self._table(["Class", "Finding", "File", "Shape", "Model says", "Conf", "Verdict"],
                                     "Y / N / U to judge · Enter / double-click to open in the viewer")
        self.audit_tbl.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.audit_tbl.currentCellChanged.connect(lambda r, _c, _pr, _pc: self._audit_follow(r))
        self.audit_tbl.cellDoubleClicked.connect(lambda r, _c: self._open_audit_item(r))
        for key, verdict in (("Y", "error"), ("N", "ok"), ("U", None)):
            sc = QShortcut(QKeySequence(key), self.audit_tbl, activated=lambda v=verdict: self._set_verdict(v))
            sc.setContext(Qt.ShortcutContext.WidgetShortcut)
        sc = QShortcut(QKeySequence(Qt.Key.Key_Return), self.audit_tbl,
                       activated=lambda: self._open_audit_item(self.audit_tbl.currentRow()))
        sc.setContext(Qt.ShortcutContext.WidgetShortcut)
        self.audit_head = QLabel("")
        hf = self.audit_head.font()
        hf.setPointSizeF(max(hf.pointSizeF(), 9.0) + 4)
        hf.setBold(True)
        self.audit_head.setFont(hf)
        self.audit_head.setWordWrap(True)
        self.est_tbl = self._table(["Class", "Labels", "Findings", "Reviewed", "Confirmed", "Est. wrong labels",
                                    "Est. error rate", "Est. missing labels"])
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.audit_tbl)
        est = QWidget()
        el = QVBoxLayout(est)
        el.setContentsMargins(0, 0, 0, 0)
        el.addWidget(self.audit_head)
        el.addWidget(self.est_tbl, 1)
        split.addWidget(est)
        lay.addWidget(intro)
        lay.addLayout(ctl)
        lay.addLayout(pick)
        lay.addWidget(split, 1)
        self.audit = None
        self._refresh_audit_list()
        return w

    # ------------------------------------------------------------------ setup info
    def _models(self) -> list[dict]:
        return [m for m in load_models_config(self.session.root).get("models", []) if m.get("enabled", True)]

    def _refresh_setup_info(self):
        svc = self.app.inference
        models = self._models()
        env = Path(svc.python).parent.parent.name if svc.python else "none"
        state = "ready" if svc.ready else ("missing " + ", ".join(svc.missing()) if svc.info else "not checked")
        names = ", ".join(f"<span style='color:{model_color(i)}'><b>{m['name']}</b></span> "
                          f"<span style='color:{hexc('muted')}'>({(m.get('trained_on') or {}).get('mode', 'all')}, "
                          f"{m.get('runtime', 'pt')})</span>" for i, m in enumerate(models))
        self.setup_info.setText(f"Models: {names or '<i>none</i>'} · env <b>{env}</b> ({state}) · "
                                f"{svc.mode} · device {svc.opts['device']} · batch {svc.opts['batch']}")

    def samples_to_scan(self):
        if self.scope.currentIndex() == 1:
            keys = set(self.app.file_dock.visible_keys())
            return [s for s in self.session.samples if s.key in keys]
        return list(self.session.samples)

    def stop_scan(self):
        if self.ctx.get("cancel") is not None:
            self.ctx["cancel"].set()                  # interrupt a running batch
        super().stop_scan()

    def _thr(self) -> dict:
        return {"conf": self.conf.value(), "iou": float(self.iou.currentData()),
                "missing": self.missing_conf.value(), "mismatch": self.mismatch_conf.value()}

    def make_context(self):
        svc = self.app.inference
        models = self._models()
        if not models:
            self.app.open_model_setup()
            raise ValueError("Add at least one model in ⚙ Models & environment.")
        if not svc.python:
            self.app.open_model_setup()
            raise ValueError("Choose a Python environment in ⚙ Models & environment.")
        agnostic = self.agnostic.isChecked()
        if agnostic and any(not m.get("names") for m in models):
            raise ValueError("Load the models once in ⚙ Models & environment (class names are needed).")
        unmapped = [m["name"] for m in models if not any((m.get("class_map") or {}).values())]
        if unmapped and not agnostic:
            raise ValueError("No classes mapped to dataset labels for: " + ", ".join(unmapped) +
                             "\nLoad the models in ⚙ Models & environment (auto-maps by name) or use class-agnostic.")
        specs = server_specs(models, svc.opts, svc.info)
        if not specs:
            raise ValueError("None of the enabled model files exist.")
        maps = {m["id"]: dict(m.get("class_map") or {}) for m in models}
        if agnostic:
            maps = {m["id"]: {k: "object" for k in (m.get("names") or {})} for m in models}
        cov = coverage({sp["id"]: maps[sp["id"]] for sp in specs})
        classes = {mid: sorted(int(k) for k, v in cmap.items() if v) for mid, cmap in maps.items()}
        order = self.samples_to_scan()
        seen_fns = {}
        for m in models:
            try:
                seen_fns[m["id"]] = make_seen_fn(m.get("trained_on"))
            except OSError as exc:
                raise ValueError(f"{m['name']}: cannot read its training list — {exc}")
        ctx = {"svc": svc, "specs": specs, "ids": [sp["id"] for sp in specs], "maps": maps, "cov": cov,
               "sigs": {sp["id"]: model_signature(sp) for sp in specs}, "classes": classes,
               "names": {m["id"]: m["name"] for m in models},
               "colors": {m["id"]: model_color(i) for i, m in enumerate(models)},
               "adapter": self.session.adapter, "cache": self.session.cache, "loaded": False,
               "order": order, "pos": {s.key: i for i, s in enumerate(order)}, "buffer": {},
               "batch": int(svc.opts["batch"]), "nms": float(svc.opts["nms_iou"]), "cancel": threading.Event(),
               "agnostic": agnostic, "seen_fns": seen_fns, "records": [], "wh": {}, "eval": None,
               "model_names": Counter(), "gt_names": Counter(), "cached": 0, "ran": 0, "load_error": None,
               "rejected": self.ledger.keys(), "show_rejected": self.show_rejected.isChecked(),
               "root": self.session.root}
        ctx["shared"] = self._shared_predictions(models, specs)
        ctx.update(self._thr())
        return ctx

    def _shared_predictions(self, models, specs):
        """{model id: (PredCache, key)} - Testing's predictions of the same weights / size / SAHI in the dataset's
        shared cache (one prediction cache per dataset); read-only here."""
        from AIO_QC.core import model_registry as R
        out = {}
        by_id = {m["id"]: m for m in models}
        for sp in specs:
            m = by_id.get(sp["id"])
            if m is None or os.path.abspath(sp["path"]) != os.path.abspath(m.get("path", "")):
                continue                                   # running an ONNX / TensorRT export: other numbers
            hit = R.shared_lookup(self.session.root, m["path"], sp.get("imgsz"), sp.get("sahi"))
            if hit is not None:
                out[sp["id"]] = hit
        return out

    # ------------------------------------------------------------- worker side
    def _prefetch(self, ctx, start_key):
        """Fill ctx['buffer'] for the next `batch` samples: cached predictions + one batched server call."""
        order, i0 = ctx["order"], ctx["pos"][start_key]
        window = order[i0:i0 + ctx["batch"]]
        need_img, per = [], {}
        cache = ctx["cache"]
        for s in window:
            if s.key in ctx["buffer"]:
                continue
            try:
                ann = ctx["adapter"].load(s, light=True)
            except Exception as exc:
                ctx["buffer"][s.key] = ("error", str(exc))
                continue
            img = ann.image_path or s.image_path
            if img is None:
                ctx["buffer"][s.key] = ("error", "no image")
                continue
            got, missing = {}, []
            wh = None
            for mid in ctx["ids"]:
                hit = cache.get_preds(ctx["sigs"][mid], img) if cache is not None else None
                if hit is None and mid in ctx.get("shared", {}):
                    hit = self._from_shared(ctx, mid, img)
                    if hit is not None and cache is not None:
                        cache.put_preds(ctx["sigs"][mid], img, hit)
                if hit is None:
                    missing.append(mid)
                else:
                    got[mid] = hit["boxes"]
                    wh = (hit["w"], hit["h"])
            per[s.key] = (ann, img, got, missing, wh)
            if missing:
                need_img.append(str(img))
            else:
                ctx["cached"] += 1
        if need_img:
            if not ctx["loaded"]:
                ctx["svc"].ensure_loaded(ctx["specs"])
                ctx["loaded"] = True
            need_ids = sorted({mid for v in per.values() for mid in v[3]})
            res = ctx["svc"].predict(need_img, need_ids, conf=0.05, batch=ctx["batch"],
                                     classes={m: ctx["classes"].get(m) for m in need_ids}, cancel=ctx["cancel"])
            ctx["ran"] += len(need_img)
            for key, (ann, img, got, missing, wh) in list(per.items()):
                r = res.get(str(img))
                if not missing or r is None:
                    continue
                if r.get("error") and not r.get("w"):
                    ctx["buffer"][key] = ("error", r["error"])
                    per[key] = None
                    continue
                for mid in missing:
                    boxes = r["by_model"].get(mid) or []
                    got[mid] = boxes
                    if cache is not None and r.get("w"):
                        cache.put_preds(ctx["sigs"][mid], img, {"w": r["w"], "h": r["h"], "boxes": boxes})
                per[key] = (ann, img, got, missing, (r["w"], r["h"]))
        for key, v in per.items():
            if v is not None:
                ctx["buffer"][key] = ("ok", v)

    @staticmethod
    def _from_shared(ctx, mid, img):
        """A Testing prediction (stored down to conf 0.001) as Model Check's {w, h, boxes} (conf >= 0.05, only the
        mapped classes)."""
        from AIO_QC.core.model_registry import qc_image_key
        from AIO_QC.core.testing.dataset import image_size
        pc, key = ctx["shared"][mid]
        got = pc.get_many(key, [qc_image_key(str(img), str(ctx["root"]))])
        if not got:
            return None
        arr = next(iter(got.values()))[0]
        w, h = image_size(str(img))
        if not w or not h:
            return None
        keep = set(ctx["classes"].get(mid) or [])
        boxes = [[float(a), float(b), float(c), float(d), float(cf), int(k)] for a, b, c, d, cf, k in arr.tolist()
                 if cf >= 0.05 and (not keep or int(k) in keep)]
        return {"w": w, "h": h, "boxes": boxes}

    def scan_sample(self, ctx, s):
        if ctx["load_error"]:
            return []
        if s.key not in ctx["buffer"]:
            try:
                self._prefetch(ctx, s.key)
            except Exception as exc:
                if not ctx["loaded"]:
                    ctx["load_error"] = str(exc)            # environment / model problem: report once
                raise
        status, v = ctx["buffer"].pop(s.key, ("error", "no prediction"))
        if status == "error":
            raise RuntimeError(v)
        ann, _img, got, _missing, wh = v
        W, H = wh or (ann.width, ann.height)
        preds = merge_predictions(got, ctx["maps"], ctx["nms"])
        for p in preds:
            ctx["model_names"][p[0]] += 1
        seen = {mid: bool(fn(s)) for mid, fn in ctx["seen_fns"].items()}
        cov = ctx["cov"]
        gt_rec = []
        for idx, label, box in gt_boxes(ann.shapes, W, H, ann.normalized):
            ev_label = "object" if ctx["agnostic"] else label
            ctx["gt_names"][label] += 1
            if ev_label not in cov:
                continue                                     # no model covers this label
            states = {seen.get(mid, True) for mid in cov[ev_label]}
            state = 1 if states == {True} else (0 if states == {False} else -1)
            gt_rec.append((idx, ev_label, box, state, label))
        pred_rec = [(p[0], p[1], p[2], p[3], seen.get(p[3], True)) for p in preds]
        light = None
        if ctx["cache"] is not None:
            hit = ctx["cache"].get_image(_img)
            light = hit.get("light") if hit else None
        ctx["records"].append(Record(s.key, s.split, gt_rec, pred_rec, W, H, s.folder, light))
        ctx["wh"][s.key] = (W, H)
        return findings_rows(ctx, s.key, gt_rec, pred_rec, W, H, ctx)

    def finalize(self, ctx):
        if ctx["cache"] is not None:
            ctx["cache"].flush()
        if ctx["load_error"]:
            raise RuntimeError(f"Inference unavailable — {ctx['load_error']}")
        labels = ["object"] if ctx["agnostic"] else sorted(ctx["cov"])
        t0 = time.monotonic()
        ctx["eval"] = build(ctx["records"], labels, ctx["ids"])
        ctx["eval_s"] = time.monotonic() - t0
        return []

    def on_scan_complete(self, ctx):
        self.table.sortItems(2 + self.columns.index("Score"), Qt.SortOrder.DescendingOrder)
        uncovered = sorted(l for l in ctx["gt_names"] if l not in ctx["cov"]) if not ctx["agnostic"] else []
        notes = []
        if uncovered:
            notes.append(f"Not checked (no model covers them): {', '.join(uncovered)}")
        notes.append(f"Inference on {ctx['ran']:,} image(s), {ctx['cached']:,} fully from cache · "
                     f"analysis prepared in {ctx.get('eval_s', 0):.1f}s.")
        self.names_info.setText("\n".join(notes))
        d = ctx.get("eval")
        if d is not None:
            has_seen = bool((d.gt_seen == 1).any())
            has_unseen = bool((d.gt_seen == 0).any() or (~d.p_seen).any())
            model = self.v_group.model()
            model.item(1).setEnabled(has_seen)
            model.item(2).setEnabled(has_unseen)
            model.item(3).setEnabled(bool(d.gt_oof.any()))
            if self.v_group.currentIndex() and not model.item(self.v_group.currentIndex()).isEnabled():
                self.v_group.setCurrentIndex(0)
        self._refresh_analysis()

    # --------------------------------------------------------------- live changes
    def _thresholds_changed(self, *_):
        if not self.ctx or self.ctx.get("eval") is None or (self.worker is not None and self.worker.isRunning()):
            return
        self.ctx.update(self._thr())
        self._refresh_analysis()                 # instant (vectorised)
        self.live_note.setText("updating findings…")
        self._refind.start()                     # findings re-matched in the background, debounced

    def _recompute_findings(self):
        self._find_gen = getattr(self, "_find_gen", 0) + 1
        gen = self._find_gen
        ctx = dict(self.ctx)
        recs = list(ctx["records"])

        def run():
            rows = []
            for r in recs:
                W, H = ctx["wh"].get(r.key, (None, None))
                rows.extend(findings_rows(ctx, r.key, r.gt, r.preds, W, H, ctx))
            return rows

        def done(rows):
            if gen != self._find_gen:
                return                          # a newer threshold change superseded this result
            self.table.setRowCount(0)
            self.rows = []
            self._append_rows(rows)
            self.table.sortItems(2 + self.columns.index("Score"), Qt.SortOrder.DescendingOrder)
            self._rebuild_filter()
            self.summary.setText(self.summarize())
            self.live_note.setText(f"{len(rows):,} findings at the current thresholds")

        self.app.inference.run_async(run, done, lambda e: self.live_note.setText(f"❌ {e}"))

    # ------------------------------------------------------------------ analysis
    def _view(self) -> View:
        g = self.v_group.currentData()
        return View(split=self.v_split.currentData(), seen="all" if g == "oof" else g, oof=g == "oof")

    def _refresh_analysis(self):
        d = self.ctx.get("eval") if self.ctx else None
        if d is None:
            return
        thr = self._thr()
        v = self._view()
        rows = report(d, thr["conf"], thr["iou"], v)
        self._last_report = rows
        self.report_tbl.setRowCount(len(rows))
        for r, row in enumerate(rows):
            is_all = row["label"].startswith("ALL")
            for c, (key, _title) in enumerate(REPORT_COLS):
                val = row[key]
                it = QTableWidgetItem()
                if isinstance(val, float):
                    if val != val:
                        it.setText("")
                    else:
                        it.setData(Qt.ItemDataRole.DisplayRole, round(val, 3))
                elif isinstance(val, (int, np.integer)):
                    it.setData(Qt.ItemDataRole.DisplayRole, int(val))
                else:
                    it.setText(str(val))
                if key in ("precision", "recall", "f1", "ap50", "ap5095") and isinstance(val, float) and val == val:
                    it.setData(BAR_ROLE, (val, gradient_hex(val, 0.68)))
                if key == "label" and not is_all:
                    f1 = row["f1"] if row["f1"] == row["f1"] else 0.0
                    it.setIcon(get_color_icon(QColor(gradient_hex(f1, 0.55, 0.70))))    # status dot, theme-safe
                if is_all:
                    it.setBackground(emphasis_bg(self.report_tbl))
                self.report_tbl.setItem(r, c, it)
        self.report_tbl.resizeColumnsToContents()
        self._refresh_confusion()
        self.insights.setText(self._insights(d, rows, thr, v))
        if self.slice_cls.count() != len(d.labels) + 1:
            self.slice_cls.blockSignals(True)
            self.slice_cls.clear()
            self.slice_cls.addItem("All classes", None)
            for l in d.labels:
                self.slice_cls.addItem(l, l)
            self.slice_cls.blockSignals(False)
        self._refresh_current_inner()

    def _refresh_confusion(self):
        d = self.ctx.get("eval") if self.ctx else None
        if d is None:
            return
        thr = self._thr()
        mat, cells = confusion(d, thr["conf"], thr["iou"], self._view())
        self._cells = cells
        names = d.labels + [BACKGROUND]
        n = len(names)
        self.conf_tbl.setRowCount(n)
        self.conf_tbl.setColumnCount(n)
        self.conf_tbl.setHorizontalHeaderLabels(names)
        self.conf_tbl.setVerticalHeaderLabels(names)
        self.conf_tbl.verticalHeader().setVisible(True)
        norm = self.v_norm.isChecked()
        for i in range(n):
            row_sum = max(int(mat[i].sum()), 1)
            for j in range(n):
                v = int(mat[i, j])
                it = QTableWidgetItem(f"{v / row_sum * 100:.1f}%" if norm and v else (str(v) if v else ""))
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if v:
                    share = v / row_sum
                    if i == j:
                        it.setBackground(QColor(heat(1.0, share)))
                    elif i == n - 1 and j == n - 1:
                        pass
                    else:
                        it.setBackground(QColor(heat(0.0, min(share * 3, 1.0))))
                    it.setToolTip(f"{names[i]} → {names[j]}: {v:,} ({share * 100:.1f}% of row)")
                self.conf_tbl.setItem(i, j, it)
        self.conf_tbl.resizeColumnsToContents()

    # ------------------------------------------------------------ inner analysis tabs
    def _refresh_current_inner(self):
        if not self.ctx or self.ctx.get("eval") is None:
            return
        {1: self._refresh_errors, 2: self._refresh_boxes, 3: self._refresh_slices,
         4: self._refresh_estimate}.get(self.inner.currentIndex(), lambda: None)()

    @staticmethod
    def _num(v, digits=3):
        it = QTableWidgetItem()
        if isinstance(v, float):
            if v == v:
                it.setData(Qt.ItemDataRole.DisplayRole, round(v, digits))
        elif isinstance(v, (int, np.integer)):
            it.setData(Qt.ItemDataRole.DisplayRole, int(v))
        else:
            it.setText(str(v))
        return it

    def _refresh_errors(self):
        d = self.ctx["eval"]
        thr = self._thr()
        eb = error_breakdown(d, thr["conf"], thr["iou"], self._view())
        self._err = eb
        per, tp = eb["per_class"], eb["tp"]
        labels = d.labels + ["ALL"]
        mat = np.vstack([per, per.sum(0, keepdims=True)]) if len(per) else np.zeros((1, len(ERROR_TYPES)), int)
        tps = list(tp) + [int(tp.sum())]
        self.err_tbl.setRowCount(len(labels))
        for r, label in enumerate(labels):
            row = mat[r]
            total = int(row.sum())
            self.err_tbl.setItem(r, 0, QTableWidgetItem(label))
            self.err_tbl.setItem(r, 1, self._num(int(tps[r])))
            for c in range(len(ERROR_TYPES)):
                v = int(row[c])
                it = self._num(v)
                if v and total:
                    share = v / total
                    it.setData(BAR_ROLE, (share, gradient_hex(1 - min(share * 1.5, 1.0), 0.72)))
                    it.setToolTip(f"{v:,} = {share * 100:.1f}% of this row's errors")
                self.err_tbl.setItem(r, 2 + c, it)
            self.err_tbl.setItem(r, 2 + len(ERROR_TYPES), self._num(total))
            main = ERROR_TYPES[int(row.argmax())] if total else None
            self.err_tbl.setItem(r, 3 + len(ERROR_TYPES), QTableWidgetItem(f"{main[1]} → {main[3]}" if main else "—"))
            if label == "ALL":
                for c in range(self.err_tbl.columnCount()):
                    if self.err_tbl.item(r, c):
                        self.err_tbl.item(r, c).setBackground(emphasis_bg(self.err_tbl))
        self.err_tbl.resizeColumnsToContents()
        tot = mat[-1]
        allsum = int(tot.sum())
        if allsum:
            order = np.argsort(-tot)[:3]
            self.err_note.setText("Where the disagreement comes from: " + " · ".join(
                f"<b>{ERROR_TYPES[i][1]}</b> {tot[i] / allsum * 100:.0f}% (<i>{ERROR_TYPES[i][3]}</i>)"
                for i in order if tot[i]))
        else:
            self.err_note.setText("No errors at these thresholds. 🎉")

    def _err_activated(self, r, c, crops):
        eb = getattr(self, "_err", None)
        d = self.ctx.get("eval") if self.ctx else None
        if eb is None or d is None or not 2 <= c < 2 + len(ERROR_TYPES):
            return
        key = ERROR_TYPES[c - 2][0]
        classes = range(len(d.labels)) if r >= len(d.labels) else [r]
        members = [m for cl in classes for m in eb["members"].get((cl, key), [])]
        if not members:
            return
        gt, files = self._members(members)
        name = "all classes" if r >= len(d.labels) else d.labels[r]
        if crops and gt:
            self.app.open_crop_instances(gt, f"{ERROR_TYPES[c - 2][1]} errors · {name}")
        else:
            self.app.apply_file_filter(files, f"🤖 {ERROR_TYPES[c - 2][1]} errors · {name}")

    def _refresh_boxes(self):
        d = self.ctx["eval"]
        thr = self._thr()
        rows = box_quality(d, thr["conf"], thr["iou"], self._view())
        self._box = rows
        spark = "▁▂▃▄▅▆▇█"
        self.box_tbl.setRowCount(len(rows))
        habits = []
        for r, row in enumerate(rows):
            self.box_tbl.setItem(r, 0, QTableWidgetItem(row["label"]))
            self.box_tbl.setItem(r, 1, self._num(row["n"]))
            if not row["n"]:
                for c in range(2, self.box_tbl.columnCount()):
                    self.box_tbl.setItem(r, c, QTableWidgetItem(""))
                continue
            for c, key in ((2, "mean_iou"), (3, "median_iou")):
                it = self._num(row[key])
                it.setData(BAR_ROLE, (row[key], gradient_hex((row[key] - 0.5) * 2, 0.7)))
                self.box_tbl.setItem(r, c, it)
            it = self._num(round(row["loose"] * 100, 1))
            it.setData(BAR_ROLE, (row["loose"], gradient_hex(1 - row["loose"] * 2, 0.72)))
            self.box_tbl.setItem(r, 4, it)
            for c, key in ((5, "left"), (6, "top"), (7, "right"), (8, "bottom")):
                v = row[key]
                it = self._num(round(v, 1), 1)
                if v >= 3:
                    it.setForeground(theme_color("warn"))
                elif v <= -3:
                    it.setForeground(theme_color("info"))
                self.box_tbl.setItem(r, c, it)
                if abs(v) >= 3 and row["label"] != "ALL" and row["n"] >= 20:
                    habits.append(f"<b>{row['label']}</b> {key} {'looser' if v > 0 else 'tighter'} by {abs(v):.1f}%")
            self.box_tbl.setItem(r, 9, self._num(round(row["area"], 3)))
            h = row["hist"]
            mx = max(h) or 1
            self.box_tbl.setItem(r, 10, QTableWidgetItem("".join(spark[min(7, int(v / mx * 7.999))] for v in h)))
            if row["label"] == "ALL":
                for c in range(self.box_tbl.columnCount()):
                    if self.box_tbl.item(r, c):
                        self.box_tbl.item(r, c).setBackground(emphasis_bg(self.box_tbl))
        self.box_tbl.resizeColumnsToContents()
        self.box_note.setText(("📐 Systematic edge offsets (annotation habit): " + ", ".join(habits[:8]))
                              if habits else "No systematic edge offsets ≥ 3% on classes with ≥ 20 matches.")

    def _box_activated(self, r, _c):
        rows = getattr(self, "_box", [])
        d = self.ctx.get("eval") if self.ctx else None
        if d is None or not 0 <= r < len(rows) or not rows[r].get("n"):
            return
        gt = [(d.keys[d.gt_img[g]], int(d.gt_idx[g])) for g, _iou in rows[r]["worst"]]
        self.app.open_crop_instances(gt, f"{rows[r]['label']}: loosest-fitting labels first")

    def _refresh_slices(self):
        d = self.ctx["eval"]
        thr = self._thr()
        rows = slices(d, thr["conf"], thr["iou"], self._view(), self.slice_cls.currentData())
        self._slices = rows
        self.slice_tbl.setRowCount(len(rows))
        weak = []
        last_dim = None
        for r, row in enumerate(rows):
            dim = QTableWidgetItem(row["dimension"] if row["dimension"] != last_dim else "")
            last_dim = row["dimension"]
            f = dim.font()
            f.setBold(True)
            dim.setFont(f)
            self.slice_tbl.setItem(r, 0, dim)
            self.slice_tbl.setItem(r, 1, QTableWidgetItem(row["value"]))
            self.slice_tbl.setItem(r, 2, self._num(row["gt"]))
            it = self._num(row["recall"])
            if row["recall"] == row["recall"]:
                it.setData(BAR_ROLE, (row["recall"], gradient_hex(row["recall"], 0.68)))
            self.slice_tbl.setItem(r, 3, it)
            dl = self._num(row["delta"])
            if row["delta"] == row["delta"]:
                dl.setForeground(theme_color("error" if row["delta"] < -0.1 else ("ok" if row["delta"] > 0.05 else "muted")))
                if row["delta"] < -0.1 and row["gt"] >= 20:
                    weak.append((row["delta"], row))
            self.slice_tbl.setItem(r, 4, dl)
            self.slice_tbl.setItem(r, 5, self._num(row["pred"]))
            pr = self._num(row["precision"])
            if row["precision"] == row["precision"]:
                pr.setData(BAR_ROLE, (row["precision"], gradient_hex(row["precision"], 0.68)))
            self.slice_tbl.setItem(r, 6, pr)
        self.slice_tbl.resizeColumnsToContents()
        if weak:
            self.slice_note.setText("🔎 Weakest slices (recall ≥ 10 pts below overall, ≥ 20 labels): " + ", ".join(
                f"<b>{w['dimension']}: {w['value']}</b> {w['recall']:.2f} ({dl:+.2f})"
                for dl, w in sorted(weak, key=lambda x: x[0])[:5]) +
                "<br><i>In-sample models: likely inconsistent labels in that slice. Held-out: also a data gap.</i>")
        else:
            self.slice_note.setText("No slice falls more than 10 points below the overall recall.")

    def _slice_activated(self, r, _c):
        rows = getattr(self, "_slices", [])
        if not 0 <= r < len(rows) or not rows[r]["members"]:
            return
        gt, _files = self._members(rows[r]["members"])
        self.app.open_crop_instances(gt, f"missed labels · {rows[r]['dimension']}: {rows[r]['value']}")

    # ------------------------------------------------------------- label-error audit
    def _refresh_audit_list(self, select: str | None = None):
        self.audit_box.blockSignals(True)
        self.audit_box.clear()
        for a in list_audits(self.session.root):
            n = len(a["items"])
            done = sum(1 for it in a["items"] if a.get("verdicts", {}).get(it["id"]))
            self.audit_box.addItem(f"{a['name']}  ({done}/{n} reviewed · {a['created']})", a)
        self.audit_box.blockSignals(False)
        if select is not None:
            for i in range(self.audit_box.count()):
                if self.audit_box.itemData(i)["name"] == select:
                    self.audit_box.setCurrentIndex(i)
                    break
        self._load_audit()

    def _create_audit(self):
        if not self.ctx or self.ctx.get("eval") is None:
            QMessageBox.information(self, self.TITLE, "Run a scan first.")
            return
        kinds = {k for k, cb in self.audit_kinds.items() if cb.isChecked()}
        if not kinds:
            return
        thr = self._thr()
        ctx = dict(self.ctx)
        findings = []
        for rec in ctx["records"]:
            W, H = ctx["wh"].get(rec.key, (None, None))
            for row in findings_rows(ctx, rec.key, rec.gt, rec.preds, W, H, ctx):
                f = row.payload[0]
                box = [round(float(v), 1) for v in f.box] if f.box else None
                findings.append({"id": f"{rec.key}|{f.kind}|{f.gt_index}|{box}", "key": rec.key, "kind": f.kind,
                                 "label": row.cells[1], "shape_index": f.gt_index, "box": box,
                                 "pred_label": f.pred_label, "conf": round(float(f.conf), 3),
                                 "model": ctx["names"].get(f.model, "")})
        d = ctx["eval"]
        gt_counts = {r["label"]: r["gt"] for r in report(d, thr["conf"], thr["iou"], View()) if not r["label"].startswith("ALL")}
        name = time.strftime("audit_%Y%m%d_%H%M%S")
        a = create_audit(name, findings, gt_counts, self.audit_n.value(), self.audit_seed.value(), kinds,
                         thr, time.strftime("%Y-%m-%d %H:%M"))
        if not a.items:
            QMessageBox.information(self, self.TITLE, "No findings of the selected kinds at these thresholds.")
            return
        save_audit(self.session.root, a.to_json())
        self.audit_seed.setValue(random.randint(1, 999_999))
        self._refresh_audit_list(select=name)
        self.audit_tbl.setFocus()
        self.audit_tbl.selectRow(0)

    def _load_audit(self):
        data = self.audit_box.currentData()
        self.audit = Audit.from_json(data) if data else None
        items = self.audit.items if self.audit else []
        self.audit_tbl.setRowCount(len(items))
        for r, it in enumerate(items):
            s = self.session.by_key.get(it["key"])
            vals = (finding_label(it), it["kind"], s.display if s else it["key"],
                    "" if it["shape_index"] is None else it["shape_index"], it.get("pred_label", ""),
                    it.get("conf", ""))
            for c, v in enumerate(vals):
                self.audit_tbl.setItem(r, c, self._num(v))
            self._paint_verdict(r)
        self.audit_tbl.resizeColumnsToContents()
        self._refresh_estimate()

    def _paint_verdict(self, r):
        it = self.audit.items[r]
        v = self.audit.verdicts.get(it["id"])
        cell = QTableWidgetItem({"error": "✔ real error", "ok": "✖ model wrong"}.get(v, "—"))
        cell.setForeground(theme_color({"error": "error", "ok": "ok"}.get(v, "faint")))
        self.audit_tbl.setItem(r, 6, cell)

    def _set_verdict(self, verdict):
        if self.audit is None:
            return
        r = self.audit_tbl.currentRow()
        if not 0 <= r < len(self.audit.items):
            return
        iid = self.audit.items[r]["id"]
        if verdict is None:
            self.audit.verdicts.pop(iid, None)
        else:
            self.audit.verdicts[iid] = verdict
        save_audit(self.session.root, self.audit.to_json())
        self._paint_verdict(r)
        self._refresh_estimate()
        if verdict is not None:                     # jump to the next unjudged finding
            for nr in list(range(r + 1, len(self.audit.items))) + list(range(0, r)):
                if self.audit.items[nr]["id"] not in self.audit.verdicts:
                    self.audit_tbl.selectRow(nr)
                    break

    def _audit_follow(self, r):
        if self.audit_follow.isChecked():
            self._open_audit_item(r)

    def _open_audit_item(self, r):
        if self.audit is None or not 0 <= r < len(self.audit.items):
            return
        it = self.audit.items[r]
        box = it.get("box")
        overlay = [{"box": box, "text": f"model: {it.get('pred_label', '')} {it.get('conf', '')}",
                    "color": "#ff2bd6"}] if box else []
        self.app.open_sample(it["key"], it["shape_index"], overlay=overlay,
                             focus_box=box if it["shape_index"] is None else None)

    def _refresh_estimate(self):
        if self.audit is None:
            self.est_tbl.setRowCount(0)
            self.audit_head.setText("Create an audit sample to estimate the label-error rate.")
            self.audit_head.setStyleSheet("")
            return
        rows, tot = estimate_audit(self.audit)
        fmt = lambda t, pct=False: (f"{t[0] * 100:.1f}%  [{t[1] * 100:.1f}–{t[2] * 100:.1f}]" if pct
                                    else f"{t[0]:,.0f}  [{t[1]:,.0f}–{t[2]:,.0f}]")
        self.est_tbl.setRowCount(len(rows))
        for r, c in enumerate(rows):
            vals = (c["label"], c["labels"], c["findings"], f"{c['reviewed']}/{c['sampled']}", c["confirmed"],
                    fmt(c["err"]), fmt(c["rate"], True), fmt(c["missing"]))
            for k, v in enumerate(vals):
                it = self._num(v)
                if k == 6:
                    score = 1 - min(c["rate"][0] / 0.10, 1.0)            # 0 % green … ≥ 10 % red
                    it.setData(BAR_ROLE, (min(c["rate"][0] / 0.10, 1.0), gradient_hex(score, 0.7)))
                if not c["complete"]:
                    it.setForeground(theme_color("muted"))
                self.est_tbl.setItem(r, k, it)
        self.est_tbl.resizeColumnsToContents()
        done = tot["reviewed"]
        if not done:
            self.audit_head.setText(f"{tot['sampled']} findings to review — select the first row and press Y / N.")
            self.audit_head.setStyleSheet("")
            return
        score = 1 - min(tot["rate"][0] / 0.10, 1.0)
        self.audit_head.setText(
            f"Estimated label-error rate: {tot['rate'][0] * 100:.2f}%  (95% CI {tot['rate'][1] * 100:.2f}–"
            f"{tot['rate'][2] * 100:.2f}%) · ≈{tot['err'][0]:,.0f} wrong labels · ≈{tot['missing'][0]:,.0f} missing"
            f"  —  {done}/{tot['sampled']} reviewed" + ("" if done >= tot["sampled"] else " (grey rows still incomplete)"))
        self.audit_head.setStyleSheet(f"color:{tone(score, 'fg')}; background:{tone(score, 'bg')};"
                                      "border-radius:6px; padding:6px;")

    def _insights(self, d, rows, thr, v) -> str:
        cls_rows = [r for r in rows if not r["label"].startswith("ALL") and r["gt"]]
        if not cls_rows:
            return "No labelled objects in this view."
        seen_share = float((d.gt_seen == 1).mean()) if len(d.gt_seen) else 0.0
        out = []
        if v.oof:
            out.append("🧪 <b>Out-of-fold view</b> — every image is judged only by models that never trained on it: "
                       "the most reliable label-error signal.")
        elif seen_share > 0.8 and v.seen != "unseen":
            out.append("🧠 <b>The models trained on these images.</b> A model normally reproduces its own training "
                       "labels, so low agreement here points to <b>inconsistent or wrong labels</b> (or genuinely "
                       "ambiguous objects), not to a weak model.")
        elif v.seen == "unseen" or seen_share < 0.2:
            out.append("🔍 <b>Held-out view.</b> Disagreement mixes model errors and label errors; look for patterns "
                       "(one class, one confusion) rather than single misses.")
        low = [r for r in sorted((r for r in cls_rows if r["recall"] == r["recall"]), key=lambda r: r["recall"])
               if r["recall"] < 0.95][:3]
        if low:
            out.append("📉 Lowest recall (labels the models could not confirm): " + ", ".join(
                f"<b>{r['label']}</b> {r['recall']:.2f} ({r['fn']:,} missed)" for r in low))
        else:
            out.append("📈 Every class has recall ≥ 0.95 at these thresholds.")
        # confusions
        mat, _ = confusion(d, thr["conf"], thr["iou"], v)
        n = len(d.labels)
        pairs = [(int(mat[i, j]), d.labels[i], d.labels[j], mat[i].sum()) for i in range(n) for j in range(n)
                 if i != j and mat[i, j]]
        if pairs:
            top = sorted(pairs, reverse=True)[:3]
            out.append("🔀 Most confused (likely mislabels between these classes): " + ", ".join(
                f"<b>{a} → {b}</b> {c:,} ({c / max(t, 1) * 100:.1f}% of {a})" for c, a, b, t in top))
        hi_fp = [(r["label"], r["fp"], r["precision"]) for r in cls_rows if r["precision"] == r["precision"]
                 and r["precision"] < 0.8 and r["fp"] >= 5]
        if hi_fp:
            out.append("➕ Many unmatched detections (possible missing annotations): " + ", ".join(
                f"<b>{l}</b> {fp:,} (precision {p:.2f})" for l, fp, p in sorted(hi_fp, key=lambda x: -x[1])[:3]))
        if v.seen == "all" and not v.oof and (d.gt_seen == 1).any() and (d.gt_seen == 0).any():
            rs = {r["label"]: r for r in report(d, thr["conf"], thr["iou"], View(v.split, "seen"))}
            ru = {r["label"]: r for r in report(d, thr["conf"], thr["iou"], View(v.split, "unseen"))}
            gaps = sorted(((rs[l]["recall"] - ru[l]["recall"], l) for l in rs if l in ru and not l.startswith("ALL")
                           and rs[l]["recall"] == rs[l]["recall"] and ru[l]["recall"] == ru[l]["recall"]), reverse=True)
            gaps = [g for g in gaps if g[0] > 0.1][:3]
            if gaps:
                out.append("↕ Seen vs unseen recall gap (over-fitting or label drift between splits): " + ", ".join(
                    f"<b>{l}</b> {g:+.2f}" for g, l in gaps))
        best = [f"{r['label']} {r['best_conf']:.2f}" for r in cls_rows if r["best_conf"] == r["best_conf"]]
        if best:
            out.append("🎚 F1-optimal confidence per class: " + ", ".join(best))
        return "<br>".join(out)

    def _members(self, members):
        d = self.ctx["eval"]
        gt = [(d.keys[d.gt_img[g]], int(d.gt_idx[g])) for kind, g in members if kind == "gt"]
        files = list(dict.fromkeys([d.keys[d.gt_img[g]] if kind == "gt" else d.keys[d.p_img[g]] for kind, g in members]))
        return gt, files

    def _cell_activated(self, r, c, crops: bool):
        d = self.ctx.get("eval") if self.ctx else None
        if d is None:
            return
        names = d.labels + [BACKGROUND]
        members = getattr(self, "_cells", {}).get((r, c), [])
        if not members:
            return
        gt, files = self._members(members)
        title = f"{names[r]} → {names[c]}"
        if crops and gt:
            self.app.open_crop_instances(gt, f"your '{names[r]}' where the model says '{names[c]}'")
        else:
            self.app.apply_file_filter(files, f"🤖 {title}")

    def _report_activated(self, row, _col):
        d = self.ctx.get("eval") if self.ctx else None
        rows = getattr(self, "_last_report", [])
        if d is None or not 0 <= row < len(rows) or rows[row]["label"].startswith("ALL"):
            return
        c = d.labels.index(rows[row]["label"])
        members = getattr(self, "_cells", {}).get((c, len(d.labels)), [])
        gt, _files = self._members(members)
        if gt:
            self.app.open_crop_instances(gt, f"'{rows[row]['label']}' not confirmed by any model")
        else:
            QMessageBox.information(self, self.TITLE, "No missed labels for this class at the current thresholds.")

    def _export_report(self):
        rows = getattr(self, "_last_report", None)
        if not rows:
            return
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Export report card", "model_report.csv", "CSV (*.csv)")
        if path:
            import csv
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow([t for _k, t in REPORT_COLS])
                for r in rows:
                    w.writerow([("" if isinstance(r[k], float) and r[k] != r[k] else r[k]) for k, _t in REPORT_COLS])

    def _refresh_cards(self):
        """Cards in the table's current order and filter."""
        if not hasattr(self, "cards"):
            return
        self.prune_marks()
        colors = self.ctx.get("colors", {}) if self.ctx else {}
        names = self.ctx.get("names", {}) if self.ctx else {}
        cards = []
        for row in self.visible_rows():
            f = row.payload[0]
            boxes, shapes = [], []
            if f.gt_box:
                boxes.append(f.gt_box)
                col = self.app.label_colors.get(row.cells[1])
                shapes.append(Shape(box_points(*f.gt_box), col.name() if col else hexc("ok")))
            if f.box:
                boxes.append(tuple(f.box))
                shapes.append(Shape(box_points(*f.box), colors.get(f.model, "#ff2bd6"), dashed=True))
            if not boxes:
                continue
            focus = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
            s = self.session.by_key.get(row.key)
            label = row.cells[1]
            line1 = {MISMATCH: f"your '{label}'  ·  model '{f.pred_label}'",
                     MISSING: f"unlabeled '{f.pred_label}' found by the model",
                     MISSED: f"your '{label}' — no matching detection",
                     POOR_FIT: f"your '{label}' — loose box"}.get(f.kind, f.kind)
            chips = [(f.kind, "error" if KIND_TONE.get(f.kind) == "error" else "warn")]
            if f.conf:
                chips.append((f"conf {f.conf:.2f}", "muted"))
            if f.gt_box and f.box:
                chips.append((f"IoU {f.iou:.2f}", "muted"))
            cards.append(Card(row, [Panel(row.key, focus, shapes)], chips,
                              [line1, f"{s.display if s else row.key}  ·  {names.get(f.model, '')}".rstrip(" ·")]))
        self.cards.set_cards(cards)
        self.tabs.setTabText(0, f"Cards ({len(cards):,})")

    def summarize(self):
        self._refresh_cards()
        live = [r for r in self.rows if r]
        c = Counter(r.cells[0] for r in live)
        parts = [f"{c[k]:,} {k}" for k in (MISMATCH, MISSING, POOR_FIT, MISSED) if c.get(k)]
        return " · ".join(parts) or "✅ The models agree with every checked label"

    def on_row_activated(self, row):
        if row is None:
            return
        f, _W, _H = row.payload
        color = self.ctx.get("colors", {}).get(f.model, "#ff2bd6")
        name = self.ctx.get("names", {}).get(f.model, "model")
        overlay = [{"box": f.box, "text": f"{name}: {f.pred_label} {f.conf:.2f}", "color": color}] if f.box else []
        self.app.open_sample(row.key, f.gt_index, overlay=overlay, focus_box=f.box if f.gt_index is None else None)

    def closeEvent(self, event):
        try:
            self.app.inference.changed.disconnect(self._refresh_setup_info)
        except TypeError:
            pass
        super().closeEvent(event)

    # --------------------------------------------------------------- actions
    def _rel(self, box, W, H):
        return (box[0] / W, box[1] / H, box[2] / W, box[3] / H) if box is not None and W and H else None

    def accept_by_confidence(self):
        from AIO_QC.core.fix_ledger import finding_key
        rejected = self.ledger.keys()
        cand = []
        for r in self.visible_rows():
            if not r.payload:
                continue
            f = r.payload[0]
            if f.kind in BULK_KINDS and finding_key(self.session.root, r.key, *r.payload) not in rejected:
                cand.append(r)
        if not cand:
            QMessageBox.information(self, self.TITLE, "No missing-box, class-mismatch or poor-fit findings in the "
                                    "list (run the check, or clear the filter).")
            return
        st = self.app.settings
        dlg = QDialog(self)
        dlg.setWindowTitle("Accept by confidence")
        lay = QVBoxLayout(dlg)
        intro = QLabel("Accept every finding below in one go — the originals are backed up (↶ Undo). Deleting "
                       "labelled shapes (“missed by model”) is never done in bulk.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        conf = dspin(0.05, 1.0, float(st.value("modelcheck/bulk_conf", 0.8)), 2, 0.05)
        form = QFormLayout()
        form.addRow("Model confidence at least:", conf)
        lay.addLayout(form)
        boxes = {}
        for k in BULK_KINDS:
            c = QCheckBox(k)
            c.setChecked(st.value(f"modelcheck/bulk_{k.replace(' ', '_')}", k == MISSING, type=bool))
            boxes[k] = c
            lay.addWidget(c)
        files = QLabel()
        apply_style(files, "font-weight:600;")
        lay.addWidget(files)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)

        def chosen():
            want = {k for k, c in boxes.items() if c.isChecked()}
            return [r for r in cand if r.payload[0].kind in want and float(r.payload[0].conf) >= conf.value()]

        def update():
            per = Counter(r.payload[0].kind for r in cand if float(r.payload[0].conf) >= conf.value())
            for k, c in boxes.items():
                c.setText(f"{k}  ({per.get(k, 0):,})")
            sel = chosen()
            files.setText(f"→ {len(sel):,} finding(s) in {len({r.key for r in sel}):,} file(s)")
            bb.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(sel))
            bb.button(QDialogButtonBox.StandardButton.Ok).setText(f"Accept {len(sel):,}")
        conf.valueChanged.connect(lambda _v: update())
        for c in boxes.values():
            c.toggled.connect(lambda _on: update())
        update()
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        st.setValue("modelcheck/bulk_conf", conf.value())
        for k, c in boxes.items():
            st.setValue(f"modelcheck/bulk_{k.replace(' ', '_')}", c.isChecked())
        rows = chosen()
        if rows:
            self.accept_rows(rows, ask=False)

    def accept_rows(self, rows=None, ask=True):
        rows = [r for r in (rows if rows is not None else self.target_rows()) if r.payload]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick or select findings first (Enter accepts the selected one).")
            return
        fixes, used, n_kind = defaultdict(list), [], Counter()
        for r in rows:
            f, W, H = r.payload
            gb, pb = self._rel(f.gt_box, W, H), self._rel(f.box, W, H)
            if f.kind == MISMATCH and f.gt_index is not None and gb:
                fixes[r.key].append(("relabel", f.gt_index, gb, f.pred_label))
            elif f.kind == MISSING and pb:
                fixes[r.key].append(("add", f.pred_label, pb))
            elif f.kind == POOR_FIT and f.gt_index is not None and gb and pb:
                fixes[r.key].append(("reshape", f.gt_index, gb, pb))
            elif f.kind == MISSED and f.gt_index is not None and gb:
                fixes[r.key].append(("delete", f.gt_index, gb))
            else:
                continue
            used.append(r)
            n_kind[f.kind] += 1
        if not used:
            return
        if len(used) > 1 and ask:
            parts = ", ".join(f"{n} {k}" for k, n in n_kind.items())
            if not self.confirm(f"Accept {len(used):,} finding(s) in {len(fixes):,} file(s)? ({parts})\n\nThe "
                                "originals are backed up (Undo)."):
                return
        nxt = self._next_after(used)
        st = self.session.apply_fixes(self, dict(fixes), what="modelcheck")
        if st["files"]:
            self._last_backup = st["backup"]
            self.undo_btn.setEnabled(True)
        ids = {id(r) for r in used}
        self.remove_rows(lambda r: id(r) in ids)
        self._compact()
        self.summary.setText(self.summarize())
        msg = f"Accepted {st['done']:,} of {len(used):,} in {st['files']:,} file(s)"
        if st["missing"]:
            msg += f" · {st['missing']:,} skipped (the shape changed since the scan)"
        if st["skipped"]:
            msg += f" · {st['skipped']:,} already annotated"
        if st["refused"]:
            msg += f" · {st['refused']:,} not possible (class not in the dataset / not a box)"
        self.app.statusBar().showMessage(msg + " — ↶ Undo puts them back", 8000)
        if st["errors"]:
            self.app.log_errors(st["errors"])
        self._go_next(nxt)

    def reject_rows(self):
        rows = [r for r in self.target_rows() if r.payload]
        if not rows:
            return
        nxt = self._next_after(rows)
        from AIO_QC.core.fix_ledger import finding_key
        keys = [finding_key(self.session.root, r.key, *r.payload) for r in rows]
        self.ledger.add(keys, "rejected in Model Check")
        self.ctx.setdefault("rejected", set()).update(keys)
        if not self.show_rejected.isChecked():
            ids = {id(r) for r in rows}
            self.remove_rows(lambda r: id(r) in ids)
            self._compact()
        self.summary.setText(self.summarize())
        self.app.statusBar().showMessage(f"Rejected {len(rows):,} finding(s) — hidden on later runs too", 5000)
        self._go_next(nxt)

    def unreject_rows(self):
        from AIO_QC.core.fix_ledger import finding_key
        rows = [r for r in self.target_rows() if r.payload]
        keys = [finding_key(self.session.root, r.key, *r.payload) for r in rows]
        keys = [k for k in keys if k in self.ledger.items]
        if not keys:
            QMessageBox.information(self, self.TITLE, "Tick grey (rejected) rows first — tick “Show rejected”.")
            return
        self.ledger.remove(keys)
        self.ctx.get("rejected", set()).difference_update(keys)
        self._recompute_findings()

    def _toggle_rejected(self, on):
        if self.ctx and self.ctx.get("records"):
            self.ctx["show_rejected"] = on
            self.ctx["rejected"] = self.ledger.keys()
            self._recompute_findings()

    def undo_accept(self):
        b = self._last_backup
        if b is None or not self.confirm(f"Put back the {len(b.after):,} file(s) changed by the last Accept?"):
            return
        keys, skipped = self.session.undo_backup(b)
        self._last_backup = None
        self.undo_btn.setEnabled(False)
        self.report("Undone", [f"Restored {len(keys):,} file(s). Re-run the check to see their findings again."] +
                    ([f"{len(skipped):,} file(s) changed after the Accept and were left alone."] if skipped else []),
                    [])

    # ---- review flow: Enter accepts, X rejects, the next finding opens
    def _visible_order(self) -> list:
        out = []
        for t in range(self.table.rowCount()):
            if self.table.isRowHidden(t):
                continue
            r = self._row_at(t)
            if r is not None:
                out.append((t, r))
        return out

    def _next_after(self, rows):
        order = self._visible_order()
        ids = {id(r) for r in rows}
        pos = [i for i, (_t, r) in enumerate(order) if id(r) in ids]
        if not pos:
            return None
        for _t, r in order[max(pos) + 1:]:
            if id(r) not in ids:
                return r
        for _t, r in reversed(order[:min(pos)]):
            if id(r) not in ids:
                return r
        return None

    def _go_next(self, row):
        if row is None or not self.auto_next.isChecked():
            return
        for t, r in self._visible_order():
            if r is row:
                self.table.clearSelection()
                self.table.selectRow(t)
                self.table.scrollToItem(self.table.item(t, 0))
                self.on_row_activated(r)
                self.table.setFocus()
                break

    def flag_files(self, rows=None, note=None):
        rows = rows if rows is not None else self.target_rows()
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick (or select) rows first.")
            return
        notes = defaultdict(set)
        for r in rows:
            notes[r.key].add(r.cells[0])
        for k, kinds in notes.items():
            self.session.set_review([k], REVIEW_FLAG, note if note is not None else
                                    "Model Check: " + ", ".join(sorted(kinds)))
        self.app.statusBar().showMessage(f"Flagged {len(notes)} file(s)", 4000)
