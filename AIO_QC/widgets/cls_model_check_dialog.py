# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Model Check for classification datasets: your classifier (an Ultralytics YOLO-cls model added in Models &
environment) looks at every image —

  · Likely mislabel      the model is sure (≥ the "sure" level) of another class
  · Model disagrees      another class, less sure
  · Model unsure         the right class, but below the "unsure" level (hard / ambiguous images)
  · No class             the image has none: the model's suggestion
  · per class            precision, recall, F1 and how many images; accuracy (top-1 / top-5); the confusion matrix

Use the model's class sets the ticked images to the model's top class (one step of Undo). Predictions are cached:
running it again only looks at images that changed.
"""
from __future__ import annotations

import threading

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QDoubleSpinBox, QHeaderView, QLabel, QMessageBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from AIO_QC.core import cls_model_check as CM
from AIO_QC.core.dataset import CLASSIFICATION_TYPES
from AIO_QC.core.model_check import model_signature
from AIO_QC.core.state import load_models_config
from AIO_QC.utils import gradient_hex
from .base_scan import BaseScanDialog, ResultRow


def classifier_models(root) -> list[dict]:
    return [m for m in load_models_config(root).get("models", [])
            if m.get("enabled", True) and (m.get("task") == "classify")]


def cached_predictions(app, samples, stop=lambda: False) -> tuple[str, dict]:
    """(model name, {key: [(class, conf), …]}) for the images the classifier has already looked at — read from the
    prediction cache (Model Check fills it), no model run. Cached predictions follow images that were moved or
    renamed. ("", {}) without a classifier."""
    from .model_setup_dialog import server_specs
    s = app.session
    models = classifier_models(s.root) if s is not None else []
    if not models or s.cache is None:
        return "", {}
    m = models[0]
    svc = app.inference
    specs = server_specs([m], svc.opts, svc.info)
    if not specs:
        return m.get("name", ""), {}
    sig = model_signature(specs[0]) + "|cls"
    cmap, names = dict(m.get("class_map") or {}), dict(m.get("names") or {})
    out = {}
    for x in samples:
        if stop():
            break
        img = x.image_path
        if img is None:
            continue
        hit = s.cache.get_preds(sig, img)
        if hit is not None:
            out[x.key] = CM.map_probs(hit.get("probs") or [], cmap, names)
    return m.get("name", ""), out


class ClsModelCheckDialog(BaseScanDialog):
    TITLE, ICON = "Model Check", ""
    SUPPORTED = CLASSIFICATION_TYPES
    HELP = ("Your classifier looks at every image: likely mislabels (it is sure of another class), images it "
            "disagrees with, images it is unsure about, images without a class (its suggestion), and per class "
            "precision / recall / F1 with the confusion matrix. Add a YOLO classification model (…-cls.pt) in "
            "Models & environment first.")
    COLUMNS = ["Finding", "Class", "Model", "Confidence", "Why"]
    FILTER_COLUMN = 0
    AUTO_START = False

    def build_params(self, form):
        self.sure = QDoubleSpinBox()
        self.sure.setRange(0.3, 0.999)
        self.sure.setSingleStep(0.05)
        self.sure.setValue(0.8)
        self.sure.setToolTip("Another class at least this sure: a likely mislabel")
        self.unsure = QDoubleSpinBox()
        self.unsure.setRange(0.05, 0.95)
        self.unsure.setSingleStep(0.05)
        self.unsure.setValue(0.5)
        self.unsure.setToolTip("The right class but below this: the model is unsure (hard / ambiguous image)")
        form.addRow("Sure of another class at", self.sure)
        form.addRow("Unsure below", self.unsure)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        form.addRow(self.info)
        self.add_action("Use the model's class", self.use_model,
                        "The ticked (or selected) images get the model's top class — Ctrl+Z undoes it")
        self.add_action("Confusion matrix…", self.show_confusion, "Which classes the model mixes up")
        self.add_action("Models & environment…", lambda: self.app.open_model_setup(),
                        "Add or load your classifier")
        self._refresh_info()

    def _refresh_info(self):
        ms = classifier_models(self.session.root)
        self.info.setText("Classifier: " + (", ".join(f"<b>{m['name']}</b>" for m in ms) if ms else
                                            "<i>none — add a YOLO classification model (…-cls.pt) in Models & "
                                            "environment and load it once</i>"))

    def focus_models(self, ids=None):
        """Check only this classifier (by id) — None: the first enabled one again (Improve Model checks its run)."""
        self.only_ids = set(ids) if ids else None
        self._refresh_info()

    def make_context(self):
        svc = self.app.inference
        models = classifier_models(self.session.root)
        only = getattr(self, "only_ids", None)
        if only:
            models = [m for m in models if m.get("id") in only] or models
        if not models:
            self.app.open_model_setup()
            raise ValueError("Add a classification model (YOLO …-cls.pt) in Models & environment and load it once "
                             "(its classes are read then).")
        if not svc.python:
            self.app.open_model_setup()
            raise ValueError("Choose a Python environment in Models & environment.")
        from .model_setup_dialog import server_specs
        m = models[0]
        specs = server_specs([m], svc.opts, svc.info)
        if not specs:
            raise ValueError(f"{m['name']}: the model file is missing.")
        s = self.session
        return {"svc": svc, "spec": specs[0], "mid": m["id"], "sig": model_signature(specs[0]) + "|cls",
                "cmap": dict(m.get("class_map") or {}), "names": dict(m.get("names") or {}),
                "adapter": s.adapter, "cache": s.cache, "batch": int(svc.opts.get("batch", 8)),
                "order": list(s.samples), "pos": {x.key: i for i, x in enumerate(s.samples)}, "buffer": {},
                "loaded": False, "cancel": threading.Event(), "recs": [], "sure": self.sure.value(),
                "unsure": self.unsure.value(), "ran": 0, "cached": 0}

    def stop_scan(self):
        if self.ctx and self.ctx.get("cancel") is not None:
            self.ctx["cancel"].set()
        super().stop_scan()

    # ------------------------------------------------------------------ worker
    def _prefetch(self, ctx, key):
        i0 = ctx["pos"][key]
        window = [x for x in ctx["order"][i0:i0 + ctx["batch"]] if x.key not in ctx["buffer"]]
        need = []
        for x in window:
            try:
                ann = ctx["adapter"].load(x, light=True)
            except Exception as exc:
                ctx["buffer"][x.key] = ("error", str(exc))
                continue
            img = ann.image_path or x.image_path
            if img is None:
                ctx["buffer"][x.key] = ("error", "no image")
                continue
            hit = ctx["cache"].get_preds(ctx["sig"], img) if ctx["cache"] is not None else None
            if hit is not None:
                ctx["buffer"][x.key] = ("ok", (ann, hit.get("probs") or []))
                ctx["cached"] += 1
            else:
                need.append((x, ann, img))
        if need:
            if not ctx["loaded"]:
                ctx["svc"].ensure_loaded([ctx["spec"]])
                ctx["loaded"] = True
            res = ctx["svc"].predict([str(i) for _x, _a, i in need], [ctx["mid"]], conf=0.0, batch=ctx["batch"],
                                     cancel=ctx["cancel"])
            ctx["ran"] += len(need)
            for x, ann, img in need:
                r = res.get(str(img)) or {}
                probs = (r.get("probs") or {}).get(ctx["mid"])
                if probs is None:
                    ctx["buffer"][x.key] = ("error", r.get("error") or "no prediction")
                    continue
                if ctx["cache"] is not None:
                    ctx["cache"].put_preds(ctx["sig"], img, {"w": r.get("w"), "h": r.get("h"), "boxes": [],
                                                             "probs": probs})
                ctx["buffer"][x.key] = ("ok", (ann, probs))

    def scan_sample(self, ctx, s):
        if s.key not in ctx["buffer"]:
            self._prefetch(ctx, s.key)
        st, v = ctx["buffer"].pop(s.key, ("error", "not predicted"))
        if st == "error":
            return [ResultRow(s.key, ("Could not check", "", "", "", str(v)[:200]), tone="error")]
        ann, probs = v
        pred = CM.map_probs(probs, ctx["cmap"], ctx["names"])
        ctx["recs"].append({"key": s.key, "split": s.split, "gt": list(ann.tags or []), "pred": pred})
        return []

    def finalize(self, ctx):
        rows = []
        for f in CM.findings(ctx["recs"], ctx["sure"], ctx["unsure"]):
            tone = {CM.LIKELY: "error", CM.NO_CLASS: "warning", CM.WRONG: "warning", CM.UNSURE: "info"}[f["kind"]]
            rows.append(ResultRow(f["key"], (f["kind"], ", ".join(f["gt"]), f["suggest"], f"{f['conf']:.2f}",
                                             f["why"]), tone=tone, payload=f))
        m = CM.metrics(ctx["recs"])
        ctx["metrics"] = m
        for c, v in sorted(m["classes"].items(), key=lambda x: x[1]["f1"]):
            rows.append(ResultRow("", ("Class", c, f"F1 {v['f1']:.2f}", f"P {v['precision']:.2f} · R {v['recall']:.2f}",
                                       f"{v['support']:,} image(s) · {v['fn']:,} missed · {v['fp']:,} taken for it"),
                                  payload={"kind": "class", "class": c},
                                  dot=gradient_hex(v["f1"], 0.55, 0.70)))
        return rows

    def on_scan_complete(self, ctx):
        super().on_scan_complete(ctx)
        gal = self.app.open_dialogs.get("class_gallery")      # an open Class Gallery picks up the new predictions
        if gal is not None:
            gal.load_predictions()

    def summarize(self):
        m = (self.ctx or {}).get("metrics")
        if not m:
            return super().summarize()
        bad = sum(1 for r in self.rows if r and r.payload and r.payload.get("kind") == CM.LIKELY)
        return (f"{m['n']:,} image(s) · accuracy {m['top1']:.1%} (top-5 {m['top5']:.1%}) · {bad:,} likely "
                f"mislabel(s) · {self.ctx.get('ran', 0):,} predicted, {self.ctx.get('cached', 0):,} from the cache")

    def on_row_activated(self, row):
        if row is not None and row.key in self.session.by_key:
            self.app.open_sample(row.key, None)

    # ------------------------------------------------------------------ actions
    def use_model(self):
        rows = [r for r in self.target_rows() if r and r.payload and r.payload.get("suggest")]
        if not rows:
            QMessageBox.information(self, self.TITLE, "Tick (or select) findings first.")
            return
        s = self.session
        by_cls = {}
        for r in rows:
            by_cls.setdefault(r.payload["suggest"], []).append(r.key)
        done, errors = [], []
        for cls, keys in by_cls.items():
            d, e = s.set_tags(keys, tags=[cls], what=f"class → {cls} (model)")
            done += d
            errors += e
        self.report("Use the model's class", [f"{len(done):,} image(s) changed (Ctrl+Z undoes it)."], errors)
        gone = set(r.key for r in rows)
        self.remove_rows(lambda r: r.key in gone)

    def show_confusion(self):
        m = (self.ctx or {}).get("metrics")
        if not m or not m["confusion"]:
            QMessageBox.information(self, self.TITLE, "Run the check first.")
            return
        ConfusionDialog(self, m["confusion"], open_cell=self._open_cell).exec()

    def _open_cell(self, gt: str, pred: str):
        """A cell of the confusion matrix: those images in the Class Gallery."""
        gal = self.app.open_class_gallery()
        if gal is not None:
            gal.show_confusion(gt, pred)
            if not gal.preds:                  # the predictions are still being read: show the cell once they are
                gal._after_preds = lambda: gal.show_confusion(gt, pred)


class ConfusionDialog(QDialog):
    """Rows: the label; columns: what the model said. The diagonal is right."""

    def __init__(self, parent, conf: dict, open_cell=None):
        super().__init__(parent)
        self.setWindowTitle("Confusion matrix")
        self.resize(720, 520)
        classes = sorted(set(conf) | {p for v in conf.values() for p in v}, key=str.lower)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Rows: the image's class · columns: the model's top class · the diagonal is right"
                             + (" · double-click a cell: those images in the Class Gallery" if open_cell else "")))
        t = QTableWidget(len(classes), len(classes))
        t.setHorizontalHeaderLabels(classes)
        t.setVerticalHeaderLabels(classes)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for i, g in enumerate(classes):
            row = conf.get(g, {})
            tot = sum(row.values()) or 1
            for j, p in enumerate(classes):
                n = row.get(p, 0)
                it = QTableWidgetItem(str(n) if n else "")
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if n:
                    from PyQt6.QtGui import QColor
                    share = n / tot
                    it.setBackground(QColor(gradient_hex(share if i == j else 1 - share, 0.55, 0.75)))
                    it.setToolTip(f"{g} → {p}: {n} ({share:.0%} of {g})")
                t.setItem(i, j, it)
        self.table = t
        if open_cell is not None:
            def cell(i, j):
                if classes[i] != classes[j]:
                    self.accept()                 # the gallery takes over (this window is modal)
                    open_cell(classes[i], classes[j])
            t.cellDoubleClicked.connect(cell)
        lay.addWidget(t, 1)
