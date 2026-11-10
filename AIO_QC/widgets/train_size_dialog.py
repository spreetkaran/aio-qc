# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
📏 Training Size — every object measured at the size the detector will actually see it: the image shrunk so its
longer side equals the training imgsz, optionally shrunk further by scale augmentation. Lists the objects that
become too small to learn, and per class the imgsz that would keep 95 % of them above the limit.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QLabel, QTableWidget

from AIO_QC.core import train_size as TS
from AIO_QC.core.dataset import DETECTION_TYPES
from AIO_QC.core.state import load_models_config
from .base_scan import TONE, BaseScanDialog, ResultRow, dspin, ispin, readable_table


def default_imgsz(root) -> int:
    for m in load_models_config(root).get("models", []):
        if m.get("enabled", True):
            v = m.get("imgsz") or m.get("train_imgsz")
            if v:
                return int(v)
    return 640


class TrainSizeDialog(BaseScanDialog):
    TITLE = "Training Size"
    ICON = "📏"
    HELP = ("How big is each object when the image is shrunk to the training size (longer side = imgsz, like "
            "Ultralytics)? Objects only a few pixels across are hard to learn and are often missed later — scale "
            "augmentation shrinks them further. Lists the objects below the limit and, per class, the imgsz that "
            "keeps 95 % of the class above it.")
    COLUMNS = ["Verdict", "Label", "In the image", "At training size", "Needs imgsz ≥"]
    FILTER_COLUMN = 0
    SUPPORTED = DETECTION_TYPES

    def build_params(self, form):
        st = self.app.settings
        self.imgsz = ispin(32, 8192, int(st.value("trainsize/imgsz", 0) or default_imgsz(self.app.session.root)))
        self.imgsz.setSingleStep(32)
        self.imgsz.setSuffix(" px")
        self.imgsz.setToolTip("The training image size (the longer side). Defaults to the imgsz of your first model "
                              "in Models & Environment, else 640.")
        self.min_side = dspin(1, 256, float(st.value("trainsize/min_side", 8)), 1, 1, " px")
        self.min_side.setToolTip("An object's shorter side at training size below this is flagged. 8 px = YOLO's "
                                 "finest stride; text you want to read usually needs 12–16 px.")
        self.min_area = dspin(0, 100000, float(st.value("trainsize/min_area", 0)), 0, 16, " px²")
        self.min_area.setToolTip("Optional: also flag objects whose area at training size is below this (0 = off)")
        self.aug = dspin(0, 0.9, float(st.value("trainsize/aug", 0.5)), 2, 0.05)
        self.aug.setToolTip("Scale augmentation: images are shrunk at random down to (1 − scale). Ultralytics "
                            "default 0.5. 0 = off.")
        form.addRow("Training imgsz:", self.imgsz)
        form.addRow("Shorter side at least:", self.min_side)
        form.addRow("Area at least:", self.min_area)
        form.addRow("Scale augmentation:", self.aug)

    def post_init(self):
        self.per_class = QTableWidget(0, 7)
        self.per_class.setHorizontalHeaderLabels(["Class", "Objects", "Too small", "At risk", "Median short side",
                                                  "imgsz for 95 %", "Advice"])
        self.per_class.verticalHeader().setVisible(False)
        self.per_class.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.per_class.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.per_class.horizontalHeader().setStretchLastSection(True)
        readable_table(self.per_class)
        self.per_class.setMaximumHeight(190)
        self.per_class.setVisible(False)
        self.per_class.cellDoubleClicked.connect(self._filter_class)
        cap = QLabel("Per class (double-click: show its objects)")
        cap.setVisible(False)
        self._cap = cap
        self.table_host.insertWidget(1, cap)
        self.table_host.insertWidget(2, self.per_class)

    def make_context(self):
        st = self.app.settings
        for k, w in (("imgsz", self.imgsz), ("min_side", self.min_side), ("min_area", self.min_area),
                     ("aug", self.aug)):
            st.setValue(f"trainsize/{k}", w.value())
        return {"adapter": self.session.adapter, "imgsz": self.imgsz.value(), "min_side": self.min_side.value(),
                "min_area": self.min_area.value(), "aug": self.aug.value(), "objs": []}

    def scan_sample(self, ctx, s):
        ann = ctx["adapter"].load(s, light=True)
        if ann.normalized or not (ann.width and ann.height):
            size = ctx["adapter"].image_size(s, ann)
            if not size:
                return []
            W, H = float(size[0]), float(size[1])
        else:
            W, H = float(ann.width), float(ann.height)
        rows = []
        for i, sh in enumerate(ann.shapes):
            if sh.get("_error"):
                continue
            wh = TS.shape_wh(sh, W, H, ann.normalized)
            if wh is None:
                continue
            o = TS.Obj(s.key, i, str(sh.get("label", "")), wh[0], wh[1], max(W, H))
            o.verdict = TS.classify(o, ctx["imgsz"], ctx["min_side"], ctx["min_area"], ctx["aug"])
            ctx["objs"].append(o)
            if o.verdict:
                rows.append(self._row(o, ctx))
        return rows

    def rescan_sample(self, ctx, s):
        ctx["objs"] = [o for o in ctx["objs"] if o.key != s.key]
        return self.scan_sample(ctx, s)

    @staticmethod
    def _row(o, ctx) -> ResultRow:
        tw, th = o.at(ctx["imgsz"])
        need = o.needed(ctx["min_side"])
        return ResultRow(o.key, (o.verdict, o.label, f"{o.w:.0f} × {o.h:.0f}", f"{tw:.1f} × {th:.1f}",
                                 TS.round_up(need) if need != float("inf") else ""),
                         o.idx, payload=o, tone="error" if o.verdict == TS.TOO_SMALL else "warning")

    def on_scan_complete(self, ctx):
        rows = TS.class_summary(ctx["objs"], ctx["imgsz"], ctx["min_side"])
        ctx["classes"] = rows
        t = self.per_class
        t.setSortingEnabled(False)
        t.setRowCount(len(rows))
        for r, d in enumerate(rows):
            vals = [d["label"], d["n"], d["too_small"], d["at_risk"], round(d["median_short"], 1),
                    d["imgsz_needed"] if d["imgsz_needed"] is not None else "", TS.advice(d, ctx["imgsz"])]
            for c, v in enumerate(vals):
                t.setItem(r, c, self._cell(v))
            if d["too_small"]:
                col = TONE("error")
                if col is not None:
                    t.item(r, 2).setForeground(col)
        t.setSortingEnabled(True)
        t.resizeColumnsToContents()
        t.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        t.setVisible(bool(rows))
        self._cap.setVisible(bool(rows))

    def _filter_class(self, r, _c):
        lab = self.per_class.item(r, 0).text()
        for t in range(self.table.rowCount()):
            row = self._row_at(t)
            self.table.setRowHidden(t, row is None or row.payload.label != lab)

    def summarize(self):
        c = self.ctx or {}
        objs = c.get("objs") or []
        if not objs:
            return super().summarize()
        small = sum(1 for o in objs if o.verdict == TS.TOO_SMALL)
        risk = sum(1 for o in objs if o.verdict == TS.AT_RISK)
        return (f"{len(objs):,} object(s) at imgsz {c['imgsz']}: {small:,} too small ({small / len(objs):.1%}), "
                f"{risk:,} at risk with augmentation")
