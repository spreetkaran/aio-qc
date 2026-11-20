# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Search in Words — a text-aware vision model (SigLIP / SigLIP2 / CLIP, the models Visual Similarity manages) used
two ways, locally, nothing leaves the machine:

    Search                "truck at night in rain" → images (or objects) ranked by how well they match; per split
                          how many matched (coverage); optionally only images WITHOUT a given label ("probably an
                          unlabelled person"). A new query after the first one is instant: the vectors stay in memory
                          and only the query is embedded.
    Class descriptions    every label checked against one line per class (core.text_search): objects that look like
                          another class, objects unlike their class, and per class how often the model agrees —
                          works without a trained model, on day one of a dataset.
Vectors come from the same cache as Visual Similarity (same model, same crop margin): what one tool embedded, the
other reuses.
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from AIO_QC.core import text_search as TS
from AIO_QC.utils import newButton
from .base_scan import ResultRow, dspin, ispin
from .inference_service import FuncWorker
from .similarity_dialog import SimilarityDialog

SEARCH, DESC = "search", "desc"


class DescriptionsDialog(QDialog):
    """One line per class — what the class looks like, in words. Saved with the dataset."""

    def __init__(self, parent, root, classes: list[str]):
        super().__init__(parent)
        self.setWindowTitle("Class descriptions")
        self.resize(720, 520)
        self.root = root
        data = TS.load_descriptions(root)
        names = list(dict.fromkeys(list(classes) + sorted(data["classes"])))
        lay = QVBoxLayout(self)
        tip = QLabel("Describe what each class LOOKS like, the way you would tell a new annotator — e.g. "
                     "<i>trailer_text</i>: “painted container number on the side of a trailer”. Say what sets it "
                     "apart from the class it is confused with. Used by Search in Words and Claude Review.")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        self.table = QTableWidget(len(names), 2)
        self.table.setHorizontalHeaderLabels(["Class", "Description"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for r, n in enumerate(names):
            it = QTableWidgetItem(n)
            it.setFlags(it.flags() & ~it.flags().ItemIsEditable)
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QTableWidgetItem(data["classes"].get(n) or TS.default_description(n)))
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Phrase as:"))
        self.template = QLineEdit(data.get("template", TS.DEFAULT_TEMPLATE))
        self.template.setToolTip("{} is replaced by the description. Clear it to use the descriptions as written.")
        row.addWidget(self.template, 1)
        row.addWidget(newButton("Reset to class names", slot=self._reset))
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _reset(self):
        for r in range(self.table.rowCount()):
            self.table.item(r, 1).setText(TS.default_description(self.table.item(r, 0).text()))

    def values(self) -> dict:
        return {"template": self.template.text().strip(),
                "classes": {self.table.item(r, 0).text(): self.table.item(r, 1).text().strip()
                            for r in range(self.table.rowCount()) if self.table.item(r, 1).text().strip()}}

    def _save(self):
        TS.save_descriptions(self.root, self.values())
        self.accept()


class TextSearchDialog(SimilarityDialog):
    TITLE, ICON = "Search in Words", ""
    HELP = ("Search the dataset in words — “truck at night in rain”, “blurred plate”, “trailer roof with a hole” — "
            "or check every label against a one-line description of its class. Needs a text-aware model (SigLIP2 / "
            "SigLIP / CLIP: Download… offers them); runs in your Python environment, nothing leaves this computer. "
            "The first run embeds the images (shared with Visual Similarity); every search after that is instant. "
            "Scores are a ranking, not a verdict — look at the results.")
    COLUMNS = ["#", "Match", "Label", "Split"]
    RESCAN_AFTER_EDIT = False

    # ------------------------------------------------------------------ parameters
    def build_params(self, form):
        self._store = None
        self._form = form
        self._model_rows(form)
        st = self.app.settings
        self.mode = QComboBox()
        self.mode.addItem("Search in words", SEARCH)
        self.mode.addItem("Check labels against class descriptions", DESC)
        self.mode.setCurrentIndex(max(0, self.mode.findData(st.value("words/mode", SEARCH))))
        form.addRow("Do:", self.mode)
        # search
        self.query = QLineEdit(st.value("words/query", "") or "")
        self.query.setPlaceholderText("e.g. truck at night in rain · blurred plate · empty gate lane — Enter searches")
        self.query.returnPressed.connect(self.start_scan)
        self.what = QComboBox()
        self.what.addItem("Images", "image")
        self.what.addItem("Objects (labelled shapes)", "shape")
        self.what.setEnabled(not self.session.is_classification)
        self.top = ispin(1, 1_000_000, int(st.value("words/top", 200)))
        self.top.setToolTip("How many of the best matches to list")
        self.phrase = QCheckBox("as “a photo of …”")
        self.phrase.setChecked(st.value("words/phrase", "true") == "true")
        self.phrase.setToolTip("CLIP-style models match “a photo of a truck at night” better than “truck at night”")
        self.without = QComboBox()
        self.without.addItem("(any image)", None)
        for lab, _n in self.session.label_totals().most_common():
            self.without.addItem(f"only images WITHOUT a '{lab}' label", lab)
        self.without.setToolTip("Find what is probably unlabelled: search “a person”, keep only images without a "
                                "'person' label")
        qrow = QWidget()
        ql = QHBoxLayout(qrow)
        ql.setContentsMargins(0, 0, 0, 0)
        ql.addWidget(self.query, 1)
        ql.addWidget(newButton("Search", icon="search", slot=self.start_scan))
        from AIO_QC.utils.qt_utils import flow_row
        self.search_opts = flow_row([("in", self.what), ("show best", self.top), self.phrase, self.without])
        self.query_row = qrow
        form.addRow("Words:", qrow)
        form.addRow("", self.search_opts)
        # descriptions
        self.margin = dspin(0.2, 5.0, 1.0, 1, step=0.1)
        self.margin.setToolTip("Report 'looks like another class' when another description scores this many "
                               "standard deviations above its own")
        self.poor = dspin(1.0, 10.0, 2.5, 1, step=0.5)
        self.poor.setToolTip("Report 'unlike its class description' this many robust SDs below its class")
        self.desc_opts = flow_row([newButton("Class descriptions…", slot=self._edit_descriptions,
                                             tip="One line per class: what it looks like"),
                                   ("other class by ≥", self.margin), ("unlike its class ≥", self.poor)])
        form.addRow("Descriptions:", self.desc_opts)
        form.addRow("Settings:", flow_row([("device", self.device), self.half, ("batch", self.batch),
                                           ("crop", self.pad)]))
        self.relabel_btn = self.add_action("✓ Use suggested label", self._relabel,
                                           "Relabel the ticked (or selected) rows to the class their look matches — "
                                           "backed up, undoable")
        self.undo_btn = self.add_action("↶ Undo", self._undo, "Put back the files changed by the last relabel")
        self.undo_btn.setEnabled(False)
        self.add_action("▦ Review in grid", self._grid, "Open the listed files in the Review Grid")
        self.add_action("📊 Class agreement…", self._agreement,
                        "Per class: how often the model's reading of the descriptions agrees with the labels")
        self._last_backup = None
        self.mode.currentIndexChanged.connect(lambda _i: self._mode_ui())
        self._mode_ui()

    def _mode_ui(self):
        s = self.mode.currentData() == SEARCH
        for w, on in ((self.query_row, s), (self.search_opts, s), (self.desc_opts, not s)):
            w.setVisible(on)
            try:
                self._form.setRowVisible(w, on)
            except (AttributeError, TypeError, RuntimeError):
                pass
        self.relabel_btn.setVisible(not s)
        self.undo_btn.setVisible(not s)
        self.label_box.setEnabled(not s or self.what.currentData() == "shape")

    def _model_info(self):
        path = self.model_box.currentData()
        if not path:
            self.model_note.setText("Needs a model that understands words: SigLIP2 (best), SigLIP or CLIP — "
                                    "Download… offers them. DINO models understand images only.")
            return
        super()._model_info()

    def _classes(self) -> list[str]:
        if self.session.is_classification:
            return list(self.session.cls_classes())
        return [lab for lab, _n in self.session.label_totals().most_common()]

    def _edit_descriptions(self):
        d = DescriptionsDialog(self, self.session.root, self._classes())
        if d.exec():
            self._store_texts = None

    # ------------------------------------------------------------------ scan
    def columns_for(self, ctx):
        return self.COLUMNS if ctx.get("mode") == SEARCH else ["Finding", "Label", "Suggests", "Score", "Why"]

    def make_context(self):
        mode = self.mode.currentData()
        q = self.query.text().strip()
        if mode == SEARCH and not q:
            raise ValueError("Type what to look for, e.g. “truck at night in rain”.")
        ctx = super().make_context()
        st = self.app.settings
        st.setValue("words/mode", mode)
        st.setValue("words/query", q)
        st.setValue("words/top", self.top.value())
        st.setValue("words/phrase", "true" if self.phrase.isChecked() else "false")
        cls = self.session.is_classification
        unit = ("image" if cls or self.what.currentData() == "image" else "shape") if mode == SEARCH else \
            ("image" if cls else "shape")
        ctx.update(mode=mode, unit=unit, query=q, top=self.top.value(), phrase=self.phrase.isChecked(),
                   without=self.without.currentData() if unit == "image" else None, margin=self.margin.value(),
                   poor=self.poor.value(), all_images=mode == SEARCH and unit == "image", splits=[], cap=0,
                   label=ctx["label"] if (mode == DESC or unit == "shape") else None)
        return ctx

    def scan_sample(self, ctx, s):
        n0 = len(ctx["ids"])
        if ctx.get("all_images"):
            img = s.image_path
            if img is not None:
                import os
                from AIO_QC.core.embeddings import file_sig
                rel = os.path.relpath(s.key, ctx["root"]).replace("\\", "/")
                ctx["keys"].append(s.key)
                ctx["idx"].append(-1)
                ctx["labels"].append((ctx["classes"] or {}).get(s.key, "") if ctx["classes"] else "")
                ctx["boxes"].append(None)
                ctx["wh"].append(None)
                ctx["images"].append(str(img))
                ctx["ids"].append(f"{rel}|image|{file_sig(img)}")
        else:
            super().scan_sample(ctx, s)
        ctx["splits"].extend([s.split] * (len(ctx["ids"]) - n0))
        return []

    def _store_key(self, ctx) -> tuple:
        return (ctx["path"], round(ctx["pad"], 3), ctx["unit"], ctx["device"], bool(ctx["half"]), ctx["label"],
                ctx.get("all_images"), hash(tuple(s.key for s in self.samples_to_scan())))

    def start_scan(self):
        """A new query on vectors already in memory: no rescan — embed the words, rank, show."""
        if self.worker is not None and self.worker.isRunning():
            return
        st = self._store
        if st is not None:
            try:
                ctx = self.make_context()
            except ValueError:
                return super().start_scan()
            if self._store_key(ctx) == st["key"]:
                for f in ("keys", "idx", "labels", "boxes", "wh", "images", "ids", "splits", "vecs", "valid",
                          "load_info"):
                    ctx[f] = st[f]
                ctx["n_new"] = 0
                self.table.setRowCount(0)
                self.rows.clear()
                self.errors.clear()
                self.ctx = ctx
                self._scan_started = time.monotonic()
                self.status.setText("Searching …")
                w = FuncWorker(lambda: self._results(ctx), self)
                w.done.connect(lambda rows: self._on_completed(rows, [], False))
                w.failed.connect(lambda e: self.status.setText(f"⚠ {e}"))
                w.finished.connect(w.deleteLater)
                w.start()
                self._qw = w
                return
        super().start_scan()

    def after_edit(self, keys):
        self._store = None                                # shapes / labels changed: the next run re-reads
        super().after_edit(keys)

    def finalize(self, ctx):
        if not ctx["ids"]:
            raise ValueError("nothing to search: no images" + (" with shapes" if ctx["unit"] == "shape" else ""))
        tmp = Path(tempfile.mkdtemp(prefix="aioqc_words_"))
        try:
            vecs, valid = self._embed_all(ctx, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if ctx["load_info"].get("text") is False:
            raise ValueError(f"{ctx['model'].get('name')} understands images only — choose a text-aware model "
                             "(SigLIP2 / SigLIP / CLIP; Download… offers them).")
        self._store = {"key": self._store_key(ctx), **{f: ctx[f] for f in (
            "keys", "idx", "labels", "boxes", "wh", "images", "ids", "splits", "load_info")},
            "vecs": vecs, "valid": valid}
        return self._results(ctx)

    def _text_vecs(self, ctx, texts: list[str]) -> np.ndarray:
        svc = self.app.embedder
        svc.ensure_loaded(ctx["python"], ctx["path"], ctx["device"], ctx["half"])
        tmp = Path(tempfile.mkdtemp(prefix="aioqc_text_"))
        try:
            svc.embed_text(texts, tmp / "t.npy", cancel=ctx["cancel"])
            return np.load(tmp / "t.npy").astype(np.float32)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _results(self, ctx) -> list[ResultRow]:
        vecs, valid = ctx["vecs"], ctx["valid"]
        if ctx["mode"] == SEARCH:
            t = self._text_vecs(ctx, [TS.phrase(ctx["query"], TS.DEFAULT_TEMPLATE if ctx["phrase"] else None)])[0]
            allowed = None
            if ctx.get("without"):
                lab = ctx["without"]
                idx = self.session.index
                allowed = np.fromiter((not (idx.get(k) or {}).get(lab) for k in ctx["keys"]), bool,
                                      count=len(ctx["keys"]))
            order, sc = TS.rank(vecs, t, valid, ctx["top"], allowed)
            allsc = vecs.astype(np.float32) @ t
            rel = TS.relative(sc, float(np.median(allsc[valid])) if valid.any() else None)
            ctx["hits"] = [int(i) for i in order]
            ctx["split_stats"] = TS.split_counts([ctx["splits"][i] for i in order],
                                                 [sp for sp, ok in zip(ctx["splits"], valid) if ok])
            rows = []
            for r, (i, s, m) in enumerate(zip(order, sc, rel), 1):
                i = int(i)
                rows.append(ResultRow(ctx["keys"][i], (r, f"{m:.0f} %", ctx["labels"][i], ctx["splits"][i] or "—"),
                                      None if ctx["idx"][i] < 0 else ctx["idx"][i], payload=("hit", i, float(s)),
                                      tone="info"))
            return rows
        # class descriptions
        data = TS.load_descriptions(ctx["root"])
        names = sorted({lab for lab, ok in zip(ctx["labels"], valid) if ok and lab})
        if not names:
            raise ValueError("no labelled objects to check")
        texts = [TS.phrase(d, data.get("template")) for d in TS.descriptions_for(names, data)]
        tmat = self._text_vecs(ctx, texts)
        rep = TS.check_descriptions(vecs, ctx["labels"], names, tmat, valid, ctx["margin"], ctx["poor"])
        ctx["report"], ctx["names"] = rep, names
        tones = {"other": "error", "poor": "muted"}
        return [ResultRow(ctx["keys"][f.i], (TS.DESC_KINDS[f.kind], f.label, f.suggest, round(f.score, 2), f.evidence),
                          None if ctx["idx"][f.i] < 0 else ctx["idx"][f.i], payload=(f, f.i), tone=tones[f.kind])
                for f in rep.findings]

    def on_scan_complete(self, ctx):
        miss = ctx.get("env_missing")
        if miss:
            self._offer_install(miss, "This model needs packages your environment does not have yet.")

    def summarize(self):
        c = self.ctx or {}
        if c.get("vecs") is None:
            return super(SimilarityDialog, self).summarize()
        unit = "images" if c["unit"] == "image" else "objects"
        n = int(np.asarray(c["valid"]).sum())
        new = c.get("n_new", 0)
        tail = f" · {new:,} newly embedded" if new else ""
        if c["mode"] == SEARCH:
            parts = [f"{sp or 'all'} {h:,} of {tot:,} ({100 * h / max(tot, 1):.1f} %)" for sp, h, tot in
                     c.get("split_stats", [])]
            return (f"“{c['query']}”: best {len(self.rows):,} of {n:,} {unit}" +
                    (f" without '{c['without']}'" if c.get("without") else "") +
                    (" · per split: " + " · ".join(parts) if parts else "") + tail)
        rep = c.get("report")
        if rep is None:
            return super(SimilarityDialog, self).summarize()
        tot = sum(rep.counts.values()) or 1
        agree = sum(rep.agreement.get(k, 0) * v for k, v in rep.counts.items()) / tot
        weak = sorted(rep.agreement.items(), key=lambda kv: kv[1])[:3]
        w = " · weakest: " + ", ".join(f"{k} {v:.0%}" + (f" (→ {rep.confused[k][0]})" if k in rep.confused else "")
                                       for k, v in weak) if len(rep.agreement) > 1 else ""
        return (f"{n:,} {unit} vs {len(rep.counts)} class descriptions · the model agrees with the labels on "
                f"{agree:.0%}{w} · {len(self.rows):,} finding(s)" + tail)

    # ------------------------------------------------------------------ actions
    def _grid(self):
        keys = list(dict.fromkeys(r.key for r in (self.target_rows() or self.visible_rows())
                                  if r.key in self.session.by_key))
        if not keys:
            QMessageBox.information(self, self.TITLE, "Nothing listed yet.")
            return
        c = self.ctx or {}
        self.app.open_review_grid(keys, f"Words: {c.get('query')}" if c.get("mode") == SEARCH else "Class descriptions")

    def _agreement(self):
        c = self.ctx or {}
        rep = c.get("report")
        if rep is None:
            QMessageBox.information(self, self.TITLE, "Run “Check labels against class descriptions” first.")
            return
        lines = ["How often the model's reading of each description agrees with the label (higher = the "
                 "description and the class are easy to tell apart):", ""]
        for k, v in sorted(rep.agreement.items(), key=lambda kv: kv[1]):
            conf = rep.confused.get(k)
            lines.append(f"  {k}: {v:.0%} of {rep.counts.get(k, 0):,}" +
                         (f" — otherwise mostly '{conf[0]}' ({conf[1]:.0%})" if conf else ""))
        if rep.close_texts:
            lines += ["", "Descriptions the model reads as nearly the same:"] + \
                [f"  {a} / {b} (similarity {v:.2f}) — say what sets them apart" for a, b, v in rep.close_texts]
        lines += ["", "A low share is not always wrong labels: the description may be too vague, or the class hard "
                      "to see. Rewrite it (Class descriptions…) and run again — the images are not embedded again."]
        self.report("Class agreement", lines, [])
