# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Testing (Model Lab) — step 3, running the models.

A mixin of AIO_QC.widgets.testing.model_lab.ModelLab: the methods use the window's own attributes, set up in its __init__.
"""
import os
import time
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QComboBox, QDoubleSpinBox,
    QCheckBox, QRadioButton, QButtonGroup, QGroupBox, QMessageBox, QPlainTextEdit,
)
from AIO_QC.widgets import motion as anim
from AIO_QC.widgets.testing.lab_common import (
    ProgressRow,
)

from AIO_QC.widgets.testing.model_lab_common import (_swatch_icon, slot_hex)


class RunMixin:

    # ------------------------------------------------------------------ 3 run
    def _page_run(self):
        w, lay = self._page("Run", step=2, subtitle="Every model on every test image (and analysed video). Predictions are cached: "
                            "stop any time, continue later; a model already run on these images costs nothing.")
        box = QGroupBox("Scoring")
        f = QFormLayout(box)
        self.ground = QComboBox()
        self.ground.addItem("Labels when present, else a reference model", "auto")
        self.ground.addItem("Labels (true accuracy)", "labels")
        self.ground.addItem("A reference model's detections", "reference")
        self.ground.setToolTip("Without labels, one model's detections can serve as the answer key: the others are "
                               "scored on how closely they reproduce it (useful to compare a new model with the one "
                               "in production - not true accuracy).")
        f.addRow("Ground truth", self.ground)
        self.reference = QComboBox()
        self.reference.setToolTip("Reference model (when there are no labels) and for video agreement.")
        f.addRow("Reference model", self.reference)
        self.unreviewed = QCheckBox("Also score unreviewed pre-labels")
        self.unreviewed.setToolTip("Off (recommended): images whose labels came from a model and were never edited "
                                   "are left out - they would favour that model.")
        f.addRow("", self.unreviewed)
        thr = QHBoxLayout()
        self.thr_best = QRadioButton("Best F1 per model")
        self.thr_best.setToolTip("Each model at the confidence where its precision/recall balance (F1) is best - "
                                 "the fair way to compare models.")
        self.thr_fixed = QRadioButton("Fixed")
        g = QButtonGroup(self)
        g.addButton(self.thr_best)
        g.addButton(self.thr_fixed)
        self.thr_best.setChecked(True)
        self.thr_val = QDoubleSpinBox()
        self.thr_val.setRange(0.01, 0.99)
        self.thr_val.setSingleStep(0.05)
        self.thr_val.setValue(0.25)
        thr.addWidget(self.thr_best)
        thr.addWidget(self.thr_fixed)
        thr.addWidget(self.thr_val)
        thr.addStretch(1)
        f.addRow("Confidence threshold", thr)
        lay.addWidget(box)
        row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.clicked.connect(self._run_all)
        self.run_info = QLabel("")
        row.addWidget(self.run_btn)
        row.addWidget(self.run_info, 1)
        lay.addLayout(row)
        self.rows_box = QWidget()
        self.rows_lay = QVBoxLayout(self.rows_box)
        self.rows_lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.rows_box)
        self.run_bar = anim.SmoothProgressBar()
        self.run_bar.setRange(0, 1000)
        self.run_bar.setVisible(False)
        lay.addWidget(self.run_bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("Log")
        lay.addWidget(self.log, 1)
        return w

    def _refresh_run_page(self):
        specs = self._sync_specs()
        cur = self.reference.currentIndex()
        self.reference.clear()
        self.reference.addItem("(none)", -1)
        for i, s in enumerate(specs):
            self.reference.addItem(_swatch_icon(slot_hex(self.slots[i])), s.label, i)
        want = self._pending_reference
        if want is not None and self.reference.findData(want) >= 0:
            self.reference.setCurrentIndex(self.reference.findData(want))
            self._pending_reference = None
        elif 0 <= cur < self.reference.count():
            self.reference.setCurrentIndex(cur)
        elif specs:
            self.reference.setCurrentIndex(1)
        for i in reversed(range(self.rows_lay.count())):
            wdg = self.rows_lay.itemAt(i).widget()
            if wdg:
                self.rows_lay.removeWidget(wdg)
                wdg.setParent(None)
        self.prog_rows = []
        for i, s in enumerate(specs):
            pr = ProgressRow(s.label, slot_hex(self.slots[i]))
            self.rows_lay.addWidget(pr)
            self.prog_rows.append(pr)
        vids = self.analysis_videos()
        n_img = len(self.data.dataset.items) if self.data is not None and not self.kind_none.isChecked() else 0
        bits = []
        if n_img:
            bits.append(f"{n_img:,} images")
        if vids:
            bits.append(f"{len(vids)} video(s)")
        self.run_info.setText(f"{len(specs)} model(s) × " + (" + ".join(bits) if bits else "nothing yet - add test data "
                                                                                          "or videos on page 1"))
        self._revalidate()

    def _log(self, text):
        self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")

    def _run_all(self, quiet=False):
        if self.job is not None:
            if self.run_btn.text().startswith("■"):     # our own run: this click stops it
                self.job.cancel()
                self._log("Stopping… (finished predictions are kept)")
            else:                                      # e.g. the project still loading: never cancel that
                self.status.showMessage("Busy — wait for the current task to finish", 4000)
            return
        specs = self._sync_specs()
        if not specs:
            return
        if not self.prog_rows:
            self._refresh_run_page()
        for s in specs:
            if not os.path.isfile(s.path):
                QMessageBox.warning(self, "Testing", f"Weights not found:\n{s.path}")
                return
        self.st.setValue("batch", self.batch_spin.value())
        self.st.setValue("env_python", self.env_edit.text().strip())
        from AIO_QC.core.testing import pipeline as P
        data = None if self.kind_none.isChecked() else self.data
        if data is not None and not data.dataset.items:
            data = None
        vids = self.analysis_videos()
        cache = self.cache if data is not None else P.cache_for()
        self.cache = cache
        ground = self.ground.currentData()
        ref = self.reference.currentData()
        ref = None if ref is None or ref < 0 else ref
        op_conf = None if self.thr_best.isChecked() else self.thr_val.value()
        only_rev = not self.unreviewed.isChecked()
        batch = self.batch_spin.value()
        every = self.vid_every.value()
        rows = self.prog_rows
        n_models = len(specs)

        last = [0.0]

        def work(job):
            run = infos = None
            if data is not None:
                def prog(mi, done, total, ms, text):
                    now = time.monotonic()
                    if now - last[0] > 0.066 or done >= total or "loading" in text:     # <= 15 UI updates/s
                        last[0] = now
                        job.progress.emit(("img", mi, done, total, ms, text))
                run, infos = P.evaluate(data, specs, cache, only_reviewed=only_rev, ground=ground, reference=ref,
                                        op_conf=op_conf, batch=batch, progress=prog, cancel=job.cancelled,
                                        log=job.log.emit)
            out = []
            if vids and not job.cancelled():
                out = P.analyse_videos(vids, specs, cache, every_s=every, op_conf=op_conf or 0.25, reference=ref,
                                       batch=batch, cancel=job.cancelled, log=job.log.emit,
                                       progress=lambda vi, d, t, txt: job.progress.emit(("vid", vi, d, t, None, txt)))
            if run is None and out and not job.cancelled():
                # videos only: score the sampled frames (vs the reference model) so every result tab has content
                run = P.video_run(out, specs, cache, ground, ref, op_conf)
                if run is not None:
                    how = (f"vs reference {specs[run.reference].label}" if run.ground == "reference"
                           else "no reference model - plots and behaviour only")
                    job.log.emit(f"Videos: {len(run.items):,} sampled frames in the result views ({how})")
            return run, infos, out, job.cancelled()

        def progress(p):
            kind, i, done, total, ms, text = p
            if kind == "img" and i < len(rows):
                extra = f" · {ms:.1f} ms/img" if ms else ""
                rows[i].set(done, total, f"{done:,}/{total:,}{extra}" if "loading" not in text else text)
                self.run_bar.setValue(int(1000 * (i + (done / total if total else 0)) / max(1, n_models)))
            elif kind == "vid":
                self.run_bar.setValue(int(1000 * done / total) if total else 0)
                self.run_bar.setFormat(text)
                self.status.showMessage(f"Video {i + 1}/{len(vids)}: {text}")

        def done(res):
            run, infos, vout, cancelled = res
            self.run, self.infos, self.videos_out = run, infos or {}, vout
            for i, s in enumerate(specs):
                info = (infos or {}).get(s.key) or {}
                if i < len(rows):
                    if info.get("error"):
                        rows[i].set(0, 1, "failed")
                        rows[i].text.setToolTip(info["error"])
                    elif info:
                        ms = info.get("speed_ms")
                        rows[i].set(1, 1, f"done · {info.get('ran', 0):,} new" + (f" · {ms:.1f} ms/img" if ms else ""))
            secs = time.monotonic() - self._t_run
            self._log(("Stopped" if cancelled else "Finished") + f" in {secs:.1f} s")
            if run is not None and not run.scored and run.ground == "none":
                self._log("No ground truth: add labels or choose a reference model to get accuracy numbers.")
            from AIO_QC.core.testing import project as PJ
            self._run_specs_keys = [s.key for s in specs]
            self._summary = PJ.summarize(run, vout)
            self._show_results()
            self._changed()
            if self._project_path:
                PJ.add_recent(self._project_path, dict(name=self._project_name, summary=self._summary))
            if not quiet and ((run is not None and run.scored) or vout):
                self.nav.setCurrentRow(3)
        for r in rows:
            r.set(0, 1, "waiting")
        self._t_run = time.monotonic()
        self._log(("Loading cached results: " if quiet else "Run: ") + f"{n_models} model(s)"
                  + (f" on {len(data.dataset.items):,} images" if data else "") + (f" + {len(vids)} video(s)" if vids else ""))
        self._start(work, done, bar=self.run_bar, button=self.run_btn, busy="■  Stop", progress=progress)
