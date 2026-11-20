# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Testing (Model Lab) — step 5, the report; jobs; projects and the start page.

A mixin of AIO_QC.widgets.testing.model_lab.ModelLab: the methods use the window's own attributes, set up in its __init__.
"""
import os
import time
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QComboBox, QCheckBox, QRadioButton, QButtonGroup, QGroupBox, QLineEdit, QFileDialog, QMessageBox, QFrame, QMenu,
)
from AIO_QC.widgets.testing.lab_common import (
    Job,
    open_path,
)
from AIO_QC.tools.frame_extractor.core.common import VIDEO_FILTER

from AIO_QC.widgets.testing.model_lab_common import (_hint)


class ReportMixin:

    # ------------------------------------------------------------------ 5 report
    def _page_report(self):
        w, lay = self._page("Report", step=4, subtitle="A single HTML file - charts, tables and images inside, opens in any browser, "
                            "light and dark, prints well. Data files (metrics.json, per-class and per-image CSV) "
                            "go next to it.")
        box = QGroupBox("Type")
        bl = QVBoxLayout(box)
        self.rep_btns = {}
        g = QButtonGroup(self)
        for key, title, desc in (
                ("summary", "Summary", "One page for a decision: verdict, leaderboard, precision-recall, per-class "
                                       "accuracy, error types, speed."),
                ("detailed", "Detailed", "Everything: all curves, recall by object size, class balance, and per "
                                         "model a confusion matrix, per-class table and its hardest images."),
                ("comparison", "Comparison (A vs B)", "Two models head to head: differences per class, curves, "
                                                      "confusion matrices, and the images where each one wins."),
                ("video", "Video behaviour", "Label-free: detections over time, flicker and blips, agreement "
                                             "between models, moments worth a look.")):
            rb = QRadioButton(title)
            rb.setToolTip(desc)
            g.addButton(rb)
            self.rep_btns[key] = rb
            bl.addWidget(rb)
            bl.addWidget(_hint("      " + desc))
            if key == "comparison":
                row = QHBoxLayout()
                row.addSpacing(24)
                self.rep_a = QComboBox()
                self.rep_b = QComboBox()
                row.addWidget(self.rep_a)
                row.addWidget(QLabel("vs"))
                row.addWidget(self.rep_b)
                row.addStretch(1)
                bl.addLayout(row)
        self.rep_btns["summary"].setChecked(True)
        lay.addWidget(box)
        f = QFormLayout()
        self.rep_title = QLineEdit()
        self.rep_title.setPlaceholderText("Report title (default: Model evaluation · <test data>)")
        f.addRow("Title", self.rep_title)
        row = QHBoxLayout()
        self.rep_dir = QLineEdit()
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: (lambda d: d and self.rep_dir.setText(d))(
            QFileDialog.getExistingDirectory(self, "Report folder", self.rep_dir.text())))
        row.addWidget(self.rep_dir, 1)
        row.addWidget(b)
        f.addRow("Folder", row)
        self.rep_data = QCheckBox("Also write metrics.json, per_class.csv, per_image.csv")
        self.rep_data.setChecked(True)
        f.addRow("", self.rep_data)
        lay.addLayout(f)
        row = QHBoxLayout()
        self.rep_go = QPushButton("Create report")
        self.rep_go.setMinimumHeight(32)
        self.rep_go.clicked.connect(self._make_report)
        self.rep_open = QPushButton("Open report")
        self.rep_open.clicked.connect(lambda: self._report_path and open_path(self._report_path))
        self.rep_open.setEnabled(False)
        self.rep_folder = QPushButton("Show folder")
        self.rep_folder.clicked.connect(lambda: self._report_path and open_path(os.path.dirname(self._report_path)))
        self.rep_folder.setEnabled(False)
        row.addWidget(self.rep_go)
        row.addWidget(self.rep_open)
        row.addWidget(self.rep_folder)
        row.addStretch(1)
        lay.addLayout(row)
        self.rep_status = QLabel("")
        self.rep_status.setWordWrap(True)
        lay.addWidget(self.rep_status)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        lay.addWidget(sep)
        lay.addWidget(QLabel("<b>Annotated video</b>"))
        lay.addWidget(_hint("Draw up to four models on a video (one picture or side by side) - to show a result, or "
                            "to watch models behave. Uses the cached predictions."))
        row = QHBoxLayout()
        self.ann_btn = QPushButton("Export annotated video…")
        self.ann_btn.clicked.connect(self._annotated)
        row.addWidget(self.ann_btn)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)
        return w

    def _refresh_report_page(self):
        r = self.run
        self._pair_combos(self.rep_a, self.rep_b)
        has_img = r is not None and bool(r.scored)
        can_cmp = has_img and (len(r.scored) > 1 or (r.ground == "reference" and r.reference is not None
                                                     and len(r.specs) > 1))
        for k in ("summary", "detailed", "comparison"):
            self.rep_btns[k].setEnabled(bool(has_img if k != "comparison" else can_cmp))
        for combo in (self.rep_a, self.rep_b):
            combo.setEnabled(bool(can_cmp))
        self.rep_btns["video"].setEnabled(bool(self.videos_out))
        if not any(b.isChecked() and b.isEnabled() for b in self.rep_btns.values()):
            for b in self.rep_btns.values():
                if b.isEnabled():
                    b.setChecked(True)
                    break
        if not self.rep_dir.text():
            if self.testset is not None:
                self.rep_dir.setText(self.testset.reports_dir)
            else:
                self.rep_dir.setText(self.st.value("report_dir", os.path.join(os.path.expanduser("~"), "Documents")))
        ok = has_img or bool(self.videos_out)
        self.rep_go.setEnabled(ok)
        self.rep_status.setText("" if ok else "Run the models first (page 3).")

    def _make_report(self):
        kind = next((k for k, b in self.rep_btns.items() if b.isChecked()), "summary")
        folder = self.rep_dir.text().strip() or os.getcwd()
        os.makedirs(folder, exist_ok=True)
        self.st.setValue("report_dir", folder)
        src = self.data.label if self.data is not None else (os.path.basename(self.videos_out[0][0]) if self.videos_out else "")
        title = self.rep_title.text().strip() or (
            ("Model comparison" if kind == "comparison" else "Video behaviour" if kind == "video" else "Model evaluation")
            + (f" · {src}" if src else ""))
        safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in title).strip().replace(" ", "_")[:60]
        path = os.path.join(folder, f"{safe}_{kind}_{time.strftime('%Y%m%d_%H%M')}.html")
        pair = (self.rep_a.currentData(), self.rep_b.currentData()) if kind == "comparison" else None
        if pair is not None and (None in pair or pair[0] == pair[1]):
            QMessageBox.information(self, "Report", "Pick two different models to compare.")
            return
        run, vids, slots, data_files = self.run, self.videos_out, self._run_slots(self.run, self.videos_out), \
            self.rep_data.isChecked()
        if kind == "video" and run is None:
            slots = self._run_slots(None, vids)
        version = self.version

        def work(job):
            from AIO_QC.core.testing import pipeline as P
            job.progress.emit((0, 1, "Drawing charts and images…"))
            return P.make_report(path, kind, run, vids if kind in ("video", "detailed") else [], title, slots=slots,
                                 pair=pair, exports=data_files, version=version)

        def done(paths):
            self._report_path = paths[0]
            self.rep_open.setEnabled(True)
            self.rep_folder.setEnabled(True)
            kb = os.path.getsize(paths[0]) / 1024
            self.rep_status.setText(f"Saved {os.path.basename(paths[0])} ({kb:,.0f} KB)"
                                    + (f" + {len(paths) - 1} data files" if len(paths) > 1 else ""))
            open_path(paths[0])
        self._start(work, done, button=self.rep_go, busy="Working…",
                    failed=lambda m: self.rep_status.setText(f"Failed: {m}"))

    def _annotated(self):
        specs = self._sync_specs()
        if not specs:
            QMessageBox.information(self, "Testing", "Add models first.")
            return
        vids = [v["path"] for v in self.analysis_videos()]
        if self.testset is not None:
            vids += [s["path"] for s in self.testset.manifest["sources"] if s.get("type") == "video"]
        vids = [v for i, v in enumerate(vids) if v not in vids[:i] and os.path.isfile(v)]
        video = None
        if vids:
            m = QMenu(self)
            for v in vids:
                m.addAction(os.path.basename(v)).setData(v)
            m.addSeparator()
            m.addAction("Other video…").setData("")
            act = m.exec(self.ann_btn.mapToGlobal(self.ann_btn.rect().bottomLeft()))
            if act is None:
                return
            video = act.data()
        if not video:
            video, _ = QFileDialog.getOpenFileName(self, "Video", self.st.value("video_dir", ""), VIDEO_FILTER)
        if not video:
            return
        self.st.setValue("video_dir", os.path.dirname(video))
        ff = self.ffmpeg_getter()
        if not ff:
            QMessageBox.warning(self, "Testing", "ffmpeg was not found (Settings → Playback → ffmpeg location).")
            return
        from AIO_QC.tools.frame_extractor.core.sources import open_source
        from AIO_QC.tools.frame_extractor.widgets.export_dialogs import AnnotatedExportDialog
        src = open_source(video, "auto", hw_accel=False)
        dur = src.duration()
        src.release()
        order = sorted(range(len(specs)), key=lambda i: self.slots[i])
        dlg = AnnotatedExportDialog(self, video, [specs[i] for i in order], ff, 0.0, dur, dur, cache=self.cache,
                                    conf=self.thr_val.value() if self.thr_fixed.isChecked() else 0.25,
                                    slots=[self.slots[i] for i in order])
        dlg.exec()

    # ------------------------------------------------------------------ jobs
    def _start(self, fn, done, bar=None, button=None, busy=None, failed=None, progress=None):
        if self.job is not None:
            QMessageBox.information(self, "Testing", "Something is still running - wait for it or stop it first.")
            return
        old = button.text() if button is not None else None
        job = Job(fn, self)
        self.job = job
        from AIO_QC.widgets import tasks
        what = (old or "").strip(" ▶■") or self.status.currentMessage() or "working"
        task = tasks.start(f"Testing · {what}", cancel=job.cancel)
        if bar is not None:
            bar.setVisible(True)
            bar.setValue(0)
            bar.setFormat("")
        if button is not None and busy:
            button.setText(busy)

        def prog(p):
            if isinstance(p, tuple) and len(p) >= 3 and isinstance(p[0], (int, float)) and \
                    isinstance(p[1], (int, float)):
                task.update(p[0] / p[1] if p[1] else None, p[2] if isinstance(p[2], str) else None)
            elif isinstance(p, tuple) and len(p) >= 4 and isinstance(p[2], (int, float)) and \
                    isinstance(p[3], (int, float)):
                task.update(p[2] / p[3] if p[3] else None)
            if progress is not None:
                progress(p)
                return
            if bar is not None and isinstance(p, tuple) and len(p) >= 3:
                d, t, text = p[0], p[1], p[2]
                bar.setValue(int(1000 * d / t) if t else 0)
                if text:
                    self.status.showMessage(str(text))
        state = {"clean": False}

        def cleanup():
            # runs once, BEFORE the result handler: a handler may start the next job (e.g. reload after build)
            if state["clean"]:
                return
            state["clean"] = True
            task.finish()
            if self.job is job:
                self.job = None
            if bar is not None:
                bar.setVisible(False)
            if button is not None and old is not None:
                button.setText(old)
            self.status.clearMessage()
            self._revalidate()

        def on_done(r):
            cleanup()
            done(r)

        def on_failed(m):
            cleanup()
            if failed is not None:
                failed(m)
            else:
                self._log(f"Error: {m}")
                QMessageBox.warning(self, "Testing", m)
        job.progress.connect(prog)
        job.log.connect(self._log)
        job.done.connect(on_done)
        job.failed.connect(on_failed)
        job.finished.connect(cleanup)
        job.start()
