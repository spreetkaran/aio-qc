# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Review ▸ QC Report — one click, one self-contained file: where the dataset stands and what changed (core.report).

HTML (opens in any browser, images embedded) or PDF. Built in the background; the last report's headline numbers
are kept (.aio_qc/reports/last.json, a few bytes) so the next report shows the changes since.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QMarginsF, QUrl
from PyQt6.QtGui import QDesktopServices, QImage, QPageLayout, QPageSize, QPdfWriter, QTextDocument
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QSpinBox,
    QVBoxLayout,
)

from AIO_QC import __version__
from AIO_QC.core import report as R
from AIO_QC.core import thumbs as T
from AIO_QC.core.history import user
from AIO_QC.core.state import REVIEW_FIXED, REVIEW_FLAG, REVIEW_OK
from AIO_QC.utils import newButton, newIcon, style as apply_style
from .inference_service import FuncWorker


class ReportDialog(QDialog):
    TITLE = "QC Report"
    SUPPORTED = None

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(newIcon("tool-report"))
        self.setMinimumWidth(600)
        self.worker = None
        self.cancel = threading.Event()
        st = app.settings
        s = self.session
        self.fmt = QComboBox()
        self.fmt.addItem("HTML — opens in any browser, images embedded", "html")
        self.fmt.addItem("PDF — for sharing / printing", "pdf")
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(st.value("report/fmt", "html"))))
        self.fmt.currentIndexChanged.connect(lambda _i: self._ext())
        self.path = QLineEdit(str(Path(st.value("report/dir", "") or s.root.parent) /
                                  time.strftime(f"{s.root.name}_QC_report_%Y%m%d.html")))
        self.browse = newButton("Browse…", icon="save", slot=self._browse)
        self.worst = QSpinBox()
        self.worst.setRange(0, 60)
        self.worst.setValue(int(st.value("report/worst", 12)))
        self.worst.setSuffix(" files")
        self.worst.setToolTip("Riskiest files shown with a thumbnail and their findings (0 = none)")
        self.open_after = QCheckBox("Open when done")
        self.open_after.setChecked(st.value("report/open", True, type=bool))
        last = R.load_snapshot(s.root)
        since = QLabel(("Changes are listed since the last report (" +
                        time.strftime("%Y-%m-%d %H:%M", time.localtime(last["when"])) + ").") if last else
                       "First report for this dataset — the next one will list what changed after it.")
        apply_style(since, "color:@muted;")
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.go = newButton("Generate", icon="tool-report", slot=self._generate)
        self.go.setDefault(True)
        close = newButton("Close", slot=self.close)
        g = QGridLayout()
        g.addWidget(QLabel("Format:"), 0, 0)
        g.addWidget(self.fmt, 0, 1, 1, 2)
        g.addWidget(QLabel("Save as:"), 1, 0)
        g.addWidget(self.path, 1, 1)
        g.addWidget(self.browse, 1, 2)
        g.addWidget(QLabel("Riskiest:"), 2, 0)
        row = QHBoxLayout()
        row.addWidget(self.worst)
        row.addStretch(1)
        row.addWidget(self.open_after)
        g.addLayout(row, 2, 1, 1, 2)
        expl = QLabel("Summary · class balance per split · review coverage · findings by tool · QC sample estimate · "
                      "riskiest files with thumbnails · changes since the last report.")
        expl.setWordWrap(True)
        apply_style(expl, "color:@muted;")
        foot = QHBoxLayout()
        foot.addWidget(self.progress, 1)
        foot.addWidget(self.go)
        foot.addWidget(close)
        lay = QVBoxLayout(self)
        lay.addWidget(expl)
        lay.addLayout(g)
        lay.addWidget(since)
        lay.addWidget(self.status)
        lay.addLayout(foot)
        self._ext()

    def _ext(self):
        p = Path(self.path.text().strip() or "report")
        self.path.setText(str(p.with_suffix("." + self.fmt.currentData())))

    def _browse(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save report", self.path.text(),
                                           "HTML (*.html)" if self.fmt.currentData() == "html" else "PDF (*.pdf)")
        if f:
            self.path.setText(f)
            self._ext()

    # ------------------------------------------------------------------ snapshot (GUI thread, cheap copies)
    def _snapshot(self) -> dict:
        s = self.session
        from AIO_QC.core.sampling import SamplePlan, estimate
        from AIO_QC.core.state import list_samples
        alive = set(s.by_key)
        iss = s.issues
        files_per = {t: sum(1 for k in d if k in alive) for t, d in iss.by_tool.items()}
        ranked = iss.ranked(alive)[: self.worst.value()]
        by_file = {k: [(i.severity, iss.titles.get(i.tool, i.tool), i.text) for i in iss.file_issues(k)]
                   for k, _r in ranked}
        samples_qc = []
        for d in list_samples(s.root)[:5]:
            try:
                plan = SamplePlan.from_json(d)
                est, _per = estimate(plan, s.review.status, plan.params.get("confidence", 0.95),
                                     plan.params.get("max_error", 0.05))
                samples_qc.append((plan.name, R.estimate_dict(est)))
            except Exception as exc:
                logger.debug(f"[report] sample {d.get('name')}: {exc}")
        return {"root": str(s.root), "name": s.root.name, "type": s.type.title, "version": __version__,
                "user": user(), "samples": [(x.key, x.split, x.display) for x in s.samples],
                "index": dict(s.index), "status": {k: s.review.status(k) for k in s.keys_with_review(
                    {REVIEW_OK, REVIEW_FLAG, REVIEW_FIXED})},
                "issues": {"summary": iss.summary(alive), "titles": dict(iss.titles), "when": dict(iss.when),
                           "files_per_tool": files_per, "ranked": ranked, "by_file": by_file},
                "samples_qc": samples_qc}

    def _generate(self):
        s = self.session
        out = Path(self.path.text().strip())
        fmt = self.fmt.currentData()
        st = self.app.settings
        st.setValue("report/fmt", fmt)
        st.setValue("report/dir", str(out.parent))
        st.setValue("report/worst", self.worst.value())
        st.setValue("report/open", self.open_after.isChecked())
        snap = self._snapshot()
        last = R.load_snapshot(s.root)
        since = last["when"] if last else time.time()
        adapter = s.adapter
        by_key = s.by_key
        hist = s.history

        def job():
            events = hist.since(since) if last else []
            data = R.gather(snap, events, last)
            thumbs = []
            colors: dict = {}
            for k, risk in snap["issues"]["ranked"]:
                smp = by_key.get(k)
                if smp is None:
                    continue
                jpeg = None
                try:
                    ann = adapter.load(smp, light=True)
                    img = smp.image_path or ann.image_path
                    if img is not None:
                        jpeg = T.render(img, T.shape_boxes(ann, colors), max_side=440)
                except Exception:
                    jpeg = None
                thumbs.append({"key": k, "display": smp.display, "risk": risk,
                               "status": snap["status"].get(k), "issues": snap["issues"]["by_file"].get(k, []),
                               "jpeg": jpeg})
            html, images = R.render(data, thumbs)
            return data, html, images

        self.go.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("Building the report…")
        self.worker = FuncWorker(job, self)
        self.worker.done.connect(lambda r: self._write(r, out, fmt))
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _failed(self, msg):
        self.worker = None
        self.progress.setVisible(False)
        self.go.setEnabled(True)
        self.status.setText(f"⚠ Report failed: {msg}")
        logger.error(f"[report] {msg}")

    def _write(self, result, out: Path, fmt: str):
        self.worker = None
        self.progress.setVisible(False)
        self.go.setEnabled(True)
        data, html, images = result
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "html":
                out.write_text(R.inline(html, images), encoding="utf-8")
            else:
                self._pdf(html, images, out)
            R.save_snapshot(self.session.root, data)
        except Exception as exc:
            self._failed(f"{type(exc).__name__}: {exc}")
            return
        self.status.setText(f"✓ Saved {out}")
        logger.info(f"[report] {out}")
        if self.open_after.isChecked():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    @staticmethod
    def _pdf(html: str, images: dict, out: Path):
        doc = QTextDocument()
        for ref, data in images.items():
            img = QImage()
            img.loadFromData(data)
            doc.addResource(QTextDocument.ResourceType.ImageResource.value, QUrl(ref), img)
        doc.setHtml(html)
        w = QPdfWriter(str(out))
        w.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        w.setPageMargins(QMarginsF(12, 12, 12, 12), QPageLayout.Unit.Millimeter)
        w.setResolution(110)
        w.setTitle("QC Report")
        doc.setPageSize(w.pageLayout().paintRectPixels(w.resolution()).size().toSizeF())
        doc.print(w)

    def closeEvent(self, e):
        if self.worker is not None:
            self.worker.wait(5000)
        if hasattr(self.app, "forget_dialog"):
            self.app.forget_dialog(self)
        super().closeEvent(e)

