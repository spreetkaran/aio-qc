# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
🩺 Image Quality — integrity (corrupt / truncated files, EXIF rotation and size disagreements: errors) and low-priority
information (blurry, under- / over-exposed, low contrast). Also: the background IntegritySweep that checks every image
after a dataset opens, and the per-image check behind the viewer's red banner.
"""
from __future__ import annotations

from collections import Counter

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QTabWidget, QWidget

from AIO_QC.core import image_quality as IQ
from AIO_QC.utils import hexc
from .base_scan import BaseScanDialog, ResultRow, dspin
from .card_gallery import Card, CardView, Panel

SIG_INTEGRITY = "integrity|v1"          # structural + EXIF header (cache key)
SIG_FULL = "iq|v1"                      # full decode + metrics
TONE = {IQ.ERROR: "error", IQ.WARN: "warning", IQ.INFO: "muted"}


def image_of(adapter, sample):
    """(image path, annotation width, height) — light load (LabelMe knows the annotated size)."""
    try:
        ann = adapter.load(sample, light=True)
    except Exception:
        return sample.image_path, None, None
    return (ann.image_path or sample.image_path), (None if ann.normalized else ann.width), \
        (None if ann.normalized else ann.height)


def integrity_findings(cache, path, ann_w, ann_h) -> list[tuple]:
    """Structural integrity + EXIF / size — cached per file version."""
    data = cache.get_preds(SIG_INTEGRITY, path) if cache is not None else None
    if data is None:
        data = {"integrity": [list(x) for x in IQ.integrity(path)], "exif": IQ.exif_info(path)}
        if cache is not None:
            cache.put_preds(SIG_INTEGRITY, path, data)
    out = [tuple(x) for x in data["integrity"]]
    if not any(s == IQ.ERROR for _c, s, _m in out) and data.get("exif"):
        out += IQ.exif_and_size(path, ann_w, ann_h)
    return out


class IntegritySweep(QThread):
    """Low-priority background check of every image (cached): emits findings in batches."""
    batch = pyqtSignal(dict, list)          # {key: [(code, severity, message)]}, keys checked
    progress = pyqtSignal(int, int)

    def __init__(self, samples, adapter, cache, parent=None):
        super().__init__(parent)
        self.samples, self.adapter, self.cache = list(samples), adapter, cache
        self._stop = False

    def stop(self):
        self._stop = True
        self.wait(3000)

    def run(self):
        self.setPriority(QThread.Priority.LowestPriority)
        found, checked = {}, []
        for i, s in enumerate(self.samples, 1):
            if self._stop:
                return
            path, w, h = image_of(self.adapter, s)
            if path is not None:
                try:
                    f = integrity_findings(self.cache, path, w, h)
                except Exception as exc:
                    f = [("unreadable", IQ.ERROR, str(exc))]
                if f:
                    found[s.key] = f
            checked.append(s.key)
            if len(checked) >= 400 or i == len(self.samples):
                self.batch.emit(found, checked)
                self.progress.emit(i, len(self.samples))
                found, checked = {}, []
        if self.cache is not None:
            self.cache.flush()


class ImageQualityDialog(BaseScanDialog):
    TITLE, ICON = "Image Quality", "🩺"
    HELP = ("Errors: corrupt or truncated image files (“Corrupt JPEG data”, “Premature end of JPEG file”, bad PNG "
            "chunks), images that cannot be decoded, EXIF rotation that disagrees with the annotation, annotation size ≠ "
            "image size. Information only: blurry, under- / over-exposed and low-contrast images.")
    COLUMNS = ["Issue", "Severity", "Detail"]
    FILTER_COLUMN = 0

    def build_params(self, form):
        self.chk_decode = QCheckBox("Also decode every image fully (slower; finds broken compressed data)")
        self.chk_info = QCheckBox("Blur, exposure and contrast (information only)")
        self.chk_info.setChecked(True)
        form.addRow(self.chk_decode)
        form.addRow(self.chk_info)

    def post_init(self):
        t = IQ.DEFAULT_THRESHOLDS
        live = QWidget()
        lv = QHBoxLayout(live)
        lv.setContentsMargins(0, 0, 0, 0)
        title = QLabel("⚡ Info thresholds")
        title.setStyleSheet("font-weight:700;")
        lv.addWidget(title)
        self.thr_w = {}
        for k, text, lo, hi, dec, step in (("blur", "sharpness <", 0, 2000, 0, 5), ("dark_mean", "dark <", 0, 255, 0, 5),
                                           ("bright_mean", "bright >", 0, 255, 0, 5), ("low_std", "contrast <", 0, 128, 0, 2)):
            w = dspin(lo, hi, t[k], dec, step)
            w.valueChanged.connect(lambda _v: self._retimer.start())
            self.thr_w[k] = w
            lv.addWidget(QLabel(text))
            lv.addWidget(w)
        lv.addStretch(1)
        self.table_host.insertWidget(0, live)
        self._retimer = QTimer(self, singleShot=True, interval=250, timeout=self._reevaluate)
        self.init_marks()
        self.cards = CardView(self, n_panels=1)
        self.tabs = QTabWidget()
        self.table_host.removeWidget(self.table)
        self.tabs.addTab(self.cards, "Cards")
        self.tabs.addTab(self.table, "Table")
        self.table_host.addWidget(self.tabs)
        self.filter_box.currentIndexChanged.connect(lambda _i: self._refresh_cards())

    def _thr(self):
        return {**IQ.DEFAULT_THRESHOLDS, **{k: w.value() for k, w in self.thr_w.items()}}

    def make_context(self):
        return {"adapter": self.session.adapter, "cache": self.session.cache, "decode": self.chk_decode.isChecked(),
                "info": self.chk_info.isChecked(), "metrics": {}, "thr": self._thr()}

    def scan_sample(self, ctx, s):
        path, w, h = image_of(ctx["adapter"], s)
        if path is None:
            return [self._row(s.key, ("unreadable", IQ.ERROR, "image not found"))]
        found = integrity_findings(ctx["cache"], path, w, h)
        broken = any(sev == IQ.ERROR and c not in ("exif_rotated", "size_mismatch") for c, sev, _m in found)
        if (ctx["decode"] or ctx["info"]) and not broken:
            data = ctx["cache"].get_preds(SIG_FULL, path) if ctx["cache"] is not None else None
            if data is None or (ctx["decode"] and "decode" not in data):
                data = dict(data or {})
                if ctx["decode"]:
                    data["decode"] = [list(x) for x in IQ.integrity(path, decode=True) if x[0] == "decode"]
                if "metrics" not in data:
                    data["metrics"] = IQ.metrics(path)
                if ctx["cache"] is not None:
                    ctx["cache"].put_preds(SIG_FULL, path, data)
            found += [tuple(x) for x in data.get("decode", [])] if ctx["decode"] else []
            if ctx["info"] and data.get("metrics"):
                ctx["metrics"][s.key] = data["metrics"]
                found += IQ.judge_metrics(data["metrics"], ctx["thr"])
        return [self._row(s.key, f) for f in found]

    def _row(self, key, f):
        code, sev, msg = f
        return ResultRow(key, (IQ.TITLES.get(code, code), {"error": "error", "warn": "warning", "info": "info"}[sev], msg),
                         payload=f, tone=TONE[sev])

    def issue_text(self, row):
        return f"{row.cells[0]}: {row.cells[2]}"

    def finalize(self, ctx):
        if ctx["cache"] is not None:
            ctx["cache"].flush()
        return []

    def _reevaluate(self):
        """New thresholds: rebuild the information rows from the cached metrics (no rescan)."""
        if not self.ctx or self.worker is not None and self.worker.isRunning():
            return
        thr = self._thr()
        keep = [r for r in self.rows if r is not None and r.payload[1] != IQ.INFO]
        info = [self._row(k, f) for k, m in self.ctx.get("metrics", {}).items() for f in IQ.judge_metrics(m, thr)]
        self.table.setRowCount(0)
        self.rows = []
        self._append_rows(keep + info)
        self._rebuild_filter()
        self.summary.setText(self.summarize())
        self.publish_issues(getattr(self, "_scan_scope", None), self.rows)

    def on_scan_complete(self, ctx):
        bad = sorted({r.key for r in self.rows if r and r.payload[1] == IQ.ERROR})
        if bad:
            self.app.notify_broken_images(bad, source=self.TITLE)

    def _refresh_cards(self):
        if not hasattr(self, "cards"):
            return
        self.prune_marks()
        cards = []
        for row in self.visible_rows()[:5000]:
            code, sev, msg = row.payload
            s = self.session.by_key.get(row.key)
            cards.append(Card(row, [Panel(row.key, (0, 0, 1, 1), [], normalized=True)],
                              [(IQ.TITLES.get(code, code), {"error": "error", "warn": "warn"}.get(sev, "muted"))],
                              [msg, s.display if s else row.key]))
        self.cards.set_cards(cards)
        self.tabs.setTabText(0, f"Cards ({len(cards):,})")

    def summarize(self):
        self._refresh_cards()
        c = Counter((r.payload[1], r.cells[0]) for r in self.rows if r)
        err = sum(n for (s, _t), n in c.items() if s == IQ.ERROR)
        parts = [f"{n:,} {t}" for (s, t), n in sorted(c.items(), key=lambda x: (IQ.SEVERITY_ORDER[x[0][0]], -x[1]))]
        lead = (f"<span style='color:{hexc('error')}'>⚠ {err:,} broken / inconsistent image(s)</span> · " if err else "")
        return (lead + " · ".join(parts)) if parts else "✅ Every image is intact"
