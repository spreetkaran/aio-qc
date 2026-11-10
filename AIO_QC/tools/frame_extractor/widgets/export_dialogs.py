# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Export dialogs

  ClipExportDialog       cut a clip (in -> out): Fast = lossless stream copy that starts on the keyframe at or
                         before the in point (instant); Exact = re-encoded, frame-accurate (GPU encoder when one
                         works on this machine)
  AnnotatedExportDialog  a video with 1-4 models' detections drawn (one picture, or side by side), track IDs,
                         a small info overlay; predictions cached, so re-exports and Model Lab runs share work
Both run in the background, can be cancelled, and never leave a half-written file behind.
"""
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton, QRadioButton, QButtonGroup,
    QCheckBox, QDoubleSpinBox, QComboBox, QFileDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QMessageBox,
)

from AIO_QC.tools.frame_extractor.core.common import model_hex
from AIO_QC.widgets import motion as anim
from AIO_QC.widgets.testing.lab_common import Job, human_time, open_path
from AIO_QC.utils.fx_settings import fx_settings



def _time_spin(value, maximum):
    s = QDoubleSpinBox()
    s.setDecimals(3)
    s.setRange(0.0, max(0.001, maximum))
    s.setSingleStep(0.1)
    s.setSuffix(" s")
    s.setValue(value)
    s.setKeyboardTracking(False)
    return s


class _ExportBase(QDialog):
    def __init__(self, parent, title):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.st = fx_settings()
        self.job = None
        self.result_path = None
        self.lay = QVBoxLayout(self)
        self.form = QFormLayout()
        self.lay.addLayout(self.form)

    def _out_row(self, default):
        row = QHBoxLayout()
        self.out_edit = QLineEdit(default)
        self.out_edit.setToolTip("Output file. Written as <name>.part and renamed when complete.")
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_out)
        row.addWidget(self.out_edit, 1)
        row.addWidget(b)
        self.form.addRow("Save as", row)

    def _browse_out(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save video as", self.out_edit.text(),
                                           "Video (*.mp4 *.mkv *.mov);;All files (*)")
        if p:
            self.out_edit.setText(p)

    def _footer(self):
        self.bar = anim.SmoothProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.lay.addWidget(self.bar)
        self.lay.addWidget(self.status)
        self.buttons = QDialogButtonBox()
        self.go = self.buttons.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        self.open_btn = self.buttons.addButton("Open", QDialogButtonBox.ButtonRole.ActionRole)
        self.open_btn.setVisible(False)
        self.open_btn.clicked.connect(lambda: self.result_path and open_path(self.result_path))
        self.folder_btn = self.buttons.addButton("Show folder", QDialogButtonBox.ButtonRole.ActionRole)
        self.folder_btn.setVisible(False)
        self.folder_btn.clicked.connect(lambda: self.result_path and open_path(os.path.dirname(self.result_path)))
        self.close_btn = self.buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self.go.clicked.connect(self.start)
        self.close_btn.clicked.connect(self.reject)
        self.lay.addWidget(self.buttons)

    def _run(self, fn):
        out = self.out_edit.text().strip()
        if not out:
            return
        if os.path.exists(out) and QMessageBox.question(self, self.windowTitle(), f"Replace {out}?") != \
                QMessageBox.StandardButton.Yes:
            return
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        self.go.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setValue(0)
        self.open_btn.setVisible(False)
        self.folder_btn.setVisible(False)
        self.close_btn.setText("Cancel")
        self.status.setText("Starting…")
        self.job = Job(fn, self)
        from AIO_QC.widgets import tasks
        self._task = tasks.start(f"{self.windowTitle()} · {os.path.basename(out)}", cancel=self.job.cancel)
        self.job.progress.connect(self._on_progress)
        self.job.done.connect(self._on_done)
        self.job.failed.connect(self._on_failed)
        self.job.finished.connect(self._finished)
        self.job.start()

    def _on_progress(self, p):
        done, total, text = p
        if getattr(self, "_task", None) is not None:
            self._task.update(done / total if total else None, text)
        self.bar.setValue(int(1000 * done / total) if total else 0)
        if text:
            self.status.setText(text)

    def _on_done(self, r):
        if not r:
            self.status.setText("Cancelled - nothing was written.")
            return
        self.result_path = r["path"]
        self.bar.setValue(1000)
        self.open_btn.setVisible(True)
        self.folder_btn.setVisible(True)
        self.status.setText(self.describe(r))

    def describe(self, r):
        return f"Saved {r['path']}"

    def _on_failed(self, msg):
        if "cancelled" in msg.lower() or "interrupted" in msg.lower():
            self.status.setText("Cancelled - nothing was written.")
            self.bar.setValue(0)
        else:
            self.status.setText(f"Failed: {msg}")

    def _finished(self):
        if getattr(self, "_task", None) is not None:
            self._task.finish()
            self._task = None
        self.go.setEnabled(True)
        self.close_btn.setText("Close")
        self.job = None

    def reject(self):
        if self.job is not None:
            self.job.cancel()
            self.status.setText("Cancelling…")
            return
        super().reject()

    def closeEvent(self, ev):
        if self.job is not None:
            self.job.cancel()
            self.job.wait(5000)
        super().closeEvent(ev)


# ------------------------------------------------------------------ clip
class ClipExportDialog(_ExportBase):
    def __init__(self, parent, video, start, end, duration, ffmpeg, keyframe_of=None):
        super().__init__(parent, "Export clip")
        from AIO_QC.core.testing.annotate import keyframe_before
        self.video, self.ffmpeg = video, ffmpeg
        self.keyframe_of = keyframe_of or (lambda t: keyframe_before(video, t))
        self.form.addRow(QLabel(f"<b>{os.path.basename(video)}</b>"))
        self.start_s = _time_spin(start, duration)
        self.end_s = _time_spin(end if end > start else duration, duration)
        self.len_lbl = QLabel()
        for s in (self.start_s, self.end_s):
            s.valueChanged.connect(self._update)
        self.form.addRow("Start", self.start_s)
        self.form.addRow("End", self.end_s)
        self.form.addRow("", self.len_lbl)
        self.fast = QRadioButton("Fast — no re-encoding, original quality, instant")
        self.fast.setToolTip("Copies the compressed video as it is. The clip can only begin on a keyframe, so it may "
                             "start up to a few seconds before the chosen start.")
        self.exact = QRadioButton("Exact — starts on the exact frame (re-encoded, high quality)")
        self.exact.setToolTip("Decodes and re-encodes the range: frame-accurate start and end. Uses the GPU encoder "
                              "when one works on this machine, else x264.")
        g = QButtonGroup(self)
        g.addButton(self.fast)
        g.addButton(self.exact)
        (self.exact if self.st.value("export/clip_exact", False, type=bool) else self.fast).setChecked(True)
        self.fast.toggled.connect(self._update)
        self.form.addRow("Mode", self.fast)
        self.form.addRow("", self.exact)
        self.audio = QCheckBox("Keep audio")
        self.audio.setChecked(True)
        self.form.addRow("", self.audio)
        base, ext = os.path.splitext(video)
        self._out_row(f"{base}_{start:.1f}-{end:.1f}s{ext if ext.lower() in ('.mp4', '.mkv', '.mov') else '.mp4'}")
        self._footer()
        self._update()
        self.resize(560, self.sizeHint().height())

    def _update(self):
        a, b = self.start_s.value(), self.end_s.value()
        txt = f"Length {human_time(max(0.0, b - a))}"
        if self.fast.isChecked() and self.keyframe_of:
            try:
                k = self.keyframe_of(a)
                if k is not None and a - k > 0.01:
                    txt += f" · fast mode starts at the keyframe {human_time(k)} ({a - k:.2f} s earlier)"
            except Exception:
                pass
        self.len_lbl.setText(txt)
        if hasattr(self, "go"):
            self.go.setEnabled(b > a)

    def start(self):
        from AIO_QC.core.testing.annotate import export_clip
        a, b = self.start_s.value(), self.end_s.value()
        mode = "fast" if self.fast.isChecked() else "exact"
        self.st.setValue("export/clip_exact", mode == "exact")
        out, video, ff, audio = self.out_edit.text().strip(), self.video, self.ffmpeg, self.audio.isChecked()

        def work(job):
            return export_clip(video, out, a, b, ff, mode=mode, audio=audio,
                               progress=lambda d, t, txt="": job.progress.emit((d, t, txt or f"{mode} cut…")),
                               cancel=job.cancelled)
        self._run(work)

    def describe(self, r):
        s = f"Saved {os.path.basename(r['path'])}"
        if r.get("mode") == "fast":
            s += f" · starts at the keyframe {human_time(r['start'])}"
        elif r.get("encoder"):
            s += f" · encoded with {r['encoder']}"
        return s


# ------------------------------------------------------------------ annotated video
class AnnotatedExportDialog(_ExportBase):
    def __init__(self, parent, video, specs, ffmpeg, start=0.0, end=None, duration=0.0, cache=None, conf=0.25,
                 slots=None):
        """specs: [core.lab.runner.ModelSpec] (all offered; the first up to four are ticked)."""
        super().__init__(parent, "Export annotated video")
        from AIO_QC.core.testing.pipeline import cache_for
        self.video, self.specs, self.ffmpeg = video, specs, ffmpeg
        self.slots = list(slots) if slots else list(range(len(specs)))
        self.cache = cache or cache_for()
        self.form.addRow(QLabel(f"<b>{os.path.basename(video)}</b>"))
        self.models = QListWidget()
        self.models.setMaximumHeight(120)
        self.models.setToolTip("Up to four models. Each is drawn in its own colour (the same as in Testing).")
        for i, s in enumerate(specs):
            it = QListWidgetItem(f"●  {s.label}")
            it.setForeground(_qcolor(model_hex(self.slots[i])))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if i < 4 else Qt.CheckState.Unchecked)
            it.setToolTip(s.path)
            self.models.addItem(it)
        self.models.itemChanged.connect(self._update)
        self.form.addRow("Models", self.models)
        self.start_s = _time_spin(start, max(duration, 0.001))
        self.end_s = _time_spin(end if end and end > start else duration, max(duration, 0.001))
        self.form.addRow("Start", self.start_s)
        self.form.addRow("End", self.end_s)
        self.layout_cb = QComboBox()
        self.layout_cb.addItem("All models on one picture", "overlay")
        self.layout_cb.addItem("Side by side (one panel per model)", "side_by_side")
        self.layout_cb.setToolTip("Side by side: 2 models left / right, 3-4 models in a 2×2 grid.")
        self.form.addRow("Layout", self.layout_cb)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.01, 0.99)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(conf)
        self.conf.setToolTip("Detections below this confidence are not drawn.")
        self.form.addRow("Confidence", self.conf)
        self.tracks = QCheckBox("Track IDs (same object keeps its number)")
        self.tracks.setChecked(True)
        self.hud = QCheckBox("Info overlay (time, frame, objects per model)")
        self.hud.setChecked(True)
        self.audio = QCheckBox("Keep audio")
        self.audio.setChecked(True)
        for w in (self.tracks, self.hud, self.audio):
            self.form.addRow("", w)
        self.enc = QComboBox()
        self.enc.addItem("Auto (GPU when it works here)", "auto")
        self.enc.setToolTip("The first encoder that passes a short test encode on this machine is used.")
        self._fill_encoders()
        self.form.addRow("Encoder", self.enc)
        self.maxw = QComboBox()
        for label, v in (("Original size", 0), ("Max 3840 px wide (4K)", 3840), ("Max 1920 px (1080p)", 1920),
                         ("Max 1280 px (720p)", 1280)):
            self.maxw.addItem(label, v)
        self.maxw.setCurrentIndex(1)
        self.form.addRow("Size", self.maxw)
        base = os.path.splitext(video)[0]
        self._out_row(f"{base}_annotated.mp4")
        self._footer()
        self._update()
        self.resize(600, self.sizeHint().height())

    def _fill_encoders(self):
        try:
            from AIO_QC.core.testing.annotate import available_encoders, ENCODER_LABELS
            for e in available_encoders(self.ffmpeg) if self.ffmpeg else []:
                self.enc.addItem(ENCODER_LABELS.get(e, e), e)
        except Exception:
            pass

    def _checked(self):
        return [i for i in range(self.models.count())
                if self.models.item(i).checkState() == Qt.CheckState.Checked]

    def _update(self, *_):
        n = len(self._checked())
        ok = 1 <= n <= 4
        self.go.setEnabled(ok)
        self.status.setText("" if ok else ("Tick at least one model." if n == 0 else "At most four models."))

    def start(self):
        from AIO_QC.core.testing.annotate import export_annotated
        idx = self._checked()
        specs = [self.specs[i] for i in idx]
        slots = [self.slots[i] % 8 for i in idx]
        a, b = self.start_s.value(), self.end_s.value()
        kw = dict(layout=self.layout_cb.currentData(), conf=self.conf.value(), slots=slots,
                  tracks=self.tracks.isChecked(), hud=self.hud.isChecked(), encoder=self.enc.currentData(),
                  audio=self.audio.isChecked(), max_width=self.maxw.currentData() or 100000)
        out, video, ff, cache = self.out_edit.text().strip(), self.video, self.ffmpeg, self.cache

        def work(job):
            return export_annotated(video, out, specs, cache, ff, start=a, end=b,
                                    progress=lambda d, t, txt="": job.progress.emit((d, t, txt)),
                                    cancel=job.cancelled, log=job.log.emit, **kw)
        self._run(work)

    def describe(self, r):
        try:
            mb = os.path.getsize(r["path"]) / 1e6
        except OSError:
            mb = 0.0
        size = r.get("size")
        res = f" · {size[0]}×{size[1]}" if isinstance(size, (tuple, list)) and len(size) == 2 else ""
        return (f"Saved {os.path.basename(r['path'])} · {r.get('frames', 0):,} frames in {r.get('seconds', 0):.1f} s"
                f"{res} · {r.get('encoder') or ''} · {mb:.1f} MB")


def _qcolor(h):
    from PyQt6.QtGui import QColor
    return QColor(h)
