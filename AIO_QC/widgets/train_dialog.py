# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Train Model (Image QC ▸ Models, Ctrl+Alt+B) — train an Ultralytics YOLO model on a YOLO dataset (dataset.yaml),
in the user's own Python environment, and follow it live. core.training does the work; this is the window.

Left — a new run, top to bottom: the dataset (checked as soon as it is chosen: counts, label format, everything
Ultralytics would trip over, with the fix), the environment (checked: torch, ultralytics, GPUs), the model (presets;
YOLO26 / YOLO11 / YOLOv8 in five sizes for the dataset's task, your earlier runs' best.pt, any .pt), the training
settings (image size suggested from the object sizes, batch Auto on a GPU, patience), more settings, where runs go.
Right — runs (this project folder, newest first, also runs made outside AIO QC) and the selected one, live: epoch and
batch progress with time left, scores, loss and mAP charts, per-class results, the log; Stop (keep the results),
Pause (resumable), Stop now, Resume; Add to Models, Validate, Open folder, Fine-tune from it.
Runs start detached: they keep going when this window — or AIO QC — closes, and are picked up again when it reopens.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSizePolicy, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from AIO_QC import utils
from AIO_QC.core import training as T
from .inference_service import FuncWorker
from AIO_QC.core.diag import ignored as _ignored

STATE_WORD = {"queued": "starting", "starting": "starting", "training": "training", "validating": "validating",
              "finished": "finished", "stopped": "stopped", "paused": "paused", "failed": "failed",
              "interrupted": "interrupted", "not responding": "not responding", "unknown": "—",
              "review": "waiting for your review"}
STATE_COLOR = {"training": "accent", "starting": "accent", "queued": "accent", "validating": "accent",
               "finished": "ok", "stopped": "muted", "paused": "warn", "failed": "error", "interrupted": "warn",
               "not responding": "warn", "review": "warn"}
SERIES_COLORS = ["#1E88E5", "#E53935", "#43A047", "#FB8C00", "#8E24AA", "#00897B"]


def _dur(sec) -> str:
    sec = int(max(0, sec or 0))
    h, m = divmod(sec // 60, 60)
    return f"{h} h {m:02d} min" if h else (f"{m} min {sec % 60:02d} s" if m else f"{sec} s")


# =====================================================================================================================
def nice_axis(lo: float, hi: float, ticks: int = 4) -> tuple[float, float, float]:
    """(lo, hi, step) widened to round numbers: a step of 1, 2, 2.5 or 5 × 10^k giving about `ticks` intervals."""
    span = max(hi - lo, 1e-12)
    raw = span / ticks
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw * 0.999)
    a, b = math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step
    if b - a < step:
        b = a + step
    return a, b, step


def tick_text(v: float, step: float) -> str:
    """A tick label with just the decimals the step needs (0.05 → 2, 2.5 → 1, 20 → 0)."""
    dec = 0
    while dec < 6 and abs(round(step, dec) - step) > step * 1e-6:
        dec += 1
    return f"{0.0 if abs(v) < step * 1e-9 else v:.{dec}f}"


class LineChart(QWidget):
    """Lines over epochs: series [(name, [(x, y)…], color, dashed)]; hover shows the values at the nearest epoch."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.series: list = []
        self.mark_x = None                      # best epoch
        self.hover_x = None
        self._ystep = 1.0
        self.setMouseTracking(True)
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_series(self, series, mark_x=None):
        self.series, self.mark_x = [s for s in series if s[1]], mark_x
        self.update()

    def _bounds(self):
        xs = [x for _n, pts, *_ in self.series for x, _y in pts]
        ys = [y for _n, pts, *_ in self.series for _x, y in pts if y is not None and math.isfinite(y)]
        if not xs or not ys:
            return None
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        nonneg = y0 >= 0
        if y0 == y1:
            y0, y1 = y0 - 0.5, y1 + 0.5
        pad = (y1 - y0) * 0.08
        y0, y1 = y0 - pad, y1 + pad
        if nonneg:
            y0 = max(0.0, y0)
        y0, y1, self._ystep = nice_axis(y0, y1)
        return x0, max(x1, x0 + 1), y0, y1

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        muted, ink, line = QColor(utils.hexc("muted")), QColor(utils.hexc("text")), QColor(utils.hexc("line"))
        f = p.font()
        b = self._bounds()
        ticks = []
        if b is not None:                       # round values, each axis label fully inside the margin
            n = int(round((b[3] - b[2]) / self._ystep))
            ticks = [(b[2] + self._ystep * i, tick_text(b[2] + self._ystep * i, self._ystep)) for i in range(n + 1)]
        lw = max([p.fontMetrics().horizontalAdvance(s) for _v, s in ticks] + [24]) + 8
        r = QRectF(self.rect()).adjusted(lw + 8, 46, -12, -24)
        p.setPen(ink)
        bf = QFont(f)
        bf.setBold(True)
        p.setFont(bf)
        p.drawText(QRectF(8, 2, self.width() - 16, 20), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   self.title)
        p.setFont(f)
        if b is None:
            p.setPen(muted)
            p.drawText(r, int(Qt.AlignmentFlag.AlignCenter), "Appears after the first epoch")
            return
        x0, x1, y0, y1 = b

        def X(x):
            return r.left() + (x - x0) / (x1 - x0) * r.width()

        def Y(y):
            return r.bottom() - (y - y0) / (y1 - y0) * r.height()
        p.setPen(QPen(line, 1))
        for v, label in ticks:
            yy = Y(v)
            p.drawLine(QPointF(r.left(), yy), QPointF(r.right(), yy))
            p.setPen(muted)
            p.drawText(QRectF(0, yy - 8, lw, 16), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                       label)
            p.setPen(QPen(line, 1))
        step = max(1, int(math.ceil((x1 - x0) / 8)))
        p.setPen(muted)
        x = int(math.ceil(x0))
        while x <= x1:
            p.drawText(QRectF(X(x) - 20, r.bottom() + 3, 40, 16), int(Qt.AlignmentFlag.AlignHCenter), str(x))
            x += step
        if self.mark_x is not None and x0 <= self.mark_x <= x1:
            c = QColor(utils.hexc("ok"))
            p.setPen(QPen(c, 1.2, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(X(self.mark_x), r.top()), QPointF(X(self.mark_x), r.bottom()))
        lx = 8.0
        for name, pts, color, *rest in self.series:
            dashed = bool(rest and rest[0])
            c = QColor(color)
            path = QPainterPath()
            first = True
            for xx, yy in pts:
                if yy is None or not math.isfinite(yy):
                    continue
                pt = QPointF(X(xx), Y(yy))
                path.moveTo(pt) if first else path.lineTo(pt)
                first = False
            p.setPen(QPen(c, 2.0, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            if len(pts) == 1:
                p.setBrush(c)
                p.drawEllipse(QPointF(X(pts[0][0]), Y(pts[0][1])), 3, 3)
            # legend
            w = p.fontMetrics().horizontalAdvance(name) + 22
            if lx + w < self.width() - 4:
                p.setPen(QPen(c, 2.5, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine))
                p.drawLine(QPointF(lx, 32), QPointF(lx + 14, 32))
                p.setPen(muted)
                p.drawText(QRectF(lx + 18, 24, w, 16), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                           name)
                lx += w + 6
        if self.hover_x is not None:
            hx = round(x0 + (self.hover_x - r.left()) / r.width() * (x1 - x0))
            hx = min(max(hx, x0), x1)
            vals = []
            for name, pts, color, *_ in self.series:
                near = min(pts, key=lambda q: abs(q[0] - hx))
                if near[1] is not None:
                    vals.append((name, near[1], color))
            if vals:
                p.setPen(QPen(muted, 1))
                p.drawLine(QPointF(X(hx), r.top()), QPointF(X(hx), r.bottom()))
                txt = [f"epoch {int(hx)}"] + [f"{n}: {v:.4f}" for n, v, _c in vals]
                fm = p.fontMetrics()
                w = max(fm.horizontalAdvance(t) for t in txt) + 16
                h = len(txt) * (fm.height() + 1) + 8
                bx = X(hx) + 10 if X(hx) + 10 + w < self.width() else X(hx) - 10 - w
                box = QRectF(bx, r.top() + 4, w, h)
                p.setPen(QPen(line, 1))
                p.setBrush(QColor(utils.hexc("surface")))
                p.drawRoundedRect(box, 6, 6)
                p.setPen(ink)
                for i, t in enumerate(txt):
                    p.drawText(QRectF(bx + 8, box.top() + 4 + i * (fm.height() + 1), w, fm.height()),
                               int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), t)
        p.end()

    def mouseMoveEvent(self, e):
        self.hover_x = e.position().x()
        self.update()

    def leaveEvent(self, _e):
        self.hover_x = None
        self.update()


class Tile(QWidget):
    """A score: big number, small caption."""

    def __init__(self, caption: str):
        super().__init__()
        self.value = QLabel("—")
        self.value.setObjectName("tileValue")
        f = self.value.font()
        f.setPointSizeF(f.pointSizeF() * 1.6)
        f.setBold(True)
        self.value.setFont(f)
        self.cap = QLabel(caption)
        utils.style(self.cap, "QLabel{color:@muted;}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(0)
        lay.addWidget(self.value)
        lay.addWidget(self.cap)
        self.setObjectName("tile")
        utils.style(self, "#tile{background:@surface; border:1px solid @line; border-radius:8px;}")

    def set(self, v, sub: str | None = None):
        self.value.setText("—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        if sub is not None:
            self.cap.setText(sub)


# =====================================================================================================================
class TrainDialog(QDialog):
    TITLE = "Train Model"
    SUPPORTED = None

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(utils.newIcon("tool-train"))
        from AIO_QC.utils.qt_utils import fit_size, scrolled
        fit_size(self, "large")
        self.settings = utils.app_settings()
        self.check: T.DataCheck | None = None
        self.env: dict | None = None
        self.cancel = threading.Event()
        self._check_cancel = None            # the dataset check now running (a newer one replaces it)
        self._check_no = 0
        self.workers: list = []
        self.current: Path | None = None          # run shown on the right
        self._sig = None                          # (mtime of status, epochs) last drawn
        self._ep_cache: tuple = (None, [])
        self._imgsz_touched = False
        self._programmatic = False

        # ------------------------------------------------------------------ dataset
        ds_box = QGroupBox("1 · Dataset")
        dl = QVBoxLayout(ds_box)
        r = QHBoxLayout()
        self.yaml = QLineEdit(self.settings.value("train/yaml", "") or "")
        self.yaml.setPlaceholderText("dataset.yaml / data.yaml of a YOLO dataset")
        self.yaml.editingFinished.connect(self.check_dataset)
        r.addWidget(self.yaml, 1)
        r.addWidget(utils.newButton("Browse…", icon="open", slot=self._browse_yaml))
        r.addWidget(utils.newButton("", icon="reset-default", slot=self.check_dataset, tip="Check again"))
        dl.addLayout(r)
        self.ds_info = QLabel("Choose the dataset file. A LabelMe dataset? Convert it first: Review ▸ Convert & "
                              "Split, format YOLO (Ctrl+Alt+Y) — it ends with a Train a model… button.")
        self.ds_info.setWordWrap(True)
        self.ds_info.setTextFormat(Qt.TextFormat.RichText)
        dl.addWidget(self.ds_info)

        # ------------------------------------------------------------------ environment
        env_box = QGroupBox("2 · Python environment")
        el = QVBoxLayout(env_box)
        r = QHBoxLayout()
        self.env_box = QComboBox()
        self.env_box.setEditable(True)
        self.env_box.setMinimumContentsLength(16)
        self.env_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.env_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.env_box.setToolTip("The Python with PyTorch + Ultralytics that trains. Pick one, or type / browse a "
                                "python executable or an environment folder.")
        r.addWidget(self.env_box, 1)
        r.addWidget(utils.newButton("Browse…", icon="open", slot=self._browse_env))
        r.addWidget(utils.newButton("Check", icon="run", slot=self.check_env, tip="Import torch / ultralytics there "
                                    "and list its GPUs"))
        el.addLayout(r)
        self.env_info = QLabel("")
        self.env_info.setWordWrap(True)
        self.env_info.setTextFormat(Qt.TextFormat.RichText)
        el.addWidget(self.env_info)
        self.install_btn = utils.newButton("Install missing", icon="import", slot=self.install_missing,
                                           tip="python -m pip install … in this environment (only what is missing)")
        self.install_btn.setVisible(False)
        ir = QHBoxLayout()
        ir.addStretch(1)
        ir.addWidget(self.install_btn)
        el.addLayout(ir)
        self._fill_envs()

        # ------------------------------------------------------------------ model
        m_box = QGroupBox("3 · Model")
        ml = QVBoxLayout(m_box)
        pr = QHBoxLayout()
        self.presets = QButtonGroup(self)
        for key, p in T.PRESETS.items():
            b = QPushButton(p["title"])
            b.setCheckable(True)
            b.setToolTip(p["tip"])
            b.setProperty("preset", key)
            self.presets.addButton(b)
            pr.addWidget(b)
            b.clicked.connect(lambda _c, k=key: self.apply_preset(k))
        pr.addStretch(1)
        ml.addLayout(pr)
        f = QFormLayout()
        self.model_box = QComboBox()
        self.model_box.setMinimumContentsLength(16)
        self.model_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model_box.currentIndexChanged.connect(self._model_changed)
        mr = QHBoxLayout()
        mr.addWidget(self.model_box, 1)
        mr.addWidget(utils.newButton("Browse…", icon="open", slot=self._browse_model,
                                     tip="Any Ultralytics .pt (e.g. a model trained before: fine-tune it), a model "
                                         ".yaml (train from scratch), or a classifier trained here (its best.pt)"))
        f.addRow("Start from", mr)
        self.goal = QComboBox()
        self.goal.addItem("Train — from the weights above", "train")
        self.goal.addItem("My model + more images (the same classes)", "images")
        self.goal.addItem("My model + new classes (the old ones kept)", "classes")
        self.goal.setToolTip("Extend a model you trained: its classes keep their ids, the new images (and classes) "
                             "are added — without forgetting what it knew (the old data mixed in, pre-labels, two "
                             "stages, a before / after check)")
        self.goal.currentIndexChanged.connect(lambda _i: self._goal_changed())
        self.extend_btn = utils.newButton("Plan…", icon="tool-train", slot=self.plan_extend,
                                          tip="The class plan, the old data, pre-labels and the stages — before "
                                              "anything runs")
        self.extend_btn.setVisible(False)
        gr = QHBoxLayout()
        gr.addWidget(self.goal, 1)
        gr.addWidget(self.extend_btn)
        f.addRow("Goal", gr)
        self.extend_plan = None
        self.pretrained = QCheckBox("Pretrained weights (recommended)")
        self.pretrained.setChecked(True)
        self.pretrained.setToolTip("Unticked: the architecture is trained from random weights — needs far more data "
                                   "and epochs")
        f.addRow("", self.pretrained)
        self.model_note = QLabel("")
        self.model_note.setWordWrap(True)
        self.model_note.setTextFormat(Qt.TextFormat.RichText)
        f.addRow("", self.model_note)
        ml.addLayout(f)

        # ------------------------------------------------------------------ training
        t_box = QGroupBox("4 · Training")
        g = QFormLayout(t_box)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5000)
        self.epochs.setValue(int(self.settings.value("train/epochs", 100)))
        self.patience = QSpinBox()
        self.patience.setRange(0, 1000)
        self.patience.setValue(int(self.settings.value("train/patience", 30)))
        self.patience.setToolTip("Stop by itself when the val score hasn't improved for this many epochs (0: never)")
        self.imgsz = QSpinBox()
        self.imgsz.setRange(64, 4096)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(int(self.settings.value("train/imgsz", 640)))
        self.imgsz.setToolTip("Images are resized so the longer side is this (a multiple of 32)")
        self.imgsz.valueChanged.connect(lambda _v: setattr(self, "_imgsz_touched", True)
                                        if not self._programmatic else None)
        self.imgsz_note = QLabel("")
        self.imgsz_note.setWordWrap(True)
        utils.style(self.imgsz_note, "QLabel{color:@muted;}")
        self.batch = QComboBox()
        self.batch.setEditable(True)
        for v in ("Auto (fits the GPU memory)", "8", "16", "32", "64"):
            self.batch.addItem(v)
        self.batch.setCurrentText(self.settings.value("train/batch", "Auto (fits the GPU memory)"))
        self.device = QComboBox()
        self.device.addItem("CPU", "cpu")
        self.workers_n = QSpinBox()
        self.workers_n.setRange(0, 64)
        self.workers_n.setValue(int(self.settings.value("train/workers", min(8, max(1, (os.cpu_count() or 2) - 1)))))
        self.workers_n.setToolTip("Processes preparing images while the model trains. Lower it if a run fails with a "
                                  "data-loader / memory error.")
        self.patience.setSuffix(" epochs")
        for lab, w in (("Epochs", self.epochs), ("Stop when no better for", self.patience), ("Image size", self.imgsz),
                       ("", self.imgsz_note), ("Batch", self.batch), ("Device", self.device),
                       ("Workers", self.workers_n)):
            g.addRow(lab, w)

        # ------------------------------------------------------------------ more settings
        self.more = QGroupBox("")
        mf = QFormLayout(self.more)
        self.cache = QComboBox()
        self.cache.addItem("Off — read images from disk each epoch", False)
        self.cache.addItem("RAM — fastest, needs memory", "ram")
        self.cache.addItem("Disk — .npy files next to the images", "disk")
        self.cache_note = QLabel("")
        self.cache_note.setWordWrap(True)
        utils.style(self.cache_note, "QLabel{color:@muted;}")
        self.optimizer = QComboBox()
        for o in ("auto", "SGD", "AdamW", "Adam", "RMSProp", "MuSGD"):
            self.optimizer.addItem(o)
        self.lr0 = QDoubleSpinBox()
        self.lr0.setDecimals(5)
        self.lr0.setRange(0.0, 1.0)
        self.lr0.setSingleStep(0.001)
        self.lr0.setSpecialValueText("auto")
        self.lr0.setValue(0.0)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.close_mosaic = QSpinBox()
        self.close_mosaic.setRange(0, 1000)
        self.close_mosaic.setValue(10)
        self.close_mosaic.setSuffix(" last epochs without mosaic")
        self.time_h = QDoubleSpinBox()
        self.time_h.setRange(0.0, 1000.0)
        self.time_h.setDecimals(1)
        self.time_h.setSpecialValueText("no limit")
        self.time_h.setSuffix(" h")
        self.time_h.setToolTip("Train for this long at most (overrides epochs)")
        self.fraction = QDoubleSpinBox()
        self.fraction.setRange(0.01, 1.0)
        self.fraction.setSingleStep(0.1)
        self.fraction.setValue(1.0)
        self.fraction.setToolTip("Use only this share of the train images (a quick experiment)")
        self.amp = QCheckBox("Mixed precision (AMP, faster on GPUs)")
        self.amp.setChecked(True)
        self.onnx = QCheckBox("Also save best.onnx (to use the classifier outside AIO QC)")
        self.onnx.setChecked(self.settings.value("train/onnx", "true") in (True, "true"))
        self.onnx.setToolTip("Classifiers of other architectures: best.pt works everywhere in AIO QC and with "
                             "Ultralytics; best.onnx runs anywhere (onnxruntime, OpenCV, …) — the image "
                             "normalisation is inside it")
        self.single_cls = QCheckBox("One class: train every object as one class")
        self.extra = QLineEdit(self.settings.value("train/extra", "") or "")
        self.extra.setPlaceholderText("any Ultralytics setting, e.g.  mosaic=0.5  fliplr=0  degrees=10  cos_lr=true")
        self.extra.textChanged.connect(lambda _t: self._hyp_changed())    # a typo disables Start at once
        self.extra_note = QLabel("")
        utils.style(self.extra_note, "QLabel{color:@muted;}")
        mf.addRow("Cache images", self.cache)
        mf.addRow("", self.cache_note)
        mf.addRow("Seed", self.seed)
        mf.addRow("Time limit", self.time_h)
        mf.addRow("Train images used", self.fraction)
        mf.addRow("", self.amp)
        mf.addRow("", self.onnx)
        mf.addRow("", self.single_cls)
        mf.addRow("Extra settings", self.extra)
        mf.addRow("", self.extra_note)
        from .drawer import OptionsDrawer
        self.more_drawer = OptionsDrawer(self.more, "More settings", "Fewer settings",
                                         tip="Cache, seed, time limit, share of images, any Ultralytics setting")
        # augmentation & learning-rate schedule (optimizer, start rate and close mosaic live there)
        from .train_hyp import HypPanel
        self.hyp = HypPanel(self.settings, self.optimizer, self.lr0, self.close_mosaic)
        self.hyp.changed.connect(self._hyp_changed)
        self.hyp.edit_recipe.connect(self._edit_recipe)
        self.hyp_drawer = OptionsDrawer(self.hyp, "Augmentation and learning rate", "Hide augmentation and learning rate",
                                        tip="Flips, mosaic, mixup, zoom, colour · optimizer, start / final learning "
                                            "rate, decay curve, warm-up, frozen layers — with presets for training "
                                            "and fine-tuning")

        # ------------------------------------------------------------------ where
        w_box = QGroupBox("5 · Save runs to")
        wl = QFormLayout(w_box)
        pr = QHBoxLayout()
        self.project = QLineEdit(self.settings.value("train/project", "") or "")
        self.project.setPlaceholderText("<dataset folder>/runs")
        self.project.editingFinished.connect(self.refresh_runs)
        pr.addWidget(self.project, 1)
        pr.addWidget(utils.newButton("Browse…", icon="open", slot=self._browse_project))
        wl.addRow("Folder", pr)
        self.run_name = QLineEdit("")
        self.run_name.setPlaceholderText("automatic: <model>_<date>_<time>")
        wl.addRow("Run name", self.run_name)
        self.disk_note = QLabel("")
        utils.style(self.disk_note, "QLabel{color:@muted;}")
        wl.addRow("", self.disk_note)

        self.start_btn = utils.newButton("Start training", icon="tool-train", slot=self.start)
        self.start_btn.setProperty("variant", "primary")
        self.start_btn.setDefault(True)
        self.start_note = QLabel("")
        self.start_note.setWordWrap(True)
        self.start_note.setTextFormat(Qt.TextFormat.RichText)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 6, 0)
        for wdg in (ds_box, env_box, m_box, t_box, self.hyp_drawer, self.more_drawer, w_box):
            ll.addWidget(wdg)
        sr = QHBoxLayout()
        sr.addWidget(self.start_note, 1)
        sr.addWidget(self.start_btn)
        ll.addLayout(sr)
        ll.addStretch(1)

        # ------------------------------------------------------------------ right: runs
        self.runs = QListWidget()
        self.runs.setMaximumHeight(120)
        self.runs.currentItemChanged.connect(lambda cur, _prev: self.show_run(cur.data(Qt.ItemDataRole.UserRole)
                                                                              if cur else None))
        self.run_title = QLabel("No run selected")
        tf = self.run_title.font()
        tf.setBold(True)
        tf.setPointSizeF(tf.pointSizeF() * 1.15)
        self.run_title.setFont(tf)
        self.run_state = QLabel("")
        self.run_state.setTextFormat(Qt.TextFormat.RichText)
        self.run_state.setWordWrap(True)
        self.ep_bar = QProgressBar()
        self.ep_bar.setFormat("epoch %v / %m")
        self.b_bar = QProgressBar()
        self.b_bar.setFormat("batch %v / %m")
        self.b_bar.setMaximumHeight(10)
        self.b_bar.setTextVisible(False)
        tiles = QHBoxLayout()
        self.t_map = Tile("mAP50-95 (best)")
        self.t_map50 = Tile("mAP50")
        self.t_p = Tile("precision")
        self.t_r = Tile("recall")
        self.t_best = Tile("best epoch")
        for t in (self.t_map, self.t_map50, self.t_p, self.t_r, self.t_best):
            tiles.addWidget(t)
        self.loss_chart = LineChart("Loss (lower is better) — solid: train, dashed: val")
        self.map_chart = LineChart("Scores on val (higher is better)")
        charts = QWidget()
        cl = QHBoxLayout(charts)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.loss_chart)
        cl.addWidget(self.map_chart)
        self.cls_table = QTableWidget(0, 6)
        self.cls_table.setHorizontalHeaderLabels(["Class", "Objects", "Precision", "Recall", "mAP50", "mAP50-95"])
        self.cls_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.cls_table.verticalHeader().setVisible(False)
        self.cls_table.setSortingEnabled(True)
        self.cls_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        lf = QFont("Monospace")
        lf.setStyleHint(QFont.StyleHint.TypeWriter)
        self.log.setFont(lf)
        self.args_view = QPlainTextEdit()
        self.args_view.setReadOnly(True)
        self.tabs = QTabWidget()
        self.tabs.addTab(charts, "Charts")
        self.tabs.addTab(self.cls_table, "Per class")
        self.tabs.addTab(self.log, "Log")
        self.tabs.addTab(self.args_view, "Settings")
        self.tabs.currentChanged.connect(lambda _i: self.poll(force=True))

        ab = QHBoxLayout()
        self.b_stop = utils.newButton("Stop", icon="stop", slot=lambda: self._control("stop"),
                                      tip="Finish now and keep the results: this epoch ends early, best.pt is checked "
                                          "and the plots are made")
        self.b_pause = utils.newButton("Pause", icon="review-pending", slot=lambda: self._control("pause"),
                                       tip="Stop after the next saved epoch — Resume continues exactly there")
        self.b_kill = utils.newButton("Stop now", icon="close-x", slot=lambda: self._control("kill"),
                                      tip="End at once (the last finished epoch can be resumed)")
        self.b_resume = utils.newButton("Resume", icon="run", slot=self.resume)
        self.b_folder = utils.newButton("Open folder", icon="open", slot=self._open_run_folder)
        self.b_add = utils.newButton("Add to Models", icon="tool-model-setup", slot=self.add_to_models,
                                     tip="Register best.pt for Model Check, Testing and Auto-Annotate (with its "
                                         "classes and image size)")
        self.b_val = utils.newButton("Validate…", icon="tool-testing", slot=self.validate,
                                     tip="Run best.pt on val or test and get per-class scores and the plots")
        self.b_tune = utils.newButton("Fine-tune from it", icon="tool-train", slot=self.fine_tune,
                                      tip="Start a new run from this run's best.pt (e.g. after adding data)")
        self.b_improve = utils.newButton("Improve…", icon="tool-improve", slot=self.improve,
                                         tip="Improve Model: check this run's best.pt on the open dataset — where it is "
                                             "weak and why — then label more and fine-tune, round by round")
        self.b_review = utils.newButton("Open the training copy", icon="open", slot=self.open_training_copy,
                                        tip="The pre-labels are in the training copy (a YOLO dataset): look at them, "
                                            "fix or delete what is wrong — your dataset is not touched")
        self.b_continue = utils.newButton("Continue training", icon="run", slot=self.continue_extend,
                                          tip="The pre-labels are reviewed: enlarge the head and train")
        self.b_fix = utils.newButton("Install and retry", icon="import", slot=self.fix_and_retry,
                                     tip="Install the missing package into the run's environment, then run it again")
        for b in (self.b_fix, self.b_review, self.b_continue, self.b_stop, self.b_pause, self.b_kill, self.b_resume,
                  self.b_folder, self.b_improve, self.b_add, self.b_val, self.b_tune):
            ab.addWidget(b)
        ab.addStretch(1)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        rh = QHBoxLayout()
        cap = QLabel("Runs")
        cap.setFont(tf)
        rh.addWidget(cap)
        rh.addStretch(1)
        rh.addWidget(utils.newButton("", icon="reset-default", slot=self.refresh_runs, tip="Refresh"))
        rl.addLayout(rh)
        rl.addWidget(self.runs)
        rl.addWidget(self.run_title)
        rl.addWidget(self.run_state)
        rl.addWidget(self.ep_bar)
        rl.addWidget(self.b_bar)
        rl.addLayout(tiles)
        rl.addLayout(ab)
        rl.addWidget(self.tabs, 1)

        split = QSplitter(Qt.Orientation.Horizontal)
        left_sa = scrolled(left)
        # never narrower than its rows need (a hidden horizontal bar would cut the labels off on the left)
        left_sa.setMinimumWidth(left.minimumSizeHint().width() + left_sa.verticalScrollBar().sizeHint().width() + 2)
        split.addWidget(left_sa)
        split.addWidget(right)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 6)
        split.setChildrenCollapsible(False)
        split.setSizes([560, 900])
        lay = QVBoxLayout(self)
        lay.addWidget(split, 1)
        foot = QHBoxLayout()
        self.foot_note = QLabel("Runs keep going when this window or AIO QC closes; they are picked up again here.")
        utils.style(self.foot_note, "QLabel{color:@muted;}")
        foot.addWidget(self.foot_note, 1)
        foot.addWidget(utils.newButton("Help", icon="question", slot=lambda: self.app.show_guide()
                                       if hasattr(self.app, "show_guide") else None))
        foot.addWidget(utils.newButton("Close", slot=self.close))
        lay.addLayout(foot)

        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        self.runs_timer = QTimer(self)
        self.runs_timer.setInterval(6000)
        self.runs_timer.timeout.connect(self.refresh_runs)
        self.runs_timer.start()
        for w in (self.epochs, self.imgsz, self.patience, self.workers_n):
            w.valueChanged.connect(lambda _v: self._update_start())
        self.batch.currentTextChanged.connect(lambda _t: self._update_start())
        self.cache.currentIndexChanged.connect(lambda _i: self._cache_hint())
        self._fill_models()
        self._show_run_buttons({})
        if self.yaml.text().strip():
            self.check_dataset()
        self.refresh_runs()
        self._update_start()
        if self.env_box.currentText().strip():
            QTimer.singleShot(200, lambda: self.check_env(quiet=True))

    # ================================================================== dataset
    def set_dataset(self, yaml_path):
        self.yaml.setText(str(yaml_path))
        self.check_dataset()

    def _browse_yaml(self):
        start = self.yaml.text().strip() or str(getattr(getattr(self.app, "session", None), "root", "") or "")
        p, _ = QFileDialog.getOpenFileName(self, "Dataset file (Cancel: a classification folder)", start,
                                           "YOLO dataset (*.yaml *.yml);;All files (*)")
        if not p:                                          # a classifier: the folder with train/ val/ class folders
            p = QFileDialog.getExistingDirectory(self, "Classification dataset (train/<class>/, val/<class>/)",
                                                 start)
        if p:
            self.set_dataset(p)

    def check_dataset(self):
        path = self.yaml.text().strip()
        if not path:
            return
        p = Path(path).expanduser()
        if p.is_dir():                                     # a folder: its dataset.yaml / data.yaml
            for n in ("dataset.yaml", "data.yaml", "data.yml"):
                if (p / n).is_file():
                    p = p / n
                    self.yaml.setText(str(p))
                    break
        if not p.is_file() and not p.is_dir():
            self.check = None
            self.ds_info.setText(f"<span style='color:{utils.hexc('error')}'>Not found: {p}</span>")
            self._update_start()
            return
        self.settings.setValue("train/yaml", str(p))
        if not self.project.text().strip():
            self.project.setPlaceholderText(str(p.parent / "runs"))
        self.ds_info.setText("Checking the dataset …")
        # Only the latest check counts: an older one (the dataset remembered from last time, a file chosen a moment
        # ago) can finish later — its result must not replace this one's. It is told to stop, and ignored.
        if self._check_cancel is not None:
            self._check_cancel.set()
        cancel = self._check_cancel = threading.Event()
        self._check_no += 1
        no = self._check_no
        self.check = None
        self._update_start()

        def work():
            return T.check_dataset(p, cancel=cancel)

        def done(chk):
            if no == self._check_no:
                self._checked(chk)

        def failed(m):
            if no == self._check_no:
                self.ds_info.setText(f"⚠ {m}")
        self._run(work, done, failed)
        self.refresh_runs()

    def _checked(self, chk: T.DataCheck):
        self.check = chk
        self.hyp.set_task(self._hyp_task())
        self.hyp.set_names(chk.names)                      # plates / characters: no mirroring
        L = []
        if chk.splits:
            parts = []
            for sp in ("train", "val", "test"):
                if sp in chk.splits:
                    i = chk.splits[sp]
                    parts.append(f"<b>{sp}</b> {i['images']:,}" + (f" ({i['empty']:,} background)" if i["empty"] else ""))
            task = {"seg": "segmentation (polygons)", "cls": "classification (class folders)"}.get(
                chk.task, "detection (boxes)")
            L.append(" · ".join(parts) + f" images · <b>{len(chk.names)}</b> classes · {task}")
        icon = {"error": ("✕", "error"), "warn": ("!", "warn"), "info": ("i", "muted")}
        for lv, txt in chk.issues:
            sym, col = icon[lv]
            L.append(f"<span style='color:{utils.hexc(col)}'><b>{sym}</b> {txt}</span>")
        if chk.ok and not any(lv == "warn" for lv, _t in chk.issues):
            L.append(f"<span style='color:{utils.hexc('ok')}'>✓ Ready to train.</span>")
        self.ds_info.setText("<br>".join(L))
        sz, why = T.suggest_imgsz(chk)
        self.imgsz_note.setText(f"Suggested image size {why}.")
        if not self._imgsz_touched:
            self._programmatic = True
            self.imgsz.setValue(sz)
            self._programmatic = False
        self._fill_models()
        self._cache_hint()
        self._update_start()

    # ================================================================== environment
    def _fill_envs(self):
        from AIO_QC.core.env_manager import discover_envs
        want = self.settings.value("train/python", "") or getattr(getattr(self.app, "inference", None), "python", "")
        self.env_box.clear()
        seen = set()
        if want:
            self.env_box.addItem(f"{want}", want)
            seen.add(os.path.normcase(want))
        try:
            envs = discover_envs()
        except Exception as _exc:
            _ignored(_exc)
            envs = []
        for name, py in envs:
            if os.path.normcase(py) not in seen:
                self.env_box.addItem(f"{name} — {py}", py)
                seen.add(os.path.normcase(py))
        self.env_box.setCurrentIndex(0)

    def _python(self) -> str:
        data = self.env_box.currentData()
        txt = self.env_box.currentText().strip()
        if data and (txt == data or txt.endswith(str(data))):
            return str(data)
        from AIO_QC.core.env_manager import python_in_env
        py = python_in_env(txt.split("—")[-1].strip()) if txt else None
        return str(py) if py else txt

    def _browse_env(self):
        p, _ = QFileDialog.getOpenFileName(self, "Python executable (python / python.exe)", "",
                                           "Python (python* python*.exe);;All files (*)")
        if p:
            self.env_box.insertItem(0, p, p)
            self.env_box.setCurrentIndex(0)
            self.check_env()

    def check_env(self, quiet=False):
        py = self._python()
        if not py or not Path(py).is_file():
            if not quiet:
                self.env_info.setText(f"<span style='color:{utils.hexc('error')}'>Choose a Python first.</span>")
            return
        self.env_info.setText("Checking (importing torch can take a while) …")

        def work():
            return T.probe(py)
        self._run(work, lambda info: self._env_checked(py, info),
                  lambda m: self.env_info.setText(f"<span style='color:{utils.hexc('error')}'>Could not check: "
                                                  f"{m[-300:]}</span>"))

    def _env_checked(self, py, info):
        if py != self._python():                           # another Python was chosen meanwhile: that check counts
            return
        self.env = info
        self.env["_python"] = py
        self.hyp.set_env(info)                             # its own defaults; settings it lacks are switched off
        self.settings.setValue("train/python", py)
        miss = [n for n in ("torch", "ultralytics") if not info.get(n)]
        deps = list(info.get("missing") or [])
        txt = T.env_summary(info)
        self._to_install = (["ultralytics"] if "ultralytics" in miss else []) + deps
        self.install_btn.setVisible(bool(self._to_install) and "torch" not in miss)
        self.install_btn.setText(f"Install {', '.join(self._to_install)}" if self._to_install else "Install missing")
        if miss:
            self.env_info.setText(f"<span style='color:{utils.hexc('error')}'>Missing: {', '.join(miss)}</span> — "
                                  f"{txt}." + (" Install PyTorch first (pytorch.org has the command for your GPU), or "
                                               "let Models &amp; Environment make an environment." if "torch" in miss
                                               else ""))
        elif deps:
            self.env_info.setText(f"<span style='color:{utils.hexc('error')}'>Ultralytics is installed without some of "
                                  f"its own packages: {', '.join(deps)}</span> — training would fail. {txt}")
        else:
            gpus = [d for d in info["devices"] if d["id"] != "cpu"]
            opt = info.get("missing_optional") or []
            self.env_info.setText(f"<span style='color:{utils.hexc('ok')}'>✓</span> {txt}"
                                  + (f" · not installed (optional): {', '.join(opt)}" if opt else "")
                                  + ("" if gpus else f" — <span style='color:{utils.hexc('warn')}'>training on a CPU "
                                                     "is slow: fine for a quick look, use a GPU for real runs</span>"))
        if self._task() == "cls":
            self._fill_models()                            # timm there or not: which architectures train here
        want = self.settings.value("train/device", "")
        self.device.clear()
        for d in info.get("devices", []):
            if d["id"] != "cpu":
                self.device.addItem(d["name"], d["id"])
        gpu_ids = [d["id"] for d in info.get("devices", []) if d["id"] not in ("cpu", "mps")]
        if len(gpu_ids) > 1:
            self.device.addItem(f"All {len(gpu_ids)} GPUs", ",".join(gpu_ids))
        self.device.addItem("CPU", "cpu")
        k = self.device.findData(want)
        self.device.setCurrentIndex(k if k >= 0 else 0)
        self._update_start()

    def install_missing(self, packages=None, python=None, then=None):
        pkgs = list(packages or getattr(self, "_to_install", []) or [])
        py = python or (self.env or {}).get("_python") or self._python()
        if not pkgs or not py:
            return
        if QMessageBox.question(self, self.TITLE, f"Install {', '.join(pkgs)} into\n{py}\n\nwith pip?") != \
                QMessageBox.StandardButton.Yes:
            return
        self.install_btn.setEnabled(False)
        self.env_info.setText(f"Installing {', '.join(pkgs)} … (pip; this can take a minute)")

        def work():
            return T.pip_install(py, pkgs)

        def done(res):
            ok, out = res
            self.install_btn.setEnabled(True)
            if not ok:
                self.env_info.setText(f"<span style='color:{utils.hexc('error')}'>pip could not install "
                                      f"{', '.join(pkgs)}:</span><br><code>{out[-600:].replace(chr(10), '<br>')}</code>")
                return
            if then is not None:
                then()
            self.check_env()
        self._run(work, done, lambda m: (self.install_btn.setEnabled(True), self.env_info.setText(f"⚠ {m}")))

    # ================================================================== model
    def _task(self) -> str:
        return self.check.task if self.check is not None else "det"

    def _torch(self, m=None) -> bool:
        """Does the chosen model train with AIO QC's PyTorch classifier trainer (any architecture but YOLO)?"""
        return T.is_torch_model(self.model_box.currentData() if m is None else m)

    def _hyp_task(self) -> str:
        return "clst" if self._torch() else self._task()

    def _fill_models(self):
        want = self.model_box.currentData() or self.settings.value("train/model", "")
        task = self._task()
        self.model_box.blockSignals(True)
        self.model_box.clear()
        for fam, sizes, _d, _s in T.FAMILIES:
            for s_ in sizes:
                name = T.weights_name(fam, s_, task)
                self.model_box.addItem(f"{fam} {s_} — {T.SIZE_WORD[s_]}  ({name})", name)
        if task == "cls":                                  # every other popular classifier (AIO QC's PyTorch trainer)
            have_timm = self.env is None or bool(self.env.get("timm"))
            for g, gtitle in T.CLS_GROUPS.items():
                self.model_box.insertSeparator(self.model_box.count())
                self.model_box.addItem(f"— {gtitle} —", None)
                self.model_box.model().item(self.model_box.count() - 1).setEnabled(False)
                for key, title, grp, tm, tv, size, params in T.CLS_ARCHS:
                    if grp != g:
                        continue
                    lib = "" if (tm and have_timm) else (" · torchvision" if tv else " · needs timm")
                    upd = T.timm_update(f"arch:{key}", self.env) if tm and have_timm else None
                    if upd:
                        lib = " · torchvision" if tv else f" · needs timm {upd[6:]}+"
                    self.model_box.addItem(f"{title} — {params:g} M · {size} px{lib}", f"arch:{key}")
            self.model_box.insertSeparator(self.model_box.count())
            self.model_box.addItem("Any timm model, by name…", "ask:timm")
            self.model_box.addItem("Any torchvision model, by name…", "ask:tv")
        self.model_box.insertSeparator(self.model_box.count())
        for r in T.list_runs(self._project())[:12]:
            if r.get("best_pt"):
                self.model_box.addItem(f"Earlier run: {r['name']}" + (
                    f" ({r.get('metric', 'mAP50-95')} {r['best']:.3f})" if r["best"] is not None else ""), r["best_pt"])
        try:
            from AIO_QC.core import model_registry as R
            root = getattr(getattr(self.app, "session", None), "root", None)
            for e in R.all_models(root)[:20]:
                if str(e.get("path", "")).lower().endswith(".pt") and self.model_box.findData(e["path"]) < 0:
                    self.model_box.addItem(f"Your model: {e.get('name') or Path(e['path']).stem}", e["path"])
        except Exception as _exc:
            _ignored(_exc)
        k = self.model_box.findData(want)
        if k < 0 and want:
            stem = Path(str(want)).stem.replace("-seg", "").replace("-cls", "")
            k = self.model_box.findData(stem + {"seg": "-seg", "cls": "-cls"}.get(task, "") + ".pt")
        if k < 0 and str(want).startswith(("timm:", "tv:")) and task == "cls":     # a model chosen by name before
            self.model_box.insertItem(0, f"{T.arch_title(want)} — {want}", want)
            k = 0
        self.model_box.setCurrentIndex(k if k >= 0 else max(0, self.model_box.findData(T.weights_name("YOLO11", "s",
                                                                                                       task))))
        self.model_box.blockSignals(False)
        self._model_changed()

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Model to start from", "", "Model (*.pt *.pth *.yaml);;All (*)")
        if p:
            self.model_box.insertItem(0, f"{Path(p).name} — {p}", p)
            self.model_box.setCurrentIndex(0)

    def _ask_model_name(self, lib: str):
        """Any timm / torchvision classifier by its name (timm.list_models() / torchvision.models.list_models())."""
        from PyQt6.QtWidgets import QInputDialog
        what = "timm" if lib == "timm" else "torchvision"
        eg = "convnext_base.fb_in22k_ft_in1k, efficientnet_b5.sw_in12k_ft_in1k, vit_base_patch14_dinov2.lvd142m" \
            if lib == "timm" else "resnet152, efficientnet_b4, regnet_y_8gf, swin_v2_s, maxvit_t"
        name, ok = QInputDialog.getText(self, f"A {what} model", f"The {what} model name (with its pretrained tag for "
                                        f"timm), e.g. {eg}:", text=self.settings.value(f"train/ask_{lib}", "") or "")
        name = (name or "").strip()
        if not ok or not name or not re.match(r"^[A-Za-z0-9_.\-]+$", name):
            return None
        self.settings.setValue(f"train/ask_{lib}", name)
        data = f"{lib}:{name}"
        self.model_box.blockSignals(True)
        k = self.model_box.findData(data)
        if k < 0:
            self.model_box.insertItem(0, f"{name} ({what})", data)
            k = 0
        self.model_box.setCurrentIndex(k)
        self.model_box.blockSignals(False)
        return data

    def _model_changed(self, _i=None):
        m = self.model_box.currentData() or ""
        if str(m).startswith("ask:"):
            m = self._ask_model_name(m[4:])
            if m is None:                                  # cancelled: back to the first model
                self.model_box.blockSignals(True)
                self.model_box.setCurrentIndex(0)
                self.model_box.blockSignals(False)
                m = self.model_box.currentData() or ""
        t = T.model_task(m)
        notes = []
        if self.check is not None and t is not None and t != self.check.task:
            if "cls" in (t, self.check.task):
                notes.append(f"<span style='color:{utils.hexc('error')}'>" + (
                    "A classifier (…-cls) needs a classification dataset (class folders)." if t == "cls" else
                    "This is a classification dataset: choose a classifier (…-cls).") + "</span>")
            elif t == "seg" and self.check.task == "det":
                notes.append(f"<span style='color:{utils.hexc('error')}'>A segmentation model needs polygon labels; "
                             "this dataset has boxes — choose a detection model.</span>")
            else:
                notes.append("A detection model on polygon labels: their bounding boxes are learned.")
        torch_ = self._torch(m)
        if torch_ and str(m).startswith(("arch:", "timm:", "tv:")):
            resolved, why = T.resolve_arch(m, self.env)
            sz = T.arch_imgsz(m)
            if why:
                notes.append(f"<span style='color:{utils.hexc('error')}'>{why}</span>")
            else:
                lib = "timm" if resolved.startswith("timm:") else "torchvision"
                notes.append(f"Trained by AIO QC's PyTorch trainer ({lib} <code>{resolved.split(':', 1)[1]}</code>"
                             + (f", {sz} px" if sz else "") + "): pretrained weights download once. Saves best.pt "
                             "(Testing, Model Check, the Class Gallery and Ultralytics use it) and best.onnx.")
                if str(m).startswith("arch:") and T.CLS_ARCH[m[5:]][2] == "found":
                    notes.append("<b>Few labelled images?</b> Augmentation and learning rate ▸ Schedule ▸ <i>Few "
                                 "labelled images (linear probe)</i>: the backbone runs once over every image and "
                                 "only the classifier on top learns — minutes, even for the large models.")
                lic = T.arch_licence(resolved)
                if lic:
                    notes.append(f"Its weights come under {lic}.")
            if sz:
                self.imgsz_note.setText(f"{T.arch_title(m)} was pretrained at {sz} px — larger can help small "
                                        "details (slower); sizes are multiples of 32.")
            if sz and not self._imgsz_touched:
                self._programmatic = True
                self.imgsz.setValue(sz)
                self._programmatic = False
        elif self.check is not None and self.check.cls_folder:
            self.imgsz_note.setText(f"Suggested image size {T.suggest_imgsz(self.check)[1]}.")
        if torch_ and not str(m).startswith(("arch:", "timm:", "tv:")):
            info = T.checkpoint_info(m) or {}
            notes.append(f"Your {info.get('arch') or 'classifier'} ({len(info.get('names') or [])} classes"
                         + (f", top-1 {info['top1']:.3f}" if info.get("top1") is not None else "") + "): its classes "
                         "are matched to this dataset's by name — they keep what they learned; new ones start "
                         "fresh.")
            if info.get("imgsz") and not self._imgsz_touched:
                self._programmatic = True
                self.imgsz.setValue(int(info["imgsz"]))
                self._programmatic = False
        if self.check is not None and self.check.flat and t == "cls" and not torch_:
            notes.append(f"<span style='color:{utils.hexc('error')}'>A YOLO classifier needs train/ and val/ "
                         "folders (Convert & Split ▸ Class folders) — or choose another architecture below: it "
                         "holds out validation images itself.</span>")
        if str(m).endswith(".yaml"):
            notes.append("A model .yaml trains the architecture from scratch.")
        elif m and not Path(m).is_absolute() and not torch_:
            have = (T.weights_dir() / m).exists()
            notes.append("Downloaded once, then kept." if not have else "Already downloaded.")
        self.model_note.setText(" ".join(notes))
        self.hyp.set_task(self._hyp_task())
        self.hyp.set_finetune_hint(str(m).endswith((".pt", ".pth")) and Path(m).is_absolute())  # trained before
        self.pretrained.setEnabled(not str(m).endswith(".yaml") and not (torch_ and Path(str(m)).is_absolute()))
        for w in (self.cache, self.single_cls):
            w.setEnabled(not torch_)
        self.onnx.setEnabled(torch_)
        self.goal.setEnabled(True)
        for b in self.presets.buttons():
            b.setChecked(False)
        self._update_start()

    def apply_preset(self, key):
        p = T.PRESETS[key]
        fam = "YOLO11"
        cur = self.model_box.currentData() or ""
        if self._torch(cur):                       # another architecture, chosen on purpose: only the run length
            self.epochs.setValue(p["epochs"])
            self.patience.setValue(min(p["patience"], 20))
            for b in self.presets.buttons():
                b.setChecked(b.property("preset") == key)
            return
        for f_, _s, _d, _g in T.FAMILIES:
            if Path(cur).stem.lower().startswith(f_.lower()):
                fam = f_
        k = self.model_box.findData(T.weights_name(fam, p["size"], self._task()))
        if k >= 0:
            self.model_box.setCurrentIndex(k)
        self.epochs.setValue(p["epochs"])
        self.patience.setValue(p["patience"])
        for b in self.presets.buttons():
            b.setChecked(b.property("preset") == key)

    # ================================================================== settings helpers
    def _check_extra(self):
        try:
            d = T.parse_extra(self.extra.text())
        except ValueError as exc:
            self.extra_note.setText(f"⚠ {exc}")
            return None
        if self._torch():
            bad = [k for k in d if k not in T.torch_keys()]
            if bad:
                self.extra_note.setText("⚠ not a setting of the PyTorch classifier trainer: " + ", ".join(bad))
                return None
        else:
            bad = T.unknown_settings(d, self.env)
        if bad:                                            # a typo would only fail once the run starts
            self.extra_note.setText(f"⚠ not an Ultralytics {(self.env or {}).get('ultralytics') or ''} setting: "
                                    + ", ".join(bad))
            return None
        txt = ", ".join(f"{k}={v}" for k, v in d.items()) if d else ""
        over = [k for k in d if k in T.HYP_KEYS or k in ("optimizer", "lr0", "close_mosaic")]
        if over:
            txt += f" — these win over Augmentation and learning rate: {', '.join(over)}"
        self.extra_note.setText(txt)
        self.extra_note.setStyleSheet("")
        return d

    def _edit_recipe(self, name: str):
        """Image Augmentation, on the recipe chosen here."""
        if not hasattr(self.app, "open_augment"):
            return
        dlg = self.app.open_augment()
        if name:
            dlg.load_named(name)

    def _hyp_changed(self):
        if hasattr(self, "start_btn"):
            self._check_extra()
            self._update_start()

    def _cache_hint(self):
        if self.check is None:
            return
        n = self.check.n("train") + (self.check.n("val") if self.check.data.get("val") != self.check.data.get("train")
                                     else 0)
        gb = n * self.imgsz.value() ** 2 * 3 * 0.6 / 2**30          # resized, letterbox aspect ~0.6
        from AIO_QC.core.procs import available_memory
        free = (available_memory() or 0) / 2**30
        self.cache_note.setText(f"RAM cache needs ≈ {gb:.1f} GB (free now: {free:.1f} GB)."
                                + (" Fits easily: worth it when the images are on a slow disk." if gb < free * 0.4
                                   else " Too big for RAM: off or disk." if gb > free * 0.7 else ""))

    def _project(self) -> Path:
        t = self.project.text().strip()
        if t:
            return Path(t).expanduser()
        if self.check is not None:
            return self.check.yaml.parent / "runs"
        y = self.yaml.text().strip()
        return Path(y).expanduser().parent / "runs" if y else Path.home() / "aioqc_runs"

    def _browse_project(self):
        d = QFileDialog.getExistingDirectory(self, "Save runs to", str(self._project()))
        if d:
            self.project.setText(d)
            self.refresh_runs()

    def _batch(self):
        t = self.batch.currentText().strip().lower()
        if t.startswith("auto"):
            return -1
        try:
            v = float(t)
        except ValueError:
            return None
        return v if 0 < v < 1 else int(v)            # 0.7 = use 70 % of the GPU memory (Ultralytics)

    def _update_start(self):
        probs = []
        if self.check is None:
            probs.append("check the dataset")
        elif not self.check.ok:
            probs.append("fix the dataset errors")
        m = self.model_box.currentData() or ""
        torch_ = self._torch(m)
        if self.env is None:
            probs.append("check the environment")
        elif torch_ and not self.env.get("torch"):
            probs.append("an environment with PyTorch")
        elif torch_:
            _r, why = T.resolve_arch(m, self.env)
            upd = T.timm_update(m, self.env)
            if why:
                probs.append(f"timm {upd[6:]} or newer (Update timm)" if upd else
                             "timm in the environment (Install timm)")
        elif not (self.env.get("torch") and self.env.get("ultralytics")):
            probs.append("an environment with torch + ultralytics")
        elif self.env.get("missing"):
            probs.append(f"the packages Ultralytics needs ({', '.join(self.env['missing'])}: Install)")
        if self.check is not None and self.check.flat and not torch_:
            probs.append("train/ and val/ folders for a YOLO classifier (or another architecture)")
        if self.check is not None and T.model_task(m) == "seg" and self.check.task == "det":
            probs.append("a detection model for box labels")
        if self._batch() is None:
            probs.append("a valid batch")
        if self._check_extra() is None:
            probs.append("valid extra settings")
        if getattr(self, "goal", None) is not None and self.goal.currentData() != "train" and \
                not str(m).lower().endswith(".pt"):
            probs.append("your model (.pt) to extend")
        self._install_hint(torch_, m)
        self.start_btn.setEnabled(not probs)
        dev = self.device.currentData() or "cpu"
        est = ""
        if self.check is not None and not probs:
            n = self.check.n("train")
            est = f"{n:,} train images × {self.epochs.value()} epochs on {self.device.currentText().split(' · ')[0]}."
            if dev == "cpu" and self._batch() == -1:
                est += " Auto batch needs a GPU: 16 is used on the CPU."
        free = T.free_disk_gb(self._project())
        self.disk_note.setText(f"{self._project()} · {free:.0f} GB free" if math.isfinite(free) else str(self._project()))
        self.start_note.setText(("Needs: " + ", ".join(probs) + ".") if probs else est)

    def _install_hint(self, torch_, m):
        """The environment lacks timm and the chosen architecture needs it: the Install button offers it."""
        if self.env is None or not self.env.get("torch"):
            return
        need = torch_ and bool(T.resolve_arch(m, self.env)[1])
        upd = T.timm_update(m, self.env) if torch_ else None
        base = [p for p in (getattr(self, "_to_install", None) or []) if not p.startswith("timm")]
        self._to_install = base + ([upd or "timm"] if need else [])
        self.install_btn.setVisible(bool(self._to_install))
        self.install_btn.setText("Update timm" if need and upd and not base else
                                 f"Install {', '.join(self._to_install)}" if self._to_install else "Install missing")

    # ================================================================== start / control
    def settings_dict(self) -> dict:
        m = self.model_box.currentData() or ""
        batch = self._batch()
        dev = self.device.currentData() or "cpu"
        if dev == "cpu" and batch == -1:
            batch = 16
        s = {"model": m, "task": self._task(), "epochs": self.epochs.value(), "imgsz": self.imgsz.value(),
             "batch": batch, "device": dev, "workers": self.workers_n.value(), "patience": self.patience.value(),
             "project": str(self._project()), "name": self.run_name.text().strip() or T.new_run_name(m),
             "pretrained": self.pretrained.isChecked(), "seed": self.seed.value(),
             "cache": self.cache.currentData(), "close_mosaic": self.close_mosaic.value(),
             "amp": self.amp.isChecked(), "fraction": round(self.fraction.value(), 3),
             "single_cls": self.single_cls.isChecked() or None, "extra": self._check_extra() or {},
             "hyp": self.hyp.values(full=self._torch(m)), "augment": self.hyp.augment()}
        if self._torch(m):                         # the PyTorch classifier trainer: what it needs on top
            s["hyp"]["onnx"] = self.onnx.isChecked()
            s["env"] = {"timm": (self.env or {}).get("timm"), "torchvision": (self.env or {}).get("torchvision")}
            for k in ("cache", "single_cls", "close_mosaic"):
                s.pop(k, None)
        if self.optimizer.currentText() != "auto":
            s["optimizer"] = self.optimizer.currentText()
        if self.lr0.value() > 0:
            s["lr0"] = self.lr0.value()
        if self.time_h.value() > 0:
            s["time_h"] = self.time_h.value()
        return s

    # ------------------------------------------------------------------ extend model
    def _goal_changed(self):
        ext = self.goal.currentData() != "train"
        self.extend_btn.setVisible(ext)
        self.pretrained.setEnabled(not ext and not str(self.model_box.currentData() or "").endswith(".yaml"))
        self.extend_plan = None
        if ext and not str(self.model_box.currentData() or "").lower().endswith(".pt"):
            self.model_note.setText(f"<span style='color:{utils.hexc('warn')}'>Choose your model (.pt) under Start "
                                    "from — an earlier run or Browse…</span>")
        elif ext:
            self.model_note.setText("Plan… shows the classes, the old data and the stages before anything runs.")
        self._update_start()

    def plan_extend(self) -> bool:
        from .extend_dialog import ExtendDialog
        base = str(self.model_box.currentData() or "")
        if not base.lower().endswith(".pt") or self.check is None:
            QMessageBox.information(self, self.TITLE, "Choose the dataset (step 1) and your model (.pt) first.")
            return False
        dlg = ExtendDialog(self, base, str(self.check.yaml), self._python(), goal=self.goal.currentData())
        self._extend_dlg = dlg
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.result_plan is None:
            return False
        self.extend_plan = dlg.result_plan
        P = self.extend_plan
        self.model_note.setText(f"Extend plan: {len(P.base_names)} class(es) kept"
                                + (f" + {len(P.new_classes)} new ({', '.join(P.new_classes[:5])})" if P.new_classes
                                   else "") + (" · old data mixed in" if P.old_yaml else " · new images only")
                                + (" · pre-labels" + (" (review first)" if P.review else "") if P.prelabel else "")
                                + (" · two stages" if P.two_stage else ""))
        return True

    def start_extend(self, s: dict):
        P = self.extend_plan
        if P is None and not self.plan_extend():
            return
        P = self.extend_plan
        py = self.env["_python"]
        self.foot_note.setText("Building the training copy (hard links) …")

        def work():
            run, st = T.prepare_extend_run(py, s, P)
            if st.get("cancelled"):
                return run, st, None
            return run, st, T.launch(run, py)

        def done(res):
            run, st, info = res
            self.foot_note.setText(f"Training copy: {sum(st.get('images', {}).values()):,} images · "
                                   f"{st.get('labels_rewritten', 0):,} label files with new ids"
                                   + (f" · {st['leak_skipped']} new image(s) left out: copies of old validation "
                                      "images" if st.get("leak_skipped") else ""))
            if info is None:
                return
            self.app_watch(run)
            self.run_name.clear()
            self.extend_plan = None
            self.refresh_runs(select=run)

        def failed(m):
            QMessageBox.warning(self, self.TITLE, f"Could not start the run:\n{m}")
        self._run(work, done, failed)

    def continue_extend(self):
        if self.current is None:
            return
        py = (self.env or {}).get("_python") or self._spec(self.current).get("python") or self._python()
        T.prepare_continue(self.current)
        T.launch(self.current, py)
        self.app_watch(self.current)
        self._sig = None
        self.poll(force=True)

    def open_training_copy(self):
        if self.current is None:
            return
        data = self.current / "data"
        opener = getattr(self.app, "open_dataset", None)
        if opener is not None and QMessageBox.question(
                self, self.TITLE, f"Open the training copy ({data}) in this window to review the pre-labels?\n\n"
                "It is a YOLO dataset of its own: fixes there change only the copy, never your dataset. prelabels.csv "
                "(in the run's aioqc folder) lists every pre-label.") == QMessageBox.StandardButton.Yes:
            try:
                opener(data)
                return
            except Exception as exc:
                _ignored(exc)
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(data)))

    def start(self):
        if self.check is None or self.env is None:
            return
        s = self.settings_dict()
        if self.goal.currentData() != "train":
            self.start_extend(s)
            return
        warn = [t for lv, t in self.check.issues if lv == "warn"]
        if warn and QMessageBox.question(self, self.TITLE, "The dataset check found:\n\n· " + "\n· ".join(warn[:6])
                                         + "\n\nTrain anyway?") != QMessageBox.StandardButton.Yes:
            return
        if s["device"] not in ("cpu", "mps"):
            svc = getattr(self.app, "inference", None)
            if svc is not None and getattr(svc, "client", None) is not None:
                svc.stop()                             # the model server's GPU memory goes back to training
        self.settings.setValue("train/onnx", "true" if self.onnx.isChecked() else "false")
        for k, v in (("model", s["model"]), ("epochs", s["epochs"]), ("patience", s["patience"]),
                     ("batch", self.batch.currentText()), ("device", s["device"]), ("workers", s["workers"]),
                     ("project", self.project.text().strip()), ("extra", self.extra.text())):
            self.settings.setValue(f"train/{k}", v)
        py = self.env["_python"]
        try:
            run = T.prepare_run(py, self.check, s)
            info = T.launch(run, py)
        except Exception as exc:
            _ignored(exc)
            QMessageBox.warning(self, self.TITLE, f"Could not start the run:\n{exc}")
            return
        self.app_watch(run)
        self.run_name.clear()
        if not info["survives"]:
            self.foot_note.setText("This run ends when AIO QC closes (Windows did not let it run on its own).")
        self.refresh_runs(select=run)

    def app_watch(self, run):
        w = getattr(self.app, "train_watch", None)
        if w is not None:
            w.add(run)

    def _control(self, what):
        if self.current is None:
            return
        if what == "kill":
            if QMessageBox.question(self, self.TITLE, "End this run at once? Its GPU and memory are freed; the last "
                                    "finished epoch can be resumed.") != QMessageBox.StandardButton.Yes:
                return
            from .training_monitor import monitor
            monitor().end(self.current, done=lambda: self.poll(force=True))
        else:
            T.control(self.current, what)
        self.run_state.setText({"stop": "Stopping — finishing this epoch, then checking best.pt …",
                                "pause": "Pausing after the next saved epoch …",
                                "kill": "Ending …"}[what])

    def resume(self):
        if self.current is None:
            return
        py = (self.env or {}).get("_python") or T.status(self.current).get("python") or self._python()
        if not py:
            return
        T.prepare_resume(self.current, py)
        T.launch(self.current, py)
        self.app_watch(self.current)
        self._sig = None
        self.poll(force=True)

    def fix_and_retry(self):
        run, pkg = self.current, getattr(self, "_fix_pkg", None)
        if run is None or not pkg:
            return
        spec = self._spec(run)
        py = spec.get("python") or (self.env or {}).get("_python") or self._python()
        miss = (self.env or {}).get("missing") if (self.env or {}).get("_python") == py else None
        pkgs = sorted(set([pkg] + list(miss or [])))

        def again():
            if (run / "weights" / "last.pt").exists() and T.status(run).get("last_epoch"):
                T.prepare_resume(run, py)
            else:                                          # failed before the first epoch: the same run again
                for f in ("epochs.jsonl",):
                    try:
                        (run / "aioqc" / f).unlink()
                    except OSError:
                        pass
            T.launch(run, py)
            self.app_watch(run)
            self._sig = None
            self.poll(force=True)
        self.install_missing(pkgs, py, then=again)

    def fine_tune(self):
        if self.current is None:
            return
        best = self.current / "weights" / "best.pt"
        if best.is_file():
            self.model_box.insertItem(0, f"From {self.current.name} — best.pt", str(best))
            self.model_box.setCurrentIndex(0)
            self.hyp.apply_lr("finetune")                  # small learning rate, short warm-up, backbone frozen
            self.hyp_drawer.set_open(True)
            self.foot_note.setText(
                "Fine-tuning settings applied (AdamW 0.0002, 1 warm-up epoch, cosine decay, the head alone for 2 "
                "epochs, then layer-wise decay) — or “Improve my model” for hard examples; change them under "
                "Augmentation and learning rate." if self._torch(str(best)) else
                "Fine-tuning settings applied (AdamW 0.0005, 1 warm-up epoch, cosine decay, first 10 layers frozen) — "
                "change them under Augmentation and learning rate.")
            self.epochs.setFocus()

    def validate(self):
        if self.current is None:
            return
        best = self.current / "weights" / "best.pt"
        spec = self._spec(self.current)
        data = (spec.get("args") or {}).get("data") or (str(self.check.yaml) if self.check else "")
        if not best.is_file() or not data:
            return
        from PyQt6.QtWidgets import QInputDialog
        splits = ["val", "test", "train"]
        sp, ok = QInputDialog.getItem(self, "Validate", "Score best.pt on", splits, 0, False)
        if not ok:
            return
        py = (self.env or {}).get("_python") or spec.get("python") or self._python()
        a = spec.get("args") or {}
        out = T.prepare_val(str(best), data, self.current / f"val_{sp}", py, sp, a.get("imgsz", 640),
                            a.get("device", ""), 16 if a.get("batch", 16) == -1 else a.get("batch", 16),
                            {"val_share": a["val_share"]} if a.get("val_share") else None)
        T.launch(out, py)
        self.refresh_runs(select=out, project=self.current)

    def add_to_models(self):
        if self.current is None:
            return
        best = self.current / "weights" / "best.pt"
        if not best.is_file():
            return
        from AIO_QC.core import model_registry as R
        spec = self._spec(self.current)
        names = {str(i): n for i, n in enumerate(spec.get("names") or [])}
        session = getattr(self.app, "session", None)
        root = getattr(session, "root", None)
        info = T.checkpoint_info(best) or {}
        R.register(str(best), root=root, name=f"{self.current.name} (best)", names=names,
                   labels=set(session.label_totals()) if session is not None else None,
                   task={"seg": "segment", "cls": "classify"}.get(spec.get("task"), "detect"),
                   train_imgsz=info.get("imgsz") or (spec.get("args") or {}).get("imgsz"))
        toast = getattr(self.app, "toast", None)
        if toast is not None:
            toast.show_message(f"<b>{self.current.name}</b> added to the models — Model Check, Testing and "
                               "Auto-Annotate can use it now.", kind="success", key="train-added")

    def improve(self):
        if self.current is not None and hasattr(self.app, "open_improve"):
            self.app.open_improve(run=self.current, check=True)

    def _open_run_folder(self):
        if self.current is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current)))

    @staticmethod
    def _spec(run) -> dict:
        try:
            return json.loads((Path(run) / "aioqc" / "spec.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # ================================================================== runs list
    def refresh_runs(self, select=None, project=None):
        proj = Path(project) if project else self._project()
        cur = Path(select) if select else self.current
        rows = T.list_runs(proj)
        if project is None and self.current is not None and self.current.parent != proj and \
                not any(r["dir"] == self.current for r in rows):
            pass
        self.runs.blockSignals(True)
        self.runs.clear()
        pick = None
        for r in rows:
            col = utils.hexc(STATE_COLOR.get(r["state"], "muted"))
            score = f" · {r.get('metric', 'mAP50-95')} {r['best']:.3f}" if r["best"] is not None else ""
            ep = f" · {r['epoch']}/{r['epochs']} epochs" if r["epochs"] and r["mode"] != "val" else ""
            it = QListWidgetItem(f"{r['name']}   [{STATE_WORD.get(r['state'], r['state'])}]{ep}{score}")
            it.setData(Qt.ItemDataRole.UserRole, str(r["dir"]))
            it.setForeground(QColor(col))
            it.setToolTip(f"{r['model']} · {time.strftime('%Y-%m-%d %H:%M', time.localtime(float(r['started'])))}")
            self.runs.addItem(it)
            if cur is not None and Path(r["dir"]) == cur:
                pick = it
        # nested validation runs of the selected run
        if cur is not None and cur.parent != proj and pick is None and (cur / "aioqc").is_dir():
            it = QListWidgetItem(f"{cur.parent.name} ▸ {cur.name}")
            it.setData(Qt.ItemDataRole.UserRole, str(cur))
            self.runs.insertItem(0, it)
            pick = it
        self.runs.blockSignals(False)
        if pick is not None:
            self.runs.setCurrentItem(pick)
            if Path(pick.data(Qt.ItemDataRole.UserRole)) != self.current:
                self.show_run(pick.data(Qt.ItemDataRole.UserRole))
        elif self.runs.count() and self.current is None:
            self.runs.setCurrentRow(0)

    def show_run(self, path):
        self.current = Path(path) if path else None
        self._sig = None
        self._ep_cache = (None, [])
        self.log.clear()
        self.poll(force=True)

    # ================================================================== live view
    def poll(self, force=False):
        T.reap()
        if not self.isVisible() and not force:
            return
        run = self.current
        if run is None:
            self.run_title.setText("No run selected")
            self.run_state.setText("Start a run on the left; it shows up here.")
            self._show_run_buttons({})
            return
        st = T.status(run)
        state = T.state_of(st) if st else ("finished" if (run / "weights" / "best.pt").exists() else "unknown")
        try:
            sig = ((run / "aioqc" / "status.json").stat().st_mtime, (run / "aioqc" / "epochs.jsonl").stat().st_mtime
                   if (run / "aioqc" / "epochs.jsonl").exists() else 0, self.tabs.currentIndex(), state)
        except OSError:
            sig = (0, 0, self.tabs.currentIndex(), state)
        if sig == self._sig and not force:
            return
        self._sig = sig
        spec = self._spec(run)
        self.run_title.setText(run.name + (f"  ·  {Path(spec.get('model', '')).name}" if spec.get("model") else ""))
        self._show_state(st, state, spec)
        self._show_run_buttons(st, state, run)
        rows = self._epochs(run)
        self._draw(rows, st)
        i = self.tabs.currentIndex()
        if i == 1:
            self._per_class(st)
        elif i == 2:
            self._show_log(run)
        elif i == 3:
            self.args_view.setPlainText(json.dumps(spec.get("args") or spec, indent=2, ensure_ascii=False))

    def _epochs(self, run):
        p = run / "aioqc" / "epochs.jsonl"
        try:
            m = p.stat().st_mtime if p.exists() else (run / "results.csv").stat().st_mtime
        except OSError:
            m = None
        if self._ep_cache[0] != (run, m):
            self._ep_cache = ((run, m), T.epochs(run))
        return self._ep_cache[1]

    def _show_state(self, st, state, spec):
        col = utils.hexc(STATE_COLOR.get(state, "muted"))
        ep, eps = st.get("epoch") or 0, st.get("epochs") or (spec.get("args") or {}).get("epochs") or 0
        txt = f"<b style='color:{col}'>{STATE_WORD.get(state, state)}</b>"
        if st.get("phase") and state in T.RUNNING:
            txt += f" · {st['phase']}"
        el = st.get("elapsed") or (time.time() - float(st["started"]) if st.get("started") and state in T.RUNNING
                                   else None)
        if el:
            txt += f" · {_dur(el)}"
        if state in T.RUNNING and st.get("epoch_time") and eps:
            left = (eps - (st.get("last_epoch") or 0)) * float(st["epoch_time"])
            txt += f" · about {_dur(left)} left (at most; it stops earlier when val stops improving)"
        if st.get("gpu_mem_gb"):
            txt += f" · GPU memory {st['gpu_mem_gb']:.1f} GB"
        if st.get("train_images"):
            txt += f" · {st['train_images']:,} train / {st.get('val_images') or 0:,} val images"
        if st.get("message") and state not in T.RUNNING:
            txt += f" — {st['message']}"
        if state == "failed":
            hint = T.explain((st.get("error") or "") + "\n" + (st.get("error_tail") or ""))
            txt += f"<br><span style='color:{utils.hexc('error')}'>{st.get('error', '')}</span>"
            if hint:
                txt += f"<br><b>What to do:</b> {hint}"
        elif state == "interrupted":
            txt += "<br>The training process is gone (the computer restarted, or it was closed)." + \
                   (" Resume continues from the last saved epoch." if (self.current / "weights" / "last.pt").exists()
                    else "")
        if st.get("request") and state in T.RUNNING:
            txt += f" · {st['request']} requested"
        if spec.get("extend") and state == "finished" and self.current is not None:
            txt += "<br>" + self._extend_summary(self.current, spec)
        self.run_state.setText(txt)
        self.ep_bar.setMaximum(max(1, int(eps or 1)))
        self.ep_bar.setValue(min(int(ep or 0), int(eps or 1)))
        self.ep_bar.setVisible(spec.get("mode") != "val")
        self.b_bar.setMaximum(max(1, int(st.get("batches") or 1)))
        self.b_bar.setValue(int(st.get("batch") or 0))
        self.b_bar.setVisible(state in T.RUNNING and spec.get("mode") != "val")

    @staticmethod
    def _extend_summary(run, spec) -> str:
        res = T.extend_result(run)
        if res is None:
            return "Extended model — the before / after check is not there (see the log)."
        learned = ", ".join(f"{r['name']} {100 * (r['after'] or 0):.0f}" for r in res["learned"]
                            if r["split"] == "val_new")
        old = [r for r in res["rows"] if r["split"] == "val_old" and r["old_class"] and r["delta"] is not None]
        lim = 100 * float((spec.get("extend") or {}).get("forget_limit") or 0.03)
        if res["forgot"]:
            f = ", ".join(f"{r['name']} {100 * r['delta']:+.0f}" for r in res["forgot"])
            o = f"<span style='color:{utils.hexc('warn')}'>old classes that dropped more than {lim:.0f} mAP points " \
                f"on the old validation images: {f}</span> — more old images, or a lower learning rate, helps"
        elif old:
            o = f"<span style='color:{utils.hexc('ok')}'>no old class dropped more than {lim:.0f} mAP points</span> " \
                f"({len(old)} checked on the old validation images)"
        else:
            o = "the old classes were not checked (no old training data)"
        return "<b>Extended:</b> " + (f"learned {learned} (mAP50-95 %, new validation images) · " if learned else "") + o

    def _show_run_buttons(self, st, state="", run=None):
        running = state in T.RUNNING
        stuck = state == "not responding"
        has_best = run is not None and (run / "weights" / "best.pt").exists()
        resumable = run is not None and (run / "weights" / "last.pt").exists() and \
            state in ("paused", "stopped", "interrupted", "failed") and st.get("resumable", True) is not False
        val = (self._spec(run).get("mode") == "val") if run is not None else False
        pkg = T.missing_module((st.get("error") or "") + (st.get("error_tail") or "")) if state == "failed" else None
        self.b_fix.setVisible(bool(pkg) and pkg not in ("torch",))
        if pkg:
            self.b_fix.setText(f"Install {pkg} and retry")
            self._fix_pkg = pkg
        for b, on in ((self.b_stop, running and not val), (self.b_pause, running and not val),
                      (self.b_kill, running or stuck), (self.b_resume, resumable and not val),
                      (self.b_add, has_best and not running), (self.b_val, has_best and not running),
                      (self.b_tune, has_best), (self.b_folder, run is not None),
                      (self.b_improve, has_best and not running and not val
                       and getattr(self.app, "session", None) is not None),
                      (self.b_review, state == "review"), (self.b_continue, state == "review")):
            b.setVisible(on)

    def _draw(self, rows, st):
        train_rows = [r for r in rows if not r.get("final")]
        loss_keys, seen = [], set()
        for r in train_rows:
            for k in list((r.get("loss") or {}).keys()) + [k for k in (r.get("metrics") or {}) if k.startswith("val/")]:
                if k not in seen:
                    seen.add(k)
                    loss_keys.append(k)
        series = []
        base = {}
        for k in loss_keys:
            short = k.split("/", 1)[-1]
            if short not in base:
                base[short] = SERIES_COLORS[len(base) % len(SERIES_COLORS)]
            pts = [(r["epoch"], (r.get("loss") or {}).get(k, (r.get("metrics") or {}).get(k))) for r in train_rows]
            pts = [(x, y) for x, y in pts if y is not None]
            series.append((("val " if k.startswith("val/") else "") + short.replace("_loss", ""), pts, base[short],
                           k.startswith("val/")))
        self.loss_chart.set_series(series)
        mkeys = [k for k in ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/mAP50-95(M)", "metrics/mAP50(M)",
                             "metrics/precision(B)", "metrics/recall(B)", T.TOP1, T.TOP5)
                 if any(k in (r.get("metrics") or {}) for r in train_rows)]
        ms = []
        for i, k in enumerate(mkeys):
            pts = [(r["epoch"], (r.get("metrics") or {}).get(k)) for r in train_rows if k in (r.get("metrics") or {})]
            name = {T.TOP1: "top-1 accuracy", T.TOP5: "top-5 accuracy"}.get(k) or \
                k.split("/", 1)[1].replace("(B)", " box").replace("(M)", " mask")
            ms.append((name, pts, SERIES_COLORS[i % len(SERIES_COLORS)], "precision" in k or "recall" in k))
        best_ep = None
        key = next((k for k in T.MAP_KEYS if any(k in (r.get("metrics") or {}) for r in train_rows)), None)
        if key:
            best_r = max(train_rows, key=lambda r: (r.get("metrics") or {}).get(key, -1))
            best_ep = best_r["epoch"]
        self.map_chart.set_series(ms, best_ep)
        # tiles: the best epoch (or the final check of best.pt)
        final = next((r for r in rows if r.get("final")), None)
        src = final or (max(train_rows, key=lambda r: (r.get("metrics") or {}).get(key, -1)) if key and train_rows
                        else (st.get("metrics") and {"metrics": st["metrics"]}) or None)
        m = (src or {}).get("metrics") or st.get("metrics") or {}
        cls = T.TOP1 in m or T.is_cls_rows(train_rows)
        self._set_cls_view(cls)
        if cls:                                     # a classifier: accuracy, not mAP
            self.t_map.set(m.get(T.TOP1), "top-1 accuracy" + (" · best.pt" if final else " (best)"))
            self.t_map50.set(m.get(T.TOP5), "top-5 accuracy")
            self.t_p.set(None)
            self.t_r.set(None)
            self.t_best.set(best_ep)
            return
        seg = "metrics/mAP50-95(M)" in m
        sfx = "(M)" if seg else "(B)"
        self.t_map.set(m.get(f"metrics/mAP50-95{sfx}", m.get("mask_map" if seg else "box_map")),
                       "mAP50-95 " + ("mask" if seg else "box") + (" · best.pt" if final else " (best)"))
        self.t_map50.set(m.get(f"metrics/mAP50{sfx}", m.get("box_map50")), "mAP50")
        self.t_p.set(m.get(f"metrics/precision{sfx}", m.get("box_p")))
        self.t_r.set(m.get(f"metrics/recall{sfx}", m.get("box_r")))
        self.t_best.set(best_ep)

    def _set_cls_view(self, cls: bool):
        """Score tiles and the per-class table for a classifier (accuracy) or a detector (mAP)."""
        if getattr(self, "_cls_view", None) == cls:
            return
        self._cls_view = cls
        self.t_p.setVisible(not cls)
        self.t_r.setVisible(not cls)
        self.cls_table.setHorizontalHeaderLabels(
            ["Class", "Images", "Precision", "Recall (= its accuracy)", "", ""] if cls else
            ["Class", "Objects", "Precision", "Recall", "mAP50", "mAP50-95"])
        for c in (4, 5):
            self.cls_table.setColumnHidden(c, cls)

    def _per_class(self, st):
        rows = st.get("per_class") or []
        seg = any("mask_map" in r for r in rows)
        tag = "mask" if seg else "box"
        self.cls_table.setSortingEnabled(False)
        self.cls_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [r.get("name"), r.get("instances", ""), r.get(f"{tag}_p"), r.get(f"{tag}_r"), r.get(f"{tag}_map50"),
                    r.get(f"{tag}_map")]
            for c, v in enumerate(vals):
                it = QTableWidgetItem()
                it.setData(Qt.ItemDataRole.DisplayRole, round(v, 3) if isinstance(v, float) else v)
                if (c == 5 or (c == 3 and getattr(self, "_cls_view", False))) and isinstance(v, float):
                    it.setForeground(QColor(utils.hexc("error" if v < 0.3 else "warn" if v < 0.5 else "ok")))
                self.cls_table.setItem(i, c, it)
        self.cls_table.setSortingEnabled(True)
        if not rows:
            self.cls_table.setRowCount(1)
            self.cls_table.setItem(0, 0, QTableWidgetItem("Per-class results appear when the run ends (or after "
                                                          "Validate)."))

    def _show_log(self, run):
        text = T.log_tail(run)
        if text != self.log.toPlainText():
            bar = self.log.verticalScrollBar()
            at_end = bar.value() >= bar.maximum() - 4
            self.log.setPlainText(text)
            if at_end:
                bar.setValue(bar.maximum())

    # ================================================================== plumbing
    def _run(self, fn, done, failed):
        from PyQt6 import sip

        def alive(cb):                     # a slow check (environment probe) may end after the window is gone
            return lambda *a: None if sip.isdeleted(self) or getattr(self, "_closed", False) else cb(*a)
        w = FuncWorker(fn, self)
        w.done.connect(alive(done))
        w.failed.connect(alive(failed))
        w.finished.connect(lambda: self.workers.remove(w) if w in self.workers else None)
        w.finished.connect(w.deleteLater)
        self.workers.append(w)
        w.start()

    def sizeHint(self):
        return QSize(1500, 950)

    def closeEvent(self, e):
        self._closed = not getattr(self, "keep_after_close", False)
        self.cancel.set()
        if self._check_cancel is not None:
            self._check_cancel.set()
        for w in list(self.workers):
            try:
                if w.isRunning():
                    w.wait(3000)
            except RuntimeError:
                pass
        self.timer.stop()
        self.runs_timer.stop()
        if hasattr(self.app, "forget_dialog"):
            try:
                self.app.forget_dialog(self)
            except Exception as _exc:
                _ignored(_exc)
        super().closeEvent(e)

    def showEvent(self, e):
        super().showEvent(e)
        self.cancel.clear()
        self.timer.start()
        self.runs_timer.start()


# =====================================================================================================================
class TrainWatch(QObject):
    """Image QC's side of the app-wide TrainingMonitor: a notification when a run ends (also one started before
    AIO QC was restarted). The monitor also frees what an ended run still holds and drives the menu-bar chip."""

    def __init__(self, app):
        super().__init__(app)                        # a child of the window: disconnected when it goes
        from .training_monitor import monitor
        self.app = app
        self.m = monitor()
        self.m.ended.connect(self._notify)

    def add(self, run):
        self.m.add(run)

    def running(self) -> list[Path]:
        return [Path(r) for r in self.m.runs() if T.state_of(T.status(r)) in T.RUNNING | {"queued"}]

    def tick(self):
        self.m.tick()

    def _notify(self, run: Path, state: str, st: dict):
        toast = getattr(self.app, "toast", None)
        if toast is None:
            return
        rows = T.epochs(run)
        best, b50 = T.best_scores(rows)
        val = st.get("mode") == "val"
        score = ""
        if val and st.get("metrics"):
            m = st["metrics"]
            score = (f" · top-1 {m[T.TOP1]:.3f}" if T.TOP1 in m else
                     f" · mAP50-95 {m.get('mask_map', m.get('box_map', 0)):.3f}")
        elif best is not None:
            score = f" · {T.score_word(T.is_cls_rows(rows))} {best:.3f}"
        word = {"finished": "finished", "stopped": "stopped", "paused": "paused", "failed": "failed",
                "interrupted": "was interrupted"}.get(state, state)
        what = "Validation" if val else "Training"
        kind = "success" if state == "finished" else "error" if state == "failed" else "warn"
        msg = f"<b>{what} {word}</b> — {run.name}{score}"
        if state == "failed":
            hint = T.explain((st.get("error") or "") + (st.get("error_tail") or ""))
            msg += f"<br>{st.get('error', '')}" + (f"<br>{hint}" if hint else "")
        actions = [("Show", lambda r=run: self.app.open_training(run=r))]
        if state == "finished" and not val and (run / "weights" / "best.pt").is_file() and \
                getattr(self.app, "session", None) is not None and hasattr(self.app, "open_improve"):
            actions.append(("Check it", lambda r=run: self.app.open_improve(run=r, check=True)))
        toast.show_message(msg, kind=kind, key=f"train-{run.name}", actions=actions)
