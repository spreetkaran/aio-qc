# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Manual frame extraction with timeline markers.

  M                 drop / remove a marker at the playhead (works while playing)
  drag on timeline  move a marker (the strip under the seekbar has a marker lane)
  Snap              move markers to the sharpest frame within +-N frames (fixes motion blur of live marking)
  Suggest           propose markers from the timeline analysis (scene starts, sharpest per window,
                    model-flagged samples, uniform) - shown hollow until extracted, delete what you don't want
  Extract pending   save marked frames (full resolution, optional ROI crop, LabelMe / YOLO pre-annotations)
  Auto-extract      every new marker is saved immediately in the background
  Send to batch     the same frames as a batch-queue row (parallel worker processes)
"""
import threading
from collections import deque

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QSize, pyqtSignal
from PyQt6.QtGui import QImage, QIcon, QColor
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QLabel, QToolButton, QPushButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QMenu,
)

from AIO_QC.tools.frame_extractor.core.markers import SUGGEST_MODES
from AIO_QC.tools.frame_extractor.utils import newIcon

GREEN, BLUE = QColor("#43A047"), QColor("#1E88E5")


class MarkerWorker(QThread):
    """Saves / snaps markers with its own decoder (never blocks playback)."""
    saved = pyqtSignal(int, str, QImage)
    snapped = pyqtSignal(int, int, float)
    failed = pyqtSignal(int, str)
    idle = pyqtSignal()

    def __init__(self, video_path, out_dir, backend="auto", out_cfg=None, model_cfg=None, preannot="none",
                 hash_db=None):
        super().__init__()
        self.out_cfg = out_cfg if isinstance(out_cfg, dict) else {"ext": out_cfg or ".jpg"}
        self.args = (video_path, out_dir, backend, self.out_cfg.get("ext", ".jpg"))
        self.model_cfg, self.preannot, self.hash_db = model_cfg, preannot, hash_db
        self._q = deque()
        self._cond = threading.Condition()
        self._quit = False

    def save(self, idx, roi):
        self._put(("save", idx, roi))

    def snap(self, idx, radius, roi):
        self._put(("snap", idx, radius, roi))

    def _put(self, cmd):
        with self._cond:
            if cmd not in self._q:
                self._q.append(cmd)
            self._cond.notify()

    def pending(self):
        with self._cond:
            return len(self._q)

    def stop(self):
        with self._cond:
            self._quit = True
            self._q.clear()
            self._cond.notify()
        self.wait()

    def run(self):
        from AIO_QC.tools.frame_extractor.core.markers import MarkerExtractor
        mx = None
        try:
            while True:
                with self._cond:
                    while not self._q and not self._quit:
                        self._cond.wait()
                    if self._quit:
                        return
                    cmd = self._q.popleft()
                if mx is None:
                    dets = None
                    c = self.model_cfg
                    if c and c.get("models") and self.preannot != "none":
                        try:
                            from AIO_QC.tools.frame_extractor.core.detector import load_detectors
                            from AIO_QC.tools.frame_extractor.widgets.models_dialog import detector_kwargs
                            dets = load_detectors(c["models"], c["conf"], **detector_kwargs(c))
                        except Exception as e:
                            self.failed.emit(-1, f"models not loaded, saving without pre-annotations: {e}")
                    video, out, backend, ext = self.args
                    oc = self.out_cfg
                    mx = MarkerExtractor(video, out, backend, ext, dets, self.preannot,
                                         (c or {}).get("conf", 0.25), self.hash_db, quality=oc.get("quality", 95),
                                         name_template=oc.get("template"), on_exists=oc.get("on_exists", "overwrite"))
                try:
                    if cmd[0] == "save":
                        path, img = mx.save(cmd[1], cmd[2])
                        self.saved.emit(cmd[1], path, _thumb(img))
                    else:
                        new, sharp = mx.snap(cmd[1], cmd[2], cmd[3])
                        self.snapped.emit(cmd[1], new, sharp)
                except Exception as e:
                    self.failed.emit(cmd[1], str(e))
                if not self.pending():
                    self.idle.emit()
        finally:
            if mx is not None:
                mx.close()


def _thumb(img, side=96):
    h, w = img.shape[:2]
    s = side / max(h, w)
    small = np.ascontiguousarray(cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA))
    return QImage(small.data, small.shape[1], small.shape[0], small.strides[0], QImage.Format.Format_BGR888).copy()


class MarkersDock(QDockWidget):
    jump_requested = pyqtSignal(int)
    remove_requested = pyqtSignal(list)
    snap_requested = pyqtSignal(list, int)          # frames ([] = all pending), radius
    suggest_requested = pyqtSignal(str, float)      # mode, every_s
    auto_pick_requested = pyqtSignal()
    extract_requested = pyqtSignal()
    batch_requested = pyqtSignal()
    review_requested = pyqtSignal()
    options_changed = pyqtSignal()

    def __init__(self, parent, toggle_action):
        super().__init__("Extract markers", parent)
        self.setObjectName("MarkersDock")
        self.time_text = lambda f: str(f)
        self.thumbs = {}
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 6, 6)

        row = QHBoxLayout()
        b = QToolButton()
        b.setDefaultAction(toggle_action)
        b.setAutoRaise(True)
        row.addWidget(b)
        snap = QToolButton()
        snap.setIcon(newIcon("snap"))
        snap.setText("Snap")
        snap.setToolTip("Move the selected (or all pending) markers to the sharpest frame within ±N frames")
        snap.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        snap.setAutoRaise(True)
        snap.clicked.connect(lambda: self.snap_requested.emit(self.selected_frames(), self.radius.value()))
        self.radius = QSpinBox()
        self.radius.setToolTip('Snap searches this many frames before and after each marker for the sharpest one (marks placed while playing often land on motion blur).')
        self.radius.setRange(1, 120)
        self.radius.setValue(8)
        self.radius.setPrefix("±")
        self.radius.setSuffix(" fr")
        row.addWidget(snap)
        row.addWidget(self.radius)
        sug = QToolButton()
        sug.setToolTip("<b>Suggest markers</b> from the timeline analysis: scene starts, sharpest frame per window, model-flagged samples, or evenly spaced. Suggestions are hollow until extracted - delete what you don't want.")
        sug.setIcon(newIcon("suggest"))
        sug.setText("Suggest")
        sug.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        sug.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        sug.setAutoRaise(True)
        m = QMenu(sug)
        for key, text in SUGGEST_MODES.items():
            a = m.addAction(text)
            a.triggered.connect(lambda _=False, k=key: self.suggest_requested.emit(k, self.every.value()))
        m.addSeparator()
        auto = m.addAction(newIcon("run"), "Auto-pick: sharpest per window → snap → extract")
        auto.triggered.connect(self.auto_pick_requested.emit)
        sug.setMenu(m)
        row.addWidget(sug)
        self.every = QDoubleSpinBox()
        self.every.setRange(0.2, 600)
        self.every.setValue(2.0)
        self.every.setSuffix(" s")
        self.every.setToolTip("Window / spacing for 'sharpest per window' and 'every N seconds'")
        row.addWidget(self.every)
        row.addStretch(1)
        lay.addLayout(row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Frame", "Time", "Sharp", "ROI", "Status"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setIconSize(QSize(64, 40))
        hh = self.table.horizontalHeader()
        for c in range(4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(lambda r, c: self.jump_requested.emit(self._frame(r)))
        lay.addWidget(self.table, 1)

        opts = QHBoxLayout()
        self.auto = QCheckBox("Auto-extract new markers")
        self.auto.setToolTip("Save each marker in the background as soon as it is placed")
        self.crop = QCheckBox("Crop to ROI")
        self.crop.setToolTip("Markers remember the ROI at the moment they are placed (tracked ROIs too)")
        self.preannot = QComboBox()
        self.preannot.addItems(["no labels", "labelme", "yolo"])
        self.preannot.setToolTip("Write the primary model's boxes next to each saved frame")
        for w in (self.auto, self.crop, self.preannot):
            opts.addWidget(w)
            (w.toggled if hasattr(w, "toggled") else w.currentIndexChanged).connect(lambda *_: self.options_changed.emit())
        opts.addStretch(1)
        lay.addLayout(opts)

        btns = QHBoxLayout()
        self.extract_btn = QPushButton("Extract pending")
        self.extract_btn.setToolTip('Save every pending marker (full resolution, ROI crop if set, label files if chosen) - Ctrl+E.')
        self.extract_btn.setIcon(newIcon("extract"))
        self.extract_btn.clicked.connect(self.extract_requested.emit)
        batch = QPushButton("Send to batch")
        batch.setIcon(newIcon("batch"))
        batch.setToolTip("Queue the marked frames as a batch job (worker processes, same outputs)")
        batch.clicked.connect(self.batch_requested.emit)
        rev = QPushButton("Review")
        rev.setToolTip('Open the extracted marker frames in the review gallery.')
        rev.setIcon(newIcon("gallery"))
        rev.clicked.connect(self.review_requested.emit)
        rm = QPushButton()
        rm.setIcon(newIcon("delete"))
        rm.setToolTip("Remove selected markers (Del). Saved files are kept.")
        rm.clicked.connect(lambda: self.remove_requested.emit(self.selected_frames()))
        for w in (self.extract_btn, batch, rev):
            btns.addWidget(w)
        btns.addStretch(1)
        btns.addWidget(rm)
        lay.addLayout(btns)
        self.status = QLabel()
        self.status.setStyleSheet("color: gray; font-size: 8pt;")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.setWidget(body)

    # ------------------------------------------------------------------
    def options(self):
        return dict(auto=self.auto.isChecked(), crop=self.crop.isChecked(),
                    preannot=["none", "labelme", "yolo"][self.preannot.currentIndex()])

    def set_options(self, o):
        for w in (self.auto, self.crop, self.preannot):
            w.blockSignals(True)
        self.auto.setChecked(bool(o.get("auto")))
        self.crop.setChecked(bool(o.get("crop", True)))
        self.preannot.setCurrentIndex(["none", "labelme", "yolo"].index(o.get("preannot", "none")))
        for w in (self.auto, self.crop, self.preannot):
            w.blockSignals(False)

    def _frame(self, r):
        return self.table.item(r, 0).data(Qt.ItemDataRole.UserRole)

    def selected_frames(self):
        return sorted({self._frame(i.row()) for i in self.table.selectedIndexes()})

    def set_markers(self, markers, sharp_of=lambda f: None, out_dir="", busy=0):
        sel = set(self.selected_frames())
        self.table.setRowCount(0)
        for f in sorted(markers):
            m = markers[f]
            r = self.table.rowCount()
            self.table.insertRow(r)
            it = QTableWidgetItem(str(f))
            it.setData(Qt.ItemDataRole.UserRole, f)
            if f in self.thumbs:
                it.setIcon(QIcon(self.thumbs[f]))
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QTableWidgetItem(self.time_text(f)))
            s = sharp_of(f)
            self.table.setItem(r, 2, QTableWidgetItem(f"{s:.0f}" if s is not None else ""))
            roi = m.get("roi")
            self.table.setItem(r, 3, QTableWidgetItem(f"{roi[2] - roi[0]}×{roi[3] - roi[1]}" if roi else "full"))
            if m.get("saved"):
                st, col = "saved", BLUE
            elif m.get("src") == "suggested":
                st, col = "suggested", None
            else:
                st, col = "pending", GREEN
            si = QTableWidgetItem(("● " if col else "○ ") + st)
            if col:
                si.setForeground(col)
            si.setToolTip(m.get("saved") or "")
            self.table.setItem(r, 4, si)
            if f in sel:
                self.table.selectRow(r)
        pend = sum(1 for m in markers.values() if not m.get("saved"))
        self.extract_btn.setText(f"Extract pending ({pend})")
        self.extract_btn.setEnabled(pend > 0)
        self.status.setText(f"{len(markers)} marker(s) · {pend} pending" + (f" · saving {busy}…" if busy else "")
                            + (f"<br>→ {out_dir}" if out_dir else ""))

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Delete:
            self.remove_requested.emit(self.selected_frames())
        else:
            super().keyPressEvent(ev)
