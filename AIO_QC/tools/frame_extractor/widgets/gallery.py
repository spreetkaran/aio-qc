# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Review gallery for extracted frames.

Browse a batch output folder (frames / blurry negatives / rejected), filter by
mining reason, preview with pre-annotations drawn, and reject bad frames before
they reach annotators. Rejecting *moves* an image together with its LabelMe
JSON / YOLO txt into `_rejected/` (mirroring the sub-folder), so every decision
can be reverted with Restore or Ctrl+Z. Rejected frames keep their entry in the
cross-video hash database, so they are not extracted again.
"""
import csv
import glob
import json
import os
import re
import shutil
import threading

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QSize, QRectF, QUrl, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon, QPainter, QColor, QPen, QFont, QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QComboBox, QLabel, QPushButton,
    QSplitter, QWidget, QSlider, QFileDialog, QAbstractItemView, QListView, QMessageBox,
)

from AIO_QC.tools.frame_extractor.core.common import IMG_EXTS
from AIO_QC.tools.frame_extractor.core.imageio import imread
REJECTED = "_rejected"
VIEWS = [("Frames", ""), ("Blurry negatives", "negatives_blurry"), ("Rejected", REJECTED)]
_num = re.compile(r"(\d+)")


def _natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in _num.split(s)]


def _sidecars(img_path):
    stem = os.path.splitext(img_path)[0]
    return [p for p in (stem + ".json", stem + ".txt") if os.path.exists(p)]


class ThumbLoader(QThread):
    loaded = pyqtSignal(str, QImage)

    def __init__(self, paths, size):
        super().__init__()
        self.paths, self.size = list(paths), size
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()
        self.wait()

    def run(self):
        for p in self.paths:
            if self._stop.is_set():
                return
            try:
                img = imread(p, cv2.IMREAD_REDUCED_COLOR_2)
            except Exception:
                img = None
            if img is None:
                continue
            h, w = img.shape[:2]
            s = self.size / max(h, w)
            if s < 1:
                img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            img = np.ascontiguousarray(img)
            q = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_BGR888).copy()
            self.loaded.emit(p, q)


class Preview(QWidget):
    """Large preview with LabelMe / YOLO boxes drawn on top."""

    def __init__(self):
        super().__init__()
        self.img = None
        self.boxes = []   # (x1, y1, x2, y2, label)
        self.setMinimumWidth(320)

    def set_image(self, path):
        self.img, self.boxes = None, []
        if path:
            arr = imread(path)
            if arr is not None:
                arr = np.ascontiguousarray(arr)
                self.img = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0],
                                  QImage.Format.Format_BGR888).copy()
                self.boxes = self._load_boxes(path, arr.shape[1], arr.shape[0])
        self.update()

    @staticmethod
    def _load_boxes(path, w, h):
        stem = os.path.splitext(path)[0]
        out = []
        if os.path.exists(stem + ".json"):
            try:
                data = json.load(open(stem + ".json", encoding="utf-8"))
                for sh in data.get("shapes", []):
                    pts = np.array(sh.get("points", []), float)
                    if len(pts):
                        conf = sh.get("description", "")
                        conf = conf.split()[0].replace("conf=", " ") if conf.startswith("conf=") else ""
                        out.append((*pts.min(0), *pts.max(0), sh.get("label", "") + conf))
            except Exception:
                pass
        elif os.path.exists(stem + ".txt"):
            names = []
            cls_file = os.path.join(os.path.dirname(path), "classes.txt")
            if os.path.exists(cls_file):
                names = [l.strip() for l in open(cls_file, encoding="utf-8") if l.strip()]
            for line in open(stem + ".txt", encoding="utf-8"):
                v = line.split()
                if len(v) >= 5:
                    c, cx, cy, bw, bh = int(float(v[0])), *map(float, v[1:5])
                    out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h,
                                names[c] if c < len(names) else str(c)))
        return out

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        if self.img is None:
            p.setPen(QColor(150, 150, 150))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a frame")
            return
        s = min(self.width() / self.img.width(), self.height() / self.img.height())
        w, h = self.img.width() * s, self.img.height() * s
        ox, oy = (self.width() - w) / 2, (self.height() - h) / 2
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(QRectF(ox, oy, w, h), self.img)
        f = QFont(self.font())
        f.setPointSize(8)
        p.setFont(f)
        for x1, y1, x2, y2, label in self.boxes:
            r = QRectF(ox + x1 * s, oy + y1 * s, (x2 - x1) * s, (y2 - y1) * s)
            p.setPen(QPen(QColor(122, 79, 214), 2))
            p.drawRect(r)
            tw = p.fontMetrics().horizontalAdvance(label) + 6
            p.fillRect(QRectF(r.left(), r.top() - 14, tw, 14), QColor(122, 79, 214))
            p.setPen(QColor(255, 255, 255))
            p.drawText(QRectF(r.left(), r.top() - 14, tw, 14), Qt.AlignmentFlag.AlignCenter, label)


class ReviewGallery(QDialog):
    def __init__(self, parent=None, folder=None):
        super().__init__(parent)
        self.setWindowTitle("Review extracted frames")
        from AIO_QC.tools.frame_extractor.utils.qt_utils import preferred_size
        preferred_size(self, 1180, 740)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.folder = None
        self.loader = None
        self.thumbs = {}
        self.reasons = {}
        self.undo = []           # list of [(src, dst), ...] moves per action
        self.thumb_size = 176

        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.folder_label = QLabel()
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        b = QPushButton("Open folder…")
        b.setToolTip('Open another folder of extracted frames.')
        b.clicked.connect(self.choose_folder)
        self.view_cb = QComboBox()
        self.view_cb.setToolTip('<b>Frames</b>: kept frames (sub-folders from the naming template included).<br><b>Blurry negatives</b>: the sampled blurry frames.<br><b>Rejected</b>: frames you rejected here - restore them any time.')
        self.view_cb.addItems([v[0] for v in VIEWS])
        self.view_cb.currentIndexChanged.connect(self.reload)
        self.reason_cb = QComboBox()
        self.reason_cb.setToolTip('Show only frames mined for this reason (model uncertainty, disagreement, flicker / blip, rare classes...).')
        self.reason_cb.currentIndexChanged.connect(self.reload)
        self.size_sl = QSlider(Qt.Orientation.Horizontal)
        self.size_sl.setToolTip('Thumbnail size.')
        self.size_sl.setRange(96, 320)
        self.size_sl.setValue(self.thumb_size)
        self.size_sl.setFixedWidth(120)
        self.size_sl.valueChanged.connect(self._resize_icons)
        bar.addWidget(b)
        bar.addWidget(self.folder_label, 1)
        for w in (QLabel("Show:"), self.view_cb, QLabel("Reason:"), self.reason_cb, QLabel("Size:"), self.size_sl):
            bar.addWidget(w)
        lay.addLayout(bar)

        split = QSplitter()
        self.list = QListWidget()
        self.list.setViewMode(QListView.ViewMode.IconMode)
        self.list.setResizeMode(QListView.ResizeMode.Adjust)
        self.list.setMovement(QListView.Movement.Static)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setUniformItemSizes(True)
        self.list.setSpacing(4)
        self.list.currentItemChanged.connect(self._on_current)
        self.list.itemSelectionChanged.connect(self._update_status)
        self.list.itemActivated.connect(lambda it: QDesktopServices.openUrl(
            QUrl.fromLocalFile(it.data(Qt.ItemDataRole.UserRole))))
        split.addWidget(self.list)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.preview = Preview()
        rl.addWidget(self.preview, 1)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rl.addWidget(self.info)
        split.addWidget(right)
        split.setSizes([760, 480])
        lay.addWidget(split, 1)

        row = QHBoxLayout()
        self.status = QLabel()
        row.addWidget(self.status, 1)
        self.reject_btn = QPushButton("Reject selected  (Del)")
        self.reject_btn.setToolTip('Move the selected frames (and their LabelMe / YOLO files) into _rejected/. Nothing is deleted: Restore or Ctrl+Z undo it.')
        self.reject_btn.clicked.connect(self.reject_selected)
        self.restore_btn = QPushButton("Restore selected  (R)")
        self.restore_btn.setToolTip('Move the selected rejected frames (and their label files) back to where they came from.')
        self.restore_btn.clicked.connect(self.restore_selected)
        undo_btn = QPushButton("Undo  (Ctrl+Z)")
        undo_btn.setToolTip('Undo the last reject / restore.')
        undo_btn.clicked.connect(self.undo_last)
        exp = QPushButton("Show in folder")
        exp.setToolTip('Open this folder in the file manager.')
        exp.clicked.connect(lambda: self.folder and QDesktopServices.openUrl(QUrl.fromLocalFile(self._view_dir())))
        qc = QPushButton("Open in Image QC")
        qc.setToolTip("Open this folder (frames + any pre-labels) as the dataset in AIO QC's Image QC workspace")
        qc.clicked.connect(self.open_in_qc)
        for w in (self.reject_btn, self.restore_btn, undo_btn, exp, qc):
            row.addWidget(w)
        lay.addLayout(row)

        for key, slot in (("Del", self.reject_selected), ("Backspace", self.reject_selected),
                          ("R", self.restore_selected), ("Ctrl+Z", self.undo_last)):
            QShortcut(QKeySequence(key), self, activated=slot)
        if folder:
            self.set_folder(folder)

    # ---------------------------------------------------------------- data
    def open_in_qc(self):
        from PyQt6.QtWidgets import QApplication
        shell = getattr(QApplication.instance(), "shell", None)
        if shell is None or not self.folder:
            return
        if shell.open_in_qc(self.folder):
            self.close()

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with extracted frames", self.folder or "")
        if d:
            self.set_folder(d)

    def set_folder(self, folder):
        self.folder = folder
        self.folder_label.setText(folder)
        self.undo.clear()
        self._load_reasons()
        self.reload()

    def _load_reasons(self):
        self.reasons = {}
        for f in glob.glob(os.path.join(self.folder, "mining_*.csv")):
            try:
                for row in csv.DictReader(open(f, encoding="utf-8")):
                    if row.get("reasons"):
                        self.reasons[row["file"]] = row["reasons"].split("|")
            except Exception:
                pass
        found = sorted({r for rs in self.reasons.values() for r in rs})
        self.reason_cb.blockSignals(True)
        self.reason_cb.clear()
        self.reason_cb.addItem("All")
        self.reason_cb.addItems(found)
        self.reason_cb.setEnabled(bool(found))
        self.reason_cb.blockSignals(False)

    def _reasons_of(self, path):
        rel = os.path.relpath(path, self.folder).replace(os.sep, "/")
        return self.reasons.get(rel) or self.reasons.get(os.path.basename(path), [])

    def _view_dir(self):
        return os.path.join(self.folder, VIEWS[self.view_cb.currentIndex()][1])

    def _files(self):
        d = self._view_dir()
        if not os.path.isdir(d):
            return []
        files = []
        top = VIEWS[self.view_cb.currentIndex()][1] == ""
        for r, dirs, fs in os.walk(d):
            if top:   # naming templates may create sub-folders; skip the special ones
                dirs[:] = [x for x in dirs if x not in ("negatives_blurry", REJECTED, ".fx_resume")]
            files += [os.path.join(r, f) for f in fs if f.lower().endswith(IMG_EXTS)]
        reason = self.reason_cb.currentText()
        if reason and reason != "All":
            files = [f for f in files if reason in self._reasons_of(f)]
        return sorted(files, key=lambda p: _natural(os.path.basename(p)))

    def reload(self):
        if not self.folder:
            return
        if self.loader:
            self.loader.stop()
        files = self._files()
        self.list.clear()
        self.list.setIconSize(QSize(self.thumb_size, self.thumb_size))
        self.list.setGridSize(QSize(self.thumb_size + 16, self.thumb_size + 34))
        placeholder = QPixmap(self.thumb_size, int(self.thumb_size * 0.56))
        placeholder.fill(QColor(60, 60, 60))
        for f in files:
            it = QListWidgetItem(os.path.relpath(f, self._view_dir()).replace(os.sep, "/"))
            it.setData(Qt.ItemDataRole.UserRole, f)
            px = self.thumbs.get((f, self.thumb_size))
            it.setIcon(QIcon(px or placeholder))
            rs = self._reasons_of(f)
            if rs:
                it.setToolTip("mined: " + ", ".join(rs))
            self.list.addItem(it)
        todo = [f for f in files if (f, self.thumb_size) not in self.thumbs]
        self.loader = ThumbLoader(todo, self.thumb_size)
        self.loader.loaded.connect(self._on_thumb)
        self.loader.start(QThread.Priority.LowPriority)
        rejected_view = VIEWS[self.view_cb.currentIndex()][1] == REJECTED
        self.reject_btn.setEnabled(not rejected_view)
        self.restore_btn.setEnabled(rejected_view)
        self.preview.set_image(None)
        self.info.clear()
        self._update_status()

    def _on_thumb(self, path, img):
        px = QPixmap.fromImage(img)
        self.thumbs[(path, self.thumb_size)] = px
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == path:
                it.setIcon(QIcon(px))
                break

    def _resize_icons(self, v):
        self.thumb_size = v
        self.reload()

    def _on_current(self, cur, prev):
        path = cur.data(Qt.ItemDataRole.UserRole) if cur else None
        self.preview.set_image(path)
        if path:
            h, w = (self.preview.img.height(), self.preview.img.width()) if self.preview.img else (0, 0)
            rs = self._reasons_of(path)
            side = ", ".join(os.path.basename(s) for s in _sidecars(path))
            self.info.setText(f"<b>{os.path.basename(path)}</b> · {w}×{h} · {len(self.preview.boxes)} box(es)"
                              + (f"<br>mined: {', '.join(rs)}" if rs else "")
                              + (f"<br>labels: {side}" if side else ""))

    def _update_status(self):
        n, sel = self.list.count(), len(self.list.selectedItems())
        rej = os.path.join(self.folder or "", REJECTED)
        nrej = sum(1 for _, _, fs in os.walk(rej) for f in fs if f.lower().endswith(IMG_EXTS)) \
            if os.path.isdir(rej) else 0
        self.status.setText(f"{n} frame(s) shown · {sel} selected · {nrej} rejected")

    # ------------------------------------------------------------- actions
    def _move(self, pairs):
        done = []
        for src, dst in pairs:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                base, ext = os.path.splitext(dst)
                dst = f"{base}_dup{ext}"
            shutil.move(src, dst)
            done.append((src, dst))
        return done

    def _selected_paths(self):
        return [it.data(Qt.ItemDataRole.UserRole) for it in self.list.selectedItems()]

    def reject_selected(self):
        if not self.folder or VIEWS[self.view_cb.currentIndex()][1] == REJECTED:
            return
        pairs = []
        for p in self._selected_paths():
            rel = os.path.relpath(p, self.folder)
            for f in [p] + _sidecars(p):
                pairs.append((f, os.path.join(self.folder, REJECTED, os.path.dirname(rel), os.path.basename(f))))
        if pairs:
            row = self.list.currentRow()
            self.undo.append(self._move(pairs))
            self.reload()
            if self.list.count():
                self.list.setCurrentRow(min(row, self.list.count() - 1))

    def restore_selected(self):
        if not self.folder or VIEWS[self.view_cb.currentIndex()][1] != REJECTED:
            return
        pairs = []
        rej = os.path.join(self.folder, REJECTED)
        for p in self._selected_paths():
            rel = os.path.relpath(p, rej)
            for f in [p] + _sidecars(p):
                pairs.append((f, os.path.join(self.folder, os.path.dirname(rel), os.path.basename(f))))
        if pairs:
            self.undo.append(self._move(pairs))
            self.reload()

    def undo_last(self):
        if not self.undo:
            return
        moves = self.undo.pop()
        try:
            self._move([(dst, src) for src, dst in reversed(moves)])
        except OSError as e:
            QMessageBox.warning(self, "Undo", f"Could not undo: {e}")
        self.reload()

    def closeEvent(self, ev):
        if self.loader:
            self.loader.stop()
        super().closeEvent(ev)
