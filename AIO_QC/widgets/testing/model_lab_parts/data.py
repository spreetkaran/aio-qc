# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Testing (Model Lab) — step 1, the test data.

A mixin of AIO_QC.widgets.testing.model_lab.ModelLab: the methods use the window's own attributes, set up in its __init__.
"""
import os
from html import escape
import numpy as np
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QTableWidgetItem, QAbstractItemView, QComboBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QRadioButton, QButtonGroup, QGroupBox, QLineEdit, QFileDialog, QMessageBox, QMenu,
)
from AIO_QC.widgets import motion as anim
from AIO_QC.widgets.testing.lab_common import (
    Job,
    human_time,
    open_path,
    bgr_pixmap,
)
from AIO_QC.tools.frame_extractor.core.common import VIDEO_FILTER, parse_range
from AIO_QC.tools.frame_extractor.core.imageio import imread

from AIO_QC.widgets.testing.model_lab_common import (ImagePreview, _hint, _item, _make_table, _scroll, _swatch_icon, _unique_tip, slot_hex)


class DataMixin:

    # ------------------------------------------------------------------ 1 data
    def _page_data(self):
        w, lay = self._page("Test data", step=0, subtitle="The images every model is measured on. Build a reusable test set from "
                            "videos and image folders (unique frames only), or use a labelled dataset you already "
                            "have. Labels are optional: without them, models are compared with each other.")
        kind = QHBoxLayout()
        self.kind_qc = QRadioButton("This dataset (Image QC)")
        self.kind_qc.setToolTip("The dataset open in Image QC: its files, labels and classes. Label fixes made here "
                                "go through AIO QC (backed up, in the file's history, visible in Image QC at once).")
        self.kind_ts = QRadioButton("Test set from videos")
        self.kind_ds = QRadioButton("Labelled dataset (YOLO / LabelMe folder or data.yaml)")
        self.kind_none = QRadioButton("Videos only")
        self.kind_ts.setToolTip("A folder AIO QC manages: frames, labels, cache and reports. Opens in any "
                                "YOLO tool (data.yaml). Re-use it to measure next month's model on the same frames.")
        self.kind_ds.setToolTip("An existing dataset: YOLO txt labels (images/ + labels/), LabelMe json next to the "
                                "images, or a YOLO data.yaml (val split).")
        self.kind_none.setToolTip("Skip accuracy on images: analyse whole videos only (stability, agreement).")
        g = QButtonGroup(self)
        for b in (self.kind_qc, self.kind_ts, self.kind_ds, self.kind_none):
            g.addButton(b)
            kind.addWidget(b)
        kind.addStretch(1)
        lay.addLayout(kind)
        self.kind_ts.setChecked(True)
        for b in (self.kind_qc, self.kind_ts, self.kind_ds, self.kind_none):
            b.toggled.connect(self._kind_changed)

        # --- the dataset open in Image QC
        self.qc_box = QGroupBox("Dataset open in Image QC")
        ql = QHBoxLayout(self.qc_box)
        self.qc_lbl = QLabel("")
        self.qc_lbl.setWordWrap(True)
        ql.addWidget(self.qc_lbl, 1)
        b = QPushButton("Use it")
        b.setToolTip("Load the open dataset (again) as the test data")
        b.clicked.connect(lambda: self._load_qc())
        ql.addWidget(b)
        lay.addWidget(self.qc_box)

        # --- test set
        self.ts_box = QGroupBox("Test set")
        tl = QVBoxLayout(self.ts_box)
        row = QHBoxLayout()
        self.ts_path = QLineEdit()
        self.ts_path.setReadOnly(True)
        self.ts_path.setPlaceholderText("Create a new test set or open one")
        b_new = QPushButton("New…")
        b_new.clicked.connect(self._new_testset)
        b_open = QPushButton("Open…")
        b_open.clicked.connect(self._open_testset)
        row.addWidget(self.ts_path, 1)
        row.addWidget(b_new)
        row.addWidget(b_open)
        tl.addLayout(row)
        self.src_table = _make_table(["Type", "Source", "Range"], stretch_col=1)
        self.src_table.setMinimumHeight(110)
        self.src_table.setToolTip("Sources to add to the test set. Double-click Range to limit a video "
                                  "(e.g. 0:30-5:00).")
        self.src_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        tl.addWidget(self.src_table)
        row = QHBoxLayout()
        for text, fn in (("Add videos…", self._add_src_videos), ("Add image folder…", self._add_src_folder),
                         ("Remove", self._remove_src)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        tl.addLayout(row)
        sam = QHBoxLayout()
        self.sampling = QComboBox()
        for label, key in (("A frame every", "interval"), ("Every Nth frame", "every_n"), ("N frames per video", "count"),
                           ("All frames", "all")):
            self.sampling.addItem(label, key)
        self.sampling.setToolTip("How frames are taken from each video before duplicates are removed.")
        self.samp_val = QDoubleSpinBox()
        self.samp_val.setRange(0.04, 100000)
        self.samp_val.setDecimals(2)
        self.samp_val.setValue(1.0)
        self.sampling.currentIndexChanged.connect(self._sampling_changed)
        self.max_per = QSpinBox()
        self.max_per.setRange(0, 1000000)
        self.max_per.setSpecialValueText("no limit")
        self.max_per.setToolTip("At most this many frames from one video (0 = no limit) - keeps one long video "
                                "from dominating the test set.")
        self.unique = QComboBox()
        for k in ("off", "exact", "conservative", "balanced", "aggressive"):
            self.unique.addItem(k.capitalize(), k)
            self.unique.setItemData(self.unique.count() - 1, _unique_tip(k), Qt.ItemDataRole.ToolTipRole)
        self.unique.setCurrentIndex(3)
        self.unique.setToolTip("Unique frames: duplicates are removed inside each video AND across all videos and "
                               "folders in the set (a scene repeating an hour later, or in another video, is kept "
                               "once). Exposure changes and sensor noise are ignored.")
        self.min_sharp = QDoubleSpinBox()
        self.min_sharp.setRange(0, 100000)
        self.min_sharp.setDecimals(0)
        self.min_sharp.setSpecialValueText("off")
        self.min_sharp.setToolTip("Skip blurry frames below this sharpness (Laplacian variance; the main window's "
                                  "status bar shows the current frame's value). 0 = keep all.")
        self.copy_folders = QCheckBox("Copy folder images")
        self.copy_folders.setToolTip("Copy images from folders into the test set (else they are referenced in place, "
                                     "and their labels are read from where they are).")
        for lab, wdg in (("Sampling", self.sampling), (None, self.samp_val), ("Max per video", self.max_per),
                         ("Unique frames", self.unique), ("Min sharpness", self.min_sharp)):
            if lab:
                sam.addWidget(QLabel(lab))
            sam.addWidget(wdg)
        sam.addWidget(self.copy_folders)
        sam.addStretch(1)
        tl.addLayout(sam)
        row = QHBoxLayout()
        self.build_btn = QPushButton("Add frames to the test set")
        self.build_btn.setToolTip("Sample, drop duplicates and blurry frames, save. Interrupting keeps what was added.")
        self.build_btn.clicked.connect(self._build_testset)
        self.build_bar = anim.SmoothProgressBar()
        self.build_bar.setRange(0, 1000)
        self.build_bar.setVisible(False)
        self.build_lbl = QLabel("")
        row.addWidget(self.build_btn)
        row.addWidget(self.build_bar, 1)
        row.addWidget(self.build_lbl, 2)
        tl.addLayout(row)
        lay.addWidget(self.ts_box)

        # --- dataset
        self.ds_box = QGroupBox("Labelled dataset")
        dl = QHBoxLayout(self.ds_box)
        self.ds_path = QLineEdit()
        self.ds_path.setPlaceholderText("Folder with images + YOLO / LabelMe labels, class folders (classifiers), "
                                        "or a data.yaml")
        self.ds_path.editingFinished.connect(lambda: self._load_data(self.ds_path.text().strip()))
        self.ds_fmt = QComboBox()
        for label, k in (("Auto-detect", "auto"), ("YOLO txt", "yolo"), ("LabelMe json", "labelme"),
                         ("Classification (class folders / LabelMe flags)", "classify")):
            self.ds_fmt.addItem(label, k)
        self.ds_fmt.setToolTip("Auto-detect also recognises classification test sets: one folder per class "
                               "(optionally inside train / val / test) or LabelMe files whose flags name the class.")
        b1 = QPushButton("Folder…")
        b1.clicked.connect(self._browse_ds_folder)
        b2 = QPushButton("data.yaml…")
        b2.clicked.connect(self._browse_ds_yaml)
        dl.addWidget(self.ds_path, 1)
        dl.addWidget(self.ds_fmt)
        dl.addWidget(b1)
        dl.addWidget(b2)
        lay.addWidget(self.ds_box)

        # --- summary + labels
        self.sum_box = QGroupBox("What is in it")
        sl = QVBoxLayout(self.sum_box)
        self.sum_lbl = QLabel("No test data yet.")
        self.sum_lbl.setWordWrap(True)
        self.sum_lbl.setTextFormat(Qt.TextFormat.RichText)
        sl.addWidget(self.sum_lbl)
        self.preview = QListWidget()
        self.preview.setViewMode(QListWidget.ViewMode.IconMode)
        self.preview.setFlow(QListWidget.Flow.LeftToRight)
        self.preview.setWrapping(False)
        self.preview.setIconSize(QSize(144, 81))
        self.preview.setFixedHeight(112)
        self.preview.setMovement(QListWidget.Movement.Static)
        self.preview.setToolTip("A sample of the test images (evenly spread). Double-click to view.")
        self.preview.itemDoubleClicked.connect(self._preview_open)
        sl.addWidget(self.preview)
        row = QHBoxLayout()
        self.prelabel_btn = QPushButton("Pre-label with a model…")
        self.prelabel_btn.setToolTip("Write labels from a model for images that have none - then fix them in any "
                                     "labelling tool. They count as ground truth only after you have edited them "
                                     "(or marked them reviewed).")
        self.prelabel_btn.clicked.connect(self._prelabel)
        self.reviewed_btn = QPushButton("Mark pre-labels reviewed")
        self.reviewed_btn.setToolTip("Accept every pre-label as ground truth (only after checking them!).")
        self.reviewed_btn.clicked.connect(self._mark_reviewed)
        self.folder_btn = QPushButton("Open folder")
        self.folder_btn.setToolTip("Open the images folder - label it with LabelImg, X-AnyLabeling, CVAT, Label Studio…")
        self.folder_btn.clicked.connect(lambda: self.data and open_path(self.testset.images_dir if self.testset
                                                                        else self.data.root))
        self.reload_btn = QPushButton("Reload labels")
        self.reload_btn.clicked.connect(lambda: self._reload_data())
        for b in (self.prelabel_btn, self.reviewed_btn, self.folder_btn, self.reload_btn):
            row.addWidget(b)
        row.addStretch(1)
        sl.addLayout(row)
        lay.addWidget(self.sum_box)

        # --- videos
        vb = QGroupBox("Videos for behaviour analysis (optional - no labels needed)")
        vl = QVBoxLayout(vb)
        vl.addWidget(_hint("Each model runs on the whole video (a frame every N seconds): detections over time, "
                           "flicker and one-frame blips, agreement between models, and the moments they disagree."))
        self.vid_list = QListWidget()
        self.vid_list.setMaximumHeight(90)
        vl.addWidget(self.vid_list)
        row = QHBoxLayout()
        b = QPushButton("Add videos…")
        b.clicked.connect(self._add_analysis_videos)
        row.addWidget(b)
        b = QPushButton("Same as test set videos")
        b.setToolTip("Analyse the videos the test set was built from.")
        b.clicked.connect(self._videos_from_testset)
        row.addWidget(b)
        b = QPushButton("Remove")
        b.clicked.connect(lambda: ([self.vid_list.takeItem(self.vid_list.row(i)) for i in self.vid_list.selectedItems()],
                                   self._changed()))
        row.addWidget(b)
        row.addStretch(1)
        row.addWidget(QLabel("A frame every"))
        self.vid_every = QDoubleSpinBox()
        self.vid_every.setRange(0.03, 60)
        self.vid_every.setSingleStep(0.1)
        self.vid_every.setValue(0.2)
        self.vid_every.setSuffix(" s")
        self.vid_every.setToolTip("Sampling step. Flicker is measured between consecutive samples: 0.1-0.5 s works "
                                  "well; every frame is the most exact but slowest.")
        row.addWidget(self.vid_every)
        vl.addLayout(row)
        lay.addWidget(vb)
        lay.addStretch(1)
        nxt = QPushButton("Next: models  →")
        nxt.clicked.connect(lambda: self.nav.setCurrentRow(1))
        lay.addWidget(nxt, 0, Qt.AlignmentFlag.AlignRight)
        self._kind_changed()
        return _scroll(w)

    def _kind_changed(self, *_):
        self.qc_box.setVisible(self.kind_qc.isChecked())
        if self.kind_qc.isChecked():
            why = self.qc.unavailable() if self.qc is not None else "Testing was opened without Image QC."
            s = self.qc.session if self.qc is not None else None
            self.qc_lbl.setText(why or f"<b>{s.root.name}</b> · {s.type.title} · {len(s.samples):,} files")
            if not why and not self._loading and not (self.qc.is_this(self.data)) and self.job is None:
                self._load_qc()
        self.ts_box.setVisible(self.kind_ts.isChecked())
        self.ds_box.setVisible(self.kind_ds.isChecked())
        self.sum_box.setVisible(not self.kind_none.isChecked())
        self._changed()

    def _sampling_changed(self, *_):
        k = self.sampling.currentData()
        self.samp_val.setVisible(k != "all")
        if k == "interval":
            self.samp_val.setDecimals(2)
            self.samp_val.setSuffix(" s")
            self.samp_val.setValue(float(self.st.value("interval_s", 1.0)))
        elif k == "every_n":
            self.samp_val.setDecimals(0)
            self.samp_val.setSuffix(" frames")
            self.samp_val.setValue(30)
        elif k == "count":
            self.samp_val.setDecimals(0)
            self.samp_val.setSuffix(" frames")
            self.samp_val.setValue(100)

    def _ts_settings(self, save=True):
        k = self.sampling.currentData()
        s = dict(sampling=k, max_per_video=self.max_per.value(), unique=self.unique.currentData(),
                 min_sharpness=self.min_sharp.value(), copy_folders=self.copy_folders.isChecked())
        if k == "interval":
            s["interval_s"] = self.samp_val.value()
            if save:
                self.st.setValue("interval_s", s["interval_s"])
        elif k == "every_n":
            s["every_n"] = int(self.samp_val.value())
        elif k == "count":
            s["count"] = int(self.samp_val.value())
        return s

    def _new_testset(self):
        from AIO_QC.utils.save_dirs import ask_dir
        d = ask_dir(self, "Folder for the new test set (an empty folder)", kind="datasets",
                    fallback=self.st.value("last_dir", ""))
        if not d:
            return
        from AIO_QC.core.testing.testset import TestSet
        if TestSet.is_testset(d):
            self._load_data(d)
            return
        if os.listdir(d) and QMessageBox.question(
                self, "Testing", "This folder is not empty. Create the test set inside it anyway "
                "(images/, labels/, testset.json are added)?") != QMessageBox.StandardButton.Yes:
            return
        TestSet.create(d, settings=self._ts_settings())
        self.st.setValue("last_dir", os.path.dirname(d))
        self._load_data(d)

    def _open_testset(self):
        d = QFileDialog.getExistingDirectory(self, "Open a test set folder", self.st.value("last_dir", ""))
        if d:
            from AIO_QC.core.testing.testset import TestSet
            if not TestSet.is_testset(d):
                QMessageBox.information(self, "Testing", "That folder is not a test set (no testset.json). To use "
                                        "an existing labelled dataset, pick 'Labelled dataset'.")
                return
            self.st.setValue("last_dir", os.path.dirname(d))
            self._load_data(d)

    def _add_src_rows(self, rows):
        for typ, path in rows:
            r = self.src_table.rowCount()
            self.src_table.insertRow(r)
            t = _item("Video" if typ == "video" else "Folder")
            t.setData(Qt.ItemDataRole.UserRole, typ)
            self.src_table.setItem(r, 0, t)
            p = _item(path, tip=path)
            self.src_table.setItem(r, 1, p)
            rng = QTableWidgetItem("whole video" if typ == "video" else "–")
            if typ != "video":
                rng.setFlags(rng.flags() & ~Qt.ItemFlag.ItemIsEditable)
            rng.setToolTip("Optional time range, e.g. 1:00-4:30 or 60-270 (seconds)")
            self.src_table.setItem(r, 2, rng)
        self._changed()

    def _add_src_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Videos for the test set", self.st.value("video_dir", ""),
                                                VIDEO_FILTER)
        if paths:
            self.st.setValue("video_dir", os.path.dirname(paths[0]))
        self._add_src_rows([("video", p) for p in paths])

    def _add_src_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Image folder (labels are picked up when present)",
                                             self.st.value("last_dir", ""))
        if d:
            self._add_src_rows([("folder", d)])

    def _remove_src(self):
        for r in sorted({i.row() for i in self.src_table.selectedIndexes()}, reverse=True):
            self.src_table.removeRow(r)
        self._changed()

    def _build_testset(self):
        if self.job is not None:
            if self.build_btn.text() == "Stop":          # our own build: this click stops it
                self.job.cancel()
            else:
                self.status.showMessage("Busy — wait for the current task to finish", 4000)
            return
        if self.testset is None:
            self._new_testset()
            if self.testset is None:
                return
        sources = []
        for r in range(self.src_table.rowCount()):
            typ = self.src_table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            path = self.src_table.item(r, 1).text()
            src = dict(type=typ, path=path)
            if typ == "video":
                try:
                    a, b = parse_range(self.src_table.item(r, 2).text())
                except ValueError:
                    QMessageBox.warning(self, "Testing", f"Could not read the range for {os.path.basename(path)}.")
                    return
                if a is not None:
                    src["start"] = a
                if b is not None:
                    src["end"] = b
            sources.append(src)
        if not sources:
            QMessageBox.information(self, "Testing", "Add videos or image folders first.")
            return
        ts, settings = self.testset, self._ts_settings()

        def work(job):
            return ts.add_sources(sources, settings, progress=lambda d, t, txt="": job.progress.emit((d, t, txt)),
                                  cancel=job.cancelled)

        def done(stats):
            msg = (f"+{stats['added']:,} images · {stats['duplicates']:,} duplicates and {stats['blurry']:,} blurry "
                   f"skipped · {stats['seconds']:.1f} s" + (" · stopped" if stats.get("cancelled") else ""))
            if stats.get("failed"):
                msg += f" · {stats['failed']} source(s) failed"
            self.build_lbl.setText(msg)
            self.src_table.setRowCount(0)
            self._load_data(ts.root)
        self._start(work, done, bar=self.build_bar, button=self.build_btn, busy="Stop")

    def _browse_ds_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Dataset folder", self.st.value("last_dir", ""))
        if d:
            self.ds_path.setText(d)
            self._load_data(d)

    def _browse_ds_yaml(self):
        p, _ = QFileDialog.getOpenFileName(self, "YOLO data.yaml", self.st.value("last_dir", ""),
                                           "YAML (*.yaml *.yml)")
        if p:
            self.ds_path.setText(p)
            self._load_data(p)

    def _reload_data(self):
        if self.data is not None:
            self._load_data(self.testset.root if self.testset else self.ds_path.text().strip())

    def _load_qc(self, then=None):
        """The dataset open in Image QC as test data (read on the UI thread, loaded on a worker)."""
        if self.qc is None or self.job is not None:
            return
        snap = self.qc.snapshot()
        if snap is None:
            self.sum_lbl.setText(self.qc.unavailable() or "No dataset open in Image QC.")
            return
        from AIO_QC.core.testing import pipeline as P
        self.sum_lbl.setText("Reading labels…")

        def work(job):
            return P.open_qc(snap["root"], snap["fmt"], snap["pairs"], snap["names"], snap["label"],
                             progress=lambda d, t, txt="": job.progress.emit((d, t, txt)), cancel=job.cancelled,
                             cls_type=snap.get("cls_type"))

        def done(data):
            self.data, self.testset = data, None
            self.cache = P.cache_for(data)
            self.run = None
            self.kind_qc.blockSignals(True)
            self.kind_qc.setChecked(True)
            self.kind_qc.blockSignals(False)
            self.qc_box.setVisible(True)
            self.ts_box.setVisible(False)
            self.ds_box.setVisible(False)
            self._show_summary()
            self._load_preview()
            self._changed()
            if then is not None:
                then()
        self._start(work, done, bar=self.build_bar, failed=lambda m: self.sum_lbl.setText(f"Could not read it: {m}"))

    def _load_data(self, path, then=None):
        if not path or self.job is not None:
            return
        from AIO_QC.core.testing import pipeline as P
        fmt = self.ds_fmt.currentData() if self.kind_ds.isChecked() else "auto"
        self.sum_lbl.setText("Reading labels…")

        def work(job):
            return P.open_data(path, fmt, progress=lambda d, t, txt="": job.progress.emit((d, t, txt)),
                               cancel=job.cancelled)

        def done(data):
            self.data, self.testset = data, data.testset
            self.cache = P.cache_for(data)
            self.run = None
            if data.testset:
                self.kind_ts.setChecked(True)
                self.ts_path.setText(data.testset.root)
                self.st.setValue("testset", data.testset.root)
            else:
                self.kind_ds.setChecked(True)
                self.ds_path.setText(path)
                self.st.setValue("dataset", path)
            self._show_summary()
            self._load_preview()
            self._changed()
            if then is not None:
                then()
        self._start(work, done, bar=self.build_bar, failed=lambda m: self.sum_lbl.setText(f"Could not read it: {m}"))

    def _show_summary(self):
        d = self.data
        if d is None:
            self.sum_lbl.setText("No test data yet.")
            return
        ds = d.dataset
        n = len(ds.items)
        lab = sum(1 for it in ds.items if it.labelled)
        unrev = sum(1 for it in ds.items if it.labelled and not it.reviewed)
        cls_set = getattr(ds, "task", "") == "classify"
        vids = len({it.meta.get("video") for it in ds.items if it.meta.get("video")})
        counts = {}
        if cls_set:                                  # a classifier's test set: images per class
            objs = 0
            for it in ds.items:
                for t in it.tags or []:
                    counts[t] = counts.get(t, 0) + 1
            name_of = str
        else:
            objs = sum(len(it.cls) for it in ds.items if it.labelled and it.reviewed and it.cls is not None)
            for it in ds.items:
                if it.labelled and it.reviewed and it.cls is not None:
                    for c in np.asarray(it.cls, int):
                        counts[int(c)] = counts.get(int(c), 0) + 1
            name_of = lambda c: ds.names.get(c, c)                        # noqa: E731
        chips = " ".join(f"<span style='white-space:nowrap'>{escape(str(name_of(c)))} <b>{counts[c]:,}</b></span> ·"
                         for c in sorted(counts, key=lambda c: -counts[c])[:24]).rstrip(" ·")
        if lab - unrev == 0:
            state = ("<b>No images with a class</b> - classifiers will be compared with each other (pick a reference "
                     "model in Run)." if cls_set else
                     "<b>No reviewed labels</b> - models will be compared with each other (pick a reference model "
                     "in Run), or pre-label, fix and mark reviewed to measure true accuracy.")
        elif cls_set:
            state = (f"Classification test set · <b>{lab:,}</b> images with a class, scored by top-1 / top-5 "
                     "accuracy and per-class F1")
        else:
            state = f"<b>{lab - unrev:,}</b> labelled images, <b>{objs:,}</b> objects"
        if unrev:
            state += f" · <b>{unrev:,}</b> with unreviewed pre-labels (not scored)"
        warn = ("<br><span style='color:#c98500'>" + "<br>".join(ds.warnings[:4]) + "</span>") if ds.warnings else ""
        self.sum_lbl.setText(f"<b>{n:,}</b> images" + (f" from <b>{vids}</b> video(s)" if vids else "") +
                             f" · {len(ds.names)} classes<br>{state}" + (f"<br>{chips}" if chips else "") + warn)
        is_ts = self.testset is not None
        self.prelabel_btn.setEnabled(is_ts)
        self.reviewed_btn.setEnabled(is_ts and unrev > 0)
        self.status.showMessage(f"{d.label}: {n:,} images", 5000)

    def _load_preview(self):
        self.preview.clear()
        d = self.data
        if d is None or not d.dataset.items:
            return
        items = d.dataset.items
        pick = sorted(set(np.linspace(0, len(items) - 1, min(40, len(items))).astype(int).tolist()))
        paths = [items[i].path for i in pick]

        def work(job):
            import cv2
            out = []
            for p in paths:
                if job.cancelled():
                    break
                img = imread(p, cv2.IMREAD_REDUCED_COLOR_4)
                if img is None:
                    img = imread(p)
                out.append((p, img))
            return out

        def done(res):
            for p, img in res:
                it = QListWidgetItem(QIcon(bgr_pixmap(img, 288, 162)), "")
                it.setToolTip(p)
                it.setData(Qt.ItemDataRole.UserRole, p)
                self.preview.addItem(it)
        j = Job(work, self)
        j.done.connect(done)
        j.finished.connect(j.deleteLater)
        self._preview_job = j
        j.start()

    def _preview_open(self, it):
        p = it.data(Qt.ItemDataRole.UserRole)
        img = imread(p) if p else None
        if img is not None:
            ImagePreview(self, img, os.path.basename(p), [("Show file", lambda: open_path(os.path.dirname(p)))]).exec()

    def _add_analysis_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Videos to analyse", self.st.value("video_dir", ""), VIDEO_FILTER)
        if paths:
            self.st.setValue("video_dir", os.path.dirname(paths[0]))
        have = {self.vid_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.vid_list.count())}
        for p in paths:
            if p not in have:
                self._add_analysis_video(p)

    def _add_analysis_video(self, p, start=None, end=None):
        txt = os.path.basename(p) + (f"   [{human_time(start or 0)} – {human_time(end) if end else 'end'}]"
                                     if start or end else "")
        it = QListWidgetItem(txt)
        it.setData(Qt.ItemDataRole.UserRole, p)
        it.setData(Qt.ItemDataRole.UserRole + 1, (start, end))
        it.setToolTip(p)
        self.vid_list.addItem(it)
        self._changed()

    def _videos_from_testset(self):
        if not self.testset:
            QMessageBox.information(self, "Testing", "Open or build a test set first.")
            return
        have = {self.vid_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.vid_list.count())}
        for s in self.testset.manifest["sources"]:
            if s.get("type") == "video" and s["path"] not in have and os.path.isfile(s["path"]):
                self._add_analysis_video(s["path"], s.get("start"), s.get("end"))
                have.add(s["path"])

    def analysis_videos(self):
        out = []
        for i in range(self.vid_list.count()):
            it = self.vid_list.item(i)
            a, b = it.data(Qt.ItemDataRole.UserRole + 1) or (None, None)
            out.append(dict(path=it.data(Qt.ItemDataRole.UserRole), start=a, end=b))
        return out

    def add_video(self, path, start=None, end=None):
        """From the main window: analyse this video (optionally only in -> out)."""
        self._add_analysis_video(path, start, end)

    def _prelabel(self):
        if self.testset is None:
            return
        self._sync_specs()
        if not self.specs:
            QMessageBox.information(self, "Testing", "Add a model on the Models page first.")
            self.nav.setCurrentRow(1)
            return
        m = QMenu(self)
        for i, s in enumerate(self.specs):
            m.addAction(_swatch_icon(slot_hex(self.slots[i])), s.label).setData(i)
        act = m.exec(self.prelabel_btn.mapToGlobal(self.prelabel_btn.rect().bottomLeft()))
        if act is None:
            return
        spec = self.specs[act.data()]
        fmt = "labelme" if QMessageBox.question(
            self, "Pre-label", "Write LabelMe json (next to the images) instead of YOLO txt (labels/)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes else "yolo"
        ts, cache = self.testset, self.cache
        conf = float(self.st.value("prelabel_conf", 0.25))

        def work(job):
            from AIO_QC.core.testing import pipeline as P
            return P.prelabel(ts, spec, cache, conf=conf, fmt=fmt,
                              progress=lambda mi, d, t, ms, txt: job.progress.emit((d, t, txt)), cancel=job.cancelled)

        def done(n):
            QMessageBox.information(self, "Pre-label", f"Labels written for {n:,} images (confidence ≥ {conf:.2f}).\n\n"
                                    "Fix them in your labelling tool; edited files count as ground truth automatically. "
                                    "'Mark pre-labels reviewed' accepts the rest.")
            self._reload_data()
        self._start(work, done, bar=self.build_bar)

    def _mark_reviewed(self):
        if self.testset and QMessageBox.question(
                self, "Testing", "Accept every unreviewed pre-label as ground truth?\n\nOnly do this after checking "
                "them - otherwise a model is scored against its own predictions.") == QMessageBox.StandardButton.Yes:
            self.testset.mark_reviewed()
            self._reload_data()
