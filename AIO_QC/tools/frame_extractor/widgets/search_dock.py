# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Search frames dock: find moments by describing them ("white van", "person on a bike at night") or by example
(the current frame / ROI, or any result). Results are thumbnails - click one to jump there.

Text search uses CLIP in a Python environment (Settings → Models → Environment; `pip install open_clip_torch`,
weights download once, then everything is offline). "Find similar" also works without it (built-in features).
Each video is indexed once (a frame every N seconds, GPU decode in ffmpeg) and the index is kept on disk.
"""
import os

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QComboBox, QLabel, QListWidget,
    QListWidgetItem, QDoubleSpinBox, QCheckBox, QMenu, QFileDialog, QToolButton, QAbstractItemView,
)

from AIO_QC.widgets import motion as anim
from AIO_QC.widgets.testing.lab_common import Job, human_time

from AIO_QC.tools.frame_extractor.core.common import VIDEO_FILTER
from AIO_QC.utils.fx_settings import fx_settings


class SearchDock(QDockWidget):
    seek_requested = pyqtSignal(str, float)          # video, seconds
    marker_requested = pyqtSignal(str, float)

    def __init__(self, parent, context):
        """context() -> dict(video, frame (BGR or None), roi (x1,y1,x2,y2) or None, env_python, ffmpeg)."""
        super().__init__("Search frames", parent)
        self.setObjectName("SearchDock")
        self.context = context
        self.st = fx_settings()
        self.job = None
        self.embedders = {}               # (kind, env) -> embedder
        self.extra_videos = [v for v in (self.st.value("search/videos", []) or []) if isinstance(v, str)]
        self.results = []

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("Describe it… e.g. red truck, person on a bike, night")
        self.query.setToolTip("Text search (CLIP). Words starting with - push results away: 'truck -night'.\n"
                              "Enter to search.")
        self.query.returnPressed.connect(self.search_text)
        self.go = QPushButton("Search")
        self.go.clicked.connect(self.search_text)
        row.addWidget(self.query, 1)
        row.addWidget(self.go)
        lay.addLayout(row)

        row = QHBoxLayout()
        self.similar = QPushButton("Find similar to this frame")
        self.similar.setToolTip("Frames that look like the current frame - or like the ROI when 'ROI only' is ticked.")
        self.similar.clicked.connect(lambda: self.search_example())
        self.roi_only = QCheckBox("ROI only")
        self.roi_only.setToolTip("Use only the ROI as the example (an object rather than the whole scene). CLIP only.")
        row.addWidget(self.similar, 1)
        row.addWidget(self.roi_only)
        lay.addLayout(row)

        row = QHBoxLayout()
        self.engine = QComboBox()
        self.engine.addItem("CLIP — text & image", "clip")
        self.engine.addItem("Built-in — similar scenes only", "classic")
        self.engine.setToolTip("CLIP understands what is in the picture (needs a Python environment with "
                               "open_clip_torch or transformers).\nBuilt-in compares colour, layout and edges - "
                               "no install, good for the same scene / camera view.")
        self.engine.setCurrentIndex(0 if self.st.value("search/engine", "clip") == "clip" else 1)
        self.engine.currentIndexChanged.connect(self._engine_changed)
        self.every = QDoubleSpinBox()
        self.every.setRange(0.2, 60.0)
        self.every.setSingleStep(0.5)
        self.every.setSuffix(" s")
        self.every.setValue(float(self.st.value("search/every", 1.0)))
        self.every.setToolTip("Index one frame every N seconds (finer = more precise, slower to index).\n"
                              "Very long videos are capped at about 6000 frames.")
        row.addWidget(self.engine, 1)
        row.addWidget(QLabel("every"))
        row.addWidget(self.every)
        lay.addLayout(row)

        row = QHBoxLayout()
        self.scope = QComboBox()
        self.scope.addItem("This video", "this")
        self.scope.addItem("This video + other videos", "all")
        self.scope.setToolTip("Search other videos too (each is indexed once).")
        self.scope.currentIndexChanged.connect(self._scope_changed)
        self.manage = QToolButton()
        self.manage.setText("Videos…")
        self.manage.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.manage_menu = QMenu(self)
        self.manage_menu.aboutToShow.connect(self._build_videos_menu)
        self.manage.setMenu(self.manage_menu)
        row.addWidget(self.scope, 1)
        row.addWidget(self.manage)
        lay.addLayout(row)

        self.bar = anim.SmoothProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedHeight(8)
        self.bar.setTextVisible(False)
        self.bar.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.bar)
        lay.addWidget(self.status)

        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setIconSize(QSize(176, 99))
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setMovement(QListWidget.Movement.Static)
        self.list.setSpacing(4)
        self.list.setUniformItemSizes(True)
        self.list.setWordWrap(True)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.itemClicked.connect(self._open)
        self.list.itemActivated.connect(self._open)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.list, 1)
        self.setWidget(w)
        self._scope_changed()
        self._engine_changed()

    # ---------------------------------------------------------------- settings
    def _engine_changed(self, *_):
        self.st.setValue("search/engine", self.engine.currentData())
        self._apply_engine_state()
        if self.engine.currentData() != "clip":
            self.status.setText("Built-in search finds frames that look like an example. Pick CLIP to search by text.")

    def _apply_engine_state(self):
        text_ok = self.engine.currentData() == "clip"
        self.query.setEnabled(text_ok)
        self.go.setEnabled(text_ok)
        self.roi_only.setEnabled(text_ok)

    def _scope_changed(self, *_):
        self.manage.setVisible(self.scope.currentData() == "all")

    def _build_videos_menu(self):
        m = self.manage_menu
        m.clear()
        m.addAction("Add videos…").triggered.connect(self._add_videos)
        if self.extra_videos:
            m.addSeparator()
            for v in self.extra_videos:
                sub = m.addMenu(os.path.basename(v))
                sub.addAction("Remove from search").triggered.connect(lambda _=False, v=v: self._remove_video(v))
            m.addSeparator()
            m.addAction("Clear list").triggered.connect(lambda: self._set_videos([]))

    def _add_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Videos to search", self.st.value("last_dir", ""), VIDEO_FILTER)
        if paths:
            self._set_videos(self.extra_videos + [p for p in paths if p not in self.extra_videos])

    def _remove_video(self, v):
        self._set_videos([x for x in self.extra_videos if x != v])

    def _set_videos(self, vids):
        self.extra_videos = vids
        self.st.setValue("search/videos", vids)
        self.status.setText(f"{len(vids)} other video(s) in the search." if vids else "")

    def _videos(self):
        ctx = self.context()
        vids = [ctx["video"]] if ctx.get("video") else []
        if self.scope.currentData() == "all":
            vids += [v for v in self.extra_videos if v not in vids and os.path.isfile(v)]
        return vids

    # ---------------------------------------------------------------- searching
    def search_text(self):
        q = self.query.text().strip()
        if not q:
            return
        pos = " ".join(w for w in q.split() if not w.startswith("-")) or q
        neg = [w[1:] for w in q.split() if w.startswith("-") and len(w) > 1]
        self._search(("text", pos, neg), f"“{q}”")

    def search_example(self, image=None, exclude=None, label=None):
        ctx = self.context()
        img = image
        if img is None:
            img = ctx.get("frame")
            if img is None:
                self.status.setText("Load a video first.")
                return
            roi = ctx.get("roi")
            if self.roi_only.isChecked() and roi and self.engine.currentData() == "clip":
                x1, y1, x2, y2 = (int(v) for v in roi)
                if x2 - x1 > 8 and y2 - y1 > 8:
                    img = img[max(0, y1):y2, max(0, x1):x2]
            if exclude is None and ctx.get("video") is not None and ctx.get("time") is not None:
                exclude = (os.path.abspath(ctx["video"]), float(ctx["time"]))
        self._search(("image", img.copy(), exclude), label or "frames like this one")

    def _embedder(self, kind, env):
        from AIO_QC.core.testing import search as SR
        key = (kind, env if kind == "clip" else "")
        if key not in self.embedders:
            if kind == "clip":
                if not env:
                    raise RuntimeError("Text search needs a Python environment with CLIP: Settings → Models → "
                                       "Environment, then `pip install open_clip_torch` in it. "
                                       "Or pick 'Built-in' to find similar scenes.")
                self.embedders[key] = SR.ClipEmbedder(env)
            else:
                self.embedders[key] = SR.ClassicEmbedder()
        return self.embedders[key]

    def _search(self, query, what):
        if self.job is not None:
            self.job.cancel()
            return
        vids = self._videos()
        if not vids:
            self.status.setText("Load a video first.")
            return
        ctx = self.context()
        kind, env, ff = self.engine.currentData(), ctx.get("env_python", ""), ctx.get("ffmpeg")
        if not ff:
            self.status.setText("ffmpeg was not found (Settings → Playback → ffmpeg location).")
            return
        every = self.every.value()
        self.st.setValue("search/every", every)

        def work(job):
            from AIO_QC.core.testing import search as SR
            job.progress.emit((0, 1, "Loading CLIP…" if kind == "clip" else "Preparing…"))
            emb = self._embedder(kind, env)
            indexes = []
            for vi, v in enumerate(vids):
                if job.cancelled():
                    return None
                ix = SR.VideoIndex(v, emb.id)
                ix.build(emb, ff, interval=every, cancel=job.cancelled,
                         progress=lambda d, t, txt, vi=vi: job.progress.emit(
                             (d, t, f"Indexing {vi + 1}/{len(vids)}: {txt}")))
                indexes.append(ix)
            job.progress.emit((1, 1, "Searching…"))
            if query[0] == "text":
                qv = emb.embed_text([query[1]])
                neg = emb.embed_text(query[2]) if query[2] else None
                exclude = None
            else:
                qv = emb.embed([query[1]])
                neg, exclude = None, query[2]
            gap = max(2.0, every * 2)
            return SR.search(indexes, qv, k=48, min_gap=gap, negative=neg, exclude=exclude), emb.label, \
                sum(len(ix.times) for ix in indexes)
        self.list.setEnabled(False)            # old results stay visible (dimmed) until the new ones arrive
        self.bar.setVisible(True)
        self.bar.setValue(0)
        self.go.setText("Stop")
        self.go.setEnabled(True)
        self.similar.setEnabled(False)
        self.status.setText(f"Searching {what}…")
        self.job = Job(work, self)
        self.job.progress.connect(self._progress)
        self.job.done.connect(lambda r: self._show(r, what))
        self.job.failed.connect(lambda m: self.status.setText(m.split(": ", 1)[1] if m.startswith(
            ("RuntimeError: ", "FileNotFoundError: ", "ValueError: ")) else m))
        self.job.finished.connect(self._finished)
        self.job.start()

    def _progress(self, p):
        d, t, text = p
        self.bar.setValue(int(1000 * d / t) if t else 0)
        self.status.setText(text)

    def _finished(self):
        self.job = None
        self.list.setEnabled(True)
        self.bar.setVisible(False)
        self.go.setText("Search")
        self.similar.setEnabled(True)
        self._apply_engine_state()

    def _show(self, r, what):
        if r is None:
            self.status.setText("Stopped.")
            return
        results, label, n = r
        self.results = results
        before = self.list.grab() if self.list.count() else None
        self.list.clear()
        multi = len({x["video"] for x in results}) > 1
        for k, x in enumerate(results):
            pm = QPixmap()
            if x["thumb"]:
                pm.loadFromData(x["thumb"], "JPG")
            name = (os.path.basename(x["video"]) + "\n") if multi else ""
            it = QListWidgetItem(QIcon(pm), f"{name}{human_time(x['time'])} · {x['score']:.2f}")
            it.setData(Qt.ItemDataRole.UserRole, k)
            it.setToolTip(f"{x['video']}\n{human_time(x['time'])} · similarity {x['score']:.3f}\nClick to jump there.")
            self.list.addItem(it)
        anim.transition(self.list, before, 1, "y", 220)
        self.status.setText(f"{len(results)} moments for {what} · {n:,} frames searched · {label}"
                            if results else f"Nothing found for {what}.")

    def _item(self, it):
        k = it.data(Qt.ItemDataRole.UserRole) if it is not None else None
        return self.results[k] if k is not None and k < len(self.results) else None

    def _open(self, it):
        x = self._item(it)
        if x:
            self.seek_requested.emit(x["video"], float(x["time"]))

    def _menu(self, pos):
        it = self.list.itemAt(pos)
        x = self._item(it)
        if not x:
            return
        m = QMenu(self)
        m.addAction("Go there").triggered.connect(lambda: self.seek_requested.emit(x["video"], float(x["time"])))
        m.addAction("Add extraction marker here").triggered.connect(
            lambda: self.marker_requested.emit(x["video"], float(x["time"])))
        m.addAction("Find more like this").triggered.connect(lambda: self._more_like(x))
        m.exec(self.list.viewport().mapToGlobal(pos))

    def _more_like(self, x):
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(x["thumb"], np.uint8), cv2.IMREAD_COLOR) if x["thumb"] else None
        if img is not None:
            self.search_example(img, exclude=(x["video"], x["time"]), label=f"frames like {human_time(x['time'])}")

    def shutdown(self):
        if self.job is not None:
            self.job.cancel()
            self.job.wait(3000)
