# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Testing (Model Lab) — step 6, reviewing videos; step checks; drag and drop.

A mixin of AIO_QC.widgets.testing.model_lab.ModelLab: the methods use the window's own attributes, set up in its __init__.
"""
import os
import time
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QTableWidgetItem, QAbstractItemView, QComboBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QRadioButton, QGroupBox, QFileDialog, QMessageBox, QApplication,
    QMenu,
)
from AIO_QC.widgets.testing.lab_common import (
    open_path,
)
from AIO_QC.tools.frame_extractor.core.common import VIDEO_FILTER
from AIO_QC.utils.qt_utils import later
from AIO_QC.core.diag import ignored as _ignored

from AIO_QC.widgets.testing.model_lab_common import (_hint, _make_table, _shared_env, _status_icon, _strip_tags, slot_hex)


class ReviewMixin:

    # ------------------------------------------------------------------ persistence
    # ------------------------------------------------------------------ projects & start page
    # ------------------------------------------------------------------ 6 review videos
    def _page_review(self):
        w, lay = self._page(
            "Review videos", step=None,
            subtitle="Play videos in the Frame Extractor with every model's detections drawn on them - one after another. "
                     "Testing only sends the playlist: the player analyses ahead of the playhead, keeps playback smooth "
                     "and shows the boxes of the exact frame on screen. Detections are saved in the prediction cache.")
        self.rv_table = _make_table(["Video", "In", "Out", "Length"], stretch_col=0)
        self.rv_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                      QAbstractItemView.EditTrigger.EditKeyPressed)
        self.rv_table.setToolTip("Double-click In / Out to play only part of a video (1:30, 90, 0:01:30.5).")
        self.rv_table.itemChanged.connect(self._review_edited)
        lay.addWidget(self.rv_table, 2)
        row = QHBoxLayout()
        for text, fn, tip in (("Add videos…", self._review_add_videos, "Several at once"),
                              ("Add folder…", self._review_add_folder, "Every video in a folder (and its sub-folders)"),
                              ("Use step 1's videos", self._review_from_step1, "The videos for behaviour analysis"),
                              ("Remove", self._review_remove, None),
                              ("↑", lambda: self._review_move(-1), "Play earlier"),
                              ("↓", lambda: self._review_move(1), "Play later")):
            b = QPushButton(text)
            if tip:
                b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        mbox = QGroupBox("Models (from step 2)")
        mv = QVBoxLayout(mbox)
        self.rv_models = QLabel("")
        self.rv_models.setWordWrap(True)
        self.rv_models.setTextFormat(Qt.TextFormat.RichText)
        self.rv_models.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        mv.addWidget(self.rv_models)
        mrow = QHBoxLayout()
        b = QPushButton("Edit models…")
        b.clicked.connect(lambda: self.nav.setCurrentRow(1))
        mrow.addWidget(b)
        b = QPushButton("Refresh details")
        b.clicked.connect(lambda: self._refresh_review_page(force=True))
        mrow.addWidget(b)
        mrow.addStretch(1)
        mv.addLayout(mrow)
        lay.addWidget(mbox)
        obox = QGroupBox("Analysis")
        f = QFormLayout(obox)
        self.rv_mode = QComboBox()
        for key, text in (("balanced", "Balanced - smooth real-time playback (recommended)"),
                          ("best", "Best quality - every frame; playback waits when needed"),
                          ("fixed", "Fixed step - every N-th frame, interpolated in between")):
            self.rv_mode.addItem(text, key)
        self.rv_mode.setToolTip("Balanced: when the models keep up, every frame is analysed live; when they can't, "
                                "the frame step adapts, frames in between are interpolated from the analysed frames on "
                                "both sides, and spare time fills them in afterwards.")
        self.rv_stride = QSpinBox()
        self.rv_stride.setRange(2, 30)
        self.rv_stride.setSuffix(" (every N-th frame)")
        self.rv_lead = QDoubleSpinBox()
        self.rv_lead.setRange(0.5, 30)
        self.rv_lead.setSingleStep(0.5)
        self.rv_lead.setSuffix(" s")
        self.rv_lead.setToolTip("How far ahead of the playhead detections are kept ready")
        self.rv_conf = QDoubleSpinBox()
        self.rv_conf.setRange(0.01, 0.99)
        self.rv_conf.setSingleStep(0.05)
        self.rv_conf.setToolTip("Boxes below this are hidden at first - a slider in the player changes it live")
        self.rv_loop = QCheckBox("Start again after the last video")
        from AIO_QC.widgets.settings_pages import review_defaults
        dflt = review_defaults()
        self.rv_mode.setCurrentIndex(max(0, self.rv_mode.findData(dflt["mode"])))
        self.rv_stride.setValue(dflt["stride"])
        self.rv_lead.setValue(dflt["lead_s"])
        self.rv_conf.setValue(dflt["conf"])
        self.rv_mode.currentIndexChanged.connect(lambda _i: (self.rv_stride.setEnabled(
            self.rv_mode.currentData() == "fixed"), self._changed()))
        self.rv_stride.setEnabled(self.rv_mode.currentData() == "fixed")
        for x in (self.rv_stride, self.rv_lead, self.rv_conf):
            x.valueChanged.connect(lambda _v: self._changed())
        self.rv_loop.toggled.connect(lambda _on: self._changed())
        f.addRow("Mode", self.rv_mode)
        f.addRow("Step", self.rv_stride)
        f.addRow("Look-ahead", self.rv_lead)
        f.addRow("Show boxes from confidence", self.rv_conf)
        f.addRow("", self.rv_loop)
        lay.addWidget(obox)
        row = QHBoxLayout()
        self.rv_hint = QLabel("")
        self.rv_hint.setWordWrap(True)
        row.addWidget(self.rv_hint, 1)
        self.rv_go = QPushButton("▶  Review in player")
        self.rv_go.setObjectName("review_in_player")
        self.rv_go.setToolTip("Switches to the Frame Extractor and plays the videos with the detections")
        self.rv_go.clicked.connect(self._review_go)
        row.addWidget(self.rv_go)
        lay.addLayout(row)
        self._rv_info = {}                          # model path -> describe() result
        return w

    def _review_videos(self):
        from AIO_QC.tools.frame_extractor.core.sources import parse_time
        out = []
        for r in range(self.rv_table.rowCount()):
            path = self.rv_table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            vals = []
            for c in (1, 2):
                t = (self.rv_table.item(r, c).text() or "").strip()
                try:
                    vals.append(parse_time(t) if t else None)
                except ValueError:
                    vals.append(None)
            out.append(dict(path=path, start=vals[0], end=vals[1]))
        return out

    def _review_add(self, path, start=None, end=None):
        from AIO_QC.tools.frame_extractor.core.sources import format_time
        have = {self.rv_table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in range(self.rv_table.rowCount())}
        if path in have:
            return
        self.rv_table.blockSignals(True)
        r = self.rv_table.rowCount()
        self.rv_table.insertRow(r)
        it = QTableWidgetItem(os.path.basename(path))
        it.setData(Qt.ItemDataRole.UserRole, path)
        it.setToolTip(path)
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.rv_table.setItem(r, 0, it)
        self.rv_table.setItem(r, 1, QTableWidgetItem(format_time(start) if start else ""))
        self.rv_table.setItem(r, 2, QTableWidgetItem(format_time(end) if end else ""))
        ln = QTableWidgetItem("")
        ln.setFlags(ln.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.rv_table.setItem(r, 3, ln)
        self.rv_table.blockSignals(False)
        self._review_probe(path)
        self._changed()

    def _review_probe(self, path):
        """Length and frame rate, off the UI thread."""
        def fn():
            from AIO_QC.tools.frame_extractor.core.sources import open_source, format_time
            src = open_source(path, "auto", hw_accel=False)
            try:
                return f"{format_time(src.duration())[:-4]} · {src.fps:.2f} fps · {src.frame_count:,} frames"
            finally:
                src.release()

        def ok(text):
            for r in range(self.rv_table.rowCount()):
                if self.rv_table.item(r, 0).data(Qt.ItemDataRole.UserRole) == path:
                    self.rv_table.blockSignals(True)
                    self.rv_table.item(r, 3).setText(text)
                    self.rv_table.blockSignals(False)

        def bad(msg):
            ok(f"can't open: {msg}")
        self.views.submit(f"rv-probe:{path}", fn, ok, bad)

    def _review_add_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Videos to review", self.st.value("video_dir", ""), VIDEO_FILTER)
        if paths:
            self.st.setValue("video_dir", os.path.dirname(paths[0]))
        for p in paths:
            self._review_add(p)
        self._refresh_review_page()

    def _review_add_folder(self):
        from AIO_QC.tools.frame_extractor.core.common import VIDEO_EXTS
        d = QFileDialog.getExistingDirectory(self, "Folder with videos", self.st.value("video_dir", ""))
        if not d:
            return
        found = []
        for root, _dirs, files in os.walk(d):
            found += [os.path.join(root, f) for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXTS]
        for p in sorted(found):
            self._review_add(p)
        self._refresh_review_page()

    def _review_from_step1(self):
        for v in self.analysis_videos():
            if os.path.isfile(v["path"]):
                self._review_add(v["path"], v.get("start"), v.get("end"))
        self._refresh_review_page()

    def _review_remove(self):
        rows = sorted({i.row() for i in self.rv_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.rv_table.removeRow(r)
        self._changed()
        self._refresh_review_page()

    def _review_move(self, d):
        r = self.rv_table.currentRow()
        j = r + d
        if r < 0 or not 0 <= j < self.rv_table.rowCount():
            return
        self.rv_table.blockSignals(True)
        for c in range(self.rv_table.columnCount()):
            a, b = self.rv_table.takeItem(r, c), self.rv_table.takeItem(j, c)
            self.rv_table.setItem(r, c, b)
            self.rv_table.setItem(j, c, a)
        self.rv_table.blockSignals(False)
        self.rv_table.setCurrentCell(j, 0)
        self._changed()

    def _review_edited(self, item):
        self._changed()
        self._refresh_review_page()

    def _review_state(self):
        return dict(videos=self._review_videos(), mode=self.rv_mode.currentData(), stride=self.rv_stride.value(),
                    lead_s=self.rv_lead.value(), conf=self.rv_conf.value(), loop=self.rv_loop.isChecked())

    def _apply_review_state(self, st):
        self.rv_table.setRowCount(0)
        for v in st.get("videos", []):
            if v.get("path"):
                self._review_add(v["path"], v.get("start"), v.get("end"))
        if st.get("mode"):
            self.rv_mode.setCurrentIndex(max(0, self.rv_mode.findData(st["mode"])))
        for x, k in ((self.rv_stride, "stride"), (self.rv_lead, "lead_s"), (self.rv_conf, "conf")):
            if k in st:
                x.setValue(st[k])
        self.rv_loop.setChecked(bool(st.get("loop")))

    def _refresh_review_page(self, force=False):
        """Model details (read off the UI thread, once per file) and what still blocks the review."""
        self._sync_specs()
        if force:
            self._rv_info = {}
        todo = [s for s in self.specs if s.path not in self._rv_info]
        for s in todo:
            self._rv_info[s.path] = None
            from AIO_QC.core.testing.review import describe

            def ok(info, path=s.path):
                self._rv_info[path] = info
                self._show_review_models()

            def bad(msg, path=s.path, label=s.label):
                self._rv_info[path] = dict(label=label, error=msg)
                self._show_review_models()
            self.views.submit(f"rv-model:{s.path}", lambda s=s: describe(s), ok, bad)
        self._show_review_models()

    def _show_review_models(self):
        from AIO_QC.core.testing.review import describe_text
        if not self.specs:
            self.rv_models.setText("<i>No models yet - add weights in step 2.</i>")
        else:
            cards = []
            for i, s in enumerate(self.specs):
                col = slot_hex(self.slots[i]) if i < len(self.slots) else "#888"
                info = self._rv_info.get(s.path)
                body = describe_text(info) if info else f"<b>{s.label}</b> <i>reading…</i>"
                cards.append(f"<table><tr><td style='background:{col}' width='6'></td>"
                             f"<td style='padding-left:6px'>{body}</td></tr></table>")
            self.rv_models.setText("".join(cards))
        vids = self._review_videos()
        problems = []
        if not vids:
            problems.append("add videos")
        if not self.specs:
            problems.append("add models (step 2)")
        bad = [os.path.basename(v["path"]) for v in vids if not os.path.isfile(v["path"])]
        if bad:
            problems.append("missing: " + ", ".join(bad[:3]))
        err = [i["label"] for i in self._rv_info.values() if i and i.get("error")]
        shell = getattr(QApplication.instance(), "shell", None)
        if shell is None:
            problems.append("the Frame Extractor is available when AIO QC is started normally")
        self.rv_go.setEnabled(not problems)
        self.rv_hint.setText(("<span style='color:#d03b3b'>" + " · ".join(problems) + "</span>") if problems else
                             f"{len(vids)} video(s) · {len(self.specs)} model(s)"
                             + (f" · <span style='color:#c77700'>{len(err)} can't be read: "
                                f"{', '.join(err)}</span>" if err else ""))
        it = self.nav.item(5)
        if it is not None:
            it.setIcon(_status_icon("ready" if not problems else "todo"))

    def _review_go(self):
        from AIO_QC.core.testing.review import ReviewItem, ReviewPlaylist
        self._sync_specs()
        pl = ReviewPlaylist(
            items=[ReviewItem(v["path"], v["start"], v["end"]) for v in self._review_videos()],
            specs=list(self.specs), colors=[slot_hex(sl) for sl in self.slots], mode=self.rv_mode.currentData(),
            stride=self.rv_stride.value(), lead_s=self.rv_lead.value(), conf=self.rv_conf.value(),
            loop=self.rv_loop.isChecked(), cache_path=getattr(self.cache, "path", "") or "",
            source=f"Testing · {self._project_name}")
        shell = getattr(QApplication.instance(), "shell", None)
        if shell is None:
            QMessageBox.information(self, "Review in player", "The Frame Extractor is available when AIO QC is started "
                                                             "normally.")
            return
        shell.review_in_player(pl)

    def _build_menu(self):
        mb = self.menuBar()
        m = mb.addMenu("Project")
        self.act_new = m.addAction("New project", self._new_project)
        self.act_new.setShortcut(QKeySequence("Ctrl+N"))
        self.act_open = m.addAction("Open project…", self._open_project_dialog)
        self.act_open.setShortcut(QKeySequence("Ctrl+O"))
        self.recent_menu = m.addMenu("Open recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        m.addSeparator()
        self.act_save = m.addAction("Save project", self._save_project)
        self.act_save.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save_as = m.addAction("Save project as…", lambda: self._save_project(ask=True))
        self.act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        m.addSeparator()
        m.addAction("Start page", self._show_home)
        m.addSeparator()
        m.addAction("Close", self.close).setShortcut(QKeySequence("Ctrl+W"))
        h = mb.addMenu("Help")
        h.addAction("Testing guide", lambda: self.help_requested.emit("09-model-lab.html"))
        h.addAction("Metrics explained", lambda: self.help_requested.emit("13-metrics.html"))

    def _page_home(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(28, 22, 28, 18)
        lay.setSpacing(12)
        t = QLabel("<span style='font-size:20pt; font-weight:600'>Testing</span>")
        lay.addWidget(t)
        lay.addWidget(_hint("Measure and compare detection and classification models on your own footage. A project remembers the test "
                            "data, models and settings - reopen it next month to test a new model on exactly the "
                            "same frames. Tip: drop videos, folders, data.yaml or weights anywhere on this window."))
        row = QHBoxLayout()
        for text, fn, tip in (("New project", self._new_project, "Start from scratch (Ctrl+N)"),
                              ("Open project…", self._open_project_dialog, "Open a .fxlab file (Ctrl+O)"),
                              ("Continue last session", self._continue_last, "Everything as you left it")):
            b = QPushButton(text)
            b.setMinimumHeight(34)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
            if text.startswith("Continue"):
                self.continue_btn = b
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(QLabel("<b>Recent projects</b>"))
        self.recent_list = QListWidget()
        self.recent_list.setSpacing(2)
        self.recent_list.itemActivated.connect(lambda it: self._open_project(it.data(Qt.ItemDataRole.UserRole)))
        self.recent_list.itemDoubleClicked.connect(lambda it: self._open_project(it.data(Qt.ItemDataRole.UserRole)))
        self.recent_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.recent_list.customContextMenuRequested.connect(self._recent_menu_at)
        lay.addWidget(self.recent_list, 1)
        steps = QLabel("<b>How it works</b> &nbsp; 1 Test data → 2 Models → 3 Run → 4 Results → 5 Report")
        steps.setEnabled(False)
        lay.addWidget(steps)
        return w

    def _fill_recent(self):
        from AIO_QC.core.testing import project as PJ
        self.recent_list.clear()
        rows = PJ.recent()
        for r in rows:
            s = r.get("summary") or {}
            when = time.strftime("%d %b %Y %H:%M", time.localtime(r.get("saved") or 0))
            res = (f"best {s['best']} · " + (f"top-1 {s['top1']:.1%}" if s.get("task") == "classify" else
                                            f"mAP50-95 {s['map5095']:.3f}") + f" · {s.get('images', 0):,} images"
                   if s.get("best") else "no results yet")
            it = QListWidgetItem(f"{r.get('name')}\n{res} · saved {when}\n{r['path']}")
            it.setData(Qt.ItemDataRole.UserRole, r["path"])
            it.setToolTip(r["path"])
            self.recent_list.addItem(it)
        if not rows:
            it = QListWidgetItem("No projects yet - start a new one, or just begin on the Test data page "
                                 "(your work is kept automatically).")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent_list.addItem(it)
        self.continue_btn.setEnabled(os.path.isfile(PJ.AUTOSAVE))

    def _fill_recent_menu(self):
        from AIO_QC.core.testing import project as PJ
        self.recent_menu.clear()
        for r in PJ.recent():
            self.recent_menu.addAction(r.get("name") or os.path.basename(r["path"]),
                                       lambda p=r["path"]: self._open_project(p))
        if self.recent_menu.isEmpty():
            self.recent_menu.addAction("(none)").setEnabled(False)

    def _recent_menu_at(self, pos):
        it = self.recent_list.itemAt(pos)
        p = it.data(Qt.ItemDataRole.UserRole) if it else None
        if not p:
            return
        from AIO_QC.core.testing import project as PJ
        m = QMenu(self)
        m.addAction("Open", lambda: self._open_project(p))
        m.addAction("Show in folder", lambda: open_path(os.path.dirname(p)))
        m.addAction("Remove from list", lambda: (PJ.remove_recent(p), self._fill_recent()))
        m.exec(self.recent_list.viewport().mapToGlobal(pos))

    def _show_home(self):
        self._fill_recent()
        self.nav.blockSignals(True)
        self.nav.clearSelection()
        self.nav.setCurrentRow(-1)
        self.nav.blockSignals(False)
        self.pages.setCurrentIndex(self.HOME)

    def _collect_state(self):
        kind = "qc" if self.kind_qc.isChecked() else "testset" if self.kind_ts.isChecked() else \
            "dataset" if self.kind_ds.isChecked() else "none"
        path = (self.testset.root if self.testset is not None else self.ts_path.text()) if kind == "testset" \
            else self.ds_path.text().strip() if kind == "dataset" \
            else (self.data.root if self.data is not None and self.data.qc else "") if kind == "qc" else ""
        models = []
        for r in range(self.m_table.rowCount()):
            models.append(dict(path=self.m_table.item(r, 2).text(), label=self.m_table.item(r, 1).text(),
                               imgsz=self.m_table.item(r, 4).text(),
                               half=self.m_table.item(r, 5).checkState() == Qt.CheckState.Checked,
                               slot=self.m_table.item(r, 0).data(Qt.ItemDataRole.UserRole),
                               class_map=self.m_table.item(r, 1).data(Qt.ItemDataRole.UserRole) or {}))
        return dict(name=self._project_name, data=dict(kind=kind, path=path, fmt=self.ds_fmt.currentData()),
                    videos=self.analysis_videos(), every_s=self.vid_every.value(), models=models,
                    env=self.env_edit.text().strip(), device=self.device_edit.currentText().strip(),
                    batch=self.batch_spin.value(), sampling=self._ts_settings(save=False),
                    run=dict(ground=self.ground.currentData(), reference=self.reference.currentData(),
                             unreviewed=self.unreviewed.isChecked(), best=self.thr_best.isChecked(),
                             thr=self.thr_val.value()),
                    report=dict(dir=self.rep_dir.text(), title=self.rep_title.text()),
                    review=self._review_state(), summary=self._summary or {})

    def _apply_state(self, st):
        self._loading = True
        try:
            self._project_name = st.get("name") or "Untitled"
            self._summary = st.get("summary") or {}
            self.m_table.setRowCount(0)
            self._add_models([m for m in st.get("models", []) if m.get("path")])
            self.env_edit.setText(_shared_env() or st.get("env", "") or self.model_cfg.get("env_python", ""))
            if st.get("device"):
                self.device_edit.setCurrentText(st["device"])
            self.batch_spin.setValue(int(st.get("batch", 8)))
            self.vid_list.clear()
            for v in st.get("videos", []):
                if v.get("path"):
                    self._add_analysis_video(v["path"], v.get("start"), v.get("end"))
            self.vid_every.setValue(float(st.get("every_s", 0.2)))
            run = st.get("run") or {}
            k = self.ground.findData(run.get("ground", "auto"))
            self.ground.setCurrentIndex(max(0, k))
            self.unreviewed.setChecked(bool(run.get("unreviewed")))
            (self.thr_best if run.get("best", True) else self.thr_fixed).setChecked(True)
            self.thr_val.setValue(float(run.get("thr", 0.25)))
            self._pending_reference = run.get("reference")
            self._apply_review_state(st.get("review") or {})
            rep = st.get("report") or {}
            self.rep_dir.setText(rep.get("dir", ""))
            self.rep_title.setText(rep.get("title", ""))
            self.run, self.videos_out, self.infos = None, [], {}
            self.data = self.testset = None
            self.ts_path.setText("")
            self.ds_path.setText("")
            self.src_table.setRowCount(0)
            d = st.get("data") or {}
            kind = d.get("kind", "testset")
            qc_ok = kind == "qc" and self.qc is not None and not self.qc.unavailable() and \
                os.path.normcase(os.path.abspath(str(self.qc.session.root))) == \
                os.path.normcase(os.path.abspath(d.get("path") or ""))
            if kind == "qc" and not qc_ok:          # made on a dataset that is not open now: read it as a dataset
                kind = "dataset"
                d = dict(d, fmt=d.get("fmt") or "auto")
                self.ds_path.setText(d.get("path") or "")
            {"qc": self.kind_qc, "testset": self.kind_ts, "dataset": self.kind_ds,
             "none": self.kind_none}.get(kind, self.kind_ts).setChecked(True)
            if d.get("fmt"):
                self.ds_fmt.setCurrentIndex(max(0, self.ds_fmt.findData(d["fmt"])))
            self._show_summary()
            self._show_results()
        finally:
            self._loading = False
        self._dirty = False
        self._update_title()
        path = d.get("path")
        if kind == "qc":
            self._load_qc(then=self._load_cached_results)
        elif kind != "none" and path and os.path.exists(path):
            self._load_data(path, then=self._load_cached_results)
        else:
            self._load_cached_results()
        self._revalidate()

    def _new_project(self):
        if not self._confirm_discard():
            return
        self._project_path = None
        self._apply_state(dict(name="Untitled", models=[dict(path=p) for p in self.model_cfg.get("models", [])
                                                         if os.path.isfile(p)]))
        self.nav.setCurrentRow(0)

    def _open_project_dialog(self):
        from AIO_QC.core.testing import project as PJ
        p, _ = QFileDialog.getOpenFileName(self, "Open Testing project", self.st.value("project_dir", ""),
                                           f"Testing project (*{PJ.EXT});;All files (*)")
        if p:
            self._open_project(p)

    def _open_project(self, path):
        from AIO_QC.core.testing import project as PJ
        if not path or not self._confirm_discard():
            return
        try:
            st = PJ.load(path)
        except Exception as e:
            _ignored(e)
            QMessageBox.warning(self, "Testing", f"Could not open the project:\n{e}")
            return
        self._project_path = path
        self.st.setValue("project_dir", os.path.dirname(path))
        PJ.add_recent(path, st)
        self._apply_state(st)
        self.nav.setCurrentRow(3 if self._summary.get("best") else 0)

    def _continue_last(self):
        from AIO_QC.core.testing import project as PJ
        if not self._dirty and (self.data is not None or self.specs):     # the last session is already open
            self.nav.setCurrentRow(3 if (self.run is not None or self.videos_out) else 0)
            return
        if os.path.isfile(PJ.AUTOSAVE):
            try:
                st = PJ.load(PJ.AUTOSAVE)
            except Exception as _exc:
                _ignored(_exc)
                return
            self._project_path = st.get("project_path") if st.get("project_path") and \
                os.path.isfile(st.get("project_path")) else None
            self._apply_state(st)
            self.nav.setCurrentRow(3 if self._summary.get("best") else 0)

    def _save_project(self, ask=False):
        from AIO_QC.core.testing import project as PJ
        path = self._project_path
        if ask or not path:
            base = self.testset.root if self.testset is not None else self.st.value("project_dir", "")
            name = (self._project_name if self._project_name != "Untitled" else "model_lab") + PJ.EXT
            from AIO_QC.utils.save_dirs import ask_save
            path, _ = ask_save(self, "Save Testing project", name, f"Testing project (*{PJ.EXT})", kind="exports",
                               fallback=base or None)
            if not path:
                return False
            if not path.endswith(PJ.EXT):
                path += PJ.EXT
            self._project_name = os.path.splitext(os.path.basename(path))[0]
        st = self._collect_state()
        PJ.save(path, st)
        PJ.add_recent(path, st)
        self._project_path = path
        self.st.setValue("project_dir", os.path.dirname(path))
        self._dirty = False
        self._update_title()
        self._autosave()
        self.status.showMessage(f"Saved {path}", 4000)
        return True

    def _confirm_discard(self):
        if not self._dirty or self._project_path is None:
            return True                     # unnamed sessions live on in the autosave
        r = QMessageBox.question(self, "Testing", f"Save changes to “{self._project_name}”?",
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                 QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if r == QMessageBox.StandardButton.Save:
            return self._save_project()
        return r == QMessageBox.StandardButton.Discard

    def _autosave(self):
        if self._loading:
            return
        from AIO_QC.core.testing import project as PJ
        try:
            st = self._collect_state()
            st["project_path"] = self._project_path
            PJ.save(PJ.AUTOSAVE, st)
            if self.data is not None and self.data.qc:             # part of the dataset's AIO QC session
                PJ.save(PJ.dataset_project(self.data.root), st)
        except Exception as e:
            self._log(f"Autosave failed: {e}")

    def _changed(self, *_):
        if self._loading:
            return
        if not self._dirty:
            self._dirty = True
            self._update_title()
        self._autosave_timer.start()
        self._revalidate()

    def _update_title(self):
        name = self._project_name or "Untitled"
        self.setWindowTitle(f"Testing — {name}{' •' if self._dirty else ''}")

    def _watch_changes(self):
        """Every setting marks the project changed (title •, autosave, step checks)."""
        for w in self.findChildren(QComboBox):
            w.currentIndexChanged.connect(self._changed)
        for w in self.findChildren(QDoubleSpinBox) + self.findChildren(QSpinBox):
            w.valueChanged.connect(self._changed)
        for w in self.findChildren(QCheckBox) + self.findChildren(QRadioButton):
            w.toggled.connect(self._changed)
        for w in (self.env_edit, self.rep_title, self.rep_dir, self.ds_path):
            w.textChanged.connect(self._changed)

    def _load_cached_results(self):
        """Opening a project: if every prediction is already cached, show the results right away (seconds)."""
        specs = self._sync_specs()
        data = None if self.kind_none.isChecked() else self.data
        vids = self.analysis_videos()
        if not specs or (data is None and not vids):
            return
        from AIO_QC.core.testing import pipeline as P
        cache = self.cache if data is not None and self.cache is not None else P.cache_for()

        def ready():
            if data is not None and data.dataset.items:
                keys = set(data.keys)
                for s in specs:
                    if not keys <= cache.keys_for(s.key):
                        return False
            for v in vids:
                for s in specs:
                    if not cache.keys_for(P.video_spec(s).key):
                        return False
            return True
        self._bg("cached?", ready, lambda ok: ok and self.job is None and self._run_all(quiet=True))

    # ------------------------------------------------------------------ step checks
    def _revalidate(self):
        self._valid_timer.start()

    def _validate(self):
        """{step: [(level, text, fix label, fix fn)]} - cheap checks only (files, settings)."""
        out = {i: [] for i in range(5)}
        d = out[0]
        vids = self.analysis_videos()
        missing_v = [v["path"] for v in vids if not os.path.isfile(v["path"])]
        if self.kind_none.isChecked():
            if not vids:
                d.append(("error", "Add at least one video to analyse (or choose test images).", "Add videos…",
                          self._add_analysis_videos))
        elif self.data is None:
            if self.kind_ts.isChecked():
                d.append(("error", "No test set yet: create one, then add videos or image folders to it.",
                          "New test set…", self._new_testset))
            else:
                d.append(("error", "Choose a labelled folder (boxes or class folders) or a data.yaml.", "Folder…", self._browse_ds_folder))
        else:
            ds = self.data.dataset
            if not ds.items:
                d.append(("error", "The test set has no images yet: add videos or folders above and press "
                                   "<b>Add frames to the test set</b>.", "Add videos…", self._add_src_videos))
            else:
                reviewed = sum(1 for it in ds.items if it.labelled and it.reviewed)
                unrev = sum(1 for it in ds.items if it.labelled and not it.reviewed)
                if not reviewed:
                    d.append(("warn", "No reviewed labels: models will be compared with a reference model instead of "
                                      "true accuracy.", "Pre-label…" if self.testset is not None else None,
                              self._prelabel if self.testset is not None else None))
                if unrev:
                    d.append(("warn", f"{unrev:,} image(s) have pre-labels nobody reviewed - they are not scored.",
                              "Review…", lambda: self._review_prelabels()))
        if missing_v:
            d.append(("error", f"{len(missing_v)} video(s) not found: {', '.join(os.path.basename(p) for p in missing_v[:3])}",
                      None, None))
        m = out[1]
        specs = self.specs
        if not specs:
            m.append(("error", "Add the weights you want to test.", "Add weights…", self._add_weights))
        miss = [s for s in specs if not os.path.isfile(s.path)]
        if miss:
            m.append(("error", f"Weights not found: {', '.join(os.path.basename(s.path) for s in miss[:3])}", None, None))
        if any(not s.path.lower().endswith(".onnx") for s in specs) and not self.env_edit.text().strip():
            m.append(("error", ".pt / .engine models need a Python environment with ultralytics.", "Choose…",
                      self._browse_env))
        r = out[2]
        if any(lv == "error" for lv, *_ in out[0] + out[1]):
            r.append(("error", "Fix the problems on the Test data and Models pages first.", None, None))
        elif self.data is not None and self.data.dataset.items and not any(
                it.labelled and it.reviewed for it in self.data.dataset.items) and len(specs) < 2:
            r.append(("warn", "No labels and a single model: there is nothing to score against (add labels or a "
                              "second model).", None, None))
        elif self.ground.currentData() == "reference" or (
                self.data is not None and not any(it.labelled and it.reviewed for it in self.data.dataset.items)):
            r.append(("info", "Scores will be relative to the reference model - not true accuracy.", None, None))
        if self.run is not None and self._run_specs_keys and \
                self._run_specs_keys != [s.key for s in specs]:
            r.append(("warn", "Models changed since the last run.", "Run again", self._run_all))
        res = out[3]
        if self.run is None and not self.videos_out:
            res.append(("info", "No results yet.", "Go to Run", lambda: self.nav.setCurrentRow(2)))
        rep = out[4]
        if self.run is None and not self.videos_out:
            rep.append(("info", "Run the models first - the report is built from the results.", "Go to Run",
                        lambda: self.nav.setCurrentRow(2)))
        return out

    def _apply_validation(self):
        v = self._validate()
        for step, issues in v.items():
            bar = self.issue_bars.get(step)
            if bar is not None:
                bar.set_issues(issues)
            level = "error" if any(i[0] == "error" for i in issues) else "warn" if any(i[0] == "warn" for i in issues) \
                else "ok"
            if step == 3:
                level = "ok" if (self.run is not None and self.run.scored) or self.videos_out else "todo"
            if step == 4:
                level = "ok" if self._report_path else ("todo" if self.run is None and not self.videos_out else "ready")
            it = self.nav.item(step)
            if it is not None:
                it.setIcon(_status_icon(level))
                tip = "\n".join(_strip_tags(i[1]) for i in issues) or {"ok": "Ready", "todo": "Not done yet",
                                                                        "ready": "Ready"}.get(level, "")
                it.setToolTip(tip)
        blocked = [_strip_tags(i[1]) for s in (0, 1, 2) for i in v[s] if i[0] == "error"]
        can = not blocked and self.job is None or (self.job is not None and self.run_btn.text().startswith("■"))
        self.run_btn.setEnabled(bool(can))
        self.run_btn.setToolTip("Start (cached predictions are reused)" if not blocked else "Not ready:\n" + "\n".join(blocked))

    def _review_prelabels(self):
        """Open the label editor on the images with unreviewed pre-labels (needs a scored or loaded run)."""
        if self.run is None:
            QMessageBox.information(self, "Testing", "Run the models once first - the editor shows their detections "
                                    "next to the pre-labels. Or open the images folder in your labelling tool.")
            return
        items = [i for i, it in enumerate(self.run.items) if it.labelled and not it.reviewed]
        if items:
            self._open_editor(items, 0, self.run.ranking()[0] if self.run.ranking() else 0)
    def handle_paths(self, paths):
        """Route dropped files: projects, weights, data.yaml, test sets, labelled folders, image folders, videos."""
        from AIO_QC.tools.frame_extractor.core.common import is_video, WEIGHT_EXTS
        from AIO_QC.core.testing import project as PJ
        from AIO_QC.core.testing.testset import TestSet
        projects = [p for p in paths if p.lower().endswith(PJ.EXT)]
        if projects:
            self._open_project(projects[0])
            return
        weights = [p for p in paths if p.lower().endswith(WEIGHT_EXTS)]
        yamls = [p for p in paths if p.lower().endswith((".yaml", ".yml"))]
        videos = [p for p in paths if is_video(p)]
        folders = [p for p in paths if os.path.isdir(p)]
        if weights:
            self._add_models([dict(path=p) for p in weights])
            self.status.showMessage(f"Added {len(weights)} model(s)", 4000)
        if yamls:
            self.kind_ds.setChecked(True)
            self.ds_path.setText(yamls[0])
            self._load_data(yamls[0])
            self.nav.setCurrentRow(0)
        for f in folders:
            if TestSet.is_testset(f):
                self.kind_ts.setChecked(True)
                self._load_data(f)
                self.nav.setCurrentRow(0)
                continue
            m = QMenu(self)
            a_ds = m.addAction(f"Use “{os.path.basename(f)}” as the labelled dataset")
            a_src = m.addAction("Add its images to the test set")
            act = m.exec(self.cursor().pos())
            if act is a_ds:
                self.kind_ds.setChecked(True)
                self.ds_path.setText(f)
                self._load_data(f)
            elif act is a_src:
                self.kind_ts.setChecked(True)
                self._add_src_rows([("folder", f)])
            self.nav.setCurrentRow(0)
        if videos:
            m = QMenu(self)
            a_ts = m.addAction(f"Add {len(videos)} video(s) to the test set (unique frames)")
            a_an = m.addAction("Analyse the whole video(s) - behaviour, no labels needed")
            a_both = m.addAction("Both")
            act = m.exec(self.cursor().pos())
            if act in (a_ts, a_both):
                self.kind_ts.setChecked(True)
                self._add_src_rows([("video", p) for p in videos])
            if act in (a_an, a_both):
                have = {v["path"] for v in self.analysis_videos()}
                for p in videos:
                    if p not in have:
                        self._add_analysis_video(p)
            if act is not None:
                self.nav.setCurrentRow(0)
        self._changed()

    def _restore(self):
        """Reopen the last session in the background and show the start page. With a dataset open in Image QC:
        that dataset's own Testing session (or a new one on it, with its models)."""
        from AIO_QC.core.testing import project as PJ
        if self.qc is not None and not self.qc.unavailable():
            st = self._dataset_state()
            later(0, self, lambda: self._apply_state(st))
            later(0, self, lambda: self.nav.setCurrentRow(3 if (st.get("summary") or {}).get("best") else 0))
            return
        st = None
        if os.path.isfile(PJ.AUTOSAVE):
            try:
                st = PJ.load(PJ.AUTOSAVE)
            except Exception as _exc:
                _ignored(_exc)
                st = None
        if st is None:                                 # first run: carry over the earlier (pre-project) settings
            import json
            try:
                rows = json.loads(self.st.value("models", "[]") or "[]")
            except ValueError:
                rows = []
            if not rows:
                rows = [dict(path=p) for p in self.model_cfg.get("models", []) if os.path.isfile(p)]
            st = dict(name="Untitled", models=[r for r in rows if os.path.isfile(r.get("path", ""))])
            ts, ds = self.st.value("testset", ""), self.st.value("dataset", "")
            if ts and os.path.isdir(ts):
                st["data"] = dict(kind="testset", path=ts)
            elif ds and os.path.exists(ds):
                st["data"] = dict(kind="dataset", path=ds)
        else:
            pp = st.get("project_path")
            self._project_path = pp if pp and os.path.isfile(pp) else None
        later(0, self, lambda: self._apply_state(st))
        self._show_home()

    def _dataset_state(self):
        """The Testing project stored with the open dataset, else a fresh one on it with the dataset's models."""
        from AIO_QC.core.testing import project as PJ
        from AIO_QC.core import model_registry as R
        root = str(self.qc.session.root)
        p = PJ.dataset_project(root)
        if os.path.isfile(p):
            try:
                st = PJ.load(p)
                st["data"] = dict(st.get("data") or {}, kind="qc", path=root)
                return st
            except Exception as _exc:
                _ignored(_exc)
        models = [dict(path=m["path"], label=m.get("name", ""), imgsz=R.effective_imgsz(m),
                       class_map=m.get("class_map") or {})
                  for m in R.dataset_models(root) if m.get("enabled", True) and os.path.isfile(m.get("path", ""))]
        return dict(name=os.path.basename(root), data=dict(kind="qc", path=root), models=models)

    def use_qc_dataset(self):
        """Image QC opened another dataset (or Testing was opened from it): switch to it."""
        if self.qc is None or self.qc.unavailable():
            return
        if self.qc.is_this(self.data):
            return
        if self.job is not None:
            return
        self._apply_state(self._dataset_state())

    def set_model_cfg(self, cfg):
        self.model_cfg = dict(cfg or {})
        if not self.env_edit.text().strip():
            self.env_edit.setText(self.model_cfg.get("env_python", ""))
