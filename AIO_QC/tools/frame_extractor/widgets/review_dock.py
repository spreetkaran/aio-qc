# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Video review in the Frame Extractor: plays a playlist from Testing with every model's detections drawn on the video.

Testing only hands over the playlist (core.testing.review.ReviewPlaylist); this panel owns it from then on:
  * each video is analysed by a core.testing.stream.StreamAnalyser ahead of the playhead (CPU decoding with the
    player's own decoder, so frame numbers match), and the player draws the boxes of the exact frame on screen -
    smooth playback with the built-in engine or libmpv alike
  * balanced (adaptive step + interpolation + backfill), best quality (every frame; playback waits when needed) or a
    fixed step; while one video plays the next one is analysed once the current one is complete
  * models can be hidden, the confidence threshold changes live (all detections are kept, nothing is re-run)
  * the videos play one after another (in / out points respected, optional loop); at the end - or on End review -
    the video that was open before comes back at its frame

Boxes on frames the models did not see themselves (interpolated between two analysed frames, or held) are dashed.

Display: boxes coloured by CLASS (Image QC's palette — each class its own, clearly different colour, the same in every
video and session; with several models a numbered badge in the model's colour leads each label) or by model; labels
"class + confidence", "class" or off; a light fill; label size. The legend lists the classes with their colour and how
many are on the current frame — click one to hide / show it. "Save frame…" writes the frame at full resolution with
its boxes as a PNG.
"""
from __future__ import annotations

import os
import threading
import time

from AIO_QC.utils.qt_utils import later

from loguru import logger
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QFileDialog, QFormLayout,
                             QApplication, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton,
                             QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget)
from AIO_QC.core.diag import ignored as _ignored

MODE_TEXT = {"balanced": "Balanced", "best": "Best quality", "fixed": "Fixed step"}
STYLE_KEY = "fx/review/style"
LABEL_SIZES = (("Small", 0.85), ("Medium", 1.0), ("Large", 1.25), ("Extra large", 1.6))


def _swatch(color, text="", size=14):
    pm = QPixmap(size, size)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(color)
    if text:                                        # a model: the same dark numbered badge its labels carry
        from AIO_QC.tools.frame_extractor.widgets.det_paint import BADGE_BG, badge_ink
        p.setBrush(BADGE_BG)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(0, 0, size, size), 3, 3)
        p.fillRect(QRectF(1, size - 2.5, size - 2, 2), c)
        f = p.font()
        f.setPixelSize(int(size * 0.72))
        f.setBold(True)
        p.setFont(f)
        p.setPen(badge_ink(c))
        p.drawText(QRectF(0, -1, size, size), Qt.AlignmentFlag.AlignCenter, text)
    else:                                           # a class: its colour
        p.setBrush(c)
        p.setPen(QPen(c.darker(150), 1))
        p.drawRoundedRect(QRectF(0.5, 0.5, size - 1, size - 1), 3, 3)
    p.end()
    return QIcon(pm)
EVENT_COLORS = {"disagree": "#e53935", "flicker": "#fb8c00", "uncertain": "#fdd835"}


class ReviewStrip(QWidget):
    """Under the seek bar while reviewing: detections per frame of each model (one lane each), what is not analysed
    yet (grey), the events (coloured ticks) and the playhead. Click / drag to seek."""
    seek_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setMouseTracking(True)
        self.setToolTip("Detections per frame of each model · grey: not analysed yet · ticks: events "
                        "(red: models disagree, orange: flicker, yellow: uncertain)")
        self.counts, self.colors, self.events = {}, {}, []
        self.first, self.total, self.playhead = 0, 1, 0

    def set_data(self, counts, colors, events, first, total):
        self.counts, self.colors, self.events = counts, colors, events
        self.first, self.total = int(first), max(1, int(total))
        self.update()

    def set_playhead(self, idx):
        if idx != self.playhead:
            self.playhead = idx
            self.update()

    def _x(self, frame):
        return (frame + 0.5) / self.total * self.width()

    def paintEvent(self, _ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), self.palette().base())
        lanes = [k for k in self.counts if k in self.colors]
        top, lane_h = 6, (h - 8) / max(1, len(lanes))
        for li, k in enumerate(lanes):
            a = self.counts[k]
            if not len(a):
                continue
            col = QColor(self.colors[k])
            peak = max(1, int(a.max()))
            y0 = top + li * lane_h
            for x in range(w):
                f0 = int(x / w * self.total) - self.first
                f1 = int((x + 1) / w * self.total) - self.first
                f0, f1 = max(0, f0), min(len(a), max(f0 + 1, f1))
                if f0 >= len(a) or f1 <= 0:
                    continue
                seg = a[f0:f1]
                if (seg < 0).all():
                    p.fillRect(QRectF(x, y0, 1, lane_h), QColor(128, 128, 128, 60))
                    continue
                v = int(seg.max())
                if v > 0:
                    bh = max(1.0, (lane_h - 1) * v / peak)
                    p.fillRect(QRectF(x, y0 + lane_h - bh, 1, bh), col)
        for e in self.events:
            p.setPen(QPen(QColor(EVENT_COLORS.get(e.kind, "#888")), 2))
            x = self._x(e.frame)
            p.drawLine(int(x), 0, int(x), 5)
        p.setPen(QPen(self.palette().text().color(), 1))
        x = self._x(self.playhead)
        p.drawLine(int(x), 0, int(x), h)
        p.end()

    def mousePressEvent(self, ev):
        self._seek(ev.position().x())

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.LeftButton:
            self._seek(ev.position().x())

    def _seek(self, x):
        self.seek_requested.emit(max(0, min(self.total - 1, int(x / max(1, self.width()) * self.total))))


class HardFramesDialog(QDialog):
    """Send frames worth labelling - where the models disagree, flicker or hesitate - to Image QC."""

    def __init__(self, parent, specs, current_only_default=False):
        super().__init__(parent)
        self.setWindowTitle("Send hard frames to Image QC")
        f = QFormLayout(self)
        self.kinds = {}
        box = QVBoxLayout()
        from AIO_QC.core.testing.review_events import KIND_TEXT
        for k, t in KIND_TEXT.items():
            cb = QCheckBox(t)
            cb.setChecked(k != "uncertain")
            self.kinds[k] = cb
            box.addWidget(cb)
        f.addRow("Frames where", box)
        self.max_n = QSpinBox()
        self.max_n.setRange(1, 5000)
        self.max_n.setValue(50)
        self.max_n.setSuffix(" per video")
        f.addRow("At most", self.max_n)
        self.gap = QSpinBox()
        self.gap.setRange(0, 600)
        self.gap.setValue(1)
        self.gap.setSuffix(" s apart")
        f.addRow("At least", self.gap)
        self.model = QComboBox()
        self.model.addItem("No pre-labels", None)
        for s in specs:
            self.model.addItem(f"Pre-labels from {s.label}", s.key)
        if specs:
            self.model.setCurrentIndex(1)
        f.addRow("Labels", self.model)
        self.scope = QComboBox()
        self.scope.addItem("Every video of the playlist", "all")
        self.scope.addItem("This video only", "current")
        self.scope.setCurrentIndex(1 if current_only_default else 0)
        f.addRow("Videos", self.scope)
        f.addRow(QLabel("<small>Only frames already analysed count. Images go to the output folder, with their video "
                        "and frame remembered, and open in Image QC as a dataset.</small>"))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def options(self):
        return dict(kinds=[k for k, cb in self.kinds.items() if cb.isChecked()], max_n=self.max_n.value(),
                    gap_s=self.gap.value(), label_model=self.model.currentData(), scope=self.scope.currentData())


def ordinal(n):
    return "every frame" if n <= 1 else f"every {n}{'nd' if n == 2 else 'rd' if n == 3 else 'th'} frame"


class ReviewDock(QDockWidget):
    def __init__(self, win):
        super().__init__("Review playlist", win)
        self.setObjectName("reviewDock")
        self.win = win
        self.playlist = None
        self.active = False
        self.index = -1
        self.analysers = {}                 # item index -> StreamAnalyser
        self.visible_models = set()
        self.prev = None                    # (path, frame) open before the review
        self._last = (None, None)           # (frame, how) drawn last
        self._autoplay_at = None
        self._buffering = False
        self._advancing = False
        self._cache_path = ""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        self.head = QLabel()
        self.head.setWordWrap(True)
        v.addWidget(self.head)
        self.state = QLabel()
        self.state.setWordWrap(True)
        self.state.setObjectName("reviewState")
        v.addWidget(self.state)
        self.list = QListWidget()
        self.list.itemActivated.connect(lambda it: self.play_item(self.list.row(it)))
        self.list.itemDoubleClicked.connect(lambda it: self.play_item(self.list.row(it)))
        v.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(lambda _c=False: self.step(-1))
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(lambda _c=False: self.step(1))
        self.loop = QCheckBox("Loop")
        self.loop.toggled.connect(lambda on: setattr(self.playlist, "loop", bool(on)) if self.playlist else None)
        for x in (self.prev_btn, self.next_btn, self.loop):
            row.addWidget(x)
        row.addStretch(1)
        v.addLayout(row)
        self.models_box = QVBoxLayout()
        v.addWidget(QLabel("<b>Models</b>"))
        v.addLayout(self.models_box)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Confidence ≥"))
        self.conf = QSlider(Qt.Orientation.Horizontal)
        self.conf.setRange(1, 99)
        self.conf.valueChanged.connect(self._conf_changed)
        crow.addWidget(self.conf, 1)
        self.conf_label = QLabel()
        self.conf_label.setMinimumWidth(36)
        crow.addWidget(self.conf_label)
        v.addLayout(crow)
        # display: colours, labels, fill, size
        st = self._style_load()
        drow = QHBoxLayout()
        self.color_by = QComboBox()
        self.color_by.addItem("Colour by class", "class")
        self.color_by.addItem("Colour by model", "model")
        self.color_by.setCurrentIndex(max(0, self.color_by.findData(st["color_by"])))
        self.color_by.setToolTip("Class: every class its own, clearly different colour (as in Image QC), the same in "
                                 "every video; with several models a numbered badge in the model's colour leads each "
                                 "label.\nModel: one colour per model.")
        self.labels_cb = QComboBox()
        self.labels_cb.addItem("Class + confidence", "full")
        self.labels_cb.addItem("Class only", "class")
        self.labels_cb.addItem("No labels", "off")
        self.labels_cb.setCurrentIndex(max(0, self.labels_cb.findData(st["labels"])))
        drow.addWidget(self.color_by, 1)
        drow.addWidget(self.labels_cb, 1)
        v.addLayout(drow)
        d2 = QHBoxLayout()
        self.fill_cb = QCheckBox("Fill boxes")
        self.fill_cb.setChecked(bool(st["fill"]))
        self.fill_cb.setToolTip("A light fill in the box colour — easier to see small boxes on busy frames")
        self.size_cb = QComboBox()
        for t, k in LABEL_SIZES:
            self.size_cb.addItem(f"Labels: {t.lower()}", k)
        self.size_cb.setCurrentIndex(max(0, self.size_cb.findData(st["size"])))
        d2.addWidget(self.fill_cb)
        d2.addWidget(self.size_cb, 1)
        v.addLayout(d2)
        d3 = QHBoxLayout()
        self.clean_cb = QCheckBox("Clean while playing")
        self.clean_cb.setToolTip("While the video plays: thin outlines, no labels (hover a box for its details); "
                                 "labels come back when you pause")
        self.shade_cb = QCheckBox("Shade by confidence")
        self.shade_cb.setToolTip("Less sure boxes are fainter")
        self.glide_cb = QCheckBox("Glide labels")
        self.glide_cb.setToolTip("A label that has to move slides to its new place instead of jumping")
        for cb, k in ((self.clean_cb, "clean"), (self.shade_cb, "shade"), (self.glide_cb, "glide")):
            cb.setChecked(bool(st[k]))
            cb.toggled.connect(lambda _on: self._style_changed())
            d3.addWidget(cb)
        d3.addStretch(1)
        v.addLayout(d3)
        self._playing = False
        self._drawn_labels = True
        for wdg in (self.color_by, self.labels_cb, self.size_cb):
            wdg.currentIndexChanged.connect(lambda _i: self._style_changed())
        self.fill_cb.toggled.connect(lambda _on: self._style_changed())
        # legend: classes, their colour, how many on this frame; click hides / shows
        lrow = QHBoxLayout()
        self.legend_label = QLabel("<b>Classes</b>")
        lrow.addWidget(self.legend_label, 1)
        self.class_btn = QToolButton()                     # kept for the shortcut menu: Show all / only …
        self.class_btn.setText("Show all")
        self.class_btn.setToolTip("Show every class again")
        self.class_btn.clicked.connect(self._show_all_classes)
        lrow.addWidget(self.class_btn)
        v.addLayout(lrow)
        self.legend = QListWidget()
        self.legend.setMaximumHeight(150)
        self.legend.setToolTip("Untick a class to hide it · the number is how many are on the current frame")
        self.legend.itemChanged.connect(self._legend_toggled)
        v.addWidget(self.legend)
        self.class_menu = QMenu(self)
        self.class_menu.aboutToShow.connect(self._fill_classes)
        self._legend_names: list[str] = []
        self._legend_counts: dict = {}
        self._colored_models: set = set()
        erow = QHBoxLayout()
        self.events_label = QLabel("<b>Events</b>")
        erow.addWidget(self.events_label, 1)
        self.ev_prev = QPushButton("◀")
        self.ev_prev.setToolTip("Previous event (Alt+[)")
        self.ev_prev.clicked.connect(lambda _c=False: self.jump_event(-1))
        self.ev_next = QPushButton("▶")
        self.ev_next.setToolTip("Next event (Alt+])")
        self.ev_next.clicked.connect(lambda _c=False: self.jump_event(1))
        erow.addWidget(self.ev_prev)
        erow.addWidget(self.ev_next)
        v.addLayout(erow)
        self.ev_list = QListWidget()
        self.ev_list.setMaximumHeight(130)
        self.ev_list.itemActivated.connect(self._event_activated)
        self.ev_list.itemClicked.connect(self._event_activated)
        v.addWidget(self.ev_list)
        brow = QHBoxLayout()
        self.hard_btn = QPushButton("Send hard frames to Image QC…")
        self.hard_btn.setToolTip("Frames where the models disagree, flicker or hesitate - with pre-labels - as an "
                                 "Image QC dataset")
        self.hard_btn.clicked.connect(self.send_hard_frames)
        brow.addWidget(self.hard_btn)
        self.shot_btn = QPushButton("Save frame…")
        self.shot_btn.setToolTip("This frame at full resolution with its boxes, as a PNG")
        self.shot_btn.clicked.connect(self.save_frame)
        brow.addWidget(self.shot_btn)
        self.csv_btn = QPushButton("CSV…")
        self.csv_btn.setToolTip("This video's detections (every analysed frame) as a CSV file")
        self.csv_btn.clicked.connect(self.export_csv)
        brow.addWidget(self.csv_btn)
        v.addLayout(brow)
        self.end_btn = QPushButton("End review")
        self.end_btn.setToolTip("Stop the review and go back to the video that was open before")
        self.end_btn.clicked.connect(lambda _c=False: self.stop(back=True))
        v.addWidget(self.end_btn)
        self.setWidget(w)
        self.hidden_classes = set()
        self.events = []
        self._ev_job = None                  # (thread, result box) computing summary + events off the UI thread
        self._ev_key = None
        self.strip = ReviewStrip()
        self.strip.hide()
        self.strip.seek_requested.connect(lambda f: self.win.goto_frame(f))
        lay = getattr(win, "canvas_layout", None)
        if lay is not None:
            lay.insertWidget(lay.indexOf(win.strip) + 1, self.strip)
        for n in range(1, 10):                               # Alt+1 … Alt+9: show / hide model n
            act = QAction(f"Show / hide review model {n}", win)
            act.setObjectName(f"review_model_{n}")
            act.setShortcut(QKeySequence(f"Alt+{n}"))
            act.triggered.connect(lambda _c=False, n=n: self.toggle_model_n(n))
            win.addAction(act)
        for key, d, text in (("Alt+]", 1, "Next review event"), ("Alt+[", -1, "Previous review event")):
            act = QAction(text, win)
            act.setObjectName(f"review_event_{'next' if d > 0 else 'prev'}")
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(lambda _c=False, d=d: self.jump_event(d))
            win.addAction(act)
        self.timer = QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ start / stop
    def start(self, playlist):
        from AIO_QC.core.testing.runner import default_cache_dir
        if self.active:
            self.stop(back=False)
        problems = playlist.validate()
        if problems:
            raise ValueError("; ".join(problems))
        w = self.win
        if w.engine is not None and w.video_path:
            self.prev = (w.video_path, w.current_frame)
        if w.is_playing:
            w.pause()
        if w.actions_.detect.isChecked():
            w.actions_.detect.setChecked(False)          # the paused-frame Detect overlay steps aside
            w.canvas.set_detections([])
        self.playlist = playlist
        self._cache_path = playlist.cache_path or os.path.join(default_cache_dir(), "video_review.db")
        self.active = True
        self.visible_models = {s.key for s in playlist.specs}
        self._build_models()
        self.conf.blockSignals(True)
        self.conf.setValue(int(round(playlist.conf * 100)))
        self.conf.blockSignals(False)
        self._conf_changed(self.conf.value())
        self.loop.setChecked(bool(playlist.loop))
        self.list.clear()
        for it in playlist.items:
            li = QListWidgetItem(self._item_text(it, "queued"))
            li.setToolTip(it.path)
            self.list.addItem(li)
        self.head.setText(f"<b>{len(playlist.items)} video(s)</b> · {len(playlist.specs)} model(s) · "
                          f"{MODE_TEXT[playlist.mode]}" + (f" (every {playlist.stride})" if playlist.mode == "fixed"
                                                         else "") + f"<br><small>from {playlist.source}</small>")
        self.hidden_classes = set()
        self._legend_names, self._legend_counts = [], {}
        self.legend.clear()
        self._style_changed(save=False)
        self.events, self._ev_key = [], None
        self.ev_list.clear()
        self.strip.show()
        self.show()
        self.raise_()
        self.timer.start()
        logger.info(f"video review: {len(playlist.items)} videos, {len(playlist.specs)} models, {playlist.mode}")
        self.play_item(0)

    def stop(self, back=True):
        if not self.active:
            return
        self.active = False
        self.timer.stop()
        for a in self.analysers.values():
            a.stop(wait=0)
        self.analysers.clear()
        self.win.canvas.set_detections([])
        self.win.canvas.set_detection_style(labels=True, thin=False)   # the paused-frame Detect overlay: labelled
        self.strip.hide()
        self.state.setText("Review ended.")
        prev, self.prev = self.prev, None
        if back and prev is not None and os.path.isfile(prev[0]):
            self.win.open_at(prev[0], frame=prev[1])
        logger.info("video review ended")

    def shutdown(self):
        """Window closing: stop analysis threads (they are daemons; this also stops their decoders)."""
        self.active = False
        self.timer.stop()
        for a in self.analysers.values():
            a.stop(wait=1.0)
        self.analysers.clear()

    # ------------------------------------------------------------------ playlist
    def _item_text(self, it, state):
        rng = ""
        if it.start is not None or it.end is not None:
            from AIO_QC.tools.frame_extractor.core.sources import format_time
            rng = f"  [{format_time(it.start or 0)} → {format_time(it.end) if it.end is not None else 'end'}]"
        return f"{it.name}{rng}   · {state}"

    def _set_item_state(self, i, state):
        if 0 <= i < self.list.count():
            self.list.item(i).setText(self._item_text(self.playlist.items[i], state))

    def step(self, d):
        if not self.active:
            return
        n = len(self.playlist.items)
        j = self.index + d
        if j >= n:
            if self.playlist.loop:
                j = 0
            else:
                self.state.setText("Playlist finished.")
                self.stop(back=True)
                return
        if j < 0:
            j = 0
        self.play_item(j)

    def play_item(self, i):
        if not self.active or not (0 <= i < len(self.playlist.items)):
            return
        if 0 <= self.index < len(self.playlist.items) and self.index != i:
            self._set_item_state(self.index, "done" if self.index < i else "queued")
        self.index = i
        self.list.setCurrentRow(i)
        it = self.playlist.items[i]
        for k in [k for k in self.analysers if k not in (i, i + 1)]:
            self.analysers.pop(k).stop(wait=0)               # keep the current and the next one only
        self._set_item_state(i, "opening")
        self.events, self._ev_key = [], None
        self.ev_list.clear()
        self._autoplay_at = None
        self._last = (None, None)
        self.win.load_video(it.path)

    def on_video_loaded(self, path):
        """The player opened a video: start (or reuse) its analysis, go to the in point, play once ready."""
        if not self.active or not (0 <= self.index < len(self.playlist.items)):
            return
        it = self.playlist.items[self.index]
        if os.path.normcase(os.path.abspath(path)) != os.path.normcase(os.path.abspath(it.path)):
            self.stop(back=False)                            # another video was opened by hand: the review ends
            return
        a = self._analyser(self.index)
        e = self.win.engine
        first = e.index_of(it.start) if it.start else 0
        if first:
            self.win.goto_frame(first)
        a.set_playhead(first, playing=False)
        self._set_item_state(self.index, "analysing")
        self._autoplay_at = time.monotonic()

    def _analyser(self, i):
        from AIO_QC.core.testing.stream import StreamAnalyser
        a = self.analysers.get(i)
        if a is None:
            pl, it = self.playlist, self.playlist.items[i]
            a = StreamAnalyser(pl.specs, it.path, self._cache_path, mode=pl.mode, stride=pl.stride,
                               lead_s=pl.lead_s, backend=getattr(self.win, "backend", "auto"), start=it.start,
                               end=it.end, log=lambda m: logger.info(f"review: {m}"))
            a.start()
            self.analysers[i] = a
        return a

    def on_video_end(self):
        """End of the file: next video. True = handled."""
        if not self.active or self._advancing:
            return False
        self._advancing = True
        try:
            self._set_item_state(self.index, "done")
            later(0, self, lambda: self.step(1))
        finally:
            self._advancing = False
        return True

    # ------------------------------------------------------------------ frames
    def current(self):
        return self.analysers.get(self.index) if self.active else None

    def on_frame(self, idx, playing):
        """Every frame the player shows (built-in engine and libmpv): draw its detections."""
        a = self.current()
        if a is None:
            return
        w = self.win
        speed = w.SPEEDS[w.speed_cb.currentIndex()] if hasattr(w, "speed_cb") else 1.0
        a.set_playhead(idx, playing=playing, speed=speed)
        it = self.playlist.items[self.index]
        if it.end is not None and playing and w.engine is not None and idx >= w.engine.index_of(it.end):
            w.pause()
            self.on_video_end()
            return
        self.strip.set_playhead(idx)
        self._playing = bool(playing)
        per, how = a.detections(idx)
        if not playing and how != "exact":
            a.request_exact(idx)
        self._draw(per, how)
        self._last = (idx, how)

    def _draw(self, per, how):
        pl = self.playlist
        a = self.current()
        boxes = []
        thr = self.conf.value() / 100.0
        multi = len(self.visible_models) > 1
        dashed = how in ("interpolated", "held")
        by_class = self.color_by.currentData() == "class"
        lab_mode = self.labels_cb.currentData()
        from AIO_QC.tools.frame_extractor.widgets.det_paint import ClassColors, label_text
        cc = ClassColors.shared()
        counts = {}
        for i, s in enumerate(pl.specs):
            if s.key not in per:
                continue
            names = a.names(s.key) if a else {}
            if by_class and s.key not in self._colored_models and names:
                cc.ensure([names[k] for k in sorted(names)])          # a model's first classes: most distinct
                self._colored_models.add(s.key)
            mcol = QColor(pl.colors[i] if i < len(pl.colors) else "#e53935")
            shown = s.key in self.visible_models
            from AIO_QC.tools.frame_extractor.widgets.det_paint import classifier_answer
            best = classifier_answer(per[s.key])
            if best is not None:                         # a classifier: its answer for the frame, as a pill
                x1, y1, x2, y2, c, k = (float(v) for v in best)
                name = names.get(int(k), str(int(k)))
                if shown:
                    counts[name] = counts.get(name, 0) + 1
                if shown and name not in self.hidden_classes:          # below the threshold: faint ("unsure")
                    text = name if lab_mode in ("off", "class") else label_text(name, c)
                    badge = (str(i + 1), mcol, s.label) if multi else None
                    boxes.append((x1, y1, x2, y2, text, cc.get(name) if by_class else mcol, dashed or c < thr,
                                  badge, c, "class"))
                continue
            for x1, y1, x2, y2, c, k in per[s.key]:
                if c < thr:
                    continue
                name = names.get(int(k), str(int(k)))
                if shown:
                    counts[name] = counts.get(name, 0) + 1
                if not shown or name in self.hidden_classes:
                    continue
                text = "" if lab_mode == "off" else (name if lab_mode == "class" else label_text(name, c))
                badge = (str(i + 1), mcol, s.label) if multi and by_class and lab_mode != "off" else None
                boxes.append((float(x1), float(y1), float(x2), float(y2), text,
                              cc.get(name) if by_class else mcol, dashed, badge, float(c)))
        clean = self.clean_cb.isChecked() and self._playing
        self.win.canvas.set_detection_style(labels=lab_mode != "off" and not clean, thin=clean)
        self._drawn_labels = not clean
        self.win.canvas.set_detections(boxes)
        self.win.pixel_label.setText(f"{len(boxes)} detection(s)" + (" · interpolated" if dashed else ""))
        self._update_legend(counts)

    # ------------------------------------------------------------------ display style / legend
    def _style_load(self):
        from AIO_QC.utils import app_settings
        d = {"color_by": "class", "labels": "full", "fill": False, "size": 1.0, "clean": False, "shade": False,
             "glide": True}
        try:
            raw = app_settings().value(STYLE_KEY, "") or ""
            if raw:
                import json
                got = json.loads(raw)
                if isinstance(got, dict):
                    d.update({k: got[k] for k in d if k in got})
        except (ValueError, TypeError):
            pass
        return d

    def _style_changed(self, save=True):
        st = {"color_by": self.color_by.currentData(), "labels": self.labels_cb.currentData(),
              "fill": self.fill_cb.isChecked(), "size": self.size_cb.currentData(), "clean": self.clean_cb.isChecked(),
              "shade": self.shade_cb.isChecked(), "glide": self.glide_cb.isChecked()}
        if save:
            import json
            from AIO_QC.utils import app_settings
            app_settings().setValue(STYLE_KEY, json.dumps(st))
        self.win.canvas.set_detection_style(fill=st["fill"], labels=st["labels"] != "off", scale=st["size"],
                                            shade=st["shade"], glide=st["glide"])
        self._legend_names = []                                   # swatches follow the colour choice
        if self.active:
            self.redraw()

    def _update_legend(self, counts):
        names = self._all_classes()
        by_class = self.color_by.currentData() == "class"
        if names != self._legend_names:
            from AIO_QC.tools.frame_extractor.widgets.det_paint import ClassColors
            cc = ClassColors.shared()
            self.legend.blockSignals(True)
            self.legend.clear()
            for n in names:
                it = QListWidgetItem(n)
                it.setData(Qt.ItemDataRole.UserRole, n)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Unchecked if n in self.hidden_classes else Qt.CheckState.Checked)
                if by_class:
                    it.setIcon(_swatch(cc.get(n)))
                self.legend.addItem(it)
            self.legend.blockSignals(False)
            self._legend_names, self._legend_counts = list(names), {}
        self.legend.blockSignals(True)                               # ticks follow hidden classes, however hidden
        for r in range(self.legend.count()):
            it = self.legend.item(r)
            want = Qt.CheckState.Unchecked if it.data(Qt.ItemDataRole.UserRole) in self.hidden_classes else \
                Qt.CheckState.Checked
            if it.checkState() != want:
                it.setCheckState(want)
        self.legend.blockSignals(False)
        if counts != self._legend_counts:
            self.legend.blockSignals(True)
            for r in range(self.legend.count()):
                it = self.legend.item(r)
                n = it.data(Qt.ItemDataRole.UserRole)
                k = counts.get(n, 0)
                it.setText(f"{n}   ·  {k}" if k else n)
            self.legend.blockSignals(False)
            self._legend_counts = dict(counts)
        hidden = len(self.hidden_classes & set(names))
        self.legend_label.setText(f"<b>Classes</b> {len(names) - hidden} of {len(names)} shown" if hidden else
                                  f"<b>Classes</b> {len(names)}" if names else "<b>Classes</b>")
        self.class_btn.setEnabled(bool(hidden))

    def _legend_toggled(self, it):
        n = it.data(Qt.ItemDataRole.UserRole)
        if n is not None:
            self._toggle_class(n, it.checkState() == Qt.CheckState.Checked)

    # ------------------------------------------------------------------ save frame
    def save_frame(self):
        if not self.active or self.win.current is None:
            return
        img = self.win.canvas.frame_with_detections()
        if img is None:
            img = self.win.canvas.grab().toImage()          # e.g. a player that keeps frames on the GPU
        it = self.playlist.items[self.index]
        stem = os.path.splitext(os.path.basename(it.path))[0]
        default = os.path.join(self.win.output_dir or os.path.dirname(it.path),
                               f"{stem}_frame{self.win.current_frame:06d}_detections.png")
        p, _ = QFileDialog.getSaveFileName(self, "Frame with detections", default, "PNG (*.png);;JPEG (*.jpg)")
        if not p:
            return
        if not img.save(p):
            QMessageBox.warning(self, "Save frame", f"Could not write\n{p}")
            return
        self.win.flash_status(f"Frame saved: {p}", 5000)

    def redraw(self):
        a = self.current()
        if a is not None and self.win.current is not None:
            per, how = a.detections(self.win.current_frame)
            self._draw(per, how)
            self._last = (self.win.current_frame, how)

    # ------------------------------------------------------------------ models / threshold
    def _build_models(self):
        while self.models_box.count():
            x = self.models_box.takeAt(0).widget()
            if x is not None:
                x.deleteLater()
        for i, s in enumerate(self.playlist.specs):
            cb = QCheckBox(s.label)
            col = self.playlist.colors[i] if i < len(self.playlist.colors) else "#e53935"
            cb.setIcon(_swatch(col, str(i + 1)))                 # the badge its labels carry
            cb.setStyleSheet(f"QCheckBox::indicator:checked {{ background: {col}; border: 1px solid {col}; }}")
            cb.setChecked(True)
            cb.setToolTip(f"{s.path}\nAlt+{i + 1} shows / hides it")
            cb.toggled.connect(lambda on, k=s.key: self._toggle_model(k, on))
            self.models_box.addWidget(cb)

    def toggle_model_n(self, n):
        """Alt+n: the n-th model's tick in the panel."""
        if not self.active or not (1 <= n <= self.models_box.count()):
            return
        cb = self.models_box.itemAt(n - 1).widget()
        if cb is not None:
            cb.setChecked(not cb.isChecked())

    def _toggle_model(self, key, on):
        (self.visible_models.add if on else self.visible_models.discard)(key)
        self.redraw()

    def _conf_changed(self, v):
        self.conf_label.setText(f"{v / 100:.2f}")
        if self.active:
            self.redraw()
            self._ev_key = None                  # events depend on the threshold: recompute

    # ------------------------------------------------------------------ classes
    def _all_classes(self):
        a = self.current()
        out = set()
        if a is not None:
            for s in self.playlist.specs:
                out |= set(a.names(s.key).values())
        return sorted(out)

    def _fill_classes(self):
        m = self.class_menu
        m.clear()
        names = self._all_classes()
        if not names:
            m.addAction("(classes appear once the models are loaded)").setEnabled(False)
            return
        m.addAction("Show all", self._show_all_classes)
        m.addSeparator()
        for n in names:
            act = m.addAction(n)
            act.setCheckable(True)
            act.setChecked(n not in self.hidden_classes)
            act.toggled.connect(lambda on, n=n: self._toggle_class(n, on))

    def _toggle_class(self, name, on):
        (self.hidden_classes.discard if on else self.hidden_classes.add)(name)
        self.redraw()

    def _show_all_classes(self):
        self.hidden_classes.clear()
        self._legend_names = []                              # rebuild the ticks
        self.redraw()

    # ------------------------------------------------------------------ events / strip
    def _update_events(self, a):
        """Recompute detections-per-frame and events in the background when the analysis grew or the threshold
        changed (at most once a second); show the result when it is ready."""
        if self._ev_job is not None:
            th, box = self._ev_job
            if th.is_alive():
                return
            self._ev_job = None
            if box.get("idx") == self.index and "events" in box:
                self.events = box["events"]
                self._show_events(box)
            return
        cov = round(a.coverage(), 2)
        key = (self.index, cov, self.conf.value(), a.stride)
        if key == self._ev_key or not a.ready.is_set():
            return
        if self._ev_key is not None and time.monotonic() - getattr(self, "_ev_t", 0) < 1.0:
            return
        self._ev_key, self._ev_t = key, time.monotonic()
        from AIO_QC.core.testing.review_events import find_events, summarize
        store = {k: dict(v) for k, v in a._store.items() if k not in a._errors}
        names = {k: a.names(k) for k in store}
        first, last, fps, thr, idx = a.first, a.last, a.fps, self.conf.value() / 100.0, self.index
        box = {"idx": idx}

        def work():
            try:
                box["counts"] = summarize(store, first, last, thr)
                box["events"] = find_events(store, names, first, last, fps, thr)
                box["first"] = first
            except Exception as e:                      # never break the review over a statistics problem
                logger.warning(f"review events: {e}")
                box["events"] = []
        th = threading.Thread(target=work, name="review-events", daemon=True)
        th.start()
        self._ev_job = (th, box)

    def _show_events(self, box):
        from AIO_QC.core.testing.review_events import KIND_TEXT
        a = self.current()
        if a is None:
            return
        colors = {s.key: (self.playlist.colors[i] if i < len(self.playlist.colors) else "#e53935")
                  for i, s in enumerate(self.playlist.specs) if s.key in self.visible_models}
        total = self.win.frame_count or (a.last + 1)
        self.strip.set_data(box.get("counts", {}), colors, self.events, box.get("first", 0), total)
        cur = self.ev_list.currentRow()
        self.ev_list.clear()
        from AIO_QC.tools.frame_extractor.core.sources import format_time
        for e in self.events[:2000]:
            it = QListWidgetItem(f"{format_time(a.fps and e.frame / a.fps)[:-4]}  {KIND_TEXT[e.kind]} · {e.text}")
            it.setData(Qt.ItemDataRole.UserRole, e.frame)
            it.setForeground(QColor(EVENT_COLORS.get(e.kind, "#888")).darker(130))
            self.ev_list.addItem(it)
        if 0 <= cur < self.ev_list.count():
            self.ev_list.setCurrentRow(cur)
        n = {k: sum(1 for e in self.events if e.kind == k) for k in KIND_TEXT}
        self.events_label.setText("<b>Events</b> " + " · ".join(
            f"<span style='color:{EVENT_COLORS[k]}'>■</span> {v}" for k, v in n.items()))

    def _event_activated(self, it):
        f = it.data(Qt.ItemDataRole.UserRole)
        if f is not None:
            if self.win.is_playing:
                self.win.pause()
            self.win.goto_frame(int(f))

    def jump_event(self, d):
        if not self.active or not self.events:
            return
        cur = self.win.current_frame
        if d > 0:
            e = next((e for e in self.events if e.frame > cur), None)
        else:
            e = next((e for e in reversed(self.events) if e.frame < cur), None)
        if e is not None:
            if self.win.is_playing:
                self.win.pause()
            self.win.goto_frame(e.frame)
            for r in range(self.ev_list.count()):
                if self.ev_list.item(r).data(Qt.ItemDataRole.UserRole) == e.frame:
                    self.ev_list.setCurrentRow(r)
                    break

    # ------------------------------------------------------------------ outputs
    def export_csv(self):
        a = self.current()
        if a is None:
            return
        it = self.playlist.items[self.index]
        p, _ = QFileDialog.getSaveFileName(self, "Detections as CSV",
                                           os.path.splitext(it.path)[0] + "_detections.csv", "CSV (*.csv)")
        if not p:
            return
        import csv
        try:
            with open(p, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["frame", "time_s", "model", "class", "confidence", "x1", "y1", "x2", "y2"])
                for s in self.playlist.specs:
                    names = a.names(s.key)
                    for i in sorted(a._store.get(s.key, {})):
                        for x1, y1, x2, y2, c, k in a._store[s.key][i]:
                            wr.writerow([i, f"{i / a.fps:.4f}", s.label, names.get(int(k), int(k)), f"{c:.4f}",
                                         f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])
        except OSError as e:
            QMessageBox.warning(self, "CSV", str(e))
            return
        self.win.flash_status(f"Detections saved: {p}", 5000)

    def send_hard_frames(self):
        """Frames where the models disagree / flicker / hesitate, with pre-labels, as an Image QC dataset."""
        if not self.active:
            return
        dlg = HardFramesDialog(self, self.playlist.specs)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        o = dlg.options()
        if not o["kinds"]:
            return
        pl, thr = self.playlist, self.conf.value() / 100.0
        idxs = [self.index] if o["scope"] == "current" else list(range(len(pl.items)))
        out_root = os.path.join(self.win.output_dir, f"review_{time.strftime('%Y%m%d_%H%M%S')}")
        jobs = []
        for i in idxs:
            a = self.analysers.get(i)
            live = None
            if a is not None:
                live = ({k: dict(v) for k, v in a._store.items()}, {s.key: a.names(s.key) for s in pl.specs},
                        a.fps, a.first, a.last)
            jobs.append((pl.items[i], live))
        cache_path, specs, backend = self._cache_path, list(pl.specs), getattr(self.win, "backend", "auto")
        from AIO_QC.widgets import tasks
        stop = threading.Event()
        task = tasks.start("Hard frames → Image QC", cancel=stop.set, owner=self)
        box = {"done": 0, "total": len(jobs), "paths": [], "error": None}

        def work():
            from AIO_QC.core.testing.review_events import export_frames, find_events, load_store, pick_hard_frames
            from AIO_QC.tools.frame_extractor.core.sources import open_source
            try:
                for item, live in jobs:
                    if stop.is_set():
                        break
                    if live is None:
                        store, names = load_store(specs, item.path, cache_path)
                        src = open_source(item.path, backend, hw_accel=False)
                        fps, n = src.fps, src.frame_count
                        src.release()
                        first, last = 0, n - 1
                    else:
                        store, names, fps, first, last = live
                    evs = find_events(store, names, first, last, fps, thr, kinds=o["kinds"])
                    picks = pick_hard_frames(evs, o["kinds"], o["max_n"], max(1, int(o["gap_s"] * (fps or 25))))
                    box["paths"] += export_frames(item.path, [e.frame for e in picks], out_root, store, names,
                                                  o["label_model"], thr, backend, cancel=stop.is_set)
                    box["done"] += 1
            except Exception as e:
                _ignored(e)
                box["error"] = f"{type(e).__name__}: {e}"

        th = threading.Thread(target=work, name="review-hard-frames", daemon=True)
        th.start()

        def poll():
            task.update(box["done"] / max(1, box["total"]), f"{len(box['paths'])} frame(s)")
            if th.is_alive():
                later(200, self, poll)
                return
            task.finish()
            if box["error"]:
                QMessageBox.warning(self, "Hard frames", box["error"])
                return
            if not box["paths"]:
                QMessageBox.information(self, "Hard frames", "No frames matched (only analysed frames count).")
                return
            shell = getattr(QApplication.instance(), "shell", None)
            self.win.flash_status(f"{len(box['paths'])} hard frame(s) saved in {out_root}", 8000)
            if shell is not None:
                shell.open_in_qc(out_root)
        later(200, self, poll)

    # ------------------------------------------------------------------ status / buffering / prefetch
    def _tick(self):
        a = self.current()
        if a is None:
            return
        w = self.win
        st = a.status()
        idx = w.current_frame
        need = max(2, int(min(self.playlist.lead_s * 0.5, 1.5) * max(1.0, a.fps)))
        remaining = max(0, a.last - idx)
        ready = st["ahead_frames"] >= min(need, remaining) or st["coverage"] >= 0.999
        if st["phase"] == "failed":
            self._set_item_state(self.index, "failed")
            self.state.setText(f"<span style='color:#d03b3b'>Analysis failed: {st['error']}</span>")
            if self._autoplay_at is not None and not w.is_playing:
                self._autoplay_at = None
                later(1500, self, lambda: self.step(1))      # skip a video that can't be analysed
            return
        # start playing once the first seconds are ready (at most a few seconds' wait)
        if self._autoplay_at is not None and a.ready.is_set():
            if ready or time.monotonic() - self._autoplay_at > 8.0:
                self._autoplay_at = None
                self._set_item_state(self.index, "playing")
                if not w.is_playing and w.engine is not None:
                    w.play()
        # best quality: wait while the analysis catches up (like video buffering)
        if self.playlist.mode == "best" and w.is_playing and not ready and st["coverage"] < 0.999:
            self._buffering = True
            w.pause()
        elif self._buffering and ready:
            self._buffering = False
            w.play()
        # a paused frame whose own detections just arrived — or, in Clean mode, its labels come back
        if not w.is_playing and self._last[0] == idx and self._last[1] != "exact" and a.has(idx):
            self.redraw()
        elif not w.is_playing and self._playing:
            self._playing = False
            if not self._drawn_labels:
                self.redraw()
        # analyse the next video once this one is complete
        nxt = self.index + 1
        if st["coverage"] >= 0.999 and nxt < len(self.playlist.items) and nxt not in self.analysers:
            self._analyser(nxt)
            self._set_item_state(nxt, "analysing (next)")
        self.state.setText(self._status_text(st))
        self._update_events(a)

    def _status_text(self, st):
        if self._buffering:
            return f"<b>Analysing…</b> {st['coverage'] * 100:.0f} % of this video · playback waits (Best quality)"
        if st["phase"] == "starting":
            return "Loading the models…"
        bits = []
        how = ordinal(st["stride"])
        if st["coverage"] >= 0.999:
            bits.append("<b>every frame analysed</b>")
        else:
            bits.append(f"<b>{how}</b>" + (" + interpolated" if st["stride"] > 1 else ""))
            if st["phase"] == "backfill":
                bits.append(f"backfilling {st['coverage'] * 100:.0f} %")
            else:
                bits.append(f"{st['ahead_s']:.1f} s ahead")
        if st["infer_fps"]:
            bits.append(f"{st['infer_fps']:.0f} frames / s")
        if st["errors"]:
            bits.append(f"<span style='color:#d03b3b'>{len(st['errors'])} model(s) failed</span>")
        return f"Video {self.index + 1} / {len(self.playlist.items)} · " + " · ".join(bits)
