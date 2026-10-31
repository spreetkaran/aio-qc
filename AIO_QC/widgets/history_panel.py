# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
History panel (dock, Ctrl+H) — everything that happened to the file on screen, and its saved versions.

Changes   every write, review mark, outside change, undo and revert: when · who · tool · what (core.history)
Versions  the originals the backup store kept before each operation that rewrote the file. "Preview" draws that
          version's shapes over the image (dashed) and says what differs from now; "Revert" restores it exactly —
          after storing the current file, so the revert itself can be reverted.
"""
from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from AIO_QC.core import annodiff
from AIO_QC.core.history import ACTIONS, revert, version_bytes, versions
from AIO_QC.utils import hexc, newButton, style as apply_style

ROLE = Qt.ItemDataRole.UserRole


def _when(ts: float) -> str:
    d = time.time() - ts
    if d < 60:
        return "just now"
    if d < 3600:
        return f"{int(d // 60)} min ago"
    if d < 86400 and time.localtime(ts).tm_yday == time.localtime().tm_yday:
        return time.strftime("today %H:%M", time.localtime(ts))
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


class HistoryPanel(QWidget):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.key = None
        self._overlay = False                 # our dashed preview is on the image (clear only what we drew)
        self.title = QLabel("No file")
        self.title.setWordWrap(True)
        apply_style(self.title, "font-weight:bold;")
        self.events = QTreeWidget()
        self.events.setHeaderLabels(["When", "Who", "What"])
        self.events.setRootIsDecorated(False)
        self.events.setUniformRowHeights(True)
        self.events.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.vers = QTreeWidget()
        self.vers.setHeaderLabels(["Saved version", "Kept before"])
        self.vers.setRootIsDecorated(False)
        self.vers.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.vers.itemSelectionChanged.connect(self._sel)
        self.vers.itemDoubleClicked.connect(lambda *_: self._preview())
        self.diff = QLabel("")
        self.diff.setWordWrap(True)
        apply_style(self.diff, "color:@muted;")
        self.preview_btn = newButton("Preview", slot=self._preview,
                                     tip="Draw this version's shapes over the image (dashed) and list what differs")
        self.revert_btn = newButton("Revert to this version", icon="undo", slot=self._revert,
                                    tip="Restore exactly this version (the current file is stored first)")
        for b in (self.preview_btn, self.revert_btn):
            b.setEnabled(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.title)
        split = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(QLabel("Changes"))
        tl.addWidget(self.events, 1)
        bot = QWidget()
        bl = QVBoxLayout(bot)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(QLabel("Saved versions"))
        bl.addWidget(self.vers, 1)
        bl.addWidget(self.diff)
        row = QHBoxLayout()
        row.addWidget(self.preview_btn)
        row.addWidget(self.revert_btn)
        row.addStretch(1)
        bl.addLayout(row)
        split.addWidget(top)
        split.addWidget(bot)
        lay.addWidget(split, 1)
        self._timer = QTimer(self, singleShot=True, interval=250, timeout=self.refresh)

    # ------------------------------------------------------------------ follow the current file
    def set_key(self, key):
        if key != self.key:
            self._overlay = False             # a new image clears the canvas overlay by itself
        self.key = key
        if self.isVisible():
            self._timer.start()

    def touched(self, keys):
        if self.key in set(keys) and self.isVisible():
            self._timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        self._timer.start()

    def refresh(self):
        s = self.app.session
        self.events.clear()
        self.vers.clear()
        self.diff.setText("")
        if s is None or not self.key or self.key not in s.by_key:
            self.title.setText("No file")
            return
        sample = s.by_key[self.key]
        self.title.setText(sample.display)
        ann = sample.ann_path
        for e in s.history.for_file(self.key):          # events are logged per sample key
            what = e["what"] or ACTIONS.get(e["action"], e["action"])
            it = QTreeWidgetItem([_when(e["ts"]), e["user"], f"{e['tool']}: {what}"])
            it.setToolTip(2, f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e['ts']))}\n{e['tool']} · "
                             f"{ACTIONS.get(e['action'], e['action'])}\n{what}")
            if e["action"] == "external":
                it.setForeground(2, self.palette().color(self.foregroundRole()))
                it.setText(2, f"⚠ {e['tool']}: {what}")
            self.events.addTopLevelItem(it)
        if not self.events.topLevelItemCount():
            self.events.addTopLevelItem(QTreeWidgetItem(["—", "", "No changes recorded for this file yet"]))
        if ann is not None:
            for v in versions(s.root, ann):
                it = QTreeWidgetItem([time.strftime("%Y-%m-%d %H:%M", time.localtime(v["created"])),
                                      v["title"] or v["kind"]])
                it.setData(0, ROLE, v)
                self.vers.addTopLevelItem(it)
        if not self.vers.topLevelItemCount():
            it = QTreeWidgetItem(["—", "No saved versions (nothing has rewritten this file yet)"])
            self.vers.addTopLevelItem(it)
        self._sel()

    def _current(self):
        it = self.vers.currentItem()
        return it.data(0, ROLE) if it is not None else None

    def _sel(self):
        v = self._current()
        for b in (self.preview_btn, self.revert_btn):
            b.setEnabled(v is not None and self.app.session is not None and self.key not in self.app.session.locked)
        if v is None:
            self._clear_overlay()

    def _clear_overlay(self):
        if self._overlay:
            self.app.canvas.clear_overlay()
            self._overlay = False

    # ------------------------------------------------------------------ preview / revert
    def _kind(self):
        return "yolo" if self.app.session.is_yolo else "labelme"

    def _preview(self):
        v = self._current()
        s = self.app.session
        if v is None or s is None:
            return
        ann_path = s.by_key[self.key].ann_path
        old = version_bytes(s.root, v["op"], ann_path)
        try:
            now = Path(ann_path).read_bytes() if Path(ann_path).is_file() else None
        except OSError:
            now = None
        names = list(s.base_adapter.class_names) if s.is_yolo else None
        a, b = annodiff.parse(old, self._kind(), names), annodiff.parse(now, self._kind(), names)
        d = annodiff.diff(a, b)
        self.diff.setText(("From that version to now: " + annodiff.summary(d)) if annodiff.is_change(d)
                          else "Same annotations as now (other bytes may differ, e.g. formatting or image data).")
        if self.key != self.app.current_key:
            self.app.open_sample(self.key)
        pix = self.app.canvas.pixmap_item.pixmap()
        W, H = pix.width(), pix.height()
        boxes = [{"box": (bb[0] * W, bb[1] * H, bb[2] * W, bb[3] * H), "text": f"then: {lab}",
                  "color": hexc("warn")} for lab, _t, bb, _p in a.get("shapes", [])]
        self.app.canvas.set_overlay(boxes)
        self._overlay = True

    def _revert(self):
        v = self._current()
        s = self.app.session
        if v is None or s is None:
            return
        if self.key in s.locked:
            QMessageBox.information(self, "History", f"The file is open in {s.locked[self.key]} — end that first.")
            return
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(v["created"]))
        if QMessageBox.question(self, "Revert", f"Restore the version kept on {when}\n({v['title']})?\n\nThe current "
                                                "file is stored first — you can revert the revert.") \
                != QMessageBox.StandardButton.Yes:
            return
        ann_path = s.by_key[self.key].ann_path
        new_op, how = revert(s.root, ann_path, v["op"])
        if how == "missing":
            QMessageBox.warning(self, "History", "That version is no longer in the backup store.")
            return
        s.log([self.key], "revert", f"reverted to the version kept {when} ({v['title']})", tool="History", op=new_op)
        self._clear_overlay()
        s.reindex_keys([self.key])
        self.app.statusBar().showMessage(f"Reverted {Path(ann_path).name} to the version kept {when}"
                                         + (" (image data rebuilt from the image file)" if how == "regen" else ""), 6000)
        self._timer.start()
