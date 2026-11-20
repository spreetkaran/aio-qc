# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Running training, everywhere in the app.

TrainingMonitor (one per app) — watches the runs Train Model started (also before a restart: they are remembered),
every few seconds while any is listed:
  · emits changed(rows) — rows: [(run folder, status dict, state)] of the runs still going (or just ended)
  · emits ended(run, state, status) once when a run finishes / stops / pauses / fails / is interrupted
  · makes sure an ended run gives back what it held: if its process is still there 20 s after it ended (a crashed run
    hanging on a data-loader worker keeps GPU memory and RAM), its whole process tree is killed.
TrainingChip — in the menu-bar corner of every mode, next to the mode switch: "Training · 12/100" while runs go;
click it for the list (progress, scores, time) with Show and End per run.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
                             QToolButton, QVBoxLayout, QWidget)

from AIO_QC import utils
from AIO_QC.core import training as T

KEY = "train/active"
_MONITOR = None


def monitor() -> "TrainingMonitor":
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = TrainingMonitor(QApplication.instance())
    return _MONITOR


class TrainingMonitor(QObject):
    changed = pyqtSignal(list)                  # [(run Path, status, state)] still going
    ended = pyqtSignal(object, str, dict)       # run, state, status
    _back = pyqtSignal(object)                   # a callable to run on the GUI thread (from a worker thread)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list = []
        self._told: set = set()
        self.timer = QTimer(self)
        self.timer.setInterval(3000)
        self.timer.timeout.connect(self.tick)
        self._back.connect(lambda fn: fn())
        if self.runs():
            self.timer.start()
            QTimer.singleShot(400, self.tick)

    @property
    def settings(self):
        return utils.app_settings()                  # fresh: the settings file can change (tests, profiles)

    def runs(self) -> list[str]:
        v = self.settings.value(KEY, []) or []
        return [v] if isinstance(v, str) else [str(x) for x in v]

    def add(self, run):
        r = [x for x in self.runs() if x != str(run)] + [str(run)]
        self.settings.setValue(KEY, r[-30:])
        self.timer.start()
        QTimer.singleShot(300, self.tick)

    def running(self) -> list[Path]:
        return [Path(r) for r, _st, s in self.rows if s in T.RUNNING | {"queued", "not responding"}]

    def tick(self):
        T.reap()
        keep, rows = [], []
        for r in self.runs():
            st = T.status(r)
            if not st:
                continue
            state = T.state_of(st)
            if state in T.RUNNING | {"queued", "not responding"}:
                keep.append(r)
                rows.append((Path(r), st, state))
                continue
            if r not in self._told and not (Path(r) / "aioqc" / "notified").exists():
                self._told.add(r)
                try:
                    (Path(r) / "aioqc" / "notified").write_text(str(time.time()), encoding="utf-8")
                except OSError:
                    pass
                self.ended.emit(Path(r), state, st)
            if T.is_runner(st.get("pid")):
                keep.append(r)                       # ended, but its process is still there: watched until gone
                T.ensure_released(r)
        self.settings.setValue(KEY, keep)
        self.rows = rows
        self.changed.emit(rows)
        if not keep:
            self.timer.stop()

    def end(self, run, done=None):
        """End a run now (in the background: it can take a few seconds)."""
        def work():
            try:
                T.end_run(run)
            finally:
                self._back.emit(self.tick)
                if done is not None:
                    self._back.emit(done)
        threading.Thread(target=work, daemon=True).start()


def _progress(st) -> str:
    ep, eps = st.get("epoch") or 0, st.get("epochs") or 0
    if st.get("mode") == "val":
        return "validating"
    return f"{ep}/{eps}" if eps else (st.get("phase") or "starting")


class TrainingChip(QToolButton):
    """The menu-bar corner chip; hidden while nothing trains."""
    show_run = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trainingChip")
        self.setIcon(utils.newIcon("tool-train"))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMinimumHeight(24)
        utils.style(self, "#trainingChip{background:@pill; color:@accent; border:1px solid @line; border-radius:12px;"
                          " padding:1px 10px 1px 6px; font-weight:600;} #trainingChip:hover{border-color:@accent;}")
        self.clicked.connect(self.open_list)
        self.show_run.connect(self._show_run)            # every mode's chip opens the run itself
        self.popup = None
        self.setVisible(False)
        m = monitor()
        m.changed.connect(self._update)
        self._update(m.rows)

    def _update(self, rows):
        going = [x for x in rows if x[2] in T.RUNNING | {"queued", "not responding"}]
        if self.isVisibleTo(self.parentWidget()) != bool(going) if self.parentWidget() else True:
            self.setVisible(bool(going))
            QTimer.singleShot(0, self._relayout)
        if not going:
            if self.popup is not None and self.popup.isVisible():
                self.popup.fill(rows)
            return
        if len(going) == 1:
            r, st, s = going[0]
            self.setText(f"Training · {_progress(st)}" + (" ⚠" if s == "not responding" else ""))
        else:
            self.setText(f"Training · {len(going)} runs")
        self.setToolTip("Running training — click to see them, show one or end it:\n" +
                        "\n".join(f"{r.name}: {s}, {_progress(st)}" for r, st, s in going))
        if self.popup is not None and self.popup.isVisible():
            self.popup.fill(rows)

    def _relayout(self):
        """The menu bar sizes its corner once: make room for (or give back) the chip's space."""
        from PyQt6.QtWidgets import QMenuBar
        w = self.parentWidget()
        mb = None
        while w is not None:
            w.adjustSize()
            if isinstance(w, QMenuBar):
                mb = w
                break
            w = w.parentWidget()
        if mb is not None:
            sz = mb.size()
            mb.resize(sz.width() + 1, sz.height())
            mb.resize(sz)

    def _show_run(self, run):
        """Show: Train Model (in Image QC) on that run — from whichever mode the chip is in. The shell switches to
        Image QC first; a window used without the shell opens it directly."""
        app = QApplication.instance()
        sh = getattr(app, "shell", None)
        if sh is not None and hasattr(sh, "show_training"):
            sh.show_training(run)
            return
        w = self.parentWidget()
        while w is not None and not hasattr(w, "open_training"):
            w = w.parentWidget()
        if w is not None:
            w.open_training(run=run)

    def open_list(self):
        if self.popup is None:
            self.popup = RunsPopup(self)
        self.popup.fill(monitor().rows)
        g = self.mapToGlobal(QPoint(self.width(), self.height() + 4))
        x, y = g.x() - self.popup.width(), g.y()
        scr = self.screen().availableGeometry() if self.screen() is not None else None
        if scr is not None:                               # never partly off the screen
            x = max(scr.left() + 4, min(x, scr.right() - self.popup.width() - 4))
            if y + self.popup.height() > scr.bottom():
                y = max(scr.top() + 4, self.mapToGlobal(QPoint(0, 0)).y() - self.popup.height() - 4)
        self.popup.move(x, y)
        self.popup.show()
        self.popup.raise_()


class RunsPopup(QFrame):
    def __init__(self, chip: TrainingChip):
        super().__init__(None, Qt.WindowType.Popup)
        self.chip = chip
        self.setObjectName("runsPopup")
        utils.style(self, "#runsPopup{background:@base; border:1px solid @line; border-radius:10px;}")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(12, 10, 12, 10)
        self.lay.setSpacing(8)
        self.setFixedWidth(440)
        self._shape = None                     # the runs listed (and their warnings): rows are rebuilt only when it
        self._live = {}                        # changes — a refresh never replaces a button under the mouse

    def fill(self, rows):
        going = [x for x in rows if x[2] in T.RUNNING | {"queued", "not responding"}]
        shape = tuple((str(r), s == "not responding") for r, _st, s in going)
        if shape == self._shape and self._live:
            for run, st, state in going:
                bar, info = self._live[str(run)]
                self._progress(bar, st)
                info.setText(self._info(st, state))
            return
        self._shape = shape
        self._live = {}
        while self.lay.count():
            it = self.lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        head = QLabel("<b>Running training</b>")
        self.lay.addWidget(head)
        going = [x for x in rows if x[2] in T.RUNNING | {"queued", "not responding"}]
        if not going:
            self.lay.addWidget(QLabel("Nothing is training now."))
        for run, st, state in going:
            self.lay.addWidget(self._row(run, st, state))
        note = QLabel("Runs keep going when AIO QC closes. End stops one at once and frees its GPU / memory (the last "
                      "saved epoch can be resumed).")
        note.setWordWrap(True)
        utils.style(note, "QLabel{color:@muted;}")
        self.lay.addWidget(note)
        self.fit()
        QTimer.singleShot(0, self.fit)                # again once the new rows are polished (styled sizes)

    def fit(self):
        for w in self.findChildren(QWidget):
            w.ensurePolished()
        self.lay.invalidate()
        self.lay.activate()
        h = self.lay.totalHeightForWidth(self.width()) if self.lay.hasHeightForWidth() else -1
        self.setFixedHeight(max(h, self.lay.totalMinimumSize().height(), self.lay.totalSizeHint().height()))

    def _row(self, run, st, state):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3)
        top = QHBoxLayout()
        name = QLabel(f"<b>{run.name}</b>")
        name.setToolTip(str(run))
        top.addWidget(name, 1)
        show = QPushButton("Show")
        show.clicked.connect(lambda: (self.hide(), self.chip.show_run.emit(run)))
        end = QPushButton("End")
        end.setToolTip("End this run now and free its GPU and memory")
        end.clicked.connect(lambda: self._end(run, end))
        for b in (show, end):
            b.setSizePolicy(b.sizePolicy().horizontalPolicy(), b.sizePolicy().Policy.Fixed)
            top.addWidget(b, 0, Qt.AlignmentFlag.AlignVCenter)
        v.addLayout(top)
        bar = QProgressBar()
        bar.setFixedHeight(16)
        self._progress(bar, st)
        v.addWidget(bar)
        info = QLabel(self._info(st, state))
        utils.style(info, "QLabel{color:@muted;}")
        v.addWidget(info)
        self._live[str(run)] = (bar, info)
        if state == "not responding":
            warn = QLabel("Not responding for a while — End frees it.")
            utils.style(warn, "QLabel{color:@warn;}")
            v.addWidget(warn)
        return w

    @staticmethod
    def _progress(bar, st):
        eps = int(st.get("epochs") or 0)
        bar.setRange(0, max(1, eps))
        bar.setValue(min(int(st.get("epoch") or 0), max(1, eps)))
        bar.setFormat("epoch %v / %m" if eps else (st.get("phase") or "starting"))

    @staticmethod
    def _info(st, state) -> str:
        bits = [state if state != "training" else (st.get("phase") or "training")]
        m = st.get("metrics") or {}
        best = m.get("metrics/mAP50-95(M)", m.get("metrics/mAP50-95(B)"))
        if best is not None:
            bits.append(f"mAP50-95 {best:.3f}")
        if st.get("started"):
            bits.append(f"{int((time.time() - float(st['started'])) // 60)} min")
        if st.get("gpu_mem_gb"):
            bits.append(f"GPU {st['gpu_mem_gb']:.1f} GB")
        if st.get("device"):
            bits.append(str(st["device"]))
        return " · ".join(bits)

    def _end(self, run, btn):
        # the question belongs to the window, not to this popup (a popup closes as soon as another window opens)
        self.hide()
        if QMessageBox.question(self.chip.window(), "End training", f"End {run.name} now?\n\nIts GPU and memory are "
                                "freed; the last saved epoch can be resumed in Train Model.") \
                != QMessageBox.StandardButton.Yes:
            return
        try:
            btn.setEnabled(False)
            btn.setText("Ending …")
        except RuntimeError:
            pass                                          # the list was rebuilt meanwhile
        self.chip.setText("Training · ending …")
        monitor().end(run)
