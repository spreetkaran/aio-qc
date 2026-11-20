# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Auto-Annotate window: videos -> unique frames -> LabelMe files -> the frames you need -> parts to upload.

Left: the five steps, each with a tick (run it or not) and what the run folder holds after it. Right: the selected
step's settings. Bottom: progress, Stop, and what to do with the result. Any set of steps can run — "Frames only"
for a plain batch extraction, or later steps alone on what an earlier run left (a new confidence, a new rule) —
and every step can be run on its own from its page. Stop is always safe; Run continues where it stopped.

The work is done by core.autoannotate.engine in a background thread; the same engine runs the command line
(aio-qc autoannotate), so a setup saved here runs unchanged on a server.
"""
from __future__ import annotations

import json
import os
import re
import threading

from loguru import logger
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
                             QScrollArea, QSizePolicy, QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem,
                             QToolButton, QVBoxLayout, QWidget)

from AIO_QC.tools.frame_extractor.core.autoannotate import engine as E
from AIO_QC.tools.frame_extractor.core.autoannotate import labelme_io as LM
from AIO_QC.tools.frame_extractor.core.autoannotate.model_info import info as model_info
from AIO_QC.tools.frame_extractor.core.autoannotate.recipe import (PRESETS, Recipe, SETUP_SUFFIX, STAGE_HINTS,
                                                                       STAGE_TITLES, STAGES)
from AIO_QC.utils import newButton, newIcon, style
from AIO_QC.widgets.lifecycle import RetainedThread
from AIO_QC.core.diag import ignored as _ignored

SETTINGS = "autoannotate"
BOX_COLOURS = ("#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00897B", "#F4511E", "#3949AB", "#C0CA33",
               "#6D4C41", "#D81B60", "#00ACC1")


def _settings():
    from AIO_QC.utils.qt_utils import app_settings
    return app_settings()


def _env_python():
    return _settings().value("inference/python", "") or ""


def setups_dir() -> str:
    from AIO_QC.core.appdirs import data_home
    d = os.path.join(str(data_home()), "autoannotate_setups")
    os.makedirs(d, exist_ok=True)
    return d


def _colour(label: str) -> QColor:
    import zlib
    return QColor(BOX_COLOURS[zlib.crc32(label.encode()) % len(BOX_COLOURS)])


# ====================================================================================================== threads
class RunWorker(RetainedThread, QThread):
    progress = pyqtSignal(str, int, int, str)
    logged = pyqtSignal(str, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str, str)
    stopped = pyqtSignal(str, str)
    confirm = pyqtSignal(str, int)

    def __init__(self, root, recipe, stages, source, env, replace_previous=False, parent=None):
        super().__init__(parent)
        self.args = (root, recipe, stages, source, env, replace_previous)
        self.cancel = threading.Event()
        self.stage = ""

    def run(self):
        root, r, stages, source, env, rep = self.args

        def prog(stage, d, t, text):
            self.stage = stage
            self.progress.emit(stage, d, t, text)

        ctx = E.Context(prog, lambda s, t: self.logged.emit(s, t), self.cancel)
        try:
            self.finished_ok.emit(E.run(root, r, stages, source, ctx, env, rep))
        except E.Stopped as exc:
            self.stopped.emit(self.stage, str(exc))
        except E.NeedsConfirmation as exc:
            self.confirm.emit(str(exc), exc.count)
        except E.StepError as exc:
            self.failed.emit(exc.stage, str(exc))
        except Exception as exc:                       # never silently: the log has the traceback
            logger.exception("auto-annotate")
            self.failed.emit(self.stage, f"{type(exc).__name__}: {exc}")


class CallWorker(RetainedThread, QThread):
    """A short background call (model details, preview, counting) -> done(result) / error(text)."""
    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except E.StepError as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            logger.opt(exception=True).debug("auto-annotate background call")
            self.error.emit(f"{type(exc).__name__}: {exc}")


# ====================================================================================================== pieces
class StepRow(QFrame):
    clicked = pyqtSignal(str)
    toggled = pyqtSignal(str, bool)

    def __init__(self, n, stage, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.setObjectName("stepRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(10)
        self.check = QCheckBox()
        self.check.setToolTip("Run this step (untick to leave it out)")
        self.check.toggled.connect(lambda on: self.toggled.emit(stage, on))
        self.num = QLabel(str(n))
        self.num.setObjectName("stepNum")
        self.num.setFixedSize(22, 22)
        self.num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text = QVBoxLayout()
        text.setSpacing(1)
        self.title = QLabel(STAGE_TITLES[stage])
        self.title.setObjectName("stepTitle")
        self.status = QLabel("")
        self.status.setObjectName("stepStatus")
        self.status.setWordWrap(True)
        text.addWidget(self.title)
        text.addWidget(self.status)
        lay.addWidget(self.check)
        lay.addWidget(self.num)
        lay.addLayout(text, 1)
        self.set_selected(False)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.stage)
        super().mousePressEvent(e)

    def set_selected(self, on):
        if self.property("selected") == bool(on):
            return
        self.setProperty("selected", bool(on))
        for w in (self, self.num, self.title):
            w.style().unpolish(w)
            w.style().polish(w)

    def set_status(self, text, kind=""):
        self.status.setText(text)
        self.status.setProperty("kind", kind)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)


def _section(text):
    lab = QLabel(text)
    style(lab, "QLabel{font-weight:600; color:@text; padding-top:10px;}")
    return lab


def _hint(text, wrap=True):
    lab = QLabel(text)
    lab.setWordWrap(wrap)
    style(lab, "QLabel{color:@muted;}")
    return lab


class _InfoLine(QLabel):
    """A hint line that takes no room while it has nothing to say (the header stays compact on small screens)."""

    def setText(self, text):
        super().setText(text)
        self.setVisible(bool(text))


def _info_hint():
    lab = _InfoLine("")
    lab.setWordWrap(True)
    lab.setVisible(False)
    style(lab, "QLabel{color:@muted;}")
    return lab


def _seg(options, current, on_change=None):
    """Segmented choice: options [(key, text, tip)] -> (widget, getter, setter)."""
    from AIO_QC.widgets.chrome import Segmented
    group = QButtonGroup()
    group.setExclusive(True)
    btns = {}
    for key, text, tip in options:
        b = QToolButton()
        b.setText(text)
        b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        b.setCheckable(True)
        b.setToolTip(tip)
        b.setMinimumHeight(26)
        group.addButton(b)
        btns[key] = b
    seg = Segmented(list(btns.values()))
    seg._group = group                                     # keep the group alive with the widget

    def get():
        return next((k for k, b in btns.items() if b.isChecked()), options[0][0])

    def set_(k):
        (btns.get(k) or btns[options[0][0]]).setChecked(True)
    set_(current)
    if on_change:
        group.buttonToggled.connect(lambda _b, on: on and on_change(get()))
    seg.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return seg, get, set_


class PathRow(QWidget):
    changed = pyqtSignal(str)

    def __init__(self, placeholder, pick, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setClearButtonEnabled(True)
        self.btn = newButton("Browse…", icon="folder-line", slot=pick)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.btn)
        self._t = QTimer(self)
        self._t.setSingleShot(True)
        self._t.setInterval(350)
        self._t.timeout.connect(lambda: self.changed.emit(self.text()))
        self.edit.textChanged.connect(lambda _t: self._t.start())

    def text(self):
        return self.edit.text().strip().strip('"')

    def setText(self, t):
        self.edit.setText(t or "")


def bgr_to_pixmap(img, shapes, max_w=460):
    import numpy as np
    h, w = img.shape[:2]
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    qi = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    pm = QPixmap.fromImage(qi)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    lw = max(2, int(round(min(w, h) / 360)))
    f = QFont(p.font())
    f.setPixelSize(max(11, int(min(w, h) / 40)))
    f.setWeight(QFont.Weight.DemiBold)
    p.setFont(f)
    fm = p.fontMetrics()
    for s in shapes:
        (x1, y1), (x2, y2) = s["points"]
        c = _colour(s["label"])
        p.setPen(QPen(c, lw))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        tag = s["label"] + (f"  {s['description'][5:]}" if s.get("description", "").startswith("conf=") else "")
        tw, th = fm.horizontalAdvance(tag) + 8, fm.height() + 2
        ty = int(y1) - th if y1 - th > 0 else int(y1)
        p.fillRect(int(x1), ty, tw, th, c)
        p.setPen(QColor("#FFFFFF"))
        p.drawText(int(x1) + 4, ty + fm.ascent() + 1, tag)
    p.end()
    return pm.scaledToWidth(min(max_w, w), Qt.TransformationMode.SmoothTransformation)


# ====================================================================================================== window
class AutoAnnotateWindow(QDialog):
    TITLE = "Auto-Annotate"

    def __init__(self, parent=None, source: str = "", run_folder: str = ""):
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(newIcon("tool-autoannotate"))
        self.setObjectName("autoAnnotate")
        self.worker = None
        self.task = None
        self._calls = []
        self._model_names = {}
        self._model_meta = {}
        self._preview_det = None
        self._preview_key = None
        self._selected = "frames"
        self._loading = False
        self._replace_ok = False
        self._last_stages = []
        st = _settings()
        try:
            r = Recipe.from_dict(json.loads(st.value(f"{SETTINGS}/recipe", "") or "{}"))
        except ValueError:
            r = Recipe()
        self._build()
        self._apply_style()
        self.set_recipe(r)
        self.source.setText(source or st.value(f"{SETTINGS}/source", "") or "")
        self.run_dir.setText(run_folder or st.value(f"{SETTINGS}/run", "") or "")
        self._select("frames")
        self._refresh_source()
        self._refresh_run()
        self._refresh_setups()
        from AIO_QC.utils.qt_utils import fit_size
        fit_size(self, "large")

    # ------------------------------------------------------------------ layout
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)
        # ---- header: what and where
        head = QHBoxLayout()
        title = QLabel(self.TITLE)
        style(title, "QLabel{font-size:15pt; font-weight:600;}")
        sub = _hint("Videos to unique frames with LabelMe files — run every step, or only the ones you need.")
        tbox = QVBoxLayout()
        tbox.setSpacing(0)
        tbox.addWidget(title)
        tbox.addWidget(sub)
        head.addLayout(tbox, 1)
        head.addWidget(QLabel("Setup"))
        self.setup_combo = QComboBox()
        self.setup_combo.setMinimumWidth(220)
        self.setup_combo.setToolTip("Saved setups: every choice of a project, ready for the next batch")
        self.setup_combo.activated.connect(self._setup_chosen)
        head.addWidget(self.setup_combo)
        head.addWidget(newButton("Save", icon="save", slot=self._save_setup,
                                 tip="Save these settings as a named setup (also runs from the command line)"))
        more = QPushButton("More")
        more.setIcon(newIcon("settings"))
        more.setToolTip("Load a setup file, save a copy, the command line for this setup")
        from PyQt6.QtWidgets import QMenu
        m = QMenu(more)
        m.addAction("Load setup file…", self._load_setup_file)
        m.addAction("Save setup as file…", self._save_setup_file)
        m.addSeparator()
        m.addAction("Open setups folder", lambda: self._open_path(setups_dir()))
        m.addAction("Command line for this setup", self._show_cli)
        m.addSeparator()
        m.addAction("Reset to defaults", lambda: self.set_recipe(Recipe(model=self.recipe().model)))
        more.setMenu(m)
        head.addWidget(more)
        outer.addLayout(head)

        where = QGridLayout()
        where.setHorizontalSpacing(10)
        where.setVerticalSpacing(6)
        self.src_seg, self.src_get, self.src_set = _seg(
            [("videos", "Videos", "Extract frames from a folder of videos"),
             ("images", "Images", "Start from a folder of images (no extraction)")], "videos",
            lambda _k: (self._refresh_source(), self._update_steps(), self._sync_frames_page()))
        self.source = PathRow("Folder of videos (sub-folders included)", self._pick_source)
        self.source.changed.connect(lambda _t: self._refresh_source())
        self.src_info = _info_hint()
        self.run_dir = PathRow("Run folder: frames, annotations and packages go here", self._pick_run)
        self.run_dir.changed.connect(lambda _t: self._refresh_run())
        self.run_info = _info_hint()
        self.open_run_btn = newButton("Open", icon="open", slot=lambda: self._open_path(self.run_dir.text()),
                                      tip="Open the run folder")
        where.addWidget(QLabel("From"), 0, 0)
        where.addWidget(self.src_seg, 0, 1)
        where.addWidget(self.source, 0, 2)
        where.addWidget(self.src_info, 1, 2)
        where.addWidget(QLabel("Run folder"), 2, 0)
        rr = QHBoxLayout()
        rr.addWidget(self.run_dir, 1)
        rr.addWidget(self.open_run_btn)
        where.addLayout(rr, 2, 1, 1, 2)
        where.addWidget(self.run_info, 3, 2)
        where.setColumnStretch(2, 1)
        outer.addLayout(where)

        # ---- body: steps | settings
        body = QHBoxLayout()
        body.setSpacing(14)
        left = QFrame()
        left.setObjectName("stepsPanel")
        left.setFixedWidth(270)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(4)
        cap = QLabel("STEPS")
        style(cap, "QLabel{color:@faint; font-size:8.5pt; font-weight:600; letter-spacing:1px; padding:2px 6px;}")
        ll.addWidget(cap)
        self.rows = {}
        for i, s in enumerate(STAGES, 1):
            row = StepRow(i, s)
            row.clicked.connect(self._select)
            row.toggled.connect(lambda _s, _on: self._update_steps())
            self.rows[s] = row
            ll.addWidget(row)
        ll.addSpacing(6)
        cap2 = QLabel("QUICK CHOICE")
        style(cap2, "QLabel{color:@faint; font-size:8.5pt; font-weight:600; letter-spacing:1px; padding:2px 6px;}")
        ll.addWidget(cap2)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Choose the steps…", None)
        for name, stages in PRESETS.items():
            self.preset_combo.addItem(name, list(stages))
        self.preset_combo.activated.connect(self._preset_chosen)
        self.preset_combo.setToolTip("Tick a common set of steps at once")
        ll.addWidget(self.preset_combo)
        ll.addStretch(1)
        self.steps_note = _hint("")
        ll.addWidget(self.steps_note)
        body.addWidget(left)

        self.pages = QStackedWidget()
        self.page_index = {}
        for s, build in (("frames", self._page_frames), ("similar", self._page_similar),
                         ("annotate", self._page_annotate), ("rules", self._page_rules),
                         ("package", self._page_package)):
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(4, 0, 12, 12)
            lay.setSpacing(8)
            top = QHBoxLayout()
            t = QLabel(STAGE_TITLES[s])
            style(t, "QLabel{font-size:13pt; font-weight:600;}")
            top.addWidget(t)
            top.addStretch(1)
            only = newButton("Run only this step", icon="run", slot=lambda _c=False, s=s: self.start([s]))
            only.setToolTip(f"Run {STAGE_TITLES[s]} alone, on what the run folder holds now")
            only.setObjectName(f"only_{s}")
            top.addWidget(only)
            lay.addLayout(top)
            lay.addWidget(_hint(STAGE_HINTS[s] + "."))
            build(lay)
            lay.addStretch(1)
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setFrameShape(QFrame.Shape.NoFrame)
            sc.setWidget(w)
            self.page_index[s] = self.pages.addWidget(sc)
        body.addWidget(self.pages, 1)
        outer.addLayout(body, 1)

        # ---- footer: progress + actions
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        self.log.setVisible(False)
        self.log.setFixedHeight(130)
        from AIO_QC.widgets.design import mono
        self.log.setFont(mono(11.5))
        outer.addWidget(self.log)
        foot = QHBoxLayout()
        foot.setSpacing(8)
        from AIO_QC.widgets.motion import SmoothProgressBar
        self.bar = SmoothProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        stat = QVBoxLayout()
        stat.setSpacing(4)
        stat.addWidget(self.status)
        stat.addWidget(self.bar)
        foot.addLayout(stat, 1)
        self.log_btn = newButton("Log", icon="logs", slot=self._toggle_log, tip="Show what each step did")
        self.summary_btn = newButton("Summary", icon="tool-report", slot=self._open_summary,
                                     tip="The run's summary (reports/summary.html)")
        self.qc_btn = newButton("Open in Image QC", icon="mode-image", slot=self._open_in_qc,
                                tip="Review the frames and their annotations in Image QC")
        self.run_btn = newButton("Run", icon="run", slot=lambda: self.start(None))
        self.run_btn.setDefault(True)
        self.run_btn.setAutoDefault(False)
        self.stop_btn = newButton("Stop", icon="stop", slot=self.stop,
                                  tip="Stop safely — Run continues where it stopped")
        self.stop_btn.setVisible(False)
        close = newButton("Close", slot=self.close)
        close.setAutoDefault(False)
        for b in (self.log_btn, self.summary_btn, self.qc_btn):
            b.setAutoDefault(False)
            foot.addWidget(b)
        foot.addSpacing(12)
        foot.addWidget(self.run_btn)
        foot.addWidget(self.stop_btn)
        foot.addWidget(close)
        outer.addLayout(foot)
        for b in self.findChildren(QPushButton):        # Enter never presses Save / Browse by accident
            b.setAutoDefault(False)
        self.run_btn.setDefault(True)

    def _apply_style(self):
        from AIO_QC.widgets.chrome import T
        t = {k: T(k) for k in ("surface", "border", "hover", "accent_soft", "accent", "on_accent", "text", "muted",
                               "faint", "ok", "warn", "error", "accent_mid")}
        css = f"""
#stepsPanel {{ background:{t['surface']}; border:1px solid {t['border']}; border-radius:10px; }}
#stepRow {{ border-radius:8px; background:transparent; }}
#stepRow:hover {{ background:{t['hover']}; }}
#stepRow[selected="true"] {{ background:{t['accent_soft']}; }}
#stepNum {{ border-radius:11px; background:{t['hover']}; color:{t['muted']}; font-weight:600; }}
#stepRow[selected="true"] #stepNum {{ background:{t['accent']}; color:{t['on_accent']}; }}
#stepNum:disabled {{ background:transparent; color:{t['faint']}; border:1px dashed {t['border']}; }}
#stepTitle {{ font-weight:600; color:{t['text']}; }}
#stepStatus {{ color:{t['faint']}; font-size:9pt; }}
#stepStatus[kind="done"] {{ color:{t['ok']}; }}
#stepStatus[kind="stale"] {{ color:{t['warn']}; }}
#stepStatus[kind="failed"] {{ color:{t['error']}; }}
#modelCard {{ background:{t['surface']}; border:1px solid {t['border']}; border-radius:8px; }}
QCheckBox#chip {{ background:{t['surface']}; border:1px solid {t['border']}; border-radius:13px; padding:4px 10px; }}
QCheckBox#chip:checked {{ background:{t['accent_soft']}; border-color:{t['accent_mid']}; }}
QCheckBox#chip[missing="true"] {{ color:{t['warn']}; }}
QCheckBox#chip:disabled {{ color:{t['faint']}; }}
#previewArea {{ background:{t['surface']}; border:1px solid {t['border']}; border-radius:8px; }}
"""
        if css != self.styleSheet():
            self.setStyleSheet(css)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() == e.Type.PaletteChange and hasattr(self, "rows"):
            self._apply_style()

    # ------------------------------------------------------------------ pages
    def _form(self, lay):
        f = QFormLayout()
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        f.setHorizontalSpacing(14)
        f.setVerticalSpacing(8)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        lay.addLayout(f)
        return f

    def _row(self, *ws, stretch=True):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(8)
        for x in ws:
            l.addWidget(x) if isinstance(x, QWidget) else l.addLayout(x)
        if stretch:
            l.addStretch(1)
        return w

    def _page_frames(self, lay):
        self.frames_video = QWidget()
        vl = QVBoxLayout(self.frames_video)
        vl.setContentsMargins(0, 0, 0, 0)
        f = self._form(vl)
        self.mode_seg, self.mode_get, self.mode_set = _seg(
            [("step", "Every Nth frame", "Keep frame 0, N, 2N …"),
             ("sharpest", "Sharpest of every N", "Of every N frames keep the sharpest (less motion blur)")],
            "step", lambda _k: self._frames_hint())
        f.addRow("Take", self.mode_seg)
        self.step = QSpinBox()
        self.step.setRange(1, 100000)
        self.step.setSuffix(" frames")
        self.step.valueChanged.connect(lambda _v: self._frames_hint())
        self.step_hint = _hint("", wrap=False)
        f.addRow("N", self._row(self.step, self.step_hint))
        self.repeats = QComboBox()
        for k, t in (("near", "Skip near-identical frames (recommended)"), ("exact", "Skip exact repeats only"),
                     ("off", "Keep every sampled frame")):
            self.repeats.addItem(t, k)
        self.repeats.setToolTip("Within one video, as it is decoded: a parked scene gives one frame, not hundreds.\n"
                                "Near-identical also looks at small regions, so a new object in a corner still "
                                "counts as a change.")
        f.addRow("Repeats", self.repeats)
        self.across = QCheckBox("Also skip frames already taken from another video or an earlier run")
        self.across.setToolTip("Remembers every frame of this run folder (perceptual hash + a check of small "
                               "regions), so the same scene from a second camera pass is not taken twice")
        f.addRow("", self.across)
        self.sharp = QDoubleSpinBox()
        self.sharp.setRange(0, 100000)
        self.sharp.setDecimals(0)
        self.sharp.setSpecialValueText("off")
        self.sharp.setToolTip("Leave out frames blurrier than this (variance of the Laplacian). 0 = off.\n"
                              "Open a video in the Frame Extractor to see the sharpness of its frames.")
        f.addRow("Blur filter", self._row(self.sharp, _hint("minimum sharpness — off keeps blurry frames", wrap=False)))
        self.ext = QComboBox()
        for e in (".jpg", ".png", ".webp", ".bmp"):
            self.ext.addItem(e[1:].upper(), e)
        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.quality.setPrefix("quality ")
        f.addRow("Format", self._row(self.ext, self.quality))
        self.template = QComboBox()
        self.template.setEditable(True)
        from AIO_QC.tools.frame_extractor.core.naming import PRESETS as NAMES
        for name, t in NAMES.items():
            if "/" not in t:
                self.template.addItem(t, t)
        self.template.setToolTip("File name of each frame: {video} {frame} {frame:06d} {time} {tc} {sec} {ms} …\n"
                                 "Every frame keeps the video's name, so names stay unique when folders are merged.")
        self.template.setMinimumWidth(260)
        f.addRow("Names", self.template)
        self.workers = QSpinBox()
        self.workers.setRange(0, 64)
        self.workers.setSpecialValueText("automatic")
        self.workers.setToolTip("Videos decoded at the same time (long videos are split between workers)")
        f.addRow("Workers", self.workers)
        lay.addWidget(self.frames_video)

        self.frames_images = QWidget()
        il = QVBoxLayout(self.frames_images)
        il.setContentsMargins(0, 0, 0, 0)
        il.addWidget(_hint("The images are added to the run folder's frames/ (sub-folders kept). Your folder is "
                           "never changed."))
        f2 = self._form(il)
        self.import_mode = QComboBox()
        self.import_mode.addItem("Link (no extra disk space; same drive)", "link")
        self.import_mode.addItem("Copy", "copy")
        f2.addRow("Add as", self.import_mode)
        lay.addWidget(self.frames_images)

        lay.addWidget(_section("Which files"))
        f3 = self._form(lay)
        self.include = QLineEdit()
        self.include.setPlaceholderText("all files — or e.g.  *cam1*; *cam2*")
        self.include.setToolTip("Only files whose names match one of these (; between patterns). Without * a "
                                "pattern matches anywhere in the name.")
        self.include.textChanged.connect(lambda _t: self._src_timer.start())
        self.exclude = QLineEdit()
        self.exclude.setPlaceholderText("nothing — or e.g.  *test*")
        for e in (self.include, self.exclude):
            e.setMinimumWidth(380)
        self.exclude.textChanged.connect(lambda _t: self._src_timer.start())
        self.recursive = QCheckBox("Include sub-folders")
        self.recursive.toggled.connect(lambda _on: self._src_timer.start())
        f3.addRow("Only names", self.include)
        f3.addRow("Leave out", self.exclude)
        f3.addRow("", self.recursive)
        self._src_timer = QTimer(self)
        self._src_timer.setSingleShot(True)
        self._src_timer.setInterval(400)
        self._src_timer.timeout.connect(self._refresh_source)

    def _page_similar(self, lay):
        lay.addWidget(_hint("Compares every frame with all the frames kept so far — across videos too — and moves "
                            "the ones that look the same to similar/ (with their annotation files). Nothing is "
                            "deleted; running this step again first puts them back."))
        f = self._form(lay)
        self.sim_seg, self.sim_get, self.sim_set = _seg(
            [("strict", "Strict", "Only frames that are nearly the same picture"),
             ("balanced", "Balanced", "Same scene, small changes (recommended)"),
             ("loose", "Loose", "Also frames with modest changes — fewer, more varied frames")], "balanced")
        f.addRow("How alike", self.sim_seg)
        self.prefer_sharp = QCheckBox("Of two similar frames keep the sharper one")
        f.addRow("", self.prefer_sharp)
        lay.addWidget(_hint("Within one video, Frames already skips repeats as it extracts. This step is for "
                            "repeats across videos and runs, or for an image folder."))

    def _page_annotate(self, lay):
        from AIO_QC.tools.frame_extractor.widgets.ai_annotate_panel import AIAnnotatePanel
        self.ann_seg, self.ann_get, self.ann_set = _seg(
            [("model", "Your trained model", "A YOLO model you trained (.onnx / .pt) draws the boxes"),
             ("ai", "AI annotate", "A vision model (online or on your computer) annotates from your description of "
                                   "the classes and a few examples — no trained model needed")], "model",
            lambda _k: self._annotator_changed())
        f0 = self._form(lay)
        f0.addRow("Who annotates", self.ann_seg)
        self.model_box = QWidget()
        ml = QVBoxLayout(self.model_box)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(8)
        self._page_annotate_model(ml)
        lay.addWidget(self.model_box)
        self.ai_panel = AIAnnotatePanel(self)
        self.ai_panel.changed.connect(self._ai_changed)
        self.ai_panel.project_changed.connect(lambda _p: self._ai_changed())
        self.ai_panel.setVisible(False)
        lay.addWidget(self.ai_panel)

    def _page_annotate_model(self, lay):
        f = self._form(lay)
        mrow = QHBoxLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setMinimumWidth(380)
        self.model.lineEdit().setPlaceholderText("Model file (.onnx or .pt)")
        self.model.setToolTip("Models you used before (here, in Testing or Models & Environment) are listed")
        self.model.currentTextChanged.connect(lambda _t: self._model_timer.start())
        mrow.addWidget(self.model, 1)
        mrow.addWidget(newButton("Browse…", icon="folder-line", slot=self._pick_model))
        mw = QWidget()
        mw.setLayout(mrow)
        mrow.setContentsMargins(0, 0, 0, 0)
        f.addRow("Model", mw)
        self.model_card = QLabel("")
        self.model_card.setObjectName("modelCard")
        self.model_card.setWordWrap(True)
        self.model_card.setTextFormat(Qt.TextFormat.RichText)
        self.model_card.setContentsMargins(10, 8, 10, 8)
        self.model_card.setMinimumWidth(560)
        f.addRow("", self.model_card)
        self._model_timer = QTimer(self)
        self._model_timer.setSingleShot(True)
        self._model_timer.setInterval(300)
        self._model_timer.timeout.connect(self._read_model)

        self.imgsz = QSpinBox()
        self.imgsz.setRange(0, 8192)
        self.imgsz.setSingleStep(32)
        self.imgsz.setSpecialValueText("the model's own")
        self.imgsz.setSuffix(" px")
        self.imgsz.setToolTip("The size images are resized to for the model. Leave it at the model's own (read "
                              "from the file) unless you know better.")
        self.imgsz_hint = _hint("", wrap=False)
        f.addRow("Image size", self._row(self.imgsz, self.imgsz_hint))
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.01, 1.0)
        self.conf.setSingleStep(0.05)
        self.conf.setDecimals(2)
        self.conf.setToolTip("Boxes below this confidence are left out (a class can have its own, below)")
        self.iou = QDoubleSpinBox()
        self.iou.setRange(0.05, 1.0)
        self.iou.setSingleStep(0.05)
        self.iou.setDecimals(2)
        self.iou.setToolTip("Overlap (IoU) above which two boxes of a class are merged into one (NMS)")
        f.addRow("Confidence", self._row(self.conf, QLabel("IoU"), self.iou))
        self.max_det = QSpinBox()
        self.max_det.setRange(1, 10000)
        self.max_det.setToolTip("At most this many boxes per frame")
        self.min_box = QSpinBox()
        self.min_box.setRange(0, 4096)
        self.min_box.setSuffix(" px")
        self.min_box.setSpecialValueText("any size")
        self.min_box.setToolTip("Leave out boxes narrower or shorter than this")
        f.addRow("Max boxes", self._row(self.max_det, QLabel("Smallest box"), self.min_box))
        self.device = QComboBox()
        self.device.setEditable(True)
        for k, t in (("auto", "Automatic (GPU when there is one)"), ("cpu", "CPU"), ("0", "GPU 0"), ("1", "GPU 1")):
            self.device.addItem(t, k)
        self.batch = QSpinBox()
        self.batch.setRange(1, 256)
        self.batch.setToolTip("Frames per model call (higher is faster on a GPU, needs more memory)")
        self.half = QCheckBox("Half precision (FP16, GPU)")
        f.addRow("Runs on", self._row(self.device, QLabel("Batch"), self.batch, self.half))

        lay.addWidget(_section("Classes"))
        lay.addWidget(_hint("Untick a class to leave it out; type a name to write it under another label; a class can "
                            "have its own minimum confidence."))
        self.class_table = QTableWidget(0, 4)
        self.class_table.setHorizontalHeaderLabels(["Model class", "Write as", "Min confidence", "ID"])
        self.class_table.verticalHeader().setVisible(False)
        hh = self.class_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.class_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.class_table.itemChanged.connect(lambda _i: self._classes_changed())
        lay.addWidget(self.class_table)

        lay.addWidget(_section("Files written"))
        f2 = self._form(lay)
        self.ofmt_seg, self.ofmt_get, self.ofmt_set = _seg(
            [("labelme", "LabelMe", "One LabelMe JSON per frame"),
             ("yolo", "YOLO txt", "One YOLO .txt per frame, in the class order of the table above, + classes.txt"),
             ("both", "Both", "LabelMe JSON and YOLO txt")], "labelme", lambda _k: self._package_hint())
        f2.addRow("Format", self.ofmt_seg)
        self.lm_seg, self.lm_get, self.lm_set = _seg(
            [(v, f"LabelMe {v}", f"Files exactly as LabelMe {v} writes them") for v in LM.VERSIONS], LM.DEFAULT_VERSION)
        f2.addRow("Version", self.lm_seg)
        self.image_data = QCheckBox("Save the image inside each JSON (imageData)")
        self.image_data.setToolTip("Off (recommended): small files that point to the image next to them.\n"
                                   "On: each JSON carries the image in base64 (about 1.35 × the image's size).")
        self.image_data.toggled.connect(lambda _on: self._package_hint())
        f2.addRow("", self.image_data)
        self.desc = QCheckBox("Write each box's confidence in its description")
        f2.addRow("", self.desc)
        self.existing = QComboBox()
        for k, t in (("update", "Redo files this run wrote with other settings (keeps files you edited)"),
                     ("keep", "Keep every file that is already there"),
                     ("replace", "Replace every file (also ones you edited)")):
            self.existing.addItem(t, k)
        f2.addRow("Existing files", self.existing)

        lay.addWidget(_section("Try it first"))
        prow = QHBoxLayout()
        self.preview_btn = newButton("Try on 4 frames", icon="ai-box", slot=self._preview,
                                     tip="Runs the model on a few frames from the source with these settings — "
                                         "nothing is written")
        self.preview_btn.setAutoDefault(False)
        self.preview_note = _hint("")
        prow.addWidget(self.preview_btn)
        prow.addWidget(self.preview_note, 1)
        lay.addLayout(prow)
        self.preview_area = QWidget()
        self.preview_area.setObjectName("previewArea")
        self.preview_grid = QGridLayout(self.preview_area)
        self.preview_grid.setContentsMargins(8, 8, 8, 8)
        self.preview_grid.setSpacing(8)
        self.preview_area.setVisible(False)
        lay.addWidget(self.preview_area)

    def _page_rules(self, lay):
        f = self._form(lay)
        self.keep_when = QComboBox()
        for k, t in (("any_of", "It has at least one of these classes"), ("all_of", "It has all of these classes"),
                     ("detections", "It has any box at all"), ("always", "Always (keep every frame)")):
            self.keep_when.addItem(t, k)
        self.keep_when.currentIndexChanged.connect(lambda _i: (self._rules_enable(), self._count_timer.start()))
        f.addRow("Keep a frame when", self.keep_when)
        self.min_boxes = QSpinBox()
        self.min_boxes.setRange(1, 1000)
        self.min_boxes.setPrefix("at least ")
        self.min_boxes.setSuffix(" box(es)")
        self.min_boxes.valueChanged.connect(lambda _v: self._count_timer.start())
        f.addRow("", self.min_boxes)
        from AIO_QC.utils.qt_utils import FlowLayout
        self.keep_box = QWidget()
        self.keep_box.setMinimumWidth(420)
        self.keep_flow = FlowLayout(self.keep_box, spacing=6)
        self.keep_chips = []
        f.addRow("Classes", self.keep_box)
        self.others_seg, self.others_get, self.others_set = _seg(
            [("move", "Move to extra_images/", "Set them aside with their files (put back if you run this again)"),
             ("keep", "Keep them", "Leave every frame where it is (the rule only counts)")], "move",
            lambda _k: self._count_timer.start())
        f.addRow("Other frames", self.others_seg)
        self.rule_count = _hint("")
        lay.addWidget(self.rule_count)
        self._count_timer = QTimer(self)
        self._count_timer.setSingleShot(True)
        self._count_timer.setInterval(500)
        self._count_timer.timeout.connect(self._count_rule)

    def _page_package(self, lay):
        f = self._form(lay)
        self.group_seg, self.group_get, self.group_set = _seg(
            [("none", "One folder", "All frames together"),
             ("video", "Folder per video", "One folder for each video"),
             ("pattern", "By name pattern", "A part of the file name names the folder")], "none",
            lambda _k: (self._pattern_enable(), self._package_hint()))
        f.addRow("Folders", self.group_seg)
        self.pattern = QLineEdit()
        self.pattern.setPlaceholderText(r"regular expression, e.g.  cam\d+   or   _(\w+?)_frame")
        self.pattern.setToolTip("Searched in each file name; the folder is the first group in ( ) if there is one, "
                                "else the whole match. Names that don't match go to _unmatched.")
        self.pattern.setMinimumWidth(360)
        self.pattern.textChanged.connect(lambda _t: self._pkg_timer.start())
        f.addRow("Pattern", self.pattern)
        self.pattern_test = _hint("")
        f.addRow("", self.pattern_test)
        self.split = QSpinBox()
        self.split.setRange(0, 1000000)
        self.split.setSpecialValueText("no parts")
        self.split.setSuffix(" frames per part")
        self.split.valueChanged.connect(lambda _v: self._pkg_timer.start())
        self.part_names = QComboBox()
        self.part_names.addItem("part_001, part_002 …", "number")
        self.part_names.addItem("part_0, part_200 … (first file's index)", "start")
        f.addRow("Parts", self._row(self.split, self.part_names))
        self.zip_parts = QCheckBox("Zip each part")
        self.keep_folders = QCheckBox("Keep the part folders next to the zip files")
        self.with_images = QCheckBox("Images go with their JSON files")
        self.with_images.setToolTip("Off only when the JSONs carry the image (imageData)")
        self.with_images.toggled.connect(lambda _on: self._package_hint())
        f.addRow("", self.zip_parts)
        f.addRow("", self.keep_folders)
        f.addRow("", self.with_images)
        self.pkg_files_seg, self.pkg_files_get, self.pkg_files_set = _seg(
            [("link", "Hard links", "No extra disk space: a part's files are the same files as in frames/. "
                                    "Editing one edits both. (Copies are made where the drive has no hard links.)"),
             ("copy", "Copies", "Independent files: change or delete the frames later without touching the parts "
                                "(uses as much space again)")], "link", lambda _k: self._package_hint())
        f.addRow("Files", self.pkg_files_seg)
        self.ylay_seg, self.ylay_get, self.ylay_set = _seg(
            [("beside", "Next to the images", "Each .txt next to its image (with classes.txt)"),
             ("split", "images/ + labels/", "YOLO training layout: images/ and labels/ in each part, with "
                                            "data.yaml")], "beside")
        f.addRow("YOLO files", self.ylay_seg)
        self.pkg_hint = _hint("")
        lay.addWidget(self.pkg_hint)
        lay.addWidget(_hint("package/ is rebuilt each time this step runs; parts.csv lists every part. With hard "
                            "links a part's files share the disk space (and the content) with frames/."))
        self._pkg_timer = QTimer(self)
        self._pkg_timer.setSingleShot(True)
        self._pkg_timer.setInterval(400)
        self._pkg_timer.timeout.connect(self._package_hint)

    # ------------------------------------------------------------------ recipe <-> widgets
    def recipe(self) -> Recipe:
        r = Recipe.from_dict(getattr(self, "_recipe_base", Recipe()).to_dict())
        r.stages = [s for s in STAGES if self.rows[s].check.isChecked()]
        r.source = self.src_get()
        r.recursive = self.recursive.isChecked()
        r.include, r.exclude = self.include.text().strip(), self.exclude.text().strip()
        r.mode, r.step = self.mode_get(), self.step.value()
        r.skip_repeats = self.repeats.currentData()
        r.across_videos = self.across.isChecked()
        r.min_sharpness = self.sharp.value()
        r.ext, r.quality = self.ext.currentData(), self.quality.value()
        r.name_template = self.template.currentText().strip() or "{video}_frame_{frame}"
        r.workers = self.workers.value()
        r.import_mode = self.import_mode.currentData()
        r.similar, r.prefer_sharper = self.sim_get(), self.prefer_sharp.isChecked()
        r.model = self.model.currentText().strip().strip('"')
        r.imgsz, r.conf, r.iou = self.imgsz.value(), round(self.conf.value(), 3), round(self.iou.value(), 3)
        r.max_det, r.min_box = self.max_det.value(), self.min_box.value()
        r.device = self.device.currentData() if self.device.currentText() == self.device.itemText(
            self.device.currentIndex()) else self.device.currentText().strip()
        r.batch, r.half = self.batch.value(), self.half.isChecked()
        r.classes = self._classes_from_table() if self.class_table.rowCount() else dict(r.classes)
        r.labelme_version = self.lm_get()
        r.image_data = self.image_data.isChecked()
        r.description = "conf" if self.desc.isChecked() else "none"
        r.also_yolo = False
        r.output_format = self.ofmt_get()
        r.annotator = self.ann_get()
        r.ai_config = self.ai_panel.path or r.ai_config
        r.yolo_layout = self.ylay_get()
        r.existing = self.existing.currentData()
        r.keep_when = self.keep_when.currentData()
        r.min_boxes = self.min_boxes.value()
        r.keep_classes = [c.text().replace("&&", "&") for c in self.keep_chips if c.isChecked()] \
            if self.keep_chips or not self._loading else list(r.keep_classes)
        r.others = self.others_get()
        r.group_by, r.group_pattern = self.group_get(), self.pattern.text().strip()
        r.split_size, r.part_names = self.split.value(), self.part_names.currentData()
        r.zip_parts, r.keep_folders = self.zip_parts.isChecked(), self.keep_folders.isChecked()
        r.package_images = self.with_images.isChecked()
        r.package_files = self.pkg_files_get()
        return r

    def set_recipe(self, r: Recipe):
        self._loading = True
        try:
            self._recipe_base = r
            for s in STAGES:
                self.rows[s].check.setChecked(s in r.stages)
            self.src_set(r.source)
            self.recursive.setChecked(r.recursive)
            self.include.setText(r.include)
            self.exclude.setText(r.exclude)
            self.mode_set(r.mode)
            self.step.setValue(int(r.step))
            self._set_combo(self.repeats, r.skip_repeats)
            self.across.setChecked(r.across_videos)
            self.sharp.setValue(float(r.min_sharpness))
            self._set_combo(self.ext, r.ext)
            self.quality.setValue(int(r.quality))
            self.template.setEditText(r.name_template)
            self.workers.setValue(int(r.workers))
            self._set_combo(self.import_mode, r.import_mode)
            self.sim_set(r.similar)
            self.prefer_sharp.setChecked(r.prefer_sharper)
            self._fill_models(r.model)
            self.imgsz.setValue(int(r.imgsz))
            self.conf.setValue(float(r.conf))
            self.iou.setValue(float(r.iou))
            self.max_det.setValue(int(r.max_det))
            self.min_box.setValue(int(r.min_box))
            i = self.device.findData(r.device)
            if i >= 0:
                self.device.setCurrentIndex(i)
            else:
                self.device.setEditText(str(r.device))
            self.batch.setValue(int(r.batch))
            self.half.setChecked(r.half)
            self.lm_set(r.labelme_version)
            self.image_data.setChecked(r.image_data)
            self.desc.setChecked(r.description == "conf")
            self.ofmt_set("both" if r.also_yolo and r.output_format == "labelme" else r.output_format)
            self.ann_set(r.annotator)
            self.ylay_set(r.yolo_layout)
            self._set_combo(self.existing, r.existing)
            self._set_combo(self.keep_when, r.keep_when)
            self.min_boxes.setValue(int(r.min_boxes))
            self.others_set(r.others)
            self.group_set(r.group_by)
            self.pattern.setText(r.group_pattern)
            self.split.setValue(int(r.split_size))
            self._set_combo(self.part_names, r.part_names)
            self.zip_parts.setChecked(r.zip_parts)
            self.keep_folders.setChecked(r.keep_folders)
            self.with_images.setChecked(r.package_images)
            self.pkg_files_set(r.package_files)
        finally:
            self._loading = False
        self._fill_classes(self._model_names, r)
        self.ai_panel.load_project(r.ai_config)
        self._annotator_changed(save=False)
        self._sync_frames_page()
        self._frames_hint()
        self._rules_enable()
        self._pattern_enable()
        self._update_steps()
        self._model_timer.start()

    @staticmethod
    def _set_combo(c, data):
        i = c.findData(data)
        c.setCurrentIndex(i if i >= 0 else 0)

    # ------------------------------------------------------------------ model
    def _fill_models(self, current):
        seen, items = set(), []
        try:
            from AIO_QC.core import model_registry as R
            for m in R.app_models():
                p = m.get("path", "")
                if p and os.path.isfile(p) and os.path.normcase(p) not in seen:
                    seen.add(os.path.normcase(p))
                    items.append(p)
        except Exception as _exc:
            _ignored(_exc)
        for p in (_settings().value(f"{SETTINGS}/models", []) or []):
            if p and os.path.isfile(p) and os.path.normcase(p) not in seen:
                seen.add(os.path.normcase(p))
                items.insert(0, p)
        self.model.blockSignals(True)
        self.model.clear()
        for p in items:
            self.model.addItem(p, p)
        self.model.setEditText(current or "")
        self.model.blockSignals(False)

    def _pick_model(self):
        start = os.path.dirname(self.model.currentText().strip()) or _settings().value(f"{SETTINGS}/model_dir", "")
        p, _ = QFileDialog.getOpenFileName(self, "Model", start, "Models (*.onnx *.pt *.engine);;All files (*)")
        if p:
            _settings().setValue(f"{SETTINGS}/model_dir", os.path.dirname(p))
            self.model.setEditText(p)

    def _read_model(self):
        path = self.model.currentText().strip().strip('"')
        if not path:
            self._model_names, self._model_meta = {}, {}
            self.model_card.setText(f"<span style='color:{self._hex('muted')}'>Choose the model that draws the "
                                    f"boxes — its classes and image size are read from the file.</span>")
            self._fill_classes({}, self.recipe())
            self._update_steps()
            return
        if not os.path.isfile(path):
            self._model_names, self._model_meta = {}, {}
            self.model_card.setText(f"<span style='color:{self._hex('error')}'>File not found</span>")
            self._update_steps()
            return

        def done(inf):
            if self.model.currentText().strip().strip('"') != path:
                return
            self._model_meta = inf
            self._model_names = dict(inf.get("names") or {})
            self._show_model_card(path, inf)
            self._fill_classes(self._model_names, self.recipe())
            self._update_steps()
        self._call(lambda: model_info(path), done)

    def _hex(self, role):
        from AIO_QC.utils import hexc
        return hexc(role)

    def _show_model_card(self, path, inf):
        ext = os.path.splitext(path)[1].lower()
        env = _env_python()
        if ext == ".onnx":
            runs = "runs here (ONNX Runtime — the GPU when CUDA / DirectML is available)"
        elif env:
            runs = f"runs in your Python environment ({os.path.basename(os.path.dirname(os.path.dirname(env))) or env})"
        else:
            import importlib.util
            runs = ("runs here (ultralytics)" if importlib.util.find_spec("ultralytics") else
                    f"<span style='color:{self._hex('warn')}'>needs a Python environment with ultralytics — "
                    f"choose one in Settings ▸ Models, or export the model to ONNX</span>")
        n = len(inf.get("names") or {})
        sz = inf.get("imgsz")
        size_txt = (f"trained at <b>{sz} px</b>" if ext == ".pt" else f"input <b>{sz} px</b>") if sz else \
            "image size not in the file (640 px is used unless you set one)"
        cls = ", ".join(list((inf.get("names") or {}).values())[:8]) + (" …" if n > 8 else "")
        err = f"<br><span style='color:{self._hex('warn')}'>{inf['error']}</span>" if inf.get("error") else ""
        self.model_card.setText(f"<b>{os.path.basename(path)}</b> · {ext[1:].upper()} · {inf.get('task') or 'detect'}"
                                f" · {n} class{'es' if n != 1 else ''} · {size_txt}<br>"
                                f"<span style='color:{self._hex('muted')}'>{cls}</span><br>"
                                f"<span style='color:{self._hex('muted')}'>{runs}</span>{err}")
        self.imgsz.setSpecialValueText(f"the model's own ({sz} px)" if sz else "the model's own (640 px)")
        fixed = ext == ".onnx" and sz and not inf.get("dynamic_size")
        self.imgsz.setEnabled(not fixed)
        self.imgsz_hint.setText(f"fixed in this ONNX export — {sz} px is always used" if fixed else "")

    def _fill_classes(self, names, r: Recipe):
        self._loading_classes = True
        t = self.class_table
        t.setRowCount(0)
        for cid in sorted(names):
            e = r.class_entry(cid)
            row = t.rowCount()
            t.insertRow(row)
            it = QTableWidgetItem(names[cid])
            it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if e["keep"] else Qt.CheckState.Unchecked)
            it.setData(Qt.ItemDataRole.UserRole, cid)
            t.setItem(row, 0, it)
            w = QTableWidgetItem(e["name"])
            w.setToolTip(f"Label written for '{names[cid]}' (empty = the same)")
            t.setItem(row, 1, w)
            c = QTableWidgetItem(f"{e['conf']:.2f}" if e["conf"] else "")
            c.setToolTip("Empty = the confidence above")
            t.setItem(row, 2, c)
            idi = QTableWidgetItem(str(cid))
            idi.setFlags(Qt.ItemFlag.ItemIsEnabled)
            t.setItem(row, 3, idi)
        t.setVisible(bool(names))
        rows_h = sum(t.rowHeight(i) for i in range(t.rowCount()))
        t.setFixedHeight(min(320, t.horizontalHeader().height() + rows_h + 6) if names else 0)
        self._loading_classes = False
        self._fill_keep_list(r)

    def _classes_from_table(self) -> dict:
        out = {}
        t = self.class_table
        for row in range(t.rowCount()):
            it = t.item(row, 0)
            cid = int(it.data(Qt.ItemDataRole.UserRole))
            name = (t.item(row, 1).text() if t.item(row, 1) else "").strip()
            raw = (t.item(row, 2).text() if t.item(row, 2) else "").strip().replace(",", ".")
            try:
                conf = float(raw) if raw else None
                conf = conf if conf is None or 0 < conf <= 1 else None
            except ValueError:
                conf = None
            keep = it.checkState() == Qt.CheckState.Checked
            if not keep or name or conf:
                out[str(cid)] = {"keep": keep, "name": name, **({"conf": conf} if conf else {})}
        return out

    def _classes_changed(self):
        if getattr(self, "_loading_classes", False) or self._loading:
            return
        self._fill_keep_list(self.recipe())
        self._update_steps()

    def _output_labels(self, r: Recipe):
        """The labels the chosen annotator writes (for the keep rule)."""
        if r.annotator == "ai":
            from AIO_QC.tools.frame_extractor.core.ai_annotate.config import class_names
            return class_names(self.ai_panel.cfg) if self.ai_panel.path else []
        return E.output_classes(r, self._model_names) if self._model_names else []

    def _annotator_changed(self, save=True):
        ai = self.ann_get() == "ai"
        self.model_box.setVisible(not ai)
        self.ai_panel.setVisible(ai)
        if ai and not self.ai_panel._ready_items:
            self.ai_panel.check_ready(True)
        self._fill_keep_list(self.recipe())
        if save:
            self._update_steps()

    def _ai_changed(self):
        if self._loading:
            return
        if self.ann_get() == "ai":
            self._fill_keep_list(self.recipe())
        self._update_steps()

    def sample_image_paths(self, n):
        """Images for AI annotate's Try: frames of the run folder, else images / frames from the source."""
        root = self.run_dir.text()
        ws = E.Workspace(root) if root else None
        imgs = ws.frame_images() if ws and os.path.isdir(ws.frames) else []
        r = self.recipe()
        src = self.source.text()
        if not imgs and src and os.path.exists(src):
            if r.source == "images":
                imgs = E.find_images(src, r.recursive, r.include, r.exclude)
            else:
                out = os.path.join(root or os.path.join(os.path.dirname(src), "aioqc_try"), E.META, "try_frames")
                os.makedirs(out, exist_ok=True)
                paths = []
                for i, (title, fr) in enumerate(E.sample_frames(r, src, n)):
                    p = os.path.join(out, re.sub(r"[^\w.-]+", "_", title) + ".jpg")
                    if not os.path.exists(p):
                        import cv2
                        cv2.imwrite(p, fr)
                    paths.append(p)
                return paths
        return imgs[::max(1, len(imgs) // max(1, n))][:n]

    def image_count(self):
        names = getattr(self, "_frame_names", None)
        if names:
            return len(names)
        r = self.recipe()
        src = self.source.text()
        if not src or not os.path.exists(src):
            return 0
        if r.source == "images":
            return len(E.find_images(src, r.recursive, r.include, r.exclude))
        total = 0
        from AIO_QC.tools.frame_extractor.core.sources import open_source
        for v in E.find_videos(src, r.recursive, r.include, r.exclude):
            try:
                sr = open_source(v, "auto", False)
                total += max(1, sr.frame_count // max(1, int(r.step)))
                sr.release()
            except Exception as _exc:
                _ignored(_exc)
        return total

    def _fill_keep_list(self, r: Recipe):
        labels = self._output_labels(r)
        written = set(labels)
        if r.annotator != "ai" and not self._model_names:
            written = set()
        for c in r.keep_classes:
            if c not in labels:
                labels.append(c)
        want = set(r.keep_classes)
        for c in self.keep_chips:
            self.keep_flow.removeWidget(c)
            c.deleteLater()
        self.keep_chips = []
        while self.keep_flow.count():
            it = self.keep_flow.takeAt(0)
            if it is not None and it.widget() is not None:
                it.widget().deleteLater()
        if not labels:
            self.keep_flow.addWidget(_hint("Choose the model on the Annotate page — its classes appear here."))
            return
        for lab in labels:
            c = QCheckBox(lab.replace("&", "&&"))
            c.setObjectName("chip")
            c.setChecked(lab in want)
            if labels and lab not in written:
                c.setToolTip("Not among the classes the model writes (renamed or left out on the Annotate page)")
                c.setProperty("missing", True)
            c.toggled.connect(lambda _on: (self._count_timer.start(), self._update_steps()))
            self.keep_flow.addWidget(c)
            self.keep_chips.append(c)
        self.keep_box.updateGeometry()

    # ------------------------------------------------------------------ hints
    def _frames_hint(self):
        n = self.step.value()
        if self.mode_get() == "step":
            self.step_hint.setText(f"≈ {60 * 30 / n:,.0f} frames per minute of 30 fps video" if n else "")
        else:
            self.step_hint.setText(f"the sharpest of every {n} — ≈ {60 * 30 / n:,.0f} per minute at 30 fps")

    def _sync_frames_page(self):
        img = self.src_get() == "images"
        self.frames_video.setVisible(not img)
        self.frames_images.setVisible(img)
        self.source.edit.setPlaceholderText("Folder of images (sub-folders included)" if img else
                                            "Folder of videos (sub-folders included)")

    def _rules_enable(self):
        k = self.keep_when.currentData()
        self.keep_box.setEnabled(k in ("any_of", "all_of"))
        self.min_boxes.setVisible(k == "detections")

    def _pattern_enable(self):
        on = self.group_get() == "pattern"
        self.pattern.setEnabled(on)
        self.pattern_test.setVisible(on)

    def _names_for_tests(self, n=6):
        if getattr(self, "_frame_names", None):          # the run folder's frames (read in the background)
            return self._frame_names
        src = self.source.text()
        if src and os.path.isdir(src) and self.src_get() == "videos":
            vids = E.find_videos(src, self.recursive.isChecked(), self.include.text(), self.exclude.text())[:200]
            stem = lambda v: os.path.splitext(os.path.basename(v))[0]
            return [f"{stem(v)}/{stem(v)}_frame_0.jpg" for v in vids]
        return []

    def _package_hint(self):
        r = self.recipe()
        names = self._names_for_tests()
        if r.group_by == "pattern":
            if not r.group_pattern:
                self.pattern_test.setText("Type a pattern; the folder each name goes to is shown here.")
            else:
                import re
                try:
                    re.compile(r.group_pattern)
                    ex = []
                    for n in names[:: max(1, len(names) // 5)][:5]:
                        ex.append(f"{os.path.basename(n)} → <b>{E.group_of(r, n)}</b>")
                    self.pattern_test.setText("<br>".join(ex) if ex else "No file names to test yet.")
                except re.error as exc:
                    self.pattern_test.setText(f"<span style='color:{self._hex('error')}'>Not a valid pattern: "
                                              f"{exc}</span>")
        groups = {}
        for n in names:
            g = E.group_of(r, n)
            groups[g] = groups.get(g, 0) + 1
        if getattr(self, "_frame_names", None):
            n = int(r.split_size)
            plan = {g: (-(-c // n) if n > 0 else 1, c) for g, c in sorted(groups.items(),
                                                                         key=lambda kv: E.natural_key(kv[0]))}
            parts = sum(p for p, _ in plan.values())
            files = sum(c for _, c in plan.values())
            top = ", ".join(f"{g or 'all'} ({plan[g][1]:,})" for g in list(plan)[:6])
            self.pkg_hint.setText(f"With the frames there now: {files:,} frames → {len(plan)} folder"
                                  f"{'s' if len(plan) != 1 else ''}, {parts} part{'s' if parts != 1 else ''}"
                                  f"{' (zipped)' if r.zip_parts else ''}.  {top}{' …' if len(plan) > 6 else ''}")
        elif groups:
            self.pkg_hint.setText("Folders from the video names: " + ", ".join(
                f"{g or 'all'} ({c} video{'s' if c != 1 else ''})" for g, c in list(groups.items())[:8]))
        else:
            self.pkg_hint.setText("")
        if not r.package_images and not r.image_data:
            self.pkg_hint.setText(f"<span style='color:{self._hex('warn')}'>Without the images the JSONs need "
                                  f"imageData (Annotate ▸ Save the image inside each JSON).</span>")

    def _count_rule(self):
        root = self.run_dir.text()
        if not root:
            self.rule_count.setText("")
            return
        ws = E.Workspace(root)
        if not os.path.isdir(ws.frames) and not os.path.isdir(ws.extra):
            self.rule_count.setText("")
            return
        r = self.recipe()

        def count():
            imgs = ws.frame_images() + E.find_images(ws.extra)
            have = keep = 0
            for p in imgs[:50000]:
                jp = os.path.splitext(p)[0] + ".json"
                if os.path.exists(jp):
                    have += 1
                    keep += E.keeps(r, LM.labels_in(jp))
            return len(imgs), have, keep

        def done(res):
            n, have, keep = res
            if not have:
                self.rule_count.setText("No annotation files yet — run Annotate first to see how many frames "
                                        "this rule keeps.")
            else:
                self.rule_count.setText(f"With the annotations there now this rule keeps <b>{keep:,}</b> of "
                                        f"{have:,} annotated frames ({100 * keep / have:.0f} %).")
        self._call(count, done)

    # ------------------------------------------------------------------ source / run folder
    def _pick_source(self):
        img = self.src_get() == "images"
        d = QFileDialog.getExistingDirectory(self, "Folder of images" if img else "Folder of videos",
                                             self.source.text() or "")
        if d:
            self.source.setText(d)

    def _pick_run(self):
        d = QFileDialog.getExistingDirectory(self, "Run folder (frames, annotations and packages go here)",
                                             self.run_dir.text() or os.path.dirname(self.source.text() or ""))
        if d:
            self.run_dir.setText(d)

    def _refresh_source(self):
        src = self.source.text()
        if not src:
            self.src_info.setText("")
            return
        if not os.path.exists(src):
            self.src_info.setText(f"<span style='color:{self._hex('error')}'>Folder not found</span>")
            return
        img = self.src_get() == "images"
        rec, inc, exc = self.recursive.isChecked(), self.include.text(), self.exclude.text()

        def scan():
            if img:
                return len(E.find_images(src, rec, inc, exc)), 0
            vids = E.find_videos(src, rec, inc, exc)
            return len(vids), sum(os.path.getsize(v) for v in vids if os.path.exists(v))

        def done(res):
            n, size = res
            what = "image" if img else "video"
            gb = f" · {size / 2 ** 30:,.1f} GB" if size > 2 ** 30 else f" · {size / 2 ** 20:,.0f} MB" if size else ""
            self.src_info.setText(f"{n:,} {what}{'s' if n != 1 else ''}{gb}" if n else
                                  f"<span style='color:{self._hex('warn')}'>No {what}s found"
                                  f"{' (check the name filters on the Frames page)' if inc or exc else ''}</span>")
            if not self.run_dir.text() and n:
                self.run_dir.setText(os.path.join(os.path.dirname(os.path.abspath(src)),
                                                  os.path.basename(os.path.abspath(src)) + "_autoannotate"))
            self._package_hint()
        self._call(scan, done)

    def _refresh_run(self):
        root = self.run_dir.text()
        self.open_run_btn.setEnabled(bool(root) and os.path.isdir(root))
        if not root or not os.path.isdir(root):
            self._frame_names = []
            self.run_info.setText("A new folder — it is created when the run starts." if root else "")
            for s in STAGES:
                self._row_status(s, {}, None, False)
            self.qc_btn.setEnabled(False)
            self.summary_btn.setEnabled(False)
            self._update_steps()
            return
        ws = E.Workspace(root)

        def look():
            names = [os.path.relpath(p, ws.frames).replace(os.sep, "/") for p in ws.frame_images()]
            c = ws.counts()
            st = {s: ws.stage_record(s) for s in STAGES}
            stale = {s: ws.stale(s) for s in STAGES}
            src = ws.state().get("source", "")
            ws.close()
            return c, st, stale, src, names

        def done(res):
            c, st, stale, src, names = res
            self._frame_names = names
            bits = [f"{c['frames']:,} frames", f"{c['annotated']:,} annotated"]
            if c["similar"]:
                bits.append(f"{c['similar']:,} similar set aside")
            if c["extra"]:
                bits.append(f"{c['extra']:,} in extra_images")
            if c["parts"]:
                bits.append(f"{c['parts']:,} zip files")
            self.run_info.setText(" · ".join(bits) if c["frames"] or c["similar"] or c["extra"] else
                                  "Empty — every step starts fresh.")
            for s in STAGES:
                self._row_status(s, st[s], c, stale[s])
            self.qc_btn.setEnabled(c["frames"] > 0)
            self.summary_btn.setEnabled(os.path.isfile(os.path.join(root, "reports", "summary.html")))
            if src and not self.source.text():
                self.source.setText(src)
            self._package_hint()
            self._count_timer.start()
            self._update_steps()
        self._call(look, done)

    def _row_status(self, s, rec, counts, stale):
        status = rec.get("status") if rec else None
        stats = (rec or {}).get("stats") or {}
        if status == "done":
            txt = {"frames": lambda: f"{stats.get('frames', 0):,} frames",
                   "similar": lambda: f"{stats.get('similar', 0):,} set aside",
                   "annotate": lambda: f"{stats.get('boxes', 0):,} boxes in {stats.get('frames', 0):,} frames",
                   "rules": lambda: f"kept {stats.get('kept', 0):,} · set aside {stats.get('set_aside', 0):,}",
                   "package": lambda: f"{stats.get('parts', 0):,} parts · {stats.get('files', 0):,} frames"}[s]()
            self.rows[s].set_status(("Out of date — run again · " if stale else "Done · ") + txt,
                                    "stale" if stale else "done")
        elif status == "stopped":
            self.rows[s].set_status("Stopped — Run continues it", "stale")
        elif status == "failed":
            self.rows[s].set_status("Failed — see the log", "failed")
        elif status == "running":
            self.rows[s].set_status("Interrupted — Run continues it", "stale")
        else:
            self.rows[s].set_status("Not run yet", "off")

    # ------------------------------------------------------------------ steps
    def _select(self, s):
        self._selected = s
        for k, row in self.rows.items():
            row.set_selected(k == s)
        if self.isVisible():
            from AIO_QC.widgets import anim
            anim.crossfade(self.pages, 140)
        self.pages.setCurrentIndex(self.page_index[s])
        if s == "rules":
            self._count_timer.start()
        if s == "package":
            self._package_hint()

    def _preset_chosen(self, i):
        stages = self.preset_combo.itemData(i)
        if stages is None:
            return
        for s in STAGES:
            self.rows[s].check.setChecked(s in stages)
        self.preset_combo.setCurrentIndex(0)

    def _update_steps(self):
        if self._loading:
            return
        chosen = [s for s in STAGES if self.rows[s].check.isChecked()]
        if self.worker is None:
            n = len(chosen)
            self.run_btn.setText("Run" if n == 0 else f"Run {STAGE_TITLES[chosen[0]]} only" if n == 1 else
                                 f"Run {n} steps")
            rr = self.recipe()
            problems = [p for p in rr.problems((self._model_names or None) if rr.annotator != "ai" else None)
                        if E._problem_applies(p, chosen)] if chosen else ["Tick at least one step."]
            if chosen and "annotate" in chosen and rr.annotator == "ai":
                problems = [p for p in problems if not p.startswith("Annotate: choose or make")] + \
                    self.ai_panel.run_problems()
            self.run_btn.setEnabled(not problems)
            self.run_btn.setToolTip("\n".join(problems) if problems else "Runs the ticked steps in order")
            note = []
            if "frames" not in chosen and chosen and not self._have_frames():
                note.append("Frames is not ticked and the run folder has no frames yet.")
            if problems:
                note += problems[:2]
            self.steps_note.setText("<br>".join(note))
            style(self.steps_note, "QLabel{color:@warn;}" if note else "QLabel{color:@muted;}")
        for s in STAGES:
            self.rows[s].num.setEnabled(self.rows[s].check.isChecked())
        self._save_ui()

    def _have_frames(self):
        root = self.run_dir.text()
        return bool(root) and os.path.isdir(os.path.join(root, E.FRAMES)) and any(
            True for _ in os.scandir(os.path.join(root, E.FRAMES)))

    def _save_ui(self):
        if self._loading:
            return
        st = _settings()
        try:
            st.setValue(f"{SETTINGS}/recipe", json.dumps(self.recipe().to_dict()))
        except Exception as _exc:
            _ignored(_exc)
        st.setValue(f"{SETTINGS}/source", self.source.text())
        st.setValue(f"{SETTINGS}/run", self.run_dir.text())

    # ------------------------------------------------------------------ setups
    def _refresh_setups(self, select=None):
        self.setup_combo.blockSignals(True)
        self.setup_combo.clear()
        self.setup_combo.addItem("(unsaved)", None)
        d = setups_dir()
        files = sorted(f for f in os.listdir(d) if f.endswith(SETUP_SUFFIX))
        for f in files:
            self.setup_combo.addItem(f[:-len(SETUP_SUFFIX)], os.path.join(d, f))
        cur = select or _settings().value(f"{SETTINGS}/setup", "")
        i = self.setup_combo.findData(cur) if cur else 0
        self.setup_combo.setCurrentIndex(max(0, i))
        self.setup_combo.blockSignals(False)

    def _setup_chosen(self, i):
        p = self.setup_combo.itemData(i)
        if not p:
            return
        try:
            self.set_recipe(Recipe.load(p))
            _settings().setValue(f"{SETTINGS}/setup", p)
            self.status.setText(f"Setup “{self.setup_combo.itemText(i)}” loaded.")
        except Exception as exc:
            _ignored(exc)
            QMessageBox.warning(self, self.TITLE, f"Could not read the setup:\n{exc}")

    def _save_setup(self):
        from PyQt6.QtWidgets import QInputDialog
        cur = self.setup_combo.currentData()
        name0 = self.setup_combo.currentText() if cur else (self.recipe().name or "")
        name, ok = QInputDialog.getText(self, "Save setup", "Name of this setup (e.g. the project):", text=name0)
        name = (name or "").strip()
        if not ok or not name:
            return
        import re
        safe = re.sub(r'[<>:"/\\|?*]+', "_", name)
        path = os.path.join(setups_dir(), safe + SETUP_SUFFIX)
        r = self.recipe()
        r.name = name
        r.save(path)
        _settings().setValue(f"{SETTINGS}/setup", path)
        self._refresh_setups(path)
        self.status.setText(f"Setup “{name}” saved.")

    def _load_setup_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load setup", setups_dir(), "Auto-Annotate setup (*.json)")
        if p:
            try:
                self.set_recipe(Recipe.load(p))
                self.status.setText(f"Setup loaded from {os.path.basename(p)}.")
            except Exception as exc:
                _ignored(exc)
                QMessageBox.warning(self, self.TITLE, f"Could not read the setup:\n{exc}")

    def _save_setup_file(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save setup as", os.path.join(setups_dir(), "setup" + SETUP_SUFFIX),
                                           "Auto-Annotate setup (*.json)")
        if p:
            self.recipe().save(p)
            self.status.setText(f"Setup saved to {p}")

    def _show_cli(self):
        cur = self.setup_combo.currentData()
        setup = cur or "<setup>" + SETUP_SUFFIX
        steps = ",".join(self.recipe().stages)
        src = self.source.text() or "<videos>"
        cmd = (f'aio-qc autoannotate "{self.run_dir.text() or "<run folder>"}" --{self.src_get()} "{src}" '
               f'--setup "{setup}" --only {steps}')
        box = QMessageBox(self)
        box.setWindowTitle("Command line")
        box.setText("The same run without the window (e.g. on a GPU server):" +
                    ("" if cur else "\n\nSave the setup first — the command uses its file."))
        box.setDetailedText(cmd)
        box.setInformativeText(cmd)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(cmd)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    # ------------------------------------------------------------------ preview
    def _preview(self):
        r = self.recipe()
        src = self.source.text()
        if not r.model or not os.path.isfile(r.model):
            self.preview_note.setText("Choose the model first.")
            return
        root = self.run_dir.text()
        ws = E.Workspace(root) if root else None
        have = bool(getattr(self, "_frame_names", None)) and ws is not None
        if not have and (not src or not os.path.exists(src)):
            self.preview_note.setText("Choose the source folder first.")
            return
        key = (r.model, r.imgsz, r.device, r.half, r.iou, r.max_det)
        det = self._preview_det if self._preview_key == key else None
        env = _env_python()
        self.preview_btn.setEnabled(False)
        self.preview_note.setText("Loading the model and reading a few frames…")

        def work():
            if have:
                from AIO_QC.tools.frame_extractor.core.imageio import imread
                imgs = ws.frame_images()
                picks = imgs[::max(1, len(imgs) // 4)][:4]
                frames = [(os.path.basename(p), imread(p)) for p in picks]
                frames = [(t, f) for t, f in frames if f is not None]
            else:
                frames = E.sample_frames(r, src, 4)
            if not frames:
                raise E.StepError("annotate", "No frames could be read from the source.")
            if det is not None:
                lo = min([float(r.conf)] + [e["conf"] for c in r.classes for e in [r.class_entry(c)] if e["conf"]])
                det.conf = lo
            return E.preview(r, frames, env, det)

        def done(res):
            out, d, _names = res
            self._preview_det, self._preview_key = d, key
            self.preview_btn.setEnabled(True)
            self._show_preview(out)

        def fail(msg):
            self.preview_btn.setEnabled(True)
            self.preview_note.setText(f"<span style='color:{self._hex('error')}'>{msg}</span>")
        self._call(work, done, fail)

    def _show_preview(self, out):
        while self.preview_grid.count():
            w = self.preview_grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        r = self.recipe()
        total = 0
        kept = 0
        for i, (title, img, shapes) in enumerate(out):
            cell = QWidget()
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(3)
            pic = QLabel()
            pic.setPixmap(bgr_to_pixmap(img, shapes, 420))
            pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
            counts = {}
            for s in shapes:
                counts[s["label"]] = counts.get(s["label"], 0) + 1
            k = E.keeps(r, [s["label"] for s in shapes])
            kept += k
            total += len(shapes)
            cap = QLabel(f"<b>{title}</b><br>" + (", ".join(f"{n} × {c}" for c, n in counts.items()) or "no boxes")
                         + f"<br><span style='color:{self._hex('ok' if k else 'warn')}'>"
                           f"{'kept by the rule' if k else 'set aside by the rule'}</span>")
            cap.setWordWrap(True)
            style(cap, "QLabel{color:@muted; font-size:9pt;}")
            cl.addWidget(pic)
            cl.addWidget(cap)
            self.preview_grid.addWidget(cell, i // 2, i % 2)
        self.preview_area.setVisible(True)
        sc = self.pages.widget(self.page_index["annotate"])
        QTimer.singleShot(60, lambda: sc.ensureWidgetVisible(self.preview_area, 0, 40))
        self.preview_note.setText(f"{total} box{'es' if total != 1 else ''} on {len(out)} frames · the keep rule "
                                  f"would keep {kept} of {len(out)}. Change a setting and try again.")

    # ------------------------------------------------------------------ run
    def start(self, stages=None, replace_previous=False):
        if self.worker is not None:
            return
        r = self.recipe()
        stages = [s for s in STAGES if s in (stages if stages is not None else r.stages)]
        if not stages:
            self.status.setText("Tick at least one step.")
            return
        self.ai_panel.flush()
        problems = [p for p in r.problems((self._model_names or None) if r.annotator != "ai" else None)
                    if E._problem_applies(p, stages)]
        if "annotate" in stages and r.annotator == "ai":
            problems = [p for p in problems if not p.startswith("Annotate: choose or make")] + \
                self.ai_panel.run_problems()
        if problems:
            QMessageBox.warning(self, self.TITLE, "\n".join(problems))
            return
        root = self.run_dir.text()
        if not root:
            QMessageBox.information(self, self.TITLE, "Choose the run folder first (where frames and annotations go).")
            return
        src = self.source.text()
        if "frames" in stages and (not src or not os.path.exists(src)):
            QMessageBox.information(self, self.TITLE, "Choose the folder of videos (or images) first.")
            return
        inside = False
        if src and r.source == "images" and os.path.isdir(src):
            a, b = os.path.normcase(os.path.abspath(root)), os.path.normcase(os.path.abspath(src))
            try:
                inside = os.path.commonpath([a, b]) == b
            except ValueError:                     # different drives
                inside = False
        if inside:
            QMessageBox.warning(self, self.TITLE, "The run folder must not be inside the image folder (the frames "
                                                  "would be read again as new images). Choose another run folder.")
            return
        if r.model:
            ms = [p for p in (_settings().value(f"{SETTINGS}/models", []) or []) if p != r.model][:9]
            _settings().setValue(f"{SETTINGS}/models", [r.model] + ms)
        self._last_stages = stages
        self._save_ui()
        self.log.appendPlainText(f"— {', '.join(STAGE_TITLES[s] for s in stages)}")
        self.worker = RunWorker(root, r, stages, src, _env_python(), replace_previous, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.logged.connect(lambda s, t: self.log.appendPlainText(f"[{STAGE_TITLES.get(s, s)}] {t}"))
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.stopped.connect(self._on_stopped)
        self.worker.confirm.connect(self._on_confirm)
        self.worker.finished.connect(self._on_thread_end)
        from AIO_QC.widgets import tasks
        self.task = tasks.start(f"Auto-Annotate · {', '.join(STAGE_TITLES[s] for s in stages)}", cancel=self.stop,
                                owner=self)
        self._busy(True)
        self.status.setText("Starting…")
        self.worker.start()

    def stop(self):
        if self.worker is not None:
            self.worker.cancel.set()
            self.stop_btn.setEnabled(False)
            self.status.setText("Stopping safely — finishing the current frames…")

    def _busy(self, on):
        self.run_btn.setVisible(not on)
        self.run_btn.setDefault(not on)
        self.stop_btn.setVisible(on)
        self.stop_btn.setEnabled(on)
        self.bar.setVisible(on)
        self.bar.setRange(0, 0) if on else None
        for w in (self.src_seg, self.source, self.run_dir, self.setup_combo, self.preset_combo, self.preview_btn):
            w.setEnabled(not on)
        for s, row in self.rows.items():
            row.check.setEnabled(not on)
        for i in range(self.pages.count()):
            for b in self.pages.widget(i).findChildren(QPushButton):
                if b.objectName().startswith("only_"):
                    b.setEnabled(not on)

    def _on_progress(self, stage, d, t, text):
        for s in STAGES:
            self.rows[s].set_selected(s == stage) if self.worker is not None else None
        if self.rows[stage].status.text() != "Running…":
            self.rows[stage].set_status("Running…", "stale")
        if t > 0:
            self.bar.setRange(0, t)
            self.bar.setValue(d)
        else:
            self.bar.setRange(0, 0)
        n = self._last_stages.index(stage) + 1 if stage in self._last_stages else 0
        step = f"Step {n} of {len(self._last_stages)} · " if len(self._last_stages) > 1 and n else ""
        pct = f" · {100 * d / t:.0f} %" if t else ""
        self.status.setText(f"<b>{step}{STAGE_TITLES.get(stage, stage)}</b>{pct}"
                            + (f" · {text}" if text else ""))
        if self.task is not None:
            self.task.update(d / t if t else None, f"{STAGE_TITLES.get(stage, stage)} {text}")

    def _on_thread_end(self):
        self.worker = None
        if self.task is not None:
            self.task.finish()
            self.task = None
        self._busy(False)
        self._select(self._selected)
        self._refresh_run()
        self._update_steps()

    def _on_done(self, res):
        lines = []
        for s, st in res.items():
            lines.append(f"[{STAGE_TITLES[s]}] " + ", ".join(
                f"{k.replace('_', ' ')} {v:,}" for k, v in st.items() if isinstance(v, int) and not isinstance(v, bool)))
        self.log.appendPlainText("\n".join(lines))
        last = list(res)[-1] if res else ""
        st = res.get(last, {})
        msg = {"frames": lambda: f"{st.get('frames', 0):,} frames in the run folder.",
               "similar": lambda: f"{st.get('similar', 0):,} similar frames set aside, {st.get('kept', 0):,} kept.",
               "annotate": lambda: self._annotate_msg(st),
               "rules": lambda: f"Kept {st.get('kept', 0):,} frames, set aside {st.get('set_aside', 0):,}.",
               "package": lambda: f"{st.get('parts', 0):,} parts with {st.get('files', 0):,} frames in package/."
               }.get(last, lambda: "")()
        self.status.setText(f"<span style='color:{self._hex('ok')}'>Done.</span> {msg}")

    @staticmethod
    def _annotate_msg(st):
        msg = f"{st.get('boxes', 0):,} boxes in {st.get('frames', 0):,} frames."
        if "rejected" in st:
            msg += f" Rejected {st['rejected']:,} (in rejected/), {st.get('needs_review', 0):,} need review"
            msg += f", cost ${st['cost_usd']:.2f}." if st.get("cost_usd") else "."
            if st.get("finished_early"):
                msg += f" Finished early: {st['finished_early']}."
        return msg

    def _on_failed(self, stage, msg):
        self.log.appendPlainText(f"[{STAGE_TITLES.get(stage, stage)}] failed: {msg}")
        self.status.setText(f"<span style='color:{self._hex('error')}'>{STAGE_TITLES.get(stage, stage)} stopped "
                            f"with a problem:</span> {msg}")
        if not self.log.isVisible():
            self._toggle_log()

    def _on_stopped(self, stage, why=""):
        self.log.appendPlainText(f"[{STAGE_TITLES.get(stage, stage)}] stopped{': ' + why if why else ''}")
        self.status.setText((f"<span style='color:{self._hex('warn')}'>{why}</span> " if why else "Stopped. ")
                            + "Everything done so far is kept — Run continues where it stopped.")

    def _on_confirm(self, msg, count):
        stages = list(self._last_stages)

        def ask():
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle(self.TITLE)
            box.setText(msg)
            box.setInformativeText("Your videos and image folders are never touched — only the run folder.")
            rep = box.addButton("Replace and extract again", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is rep:
                self.start(stages, replace_previous=True)
            else:
                self.status.setText("Nothing changed. Choose another run folder to keep the previous frames.")
        QTimer.singleShot(0, ask)

    # ------------------------------------------------------------------ misc
    def _call(self, fn, done, fail=None):
        w = CallWorker(fn, self)
        self._calls.append(w)

        def end():
            if w in self._calls:
                self._calls.remove(w)
        def ok(res):
            try:
                done(res)
            except RuntimeError:                 # the window closed meanwhile
                pass
        w.done.connect(ok)
        w.error.connect(fail or (lambda m: logger.debug(f"auto-annotate: {m}")))
        w.finished.connect(end)
        w.start()

    def _toggle_log(self):
        self.log.setVisible(not self.log.isVisible())

    def _open_path(self, p):
        if p and os.path.exists(p):
            from PyQt6.QtCore import QUrl
            from PyQt6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def _open_summary(self):
        self._open_path(os.path.join(self.run_dir.text(), "reports", "summary.html"))

    def _open_in_qc(self):
        from PyQt6.QtWidgets import QApplication
        folder = os.path.join(self.run_dir.text(), E.FRAMES)
        shell = getattr(QApplication.instance(), "shell", None)
        if shell is None:
            self._open_path(folder)
            return
        shell.open_in_qc(folder)

    def closeEvent(self, e):
        if self.worker is not None:
            box = QMessageBox.question(self, self.TITLE, "A run is in progress. Stop it and close?\n\nEverything "
                                                         "done so far is kept; Run continues later.")
            if box != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.worker.cancel.set()
            self.worker.wait(15000)
        self._save_ui()
        self.ai_panel.flush()
        self.ai_panel.wait_calls()
        for w in list(self._calls):
            w.wait(3000)
        super().closeEvent(e)


def open_window(parent=None, source="", run_folder=""):
    """One Auto-Annotate window per app (brought to the front if it is open)."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    win = getattr(app, "_autoannotate_window", None)
    try:
        if win is not None and win.isVisible():
            if source and win.worker is None:
                win.source.setText(source)
            win.raise_()
            win.activateWindow()
            return win
    except RuntimeError:
        win = None
    win = AutoAnnotateWindow(parent, source, run_folder)
    win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    app._autoannotate_window = win
    win.show()
    return win
