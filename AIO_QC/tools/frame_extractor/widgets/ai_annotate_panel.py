# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
"AI annotate" — the Annotate step's second annotator: a project (config.yaml) edited entirely in the app.

Top: "Ready?" — the app, the Python environment, the key and the AI service, each checked with a button that fixes
it (install, choose the environment, enter the key once, test). Below: the project, and seven plain-language tabs —
Connect, Model, Classes, Examples, Checks, Output, Try — plus the YAML for those who want it. Every change is saved
to the project file a moment later; nothing needs to be typed into the file by hand.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys

from loguru import logger
from PyQt6.QtCore import QProcess, QProcessEnvironment, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                             QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                             QPlainTextEdit, QRadioButton, QSpinBox,
                             QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from AIO_QC.tools.frame_extractor.core.ai_annotate import config as C
from AIO_QC.tools.frame_extractor.core.ai_annotate import keys as K
from AIO_QC.tools.frame_extractor.core.ai_annotate import profiles as PF
from AIO_QC.tools.frame_extractor.core.ai_annotate import readiness as RD
from AIO_QC.utils import newButton, style
from AIO_QC.widgets.lifecycle import RetainedThread
from AIO_QC.core.diag import ignored as _ignored

STATUS_MARK = {"ok": ("✓", "ok"), "warn": ("!", "warn"), "fail": ("✕", "error"), "skip": ("–", "faint"),
               "pending": ("…", "faint")}


def _hex(role):
    from AIO_QC.utils import hexc
    return hexc(role)


def _hint(text, wrap=True):
    lab = QLabel(text)
    lab.setWordWrap(wrap)
    style(lab, "QLabel{color:@muted;}")
    return lab


def _section(text):
    lab = QLabel(text)
    style(lab, "QLabel{font-weight:600; color:@text; padding-top:8px;}")
    return lab


def projects_dir() -> str:
    from AIO_QC.tools.frame_extractor.widgets.autoannotate_window import setups_dir
    d = os.path.join(setups_dir(), "ai_projects")
    os.makedirs(d, exist_ok=True)
    return d


def env_python() -> str:
    from AIO_QC.utils.qt_utils import app_settings
    return app_settings().value("inference/python", "") or ""


class Call(RetainedThread, QThread):
    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as exc:
            logger.opt(exception=True).debug("AI annotate background call")
            self.error.emit(str(exc) or type(exc).__name__)


# ====================================================================================================== dialogs
class InstallDialog(QDialog):
    """pip install into a Python, with its output shown."""

    def __init__(self, python, packages, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.ok = False
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"Installing <b>{', '.join(packages)}</b> into<br><code>{python}</code>"))
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        from AIO_QC.widgets.design import mono
        self.out.setFont(mono(11))
        self.out.setMinimumSize(640, 260)
        lay.addWidget(self.out)
        self.bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.bb.rejected.connect(self.reject)
        lay.addWidget(self.bb)
        self.p = QProcess(self)
        self.p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.p.setProcessEnvironment(QProcessEnvironment.systemEnvironment())
        self.p.readyRead.connect(lambda: self.out.appendPlainText(
            bytes(self.p.readAll()).decode("utf-8", "replace").rstrip()))
        self.p.finished.connect(self._end)
        self.p.start(python, ["-m", "pip", "install", "--disable-pip-version-check"] + list(packages))

    def _end(self, code, _status):
        self.ok = code == 0
        self.out.appendPlainText("\n✓ Installed." if self.ok else f"\n✕ pip stopped with code {code}.")

    def reject(self):
        if self.p.state() != QProcess.ProcessState.NotRunning:
            self.p.kill()
            self.p.waitForFinished(3000)
        super().reject()


class EnvDialog(QDialog):
    """Choose the Python environment for local models (shared with the app's .pt models)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Python environment")
        lay = QVBoxLayout(self)
        lay.addWidget(_hint("The Python environment that runs models on your computer (the same one your .pt models "
                            "use). It needs PyTorch; the check shows what else is missing and installs it."))
        self.list = QListWidget()
        self.list.setMinimumSize(560, 200)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        row.addWidget(newButton("Browse for python…", icon="folder-line", slot=self._browse))
        row.addStretch(1)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        cur = env_python()
        try:
            from AIO_QC.tools.frame_extractor.core import env_manager as EM
            envs = EM.discover_envs()
        except Exception as _exc:
            _ignored(_exc)
            envs = []
        if cur and all(p != cur for _, p in envs):
            envs.insert(0, ("current", cur))
        for name, py in envs:
            it = QListWidgetItem(f"{name}   —   {py}")
            it.setData(Qt.ItemDataRole.UserRole, py)
            self.list.addItem(it)
            if py == cur:
                self.list.setCurrentItem(it)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Python of the environment", "",
                                           "Python (python.exe python python3*);;All files (*)")
        if p:
            it = QListWidgetItem(p)
            it.setData(Qt.ItemDataRole.UserRole, p)
            self.list.addItem(it)
            self.list.setCurrentItem(it)

    def chosen(self):
        it = self.list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else ""


class BoxDialog(QDialog):
    """Draw the boxes of an example image (saved as a LabelMe file next to it)."""

    def __init__(self, path, classes, parent=None):
        super().__init__(parent)
        from AIO_QC.tools.frame_extractor.core.ai_annotate.writers import read_labels
        self.setWindowTitle(f"Boxes of {os.path.basename(path)}")
        self.path = path
        self.boxes = [list(b) for b in [(lab, *box) for lab, box in read_labels(path)]]
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Class"))
        self.cls = QComboBox()
        self.cls.addItems(classes or ["object"])
        self.cls.setEditable(True)
        top.addWidget(self.cls)
        top.addWidget(_hint("Drag on the picture to draw a box · click a box and press Delete to remove it", False))
        top.addStretch(1)
        lay.addLayout(top)
        self.canvas = _BoxCanvas(path, self)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(_hint(f"Saved as a LabelMe file next to the image ({os.path.splitext(os.path.basename(path))[0]}"
                            f".json) — the examples then show the AI exactly which objects you want, and how tight."))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        from AIO_QC.utils.qt_utils import fit_size
        fit_size(self, "large")

    def _save(self):
        from AIO_QC.tools.frame_extractor.core.autoannotate import labelme_io as LM
        h, w = self.canvas.img_size
        jp = os.path.splitext(self.path)[0] + ".json"
        shapes = [LM.rectangle(b[0], *b[1:5]) for b in self.boxes]
        try:
            LM.write(jp, LM.document(self.path, jp, w, h, shapes))
        except OSError as exc:
            QMessageBox.warning(self, "Boxes", f"Could not save next to the image:\n{exc}")
            return
        self.accept()


class _BoxCanvas(QWidget):
    def __init__(self, path, dlg):
        super().__init__(dlg)
        from AIO_QC.tools.frame_extractor.core.imageio import imread
        self.dlg = dlg
        img = imread(path)
        self.img_size = img.shape[:2] if img is not None else (1, 1)
        import numpy as np
        if img is not None:
            from PyQt6.QtGui import QImage
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            self.pm = QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
                                               QImage.Format.Format_RGB888).copy())
        else:
            self.pm = QPixmap()
        self.setMinimumSize(640, 400)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.drag = None
        self.sel = -1

    def _xf(self):
        if self.pm.isNull():
            return 1.0, 0, 0
        s = min(self.width() / self.pm.width(), self.height() / self.pm.height())
        return s, (self.width() - self.pm.width() * s) / 2, (self.height() - self.pm.height() * s) / 2

    def _to_img(self, p):
        s, ox, oy = self._xf()
        return ((p.x() - ox) / s, (p.y() - oy) / s)

    def paintEvent(self, _e):
        from PyQt6.QtCore import QRectF
        from PyQt6.QtGui import QPainter, QPen
        p = QPainter(self)
        s, ox, oy = self._xf()
        if not self.pm.isNull():
            p.drawPixmap(QRectF(ox, oy, self.pm.width() * s, self.pm.height() * s), self.pm,
                         QRectF(0, 0, self.pm.width(), self.pm.height()))
        for i, b in enumerate(self.dlg.boxes + ([["", *self.drag]] if self.drag else [])):
            c = QColor("#FF3B30") if i != self.sel else QColor("#FFCC00")
            p.setPen(QPen(c, 2))
            r = QRectF(ox + b[1] * s, oy + b[2] * s, (b[3] - b[1]) * s, (b[4] - b[2]) * s)
            p.drawRect(r)
            if b[0]:
                fm = p.fontMetrics()
                tw, th = fm.horizontalAdvance(b[0]) + 8, fm.height() + 2
                tag = QRectF(r.left(), max(0.0, r.top() - th), tw, th)
                p.fillRect(tag, c)
                p.setPen(QColor("#FFFFFF"))
                p.drawText(tag, int(Qt.AlignmentFlag.AlignCenter), b[0])
        p.end()

    def mousePressEvent(self, e):
        x, y = self._to_img(e.position())
        hit = [i for i, b in enumerate(self.dlg.boxes) if b[1] <= x <= b[3] and b[2] <= y <= b[4]]
        if hit:
            self.sel = hit[-1]
        else:
            self.sel = -1
            self.drag = [x, y, x, y]
        self.update()

    def mouseMoveEvent(self, e):
        if self.drag:
            x, y = self._to_img(e.position())
            self.drag[2], self.drag[3] = x, y
            self.update()

    def mouseReleaseEvent(self, e):
        if self.drag:
            h, w = self.img_size
            x1, y1, x2, y2 = self.drag
            x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
            y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
            if x2 - x1 > 3 and y2 - y1 > 3:
                self.dlg.boxes.append([self.dlg.cls.currentText().strip() or "object", x1, y1, x2, y2])
            self.drag = None
            self.update()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and 0 <= self.sel < len(self.dlg.boxes):
            del self.dlg.boxes[self.sel]
            self.sel = -1
            self.update()


# ====================================================================================================== panel
class AIAnnotatePanel(QWidget):
    changed = pyqtSignal()                 # the project (path or content) changed: the window re-checks Run
    project_changed = pyqtSignal(str)

    def __init__(self, window=None, parent=None):
        super().__init__(parent)
        self.win = window
        self.cfg = C.default()
        self.path = ""
        self._binds = []
        self._loading = False
        self._calls = []
        self._ready_items = []
        self._env_info_cache = {}
        self._save_t = QTimer(self, singleShot=True, interval=500, timeout=self._save_now)
        self._ready_t = QTimer(self, singleShot=True, interval=700, timeout=lambda: self.check_ready(True))
        self._yaml_t = QTimer(self, singleShot=True, interval=500, timeout=self._yaml_live)
        self._loading = True                       # widgets being made must not write into the project
        self._build()
        self._loading = False

    # ------------------------------------------------------------------ layout
    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        t = QLabel("AI annotate")
        style(t, "QLabel{font-size:12pt; font-weight:600;}")
        lay.addWidget(t)
        lay.addWidget(_hint("No trained model needed: describe your classes in plain words, add a few examples, and a "
                            "vision model (online or on your computer) annotates the frames."))
        # ---- ready card (first)
        self.ready_box = QFrame()
        self.ready_box.setObjectName("readyCard")
        rl = QVBoxLayout(self.ready_box)
        rl.setContentsMargins(12, 10, 12, 10)
        rl.setSpacing(6)
        rh = QHBoxLayout()
        self.ready_title = QLabel("<b>Is everything ready?</b>")
        rh.addWidget(self.ready_title)
        rh.addStretch(1)
        self.env_lbl = _hint("", False)
        rh.addWidget(self.env_lbl)
        rh.addWidget(newButton("Environment…", icon="settings", slot=self._choose_env,
                               tip="The Python environment for models that run on your computer"))
        self.recheck_btn = newButton("Check again", slot=lambda: self.check_ready(True))
        rh.addWidget(self.recheck_btn)
        rl.addLayout(rh)
        self.ready_rows = QGridLayout()
        self.ready_rows.setHorizontalSpacing(10)
        self.ready_rows.setVerticalSpacing(4)
        rl.addLayout(self.ready_rows)
        lay.addWidget(self.ready_box)

        # ---- the rest (enabled once the app itself is ready)
        self.body = QWidget()
        bl = QVBoxLayout(self.body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        pr = QHBoxLayout()
        pr.addWidget(QLabel("Project"))
        self.proj = QComboBox()
        self.proj.setMinimumWidth(260)
        self.proj.activated.connect(self._project_chosen)
        pr.addWidget(self.proj, 1)
        pr.addWidget(newButton("New…", icon="new", slot=self._new_project, tip="Start a project from a template"))
        pr.addWidget(newButton("Import…", icon="open", slot=self._import, tip="Use a config.yaml you have"))
        pr.addWidget(newButton("Export…", icon="save", slot=self._export,
                               tip="Save this project as a config.yaml (to share it or run it from the command line)"))
        bl.addLayout(pr)
        about = QFormLayout()
        about.setHorizontalSpacing(14)
        self.p_name = self._bind(QLineEdit(), "project")
        self.p_name.setPlaceholderText("a short name, used in reports")
        self.p_desc = self._bind(QLineEdit(), "description")
        self.p_desc.setPlaceholderText("optional: one line about the project — it is told to the AI, e.g. "
                                       "“yard cameras, trucks and trailers at the gate”")
        about.addRow("Name", self._row(self.p_name, QLabel("About"), self.p_desc))
        self.p_desc.setMinimumWidth(460)
        bl.addLayout(about)
        self.problems = QLabel("")
        self.problems.setWordWrap(True)
        bl.addWidget(self.problems)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        for title, build in (("1 · Connect", self._tab_connect), ("2 · Model", self._tab_model),
                             ("3 · Classes", self._tab_classes), ("4 · Examples", self._tab_examples),
                             ("5 · Checks", self._tab_checks), ("6 · Output", self._tab_output),
                             ("7 · Try", self._tab_try), ("YAML", self._tab_yaml)):
            w = QWidget()
            l = QVBoxLayout(w)
            l.setContentsMargins(8, 10, 8, 8)
            l.setSpacing(8)
            build(l)
            l.addStretch(1)
            self.tabs.addTab(w, title)
        self.tabs.currentChanged.connect(self._tab_changed)
        bl.addWidget(self.tabs)
        lay.addWidget(self.body)
        self.body.setEnabled(False)
        self._apply_style()

    def _apply_style(self):
        from AIO_QC.widgets.chrome import T
        self.ready_box.setStyleSheet(f"#readyCard {{ background:{T('surface')}; border:1px solid {T('border')}; "
                                     f"border-radius:8px; }}")

    def _form(self, lay):
        f = QFormLayout()
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        f.setHorizontalSpacing(14)
        f.setVerticalSpacing(8)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        lay.addLayout(f)
        return f

    def _row(self, *ws):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(8)
        for x in ws:
            l.addWidget(x)
        l.addStretch(1)
        return w

    # ------------------------------------------------------------------ binding
    def _get(self, path):
        d = self.cfg
        for k in path.split("."):
            d = d.get(k) if isinstance(d, dict) else None
        return d

    def _set(self, path, v):
        d = self.cfg
        ks = path.split(".")
        for k in ks[:-1]:
            d = d.setdefault(k, {})
        if d.get(ks[-1]) != v:
            d[ks[-1]] = v
            self._edited()

    # ------------------------------------------------------------------ models that run their own code
    def _trust_box(self, model_of, kind_of):
        """“Allow this model to run its own code” — per model, on this computer only (ai_annotate/trust.py)."""
        cb = QCheckBox("Allow this model to run its own code")
        cb.setToolTip("Some models (Florence-2, Moondream, several chat models) ship Python code that runs when they "
                      "load. It can do anything your account can — allow it only for a model from a source you trust. "
                      "Remembered for this model on this computer; never taken from a shared config file.")
        cb.model_of = model_of
        cb.kind_of = kind_of if callable(kind_of) else (lambda: kind_of)
        cb.clicked.connect(lambda *_a, cb=cb: self._trust_clicked(cb, cb.isChecked()))
        self._sync_trust(cb)
        return cb

    def _sync_trust(self, cb):
        from AIO_QC.tools.frame_extractor.core.ai_annotate import trust
        m = cb.model_of()
        cb.blockSignals(True)
        cb.setChecked(trust.trusted(m))
        cb.blockSignals(False)
        cb.setVisible(bool(m) and (trust.needs_own_code(cb.kind_of(), m) or trust.trusted(m)))

    def _trust_clicked(self, cb, on):
        from AIO_QC.tools.frame_extractor.core.ai_annotate import trust
        m = cb.model_of()
        if on and QMessageBox.question(
                self, "Allow the model's own code?",
                f"“{m}” runs Python code from its download when it loads. That code can read, change or send "
                "anything your user account can.\n\nAllow it only if the model comes from a source you trust "
                "(for example the official page of its authors).\n\nAllow it on this computer?") \
                != QMessageBox.StandardButton.Yes:
            on = False
        trust.set_trusted(m, on)
        self._sync_trust(cb)

    def _bind(self, w, path):
        self._binds.append((w, path))
        if isinstance(w, QLineEdit):
            w.textEdited.connect(lambda t, p=path: self._loading or self._set(p, t.strip()))
        elif isinstance(w, QDoubleSpinBox):
            w.valueChanged.connect(lambda v, p=path: self._loading or self._set(p, round(float(v), 4)))
        elif isinstance(w, QSpinBox):
            w.valueChanged.connect(lambda v, p=path: self._loading or self._set(p, int(v)))
        elif isinstance(w, QCheckBox):
            w.toggled.connect(lambda v, p=path: self._loading or self._set(p, bool(v)))
        elif isinstance(w, QComboBox):
            if w.isEditable():
                w.currentTextChanged.connect(lambda t, p=path, w=w: self._loading or self._set(
                    p, (w.currentData() if w.currentText() == w.itemText(w.currentIndex()) and w.currentData()
                        is not None else t).strip() if isinstance(t, str) else t))
            else:
                w.currentIndexChanged.connect(lambda _i, p=path, w=w: self._loading or self._set(p, w.currentData()))
        elif isinstance(w, QPlainTextEdit):
            w.textChanged.connect(lambda p=path, w=w: self._loading or self._set(p, w.toPlainText()))
        return w

    def _push(self):
        """cfg -> widgets."""
        self._loading = True
        try:
            for w, path in self._binds:
                v = self._get(path)
                if isinstance(w, QLineEdit):
                    if w.text() != str(v or ""):
                        w.setText(str(v or ""))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(v or 0))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(v or 0))
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(v))
                elif isinstance(w, QComboBox):
                    i = w.findData(v)
                    if i >= 0:
                        w.setCurrentIndex(i)
                    elif w.isEditable():
                        w.setEditText(str(v or ""))
                elif isinstance(w, QPlainTextEdit):
                    if w.toPlainText() != str(v or ""):
                        w.setPlainText(str(v or ""))
            self._fill_classes()
            self._fill_flags()
            self._fill_views()
            self._fill_fields()
            self._fill_rules()
            self._fill_targets()
            self._sync_visibility()
            self._show_profile()
            self._show_key()
            b = self.mode_btns.get(self.cfg["mode"])
            if b is not None:
                b.setChecked(True)
            self.fmt_set(self.cfg["output"]["format"])
            self.lm_set(self.cfg["output"]["labelme_version"])
        finally:
            self._loading = False
        self._validate()

    def _edited(self):
        if self._loading:
            return
        self._save_t.start()
        self._sync_visibility()
        self._validate()
        self.changed.emit()

    # ------------------------------------------------------------------ tabs
    def _tab_connect(self, lay):
        lay.addWidget(_hint("Where the vision model runs. A LiteLLM proxy or any OpenAI-compatible address works for "
                            "most models; Gemini and Claude can also be reached directly."))
        f = self.connect_form = self._form(lay)
        self.provider = QComboBox()
        for k, (title, *_rest) in PF.PROVIDERS.items():
            self.provider.addItem(title, k)
        self.provider.currentIndexChanged.connect(self._provider_changed)
        f.addRow("AI service", self.provider)
        self.provider_note = _hint("")
        f.addRow("", self.provider_note)
        self.base_url = self._bind(QLineEdit(), "connection.base_url")
        self.base_url.setMinimumWidth(420)
        self.base_url.setPlaceholderText("https://… or http://host:port")
        self.base_url.textEdited.connect(lambda _t: self._ready_t.start())
        self.url_row = self.base_url
        f.addRow("Address", self.base_url)
        self.api_version = self._bind(QLineEdit(), "connection.api_version")
        self.api_version.setPlaceholderText("2024-10-21")
        f.addRow("API version", self.api_version)
        self.region = self._bind(QLineEdit(), "connection.region")
        self.region.setPlaceholderText("us-east-1")
        f.addRow("Region", self.region)
        self.key_lbl = QLabel("")
        self.key_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.key_btn = newButton("Enter key…", slot=self._enter_key,
                                 tip="Typed once and saved as an environment variable on this computer")
        self.forget_btn = newButton("Forget", slot=self._forget_key, tip="Remove the saved key from this computer")
        f.addRow("API key", self._row(self.key_lbl, self.key_btn, self.forget_btn))
        self.key_env = self._bind(QLineEdit(), "connection.api_key_env")
        self.key_env.setPlaceholderText("e.g. LITELLM_API_KEY")
        self.key_env.textEdited.connect(lambda _t: (self._show_key(), self._ready_t.start()))
        self.key_env.setToolTip("The NAME of the environment variable that holds the key (the key itself is never "
                                "written in the project)")
        f.addRow("Key variable", self._row(self.key_env, _hint("the key is never stored in the project", False)))
        self.test_btn = newButton("Test connection", icon="run", slot=lambda: self.check_ready(True))
        self.test_lbl = _hint("")
        f.addRow("", self._row(self.test_btn, self.test_lbl))
        lay.addWidget(_section("Speed"))
        f2 = self._form(lay)
        w = self._bind(QSpinBox(), "connection.workers")
        w.setRange(1, 128)
        w.setToolTip("Requests sent at the same time — the main speed lever. Lower it if you see many 429 errors.")
        f2.addRow("Requests at a time", self._row(w, _hint("lower it if the service says “too many requests”",
                                                           False)))
        t = self._bind(QSpinBox(), "connection.timeout_s")
        t.setRange(5, 3600)
        t.setSuffix(" s")
        r = self._bind(QSpinBox(), "connection.retries")
        r.setRange(0, 20)
        f2.addRow("Wait for an answer", self._row(t, QLabel("Retries"), r))

    def _tab_model(self, lay):
        f = self._form(lay)
        self.model = self._bind(QComboBox(), "vlm.model")
        self.model.setEditable(True)
        self.model.setMinimumWidth(360)
        self.model.lineEdit().setPlaceholderText("e.g. gemini-2.5-flash, gpt-4o, qwen2.5-vl-72b-instruct …")
        self.model.currentTextChanged.connect(lambda _t: (self._show_profile(), self._ready_t.start()))
        f.addRow("Vision model", self._row(self.model, newButton("List models", icon="list", slot=self._list_models,
                                                                tip="Ask the service which models it offers")))
        self.vlm_trust = self._trust_box(lambda: self.model.currentText().strip(), "")
        self.model.currentTextChanged.connect(lambda _t: self._sync_trust(self.vlm_trust))
        f.addRow("", self.vlm_trust)
        self.profile_lbl = QLabel("")
        self.profile_lbl.setWordWrap(True)
        self.profile_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.profile_lbl.setMinimumWidth(520)
        f.addRow("", self.profile_lbl)
        lay.addWidget(_section("What should the AI do?"))
        self.mode_group = QButtonGroup(self)
        self.mode_btns = {}
        for k, title in C.MODES.items():
            rb = QRadioButton(title)
            rb.setToolTip({"vlm_detect": "The vision model finds the objects and draws the boxes. One request per "
                                         "image.",
                           "detect_vlm": "A detector on your GPU draws tight boxes; the vision model keeps the right "
                                         "ones and fixes their labels. Most reliable.",
                           "detect": "A text-prompted detector on your GPU only — free, no API.",
                           "classify": "No boxes: the AI answers your yes/no questions about each image."}[k])
            self.mode_group.addButton(rb)
            self.mode_btns[k] = rb
            lay.addWidget(rb)
            rb.toggled.connect(lambda on, k=k: on and not self._loading and self._set("mode", k))
        self.det_box = QGroupBox("Detector (runs on your computer)")
        dl = QFormLayout(self.det_box)
        self.det_type = self._bind(QComboBox(), "detector.type")
        for k, t in C.DETECTORS.items():
            self.det_type.addItem(t, k)
        dl.addRow("Detector", self.det_type)
        self.det_model = self._bind(QLineEdit(), "detector.model")
        self.det_model.setMinimumWidth(380)
        self.det_model.setPlaceholderText("a model folder (Hugging Face) or a .pt / .onnx file")
        dl.addRow("Model", self._row(self.det_model, newButton("Folder…", icon="folder-line",
                                                               slot=lambda: self._pick_into(self.det_model, True)),
                                     newButton("File…", slot=lambda: self._pick_into(self.det_model, False))))
        self.det_trust = self._trust_box(lambda: self.det_model.text().strip(),
                                         lambda: self.det_type.currentData() or "")
        self.det_model.textChanged.connect(lambda _t: self._sync_trust(self.det_trust))
        self.det_type.currentIndexChanged.connect(lambda _i: self._sync_trust(self.det_trust))
        dl.addRow("", self.det_trust)
        bt = self._bind(QDoubleSpinBox(), "detector.box_threshold")
        bt.setRange(0.01, 1)
        bt.setSingleStep(0.05)
        tt = self._bind(QDoubleSpinBox(), "detector.text_threshold")
        tt.setRange(0.01, 1)
        tt.setSingleStep(0.05)
        dl.addRow("Box threshold", self._row(bt, QLabel("Text threshold"), tt))
        io = self._bind(QDoubleSpinBox(), "detector.iou")
        io.setRange(0.05, 1)
        io.setSingleStep(0.05)
        io.setToolTip("Boxes of one class overlapping more than this are merged")
        dv = self._bind(QComboBox(), "detector.device")
        dv.setEditable(True)
        for k, t in (("auto", "Automatic"), ("cpu", "CPU"), ("0", "GPU 0"), ("1", "GPU 1")):
            dv.addItem(t, k)
        dl.addRow("Overlap merge", self._row(io, QLabel("Runs on"), dv))
        lay.addWidget(self.det_box)
        self.seg_check = QCheckBox("Polygons instead of boxes (SAM / SAM2 — on your computer)")
        self.seg_check.toggled.connect(lambda on: self._loading or self._set("segment.type", "sam2" if on else "none"))
        lay.addWidget(self.seg_check)
        self.seg_model = self._bind(QLineEdit(), "segment.model")
        self.seg_model.setPlaceholderText("SAM weights, e.g. sam2.1_b.pt (downloaded by ultralytics on first use)")
        self.seg_model.setMinimumWidth(420)
        lay.addWidget(self.seg_model)
        inner = QWidget()
        il = QFormLayout(inner)
        il.setContentsMargins(0, 0, 0, 0)
        ms = self._bind(QSpinBox(), "vlm.image_max_side")
        ms.setRange(256, 8192)
        ms.setSingleStep(128)
        ms.setSuffix(" px")
        ms.setToolTip("Images are made this small (longest side) before sending — smaller is cheaper and faster; "
                      "boxes are mapped back to full size")
        il.addRow("Image size sent", ms)
        self.boxfmt = self._bind(QComboBox(), "vlm.box_format")
        self.boxfmt.addItem("Automatic (from the model)", "auto")
        for k, t in C.BOX_FORMATS.items():
            self.boxfmt.addItem(t, k)
        self.boxfmt.currentIndexChanged.connect(lambda _i: self._show_profile())
        il.addRow("Boxes written as", self._row(self.boxfmt, newButton("Detect…", slot=self._calibrate,
                                                                      tip="Ask the model to box the known objects of "
                                                                          "an example image and keep the format that "
                                                                          "fits")))
        self.temp = QDoubleSpinBox()
        self.temp.setRange(0, 2)
        self.temp.setSingleStep(0.1)
        self.temp.valueChanged.connect(lambda v: self._loading or self._set_param("temperature", round(v, 2)))
        self.maxtok = QSpinBox()
        self.maxtok.setRange(64, 200000)
        self.maxtok.setSingleStep(256)
        self.maxtok.valueChanged.connect(lambda v: self._loading or self._set_param("max_tokens", int(v)))
        il.addRow("Temperature", self._row(self.temp, QLabel("Longest answer"), self.maxtok, QLabel("tokens")))
        jm = self._bind(QCheckBox("Ask for JSON answers (recommended; turned off by itself if the service refuses)"),
                        "vlm.json_mode")
        il.addRow("", jm)
        self.params = QLineEdit()
        self.params.setPlaceholderText('more model settings as JSON, e.g. {"reasoning_effort": "low"}')
        self.params.setMinimumWidth(420)
        self.params.editingFinished.connect(self._params_edited)
        il.addRow("Model settings", self.params)
        ins = self._bind(QPlainTextEdit(), "vlm.instructions")
        ins.setPlaceholderText("Project rules for the AI, in plain words — e.g. “Annotate only objects that are "
                               "fully visible. Ignore reflections.”")
        ins.setFixedHeight(90)
        il.addRow("Project rules", ins)
        from AIO_QC.widgets.drawer import OptionsDrawer
        self.adv_drawer = OptionsDrawer(inner, "Advanced settings", "Hide advanced settings",
                                        tip="Image size sent, box format, temperature, JSON answers, model settings, "
                                            "project rules")
        lay.addWidget(self.adv_drawer)

    def _tab_classes(self, lay):
        lay.addWidget(_hint("What to annotate. Describe each class the way you would brief a new annotator: what it "
                            "is, what it is not, how tight the box should be. The order is the YOLO class order "
                            "(0, 1, 2 …)."))
        self.cls_table = QTableWidget(0, 6)
        self.cls_table.setHorizontalHeaderLabels(["ID", "Class name", "Description (plain words)", "Min confidence",
                                                  "At most per image", "Must lie inside"])
        self.cls_table.verticalHeader().setVisible(False)
        hh = self.cls_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for c in (3, 4, 5):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.cls_table.setColumnWidth(1, 170)
        self.cls_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.cls_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.cls_table.setMinimumHeight(170)
        self.cls_table.itemChanged.connect(self._cls_item_changed)
        self.cls_table.currentCellChanged.connect(lambda *_a: self._cls_details())
        lay.addWidget(self.cls_table)
        row = QHBoxLayout()
        for text, icon, slot in (("Add class", "new", self._cls_add), ("Remove", "delete", self._cls_remove),
                                 ("Move up ↑", None, lambda: self._cls_move(-1)),
                                 ("Move down ↓", None, lambda: self._cls_move(1))):
            row.addWidget(newButton(text, icon=icon, slot=slot))
        row.addStretch(1)
        lay.addLayout(row)
        self.cls_detail = QGroupBox("Selected class")
        dl = QFormLayout(self.cls_detail)
        self.cd_text = QCheckBox("Also read the text in the box (written with the box)")
        self.cd_text.toggled.connect(lambda v: self._cls_set("text", bool(v)))
        dl.addRow("", self.cd_text)
        self.cd_pattern = QLineEdit()
        self.cd_pattern.setPlaceholderText(r"optional: the shape of the text, e.g.  ^[A-Z]{4}\d{7}$")
        self.cd_pattern.setMinimumWidth(380)
        self.cd_pattern.textEdited.connect(lambda t: self._cls_set("text_pattern", t.strip()))
        dl.addRow("Text pattern", self.cd_pattern)
        self.cd_check = QComboBox()
        for k, t in C.TEXT_CHECKS.items():
            self.cd_check.addItem(t, k)
        self.cd_check.currentIndexChanged.connect(lambda _i: self._cls_set("text_check", self.cd_check.currentData()))
        dl.addRow("Text check", self.cd_check)
        self.cd_parts = QLineEdit()
        self.cd_parts.setPlaceholderText("optional: its parts, e.g.  owner_code, serial_number, check_digit")
        self.cd_parts.setMinimumWidth(380)
        self.cd_parts.editingFinished.connect(self._cls_parts)
        dl.addRow("Parts", self._row(self.cd_parts, _hint("each part becomes a class too", False)))
        self.cd_size = QSpinBox()
        self.cd_size.setRange(0, 4096)
        self.cd_size.setSuffix(" px")
        self.cd_size.setSpecialValueText("any size")
        self.cd_size.valueChanged.connect(lambda v: self._cls_set("min_size_px", int(v)))
        dl.addRow("Smallest box", self.cd_size)
        self.cd_prompt = QLineEdit()
        self.cd_prompt.setPlaceholderText("what the detector looks for, in a few words, e.g. “small dark hole in a "
                                          "metal roof” (empty = the description)")
        self.cd_prompt.setMinimumWidth(380)
        self.cd_prompt.textEdited.connect(self._cls_prompt)
        dl.addRow("Detector words", self.cd_prompt)
        lay.addWidget(self.cls_detail)
        lay.addWidget(_section("Questions about each image (optional)"))
        lay.addWidget(_hint("Yes/no answers written as LabelMe flags (and usable in Checks), e.g. "
                            "“damage_visible — true when a tear or hole is visible”."))
        self.flag_table = QTableWidget(0, 2)
        self.flag_table.setHorizontalHeaderLabels(["Name", "True when …"])
        self.flag_table.verticalHeader().setVisible(False)
        self.flag_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.flag_table.setColumnWidth(0, 180)
        self.flag_table.setFixedHeight(110)
        self.flag_table.itemChanged.connect(self._flags_from_table)
        lay.addWidget(self.flag_table)
        fr = QHBoxLayout()
        fr.addWidget(newButton("Add question", icon="new", slot=lambda: self._table_add(self.flag_table)))
        fr.addWidget(newButton("Remove", icon="delete", slot=lambda: self._table_remove(self.flag_table,
                                                                                         self._flags_from_table)))
        fr.addStretch(1)
        lay.addLayout(fr)

    def _tab_examples(self, lay):
        lay.addWidget(_hint("A few example images show the AI what you mean (1–4 per camera is plenty). An example "
                            "with its boxes drawn shows exactly which objects you want, and how tight. Different "
                            "cameras or angles can have their own examples."))
        f = self._form(lay)
        ms = self._bind(QSpinBox(), "references.max_side")
        ms.setRange(256, 4096)
        ms.setSingleStep(128)
        ms.setSuffix(" px")
        db = self._bind(QCheckBox("Draw an example's own boxes onto it"), "references.draw_boxes")
        f.addRow("Example size", self._row(ms, db))
        split = QHBoxLayout()
        left = QVBoxLayout()
        self.views = QListWidget()
        self.views.setFixedWidth(200)
        self.views.currentRowChanged.connect(lambda _r: self._view_show())
        left.addWidget(QLabel("Cameras / views"))
        left.addWidget(self.views, 1)
        vb = QHBoxLayout()
        vb.addWidget(newButton("Add", icon="new", slot=self._view_add))
        vb.addWidget(newButton("Remove", icon="delete", slot=self._view_remove))
        left.addLayout(vb)
        split.addLayout(left)
        right = QVBoxLayout()
        vf = QFormLayout()
        self.v_match = QLineEdit()
        self.v_match.setPlaceholderText("which images: part of the file or folder name, e.g.  CAM1  or  *GRC2_*")
        self.v_match.setMinimumWidth(360)
        self.v_match.textEdited.connect(lambda t: self._view_set("match", t.strip()))
        vf.addRow("Images whose name has", self.v_match)
        self.v_note = QLineEdit()
        self.v_note.setPlaceholderText("optional: a word about this camera, e.g. “high on a pole, looking down”")
        self.v_note.textEdited.connect(lambda t: self._view_set("note", t.strip()))
        vf.addRow("About this camera", self.v_note)
        right.addLayout(vf)
        self.v_imgs = QListWidget()
        self.v_imgs.setViewMode(QListWidget.ViewMode.IconMode)
        self.v_imgs.setIconSize(QSize(150, 100))
        self.v_imgs.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.v_imgs.setMinimumHeight(170)
        self.v_imgs.setAcceptDrops(True)
        self.v_imgs.itemDoubleClicked.connect(lambda it: self._draw_boxes(it))
        right.addWidget(self.v_imgs, 1)
        ib = QHBoxLayout()
        ib.addWidget(newButton("Add images…", icon="new", slot=self._ref_add_files))
        ib.addWidget(newButton("Add folder…", icon="folder-line", slot=self._ref_add_folder))
        ib.addWidget(newButton("Draw boxes…", icon="sketch-rect", slot=lambda: self._draw_boxes(
            self.v_imgs.currentItem())))
        ib.addWidget(newButton("Remove", icon="delete", slot=self._ref_remove))
        ib.addStretch(1)
        right.addLayout(ib)
        right.addWidget(_hint("Double-click an example to draw or fix its boxes. Images that match no camera use "
                              "the Default examples."))
        split.addLayout(right, 1)
        lay.addLayout(split)

    def _tab_checks(self, lay):
        lay.addWidget(_hint("Checks decide whether an image is kept. Rejected images go to rejected/ with their "
                            "reasons (reports/rejected.csv) and come back if you run the step again."))
        rb = self._bind(QCheckBox("Reject an image when the AI finds no box"), "checks.require_boxes")
        lay.addWidget(rb)
        f = self._form(lay)
        self.text_fail = self._bind(QComboBox(), "checks.on_text_fail")
        self.text_fail.addItem("Keep the box and mark the image “needs review”", "review")
        self.text_fail.addItem("Drop the box", "drop")
        f.addRow("Text that fails its check", self.text_fail)
        kr = self._bind(QCheckBox("Keep the AI's boxes for rejected images (to tune the prompts)"),
                        "checks.keep_rejected")
        f.addRow("", kr)
        lay.addWidget(_section("Also ask about each image (optional)"))
        lay.addWidget(_hint("Extra answers to judge an image by — e.g. “photorealistic — 1 to 5, 5 = looks like a real "
                            "photo”, or “artifacts — a list of real problems, empty if none”."))
        self.field_table = QTableWidget(0, 2)
        self.field_table.setHorizontalHeaderLabels(["Name", "What it means"])
        self.field_table.verticalHeader().setVisible(False)
        self.field_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.field_table.setColumnWidth(0, 180)
        self.field_table.setFixedHeight(110)
        self.field_table.itemChanged.connect(self._fields_from_table)
        lay.addWidget(self.field_table)
        fb = QHBoxLayout()
        fb.addWidget(newButton("Add", icon="new", slot=lambda: self._table_add(self.field_table)))
        fb.addWidget(newButton("Remove", icon="delete", slot=lambda: self._table_remove(self.field_table,
                                                                                         self._fields_from_table)))
        fb.addStretch(1)
        lay.addLayout(fb)
        lay.addWidget(_section("Keep an image only when … (optional)"))
        rr = QHBoxLayout()
        self.rule_left = QComboBox()
        self.rule_left.setMinimumWidth(200)
        self.rule_op = QComboBox()
        for op, t in ((">=", "is at least"), ("<=", "is at most"), ("==", "is"), ("!=", "is not"), (">", "is more than"),
                      ("<", "is less than")):
            self.rule_op.addItem(t, op)
        self.rule_val = QLineEdit()
        self.rule_val.setPlaceholderText("value, e.g. 4 or true")
        self.rule_val.setFixedWidth(150)
        rr.addWidget(self.rule_left)
        rr.addWidget(self.rule_op)
        rr.addWidget(self.rule_val)
        rr.addWidget(newButton("Add rule", icon="new", slot=self._rule_add))
        rr.addStretch(1)
        lay.addLayout(rr)
        self.rules = QListWidget()
        self.rules.setFixedHeight(90)
        lay.addWidget(self.rules)
        rb2 = QHBoxLayout()
        rb2.addWidget(newButton("Remove rule", icon="delete", slot=self._rule_remove))
        rb2.addStretch(1)
        lay.addLayout(rb2)
        lay.addWidget(_section("Look only inside a bigger object (optional)"))
        f3 = self._form(lay)
        self.roi = self._bind(QComboBox(), "roi.class")
        self.roi.setToolTip("Uses the boxes of this class already in each image's file (e.g. from your trained "
                            "model) and asks the AI about each one close up — better for small objects")
        f3.addRow("Inside boxes of", self.roi)
        pad = self._bind(QDoubleSpinBox(), "roi.padding")
        pad.setRange(0, 2)
        pad.setSingleStep(0.05)
        mc = self._bind(QSpinBox(), "roi.max_crops_per_image")
        mc.setRange(1, 64)
        f3.addRow("Margin", self._row(pad, QLabel("At most"), mc, QLabel("per image")))
        lay.addWidget(_section("Limits"))
        f4 = self._form(lay)
        bud = self._bind(QDoubleSpinBox(), "limits.budget_usd")
        bud.setRange(0, 1e6)
        bud.setPrefix("$ ")
        bud.setSpecialValueText("no limit")
        bud.setToolTip("The run stops safely when the cost estimate reaches this; Run continues later")
        mi = self._bind(QSpinBox(), "limits.max_images")
        mi.setRange(0, 10 ** 8)
        mi.setSpecialValueText("all")
        f4.addRow("Budget", self._row(bud, QLabel("Images at most"), mi))
        pin = QDoubleSpinBox()
        pin.setRange(0, 1000)
        pin.setDecimals(3)
        pin.setPrefix("$ ")
        pout = QDoubleSpinBox()
        pout.setRange(0, 1000)
        pout.setDecimals(3)
        pout.setPrefix("$ ")
        self._bind(pin, "pricing.input")
        self._bind(pout, "pricing.output")
        f4.addRow("Price per million tokens", self._row(QLabel("in"), pin, QLabel("out"), pout,
                                                        _hint("from the provider's price page", False)))
        self.target_table = QTableWidget(0, 2)
        self.target_table.setHorizontalHeaderLabels(["Class", "Stop after this many images with it (0 = no target)"])
        self.target_table.verticalHeader().setVisible(False)
        self.target_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.target_table.setFixedHeight(110)
        self.target_table.itemChanged.connect(self._targets_from_table)
        lay.addWidget(self.target_table)
        f5 = self._form(lay)
        af = self._bind(QDoubleSpinBox(), "limits.max_attempts_factor")
        af.setRange(1, 100)
        af.setSingleStep(0.5)
        af.setToolTip("With targets: give up after this many × the target images were tried")
        sd = self._bind(QSpinBox(), "limits.seed")
        sd.setRange(0, 2 ** 31 - 1)
        f5.addRow("With targets, try at most", self._row(af, QLabel("× the target · order seed"), sd))

    def _tab_output(self, lay):
        from AIO_QC.tools.frame_extractor.widgets.autoannotate_window import _seg
        f = self._form(lay)
        self.fmt_seg, self.fmt_get, self.fmt_set = _seg([(k, t, "") for k, t in C.FORMATS.items()], "yolo",
                                                        lambda k: self._loading or self._set("output.format", k))
        f.addRow("Files", self.fmt_seg)
        self.order_lbl = _hint("")
        f.addRow("", self.order_lbl)
        dy = self._bind(QCheckBox("Write data.yaml (ready to train with YOLO)"), "output.data_yaml")
        f.addRow("", dy)
        self.lm_seg, self.lm_get, self.lm_set = _seg([(v, f"LabelMe {v}", "") for v in ("6.3.1", "5.11.4")], "6.3.1",
                                                     lambda k: self._loading or self._set("output.labelme_version", k))
        f.addRow("LabelMe version", self.lm_seg)
        idt = self._bind(QCheckBox("Save the image inside each LabelMe file (imageData)"), "output.image_data")
        f.addRow("", idt)
        wc = self._bind(QCheckBox("Write the confidence (and text read) in each box's description"),
                        "output.write_confidence")
        f.addRow("", wc)
        nr = self._bind(QDoubleSpinBox(), "output.needs_review_below")
        nr.setRange(0, 1)
        nr.setSingleStep(0.05)
        nr.setToolTip("An image with a box below this confidence is flagged “needs_review” — find them in Image QC")
        f.addRow("Mark for review below", self._row(nr, _hint("confidence", False)))
        mg = self._bind(QComboBox(), "output.merge_with_existing")
        for k, t in C.MERGE.items():
            mg.addItem(t, k)
        f.addRow("An image that has annotations", mg)
        au = self._bind(QCheckBox("Keep every request's answer (reports/ai_audit.jsonl — never the key)"),
                        "output.audit")
        f.addRow("", au)
        lay.addWidget(_hint("YOLO files have no place for read text, confidence or flags: those are kept in the run "
                            "and in a meta.jsonl in each packaged part."))

    def _tab_try(self, lay):
        lay.addWidget(_hint("Runs the AI on a few images with these settings and shows the boxes next to its answer. "
                            "Nothing is written. Change a description or a check and try again — answers already "
                            "given are reused, so trying again on the same images is instant and free."))
        row = QHBoxLayout()
        self.try_n = QSpinBox()
        self.try_n.setRange(1, 24)
        self.try_n.setValue(4)
        self.try_n.setSuffix(" images")
        self.try_btn = newButton("Try", icon="run", slot=self._try)
        self.est_btn = newButton("Estimate the whole run", icon="tool-report", slot=self._estimate,
                                 tip="Cost and time for every image, from the images tried")
        row.addWidget(self.try_n)
        row.addWidget(self.try_btn)
        row.addWidget(self.est_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.try_note = _hint("")
        lay.addWidget(self.try_note)
        self.try_area = QWidget()
        self.try_grid = QVBoxLayout(self.try_area)
        self.try_grid.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.try_area)

    def _tab_yaml(self, lay):
        lay.addWidget(_hint("The project file itself. Everything here is also on the other tabs; edit here only if "
                            "you prefer. Problems are shown with their line."))
        from AIO_QC.widgets.design import mono
        self.yaml = QPlainTextEdit()
        self.yaml.setFont(mono(11.5))
        self.yaml.setMinimumHeight(380)
        self.yaml.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.yaml.textChanged.connect(lambda: self._yaml_t.start() if self.yaml.hasFocus() else None)
        lay.addWidget(self.yaml)
        row = QHBoxLayout()
        self.yaml_lbl = _hint("")
        row.addWidget(self.yaml_lbl, 1)
        self.yaml_apply = newButton("Apply", icon="status-ok", slot=self._yaml_apply)
        row.addWidget(self.yaml_apply)
        lay.addLayout(row)

    # ------------------------------------------------------------------ project files
    def load_project(self, path: str):
        """Open a project (called by the window with the setup's ai_config)."""
        self._refresh_projects(path)
        if path and os.path.isfile(path):
            try:
                self.cfg = C.load(path)
                self.path = path
            except (C.ConfigError, RuntimeError, OSError) as exc:
                self.cfg, self.path = C.default(), path
                self.problems.setText(f"<span style='color:{_hex('error')}'>Could not read {os.path.basename(path)}: "
                                      f"{exc}</span>")
        else:
            self.cfg, self.path = C.default(), ""
        self._push()
        self._yaml_from_cfg()
        self._ready_t.start()

    def _refresh_projects(self, select=""):
        self.proj.blockSignals(True)
        self.proj.clear()
        self.proj.addItem("(choose or make a project)", "")
        d = projects_dir()
        files = sorted(f for f in os.listdir(d) if f.endswith((".yaml", ".yml")))
        paths = [os.path.join(d, f) for f in files]
        if select and select not in paths and os.path.isfile(select):
            paths.append(select)
        for p in paths:
            self.proj.addItem(os.path.splitext(os.path.basename(p))[0] + ("" if os.path.dirname(p) == d else
                                                                         f"   —   {os.path.dirname(p)}"), p)
        i = self.proj.findData(select) if select else 0
        self.proj.setCurrentIndex(max(0, i))
        self.proj.blockSignals(False)

    def _project_chosen(self, i):
        p = self.proj.itemData(i) or ""
        self.load_project(p)
        self.project_changed.emit(p)

    def _new_project(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("New AI annotate project")
        l = QVBoxLayout(dlg)
        l.addWidget(QLabel("<b>What should the AI do?</b>"))
        grp = QButtonGroup(dlg)
        btns = {}
        for k, (title, desc, _c) in C.TEMPLATES.items():
            rb = QRadioButton(title)
            grp.addButton(rb)
            btns[k] = rb
            l.addWidget(rb)
            h = _hint(desc)
            h.setContentsMargins(24, 0, 0, 6)
            l.addWidget(h)
        btns["find"].setChecked(True)
        f = QFormLayout()
        ex_dir = os.path.join(os.path.dirname(C.__file__), "examples")
        example = QComboBox()
        example.addItem("— the choice above —", "")
        for fn in sorted(os.listdir(ex_dir)) if os.path.isdir(ex_dir) else []:
            if fn.endswith(".yaml"):
                example.addItem(f"Example: {fn[:-5].replace('_', ' ')}", os.path.join(ex_dir, fn))
        example.setToolTip("A finished example project to adapt (its classes, checks and settings filled in)")
        f.addRow("Or start from", example)
        name = QLineEdit()
        name.setPlaceholderText("e.g. trailer-damage, container-ids")
        f.addRow("Project name", name)
        prov = QComboBox()
        for k, (title, *_r) in PF.PROVIDERS.items():
            prov.addItem(title, k)
        prov.setCurrentIndex(prov.findData(self.cfg["connection"].get("provider") or "litellm"))
        f.addRow("AI service", prov)
        l.addLayout(f)
        l.addWidget(_hint("Next: the tabs guide you — Connect, Model, Classes, Examples, Checks, Output, then Try."))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        l.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        nm = re.sub(r'[<>:"/\\|?*\s]+', "-", name.text().strip()) or "my-project"
        path = os.path.join(projects_dir(), nm + ".yaml")
        if os.path.exists(path) and QMessageBox.question(self, "New project", f"“{nm}” exists. Replace it?") != \
                QMessageBox.StandardButton.Yes:
            return
        key = next(k for k, b in btns.items() if b.isChecked())
        if example.currentData():
            cfg = C.load(example.currentData())
            cfg.pop("_path", None)
            cfg["project"] = nm
        else:
            cfg = C.from_template(key, nm)
        pk = prov.currentData()
        _t, _kind, url, env, _n = PF.PROVIDERS[pk]
        cfg["connection"].update(provider=pk, base_url=url, api_key_env=env)
        try:
            C.save(cfg, path)
        except (OSError, RuntimeError) as exc:
            QMessageBox.warning(self, "New project", str(exc))
            return
        self.load_project(path)
        self.project_changed.emit(path)
        self.tabs.setCurrentIndex(0)

    def _import(self):
        p, _ = QFileDialog.getOpenFileName(self, "Use a config.yaml", "", "YAML (*.yaml *.yml)")
        if not p:
            return
        try:
            cfg = C.load(p)
        except (C.ConfigError, RuntimeError) as exc:
            QMessageBox.warning(self, "Import", f"Could not read it{f' (line {exc.line})' if getattr(exc, 'line', None) else ''}:\n{exc}")
            return
        dst = os.path.join(projects_dir(), os.path.basename(p))
        if os.path.abspath(p) != os.path.abspath(dst):
            if os.path.exists(dst) and QMessageBox.question(self, "Import", f"Replace the project "
                                                                           f"“{os.path.basename(p)}”?") != \
                    QMessageBox.StandardButton.Yes:
                return
            base = os.path.dirname(p)              # examples stay where they are: make their paths absolute
            for v in cfg["references"]["views"].values():
                v["images"] = [x if os.path.isabs(x) else os.path.normpath(os.path.join(base, x)) for x in v["images"]]
            C.save(cfg, dst)
        self.load_project(dst)
        self.project_changed.emit(dst)

    def _export(self):
        if not self.path:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Export the project", os.path.basename(self.path), "YAML (*.yaml)")
        if p:
            self._save_now()
            C.save(self.cfg, p)

    def _save_now(self):
        if not self.path:
            return
        try:
            C.save(self.cfg, self.path)
        except (OSError, RuntimeError) as exc:
            self.problems.setText(f"<span style='color:{_hex('error')}'>Could not save: {exc}</span>")
            return
        if not self.yaml.hasFocus():
            self._yaml_from_cfg()

    # ------------------------------------------------------------------ readiness
    def check_ready(self, connection=True):
        self.recheck_btn.setEnabled(False)
        self.ready_title.setText("<b>Is everything ready?</b> <span style='color:%s'>checking…</span>" % _hex("muted"))
        cfg = copy.deepcopy(self.cfg)
        py = env_python()
        self.env_lbl.setText(f"Environment: {self._env_name(py)}")
        info = self._env_info_cache.get(py)

        def work():
            inf = info
            if C.needs_env(cfg) and py and inf is None:
                try:
                    from AIO_QC.tools.frame_extractor.core.ai_annotate.runner import probe_env
                    inf = probe_env(py)
                except Exception as exc:
                    _ignored(exc)
                    inf = {"error": str(exc)}
            return RD.check(cfg, py, connection, inf), inf

        def done(res):
            items, inf = res
            if inf is not None and not inf.get("error"):
                self._env_info_cache[py] = inf
            self._show_ready(items)
        self._call(work, done, lambda m: self._show_ready([RD.Item("app", "Check", "fail", m)]))

    def _env_name(self, py):
        if not py:
            return "not chosen"
        parts = py.replace("\\", "/").split("/")
        for i in range(len(parts) - 1, 0, -1):
            if parts[i] in ("bin", "Scripts"):
                return parts[i - 1]
        return os.path.basename(os.path.dirname(py)) or py

    def _show_ready(self, items):
        self._ready_items = items
        self.recheck_btn.setEnabled(True)
        while self.ready_rows.count():
            w = self.ready_rows.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for r, it in enumerate(items):
            mark, role = STATUS_MARK.get(it.status, ("?", "muted"))
            m = QLabel(mark)
            m.setFixedWidth(18)
            m.setAlignment(Qt.AlignmentFlag.AlignCenter)
            style(m, f"QLabel{{color:@{role if role in ('ok', 'warn', 'error') else 'faint'}; font-weight:700;}}")
            t = QLabel(f"<b>{it.title}</b>")
            d = QLabel(it.detail)
            d.setWordWrap(True)
            style(d, "QLabel{color:@muted;}")
            self.ready_rows.addWidget(m, r, 0)
            self.ready_rows.addWidget(t, r, 1)
            self.ready_rows.addWidget(d, r, 2)
            if it.fix:
                text = {"install_app": "Install", "install_env": "Install", "choose_env": "Choose…",
                        "enter_key": "Enter key…", "test_connection": "Test again"}.get(it.fix, "Fix")
                b = newButton(text, slot=lambda _c=False, it=it: self._fix(it))
                b.setAutoDefault(False)
                self.ready_rows.addWidget(b, r, 3)
        self.ready_rows.setColumnStretch(2, 1)
        app_ok = all(i.status != "fail" for i in items if i.key == "app")
        self.body.setEnabled(app_ok)
        n_fail = sum(i.status == "fail" for i in items)
        self.ready_title.setText("<b>Is everything ready?</b> " + (
            f"<span style='color:{_hex('ok')}'>Yes</span>" if not n_fail else
            f"<span style='color:{_hex('error')}'>{n_fail} thing{'s' if n_fail != 1 else ''} to fix</span>"))
        svc = next((i for i in items if i.key == "service"), None)
        if svc is not None:
            self.test_lbl.setText(svc.detail)
        self.changed.emit()

    def ready(self) -> bool:
        return bool(self._ready_items) and RD.ready(self._ready_items)

    def _fix(self, it):
        if it.fix == "install_app":
            dlg = InstallDialog(sys.executable, it.fix_args, "Install into AIO QC's Python", self)
            dlg.exec()
        elif it.fix == "install_env":
            py = env_python()
            dlg = InstallDialog(py, it.fix_args, "Install into your Python environment", self)
            dlg.exec()
            self._env_info_cache.pop(py, None)
        elif it.fix == "choose_env":
            self._choose_env()
            return
        elif it.fix == "enter_key":
            self._enter_key()
            return
        self.check_ready(True)

    def _choose_env(self):
        d = EnvDialog(self)
        if d.exec() == QDialog.DialogCode.Accepted and d.chosen():
            from AIO_QC.utils.qt_utils import app_settings
            app_settings().setValue("inference/python", d.chosen())
            self._env_info_cache.pop(d.chosen(), None)
            self.check_ready(True)

    # ------------------------------------------------------------------ connect
    def _provider_changed(self, _i):
        k = self.provider.currentData()
        title, kind, url, env, note = PF.PROVIDERS.get(k, PF.PROVIDERS["custom"])
        self.provider_note.setText(note)
        if self._loading:
            return
        con = self.cfg["connection"]
        old = PF.PROVIDERS.get(con.get("provider") or "custom", PF.PROVIDERS["custom"])
        con["provider"] = k
        if not con.get("base_url") or con.get("base_url") == old[2]:
            con["base_url"] = url
        if not con.get("api_key_env") or con.get("api_key_env") == old[3]:
            con["api_key_env"] = env
        self._push()
        self._edited()
        self._ready_t.start()

    def _sync_visibility(self):
        con = self.cfg["connection"]
        kind = PF.PROVIDERS.get(con.get("provider") or "custom", PF.PROVIDERS["custom"])[1]
        i = self.provider.findData(con.get("provider") or "custom")
        if i >= 0 and self.provider.currentIndex() != i:
            was, self._loading = self._loading, True
            self.provider.setCurrentIndex(i)
            self._loading = was
        self.provider_note.setText(PF.PROVIDERS.get(con.get("provider") or "custom", PF.PROVIDERS["custom"])[4])
        for w, show in ((self.base_url, kind not in ("bedrock", "local")), (self.api_version, kind == "azure"),
                        (self.region, kind == "bedrock")):
            self.connect_form.setRowVisible(w, show)
        mode = self.cfg["mode"]
        self.det_box.setVisible(mode in ("detect", "detect_vlm"))
        seg = self.cfg["segment"].get("type", "none") != "none"
        self.seg_check.blockSignals(True)
        self.seg_check.setChecked(seg)
        self.seg_check.blockSignals(False)
        self.seg_model.setVisible(seg)
        names = C.class_names(self.cfg)
        cur = self.cfg["roi"].get("class", "")
        self.roi.blockSignals(True)
        self.roi.clear()
        self.roi.addItem("the whole image", "")
        for n in names:
            self.roi.addItem(n, n)
        self.roi.setCurrentIndex(max(0, self.roi.findData(cur)))
        self.roi.blockSignals(False)
        self.order_lbl.setText("YOLO class IDs follow the order on the Classes tab: " +
                               (", ".join(f"{i} {n}" for i, n in enumerate(names)) or "(no classes yet)"))
        left = [("boxes", "number of boxes")] + [(f"count({n})", f"number of “{n}” boxes") for n in names] + \
               [(f, f"answer “{f}”") for f in (self.cfg["checks"].get("fields") or {})] + \
               [(f"len({f})", f"length of “{f}” (a list)") for f in (self.cfg["checks"].get("fields") or {})]
        self.rule_left.clear()
        for k, t in left:
            self.rule_left.addItem(t, k)
        self.temp.blockSignals(True)
        self.maxtok.blockSignals(True)
        p = self.cfg["vlm"].get("params") or {}
        self.temp.setValue(float(p.get("temperature", 0) or 0))
        self.maxtok.setValue(int(p.get("max_tokens", 3000) or 3000))
        self.temp.blockSignals(False)
        self.maxtok.blockSignals(False)
        extra = {k: v for k, v in p.items() if k not in ("temperature", "max_tokens")}
        if not self.params.hasFocus():
            self.params.setText(json.dumps(extra) if extra else "")

    def _show_key(self):
        name = self.cfg["connection"].get("api_key_env") or ""
        kind = PF.PROVIDERS.get(self.cfg["connection"].get("provider") or "custom", PF.PROVIDERS["custom"])[1]
        if kind in ("bedrock", "local"):
            self.key_lbl.setText("not needed")
            self.key_btn.setVisible(False)
            self.forget_btn.setVisible(False)
            return
        if not name:
            self.key_lbl.setText("this service needs no key")
            self.key_btn.setVisible(False)
            self.forget_btn.setVisible(False)
            return
        v = K.get(name)
        self.key_btn.setVisible(True)
        self.key_btn.setText("Replace key…" if v else "Enter key…")
        self.forget_btn.setVisible(bool(v))
        self.key_lbl.setText(f"<span style='color:{_hex('ok')}'>found</span> in {name} ({K.mask(v)})" if v else
                             f"<span style='color:{_hex('error')}'>not set</span> — {name} is empty")

    def _enter_key(self):
        name = self.cfg["connection"].get("api_key_env") or ""
        if not K.valid_name(name):
            QMessageBox.information(self, "API key", "Type the name of the variable first (Key variable), e.g. "
                                                     "LITELLM_API_KEY.")
            return
        v, ok = QInputDialog.getText(self, "API key", f"Paste the key for {name}.\n\nIt is saved as the environment "
                                                      f"variable {name} on this computer (your user only) and used "
                                                      f"by every later run and the command line. It is never written "
                                                      f"in the project file.", QLineEdit.EchoMode.Password)
        if not ok or not v.strip():
            return
        try:
            where = K.save(name, v.strip())
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "API key", f"Could not save it: {exc}")
            return
        self._show_key()
        self.test_lbl.setText(f"Saved in {where}.")
        self.check_ready(True)

    def _forget_key(self):
        name = self.cfg["connection"].get("api_key_env") or ""
        if name and QMessageBox.question(self, "API key", f"Remove the key saved in {name} from this computer?") == \
                QMessageBox.StandardButton.Yes:
            K.forget(name)
            self._show_key()
            self.check_ready(True)

    # ------------------------------------------------------------------ model
    def _show_profile(self):
        m = self.model.currentText().strip()
        if not m:
            self.profile_lbl.setText(f"<span style='color:{_hex('muted')}'>Pick a model; its box format and "
                                     f"settings are filled in for you.</span>")
            return
        p = PF.profile(m)
        fmt = PF.box_format(self.cfg)
        self.profile_lbl.setText(f"<b>{p['family']}</b> · boxes as {C.BOX_FORMATS.get(fmt, fmt)}"
                                 f"{' (automatic)' if self.cfg['vlm'].get('box_format', 'auto') == 'auto' else ''}"
                                 f"<br><span style='color:{_hex('muted')}'>{p['note']}</span>")

    def _list_models(self):
        cfg = copy.deepcopy(self.cfg)
        self.profile_lbl.setText("Asking the service…")

        def done(models):
            cur = self.model.currentText()
            self.model.blockSignals(True)
            self.model.clear()
            vis = [m for m in models if not re.search(r"embed|whisper|tts|dall|moderation|audio", m, re.I)]
            for m in vis:
                self.model.addItem(m, m)
            self.model.setEditText(cur)
            self.model.blockSignals(False)
            self._show_profile()
            self.test_lbl.setText(f"{len(vis)} models listed — open the list to choose." if vis else
                                  "The service does not list its models — type the name.")
            self.model.showPopup() if vis else None

        from AIO_QC.tools.frame_extractor.core.ai_annotate import providers as PV
        self._call(lambda: PV.list_models(cfg), done, lambda m: self.profile_lbl.setText(
            f"<span style='color:{_hex('error')}'>{m}</span>"))

    def _set_param(self, k, v):
        p = dict(self.cfg["vlm"].get("params") or {})
        p[k] = v
        self._set("vlm.params", p)

    def _params_edited(self):
        t = self.params.text().strip()
        try:
            extra = json.loads(t) if t else {}
            if not isinstance(extra, dict):
                raise ValueError
        except ValueError:
            self.params.setStyleSheet(f"border:1px solid {_hex('error')};")
            return
        self.params.setStyleSheet("")
        p = {k: v for k, v in (self.cfg["vlm"].get("params") or {}).items() if k in ("temperature", "max_tokens")}
        p.update(extra)
        self._set("vlm.params", p)

    def _pick_into(self, edit, folder):
        if folder:
            p = QFileDialog.getExistingDirectory(self, "Model folder", edit.text())
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Model file", edit.text(), "Models (*.pt *.onnx *.pth);;All (*)")
        if p:
            edit.setText(p)
            edit.textEdited.emit(p)

    def _calibrate(self):
        p, _ = QFileDialog.getOpenFileName(self, "An example image that already has boxes (LabelMe or YOLO next to it)",
                                           "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not p:
            return
        cfg = copy.deepcopy(self.cfg)
        self.profile_lbl.setText("Asking the model to box the known objects in 4 ways…")
        from AIO_QC.tools.frame_extractor.core.ai_annotate.runner import calibrate

        def done(res):
            sc = ", ".join(f"{C.BOX_FORMATS[k].split(',')[0]} {v:.2f}" for k, v in res["scores"].items())
            if res["best"]:
                self.cfg["vlm"]["box_format"] = res["best"]
                self._push()
                self._edited()
                self.profile_lbl.setText(self.profile_lbl.text() + f"<br>Detected: {C.BOX_FORMATS[res['best']]} "
                                                                   f"(overlap {sc})")
            else:
                self.profile_lbl.setText(f"No format matched the known boxes (overlap {sc}) — the model may not "
                                         f"draw boxes well; try “A detector finds boxes, the vision model checks "
                                         f"them”.")
        self._call(lambda: calibrate(cfg, p, env_python()), done,
                   lambda m: self.profile_lbl.setText(f"<span style='color:{_hex('error')}'>{m}</span>"))

    # ------------------------------------------------------------------ classes
    def _fill_classes(self):
        t = self.cls_table
        t.blockSignals(True)
        t.setRowCount(0)
        names = C.class_names(self.cfg)
        for i, c in enumerate(self.cfg["classes"]):
            t.insertRow(i)
            idi = QTableWidgetItem(str(i))
            idi.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            t.setItem(i, 0, idi)
            t.setItem(i, 1, QTableWidgetItem(c.get("name", "")))
            t.setItem(i, 2, QTableWidgetItem((c.get("description") or "").strip()))
            t.setItem(i, 3, QTableWidgetItem(f"{float(c.get('min_confidence') or 0):.2f}" if c.get("min_confidence")
                                             else ""))
            t.setItem(i, 4, QTableWidgetItem(str(c.get("max_per_image") or "")))
            box = QComboBox()
            box.addItem("anywhere", "")
            for n in names:
                if n != c.get("name"):
                    box.addItem(n, n)
            box.setCurrentIndex(max(0, box.findData(c.get("inside") or "")))
            box.currentIndexChanged.connect(lambda _i, row=i, b=box: self._cls_set("inside", b.currentData(), row))
            t.setCellWidget(i, 5, box)
        t.blockSignals(False)
        self._cls_details()

    def _cls_item_changed(self, it):
        if self._loading:
            return
        r, c = it.row(), it.column()
        if r >= len(self.cfg["classes"]):
            return
        cls = self.cfg["classes"][r]
        txt = it.text().strip()
        if c == 1:
            old = cls.get("name", "")
            new = re.sub(r"[\s/\\]+", "_", txt)
            cls["name"] = new
            for o in self.cfg["classes"]:                 # keep references to the class consistent
                if o.get("inside") == old:
                    o["inside"] = new
                o["parts"] = [new if p == old else p for p in o.get("parts") or []]
            if self.cfg["roi"].get("class") == old:
                self.cfg["roi"]["class"] = new
            t = self.cfg["limits"].get("targets") or {}
            if old in t:
                t[new] = t.pop(old)
            pr = self.cfg["detector"].get("prompts") or {}
            if old in pr:
                pr[new] = pr.pop(old)
            if new != txt:
                QTimer.singleShot(0, self._fill_classes)
            else:
                QTimer.singleShot(0, self._refresh_class_refs)
        elif c == 2:
            cls["description"] = txt
        elif c == 3:
            try:
                v = float(txt.replace(",", ".")) if txt else 0.0
                cls["min_confidence"] = max(0.0, min(1.0, v))
            except ValueError:
                pass
        elif c == 4:
            try:
                cls["max_per_image"] = max(0, int(txt)) if txt else 0
            except ValueError:
                pass
        self._edited()

    def _refresh_class_refs(self):
        self._fill_classes()
        self._fill_targets()
        self._sync_visibility()

    def _sel_class(self):
        r = self.cls_table.currentRow()
        return r if 0 <= r < len(self.cfg["classes"]) else -1

    def _cls_details(self):
        r = self._sel_class()
        self.cls_detail.setEnabled(r >= 0)
        if r < 0:
            self.cls_detail.setTitle("Selected class")
            return
        c = self.cfg["classes"][r]
        self.cls_detail.setTitle(f"Class “{c.get('name') or '?'}”")
        was, self._loading = self._loading, True
        self.cd_text.setChecked(bool(c.get("text")))
        self.cd_pattern.setText(c.get("text_pattern") or "")
        self.cd_check.setCurrentIndex(max(0, self.cd_check.findData(c.get("text_check") or "none")))
        self.cd_parts.setText(", ".join(c.get("parts") or []))
        self.cd_size.setValue(int(c.get("min_size_px") or 0))
        self.cd_prompt.setText((self.cfg["detector"].get("prompts") or {}).get(c.get("name"), ""))
        det = self.cfg["mode"] in ("detect", "detect_vlm")
        self.cd_prompt.setVisible(det)
        lab = self.cls_detail.layout().labelForField(self.cd_prompt)
        if lab is not None:
            lab.setVisible(det)
        for w in (self.cd_pattern, self.cd_check):
            w.setEnabled(bool(c.get("text")))
        self._loading = was

    def _cls_set(self, key, value, row=None):
        if self._loading:
            return
        r = self._sel_class() if row is None else row
        if r < 0:
            return
        self.cfg["classes"][r][key] = value
        if key == "text":
            for w in (self.cd_pattern, self.cd_check):
                w.setEnabled(bool(value))
        self._edited()

    def _cls_prompt(self, t):
        r = self._sel_class()
        if r < 0 or self._loading:
            return
        pr = dict(self.cfg["detector"].get("prompts") or {})
        name = self.cfg["classes"][r]["name"]
        if t.strip():
            pr[name] = t.strip()
        else:
            pr.pop(name, None)
        self._set("detector.prompts", pr)

    def _cls_parts(self):
        r = self._sel_class()
        if r < 0:
            return
        parts = [p for p in (re.sub(r"[\s/\\]+", "_", x.strip()) for x in self.cd_parts.text().split(",")) if p]
        self.cfg["classes"][r]["parts"] = parts
        names = C.class_names(self.cfg)
        for p in parts:                                   # each part is a class too
            if p not in names:
                self.cfg["classes"].append(C.new_class(p, f"the {p.replace('_', ' ')} of the "
                                                          f"{self.cfg['classes'][r]['name']}"))
                self.cfg["classes"][-1]["inside"] = self.cfg["classes"][r]["name"]
        self._fill_classes()
        self.cls_table.selectRow(r)
        self._fill_targets()
        self._sync_visibility()
        self._edited()

    def _cls_add(self):
        names = C.class_names(self.cfg)
        n = 1
        while f"class_{n}" in names:
            n += 1
        self.cfg["classes"].append(C.new_class(f"class_{n}", ""))
        self._fill_classes()
        self._fill_targets()
        self._sync_visibility()
        r = len(self.cfg["classes"]) - 1
        self.cls_table.setCurrentCell(r, 1)
        self.cls_table.editItem(self.cls_table.item(r, 1))
        self._edited()

    def _cls_remove(self):
        r = self._sel_class()
        if r < 0:
            return
        gone = self.cfg["classes"].pop(r).get("name")
        for o in self.cfg["classes"]:
            if o.get("inside") == gone:
                o["inside"] = ""
            o["parts"] = [p for p in o.get("parts") or [] if p != gone]
        (self.cfg["limits"].get("targets") or {}).pop(gone, None)
        if self.cfg["roi"].get("class") == gone:
            self.cfg["roi"]["class"] = ""
        self._fill_classes()
        self._fill_targets()
        self._sync_visibility()
        self._edited()

    def _cls_move(self, d):
        r = self._sel_class()
        j = r + d
        if r < 0 or not 0 <= j < len(self.cfg["classes"]):
            return
        cl = self.cfg["classes"]
        cl[r], cl[j] = cl[j], cl[r]
        self._fill_classes()
        self.cls_table.selectRow(j)
        self._sync_visibility()
        self._edited()

    # ------------------------------------------------------------------ flags / fields / rules / targets
    def _table_add(self, t):
        r = t.rowCount()
        t.insertRow(r)
        t.setItem(r, 0, QTableWidgetItem(""))
        t.setItem(r, 1, QTableWidgetItem(""))
        t.setCurrentCell(r, 0)
        t.editItem(t.item(r, 0))

    def _table_remove(self, t, sync):
        r = t.currentRow()
        if r >= 0:
            t.removeRow(r)
            sync()

    def _pairs(self, t):
        out = []
        for r in range(t.rowCount()):
            a = (t.item(r, 0).text() if t.item(r, 0) else "").strip()
            b = (t.item(r, 1).text() if t.item(r, 1) else "").strip()
            if a:
                out.append((re.sub(r"[\s/\\]+", "_", a), b))
        return out

    def _fill_pairs(self, t, pairs):
        t.blockSignals(True)
        t.setRowCount(0)
        for a, b in pairs:
            r = t.rowCount()
            t.insertRow(r)
            t.setItem(r, 0, QTableWidgetItem(a))
            t.setItem(r, 1, QTableWidgetItem(b))
        t.blockSignals(False)

    def _fill_flags(self):
        self._fill_pairs(self.flag_table, [(f["name"], f.get("description", "")) for f in self.cfg["flags"]])

    def _flags_from_table(self, *_a):
        if self._loading:
            return
        self.cfg["flags"] = [{"name": a, "description": b} for a, b in self._pairs(self.flag_table)]
        self._sync_visibility()
        self._edited()

    def _fill_fields(self):
        self._fill_pairs(self.field_table, list((self.cfg["checks"].get("fields") or {}).items()))

    def _fields_from_table(self, *_a):
        if self._loading:
            return
        self.cfg["checks"]["fields"] = dict(self._pairs(self.field_table))
        self._sync_visibility()
        self._edited()

    def _fill_rules(self):
        self.rules.clear()
        for r in self.cfg["checks"].get("accept") or []:
            self.rules.addItem(str(r))

    def _rule_add(self):
        left, op, val = self.rule_left.currentData(), self.rule_op.currentData(), self.rule_val.text().strip()
        if not left or not val:
            return
        self.cfg["checks"].setdefault("accept", []).append(f"{left} {op} {val}")
        self.rule_val.clear()
        self._fill_rules()
        self._edited()

    def _rule_remove(self):
        r = self.rules.currentRow()
        if r >= 0:
            del self.cfg["checks"]["accept"][r]
            self._fill_rules()
            self._edited()

    def _fill_targets(self):
        t = self.target_table
        t.blockSignals(True)
        t.setRowCount(0)
        tg = self.cfg["limits"].get("targets") or {}
        for n in C.class_names(self.cfg):
            r = t.rowCount()
            t.insertRow(r)
            a = QTableWidgetItem(n)
            a.setFlags(Qt.ItemFlag.ItemIsEnabled)
            t.setItem(r, 0, a)
            t.setItem(r, 1, QTableWidgetItem(str(tg.get(n, "") or "")))
        t.blockSignals(False)

    def _targets_from_table(self, *_a):
        if self._loading:
            return
        tg = {}
        for r in range(self.target_table.rowCount()):
            n = self.target_table.item(r, 0).text()
            v = (self.target_table.item(r, 1).text() if self.target_table.item(r, 1) else "").strip()
            try:
                if v and int(v) > 0:
                    tg[n] = int(v)
            except ValueError:
                pass
        self.cfg["limits"]["targets"] = tg
        self._edited()

    # ------------------------------------------------------------------ examples
    def _fill_views(self):
        cur = self.views.currentItem().text() if self.views.currentItem() else "default"
        self.views.blockSignals(True)
        self.views.clear()
        for k in self.cfg["references"]["views"]:
            self.views.addItem(k)
        items = self.views.findItems(cur, Qt.MatchFlag.MatchExactly)
        self.views.setCurrentItem(items[0] if items else self.views.item(0))
        self.views.blockSignals(False)
        self._view_show()

    def _view(self):
        it = self.views.currentItem()
        return it.text() if it else "default"

    def _view_show(self):
        v = self.cfg["references"]["views"].get(self._view()) or {}
        is_def = self._view() == "default"
        self.v_match.setEnabled(not is_def)
        self.v_match.setText("" if is_def else v.get("match", ""))
        self.v_match.setPlaceholderText("every image no camera below matches" if is_def else
                                        "which images: part of the file or folder name, e.g.  CAM1  or  *GRC2_*")
        self.v_note.setText(v.get("note", ""))
        self.v_imgs.clear()
        from AIO_QC.tools.frame_extractor.core.ai_annotate.prompt import expand
        from AIO_QC.tools.frame_extractor.core.ai_annotate.writers import read_labels
        base = os.path.dirname(self.path) if self.path else ""
        for entry in v.get("images") or []:
            files = expand([entry], base)
            for p in files[:40]:
                it = QListWidgetItem(os.path.basename(p) + (" (folder)" if len(files) > 1 else ""))
                it.setData(Qt.ItemDataRole.UserRole, (entry, p))
                n = len(read_labels(p))
                it.setToolTip(f"{p}\n{n} box{'es' if n != 1 else ''} drawn" if n else f"{p}\nno boxes drawn")
                pm = QPixmap(p)
                if not pm.isNull():
                    it.setIcon(QIcon(pm.scaled(150, 100, Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)))
                if n:
                    it.setText(it.text() + f" · {n} box{'es' if n != 1 else ''}")
                self.v_imgs.addItem(it)
            if not files:
                it = QListWidgetItem(f"missing: {entry}")
                it.setData(Qt.ItemDataRole.UserRole, (entry, ""))
                it.setForeground(QColor(_hex("error")))
                self.v_imgs.addItem(it)

    def _view_set(self, key, value):
        self.cfg["references"]["views"].setdefault(self._view(), {"match": "", "images": [], "note": ""})[key] = value
        self._edited()

    def _view_add(self):
        name, ok = QInputDialog.getText(self, "Camera / view", "A short name for this camera or angle:")
        name = re.sub(r"[^\w\-]+", "_", (name or "").strip())
        if not ok or not name or name in self.cfg["references"]["views"]:
            return
        self.cfg["references"]["views"][name] = {"match": "", "images": [], "note": ""}
        self._fill_views()
        self.views.setCurrentItem(self.views.findItems(name, Qt.MatchFlag.MatchExactly)[0])
        self.v_match.setFocus()
        self._edited()

    def _view_remove(self):
        v = self._view()
        if v == "default":
            return
        self.cfg["references"]["views"].pop(v, None)
        self._fill_views()
        self._edited()

    def _ref_add(self, paths):
        imgs = self.cfg["references"]["views"].setdefault(self._view(), {"match": "", "images": [],
                                                                         "note": ""})["images"]
        for p in paths:
            if p not in imgs:
                imgs.append(p)
        self._view_show()
        self._edited()

    def _ref_add_files(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Example images", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if ps:
            self._ref_add(ps)

    def _ref_add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder of example images")
        if d:
            self._ref_add([d])

    def _ref_remove(self):
        it = self.v_imgs.currentItem()
        if not it:
            return
        entry = it.data(Qt.ItemDataRole.UserRole)[0]
        imgs = self.cfg["references"]["views"][self._view()]["images"]
        if entry in imgs:
            imgs.remove(entry)
        self._view_show()
        self._edited()

    def _draw_boxes(self, it):
        if not it:
            return
        p = it.data(Qt.ItemDataRole.UserRole)[1]
        if not p:
            return
        if BoxDialog(p, C.class_names(self.cfg), self).exec():
            self._view_show()

    # ------------------------------------------------------------------ try / estimate
    def _sample_paths(self, n):
        if self.win is not None and hasattr(self.win, "sample_image_paths"):
            return self.win.sample_image_paths(n)
        return []

    def _try(self):
        if not self.path:
            self.try_note.setText("Make or choose a project first.")
            return
        probs = C.check(self.cfg, os.path.dirname(self.path))
        if probs:
            self.try_note.setText(f"<span style='color:{_hex('error')}'>Fix first: {probs[0][1]}</span>")
            return
        n = self.try_n.value()
        self.try_btn.setEnabled(False)
        self.try_note.setText("Reading a few images and asking the AI…")
        cfg = copy.deepcopy(self.cfg)
        cfg["_path"] = self.path
        cache = os.path.join(projects_dir(), ".try_cache.db")
        py = env_python()

        def work():
            from AIO_QC.tools.frame_extractor.core.ai_annotate.runner import Annotator
            paths = self._sample_paths(n)
            if not paths:
                raise RuntimeError("No images to try: choose the source folder (or run Frames) first.")
            a = Annotator(cfg, py, cache, "", base_dir=os.path.dirname(self.path))
            try:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max(1, min(len(paths), int(cfg["connection"].get("workers") or 4)))) as ex:
                    return list(ex.map(a.image, paths))
            finally:
                a.close()

        def done(results):
            self.try_btn.setEnabled(True)
            self._last_try = results
            self._show_try(results)

        def fail(m):
            self.try_btn.setEnabled(True)
            self.try_note.setText(f"<span style='color:{_hex('error')}'>{m}</span>")
        self._call(work, done, fail)

    def _show_try(self, results):
        from AIO_QC.tools.frame_extractor.core.imageio import imread
        from AIO_QC.tools.frame_extractor.widgets.autoannotate_window import bgr_to_pixmap
        from AIO_QC.widgets.design import mono
        while self.try_grid.count():
            w = self.try_grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        cost = sum(r.cost for r in results)
        secs = max((r.seconds for r in results), default=0)
        kept = sum(r.accepted for r in results)
        errs = [r for r in results if r.error]
        for r in results:
            card = QFrame()
            card.setObjectName("tryCard")
            cl = QHBoxLayout(card)
            img = imread(r.path)
            pic = QLabel()
            if img is not None:
                shapes = [{"label": s["label"] + (f" {s['text']}" if s.get("text") else ""),
                           "points": [[s["box"][0], s["box"][1]], [s["box"][2], s["box"][3]]],
                           "description": f"conf={s['conf']:.2f}"} for s in r.shapes]
                pic.setPixmap(bgr_to_pixmap(img, shapes, 420))
            cl.addWidget(pic)
            right = QVBoxLayout()
            verdict = ("<span style='color:%s'>✕ %s</span>" % (_hex("error"), r.error) if r.error else
                       "<span style='color:%s'>kept</span>" % _hex("ok") if r.accepted else
                       "<span style='color:%s'>rejected: %s</span>" % (_hex("warn"), "; ".join(r.reasons)))
            counts = ", ".join(f"{n} × {c}" for c, n in r.counts().items()) or "no boxes"
            extra = []
            if r.review:
                extra.append("needs review")
            if r.flags:
                extra.append(", ".join(f"{k}={v}" for k, v in r.flags.items()))
            if r.cached:
                extra.append("answer reused")
            head = QLabel(f"<b>{os.path.basename(r.path)}</b><br>{counts} · {verdict}"
                          + (f"<br><span style='color:{_hex('muted')}'>{' · '.join(extra)}</span>" if extra else ""))
            head.setWordWrap(True)
            right.addWidget(head)
            def pretty(a):
                try:
                    from AIO_QC.tools.frame_extractor.core.ai_annotate.parse import extract_json
                    return json.dumps(extract_json(a), indent=1, ensure_ascii=False)
                except ValueError:
                    return a
            raw = QPlainTextEdit("\n\n".join(pretty(a) for a in r.answers) or "(no answer)")
            raw.setReadOnly(True)
            raw.setFont(mono(10.5))
            raw.setMinimumHeight(140)
            right.addWidget(raw, 1)
            cl.addLayout(right, 1)
            self.try_grid.addWidget(card)
        self.try_note.setText(f"{kept} of {len(results)} kept · {sum(len(r.shapes) for r in results)} boxes"
                              + (f" · {len(errs)} failed" if errs else "")
                              + (f" · ${cost:.4f} for these" if cost else "") + f" · slowest {secs:.1f} s")

    def _estimate(self):
        res = getattr(self, "_last_try", None)
        if not res:
            self.try_note.setText("Try a few images first — the estimate is made from them.")
            return
        n = self.win.image_count() if self.win is not None and hasattr(self.win, "image_count") else 0
        if not n:
            self.try_note.setText("No images found yet (choose the source, or run Frames).")
            return
        fresh = [r for r in res if r.calls]
        per_cost = sum(r.cost for r in fresh) / len(fresh) if fresh else 0
        per_sec = sum(r.seconds for r in fresh) / len(fresh) if fresh else 0
        if not fresh:
            self.try_note.setText("These answers were reused (no new calls) — try other images for an estimate.")
            return
        workers = max(1, int(self.cfg["connection"].get("workers") or 1))
        mi = int(self.cfg["limits"].get("max_images") or 0)
        n = min(n, mi) if mi else n
        secs = n * per_sec / workers
        t = f"{secs / 3600:.1f} h" if secs > 3600 else f"{secs / 60:.0f} min" if secs > 90 else f"{secs:.0f} s"
        price = (f"≈ ${per_cost * n:,.2f}" if per_cost else "cost unknown (set the prices on the Checks tab)")
        bud = float(self.cfg["limits"].get("budget_usd") or 0)
        self.try_note.setText(f"Whole run: {n:,} images · {price} · ≈ {t} with {workers} at a time"
                              + (f" · the budget (${bud:g}) stops it after ≈ {int(bud / per_cost):,} images"
                                 if bud and per_cost and per_cost * n > bud else ""))

    # ------------------------------------------------------------------ yaml
    def _yaml_from_cfg(self):
        try:
            t = C.dumps(self.cfg)
        except RuntimeError as exc:
            t = f"# {exc}"
        self.yaml.blockSignals(True)
        self.yaml.setPlainText(t)
        self.yaml.blockSignals(False)
        self.yaml_lbl.setText("")

    def _yaml_live(self):
        try:
            cfg = C.loads(self.yaml.toPlainText())
        except C.ConfigError as exc:
            self.yaml_lbl.setText(f"<span style='color:{_hex('error')}'>Line {exc.line or '?'}: {exc}</span>")
            self.yaml_apply.setEnabled(False)
            return
        except RuntimeError as exc:
            self.yaml_lbl.setText(str(exc))
            return
        probs = C.check(cfg, os.path.dirname(self.path) if self.path else "")
        text = self.yaml.toPlainText()
        if probs:
            k, m = probs[0]
            ln = C.line_of(text, k)
            self.yaml_lbl.setText(f"<span style='color:{_hex('warn')}'>{'Line ' + str(ln) + ': ' if ln else ''}"
                                  f"{m}</span>" + (f" (+{len(probs) - 1} more)" if len(probs) > 1 else ""))
        else:
            self.yaml_lbl.setText(f"<span style='color:{_hex('ok')}'>Looks right — Apply to use it.</span>")
        self.yaml_apply.setEnabled(True)

    def _yaml_apply(self):
        try:
            cfg = C.loads(self.yaml.toPlainText())
        except (C.ConfigError, RuntimeError) as exc:
            self.yaml_lbl.setText(f"<span style='color:{_hex('error')}'>{exc}</span>")
            return
        self.cfg = cfg
        self._push()
        self._edited()
        self.yaml_lbl.setText("Applied.")

    def _tab_changed(self, i):
        if self.tabs.tabText(i) == "YAML":
            self._save_now()
            self._yaml_from_cfg()

    # ------------------------------------------------------------------ validation
    def _validate(self):
        if not self.path:
            self.problems.setText(f"<span style='color:{_hex('muted')}'>Start with <b>New…</b> (a template asks "
                                  f"what the AI should do) or <b>Import…</b> a config.yaml.</span>")
            return
        probs = C.check(self.cfg, os.path.dirname(self.path))
        if probs:
            self.problems.setText(f"<span style='color:{_hex('warn')}'>To do: " + " · ".join(m for _, m in probs[:3])
                                  + (f" (+{len(probs) - 3} more)" if len(probs) > 3 else "") + "</span>")
        else:
            self.problems.setText(f"<span style='color:{_hex('ok')}'>The project is complete.</span> Saved in "
                                  f"{self.path}")

    def run_problems(self) -> list[str]:
        """What stops a run (the window's Run button shows these)."""
        out = []
        if not self.path:
            return ["AI annotate: make or choose a project."]
        out += [f"AI annotate: {m}" for _, m in C.check(self.cfg, os.path.dirname(self.path))]
        if self._ready_items:
            out += [f"AI annotate: {i.title} — {i.detail}" for i in self._ready_items if i.status == "fail"]
        return out

    def flush(self):
        if self._save_t.isActive():
            self._save_t.stop()
            self._save_now()

    # ------------------------------------------------------------------ misc
    def _call(self, fn, done, fail=None):
        w = Call(fn, self)
        self._calls.append(w)

        def ok(res):
            try:
                done(res)
            except RuntimeError:
                pass

        def bad(m):
            try:
                (fail or (lambda x: logger.debug(f"AI annotate: {x}")))(m)
            except RuntimeError:
                pass
        w.done.connect(ok)
        w.error.connect(bad)
        w.finished.connect(lambda: w in self._calls and self._calls.remove(w))
        w.start()

    def wait_calls(self, ms=3000):
        for w in list(self._calls):
            w.wait(ms)
