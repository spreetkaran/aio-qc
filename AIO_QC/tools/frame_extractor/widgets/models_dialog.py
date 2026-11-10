# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Settings -> Models: one shared model configuration used by the detection overlay, the
timeline model track, marker pre-annotations and batch mining.

Loading YOLO models effortlessly (the AIO QC approach):
  * .onnx runs right here (onnxruntime / cv2.dnn) - no setup at all.
  * .pt / .engine run in any Python environment that has ultralytics + torch (conda, venv, system),
    through a small server subprocess: the app never imports torch, GPU / CUDA stay the env's business.
  * Adding a .pt with no environment chosen searches the discovered environments automatically
    (a few seconds - packages are only looked up, not imported). If none has ultralytics, one click builds
    a temporary environment (optionally reusing an existing PyTorch) and removes it when the app closes.
  * Class names and the training image size come from the model file; GPU devices from the environment.
"""
import os
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QFileDialog,
    QPlainTextEdit, QDialogButtonBox, QTabWidget, QWidget, QGroupBox, QMessageBox, QSplitter,
)

from AIO_QC.widgets.motion import animate_tabs
from AIO_QC.tools.frame_extractor.core import env_manager as envm
from AIO_QC.tools.frame_extractor.core.detector import HAVE_ORT, model_backend
from AIO_QC.tools.frame_extractor.utils import newIcon
from AIO_QC.widgets.testing.lab_common import FuncWorker  # noqa: F401  (re-exported)

DEFAULTS = dict(models=[], conf=0.25, iou=0.45, imgsz=0, device="auto", unc_low=0.25, unc_high=0.60,
                env_python="", half=False, sahi=False, sahi_slice=640, sahi_overlap=0.2,
                temp_env="", delete_temp_on_exit=True)
OK, BAD, WARN = "✅", "❌", "⚠️"
from AIO_QC.tools.frame_extractor.core.common import PT_EXTS
from AIO_QC.utils.fx_settings import fx_settings


def load_model_cfg():
    st = fx_settings()
    cfg = dict(DEFAULTS)
    models = st.value("models/paths", [])
    cfg["models"] = [m for m in (models if isinstance(models, list) else [models]) if m]
    for k, v in DEFAULTS.items():
        if k == "models":
            continue
        raw = st.value(f"models/{k}", v)
        cfg[k] = raw in (True, "true", "1", 1) if isinstance(v, bool) else type(v)(raw if raw is not None else v)
    # a temporary env that was left behind (locked files at exit) - retry the deletion
    for p in st.value("models/pending_delete", []) or []:
        if os.path.isdir(p):
            envm.remove_env(Path(p))
    st.setValue("models/pending_delete", [p for p in (st.value("models/pending_delete", []) or []) if os.path.isdir(p)])
    # one model registry / one environment for the whole app (Image QC ▸ Models & Environment, Testing, here)
    shared = _shared_env()
    if shared:
        cfg["env_python"] = shared
    _register(cfg["models"])
    return cfg


def _shared_env():
    from AIO_QC.utils.qt_utils import app_settings
    return app_settings().value("inference/python", "") or ""


def _register(paths):
    try:
        from AIO_QC.core import model_registry as R
        known = {os.path.normcase(os.path.abspath(m.get("path", ""))) for m in R.app_models()}
        for p in paths:
            if p and os.path.isfile(p) and os.path.normcase(os.path.abspath(p)) not in known:
                R.register(p)
    except Exception:
        pass


def save_model_cfg(cfg):
    st = fx_settings()
    st.setValue("models/paths", list(cfg["models"]))
    _register(cfg["models"])
    if cfg.get("env_python"):                    # the one environment for .pt / .engine models
        from AIO_QC.utils.qt_utils import app_settings
        app_settings().setValue("inference/python", cfg["env_python"])
    for k in DEFAULTS:
        if k != "models":
            st.setValue(f"models/{k}", cfg[k])


def detector_kwargs(cfg, threads=0):
    """cfg -> keyword arguments for core.detector.load_detectors (minus conf)."""
    return dict(iou=cfg["iou"], imgsz=cfg["imgsz"], device=cfg["device"], threads=threads,
                env_python=cfg.get("env_python", ""), half=cfg.get("half", False),
                sahi={"slice": cfg["sahi_slice"], "overlap": cfg["sahi_overlap"]} if cfg.get("sahi") else None)


def shutdown_models(cfg):
    """App exit: stop env servers; delete the temporary environment if asked to."""
    from AIO_QC.tools.frame_extractor.core.detector import shutdown_clients
    shutdown_clients()
    tmp = cfg.get("temp_env")
    if tmp and cfg.get("delete_temp_on_exit") and os.path.isdir(tmp):
        if not envm.remove_env(Path(tmp)):
            st = fx_settings()
            st.setValue("models/pending_delete", sorted(set((st.value("models/pending_delete", []) or []) + [tmp])))
        else:
            if cfg.get("env_python", "").startswith(tmp):
                cfg["env_python"] = ""
            cfg["temp_env"] = ""
            save_model_cfg(cfg)


class ModelsDialog(QDialog):
    def __init__(self, parent, cfg, frame_getter=None, show_detections=None):
        super().__init__(parent)
        self.setWindowTitle("Models & environment")
        self.setWindowIcon(newIcon("models"))
        from AIO_QC.tools.frame_extractor.utils.qt_utils import preferred_size
        preferred_size(self, 880, 680)
        self.cfg = dict(cfg)
        self.frame_getter = frame_getter
        self.show_detections = show_detections
        self.info = None            # probe of the environment
        self.workers = []
        self.proc = None
        self.model_info = {}        # path -> text

        lay = QVBoxLayout(self)
        split = QSplitter(Qt.Orientation.Vertical)
        self.tabs = QTabWidget()
        animate_tabs(self.tabs)
        self.tabs.addTab(self._build_models_tab(), newIcon("models"), "Models")
        self.tabs.addTab(self._build_env_tab(), newIcon("environment"), "Python environment (.pt models)")
        split.addWidget(self.tabs)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(4000)
        self.console.setPlaceholderText("Environment checks, installs and model loading are logged here.")
        f = self.console.font()
        f.setFamily("Consolas")
        self.console.setFont(f)
        split.addWidget(self.console)
        split.setSizes([520, 160])
        lay.addWidget(split, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self._refresh_models()
        self._refresh_env_labels()
        if self.cfg["env_python"]:
            self.check_env(self.cfg["env_python"], quiet=True)

    # ================================================================ models tab
    def _build_models_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setTextFormat(Qt.TextFormat.RichText)
        self.banner.linkActivated.connect(self._banner_link)
        lay.addWidget(self.banner)

        row = QHBoxLayout()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Model (first = primary)", "Runs with", "Classes", "Input"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        row.addWidget(self.table, 1)
        col = QVBoxLayout()
        for text, icon, slot in (("Add…", "open", self.add), ("Remove", "delete", self.remove),
                                 ("Up", None, lambda: self.move(-1)), ("Down", None, lambda: self.move(1)),
                                 ("Test load", "run", self.test_load),
                                 ("Test on current frame", "detect", self.test_frame),
                                 ("Export .pt → ONNX", "extract", self.export_onnx)):
            b = QPushButton(text)
            if icon:
                b.setIcon(newIcon(icon))
            b.clicked.connect(slot)
            col.addWidget(b)
            if text == "Test on current frame":
                b.setEnabled(self.frame_getter is not None)
                b.setToolTip("Run every model on the frame shown in the player and draw the boxes")
            if text.startswith("Export"):
                b.setToolTip("Export the selected .pt with ultralytics in the environment. ONNX models run "
                             "without any environment (also in batch workers).")
        col.addStretch(1)
        row.addLayout(col)
        lay.addLayout(row, 1)

        g = QGridLayout()
        c = self.cfg
        self.conf = self._dspin(c["conf"], "Detections below this are ignored")
        self.iou = self._dspin(c["iou"], "NMS IoU")
        self.unc_low = self._dspin(c["unc_low"], "Uncertainty band low (mining, timeline, dashed boxes)")
        self.unc_high = self._dspin(c["unc_high"], "Uncertainty band high")
        self.imgsz = QSpinBox()
        self.imgsz.setToolTip("Inference image size. 'from model' uses the size the model was trained at (best accuracy). Larger finds smaller objects but is slower.")
        self.imgsz.setRange(0, 4096)
        self.imgsz.setSingleStep(32)
        self.imgsz.setSpecialValueText("from model (training size)")
        self.imgsz.setValue(c["imgsz"])
        self.device = QComboBox()
        self.device.setToolTip(".onnx: auto = CUDA / DirectML when onnxruntime-gpu / -directml is installed.\n"
                               ".pt: devices found in the environment (Check environment).")
        self.half = QCheckBox("FP16 (GPU, .pt)")
        self.half.setToolTip('Half precision on NVIDIA GPUs: ~2x faster .pt inference, same accuracy for practically all YOLO models. Ignored on CPU.')
        self.half.setChecked(c["half"])
        self.sahi = QCheckBox("SAHI sliced inference (.pt)")
        self.sahi.setToolTip("Tiles large frames so small objects (plates, signs) are found. Needs `sahi` in the "
                             "environment. Slower.")
        self.sahi.setChecked(c["sahi"])
        self.slice = QSpinBox()
        self.slice.setToolTip('SAHI tile size in pixels: the frame is cut into tiles of this size, each run through the model at full detail. Smaller = finds tinier objects, slower.')
        self.slice.setRange(128, 2048)
        self.slice.setSingleStep(64)
        self.slice.setPrefix("slice ")
        self.slice.setValue(c["sahi_slice"])
        self.overlap = QDoubleSpinBox()
        self.overlap.setToolTip("Overlap between SAHI tiles (fraction), so objects on tile borders aren't cut in half. 0.2 is a good default.")
        self.overlap.setRange(0.0, 0.8)
        self.overlap.setSingleStep(0.05)
        self.overlap.setPrefix("overlap ")
        self.overlap.setValue(c["sahi_overlap"])
        for r, (label, wdg) in enumerate((("Confidence", self.conf), ("NMS IoU", self.iou), ("Input size", self.imgsz))):
            g.addWidget(QLabel(label), r, 0)
            g.addWidget(wdg, r, 1)
        for r, (label, wdg) in enumerate((("Uncertain from", self.unc_low), ("Uncertain below", self.unc_high),
                                          ("Device", self.device))):
            g.addWidget(QLabel(label), r, 2)
            g.addWidget(wdg, r, 3)
        sr = QHBoxLayout()
        for wdg in (self.half, self.sahi, self.slice, self.overlap):
            sr.addWidget(wdg)
        sr.addStretch(1)
        g.addLayout(sr, 3, 0, 1, 4)
        lay.addLayout(g)
        self._fill_devices()
        return w

    def _dspin(self, v, tip):
        s = QDoubleSpinBox()
        s.setRange(0.01, 0.99)
        s.setSingleStep(0.05)
        s.setDecimals(2)
        s.setValue(v)
        s.setToolTip(tip)
        return s

    def _models(self):
        return [self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in range(self.table.rowCount())]

    def _needs_env(self):
        return any(m.lower().endswith(PT_EXTS) for m in self._models())

    def _refresh_models(self, models=None):
        models = self._models() if models is None else models
        self.table.setRowCount(0)
        for m in (models if models is not None else self.cfg["models"]):
            r = self.table.rowCount()
            self.table.insertRow(r)
            it = QTableWidgetItem(("★ " if r == 0 else "") + os.path.basename(m))
            it.setToolTip(m)
            it.setData(Qt.ItemDataRole.UserRole, m)
            self.table.setItem(r, 0, it)
            backend = model_backend(m, self.cfg["env_python"])
            self.table.setItem(r, 1, QTableWidgetItem(
                {"environment": "PyTorch · environment", "needs environment": "⚠ needs an environment"}.get(backend, backend)))
            info = self.model_info.get(m, ("", ""))
            self.table.setItem(r, 2, QTableWidgetItem(info[0]))
            self.table.setItem(r, 3, QTableWidgetItem(info[1]))
        self._refresh_banner()

    def _refresh_banner(self):
        py = self.cfg["env_python"]
        if not self._needs_env():
            txt = "ONNX models run inside AIO QC — nothing to set up." if self._models() else \
                "Add YOLO models (.pt, .onnx, .engine). <b>.onnx</b> runs directly; <b>.pt</b> runs in a Python " \
                "environment with ultralytics, found automatically."
            if not HAVE_ORT and self._models():
                txt += f" {WARN} onnxruntime is not installed — using cv2.dnn (slower). <i>pip install onnxruntime</i>"
        elif py and self.info and not envm.missing_packages(self.info, "sahi" if self.sahi.isChecked() else "standard"):
            gpus = [d["name"] for d in self.info.get("devices", []) if d["id"] != "cpu"]
            txt = (f"{OK} .pt models run in <b>{self._env_name(py)}</b> · ultralytics {self.info.get('ultralytics')} · "
                   f"torch {self.info.get('torch')} · " + (" | ".join(gpus) if gpus else "CPU only"))
        elif py and self.info is None:
            txt = f"… checking <b>{self._env_name(py)}</b>"
        else:
            txt = (f"{WARN} .pt models need a Python environment with ultralytics. "
                   "<a href='find'>Find one automatically</a> · <a href='temp'>Create a temporary one</a> · "
                   "<a href='onnx'>Export to ONNX instead</a>")
        self.banner.setText(txt)

    def _banner_link(self, link):
        if link == "find":
            self.find_env()
        elif link == "temp":
            self.tabs.setCurrentIndex(1)
            self.start_install()
        elif link == "onnx":
            self.export_onnx()

    def _env_name(self, py):
        for name, p in envm.discover_envs() if not hasattr(self, "_envs_cache") else self._envs_cache:
            if os.path.normcase(p) == os.path.normcase(py):
                return name
        return py

    def add(self):
        files = self._pick_known()
        if files is None:
            files, _ = QFileDialog.getOpenFileNames(self, "Add YOLO models", "",
                                                    "YOLO models (*.pt *.onnx *.engine);;All files (*)")
        if not files:
            return
        _register(files)
        self._refresh_models(self._models() + [f for f in files if f not in self._models()])
        if any(f.lower().endswith(PT_EXTS) for f in files) and not self.cfg["env_python"]:
            self.find_env()          # effortless: go and look for an environment right away
        else:
            self.test_load()

    def _pick_known(self):
        """AIO QC's models (Image QC, Testing, here) not in the list yet, plus Browse… -> paths, or None = browse."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QCursor
        from AIO_QC.core import model_registry as R
        have = {os.path.normcase(os.path.abspath(p)) for p in self._models()}
        known = [m for m in R.app_models() if os.path.isfile(m.get("path", "")) and
                 os.path.normcase(os.path.abspath(m["path"])) not in have]
        if not known:
            return None
        m = QMenu(self)
        browse = m.addAction("Browse…")
        m.addSeparator()
        m.addSection("AIO QC's models")
        for e in known:
            m.addAction(f"{e.get('name') or os.path.basename(e['path'])}  —  {e['path']}").setData(e["path"])
        act = m.exec(QCursor.pos())
        if act is None:
            return []
        if act is browse:
            return None
        return [act.data()]

    def remove(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        models = self._models()
        for r in rows:
            models.pop(r)
        self._refresh_models(models)

    def move(self, d):
        r = self.table.currentRow()
        models = self._models()
        if 0 <= r < len(models) and 0 <= r + d < len(models):
            models[r], models[r + d] = models[r + d], models[r]
            self._refresh_models(models)
            self.table.selectRow(r + d)

    def cfg_out(self):
        c = dict(self.cfg)
        c.update(models=self._models(), conf=self.conf.value(), iou=self.iou.value(), imgsz=self.imgsz.value(),
                 device=self.device.currentData() or "auto", unc_low=self.unc_low.value(),
                 unc_high=self.unc_high.value(), half=self.half.isChecked(), sahi=self.sahi.isChecked(),
                 sahi_slice=self.slice.value(), sahi_overlap=self.overlap.value())
        return c

    def _fill_devices(self):
        want = self.cfg.get("device", "auto")
        self.device.blockSignals(True)
        self.device.clear()
        self.device.addItem("auto (best available)", "auto")
        self.device.addItem("CPU", "cpu")
        for d in (self.info or {}).get("devices", []):
            if d["id"] != "cpu":
                self.device.addItem(d["name"], d["id"])
        i = self.device.findData(want)
        self.device.setCurrentIndex(i if i >= 0 else 0)
        self.device.blockSignals(False)

    # --------------------------------------------------------------- testing
    def _run(self, fn, done, busy_text):
        self._log(busy_text)
        w = FuncWorker(fn, self)
        w.note.connect(self._log)
        w.done.connect(done)
        w.failed.connect(lambda e: self._log(f"{BAD} {e}"))
        w.finished.connect(lambda: self.workers.remove(w) if w in self.workers else None)
        self.workers.append(w)
        w.start()

    def test_load(self):
        models = self._models()
        if not models:
            return
        c = self.cfg_out()
        from AIO_QC.tools.frame_extractor.core.detector import load_detector

        def work(note):
            import numpy as np
            out = {}
            for m in models:
                try:
                    t0 = time.perf_counter()
                    d = load_detector(m, c["conf"], **detector_kwargs(c))
                    t1 = time.perf_counter()
                    d.detect(np.zeros((720, 1280, 3), np.uint8))
                    ms = (time.perf_counter() - t1) * 1000
                    names = ", ".join(d.names[k] for k in sorted(d.names)) if d.names else "unknown (add <model>.names)"
                    out[m] = (names, f"{d.imgsz[0]} px", f"{os.path.basename(m)}: {d.provider} · load {t1 - t0:.1f}s · "
                                                         f"~{ms:.0f} ms/frame (first run)")
                    note(f"{OK} {out[m][2]}")
                except Exception as e:
                    out[m] = ("", "", f"{os.path.basename(m)}: ERROR {e}")
                    note(f"{BAD} {out[m][2]}")
            return out

        def done(out):
            for m, (names, size, _) in out.items():
                self.model_info[m] = (names, size)
            self._refresh_models()
        self._run(work, done, f"▶ loading {len(models)} model(s)…")

    def test_frame(self):
        frame = self.frame_getter() if self.frame_getter else None
        if frame is None:
            QMessageBox.information(self, "Models", "Load a video and pause on a frame first.")
            return
        models, c = self._models(), self.cfg_out()
        from AIO_QC.tools.frame_extractor.core.detector import load_detectors

        def work(note):
            dets = load_detectors(models, min(c["conf"], c["unc_low"]), **detector_kwargs(c))
            return [d.detect(frame) for d in dets]

        def done(per_model):
            for m, dets in zip(models, per_model):
                self._log(f"  {os.path.basename(m)}: {len(dets)} box(es) " +
                          ", ".join(f"{d.name} {d.conf:.2f}" for d in dets[:8]) + (" …" if len(dets) > 8 else ""))
            if self.show_detections:
                self.show_detections(per_model, c)
        self._run(work, done, "▶ running every model on the current frame…")

    def export_onnx(self):
        pts = [m for m in self._models() if m.lower().endswith(".pt")]
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        sel = [self._models()[r] for r in rows if self._models()[r].lower().endswith(".pt")] or pts
        if not sel:
            QMessageBox.information(self, "Export", "Add a .pt model first.")
            return
        if not self.cfg["env_python"]:
            QMessageBox.information(self, "Export", "Exporting uses ultralytics in a Python environment — "
                                    "pick or create one on the environment tab first.")
            self.tabs.setCurrentIndex(1)
            return
        py, c = self.cfg["env_python"], self.cfg_out()

        def work(note):
            from AIO_QC.tools.frame_extractor.core.detector import _client
            client = _client(py)
            out = {}
            for m in sel:
                note(f"exporting {os.path.basename(m)} → ONNX (dynamic, simplified)…")
                res = client.export({"path": m, "format": "onnx", "imgsz": c["imgsz"] or 640})
                out[m] = res["path"]
            return out

        def done(out):
            models = self._models()
            for pt, onnx in out.items():
                self._log(f"{OK} {os.path.basename(pt)} → {onnx}")
                if onnx not in models:
                    models.insert(models.index(pt) + 1, onnx)
            self._refresh_models(models)
            QMessageBox.information(self, "Export", "ONNX export(s) added to the list. Remove the .pt entries if "
                                    "you want to run without the environment.")
        self._run(work, done, f"▶ export {len(sel)} model(s) with ultralytics in {py}")

    # =========================================================== environment tab
    def _build_env_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        intro = QLabel("<b>.pt</b> models run in a Python environment that has <i>ultralytics</i> and <i>torch</i> "
                       "(your training env is ideal). AIO QC starts a small server in it — it never imports "
                       "torch itself, so versions and CUDA builds can't clash. <b>.onnx</b> models need none of this.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        ex = QGroupBox("Use an existing environment")
        g = QGridLayout(ex)
        self.env_box = QComboBox()
        self.env_box.setToolTip('Python environment (conda / venv / system) that has ultralytics + torch - your training environment is ideal. Pick one, browse, or type a path.')
        self.env_box.setEditable(True)
        self.env_box.setMinimumWidth(460)
        self._fill_envs()
        for c, (text, slot, tip) in enumerate((
                ("↻", self._fill_envs, "Re-scan conda / venv environments"),
                ("Browse…", self._browse_env, "Pick an environment folder or a python executable"),
                ("Check", lambda: self.check_env(), "Start the server in it and list versions / GPUs"),
                ("Find automatically", self.find_env, "Look through all environments for ultralytics + torch"))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            g.addWidget(b, 0, 2 + c)
        g.addWidget(QLabel("Environment:"), 0, 0)
        g.addWidget(self.env_box, 0, 1)
        self.status_lbls = {}
        for r, (key, title) in enumerate((("python", "Python"), ("ultralytics", "ultralytics"), ("torch", "PyTorch"),
                                          ("cuda", "GPU"), ("sahi", "sahi")), start=1):
            g.addWidget(QLabel(title + ":"), r, 0)
            lbl = QLabel("–")
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            g.addWidget(lbl, r, 1, 1, 5)
            self.status_lbls[key] = lbl
        lay.addWidget(ex)

        tmp = QGroupBox("…or create a temporary environment (deleted when AIO QC closes)")
        f = QFormLayout(tmp)
        self.driver = envm.detect_cuda_driver()
        self.torch_box = QComboBox()
        self.torch_box.setToolTip("PyTorch build for the temporary environment. 'auto' picks the CUDA version your NVIDIA driver supports, or CPU without an NVIDIA GPU.")
        for key, text in envm.TORCH_CHOICES:
            self.torch_box.addItem(text, key)
        self.reuse_torch = QCheckBox("Reuse PyTorch from the environment above (much smaller download)")
        self.reuse_torch.setToolTip('Reuse the PyTorch already installed in the selected environment: the temporary environment then only adds ultralytics (~150 MB instead of 2-4 GB).')
        self.keep_cache = QCheckBox("Keep pip's download cache (faster next time)")
        self.keep_cache.setToolTip("Keep pip's download cache so recreating the environment later doesn't download everything again.")
        self.keep_cache.setChecked(True)
        self.delete_on_exit = QCheckBox("Delete the temporary environment when AIO QC closes")
        self.delete_on_exit.setToolTip('Remove the temporary environment when AIO QC closes (it is re-created on demand). Untick to keep it between sessions.')
        self.delete_on_exit.setChecked(self.cfg["delete_temp_on_exit"])
        drv = (f"NVIDIA driver supports CUDA {self.driver} → auto = {envm.pick_torch_build(self.driver)}"
               if self.driver else "no NVIDIA GPU detected → auto = CPU build")
        btns = QHBoxLayout()
        self.install_btn = QPushButton("Install into a temporary environment")
        self.install_btn.setToolTip('Create a temporary environment and install what .pt models need (progress in the console below).')
        self.install_btn.setIcon(newIcon("extract"))
        self.install_btn.clicked.connect(self.start_install)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip('Stop the installation; the partial environment is removed.')
        self.cancel_btn.setIcon(newIcon("stop"))
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(lambda: self.proc and self.proc.kill())
        self.remove_btn = QPushButton("Remove it now")
        self.remove_btn.setToolTip('Delete the temporary environment now.')
        self.remove_btn.setIcon(newIcon("delete"))
        self.remove_btn.clicked.connect(self.remove_temp)
        for b in (self.install_btn, self.cancel_btn, self.remove_btn):
            btns.addWidget(b)
        btns.addStretch(1)
        self.tmp_info = QLabel()
        self.tmp_info.setStyleSheet("color: gray;")
        f.addRow("PyTorch build:", self.torch_box)
        f.addRow("", QLabel(drv))
        f.addRow(self.reuse_torch)
        f.addRow(self.keep_cache)
        f.addRow(self.delete_on_exit)
        f.addRow(btns)
        f.addRow(self.tmp_info)
        lay.addWidget(tmp)
        lay.addStretch(1)
        return w

    def _fill_envs(self):
        self._envs_cache = envm.discover_envs()
        cur = self.cfg["env_python"]
        self.env_box.clear()
        for name, py in self._envs_cache:
            self.env_box.addItem(f"{name}  —  {py}", py)
        tmp = self.cfg.get("temp_env")
        if tmp:
            py = str(envm.env_python(Path(tmp)))
            if os.path.isfile(py) and self.env_box.findData(py) < 0:
                self.env_box.addItem(f"temporary  —  {py}", py)
        if cur:
            i = self.env_box.findData(cur)
            if i < 0:
                self.env_box.addItem(f"current  —  {cur}", cur)
                i = self.env_box.count() - 1
            self.env_box.setCurrentIndex(i)
        self.tmp_info.setText(f"Temporary environment: {tmp}" if tmp else "No temporary environment.") \
            if hasattr(self, "tmp_info") else None

    def _browse_env(self):
        d = QFileDialog.getExistingDirectory(self, "Environment folder (conda env / venv root)")
        if not d:
            return
        py = envm.python_in_env(d)
        if py is None:
            QMessageBox.warning(self, "Environment", "No python executable found in that folder.")
            return
        self.env_box.addItem(f"browsed  —  {py}", str(py))
        self.env_box.setCurrentIndex(self.env_box.count() - 1)
        self.check_env(str(py))

    def _selected_python(self):
        data = self.env_box.currentData()
        if data and self.env_box.currentText().endswith(str(data)):
            return str(data)
        text = self.env_box.currentText().split("—")[-1].strip()
        py = envm.python_in_env(text) if text else None
        return str(py) if py else text

    def find_env(self):
        self._log("▶ looking for an environment with ultralytics + torch (packages are looked up, not imported)…")
        self.banner.setText("… searching environments for ultralytics + torch")

        def work(note):
            return envm.find_ready_env(progress=lambda i, n, name: note(f"  [{i + 1}/{n}] {name}"))

        def done(res):
            if res:
                name, py = res
                self._log(f"{OK} found: {name}")
                i = self.env_box.findData(py)
                if i >= 0:
                    self.env_box.setCurrentIndex(i)
                self.check_env(py)
            else:
                self._log(f"{WARN} no environment has ultralytics + torch.")
                self._refresh_banner()
                if QMessageBox.question(self, "No environment found",
                                        "None of your Python environments has ultralytics + torch.\n\n"
                                        "Create a temporary one now? (It is deleted when AIO QC closes.)") \
                        == QMessageBox.StandardButton.Yes:
                    self.tabs.setCurrentIndex(1)
                    self.start_install()
        self._run(work, done, "")

    def check_env(self, python=None, quiet=False):
        py = python or self._selected_python()
        if not py or not os.path.isfile(py):
            if not quiet:
                QMessageBox.warning(self, "Environment", "Pick a Python environment first.")
            return
        from AIO_QC.tools.frame_extractor.core import detector as D
        if py != self.cfg["env_python"]:
            D.shutdown_clients()
        self.cfg["env_python"] = py
        self.info = None
        self._refresh_env_labels()
        self._refresh_models()

        def work(note):
            note(f"=== checking {py} (importing torch can take a while) ===")
            return D._client(py).info

        def done(info):
            self.info = info
            miss = envm.missing_packages(info, "sahi" if self.sahi.isChecked() else "standard")
            self._log(f"{OK} environment ready" if not miss else f"{WARN} missing: {', '.join(miss)}")
            self._fill_devices()
            if self.device.currentData() == "auto" and self.device.count() > 2:
                self.device.setCurrentIndex(2)          # first GPU by default
            self.reuse_torch.setChecked(bool(info.get("torch")))
            self._refresh_env_labels()
            self._refresh_models()
            if miss and not quiet and QMessageBox.question(
                    self, "Environment", f"This environment lacks: {', '.join(miss)}.\n\nInstall them into a temporary "
                    "environment" + (" that reuses its PyTorch?" if info.get("torch") else "?")) \
                    == QMessageBox.StandardButton.Yes:
                self.start_install()
        self._run(work, done, "")

    def _refresh_env_labels(self):
        info, L = self.info, self.status_lbls
        if not info:
            for k, lbl in L.items():
                lbl.setText("– (not checked)" if k == "python" else "–")
            return
        L["python"].setText(f"{OK} {info['python']} · {info.get('executable', '')}")
        L["ultralytics"].setText(f"{OK} {info['ultralytics']}" if info.get("ultralytics") else f"{BAD} not installed")
        t = info.get("torch")
        L["torch"].setText(f"{OK} {t}" if t else f"{BAD} not installed" +
                           (f" ({info['torch_error']})" if info.get("torch_error") else ""))
        gpus = [d["name"] for d in info.get("devices", []) if d["id"] != "cpu"]
        L["cuda"].setText(f"{OK} " + " | ".join(gpus) if gpus else f"{WARN} CPU only" +
                          (" — an NVIDIA driver is present: a CUDA build of PyTorch would be much faster"
                           if self.driver else ""))
        L["sahi"].setText(f"{OK} {info['sahi']}" if info.get("sahi") else "– not installed (only for SAHI)")

    # ------------------------------------------------------ temporary env
    def start_install(self):
        if self.proc is not None:
            return
        mode = "sahi" if self.sahi.isChecked() else "standard"
        info = self.info
        reuse = self.reuse_torch.isChecked() and info and info.get("torch")
        base = info.get("executable") if reuse else None
        if reuse:
            packages = [p for p in envm.required_packages(mode) if not info.get(p)] or envm.required_packages(mode)
            torch_build = None
        else:
            packages = envm.required_packages(mode)
            key = self.torch_box.currentData()
            torch_build = envm.pick_torch_build(self.driver) if key == "auto" else key
        env_dir = envm.new_temp_env_dir()
        self.steps = envm.install_plan(env_dir, packages, torch_build, self.keep_cache.isChecked(), base, bool(reuse))
        size = "~150 MB" if reuse else ("~1 GB" if torch_build == "cpu" else "~2–4 GB")
        if QMessageBox.question(self, "Temporary environment",
                                f"Create {env_dir}\nand install: {', '.join((['torch'] if torch_build else []) + packages)}"
                                f"\n\nDownload size {size}. Continue?") != QMessageBox.StandardButton.Yes:
            return
        from AIO_QC.tools.frame_extractor.core.detector import shutdown_clients
        shutdown_clients()
        self._pending_env, self._step, self._t0 = env_dir, 0, time.monotonic()
        self._log(f"\n=== Temporary environment: {env_dir} ===")
        self.install_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self._next_step()

    def _next_step(self):
        if self._step >= len(self.steps):
            return self._install_finished(True)
        title, argv = self.steps[self._step]
        self._log(f"\n▶ [{self._step + 1}/{len(self.steps)}] {title}\n$ {' '.join(argv)}")
        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        for k in ("PYTHONHOME", "PYTHONPATH"):
            env.remove(k)
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PIP_NO_INPUT", "1")
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda p=proc: self._on_output(p))
        proc.finished.connect(self._on_step_finished)
        self.proc = proc
        proc.start(argv[0], argv[1:])

    def _on_output(self, proc):
        text = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.replace("\r\n", "\n").split("\n"):
            if line.strip():
                self._log(line.split("\r")[-1])

    def _on_step_finished(self, code, _status):
        self.proc = None
        if code != 0:
            self._log(f"{BAD} step failed (exit code {code})")
            return self._install_finished(False)
        self._step += 1
        self._next_step()

    def _install_finished(self, ok):
        self.install_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        env_dir = self._pending_env
        mins = (time.monotonic() - self._t0) / 60
        if not ok:
            self._log(f"Installation stopped after {mins:.1f} min. Removing the partial environment…")
            envm.remove_env(env_dir)
            return
        old = self.cfg.get("temp_env")
        if old and old != str(env_dir):
            envm.remove_env(Path(old))
        self.cfg["temp_env"] = str(env_dir)
        self.cfg["delete_temp_on_exit"] = self.delete_on_exit.isChecked()
        self._log(f"{OK} installed in {mins:.1f} min ({envm.dir_size_mb(env_dir):,.0f} MB)")
        self._fill_envs()
        self.check_env(str(envm.env_python(env_dir)))

    def remove_temp(self):
        tmp = self.cfg.get("temp_env")
        if not tmp:
            return
        if QMessageBox.question(self, "Remove", f"Delete {tmp}?") != QMessageBox.StandardButton.Yes:
            return
        from AIO_QC.tools.frame_extractor.core.detector import shutdown_clients
        shutdown_clients()
        if self.cfg["env_python"].startswith(tmp):
            self.cfg["env_python"] = ""
            self.info = None
        ok = envm.remove_env(Path(tmp))
        self._log(f"{OK + ' removed' if ok else BAD + ' could not remove'} {tmp}")
        if ok:
            self.cfg["temp_env"] = ""
        self._fill_envs()
        self._refresh_env_labels()
        self._refresh_models()

    # ----------------------------------------------------------------- misc
    def _log(self, text):
        if not text:
            return
        self.console.appendPlainText(text)
        self.console.moveCursor(QTextCursor.MoveOperation.End)

    def accept(self):
        self.cfg["delete_temp_on_exit"] = self.delete_on_exit.isChecked()
        if self.proc is not None:
            QMessageBox.information(self, "Install running", "Wait for the installation to finish or cancel it.")
            return
        super().accept()

    def reject(self):
        if self.proc is not None:
            self.proc.kill()
        super().reject()

    def closeEvent(self, ev):
        for w in list(self.workers):
            w.wait(2000)
        super().closeEvent(ev)
