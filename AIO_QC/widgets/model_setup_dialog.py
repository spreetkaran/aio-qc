# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
🤖 Models & Environment

Tab 1 · Environment: pick an existing Python env (conda / venv / system; auto-discovered or browsed),
        check it (Python, ultralytics, torch + CUDA devices, sahi), or build a TEMPORARY env — optionally
        reusing an existing PyTorch — with the install streamed into the console. Temporary envs are
        deleted when AIO QC closes (and pip's download cache can be skipped) — nothing is left behind.
Tab 2 · Models: several models per project, each with its own class → label map and image size;
        standard or SAHI (sliced) inference, device from the detected list, FP16, batch size,
        cross-model NMS; label-coverage overview; "Test on current image" overlays every model's boxes.
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt
from PyQt6.QtGui import QColor, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPlainTextEdit, QProgressBar,
    QRadioButton, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from AIO_QC.core.checks import normalize_label
from AIO_QC.core.env_manager import (
    TORCH_CHOICES, detect_cuda_driver, dir_size_mb, discover_envs, env_python, install_plan, missing_packages,
    new_temp_env_dir, pick_torch_build, python_in_env, remove_env, required_packages,
)
from AIO_QC.core.model_check import coverage, merge_predictions
from AIO_QC.core.state import load_models_config, save_models_config
from AIO_QC.utils import hexc, style as apply_style
from AIO_QC.utils import labelValidator, newButton
from .base_scan import dspin, ispin, readable_table

MODEL_COLORS = ["#ff2bd6", "#00b7ff", "#ffae00", "#5bd900", "#b36bff", "#ff5252", "#00d5a0", "#ffd400"]
IMGSZ_PRESETS = [0, 320, 416, 512, 640, 768, 960, 1024, 1280, 1536, 1920]
OK, BAD, WARN = "✅", "❌", "⚠️"


def model_color(i: int) -> str:
    return MODEL_COLORS[i % len(MODEL_COLORS)]


TRAINED_ON = [("all", "this dataset (all)"), ("train", "train split only"), ("none", "held-out (not this data)"),
              ("list", "image list…")]
RUNTIMES = [("pt", "PyTorch (.pt)"), ("onnx", "ONNX"), ("engine", "TensorRT")]


def src_sig(path) -> str:
    try:
        st = Path(path).stat()
        return f"{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        return "missing"


def device_name(info: dict | None, dev: str) -> str:
    for d in (info or {}).get("devices", []):
        if d["id"] == dev:
            return d["name"]
    return dev


def effective_imgsz(m: dict) -> int:
    return int(m.get("imgsz") or m.get("train_imgsz") or 640)


def runtime_path(m: dict, opts: dict, info: dict | None) -> tuple[str, str]:
    """(file to load, note). Falls back to the .pt whenever the chosen export is missing or stale."""
    rt = m.get("runtime", "pt")
    if rt == "pt":
        return m["path"], ""
    e = (m.get("exports") or {}).get(rt)
    if not e or not Path(e.get("path", "")).is_file():
        return m["path"], f"no {rt} export → using .pt"
    if e.get("src_sig") != src_sig(m["path"]):
        return m["path"], f"{rt} export outdated (model file changed) → .pt"
    if int(e.get("imgsz", 0)) != effective_imgsz(m):
        return m["path"], f"{rt} export was built for imgsz {e.get('imgsz')} → .pt"
    if rt == "engine":
        dev = str(opts.get("device") or "cpu")
        if dev in ("cpu", "mps"):
            return m["path"], "TensorRT needs a CUDA device → .pt"
        if e.get("device_name") != device_name(info, dev):
            return m["path"], f"engine built on {e.get('device_name')} → .pt"
        if info and info.get("tensorrt") and e.get("trt") != info.get("tensorrt"):
            return m["path"], f"engine built with TensorRT {e.get('trt')} → .pt"
    return e["path"], ""


def server_specs(models: list[dict], opts: dict, info: dict | None = None) -> list[dict]:
    """GUI model entries + shared options -> what infer_server.load() expects."""
    dev = str(opts.get("device") or "cpu")
    half = bool(opts.get("half")) and dev not in ("cpu", "mps")
    out = []
    for m in models:
        if not m.get("enabled", True) or not Path(m.get("path", "")).is_file():
            continue
        path, _note = runtime_path(m, opts, info)
        out.append({"id": m["id"], "path": path, "imgsz": effective_imgsz(m),
                    "device": dev, "half": half, "warmup": True, "conf": 0.05, "batch": int(opts.get("batch", 8)),
                    "sahi": {"slice": int(opts.get("slice", 640)), "overlap": float(opts.get("overlap", 0.2))}
                    if opts.get("mode") == "sahi" else None})
    return out


def auto_class_map(names: dict[str, str], labels: list[str], map_label=lambda x: x) -> dict[str, str]:
    """model class -> dataset label: exact, then normalised, then through the view mapping; else ignore."""
    norm = {normalize_label(l): l for l in labels}
    out = {}
    for cid, name in names.items():
        if name in labels:
            out[cid] = name
        elif normalize_label(name) in norm:
            out[cid] = norm[normalize_label(name)]
        elif map_label(name) in labels:
            out[cid] = map_label(name)
        else:
            out[cid] = ""
    return out


# =============================================================================
class ClassMapDialog(QDialog):
    def __init__(self, parent, model: dict, labels: list[str], map_label):
        super().__init__(parent)
        self.model, self.labels, self.map_label = model, labels, map_label
        self.setWindowTitle(f"Class mapping · {model.get('name')}")
        self.resize(520, 460)
        names = model.get("names") or {}
        cmap = dict(model.get("class_map") or {})
        self.table = QTableWidget(len(names), 3)
        self.table.setHorizontalHeaderLabels(["ID", "Model class", "→ Dataset label"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        readable_table(self.table)
        self.combos = {}
        for r, (cid, name) in enumerate(sorted(names.items(), key=lambda kv: int(kv[0]))):
            self.table.setItem(r, 0, QTableWidgetItem(cid))
            self.table.setItem(r, 1, QTableWidgetItem(name))
            box = QComboBox()
            box.setEditable(True)
            box.addItems(["(ignore)"] + labels)
            box.setCurrentText(cmap.get(cid) or "(ignore)")
            box.setValidator(labelValidator())
            self.table.setCellWidget(r, 2, box)
            self.combos[cid] = box
        auto = newButton("✨ Auto-match", slot=lambda: self._auto())
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        info = QLabel("Map each model class to a dataset label (as shown in the viewer, i.e. after 🔀 label mapping). "
                      "'(ignore)' classes are skipped entirely.")
        info.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addWidget(info)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(auto)
        row.addStretch(1)
        row.addWidget(buttons)
        lay.addLayout(row)

    def _auto(self):
        for cid, label in auto_class_map(self.model.get("names") or {}, self.labels, self.map_label).items():
            self.combos[cid].setCurrentText(label or "(ignore)")

    def result_map(self) -> dict[str, str]:
        out = {}
        for cid, box in self.combos.items():
            t = box.currentText().strip()
            out[cid] = "" if t in ("", "(ignore)") else t
        return out


# =============================================================================
class ModelSetupDialog(QDialog):
    def __init__(self, app):
        super().__init__(app)
        self.app, self.svc, self.session = app, app.inference, app.session
        self.setWindowTitle("Models & Environment")
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(int(app.width() * 0.7), int(app.height() * 0.88))
        self.proc: QProcess | None = None
        self.steps: list = []
        self.models: list[dict] = list((load_models_config(self.session.root) if self.session else {}).get("models", []))
        self._busy_count = 0

        tabs = QTabWidget()
        tabs.addTab(self._build_env_tab(), "1 · Environment")
        tabs.addTab(self._build_models_tab(), "2 · Models")
        self.tabs = tabs

        # console ---------------------------------------------------------------
        self.console = QPlainTextEdit(readOnly=True)
        self.console.setMaximumBlockCount(5000)
        f = QFont("Consolas" if sys.platform == "win32" else "Monospace")
        f.setStyleHint(QFont.StyleHint.Monospace)
        f.setPointSizeF(9.5)
        self.console.setFont(f)
        self.console.setStyleSheet("QPlainTextEdit{background:#15171a; color:#d7dde4; border-radius:6px;}")
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setMaximumHeight(8)
        self.busy.setTextVisible(False)
        self.busy.setVisible(False)
        cbar = QHBoxLayout()
        self.cancel_btn = newButton("⏹ Cancel install", slot=lambda: self._cancel_install())
        self.cancel_btn.setVisible(False)
        clear = newButton("Clear", slot=self.console.clear)
        copy = newButton("Copy", slot=lambda: (self.console.selectAll(), self.console.copy()))
        cbar.addWidget(QLabel("Console"))
        cbar.addStretch(1)
        for b in (self.cancel_btn, copy, clear):
            cbar.addWidget(b)
        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addLayout(cbar)
        cl.addWidget(self.busy)
        cl.addWidget(self.console, 1)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(tabs)
        split.addWidget(cw)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)

        foot = QHBoxLayout()
        self.state = QLabel()
        close = newButton("Close", slot=self.close)
        foot.addWidget(self.state, 1)
        foot.addWidget(close)
        lay = QVBoxLayout(self)
        lay.addWidget(split, 1)
        lay.addLayout(foot)

        self.svc.log.connect(self._log)
        self.svc.changed.connect(self._refresh_state)
        self._refresh_env_status()
        self._refresh_models()
        self._refresh_state()

    # =========================================================== environment tab
    def _build_env_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        # --- existing env
        self.rb_existing = QRadioButton("Use an existing environment (ultralytics / sahi already installed)")
        self.rb_temp = QRadioButton("Create a temporary environment and install what is missing")
        grp = QButtonGroup(self)
        grp.addButton(self.rb_existing)
        grp.addButton(self.rb_temp)
        is_temp = bool(self.svc.temp_env) and self.svc.python.startswith(self.svc.temp_env)
        (self.rb_temp if is_temp else self.rb_existing).setChecked(True)

        ex = QGroupBox()
        g = QGridLayout(ex)
        self.env_box = QComboBox()
        self.env_box.setEditable(True)
        self.env_box.setMinimumWidth(420)
        self._fill_envs()
        rescan = newButton("↻", slot=lambda: self._fill_envs(), tip="Re-scan for conda / venv environments")
        browse = newButton("Browse…", slot=lambda: self._browse_env())
        check = newButton("🔍 Check environment", slot=lambda: self.check_env())
        g.addWidget(QLabel("Environment:"), 0, 0)
        g.addWidget(self.env_box, 0, 1)
        g.addWidget(rescan, 0, 2)
        g.addWidget(browse, 0, 3)
        g.addWidget(check, 0, 4)
        self.status_lbls = {}
        for r, (key, title) in enumerate((("python", "Python"), ("ultralytics", "ultralytics"), ("torch", "PyTorch"),
                                          ("cuda", "GPU"), ("sahi", "sahi")), start=1):
            g.addWidget(QLabel(title + ":"), r, 0)
            lbl = QLabel("–")
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            g.addWidget(lbl, r, 1, 1, 4)
            self.status_lbls[key] = lbl

        # --- temporary env
        tmp = QGroupBox()
        f = QFormLayout(tmp)
        self.driver = detect_cuda_driver()
        self.torch_box = QComboBox()
        for key, text in TORCH_CHOICES:
            self.torch_box.addItem(text, key)
        self.reuse_torch = QCheckBox("Reuse PyTorch from the environment above (much smaller download)")
        self.keep_cache = QCheckBox("Keep pip's download cache (faster next time; untick to leave nothing behind)")
        self.keep_cache.setChecked(True)
        self.delete_on_exit = QCheckBox("Delete the temporary environment when AIO QC closes")
        self.delete_on_exit.setChecked(self.svc.delete_temp_on_exit)
        self.delete_on_exit.toggled.connect(lambda v: self.svc.set_temp_env(self.svc.temp_env, v))
        drv = f"NVIDIA driver supports CUDA {self.driver} → auto = {pick_torch_build(self.driver)}" if self.driver \
            else "no NVIDIA GPU detected → auto = CPU build"
        btns = QHBoxLayout()
        self.install_btn = newButton("⬇ Install into a temporary environment", slot=lambda: self.start_install())
        self.remove_tmp_btn = newButton("🗑 Remove temporary environment now", slot=lambda: self.remove_temp())
        btns.addWidget(self.install_btn)
        btns.addWidget(self.remove_tmp_btn)
        btns.addStretch(1)
        self.tmp_info = QLabel()
        apply_style(self.tmp_info, "color:@muted;")
        f.addRow("PyTorch build:", self.torch_box)
        f.addRow("", QLabel(drv))
        f.addRow(self.reuse_torch)
        f.addRow(self.keep_cache)
        f.addRow(self.delete_on_exit)
        f.addRow(btns)
        f.addRow(self.tmp_info)

        # --- mode
        mode = QGroupBox("Inference mode")
        m = QHBoxLayout(mode)
        self.rb_std = QRadioButton("Standard (ultralytics)")
        self.rb_sahi = QRadioButton("SAHI sliced inference (sahi[ultralytics]) — small objects in large images")
        (self.rb_sahi if self.svc.mode == "sahi" else self.rb_std).setChecked(True)
        self.rb_sahi.toggled.connect(self._on_mode)
        m.addWidget(self.rb_std)
        m.addWidget(self.rb_sahi)
        m.addStretch(1)

        lay.addWidget(mode)
        lay.addWidget(self.rb_existing)
        lay.addWidget(ex)
        lay.addWidget(self.rb_temp)
        lay.addWidget(tmp)
        lay.addStretch(1)
        return w

    def _fill_envs(self):
        current = self.svc.python
        self.env_box.clear()
        seen = set()
        for name, py in discover_envs():
            self.env_box.addItem(f"{name}  —  {py}", py)
            seen.add(py)
        for py in self.app.settings.value("inference/recent_envs", []) or []:
            if py not in seen and Path(py).is_file():
                self.env_box.addItem(f"recent  —  {py}", py)
        if current:
            i = self.env_box.findData(current)
            if i < 0:
                self.env_box.addItem(f"current  —  {current}", current)
                i = self.env_box.count() - 1
            self.env_box.setCurrentIndex(i)

    def _browse_env(self):
        d = QFileDialog.getExistingDirectory(self, "Environment folder (conda env / venv root)")
        if not d:
            return
        py = python_in_env(d)
        if py is None:
            QMessageBox.warning(self, "Environment", "No python executable found in that folder.")
            return
        self.env_box.addItem(f"browsed  —  {py}", str(py))
        self.env_box.setCurrentIndex(self.env_box.count() - 1)

    def _selected_python(self) -> str:
        data = self.env_box.currentData()
        if data and self.env_box.currentText().endswith(str(data)):
            return str(data)
        text = self.env_box.currentText().split("—")[-1].strip()
        py = python_in_env(text) if text else None
        return str(py) if py else text

    def _on_mode(self, sahi: bool):
        self.svc.set_option("mode", "sahi" if sahi else "standard")
        self.sahi_box.setEnabled(sahi)
        self._refresh_env_status()
        self._refresh_state()

    # -------------------------------------------------------------- probing
    def check_env(self, python: str | None = None):
        py = python or self._selected_python()
        if not py or not Path(py).is_file():
            QMessageBox.warning(self, "Environment", "Pick a Python environment first.")
            return
        recent = [p for p in (self.app.settings.value("inference/recent_envs", []) or []) if p != py]
        self.app.settings.setValue("inference/recent_envs", [py] + recent[:7])
        self.svc.set_python(py)
        self._log(f"\n=== Checking {py} (importing torch can take a while) ===")
        self._busy(True)
        self.svc.run_async(self.svc.ensure_started, self._on_probed, self._on_probe_failed)

    def _on_probed(self, info):
        self._busy(False)
        self._refresh_env_status()
        self._fill_devices()
        miss = missing_packages(info, self.svc.mode)
        if miss:
            self._log(f"Missing for {self.svc.mode} mode: {', '.join(miss)}")
            self.rb_temp.setChecked(True)
            self.reuse_torch.setChecked(bool(info.get("torch")))
            QMessageBox.information(self, "Environment",
                                    f"This environment lacks: {', '.join(miss)}.\n\n"
                                    "Use 'Install into a temporary environment' — "
                                    + ("its PyTorch can be reused, so only the missing packages are downloaded."
                                       if info.get("torch") else "PyTorch will be installed too."))
        else:
            self._log("✅ Environment is ready.")

    def _on_probe_failed(self, err):
        self._busy(False)
        self._log(f"❌ {err}")
        self._refresh_env_status()

    def _refresh_env_status(self):
        info = self.svc.info
        L = self.status_lbls
        if not info:
            for k, lbl in L.items():
                lbl.setText("– (not checked)" if k == "python" else "–")
            return
        need_sahi = self.svc.mode == "sahi"
        L["python"].setText(f"{OK} {info['python']} · {info.get('executable', '')}")
        L["ultralytics"].setText(f"{OK} {info['ultralytics']}" if info.get("ultralytics") else f"{BAD} not installed")
        t = info.get("torch")
        L["torch"].setText(f"{OK} {t}" + (f" · torchvision {info.get('torchvision')}" if info.get("torchvision") else "")
                           if t else f"{BAD} not installed" + (f" ({info['torch_error']})" if info.get("torch_error") else ""))
        gpus = [d["name"] for d in info.get("devices", []) if d["id"] != "cpu"]
        L["cuda"].setText(f"{OK} " + " | ".join(gpus) + (f" · CUDA {info['cuda_version']}" if info.get("cuda_version") else "")
                          if gpus else f"{WARN} CPU only" + (" — the NVIDIA driver is present: a CUDA build of PyTorch "
                                                             "would be much faster" if self.driver else ""))
        s = info.get("sahi")
        L["sahi"].setText(f"{OK} {s}" if s else (f"{BAD} not installed (needed for SAHI mode)" if need_sahi
                                                  else "– not installed (only needed for SAHI mode)"))
        self.reuse_torch.setEnabled(bool(t))

    # --------------------------------------------------------- temporary env
    def start_install(self):
        if self.proc is not None:
            return
        mode = self.svc.mode
        info = self.svc.info
        reuse = self.reuse_torch.isChecked() and self.reuse_torch.isEnabled() and info and info.get("torch")
        base = info.get("executable") if reuse else None
        if reuse:
            packages = [p for p in required_packages(mode) if not info.get(p)] or required_packages(mode)
            torch_build = None
        else:
            packages = required_packages(mode)
            key = self.torch_box.currentData()
            torch_build = pick_torch_build(self.driver) if key == "auto" else key
        env_dir = new_temp_env_dir()
        self.steps = install_plan(env_dir, packages, torch_build, self.keep_cache.isChecked(), base, bool(reuse))
        size = "~150 MB" if reuse else ("~1 GB" if torch_build == "cpu" else "~2–4 GB")
        if QMessageBox.question(self, "Temporary environment",
                                f"Create {env_dir}\nand install: {', '.join((['torch'] if torch_build else []) + packages)}"
                                f"\n\nDownload size {size}. Continue?") != QMessageBox.StandardButton.Yes:
            return
        self._pending_env = env_dir
        self._step = 0
        self._t0 = time.monotonic()
        self._log(f"\n=== Temporary environment: {env_dir} ===")
        self.install_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self._busy(True)
        self.svc.stop()        # never install into an env the server is using
        self._run_next_step()

    def _run_next_step(self):
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
        proc.readyReadStandardOutput.connect(lambda p=proc: self._on_proc_output(p))
        proc.finished.connect(self._on_step_finished)
        proc.errorOccurred.connect(lambda e, p=proc: self._log(f"process error: {e.name if hasattr(e, 'name') else e}"))
        self.proc = proc
        proc.start(argv[0], argv[1:])

    def _on_proc_output(self, proc):
        text = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.replace("\r\n", "\n").split("\n"):
            if line.strip():
                self._log(line.split("\r")[-1])            # keep the last state of \r progress lines

    def _on_step_finished(self, code, _status):
        self.proc = None
        if code != 0:
            self._log(f"❌ step failed (exit code {code})")
            return self._install_finished(False)
        self._step += 1
        self._run_next_step()

    def _cancel_install(self):
        if self.proc is not None:
            self._log("⏹ cancelling…")
            self.proc.kill()

    def _install_finished(self, ok):
        self._busy(False)
        self.install_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        env_dir = getattr(self, "_pending_env", None)
        mins = (time.monotonic() - self._t0) / 60
        if not ok:
            self._log(f"Installation stopped after {mins:.1f} min. Removing the partial environment…")
            if env_dir is not None:
                remove_env(env_dir)
            return
        # the previous temporary env (if any) is replaced
        if self.svc.temp_env and self.svc.temp_env != str(env_dir):
            remove_env(Path(self.svc.temp_env))
        self.svc.set_temp_env(str(env_dir), self.delete_on_exit.isChecked())
        self._log(f"✅ Installed in {mins:.1f} min ({dir_size_mb(env_dir):,.0f} MB). "
                  + ("It will be deleted when AIO QC closes." if self.delete_on_exit.isChecked() else ""))
        self.check_env(str(env_python(env_dir)))
        self._refresh_state()

    def remove_temp(self):
        if not self.svc.temp_env:
            return
        path = Path(self.svc.temp_env)
        if QMessageBox.question(self, "Remove", f"Delete {path}?") != QMessageBox.StandardButton.Yes:
            return
        if self.svc.python.startswith(str(path)):
            self.svc.set_python("")
        self.svc.stop()
        ok = remove_env(path)
        self._log(f"{'✅ removed' if ok else '❌ could not remove'} {path}")
        if ok:
            self.svc.set_temp_env("", self.delete_on_exit.isChecked())
        self._refresh_state()

    # ================================================================ models tab
    def _build_models_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        # shared inference options
        opts = QGroupBox("Inference options (shared by all models)")
        o = QHBoxLayout(opts)
        # FP16 first: filling the device list calls _on_device(), which enables / disables it
        self.half = QCheckBox("FP16")
        self.half.setToolTip("Half precision on CUDA: ~2x faster, same results for QC purposes")
        self.half.setChecked(bool(self.svc.opts["half"]))
        self.half.toggled.connect(lambda v: self.svc.set_option("half", v))
        self.device = QComboBox()
        self.device.setMinimumWidth(260)
        self._fill_devices()
        self.device.currentIndexChanged.connect(lambda _i: self._on_device())
        self.batch = ispin(1, 256, int(self.svc.opts["batch"]))
        self.batch.valueChanged.connect(lambda v: self.svc.set_option("batch", v))
        self.nms = dspin(0.3, 0.95, float(self.svc.opts["nms_iou"]), 2, 0.05)
        self.nms.setToolTip("Boxes of the same label from different models are merged above this IoU")
        self.nms.valueChanged.connect(lambda v: self.svc.set_option("nms_iou", v))
        self.sahi_box = QWidget()
        sb = QHBoxLayout(self.sahi_box)
        sb.setContentsMargins(0, 0, 0, 0)
        self.slice = ispin(128, 4096, int(self.svc.opts["slice"]))
        self.slice.valueChanged.connect(lambda v: self.svc.set_option("slice", v))
        self.overlap = dspin(0.0, 0.8, float(self.svc.opts["overlap"]), 2, 0.05)
        self.overlap.valueChanged.connect(lambda v: self.svc.set_option("overlap", v))
        for t, wd in (("SAHI slice", self.slice), ("overlap", self.overlap)):
            sb.addWidget(QLabel(t))
            sb.addWidget(wd)
        self.sahi_box.setEnabled(self.svc.mode == "sahi")
        for t, wd in (("Device", self.device), ("", self.half), ("Batch", self.batch), ("Cross-model NMS IoU", self.nms)):
            if t:
                o.addWidget(QLabel(t))
            o.addWidget(wd)
        o.addWidget(self.sahi_box)
        o.addStretch(1)

        # models table
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Use", "Name", "Task", "Classes → labels", "imgsz", "Trained on",
                                              "Runtime", "Status", "File"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeMode.Stretch)
        readable_table(self.table)
        self.table.cellDoubleClicked.connect(lambda r, c: self.edit_class_map(r))
        btns = QHBoxLayout()
        for text, slot, tip in (
            ("➕ Add models…", self.add_models, "Add one or more .pt / .onnx / .engine files"),
            ("➖ Remove", self.remove_model, ""),
            ("🏷 Class mapping…", lambda: self.edit_class_map(self.table.currentRow()),
             "Map the selected model's classes to dataset labels (or double-click a row)"),
            ("⟳ Load / reload models", self.load_models, "Start the environment and load every enabled model"),
            ("⚡ Export ▸ ONNX", lambda: self.export_model("onnx"),
             "Export the selected model to ONNX (portable; fast on CPU / onnxruntime-gpu)"),
            ("⚡ Export ▸ TensorRT", lambda: self.export_model("engine"),
             "Build a TensorRT engine for the selected GPU (fastest on NVIDIA; tied to this GPU + TensorRT)"),
            ("🎯 Test on current image", self.test_current, "Run every model on the image shown in the viewer"),
        ):
            b = newButton(text, slot=slot, tip=tip)
            btns.addWidget(b)
        btns.addStretch(1)
        self.coverage = QLabel()
        self.coverage.setWordWrap(True)
        self.coverage.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(opts)
        lay.addLayout(btns)
        lay.addWidget(self.table, 1)
        lay.addWidget(QLabel("<b>Label coverage</b> — which model can confirm each dataset label"))
        lay.addWidget(self.coverage)
        return w

    def _fill_devices(self):
        if not hasattr(self, "device"):
            return
        want = str(self.svc.opts["device"])
        self.device.blockSignals(True)
        self.device.clear()
        devices = (self.svc.info or {}).get("devices") or [{"id": "cpu", "name": "CPU"}]
        for d in devices:
            self.device.addItem(d["name"], d["id"])
        i = self.device.findData(want)
        if i < 0:   # best available by default: first GPU
            i = 1 if self.device.count() > 1 else 0
        self.device.setCurrentIndex(i)
        self.device.blockSignals(False)
        self._on_device()

    def _on_device(self):
        dev = self.device.currentData() or "cpu"
        self.svc.set_option("device", dev)
        gpu = dev not in ("cpu", "mps")
        self.half.setEnabled(gpu)
        if not gpu:
            self.half.setChecked(False)

    def _labels(self) -> list[str]:
        if not self.session:
            return []
        labels = [l for l, _ in self.session.label_totals().most_common()]
        return labels or list(self.session.adapter.class_names)

    def add_models(self):
        start = str(Path(self.models[-1]["path"]).parent) if self.models else str(self.session.root if self.session else "")
        paths, _ = QFileDialog.getOpenFileNames(self, "Add models", start,
                                                "Models (*.pt *.onnx *.engine *.torchscript *.mlpackage);;All files (*)")
        for p in paths:
            if any(m["path"] == p for m in self.models):
                continue
            self.models.append({"id": uuid.uuid4().hex[:8], "name": Path(p).stem, "path": p, "enabled": True,
                                "imgsz": 0, "class_map": {}, "names": {}, "task": "", "train_imgsz": None})
        self._save()
        self._refresh_models()
        if paths:
            self._log(f"Added {len(paths)} model(s). Press ⟳ Load to read their classes.")

    def remove_model(self):
        r = self.table.currentRow()
        if 0 <= r < len(self.models):
            del self.models[r]
            self._save()
            self._refresh_models()

    def edit_class_map(self, row):
        if not 0 <= row < len(self.models):
            return
        m = self.models[row]
        if not m.get("names"):
            QMessageBox.information(self, "Class mapping", "Load the model first (⟳ Load) so its classes are known.")
            return
        dlg = ClassMapDialog(self, m, self._labels(), self.session.adapter.map_label if self.session else (lambda x: x))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            m["class_map"] = dlg.result_map()
            self._save()
            self._refresh_models()

    def _refresh_models(self):
        try:
            self.table.itemChanged.disconnect(self._on_item_changed)
        except TypeError:
            pass
        self.table.setRowCount(len(self.models))
        loaded = self.svc.loaded
        for r, m in enumerate(self.models):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            chk.setCheckState(Qt.CheckState.Checked if m.get("enabled", True) else Qt.CheckState.Unchecked)
            self.table.setItem(r, 0, chk)
            name = QTableWidgetItem(m["name"])
            name.setForeground(QColor(model_color(r)))
            f = name.font()
            f.setBold(True)
            name.setFont(f)
            self.table.setItem(r, 1, name)
            self.table.setItem(r, 2, QTableWidgetItem(m.get("task") or "?"))
            names, cmap = m.get("names") or {}, m.get("class_map") or {}
            used = [f"{names[c]}→{l}" if names.get(c) != l else l for c, l in cmap.items() if l and c in names]
            self.table.setItem(r, 3, QTableWidgetItem(f"{len(used)}/{len(names)}: " + ", ".join(used[:6])
                                                      + ("…" if len(used) > 6 else "") if names else "(load to read)"))
            box = QComboBox()
            for v in IMGSZ_PRESETS:
                box.addItem(f"auto ({m.get('train_imgsz') or 640})" if v == 0 else str(v), v)
            box.setEditable(True)
            i = box.findData(int(m.get("imgsz") or 0))
            box.setCurrentIndex(i) if i >= 0 else box.setCurrentText(str(m.get("imgsz")))
            box.currentTextChanged.connect(lambda t, mm=m: self._set_imgsz(mm, t))
            self.table.setCellWidget(r, 4, box)
            tr = QComboBox()
            for key, text in TRAINED_ON:
                tr.addItem(text, key)
            t_on = m.get("trained_on") or {"mode": "all"}
            tr.setCurrentIndex(max(0, tr.findData(t_on.get("mode", "all"))))
            if t_on.get("mode") == "list":
                tr.setItemText(tr.currentIndex(), f"list: {Path(t_on.get('path', '')).name}")
                tr.setToolTip(t_on.get("path", ""))
            tr.activated.connect(lambda _i, mm=m, cb=tr: self._set_trained_on(mm, cb))
            self.table.setCellWidget(r, 5, tr)
            rt = QComboBox()
            for key, text in RUNTIMES:
                have = key == "pt" or key in (m.get("exports") or {})
                rt.addItem(text + ("" if have else " (not exported)"), key)
            rt.setCurrentIndex(max(0, rt.findData(m.get("runtime", "pt"))))
            rt.activated.connect(lambda _i, mm=m, cb=rt: self._set_runtime(mm, cb))
            self.table.setCellWidget(r, 6, rt)
            li = loaded.get(m["id"])
            _path, note = runtime_path(m, self.svc.opts, self.svc.info)
            if not Path(m["path"]).is_file():
                st = f"{BAD} file missing"
            elif li is None:
                st = "not loaded"
            elif li.get("ok"):
                st = f"{OK} loaded" + (f" ({li.get('load_s')}s)" if li.get("load_s") is not None else "")
            else:
                st = f"{BAD} {li.get('error', '')[:60]}"
            if note:
                st += f" · {WARN} {note}"
            self.table.setItem(r, 7, QTableWidgetItem(st))
            self.table.setItem(r, 8, QTableWidgetItem(m["path"]))
        self.table.itemChanged.connect(self._on_item_changed)
        self._refresh_coverage()

    def _on_item_changed(self, item):
        if item.column() == 0 and item.row() < len(self.models):
            self.models[item.row()]["enabled"] = item.checkState() == Qt.CheckState.Checked
            self._save()
            self._refresh_coverage()

    def _set_trained_on(self, m, combo):
        mode = combo.currentData()
        if mode == "list":
            path, _ = QFileDialog.getOpenFileName(self, "Training image list (one path per line, e.g. train.txt)",
                                                  str(self.session.root if self.session else ""),
                                                  "Lists (*.txt);;All files (*)")
            if not path:
                self._refresh_models()
                return
            m["trained_on"] = {"mode": "list", "path": path}
        else:
            m["trained_on"] = {"mode": mode}
        self._save()
        self._refresh_models()

    def _set_runtime(self, m, combo):
        m["runtime"] = combo.currentData()
        self._save()
        self._refresh_models()

    def export_model(self, fmt: str):
        r = self.table.currentRow()
        if not 0 <= r < len(self.models):
            QMessageBox.information(self, "Export", "Select a model row first.")
            return
        m = self.models[r]
        if not self.svc.python or not self.svc.info:
            self.tabs.setCurrentIndex(0)
            QMessageBox.information(self, "Export", "Check an environment first (tab 1).")
            return
        dev = str(self.svc.opts.get("device") or "cpu")
        if fmt == "engine" and dev in ("cpu", "mps"):
            QMessageBox.information(self, "Export", "TensorRT needs a CUDA device — choose one in the Device list.")
            return
        spec = {"path": m["path"], "format": fmt, "imgsz": effective_imgsz(m),
                "half": bool(self.svc.opts.get("half")) and dev not in ("cpu", "mps"),
                "device": dev, "batch": int(self.svc.opts.get("batch", 8))}
        src = Path(m["path"])
        before = {p.name for p in src.parent.glob(src.stem + ".*")}
        self._log(f"\n=== Export {m['name']} → {fmt} (imgsz {spec['imgsz']}, half {spec['half']}) — "
                  "ultralytics installs missing export packages itself; this can take several minutes ===")
        self._busy(True)
        info = dict(self.svc.info)

        def run():
            self.svc.ensure_started()
            with self.svc._lock:
                return self.svc.client.export(spec)

        def done(res):
            self._busy(False)
            out = Path(res["path"])
            dest_dir = (self.session.root / ".aio_qc" / "exports") if self.session else out.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{m['id']}_{spec['imgsz']}{out.suffix}"
            try:
                import shutil
                if dest.exists():
                    dest.unlink()
                shutil.move(str(out), str(dest))
                for p in src.parent.glob(src.stem + ".*"):          # intermediates this export created
                    if p.name not in before and p != out and p.suffix in (".onnx", ".cache"):
                        p.unlink(missing_ok=True)
            except OSError as exc:
                self._log(f"{WARN} could not move the export ({exc}); keeping {out}")
                dest = out
            m.setdefault("exports", {})[fmt] = {
                "path": str(dest), "src_sig": src_sig(m["path"]), "imgsz": spec["imgsz"], "half": spec["half"],
                "device_name": device_name(info, dev) if fmt == "engine" else "",
                "trt": (self.svc.info or {}).get("tensorrt") if fmt == "engine" else ""}
            m["runtime"] = fmt
            self._save()
            self._refresh_models()
            self._log(f"✅ {fmt} ready in {res.get('seconds')}s → {dest} (now used for {m['name']})")

        self.svc.run_async(run, done, self._failed)

    def _set_imgsz(self, m, text):
        t = text.split("(")[0].strip()
        m["imgsz"] = 0 if t.startswith("auto") or not t else int(t) if t.isdigit() else m.get("imgsz", 0)
        self._save()

    def _refresh_coverage(self):
        maps = {m["id"]: m.get("class_map") or {} for m in self.models if m.get("enabled", True)}
        cov = coverage(maps)
        names = {m["id"]: m["name"] for m in self.models}
        parts = []
        for label in self._labels():
            ms = cov.get(label)
            if ms:
                parts.append(f"<span style='color:{hexc('ok')}'>✔ <b>{label}</b> ← {', '.join(names[i] for i in ms)}</span>")
            else:
                parts.append(f"<span style='color:{hexc('warn')}'>✖ <b>{label}</b> (no model — not checked)</span>")
        extra = sorted({l for l in cov if l not in set(self._labels())})
        if extra:
            parts.append(f"<span style='color:{hexc('muted')}'>model-only labels: {', '.join(extra)}</span>")
        self.coverage.setText(" &nbsp;·&nbsp; ".join(parts) if parts else "Open a dataset to see coverage.")

    def _save(self):
        if self.session:
            save_models_config(self.session.root, {"version": 1, "models": self.models})

    # ------------------------------------------------------------ load & test
    def load_models(self, then=None):
        specs = server_specs(self.models, self.svc.opts, self.svc.info)
        if not specs:
            QMessageBox.information(self, "Models", "Add and enable at least one model file.")
            return
        if not self.svc.python:
            self.tabs.setCurrentIndex(0)
            QMessageBox.information(self, "Models", "Choose and check an environment first (tab 1).")
            return
        self._busy(True)
        t0 = time.monotonic()

        def done(loaded):
            self._busy(False)
            labels = self._labels()
            for m in self.models:
                li = loaded.get(m["id"])
                if li and li.get("ok"):
                    m["names"], m["task"], m["train_imgsz"] = li["names"], li.get("task"), li.get("train_imgsz")
                    if not m.get("class_map") or set(m["class_map"]) != set(m["names"]):
                        m["class_map"] = auto_class_map(m["names"], labels,
                                                        self.session.adapter.map_label if self.session else (lambda x: x))
            self._save()
            self._refresh_models()
            n_ok = sum(1 for v in loaded.values() if v.get("ok"))
            self._log(f"✅ {n_ok}/{len(loaded)} model(s) ready in {time.monotonic() - t0:.1f}s")
            if then:
                then()

        self.svc.run_async(lambda: self.svc.ensure_loaded(specs), done, self._failed)

    def test_current(self):
        key = self.app.current_key
        if not key or self.app.current_ann is None:
            QMessageBox.information(self, "Test", "Open an image in the viewer first.")
            return
        s = self.session.by_key[key]
        img = self.app.current_ann.image_path or s.image_path
        specs = server_specs(self.models, self.svc.opts, self.svc.info)
        if not specs or img is None:
            return
        ids = [sp["id"] for sp in specs]
        colors = {m["id"]: model_color(i) for i, m in enumerate(self.models)}
        names = {m["id"]: m["name"] for m in self.models}
        maps = {m["id"]: m.get("class_map") or {} for m in self.models}
        conf = 0.25

        def run():
            self.svc.ensure_loaded(specs)
            t0 = time.monotonic()
            res = self.svc.predict([img], ids, conf=conf)
            return res, time.monotonic() - t0

        def done(out):
            res, dt = out
            self._busy(False)
            r = res.get(str(img)) or next(iter(res.values()))
            if r.get("error"):
                self._log(f"❌ {r['error']}")
            overlay, counts = [], []
            for mid in ids:
                boxes = r["by_model"].get(mid) or []
                shown = 0
                for x1, y1, x2, y2, c, k in boxes:
                    label = maps[mid].get(str(int(k)), "") or f"({self._model_name(mid, k)})"
                    overlay.append({"box": (x1, y1, x2, y2), "text": f"{label} {c:.2f} · {names[mid]}",
                                    "color": colors[mid]})
                    shown += 1
                counts.append(f"{names[mid]}: {shown}")
            merged = merge_predictions(r["by_model"], maps, float(self.svc.opts["nms_iou"]))
            self.app.canvas.set_overlay(overlay)
            self._log(f"🎯 {Path(str(img)).name}: {', '.join(counts)} (conf ≥ {conf}) → {len(merged)} after "
                      f"mapping + cross-model NMS · {dt * 1000:.0f} ms")
            self.app.statusBar().showMessage("Model boxes shown as dashed overlay (cleared on next image)", 6000)

        self._busy(True)
        self.svc.run_async(run, done, self._failed)

    def _model_name(self, mid, k):
        for m in self.models:
            if m["id"] == mid:
                return (m.get("names") or {}).get(str(int(k)), f"class {k}")
        return f"class {k}"

    # ------------------------------------------------------------------ misc
    def _failed(self, err):
        self._busy(False)
        self._log(f"❌ {err}")
        self._refresh_models()

    def _busy(self, on: bool):
        self._busy_count = max(0, self._busy_count + (1 if on else -1))
        self.busy.setVisible(self._busy_count > 0)

    def _log(self, text: str):
        self.console.appendPlainText(text)
        self.console.moveCursor(QTextCursor.MoveOperation.End)

    def _refresh_state(self):
        svc = self.svc
        n = sum(1 for m in self.models if m.get("enabled", True))
        env = Path(svc.python).parent.parent.name if svc.python else "—"
        tmp = " (temporary)" if svc.temp_env and svc.python.startswith(svc.temp_env) else ""
        ready = "✅ ready" if svc.ready else ("⚠ missing: " + ", ".join(svc.missing()) if svc.info else "⚪ not checked")
        self.state.setText(f"Env: <b>{env}</b>{tmp} · {ready} · mode: <b>{svc.mode}</b> · device: "
                           f"<b>{svc.opts['device']}</b> · models: <b>{n}</b>")
        if svc.temp_env and Path(svc.temp_env).exists():
            self.tmp_info.setText(f"Temporary env: {svc.temp_env}"
                                  + (" · deleted on exit" if svc.delete_temp_on_exit else " · kept"))
            self.remove_tmp_btn.setEnabled(True)
        else:
            self.tmp_info.setText("No temporary environment.")
            self.remove_tmp_btn.setEnabled(False)

    def enabled_models(self) -> list[dict]:
        return [m for m in self.models if m.get("enabled", True)]

    def closeEvent(self, event):
        if self.proc is not None:
            if QMessageBox.question(self, "Install running", "Cancel the running installation?") \
                    != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.proc.kill()
        self._save()
        for sig, slot in ((self.svc.log, self._log), (self.svc.changed, self._refresh_state)):
            try:
                sig.disconnect(slot)
            except TypeError:
                pass
        self.app.forget_dialog(self)
        super().closeEvent(event)
