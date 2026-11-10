# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
🔧 Fix in Editor — open flagged files in labelme (primary) or labelImg (secondary, YOLO boxes) and sync every
save back into the dataset (see core.edit_session for the safety rules).

Page 1  Setup    files · editor · version (+ Add source zip…) · image data · environment status
Page 2  Session  install progress → live table of files (waiting / fixed / unchanged / rejected / conflict)
                 · editor log · Reopen editor · Sync now · Undo session · Finish
"""
from __future__ import annotations

from AIO_QC.core.dataset import DETECTION_TYPES, DatasetType

import time
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QEasingCurve, QProcess, QRect, QSettings, Qt, QTimer
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QMenu,
    QMessageBox, QPlainTextEdit, QProgressBar, QRadioButton, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from AIO_QC.core.edit_session import EditSession, Rejected
from AIO_QC.core.editors import (
    LABELIMG, LABELME, MIN_PYTHON, TITLES, build_command, env_dir, find_base_python, import_source_zip,
    install_steps, is_installed, list_sources, mark_installed, python_version, remove_editor_env,
)
from AIO_QC.core.state import REVIEW_FIXED, REVIEW_FLAG
from AIO_QC.utils import hexc, newButton, newIcon, style as apply_style, theme_color
from .inference_service import FuncWorker
from .process_runner import StepRunner, clean_process_env
from .mini_session import FrameAnimator, Ghost, MiniSessionWindow, lerp_rect

STATUS_UI = {  # status -> (text, theme role, icon)
    "waiting": ("waiting", "muted", "review-next"),
    "fixed": ("fixed", "info", "review-fixed"),
    "unchanged": ("unchanged", "muted", "review-ok"),
    "rejected": ("needs attention", "error", "review-flag"),
    "conflict": ("conflict", "warn", "review-flag"),
    "error": ("error", "error", "review-clear"),
}
SIZE_HINT = {LABELME: "about 450 MB (PyQt5, onnxruntime, scikit-image, matplotlib …)",
             LABELIMG: "about 80 MB (PyQt5, lxml)"}
POLL_MS = 700


class EditorSessionDialog(QDialog):
    TITLE, ICON = "Fix in Editor", "🔧"
    SUPPORTED = DETECTION_TYPES | {DatasetType.CLS_LABELME}     # class folders have nothing to edit

    def __init__(self, app, keys: list[str] | None = None, resume: Path | None = None):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.settings: QSettings = app.settings
        self.es: EditSession | None = None
        self.mini: MiniSessionWindow | None = None          # picture-in-picture window
        self._normal_geo: QRect | None = None               # last normal (not minimised) geometry
        self._last_snapshot = None
        self.proc: QProcess | None = None
        self.runner: StepRunner | None = None
        self.worker = None
        self.source = None
        self.python: str | None = None
        self._explicit_keys = keys
        self._rows: dict[str, int] = {}
        self._stray_warned = False
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(newIcon("tool-editor"))
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinimizeButtonHint |
                            Qt.WindowType.WindowMaximizeButtonHint)       # minimise = in-app mini window
        self.resize(int(app.width() * 0.62), int(app.height() * 0.72))

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_setup())
        self.stack.addWidget(self._build_session())
        lay = QVBoxLayout(self)
        lay.addWidget(self.stack)

        self.timer = QTimer(self, interval=POLL_MS, timeout=self._sync_tick)
        if resume is not None:
            self._resume(resume)
        else:
            self._refresh_setup()

    # =====================================================================================
    # Setup page
    # =====================================================================================
    def _build_setup(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        intro = QLabel("Open files in an annotation editor. Every save in the editor is checked and written back to "
                       "the dataset automatically, the file is marked <b>Fixed</b>, and the originals are backed up "
                       "(Undo restores them).")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QFormLayout()
        self.scope = QComboBox()
        self.scope.currentIndexChanged.connect(lambda _i: self._update_start())
        form.addRow("Files:", self.scope)

        ed = QWidget()
        el = QHBoxLayout(ed)
        el.setContentsMargins(0, 0, 0, 0)
        self.rb_labelme = QRadioButton("labelme  (recommended — all shape types, both dataset types)")
        self.rb_labelimg = QRadioButton("labelImg  (YOLO boxes only)")
        self.editor_group = QButtonGroup(self)
        for i, rb in enumerate((self.rb_labelme, self.rb_labelimg)):
            self.editor_group.addButton(rb, i)
            el.addWidget(rb)
        el.addStretch(1)
        self.rb_labelme.setChecked(True)
        self.editor_group.idToggled.connect(lambda _i, on: on and self._refresh_setup())
        form.addRow("Editor:", ed)

        vr = QWidget()
        vl = QHBoxLayout(vr)
        vl.setContentsMargins(0, 0, 0, 0)
        self.version = QComboBox()
        self.version.currentIndexChanged.connect(lambda _i: self._on_version())
        vl.addWidget(self.version, 1)
        vl.addWidget(newButton("Add source zip…", icon="open", slot=self._add_source,
                               tip="Add a labelme / labelImg version from its source zip (e.g. labelme-6_4_0.zip)"))
        form.addRow("Version:", vr)

        self.image_data = QCheckBox("Save image data in the JSON (imageData)")
        self.image_data.setChecked(False)
        self.image_data.setToolTip(
            "Unticked (default): every JSON you change is written with \"imageData\": null — small files, the image "
            "is read from imagePath.\nTicked: the JSON embeds the image (base64) — a JSON that already embeds it keeps "
            "its image, others embed the image file the way the selected labelme version does.\nSame key in labelme 5 "
            "and 6; only files you change are rewritten. The editor itself always works without image data.")
        self.image_data.toggled.connect(lambda _on: self._image_note())
        form.addRow("Image data:", self.image_data)
        self.image_note = QLabel("")
        self.image_note.setWordWrap(True)
        apply_style(self.image_note, "color:@muted;")
        form.addRow("", self.image_note)

        er = QWidget()
        erl = QHBoxLayout(er)
        erl.setContentsMargins(0, 0, 0, 0)
        self.env_label = QLabel("")
        self.env_label.setWordWrap(True)
        erl.addWidget(self.env_label, 1)
        erl.addWidget(newButton("Use existing Python…", icon="open", slot=self._choose_python,
                                tip="Use a Python environment that already has the editor's packages (no download)"))
        self.remove_env_btn = newButton("Remove environment", icon="delete", slot=self._remove_env,
                                        tip="Delete this editor environment (it is re-created when needed)")
        erl.addWidget(self.remove_env_btn)
        form.addRow("Environment:", er)
        lay.addLayout(form)
        lay.addStretch(1)
        foot = QHBoxLayout()
        foot.addStretch(1)
        self.start_btn = newButton("Open in editor", icon="tool-editor", slot=self._start)
        self.start_btn.setDefault(True)
        foot.addWidget(self.start_btn)
        foot.addWidget(newButton("Close", slot=self.close))
        lay.addLayout(foot)
        return w

    def _scope_options(self) -> list[tuple[str, list[str]]]:
        s = self.session
        dock = self.app.file_dock
        flagged = s.keys_with_review({REVIEW_FLAG}, rank=dock.model.row_of)
        filtering = dock.proxy.filtering()
        listed = dock.visible_keys() if filtering else None
        if filtering:
            shown = set(listed)
            flagged_listed = [k for k in flagged if k in shown]
        opts = []
        if self._explicit_keys:
            opts.append((f"Selected files ({len(self._explicit_keys):,})", list(self._explicit_keys)))
        opts.append((f"All flagged files ({len(flagged):,})", flagged))
        if filtering and len(listed) != len(s.samples):
            opts.append((f"Flagged files in the current list ({len(flagged_listed):,})", flagged_listed))
            opts.append((f"All files in the current list ({len(listed):,})", listed))
        if self.app.current_key:
            opts.append(("Current file", [self.app.current_key]))
        return opts

    def _refresh_setup(self):
        is_yolo = self.session.is_yolo
        self.rb_labelimg.setEnabled(is_yolo)
        if not is_yolo:
            self.rb_labelimg.setToolTip("labelImg edits YOLO boxes only — use labelme for LabelMe datasets")
            self.rb_labelme.setChecked(True)
        cur = self.scope.currentText()
        self.scope.blockSignals(True)
        self.scope.clear()
        for text, keys in self._scope_options():
            self.scope.addItem(text, keys)
        i = self.scope.findText(cur)
        self.scope.setCurrentIndex(i if i >= 0 else 0)
        self.scope.blockSignals(False)
        kind = self.kind
        self.version.blockSignals(True)
        self.version.clear()
        for src in list_sources(kind):
            self.version.addItem(f"{src.title}" + ("" if src.origin == "bundled" else "  (added)"), src)
        self.version.blockSignals(False)
        self._on_version()

    @property
    def kind(self) -> str:
        return LABELIMG if self.rb_labelimg.isChecked() else LABELME

    def _on_version(self):
        self.source = self.version.currentData()
        show_img = self.kind == LABELME and not self.session.is_yolo
        self.image_data.setVisible(show_img)
        self.image_note.setVisible(show_img)
        if show_img and self.source is not None:
            self._image_note()
        self._refresh_env()
        self._update_start()

    def _image_note(self):
        if self.source is None:
            return
        if self.image_data.isChecked():
            text = (f"Changed files embed the image (as labelme {self.source.version} would) — "
                    "self-contained JSONs, but each grows by roughly the image size.")
        else:
            text = ("Changed files are saved with \"imageData\": null (works in labelme 5 and 6); imagePath is kept. "
                    "JSONs whose only copy of the image is embedded keep it.")
        self.image_note.setText(text)

    def _update_start(self):
        keys = self.scope.currentData() or []
        ok = bool(keys) and self.source is not None
        self.start_btn.setEnabled(ok)
        self.start_btn.setText(f"Open {len(keys):,} file(s) in {TITLES[self.kind]}" if keys else "No files to open")

    # ------------------------------------------------------------------ environment
    def _custom_python(self) -> str | None:
        py = self.settings.value(f"editors/{self.kind}/python", "") or ""
        return py if py and Path(py).is_file() else None

    def _refresh_env(self):
        if self.source is None:
            self.env_label.setText(f"No {TITLES[self.kind]} source available — add one with “Add source zip…”.")
            self.remove_env_btn.setEnabled(False)
            return
        custom = self._custom_python()
        verified = self.source.deps_hash() in (self.settings.value(f"editors/{self.kind}/verified", []) or [])
        if custom and verified:
            self.env_label.setText(f"✓ Using {custom}")
            self.remove_env_btn.setEnabled(False)
            return
        base = self.settings.value("editors/base_python", "") or ""
        pyver = python_version(base) if base and Path(base).is_file() else None
        if pyver and pyver >= MIN_PYTHON[self.kind]:
            env = env_dir(self.kind, pyver)
            if is_installed(env, self.source):
                self.env_label.setText(f"✓ Ready — {env}")
                self.remove_env_btn.setEnabled(True)
                return
            self.env_label.setText(f"Not installed yet — {SIZE_HINT[self.kind]}, downloaded once and reused.\n{env}")
            self.remove_env_btn.setEnabled(env.exists())
            return
        self.env_label.setText(f"Not installed yet — {SIZE_HINT[self.kind]}, downloaded once and reused. "
                               f"Needs Python ≥ {'.'.join(map(str, MIN_PYTHON[self.kind]))}.")
        self.remove_env_btn.setEnabled(False)

    def _choose_python(self):
        path, _ = QFileDialog.getOpenFileName(self, "Python executable with the editor's packages", "",
                                              "Python (python*);;All files (*)")
        if not path:
            return
        v = python_version(path)
        if not v or v < MIN_PYTHON[self.kind]:
            QMessageBox.warning(self, self.TITLE, f"{TITLES[self.kind]} needs Python ≥ "
                                                  f"{'.'.join(map(str, MIN_PYTHON[self.kind]))} (this is {v}).")
            return
        from AIO_QC.core.editors import CHECK_SNIPPET
        import subprocess
        r = subprocess.run([path, "-c", CHECK_SNIPPET[self.kind]], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            QMessageBox.warning(self, self.TITLE, "That environment is missing packages the editor needs:\n\n" +
                                (r.stderr.strip().splitlines() or ["unknown error"])[-1] +
                                "\n\nAIO QC will create its own environment instead.")
            return
        self.settings.setValue(f"editors/{self.kind}/python", path)
        ver = set(self.settings.value(f"editors/{self.kind}/verified", []) or [])
        ver.add(self.source.deps_hash())
        self.settings.setValue(f"editors/{self.kind}/verified", sorted(ver))
        self._refresh_env()

    def _remove_env(self):
        base = self.settings.value("editors/base_python", "") or ""
        pyver = python_version(base) if base else None
        if not pyver:
            return
        env = env_dir(self.kind, pyver)
        if QMessageBox.question(self, self.TITLE, f"Delete the {TITLES[self.kind]} environment?\n{env}") \
                == QMessageBox.StandardButton.Yes:
            if not remove_editor_env(env):
                QMessageBox.warning(self, self.TITLE, "Could not delete it (is the editor still open?).")
            self._refresh_env()

    def _add_source(self):
        path, _ = QFileDialog.getOpenFileName(self, "labelme / labelImg source zip", "", "Zip archives (*.zip)")
        if not path:
            return
        self._busy(True, "Importing the source …")

        def done(src):
            self._busy(False)
            self.rb_labelimg.setChecked(src.kind == LABELIMG) if self.session.is_yolo else None
            self._refresh_setup()
            i = next((i for i in range(self.version.count()) if self.version.itemData(i).version == src.version), 0)
            self.version.setCurrentIndex(i)
            extra = f"\n\nPatched for modern Python:\n• " + "\n• ".join(src.patches) if src.patches else ""
            QMessageBox.information(self, self.TITLE, f"Added {src.title}.{extra}")

        def failed(msg):
            self._busy(False)
            QMessageBox.warning(self, self.TITLE, f"Could not add this source:\n{msg}")

        self._run(lambda: import_source_zip(path), done, failed)

    def _busy(self, on: bool, text: str = ""):
        self.start_btn.setEnabled(not on)
        if on:
            self.env_label.setText(text)
        else:
            self._refresh_env()
            self._update_start()

    def _run(self, fn, done, failed):
        w = FuncWorker(fn, self)
        w.done.connect(done)
        w.failed.connect(failed)
        w.finished.connect(w.deleteLater)
        self.worker = w
        w.start()

    # =====================================================================================
    # Start: environment -> session -> editor
    # =====================================================================================
    def _start(self):
        keys = [k for k in (self.scope.currentData() or []) if k in self.session.by_key]
        busy = [k for k in keys if k in self.session.locked]
        if busy:
            QMessageBox.information(self, self.TITLE, f"{len(busy)} of these files are already open in another editor "
                                                      "session. Finish that session first.")
            return
        if not keys:
            return
        if self.session.is_yolo and not self.session.base_adapter.class_names:
            QMessageBox.warning(self, self.TITLE, "Load the classes.txt of this YOLO dataset first (class ids need names).")
            return
        self._keys = keys
        custom = self._custom_python()
        if custom and self.source.deps_hash() in (self.settings.value(f"editors/{self.kind}/verified", []) or []):
            self.python = custom
            self._create_session()
            return
        self._busy(True, "Looking for a suitable Python …")

        def find():
            base = self.settings.value("editors/base_python", "") or ""
            v = python_version(base) if base and Path(base).is_file() else None
            if v and v >= MIN_PYTHON[self.kind]:
                return base, v
            return find_base_python(MIN_PYTHON[self.kind])

        def found(res):
            self._busy(False)
            if res is None:
                QMessageBox.warning(self, self.TITLE,
                                    f"{TITLES[self.kind]} needs Python ≥ {'.'.join(map(str, MIN_PYTHON[self.kind]))} and "
                                    "none was found. Install Python 3.11/3.12, or use “Use existing Python…”.")
                return
            base, v = res
            self.settings.setValue("editors/base_python", base)
            env = env_dir(self.kind, v)
            py = str(env / ("Scripts/python.exe" if __import__("sys").platform == "win32" else "bin/python"))
            if is_installed(env, self.source):
                self.python = py
                self._create_session()
                return
            if QMessageBox.question(
                    self, self.TITLE,
                    f"Set up {self.source.title} now?\n\nThis downloads {SIZE_HINT[self.kind]} into\n{env}\n"
                    "once; later sessions start instantly. Progress is shown in the next window.") \
                    != QMessageBox.StandardButton.Yes:
                return
            self._show_session_page(installing=True)
            steps = install_steps(env, self.source, base, create=not Path(py).is_file())
            self.runner = StepRunner(steps, self)
            self.runner.output.connect(self._log)
            self.runner.stepStarted.connect(lambda i, n, t: self._set_progress(i, n, t))
            self.runner.finished.connect(lambda ok, msg: self._install_done(ok, msg, env, py, base))
            self.runner.start()

        self._run(find, found, lambda m: (self._busy(False), QMessageBox.warning(self, self.TITLE, m)))

    def _install_done(self, ok, msg, env, py, base):
        self.progress.setVisible(False)
        self.runner = None
        if not ok:
            self.header.setText(f"<b>Setup failed</b> — {msg}")
            apply_style(self.header, "color:@error;")
            self.back_btn.setVisible(True)
            return
        mark_installed(env, self.source, base)
        self._log(f"\n✓ {self.source.title} is ready.\n")
        self.python = py
        self._create_session()

    def _create_session(self):
        s = self.session
        ad = s.base_adapter
        if s.is_yolo:
            names, hints = list(ad.class_names), None
        else:
            names = list(s.user_classes or [])
            hints = sorted(s.label_totals(raw=True))
        policy = "embed" if (self.image_data.isVisible() and self.image_data.isChecked()) else "none"
        samples = [s.by_key[k] for k in self._keys]
        src, kind = self.source, self.kind
        self._show_session_page(installing=False)
        self.header.setText(f"Preparing {len(samples):,} file(s) …")

        def make():
            return EditSession.create(s.root, samples, ad, kind, names, policy, src.title, review=s.review,
                                      source_version=src.version, label_hints=hints)

        def made(es):
            self._attach(es)
            self._launch()

        def failed(msg):
            self.header.setText(f"<b>Could not start</b> — {msg.split(': ', 1)[-1]}")
            apply_style(self.header, "color:@error;")
            self.back_btn.setVisible(True)

        self._run(make, made, failed)

    def _attach(self, es: EditSession):
        self.es = es
        es.bind_targets(self.session.by_key)
        for e in es.entries:
            if e.status != "error":
                self.session.locked[e.key] = es.source_title
        self._fill_table()
        self.timer.start()
        self.app.editor_dialog = self

    def _launch(self):
        es = self.es
        labels = es.work / ("classes.txt" if es.editor == LABELIMG else "labels.txt")
        validate = bool(es.class_names)
        flags_file = None
        if getattr(es, "flags_matter", False):          # classification (LabelMe flags): offer the classes as flags
            flags_file = es.work / "flags.txt"
            flags_file.write_text("\n".join(es.class_names or sorted(self.session.label_totals(raw=True))) + "\n",
                                  encoding="utf-8")
        prog, args, cwd = build_command(self.source, self.python, es.work, labels if labels.is_file() else None,
                                        es.image_data, validate, flags_file)
        p = QProcess(self)
        p.setProcessEnvironment(clean_process_env())
        p.setWorkingDirectory(cwd)
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        p.readyReadStandardOutput.connect(lambda: self._log(bytes(p.readAllStandardOutput()).decode("utf-8", "replace")))
        p.finished.connect(self._editor_closed)
        p.errorOccurred.connect(lambda err: err == QProcess.ProcessError.FailedToStart and
                                self._editor_closed(-1, None, failed=True))
        self.proc = p
        self._log(f"\n▶ {self.source.title}: {Path(prog).name} {' '.join(args)}\n")
        p.start(prog, args)
        self._update_header()
        self._set_buttons(running=True)

    # =====================================================================================
    # Session page
    # =====================================================================================
    def _build_session(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.header = QLabel("")
        self.header.setWordWrap(True)
        self.header.setTextFormat(Qt.TextFormat.RichText)
        f = self.header.font()
        f.setPointSizeF(f.pointSizeF() + 1)
        self.header.setFont(f)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File", "Status", "Details"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._open_row(r))
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._row_menu)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(5000)
        self.console.setFont(QFont("Monospace", 9))
        self.console.setStyleSheet("QPlainTextEdit{background:#15171a; color:#d7dde4; border-radius:6px;}")
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(self.console)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        hint = QLabel("Save in the editor (autosave is on) — each saved file is checked and written back within a "
                      "second. Double-click a row to see the file here; right-click a conflict to resolve it.")
        hint.setWordWrap(True)
        apply_style(hint, "color:@muted;")
        foot = QHBoxLayout()
        self.back_btn = newButton("Back", icon="prev", slot=lambda: self.stack.setCurrentIndex(0))
        self.back_btn.setVisible(False)
        self.reopen_btn = newButton("Reopen editor", icon="tool-editor", slot=self._launch)
        self.sync_btn = newButton("Sync now", icon="undo", slot=lambda: [self._sync_tick(), self._sync_tick()])
        self.undo_btn = newButton("Undo session…", icon="undo", slot=self._undo)
        self.finish_btn = newButton("Finish session", icon="review-ok", slot=self._finish)
        for b in (self.back_btn, self.reopen_btn, self.sync_btn, self.undo_btn):
            foot.addWidget(b)
        foot.addStretch(1)
        foot.addWidget(self.finish_btn)
        lay.addWidget(self.header)
        lay.addWidget(self.progress)
        lay.addWidget(split, 1)
        lay.addWidget(hint)
        lay.addLayout(foot)
        return w

    def _show_session_page(self, installing: bool):
        self.stack.setCurrentIndex(1)
        self.progress.setVisible(installing)
        self.back_btn.setVisible(False)
        for b in (self.reopen_btn, self.sync_btn, self.undo_btn, self.finish_btn):
            b.setEnabled(False)
        if installing:
            self.header.setText(f"Setting up {self.source.title} …")
            apply_style(self.header, "")

    def _set_progress(self, i, n, title):
        self.progress.setRange(0, n)
        self.progress.setValue(i)
        self.progress.setFormat(f"{i + 1}/{n} · {title}")
        self.header.setText(f"Setting up {self.source.title} — {title} …")

    def _log(self, text: str):
        self.console.moveCursor(QTextCursor.MoveOperation.End)
        self.console.insertPlainText(text)
        self.console.ensureCursorVisible()

    def _set_buttons(self, running: bool):
        active = self.es is not None and self.es.active
        self.reopen_btn.setEnabled(active and not running)
        self.sync_btn.setEnabled(active)
        self.undo_btn.setEnabled(active and not running and any(e.writes for e in self.es.entries))
        self.finish_btn.setEnabled(active)
        self.finish_btn.setText("End session" if running else "Finish session")

    def _fill_table(self):
        self.table.setRowCount(len(self.es.entries))
        self._rows = {}
        for r, e in enumerate(self.es.entries):
            self._rows[e.key] = r
            self.table.setItem(r, 0, QTableWidgetItem(e.name))
            self._paint_row(e)
        self.table.resizeColumnToContents(0)

    def _paint_row(self, e):
        if self.mini is not None:
            self.mini.set_entry(e, flash=self.mini.isVisible())
        r = self._rows.get(e.key)
        if r is None:
            return
        text, role, icon = STATUS_UI.get(e.status, (e.status, "muted", "review-next"))
        st = QTableWidgetItem(newIcon(icon), text)
        st.setForeground(theme_color(role))
        self.table.setItem(r, 1, st)
        self.table.setItem(r, 2, QTableWidgetItem(e.message))

    def _update_header(self):
        if self.es is None:
            return
        c = self.es.counts()
        running = self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning
        state = f"{self.es.source_title} is open" if running else "Editor closed"
        parts = [f"<b>{state}</b> · {len(self.es.entries):,} file(s)"]
        for k in ("fixed", "unchanged", "waiting", "rejected", "conflict", "error"):
            if c.get(k):
                parts.append(f"<span style='color:{hexc(STATUS_UI[k][1])}'>{c[k]:,} {STATUS_UI[k][0]}</span>")
        self.header.setText(" · ".join(parts))
        apply_style(self.header, "")
        if self.mini is not None:
            self.mini.set_header(f"<b>{self.es.source_title}</b> · {'open' if running else 'closed'}",
                                 " · ".join(parts[1:]) or f"{len(self.es.entries):,} file(s) waiting")

    # =====================================================================================
    # Picture-in-picture
    # =====================================================================================
    def to_mini(self, snapshot=None):
        """Shrink into the in-app mini window: a snapshot of this window glides into place inside AIO QC."""
        if self.es is None or (self.mini is not None and self.mini.isVisible()):
            return
        host = self.app
        if self.mini is None:
            self.mini = MiniSessionWindow(host)
            self.mini.restoreRequested.connect(self.from_mini)
            self.mini.openRequested.connect(self._open_from_mini)
        self.mini.set_entries(self.es.entries)
        self._update_header()
        full = self._normal_geo or self.geometry()               # where to come back to (never the minimised one)
        self._normal_geo = QRect(full)
        start = self._in_host(QRect(full.topLeft(), full.size()))
        target = self.mini.default_rect()
        ghost = self._ghost(snapshot if snapshot is not None else self.grab(), start)
        self.hide()
        if host.isMinimized():
            host.showNormal()
        host.raise_()
        host.activateWindow()

        def step(t):
            ghost.setGeometry(lerp_rect(start, target, t))
            ghost.opacity = 1.0 - 0.8 * t
            ghost.update()

        def done():
            ghost.hide()
            self.mini.setGeometry(target)
            self.mini.clamp()
            self.mini.fx.setOpacity(0.0)
            self.mini.show()
            self.mini.raise_()
            FrameAnimator(self.mini, 170, lambda t: self.mini.fx.setOpacity(t)).start()

        FrameAnimator(host, 280, step, done).start()
        logger.debug("[Fix in Editor] session window → mini window")

    def from_mini(self):
        if self.mini is None or not self.mini.isVisible():
            return
        host = self.app
        start = self.mini.geometry()
        self.mini.remember()
        full = QRect(self._normal_geo or self.geometry())
        end = self._in_host(full)
        ghost = self._ghost(self._last_snapshot if self._last_snapshot is not None else self.grab(), start)
        ghost.opacity = 0.2
        ghost.hide()

        def fade_out(t):
            self.mini.fx.setOpacity(1.0 - t)

        def grow():
            self.mini.hide()
            self.mini.fx.setOpacity(1.0)
            ghost.setGeometry(start)
            ghost.show()
            ghost.raise_()

            def step(t):
                ghost.setGeometry(lerp_rect(start, end, t))
                ghost.opacity = 0.2 + 0.8 * t
                ghost.update()

            def done():
                self.setWindowState(Qt.WindowState.WindowNoState)
                self.setGeometry(full)
                self.show()
                self.raise_()
                self.activateWindow()
                ghost.hide()

            FrameAnimator(host, 300, step, done).start()

        FrameAnimator(self.mini, 110, fade_out, grow, QEasingCurve.Type.InCubic).start()
        logger.debug("[Fix in Editor] mini window → session window")

    def _in_host(self, global_rect: QRect) -> QRect:
        """A screen rectangle in the main window's coordinates (the ghost lives inside the app)."""
        return QRect(self.app.mapFromGlobal(global_rect.topLeft()), global_rect.size())

    def _ghost(self, pix, rect: QRect):
        g = getattr(self, "_ghost_w", None)
        if g is None:
            g = self._ghost_w = Ghost(self.app, pix)
        g.pix, g.opacity = pix, 1.0
        self._last_snapshot = pix
        g.setGeometry(rect)
        g.show()
        g.raise_()
        return g

    def changeEvent(self, e):
        """The minimise button: not to the taskbar — into the in-app mini window."""
        if e.type() == e.Type.WindowStateChange and self.isMinimized() and self.es is not None \
                and not getattr(self, "_minimising", False):
            self._minimising = True
            QTimer.singleShot(0, self._minimise_to_mini)
        super().changeEvent(e)

    def _minimise_to_mini(self):
        try:
            snap = self.grab()                     # renders the widget tree: valid even while minimised
            self.hide()
            self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
            self.to_mini(snap)
        finally:
            self._minimising = False

    def moveEvent(self, e):
        super().moveEvent(e)
        self._track_geometry()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._track_geometry()

    def _track_geometry(self):
        if self.isVisible() and not self.isMinimized() and not self.isMaximized():
            self._normal_geo = QRect(self.geometry())

    def _open_from_mini(self, key):
        if key in self.session.by_key:
            self.app.open_sample(key)
            self.app.raise_()
            self.app.activateWindow()

    # =====================================================================================
    # Sync
    # =====================================================================================
    def _sync_tick(self):
        es = self.es
        if es is None or not es.active:
            return
        changed = False
        for e in es.poll():
            self._sync_entry(e)
            changed = True
        if es.stray and not self._stray_warned:
            self._stray_warned = True
            self._log(f"\n⚠ labelImg saved Pascal VOC files ({', '.join(sorted(es.stray)[:3])}) — they are ignored. "
                      "Switch labelImg back to YOLO (View ▸ … or the format button) and save again.\n")
        if changed:
            es.save()
            self._update_header()
            self._set_buttons(running=self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning)

    def _sync_entry(self, e):
        es, s = self.es, self.session
        sample = s.by_key.get(e.key)
        if sample is None:
            e.status, e.message = "error", "the file is no longer in the dataset"
            es.mark_seen(e)
            self._paint_row(e)
            return
        try:
            upd = es.build_update(e, s.base_adapter, sample)
        except Rejected as exc:
            e.status, e.message = "rejected", f"{exc} — fix it in the editor and save again"
            es.mark_seen(e)
            self._paint_row(e)
            return
        except ValueError as exc:
            msg = str(exc)
            e.status = "conflict" if msg.startswith("conflict") else "error"
            e.message = ("the file changed outside this session — right-click to choose which version to keep"
                         if e.status == "conflict" else f"{msg} (save it again in the editor)")
            es.mark_seen(e)
            self._paint_row(e)
            return
        if upd is None:
            if e.status in ("waiting", "rejected", "error"):
                e.status, e.message = "unchanged", "saved without changes"
            es.mark_seen(e)
            self._paint_row(e)
            return
        try:
            es.backup_original(e, sample)
            s.locked.pop(e.key, None)                  # our own write is allowed
            s.write_external(e.key, lambda ann: (upd.apply(ann), True)[1])
        except Exception as exc:
            e.status, e.message = "error", f"could not write: {exc}"
            es.mark_seen(e)
            self._paint_row(e)
            return
        finally:
            if es.active:
                s.locked[e.key] = es.source_title
        es.after_write(e, sample, upd.summary)
        s.set_review([e.key], REVIEW_FIXED, f"Fixed in {es.source_title} · {time.strftime('%Y-%m-%d %H:%M')} · "
                                            f"{upd.summary}")
        self._paint_row(e)

    def _editor_closed(self, code, _status, failed: bool = False):
        self._log(f"\n■ editor closed{' — it could not be started' if failed else f' (exit code {code})'}\n")
        self.proc = None
        # two more polls so a save made right before closing is not missed
        QTimer.singleShot(POLL_MS, self._sync_tick)
        QTimer.singleShot(2 * POLL_MS + 100, lambda: (self._sync_tick(), self._update_header(),
                                                      self._set_buttons(running=False)))
        crashed = _status == QProcess.ExitStatus.CrashExit
        if failed or crashed or code not in (0, None):
            self.header.setText("<b>The editor stopped unexpectedly</b> — see the log below. "
                                "Your synced changes are safe; “Reopen editor” to continue.")
            apply_style(self.header, "color:@error;")
        self._set_buttons(running=False)

    # =====================================================================================
    # Row actions
    # =====================================================================================
    def _entry_at(self, r):
        return self.es.entries[r] if self.es and 0 <= r < len(self.es.entries) else None

    def _open_row(self, r):
        e = self._entry_at(r)
        if e is not None and e.key in self.session.by_key:
            self.app.open_sample(e.key)

    def _row_menu(self, pos):
        r = self.table.indexAt(pos).row()
        e = self._entry_at(r)
        if e is None:
            return
        m = QMenu(self)
        m.addAction(newIcon("eye"), "Show in AIO QC", lambda: self._open_row(r))
        if e.status == "conflict":
            sample = self.session.by_key.get(e.key)
            m.addSeparator()
            m.addAction(newIcon("tool-editor"), "Write the editor's version",
                        lambda: (self.es.resolve_conflict(e, sample, True), self._paint_row(e)))
            m.addAction(newIcon("review-ok"), "Keep the file as it is now",
                        lambda: (self.es.resolve_conflict(e, sample, False), self._paint_row(e)))
        m.exec(self.table.viewport().mapToGlobal(pos))

    # =====================================================================================
    # Undo / finish / resume
    # =====================================================================================
    def _stop_editor(self) -> bool:
        if self.proc is None or self.proc.state() == QProcess.ProcessState.NotRunning:
            return True
        if QMessageBox.question(self, self.TITLE, f"{self.es.source_title} is still open. Close it now?\n"
                                                  "(Unsaved edits in the editor are lost.)") \
                != QMessageBox.StandardButton.Yes:
            return False
        self.proc.terminate()
        if not self.proc.waitForFinished(4000):
            self.proc.kill()
            self.proc.waitForFinished(2000)
        self.proc = None
        return True

    def _release(self):
        for e in self.es.entries:
            self.session.locked.pop(e.key, None)
        self.timer.stop()
        if getattr(self.app, "editor_dialog", None) is self:
            self.app.editor_dialog = None

    def _undo(self):
        if not self._stop_editor():
            return
        written = [e for e in self.es.entries if e.writes]
        if QMessageBox.question(self, self.TITLE, f"Restore the {len(written)} file(s) written in this session from "
                                                  "their backups, restore their review marks, and end the session?") \
                != QMessageBox.StandardButton.Yes:
            return
        restored = self.es.undo()
        self.session.reindex_keys([k for k, _p in restored])
        for key, prev in restored:
            self.session.set_review([key], prev[0] if prev else None, prev[1] if prev else None)
        self._end(f"Undone — {len(restored)} file(s) restored from backup.")

    def _finish(self):
        if not self._stop_editor():
            return
        self._sync_tick()
        self._sync_tick()
        pending = [e for e in self.es.entries if e.status in ("rejected", "conflict")]
        if pending and QMessageBox.question(
                self, self.TITLE, f"{len(pending)} file(s) still need attention (rejected or conflict). "
                                  "Finish anyway? Their edits in the editor are discarded; the dataset files stay "
                                  "as they are.") != QMessageBox.StandardButton.Yes:
            return
        c = self.es.counts()
        self._end(f"Session finished — {c['fixed']:,} fixed · {c['unchanged']:,} unchanged. "
                  + (f"Originals of {self.es.backed_up():,} file(s) are in the compact backup store "
                     "(Tools ▸ Storage…)." if self.es.backed_up() else "No backups needed."))

    def _end(self, message: str):
        if self.mini is not None and self.mini.isVisible():
            self.from_mini()                                 # show the summary in the full window
        self._release()
        self.es.finish(remove_work=True)
        self.header.setText(f"<b>{message}</b>")
        apply_style(self.header, "")
        for b in (self.reopen_btn, self.sync_btn, self.undo_btn):
            b.setEnabled(False)
        self.finish_btn.setText("Close")
        self.finish_btn.setEnabled(True)
        try:
            self.finish_btn.clicked.disconnect()
        except TypeError:
            pass
        self.finish_btn.clicked.connect(self.close)
        self.app.statusBar().showMessage(message, 8000)
        self.app.toast.show_message(f"<b>{self.es.source_title}</b><br>{message}",
                                    kind="info" if message.startswith("Undone") else "success")

    def _resume(self, work: Path):
        es = EditSession.load(work, self.session.root)
        src = next((x for x in list_sources() if x.title == es.source_title), None)
        if src is None:
            src = next(iter(list_sources(es.editor)), None)
        self.source = src
        self.rb_labelimg.setChecked(es.editor == LABELIMG)
        self._show_session_page(installing=False)
        self._attach(es)
        custom = self._custom_python()
        base = self.settings.value("editors/base_python", "") or ""
        v = python_version(base) if base else None
        env_py = None
        if v and src is not None:
            env = env_dir(es.editor, v)
            env_py = str(env / ("Scripts/python.exe" if __import__("sys").platform == "win32" else "bin/python"))
            if not is_installed(env, src):
                env_py = None
        self.python = custom or env_py
        # catch up on saves made while AIO QC was closed
        for e in es.entries:
            if e.status not in ("error",) or e.last_mirror_sig is not None:
                e.pending_sig = None
        self._sync_tick()
        self._sync_tick()
        self._update_header()
        self._set_buttons(running=False)
        if self.python is None:
            self.reopen_btn.setEnabled(False)
            self.reopen_btn.setToolTip("The editor environment is not installed any more")

    def closeEvent(self, event):
        if self.mini is not None and self.mini.isVisible() and not self.isVisible():
            event.ignore()                                   # minimised: the mini window stands in for this one
            return
        if self.es is not None and self.es.active:
            running = self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning
            box = QMessageBox(self)
            box.setWindowTitle(self.TITLE)
            box.setText(f"The editor session is still active{' and ' + self.es.source_title + ' is open' if running else ''}.")
            fin = box.addButton("Finish session", QMessageBox.ButtonRole.AcceptRole)
            later = box.addButton("Keep for later", QMessageBox.ButtonRole.DestructiveRole) if not running else None
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is fin:
                self._finish()
                if self.es.active:
                    event.ignore()
                    return
            elif later is not None and box.clickedButton() is later:
                self.keep_for_later("kept for later by the user")
            else:
                event.ignore()
                return
        self._shutdown("session window closed")
        super().closeEvent(event)

    def keep_for_later(self, why: str):
        """Leave the session unfinished (offered again when the dataset opens); close the editor if it is open."""
        if self.es is None or not self.es.active:
            return
        if self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning:
            self._sync_tick()
            self._sync_tick()                            # take the last saves before closing the editor
            self.proc.terminate()
            if not self.proc.waitForFinished(4000):
                self.proc.kill()
                self.proc.waitForFinished(2000)
            self.proc = None
        c = self.es.counts()
        self.es.save()
        self._release()
        logger.info(f"[Fix in Editor] session {self.es.work.name} {why}: {c.get('fixed', 0)} fixed, "
                     f"{c.get('waiting', 0)} waiting, {c.get('rejected', 0) + c.get('conflict', 0)} need attention — "
                     f"work folder {self.es.work}")

    def _shutdown(self, why: str):
        """Everything this window started stops BEFORE Qt destroys it (no 'QThread destroyed while running')."""
        from .lifecycle import stop_threads
        if self.runner is not None:
            self.runner.cancel()
        self.timer.stop()
        if self.mini is not None:
            self.mini.hide()
            self.mini.deleteLater()
            self.mini = None
        stop_threads(self, why)
        self.app.forget_dialog(self)
