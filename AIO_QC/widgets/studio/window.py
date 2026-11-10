# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Data Studio — the table-data mode of AIO QC: its own window, menus, panels, recent files and settings.

    Tables panel   every table of the workspace (files you opened, the image dataset's annotations …)
    Recipe panel   the steps of the current table, in order: tick to switch one off, double-click to edit, drag to
                   reorder (or Alt+↑ / ↓); click a step to see the data right after it
    Data · Profile · Chart · Code   the table (virtual grid), its health with one-click fixes, charts, and the
                   recipe as runnable pandas / PySpark code
Work runs in the background on pandas; big files open as a sample, while exports and Spark use the full data.
"""
from __future__ import annotations

import copy
import os
import time
import webbrowser
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QHeaderView, QInputDialog,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPlainTextEdit,
    QStackedWidget, QTableView, QTabWidget, QToolBar, QToolButton, QVBoxLayout, QWidget,
)

from AIO_QC.core.tabular import io as TIO
from AIO_QC.core.tabular import profile as PR
from AIO_QC.core.tabular import report as RP
from AIO_QC.core.tabular import steps as S
from AIO_QC.core.tabular.workspace import StepError, Workspace
from AIO_QC.utils import app_settings, newAction, newButton, newIcon, style as apply_style

from .chart_view import ChartPanel
from .profile_view import ProfileView
from .step_form import StepDialog, _mono
from .table_model import FrameModel
from AIO_QC.widgets.lifecycle import RetainedThread

WS_FILTER = "Data Studio workspace (*.aioqc-data.json);;JSON (*.json)"
GROUPS = ["Columns", "Rows", "Clean", "Reshape", "Features"]


class Job(RetainedThread, QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str, object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as exc:                       # reported in the window; never kills the app
            logger.opt(exception=exc).debug("data studio job failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}" if not isinstance(exc, StepError) else str(exc), exc)


class StudioWindow(QMainWindow):
    def __init__(self, shell=None):
        super().__init__()
        self.shell = shell
        self.st = app_settings()
        from AIO_QC.widgets.settings_pages import apply_studio_tunables
        apply_studio_tunables(self.st)                      # large-file / profile / chart / memory settings
        self.ws = Workspace()
        self.current: str | None = None
        self.at_step: int | None = None              # show the data after this many steps (None = all)
        self.df = None
        self.kinds: dict[str, str] = {}
        self.jobs: list[Job] = []
        self._busy = None
        self._pending = False
        self._undo: list = []
        self._redo: list = []
        self._prof_key = None
        self.setWindowTitle("Data Studio — AIO QC App")
        self.setWindowIcon(newIcon("mode-data"))
        self.setAcceptDrops(True)
        self.resize(1360, 860)
        self._build()
        g = self.st.value("studio/geometry")
        if g is not None:
            self.restoreGeometry(g)
        s = self.st.value("studio/state")
        if s is not None:
            self.restoreState(s)
        self._show_welcome()

    # ================================================================== UI
    def _build(self):
        A = lambda *a, **k: newAction(self, *a, **k)
        self.a = type("Actions", (), {})()
        a = self.a
        a.open = A("Open Data…", self.open_data_dialog, "Ctrl+O", "open", "Open CSV / Excel / Parquet / JSON / SQLite")
        a.open_ws = A("Open Workspace…", self.open_workspace_dialog, "Ctrl+Shift+O", "open")
        a.save = A("Save Workspace", self.save, "Ctrl+S", "save", "Save the tables and their recipes (not the data)")
        a.save_as = A("Save Workspace As…", self.save_as, "Ctrl+Shift+S")
        a.new = A("New Workspace", self.new_workspace, "Ctrl+N")
        a.export = A("Export Data…", self.export_data, "Ctrl+E", "export", "Write the result as CSV / Parquet / "
                                                                             "Excel / JSON")
        a.code_pd = A("Export pandas Code…", lambda: self.export_code("pandas"), None, "data-code")
        a.code_sp = A("Export PySpark Code…", lambda: self.export_code("pyspark"), None, "data-code")
        a.spark = A("Run on Spark…", self.run_spark, "Ctrl+Shift+R", "data-spark",
                    "Run the whole recipe on the full data with Spark (local or cluster)")
        a.report = A("Profile Report…", self.export_report, "Ctrl+Alt+R", "tool-report",
                     "One HTML page: every column's health, charts and the recipe")
        a.undo = A("Undo", self.undo, "Ctrl+Z", "undo")
        a.redo = A("Redo", self.redo, ["Ctrl+Shift+Z", "Ctrl+Y"], "redo")
        a.add = A("Add Step…", self._add_step_menu, "Ctrl+T", "data-add-step", "Add a step to the recipe")
        a.reload = A("Reload Source", self.reload_source, "F5", None, "Read the file again")
        a.switch = A("Switch to Image QC", lambda: self.shell and self.shell.switch("image"), None,
                     "mode-image", "The image-dataset mode of AIO QC")
        a.next_mode = A("Next Mode", lambda: self.shell and self.shell.cycle(), "Ctrl+Alt+M", None,
                        "Frame Extractor → Image QC → Data Studio")
        a.to_video = A("Switch to Frame Extractor", lambda: self.shell and self.shell.switch("video"), None,
                       "mode-video", "The video mode of AIO QC: play, mark and extract frames for annotation")
        a.quit = A("Quit", self.close, "Ctrl+Q")
        mb = self.menuBar()
        f = mb.addMenu("&File")
        for x in (a.new, a.open, a.open_ws):
            f.addAction(x)
        self.recent_menu = f.addMenu("Open Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        f.addSeparator()
        for x in (a.save, a.save_as):
            f.addAction(x)
        f.addSeparator()
        for x in (a.export, a.code_pd, a.code_sp, a.report, a.spark):
            f.addAction(x)
        f.addSeparator()
        f.addAction(a.next_mode)
        f.addAction(a.switch)
        f.addAction(a.to_video)
        f.addAction(a.quit)
        e = mb.addMenu("&Edit")
        for x in (a.undo, a.redo):
            e.addAction(x)
        e.addSeparator()
        e.addAction(A("Edit Step…", lambda: self.edit_step(self._sel_step()), "Return"))
        e.addAction(A("Delete Step", lambda: self.delete_step(self._sel_step()), "Delete"))
        e.addAction(A("Move Step Up", lambda: self.move_step(self._sel_step(), -1), "Alt+Up"))
        e.addAction(A("Move Step Down", lambda: self.move_step(self._sel_step(), 1), "Alt+Down"))
        d = mb.addMenu("&Data")
        d.addAction(a.add)
        for g in GROUPS:
            sub = d.addMenu(g)
            for op in S.OPS.values():
                if op.group == g:
                    sub.addAction(op.title + "…", lambda k=op.key: self.add_step(k))
        d.addSeparator()
        d.addAction(a.reload)
        d.addAction(A("Duplicate Table", self.duplicate_table))
        d.addAction(A("Rename Table…", self.rename_table))
        d.addAction(A("Close Table", self.close_table))
        v = mb.addMenu("&View")
        for i, (name, key) in enumerate((("Data", "Ctrl+1"), ("Profile", "Ctrl+2"), ("Chart", "Ctrl+3"),
                                         ("Code", "Ctrl+4"))):
            v.addAction(A(name, lambda n=i: self.tabs.setCurrentIndex(n), key))
        v.addSeparator()
        h = mb.addMenu("&Help")
        h.addAction(A("Data Studio Guide", self.show_help, "F1"))
        h.addAction(A("Formula Reference", lambda: QMessageBox.information(self, "Formulas", S.OPS["derive"].help)))
        h.addSeparator()
        from AIO_QC.widgets.about import show_about
        h.addAction(A("About Data Studio", lambda: show_about(
            self, "Data Studio", "Analysis, cleaning and features for tables (CSV, Excel, Parquet, Spark)",
            newIcon("mode-data"))))

        from AIO_QC.widgets.mode_switch import ModeSwitch
        self.mode_switch = ModeSwitch("data", self)
        self.mode_switch.switched.connect(lambda m: self.shell and self.shell.switch(m))
        mb.setCornerWidget(self.mode_switch, Qt.Corner.TopRightCorner)
        self.mode_switch.setVisible(self.shell is not None)
        from AIO_QC.widgets.settings_dialog import install_settings
        install_settings(self)                              # gear in the corner + File ▸ Settings… (Ctrl+,)

        tb = QToolBar("Main")
        tb.setObjectName("studioToolbar")
        tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        for x in (a.open, a.save):
            tb.addAction(x)
        tb.addSeparator()
        self.add_btn = QToolButton()
        self.add_btn.setDefaultAction(a.add)
        self.add_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.add_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_btn.setMenu(self._steps_menu())
        tb.addWidget(self.add_btn)
        for x in (a.undo, a.redo):
            tb.addAction(x)
        tb.addSeparator()
        for x in (a.export, a.report, a.spark):
            tb.addAction(x)
        self.addToolBar(tb)

        # docks
        self.tables = QListWidget()
        self.tables.currentItemChanged.connect(lambda cur, _p: cur and self.set_current(cur.data(Qt.ItemDataRole.UserRole)))
        self.tables.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tables.customContextMenuRequested.connect(self._tables_menu)
        dk = QDockWidget("Tables", self)
        dk.setObjectName("studioTables")
        dk.setWidget(self.tables)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dk)
        rec = QWidget()
        rl = QVBoxLayout(rec)
        rl.setContentsMargins(0, 0, 0, 0)
        self.steps = QListWidget()
        self.steps.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.steps.model().rowsMoved.connect(self._steps_dragged)
        self.steps.itemChanged.connect(self._step_toggled)
        self.steps.currentRowChanged.connect(self._step_selected)
        self.steps.itemDoubleClicked.connect(lambda it: self.edit_step(self.steps.row(it) - 1))
        self.steps.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.steps.customContextMenuRequested.connect(self._steps_ctx)
        rl.addWidget(self.steps, 1)
        add = newButton("Add step", icon="data-add-step", slot=self._add_step_menu)
        rl.addWidget(add)
        dr = QDockWidget("Recipe", self)
        dr.setObjectName("studioRecipe")
        dr.setWidget(rec)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dr)
        self.docks = (dk, dr)

        # centre
        self.stack = QStackedWidget()
        self.welcome = self._build_welcome()
        self.stack.addWidget(self.welcome)
        work = QWidget()
        wl = QVBoxLayout(work)
        wl.setContentsMargins(0, 0, 0, 0)
        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        apply_style(self.banner, "QLabel{background:@banner_bg; color:@banner_fg; border:1px solid @banner_line; "
                                 "border-radius:4px; padding:5px 9px;}")
        self.banner.setVisible(False)
        wl.addWidget(self.banner)
        self.tabs = QTabWidget()
        self.grid = QTableView()
        self.model = FrameModel(self)
        self.grid.setModel(self.model)
        self.grid.setAlternatingRowColors(True)
        self.grid.verticalHeader().setDefaultSectionSize(24)
        self.grid.horizontalHeader().setDefaultSectionSize(130)
        self.grid.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.grid.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.grid.horizontalHeader().customContextMenuRequested.connect(self._column_menu)
        self.grid.horizontalHeader().sectionDoubleClicked.connect(lambda c: self._column_menu_at(c))
        self.tabs.addTab(self.grid, newIcon("mode-data"), "Data")
        self.profile = ProfileView()
        self.profile.fix.connect(lambda st: self.add_step(st["op"], st["p"], ask=False))
        self.tabs.addTab(self.profile, newIcon("data-profile"), "Profile")
        self.chart = ChartPanel(self.st)
        self.tabs.addTab(self.chart, newIcon("data-chart"), "Chart")
        codew = QWidget()
        cl = QVBoxLayout(codew)
        crow = QHBoxLayout()
        self.code_engine = QComboBox()
        self.code_engine.addItem("pandas", "pandas")
        self.code_engine.addItem("PySpark", "pyspark")
        self.code_engine.currentIndexChanged.connect(self._fill_code)
        crow.addWidget(QLabel("Recipe as"))
        crow.addWidget(self.code_engine)
        crow.addStretch(1)
        crow.addWidget(newButton("Copy", slot=lambda: QApplication.clipboard().setText(self.code.toPlainText())))
        crow.addWidget(newButton("Save .py…", slot=lambda: self.export_code(self.code_engine.currentData())))
        cl.addLayout(crow)
        self.code = QPlainTextEdit()
        self.code.setReadOnly(True)
        self.code.setFont(_mono())
        cl.addWidget(self.code, 1)
        self.tabs.addTab(codew, newIcon("data-code"), "Code")
        self.tabs.currentChanged.connect(self._tab_changed)
        wl.addWidget(self.tabs, 1)
        self.stack.addWidget(work)
        self.setCentralWidget(self.stack)
        self.status_lbl = QLabel("")
        self.statusBar().addWidget(self.status_lbl, 1)
        self.engine_lbl = QLabel("engine: pandas")
        apply_style(self.engine_lbl, "color:@muted;")
        self.statusBar().addPermanentWidget(self.engine_lbl)
        self._update_actions()

    def _steps_menu(self, parent=None) -> QMenu:
        m = QMenu(parent or self)
        for g in GROUPS:
            m.addSection(g)
            for op in S.OPS.values():
                if op.group == g:
                    m.addAction(op.title + "…", lambda k=op.key: self.add_step(k))
        return m

    def _add_step_menu(self):
        if self.current is None:
            return
        self._steps_menu().exec(self.cursor().pos())

    def _build_welcome(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addStretch(1)
        t = QLabel("Data Studio")
        apply_style(t, "font-size:26pt; font-weight:600;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = QLabel("Open a table — CSV, Excel, Parquet, JSON or SQLite — then clean it, reshape it, build features "
                     "and chart it step by step. Every step is undoable, shown as code, and runs at full scale on "
                     "Spark when you need it.")
        sub.setWordWrap(True)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setMaximumWidth(640)
        apply_style(sub, "color:@muted; font-size:11pt;")
        lay.addWidget(t)
        h = QHBoxLayout()
        h.addStretch(1)
        h.addWidget(sub)
        h.addStretch(1)
        lay.addLayout(h)
        btns = QHBoxLayout()
        btns.addStretch(1)
        for text, icon, slot in (("Open Data…", "open", self.open_data_dialog),
                                 ("Open Workspace…", "open", self.open_workspace_dialog)):
            b = newButton(text, icon=icon, slot=slot)
            b.setMinimumHeight(36)
            btns.addWidget(b)
        self.ann_btn = newButton("Annotations of the image dataset", icon="mode-image",
                                 slot=lambda: self.shell and self.shell.annotations_to_studio())
        self.ann_btn.setMinimumHeight(36)
        self.ann_btn.setToolTip("One row per annotation shape of the dataset open in Image QC")
        btns.addWidget(self.ann_btn)
        btns.addStretch(1)
        lay.addSpacing(18)
        lay.addLayout(btns)
        lay.addSpacing(14)
        self.recent_list = QListWidget()
        self.recent_list.setMaximumWidth(640)
        self.recent_list.setMaximumHeight(220)
        self.recent_list.itemActivated.connect(lambda it: self._open_path(it.data(Qt.ItemDataRole.UserRole)))
        self.recent_list.itemClicked.connect(lambda it: self._open_path(it.data(Qt.ItemDataRole.UserRole)))
        rh = QHBoxLayout()
        rh.addStretch(1)
        rh.addWidget(self.recent_list)
        rh.addStretch(1)
        self.recent_caption = QLabel("Recent")
        self.recent_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        apply_style(self.recent_caption, "color:@muted;")
        lay.addWidget(self.recent_caption)
        lay.addLayout(rh)
        hint = QLabel("…or drop files here")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        apply_style(hint, "color:@faint;")
        lay.addWidget(hint)
        lay.addStretch(2)
        return w

    def _show_welcome(self):
        self.recent_list.clear()
        for p in self._recent():
            it = QListWidgetItem(newIcon("mode-data" if not p.endswith(".json") else "open"),
                                 f"{os.path.basename(p)}   —   {os.path.dirname(p)}")
            it.setData(Qt.ItemDataRole.UserRole, p)
            self.recent_list.addItem(it)
        self.recent_caption.setVisible(self.recent_list.count() > 0)
        self.recent_list.setVisible(self.recent_list.count() > 0)
        self.ann_btn.setVisible(bool(self.shell and self.shell.image_session() is not None))
        self.stack.setCurrentIndex(0 if not self.ws.tables else 1)

    def showEvent(self, e):
        super().showEvent(e)
        if not self.ws.tables:
            self._show_welcome()

    # ================================================================== recent
    def _recent(self) -> list[str]:
        v = self.st.value("studio/recent", []) or []
        v = [v] if isinstance(v, str) else list(v)
        return [p for p in v if os.path.exists(p)][:12]

    def _remember(self, path):
        r = [p for p in self._recent() if p != path]
        self.st.setValue("studio/recent", [path] + r[:11])

    def _fill_recent(self):
        self.recent_menu.clear()
        for p in self._recent():
            self.recent_menu.addAction(os.path.basename(p), lambda pp=p: self._open_path(pp)).setToolTip(p)
        if self.recent_menu.isEmpty():
            self.recent_menu.addAction("(none)").setEnabled(False)

    # ================================================================== open / save
    def open_data_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open data", self.st.value("studio/last_dir", "") or "",
                                                TIO.OPEN_FILTER)
        if paths:
            self.st.setValue("studio/last_dir", os.path.dirname(paths[0]))
            for p in paths:
                self._open_path(p)

    def open_workspace_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open workspace", self.st.value("studio/last_dir", "") or "",
                                           WS_FILTER)
        if p:
            self._open_path(p)

    def _open_path(self, p):
        if not p or not os.path.exists(p):
            return
        if p.endswith(".aioqc-data.json"):
            return self.open_workspace(p)
        if p.lower().endswith(".json"):
            try:
                with open(p, "rb") as f:
                    if b'"app": "AIO QC Data Studio"' in f.read(400):
                        return self.open_workspace(p)
            except OSError:
                pass
        self.open_data(p)

    def open_data(self, path, name=None):
        try:
            src = TIO.detect(path)
        except Exception as exc:
            QMessageBox.warning(self, "Open data", f"{os.path.basename(path)}: {exc}")
            return
        miss = TIO.missing_engine_for(src["format"])
        if miss:
            QMessageBox.warning(self, "Open data", f"Reading {TIO.FORMATS[src['format']]} needs the package "
                                                   f"“{miss}”: pip install {miss}")
            return
        if src["format"] == "excel":
            sh = TIO.sheets(path)
            if len(sh) > 1:
                s, ok = QInputDialog.getItem(self, "Excel sheet", f"{os.path.basename(path)} — which sheet?", sh, 0,
                                             False)
                if not ok:
                    return
                src["opts"]["sheet"] = s
        if src["format"] == "sqlite":
            tb = TIO.sqlite_tables(path)
            if not tb:
                QMessageBox.information(self, "Open data", "No tables in this database.")
                return
            if len(tb) > 1:
                s, ok = QInputDialog.getItem(self, "SQLite table", "Which table?", tb, 0, False)
                if not ok:
                    return
                src["opts"]["table"] = s
        self._push_undo()
        t = self.ws.add_source(path, src, name)
        self._remember(os.path.abspath(path))
        self._refresh_tables(select=t.name)

    def add_frame(self, df, name, note=""):
        """A table made inside the app (e.g. the image dataset's annotations)."""
        self._push_undo()
        t = self.ws.add_frame(df, name, note)
        self._refresh_tables(select=t.name)
        return t.name

    def open_workspace(self, path):
        if not self.maybe_save():
            return
        try:
            ws = Workspace.open(path)
        except Exception as exc:
            QMessageBox.warning(self, "Open workspace", f"{exc}")
            return
        mem = [n for n, t in ws.tables.items() if t.source.get("format") == "memory"]
        for n in mem:                                     # in-app tables: take them over from this session if open
            if n in self.ws.tables and n in self.ws._raw:
                ws._raw[n] = self.ws._raw[n]
        missing = [t.source.get("path") for t in ws.tables.values()
                   if t.source.get("format") != "memory" and not os.path.exists(t.source.get("path", ""))]
        self.ws = ws
        self._undo.clear()
        self._redo.clear()
        self._remember(os.path.abspath(path))
        self._refresh_tables(select=next(iter(ws.tables), None))
        if missing:
            QMessageBox.warning(self, "Open workspace", "These source files were not found:\n" + "\n".join(missing))

    def new_workspace(self):
        if not self.maybe_save():
            return
        self.ws = Workspace()
        self.current = None
        self._undo.clear()
        self._redo.clear()
        self._refresh_tables()
        self._show_welcome()

    def save(self) -> bool:
        if not self.ws.tables:
            return True
        if not self.ws.path:
            return self.save_as()
        try:
            self.ws.save(self.ws.path)
        except OSError as exc:
            QMessageBox.warning(self, "Save", str(exc))
            return False
        self._remember(self.ws.path)
        self.statusBar().showMessage(f"Saved {self.ws.path}", 4000)
        self._update_title()
        return True

    def save_as(self) -> bool:
        d = self.st.value("studio/last_dir", "") or ""
        p, _ = QFileDialog.getSaveFileName(self, "Save workspace", os.path.join(d, "workspace.aioqc-data.json"),
                                           WS_FILTER)
        if not p:
            return False
        if not p.endswith(".json"):
            p += ".aioqc-data.json"
        self.ws.path = p
        return self.save()

    def maybe_save(self) -> bool:
        if not self.ws.tables or not self.ws.dirty:
            return True
        r = QMessageBox.question(self, "Data Studio", "Save the workspace (tables and recipes) first?",
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                 QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Cancel:
            return False
        return self.save() if r == QMessageBox.StandardButton.Save else True

    # ================================================================== tables
    def _refresh_tables(self, select=None):
        self.tables.blockSignals(True)
        self.tables.clear()
        for name, t in self.ws.tables.items():
            src = t.source
            sub = t.note or (os.path.basename(src.get("path", "")) if src.get("path") else "in-app")
            it = QListWidgetItem(newIcon("mode-image" if src.get("format") == "memory" else "mode-data"),
                                 f"{name}\n{sub} · {len(t.steps)} step(s)")
            it.setData(Qt.ItemDataRole.UserRole, name)
            it.setToolTip(src.get("path") or t.note)
            self.tables.addItem(it)
        self.tables.blockSignals(False)
        names = list(self.ws.tables)
        target = select if select in names else (self.current if self.current in names else (names[0] if names
                                                                                             else None))
        if target is None:
            self.current = None
            self._show_welcome()
            self._update_actions()
            return
        self.tables.setCurrentRow(names.index(target))
        self.set_current(target, force=True)

    def set_current(self, name, force=False):
        if name == self.current and not force:
            return
        self.current = name
        self.at_step = None
        self.stack.setCurrentIndex(1)
        self._fill_steps()
        self.recompute()

    def _tables_menu(self, pos):
        it = self.tables.itemAt(pos)
        if it is None:
            return
        self.set_current(it.data(Qt.ItemDataRole.UserRole))
        m = QMenu(self)
        m.addAction("Rename…", self.rename_table)
        m.addAction("Duplicate", self.duplicate_table)
        m.addAction("Reload source", self.reload_source)
        m.addSeparator()
        m.addAction("Close", self.close_table)
        m.exec(self.tables.viewport().mapToGlobal(pos))

    def rename_table(self):
        if not self.current:
            return
        new, ok = QInputDialog.getText(self, "Rename table", "New name:", text=self.current)
        new = new.strip()
        if ok and new and new != self.current:
            try:
                self._push_undo()
                self.ws.rename(self.current, new)
            except ValueError as exc:
                QMessageBox.information(self, "Rename", str(exc))
                return
            self.current = new
            self._refresh_tables(select=new)

    def duplicate_table(self):
        if not self.current:
            return
        self._push_undo()
        t = self.ws.tables[self.current]
        name = self.ws.unique_name(t.name + "_copy")
        from AIO_QC.core.tabular.workspace import Table
        self.ws.tables[name] = Table(name, dict(t.source), copy.deepcopy(t.steps), t.note)
        if self.current in self.ws._raw:
            self.ws._raw[name] = self.ws._raw[self.current]
        self._refresh_tables(select=name)

    def close_table(self):
        if not self.current:
            return
        users = [n for n, t in self.ws.tables.items() if any(st["op"] == "join" and st["p"].get("table") ==
                                                             self.current for st in t.steps)]
        msg = f"Close “{self.current}”?" + (f"\n\n{', '.join(users)} join it — those steps will fail." if users else "")
        if QMessageBox.question(self, "Close table", msg) != QMessageBox.StandardButton.Yes:
            return
        self._push_undo()
        self.ws.remove(self.current)
        self.current = None
        self._refresh_tables()

    def reload_source(self):
        if self.current:
            self.ws.reload(self.current)
            self.recompute()

    # ================================================================== steps
    def _fill_steps(self):
        self.steps.blockSignals(True)
        self.steps.clear()
        t = self.ws.tables.get(self.current) if self.current else None
        if t is not None:
            src = QListWidgetItem(newIcon("open"), f"Source: {os.path.basename(t.source.get('path', '')) or t.note}")
            src.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.steps.addItem(src)
            for i, st in enumerate(t.steps, 1):
                it = QListWidgetItem(f"{i}. {S.OPS[st['op']].describe(st['p'])}")
                it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
                            | Qt.ItemFlag.ItemIsDragEnabled)
                it.setCheckState(Qt.CheckState.Checked if st.get("on", True) else Qt.CheckState.Unchecked)
                it.setToolTip(S.OPS[st["op"]].title + " — double-click to edit, untick to switch off")
                self.steps.addItem(it)
            row = len(t.steps) if self.at_step is None else self.at_step
            self.steps.setCurrentRow(row)
        self.steps.blockSignals(False)
        self._update_actions()

    def _sel_step(self) -> int | None:
        r = self.steps.currentRow()
        return r - 1 if r >= 1 else None

    def _step_selected(self, row):
        if self.current is None or row < 0:
            return
        t = self.ws.tables[self.current]
        at = None if row >= len(t.steps) else row
        if at != self.at_step:
            self.at_step = at
            self.recompute()

    def _data_for_step(self, index):
        """The data a step at position `index` receives (for its form)."""
        return self.ws.result(self.current, index)

    def add_step(self, op_key, preset=None, ask=True):
        if self.current is None:
            QMessageBox.information(self, "Add step", "Open a table first.")
            return
        t = self.ws.tables[self.current]
        pos = len(t.steps) if self.at_step is None else self.at_step
        try:
            df = self._data_for_step(pos)
        except Exception as exc:
            QMessageBox.warning(self, "Add step", f"The data before this step cannot be computed: {exc}")
            return
        p = dict(preset or {})
        if ask or S.OPS[op_key].check(p, list(df.columns)):
            dlg = StepDialog(self, op_key, df, [n for n in self.ws.tables if n != self.current], p)
            if dlg.exec() != StepDialog.DialogCode.Accepted:
                return
            p = dlg.result_params
        self._push_undo()
        t.steps.insert(pos, {"op": op_key, "p": p, "on": True})
        self.ws.invalidate(self.current, pos)
        self.at_step = None if self.at_step is None else pos + 1
        self._fill_steps()
        self.recompute()

    def edit_step(self, i):
        if self.current is None or i is None:
            return
        t = self.ws.tables[self.current]
        st = t.steps[i]
        try:
            df = self._data_for_step(i)
        except Exception as exc:
            QMessageBox.warning(self, "Edit step", str(exc))
            return
        dlg = StepDialog(self, st["op"], df, [n for n in self.ws.tables if n != self.current], st["p"])
        if dlg.exec() != StepDialog.DialogCode.Accepted:
            return
        self._push_undo()
        st["p"] = dlg.result_params
        self.ws.invalidate(self.current, i)
        self._fill_steps()
        self.recompute()

    def delete_step(self, i):
        if self.current is None or i is None:
            return
        self._push_undo()
        del self.ws.tables[self.current].steps[i]
        self.ws.invalidate(self.current, i)
        self.at_step = None
        self._fill_steps()
        self.recompute()

    def move_step(self, i, d):
        if self.current is None or i is None:
            return
        steps = self.ws.tables[self.current].steps
        j = i + d
        if not 0 <= j < len(steps):
            return
        self._push_undo()
        steps[i], steps[j] = steps[j], steps[i]
        self.ws.invalidate(self.current, min(i, j))
        self.at_step = None
        self._fill_steps()
        self.steps.setCurrentRow(j + 1)
        self.recompute()

    def _step_toggled(self, it):
        r = self.steps.row(it)
        if r < 1 or self.current is None:
            return
        st = self.ws.tables[self.current].steps[r - 1]
        on = it.checkState() == Qt.CheckState.Checked
        if st.get("on", True) != on:
            self._push_undo()
            st["on"] = on
            self.ws.invalidate(self.current, r - 1)
            self.recompute()

    def _steps_dragged(self, *_a):
        """Drag-and-drop reorder: read the new order from the list."""
        if self.current is None:
            return
        steps = self.ws.tables[self.current].steps
        order = []
        for r in range(self.steps.count()):
            txt = self.steps.item(r).text()
            if r == 0 and txt.startswith("Source:"):
                continue
            try:
                order.append(int(txt.split(".", 1)[0]) - 1)
            except ValueError:
                pass
        if self.steps.item(0) is None or not self.steps.item(0).text().startswith("Source:") or \
                sorted(order) != list(range(len(steps))):
            QTimer.singleShot(0, self._fill_steps)                    # dropped above the source: put it back
            return
        if order != list(range(len(steps))):
            self._push_undo()
            self.ws.tables[self.current].steps = [steps[i] for i in order]
            self.ws.invalidate(self.current, min(i for i, o in enumerate(order) if o != i))
        QTimer.singleShot(0, self._fill_steps)
        self.recompute()

    def _steps_ctx(self, pos):
        it = self.steps.itemAt(pos)
        r = self.steps.row(it) if it else -1
        m = QMenu(self)
        if r >= 1:
            m.addAction("Edit…", lambda: self.edit_step(r - 1))
            m.addAction("Move up", lambda: self.move_step(r - 1, -1))
            m.addAction("Move down", lambda: self.move_step(r - 1, 1))
            m.addAction("Delete", lambda: self.delete_step(r - 1))
            m.addSeparator()
        sub = self._steps_menu(m)
        sub.setTitle("Insert a step here")
        m.addMenu(sub)
        m.exec(self.steps.viewport().mapToGlobal(pos))

    # ---- undo / redo: snapshots of every table's steps (+ the table list)
    def _snapshot(self):
        return ({n: (copy.deepcopy(t.steps), dict(t.source), t.note) for n, t in self.ws.tables.items()},
                self.current)

    def _push_undo(self):
        self._undo.append(self._snapshot())
        self._undo = self._undo[-200:]
        self._redo.clear()
        self.ws.dirty = True
        self._update_actions()

    def _restore(self, snap):
        from AIO_QC.core.tabular.workspace import Table
        tables, cur = snap
        raw = dict(self.ws._raw)
        self.ws.tables = {n: Table(n, src, steps, note) for n, (steps, src, note) in tables.items()}
        self.ws._raw = {n: v for n, v in raw.items() if n in tables}
        for n in list(self.ws.tables):
            self.ws.invalidate(n, 0)
        self.current = cur if cur in tables else None
        self.at_step = None
        self._refresh_tables(select=self.current)

    def undo(self):
        if self._undo:
            self._redo.append(self._snapshot())
            self._restore(self._undo.pop())
            self._update_actions()

    def redo(self):
        if self._redo:
            self._undo.append(self._snapshot())
            self._restore(self._redo.pop())
            self._update_actions()

    # ================================================================== compute
    def recompute(self):
        """Compute the current table (at the selected step) in the background; the newest request wins."""
        if self.current is None:
            return
        if self._busy is not None:
            self._pending = True
            return
        name, at = self.current, self.at_step
        t0 = time.monotonic()
        self.status_lbl.setText(f"Computing {name}…")
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        job = Job(lambda: (name, at, self.ws.result(name, at), time.monotonic() - t0), self)
        job.done.connect(self._computed)
        job.failed.connect(self._compute_failed)
        job.finished.connect(lambda j=job: self._job_finished(j))
        self._busy = job
        self.jobs.append(job)
        job.start()

    def _job_finished(self, job):
        QApplication.restoreOverrideCursor()
        if job in self.jobs:
            self.jobs.remove(job)
        self._busy = None
        if self._pending:
            self._pending = False
            self.recompute()

    def _computed(self, res):
        name, at, df, took = res
        if name != self.current or at != self.at_step:
            self._pending = True                                         # outdated: the newest is queued
            return
        self.df = df
        self.kinds = {str(c): PR.kind_of(df[c]) for c in df.columns}
        self.model.set_frame(df, [self.kinds[str(c)] for c in df.columns])
        info = {}
        try:
            info = self.ws.info(name)
        except Exception:
            pass
        t = self.ws.tables[name]
        nsteps = len(t.steps) if at is None else at
        self.status_lbl.setText(f"{name}: {len(df):,} rows × {df.shape[1]} columns · after {nsteps} of "
                                f"{len(t.steps)} step(s) · {took * 1000:.0f} ms")
        b = []
        if info.get("sampled"):
            b.append(f"Showing the first {info['rows']:,} rows of a {TIO.FORMATS.get(t.source.get('format'), '')} "
                     f"file of {info['bytes'] / 1024 ** 2:,.0f} MB — Export Data and Run on Spark use the whole file.")
        if at is not None:
            b.append(f"Showing the data after step {at} — click the last step (or Esc) to see the result.")
        self.banner.setText("  ".join(b))
        self.banner.setVisible(bool(b))
        self._prof_key = None
        self._tab_changed(self.tabs.currentIndex())
        self._mark_error(None)
        self._update_title()

    def _compute_failed(self, msg, exc):
        idx = getattr(exc, "index", None)
        self.status_lbl.setText(f"⚠ {msg}")
        self._mark_error(idx)
        self.banner.setText(f"⚠ {msg}" + ("  — fix or switch off that step (it is marked in the recipe)."
                                          if idx is not None else ""))
        self.banner.setVisible(True)

    def _mark_error(self, idx):
        for r in range(1, self.steps.count()):
            it = self.steps.item(r)
            f = it.font()
            f.setStrikeOut(False)
            it.setForeground(self.palette().text())
            if idx is not None and r - 1 == idx:
                from AIO_QC.utils import theme_color
                it.setForeground(theme_color("error"))
            it.setFont(f)

    def _tab_changed(self, i):
        if self.df is None:
            return
        key = (self.current, self.at_step, id(self.df))
        if i == 1 and self._prof_key != key:
            self._prof_key = key
            df = self.df
            job = Job(lambda: PR.profile(df), self)
            job.done.connect(lambda prof, k=key: self.profile.set_profile(prof) if self._prof_key == k else None)
            job.failed.connect(lambda m, _e: self.status_lbl.setText(f"⚠ profile: {m}"))
            job.finished.connect(lambda j=job: self.jobs.remove(j) if j in self.jobs else None)
            self.jobs.append(job)
            job.start()
        elif i == 2:
            self.chart.set_frame(self.df, self.kinds)
        elif i == 3:
            self._fill_code()

    def _fill_code(self):
        if self.current is None:
            return
        try:
            self.code.setPlainText(self.ws.code(self.current, self.code_engine.currentData()))
        except Exception as exc:
            self.code.setPlainText(f"# cannot write the code: {exc}")

    # ================================================================== column menu (grid header)
    def _column_menu(self, pos):
        self._column_menu_at(self.grid.horizontalHeader().logicalIndexAt(pos))

    def _column_menu_at(self, c):
        if self.df is None or c < 0:
            return
        col = str(self.df.columns[c])
        k = self.kinds.get(col, "")
        m = QMenu(self)
        m.addSection(col)
        m.addAction("Sort ascending", lambda: self.add_step("sort", {"cols": [col], "order": "ascending"}, False))
        m.addAction("Sort descending", lambda: self.add_step("sort", {"cols": [col], "order": "descending"}, False))
        m.addAction("Filter…", lambda: self.add_step("filter", {"col": col, "cmp": "=" if k not in ("text",) else
                                                                "contains", "mode": "keep", "ignore_case": True}))
        sub = m.addMenu("Change type")
        for t in S.TYPES:
            sub.addAction(t, lambda tt=t: self.add_step("cast", {"cols": [col], "type": tt}, False))
        m.addAction("Rename…", lambda: self.add_step("rename", {"col": col, "new": col}))
        m.addAction("Fill missing…", lambda: self.add_step("fill", {"cols": [col], "how": "median" if k in
                                                                     ("number", "integer") else "most common"}))
        if k in ("text", "category"):
            m.addAction("Clean text…", lambda: self.add_step("text", {"cols": [col], "trim": True, "spaces": True,
                                                                       "case": "keep"}))
            m.addAction("Split…", lambda: self.add_step("split", {"col": col, "sep": ",", "parts": 2}))
        if k in ("number", "integer"):
            m.addAction("Bin…", lambda: self.add_step("bin", {"col": col, "bins": 5, "method": "equal width"}))
            m.addAction("Outliers…", lambda: self.add_step("outliers", {"col": col, "method": "IQR", "k": 1.5,
                                                                        "action": "flag"}))
        if k == "date":
            m.addAction("Date parts", lambda: self.add_step("dateparts", {"col": col, "parts": ["year", "month",
                                                                                                 "day", "weekday"]},
                                                              False))
        m.addAction("Group by this…", lambda: self.add_step("group", {"by": [col], "aggs": [[col, "rows"]]}))
        m.addSeparator()
        m.addAction("Remove column", lambda: self.add_step("drop", {"cols": [col]}, False))
        m.addAction("Chart it", lambda: (self.tabs.setCurrentIndex(2), self.chart.x.setCurrentText(col),
                                         self.chart.type.setCurrentText("Histogram" if k in ("number", "integer",
                                                                                            "date") else "Bar")))
        m.exec(self.cursor().pos())

    # ================================================================== export
    def export_data(self):
        if self.current is None:
            return
        info = {}
        try:
            info = self.ws.info(self.current)
        except Exception:
            pass
        d = self.st.value("studio/last_dir", "") or ""
        p, flt = QFileDialog.getSaveFileName(self, "Export data", os.path.join(d, f"{self.current}.csv"),
                                             "CSV (*.csv);;Parquet (*.parquet);;Excel (*.xlsx);;JSON (*.json);;"
                                             "JSON lines (*.jsonl)")
        if not p:
            return
        ext = os.path.splitext(p)[1].lower()
        fmt = {".csv": "csv", ".parquet": "parquet", ".xlsx": "excel", ".json": "json", ".jsonl": "jsonl"}.get(ext)
        if fmt is None:
            fmt = {"CSV": "csv", "Parquet": "parquet", "Excel": "excel", "JSON lines": "jsonl", "JSON": "json"}[
                flt.split(" (")[0]]
            p += TIO.WRITE_FORMATS[fmt]
        miss = TIO.missing_engine_for(fmt)
        if miss:
            QMessageBox.warning(self, "Export", f"Writing {fmt} needs “{miss}”: pip install {miss}")
            return
        full = False
        if info.get("sampled"):
            r = QMessageBox.question(self, "Export", "The app shows a sample of this file. Export the FULL data "
                                     "(reads the whole file and runs the recipe — needs the memory for it), or just "
                                     "the sample?\n\nFor data bigger than memory use Run on Spark.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
                                     QMessageBox.StandardButton.Cancel)
            if r == QMessageBox.StandardButton.Cancel:
                return
            full = r == QMessageBox.StandardButton.Yes
        name, at = self.current, self.at_step
        ws = self.ws.full_copy() if full else self.ws

        def work():
            df = ws.result(name, at)
            TIO.write(df, p, fmt)
            return len(df)
        self.status_lbl.setText("Exporting…")
        job = Job(work, self)
        from AIO_QC.widgets import tasks
        task = tasks.start(f"Data Studio · export {os.path.basename(p)}")
        job.finished.connect(task.finish)
        job.done.connect(lambda n: self.statusBar().showMessage(f"Exported {n:,} rows → {p}", 8000))
        job.failed.connect(lambda m, _e: QMessageBox.warning(self, "Export", m))
        job.finished.connect(lambda j=job: self.jobs.remove(j) if j in self.jobs else None)
        self.jobs.append(job)
        job.start()

    def export_code(self, engine):
        if self.current is None:
            return
        out, _ = QFileDialog.getSaveFileName(self, "Where should the script write the result? (Cancel = nowhere, "
                                                   "it prints it)", f"{self.current}_out.parquet",
                                             "Parquet (*.parquet);;CSV (*.csv)")
        fmt = "csv" if out.lower().endswith(".csv") else "parquet"
        code = self.ws.code(self.current, engine, out or None, fmt)
        p, _ = QFileDialog.getSaveFileName(self, f"Save {engine} script", f"{self.current}_{engine}.py",
                                           "Python (*.py)")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(code)
            self.statusBar().showMessage(f"Saved {p}", 6000)

    def export_report(self):
        if self.df is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Profile report", f"{self.current}_profile.html", "HTML (*.html)")
        if not p:
            return
        df, name = self.df, self.current
        t = self.ws.tables[name]
        sampled = bool(self.ws.info(name).get("sampled")) if t.source.get("format") != "memory" else False

        def work():
            html = RP.build(name, PR.profile(df), t.steps, t.source.get("path") or t.note, sampled)
            with open(p, "w", encoding="utf-8") as f:
                f.write(html)
            return p
        job = Job(work, self)
        from AIO_QC.widgets import tasks
        task = tasks.start(f"Data Studio · profile report {os.path.basename(p)}")
        job.finished.connect(task.finish)
        job.done.connect(lambda path: webbrowser.open(Path(path).as_uri()))
        job.failed.connect(lambda m, _e: QMessageBox.warning(self, "Report", m))
        job.finished.connect(lambda j=job: self.jobs.remove(j) if j in self.jobs else None)
        self.jobs.append(job)
        job.start()

    def run_spark(self):
        if self.current is None:
            return
        from .spark_dialog import SparkDialog
        SparkDialog(self, self.ws, self.current, self.st).exec()

    # ================================================================== misc
    def _update_actions(self):
        has = self.current is not None
        for x in (self.a.export, self.a.code_pd, self.a.code_sp, self.a.spark, self.a.report, self.a.add,
                  self.a.reload, self.a.save, self.a.save_as):
            x.setEnabled(has)
        self.a.undo.setEnabled(bool(self._undo))
        self.a.redo.setEnabled(bool(self._redo))
        self._update_title()

    def _update_title(self):
        ws = os.path.basename(self.ws.path) if self.ws.path else "unsaved workspace"
        self.setWindowTitle(f"Data Studio — {ws}{' •' if self.ws.dirty and self.ws.tables else ''} — AIO QC App")

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape and self.at_step is not None:
            self.steps.setCurrentRow(self.steps.count() - 1)
            return
        super().keyPressEvent(e)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            if u.isLocalFile():
                self._open_path(u.toLocalFile())

    # ------------------------------------------------------------------ settings
    def settings_sections(self):
        from AIO_QC.widgets.settings_pages import studio_sections
        return studio_sections(self)

    def open_settings(self, key=None):
        from AIO_QC.widgets.settings_dialog import open_settings
        return open_settings(self, self.settings_sections(), "Settings — Data Studio", key)

    def show_help(self):
        from AIO_QC.widgets.help_center_bridge import open_help
        open_help(self, "data")

    def shutdown(self):
        self.st.setValue("studio/geometry", self.saveGeometry())
        self.st.setValue("studio/state", self.saveState())
        for j in list(self.jobs):
            j.wait(5000)

