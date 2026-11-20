# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Review ▸ Save Flagged Files… — one window, two exclusive choices:
  ● copy the flagged files (annotation + image) into a folder you choose, or
  ● write their names into a text file you choose.
"""
from __future__ import annotations

import threading
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QRadioButton, QVBoxLayout, QWidget,
)

from AIO_QC.core.flagged_export import LIST_FORMATS, LIST_MODES, FlaggedItem, copy_items, write_list
from AIO_QC.core.state import REVIEW_FIXED, REVIEW_FLAG
from AIO_QC.utils import newButton, newIcon, style as apply_style
from .inference_service import FuncWorker
from AIO_QC.core.diag import ignored as _ignored


SCOPE_ROLE = 0x0100 + 1           # Qt.ItemDataRole.UserRole + 1: "all" | "list"


class FlaggedFilesDialog(QDialog):
    TITLE = "Save Flagged Files"

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(newIcon("review-flag"))
        self.setMinimumWidth(640)
        self.cancel = threading.Event()
        self.worker = None
        st = app.settings

        # ---------------------------------------------------------------- which files
        self.scope = QComboBox()
        self.with_fixed = QCheckBox("Also files marked Fixed")
        self.with_fixed.setToolTip("Files corrected in an editor but not yet checked (S)")
        self.with_fixed.toggled.connect(lambda _on: self._fill_scope())
        top = QHBoxLayout()
        top.addWidget(QLabel("Files:"))
        top.addWidget(self.scope, 1)
        top.addWidget(self.with_fixed)

        # ---------------------------------------------------------------- option 1: copy to a folder
        self.rb_copy = QRadioButton("Copy the files to a folder")
        # the place used last time, the folder always named after THIS dataset: <dataset>_flagged
        # (a place you chose yourself is kept; otherwise next to the dataset)
        chosen = st.value("flagged/base", "") or ""
        base = Path(chosen) if chosen and Path(chosen).is_dir() else self.session.root.parent
        self.folder = QLineEdit(str(base / f"{self.session.root.name}_flagged"))
        self.folder_btn = newButton("Browse…", icon="open", slot=self._browse_folder)
        self.keep_tree = QCheckBox("Keep the dataset's folder structure (YOLO images / labels stay paired)")
        self.keep_tree.setChecked(st.value("flagged/keep_tree", True, type=bool))
        self.notes_csv = QCheckBox("Add flagged_files.csv with the notes")
        self.notes_csv.setChecked(st.value("flagged/notes_csv", True, type=bool))
        copy_box = QFrame()
        cg = QGridLayout(copy_box)
        cg.setContentsMargins(28, 0, 0, 4)
        cg.addWidget(self.folder, 0, 0)
        cg.addWidget(self.folder_btn, 0, 1)
        cg.addWidget(self.keep_tree, 1, 0, 1, 2)
        cg.addWidget(self.notes_csv, 2, 0, 1, 2)

        # ---------------------------------------------------------------- option 2: names in a text file
        self.rb_list = QRadioButton("Save a list of the files (CSV or TXT)")
        self.fmt = QComboBox()
        for k, t in LIST_FORMATS.items():
            self.fmt.addItem(t, k)
        fmt = str(st.value("flagged/list_format", "csv") or "csv")
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(fmt)))
        # the folder used last time, the name always <dataset>_flagged.<csv|txt>
        last = str(st.value("flagged/txt", "") or "")
        folder = Path(last).parent if last and Path(last).parent.is_dir() else self.session.root.parent
        self.txt = QLineEdit(str(folder / f"{self.session.root.name}_flagged.{self.fmt.currentData()}"))
        self.fmt.currentIndexChanged.connect(lambda _i: self._format_changed())
        self.txt_btn = newButton("Browse…", icon="save", slot=self._browse_txt)
        self.list_mode = QComboBox()
        for k, t in LIST_MODES.items():
            self.list_mode.addItem(t, k)
        self.list_mode.setCurrentIndex(max(0, self.list_mode.findData(st.value("flagged/list_mode", "relative"))))
        self.list_notes = QCheckBox("Add the note after each name (tab-separated)")
        list_box = QFrame()
        lg = QGridLayout(list_box)
        lg.setContentsMargins(28, 0, 0, 4)
        lg.addWidget(self.txt, 0, 0)
        lg.addWidget(self.txt_btn, 0, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Format:"))
        row.addWidget(self.fmt)
        row.addWidget(QLabel("Write:"))
        row.addWidget(self.list_mode)
        row.addWidget(self.list_notes)
        row.addStretch(1)
        w = QWidget()
        w.setLayout(row)
        lg.addWidget(w, 1, 0, 1, 2)

        self.group = QButtonGroup(self)                  # exactly one of the two
        self.group.setExclusive(True)
        self.group.addButton(self.rb_copy, 0)
        self.group.addButton(self.rb_list, 1)
        (self.rb_list if st.value("flagged/choice", "copy") == "list" else self.rb_copy).setChecked(True)
        self.group.idToggled.connect(lambda _i, _on: self._update())
        self._copy_widgets = (self.folder, self.folder_btn, self.keep_tree, self.notes_csv)
        self._list_widgets = (self.txt, self.txt_btn, self.fmt, self.list_mode, self.list_notes)
        for e in (self.folder, self.txt):
            e.textChanged.connect(lambda _t: self._update())

        self.status = QLabel("")
        self.status.setWordWrap(True)
        apply_style(self.status, "color:@muted;")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        foot = QHBoxLayout()
        foot.addWidget(self.progress, 1)
        self.save_btn = newButton("Save", icon="save", slot=self._save)
        self.save_btn.setDefault(True)
        self.close_btn = newButton("Close", slot=self.close)
        foot.addWidget(self.save_btn)
        foot.addWidget(self.close_btn)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addSpacing(6)
        lay.addWidget(self.rb_copy)
        lay.addWidget(copy_box)
        lay.addWidget(self.rb_list)
        lay.addWidget(list_box)
        lay.addWidget(self.status)
        lay.addLayout(foot)
        self.list_notes.setVisible(self.fmt.currentData() == "txt")
        self._fill_scope()
        self._update()
        # a mark set or cleared (or the list filter changed) while this is open: the counts follow
        from PyQt6.QtCore import QTimer
        self._live = QTimer(self, singleShot=True, interval=150, timeout=self._live_fill)
        self.session.reviewChanged.connect(self._schedule)        # bound methods: dropped with the window
        app.file_dock.countChanged.connect(self._schedule)

    def _schedule(self, *_a):
        self._live.start()

    def _live_fill(self):
        if self.worker is None and self.isVisible():
            self._fill_scope()

    # ------------------------------------------------------------------ helpers
    def _statuses(self):
        return {REVIEW_FLAG, REVIEW_FIXED} if self.with_fixed.isChecked() else {REVIEW_FLAG}

    def _fill_scope(self):
        s, sts = self.session, self._statuses()
        dock = self.app.file_dock
        every = s.keys_with_review(sts, rank=dock.model.row_of)
        if dock.proxy.filtering():
            shown = set(dock.visible_keys())
            listed = [k for k in every if k in shown]
        else:
            listed = every
        cur = self.scope.currentData(SCOPE_ROLE) if self.scope.count() else "all"
        self.scope.clear()
        self.scope.addItem(f"All flagged files in the dataset ({len(every):,})", every)
        self.scope.setItemData(0, "all", SCOPE_ROLE)
        if len(listed) != len(every):
            self.scope.addItem(f"Flagged files in the current list ({len(listed):,})", listed)
            self.scope.setItemData(1, "list", SCOPE_ROLE)
            if cur == "list":
                self.scope.setCurrentIndex(1)
        self._update()

    def _update(self):
        copy = self.rb_copy.isChecked()
        for w in self._copy_widgets:
            w.setEnabled(copy)
        for w in self._list_widgets:
            w.setEnabled(not copy)
        n = len(self.scope.currentData() or [])
        target = (self.folder if copy else self.txt).text().strip()
        bad = ""
        if copy and target:
            try:
                out, root = Path(target).resolve(), self.session.root.resolve()
                if out == root or root in out.parents:
                    bad = "Choose a folder outside the dataset (copies inside it would be picked up as new files)."
            except OSError:
                pass
        self.save_btn.setEnabled(bool(n and target and not bad) and self.worker is None)
        self.save_btn.setText(f"Copy {n:,} file(s)" if copy else f"Save {n:,} name(s)")
        self.status.setText(bad or ("No flagged files." if not n else ""))

    def _browse_folder(self):
        from AIO_QC.utils.save_dirs import ask_dir
        d = ask_dir(self, "Folder for the flagged files", kind="datasets", current=self.folder.text())
        if d:
            self.folder.setText(d)

    def _format_changed(self):
        ext = self.fmt.currentData()
        p = Path(self.txt.text().strip() or f"{self.session.root.name}_flagged.{ext}")
        self.txt.setText(str(p.with_suffix("." + ext)))
        self.list_notes.setVisible(ext == "txt")              # the CSV always has a note column
        self._update()

    def _browse_txt(self):
        from AIO_QC.utils.save_dirs import ask_save
        ext = self.fmt.currentData()
        filt = "CSV (*.csv);;Text (*.txt)" if ext == "csv" else "Text (*.txt);;CSV (*.csv)"
        p, chosen = ask_save(self, "List of the flagged files", self.txt.text(), filt, kind="exports", keep_dir=True)
        if p:
            if not Path(p).suffix:
                p += ".csv" if "csv" in chosen.lower() else ".txt"
            suf = Path(p).suffix.lower().lstrip(".")
            if suf in LIST_FORMATS and suf != ext:
                self.fmt.blockSignals(True)
                self.fmt.setCurrentIndex(self.fmt.findData(suf))
                self.fmt.blockSignals(False)
                self.list_notes.setVisible(suf == "txt")
            self.txt.setText(p)

    def _items(self, keys):
        """Annotation + image of each file (LabelMe images are resolved like the viewer does)."""
        s = self.session
        out = []
        for k in keys:
            smp = s.by_key.get(k)
            if smp is None:
                continue
            img = smp.image_path
            if img is None and smp.ann_path is not None:
                try:
                    img = s.base_adapter.load(smp, light=True).image_path
                except Exception as _exc:
                    _ignored(_exc)
                    img = None
            e = s.review.get(k) or {}
            files = [p for p in (smp.ann_path, img) if p is not None]
            out.append(FlaggedItem(smp.display, files, e.get("status", ""), e.get("note", ""), e.get("ts", "")))
        return out

    def _remember(self):
        st = self.app.settings
        st.setValue("flagged/choice", "copy" if self.rb_copy.isChecked() else "list")
        st.setValue("flagged/folder", self.folder.text().strip())
        parent = Path(self.folder.text().strip()).parent if self.folder.text().strip() else None
        st.setValue("flagged/base", str(parent) if parent is not None and parent != self.session.root.parent else "")
        st.setValue("flagged/txt", self.txt.text().strip())
        st.setValue("flagged/keep_tree", self.keep_tree.isChecked())
        st.setValue("flagged/notes_csv", self.notes_csv.isChecked())
        st.setValue("flagged/list_mode", self.list_mode.currentData())
        st.setValue("flagged/list_format", self.fmt.currentData())

    # ------------------------------------------------------------------ save
    def _save(self):
        keys = list(self.scope.currentData() or [])
        if not keys:
            return
        self._remember()
        items = self._items(keys)
        root = self.session.root
        if self.rb_list.isChecked():
            path = Path(self.txt.text().strip())
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                n = write_list(items, root, path, self.list_mode.currentData(), self.list_notes.isChecked())
            except OSError as exc:
                self.status.setText(f"⚠ Could not write {path}: {exc}")
                return
            logger.info(f"[review] {n} flagged file name(s) written to {path}")
            self._done(f"Saved {n:,} file name(s) to {path.name}", path.parent)
            return
        out = Path(self.folder.text().strip())
        self.cancel.clear()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.save_btn.setEnabled(False)
        self.close_btn.setText("Stop")
        try:
            self.close_btn.clicked.disconnect()
        except TypeError:
            pass
        self.close_btn.clicked.connect(self.cancel.set)
        keep, notes = self.keep_tree.isChecked(), self.notes_csv.isChecked()
        w = FuncWorker(lambda: copy_items(items, root, out, keep, notes, cancel=self.cancel), self)
        w.done.connect(lambda st: self._copied(st, out))
        w.failed.connect(lambda m: self._copied({"error": m}, out))
        self.worker = w
        w.start()

    def _copied(self, st, out):
        self.worker = None
        self.progress.setVisible(False)
        self.close_btn.setText("Close")
        try:
            self.close_btn.clicked.disconnect()
        except TypeError:
            pass
        self.close_btn.clicked.connect(self.close)
        if "error" in st:
            self.status.setText(f"⚠ Copy failed: {st['error']}")
            logger.error(f"[review] copying flagged files failed: {st['error']}")
            self._update()
            return
        msg = (f"Copied {st['items']:,} flagged file(s) ({st['files']:,} files) to {out.name}"
               + (" — stopped" if st["cancelled"] else "")
               + (f" · {len(st['missing'])} not found" if st["missing"] else "")
               + (f" · {len(st['errors'])} error(s)" if st["errors"] else ""))
        logger.info(f"[review] {msg} ({out})")
        if st["missing"] or st["errors"]:
            self.app.log_errors([f"[save flagged] missing: {m}" for m in st["missing"]] +
                                [f"[save flagged] {e}" for e in st["errors"]], quiet=True)
        self._done(msg, out)

    def _done(self, msg, folder):
        self.status.setText(msg)
        self._update()
        self.app.toast.show_message(f"<b>Flagged files saved</b><br>{msg}", kind="success",
                                    actions=[("Open folder", lambda: QDesktopServices.openUrl(
                                        QUrl.fromLocalFile(str(folder))))])

    def closeEvent(self, e):
        self.cancel.set()
        if self.worker is not None:
            self.worker.wait(5000)
        super().closeEvent(e)
