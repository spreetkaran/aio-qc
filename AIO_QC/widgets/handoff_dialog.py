# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
"""
Review ▸ Hand-off — send files to an annotator and merge their corrections back (core.handoff).

Export Batch…   which files (flagged · flagged + fixed · files in the list · riskiest in the Issues Inbox), a name, a
                folder → images + annotations in the dataset's layout, notes, findings and hint images.
Import Batch…   pick the returned folder / zip → every file compared three ways (as exported · as returned · the
                dataset now) → ready / conflict / unchanged / … with what each side changed. Double-click shows the
                returned shapes over the image (dashed). Merge writes the ticked files (backup first), records them
                in each file's History and marks them 🔧 Fixed so they get a final look.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QSpinBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
)

from AIO_QC.core import annodiff, handoff
from AIO_QC.core.dataset import DETECTION_TYPES, DatasetType
from AIO_QC.core.state import REVIEW_FIXED, REVIEW_FLAG
from AIO_QC.utils import hexc, newButton, newIcon, style as apply_style
from .inference_service import FuncWorker

ROLE = Qt.ItemDataRole.UserRole


def _inside(child: Path, parent: Path) -> bool:
    try:
        Path(os.path.abspath(child)).relative_to(os.path.abspath(parent))
        return True
    except ValueError:
        return False


# ====================================================================================================== export
class ExportBatchDialog(QDialog):
    TITLE = "Export Hand-off Batch"
    SUPPORTED = DETECTION_TYPES | {DatasetType.CLS_LABELME}      # files with annotations

    def __init__(self, app):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(newIcon("export"))
        self.setMinimumWidth(620)
        self.cancel = threading.Event()
        self.worker = None
        st = app.settings
        s = self.session
        self.scope = QComboBox()
        self.top_n = QSpinBox()
        self.top_n.setRange(1, 1_000_000)
        self.top_n.setValue(int(st.value("handoff/top_n", 200)))
        self.top_n.setPrefix("top ")
        self.top_n.valueChanged.connect(lambda _v: self._update())
        self.name = QLineEdit(time.strftime(f"{s.root.name}_batch_%Y%m%d_%H%M"))
        self.folder = QLineEdit(st.value("handoff/parent", "") or str(s.root.parent))
        self.browse = newButton("Browse…", icon="open", slot=self._browse)
        self.hints = QCheckBox("Hint images: the image with the flagged regions drawn (_hints/)")
        self.hints.setChecked(st.value("handoff/hints", True, type=bool))
        self.status = QLabel("")
        self.status.setWordWrap(True)
        apply_style(self.status, "color:@muted;")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.go = newButton("Export", icon="export", slot=self._export)
        self.go.setDefault(True)
        self.close_btn = newButton("Close", slot=self.close)
        g = QGridLayout()
        g.addWidget(QLabel("Files:"), 0, 0)
        row = QHBoxLayout()
        row.addWidget(self.scope, 1)
        row.addWidget(self.top_n)
        g.addLayout(row, 0, 1, 1, 2)
        g.addWidget(QLabel("Batch name:"), 1, 0)
        g.addWidget(self.name, 1, 1, 1, 2)
        g.addWidget(QLabel("Into folder:"), 2, 0)
        g.addWidget(self.folder, 2, 1)
        g.addWidget(self.browse, 2, 2)
        g.addWidget(self.hints, 3, 1, 1, 2)
        expl = QLabel("The annotator gets the files in the dataset's own layout with your notes and the findings "
                      "(notes.csv, README.txt). Keep batch.json in the folder — Import Batch uses it to merge the "
                      "corrections back and to spot files that changed on both sides.")
        expl.setWordWrap(True)
        apply_style(expl, "color:@muted;")
        foot = QHBoxLayout()
        foot.addWidget(self.progress, 1)
        foot.addWidget(self.go)
        foot.addWidget(self.close_btn)
        lay = QVBoxLayout(self)
        lay.addWidget(expl)
        lay.addLayout(g)
        lay.addWidget(self.status)
        lay.addLayout(foot)
        self._fill()
        self.scope.currentIndexChanged.connect(lambda _i: self._update())
        for e in (self.name, self.folder):
            e.textChanged.connect(lambda _t: self._update())
        self._update()

    def _fill(self):
        s, dock = self.session, self.app.file_dock
        rank = dock.model.row_of
        flagged = s.keys_with_review({REVIEW_FLAG}, rank=rank)
        both = s.keys_with_review({REVIEW_FLAG, REVIEW_FIXED}, rank=rank)
        self.scope.addItem(f"Flagged files ({len(flagged):,})", ("keys", flagged))
        self.scope.addItem(f"Flagged + Fixed files ({len(both):,})", ("keys", both))
        if dock.proxy.filtering():
            shown = dock.visible_keys()
            self.scope.addItem(f"Files in the list now ({len(shown):,})", ("keys", shown))
        if s.issues.by_tool:
            self.scope.addItem("Riskiest files in the Issues Inbox", ("risk", None))
        self.scope.setCurrentIndex(0 if flagged or self.scope.count() == 1 else self.scope.count() - 1)

    def _keys(self) -> list[str]:
        kind, data = self.scope.currentData() or ("keys", [])
        if kind == "risk":
            alive = set(self.session.by_key)
            return [k for k, _r in self.session.issues.ranked(alive)[: self.top_n.value()]]
        return list(data)

    def _out(self) -> Path:
        return Path(self.folder.text().strip()) / self.name.text().strip()

    def _update(self):
        self.top_n.setVisible((self.scope.currentData() or ("",))[0] == "risk")
        n = len(self._keys())
        out = self._out()
        err = ""
        if not n:
            err = "No files in this choice."
        elif not self.name.text().strip() or not self.folder.text().strip():
            err = "Name the batch and choose a folder."
        elif _inside(out, self.session.root):
            err = "Choose a folder outside the dataset (a copy inside it would be loaded as more data)."
        elif out.exists() and any(out.iterdir()):
            err = f"{out} already exists and is not empty."
        self.status.setText(f"⚠ {err}" if err else f"{n:,} file(s) → {out}")
        self.go.setEnabled(not err and self.worker is None)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Put the batch folder in", self.folder.text())
        if d:
            self.folder.setText(d)

    def _items(self, keys):
        """Runs in the worker: annotation + image + note + findings + marked regions per file."""
        s = self.session
        out = []
        for k in keys:
            smp = s.by_key.get(k)
            if smp is None or self.cancel.is_set():
                continue
            ann = None
            img = smp.image_path
            issues = s.issues.file_issues(k)
            need_shapes = any(i.shape is not None for i in issues)
            if (img is None or need_shapes) and smp.ann_path is not None:
                try:
                    ann = s.base_adapter.load(smp, light=True)
                    img = img or ann.image_path
                except Exception:
                    ann = None
            marks = []
            if ann is not None:
                for i in issues:
                    if i.shape is not None and 0 <= i.shape < len(ann.shapes):
                        pts = ann.shapes[i.shape].get("points") or []
                        if pts:
                            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                            marks.append((min(xs), min(ys), max(xs), max(ys), i.text[:40]))
            e = s.review.get(k) or {}
            out.append(handoff.BatchItem(
                k, smp.display, str(smp.ann_path) if smp.ann_path is not None else None,
                str(img) if img is not None else None, e.get("note", ""), e.get("status", ""),
                [f"{s.issues.titles.get(i.tool, i.tool)}: {i.text}" for i in issues], marks))
        return out

    def _export(self):
        keys = self._keys()
        if not keys:
            return
        st, s = self.app.settings, self.session
        st.setValue("handoff/parent", self.folder.text().strip())
        st.setValue("handoff/hints", self.hints.isChecked())
        st.setValue("handoff/top_n", self.top_n.value())
        out, title, hints = self._out(), self.name.text().strip(), self.hints.isChecked()
        kind = "yolo" if s.is_yolo else "labelme"
        names = list(s.base_adapter.class_names) if s.is_yolo else []
        cm = getattr(s.base_adapter, "class_map", None)
        class_files = [cm.source] if cm is not None and getattr(cm, "source", None) else []
        self.cancel.clear()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.go.setEnabled(False)
        self.close_btn.setText("Stop")
        self.close_btn.clicked.disconnect()
        self.close_btn.clicked.connect(self.cancel.set)

        def job():
            items = self._items(keys)
            return handoff.export_batch(s.root, items, out, kind, names, title, hints, class_files,
                                        cancel=self.cancel)
        self.worker = FuncWorker(job, self)
        self.worker.done.connect(lambda r: self._done(r, out))
        self.worker.failed.connect(lambda m: self._done({"error": m}, out))
        self.worker.start()

    def _done(self, st, out):
        self.worker = None
        self.progress.setVisible(False)
        self.close_btn.setText("Close")
        self.close_btn.clicked.disconnect()
        self.close_btn.clicked.connect(self.close)
        if "error" in st:
            self.status.setText(f"⚠ Export failed: {st['error']}")
            logger.error(f"[hand-off] export failed: {st['error']}")
            self._update()
            return
        msg = f"✓ Exported {st['items']:,} file(s) ({st['files']:,} copied"
        msg += f", {st['hints']:,} hint images)" if st.get("hints") else ")"
        if st["missing"]:
            msg += f" · {len(st['missing'])} skipped (no files)"
        if st["errors"]:
            msg += f" · {len(st['errors'])} error(s): {st['errors'][0]}"
        if st.get("cancelled"):
            msg += " · stopped early (the batch is still valid for what was exported)"
        self.status.setText(msg)
        logger.info(f"[hand-off] {msg} → {out}")
        open_btn = newButton("Open Folder", icon="open", slot=lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(out))))
        self.layout().itemAt(self.layout().count() - 1).layout().insertWidget(1, open_btn)
        self.go.setEnabled(False)

    def closeEvent(self, e):
        if self.worker is not None:
            getattr(self, "cancel", threading.Event()).set()
            self.worker.wait(3000)
        if hasattr(self.app, "forget_dialog"):
            self.app.forget_dialog(self)
        super().closeEvent(e)


# ====================================================================================================== import
STATE_ORDER = {"conflict": 0, "ready": 1, "invalid": 2, "missing": 3, "unchanged": 4, "same": 5}
STATE_ICON = {"ready": "✓", "conflict": "⚠", "unchanged": "·", "same": "=", "missing": "?", "invalid": "✕"}


class ImportBatchDialog(QDialog):
    TITLE = "Import Hand-off Batch"
    SUPPORTED = DETECTION_TYPES | {DatasetType.CLS_LABELME}      # files with annotations

    def __init__(self, app, path=None):
        super().__init__(app)
        self.app, self.session = app, app.session
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(newIcon("import"))
        self.resize(980, 600)
        self.batch_dir = None
        self.manifest = None
        self.rows: list[handoff.MergeRow] = []
        self.worker = None
        self.path = QLineEdit()
        self.path.setPlaceholderText("The returned batch folder, its batch.json, or a zip of it")
        self.browse = newButton("Folder…", icon="open", slot=self._browse)
        self.browse_zip = newButton("Zip…", slot=self._browse_zip)
        self.load_btn = newButton("Compare", icon="refresh", slot=self._load)
        self.head = QLabel("")
        self.head.setWordWrap(True)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File", "State", "Annotator changed", "Changed here since export", "Note"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSortingEnabled(False)
        h = self.tree.header()
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tree.itemDoubleClicked.connect(self._show)
        self.tree.itemChanged.connect(lambda *_: self._count())
        self.only = QComboBox()
        self.only.addItem("All files", None)
        for k, t in handoff.STATES.items():
            self.only.addItem(t, k)
        self.only.currentIndexChanged.connect(lambda _i: self._filter())
        self.take_all = newButton("Take theirs for all conflicts", slot=lambda: self._tick_conflicts(True))
        self.keep_all = newButton("Keep mine for all conflicts", slot=lambda: self._tick_conflicts(False))
        self.mark_fixed = QCheckBox("Mark merged files 🔧 Fixed (needs a final look)")
        self.mark_fixed.setChecked(app.settings.value("handoff/mark_fixed", True, type=bool))
        self.status = QLabel("")
        self.status.setWordWrap(True)
        apply_style(self.status, "color:@muted;")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.merge_btn = newButton("Merge Ticked Files", icon="save", slot=self._merge)
        self.merge_btn.setEnabled(False)
        close = newButton("Close", slot=self.close)
        top = QHBoxLayout()
        top.addWidget(self.path, 1)
        top.addWidget(self.browse)
        top.addWidget(self.browse_zip)
        top.addWidget(self.load_btn)
        tools = QHBoxLayout()
        tools.addWidget(QLabel("Show:"))
        tools.addWidget(self.only)
        tools.addStretch(1)
        tools.addWidget(self.take_all)
        tools.addWidget(self.keep_all)
        foot = QHBoxLayout()
        foot.addWidget(self.mark_fixed)
        foot.addWidget(self.progress, 1)
        foot.addWidget(self.merge_btn)
        foot.addWidget(close)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.head)
        lay.addLayout(tools)
        lay.addWidget(self.tree, 1)
        lay.addWidget(self.status)
        lay.addLayout(foot)
        hint = QLabel("Ticked files are written (each backed up first — undo in Backups or the file's History). "
                      "Ready files are ticked; conflicts are not until you choose. Double-click a row to see the "
                      "returned shapes over the image.")
        hint.setWordWrap(True)
        apply_style(hint, "color:@muted;")
        lay.insertWidget(1, hint)
        if path:
            self.path.setText(str(path))
            self._load()

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Returned batch folder", self.app.settings.value("handoff/parent", ""))
        if d:
            self.path.setText(d)
            self._load()

    def _browse_zip(self):
        f, _ = QFileDialog.getOpenFileName(self, "Returned batch (zip)", self.app.settings.value("handoff/parent", ""),
                                           "Zip (*.zip)")
        if f:
            self.path.setText(f)
            self._load()

    # ------------------------------------------------------------------ compare
    def _load(self):
        s = self.session
        p = self.path.text().strip()
        if not p or self.worker is not None:
            return
        try:
            bd = handoff.find_batch(p)
            if bd is None:
                raise ValueError("no batch.json found there")
            m = handoff.load_manifest(bd)
        except Exception as exc:
            self.status.setText(f"⚠ {exc}")
            return
        ds = os.path.abspath(m.get("dataset", ""))
        warn = ""
        if ds and os.path.normcase(ds) != os.path.normcase(os.path.abspath(s.root)):
            warn = f"\n⚠ Exported from {ds} — merging into {s.root}: files are matched by their path in the dataset."
        kind = "yolo" if s.is_yolo else "labelme"
        if m.get("kind") and m["kind"] != kind:
            self.status.setText(f"⚠ This batch is {m['kind']}; the open dataset is {kind}.")
            return
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.get("created", 0)))
        self.head.setText(f"<b>{m.get('title', '')}</b> · {len(m.get('items', [])):,} files · exported {when} by "
                          f"{m.get('user', '?')}{warn.replace(chr(10), '<br>')}")
        self.batch_dir, self.manifest = bd, m
        names = list(s.base_adapter.class_names) if s.is_yolo else None
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.worker = FuncWorker(lambda: handoff.compare(s.root, m, bd, kind, names), self)
        self.worker.done.connect(self._compared)
        self.worker.failed.connect(lambda e: (self._idle(), self.status.setText(f"⚠ Compare failed: {e}")))
        self.worker.start()

    def _idle(self):
        self.worker = None
        self.progress.setVisible(False)

    def _compared(self, rows):
        self._idle()
        self.rows = sorted(rows, key=lambda r: (STATE_ORDER.get(r.state, 9), r.entry.get("ann") or ""))
        self.tree.blockSignals(True)
        self.tree.clear()
        warn, ok = hexc("warn"), hexc("ok")
        for i, r in enumerate(self.rows):
            name = r.entry.get("ann") or r.entry.get("key")
            it = QTreeWidgetItem([name, f"{STATE_ICON.get(r.state, '')} {handoff.STATES[r.state]}",
                                  r.theirs, r.mine or r.reason, r.entry.get("note", "")])
            it.setData(0, ROLE, i)
            it.setToolTip(0, "\n".join([name] + r.entry.get("issues", [])[:10]))
            it.setToolTip(3, r.reason or r.mine)
            if r.state in ("ready", "conflict"):
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(0, Qt.CheckState.Checked if r.state == "ready" else Qt.CheckState.Unchecked)
            if r.state == "conflict":
                it.setForeground(1, self._color(warn))
            elif r.state == "ready":
                it.setForeground(1, self._color(ok))
            self.tree.addTopLevelItem(it)
        self.tree.blockSignals(False)
        for c in (0, 1):
            self.tree.resizeColumnToContents(c)
        self._filter()
        self._count()

    @staticmethod
    def _color(h):
        from PyQt6.QtGui import QColor
        return QColor(h)

    def _filter(self):
        want = self.only.currentData()
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            it.setHidden(want is not None and self.rows[it.data(0, ROLE)].state != want)

    def _tick_conflicts(self, on: bool):
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if self.rows[it.data(0, ROLE)].state == "conflict":
                it.setCheckState(0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def _ticked(self) -> list[handoff.MergeRow]:
        out = []
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            r = self.rows[it.data(0, ROLE)]
            r.choice = "theirs" if it.checkState(0) == Qt.CheckState.Checked and r.state in ("ready", "conflict") \
                else ""
            if r.choice:
                out.append(r)
        return out

    def _count(self):
        from collections import Counter
        c = Counter(r.state for r in self.rows)
        n = len(self._ticked())
        parts = [f"{c[k]:,} {handoff.STATES[k].lower()}" for k in STATE_ORDER if c.get(k)]
        self.status.setText(" · ".join(parts) + (f" — {n:,} ticked to merge" if self.rows else ""))
        self.merge_btn.setEnabled(n > 0 and self.worker is None)

    # ------------------------------------------------------------------ preview
    def _show(self, it, _col=0):
        r = self.rows[it.data(0, ROLE)]
        s = self.session
        key = self._key(r)
        if key is None:
            return
        self.app.open_sample(key)
        if r.data is None:
            return
        names = list(s.base_adapter.class_names) if s.is_yolo else None
        a = annodiff.parse(r.data, "yolo" if s.is_yolo else "labelme", names)
        pix = self.app.canvas.pixmap_item.pixmap()
        W, H = pix.width(), pix.height()
        self.app.canvas.set_overlay([{"box": (b[0] * W, b[1] * H, b[2] * W, b[3] * H), "text": f"returned: {lab}",
                                      "color": hexc("accent")} for lab, _t, b, _p in a.get("shapes", [])])
        self.app.statusBar().showMessage(f"Dashed = the returned version · {r.theirs}", 8000)

    def _key(self, r) -> str | None:
        s = self.session
        rel = r.entry.get("key")
        if not rel:
            return None
        k = os.path.abspath(os.path.join(str(s.root), rel))
        if k in s.by_key:
            return k
        k = str(Path(s.root) / rel)
        return k if k in s.by_key else None

    # ------------------------------------------------------------------ merge
    def _merge(self):
        s = self.session
        todo = self._ticked()
        if not todo:
            return
        conflicts = sum(r.state == "conflict" for r in todo)
        locked = [r for r in todo if (self._key(r) or "") in s.locked]
        if locked:
            QMessageBox.information(self, self.TITLE, f"{len(locked)} file(s) are open in an editor session — end it "
                                                      "first.")
            return
        msg = f"Write {len(todo):,} returned file(s) into the dataset?"
        if conflicts:
            msg += f"\n\n{conflicts:,} of them changed here since the export — those changes are replaced by the " \
                   "annotator's version."
        msg += "\n\nEvery file is backed up first (Backups ▸ undo, or the file's History)."
        if QMessageBox.question(self, self.TITLE, msg) != QMessageBox.StandardButton.Yes:
            return
        self.app.settings.setValue("handoff/mark_fixed", self.mark_fixed.isChecked())
        title = (self.manifest or {}).get("title", "batch")
        s.ownPaths.emit([str(r.target) for r in todo])          # our writes, not "changed outside AIO QC"
        written, errors, op = handoff.apply_rows(s.root, todo, title)
        keys = [k for k in (self._key(r) for r in written) if k]
        if keys:
            details = {}
            for r in written:
                k = self._key(r)
                if k:
                    details.setdefault(r.theirs, []).append(k)
            for what, ks in details.items():
                s.log(ks, "batch", f"merged from “{title}”: {what}", tool="Hand-off batch", op=op)
            if self.mark_fixed.isChecked():
                s.set_review(keys, REVIEW_FIXED, f"Merged from batch “{title}” · {time.strftime('%Y-%m-%d %H:%M')}")
            s.reindex_keys(keys)
            if self.app.current_key in keys:
                self.app.reload_current_annotation()
        logger.info(f"[hand-off] merged {len(written)} file(s) from {title}; {len(errors)} error(s)")
        if errors:
            QMessageBox.warning(self, self.TITLE, f"{len(errors)} file(s) could not be written:\n" + "\n".join(errors[:10]))
        self.app.statusBar().showMessage(f"Merged {len(written):,} file(s) from “{title}”", 8000)
        self._load()                                    # re-compare: merged rows now read "Already merged"

    def closeEvent(self, e):
        if self.worker is not None:
            getattr(self, "cancel", threading.Event()).set()
            self.worker.wait(3000)
        if hasattr(self.app, "forget_dialog"):
            self.app.forget_dialog(self)
        super().closeEvent(e)
