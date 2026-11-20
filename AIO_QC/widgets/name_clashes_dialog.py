# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""
Name Clashes (QC ▸ Dataset): file names that would collide — when the dataset is flattened (Convert & Split flat,
Combine & Flatten, most training tools), copied to Windows, or already today:

  · the same name in different sub-folders (camA/img001.jpg and camB/img001.jpg → one img001 when flattened)
  · names that differ only in upper / lower case (Windows and macOS treat them as ONE file)
  · two images sharing one label file (a.jpg and a.png both pair with a.json / a.txt)
  · names Windows refuses (\\ / : * ? " < > |, CON / PRN / AUX / NUL / COM1 … / LPT1 …, a trailing dot or
    space) and paths longer than Windows allows

For each clash: whether the images are identical (a duplicate — Duplicate Images removes one) or different
(renaming keeps both), and what would break. Rename… gives them new names with the naming panel (presets, your
own pattern, live check — it will not start while a name still clashes): image and label together, imagePath
fixed, review marks / notes / issues / cached results / backups follow; renamed.csv written; one Undo.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from PyQt6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLabel, QMessageBox, QVBoxLayout

from AIO_QC.core import naming as N
from AIO_QC.core import renamer as R
from .base_scan import BaseScanDialog, ResultRow

SAME_NAME, CASE_ONLY, ONE_LABEL, UNSAFE, TOO_LONG = "same_name", "case_only", "one_label", "unsafe", "too_long"
TEXT = {SAME_NAME: "Same name in different folders", CASE_ONLY: "Only upper / lower case differs",
        ONE_LABEL: "Two images share one label file", UNSAFE: "Name Windows refuses", TOO_LONG: "Path too long"}
BREAKS = {SAME_NAME: "flattening keeps only one of them (and its label)",
          CASE_ONLY: "Windows / macOS see one file: one overwrites the other",
          ONE_LABEL: "both images read the same label file",
          UNSAFE: "the file cannot be copied to Windows",
          TOO_LONG: "Windows tools fail on it (unless long paths are turned on)"}


def _same_content(paths) -> bool | None:
    """All files identical? (size first, then content). None when a file cannot be read."""
    try:
        sizes = {os.path.getsize(p) for p in paths}
        if len(sizes) > 1:
            return False
        return len({N.file_hash8(p) for p in paths}) == 1
    except OSError:
        return None


def find(items, root_len: int, per_split: bool = True) -> list[dict]:
    """[{kind, keys, items, identical, detail}] — Qt-free (runs in the scan thread)."""
    out = []
    flat_items = items if per_split else [N.Item(it.key, it.rel, "", it.path, it.extra_exts) for it in items]
    for g in N.clashes(flat_items, flat=True):
        its = [it for it in items if it.key in set(g)]
        names = {it.stem for it in its}
        folders = {it.parent_l for it in its}
        if len(folders) == 1:
            continue                                         # same folder: the in-place checks below say it
        kind = SAME_NAME if len(names) == 1 else CASE_ONLY
        out.append({"kind": kind, "keys": g, "items": its, "identical": _same_content([it.path for it in its]),
                    "detail": ", ".join(sorted(names))})
    by_key = {it.key: it for it in items}
    for g in N.clashes(items, flat=False):                   # in one folder
        its = [by_key[k] for k in g]
        stems = {it.stem for it in its}
        kind = ONE_LABEL if len(stems) == 1 else CASE_ONLY
        out.append({"kind": kind, "keys": g, "items": its, "identical": _same_content([it.path for it in its]),
                    "detail": " · ".join(Path(it.rel).name for it in its)})
    sc = N.Scheme(preset="custom", pattern="{stem}", only_clashes=False)
    for it in items:
        msgs = N._win_problems(it.stem, len(it.rel), root_len, sc, it)
        for m in msgs:
            kind = TOO_LONG if m.startswith("too long") else UNSAFE
            out.append({"kind": kind, "keys": [it.key], "items": [it], "identical": None, "detail": m})
    return out


class NameClashesDialog(BaseScanDialog):
    TITLE, ICON = "Name Clashes", ""
    HELP = ("File names that would collide: the same name in different sub-folders (flattening keeps only one), "
            "names that differ only in upper / lower case (Windows and macOS see one file), two images sharing one "
            "label file (a.jpg and a.png both use a.json), and names or paths Windows refuses. For each clash: "
            "whether the images are identical (a duplicate) or different (rename to keep both). Rename… gives "
            "them new names — presets or your own pattern, checked as you type; it will not start while a name "
            "still clashes. Image and label are renamed together, review marks, notes, issues, cached results and "
            "backups follow, renamed.csv lists every change, and ↶ Undo puts the names back.")
    COLUMNS = ["Problem", "Files", "Images", "What would break", "Detail"]
    FILTER_COLUMN = 0
    AUTO_START = True
    RESCAN_AFTER_EDIT = False

    def build_params(self, form):
        self.scope = QComboBox()
        self.scope.addItem("Flattened within each split (train / val / test)", True)
        self.scope.addItem("Flattened into one folder", False)
        self.scope.setToolTip("How the dataset will be flattened: names must be unique per split, or across all")
        self.scope.currentIndexChanged.connect(lambda _i: self.start_scan())
        form.addRow("Check for", self.scope)
        self.add_action("✎ Rename…", self.rename_rows,
                        "New names for the ticked (or selected) rows' files — or every clashing file when nothing "
                        "is ticked")
        self.undo_btn = self.add_action("↶ Undo rename", self.undo_rename, "Put back the names of the last Rename")
        self.undo_btn.setEnabled(False)
        self._log = []

    def post_init(self):
        self.progress.setFormat("")

    def samples_to_scan(self):
        return []

    def make_context(self):
        root = os.path.abspath(str(self.session.root))
        return {"items": R.items_for(self.session.samples, root), "root_len": len(root),
                "per_split": bool(self.scope.currentData())}

    def finalize(self, ctx):
        found = find(ctx["items"], ctx["root_len"], ctx["per_split"])
        ctx["found"] = found
        rows = []
        for f in found:
            same = {True: "identical (duplicate)", False: "different", None: "—"}[f["identical"]]
            if len(f["items"]) < 2:
                same = "—"
            files = " · ".join(it.rel for it in f["items"][:4]) + (" …" if len(f["items"]) > 4 else "")
            rows.append(ResultRow(f["keys"][0], (TEXT[f["kind"]], files, same, BREAKS[f["kind"]], f["detail"]),
                                  payload=f, tone="error" if f["kind"] in (SAME_NAME, CASE_ONLY, ONE_LABEL)
                                  else "warning"))
        return rows

    def summarize(self):
        found = (self.ctx or {}).get("found")
        if found is None:
            return super().summarize()
        if not found:
            return f"{len(self.session.samples):,} file(s) — no name clashes"
        c = Counter(f["kind"] for f in found)
        return " · ".join(f"{TEXT[k]}: {n:,}" for k, n in c.most_common())

    def on_row_activated(self, row):
        if row is not None and row.key in self.session.by_key:
            self.app.open_sample(row.key, None)

    # ---------------------------------------------------------------- rename
    def rename_rows(self):
        rows = self.target_rows() if any(self.is_marked(r) for r in self.rows if r) or \
            self.table.selectionModel().hasSelection() else [r for r in self.rows if r]
        keys = list(dict.fromkeys(k for r in rows if r.payload for k in r.payload["keys"]))
        if not keys:
            QMessageBox.information(self, self.TITLE, "No clashing files to rename.")
            return
        root = os.path.abspath(str(self.session.root))
        dlg = RenameDialog(self, R.items_for(self.session.samples, root), set(keys),
                           per_split=bool(self.scope.currentData()), root_len=len(root))
        self._rename_dlg = dlg
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        P = dlg.panel.plan()
        renames = [(it.key, P.new[it.key]) for it in P.items if P.new[it.key] != it.stem]
        self.run_renames(renames)

    def run_renames(self, renames):
        from AIO_QC.core.state import state_dir
        from AIO_QC.widgets.jobs import JobBar
        job = getattr(self, "job", None) or JobBar(self, "Renaming files")
        self.job = job
        if job.parent() is self and self.layout() is not None and self.layout().indexOf(job) < 0:
            self.layout().addWidget(job)
        stop = {"on": False}
        job.start(f"Renaming {len(renames):,} file(s)", len(renames), stop=lambda: stop.__setitem__("on", True))

        def progress(i, n):
            job.step(i, n)
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
        done, errors, log = self.session.rename_samples(renames, progress=progress, stop=lambda: stop["on"])
        self._log = log
        csv_path = None
        if log:
            d = state_dir(self.session.root) / "renames"
            d.mkdir(parents=True, exist_ok=True)
            import time
            csv_path = d / f"renamed_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            R.write_csv(csv_path, log, self.session.root)
        self.undo_btn.setEnabled(bool(log))
        msg = f"Renamed {len(done):,} of {len(renames):,} file(s)" + (f" · {len(errors)} could not be" if errors
                                                                      else "")
        undo = self._undo_from_bar if log else None
        if stop["on"]:
            job.stopped(undo=undo)
        else:
            job.finish(msg + (" · renamed.csv kept" if csv_path else ""), undo=undo,
                       open_path=str(csv_path.parent) if csv_path else None)
        if errors:
            self.report("Rename", [msg], errors)
        self.start_scan()
        return done, errors

    def _undo_from_bar(self):
        self.undo_rename()
        return "Names put back"

    def undo_rename(self):
        if not self._log:
            return
        back, errors = self.session.undo_renames(self._log)
        self._log = []
        self.undo_btn.setEnabled(False)
        if errors:
            self.report("Undo rename", [f"{len(back):,} name(s) put back."], errors)
        self.start_scan()


class RenameDialog(QDialog):
    """The naming panel over the dataset's files: the clashing ones are renamed (or all, if chosen)."""

    def __init__(self, parent, items, clashing: set, per_split: bool = True, root_len: int = 0):
        super().__init__(parent)
        from .naming_panel import NamingPanel
        self.setWindowTitle("Rename files")
        self.resize(900, 560)
        lay = QVBoxLayout(self)
        note = QLabel(f"{len(clashing):,} file(s) clash. They are renamed in their own folders — image and label "
                      "together; names are checked against every other file of the dataset.")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.panel = NamingPanel(self, context="name_clashes")
        lay.addWidget(self.panel, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok = bb.button(QDialogButtonBox.StandardButton.Ok)
        self.ok.setText("Rename")
        self.ok.setEnabled(False)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        if not per_split:
            items = [_without_split(it) for it in items]
        self.panel.planChanged.connect(lambda P: self.ok.setEnabled(P is not None and P.ok and P.renamed > 0))
        self.panel.set_items(items, flat=True, root_len=root_len, touch=clashing)


def _without_split(it):
    n = N.Item(it.key, it.rel, "", it.path, it.extra_exts)
    n.parent_l = it.parent_l
    return n
